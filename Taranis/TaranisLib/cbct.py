"""CBCT territory planning, Slicer side: working grid, CBCT field of view, markups, arterial tree segmentation,
centerline model, and writing predicted territories into the case segmentation as perfused-volume candidates.

All heavy work is done by vascular.py (pure numpy / scipy) on an axis-aligned working grid with isotropic voxels
(0.5-1.5 mm) around the whole liver, never on the full CBCT volume. The only output to the rest of Taranis is an
ordinary perfused-volume segment (a yellow candidate until the user accepts it), so dosimetry needs no change.
"""

import datetime
import json
import logging

import numpy as np
import vtk
import slicer

from . import roles as R
from . import segmentops as S
from . import segtools
from . import vascular as V
from . import workflow as W
from .case import (TaranisCase, P_PLANNING_RESULTS, P_PLANNING_SETTINGS, P_PLANNING_REVIEWED,
                   REF_PLANNING_INJECTION, REF_PLANNING_TIPS, REF_PLANNING_TREE, REF_PLANNING_CENTERLINES)
from .controller import (controlPoints, planningInputsKey, planningResults, segmentIDs, segmentRole, isCandidate,
                         setSegmentRole, segmentInfos, segmentsVoxels, _placement)

MODULE_NAME = "CBCTPlanning"
PLANNING_LAYOUT_ID = 50501
SPACING_CHOICES_MM = (0.5, 0.75, 1.0, 1.5)
DEFAULT_SPACING_MM = 1.0
MAX_GRID_VOXELS = 40e6           # larger working grids are coarsened automatically
GRID_MARGIN_MM = 20.0            # around the whole liver (vessels at the hilum, the tree outside the liver)
INJECTION_MARGIN_MM = 20.0       # the grid also covers the injection point and this margin around it
FOV_SAMPLE_VOXELS = 160          # the field of view is found on a subsampled CBCT (about this many voxels per axis)
PAD_VALUE = -1024.0              # outside the CBCT field of view: air (HU)
TUMOUR_MARGIN_MM = 5.0           # tumours this close to the whole-liver segment count as liver
DEFAULT_UPTAKE_PERCENT = 5.0     # MAA perfused volume for the comparison (as the hub's 'Perfused volume from uptake')
RULE_EUCLIDEAN = "euclidean"
RULE_GEODESIC = "geodesic"
RULE_LABELS = {RULE_EUCLIDEAN: "Nearest branch (Euclidean)", RULE_GEODESIC: "Shortest path inside the liver"}
RULE_SHORT = {RULE_EUCLIDEAN: "nearest branch", RULE_GEODESIC: "shortest path"}   # in segment names (no brackets)
TREE_NAME = "CBCT arterial tree"
TREE_COLOR = (0.86, 0.16, 0.16)
CENTERLINE_COLOR = (0.85, 0.08, 0.08)  # red lines on the white 3D view and over the grey MIP
CENTERLINE_COLOR_SHADED = (0.0, 0.5, 0.5)  # teal: the shaded rendering itself is red
CENTERLINE_WIDTH = 4
CENTERLINE_OPACITY = 0.5
CENTERLINE_OPACITY_DIMMED = 0.2         # the other vessels once territories are predicted
TERRITORY_STEM = "Territory"
ENHANCEMENT_STEM = "Perfused volume (CBCT enhancement"
SEARCH_NAME = "CBCT search mask"
SEARCH_COLOR = (0.3, 0.7, 1.0)
DEFAULT_SETTINGS = {"spacingMM": DEFAULT_SPACING_MM, "vesselSD": V.VESSEL_SD, "growSD": V.GROW_SD,
                    "tubularOnly": True, "showSearchMask": False,
                    "rule": RULE_EUCLIDEAN, "subtractNested": True, "enhancementPercent": 50.0,
                    "dsaStartPercentile": 10.0, "dsaTopPercentile": 99.0, "dsaMode": "mip",
                    "shadedStartPercentile": 97.0, "shadedMidPercentile": 98.5, "shadedTopPercentile": 100.0,
                    "negative": False,
                    "limitTerritories": False, "supplyDistanceMM": V.SUPPLY_DISTANCE_MM, "show3DLiver": True, "show3DTumours": True, "show3DPerfused": True, "show3DMip": True, "show3DVessels": True, "advanced": {}}

LAYOUT_XML = (
    '<layout type="horizontal" split="true">'
    ' <item splitSize="600"><view class="vtkMRMLViewNode" singletontag="CBCTPlanning3D">'
    '  <property name="viewlabel" action="default">P</property></view></item>'
    ' <item splitSize="400"><layout type="vertical" split="true">'
    '  <item><view class="vtkMRMLSliceNode" singletontag="Red">'
    '   <property name="orientation" action="default">Axial</property>'
    '   <property name="viewlabel" action="default">R</property>'
    '   <property name="viewcolor" action="default">#F34A33</property></view></item>'
    '  <item><view class="vtkMRMLSliceNode" singletontag="Green">'
    '   <property name="orientation" action="default">Coronal</property>'
    '   <property name="viewlabel" action="default">G</property>'
    '   <property name="viewcolor" action="default">#6EB04B</property></view></item>'
    ' </layout></item>'
    '</layout>')


# -- Store: the case node, or the module's own node without a case ------------------------------------------------

def storeNode():
    """Node holding the planning parameters and references: the Taranis case, or the module's parameter node."""
    node = TaranisCase.findNode()
    if node is not None:
        return node
    node = slicer.mrmlScene.GetSingletonNode(MODULE_NAME, "vtkMRMLScriptedModuleNode")
    if node is None:
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScriptedModuleNode")
        node.SetSingletonTag(MODULE_NAME)
        node.SetName("CBCT planning")
        node.SetHideFromEditors(True)
    return node


def settings(store):
    values = dict(DEFAULT_SETTINGS)
    try:
        values.update(json.loads(store.GetParameter(P_PLANNING_SETTINGS) or "{}"))
    except ValueError:
        pass
    return values


def setSettings(store, **values):
    current = settings(store)
    current.update(values)
    text = json.dumps(current, sort_keys=True)
    if store.GetParameter(P_PLANNING_SETTINGS) != text:
        store.SetParameter(P_PLANNING_SETTINGS, text)


def markupsNode(store, reference, create=False):
    node = store.GetNodeReference(reference)
    if node is None and create:
        name, color, label = {REF_PLANNING_INJECTION: ("CBCT injection point", (0.2, 0.8, 1.0), "I"),
                              REF_PLANNING_TIPS: ("Planned catheter tips", (1.0, 0.6, 0.0), "P")}[reference]
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", name)
        node.CreateDefaultDisplayNodes()
        display = node.GetDisplayNode()
        display.SetSelectedColor(*color)
        display.SetColor(*color)
        display.SetGlyphScale(2.5)
        node.SetMarkupLabelFormat(f"{label}%d")
        if reference == REF_PLANNING_INJECTION:
            node.SetMaximumNumberOfControlPoints(1)
        store.SetNodeReferenceID(reference, node.GetID())
    return node


def startPlacing(node):
    """Interactive placement of one control point in the markups node."""
    selection = slicer.app.applicationLogic().GetSelectionNode()
    selection.SetReferenceActivePlaceNodeClassName("vtkMRMLMarkupsFiducialNode")
    selection.SetActivePlaceNodeID(node.GetID())
    interaction = slicer.app.applicationLogic().GetInteractionNode()
    interaction.SetPlaceModePersistence(0)
    interaction.SetCurrentInteractionMode(interaction.Place)


def tipLabels(node):
    return [node.GetNthControlPointLabel(i) or f"P{i + 1}" for i in range(node.GetNumberOfControlPoints())] \
        if node is not None else []


# -- Working grid ----------------------------------------------------------------------------------------------

def _matrix(volumeNode):
    matrix = vtk.vtkMatrix4x4()
    volumeNode.GetIJKToRASMatrix(matrix)
    return np.array([[matrix.GetElement(r, c) for c in range(4)] for r in range(4)])


def _setMatrix(volumeNode, array):
    matrix = vtk.vtkMatrix4x4()
    for r in range(4):
        for c in range(4):
            matrix.SetElement(r, c, float(array[r][c]))
    volumeNode.SetIJKToRASMatrix(matrix)


def _segmentBounds(segmentationNode, segmentID):
    segment = segmentationNode.GetSegmentation().GetSegment(segmentID)
    bounds = [0.0] * 6
    segment.GetBounds(bounds)
    return bounds


class WorkingGrid:
    """Axis-aligned grid (RAS directions) with isotropic voxels; arrays are (k, j, i) like arrayFromVolume."""

    def __init__(self, bounds, spacingMM, name="CBCT planning grid"):
        spacing = float(spacingMM)
        lower = np.floor(np.array(bounds[0::2], dtype=float) / spacing) * spacing
        upper = np.ceil(np.array(bounds[1::2], dtype=float) / spacing) * spacing
        dims = np.round((upper - lower) / spacing).astype(int) + 1
        while float(np.prod(dims)) > MAX_GRID_VOXELS:
            spacing *= 1.25
            lower = np.floor(np.array(bounds[0::2], dtype=float) / spacing) * spacing
            upper = np.ceil(np.array(bounds[1::2], dtype=float) / spacing) * spacing
            dims = np.round((upper - lower) / spacing).astype(int) + 1
        self.spacingMM = spacing
        self.spacing = (spacing, spacing, spacing)
        self.shape = (int(dims[2]), int(dims[1]), int(dims[0]))
        self.voxelML = spacing ** 3 / 1000.0
        image = vtk.vtkImageData()
        image.SetDimensions(int(dims[0]), int(dims[1]), int(dims[2]))
        image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
        image.GetPointData().GetScalars().Fill(0)
        self.node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", name)
        self.node.SetHideFromEditors(True)
        self.node.SetAttribute("Taranis.Role", "PlanningGrid")
        self.node.SetSpacing(spacing, spacing, spacing)
        self.node.SetOrigin(*lower)
        self.node.SetAndObserveImageData(image)
        self.ijkToRas = _matrix(self.node)
        self.rasToIjk = np.linalg.inv(self.ijkToRas)

    def close(self):
        if self.node is not None and self.node.GetScene() is not None:
            slicer.mrmlScene.RemoveNode(self.node)
        self.node = None

    def toKji(self, ras):
        ijk = self.rasToIjk @ np.array([ras[0], ras[1], ras[2], 1.0])
        return (ijk[2], ijk[1], ijk[0])

    def toRas(self, kji):
        kji = np.atleast_2d(np.asarray(kji, dtype=float))
        ijk = np.column_stack([kji[:, 2], kji[:, 1], kji[:, 0], np.ones(len(kji))])
        return (self.ijkToRas @ ijk.T)[:3].T

    def inside(self, kji):
        return all(-0.5 <= value <= size - 0.5 for value, size in zip(kji, self.shape))

    def resample(self, volumeNode, dtype=np.float32):
        """The volume (with its transforms) resampled onto this grid (linear interpolation, 0 outside)."""
        resampled = slicer.vtkSlicerVolumesLogic().ResampleVolumeToReferenceVolume(volumeNode, self.node)
        try:
            return np.array(slicer.util.arrayFromVolume(resampled), dtype=dtype)
        finally:
            if resampled is not None and resampled.GetScene() is not None:
                slicer.mrmlScene.RemoveNode(resampled)

    def segment(self, segmentationNode, segmentID):
        return np.asarray(slicer.util.arrayFromSegmentBinaryLabelmap(segmentationNode, segmentID, self.node)) > 0

    def temporaryVolume(self, array, name="CBCT planning temporary"):
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", name)
        node.SetHideFromEditors(True)
        node.SetAttribute("Taranis.Role", "PlanningTemporary")
        _setMatrix(node, self.ijkToRas)
        slicer.util.updateVolumeFromArray(node, np.ascontiguousarray(array))
        _setMatrix(node, self.ijkToRas)
        return node


def fieldOfViewOnGrid(cbctNode, grid):
    """(trusted, inside) bool masks on the grid: inside the CBCT volume and its reconstructed field of view eroded by
    FOV_EROSION_MM, and not eroded. Found on a subsampled copy of the CBCT, resampled with the CBCT's transforms."""
    array = slicer.util.arrayFromVolume(cbctNode)
    step = max(1, int(round(max(array.shape) / FOV_SAMPLE_VOXELS)))
    sample = np.asarray(array[::step, ::step, ::step])
    fov = V.fieldOfViewMask(sample).astype(np.float32)
    node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "CBCT field of view")
    node.SetHideFromEditors(True)
    try:
        slicer.util.updateVolumeFromArray(node, fov)
        matrix = _matrix(cbctNode) @ np.diag([step, step, step, 1.0])
        _setMatrix(node, matrix)
        node.SetAndObserveTransformNodeID(cbctNode.GetTransformNodeID())
        mask = grid.resample(node) >= 0.5
    finally:
        slicer.mrmlScene.RemoveNode(node)
    return V.erodeMM(mask, V.FOV_EROSION_MM, grid.spacing), mask


# -- Arterial tree segmentation and centerline model ----------------------------------------------------------

def treeNode(store, create=False):
    node = store.GetNodeReference(REF_PLANNING_TREE)
    if node is None and create:
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", TREE_NAME)
        node.CreateDefaultDisplayNodes()
        store.SetNodeReferenceID(REF_PLANNING_TREE, node.GetID())
    return node


def treeSegmentID(node, create=False):
    """The arterial tree segment (the search mask, shown for review, is a second segment of the same node)."""
    segmentation = node.GetSegmentation()
    ids = [i for i in segmentIDs(node) if segmentation.GetSegment(i).GetName() != SEARCH_NAME]
    if ids:
        return ids[0]
    if not create:
        return None
    segmentID = node.GetSegmentation().AddEmptySegment("", TREE_NAME, TREE_COLOR)
    setSegmentRole(node, segmentID, W.SEGMENT_VESSELS, recolor=False)
    return segmentID


def writeTree(store, mask, grid):
    node = treeNode(store, create=True)
    node.SetReferenceImageGeometryParameterFromVolumeNode(grid.node)
    segmentID = treeSegmentID(node, create=True)
    slicer.util.updateSegmentBinaryLabelmapFromArray(mask.astype(np.uint8), node, segmentID, grid.node)
    node.CreateClosedSurfaceRepresentation()
    display = node.GetDisplayNode()
    if display is not None:
        display.SetVisibility3D(False)   # 3D: the centerlines (red lines) over the MIP; the segment in the slice views
        display.SetOpacity2DFill(0.25)
    return node


def writeSearchMask(store, mask, grid, visible=False):
    """The region the arteries were searched in, as a segment of the tree node (hidden unless asked for)."""
    node = treeNode(store, create=True)
    segmentation = node.GetSegmentation()
    segmentID = segmentation.GetSegmentIdBySegmentName(SEARCH_NAME)
    if not segmentID:
        segmentID = segmentation.AddEmptySegment("", SEARCH_NAME, SEARCH_COLOR)
    slicer.util.updateSegmentBinaryLabelmapFromArray(mask.astype(np.uint8), node, segmentID, grid.node)
    setSearchMaskVisible(store, visible)


def setSearchMaskVisible(store, visible):
    node = treeNode(store)
    if node is None or node.GetDisplayNode() is None:
        return False
    segmentID = node.GetSegmentation().GetSegmentIdBySegmentName(SEARCH_NAME)
    if not segmentID:
        return False
    display = node.GetDisplayNode()
    display.SetSegmentVisibility(segmentID, bool(visible))
    display.SetSegmentOpacity3D(segmentID, 0.08)
    display.SetSegmentOpacity2DFill(segmentID, 0.15)
    return True


def readTree(store, grid):
    node = treeNode(store)
    segmentID = treeSegmentID(node) if node is not None else None
    if segmentID is None:
        return None
    mask = grid.segment(node, segmentID)
    return mask if mask.any() else None


def updateCenterlineModel(store, tree, grid, highlight=None):
    """Model of the centerlines (lines child -> parent, RAS); point scalar 'Position': number of the planned tip
    whose downstream part contains the point (0: none)."""
    child = np.flatnonzero(tree.parent >= 0)
    points = vtk.vtkPoints()
    ras = grid.toRas(tree.points)
    points.SetNumberOfPoints(tree.count)
    for index, position in enumerate(ras):
        points.SetPoint(index, *position)
    lines = vtk.vtkCellArray()
    for c in child:
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, int(c))
        line.GetPointIds().SetId(1, int(tree.parent[c]))
        lines.InsertNextCell(line)
    polyData = vtk.vtkPolyData()
    polyData.SetPoints(points)
    polyData.SetLines(lines)
    radius = vtk.vtkFloatArray()
    radius.SetName("Radius")
    position = vtk.vtkFloatArray()
    position.SetName("Position")
    values = np.zeros(tree.count) if highlight is None else np.asarray(highlight, dtype=float)
    for index in range(tree.count):
        radius.InsertNextValue(float(tree.radius[index]))
        position.InsertNextValue(float(values[index]))
    polyData.GetPointData().AddArray(radius)
    polyData.GetPointData().AddArray(position)
    node = store.GetNodeReference(REF_PLANNING_CENTERLINES)
    if node is None:
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "CBCT centerlines")
        node.CreateDefaultDisplayNodes()
        store.SetNodeReferenceID(REF_PLANNING_CENTERLINES, node.GetID())
    node.SetAndObservePolyData(polyData)
    display = node.GetDisplayNode()
    display.SetColor(*CENTERLINE_COLOR)
    display.SetLineWidth(CENTERLINE_WIDTH)
    display.SetOpacity(CENTERLINE_OPACITY)
    display.SetVisibility2D(True)
    display.SetActiveScalarName("Position")
    display.SetScalarVisibility(bool(np.any(values > 0)))
    colorNode = slicer.mrmlScene.GetFirstNodeByName("GenericColors")
    if colorNode is not None:
        display.SetAndObserveColorNodeID(colorNode.GetID())
        display.SetScalarRangeFlag(slicer.vtkMRMLDisplayNode.UseColorNodeScalarRange)
    overlayCenterlines(store)
    return node


# The centerlines are drawn in the 3D view as an overlay: in a renderer layer above the scene (same camera), so
# they stay visible over the dark vessels of the MIP and inside the segments instead of being hidden by them.
def _overlays():
    """{render window id: entry}, kept on the slicer module so a module reload finds (and reuses) its renderers."""
    if not hasattr(slicer, "_cbctPlanningOverlays"):
        slicer._cbctPlanningOverlays = {}
    return slicer._cbctPlanningOverlays


def overlayCenterlines(store):
    layoutManager = slicer.app.layoutManager()
    node = store.GetNodeReference(REF_PLANNING_CENTERLINES) if store is not None else None
    if layoutManager is None or layoutManager.threeDViewCount == 0:
        return
    widget = threeDWidget()
    if widget is None:
        return
    view = widget.threeDView()
    renderWindow = view.renderWindow()
    main = renderWindow.GetRenderers().GetFirstRenderer()
    entry = _overlays().get(id(renderWindow))
    if entry is None or entry["window"] is not renderWindow:
        renderer = vtk.vtkRenderer()
        renderer.SetLayer(max(1, renderWindow.GetNumberOfLayers()))
        renderer.SetPreserveColorBuffer(True)   # draw over the scene, never clear it (else the view is blank)
        renderer.SetPreserveDepthBuffer(False)
        renderer.InteractiveOff()
        renderer.SetActiveCamera(main.GetActiveCamera())
        renderWindow.SetNumberOfLayers(renderer.GetLayer() + 1)
        renderWindow.AddRenderer(renderer)
        mapper = vtk.vtkPolyDataMapper()
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.PickableOff()
        renderer.AddActor(actor)
        entry = dict(window=renderWindow, renderer=renderer, mapper=mapper, actor=actor)
        _overlays()[id(renderWindow)] = entry
    entry["renderer"].SetActiveCamera(main.GetActiveCamera())
    actor, mapper = entry["actor"], entry["mapper"]
    polyData = node.GetPolyData() if node is not None else None
    display = node.GetDisplayNode() if node is not None else None
    if polyData is None or display is None or not display.GetVisibility() or not settings(store)["show3DVessels"]:
        actor.VisibilityOff()
        view.scheduleRender()
        return
    display.SetVisibility3D(False)   # drawn by the overlay instead (the model stays in the slice views)
    mapper.SetInputData(polyData)
    mapper.ScalarVisibilityOff()
    paths = _tipPaths(polyData, store)
    shaded = settings(store).get("dsaMode") == DSA_MODE_COMPOSITE
    _styleLines(actor, CENTERLINE_COLOR_SHADED if shaded else CENTERLINE_COLOR,
                CENTERLINE_OPACITY_DIMMED if paths else CENTERLINE_OPACITY)
    actor.VisibilityOn()
    # one actor per planned tip: its downstream path, blue to turquoise
    for old in entry.get("paths", []):
        entry["renderer"].RemoveActor(old)
    entry["paths"] = []
    for label, pathData in paths:
        pathMapper = vtk.vtkPolyDataMapper()
        pathMapper.SetInputData(pathData)
        pathMapper.ScalarVisibilityOff()
        pathActor = vtk.vtkActor()
        pathActor.SetMapper(pathMapper)
        pathActor.PickableOff()
        _styleLines(pathActor, pathColor(label), CENTERLINE_OPACITY)
        entry["renderer"].AddActor(pathActor)
        entry["paths"].append(pathActor)
    view.scheduleRender()


def _styleLines(actor, color, opacity):
    properties = actor.GetProperty()
    properties.SetColor(*color)
    properties.SetLineWidth(CENTERLINE_WIDTH)
    properties.SetOpacity(opacity)
    properties.SetAmbient(1.0)
    properties.SetDiffuse(0.0)
    properties.LightingOff()


def pathColor(label):
    """A random but stable colour per tip label, between turquoise and blue."""
    import colorsys
    import zlib
    value = zlib.crc32(str(label).encode("utf-8"))
    hue = 0.47 + 0.18 * ((value % 1000) / 999.0)           # turquoise (0.47) .. blue (0.65)
    saturation = 0.75 + 0.25 * (((value // 1000) % 100) / 99.0)
    return colorsys.hsv_to_rgb(hue, saturation, 0.85)


def _tipPaths(polyData, store):
    """[(tip label, polydata)] of the centerline segments downstream of each predicted tip ('Position' array)."""
    from vtk.util.numpy_support import vtk_to_numpy
    array = polyData.GetPointData().GetArray("Position")
    lines = polyData.GetLines()
    if array is None or lines is None or array.GetRange()[1] <= 0:
        return []
    position = vtk_to_numpy(array).astype(int)
    cells = vtk_to_numpy(lines.GetData()).reshape(-1, 3)[:, 1:]   # [2, a, b] per line
    results = planningResults(store) or {}
    labels = [p.get("label", f"P{i + 1}") for i, p in enumerate(results.get("positions", []))]
    paths = []
    for number in sorted(set(position[position > 0].tolist())):
        selected = cells[(position[cells[:, 0]] == number) & (position[cells[:, 1]] == number)]
        if not len(selected):
            continue
        cellArray = vtk.vtkCellArray()
        for a, b in selected:
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, int(a))
            line.GetPointIds().SetId(1, int(b))
            cellArray.InsertNextCell(line)
        data = vtk.vtkPolyData()
        data.SetPoints(polyData.GetPoints())
        data.SetLines(cellArray)
        radius = polyData.GetPointData().GetArray("Radius")
        if radius is not None:
            data.GetPointData().AddArray(radius)
        paths.append((labels[number - 1] if number - 1 < len(labels) else f"P{number}", data))
    return paths


# -- Session ---------------------------------------------------------------------------------------------------

class PlanningSession:
    """CBCT, liver, field of view, arterial tree and territory maps on one working grid. Kept by the module widget
    between the steps; call close() to remove the temporary grid volume."""

    def __init__(self, cbctNode, segmentationNode, liverSegmentID, spacingMM, store, progress=None):
        self.cbctNode = cbctNode
        self.segmentationNode = segmentationNode
        self.liverSegmentID = liverSegmentID
        self.store = store
        self.progress = progress or (lambda text: None)
        injection = controlPoints(markupsNode(store, REF_PLANNING_INJECTION))
        bounds = _segmentBounds(segmentationNode, liverSegmentID)
        if bounds[0] > bounds[1]:
            raise ValueError("The whole-liver segment is empty.")
        bounds = [bounds[0] - GRID_MARGIN_MM, bounds[1] + GRID_MARGIN_MM, bounds[2] - GRID_MARGIN_MM,
                  bounds[3] + GRID_MARGIN_MM, bounds[4] - GRID_MARGIN_MM, bounds[5] + GRID_MARGIN_MM]
        for point in injection:
            for axis in range(3):
                bounds[2 * axis] = min(bounds[2 * axis], point[axis] - INJECTION_MARGIN_MM)
                bounds[2 * axis + 1] = max(bounds[2 * axis + 1], point[axis] + INJECTION_MARGIN_MM)
        self.grid = WorkingGrid(bounds, spacingMM)
        self.key = sessionKey(cbctNode, segmentationNode, liverSegmentID, spacingMM, store)
        self.progress("Resampling the CBCT and the whole liver")
        self.values = self.grid.resample(cbctNode)
        self.liver = self.grid.segment(segmentationNode, liverSegmentID)
        tumours = [mask for _, mask in self.tumours().values()]
        self.parenchyma = self.liver.copy()
        for tumour in tumours:
            self.parenchyma &= ~tumour
        # tumours are liver tissue for the territories (a whole-liver segment drawn without them would leave holes)
        near = V.dilateMM(self.liver, TUMOUR_MARGIN_MM, self.grid.spacing)
        for tumour in tumours:
            self.liver |= tumour & near
        self.progress("Finding the CBCT field of view")
        self.fov, inside = fieldOfViewOnGrid(cbctNode, self.grid)
        # outside the CBCT (resampling fills 0) and outside its reconstructed cylinder (scanner fill value): air, so
        # that smoothing and the local background do not mix a false 0 (soft tissue) into the edge of the image
        self.values[~inside] = PAD_VALUE
        del inside
        self.liverCoverage = V.coverageFraction(self.liver, self.fov)
        self.statistics = V.liverStatistics(self.values, self.parenchyma if self.parenchyma.any() else self.liver,
                                            self.fov)
        self.airThreshold = V.airThreshold(self.values, self.statistics, self.fov)
        self.denoised = None
        self.contrast = None
        self.seeds = None
        self.tube = None
        self.region = None
        self.metalML = None
        self.tree = None
        self.treeMask = None
        self.treeKey = None
        self._maps = {}

    def close(self):
        self.grid.close()
        self._maps = {}

    @property
    def spacing(self):
        return self.grid.spacing

    def injectionKji(self):
        points = controlPoints(markupsNode(self.store, REF_PLANNING_INJECTION))
        if not points:
            raise ValueError("Place the injection point first: the catheter tip during the CBCT, inside the "
                             "contrast-filled artery.")
        kji = self.grid.toKji(points[0])
        if not self.grid.inside(kji):
            raise ValueError("The injection point is outside the working grid: is it on the CBCT?")
        return kji

    # -- Tree --

    def searchRegion(self):
        return V.searchRegion(self.values, self.parenchyma if self.parenchyma.any() else self.liver, self.spacing,
                              self.statistics, self.injectionKji(), self.fov)

    def localContrast(self):
        """Denoised CBCT and its contrast against the surrounding liver (computed once per session)."""
        if self.contrast is None:
            self.progress("Denoising the CBCT (Gaussian)")
            self.denoised = V.denoise(self.values, self.spacing)
            self.progress(f"Local liver background ({V.LOCAL_WINDOW_MM:g} mm, liver voxels only)")
            self.contrast = V.localContrast(self.denoised, self.liver, self.spacing, fov=self.fov)
        return self.contrast

    def extractTree(self, vesselSD=V.VESSEL_SD, growSD=V.GROW_SD, tubularOnly=True, showSearchMask=False):
        """Arterial tree: hysteresis at vesselSD / growSD local SD above the local liver background, inside the
        search mask (without metal)."""
        injection = self.injectionKji()
        self.progress("Search mask (liver, around the injection point, no air / lung)")
        region = self.searchRegion()
        contrast = self.localContrast()
        self.tube = None
        if tubularOnly:
            self.progress("Tube-shape filter (Hessian, several scales)")
            self.tube = V.tubeShape(contrast.z, self.spacing, region & (contrast.z >= min(growSD, vesselSD)),
                                    progress=self.progress)
        self.progress("Arterial tree from the injection point")
        segment = V.arterialTree(self.denoised, contrast, region, injection, self.spacing, self.parenchyma, vesselSD,
                                 growSD, self.tube, voxelML=self.grid.voxelML)
        self.region = region
        self.metalML = segment.metalML
        self.seeds = segment.seeds
        mask = segment.mask
        writeTree(self.store, mask, self.grid)
        writeSearchMask(self.store, region, self.grid, showSearchMask)
        self.treeMask = mask
        return self.buildCenterlines(mask)

    def buildCenterlines(self, mask=None):
        """Centerlines of the arterial tree segment (as edited by the user)."""
        if mask is None:
            mask = readTree(self.store, self.grid)
            if mask is None:
                raise ValueError("No arterial tree yet: extract it first.")
        self.progress("Centerlines (3D thinning)")
        skeleton = V.skeletonize(mask)
        self.progress("Orienting the tree from the injection point")
        tree = V.buildTree(skeleton, self.spacing, self.injectionKji(), mask)
        if self.seeds is not None:
            # twigs grown into noise (no seed voxel on the way to their tip) are pruned, with their part of the segment
            self.progress("Pruning branches without a contrast-filled vessel")
            keep = V.pruneTwigs(tree, self.seeds[tuple(tree.points.T)])
            if not keep.all():
                mask = V.maskOfKept(mask, tree, keep)
                pruned = np.zeros(mask.shape, bool)
                pruned[tuple(tree.points[keep].T)] = True
                tree = V.buildTree(pruned, self.spacing, self.injectionKji(), mask)
                writeTree(self.store, mask, self.grid)
        self.tree = tree
        self.treeMask = mask
        self.treeKey = _treeKey(self.store)
        self._maps = {}
        updateCenterlineModel(self.store, self.tree, self.grid)
        return self.tree

    def ensureTree(self):
        if self.tree is None or self.treeKey != _treeKey(self.store):
            self.buildCenterlines()
        return self.tree

    def territoryLimit(self):
        """mm a territory may extend from its own branches (None: no limit), from the settings."""
        values = settings(self.store)
        return float(values["supplyDistanceMM"]) if values.get("limitTerritories") else None

    def territoryMap(self, rule):
        tree = self.ensureTree()
        if rule not in self._maps:
            self.progress("Assigning the liver to the branches" + (" (shortest paths inside the liver)"
                                                                   if rule == RULE_GEODESIC else ""))
            self._maps[rule] = (V.territoryGeodesic(tree, self.liver) if rule == RULE_GEODESIC
                                else V.territoryEuclidean(tree, self.liver))
        self._maps[rule].limitMM = self.territoryLimit()
        return self._maps[rule]

    def supplyDistance(self):
        """Distance of every liver voxel to the visible tree (mm), cached with the tree."""
        tree = self.ensureTree()
        if getattr(self, "_distanceKey", None) != self.treeKey or getattr(self, "_distance", None) is None:
            self._distance = V.treeDistance(tree, self.liver)
            self._distanceKey = self.treeKey
        return self._distance

    def suggestSupplyDistance(self):
        return V.suggestSupplyDistance(self.supplyDistance(), self.liver)

    def unsuppliedPercent(self, distanceMM):
        """Share of the whole liver farther than distanceMM from every visible centerline."""
        distance = self.supplyDistance()
        total = int(np.count_nonzero(self.liver))
        return 100.0 * np.count_nonzero(self.liver & (distance > distanceMM)) / total if total else 0.0

    # -- Case segments --

    def segmentMasks(self, roles):
        """{segment ID: (name, mask on the grid)} of the accepted segments with these roles."""
        result = {}
        segmentation = self.segmentationNode.GetSegmentation()
        for segmentID in segmentIDs(self.segmentationNode):
            segment = segmentation.GetSegment(segmentID)
            if isCandidate(segment) or segmentRole(self.segmentationNode, segmentID) not in roles:
                continue
            result[segmentID] = (segment.GetName(), self.grid.segment(self.segmentationNode, segmentID))
        return result

    def tumours(self):
        return self.segmentMasks((W.SEGMENT_TUMOR, W.SEGMENT_VIABLE))

    # -- Prediction --

    def snap(self, tipsNode):
        """[(label, ras, index or None, distance mm)] of the planned tips."""
        tree = self.ensureTree()
        result = []
        for number, (label, ras) in enumerate(zip(tipLabels(tipsNode), controlPoints(tipsNode))):
            kji = self.grid.toKji(ras)
            index, distance = V.snapTip(tree, kji)
            result.append((label or f"P{number + 1}", ras, index, distance))
        return result

    def feeders(self, tumourSegmentID, rule):
        tree = self.ensureTree()
        tumours = self.tumours()
        if tumourSegmentID not in tumours:
            raise ValueError("Select a tumour segment.")
        name, mask = tumours[tumourSegmentID]
        others = [m for segmentID, (_, m) in tumours.items() if segmentID != tumourSegmentID]
        feeders = V.tumourFeeders(tree, mask)
        if feeders is None:
            return name, []
        candidates = V.candidatePositions(tree, self.territoryMap(rule), mask & self.liver, self.liver,
                                          self.grid.voxelML, others, feeders=feeders)
        for candidate in candidates:
            candidate["ras"] = [float(v) for v in self.grid.toRas(tree.points[candidate["index"]])[0]]
        return name, candidates


def _treeKey(store):
    node = treeNode(store)
    return json.dumps(segmentsVoxels(node)) if node is not None else None


def sessionKey(cbctNode, segmentationNode, liverSegmentID, spacingMM, store):
    """A session stays valid while the CBCT and its position, the whole liver, the working resolution and the
    injection point (it sets the grid bounds) do not change."""
    if cbctNode is None or segmentationNode is None or not liverSegmentID:
        return None
    liver = [x.voxels for x in segmentInfos(segmentationNode) if x.segmentID == liverSegmentID]
    injection = [[round(v, 1) for v in p] for p in controlPoints(markupsNode(store, REF_PLANNING_INJECTION))]
    advanced = applyAdvanced(store)   # tuned preprocessing parameters: a change starts a new session
    return json.dumps([cbctNode.GetID(), _placement(cbctNode), segmentationNode.GetID(), liverSegmentID, liver,
                       float(spacingMM), injection, advanced], default=str, sort_keys=True)


def applyAdvanced(store):
    """Set the preprocessing / extraction parameters tuned in Advanced settings (stored with the case) in the
    vascular library; returns them."""
    advanced = settings(store).get("advanced") or {}
    V.setParameters(advanced)
    return advanced


# -- Filtered images (for review) -----------------------------------------------------------------------------

FILTERED_ATTRIBUTE = "CBCTPlanning.Filtered"
FILTERED_IMAGES = [
    # key, name, (window, level) or None, colour table
    ("contrast", "CBCT local contrast (SD above the liver around)", (12.0, 3.0), "vtkMRMLColorTableNodeGrey"),
    ("denoised", "CBCT denoised", None, "vtkMRMLColorTableNodeGrey"),
    ("background", "CBCT local liver background", None, "vtkMRMLColorTableNodeGrey"),
    ("tube", "CBCT tube shape (0-1)", (1.0, 0.5), "vtkMRMLColorTableNodeGrey"),
]


def exportFilteredImages(session):
    """The intermediate images of the tree extraction as ordinary volumes (on the working grid), replaced on every
    export: the local contrast z (what the thresholds are applied to), the denoised CBCT, the local liver background
    and the tube-shape score. Returns the contrast volume node."""
    contrast = session.localContrast()
    arrays = {"contrast": contrast.z, "denoised": session.denoised, "background": contrast.background,
              "tube": session.tube}
    first = None
    for key, name, windowLevel, colorID in FILTERED_IMAGES:
        array = arrays.get(key)
        node = next((n for n in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
                     if n.GetAttribute(FILTERED_ATTRIBUTE) == key), None)
        if array is None:
            continue
        if node is None:
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", name)
            node.SetAttribute(FILTERED_ATTRIBUTE, key)
            node.SetAttribute("Taranis.Role", "PlanningFiltered")   # never offered as a case image
        _setMatrix(node, session.grid.ijkToRas)
        slicer.util.updateVolumeFromArray(node, np.ascontiguousarray(array, dtype=np.float32))
        _setMatrix(node, session.grid.ijkToRas)
        node.CreateDefaultDisplayNodes()
        display = node.GetDisplayNode()
        display.SetAndObserveColorNodeID(colorID)
        if windowLevel is not None:
            display.AutoWindowLevelOff()
            display.SetWindowLevel(*windowLevel)
        first = first or node
    if session.region is None:
        try:
            session.region = session.searchRegion()
        except ValueError:
            pass   # no injection point yet: no search mask
    if session.region is not None:
        writeSearchMask(session.store, session.region, session.grid, True)
    return first


# -- Writing candidates into the case segmentation ---------------------------------------------------------------

def _caseGrid(segmentationNode, session):
    case = TaranisCase.find()
    reference = case.primaryVolume() if case is not None else None
    if reference is None:
        reference = session.grid.node
    volumes = segtools.caseVolumes(case) if case is not None else ()
    return segtools.SegmentGrid(segmentationNode, reference, volumes, tight=True)


def _toCaseGrid(session, mask, caseGrid):
    node = session.grid.temporaryVolume(mask.astype(np.float32))
    try:
        resampled = slicer.vtkSlicerVolumesLogic().ResampleVolumeToReferenceVolume(node, caseGrid.reference)
        try:
            return np.asarray(slicer.util.arrayFromVolume(resampled)) >= 0.5
        finally:
            slicer.mrmlScene.RemoveNode(resampled)
    finally:
        slicer.mrmlScene.RemoveNode(node)


def _liverOnCase(caseGrid, session):
    """Whole liver and the tumours on the case grid (the territory domain, as in the session)."""
    liver = caseGrid.mask(session.liverSegmentID).copy()
    tumours = caseGrid.ids(W.SEGMENT_TUMOR) + caseGrid.ids(W.SEGMENT_VIABLE)
    return liver | caseGrid.union(tumours) if tumours else liver


def _removeOldCandidates(segmentationNode, segmentIDList):
    segmentation = segmentationNode.GetSegmentation()
    for segmentID in segmentIDList or []:
        segment = segmentation.GetSegment(segmentID)
        if segment is not None and isCandidate(segment):
            segmentation.RemoveSegment(segmentID)


def predictTerritories(session, rule=RULE_EUCLIDEAN, subtractNested=True, snapMarkers=True, uptakeNode=None,
                       uptakePercent=DEFAULT_UPTAKE_PERCENT):
    """Territory of every planned tip, written as perfused-volume candidates; stores and returns the results."""
    store = session.store
    tipsNode = markupsNode(store, REF_PLANNING_TIPS)
    if tipsNode is None or tipsNode.GetNumberOfControlPoints() == 0:
        raise ValueError("Place at least one planned catheter tip on the arterial tree.")
    tree = session.ensureTree()
    tmap = session.territoryMap(rule)
    tips = session.snap(tipsNode)
    if snapMarkers:
        wasModifying = tipsNode.StartModify()
        try:
            for number, (_, _, index, _) in enumerate(tips):
                if index is not None:
                    tipsNode.SetNthControlPointPositionWorld(number, *session.grid.toRas(tree.points[index])[0])
        finally:
            tipsNode.EndModify(wasModifying)
    indices = [index for _, _, index, _ in tips]
    session.progress("Territories of the planned tips")
    masks = [tmap.territory(tree, index) if index is not None else None for index in indices]
    raw = [m for m in masks if m is not None]
    usedIndices = [i for i in indices if i is not None]
    if subtractNested and len(raw) > 1:
        raw = V.subtractNested(raw, usedIndices, tree)
    iterator = iter(raw)
    masks = [next(iterator) if m is not None else None for m in masks]
    tumours = session.tumours()
    positions, highlight = [], np.zeros(tree.count)
    for number, ((label, ras, index, distance), mask) in enumerate(zip(tips, masks)):
        position = dict(label=label, ras=[round(float(v), 2) for v in ras], snapped=index is not None,
                        snapDistanceMM=round(distance, 1))
        if index is not None:
            highlight[tree.downstream(index) & (highlight == 0)] = number + 1
            position["territoryML"] = float(np.count_nonzero(mask) * session.grid.voxelML)
            position["upstreamMM"] = float(tree.distance[index])
            position["radiusMM"] = float(tree.radius[index])
            position["extrahepatic"] = []
            for finding in V.extrahepaticBranches(tree, session.liver, index):
                finding["ras"] = [round(float(v), 1) for v in session.grid.toRas(tree.points[finding["deepest"]])[0]]
                position["extrahepatic"].append({k: finding[k] for k in ("distanceMM", "lengthMM", "ras")})
            position["tumours"] = []
            for name, tumour in tumours.values():
                total = int(np.count_nonzero(tumour))
                if total:
                    position["tumours"].append([name, 100.0 * np.count_nonzero(tumour & mask) / total])
            position["outsideFovPercent"] = 100.0 * (1.0 - V.coverageFraction(mask, session.fov)) if mask.any() \
                else 0.0
        positions.append(position)
    updateCenterlineModel(store, tree, session.grid, highlight)

    union = np.zeros(session.grid.shape, bool)
    for mask in masks:
        if mask is not None:
            union |= mask
    overlaps = []
    for a in range(len(masks)):
        for b in range(a + 1, len(masks)):
            if masks[a] is not None and masks[b] is not None:
                volume = np.count_nonzero(masks[a] & masks[b]) * session.grid.voxelML
                if volume >= segtools.MIN_REPORT_ML:
                    overlaps.append([positions[a]["label"], positions[b]["label"], float(volume)])
    uncovered = []
    for name, tumour in tumours.values():
        total = int(np.count_nonzero(tumour))
        if total:
            percent = 100.0 * np.count_nonzero(tumour & union) / total
            if percent < W.TUMOUR_COVERAGE_LOW:
                uncovered.append([name, float(percent)])
    maa = compareWithUptake(session, union, uptakeNode, uptakePercent) if uptakeNode is not None else None

    # candidates in the case segmentation (previous candidates of the planner are replaced)
    previous = planningResults(store) or {}
    _removeOldCandidates(session.segmentationNode, [i for i in previous.get("segmentIDs", [])
                                                    if i not in previous.get("enhancementIDs", [])])
    segmentIDList = []
    session.progress("Writing the territory candidates")
    with _caseGrid(session.segmentationNode, session) as caseGrid:
        liverOnCase = _liverOnCase(caseGrid, session)
        for position, mask in zip(positions, masks):
            if mask is None or not mask.any():
                continue
            candidate = _toCaseGrid(session, mask, caseGrid) & liverOnCase
            name = segtools.uniqueName(session.segmentationNode,
                                       f"{TERRITORY_STEM} {position['label']} ({RULE_SHORT[rule]} - evaluate)")
            segmentID = caseGrid.add(name, W.SEGMENT_PERFUSED, candidate, candidate=True)
            position["segmentID"] = segmentID
            segmentIDList.append(segmentID)
    enhancement = previous.get("enhancement", [])
    limit = session.territoryLimit()
    unsupplied = session.unsuppliedPercent(float(settings(store).get("supplyDistanceMM", V.SUPPLY_DISTANCE_MM)))
    results = dict(key=planningInputsKey(store, session.segmentationNode, session.cbctNode), rule=rule,
                   territoryLimitMM=limit, unsuppliedPercent=unsupplied,
                   subtractNested=bool(subtractNested), liverCoverage=session.liverCoverage, loops=tree.loops,
                   treeLengthMM=tree.lengthMM(), positions=positions, overlaps=overlaps, uncovered=uncovered,
                   maa=maa, segmentIDs=segmentIDList + [e["segmentID"] for e in enhancement if e.get("segmentID")],
                   enhancementIDs=[e["segmentID"] for e in enhancement if e.get("segmentID")],
                   enhancement=enhancement, spacingMM=session.grid.spacingMM,
                   date=datetime.datetime.now().isoformat(timespec="seconds"))
    store.SetParameter(P_PLANNING_RESULTS, json.dumps(results))
    return results


def compareWithUptake(session, union, uptakeNode, percent=DEFAULT_UPTAKE_PERCENT):
    """Dice of the predicted territories with the MAA perfused volume (uptake >= percent of the maximum in the
    liver) and the fraction of the liver MAA counts inside them."""
    try:
        values = session.grid.resample(uptakeNode)
        parts = S.perfusedFromUptake(values, session.liver, percent,
                                     max(1, int(round(V.ENHANCEMENT_MIN_ML / session.grid.voxelML))))
    except Exception as e:
        logging.warning(f"CBCT planning: no comparison with '{uptakeNode.GetName()}': {e}")
        return None
    perfused = np.zeros(session.grid.shape, bool)
    for part in parts:
        perfused |= part
    return dict(image=uptakeNode.GetName(), percent=percent, dice=V.dice(union, perfused),
                countsFraction=V.countsInside(values, session.liver, union),
                perfusedML=float(np.count_nonzero(perfused) * session.grid.voxelML))


def enhancementPercentAuto(session, volumeNode):
    values = session.values if volumeNode is session.cbctNode else session.grid.resample(volumeNode)
    return V.autoEnhancementPercent(values, session.liver, session.fov)


def enhancementTerritory(session, volumeNode, percent):
    """Perfused-volume candidate from contrast enhancement of a selective / parenchymal CBCT."""
    values = session.values if volumeNode is session.cbctNode else session.grid.resample(volumeNode)
    threshold = V.thresholdFromPercent(values, session.liver, percent, session.fov)
    session.progress("Perfused volume from CBCT enhancement")
    vessels = session.treeMask if volumeNode is session.cbctNode else None
    mask = V.enhancementTerritory(values, session.liver, threshold, session.grid.voxelML, session.fov,
                                  vessels=vessels)
    if not mask.any():
        raise ValueError(f"No enhanced liver above {percent:g}% (threshold {threshold:.0f}): lower the threshold.")
    store = session.store
    with _caseGrid(session.segmentationNode, session) as caseGrid:
        candidate = _toCaseGrid(session, mask, caseGrid) & _liverOnCase(caseGrid, session)
        name = segtools.uniqueName(session.segmentationNode, f"{ENHANCEMENT_STEM} {percent:g}% - evaluate)")
        segmentID = caseGrid.add(name, W.SEGMENT_PERFUSED, candidate, candidate=True)
    volume = float(np.count_nonzero(mask) * session.grid.voxelML)
    liverML = float(np.count_nonzero(session.liver) * session.grid.voxelML)
    results = planningResults(store) or dict(positions=[], overlaps=[], uncovered=[], maa=None, segmentIDs=[],
                                             liverCoverage=session.liverCoverage)
    results.setdefault("enhancement", []).append(dict(image=volumeNode.GetName(), percent=percent,
                                                      threshold=threshold, volumeML=volume, segmentID=segmentID))
    results["segmentIDs"] = list(results.get("segmentIDs", [])) + [segmentID]
    results["enhancementIDs"] = list(results.get("enhancementIDs", [])) + [segmentID]
    results["key"] = planningInputsKey(store, session.segmentationNode, session.cbctNode)
    store.SetParameter(P_PLANNING_RESULTS, json.dumps(results))
    return segmentID, (f"Perfused volume from CBCT enhancement ≥ {percent:g}% ({threshold:.0f}): {volume:.0f} mL "
                       f"({100.0 * volume / liverML:.0f}% of the liver). Review it, then accept it in the "
                       "Segmentation step.")


def markReviewed(store):
    results = planningResults(store)
    if results:
        store.SetParameter(P_PLANNING_REVIEWED, results.get("key", ""))


# -- Layout ---------------------------------------------------------------------------------------------------

PLANNING_3D_ATTRIBUTE = "CBCTPlanning.3D"
CONTRAST_RENDER_ATTRIBUTE = "CBCTPlanning.ContrastRendering"
LIVER_3D_COLOR = (0.0, 0.6, 0.2)
SEGMENT_3D_OPACITY = {W.SEGMENT_LIVER: 0.03, W.SEGMENT_TUMOR: 0.15, W.SEGMENT_VIABLE: 0.15, W.SEGMENT_PERFUSED: 0.10}
CANDIDATE_3D_OPACITY = 0.18
DSA_START_PERCENTILE = 10.0      # the DSA-like MIP of the CBCT is white up to this percentile of the search mask
DSA_TOP_PERCENTILE = 99.0        # ... and black from this percentile of the search mask (densest contrast)
V_PAD = -1024.0                  # outside the search mask


THREED_VIEW_TAG = "CBCTPlanning3D"   # own 3D view node: other modules' layouts (e.g. the hub's) keep theirs


def threeDWidget():
    """The 3D widget of the planning layout (not simply the first one: other layouts' 3D views stay registered)."""
    layoutManager = slicer.app.layoutManager()
    if layoutManager is None:
        return None
    for index in range(layoutManager.threeDViewCount):
        widget = layoutManager.threeDWidget(index)
        node = widget.mrmlViewNode() if widget is not None else None
        if node is not None and node.GetSingletonTag() == THREED_VIEW_TAG:
            return widget
    return None


def threeDViewNode():
    widget = threeDWidget()
    return widget.mrmlViewNode() if widget is not None else None


def styleThreeDView(viewNode, negative=None):
    """White background (black in negative), orthographic projection, no box or axis labels."""
    if viewNode is None:
        return
    if negative is None:
        negative = bool(settings(storeNode()).get("negative", False))
    shade = 0.0 if negative else 1.0
    wasModifying = viewNode.StartModify()
    viewNode.SetBackgroundColor(shade, shade, shade)
    viewNode.SetBackgroundColor2(shade, shade, shade)
    viewNode.SetRenderMode(slicer.vtkMRMLViewNode.Orthographic)
    viewNode.SetBoxVisible(False)
    viewNode.SetAxisLabelsVisible(False)
    viewNode.EndModify(wasModifying)


def resetThreeDView(viewNode):
    """Centre the 3D view on what it shows, looking from anterior."""
    from .views import resetThreeDView as reset
    reset(viewNode, rotate=True)


def styleCaseSegments3D(segmentationNode, viewNode):
    """The whole liver (black), tumours and perfused volumes (territory candidates included) in the 3D view as
    wireframes, as in the Segmentation step, on an extra display node for this view only."""
    if segmentationNode is None or viewNode is None:
        return
    from .controller import TARANIS_CANDIDATE_TAG, segmentTag
    displayNode = None
    for index in range(segmentationNode.GetNumberOfDisplayNodes()):
        node = segmentationNode.GetNthDisplayNode(index)
        if node is not None and node.GetAttribute(PLANNING_3D_ATTRIBUTE):
            displayNode = node
    if displayNode is None:
        displayNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationDisplayNode", "CBCT planning segments 3D")
        displayNode.SetAttribute(PLANNING_3D_ATTRIBUTE, "1")
        segmentationNode.AddAndObserveDisplayNodeID(displayNode.GetID())
    segmentation = segmentationNode.GetSegmentation()
    wasModifying = displayNode.StartModify()
    displayNode.SetViewNodeIDs([viewNode.GetID()])
    displayNode.SetVisibility(True)
    displayNode.SetVisibility2D(False)
    displayNode.SetVisibility3D(True)
    displayNode.SetRepresentation(slicer.vtkMRMLDisplayNode.WireframeRepresentation)
    displayNode.SetAmbient(1.0)
    displayNode.SetDiffuse(0.0)
    displayNode.SetSpecular(0.0)
    displayNode.SetBackfaceCulling(False)
    displayNode.SetOpacity3D(1.0)
    for segmentID in segmentIDs(segmentationNode):
        segment = segmentation.GetSegment(segmentID)
        candidate = isCandidate(segment)
        role = segmentTag(segment, TARANIS_CANDIDATE_TAG) if candidate else segmentRole(segmentationNode, segmentID)
        visible = role in SEGMENT_3D_OPACITY and _roleShown(role)
        displayNode.SetSegmentVisibility(segmentID, True)
        displayNode.SetSegmentVisibility3D(segmentID, visible)
        if visible:
            displayNode.SetSegmentOpacity3D(segmentID, CANDIDATE_3D_OPACITY if candidate else SEGMENT_3D_OPACITY[role])
        if role == W.SEGMENT_LIVER:
            displayNode.SetSegmentOverrideColor(segmentID, *LIVER_3D_COLOR)
    displayNode.EndModify(wasModifying)
    segmentationNode.CreateClosedSurfaceRepresentation()
    # the segmentation's own display nodes stay out of this 3D view (slice views only)
    for index in range(segmentationNode.GetNumberOfDisplayNodes()):
        node = segmentationNode.GetNthDisplayNode(index)
        if node is None or node is displayNode or node.GetAttribute(PLANNING_3D_ATTRIBUTE):
            continue
        views = list(node.GetViewNodeIDs())
        if not views:
            node.SetVisibility3D(False)   # shown in every view: hide it in 3D (slice views unchanged)


def maskedNativeCbct(session):
    """(array, IJK-to-RAS matrix, transform node ID) of the original CBCT voxels inside the search mask, cropped to
    the mask, outside it V_PAD; at the CBCT's own resolution (the working grid is coarser and smoother). None when
    the CBCT is under a non-linear transform."""
    cbct = session.cbctNode
    parent = cbct.GetParentTransformNode()
    toWorld = vtk.vtkMatrix4x4()
    if parent is not None and not slicer.vtkMRMLTransformNode.GetMatrixTransformBetweenNodes(parent, None, toWorld):
        return None
    ijkToWorld = np.array([[toWorld.GetElement(r, c) for c in range(4)] for r in range(4)]) @ _matrix(cbct)
    # the search mask's bounding box in CBCT voxel indices
    points = np.argwhere(session.region)
    lower, upper = points.min(axis=0), points.max(axis=0)
    corners = np.array([[k, j, i] for k in (lower[0], upper[0]) for j in (lower[1], upper[1])
                        for i in (lower[2], upper[2])], dtype=float)
    ras = session.grid.toRas(corners)
    worldToIjk = np.linalg.inv(ijkToWorld)
    ijk = (worldToIjk @ np.column_stack([ras, np.ones(len(ras))]).T)[:3].T
    source = slicer.util.arrayFromVolume(cbct)           # (k, j, i)
    size = np.array(source.shape[::-1])                  # (i, j, k)
    low = np.clip(np.floor(ijk.min(axis=0)).astype(int) - 1, 0, size - 1)
    high = np.clip(np.ceil(ijk.max(axis=0)).astype(int) + 1, 0, size - 1)
    if np.any(high < low):
        return None
    crop = np.array(source[low[2]:high[2] + 1, low[1]:high[1] + 1, low[0]:high[0] + 1], dtype=np.float32)
    # every cropped voxel -> working grid voxel -> inside the search mask?
    gridFromIjk = session.grid.rasToIjk @ ijkToWorld
    shape = np.array(session.region.shape)               # (k, j, i)
    jj, ii = np.meshgrid(np.arange(low[1], high[1] + 1), np.arange(low[0], high[0] + 1), indexing="ij")
    for index, k in enumerate(range(low[2], high[2] + 1)):
        coordinates = np.stack([ii.ravel(), jj.ravel(), np.full(ii.size, k), np.ones(ii.size)])
        g = np.rint(gridFromIjk @ coordinates)[:3]       # (i, j, k) on the working grid
        valid = (g[0] >= 0) & (g[0] < shape[2]) & (g[1] >= 0) & (g[1] < shape[1]) & (g[2] >= 0) & (g[2] < shape[0])
        inside = np.zeros(ii.size, bool)
        inside[valid] = session.region[g[2, valid].astype(int), g[1, valid].astype(int), g[0, valid].astype(int)]
        plane = crop[index].reshape(-1)
        plane[~inside] = V_PAD
    matrix = _matrix(cbct).copy()
    matrix[:3, 3] = (matrix @ np.array([low[0], low[1], low[2], 1.0]))[:3]
    return crop, matrix, cbct.GetTransformNodeID()


DSA_MODE_MIP = "mip"
DSA_MODE_COMPOSITE = "composite"
SHADED_COLORS = ((1.0, 0.0, 60 / 255.0), (1.0, 0.0, 0.0), (1.0, 1.0, 1.0))   # opacity 0 / 0.5 / 1: #ff003c, #ff0000,
#                                          #ffffff; white already half way from the 0.5 point to the 1.0 point


def lookFromSettings(values):
    """The rendering choices of the 3D view (MIP / shaded windowing, mode, negative) from the stored settings."""
    keys = ("dsaStartPercentile", "dsaTopPercentile", "dsaMode", "shadedStartPercentile", "shadedMidPercentile",
            "shadedTopPercentile", "negative")
    return {key: values.get(key, DEFAULT_SETTINGS[key]) for key in keys}


def renderMaskedContrast(session, viewNode, look=None):
    """Maximum intensity projection of the original CBCT (its own voxels, not resampled) inside the search mask
    (nothing outside it) in the 3D view. look (lookFromSettings): mode "mip": grey on white like a subtracted
    angiogram (DSA), white up to the dsaStartPercentile, black from the dsaTopPercentile of the CBCT inside the search
    mask; mode "composite": shaded volume rendering, opacity 0 / 0.5 / 1 at the shadedStart / Mid / Top percentiles,
    #ff003c -> #ff0000 -> white as the opacity rises. negative: black background (MIP: white vessels; shaded: same colours). The masked copy is made once per search mask;
    the settings only change the transfer functions. The extracted tree and centerlines are drawn over
    it."""
    if viewNode is None:
        return None
    if session.region is None:
        try:
            session.region = session.searchRegion()   # e.g. a new session (settings changed) before extraction
        except ValueError:
            return None
    if not session.region.any():
        return None
    node = next((n for n in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
                 if n.GetAttribute(CONTRAST_RENDER_ATTRIBUTE)), None)
    if node is None:
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "CBCT in search mask")
        node.SetAttribute(CONTRAST_RENDER_ATTRIBUTE, "1")
        node.SetAttribute("Taranis.Role", "PlanningFiltered")   # never offered as a case image
    key = (id(session.region), session.key)
    if getattr(session, "renderedKey", None) == key and node.GetImageData() is not None:
        array = slicer.util.arrayFromVolume(node)
        inside = array[array > V_PAD]
        native = False
    else:
        native = maskedNativeCbct(session)
        session.renderedKey = key
    if native is False:
        pass
    elif native is not None:
        # the CBCT at its own voxel size (not resampled), cropped to the search mask, under its own transform
        array, matrix, transformID = native
        _setMatrix(node, matrix)
        slicer.util.updateVolumeFromArray(node, array)
        _setMatrix(node, matrix)
        node.SetAndObserveTransformNodeID(transformID)
        inside = array[array > V_PAD]
    else:
        array = np.where(session.region, session.values, V_PAD).astype(np.float32)
        _setMatrix(node, session.grid.ijkToRas)
        slicer.util.updateVolumeFromArray(node, array)
        _setMatrix(node, session.grid.ijkToRas)
        node.SetAndObserveTransformNodeID(None)
        inside = session.values[session.region]
    look = dict(lookFromSettings({}), **(look or {}))
    composite = look["dsaMode"] == DSA_MODE_COMPOSITE
    low, high = ((look["shadedStartPercentile"], look["shadedTopPercentile"]) if composite
                 else (look["dsaStartPercentile"], look["dsaTopPercentile"]))
    start = float(np.percentile(inside, low)) if inside.size else session.statistics.mean
    top = max(float(np.percentile(inside, high)) if inside.size else start + 1.0, start + 1.0)
    middle = float(np.percentile(inside, look["shadedMidPercentile"])) if inside.size else (start + top) / 2.0
    middle = min(max(middle, start + 0.25), top - 0.25)
    negative = bool(look["negative"])
    styleThreeDView(viewNode, negative)
    try:
        logic = slicer.modules.volumerendering.logic()
    except AttributeError:
        return node
    node.CreateDefaultDisplayNodes()   # volume rendering copies its window from the scalar display node
    displayNode = logic.GetFirstVolumeRenderingDisplayNode(node) or logic.CreateDefaultVolumeRenderingNodes(node)
    volumeProperty = displayNode.GetVolumePropertyNode().GetVolumeProperty()
    opacity = vtk.vtkPiecewiseFunction()
    points = ((V_PAD, 0.0), (start, 0.0), (middle, 0.5), (top, 1.0), (top + 10000.0, 1.0)) if composite else \
        ((V_PAD, 0.0), (start, 0.0), (top, 1.0), (top + 10000.0, 1.0))
    for value, alpha in points:
        opacity.AddPoint(value, alpha)
    background, ink = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)) if negative else ((1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    color = vtk.vtkColorTransferFunction()
    if composite:
        # shaded: #ff003c (faint) -> #ff0000 (opacity 0.5) -> white (half way to full opacity, and beyond);
        # negative changes only the background
        for value, rgb in ((start, SHADED_COLORS[0]), (middle, SHADED_COLORS[1]),
                           ((middle + top) / 2.0, SHADED_COLORS[2]), (top, SHADED_COLORS[2])):
            color.AddRGBPoint(value, *rgb)
    else:
        # DSA-like MIP: from the background colour to black (white in negative) as the contrast rises
        color.AddRGBPoint(start, *background)
        color.AddRGBPoint(top, *ink)
    volumeProperty.SetScalarOpacity(opacity)
    volumeProperty.SetColor(color)
    # maximum intensity projection (DSA-like), or composite with shading (depth and surface of the vessels)
    volumeProperty.SetShade(composite)
    viewNode.SetRaycastTechnique(slicer.vtkMRMLViewNode.Composite if composite
                                 else slicer.vtkMRMLViewNode.MaximumIntensityProjection)
    displayNode.SetViewNodeIDs([viewNode.GetID()])
    displayNode.SetVisibility(True)
    return node


def showLayout(cbctNode, treeSegmentationNode=None, caseSegmentationNode=None, session=None, look=None,
               resetView=True):
    """Planning layout: 3D view (white, orthographic, from anterior) with the whole liver, tumours, perfused volumes,
    the arterial tree, the centerlines and the local contrast inside the search mask; axial and coronal CBCT."""
    layoutManager = slicer.app.layoutManager()
    if layoutManager is None:
        return
    layoutNode = layoutManager.layoutLogic().GetLayoutNode()
    if not layoutNode.IsLayoutDescription(PLANNING_LAYOUT_ID):
        layoutNode.AddLayoutDescription(PLANNING_LAYOUT_ID, LAYOUT_XML)
    elif layoutNode.GetLayoutDescription(PLANNING_LAYOUT_ID) != LAYOUT_XML:
        layoutNode.SetLayoutDescription(PLANNING_LAYOUT_ID, LAYOUT_XML)   # an older version registered in this session
    layoutManager.setLayout(PLANNING_LAYOUT_ID)
    slicer.app.processEvents()
    if cbctNode is not None:
        slicer.util.setSliceViewerLayers(background=cbctNode, foreground=None, fit=resetView)
    if treeSegmentationNode is not None and treeSegmentationNode.GetDisplayNode() is not None:
        treeSegmentationNode.GetDisplayNode().SetVisibility(True)
        treeSegmentationNode.GetDisplayNode().SetVisibility3D(False)   # 3D: the red centerlines instead
    outlineSegments2D(caseSegmentationNode)
    _showInPlanningSliceViews([caseSegmentationNode, treeSegmentationNode,
                               storeNode().GetNodeReference(REF_PLANNING_CENTERLINES)])
    viewNode = threeDViewNode()
    styleThreeDView(viewNode)
    try:
        styleCaseSegments3D(caseSegmentationNode, viewNode)
    except Exception as e:
        logging.warning(f"CBCT planning: could not show the segments in 3D: {e}")
    if session is not None:
        try:
            renderMaskedContrast(session, viewNode, look or lookFromSettings(settings(storeNode())))
        except Exception as e:
            logging.warning(f"CBCT planning: could not render the local contrast: {e}")
    try:
        overlayCenterlines(storeNode())
        applyThreeDVisibility(storeNode(), caseSegmentationNode)
    except Exception as e:
        logging.warning(f"CBCT planning: could not overlay the centerlines: {e}")
    if resetView:
        slicer.app.processEvents()
        resetThreeDView(viewNode)
    try:
        installCarmAnnotation()
    except Exception as e:
        logging.warning(f"CBCT planning: could not show the C-arm angles: {e}")


# -- Show / hide in the 3D view ---------------------------------------------------------------------------------

def _planningSegmentsDisplay(segmentationNode):
    if segmentationNode is None:
        return None
    for index in range(segmentationNode.GetNumberOfDisplayNodes()):
        node = segmentationNode.GetNthDisplayNode(index)
        if node is not None and node.GetAttribute(PLANNING_3D_ATTRIBUTE):
            return node
    return None


def _mipDisplay():
    node = next((n for n in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
                 if n.GetAttribute(CONTRAST_RENDER_ATTRIBUTE)), None)
    if node is None:
        return None
    try:
        return slicer.modules.volumerendering.logic().GetFirstVolumeRenderingDisplayNode(node)
    except AttributeError:
        return None


ROLE_SWITCHES = {W.SEGMENT_LIVER: "Liver", W.SEGMENT_TUMOR: "Tumours", W.SEGMENT_VIABLE: "Tumours",
                 W.SEGMENT_PERFUSED: "Perfused"}


def _roleShown(role):
    """Is the 3D switch of this segment role on (whole liver, tumours, perfused volumes / territories)?"""
    switch = ROLE_SWITCHES.get(role)
    return switch is None or bool(settings(storeNode()).get(f"show3D{switch}", True))


def applyThreeDVisibility(store, segmentationNode):
    """Liver, tumours, perfused volumes, MIP and vessels (centerlines) shown or hidden in the 3D view as stored in
    the settings."""
    values = settings(store)
    display = _planningSegmentsDisplay(segmentationNode)
    if display is not None:
        display.SetVisibility(True)
        from .controller import TARANIS_CANDIDATE_TAG, segmentTag
        segmentation = segmentationNode.GetSegmentation()
        for segmentID in segmentIDs(segmentationNode):
            segment = segmentation.GetSegment(segmentID)
            role = segmentTag(segment, TARANIS_CANDIDATE_TAG) if isCandidate(segment) else                 segmentRole(segmentationNode, segmentID)
            display.SetSegmentVisibility3D(segmentID, role in SEGMENT_3D_OPACITY and _roleShown(role))
    display = _mipDisplay()
    if display is not None:
        display.SetVisibility(bool(values["show3DMip"]))
    overlayCenterlines(store)


def setThreeDVisible(store, segmentationNode, what, visible):
    """what: "Liver", "Tumours", "Perfused", "Mip" or "Vessels"."""
    setSettings(store, **{f"show3D{what}": bool(visible)})
    applyThreeDVisibility(store, segmentationNode)


def outlineSegments2D(segmentationNode, thickness=2):
    """Slice views: the segments as outlines of the given thickness (px), no fill."""
    if segmentationNode is None:
        return
    segmentationNode.CreateDefaultDisplayNodes()
    for index in range(segmentationNode.GetNumberOfDisplayNodes()):
        display = segmentationNode.GetNthDisplayNode(index)
        if display is None or display.GetAttribute(PLANNING_3D_ATTRIBUTE) or display.GetAttribute("Taranis.Segmentation3D"):
            continue
        display.SetVisibility2DFill(True)
        display.SetOpacity2DFill(0.0)
        display.SetVisibility2DOutline(True)
        display.SetOpacity2DOutline(1.0)
        display.SetSliceIntersectionThickness(thickness)


# -- Selected candidate position (feeder finder) -----------------------------------------------------------------

CANDIDATE_POINT_NAME = "Candidate position"


def showCandidatePoint(ras):
    """One point (not a planned tip) marking the position selected in the feeder table; None hides it."""
    node = next((n for n in slicer.util.getNodesByClass("vtkMRMLMarkupsFiducialNode")
                 if n.GetAttribute("CBCTPlanning.Candidate")), None)
    if ras is None:
        if node is not None:
            node.RemoveAllControlPoints()
        return None
    if node is None:
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", CANDIDATE_POINT_NAME)
        node.SetAttribute("CBCTPlanning.Candidate", "1")
        node.CreateDefaultDisplayNodes()
        display = node.GetDisplayNode()
        display.SetSelectedColor(0.0, 0.75, 0.2)
        display.SetColor(0.0, 0.75, 0.2)
        display.SetGlyphScale(4.0)
        display.SetTextScale(0.0)
        node.SetLocked(True)
    node.RemoveAllControlPoints()
    node.AddControlPointWorld(vtk.vtkVector3d(*ras), "")
    return node


def clearTerritories(store, segmentationNode):
    """No planned tip left: remove the territory candidates of the planner and its tip results (enhancement
    territories are kept)."""
    results = planningResults(store)
    if not results:
        return
    enhancementIDs = results.get("enhancementIDs", [])
    _removeOldCandidates(segmentationNode, [i for i in results.get("segmentIDs", []) if i not in enhancementIDs])
    results["positions"], results["overlaps"], results["uncovered"], results["maa"] = [], [], [], None
    results["segmentIDs"] = list(enhancementIDs)
    if results.get("enhancement"):
        store.SetParameter(P_PLANNING_RESULTS, json.dumps(results))
    else:
        store.SetParameter(P_PLANNING_RESULTS, "")
    node = store.GetNodeReference(REF_PLANNING_CENTERLINES)
    if node is not None and node.GetPolyData() is not None:
        array = node.GetPolyData().GetPointData().GetArray("Position")
        if array is not None:
            array.Fill(0)
            node.GetPolyData().Modified()
    overlayCenterlines(store)


# -- C-arm angulation in the 3D view ---------------------------------------------------------------------------

def _carmState():
    if not hasattr(slicer, "_cbctPlanningCarm"):
        slicer._cbctPlanningCarm = {}
    return slicer._cbctPlanningCarm


def updateCarmAnnotation(caller=None, event=None):
    """Lower left corner of the 3D view: the C-arm angles that show the anatomy as the 3D view does (LAO / RAO,
    CRA / CAU), so the view can be reproduced on the angiography system."""
    state = _carmState()
    view, cameraNode = state.get("view"), state.get("camera")
    if view is None or cameraNode is None:
        return
    position, focal = [0.0] * 3, [0.0] * 3
    cameraNode.GetPosition(position)
    cameraNode.GetFocalPoint(focal)
    primary, secondary = V.carmAngles(np.array(position) - np.array(focal))
    corner = view.cornerAnnotation()
    corner.SetText(0, "C-arm (detector): " + V.carmLabel(primary, secondary))
    light = bool(settings(storeNode()).get("negative", False))
    corner.GetTextProperty().SetColor(*((0.9, 0.9, 0.9) if light else (0.1, 0.1, 0.1)))
    corner.SetMaximumFontSize(16)
    corner.SetMinimumFontSize(10)
    view.scheduleRender()


def installCarmAnnotation():
    layoutManager = slicer.app.layoutManager()
    if layoutManager is None or layoutManager.threeDViewCount == 0:
        return
    widget = threeDWidget()
    if widget is None:
        return
    viewNode = widget.mrmlViewNode()
    cameraNode = slicer.modules.cameras.logic().GetViewActiveCameraNode(viewNode)
    if cameraNode is None:
        return
    state = _carmState()
    if state.get("camera") is not cameraNode:
        if state.get("camera") is not None and state.get("tag") is not None:
            try:
                state["camera"].RemoveObserver(state["tag"])
            except Exception:
                pass
        state["tag"] = cameraNode.AddObserver(vtk.vtkCommand.ModifiedEvent, updateCarmAnnotation)
        state["camera"] = cameraNode
    state["view"] = widget.threeDView()
    updateCarmAnnotation()


def _showInPlanningSliceViews(nodes):
    """Display nodes restricted to other views (e.g. by the hub's layout, which keeps other modules' renderings out
    of its own views) are shown in the planning layout's slice views too."""
    layoutManager = slicer.app.layoutManager()
    sliceIDs = []
    for name in ("Red", "Green"):
        widget = layoutManager.sliceWidget(name) if layoutManager is not None else None
        if widget is not None:
            sliceIDs.append(widget.mrmlSliceNode().GetID())
    for node in nodes:
        if node is None:
            continue
        for index in range(node.GetNumberOfDisplayNodes()):
            display = node.GetNthDisplayNode(index)
            if display is None or display.GetAttribute(PLANNING_3D_ATTRIBUTE) or                     display.GetAttribute("Taranis.Segmentation3D"):
                continue
            current = [display.GetNthViewNodeID(i) for i in range(display.GetNumberOfViewNodeIDs())]
            if current:   # empty: shown everywhere already
                for viewID in sliceIDs:
                    if viewID not in current:
                        display.AddViewNodeID(viewID)



def removeTipResults(store, segmentationNode, label):
    """A removed tip: its territory segment leaves the segmentation (candidate or already accepted), its entry the
    stored results, and its downstream path the 3D view. Returns the removed segment's name ("" if none)."""
    results = planningResults(store)
    if not results:
        return ""
    positions = results.get("positions", [])
    number = next((n for n, p in enumerate(positions, start=1) if p.get("label") == label), None)
    if number is None:
        return ""
    position = positions.pop(number - 1)
    name = ""
    segmentID = position.get("segmentID")
    if segmentID and segmentationNode is not None:
        segment = segmentationNode.GetSegmentation().GetSegment(segmentID)
        if segment is not None:
            name = segment.GetName()
            segmentationNode.GetSegmentation().RemoveSegment(segmentID)
    results["segmentIDs"] = [i for i in results.get("segmentIDs", []) if i != segmentID]
    results["overlaps"] = [o for o in results.get("overlaps", []) if label not in o[:2]]
    store.SetParameter(P_PLANNING_RESULTS, json.dumps(results))
    # the downstream path: 'Position' of the centerline points (tip numbers after it move up by one)
    node = store.GetNodeReference(REF_PLANNING_CENTERLINES)
    array = node.GetPolyData().GetPointData().GetArray("Position") if node is not None and node.GetPolyData() else None
    if array is not None:
        from vtk.util.numpy_support import vtk_to_numpy
        values = vtk_to_numpy(array)
        values[values == number] = 0
        values[values > number] -= 1
        array.Modified()
        node.GetPolyData().Modified()
    overlayCenterlines(store)
    return name


def acceptTerritories(segmentationNode, segmentIDList):
    """Accept territory / enhancement candidates of the planner as perfused volumes (as Accept in the hub's
    Segmentation step): the '(… - evaluate)' suffix goes, the role and a red of their own are set. Returns the new
    names."""
    import re
    from .ai import acceptCandidate
    suffix = re.compile(r"\s*\(.*evaluate\)\s*$")   # from the first bracket: names may nest brackets
    segmentation = segmentationNode.GetSegmentation()
    names = []
    for segmentID in segmentIDList:
        segment = segmentation.GetSegment(segmentID)
        if segment is None or not isCandidate(segment):
            continue
        base = suffix.sub("", segment.GetName()).strip() or W.SEGMENT_ROLES[W.SEGMENT_PERFUSED][1]
        segment.SetName(base + "__accepting")   # so the new name can be the base name itself
        name = segtools.uniqueName(segmentationNode, base)
        acceptCandidate(segmentationNode, segmentID, name, W.SEGMENT_PERFUSED)
        names.append(name)
    return names
