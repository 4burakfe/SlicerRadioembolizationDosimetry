"""Segmentation helpers of the hub (Slicer side of segmentops): normal liver, clipping, splitting, perfused volume
from uptake and the geometry check. Masks are read and written on a grid with the voxel size and orientation of
the primary image of the case, extended to cover every segment and every case image: segments reaching beyond the
primary image (e.g. the lungs of a SPECT/CT when the primary image is a liver MRI) are never cropped. The hub's tools
<<<<<<< Updated upstream
use a tight grid: only the region of the voxels set in the segments (every tool reads and writes inside existing
segments), which needs far less memory than grids covering whole CT / SPECT volumes."""
=======
use a tight grid: only the bounds of the segmentation (every tool reads and writes inside existing segments), which
needs far less memory than grids covering whole CT / SPECT volumes."""
>>>>>>> Stashed changes

import json

import numpy as np
import vtk
import slicer

from . import roles as R
from . import segmentops as S
from . import workflow as W
from .case import P_SEGMENT_GEOMETRY
from .controller import segmentIDs, segmentRole, setSegmentRole, segmentInfos, segmentTag, TARANIS_CANDIDATE_TAG

MIN_REPORT_ML = 0.5          # overlaps / outside parts smaller than this are ignored
MIN_REPORT_FRACTION = 0.01
UNPERFUSED_TUMOUR_FRACTION = 0.10   # more of a tumour outside the perfused volumes: warning
TUMOUR_BURDEN_WARNING = 0.50        # tumours / whole liver
NORMAL_MISMATCH_NOTE = 0.10         # normal tissue differs from liver − tumours (and perfused − tumours) by more
LUNGS_MIN_ML = 1500.0


MAX_GRID_VOXELS = 250e6     # larger extended grids fall back to the primary image grid
<<<<<<< Updated upstream
TIGHT_MARGIN_VOXELS = 2     # tight grids (hub tools): margin around the voxels set in the segments
=======
TIGHT_MARGIN_VOXELS = 2     # tight grids (hub tools): margin around the bounds of the segmentation
>>>>>>> Stashed changes


def ownLayer(segmentationNode, segmentID):
    """Move the segment to a labelmap layer of its own if it shares one. Segments on a shared layer cannot overlap:
    painting (or writing) a new segment that shares the whole liver's layer removes those voxels from the liver,
    whatever the editor's "Modify other segments" setting. Returns True if the segment was moved."""
    segmentation = segmentationNode.GetSegmentation() if segmentationNode is not None else None
    if segmentation is None or segmentation.GetSegment(segmentID) is None:
        return False
    layer = segmentation.GetLayerIndex(segmentID)
    if layer < 0:
        return False
    for index in range(segmentation.GetNumberOfSegments()):
        otherID = segmentation.GetNthSegmentID(index)
        if otherID != segmentID and segmentation.GetLayerIndex(otherID) == layer:
            segmentation.SeparateSegmentLabelmap(segmentID)
            return True
    return False


def separateSharedLayers(segmentationNode):
    """Every segment of the case segmentation on its own layer (segments of a case may overlap). Returns the number
    of segments moved. The voxels do not change."""
    segmentation = segmentationNode.GetSegmentation() if segmentationNode is not None else None
    if segmentation is None:
        return 0
    seen, moved = set(), 0
    for segmentID in [segmentation.GetNthSegmentID(i) for i in range(segmentation.GetNumberOfSegments())]:
        layer = segmentation.GetLayerIndex(segmentID)
        if layer < 0:
            continue
        if layer in seen:
            segmentation.SeparateSegmentLabelmap(segmentID)
            moved += 1
        else:
            seen.add(layer)
    return moved


def guardAutoCompleteEffect():
    """Slicer's auto-complete effects (Fill between slices, Grow from seeds) listen to segment changes as soon as
    they are activated, but their segment list is only set by Initialize: a segment change in between raised
    "AttributeError: 'NoneType' object has no attribute 'GetNumberOfValues'" (harmless, but printed on every
    edit). Ignore segment changes until the effect is initialised. Safe to call repeatedly."""
    try:
        from SegmentEditorEffects import AbstractScriptedSegmentEditorAutoCompleteEffect as effectClass
    except ImportError:
        return False
    effectClass = getattr(effectClass, "AbstractScriptedSegmentEditorAutoCompleteEffect", effectClass)
    original = getattr(effectClass, "onSegmentationModified", None)
    if original is None or getattr(original, "_taranisGuard", False):
        return False

    def onSegmentationModified(self, caller, event):
        if getattr(self, "selectedSegmentIds", None) is None:
            return   # not initialised yet: nothing to update
        return original(self, caller, event)

    onSegmentationModified._taranisGuard = True
    onSegmentationModified.__doc__ = original.__doc__
    effectClass.onSegmentationModified = onSegmentationModified
    return True


def _matrix(volumeNode):
    matrix = vtk.vtkMatrix4x4()
    volumeNode.GetIJKToRASMatrix(matrix)
    return np.array([[matrix.GetElement(r, c) for c in range(4)] for r in range(4)])


def gridExtent(referenceVolume, boundsList, includeReference=True, marginVoxels=0):
    """(lower [i, j, k], dimensions) of the reference voxel grid extended to cover every RAS bounds box (voxel
    centres or corners, transforms included). includeReference False: only the bounds boxes (plus marginVoxels),
    not the whole reference image. Pure geometry, used by gridVolume."""
    rasToIjk = np.linalg.inv(_matrix(referenceVolume))
    dims = np.array(referenceVolume.GetImageData().GetDimensions())
    lower, upper = np.zeros(3), dims - 1.0
    if not includeReference:
        lower, upper = np.full(3, np.inf), np.full(3, -np.inf)
    for bounds in boundsList:
        if bounds is None or bounds[0] > bounds[1]:
            continue
        corners = np.array([[x, y, z, 1.0] for x in bounds[0:2] for y in bounds[2:4] for z in bounds[4:6]])
        ijk = (rasToIjk @ corners.T)[:3].T
        # corner-based bounds lie half a voxel outside the outermost voxel centres
        lower = np.minimum(lower, np.floor(ijk.min(axis=0) + 0.5 + 1e-3))
        upper = np.maximum(upper, np.ceil(ijk.max(axis=0) - 0.5 - 1e-3))
    if not np.all(np.isfinite(lower)) or np.any(upper < lower):   # no bounds: the whole reference image
        return np.zeros(3, int), dims.astype(int)
    lower, upper = lower - marginVoxels, upper + marginVoxels
    return lower.astype(int), (upper - lower + 1).astype(int)


<<<<<<< Updated upstream
def segmentationDataBounds(segmentationNode):
    """RAS bounds (voxel corners) of the voxels actually set in the segmentation's labelmap layers, or None (no
    voxels, or a segmentation under a transform: then its full bounds are used). Much smaller than the images when
    the segments cover only the liver region."""
    if segmentationNode is None or segmentationNode.GetParentTransformNode() is not None:
        return None
    labelmapName = slicer.vtkSegmentationConverter.GetBinaryLabelmapRepresentationName()
    segmentation = segmentationNode.GetSegmentation()
    lower, upper, seen = None, None, set()
    for index in range(segmentation.GetNumberOfSegments()):
        image = segmentation.GetNthSegment(index).GetRepresentation(labelmapName)
        if image is None or id(image) in seen or image.GetPointData().GetScalars() is None:
            continue
        seen.add(id(image))
        extent = [0, -1, 0, -1, 0, -1]
        slicer.vtkOrientedImageDataResample.CalculateEffectiveExtent(image, extent)
        if extent[0] > extent[1] or extent[2] > extent[3] or extent[4] > extent[5]:
            continue
        matrix = vtk.vtkMatrix4x4()
        image.GetImageToWorldMatrix(matrix)
        ijkToRas = np.array([[matrix.GetElement(r, c) for c in range(4)] for r in range(4)])
        corners = np.array([[i, j, k, 1.0] for i in (extent[0] - 0.5, extent[1] + 0.5)
                            for j in (extent[2] - 0.5, extent[3] + 0.5) for k in (extent[4] - 0.5, extent[5] + 0.5)])
        ras = (ijkToRas @ corners.T)[:3].T
        lower = ras.min(axis=0) if lower is None else np.minimum(lower, ras.min(axis=0))
        upper = ras.max(axis=0) if upper is None else np.maximum(upper, ras.max(axis=0))
    if lower is None:
        return None
    return [lower[0], upper[0], lower[1], upper[1], lower[2], upper[2]]


def gridVolume(referenceVolume, volumes=(), segmentationNode=None, name="Taranis grid", tight=False):
    """Hidden temporary volume: voxel size and orientation of referenceVolume, extent covering it, the volumes and
    the segments (the caller removes it). tight: only the voxels set in the segments (plus a margin) when there are
    any – every segment still fits entirely, at a fraction of the memory of the whole images."""
    if tight:
        dataBounds = segmentationDataBounds(segmentationNode)
        if dataBounds is not None:
            lower, dims = gridExtent(referenceVolume, [dataBounds], includeReference=False,
                                     marginVoxels=TIGHT_MARGIN_VOXELS)
            return _gridNode(referenceVolume, lower, dims, name)
=======
def gridVolume(referenceVolume, volumes=(), segmentationNode=None, name="Taranis grid", tight=False):
    """Hidden temporary volume: voxel size and orientation of referenceVolume, extent covering it, the volumes and
    the segments (the caller removes it). tight: only the bounds of the segmentation (all its labelmap layers, plus a
    margin) – every segment fits entirely (a write replaces the whole segment, so nothing may lie outside the grid),
    without the whole CT / SPECT volumes around it."""
    if tight and segmentationNode is not None:
        bounds = [0.0] * 6
        segmentationNode.GetRASBounds(bounds)
        if bounds[0] <= bounds[1] and bounds[2] <= bounds[3] and bounds[4] <= bounds[5]:
            lower, dims = gridExtent(referenceVolume, [bounds], includeReference=False,
                                     marginVoxels=TIGHT_MARGIN_VOXELS)
            if float(np.prod(dims)) <= MAX_GRID_VOXELS:
                return _gridNode(referenceVolume, lower, dims, name)
>>>>>>> Stashed changes
    boundsList = []
    for node in list(volumes) + [segmentationNode]:
        if node is None:
            continue
        bounds = [0.0] * 6
        node.GetRASBounds(bounds)
        boundsList.append(bounds)
    lower, dims = gridExtent(referenceVolume, boundsList)
    if float(np.prod(dims)) > MAX_GRID_VOXELS:
        lower, dims = np.zeros(3, int), np.array(referenceVolume.GetImageData().GetDimensions())
    return _gridNode(referenceVolume, lower, dims, name)


def _gridNode(referenceVolume, lower, dims, name):
    image = vtk.vtkImageData()
    image.SetDimensions(int(dims[0]), int(dims[1]), int(dims[2]))
    image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    image.GetPointData().GetScalars().Fill(0)
    node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", name)
    node.SetHideFromEditors(True)
    node.SetAndObserveImageData(image)
    matrix = _matrix(referenceVolume)
    matrix[:3, 3] = (matrix @ np.array([lower[0], lower[1], lower[2], 1.0]))[:3]
    vtkMatrix = vtk.vtkMatrix4x4()
    for r in range(4):
        for c in range(4):
            vtkMatrix.SetElement(r, c, float(matrix[r][c]))
    node.SetIJKToRASMatrix(vtkMatrix)
    return node


def caseVolumes(case):
    return [node for node in case.roleNodes().values() if node is not None and node.IsA("vtkMRMLScalarVolumeNode")]


def extendSegmentationGeometry(segmentationNode, referenceVolume, volumes=()):
    """Reference geometry of the segmentation (used by the Segment Editor and imports): voxel size of the primary
    image, extent covering all case images and segments. Returns True if it changed."""
    if segmentationNode is None or referenceVolume is None or referenceVolume.GetImageData() is None:
        return False
    grid = gridVolume(referenceVolume, volumes, segmentationNode, "Taranis geometry")
    try:
        before = segmentationNode.GetSegmentation().GetConversionParameter(
            slicer.vtkSegmentationConverter.GetReferenceImageGeometryParameterName())
        segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(grid)
        after = segmentationNode.GetSegmentation().GetConversionParameter(
            slicer.vtkSegmentationConverter.GetReferenceImageGeometryParameterName())
        return before != after
    finally:
        slicer.mrmlScene.RemoveNode(grid)


class SegmentGrid:
    """Reads and writes segments of one segmentation on the extended grid of the primary image (see module doc).
    Use as a context manager (or call close()) to remove the temporary grid volume."""

    def __init__(self, segmentationNode, referenceVolume, volumes=(), tight=False):
        if segmentationNode is None:
            raise ValueError("No segmentation in the case.")
        if referenceVolume is None or referenceVolume.GetImageData() is None:
            raise ValueError("No image to define the voxel grid (assign the images in the Data step).")
        self.node = segmentationNode
        self.reference = gridVolume(referenceVolume, volumes, segmentationNode, tight=tight)
        self.shape = tuple(slicer.util.arrayFromVolume(self.reference).shape)
        spacing = referenceVolume.GetSpacing()
        self.voxelML = spacing[0] * spacing[1] * spacing[2] / 1000.0
        self._cache = {}

    def close(self):
        if self.reference is not None and self.reference.GetScene() is not None:
            slicer.mrmlScene.RemoveNode(self.reference)
        self.reference = None
        self._cache = {}
        try:
            from .memory import collectSoon
            collectSoon()   # the masks can be large: give the memory back soon
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
        return False

    def ids(self, role=None):
        return [i for i in segmentIDs(self.node) if role is None or segmentRole(self.node, i) == role]

    def name(self, segmentID):
        return self.node.GetSegmentation().GetSegment(segmentID).GetName()

    def mask(self, segmentID):
        if segmentID not in self._cache:
            array = slicer.util.arrayFromSegmentBinaryLabelmap(self.node, segmentID, self.reference)
            self._cache[segmentID] = np.asarray(array) > 0
        return self._cache[segmentID]

    def union(self, segmentIDList):
        result = np.zeros(self.shape, bool)
        for segmentID in segmentIDList:
            result |= self.mask(segmentID)
        return result

    def write(self, segmentID, mask):
<<<<<<< Updated upstream
=======
        """Replace the segment with the mask. Refused if the segment has voxels outside this grid (they would be
        lost: the whole segment is replaced)."""
        outside = self._voxelsOutside(segmentID)
        if outside:
            raise RuntimeError(f"'{self.name(segmentID)}' reaches beyond the working grid ({outside} voxels "
                               "outside); it was not changed.")
>>>>>>> Stashed changes
        ownLayer(self.node, segmentID)   # on a shared layer the write would take voxels from the other segments
        slicer.util.updateSegmentBinaryLabelmapFromArray(mask.astype(np.uint8), self.node, segmentID, self.reference)
        self._cache[segmentID] = mask.astype(bool)

<<<<<<< Updated upstream
=======
    def _voxelsOutside(self, segmentID):
        """Voxels of the segment (in its own labelmap) that lie outside this grid; 0 if it cannot be checked."""
        segment = self.node.GetSegmentation().GetSegment(segmentID)
        image = segment.GetRepresentation(slicer.vtkSegmentationConverter.GetBinaryLabelmapRepresentationName()) \
            if segment is not None else None
        if image is None or image.GetPointData() is None or image.GetPointData().GetScalars() is None \
                or self.node.GetParentTransformNode() is not None:
            return 0
        extent = image.GetExtent()
        if extent[0] > extent[1] or extent[2] > extent[3] or extent[4] > extent[5]:
            return 0
        from vtk.util.numpy_support import vtk_to_numpy
        shape = (extent[5] - extent[4] + 1, extent[3] - extent[2] + 1, extent[1] - extent[0] + 1)
        label = segment.GetLabelValue() if hasattr(segment, "GetLabelValue") else 1
        inside = vtk_to_numpy(image.GetPointData().GetScalars()).reshape(shape) == label
        k = np.flatnonzero(inside.any(axis=(1, 2)))
        j = np.flatnonzero(inside.any(axis=(0, 2)))
        i = np.flatnonzero(inside.any(axis=(0, 1)))
        if len(i) == 0:
            return 0
        # voxel centres of the segment -> grid voxel indices
        matrix = vtk.vtkMatrix4x4()
        image.GetImageToWorldMatrix(matrix)
        imageToWorld = np.array([[matrix.GetElement(r, c) for c in range(4)] for r in range(4)])
        worldToGrid = np.linalg.inv(_matrix(self.reference))
        corners = np.array([[a, b, c, 1.0] for a in (i.min(), i.max()) for b in (j.min(), j.max())
                            for c in (k.min(), k.max())], dtype=float)
        corners[:, :3] += np.array(extent[0::2], dtype=float)
        gridIjk = (worldToGrid @ imageToWorld @ corners.T)[:3].T
        dims = np.array(self.shape[::-1], dtype=float)   # (i, j, k)
        if np.all(gridIjk >= -0.5 - 1e-3) and np.all(gridIjk <= dims - 0.5 + 1e-3):
            return 0
        return int(np.count_nonzero(inside))   # the bounding box pokes out: report the segment size

>>>>>>> Stashed changes
    def add(self, name, role, mask, candidate=False):
        segmentation = self.node.GetSegmentation()
        color = W.SEGMENT_ROLES[role][2] if role in W.SEGMENT_ROLES else (0.6, 0.6, 0.6)
        segmentID = segmentation.AddEmptySegment("", name, color)
        self.write(segmentID, mask)
        if candidate:
            segment = segmentation.GetSegment(segmentID)
            segment.SetTag(TARANIS_CANDIDATE_TAG, role or "1")
            segment.SetColor(1.0, 0.9, 0.0)
        elif role:
            setSegmentRole(self.node, segmentID, role)
        return segmentID

    def tag(self, segmentID, tag):
        return segmentTag(self.node.GetSegmentation().GetSegment(segmentID), tag)

    def setTag(self, segmentID, tag, value):
        self.node.GetSegmentation().GetSegment(segmentID).SetTag(tag, value)

    def remove(self, segmentID):
        self.node.GetSegmentation().RemoveSegment(segmentID)
        self._cache.pop(segmentID, None)

    def liver(self):
        livers = self.ids(W.SEGMENT_LIVER)
        if not livers:
            raise ValueError("No whole-liver segment.")
        if len(livers) > 1:
            raise ValueError("Several segments are marked as whole liver; keep one.")
        return livers[0], self.mask(livers[0])


def uniqueName(segmentationNode, base):
    existing = {segmentationNode.GetSegmentation().GetSegment(i).GetName() for i in segmentIDs(segmentationNode)}
    if base not in existing:
        return base
    number = 2
    while f"{base} {number}" in existing:
        number += 1
    return f"{base} {number}"


def numberedName(segmentationNode, stem):
    existing = {segmentationNode.GetSegmentation().GetSegment(i).GetName() for i in segmentIDs(segmentationNode)}
    number = 1
    while f"{stem} {number}" in existing:
        number += 1
    return f"{stem} {number}"


# -- Helpers ----------------------------------------------------------------------------------------------------------

PERFUSED_NORMAL_NAME = "Perfused normal liver"


def normalScope(grid, segmentID):
    """W.NORMAL_SCOPE_PERFUSED or W.NORMAL_SCOPE_WHOLE of a normal tissue segment: its tag, otherwise its name."""
    scope = grid.tag(segmentID, W.NORMAL_SCOPE_TAG)
    if scope:
        return scope
    return W.NORMAL_SCOPE_PERFUSED if "perfus" in grid.name(segmentID).lower() else W.NORMAL_SCOPE_WHOLE


def makeNormalLiver(grid):
    """Create or update the normal-liver segment (whole liver minus all tumours). Returns a message. Perfused normal
    liver segments are left alone."""
    _, liver = grid.liver()
    tumours = grid.ids(W.SEGMENT_TUMOR) + grid.ids(W.SEGMENT_VIABLE)
    normal = S.normalTissue(liver, [grid.mask(i) for i in tumours])
    existing = [i for i in grid.ids(W.SEGMENT_NORMAL) if normalScope(grid, i) != W.NORMAL_SCOPE_PERFUSED]
    if existing:
        segmentID = existing[0]
        grid.write(segmentID, normal)
        action = f"'{grid.name(segmentID)}' updated"
    else:
        segmentID = grid.add(uniqueName(grid.node, W.SEGMENT_ROLES[W.SEGMENT_NORMAL][1]), W.SEGMENT_NORMAL, normal)
        action = "created"
    grid.setTag(segmentID, W.NORMAL_SCOPE_TAG, W.NORMAL_SCOPE_WHOLE)
    return (f"Normal liver {action}: {S.volumeML(normal, grid.voxelML):.0f} mL "
            f"(whole liver minus {len(tumours)} tumour segment(s)).")


def makePerfusedNormal(grid):
    """Create or update the perfused normal liver: each perfused volume inside the whole liver, minus all tumours
    (one segment per perfused volume). Used for the tumour-to-normal ratio and the normal tissue dose. Returns a
    message."""
    _, liver = grid.liver()
    perfusedIDs = grid.ids(W.SEGMENT_PERFUSED)
    if not perfusedIDs:
        raise ValueError("No perfused volume segment: create the perfused volume(s) first.")
    tumours = [grid.mask(i) for i in grid.ids(W.SEGMENT_TUMOR) + grid.ids(W.SEGMENT_VIABLE)]
    perfusedNormals = [i for i in grid.ids(W.SEGMENT_NORMAL) if normalScope(grid, i) == W.NORMAL_SCOPE_PERFUSED]
    bySource = {grid.tag(i, W.NORMAL_SOURCE_TAG): i for i in perfusedNormals}
    untagged = [i for i in perfusedNormals if not grid.tag(i, W.NORMAL_SOURCE_TAG)]
    parts = []
    for perfusedID in perfusedIDs:
        normal = S.normalTissue(grid.mask(perfusedID) & liver, tumours)
        segmentID = bySource.get(perfusedID)
        if segmentID is None and len(perfusedIDs) == 1 and untagged:
            segmentID = untagged[0]   # e.g. drawn by hand and named "perfused normal"
        if segmentID is not None:
            grid.write(segmentID, normal)
            action = "updated"
        else:
            name = PERFUSED_NORMAL_NAME if len(perfusedIDs) == 1 else \
                f"{PERFUSED_NORMAL_NAME} ({grid.name(perfusedID)})"
            segmentID = grid.add(uniqueName(grid.node, name), W.SEGMENT_NORMAL, normal)
            action = "created"
        grid.setTag(segmentID, W.NORMAL_SCOPE_TAG, W.NORMAL_SCOPE_PERFUSED)
        grid.setTag(segmentID, W.NORMAL_SOURCE_TAG, perfusedID)
        parts.append(f"'{grid.name(segmentID)}' {action} ({S.volumeML(normal, grid.voxelML):.0f} mL)")
    return (f"Perfused normal liver: " + ", ".join(parts) +
            f" = perfused volume inside the whole liver minus {len(tumours)} tumour segment(s).")


def clipToLiver(grid, roles=(W.SEGMENT_TUMOR, W.SEGMENT_PERFUSED, W.SEGMENT_NORMAL)):
    """Remove the parts of tumour, perfused and normal-tissue segments outside the whole liver."""
    _, liver = grid.liver()
    changes = []
    for role in roles:
        for segmentID in grid.ids(role):
            inside, removed = S.clipToContainer(grid.mask(segmentID), liver)
            if removed:
                grid.write(segmentID, inside)
                changes.append(f"'{grid.name(segmentID)}' −{removed * grid.voxelML:.1f} mL")
    return ("Clipped to the liver: " + ", ".join(changes) + ".") if changes else "Nothing outside the liver."


def removeLiverFromLungs(grid):
    """Lung voxels shared with the whole liver belong to the liver (boundary partial volume). Returns mL removed."""
    livers = grid.ids(W.SEGMENT_LIVER)
    if len(livers) != 1:
        return 0.0
    liver = grid.mask(livers[0])
    removed = 0
    for lungID in grid.ids(W.SEGMENT_LUNGS):
        lungs = grid.mask(lungID)
        shared = lungs & liver
        if shared.any():
            grid.write(lungID, lungs & ~liver)
            removed += int(np.count_nonzero(shared))
    return removed * grid.voxelML


def splitSegment(grid, segmentID, minML=0.5):
    """Split a segment into its connected parts (largest first): the first part keeps the segment, the others become
    new segments with the same role. Returns a message."""
    role = segmentRole(grid.node, segmentID)
    baseName = grid.name(segmentID)
    minVoxels = max(1, int(round(minML / grid.voxelML)))
    mask = grid.mask(segmentID)
    parts = S.splitComponents(mask, minVoxels)
    if not parts:
        return f"'{baseName}' has no part of at least {minML} mL."
    dropped = int(np.count_nonzero(mask) - sum(int(np.count_nonzero(p)) for p in parts))
    if len(parts) == 1 and not dropped:
        return f"'{baseName}' is a single connected region."
    stem = W.SEGMENT_ROLES[role][1] if role in W.SEGMENT_ROLES else baseName
    grid.write(segmentID, parts[0])
    if role in W.SEGMENT_ROLES and not baseName.startswith(stem + " "):
        grid.node.GetSegmentation().GetSegment(segmentID).SetName(numberedName(grid.node, stem))
    for part in parts[1:]:
        grid.add(numberedName(grid.node, stem), role, part)
    text = f"'{baseName}' split into {len(parts)} region(s)"
    if dropped:
        text += f"; {dropped * grid.voxelML:.1f} mL in parts smaller than {minML} mL removed"
    return text + "."


def perfusedFromUptake(grid, uptakeVolume, percent, minML=5.0):
    """Perfused-volume candidate from the uptake image (MAA / Y-90) inside the whole liver. Returns (segment ID,
    message)."""
    _, liver = grid.liver()
    resampled = slicer.vtkSlicerVolumesLogic().ResampleVolumeToReferenceVolume(uptakeVolume, grid.reference)
    try:
        values = np.array(slicer.util.arrayFromVolume(resampled), dtype=np.float32)
    finally:
        if resampled is not None and resampled.GetScene() is not None:
            slicer.mrmlScene.RemoveNode(resampled)
    minVoxels = max(1, int(round(minML / grid.voxelML)))
    parts = S.perfusedFromUptake(values, liver, percent, minVoxels)
    if not parts:
        raise ValueError(f"No region above {percent:g}% of the maximum uptake in the liver.")
    mask = np.zeros_like(liver)
    for part in parts:
        mask |= part
    name = uniqueName(grid.node, f"{W.SEGMENT_ROLES[W.SEGMENT_PERFUSED][1]} ({percent:g}% uptake - evaluate)")
    segmentID = grid.add(name, W.SEGMENT_PERFUSED, mask, candidate=True)
    liverML = S.volumeML(liver, grid.voxelML)
    volume = S.volumeML(mask, grid.voxelML)
    return segmentID, (f"Perfused volume from uptake ≥ {percent:g}% of the maximum: {volume:.0f} mL "
                       f"({100.0 * volume / liverML:.0f}% of the liver) in {len(parts)} region(s). Review it in the "
                       "Segment Editor, then accept it.")


# -- Geometry check -----------------------------------------------------------------------------------------------

def geometryCheck(grid, mode=None):
    """[(severity, text)] of overlaps, parts outside the liver or the perfused volumes, tumour burden and outdated
    normal tissue. mode: the case's processing mode (the texts mention what patient-relative dosimetry does).
    The whole-liver and perfused volume sizes are checked by the workflow validator (always up to date)."""
    issues = []
    ml = grid.voxelML
    names = {i: grid.name(i) for i in grid.ids()}
    byRole = {role: grid.ids(role) for role in W.SEGMENT_ROLE_KEYS}
    livers = byRole[W.SEGMENT_LIVER]
    liver = grid.mask(livers[0]) if len(livers) == 1 else None

    def significant(volume, fraction):
        return volume >= MIN_REPORT_ML and fraction >= MIN_REPORT_FRACTION

    if liver is not None:
        inside = {names[i]: grid.mask(i) for role in (W.SEGMENT_TUMOR, W.SEGMENT_VIABLE, W.SEGMENT_PERFUSED,
                                                      W.SEGMENT_NORMAL)
                  for i in byRole[role]}
        for name, volume, fraction in S.outsideReport(inside, liver, ml):
            if significant(volume, fraction):
                issues.append((W.SEVERITY_WARNING, f"'{name}': {volume:.1f} mL ({100 * fraction:.0f}%) outside the "
                                                   "whole liver (use 'Clip to liver')."))
        for lungID in byRole[W.SEGMENT_LUNGS]:
            overlap = np.count_nonzero(grid.mask(lungID) & liver) * ml
            if overlap >= MIN_REPORT_ML:
                issues.append((W.SEVERITY_WARNING, f"'{names[lungID]}' overlaps the whole liver by {overlap:.1f} mL."))
    for lungID in byRole[W.SEGMENT_LUNGS]:
        lungML = S.volumeML(grid.mask(lungID), ml)
        if 0 < lungML < LUNGS_MIN_ML:
            issues.append((W.SEVERITY_INFO, f"'{names[lungID]}' is {lungML:.0f} mL: the lungs may be cut by the "
                                            "field of view (lung mass and lung dose would be too high)."))
    perfused = {names[i]: grid.mask(i) for i in byRole[W.SEGMENT_PERFUSED]}
    for a, b, volume in S.overlapReport(perfused, ml):
        if volume >= MIN_REPORT_ML:
            issues.append((W.SEVERITY_WARNING, f"Perfused volumes '{a}' and '{b}' overlap by {volume:.1f} mL "
                                               "(counted twice)."))
    tumours = {names[i]: grid.mask(i) for i in byRole[W.SEGMENT_TUMOR]}
    for a, b, volume in S.overlapReport(tumours, ml):
        if volume >= MIN_REPORT_ML:
            issues.append((W.SEVERITY_INFO, f"Tumours '{a}' and '{b}' overlap by {volume:.1f} mL."))
    normals = {names[i]: grid.mask(i) for i in byRole[W.SEGMENT_NORMAL]}
    if normals and (tumours or byRole[W.SEGMENT_VIABLE]):
        allTumours = np.zeros_like(next(iter(normals.values())))
        for mask in list(tumours.values()) + [grid.mask(i) for i in byRole[W.SEGMENT_VIABLE]]:
            allTumours |= mask
        for name, mask in normals.items():
            volume = np.count_nonzero(mask & allTumours) * ml
            if volume >= MIN_REPORT_ML:
                issues.append((W.SEVERITY_WARNING, f"Normal tissue '{name}' contains {volume:.1f} mL of tumour "
                                                   "(use 'Normal liver = liver − tumours' or 'Perfused normal = "
                                                   "perfused − tumours')."))

    # Tumours and the perfused volumes
    tumourMasks = {names[i]: grid.mask(i) for role in (W.SEGMENT_TUMOR, W.SEGMENT_VIABLE) for i in byRole[role]}
    perfusedIDs = byRole[W.SEGMENT_PERFUSED]
    relative = mode == "relative"
    if tumourMasks and perfusedIDs:
        perfusedUnion = grid.union(perfusedIDs)
        zeroDose = " (0 Gy in patient-relative dosimetry)" if relative else ""
        for name, totalML, outsideML, fraction in S.perfusionReport(tumourMasks, perfusedUnion, ml):
            if fraction >= 1.0:
                issues.append((W.SEVERITY_WARNING, f"'{name}' does not intersect any perfused volume{zeroDose}: is "
                                                   "it outside the treated territory, or is a perfused volume "
                                                   "missing?"))
            elif fraction > UNPERFUSED_TUMOUR_FRACTION and outsideML >= MIN_REPORT_ML:
                issues.append((W.SEVERITY_WARNING, f"{100 * fraction:.0f}% of '{name}' ({outsideML:.1f} of "
                                                   f"{totalML:.1f} mL) is outside the perfused volume(s)"
                                                   f"{' – that part gets 0 Gy in patient-relative dosimetry' if relative else ''}: "
                                                   "check the perfused volumes."))

    # Tumour burden, and normal tissue that no longer matches the liver and tumours
    if liver is not None and tumourMasks:
        tumourUnion = np.zeros_like(liver)
        for mask in tumourMasks.values():
            tumourUnion |= mask
        liverVoxels = int(np.count_nonzero(liver))
        burden = int(np.count_nonzero(tumourUnion & liver)) / liverVoxels if liverVoxels else 0.0
        if burden > TUMOUR_BURDEN_WARNING:
            issues.append((W.SEVERITY_WARNING, f"Tumour burden {100 * burden:.0f}% of the whole liver (more than "
                                               f"{100 * TUMOUR_BURDEN_WARNING:.0f}%)."))
        expected = [liver & ~tumourUnion]
        if perfusedIDs:
            expected.append(grid.union(perfusedIDs) & ~tumourUnion)
            expected.append(grid.union(perfusedIDs) & liver & ~tumourUnion)
            if len(perfusedIDs) > 1:   # 'Perfused normal' makes one segment per perfused volume
                expected += [grid.mask(i) & liver & ~tumourUnion for i in perfusedIDs]
        for name, mask in normals.items():
            mismatch = min(S.mismatchFraction(mask, target) for target in expected)
            if mismatch > NORMAL_MISMATCH_NOTE:
                issues.append((W.SEVERITY_INFO, f"Normal tissue '{name}' differs by {100 * mismatch:.0f}% from "
                                                "whole liver − tumours and from perfused − tumours: if the "
                                                "segments were edited afterwards, recreate it with 'Normal liver = "
                                                "liver − tumours' or 'Perfused normal = perfused − tumours'."))
    return issues


def storeGeometryCheck(case, segmentationNode, issues):
    """Store the result with the key of the current segments (the toolbar shows it until the segments change)."""
    key = W.segmentsKey(segmentInfos(segmentationNode))
    case.setParam(P_SEGMENT_GEOMETRY, json.dumps({"key": key, "issues": [list(issue) for issue in issues]}))


def uptakeVolume(case):
    """Image used for the perfused volume: the dosimetry image (MAA or Y-90)."""
    return case.roleNode(R.ROLE_DOSIMETRY)
