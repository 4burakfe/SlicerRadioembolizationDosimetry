import os
import re
import time
import logging

import numpy as np
import qt
import ctk
import vtk
import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

from EasyRegLib import functional as F


MODULE_VERSION = "2.0"

# (name, BRAINSFit transformType chain, result is linear, description)
# Each chain starts with the simpler stages, so the later stages start from a good alignment.
METHODS = [
    ("Rigid", "Rigid", True,
     "Rotation and translation only. Recommended default: the images are not distorted."),
    ("Affine", "Rigid,Affine", True,
     "Rigid, then affine (adds scaling and shearing). Compensates small size/scale differences."),
]
# Deformable (B-spline) registration was removed: unacceptably slow and >10 GB of memory even at 0.2 % sampling.
# Deformable transforms of older scenes are still recognised (NON_RIGID_METHODS) and can be undone.
METHOD_BY_NAME = {name: (chain, linear, text) for name, chain, linear, text in METHODS}

# Functional-only path (a SPECT/PET without its own CT registered directly to the reference CT/MRI)
PATH_HYBRID = "hybrid"
PATH_FUNCTIONAL = "functional"
REFINE_METHOD = "Rigid (liver mask)"
METHOD_BY_NAME[REFINE_METHOD] = ("Rigid", True, "Rigid mutual-information refinement inside the dilated liver.")
NON_RIGID_METHODS = ("Affine", "Deformable")
DEFAULT_OUTLINE_PERCENT = 3.0      # SPECT/PET body outline threshold, % of the robust maximum
DEFAULT_MASK_MARGIN_MM = 20.0      # liver mask dilation for the refinement
REFINE_WARNING_MM = 20.0           # plausibility check of the refinement (displacement of the liver centre)
REFINE_WARNING_DEGREES = 10.0
MAX_MASK_VOXELS = 8_000_000        # images are subsampled above this for the body outline
LANDMARK_FIXED_COLOR = (0.15, 0.85, 0.35)
LANDMARK_MOVING_COLOR = (1.0, 0.55, 0.0)
FUNCTIONAL_LAYOUT_ID = 50202
# (view label, background, foreground)
FUNCTIONAL_VIEWS = [("Reference", "reference", None), ("SPECT / PET", "spect", None),
                    ("SPECT / PET on reference", "reference", "spect")]


def isRigidMethod(method):
    return bool(method) and not any(name in method for name in NON_RIGID_METHODS)

# (label, BRAINSFit samplingPercentage)
QUALITY_PRESETS = [("Fast", 0.002), ("Standard", 0.01), ("Thorough", 0.05)]
# (label, BRAINSFit initializeTransformMode)
INITIALIZATION_MODES = [
    ("Align centres of the images / ROIs", "useGeometryAlign"),
    ("Keep current position", "Off"),
]
OVERLAY_CT_COLORMAP = "Inferno"
SPECT_COLORMAP = "Inferno"  # SPECT/PET in the fusion views

# Fusion layout shown after registration: 4 columns x (axial row, coronal row).
# (view label, background, foreground); "overlayCT" is the SPECT/CT CT shown with OVERLAY_CT_COLORMAP.
FUSION_LAYOUT_ID = 50201  # the dosimetry modules use 50101 / 50102
FUSION_VIEWS = [
    ("SPECT on SPECT/CT", "spectCT", "spect"),
    ("SPECT on reference", "reference", "spect"),
    ("Reference", "reference", None),
    ("SPECT/CT on reference", "reference", "overlayCT"),
]
FUSION_ORIENTATIONS = [("Axial", "#F34A33"), ("Coronal", "#6EB04B")]

# Taranis workflow hub: EasyReg's navigation buttons go to these steps (Data, Registration, Segmentation)
TARANIS_MODULE = "Taranis"
WORKFLOW_STEPS = {"previous": "data", "overview": "registration", "next": "segmentation"}

DEFAULT_ROI_SIZE_MM = (400.0, 300.0, 250.0)
DEFAULT_SPLINE_GRID = (14, 10, 12)
SPECT_ROI_COLOR = (1.0, 0.55, 0.0)
REFERENCE_ROI_COLOR = (0.0, 0.75, 0.3)

# Attributes of nodes created by this module
TRANSFORM_ATTRIBUTE = "EasyReg.Transform"             # "1" on every registration transform
METHOD_ATTRIBUTE = "EasyReg.Method"
PREVIOUS_TRANSFORM_ATTRIBUTE = "EasyReg.PreviousTransformID"  # parent transform before the registration
HARDENED_NODES_ATTRIBUTE = "EasyReg.HardenedNodeIDs"  # rigid transforms hardened into these nodes (undoable)
HARDENED_ATTRIBUTE = "EasyReg.Hardened"
CENTRE_METHOD = "Centred on reference"    # method of a transform made by 'Centre moving image on reference'
REGISTERED_NODES_ATTRIBUTE = "EasyReg.RegisteredNodeIDs"  # moving image and followers (read by the Taranis workflow)
ROI_INITIALIZED_ATTRIBUTE = "EasyReg.ROIInitialized"
OVERLAY_ATTRIBUTE = "EasyReg.OverlayOf"               # display-only copy of a volume (shared voxel data)

INFO_TEXT = (
    "Registers a SPECT/CT (or PET/CT) to a diagnostic CT or MRI.\n"
    "1. Select the CT of the SPECT/CT (moving image), the SPECT itself and the reference CT/MRI (fixed image). "
    "The SPECT, and optionally a segmentation made on the SPECT/CT, follow the CT.\n"
    "2. Optional but recommended: create an ROI around the liver in each image. Only temporary copies are "
    "cropped for the registration; your images are never modified.\n"
    "3. Choose the method and press Register. If the CT already has a transform (a previous rigid run or a "
    "manual pre-alignment), the registration starts from it.\n"
    "4. After registration the views switch to a 4x2 fusion layout (axial top, coronal bottom): SPECT on "
    "its CT, SPECT on reference, reference only, SPECT/CT CT (inferno) on reference. Scrolling, zooming and "
    "panning are synchronised within each row. Evaluate the alignment (liver dome, liver edges, spine, "
    "kidneys).\n"
    "5. If the alignment is acceptable, press Harden transform, then Next to continue the Taranis workflow "
    "(Registration overview: back to the list of registrations of the case). Otherwise press Undo and change the "
    "ROIs or method, or fine-tune manually.\n"
    "This module is NOT a medical device. Research use only.\n"
    "Developed by: Burak Demir, MD, FEBNM\n"
    "For support and feedback: 4burakfe@gmail.com\n"
    f"Version: {MODULE_VERSION}"
)


# -- Helpers (no GUI state) ----------------------------------------------------

def registrationParameters(fixedVolume, movingVolume, outputTransform, transformType, samplingPercentage,
                           initializeMode="useGeometryAlign", initialTransform=None,
                           splineGridSize=DEFAULT_SPLINE_GRID, fixedMask=None, movingMask=None):
    """BRAINSFit parameters. Mattes mutual information works for CT-CT, CT-MRI and SPECT-CT/MRI inside a mask."""
    parameters = {
        "fixedVolume": fixedVolume,
        "movingVolume": movingVolume,
        "outputTransform": outputTransform,
        "transformType": transformType,
        "samplingPercentage": samplingPercentage,
        "costMetric": "MMI",
        "interpolationMode": "Linear",
    }
    if initialTransform is not None:
        # BRAINSFit ignores the initialization mode when an initial transform is given
        parameters["initialTransform"] = initialTransform
        parameters["initializeTransformMode"] = "Off"
    else:
        parameters["initializeTransformMode"] = initializeMode
    if fixedMask is not None and movingMask is not None:
        parameters["maskProcessingMode"] = "ROI"
        parameters["fixedBinaryVolume"] = fixedMask
        parameters["movingBinaryVolume"] = movingMask
    if "BSpline" in transformType:
        parameters["splineGridSize"] = ",".join(str(int(v)) for v in splineGridSize)
    return parameters


def defaultRoiGeometry(volumeNode, defaultSize=DEFAULT_ROI_SIZE_MM):
    """(center, size) in world RAS: centred on the volume, default size limited to the volume extent."""
    bounds = [0.0] * 6
    volumeNode.GetRASBounds(bounds)
    center = [(bounds[2 * i] + bounds[2 * i + 1]) / 2.0 for i in range(3)]
    extent = [bounds[2 * i + 1] - bounds[2 * i] for i in range(3)]
    size = [min(s, e) if e > 0 else s for s, e in zip(defaultSize, extent)]
    return center, size


def roiSize(roiNode):
    try:
        return list(roiNode.GetSize())
    except TypeError:
        size = [0.0] * 3
        roiNode.GetSize(size)
        return size


def initializeRoi(roiNode, volumeNode, color):
    """Colour the ROI and, only if it is new (never placed), centre it on the volume with the default size.
    An existing ROI the user has already adjusted is left untouched."""
    if roiNode is None:
        return
    if roiNode.GetDisplayNode() is None:
        roiNode.CreateDefaultDisplayNodes()
    displayNode = roiNode.GetDisplayNode()
    if displayNode:
        displayNode.SetSelectedColor(*color)
        displayNode.SetColor(*color)
    if volumeNode is None or roiNode.GetAttribute(ROI_INITIALIZED_ATTRIBUTE) == "1":
        return
    if roiNode.GetNumberOfControlPoints() > 0 and any(s > 0 for s in roiSize(roiNode)):
        roiNode.SetAttribute(ROI_INITIALIZED_ATTRIBUTE, "1")  # placed by the user elsewhere
        return
    center, size = defaultRoiGeometry(volumeNode)
    roiNode.SetSize(*size)
    roiNode.SetCenter(*center)
    roiNode.SetAttribute(ROI_INITIALIZED_ATTRIBUTE, "1")


def setRoiVisible(roiNode, visible):
    if roiNode is not None and roiNode.GetDisplayNode():
        roiNode.GetDisplayNode().SetVisibility(visible)


def findColorNode(name):
    """Color table by name (case-insensitive), searching only color nodes, so a volume or other node with the
    same name is never picked up by mistake."""
    for colorNode in slicer.util.getNodesByClass("vtkMRMLColorNode"):
        if (colorNode.GetName() or "").lower() == name.lower():
            return colorNode
    return None


def setColormap(volumeNode, colorName):
    if volumeNode is None:
        return
    if volumeNode.GetDisplayNode() is None:
        volumeNode.CreateDefaultDisplayNodes()
    displayNode = volumeNode.GetDisplayNode()
    colorNode = findColorNode(colorName)
    if displayNode and colorNode:
        displayNode.SetAndObserveColorNodeID(colorNode.GetID())
    elif colorNode is None:
        logging.warning(f"Color table '{colorName}' not found")


def readableFunctionalWindow(volumeNode, percentile=99.9, fullRangeFraction=0.9):
    """A SPECT/PET windowed over its whole value range (e.g. 0 - 9800 counts with a few hot voxels) looks almost
    black: window it 0 - 99.9th percentile instead. A window the user chose (narrower) is kept."""
    displayNode = volumeNode.GetDisplayNode() if volumeNode is not None else None
    imageData = volumeNode.GetImageData() if volumeNode is not None else None
    if displayNode is None or imageData is None:
        return
    low, high = imageData.GetScalarRange()
    if high <= low or displayNode.GetWindow() < fullRangeFraction * (high - low):
        return
    values = slicer.util.arrayFromVolume(volumeNode)
    top = float(np.percentile(values[values > 0], percentile)) if np.any(values > 0) else high
    if top <= 0 or top >= high:
        return
    displayNode.AutoWindowLevelOff()
    displayNode.SetWindowLevelMinMax(0.0, top)


def showLayers(background, foreground=None, opacity=0.5, fit=True):
    slicer.util.setSliceViewerLayers(background=background, foreground=foreground,
                                     foregroundOpacity=opacity, fit=fit)


def setForegroundOpacity(opacity):
    for compositeNode in slicer.util.getNodesByClass("vtkMRMLSliceCompositeNode"):
        compositeNode.SetForegroundOpacity(opacity)


def isOverlayVolume(node):
    return node.GetAttribute(OVERLAY_ATTRIBUTE) is not None


def nodesUnderTransform(transformNode):
    """Nodes whose parent transform is transformNode (display-only overlay copies excluded)."""
    if transformNode is None:
        return []
    transformID = transformNode.GetID()
    return [node for node in slicer.util.getNodesByClass("vtkMRMLTransformableNode")
            if node.GetTransformNodeID() == transformID and not isOverlayVolume(node)]


def hasPendingRegistration(volumeNode):
    """True if the volume is under a live (not yet hardened) EasyReg transform."""
    transformNode = volumeNode.GetParentTransformNode() if volumeNode is not None else None
    return transformNode is not None and transformNode.GetAttribute(TRANSFORM_ATTRIBUTE) == "1"


def updateOverlayVolume(volumeNode, colorName):
    """Display-only copy of the volume with its own colour table. A volume has one display node, so the
    SPECT/CT CT cannot be grey in one view and coloured in another; the copy shares the voxel data (no extra
    memory), geometry and parent transform of the original and is not saved with the scene."""
    overlay = None
    for node in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode"):
        if node.GetAttribute(OVERLAY_ATTRIBUTE) == volumeNode.GetID():
            overlay = node
            break
    if overlay is None:
        overlay = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        overlay.SetAttribute(OVERLAY_ATTRIBUTE, volumeNode.GetID())
        overlay.SetHideFromEditors(True)
        overlay.SetSaveWithScene(False)
    overlay.SetName(f"{volumeNode.GetName()} ({colorName.lower()})")
    ijkToRas = vtk.vtkMatrix4x4()
    volumeNode.GetIJKToRASMatrix(ijkToRas)
    overlay.SetIJKToRASMatrix(ijkToRas)
    overlay.SetAndObserveImageData(volumeNode.GetImageData())
    overlay.SetAndObserveTransformNodeID(volumeNode.GetTransformNodeID())
    setColormap(overlay, colorName)
    sourceDisplay, overlayDisplay = volumeNode.GetDisplayNode(), overlay.GetDisplayNode()
    if sourceDisplay and overlayDisplay:
        overlayDisplay.SetAutoWindowLevel(False)
        overlayDisplay.SetWindowLevel(sourceDisplay.GetWindow(), sourceDisplay.GetLevel())
    return overlay


def removeOverlayVolumes():
    for node in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode"):
        if isOverlayVolume(node):
            slicer.mrmlScene.RemoveNode(node)


def fusionViewTag(orientation, index):
    return f"EasyReg{orientation}{index + 1}"


def fusionLayoutXml():
    rows = []
    for orientation, color in FUSION_ORIENTATIONS:
        items = "".join(
            f'<item><view class="vtkMRMLSliceNode" singletontag="{fusionViewTag(orientation, index)}">'
            f'<property name="orientation" action="default">{orientation}</property>'
            f'<property name="viewlabel" action="default">{orientation[0]}{index + 1}</property>'
            f'<property name="viewcolor" action="default">{color}</property></view></item>'
            for index in range(len(FUSION_VIEWS)))
        rows.append(f'<item><layout type="horizontal" split="true">{items}</layout></item>')
    return f'<layout type="vertical" split="true">{"".join(rows)}</layout>'


def functionalViewTag(orientation, index):
    return f"EasyRegF{orientation}{index + 1}"


def functionalLayoutXml():
    rows = []
    for orientation, color in FUSION_ORIENTATIONS:
        items = "".join(
            f'<item><view class="vtkMRMLSliceNode" singletontag="{functionalViewTag(orientation, index)}">'
            f'<property name="orientation" action="default">{orientation}</property>'
            f'<property name="viewlabel" action="default">{orientation[0]}{index + 1}</property>'
            f'<property name="viewcolor" action="default">{color}</property></view></item>'
            for index in range(len(FUNCTIONAL_VIEWS)))
        rows.append(f'<item><layout type="horizontal" split="true">{items}</layout></item>')
    return f'<layout type="vertical" split="true">{"".join(rows)}</layout>'


def registerFusionLayout(*args):
    """Register the fusion layouts (at startup, so a scene saved with one can be restored, and before use)."""
    layoutManager = slicer.app.layoutManager()
    if layoutManager is None:
        return
    layoutNode = layoutManager.layoutLogic().GetLayoutNode()
    for layoutID, xml in ((FUSION_LAYOUT_ID, fusionLayoutXml()), (FUNCTIONAL_LAYOUT_ID, functionalLayoutXml())):
        if layoutNode.IsLayoutDescription(layoutID):
            layoutNode.SetLayoutDescription(layoutID, xml)
        else:
            layoutNode.AddLayoutDescription(layoutID, xml)


class SliceViewSynchronizer:
    """Keeps slice position, orientation, pan and zoom identical within groups of slice views (one group per
    row of the fusion layout). Slicer's own view linking is not used because it also synchronises the displayed
    volumes, which must differ between the views."""

    def __init__(self):
        self.groups = []
        self.observations = []
        self._syncing = False

    def setGroups(self, groups):
        self.clear()
        self.groups = [list(group) for group in groups]
        for group in self.groups:
            for node in group:
                self.observations.append((node, node.AddObserver(vtk.vtkCommand.ModifiedEvent, self._onModified)))

    def clear(self):
        for node, tag in self.observations:
            node.RemoveObserver(tag)
        self.observations = []
        self.groups = []

    def _onModified(self, caller, event):
        if self._syncing:
            return
        for group in self.groups:
            if any(node.GetID() == caller.GetID() for node in group):
                self.syncFrom(caller, group)
                return

    def syncFrom(self, source, group):
        self._syncing = True
        try:
            sourceMatrix = source.GetSliceToRAS()
            sourceFov = source.GetFieldOfView()
            for node in group:
                if node.GetID() == source.GetID():
                    continue
                matrix = node.GetSliceToRAS()
                same = all(abs(matrix.GetElement(r, c) - sourceMatrix.GetElement(r, c)) < 1e-6
                           for r in range(4) for c in range(4))
                sameFov = all(abs(a - b) < 1e-3 for a, b in zip(node.GetFieldOfView(), sourceFov))
                if same and sameFov:
                    continue  # avoids ping-pong between views
                wasModifying = node.StartModify()
                matrix.DeepCopy(sourceMatrix)
                node.SetFieldOfView(*sourceFov)
                node.SetXYZOrigin(*source.GetXYZOrigin())
                node.UpdateMatrices()
                node.EndModify(wasModifying)
        finally:
            self._syncing = False


def removeNodeIfInScene(node):
    if node is not None and node.GetScene() is not None:
        slicer.mrmlScene.RemoveNode(node)


def formatBounds(bounds):
    return ", ".join(f"{axis} {bounds[2 * i]:.0f} to {bounds[2 * i + 1]:.0f}" for i, axis in enumerate("RAS"))


def matrixToNumpy(vtkMatrix):
    return np.array([[vtkMatrix.GetElement(r, c) for c in range(4)] for r in range(4)])


def numpyToMatrix(array):
    matrix = vtk.vtkMatrix4x4()
    for r in range(4):
        for c in range(4):
            matrix.SetElement(r, c, float(array[r][c]))
    return matrix


def transformToWorld(node):
    """4x4 numpy matrix from the node's local coordinates to world (identity without a parent transform).
    Raises ValueError for a non-linear parent transform."""
    parent = node.GetParentTransformNode()
    if parent is None:
        return np.eye(4)
    matrix = vtk.vtkMatrix4x4()
    if not slicer.vtkMRMLTransformNode.GetMatrixTransformBetweenNodes(parent, None, matrix):
        raise ValueError(f"'{node.GetName()}' is under a non-linear transform.")
    return matrixToNumpy(matrix)


def ijkToWorld(volumeNode):
    """4x4 numpy matrix from voxel (i, j, k) to world RAS, including a linear parent transform."""
    ijkToRas = vtk.vtkMatrix4x4()
    volumeNode.GetIJKToRASMatrix(ijkToRas)
    return transformToWorld(volumeNode) @ matrixToNumpy(ijkToRas)


def subsampled(array, matrix, maxVoxels=MAX_MASK_VOXELS):
    """(array[::s, ::s, ::s], matching ijk-to-world matrix) with s chosen to keep at most maxVoxels voxels."""
    step = 1
    while array[::step, ::step, ::step].size > maxVoxels:
        step += 1
    scale = np.diag([step, step, step, 1.0])
    return array[::step, ::step, ::step], matrix @ scale


def looksLikeCT(volumeNode):
    imageData = volumeNode.GetImageData()
    return imageData is not None and imageData.GetScalarRange()[0] <= -500


def removeTemporaryLabelmap(labelmapNode):
    """Remove a temporary label map node and the colour table ("..._ColorTable") that exporting segments created for
    it, if nothing else uses that table (otherwise one table is left in the scene per export)."""
    if labelmapNode is None or labelmapNode.GetScene() is None:
        return
    displayNode = labelmapNode.GetDisplayNode()
    colorNode = displayNode.GetColorNode() if displayNode is not None and hasattr(displayNode, "GetColorNode") else None
    slicer.mrmlScene.RemoveNode(labelmapNode)
    if colorNode is None or colorNode.GetScene() is None or colorNode.GetSingletonTag() \
            or not re.search(r"_ColorTable(_\d+)?$", colorNode.GetName() or ""):
        return
    if not any(node.GetColorNodeID() == colorNode.GetID()
               for node in slicer.util.getNodesByClass("vtkMRMLDisplayNode") if hasattr(node, "GetColorNodeID")):
        slicer.mrmlScene.RemoveNode(colorNode)


def segmentMaskOnVolume(segmentationNode, segmentID, volumeNode):
    """Boolean array [k, j, i] of a segment on the volume's voxel grid."""
    labelmap = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
    labelmap.SetHideFromEditors(True)
    try:
        if not slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(
                segmentationNode, [segmentID], labelmap, volumeNode):
            raise ValueError("Could not export the liver segment onto the reference image.")
        return slicer.util.arrayFromVolume(labelmap) > 0
    finally:
        removeTemporaryLabelmap(labelmap)


def labelmapFromArray(mask, referenceVolume, name):
    """Temporary label map with the reference volume's geometry (no parent transform)."""
    labelmap = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", name)
    labelmap.SetHideFromEditors(True)
    labelmap.SetSaveWithScene(False)
    slicer.util.updateVolumeFromArray(labelmap, mask.astype(np.uint8))
    ijkToRas = vtk.vtkMatrix4x4()
    referenceVolume.GetIJKToRASMatrix(ijkToRas)
    labelmap.SetIJKToRASMatrix(ijkToRas)
    return labelmap


def cropVolumeToRoi(volumeNode, roiNode, name):
    """New volume node with the voxels of volumeNode inside the ROI's bounding box, or None if they do not
    overlap. Pure voxel extraction: no resampling, and the IJK-to-RAS matrix (including oblique or flipped
    axis directions, common for MRI) is kept, so every voxel stays at its physical position. The ROI is in
    world coordinates; a parent transform of the volume is taken into account. The result has no parent
    transform (it is in the volume's local coordinates)."""
    roiBounds = [0.0] * 6
    roiNode.GetRASBounds(roiBounds)
    if roiBounds[1] < roiBounds[0]:
        return None  # ROI not placed
    corners = [(r, a, s) for r in roiBounds[0:2] for a in roiBounds[2:4] for s in roiBounds[4:6]]
    parent = volumeNode.GetParentTransformNode()
    worldToLocal = None
    if parent is not None:
        worldToLocal = vtk.vtkGeneralTransform()
        parent.GetTransformFromWorld(worldToLocal)
    rasToIjk = vtk.vtkMatrix4x4()
    volumeNode.GetRASToIJKMatrix(rasToIjk)
    ijkCorners = []
    for corner in corners:
        local = worldToLocal.TransformPoint(corner) if worldToLocal is not None else corner
        ijkCorners.append(rasToIjk.MultiplyPoint(list(local) + [1.0])[:3])
    ijkCorners = np.array(ijkCorners)

    dimensions = np.array(volumeNode.GetImageData().GetDimensions())  # (i, j, k)
    lower = np.maximum(np.floor(ijkCorners.min(axis=0)).astype(int), 0)
    upper = np.minimum(np.ceil(ijkCorners.max(axis=0)).astype(int), dimensions - 1)
    if np.any(upper - lower < 1):
        return None

    array = slicer.util.arrayFromVolume(volumeNode)  # (k, j, i)
    cropped = np.ascontiguousarray(array[lower[2]:upper[2] + 1, lower[1]:upper[1] + 1, lower[0]:upper[0] + 1])
    ijkToRas = vtk.vtkMatrix4x4()
    volumeNode.GetIJKToRASMatrix(ijkToRas)
    offset = vtk.vtkMatrix4x4()
    for axis in range(3):
        offset.SetElement(axis, 3, float(lower[axis]))
    croppedIjkToRas = vtk.vtkMatrix4x4()
    vtk.vtkMatrix4x4.Multiply4x4(ijkToRas, offset, croppedIjkToRas)

    croppedNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", name)
    slicer.util.updateVolumeFromArray(croppedNode, cropped)
    croppedNode.SetIJKToRASMatrix(croppedIjkToRas)
    return croppedNode


# -- Module ----------------------------------------------------------------------

class easy_reg(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "Taranis - EasyReg"
        parent.categories = ["Nuclear Medicine"]
        parent.dependencies = []
        parent.contributors = ["Burak Demir, MD, FEBNM"]
        parent.helpText = """
        Easy workflow for registration of SPECT/CT (or PET/CT) images to diagnostic CT/MR images.<br>
        The CT of the SPECT/CT is registered to the reference image; the SPECT (and optionally a segmentation)
        follows it. Optional ROIs restrict the registration to the liver region without modifying the images.
        """
        parent.acknowledgementText = """
        This file was developed by Burak Demir.
        """
        iconPath = os.path.join(os.path.dirname(__file__), "Resources", "taranis_logo.png")
        self.parent.icon = qt.QIcon(iconPath)
        self.parent = parent
        slicer.app.connect("startupCompleted()", registerFusionLayout)


# -- Widget ----------------------------------------------------------------------

class easy_regWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):

    # (parameter node reference role, selector attribute)
    NODE_REFERENCES = [("SpectCT", "spectCTSelector"), ("Spect", "spectSelector"),
                       ("Reference", "referenceSelector"), ("Segmentation", "segmentationSelector"),
                       ("SpectROI", "spectRoiSelector"), ("ReferenceROI", "referenceRoiSelector"),
                       ("LiverSegmentation", "liverSegmentationSelector")]

    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic = None
        self._parameterNode = None
        self._updatingGUI = False
        self._startTime = None
        self._previousLayout = None
        self.sliceSync = SliceViewSynchronizer()

    # ---- UI construction ----

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = easy_regLogic()

        bannerPath = os.path.join(os.path.dirname(__file__), "Resources", "banner.png")
        if os.path.exists(bannerPath):
            bannerLabel = qt.QLabel()
            bannerLabel.setPixmap(qt.QPixmap(bannerPath).scaledToWidth(400, qt.Qt.SmoothTransformation))
            bannerLabel.setAlignment(qt.Qt.AlignCenter)
            self.layout.addWidget(bannerLabel)
        else:
            logging.warning(f"Banner file not found at {bannerPath}")

        # ---- 1. Images ----
        imagesBox = ctk.ctkCollapsibleButton()
        imagesBox.text = "1. Images"
        self.layout.addWidget(imagesBox)
        imagesLayout = qt.QFormLayout(imagesBox)

        pathRow = qt.QVBoxLayout()
        self.pathGroup = qt.QButtonGroup()
        self.pathButtons = {}
        for path, text, toolTip in (
                (PATH_HYBRID, "SPECT/CT or PET/CT \u2192 reference (anatomy to anatomy)",
                 "The CT of the hybrid scan is registered to the reference; the SPECT/PET follows it."),
                (PATH_FUNCTIONAL, "SPECT or PET only \u2192 reference (no CT of its own)",
                 "Functional-only: body outline, landmarks, manual adjustment and a rigid refinement inside the "
                 "liver. Affine and deformable registration are not offered.")):
            button = qt.QRadioButton(text)
            button.setToolTip(toolTip)
            self.pathGroup.addButton(button)
            self.pathButtons[path] = button
            pathRow.addWidget(button)
        self.pathButtons[PATH_HYBRID].setChecked(True)
        imagesLayout.addRow("Registration path: ", pathRow)

        self.spectCTSelector = self._makeNodeSelector(
            "vtkMRMLScalarVolumeNode", "CT of the SPECT/CT. This image is registered to the reference.")
        self.spectCTLabel = qt.QLabel("CT of SPECT/CT (moving): ")
        imagesLayout.addRow(self.spectCTLabel, self.spectCTSelector)
        self.spectSelector = self._makeNodeSelector(
            "vtkMRMLScalarVolumeNode", "SPECT (or PET) acquired with the CT above. It follows the CT.",
            allowNone=True)
        self.spectLabel = qt.QLabel("SPECT / PET (follows): ")
        imagesLayout.addRow(self.spectLabel, self.spectSelector)
        self.referenceSelector = self._makeNodeSelector(
            "vtkMRMLScalarVolumeNode", "Diagnostic CT or MRI. This image does not move.")
        imagesLayout.addRow("Reference CT/MRI (fixed): ", self.referenceSelector)
        self.segmentationSelector = self._makeNodeSelector(
            "vtkMRMLSegmentationNode",
            "Optional: a segmentation drawn on the SPECT/CT (e.g. automatic liver segmentation). It follows "
            "the CT. Leave empty for segmentations drawn on the reference image.", allowNone=True)
        self.segmentationLabel = qt.QLabel("Segmentation on SPECT/CT: ")
        imagesLayout.addRow(self.segmentationLabel, self.segmentationSelector)

        self.liverSegmentationSelector = self._makeNodeSelector(
            "vtkMRMLSegmentationNode", "Segmentation drawn on the reference image that contains the whole liver. "
            "Used to place the uptake head-feet and to restrict the automatic refinement.", allowNone=True)
        self.liverSegmentationLabel = qt.QLabel("Liver segmentation on reference: ")
        imagesLayout.addRow(self.liverSegmentationLabel, self.liverSegmentationSelector)
        self.liverSegmentComboBox = qt.QComboBox()
        self.liverSegmentComboBox.setToolTip("The whole-liver segment (on the reference image).")
        self.liverSegmentLabel = qt.QLabel("Liver segment: ")
        imagesLayout.addRow(self.liverSegmentLabel, self.liverSegmentComboBox)

        # Very different bed positions: bring the moving image next to the reference first
        positionRow = qt.QHBoxLayout()
        self.centreButton = qt.QPushButton("Centre moving image on reference")
        self.centreButton.setToolTip(
            "Moves the moving image (the CT of the SPECT/CT, or the SPECT/PET itself) so that the centre of its "
            "field of view lies on the centre of the reference image. A quick first step when the two scans have "
            "very different bed positions; register or align afterwards. Nothing is resampled.")
        self.resetPositionButton = qt.QPushButton("Reset to original position")
        self.resetPositionButton.setToolTip(
            "Removes EasyReg's transform: the moving image (and the images and landmarks that follow it) go back to "
            "where they were before EasyReg moved them.")
        for button in (self.centreButton, self.resetPositionButton):
            button.setMinimumHeight(38)
            font = button.font
            font.setBold(True)
            button.setFont(font)
            positionRow.addWidget(button)
        imagesLayout.addRow(positionRow)
        self.positionStatusLabel = qt.QLabel("")
        self.positionStatusLabel.setWordWrap(True)
        imagesLayout.addRow(self.positionStatusLabel)

        # ---- 2. Registration region ----
        regionBox = ctk.ctkCollapsibleButton()
        regionBox.text = "2. Registration region (optional, recommended)"
        self.layout.addWidget(regionBox)
        regionLayout = qt.QFormLayout(regionBox)
        regionInfo = qt.QLabel(
            "Restrict the registration to the liver region. Create a new ROI in each selector: it is centred on "
            "the image and can be moved and resized in the slice views. Without ROIs the whole images are used. "
            "Only temporary copies are cropped; your images are never modified.")
        regionInfo.setWordWrap(True)
        regionLayout.addRow(regionInfo)

        self.spectRoiSelector = self._makeNodeSelector(
            "vtkMRMLMarkupsROINode", "ROI around the liver on the SPECT/CT (orange).", allowCreate=True)
        self.spectRoiSelector.baseName = "EasyReg ROI SPECT-CT"
        self.showSpectCTButton = qt.QPushButton("Show")
        self.showSpectCTButton.setToolTip("Show the SPECT/CT and its ROI in the slice views.")
        self.spectRoiLabel = qt.QLabel("ROI on SPECT/CT: ")
        regionLayout.addRow(self.spectRoiLabel, self._row(self.spectRoiSelector, self.showSpectCTButton))

        self.referenceRoiSelector = self._makeNodeSelector(
            "vtkMRMLMarkupsROINode", "ROI around the liver on the reference image (green).", allowCreate=True)
        self.referenceRoiSelector.baseName = "EasyReg ROI reference"
        self.showReferenceButton = qt.QPushButton("Show")
        self.showReferenceButton.setToolTip("Show the reference image and its ROI in the slice views.")
        regionLayout.addRow("ROI on reference: ", self._row(self.referenceRoiSelector, self.showReferenceButton))

        # ---- 3. Registration ----
        registerBox = ctk.ctkCollapsibleButton()
        registerBox.text = "3. Registration"
        self.layout.addWidget(registerBox)
        self.registerBox = registerBox
        registerLayout = qt.QFormLayout(registerBox)

        self.methodGroup = qt.QButtonGroup()
        self.methodButtons = {}
        methodColumn = qt.QVBoxLayout()
        for name, _, _, text in METHODS:
            button = qt.QRadioButton(name)
            button.setToolTip(text)
            self.methodGroup.addButton(button)
            self.methodButtons[name] = button
            methodColumn.addWidget(button)
        self.methodButtons["Rigid"].setChecked(True)
        registerLayout.addRow("Method: ", methodColumn)

        self.qualityComboBox = qt.QComboBox()
        for label, percentage in QUALITY_PRESETS:
            self.qualityComboBox.addItem(f"{label} ({100 * percentage:g}% of voxels sampled)", label)
        self.qualityComboBox.setToolTip("More samples are more robust but slower.")
        registerLayout.addRow("Quality: ", self.qualityComboBox)

        self.initializationComboBox = qt.QComboBox()
        for label, mode in INITIALIZATION_MODES:
            self.initializationComboBox.addItem(label, mode)
        self.initializationComboBox.setToolTip(
            "Starting position when the CT has no transform yet.\n"
            "Align centres: works well when both images (or both ROIs) cover the same region.\n"
            "Keep current position: use when the images are already roughly aligned.\n"
            "If the CT already has a transform (previous registration or manual pre-alignment in the "
            "Transforms module), the registration always starts from it.")
        registerLayout.addRow("Start position: ", self.initializationComboBox)

        self.hardenRigidCheckBox = qt.QCheckBox("Harden rigid result automatically")
        self.hardenRigidCheckBox.setChecked(False)
        self.hardenRigidCheckBox.setToolTip(
            "Off (recommended): evaluate the result first, then press Harden transform.\n"
            "On: rigid results are hardened immediately (lossless, can still be undone). "
            "Affine results are never hardened automatically.")
        registerLayout.addRow("", self.hardenRigidCheckBox)


        buttonRow = qt.QHBoxLayout()
        self.registerButton = qt.QPushButton("Register")
        self.registerButton.setToolTip("Register the CT of the SPECT/CT to the reference image.")
        self.cancelButton = qt.QPushButton("Cancel")
        self.cancelButton.enabled = False
        buttonRow.addWidget(self.registerButton, 3)
        buttonRow.addWidget(self.cancelButton, 1)
        registerLayout.addRow(buttonRow)

        self.progressBar = qt.QProgressBar()
        self.progressBar.setRange(0, 0)  # busy indicator: BRAINSFit does not report reliable progress
        self.progressBar.visible = False
        registerLayout.addRow(self.progressBar)
        self.statusLabel = qt.QLabel("")
        self.statusLabel.setWordWrap(True)
        registerLayout.addRow(self.statusLabel)

        # ---- 3. Functional-only registration ----
        self.functionalBox = ctk.ctkCollapsibleButton()
        self.functionalBox.text = "3. Functional-only registration"
        self.layout.addWidget(self.functionalBox)
        functionalLayout = qt.QVBoxLayout(self.functionalBox)
        functionalInfo = qt.QLabel(
            "A SPECT/PET without its own CT is registered in two stages. A: bring it roughly into place (body "
            "outline, landmarks or by hand). B (optional): refine rigidly inside the liver. Always check the result "
            "visually; the step is marked as functional-only in the workflow and the report.")
        functionalInfo.setWordWrap(True)
        functionalLayout.addWidget(functionalInfo)

        outlineBox = qt.QGroupBox("A1. Body outline")
        outlineLayout = qt.QFormLayout(outlineBox)
        self.outlinePercentSpinBox = qt.QDoubleSpinBox()
        self.outlinePercentSpinBox.setRange(0.2, 30.0)
        self.outlinePercentSpinBox.setSingleStep(0.5)
        self.outlinePercentSpinBox.setDecimals(1)
        self.outlinePercentSpinBox.setSuffix(" % of max")
        self.outlinePercentSpinBox.value = DEFAULT_OUTLINE_PERCENT
        self.outlinePercentSpinBox.setToolTip(
            "Threshold for the body outline of the SPECT/PET (scatter and background activity). Lower it if the "
            "outline is incomplete, raise it if noise outside the body is included.")
        outlineLayout.addRow("SPECT/PET outline threshold: ", self.outlinePercentSpinBox)
        self.outlineButton = qt.QPushButton("Align body outlines")
        self.outlineButton.setToolTip(
            "Centres the SPECT/PET body outline on the reference body outline (left-right, anterior-posterior). "
            "With a liver segment selected, the uptake centre is placed on the liver centre (head-feet).")
        outlineLayout.addRow(self.outlineButton)
        functionalLayout.addWidget(outlineBox)

        landmarkBox = qt.QGroupBox("A2. Landmarks (at least 3 pairs, in the same order)")
        landmarkLayout = qt.QFormLayout(landmarkBox)
        landmarkInfo = qt.QLabel(
            "Place the same points on both images in the same order, e.g. liver dome, porta hepatis, focal uptake "
            "\u2194 tumour, stomach or kidney activity, spleen tip. Spread them out. The SPECT/PET landmarks move "
            "with the image.<br>Click the place button, then click in the views: <b>R points in the left column</b> "
            "(reference), <b>S points in the middle column</b> (SPECT/PET alone); both appear in the fusion column.")
        landmarkInfo.setTextFormat(qt.Qt.RichText)
        landmarkInfo.setWordWrap(True)
        landmarkLayout.addRow(landmarkInfo)
        self.fixedLandmarksPlace = self._makePlaceWidget()
        self.fixedLandmarksCount = qt.QLabel("0")
        landmarkLayout.addRow("On reference (R1, R2, ...): ", self._row(self.fixedLandmarksPlace,
                                                                        self.fixedLandmarksCount))
        self.movingLandmarksPlace = self._makePlaceWidget()
        self.movingLandmarksCount = qt.QLabel("0")
        landmarkLayout.addRow("On SPECT/PET (S1, S2, ...): ", self._row(self.movingLandmarksPlace,
                                                                        self.movingLandmarksCount))
        self.landmarkButton = qt.QPushButton("Align with landmarks")
        landmarkLayout.addRow(self.landmarkButton)
        self.landmarkResultLabel = qt.QLabel("")
        self.landmarkResultLabel.setWordWrap(True)
        landmarkLayout.addRow(self.landmarkResultLabel)
        functionalLayout.addWidget(landmarkBox)

        manualBox = qt.QGroupBox("A3. By hand")
        manualLayout = qt.QHBoxLayout(manualBox)
        self.handlesButton = qt.QPushButton("Show move / rotate handles")
        self.handlesButton.checkable = True
        self.handlesButton.setToolTip("Interaction handles on the SPECT/PET in the slice and 3D views.")
        self.transformsModuleButton = qt.QPushButton("Transforms module")
        self.transformsModuleButton.setToolTip("Translation and rotation sliders for fine adjustment.")
        manualLayout.addWidget(self.handlesButton)
        manualLayout.addWidget(self.transformsModuleButton)
        functionalLayout.addWidget(manualBox)

        refineBox = qt.QGroupBox("B. Automatic refinement inside the liver (optional)")
        refineLayout = qt.QFormLayout(refineBox)
        self.maskMarginSpinBox = qt.QDoubleSpinBox()
        self.maskMarginSpinBox.setRange(0.0, 60.0)
        self.maskMarginSpinBox.setSuffix(" mm")
        self.maskMarginSpinBox.value = DEFAULT_MASK_MARGIN_MM
        self.maskMarginSpinBox.setToolTip("The liver segment is grown by this margin; only this region is compared.")
        refineLayout.addRow("Liver mask margin: ", self.maskMarginSpinBox)
        self.refineQualityComboBox = qt.QComboBox()
        for label, percentage in QUALITY_PRESETS:
            self.refineQualityComboBox.addItem(f"{label} ({100 * percentage:g}% of voxels sampled)", label)
        self.refineQualityComboBox.setCurrentIndex(len(QUALITY_PRESETS) - 1)
        refineLayout.addRow("Quality: ", self.refineQualityComboBox)
        refineButtons = qt.QHBoxLayout()
        self.refineButton = qt.QPushButton("Refine rigidly inside the liver")
        self.refineButton.setToolTip(
            "Rigid mutual-information registration of the SPECT/PET to the reference, restricted to the dilated "
            "liver, starting from the current alignment. Do an initial alignment (A) first.")
        self.refineCancelButton = qt.QPushButton("Cancel")
        self.refineCancelButton.enabled = False
        refineButtons.addWidget(self.refineButton, 3)
        refineButtons.addWidget(self.refineCancelButton, 1)
        refineLayout.addRow(refineButtons)
        functionalLayout.addWidget(refineBox)
        self.functionalStatusLabel = qt.QLabel("")
        self.functionalStatusLabel.setWordWrap(True)
        functionalLayout.addWidget(self.functionalStatusLabel)

        # ---- 4. Check and finish ----
        checkBox = ctk.ctkCollapsibleButton()
        checkBox.text = "4. Check and finish"
        self.layout.addWidget(checkBox)
        checkLayout = qt.QFormLayout(checkBox)

        self.resultLabel = qt.QLabel("No registration yet.")
        self.resultLabel.setWordWrap(True)
        checkLayout.addRow(self.resultLabel)

        self.fusionLayoutButton = qt.QPushButton("Show fusion layout")
        self.fusionLayoutButton.setToolTip(
            "4x2 views, axial (top) and coronal (bottom):\n"
            "1 SPECT on SPECT/CT, 2 SPECT on reference, 3 reference only, 4 SPECT/CT CT (inferno) on reference.")
        checkLayout.addRow(self.fusionLayoutButton)
        self.opacitySlider = ctk.ctkSliderWidget()
        self.opacitySlider.minimum = 0.0
        self.opacitySlider.maximum = 1.0
        self.opacitySlider.singleStep = 0.05
        self.opacitySlider.value = 0.5
        self.opacitySlider.setToolTip("Foreground opacity. Move back and forth to check edges (liver dome, "
                                      "spine, kidneys).")
        checkLayout.addRow("Overlay opacity: ", self.opacitySlider)

        actionGrid = qt.QGridLayout()
        self.hardenButton = qt.QPushButton("Harden transform")
        self.hardenButton.setToolTip("Apply the transform permanently to the images.")
        self.undoButton = qt.QPushButton("Undo registration")
        self.undoButton.setToolTip("Return the images to their position before the last registration.")
        self.fineTuneButton = qt.QPushButton("Fine-tune manually")
        self.fineTuneButton.setToolTip("Open the transform in the Transforms module for manual adjustment "
                                       "(linear transforms that are not hardened).")
        actionGrid.addWidget(self.hardenButton, 0, 0)
        actionGrid.addWidget(self.undoButton, 0, 1)
        actionGrid.addWidget(self.fineTuneButton, 1, 0, 1, 2)
        checkLayout.addRow(actionGrid)
        # Taranis workflow navigation (replaces the former "Open dosimetry" buttons)
        navigationRow = qt.QHBoxLayout()
        self.workflowButtons = {}
        for key, text, toolTip in (
                ("previous", "\u25c0 Previous", "Back to the Data and imaging roles step of the Taranis workflow."),
                ("overview", "Registration overview", "The registrations of the case in the Taranis workflow."),
                ("next", "Next \u25b6", "Continue the Taranis workflow with the Segmentation step.")):
            button = qt.QPushButton(text)
            button.setToolTip(toolTip)
            button.connect("clicked()", lambda key=key: self.onWorkflowButton(key))
            self.workflowButtons[key] = button
            navigationRow.addWidget(button)
        checkLayout.addRow(navigationRow)

        infoBox = ctk.ctkCollapsibleButton()
        infoBox.text = "Instructions"
        infoBox.collapsed = True
        self.layout.addWidget(infoBox)
        infoLayout = qt.QVBoxLayout(infoBox)
        infoTextBox = qt.QTextEdit()
        infoTextBox.setReadOnly(True)
        infoTextBox.setPlainText(INFO_TEXT)
        infoLayout.addWidget(infoTextBox)

        self.layout.addStretch(1)

        # ---- Connections ----
        for _, attribute in self.NODE_REFERENCES:
            getattr(self, attribute).connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.spectRoiSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSpectRoiChanged)
        self.referenceRoiSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onReferenceRoiChanged)
        self.showSpectCTButton.connect("clicked()", self.showSpectCT)
        self.showReferenceButton.connect("clicked()", self.showReference)
        self.methodGroup.connect("buttonClicked(QAbstractButton*)", self.updateParameterNodeFromGUI)
        self.qualityComboBox.connect("currentIndexChanged(int)", self.updateParameterNodeFromGUI)
        self.initializationComboBox.connect("currentIndexChanged(int)", self.updateParameterNodeFromGUI)
        self.hardenRigidCheckBox.connect("toggled(bool)", self.updateParameterNodeFromGUI)
        self.registerButton.connect("clicked()", self.onRegisterButton)
        self.pathGroup.connect("buttonClicked(QAbstractButton*)", self.onPathChanged)
        self.spectSelector.connect("currentNodeChanged(vtkMRMLNode*)", self._syncMovingLandmarksParent)
        self.liverSegmentationSelector.connect("currentNodeChanged(vtkMRMLNode*)", self._refreshLiverSegments)
        self.liverSegmentComboBox.connect("currentIndexChanged(int)", self.updateParameterNodeFromGUI)
        self.outlinePercentSpinBox.connect("valueChanged(double)", self.updateParameterNodeFromGUI)
        self.maskMarginSpinBox.connect("valueChanged(double)", self.updateParameterNodeFromGUI)
        self.outlineButton.connect("clicked()", self.onAlignOutlines)
        self.landmarkButton.connect("clicked()", self.onAlignLandmarks)
        self.centreButton.connect("clicked()", self.onCentreOnReference)
        self.resetPositionButton.connect("clicked()", self.onResetPosition)
        self.handlesButton.connect("toggled(bool)", self.onHandlesToggled)
        self.transformsModuleButton.connect("clicked()", self.onTransformsModule)
        self.refineButton.connect("clicked()", self.onRefine)
        self.refineCancelButton.connect("clicked()", self.onCancelButton)
        self.cancelButton.connect("clicked()", self.onCancelButton)
        self.fusionLayoutButton.connect("clicked()", self.showFusionLayout)
        self.opacitySlider.connect("valueChanged(double)", setForegroundOpacity)
        self.hardenButton.connect("clicked()", self.onHardenButton)
        self.undoButton.connect("clicked()", self.onUndoButton)
        self.fineTuneButton.connect("clicked()", self.onFineTuneButton)

        self._elapsedTimer = qt.QTimer()
        self._elapsedTimer.setInterval(1000)
        self._elapsedTimer.connect("timeout()", self._updateElapsed)

        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)
        self.initializeParameterNode()

    def _makeNodeSelector(self, nodeType, toolTip, allowNone=False, allowCreate=False):
        selector = slicer.qMRMLNodeComboBox()
        selector.nodeTypes = [nodeType]
        selector.selectNodeUponCreation = True
        selector.addEnabled = allowCreate
        selector.removeEnabled = allowCreate
        selector.noneEnabled = allowNone or allowCreate
        selector.showHidden = False
        selector.showChildNodeTypes = False
        selector.setMRMLScene(slicer.mrmlScene)
        selector.setToolTip(toolTip)
        return selector

    def _makePlaceWidget(self):
        widget = slicer.qSlicerMarkupsPlaceWidget()
        widget.setMRMLScene(slicer.mrmlScene)
        widget.placeMultipleMarkups = slicer.qSlicerMarkupsPlaceWidget.ForcePlaceMultipleMarkups
        widget.setToolTip("Toggle placement, then click the points in the slice views.")
        try:
            widget.connect("activeMarkupsPlaceModeChanged(bool)", self._onLandmarkPlaceMode)
        except Exception as e:
            logging.debug(f"EasyReg: no place mode signal: {e}")
        return widget

    def _onLandmarkPlaceMode(self, placing):
        """The landmarks are shown (and can be placed) only in EasyReg's views: when placement starts while another
        layout is shown (e.g. the Taranis segmentation layout), switch to the landmark layout."""
        if not placing or not self.isFunctional():
            return
        layoutManager = slicer.app.layoutManager()
        if layoutManager is not None and layoutManager.layout != FUNCTIONAL_LAYOUT_ID:
            # from the event loop: inside the button's click handling the new views are not created yet
            qt.QTimer.singleShot(0, self._showLandmarkLayout)

    def _showLandmarkLayout(self):
        try:
            self._showFunctionalLayout()
        except Exception as e:
            logging.warning(f"EasyReg: could not show the landmark layout: {e}")

    @staticmethod
    def _row(*widgets):
        row = qt.QHBoxLayout()
        for widget in widgets:
            row.addWidget(widget)
        return row

    # ---- Module lifecycle and parameter node ----

    def cleanup(self):
        self.logic.cancel()
        self.sliceSync.clear()
        self.removeObservers()

    def enter(self):
        self.initializeParameterNode()
        # show the images in EasyReg's own layout right away (after the inputs are restored or filled in)
        qt.QTimer.singleShot(0, self._showLayoutOnEnter)

    def _showLayoutOnEnter(self):
        if not self.parent.isEntered:
            return
        try:
            self.showFusionLayout()
        except Exception as e:
            logging.warning(f"EasyReg: could not show the registration layout: {e}")

    def onSceneStartClose(self, caller, event):
        self.logic.cancel()
        self.sliceSync.clear()
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event):
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self):
        parameterNode = self.logic.getParameterNode()
        self.logic.setDefaultParameters(parameterNode)
        self.setParameterNode(parameterNode)

    def setParameterNode(self, parameterNode):
        if self._parameterNode is not None:
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
        self._parameterNode = parameterNode
        if parameterNode is not None:
            self.addObserver(parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
        self.updateGUIFromParameterNode()

    def updateGUIFromParameterNode(self, caller=None, event=None):
        if self._parameterNode is None or self._updatingGUI:
            return
        self._updatingGUI = True
        try:
            p = self._parameterNode
            for role, attribute in self.NODE_REFERENCES:
                selector = getattr(self, attribute)
                node = p.GetNodeReference(role)
                if node is not None or selector.noneEnabled:
                    selector.setCurrentNode(node)
            method = p.GetParameter("Method")
            if method in self.methodButtons:
                self.methodButtons[method].setChecked(True)
            path = p.GetParameter("Path") or PATH_HYBRID
            if path in self.pathButtons:
                self.pathButtons[path].setChecked(True)
            self._refreshLiverSegments()
            index = self.liverSegmentComboBox.findData(p.GetParameter("LiverSegmentID"))
            if index >= 0:
                self.liverSegmentComboBox.setCurrentIndex(index)
            for key, spinBox in (("OutlinePercent", self.outlinePercentSpinBox),
                                 ("MaskMargin", self.maskMarginSpinBox)):
                if p.GetParameter(key):
                    spinBox.value = float(p.GetParameter(key))
            self.fixedLandmarksPlace.setCurrentNode(p.GetNodeReference("LandmarksFixed"))
            self.movingLandmarksPlace.setCurrentNode(p.GetNodeReference("LandmarksMoving"))
            self._setComboData(self.qualityComboBox, p.GetParameter("Quality"))
            self._setComboData(self.initializationComboBox, p.GetParameter("Initialization"))
            self.hardenRigidCheckBox.setChecked(p.GetParameter("HardenRigid") == "true")
        finally:
            self._updatingGUI = False
        self._applyPath()
        self.updateButtonStates()

    def updateParameterNodeFromGUI(self, *args):
        if self._parameterNode is None or self._updatingGUI:
            return
        p = self._parameterNode
        wasModified = p.StartModify()
        try:
            for role, attribute in self.NODE_REFERENCES:
                p.SetNodeReferenceID(role, getattr(self, attribute).currentNodeID)
            p.SetParameter("Method", self.currentMethod())
            p.SetParameter("Quality", self.qualityComboBox.currentData)
            p.SetParameter("Initialization", self.initializationComboBox.currentData)
            p.SetParameter("HardenRigid", "true" if self.hardenRigidCheckBox.checked else "false")
            p.SetParameter("Path", self.currentPath())
            p.SetParameter("LiverSegmentID", self.liverSegmentComboBox.currentData or "")
            p.SetParameter("OutlinePercent", str(self.outlinePercentSpinBox.value))
            p.SetParameter("MaskMargin", str(self.maskMarginSpinBox.value))
        finally:
            p.EndModify(wasModified)
        self.updateButtonStates()

    @staticmethod
    def _setComboData(comboBox, data):
        index = comboBox.findData(data)
        if index >= 0:
            comboBox.setCurrentIndex(index)

    def currentMethod(self):
        for name, button in self.methodButtons.items():
            if button.isChecked():
                return name
        return "Rigid"

    def currentPath(self):
        return PATH_FUNCTIONAL if self.pathButtons[PATH_FUNCTIONAL].isChecked() else PATH_HYBRID

    def isFunctional(self):
        return self.currentPath() == PATH_FUNCTIONAL

    def movingVolume(self):
        """The image that is registered: the CT of the hybrid scan, or the SPECT/PET itself (functional-only)."""
        return self.spectSelector.currentNode() if self.isFunctional() else self.spectCTSelector.currentNode()

    def setPath(self, path):
        """Select the registration path (also used by the Taranis workflow)."""
        self.pathButtons[path].setChecked(True)
        self.onPathChanged()

    def onPathChanged(self, *args):
        # store the path first: _applyPath may create the landmark nodes, which refreshes the GUI from the
        # parameter node and would otherwise switch the path back
        if self._parameterNode is not None and not self._updatingGUI:
            self._parameterNode.SetParameter("Path", self.currentPath())
        self._applyPath()
        self.updateParameterNodeFromGUI()

    def _applyPath(self):
        functional = self.isFunctional()
        for widget in (self.spectCTLabel, self.spectCTSelector, self.segmentationLabel, self.segmentationSelector):
            widget.visible = not functional
        for widget in (self.liverSegmentationLabel, self.liverSegmentationSelector, self.liverSegmentLabel,
                       self.liverSegmentComboBox):
            widget.visible = functional
        self.spectLabel.text = "SPECT / PET (moving): " if functional else "SPECT / PET (follows): "
        self.spectRoiLabel.text = "ROI on SPECT/PET: " if functional else "ROI on SPECT/CT: "
        self.registerBox.visible = not functional
        self.functionalBox.visible = functional
        if functional:
            self._ensureLandmarkNodes()

    def _refreshLiverSegments(self, *args):
        segmentationNode = self.liverSegmentationSelector.currentNode()
        previous = self.liverSegmentComboBox.currentData
        wasBlocked = self.liverSegmentComboBox.blockSignals(True)
        self.liverSegmentComboBox.clear()
        guess = None
        if segmentationNode is not None:
            segmentation = segmentationNode.GetSegmentation()
            for index in range(segmentation.GetNumberOfSegments()):
                segmentID = segmentation.GetNthSegmentID(index)
                name = segmentation.GetSegment(segmentID).GetName()
                self.liverSegmentComboBox.addItem(name, segmentID)
                lowered = name.lower()
                if guess is None and "liver" in lowered and not any(w in lowered for w in ("normal", "tumo", "perfus")):
                    guess = segmentID
        index = self.liverSegmentComboBox.findData(previous)
        if index < 0 and guess is not None:
            index = self.liverSegmentComboBox.findData(guess)
        if index >= 0:
            self.liverSegmentComboBox.setCurrentIndex(index)
        self.liverSegmentComboBox.blockSignals(wasBlocked)
        if not self._updatingGUI:
            self.updateParameterNodeFromGUI()

    def liverSegment(self):
        """(segmentation node, segment ID) of the liver on the reference, or (None, None)."""
        segmentationNode = self.liverSegmentationSelector.currentNode()
        segmentID = self.liverSegmentComboBox.currentData if segmentationNode is not None else None
        return (segmentationNode, segmentID) if segmentID else (None, None)

    # ---- Landmarks ----

    def _ensureLandmarkNodes(self):
        if self._parameterNode is None:
            return
        for role, name, labelFormat, color, placeWidget in (
                ("LandmarksFixed", "EasyReg landmarks on reference", "R%d", LANDMARK_FIXED_COLOR,
                 self.fixedLandmarksPlace),
                ("LandmarksMoving", "EasyReg landmarks on SPECT-PET", "S%d", LANDMARK_MOVING_COLOR,
                 self.movingLandmarksPlace)):
            node = self._parameterNode.GetNodeReference(role)
            if node is None:
                node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", name)
                node.CreateDefaultDisplayNodes()
                node.SetControlPointLabelFormat(labelFormat)
                display = node.GetDisplayNode()
                display.SetSelectedColor(*color)
                display.SetColor(*color)
                self._parameterNode.SetNodeReferenceID(role, node.GetID())
            placeWidget.setCurrentNode(node)
            for event in (slicer.vtkMRMLMarkupsNode.PointAddedEvent, slicer.vtkMRMLMarkupsNode.PointRemovedEvent):
                if not self.hasObserver(node, event, self._updateLandmarkCounts):
                    self.addObserver(node, event, self._updateLandmarkCounts)
        self._syncMovingLandmarksParent()
        self._updateLandmarkCounts()

    def landmarkNodes(self):
        if self._parameterNode is None:
            return None, None
        return (self._parameterNode.GetNodeReference("LandmarksFixed"),
                self._parameterNode.GetNodeReference("LandmarksMoving"))

    def _syncMovingLandmarksParent(self, *args):
        """The SPECT/PET landmarks are kept under the same transform as the SPECT/PET, so they stay on it."""
        _, moving = self.landmarkNodes()
        spect = self.spectSelector.currentNode()
        if moving is not None and spect is not None and moving.GetTransformNodeID() != spect.GetTransformNodeID():
            moving.SetAndObserveTransformNodeID(spect.GetTransformNodeID())

    def _updateLandmarkCounts(self, *args):
        fixed, moving = self.landmarkNodes()
        fixedCount = fixed.GetNumberOfControlPoints() if fixed else 0
        movingCount = moving.GetNumberOfControlPoints() if moving else 0
        self.fixedLandmarksCount.text = str(fixedCount)
        self.movingLandmarksCount.text = str(movingCount)
        self.landmarkButton.enabled = (not self.logic.isBusy() and fixedCount >= F.MIN_LANDMARK_PAIRS
                                       and fixedCount == movingCount)

    def _functionalFollowers(self):
        _, moving = self.landmarkNodes()
        return [moving] if moving is not None else []

    def _checkFunctionalInputs(self):
        spect, reference = self.spectSelector.currentNode(), self.referenceSelector.currentNode()
        if spect is None or reference is None:
            slicer.util.errorDisplay("Select the SPECT/PET and the reference image.")
            return None, None
        if spect is reference:
            slicer.util.errorDisplay("The SPECT/PET and the reference must be different images.")
            return None, None
        return spect, reference

    def _afterAlignment(self, transform, text):
        if self._parameterNode is not None:
            self._parameterNode.SetNodeReferenceID("Transform", transform.GetID())
        self.functionalStatusLabel.text = text
        self.showFusionLayout()
        self.updateButtonStates()

    def onAlignOutlines(self):
        spect, reference = self._checkFunctionalInputs()
        if spect is None:
            return
        liverSegmentation, liverSegmentID = self.liverSegment()
        try:
            with slicer.util.WaitCursor():
                transform, message = self.logic.alignBodyOutlines(
                    spect, reference, self.outlinePercentSpinBox.value, liverSegmentation, liverSegmentID,
                    followers=self._functionalFollowers())
        except (ValueError, RuntimeError) as error:
            slicer.util.errorDisplay(f"Body outline alignment failed: {error}")
            return
        self._afterAlignment(transform, message)

    def onAlignLandmarks(self):
        spect, reference = self._checkFunctionalInputs()
        if spect is None:
            return
        fixed, moving = self.landmarkNodes()
        try:
            transform, rms, residuals = self.logic.alignLandmarks(spect, reference, moving, fixed,
                                                                  followers=[])
        except (ValueError, RuntimeError) as error:
            slicer.util.errorDisplay(f"Landmark alignment failed: {error}")
            return
        worst = int(np.argmax(residuals)) + 1
        self.landmarkResultLabel.text = (f"RMS distance {rms:.1f} mm over {len(residuals)} pairs; largest "
                                         f"{residuals.max():.1f} mm (pair {worst}). "
                                         + ("Check that the pairs match." if rms > 10 else ""))
        self._afterAlignment(transform, f"Aligned with {len(residuals)} landmark pairs (RMS {rms:.1f} mm).")

    def onHandlesToggled(self, checked):
        spect, reference = self.spectSelector.currentNode(), self.referenceSelector.currentNode()
        if checked and (spect is None or reference is None):
            self.handlesButton.checked = False
            slicer.util.errorDisplay("Select the SPECT/PET and the reference image.")
            return
        transform = None
        if checked:
            try:
                transform = self.logic.alignmentTransform(spect, reference, self._functionalFollowers())
            except ValueError as error:
                self.handlesButton.checked = False
                slicer.util.errorDisplay(str(error))
                return
            method = transform.GetAttribute(METHOD_ATTRIBUTE)
            if not method:
                self.logic.alignmentTransform(spect, reference, self._functionalFollowers(), method="Manual")
            elif "manual" not in method.lower():
                transform.SetAttribute(METHOD_ATTRIBUTE, f"{method} + manual")
            if self._parameterNode is not None:
                self._parameterNode.SetNodeReferenceID("Transform", transform.GetID())
        else:
            transform = self.currentTransform()
        if transform is None:
            return
        transform.CreateDefaultDisplayNodes()
        display = transform.GetDisplayNode()
        if display is not None:
            display.SetEditorVisibility(checked)
            if hasattr(display, "SetEditorSliceIntersectionVisibility"):
                display.SetEditorSliceIntersectionVisibility(checked)
        if checked:
            self.showFusionLayout()
        self.updateButtonStates()

    def onTransformsModule(self):
        spect, reference = self._checkFunctionalInputs()
        if spect is None:
            return
        transform = self.logic.alignmentTransform(spect, reference, self._functionalFollowers())
        if not transform.GetAttribute(METHOD_ATTRIBUTE):
            self.logic.alignmentTransform(spect, reference, self._functionalFollowers(), method="Manual")
        if self._parameterNode is not None:
            self._parameterNode.SetNodeReferenceID("Transform", transform.GetID())
        self.onFineTuneButton()

    def onRefine(self):
        spect, reference = self._checkFunctionalInputs()
        if spect is None:
            return
        liverSegmentation, liverSegmentID = self.liverSegment()
        transform = spect.GetParentTransformNode()
        if transform is None or not transform.GetAttribute(METHOD_ATTRIBUTE):
            if not slicer.util.confirmOkCancelDisplay(
                    "The SPECT/PET has no initial alignment yet (body outline, landmarks or by hand). The refinement "
                    "only corrects small misalignments.\n\nRefine from the current position anyway?"):
                return
        self._refineStart = self.logic.transformMatrix(transform) if transform is not None else np.eye(4)
        try:
            self._refineCentre = self.logic.liverCentre(liverSegmentation, liverSegmentID, reference) \
                if liverSegmentation is not None else None
            self.logic.refineInLiver(
                reference, spect, liverSegmentation, liverSegmentID, self.maskMarginSpinBox.value,
                dict(QUALITY_PRESETS)[self.refineQualityComboBox.currentData],
                followers=self._functionalFollowers(), onFinished=self.onRefineFinished)
        except (ValueError, RuntimeError) as error:
            slicer.util.errorDisplay(str(error))
            return
        self._startTime = time.time()
        self._statusPrefix = "Rigid refinement inside the liver running"
        self._updateElapsed()
        self._elapsedTimer.start()
        self.updateButtonStates()

    def onRefineFinished(self, result):
        self._elapsedTimer.stop()
        elapsed = f" in {int(time.time() - self._startTime)} s" if self._startTime else ""
        self._startTime = None
        if result["status"] != "completed":
            text = ("Refinement cancelled." if result["status"] == "cancelled" else "Refinement failed.") + \
                " The images were not changed."
            self.functionalStatusLabel.text = text
            if result["status"] == "failed":
                slicer.util.errorDisplay(text, detailedText=result["message"])
            self.updateButtonStates()
            return
        transform = result["transform"]
        if self._parameterNode is not None:
            self._parameterNode.SetNodeReferenceID("Transform", transform.GetID())
        text = f"Refinement completed{elapsed}."
        warning = ""
        if transform.IsLinear():
            centre = self._refineCentre if self._refineCentre is not None else defaultRoiGeometry(
                self.referenceSelector.currentNode())[0]
            displacement, angle = F.changeBetween(self._refineStart, self.logic.transformMatrix(transform), centre)
            text += f" It moved the liver region by {displacement:.1f} mm and rotated by {angle:.1f}\u00b0."
            if displacement > REFINE_WARNING_MM or angle > REFINE_WARNING_DEGREES:
                warning = ("The refinement changed the alignment a lot. It may have converged to a wrong position: "
                           "check carefully, and use Undo registration if it is worse.")
        self.functionalStatusLabel.text = text + (" " + warning if warning else "")
        self.showFusionLayout()
        self.updateButtonStates()
        if warning:
            qt.QTimer.singleShot(200, lambda: slicer.util.warningDisplay(warning, windowTitle="EasyReg"))

    def currentTransform(self):
        return self._parameterNode.GetNodeReference("Transform") if self._parameterNode else None

    # ---- Bed position ----

    def _positionFollowers(self):
        """What moves with the moving image: the SPECT/PET (and a segmentation drawn on the SPECT/CT) on the
        hybrid path, the SPECT/PET landmarks on the functional-only path."""
        if self.isFunctional():
            return self._functionalFollowers()
        return [node for node in (self.spectSelector.currentNode(), self.segmentationSelector.currentNode())
                if node is not None]

    def _easyRegTransformOf(self, node):
        """The EasyReg transform the node is under (the current one first), or None."""
        current = self.currentTransform()
        if current is not None and current.GetScene() is not None:
            return current
        parent = node.GetParentTransformNode() if node is not None else None
        if parent is not None and parent.GetAttribute(TRANSFORM_ATTRIBUTE) == "1":
            return parent
        return None

    def onCentreOnReference(self):
        moving, reference = self.movingVolume(), self.referenceSelector.currentNode()
        if moving is None or reference is None:
            slicer.util.errorDisplay("Select the moving image and the reference image first.")
            return
        if moving is reference:
            slicer.util.errorDisplay("The moving image and the reference must be different images.")
            return
        try:
            shift = self.logic.centreOnReference(moving, reference, self._positionFollowers())
        except ValueError as error:
            slicer.util.errorDisplay(str(error))
            return
        transform = self._easyRegTransformOf(moving)
        if self._parameterNode is not None and transform is not None:
            self._parameterNode.SetNodeReferenceID("Transform", transform.GetID())
        self.positionStatusLabel.text = (f"'{moving.GetName()}' moved by {np.linalg.norm(shift):.0f} mm "
                                         f"({shift[0]:+.0f}, {shift[1]:+.0f}, {shift[2]:+.0f} mm R-A-S) onto the "
                                         "reference. Now register or align it; 'Reset to original position' "
                                         "undoes this.")
        self.showFusionLayout()
        self.updateButtonStates()

    def onResetPosition(self):
        moving = self.movingVolume()
        transform = self._easyRegTransformOf(moving)
        if transform is None:
            self.positionStatusLabel.text = "The moving image is already in its original position."
            return
        method = transform.GetAttribute(METHOD_ATTRIBUTE) or ""
        if method and method != CENTRE_METHOD and not slicer.util.confirmOkCancelDisplay(
                f"Return the images to their original position? The current alignment ({method}) is removed.",
                windowTitle="EasyReg"):
            return
        with slicer.util.WaitCursor():
            restored = self.logic.undoRegistration(transform)
            removeOverlayVolumes()
            self.sliceSync.clear()
        if self._parameterNode is not None:
            self._parameterNode.SetNodeReferenceID("Transform", None)
        self.positionStatusLabel.text = ("Back to the original position: " + ", ".join(restored) + "."
                                         if restored else "EasyReg's transform removed.")
        self.showSpectCT()
        self.updateButtonStates()

    def updateButtonStates(self):
        busy = self.logic.isBusy()
        moving, fixed = self.movingVolume(), self.referenceSelector.currentNode()
        self.centreButton.enabled = (not busy and moving is not None and fixed is not None and moving is not fixed)
        self.resetPositionButton.enabled = not busy and self._easyRegTransformOf(moving) is not None
        spectCT = self.movingVolume()
        reference = self.referenceSelector.currentNode()
        functionalReady = (self.isFunctional() and not busy and spectCT is not None and reference is not None
                           and spectCT is not reference)
        for button in (self.outlineButton, self.transformsModuleButton, self.refineButton):
            button.enabled = functionalReady
        self.refineButton.enabled = functionalReady and self.liverSegment()[1] is not None
        self.refineButton.setToolTip(
            "Rigid mutual-information registration restricted to the dilated liver, starting from the current "
            "alignment." if self.liverSegment()[1] else "Select the liver segmentation and segment on the "
            "reference first.")
        self.refineCancelButton.enabled = busy and self.isFunctional()
        self.handlesButton.enabled = functionalReady or self.handlesButton.checked
        self._updateLandmarkCounts()
        self.registerButton.enabled = (not busy and spectCT is not None and reference is not None
                                       and spectCT is not reference)
        self.cancelButton.enabled = busy

        transform = self.currentTransform()
        observed = bool(nodesUnderTransform(transform))
        undoable = observed or bool(transform and transform.GetAttribute(HARDENED_NODES_ATTRIBUTE))
        self.hardenButton.enabled = not busy and observed
        self.undoButton.enabled = not busy and undoable
        self.fineTuneButton.enabled = not busy and observed and transform.IsLinear()
        self.fusionLayoutButton.enabled = not busy and spectCT is not None and reference is not None
        taranis = hasattr(slicer.modules, TARANIS_MODULE.lower())
        for button in self.workflowButtons.values():
            button.enabled = not busy and taranis
        self._updateResultLabel()

    def _updateResultLabel(self):
        transform = self.currentTransform()
        if transform is None:
            self.resultLabel.text = "No registration yet."
            return
        method = transform.GetAttribute(METHOD_ATTRIBUTE) or "Registration"
        observed = nodesUnderTransform(transform)
        if observed:
            names = ", ".join(node.GetName() for node in observed)
            self.resultLabel.text = (f"{method} transform applied (live) to: {names}. Evaluate the alignment; "
                                     "if it is acceptable, press Harden transform to continue.")
        elif transform.GetAttribute(HARDENED_ATTRIBUTE):
            self.resultLabel.text = f"{method} transform hardened into the images."
        else:
            self.resultLabel.text = f"{method} transform '{transform.GetName()}' is not applied to any image."

    # ---- ROI placement ----

    def onSpectRoiChanged(self, roiNode):
        if self._updatingGUI or roiNode is None:
            return
        initializeRoi(roiNode, self.movingVolume(), SPECT_ROI_COLOR)
        self.showSpectCT()

    def onReferenceRoiChanged(self, roiNode):
        if self._updatingGUI or roiNode is None:
            return
        initializeRoi(roiNode, self.referenceSelector.currentNode(), REFERENCE_ROI_COLOR)
        self.showReference()

    def showSpectCT(self):
        spect = self.spectSelector.currentNode()
        if self.isFunctional():
            if spect is None:
                return
            self._leaveFusionLayout()
            setColormap(spect, SPECT_COLORMAP)
            showLayers(spect, None)
            setRoiVisible(self.spectRoiSelector.currentNode(), True)
            setRoiVisible(self.referenceRoiSelector.currentNode(), False)
            return
        spectCT = self.spectCTSelector.currentNode()
        if spectCT is None:
            return
        self._leaveFusionLayout()
        setColormap(spectCT, "Grey")
        setColormap(spect, SPECT_COLORMAP)
        showLayers(spectCT, spect, self.opacitySlider.value)
        setRoiVisible(self.spectRoiSelector.currentNode(), True)
        setRoiVisible(self.referenceRoiSelector.currentNode(), False)

    def showReference(self):
        reference = self.referenceSelector.currentNode()
        if reference is None:
            return
        self._leaveFusionLayout()
        setColormap(reference, "Grey")
        showLayers(reference, None)
        setRoiVisible(self.referenceRoiSelector.currentNode(), True)
        setRoiVisible(self.spectRoiSelector.currentNode(), False)

    # ---- Registration ----

    def onRegisterButton(self):
        spectCT = self.spectCTSelector.currentNode()
        reference = self.referenceSelector.currentNode()
        spect = self.spectSelector.currentNode()
        segmentation = self.segmentationSelector.currentNode()
        if spectCT is None or reference is None:
            slicer.util.errorDisplay("Select the CT of the SPECT/CT and the reference image.")
            return
        if spectCT is reference:
            slicer.util.errorDisplay("The CT of the SPECT/CT and the reference must be different images.")
            return
        if spect in (spectCT, reference):
            slicer.util.errorDisplay("The SPECT must be a different image from the CT and the reference.")
            return

        method = self.currentMethod()
        startsFromTransform = spectCT.GetParentTransformNode() is not None
        try:
            self.logic.startRegistration(
                fixedVolume=reference, movingVolume=spectCT, method=method,
                samplingPercentage=dict(QUALITY_PRESETS)[self.qualityComboBox.currentData],
                initializeMode=self.initializationComboBox.currentData,
                fixedRoi=self.referenceRoiSelector.currentNode(), movingRoi=self.spectRoiSelector.currentNode(),
                followers=[node for node in (spect, segmentation) if node is not None],
                hardenRigid=self.hardenRigidCheckBox.checked,
                onFinished=self.onRegistrationFinished)
        except (ValueError, RuntimeError) as error:
            slicer.util.errorDisplay(str(error))
            return

        setRoiVisible(self.spectRoiSelector.currentNode(), False)
        setRoiVisible(self.referenceRoiSelector.currentNode(), False)
        self._startTime = time.time()
        self._statusPrefix = f"{method} registration running"
        if startsFromTransform:
            self._statusPrefix += " (starting from the current transform of the CT)"
        self.progressBar.visible = True
        self._updateElapsed()
        self._elapsedTimer.start()
        self.updateButtonStates()

    def _updateElapsed(self):
        if self._startTime is not None:
            label = self.functionalStatusLabel if self.isFunctional() else self.statusLabel
            label.text = f"{self._statusPrefix}... {int(time.time() - self._startTime)} s"

    def onCancelButton(self):
        self.logic.cancel()
        self.statusLabel.text = "Cancelling..."

    def onRegistrationFinished(self, result):
        self._elapsedTimer.stop()
        self.progressBar.visible = False
        elapsed = f" in {int(time.time() - self._startTime)} s" if self._startTime else ""
        self._startTime = None
        if result["status"] == "completed":
            if self._parameterNode is not None:
                self._parameterNode.SetNodeReferenceID("Transform", result["transform"].GetID())
            self.statusLabel.text = f"Registration completed{elapsed}. {result['message']}".strip()
            self.showFusionLayout()
            # after the event loop has drawn the new layout, so the user sees the images behind the message
            qt.QTimer.singleShot(200, lambda: self._promptEvaluation(result["transform"]))
        elif result["status"] == "cancelled":
            self.statusLabel.text = "Registration cancelled. The images were not changed."
        else:
            self.statusLabel.text = "Registration failed. The images were not changed."
            slicer.util.errorDisplay("Registration failed. The images were not changed.",
                                     detailedText=result["message"])
        self.updateButtonStates()

    # ---- Check and finish ----

    def _promptEvaluation(self, transform):
        if transform is None or transform.GetScene() is None:
            return
        views = "\n".join(f"  {i + 1}. {label}" for i, (label, _, _) in enumerate(FUSION_VIEWS))
        if transform.GetAttribute(HARDENED_ATTRIBUTE):
            action = ("The rigid transform was hardened automatically. If the alignment is not acceptable, "
                      "press Undo registration.")
        else:
            action = ("If the alignment is acceptable, press 'Harden transform', then 'Next' to continue.\n"
                      "If it is not acceptable, press 'Undo registration' and change the ROIs or the method, "
                      "or use 'Fine-tune manually'.")
        slicer.util.infoDisplay(
            "Registration completed. Please evaluate it before continuing.\n\n"
            "Views (axial top row, coronal bottom row; scrolling and zoom are synchronised within each row):\n"
            f"{views}\n\n"
            "Check the liver dome, liver edges, spine and kidneys, and use the overlay opacity slider.\n\n"
            + action, windowTitle="EasyReg - evaluate registration")

    def _leaveFusionLayout(self):
        layoutManager = slicer.app.layoutManager()
        if layoutManager.layout in (FUSION_LAYOUT_ID, FUNCTIONAL_LAYOUT_ID):
            layoutManager.setLayout(self._previousLayout or slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)

    def _refreshOverlay(self):
        """Re-sync the overlay copy after the CT changed (hardening replaces geometry or voxel data)."""
        spectCT = self.spectCTSelector.currentNode()
        if spectCT is not None and any(isOverlayVolume(n) for n in slicer.util.getNodesByClass(
                "vtkMRMLScalarVolumeNode")):
            updateOverlayVolume(spectCT, OVERLAY_CT_COLORMAP)

    def showFusionLayout(self):
        if self.isFunctional():
            self._showFunctionalLayout()
            return
        reference = self.referenceSelector.currentNode()
        spectCT = self.spectCTSelector.currentNode()
        spect = self.spectSelector.currentNode()
        if reference is None or spectCT is None:
            return
        registerFusionLayout()
        layoutManager = slicer.app.layoutManager()
        if layoutManager.layout not in (FUSION_LAYOUT_ID, FUNCTIONAL_LAYOUT_ID):
            self._previousLayout = layoutManager.layout
            layoutManager.setLayout(FUSION_LAYOUT_ID)
        slicer.app.processEvents()  # create the slice widgets of the new layout

        setColormap(reference, "Grey")
        setColormap(spectCT, "Grey")
        setColormap(spect, SPECT_COLORMAP)
        volumes = {"spectCT": spectCT, "spect": spect, "reference": reference,
                   "overlayCT": updateOverlayVolume(spectCT, OVERLAY_CT_COLORMAP)}
        setRoiVisible(self.spectRoiSelector.currentNode(), False)
        setRoiVisible(self.referenceRoiSelector.currentNode(), False)

        # Centre on the reference ROI (liver) when there is one, otherwise on the reference image
        center = defaultRoiGeometry(reference)[0]
        referenceRoi = self.referenceRoiSelector.currentNode()
        if referenceRoi is not None and referenceRoi.GetNumberOfControlPoints() > 0:
            roiCenter = [0.0, 0.0, 0.0]
            referenceRoi.GetCenterWorld(roiCenter)
            center = roiCenter

        opacity = self.opacitySlider.value
        groups, referenceViews = [], []
        for orientation, _ in FUSION_ORIENTATIONS:
            group = []
            for index, (_, backgroundKey, foregroundKey) in enumerate(FUSION_VIEWS):
                sliceWidget = layoutManager.sliceWidget(fusionViewTag(orientation, index))
                if sliceWidget is None:
                    continue
                compositeNode = sliceWidget.mrmlSliceCompositeNode()
                foreground = volumes.get(foregroundKey) if foregroundKey else None
                compositeNode.SetBackgroundVolumeID(volumes[backgroundKey].GetID())
                compositeNode.SetForegroundVolumeID(foreground.GetID() if foreground else None)
                compositeNode.SetForegroundOpacity(opacity)
                compositeNode.SetLinkedControl(False)
                group.append(sliceWidget.mrmlSliceNode())
                if backgroundKey == "reference" and foregroundKey is None:
                    sliceWidget.sliceController().fitSliceToBackground()
                    referenceViews.append((sliceWidget.mrmlSliceNode(), len(groups)))
            groups.append(group)

        self.sliceSync.setGroups(groups)
        for sliceNode, groupIndex in referenceViews:
            self.sliceSync.syncFrom(sliceNode, groups[groupIndex])  # same zoom as the reference-only view
            sliceNode.JumpSliceByCentering(*center)                 # propagated by the synchronizer

    def _showFunctionalLayout(self):
        """3 x 2 views (axial top, coronal bottom): reference, SPECT/PET alone, SPECT/PET on reference. The
        reference and fusion views are synchronised; the SPECT/PET view is fitted to the SPECT/PET. Reference
        landmarks are shown on the reference and fusion views, SPECT/PET landmarks on the SPECT/PET and fusion
        views."""
        reference = self.referenceSelector.currentNode()
        spect = self.spectSelector.currentNode()
        if reference is None or spect is None:
            return
        registerFusionLayout()
        layoutManager = slicer.app.layoutManager()
        if layoutManager.layout not in (FUSION_LAYOUT_ID, FUNCTIONAL_LAYOUT_ID):
            self._previousLayout = layoutManager.layout
        layoutManager.setLayout(FUNCTIONAL_LAYOUT_ID)
        slicer.app.processEvents()
        setColormap(reference, "Grey")
        setColormap(spect, SPECT_COLORMAP)
        try:
            readableFunctionalWindow(spect)
        except Exception as e:
            logging.debug(f"EasyReg: SPECT/PET window: {e}")
        volumes = {"spect": spect, "reference": reference}
        setRoiVisible(self.spectRoiSelector.currentNode(), False)
        setRoiVisible(self.referenceRoiSelector.currentNode(), False)
        centre = defaultRoiGeometry(reference)[0]
        liverSegmentation, liverSegmentID = self.liverSegment()
        if liverSegmentation is not None:
            try:
                centre = list(self.logic.liverCentre(liverSegmentation, liverSegmentID, reference))
            except ValueError:
                pass
        opacity = self.opacitySlider.value
        groups, fitViews, viewIDs = [], [], {0: [], 1: [], 2: []}
        for orientation, _ in FUSION_ORIENTATIONS:
            group = []
            for index, (_, backgroundKey, foregroundKey) in enumerate(FUNCTIONAL_VIEWS):
                sliceWidget = layoutManager.sliceWidget(functionalViewTag(orientation, index))
                if sliceWidget is None:
                    continue
                compositeNode = sliceWidget.mrmlSliceCompositeNode()
                compositeNode.SetBackgroundVolumeID(volumes[backgroundKey].GetID())
                compositeNode.SetForegroundVolumeID(volumes[foregroundKey].GetID() if foregroundKey else None)
                compositeNode.SetForegroundOpacity(opacity)
                compositeNode.SetLinkedControl(False)
                sliceNode = sliceWidget.mrmlSliceNode()
                viewIDs[index].append(sliceNode.GetID())
                if backgroundKey == "spect":
                    sliceWidget.sliceController().fitSliceToBackground()
                    fitViews.append((sliceNode, None))
                else:
                    group.append(sliceNode)
                    if foregroundKey is None:
                        sliceWidget.sliceController().fitSliceToBackground()
                        fitViews.append((sliceNode, len(groups)))
            groups.append(group)
        self.sliceSync.setGroups(groups)
        uptake = None
        try:  # the SPECT/PET-only views go through the uptake (the volume centre may be an empty slice)
            array, matrix = subsampled(slicer.util.arrayFromVolume(spect), ijkToWorld(spect))
            uptake = F.weightedCentroid(array, F.hotRegionMask(array, None), matrix)
        except Exception as e:
            logging.debug(f"EasyReg: uptake centre: {e}")
        for sliceNode, groupIndex in fitViews:
            if groupIndex is not None:
                self.sliceSync.syncFrom(sliceNode, groups[groupIndex])
                sliceNode.JumpSliceByCentering(*centre)
        spectViews = [sliceNode for sliceNode, groupIndex in fitViews if groupIndex is None]
        if uptake is not None and spectViews:
            # after the new views have been sized and fitted (a fit right after the layout switch undoes it)
            centre3 = [float(v) for v in uptake]
            for delay in (300, 1200):   # new views are fitted again once they are shown
                qt.QTimer.singleShot(delay, lambda: [n.JumpSliceByCentering(*centre3) for n in spectViews
                                                     if n.GetScene() is not None])
        fixedLandmarks, movingLandmarks = self.landmarkNodes()
        threeD = [node.GetID() for node in slicer.util.getNodesByClass("vtkMRMLViewNode")]
        for node, views in ((fixedLandmarks, viewIDs[0] + viewIDs[2]), (movingLandmarks, viewIDs[1] + viewIDs[2])):
            if node is not None and node.GetDisplayNode() is not None:
                node.GetDisplayNode().SetViewNodeIDs(views + threeD)

    def onHardenButton(self):
        transform = self.currentTransform()
        if transform is None:
            return
        if not isRigidMethod(transform.GetAttribute(METHOD_ATTRIBUTE)):
            if not slicer.util.confirmOkCancelDisplay(
                    "This transform is not rigid. Hardening can resample the images (interpolating the quantitative "
                    "SPECT voxel values) and cannot be undone.\n\nHarden now?"):
                return
        with slicer.util.WaitCursor():
            self.logic.hardenRegistration(transform)
            self._refreshOverlay()
        self.statusLabel.text = "Transform hardened. Press Next to continue the workflow."
        self.updateButtonStates()

    def onUndoButton(self):
        transform = self.currentTransform()
        if transform is None:
            return
        with slicer.util.WaitCursor():
            restored = self.logic.undoRegistration(transform)
            removeOverlayVolumes()
            self.sliceSync.clear()
        if self._parameterNode is not None:
            self._parameterNode.SetNodeReferenceID("Transform", None)
        self.statusLabel.text = ("Registration undone: " + ", ".join(restored) + " returned to the previous "
                                 "position.") if restored else "Registration removed."
        self.showSpectCT()
        self.updateButtonStates()

    def onFineTuneButton(self):
        transform = self.currentTransform()
        if transform is None:
            return
        slicer.util.selectModule("Transforms")
        try:
            slicer.modules.transforms.widgetRepresentation().setEditedNode(transform)
        except Exception as error:  # older Slicer: the user selects the transform manually
            logging.warning(f"Could not select the transform in the Transforms module: {error}")

    def onWorkflowButton(self, key):
        """Go to a step of the Taranis workflow hub (Previous: Data, overview: Registration, Next: Segmentation)."""
        if not hasattr(slicer.modules, TARANIS_MODULE.lower()):
            slicer.util.errorDisplay("The Taranis module is not loaded.")
            return
        if key == "next":
            spect, spectCT = self.spectSelector.currentNode(), self.spectCTSelector.currentNode()
            if (hasPendingRegistration(spect) or hasPendingRegistration(spectCT)) and not \
                    slicer.util.confirmYesNoDisplay(
                        "The registration is not hardened yet (the images still follow a live transform).\n\n"
                        "Continue anyway?", windowTitle="Registration not hardened"):
                return
        slicer.util.selectModule(TARANIS_MODULE)
        try:
            widget = slicer.util.getModuleWidget(TARANIS_MODULE)
            if hasattr(widget, "showStep"):
                widget.showStep(WORKFLOW_STEPS[key])
        except Exception as error:
            logging.warning(f"Could not open the Taranis step: {error}")


# -- Logic -----------------------------------------------------------------------

class easy_regLogic(ScriptedLoadableModuleLogic):

    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)
        self.job = None

    def setDefaultParameters(self, parameterNode):
        defaults = {"Method": "Rigid", "Path": PATH_HYBRID, "Quality": "Standard", "Initialization": "useGeometryAlign",
                    "HardenRigid": "false", "SplineGridSize": ",".join(str(v) for v in DEFAULT_SPLINE_GRID)}
        for name, value in defaults.items():
            if not parameterNode.GetParameter(name):
                parameterNode.SetParameter(name, value)

    def isBusy(self):
        return self.job is not None

    def makeWorkingVolume(self, volumeNode, roiNode, name, forceCopy=False):
        """(volume for registration, is temporary). A temporary copy, cropped to the ROI when one is given,
        without parent transform. The input volume is never modified. Cropping keeps the physical position of
        every voxel, so a transform computed on the copy is valid for the original image."""
        if roiNode is None and not forceCopy:
            return volumeNode, False
        if roiNode is None:
            workingNode = slicer.modules.volumes.logic().CloneVolume(slicer.mrmlScene, volumeNode, name)
        else:
            workingNode = cropVolumeToRoi(volumeNode, roiNode, name)
            if workingNode is None:
                roiBounds, volumeBounds = [0.0] * 6, [0.0] * 6
                roiNode.GetRASBounds(roiBounds)
                volumeNode.GetRASBounds(volumeBounds)
                raise ValueError(
                    f"The ROI '{roiNode.GetName()}' does not overlap '{volumeNode.GetName()}'. Move the ROI onto "
                    f"the image or remove it.\nROI: {formatBounds(roiBounds)} mm\n"
                    f"Image: {formatBounds(volumeBounds)} mm")
        workingNode.SetHideFromEditors(True)
        workingNode.SetSaveWithScene(False)
        workingNode.SetAndObserveTransformNodeID(None)  # the moving image's transform is passed separately
        return workingNode, True

    def startRegistration(self, fixedVolume, movingVolume, method, samplingPercentage,
                          initializeMode="useGeometryAlign", fixedRoi=None, movingRoi=None, followers=(),
                          splineGridSize=DEFAULT_SPLINE_GRID, hardenRigid=False, onFinished=None, wait=False,
                          fixedMask=None, preprocessMoving=False, temporaryNodes=None):
        """Register movingVolume to fixedVolume with BRAINSFit and apply the result to movingVolume and the
        followers. Runs in the background unless wait=True. onFinished(result) receives
        {"status": "completed" | "cancelled" | "failed", "message": str, "transform": node or None}."""
        if self.job is not None:
            raise RuntimeError("A registration is already running.")
        if not hasattr(slicer.modules, "brainsfit"):
            raise RuntimeError("The BRAINSFit module (General Registration) is not available in this Slicer.")
        if method not in METHOD_BY_NAME:
            raise ValueError(f"Unknown registration method '{method}'.")
        transformType, linear, _ = METHOD_BY_NAME[method]

        if fixedVolume.GetParentTransformNode() is not None:
            raise ValueError(f"The reference image '{fixedVolume.GetName()}' is under a transform. "
                             "Harden or remove that transform first (Transforms module).")
        initialTransform = movingVolume.GetParentTransformNode()
        if initialTransform is not None and (not initialTransform.IsLinear()
                                             or initialTransform.GetParentTransformNode() is not None):
            raise ValueError(f"'{movingVolume.GetName()}' is under a deformable or nested transform. Undo the "
                             "previous registration (or harden it) before registering again.")

        temporaryNodes = list(temporaryNodes or [])
        try:
            fixedWorking, isTemporary = self.makeWorkingVolume(fixedVolume, fixedRoi, "EasyReg fixed (temporary)")
            if isTemporary:
                temporaryNodes.append(fixedWorking)
            # The moving image is always copied, so its parent transform can be detached from the copy
            movingWorking, _ = self.makeWorkingVolume(movingVolume, movingRoi, "EasyReg moving (temporary)",
                                                      forceCopy=initialTransform is not None or preprocessMoving)
            if movingWorking is not movingVolume:
                temporaryNodes.append(movingWorking)
            if preprocessMoving:
                # functional image: no negative noise, light smoothing for a smoother mutual-information metric
                from scipy import ndimage
                array = slicer.util.arrayFromVolume(movingWorking)
                array[:] = ndimage.gaussian_filter(np.clip(array.astype(np.float32), 0, None), 1.0)
                slicer.util.arrayFromVolumeModified(movingWorking)
            movingMask = None
            if fixedMask is not None:
                # BRAINSFit's ROI mode needs both masks: the moving mask covers the whole moving image
                movingMask = labelmapFromArray(np.ones(slicer.util.arrayFromVolume(movingWorking).shape, bool),
                                               movingWorking, "EasyReg moving mask (temporary)")
                temporaryNodes.append(movingMask)
        except Exception:
            for node in temporaryNodes:
                removeNodeIfInScene(node)
            raise

        transformClass = "vtkMRMLLinearTransformNode" if linear else "vtkMRMLTransformNode"
        outputTransform = slicer.mrmlScene.AddNewNodeByClass(
            transformClass, f"EasyReg {method}: {movingVolume.GetName()} to {fixedVolume.GetName()}")
        outputTransform.SetAttribute(TRANSFORM_ATTRIBUTE, "1")
        outputTransform.SetAttribute(METHOD_ATTRIBUTE, method)
        if initialTransform is not None:
            outputTransform.SetAttribute(PREVIOUS_TRANSFORM_ATTRIBUTE, initialTransform.GetID())

        parameters = registrationParameters(fixedWorking, movingWorking, outputTransform, transformType,
                                            samplingPercentage, initializeMode, initialTransform, splineGridSize,
                                            fixedMask=fixedMask, movingMask=movingMask)
        logging.info(f"EasyReg: starting {method} registration ({transformType}), sampling {samplingPercentage}")
        self.job = {"method": method, "linear": linear, "moving": movingVolume, "followers": list(followers),
                    "transform": outputTransform, "initialTransform": initialTransform,
                    "temporaryNodes": temporaryNodes, "hardenRigid": hardenRigid, "onFinished": onFinished,
                    "cliNode": None, "observer": None}
        try:
            cliNode = slicer.cli.run(slicer.modules.brainsfit, None, parameters,
                                     wait_for_completion=wait, update_display=False)
        except Exception as error:
            # runs synchronously raise on failure; the error is reported like a failed background run
            return self._finish(errorText=str(error))
        self.job["cliNode"] = cliNode
        if wait or not cliNode.IsBusy():
            return self._finish()
        self.job["observer"] = cliNode.AddObserver(vtk.vtkCommand.ModifiedEvent, self._onCliModified)
        return None

    def cancel(self):
        if self.job is not None and self.job["cliNode"] is not None:
            self.job["cliNode"].Cancel()

    def _onCliModified(self, cliNode, event):
        if self.job is None or cliNode is not self.job["cliNode"] or cliNode.IsBusy():
            return
        self._finish()

    def _finish(self, errorText=""):
        job, self.job = self.job, None
        cliNode = job["cliNode"]
        if cliNode is not None:
            if job["observer"] is not None:
                cliNode.RemoveObserver(job["observer"])
            status = cliNode.GetStatus()
            if status == cliNode.Cancelled:
                state = "cancelled"
            elif status == cliNode.Completed:
                state = "completed"
            else:
                state = "failed"
                errorText = cliNode.GetErrorText() or cliNode.GetStatusString()
            # Removing the CLI node inside its own event is unsafe: remove it on the next event loop cycle
            qt.QTimer.singleShot(0, lambda: removeNodeIfInScene(cliNode))
        else:
            state = "failed"
        for node in job["temporaryNodes"]:
            removeNodeIfInScene(node)

        transform = job["transform"]
        message = ""
        if state == "completed" and transform.GetScene() is not None:
            message = self._applyResult(job)
        else:
            removeNodeIfInScene(transform)
            transform = None
            if state == "failed":
                message = errorText
                logging.error(f"EasyReg: registration failed: {errorText}")
        result = {"status": state, "message": message, "transform": transform}
        if job["onFinished"] is not None:
            job["onFinished"](result)
        return result

    def _applyResult(self, job):
        transform = job["transform"]
        nodes = [job["moving"]] + [node for node in job["followers"] if node.GetScene() is not None]
        for node in nodes:
            node.SetAndObserveTransformNodeID(transform.GetID())
        transform.SetAttribute(REGISTERED_NODES_ATTRIBUTE, ",".join(node.GetID() for node in nodes))
        # The new transform already contains the previous one: a superseded EasyReg transform is removed
        previous = job["initialTransform"]
        if (previous is not None and previous.GetAttribute(TRANSFORM_ATTRIBUTE) == "1"
                and not nodesUnderTransform(previous)):
            transform.SetAttribute(PREVIOUS_TRANSFORM_ATTRIBUTE, previous.GetAttribute(PREVIOUS_TRANSFORM_ATTRIBUTE))
            slicer.mrmlScene.RemoveNode(previous)
        if job["method"] == "Rigid" and job["hardenRigid"]:
            self.hardenRegistration(transform)
            return "Rigid transform hardened (can still be undone)."
        if not job["linear"]:
            return "The deformable transform is kept live: SPECT voxel values are not interpolated."
        return "The transform is applied live (not hardened)."

    def hardenRegistration(self, transformNode):
        nodes = nodesUnderTransform(transformNode)
        rigid = isRigidMethod(transformNode.GetAttribute(METHOD_ATTRIBUTE)) and transformNode.IsLinear()
        for node in nodes:
            slicer.vtkSlicerTransformLogic().hardenTransform(node)
        transformNode.SetAttribute(HARDENED_ATTRIBUTE, "1")
        # Only a rigid transform can be reversed exactly after hardening (inverse matrix, no resampling)
        transformNode.SetAttribute(HARDENED_NODES_ATTRIBUTE,
                                   ",".join(node.GetID() for node in nodes) if rigid else "")
        return nodes

    def undoRegistration(self, transformNode):
        """Return the images to their position before the registration and remove the transform.
        Returns the names of the images that were moved back."""
        restored = []
        previousID = transformNode.GetAttribute(PREVIOUS_TRANSFORM_ATTRIBUTE)
        previous = slicer.mrmlScene.GetNodeByID(previousID) if previousID else None
        observed = nodesUnderTransform(transformNode)
        if observed:
            for node in observed:
                node.SetAndObserveTransformNodeID(previous.GetID() if previous else None)
                restored.append(node.GetName())
        else:
            hardenedIDs = [i for i in (transformNode.GetAttribute(HARDENED_NODES_ATTRIBUTE) or "").split(",") if i]
            hardenedNodes = [slicer.mrmlScene.GetNodeByID(i) for i in hardenedIDs]
            hardenedNodes = [node for node in hardenedNodes if node is not None]
            if hardenedNodes:
                matrix = vtk.vtkMatrix4x4()
                transformNode.GetMatrixTransformToParent(matrix)
                matrix.Invert()
                inverse = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", "EasyReg undo")
                inverse.SetMatrixTransformToParent(matrix)
                for node in hardenedNodes:
                    node.SetAndObserveTransformNodeID(inverse.GetID())
                    slicer.vtkSlicerTransformLogic().hardenTransform(node)
                    if previous is not None:
                        node.SetAndObserveTransformNodeID(previous.GetID())
                    restored.append(node.GetName())
                slicer.mrmlScene.RemoveNode(inverse)
        slicer.mrmlScene.RemoveNode(transformNode)
        return restored


    # ---- Functional-only path (SPECT/PET without its own CT) ----

    def centreOnReference(self, movingVolume, fixedVolume, followers=()):
        """Translate movingVolume (and its followers) so that the centre of its field of view (world, current
        position) lies on the centre of the reference's field of view. Returns the shift (mm, RAS)."""
        if fixedVolume.GetParentTransformNode() is not None:
            raise ValueError(f"The reference image '{fixedVolume.GetName()}' is under a transform. Harden or remove "
                             "it first.")
        movingBounds, fixedBounds = [0.0] * 6, [0.0] * 6
        movingVolume.GetRASBounds(movingBounds)   # world coordinates (current transform included)
        fixedVolume.GetRASBounds(fixedBounds)
        movingCentre = np.array(movingBounds).reshape(3, 2).mean(axis=1)
        fixedCentre = np.array(fixedBounds).reshape(3, 2).mean(axis=1)
        if not (np.all(np.isfinite(movingCentre)) and np.all(np.isfinite(fixedCentre))):
            raise ValueError("The images have no valid extent.")
        shift = fixedCentre - movingCentre
        transform = self.alignmentTransform(movingVolume, fixedVolume, followers)
        self.setTransformMatrix(transform, F.translationMatrix(shift) @ self.transformMatrix(transform))
        self.alignmentTransform(movingVolume, fixedVolume, followers, method=CENTRE_METHOD)
        return shift

    def alignmentTransform(self, movingVolume, fixedVolume, followers=(), method=None):
        """The live EasyReg linear transform that positions movingVolume, created if needed. A new transform
        starts from the current position (an existing non-EasyReg linear parent transform is taken over and
        restored by Undo). Followers are placed under it too. method, if given, is recorded on the transform."""
        parent = movingVolume.GetParentTransformNode()
        if parent is not None and (not parent.IsLinear() or parent.GetParentTransformNode() is not None):
            raise ValueError(f"'{movingVolume.GetName()}' is under a deformable or nested transform. Undo or harden "
                             "it first.")
        if parent is not None and parent.GetAttribute(TRANSFORM_ATTRIBUTE) == "1" \
                and not parent.GetAttribute(HARDENED_ATTRIBUTE):
            transform = parent
        else:
            transform = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLLinearTransformNode", f"EasyReg alignment: {movingVolume.GetName()} to {fixedVolume.GetName()}")
            transform.SetAttribute(TRANSFORM_ATTRIBUTE, "1")
            if parent is not None:
                transform.SetAttribute(PREVIOUS_TRANSFORM_ATTRIBUTE, parent.GetID())
                transform.SetMatrixTransformToParent(numpyToMatrix(transformToWorld(movingVolume)))
        nodes = [movingVolume] + [node for node in followers if node is not None and node.GetScene() is not None]
        for node in nodes:
            if node.GetTransformNodeID() != transform.GetID():
                node.SetAndObserveTransformNodeID(transform.GetID())
        if method:
            transform.SetAttribute(METHOD_ATTRIBUTE, method)
            transform.SetAttribute(REGISTERED_NODES_ATTRIBUTE, ",".join(node.GetID() for node in nodes))
        return transform

    @staticmethod
    def setTransformMatrix(transformNode, matrix):
        transformNode.SetMatrixTransformToParent(numpyToMatrix(matrix))

    @staticmethod
    def transformMatrix(transformNode):
        if transformNode is None:
            return np.eye(4)
        matrix = vtk.vtkMatrix4x4()
        transformNode.GetMatrixTransformToParent(matrix)
        return matrixToNumpy(matrix)

    @staticmethod
    def nativeIjkToRas(volumeNode):
        ijkToRas = vtk.vtkMatrix4x4()
        volumeNode.GetIJKToRASMatrix(ijkToRas)
        return matrixToNumpy(ijkToRas)

    def liverCentre(self, segmentationNode, segmentID, referenceVolume):
        mask = segmentMaskOnVolume(segmentationNode, segmentID, referenceVolume)
        if not mask.any():
            raise ValueError("The liver segment is empty.")
        return F.weightedCentroid(mask.astype(np.float32), mask, ijkToWorld(referenceVolume))

    def alignBodyOutlines(self, movingVolume, fixedVolume, outlinePercent=DEFAULT_OUTLINE_PERCENT,
                          liverSegmentation=None, liverSegmentID=None, followers=()):
        """Initial alignment of a SPECT/PET without its own CT.
        1. Head-feet (with a liver segment on the reference): the uptake centre is placed on the liver centre.
        2. Left-right and anterior-posterior: the SPECT/PET body outline (scatter / background) is centred on the
           reference body outline over the head-feet range both cover. If the SPECT/PET shows no usable outline
           (its cross-section is much smaller than the body's, e.g. MAA without background), lower thresholds are
           tried; if none works, the uptake centre is placed on the liver centre in all three axes.
        Returns (transform, message)."""
        if fixedVolume.GetParentTransformNode() is not None:
            raise ValueError(f"The reference image '{fixedVolume.GetName()}' is under a transform. Harden or remove "
                             "it first.")
        movingArray, movingIjk = subsampled(slicer.util.arrayFromVolume(movingVolume),
                                            self.nativeIjkToRas(movingVolume))
        fixedArray, fixedIjk = subsampled(slicer.util.arrayFromVolume(fixedVolume), ijkToWorld(fixedVolume))
        modality = "CT" if looksLikeCT(fixedVolume) else "MR"
        fixedMask = F.bodyMask(fixedArray, F.anatomicalOutlineThreshold(fixedArray, modality), 1.0)

        transform = self.alignmentTransform(movingVolume, fixedVolume, followers)
        current = self.transformMatrix(transform)
        offset = np.zeros(3)
        notes = []
        liver = uptake = None
        if liverSegmentation is not None and liverSegmentID:
            liver = self.liverCentre(liverSegmentation, liverSegmentID, fixedVolume)
            uptake = F.weightedCentroid(movingArray, F.hotRegionMask(movingArray, None), current @ movingIjk)
            offset[2] = liver[2] - uptake[2]
            notes.append(f"Head-feet: uptake centre placed on the liver centre ({offset[2]:+.1f} mm).")
        else:
            notes.append("Head-feet position unchanged (select a liver segment on the reference, or use landmarks).")
        shifted = F.translationMatrix(offset) @ current @ movingIjk

        percents = [outlinePercent] + [p for p in F.FALLBACK_OUTLINE_PERCENTS if p < outlinePercent]
        chosen, tried = None, []
        for percent in percents:
            try:
                mask = F.bodyMask(movingArray, F.functionalOutlineThreshold(movingArray, percent), 1.0)
            except ValueError:
                continue
            ratio = F.outlineAreaRatio(mask, shifted, fixedMask, fixedIjk)
            tried.append(f"{percent:g} %: {ratio:.2f}")
            if ratio >= F.OUTLINE_AREA_RATIO:
                chosen = (percent, mask, ratio)
                break
        if chosen is not None:
            percent, mask, ratio = chosen
            xy, overlap = F.outlineOffset(F.maskWorldPoints(mask, shifted), F.maskWorldPoints(fixedMask, fixedIjk))
            offset[:2] = xy[:2]
            notes.insert(0, f"Body outlines centred (SPECT/PET outline at {percent:g} % of max, cross-section "
                            f"{100 * ratio:.0f} % of the reference): {offset[0]:+.1f} mm left-right, "
                            f"{offset[1]:+.1f} mm anterior-posterior over {overlap:.0f} mm head-feet.")
        elif liver is not None:
            offset[:2] = (liver - uptake)[:2]
            notes.insert(0, "The SPECT/PET shows no usable body outline (outline / body cross-section: "
                            + ", ".join(tried) + f"). Left-right and anterior-posterior: uptake centre placed on "
                            f"the liver centre ({offset[0]:+.1f}, {offset[1]:+.1f} mm).")
        else:
            raise ValueError("The SPECT/PET shows no usable body outline (outline / body cross-section: "
                             + ", ".join(tried) + "). Select the liver segment on the reference, or use landmarks.")
        notes.append("Check the result; with a lobar or selective injection the uptake is not centred in the liver. "
                     "Refine inside the liver next.")
        self.setTransformMatrix(transform, F.translationMatrix(offset) @ current)
        self.alignmentTransform(movingVolume, fixedVolume, followers, method="Body outline")
        return transform, " ".join(notes)

    def alignLandmarks(self, movingVolume, fixedVolume, movingLandmarks, fixedLandmarks, followers=()):
        """Rigid alignment from landmark pairs (same order on both images). The moving landmarks are placed on the
        SPECT/PET and move with it, so their local coordinates are SPECT coordinates. Returns (transform, rms,
        residuals)."""
        count = movingLandmarks.GetNumberOfControlPoints()
        movingPoints = np.array([movingLandmarks.GetNthControlPointPosition(i) for i in range(count)])
        fixedPoints = np.array([fixedLandmarks.GetNthControlPointPositionWorld(i)
                                for i in range(fixedLandmarks.GetNumberOfControlPoints())])
        # local coordinates of the moving landmarks are relative to the transform they are under; express
        # them in the SPECT's native coordinates
        toSpect = np.linalg.inv(transformToWorld(movingVolume)) @ transformToWorld(movingLandmarks)
        movingNative = (np.c_[movingPoints, np.ones(len(movingPoints))] @ toSpect.T)[:, :3] if count else movingPoints
        matrix, rms, residuals = F.rigidFromLandmarks(movingNative, fixedPoints)
        transform = self.alignmentTransform(movingVolume, fixedVolume, list(followers) + [movingLandmarks])
        self.setTransformMatrix(transform, matrix)
        self.alignmentTransform(movingVolume, fixedVolume, list(followers) + [movingLandmarks], method="Landmarks")
        return transform, rms, residuals

    def liverMaskLabelmap(self, segmentationNode, segmentID, referenceVolume, marginMm):
        mask = segmentMaskOnVolume(segmentationNode, segmentID, referenceVolume)
        if not mask.any():
            raise ValueError("The liver segment is empty.")
        spacing = referenceVolume.GetSpacing()
        grown = F.dilateMask(mask, (spacing[2], spacing[1], spacing[0]), marginMm)
        return labelmapFromArray(grown, referenceVolume, "EasyReg liver mask (temporary)")

    def refineInLiver(self, fixedVolume, movingVolume, liverSegmentation, liverSegmentID, marginMm,
                      samplingPercentage, followers=(), onFinished=None, wait=False):
        """Rigid mutual-information registration restricted to the dilated liver of the reference, starting from
        the current alignment. Affine / deformable are not offered for functional-only registration."""
        if liverSegmentation is None or not liverSegmentID:
            raise ValueError("Select the liver segment on the reference image: the refinement is restricted to it.")
        fixedMask = self.liverMaskLabelmap(liverSegmentation, liverSegmentID, fixedVolume, marginMm)
        # Only the region of the mask is compared: crop a temporary copy of the reference to it (much faster for
        # large MRI / CT volumes; the images themselves are never modified)
        cropRoi = self.maskBoundingRoi(fixedMask, extraMm=10.0)
        temporary = [fixedMask, cropRoi]
        try:
            return self.startRegistration(fixedVolume, movingVolume, REFINE_METHOD, samplingPercentage,
                                          initializeMode="Off", fixedRoi=cropRoi, followers=followers,
                                          onFinished=onFinished, wait=wait, fixedMask=fixedMask,
                                          preprocessMoving=True, temporaryNodes=temporary)
        except Exception:
            for node in temporary:
                removeNodeIfInScene(node)
            raise

    @staticmethod
    def maskBoundingRoi(labelmapNode, extraMm=0.0):
        """Temporary ROI node around the non-zero voxels of a label map (world coordinates)."""
        array = slicer.util.arrayFromVolume(labelmapNode)
        k, j, i = np.nonzero(array)
        if len(i) == 0:
            raise ValueError("The liver mask is empty.")
        corners = np.array([[a, b, c, 1.0] for a in (i.min(), i.max()) for b in (j.min(), j.max())
                            for c in (k.min(), k.max())])
        world = (ijkToWorld(labelmapNode) @ corners.T)[:3].T
        low, high = world.min(axis=0) - extraMm, world.max(axis=0) + extraMm
        roi = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode", "EasyReg liver region (temporary)")
        roi.SetHideFromEditors(True)
        roi.SetSaveWithScene(False)
        roi.SetCenter(*((low + high) / 2.0))
        roi.SetSize(*(high - low))
        roi.CreateDefaultDisplayNodes()
        if roi.GetDisplayNode() is not None:
            roi.GetDisplayNode().SetVisibility(False)
        return roi


# -- Tests -----------------------------------------------------------------------

class easy_regTest(ScriptedLoadableModuleTest):
    """Synthetic-data checks ("Reload and Test" in developer mode)."""

    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_parameters()
        self.setUp()
        self.test_cropKeepsGeometry()
        self.setUp()
        self.test_hardenAndUndo()
        self.setUp()
        self.test_rigidRecoversShift()
        self.setUp()
        self.test_functionalBodyOutline()
        self.setUp()
        self.test_functionalLandmarks()
        self.setUp()
        self.test_functionalRefinement()
        self.setUp()
        self.test_centreOnReference()
        self.delayDisplay("EasyReg tests passed")

    @staticmethod
    def _phantom(name, origin=(0.0, 0.0, 0.0)):
        """Asymmetric body with a 'liver' and a 'spine' (CT-like values), 3 mm voxels."""
        k, j, i = np.mgrid[0:40, 0:56, 0:56].astype(float)
        array = np.full(k.shape, -1000.0)
        body = ((i - 28) / 24) ** 2 + ((j - 28) / 18) ** 2 + ((k - 20) / 17) ** 2 <= 1
        array[body] = 40
        array[(((i - 36) / 9) ** 2 + ((j - 24) / 7) ** 2 + ((k - 22) / 8) ** 2 <= 1)] = 120
        array[body & (((i - 24) / 3.5) ** 2 + ((j - 38) / 3.5) ** 2 <= 1)] = 700
        node = slicer.util.addVolumeFromArray(array.astype(np.float32), name=name)
        node.SetSpacing(3.0, 3.0, 3.0)
        node.SetOrigin(*origin)
        return node

    def test_parameters(self):
        p = registrationParameters("f", "m", "t", "Rigid,Affine,BSpline", 0.01, splineGridSize=(8, 9, 10))
        assert p["initializeTransformMode"] == "useGeometryAlign" and p["splineGridSize"] == "8,9,10"
        p = registrationParameters("f", "m", "t", "Rigid", 0.01, initialTransform="i")
        assert p["initializeTransformMode"] == "Off" and p["initialTransform"] == "i"
        assert "splineGridSize" not in p
        self.delayDisplay("Parameters OK")

    def test_cropKeepsGeometry(self):
        roi = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode")
        roi.SetCenter(10.0, 0.0, 0.0)
        roi.SetSize(60.0, 60.0, 60.0)
        axial = self._phantom("axial", origin=(-80.0, -80.0, -60.0))
        # MRI-like: flipped (LPS-style) and slightly oblique axis directions
        oblique = self._phantom("oblique", origin=(80.0, 80.0, -60.0))
        angle = np.radians(12.0)
        oblique.SetIJKToRASDirections(-np.cos(angle), -np.sin(angle), 0.0,
                                      np.sin(angle), -np.cos(angle), 0.0,
                                      0.0, 0.0, 1.0)
        for volume in (axial, oblique):
            originalShape = slicer.util.arrayFromVolume(volume).shape
            cropped, temporary = easy_regLogic().makeWorkingVolume(volume, roi, f"cropped {volume.GetName()}")
            assert temporary and slicer.util.arrayFromVolume(volume).shape == originalShape  # input untouched
            croppedArray = slicer.util.arrayFromVolume(cropped)
            assert 0 < croppedArray.size < slicer.util.arrayFromVolume(volume).size, volume.GetName()
            ijkToRas, rasToIjk = vtk.vtkMatrix4x4(), vtk.vtkMatrix4x4()
            cropped.GetIJKToRASMatrix(ijkToRas)
            volume.GetRASToIJKMatrix(rasToIjk)
            for ijk in ((0, 0, 0), (5, 7, 3)):
                ras = ijkToRas.MultiplyPoint(list(ijk) + [1.0])
                i, j, k = [int(round(v)) for v in rasToIjk.MultiplyPoint(ras)[:3]]
                assert croppedArray[ijk[2], ijk[1], ijk[0]] == slicer.util.arrayFromVolume(volume)[k, j, i]
        farRoi = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode")
        farRoi.SetCenter(1000.0, 0.0, 0.0)
        farRoi.SetSize(50.0, 50.0, 50.0)
        try:
            easy_regLogic().makeWorkingVolume(axial, farRoi, "outside")
            raise AssertionError("an ROI outside the image must be rejected")
        except ValueError:
            pass
        self.delayDisplay("Crop keeps geometry OK (axial and oblique/flipped)")

    def test_hardenAndUndo(self):
        volume = self._phantom("undo test")
        transform = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode")
        transform.SetAttribute(TRANSFORM_ATTRIBUTE, "1")
        transform.SetAttribute(METHOD_ATTRIBUTE, "Rigid")
        matrix = vtk.vtkMatrix4x4()
        matrix.SetElement(0, 3, 12.0)
        matrix.SetElement(2, 3, -5.0)
        transform.SetMatrixTransformToParent(matrix)
        volume.SetAndObserveTransformNodeID(transform.GetID())
        logic = easy_regLogic()
        logic.hardenRegistration(transform)
        assert volume.GetTransformNodeID() is None and abs(volume.GetOrigin()[0] - 12.0) < 1e-6
        logic.undoRegistration(transform)
        assert np.allclose(volume.GetOrigin(), (0.0, 0.0, 0.0), atol=1e-6)
        self.delayDisplay("Harden and undo OK")

    def test_rigidRecoversShift(self):
        shift = np.array([9.0, -6.0, 6.0])
        fixed = self._phantom("fixed")
        moving = self._phantom("moving", origin=tuple(shift))
        result = easy_regLogic().startRegistration(fixed, moving, "Rigid", 0.05, initializeMode="Off", wait=True)
        assert result["status"] == "completed", result["message"]
        matrix = vtk.vtkMatrix4x4()
        result["transform"].GetMatrixTransformToParent(matrix)
        translation = np.array([matrix.GetElement(r, 3) for r in range(3)])
        error = np.linalg.norm(translation + shift)
        assert error < 1.5, f"translation {translation}, expected {-shift}"
        self.delayDisplay(f"Rigid registration recovered the shift (error {error:.2f} mm)")

    SPECT_SHIFT = np.array([12.0, -9.0, 15.0])

    def _spectPhantom(self, name, origin=(0.0, 0.0, 0.0)):
        """SPECT-like version of the phantom: low body background (scatter), hot liver, no anatomy."""
        ct = self._phantom(name + " source", origin)
        array = slicer.util.arrayFromVolume(ct)
        spect = np.zeros(array.shape, dtype=np.float32)
        spect[array > -500] = 3.0
        spect[array == 120] = 100.0
        slicer.mrmlScene.RemoveNode(ct)
        node = slicer.util.addVolumeFromArray(spect, name=name)
        node.SetSpacing(3.0, 3.0, 3.0)
        node.SetOrigin(*origin)
        return node

    def _liverSegmentation(self, ctNode):
        array = slicer.util.arrayFromVolume(ctNode)
        labelmap = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
        slicer.util.updateVolumeFromArray(labelmap, (array == 120).astype(np.uint8))
        ijkToRas = vtk.vtkMatrix4x4()
        ctNode.GetIJKToRASMatrix(ijkToRas)
        labelmap.SetIJKToRASMatrix(ijkToRas)
        segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelmap, segmentationNode)
        slicer.mrmlScene.RemoveNode(labelmap)
        segmentation = segmentationNode.GetSegmentation()
        segmentID = segmentation.GetNthSegmentID(0)
        segmentation.GetSegment(segmentID).SetName("Liver")
        return segmentationNode, segmentID

    def _translation(self, transformNode):
        matrix = easy_regLogic.transformMatrix(transformNode)
        return matrix[:3, 3]

    def test_functionalBodyOutline(self):
        fixed = self._phantom("reference CT")
        moving = self._spectPhantom("SPECT only", origin=tuple(self.SPECT_SHIFT))
        segmentationNode, liverID = self._liverSegmentation(fixed)
        logic = easy_regLogic()
        transform, message = logic.alignBodyOutlines(moving, fixed, 1.0, segmentationNode, liverID)
        translation = self._translation(transform)
        error = np.linalg.norm(translation + self.SPECT_SHIFT)
        assert error < 3.0, f"translation {translation}, expected {-self.SPECT_SHIFT} ({message})"
        assert transform.GetAttribute(METHOD_ATTRIBUTE) == "Body outline"
        assert moving.GetParentTransformNode() is transform
        self.delayDisplay(f"Body outline alignment recovered the shift (error {error:.1f} mm)")

    def test_functionalLandmarks(self):
        fixed = self._phantom("reference CT")
        moving = self._spectPhantom("SPECT only", origin=tuple(self.SPECT_SHIFT))
        points = np.array([[20.0, 30.0, 10.0], [110.0, 60.0, 40.0], [60.0, 120.0, 90.0], [30.0, 90.0, 70.0]])
        fixedLandmarks = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
        movingLandmarks = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
        for point in points:
            fixedLandmarks.AddControlPoint(*point)
            movingLandmarks.AddControlPoint(*(point + self.SPECT_SHIFT))
        transform, rms, residuals = easy_regLogic().alignLandmarks(moving, fixed, movingLandmarks, fixedLandmarks)
        assert rms < 1e-3, rms
        assert np.allclose(self._translation(transform), -self.SPECT_SHIFT, atol=1e-3)
        assert movingLandmarks.GetParentTransformNode() is transform  # the landmarks move with the SPECT
        world = movingLandmarks.GetNthControlPointPositionWorld(1)
        assert np.allclose(world, points[1], atol=1e-3)
        self.delayDisplay("Landmark alignment OK")

    def test_centreOnReference(self):
        fixed = self._phantom("reference CT")
        moving = self._spectPhantom("SPECT far away", origin=(40.0, -30.0, 1300.0))
        follower = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
        follower.AddControlPoint(10.0, 10.0, 1310.0)
        logic = easy_regLogic()
        shift = logic.centreOnReference(moving, fixed, [follower])
        assert np.allclose(shift, (-40.0, 30.0, -1300.0), atol=1e-3), shift
        movingBounds, fixedBounds = [0.0] * 6, [0.0] * 6
        moving.GetRASBounds(movingBounds)
        fixed.GetRASBounds(fixedBounds)
        assert np.allclose(np.array(movingBounds).reshape(3, 2).mean(1), np.array(fixedBounds).reshape(3, 2).mean(1))
        transform = moving.GetParentTransformNode()
        assert transform is not None and transform.GetAttribute(METHOD_ATTRIBUTE) == CENTRE_METHOD
        assert follower.GetParentTransformNode() is transform
        assert np.allclose(follower.GetNthControlPointPositionWorld(0), (-30.0, 40.0, 10.0), atol=1e-3)
        logic.undoRegistration(transform)   # "Reset to original position"
        assert moving.GetParentTransformNode() is None and follower.GetParentTransformNode() is None
        assert np.allclose(follower.GetNthControlPointPositionWorld(0), (10.0, 10.0, 1310.0), atol=1e-3)
        self.delayDisplay("Centre on reference and reset OK")

    def test_functionalRefinement(self):
        fixed = self._phantom("reference CT")
        moving = self._spectPhantom("SPECT only", origin=tuple(self.SPECT_SHIFT))
        segmentationNode, liverID = self._liverSegmentation(fixed)
        logic = easy_regLogic()
        # start 5-6 mm away from the right position, as after a rough initial alignment
        start = logic.alignmentTransform(moving, fixed, method="Manual")
        logic.setTransformMatrix(start, F.translationMatrix(-self.SPECT_SHIFT + np.array([4.0, -3.0, 3.0])))
        result = logic.refineInLiver(fixed, moving, segmentationNode, liverID, 20.0, 0.2, wait=True)
        assert result["status"] == "completed", result["message"]
        translation = self._translation(result["transform"])
        error = np.linalg.norm(translation + self.SPECT_SHIFT)
        assert error < 3.0, f"translation {translation}, expected {-self.SPECT_SHIFT}"
        assert result["transform"].GetAttribute(METHOD_ATTRIBUTE) == REFINE_METHOD
        assert not [n for n in slicer.util.getNodesByClass("vtkMRMLLabelMapVolumeNode") if "temporary" in n.GetName()]
        self.delayDisplay(f"Liver-masked refinement converged (error {error:.1f} mm)")
