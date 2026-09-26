import os
import re
import ast
import time
import logging

import numpy as np
import qt
import ctk
import vtk
import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin


MODULE_VERSION = "2.0"

# ---- AI models: files in the module folder (or Resources/Models), each with a same-name .txt descriptor ----
# organ key -> (segment name, model file, segment colour, default ROI size in mm)
ORGANS = {
    "lung": ("Lungs", "CTLUNGswin12_segmenter.pth", (0.45, 0.65, 1.00), (380.0, 260.0, 320.0)),
    "liver": ("Liver", "Liver_SwinUNETR24.pth", (0.85, 0.35, 0.25), (260.0, 240.0, 240.0)),
}
CANDIDATE_COLOR = (1.0, 0.9, 0.0)
CANDIDATE_TAG = "LSFcalc.Candidate"     # segment tag of an AI result that is not accepted yet
ROI_INITIALIZED_ATTRIBUTE = "LSFcalc.ROIInitialized"

# Same defaults and preprocessing as the Aether module, so the models see the input they were used with
DESCRIPTOR_DEFAULTS = {
    "architecture": "SwinUNETR",
    "channels": (128, 256, 512, 1024, 2048),
    "strides": (2, 2, 2, 2),
    "res_units": 2,
    "down_kernel": 3,
    "up_kernel": 3,
    "depths": (2, 2, 2, 2),
    "num_heads": (3, 6, 12, 24),
    "feature_size": 24,
    "do_rate": 0.0,
    "voxel_spacing": (2.0, 2.0, 2.0),
    "input_intensity_vol1": (-135.0, 215.0),
    "output_intensity_vol1": (0.0, 10.0),
    "no_rescale_vol1": False,
    "dual_channel": False,
    "threshold": 0.5,
}
INFERENCE_ROI_SIZE = (96, 96, 96)   # sliding window size (Aether uses 96^3 for single-channel models)
CROP_FILL_VALUE = 0.0               # value outside the image when the ROI extends beyond it (as in Aether)
MIN_CUDA_MEMORY_GB = 1.9

SPECT_COLORMAP = "Inferno"
# CT window presets: (label, window, level) in HU
CT_WINDOWS = [("Lung", 1500.0, -600.0), ("Soft tissue", 400.0, 40.0)]
# SPECT window presets: upper limit as % of the image maximum (lower limit 0); fusion and MIP use the same window
SPECT_WINDOW_PERCENTS = [10, 25, 50, 75, 100]
DEFAULT_SPECT_WINDOW_PERCENT = 50
LUNG_SEGMENTATION_ATTRIBUTE = "LSFcalc.LungShuntPercent"  # on the hidden segmentation holding the moved lungs

LSF_LAYOUT_ID = 50301  # EasyReg uses 50201, the dosimetry modules 50101 / 50102
MIP_VIEW_TAG = "LSFMIP"
# Own 3D view for the segments (not Slicer's view "1", which other modules such as Epona use for their renderings)
SEGMENTS_VIEW_TAG = "LSFSegments3D"
FOREIGN_DISPLAY_CLASSES = ("vtkMRMLVolumeRenderingDisplayNode", "vtkMRMLModelDisplayNode",
                           "vtkMRMLSegmentationDisplayNode")
# Own volume rendering display node of the SPECT for the MIP view. Sharing one display node with Epona (which also
# shows a MIP of the SPECT) made each module move the other's MIP to its own view (Epona's MIP turned white).
MIP_DISPLAY_ATTRIBUTE = "LSFcalc.MIPDisplay"
LSF_LAYOUT_XML = f"""
<layout type="horizontal" split="true">
 <item splitSize="500">
  <layout type="vertical" split="true">
   <item splitSize="500"><view class="vtkMRMLSliceNode" singletontag="Red">
    <property name="orientation" action="default">Axial</property>
    <property name="viewlabel" action="default">R</property>
    <property name="viewcolor" action="default">#F34A33</property></view></item>
   <item splitSize="500"><view class="vtkMRMLSliceNode" singletontag="Green">
    <property name="orientation" action="default">Coronal</property>
    <property name="viewlabel" action="default">G</property>
    <property name="viewcolor" action="default">#6EB04B</property></view></item>
  </layout>
 </item>
 <item splitSize="500">
  <layout type="vertical" split="true">
   <item splitSize="500"><view class="vtkMRMLViewNode" singletontag="{SEGMENTS_VIEW_TAG}">
    <property name="viewlabel" action="default">1</property></view></item>
   <item splitSize="500"><view class="vtkMRMLViewNode" singletontag="{MIP_VIEW_TAG}">
    <property name="viewlabel" action="default">MIP</property></view></item>
  </layout>
 </item>
</layout>
"""

RELATIVE_MODULE = "RadioembolizationDosimetryRelative"
MODELS_FOLDER_SETTING = "Taranis/ModelsFolder"  # AI models folder chosen in the Taranis hub settings

INTRO_TEXT = (
    "The lung shunt fraction needs two segments on the SPECT/CT: the lungs and the whole liver. "
    "Use segments you already have (or draw them in the Segment Editor), or create them with AI-assisted "
    "segmentation. AI segmentation requires an ROI around each organ: one ROI covering both lungs completely "
    "and one ROI covering the whole liver.")
INFO_TEXT = (
    "Workflow\n"
    "1. Select the SPECT (Tc-99m MAA), the CT of the SPECT/CT and a segmentation (or create a new one).\n"
    "2. Lung and liver segments: select existing segments, draw them in the Segment Editor, or use AI-assisted "
    "segmentation. For AI segmentation, create an ROI around both lungs and an ROI around the whole liver; "
    "segmentation is only run inside the ROI.\n"
    "3. Evaluate every AI result: Accept, Try again (after adjusting the ROI) or Discard.\n"
    "4. Calculate. Lung and liver segments must not overlap; an overlap can be removed from the lung segment "
    "(the liver keeps the overlapping voxels).\n"
    "5. Evaluate the segments and results, press Accept, then send the LSF to the relative dosimetry module.\n"
    "LSF = lung counts / (lung counts + liver counts), counted on the SPECT voxel grid.\n\n"
    "AI segmentation needs PyTorch (installed with the PyTorch Utils module of the SlicerPyTorch extension) "
    "and MONAI.\n"
    "This module is NOT a medical device. It is for research purposes only.\n"
    "Prepared by: Burak Demir, MD, FEBNM\n"
    "For support, feedback, and suggestions: 4burakfe@gmail.com\n"
    f"Version: {MODULE_VERSION}")


# ---------------------------------------------------------------------------------------------------
# Counting (pure numpy, no GUI)
# ---------------------------------------------------------------------------------------------------

def computeLungShunt(spectArray, lungMask, liverMask, clipNegativeValues=True):
    """Lung shunt from the SPECT values inside the two masks (same grid).
    A voxel claimed by both masks on the SPECT grid (partial-volume voxels at the lung-liver boundary) is
    counted once, for the liver. Raises ValueError for empty masks or non-positive totals."""
    lungMask = np.asarray(lungMask, dtype=bool)
    liverMask = np.asarray(liverMask, dtype=bool)
    if not lungMask.any():
        raise ValueError("The lung segment does not cover any SPECT voxel. Check the segment and that it "
                         "lies inside the SPECT field of view.")
    if not liverMask.any():
        raise ValueError("The liver segment does not cover any SPECT voxel. Check the segment and that it "
                         "lies inside the SPECT field of view.")
    shared = lungMask & liverMask
    lungMask = lungMask & ~shared
    values = np.asarray(spectArray, dtype=np.float64)
    negativeVoxels = int(np.count_nonzero(values[lungMask | liverMask] < 0))
    if clipNegativeValues:
        values = np.where(values < 0, 0.0, values)
    lungCounts = float(values[lungMask].sum())
    liverCounts = float(values[liverMask].sum())
    if lungCounts < 0 or liverCounts <= 0:
        raise ValueError("Lung or liver counts are not positive (negative voxel values?). "
                         "Enable 'Set negative voxel values to 0' or check the image.")
    return {
        "lungCounts": lungCounts,
        "liverCounts": liverCounts,
        "lsfPercent": 100.0 * lungCounts / (lungCounts + liverCounts),
        "lungVoxels": int(np.count_nonzero(lungMask)),
        "liverVoxels": int(np.count_nonzero(liverMask)),
        "sharedVoxels": int(np.count_nonzero(shared)),
        "negativeVoxels": negativeVoxels,
    }


def rescaleIntensity(array, inputRange, outputRange):
    """Same as MONAI ScaleIntensityRange(a_min, a_max, b_min, b_max, clip=True)."""
    aMin, aMax = (float(v) for v in inputRange)
    bMin, bMax = (float(v) for v in outputRange)
    scaled = (np.asarray(array, dtype=np.float32) - aMin) / (aMax - aMin) * (bMax - bMin) + bMin
    return np.clip(scaled, min(bMin, bMax), max(bMin, bMax)).astype(np.float32)


# ---------------------------------------------------------------------------------------------------
# Model files and descriptors
# ---------------------------------------------------------------------------------------------------

def modelSearchFolders():
    """The AI models folder chosen in the Taranis settings (if any), then the module folder."""
    moduleDir = os.path.dirname(os.path.abspath(__file__))
    folders = []
    customFolder = slicer.util.settingsValue(MODELS_FOLDER_SETTING, "")
    if isinstance(customFolder, str) and customFolder:
        folders.append(customFolder)
    return folders + [moduleDir, os.path.join(moduleDir, "Resources", "Models")]


def findModelFile(fileName):
    for folder in modelSearchFolders():
        path = os.path.join(folder, fileName)
        if os.path.isfile(path):
            return path
    return None


def parseDescriptorValue(text):
    value = text.strip()
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    try:
        return ast.literal_eval(value)  # numbers, tuples, lists; never executes code
    except (ValueError, SyntaxError):
        return value


def readModelDescriptor(modelPath):
    """Descriptor '<model>.txt' next to the model ('key: value' lines, as used by Aether), over the defaults."""
    descriptor = dict(DESCRIPTOR_DEFAULTS)
    descriptorPath = os.path.splitext(modelPath)[0] + ".txt"
    if not os.path.isfile(descriptorPath):
        raise ValueError(f"Descriptor file not found: {descriptorPath}")
    with open(descriptorPath, "r", encoding="utf-8") as f:
        for line in f:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            descriptor[key.strip().lower()] = parseDescriptorValue(value)
    return descriptor


def missingAiPackage():
    """None if PyTorch, MONAI and einops can be imported, otherwise the missing package name."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return "torch"
    try:
        import monai  # noqa: F401
        import einops  # noqa: F401
    except ImportError:
        return "monai"
    return None


def buildModel(descriptor, device):
    """Network as in Aether (the wrapper classes define the state_dict key names, so they are kept)."""
    import torch.nn as nn
    import torch.nn.functional as F
    from monai.networks.nets import UNet, SwinUNETR
    from monai import __version__ as monaiVersion
    from packaging import version
    oldMonai = version.parse(monaiVersion) < version.parse("1.5")

    class DenoiseUNet(nn.Module):
        def __init__(self, in_channels=1, out_channels=1, channels=(32, 64, 128, 256, 512), num_res_units=2,
                     strides=(2, 2, 2, 2), kernel_size=3, up_kernel_size=3):
            super().__init__()
            self.unet = UNet(strides=strides, num_res_units=num_res_units, kernel_size=kernel_size,
                             up_kernel_size=up_kernel_size, spatial_dims=3, in_channels=in_channels,
                             out_channels=out_channels, channels=channels)

        def forward(self, x):
            return self.unet(x)

    class SwinDenoiser(nn.Module):
        def __init__(self, in_channels=1, out_channels=1, feature_size=48, heads=(6, 12, 24, 48),
                     depths=(2, 3, 3, 2), do_rate=0.1):
            super().__init__()
            self.model = SwinUNETR(num_heads=heads, use_v2=True, in_channels=in_channels,
                                   out_channels=out_channels, feature_size=feature_size, depths=depths,
                                   dropout_path_rate=do_rate,
                                   **({"img_size": (96, 96, 96)} if oldMonai else {}), use_checkpoint=True)

        def forward(self, x):
            return self.model(x)

    class GCFN(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.norm = nn.LayerNorm(dim)
            self.fc1 = nn.Linear(dim, dim)
            self.fc2 = nn.Linear(dim, dim)
            self.fc0 = nn.Linear(dim, dim)
            self.conv1 = nn.Conv3d(dim, dim, kernel_size=5, padding=2, groups=dim)
            self.conv2 = nn.Conv3d(dim, dim, kernel_size=5, padding=2, groups=dim)

        def forward(self, x):
            B, C, D, H, W = x.shape
            x_ = x.permute(0, 2, 3, 4, 1).contiguous().view(B * D * H * W, C)
            x1 = self.fc1(self.norm(x_)).view(B, D, H, W, C).permute(0, 4, 1, 2, 3)
            x2 = self.fc2(self.norm(x_)).view(B, D, H, W, C).permute(0, 4, 1, 2, 3)
            gate = F.gelu(self.conv1(x1)) * self.conv2(x2)
            gate = gate.permute(0, 2, 3, 4, 1).contiguous().view(B * D * H * W, C)
            out = self.fc0(gate).view(B, D, H, W, C).permute(0, 4, 1, 2, 3)
            return out + x

    class SwinGCFN(nn.Module):
        def __init__(self, in_channels=1, out_channels=1, feature_size=48, heads=(6, 12, 24, 48),
                     depths=(2, 3, 3, 2), do_rate=0.1):
            super().__init__()
            self.model = SwinUNETR(num_heads=heads, use_v2=True, in_channels=in_channels,
                                   out_channels=out_channels, feature_size=feature_size, depths=depths,
                                   dropout_path_rate=do_rate,
                                   **({"img_size": (64, 64, 64)} if oldMonai else {}), use_checkpoint=True)
            self.gcfn = GCFN(dim=out_channels)

        def forward(self, x):
            return self.gcfn(self.model(x))

    d = descriptor
    architecture = str(d["architecture"])
    if architecture == "UNET":
        model = DenoiseUNet(in_channels=1, channels=tuple(d["channels"]), num_res_units=int(d["res_units"]),
                            strides=tuple(d["strides"]), kernel_size=int(d["down_kernel"]),
                            up_kernel_size=int(d["up_kernel"]))
    elif architecture == "SwinUNETR":
        model = SwinDenoiser(in_channels=1, feature_size=int(d["feature_size"]), heads=tuple(d["num_heads"]),
                             depths=tuple(d["depths"]), do_rate=float(d["do_rate"]))
    else:  # Aether treats every other architecture name as SwinUNETR+GCFN
        model = SwinGCFN(in_channels=1, feature_size=int(d["feature_size"]), heads=tuple(d["num_heads"]),
                         depths=tuple(d["depths"]), do_rate=float(d["do_rate"]))
    return model.to(device)


# ---------------------------------------------------------------------------------------------------
# Segmentation helpers
# ---------------------------------------------------------------------------------------------------

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


def segmentMaskOnVolumeGrid(segmentationNode, segmentID, referenceVolumeNode):
    """Boolean mask of one segment on the reference volume grid; the temporary labelmap is always removed."""
    labelmapNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
    labelmapNode.SetHideFromEditors(True)
    try:
        success = slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(
            segmentationNode, [segmentID], labelmapNode, referenceVolumeNode)
        if not success or labelmapNode.GetImageData() is None:
            raise RuntimeError(f"Could not export segment '{segmentID}' onto the image grid.")
        return slicer.util.arrayFromVolume(labelmapNode) > 0  # copy, valid after the node is removed
    finally:
        removeTemporaryLabelmap(labelmapNode)


def segmentOverlap(segmentationNode, segmentIDA, segmentIDB, referenceVolumeNode):
    """(overlapping voxels, overlap mL) of two segments on the reference volume grid."""
    a = slicer.util.arrayFromSegmentBinaryLabelmap(segmentationNode, segmentIDA, referenceVolumeNode) > 0
    b = slicer.util.arrayFromSegmentBinaryLabelmap(segmentationNode, segmentIDB, referenceVolumeNode) > 0
    voxels = int(np.count_nonzero(a & b))
    sx, sy, sz = referenceVolumeNode.GetSpacing()
    return voxels, voxels * sx * sy * sz / 1000.0


def removeOverlap(segmentationNode, keepSegmentID, trimSegmentID, referenceVolumeNode):
    """Remove from trimSegmentID the voxels it shares with keepSegmentID. Returns the removed voxel count."""
    keep = slicer.util.arrayFromSegmentBinaryLabelmap(segmentationNode, keepSegmentID, referenceVolumeNode) > 0
    trim = slicer.util.arrayFromSegmentBinaryLabelmap(segmentationNode, trimSegmentID, referenceVolumeNode) > 0
    overlap = keep & trim
    removed = int(np.count_nonzero(overlap))
    if removed:
        trim[overlap] = False
        slicer.util.updateSegmentBinaryLabelmapFromArray(trim.astype(np.uint8), segmentationNode, trimSegmentID,
                                                         referenceVolumeNode)
    return removed


def isCandidate(segment):
    return segment is not None and segment.HasTag(CANDIDATE_TAG)


def findColorNode(name):
    for colorNode in slicer.util.getNodesByClass("vtkMRMLColorNode"):
        if (colorNode.GetName() or "").lower() == name.lower():
            return colorNode
    return None


def setColormap(volumeNode, colorName):
    if volumeNode is None:
        return
    if volumeNode.GetDisplayNode() is None:
        volumeNode.CreateDefaultDisplayNodes()
    colorNode = findColorNode(colorName)
    if volumeNode.GetDisplayNode() and colorNode:
        volumeNode.GetDisplayNode().SetAndObserveColorNodeID(colorNode.GetID())


def roiSize(roiNode):
    try:
        return list(roiNode.GetSize())
    except TypeError:
        size = [0.0] * 3
        roiNode.GetSize(size)
        return size


def boundsOverlap(boundsA, boundsB):
    return all(boundsA[2 * i] < boundsB[2 * i + 1] and boundsB[2 * i] < boundsA[2 * i + 1] for i in range(3))


def initializeRoi(roiNode, volumeNode, size, color):
    """Colour the ROI; centre a new (never placed) ROI on the volume with the organ's default size."""
    if roiNode is None:
        return
    if roiNode.GetDisplayNode() is None:
        roiNode.CreateDefaultDisplayNodes()
    if roiNode.GetDisplayNode():
        roiNode.GetDisplayNode().SetSelectedColor(*color)
        roiNode.GetDisplayNode().SetColor(*color)
    if volumeNode is None or roiNode.GetAttribute(ROI_INITIALIZED_ATTRIBUTE) == "1":
        return
    roiNode.SetAttribute(ROI_INITIALIZED_ATTRIBUTE, "1")
    if roiNode.GetNumberOfControlPoints() > 0 and any(s > 0 for s in roiSize(roiNode)):
        return  # already placed by the user
    bounds = [0.0] * 6
    volumeNode.GetRASBounds(bounds)
    extent = [bounds[2 * i + 1] - bounds[2 * i] for i in range(3)]
    roiNode.SetSize(*[min(s, e) if e > 0 else s for s, e in zip(size, extent)])
    roiNode.SetCenter(*[(bounds[2 * i] + bounds[2 * i + 1]) / 2.0 for i in range(3)])


def roiIsPlaced(roiNode):
    return roiNode is not None and roiNode.GetNumberOfControlPoints() > 0 and all(s > 0 for s in roiSize(roiNode))


def excludeForeignRenderings(viewNodeIDs, keepNodes=()):
    """Keep volume renderings, models and segmentations of other modules (e.g. the Epona MIP, dosimetry models)
    out of the given 3D views. A display node without view IDs is shown in every view, so it gets the explicit
    list of all other views (it stays visible where it was)."""
    targets = {viewID for viewID in viewNodeIDs if viewID}
    keepIDs = {node.GetID() for node in keepNodes if node is not None}
    others = [node.GetID() for className in ("vtkMRMLViewNode", "vtkMRMLSliceNode")
              for node in slicer.util.getNodesByClass(className) if node.GetID() not in targets]
    for className in FOREIGN_DISPLAY_CLASSES:
        for displayNode in slicer.util.getNodesByClass(className):
            displayable = displayNode.GetDisplayableNode()
            if (displayable is None or displayable.GetHideFromEditors() or displayable.GetID() in keepIDs
                    or displayNode.GetAttribute(MIP_DISPLAY_ATTRIBUTE)):
                continue
            current = [displayNode.GetNthViewNodeID(i) for i in range(displayNode.GetNumberOfViewNodeIDs())]
            if current and not targets.intersection(current):
                continue
            remaining = [viewID for viewID in (current or others) if viewID not in targets]
            if remaining:
                displayNode.SetViewNodeIDs(remaining)
            else:
                displayNode.SetVisibility(False)


def mipDisplayNode(volumeNode):
    """The LSF calculator's own volume rendering display node of the SPECT (created if needed), never the one of
    another module (e.g. the Epona MIP of the same SPECT)."""
    for index in range(volumeNode.GetNumberOfDisplayNodes()):
        node = volumeNode.GetNthDisplayNode(index)
        if node is not None and node.IsA("vtkMRMLVolumeRenderingDisplayNode") and node.GetAttribute(MIP_DISPLAY_ATTRIBUTE):
            return node
    logic = slicer.modules.volumerendering.logic()
    displayNode = logic.CreateVolumeRenderingDisplayNode()
    displayNode.UnRegister(logic)
    displayNode.SetAttribute(MIP_DISPLAY_ATTRIBUTE, "1")
    slicer.mrmlScene.AddNode(displayNode)
    volumeNode.AddAndObserveDisplayNodeID(displayNode.GetID())
    logic.UpdateDisplayNodeFromVolumeNode(displayNode, volumeNode)
    return displayNode


def registerLsfLayout(*args):
    layoutManager = slicer.app.layoutManager()
    if layoutManager is None:
        return
    layoutNode = layoutManager.layoutLogic().GetLayoutNode()
    if layoutNode.IsLayoutDescription(LSF_LAYOUT_ID):
        layoutNode.SetLayoutDescription(LSF_LAYOUT_ID, LSF_LAYOUT_XML)
    else:
        layoutNode.AddLayoutDescription(LSF_LAYOUT_ID, LSF_LAYOUT_XML)


def threeDViewForNode(viewNode):
    layoutManager = slicer.app.layoutManager()
    for index in range(layoutManager.threeDViewCount):
        widget = layoutManager.threeDWidget(index)
        if widget.mrmlViewNode().GetID() == viewNode.GetID():
            return widget.threeDView()
    return None


def lookFromAnterior(viewNode):
    view = threeDViewForNode(viewNode)
    if view is None:
        return
    try:
        view.lookFromViewAxis(ctk.ctkAxesWidget.Anterior)
        view.renderWindow().GetRenderers().GetFirstRenderer().ResetCamera()
        view.forceRender()
    except Exception as error:
        logging.debug(f"Could not reset the 3D camera: {error}")


# ---------------------------------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------------------------------

class LSFcalc(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "Taranis - LSF Calculator"
        parent.categories = ["Nuclear Medicine"]
        parent.dependencies = []  # PyTorch is optional (AI segmentation only), so no hard dependency
        parent.contributors = ["Burak Demir, MD, FEBNM"]
        parent.helpText = """
        Calculates the lung shunt fraction from a Tc-99m MAA SPECT/CT before radioembolization.<br>
        Lung and liver segments can be provided by the user or created with AI-assisted segmentation
        (an ROI around each organ is required).
        """
        parent.acknowledgementText = """
        This file was developed by Burak Demir.
        """
        iconPath = os.path.join(os.path.dirname(__file__), "Resources", "taranis_logo.png")
        self.parent.icon = qt.QIcon(iconPath)
        self.parent = parent
        slicer.app.connect("startupCompleted()", registerLsfLayout)


class SegmentComboBox:
    """Segment selector for one segmentation. Unaccepted AI candidates are not listed. Preselects the first
    segment whose name contains one of the keywords."""

    def __init__(self, keywords, toolTip, onChanged):
        self.widget = qt.QComboBox()
        self.widget.setToolTip(toolTip)
        self.keywords = keywords
        self._segmentationNode = None
        self._onChanged = onChanged
        self._updating = False
        self.widget.connect("currentIndexChanged(int)", self._onIndexChanged)

    def setSegmentationNode(self, node):
        self._segmentationNode = node
        self.refresh()

    def currentSegmentID(self):
        data = self.widget.currentData if self.widget.currentIndex >= 0 else None
        return data or None

    def setCurrentSegmentID(self, segmentID):
        index = self.widget.findData(segmentID or "")
        if index >= 0:
            self.widget.setCurrentIndex(index)

    def refresh(self):
        previous = self.currentSegmentID()
        self._updating = True
        try:
            self.widget.clear()
            self.widget.addItem("(none)", "")
            segmentIDs = []
            if self._segmentationNode is not None:
                segmentation = self._segmentationNode.GetSegmentation()
                for index in range(segmentation.GetNumberOfSegments()):
                    segmentID = segmentation.GetNthSegmentID(index)
                    segment = segmentation.GetSegment(segmentID)
                    if isCandidate(segment):
                        continue
                    self.widget.addItem(segment.GetName(), segmentID)
                    segmentIDs.append((segmentID, segment.GetName()))
            target = previous if previous in [s for s, _ in segmentIDs] else None
            if target is None:
                target = next((s for s, name in segmentIDs
                               if any(k in (name or "").lower() for k in self.keywords)), None)
            self.setCurrentSegmentID(target)
        finally:
            self._updating = False
        if self.currentSegmentID() != previous:
            self._onChanged()

    def _onIndexChanged(self, index):
        if not self._updating:
            self._onChanged()


class LSFcalcWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):

    NODE_REFERENCES = [("Spect", "spectSelector"), ("CT", "ctSelector"), ("Segmentation", "segmentationSelector"),
                       ("LungROI", "lungRoiSelector"), ("LiverROI", "liverRoiSelector")]

    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self._parameterNode = None
        self._updatingGUI = False
        self._suppressSegmentationEvents = False
        self._observedSegmentation = None
        self._candidate = None     # {"organ", "segmentID", "segmentation"}
        self._result = None        # last calculation
        self._accepted = False
        self._observedSpectDisplay = None  # SPECT display node whose window drives the MIP
        self._keepResults = False          # own segment moves after sending must not clear the results
        self._sent = False

    # ---- UI ----

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = LSFcalcLogic()

        bannerPath = os.path.join(os.path.dirname(__file__), "Resources", "banner.png")
        if os.path.exists(bannerPath):
            bannerLabel = qt.QLabel()
            bannerLabel.setPixmap(qt.QPixmap(bannerPath).scaledToWidth(400, qt.Qt.SmoothTransformation))
            bannerLabel.setAlignment(qt.Qt.AlignCenter)
            self.layout.addWidget(bannerLabel)
        else:
            logging.warning(f"Banner file not found at {bannerPath}")

        introLabel = qt.QLabel(INTRO_TEXT)
        introLabel.setWordWrap(True)
        self.layout.addWidget(introLabel)

        # ---- 1. Images ----
        imagesBox = ctk.ctkCollapsibleButton()
        imagesBox.text = "1. Images"
        self.layout.addWidget(imagesBox)
        imagesLayout = qt.QFormLayout(imagesBox)
        self.spectSelector = self._makeNodeSelector("vtkMRMLScalarVolumeNode", "Tc-99m MAA SPECT (or PET).")
        imagesLayout.addRow("SPECT/PET: ", self.spectSelector)
        self.ctSelector = self._makeNodeSelector(
            "vtkMRMLScalarVolumeNode", "CT of the SPECT/CT. Used for AI segmentation, display and editing.",
            allowNone=True)
        imagesLayout.addRow("CT of SPECT/CT: ", self.ctSelector)
        self.segmentationSelector = self._makeNodeSelector(
            "vtkMRMLSegmentationNode", "Segmentation holding the lung and liver segments. "
            "Create a new one for AI segmentation or drawing.", allowNone=True, allowCreate=True)
        self.segmentationSelector.baseName = "LSF segmentation"
        imagesLayout.addRow("Segmentation: ", self.segmentationSelector)

        # ---- 2. Segments ----
        segmentsBox = ctk.ctkCollapsibleButton()
        segmentsBox.text = "2. Lung and liver segments"
        self.layout.addWidget(segmentsBox)
        segmentsLayout = qt.QFormLayout(segmentsBox)
        self.lungSegmentCombo = SegmentComboBox(["lung"], "Segment containing both lungs.", self._onSegmentSelection)
        segmentsLayout.addRow("Lung segment: ", self.lungSegmentCombo.widget)
        self.liverSegmentCombo = SegmentComboBox(["liver"], "Segment of the whole liver.", self._onSegmentSelection)
        segmentsLayout.addRow("Liver segment: ", self.liverSegmentCombo.widget)
        segmentButtons = qt.QHBoxLayout()
        self.segmentEditorButton = qt.QPushButton("Open Segment Editor")
        self.segmentEditorButton.setToolTip("Draw or correct the lung and liver segments manually "
                                            "(the CT is set as the source volume).")
        self.showLayoutButton = qt.QPushButton("Show LSF layout")
        self.showLayoutButton.setToolTip("Axial and coronal fusion (left), segments and SPECT MIP in 3D (right).")
        segmentButtons.addWidget(self.segmentEditorButton)
        segmentButtons.addWidget(self.showLayoutButton)
        segmentsLayout.addRow(segmentButtons)

        # ---- 3. AI-assisted segmentation ----
        aiBox = ctk.ctkCollapsibleButton()
        aiBox.text = "3. AI-assisted segmentation (optional)"
        self.layout.addWidget(aiBox)
        aiLayout = qt.QFormLayout(aiBox)
        aiInfo = qt.QLabel(
            "An ROI is required for each organ. Create the lung ROI so that it covers both lungs completely, "
            "and the liver ROI so that it covers the whole liver. The model only segments inside the ROI.")
        aiInfo.setWordWrap(True)
        aiLayout.addRow(aiInfo)
        self.modelStatusLabel = qt.QLabel()
        self.modelStatusLabel.setWordWrap(True)
        aiLayout.addRow(self.modelStatusLabel)

        self.lungRoiSelector = self._makeNodeSelector("vtkMRMLMarkupsROINode", "ROI covering both lungs.",
                                                      allowNone=True, allowCreate=True)
        self.lungRoiSelector.baseName = "LSF ROI lungs"
        self.segmentLungsButton = qt.QPushButton("Segment lungs")
        aiLayout.addRow("Lung ROI: ", self._row(self.lungRoiSelector, self.segmentLungsButton))
        self.liverRoiSelector = self._makeNodeSelector("vtkMRMLMarkupsROINode", "ROI covering the whole liver.",
                                                       allowNone=True, allowCreate=True)
        self.liverRoiSelector.baseName = "LSF ROI liver"
        self.segmentLiverButton = qt.QPushButton("Segment liver")
        aiLayout.addRow("Liver ROI: ", self._row(self.liverRoiSelector, self.segmentLiverButton))
        self.forceCpuCheckBox = qt.QCheckBox("Force CPU")
        self.forceCpuCheckBox.setToolTip("Run on the CPU even when a CUDA GPU is available (slower).")
        aiLayout.addRow("", self.forceCpuCheckBox)

        self.candidateLabel = qt.QLabel("")
        self.candidateLabel.setWordWrap(True)
        aiLayout.addRow(self.candidateLabel)
        evaluationRow = qt.QHBoxLayout()
        self.acceptCandidateButton = qt.QPushButton("Accept")
        self.acceptCandidateButton.setToolTip("Keep the AI segment (overlap with the other organ is removed from "
                                              "the lungs).")
        self.retryCandidateButton = qt.QPushButton("Try again")
        self.retryCandidateButton.setToolTip("Discard this result and run again (adjust the ROI first if needed).")
        self.discardCandidateButton = qt.QPushButton("Discard")
        self.discardCandidateButton.setToolTip("Remove the AI result.")
        for button in (self.acceptCandidateButton, self.retryCandidateButton, self.discardCandidateButton):
            evaluationRow.addWidget(button)
        aiLayout.addRow(evaluationRow)

        # ---- 4. Quantification ----
        quantBox = ctk.ctkCollapsibleButton()
        quantBox.text = "4. Quantification"
        self.layout.addWidget(quantBox)
        quantLayout = qt.QFormLayout(quantBox)
        self.clipNegativeCheckBox = qt.QCheckBox("Set negative voxel values to 0")
        self.clipNegativeCheckBox.setChecked(True)
        self.clipNegativeCheckBox.setToolTip("Negative values (reconstruction noise) would otherwise reduce the "
                                             "counts. The number of negative voxels is reported either way.")
        quantLayout.addRow("", self.clipNegativeCheckBox)
        self.calculateButton = qt.QPushButton("Calculate")
        self.calculateButton.setToolTip("Count lung and liver activity on the SPECT and calculate the LSF.")
        quantLayout.addRow(self.calculateButton)

        self.resultsTable = qt.QTableWidget()
        self.resultsTable.setColumnCount(2)
        self.resultsTable.setHorizontalHeaderLabels(["Quantity", "Value"])
        self.resultsTable.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)
        self.resultsTable.horizontalHeader().setStretchLastSection(True)
        self.resultsTable.verticalHeader().setVisible(False)
        self.resultsTable.setMinimumHeight(170)
        quantLayout.addRow(self.resultsTable)
        self.resultNotesLabel = qt.QLabel("")
        self.resultNotesLabel.setWordWrap(True)
        quantLayout.addRow(self.resultNotesLabel)

        finishRow = qt.QHBoxLayout()
        self.acceptResultsButton = qt.QPushButton("Accept segmentation and results")
        self.acceptResultsButton.setToolTip("Confirm that you evaluated the segments and the result.")
        self.sendLsfButton = qt.QPushButton("Send LSF to Relative Dosimetry")
        self.sendLsfButton.setToolTip("Enabled after the results are accepted.")
        finishRow.addWidget(self.acceptResultsButton)
        finishRow.addWidget(self.sendLsfButton)
        quantLayout.addRow(finishRow)
        self.moveLungCheckBox = qt.QCheckBox("Move the lung segment out of the segmentation when sending")
        self.moveLungCheckBox.setChecked(True)
        self.moveLungCheckBox.setToolTip(
            "The relative dosimetry module uses the lung shunt with a lung mass, so the lung segment is not needed "
            "there and should not be dosed with the liver density.\nIt is moved to a separate, hidden "
            "segmentation (kept with the scene for traceability, not listed in the dosimetry selectors).")
        quantLayout.addRow(self.moveLungCheckBox)

        self.statusLabel = qt.QLabel("")
        self.statusLabel.setWordWrap(True)
        self.layout.addWidget(self.statusLabel)

        # ---- Display: window presets ----
        displayBox = ctk.ctkCollapsibleButton()
        displayBox.text = "Display"
        self.layout.addWidget(displayBox)
        displayLayout = qt.QFormLayout(displayBox)
        ctRow = qt.QHBoxLayout()
        for label, window, level in CT_WINDOWS:
            button = qt.QPushButton(label)
            button.setToolTip(f"CT window {window:g} / level {level:g} HU")
            button.connect("clicked()", lambda w=window, l=level: self.setCtWindow(w, l))
            ctRow.addWidget(button)
        displayLayout.addRow("CT window: ", ctRow)
        spectRow = qt.QHBoxLayout()
        for percent in SPECT_WINDOW_PERCENTS:
            button = qt.QPushButton(f"0-{percent}%")
            button.setToolTip(f"SPECT window from 0 to {percent}% of the image maximum (fusion views and MIP).")
            button.connect("clicked()", lambda p=percent: self.setSpectWindowPercent(p))
            spectRow.addWidget(button)
        displayLayout.addRow("SPECT window: ", spectRow)
        self.fusionOpacitySlider = ctk.ctkSliderWidget()
        self.fusionOpacitySlider.minimum = 0.0
        self.fusionOpacitySlider.maximum = 1.0
        self.fusionOpacitySlider.singleStep = 0.05
        self.fusionOpacitySlider.value = 0.5
        displayLayout.addRow("Fusion opacity: ", self.fusionOpacitySlider)

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
        self.spectSelector.connect("currentNodeChanged(vtkMRMLNode*)", self._onInputChanged)
        self.segmentationSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSegmentationChanged)
        self.lungRoiSelector.connect("currentNodeChanged(vtkMRMLNode*)", lambda n: self._onRoiChanged("lung", n))
        self.liverRoiSelector.connect("currentNodeChanged(vtkMRMLNode*)", lambda n: self._onRoiChanged("liver", n))
        self.segmentEditorButton.connect("clicked()", self.onOpenSegmentEditor)
        self.showLayoutButton.connect("clicked()", self.showLsfLayout)
        self.segmentLungsButton.connect("clicked()", lambda: self.onSegmentOrgan("lung"))
        self.segmentLiverButton.connect("clicked()", lambda: self.onSegmentOrgan("liver"))
        self.acceptCandidateButton.connect("clicked()", self.onAcceptCandidate)
        self.retryCandidateButton.connect("clicked()", self.onRetryCandidate)
        self.discardCandidateButton.connect("clicked()", self.onDiscardCandidate)
        self.clipNegativeCheckBox.connect("toggled(bool)", self._onInputChanged)
        self.forceCpuCheckBox.connect("toggled(bool)", self.updateParameterNodeFromGUI)
        self.calculateButton.connect("clicked()", self.onCalculate)
        self.acceptResultsButton.connect("clicked()", self.onAcceptResults)
        self.sendLsfButton.connect("clicked()", self.onSendLsf)
        self.fusionOpacitySlider.connect("valueChanged(double)", self._setFusionOpacity)

        self._segmentRefreshTimer = qt.QTimer()  # coalesces segment edit events
        self._segmentRefreshTimer.setSingleShot(True)
        self._segmentRefreshTimer.setInterval(300)
        self._segmentRefreshTimer.connect("timeout()", self._refreshSegmentCombos)

        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)
        self._updateModelStatus()
        self.initializeParameterNode()
        self.onSegmentationChanged(self.segmentationSelector.currentNode())

    def _makeNodeSelector(self, nodeType, toolTip, allowNone=False, allowCreate=False):
        selector = slicer.qMRMLNodeComboBox()
        selector.nodeTypes = [nodeType]
        selector.selectNodeUponCreation = True
        selector.addEnabled = allowCreate
        selector.removeEnabled = allowCreate
        selector.renameEnabled = allowCreate
        selector.noneEnabled = allowNone
        selector.showHidden = False
        selector.showChildNodeTypes = False
        selector.setMRMLScene(slicer.mrmlScene)
        selector.setToolTip(toolTip)
        return selector

    @staticmethod
    def _row(*widgets):
        row = qt.QHBoxLayout()
        for widget in widgets:
            row.addWidget(widget)
        return row

    def _updateModelStatus(self):
        lines = []
        for organ, (name, fileName, _, _) in ORGANS.items():
            path = findModelFile(fileName)
            if path is None:
                lines.append(f"{name} model: not found ({fileName})")
            elif not os.path.isfile(os.path.splitext(path)[0] + ".txt"):
                lines.append(f"{name} model: descriptor missing ({os.path.splitext(fileName)[0]}.txt)")
            else:
                lines.append(f"{name} model: available")
        self.modelStatusLabel.text = "\n".join(lines)
        self.modelStatusLabel.setToolTip("Model files are searched in:\n" + "\n".join(modelSearchFolders()))

    # ---- Lifecycle and parameter node ----

    def cleanup(self):
        self._observeSegmentation(None)
        self._observeSpectDisplay(None)
        self.removeObservers()

    def enter(self):
        self.initializeParameterNode()

    def onSceneStartClose(self, caller, event):
        self._observeSegmentation(None)
        self._observeSpectDisplay(None)
        self._candidate = None
        self._clearResults()
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event):
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self):
        parameterNode = self.logic.getParameterNode()
        if not parameterNode.GetParameter("ClipNegative"):
            parameterNode.SetParameter("ClipNegative", "true")
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
            self.clipNegativeCheckBox.setChecked(p.GetParameter("ClipNegative") != "false")
            self.forceCpuCheckBox.setChecked(p.GetParameter("ForceCPU") == "true")
            self.lungSegmentCombo.setCurrentSegmentID(p.GetParameter("LungSegmentID"))
            self.liverSegmentCombo.setCurrentSegmentID(p.GetParameter("LiverSegmentID"))
        finally:
            self._updatingGUI = False
        self.updateButtonStates()

    def updateParameterNodeFromGUI(self, *args):
        if self._parameterNode is None or self._updatingGUI:
            return
        p = self._parameterNode
        wasModified = p.StartModify()
        try:
            for role, attribute in self.NODE_REFERENCES:
                p.SetNodeReferenceID(role, getattr(self, attribute).currentNodeID)
            p.SetParameter("ClipNegative", "true" if self.clipNegativeCheckBox.checked else "false")
            p.SetParameter("ForceCPU", "true" if self.forceCpuCheckBox.checked else "false")
            p.SetParameter("LungSegmentID", self.lungSegmentCombo.currentSegmentID() or "")
            p.SetParameter("LiverSegmentID", self.liverSegmentCombo.currentSegmentID() or "")
        finally:
            p.EndModify(wasModified)
        self.updateButtonStates()

    def updateButtonStates(self):
        spect = self.spectSelector.currentNode()
        ct = self.ctSelector.currentNode()
        segmentation = self.segmentationSelector.currentNode()
        pending = self._candidate is not None
        self.segmentEditorButton.enabled = not pending
        self.showLayoutButton.enabled = spect is not None or ct is not None
        for organ, button, roiSelector in (("lung", self.segmentLungsButton, self.lungRoiSelector),
                                           ("liver", self.segmentLiverButton, self.liverRoiSelector)):
            roi = roiSelector.currentNode()
            button.enabled = not pending and ct is not None and roiIsPlaced(roi)
            button.setToolTip(f"Run AI {ORGANS[organ][0].lower()} segmentation on the CT inside the ROI." if
                              roiIsPlaced(roi) else "Select the CT and create the ROI first (required).")
        for button in (self.acceptCandidateButton, self.retryCandidateButton, self.discardCandidateButton):
            button.enabled = pending
        lungID = self.lungSegmentCombo.currentSegmentID()
        liverID = self.liverSegmentCombo.currentSegmentID()
        self.calculateButton.enabled = (not pending and spect is not None and segmentation is not None
                                        and lungID is not None and liverID is not None and lungID != liverID)
        self.acceptResultsButton.enabled = self._result is not None and not self._accepted
        self.sendLsfButton.enabled = self._result is not None and self._accepted and not self._sent

    # ---- Inputs and invalidation ----

    def _onInputChanged(self, *args):
        self._clearResults()
        self.updateParameterNodeFromGUI()

    def _onSegmentSelection(self):
        if not self._keepResults:
            self._clearResults()
        self.updateParameterNodeFromGUI()

    def _clearResults(self):
        if self._result is not None or self._accepted:
            self.statusLabel.text = "Inputs or segments changed: results cleared, calculate again."
        self._result = None
        self._accepted = False
        self._sent = False
        if hasattr(self, "resultsTable"):
            self.resultsTable.setRowCount(0)
            self.resultNotesLabel.text = ""
            self.updateButtonStates()

    def onSegmentationChanged(self, segmentationNode):
        reference = self.ctSelector.currentNode() or self.spectSelector.currentNode()
        if (segmentationNode is not None and reference is not None
                and segmentationNode.GetSegmentation().GetNumberOfSegments() == 0):
            segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(reference)
            segmentationNode.CreateDefaultDisplayNodes()
        if self._candidate is not None and self._candidate["segmentation"] is not segmentationNode:
            self._removeCandidateSegment()
        self._observeSegmentation(segmentationNode)
        self.lungSegmentCombo.setSegmentationNode(segmentationNode)
        self.liverSegmentCombo.setSegmentationNode(segmentationNode)
        if not self._updatingGUI and self._parameterNode is not None:
            self.lungSegmentCombo.setCurrentSegmentID(self._parameterNode.GetParameter("LungSegmentID"))
            self.liverSegmentCombo.setCurrentSegmentID(self._parameterNode.GetParameter("LiverSegmentID"))
        self._clearResults()

    def _segmentationEvents(self):
        events = [slicer.vtkSegmentation.SegmentAdded, slicer.vtkSegmentation.SegmentRemoved,
                  slicer.vtkSegmentation.SegmentModified]
        for name in ("SourceRepresentationModified", "MasterRepresentationModified"):
            if hasattr(slicer.vtkSegmentation, name):
                events.append(getattr(slicer.vtkSegmentation, name))
                break
        return events

    def _observeSegmentation(self, segmentationNode):
        if self._observedSegmentation is not None:
            for event in self._segmentationEvents():
                self.removeObserver(self._observedSegmentation, event, self._onSegmentationEdited)
        self._observedSegmentation = segmentationNode.GetSegmentation() if segmentationNode else None
        if self._observedSegmentation is not None:
            for event in self._segmentationEvents():
                self.addObserver(self._observedSegmentation, event, self._onSegmentationEdited)

    def _onSegmentationEdited(self, caller, event):
        if self._suppressSegmentationEvents:
            return
        self._clearResults()
        self._segmentRefreshTimer.start()

    def _refreshSegmentCombos(self):
        self.lungSegmentCombo.refresh()
        self.liverSegmentCombo.refresh()
        self.updateButtonStates()

    def _onRoiChanged(self, organ, roiNode):
        if roiNode is not None and not self._updatingGUI:
            _, _, color, size = ORGANS[organ]
            initializeRoi(roiNode, self.ctSelector.currentNode() or self.spectSelector.currentNode(), size, color)
            if not self.hasObserver(roiNode, slicer.vtkMRMLMarkupsNode.PointModifiedEvent, self._onRoiModified):
                self.addObserver(roiNode, slicer.vtkMRMLMarkupsNode.PointModifiedEvent, self._onRoiModified)
            self.showLsfLayout()
        self.updateButtonStates()

    def _onRoiModified(self, caller, event):
        self.updateButtonStates()

    # ---- Layout and display ----

    def showLsfLayout(self):
        spect = self.spectSelector.currentNode()
        ct = self.ctSelector.currentNode()
        if spect is None and ct is None:
            return
        registerLsfLayout()
        layoutManager = slicer.app.layoutManager()
        if layoutManager.layout != LSF_LAYOUT_ID:
            layoutManager.setLayout(LSF_LAYOUT_ID)
        slicer.app.processEvents()

        # Fusion slice views (left): CT with SPECT on top
        setColormap(ct, "Grey")
        setColormap(spect, SPECT_COLORMAP)
        for sliceName, orientation in (("Red", "Axial"), ("Green", "Coronal")):
            sliceWidget = layoutManager.sliceWidget(sliceName)
            if sliceWidget is None:
                continue
            sliceWidget.mrmlSliceNode().SetOrientation(orientation)
            compositeNode = sliceWidget.mrmlSliceCompositeNode()
            compositeNode.SetBackgroundVolumeID((ct or spect).GetID())
            compositeNode.SetForegroundVolumeID(spect.GetID() if (spect and ct) else None)
            compositeNode.SetForegroundOpacity(self.fusionOpacitySlider.value)
            sliceWidget.sliceController().fitSliceToBackground()

        segmentsView = slicer.mrmlScene.GetSingletonNode(SEGMENTS_VIEW_TAG, "vtkMRMLViewNode")
        mipView = slicer.mrmlScene.GetSingletonNode(MIP_VIEW_TAG, "vtkMRMLViewNode")

        # Segments in 3D (top right), not in the MIP view
        segmentation = self.segmentationSelector.currentNode()
        if segmentation is not None and segmentsView is not None:
            self._suppressSegmentationEvents = True
            try:
                segmentation.CreateDefaultDisplayNodes()
                segmentation.CreateClosedSurfaceRepresentation()
                displayNode = segmentation.GetDisplayNode()
                viewIDs = [segmentsView.GetID()] + [
                    node.GetID() for node in slicer.util.getNodesByClass("vtkMRMLSliceNode")]
                displayNode.SetViewNodeIDs(viewIDs)
                displayNode.SetVisibility(True)
                displayNode.SetVisibility3D(True)   # the dosimetry modules turn the 3D segmentation off
                displayNode.SetOpacity2DFill(0.15)
                displayNode.SetOpacity3D(0.8)
            finally:
                self._suppressSegmentationEvents = False
            lookFromAnterior(segmentsView)

        spectDisplay = self._spectDisplayNode(spect)
        if spectDisplay is not None and spectDisplay.GetAutoWindowLevel():
            self.setSpectWindowPercent(DEFAULT_SPECT_WINDOW_PERCENT)

        # SPECT MIP (bottom right)
        if spect is not None and mipView is not None:
            mipView.SetRaycastTechnique(slicer.vtkMRMLViewNode.MaximumIntensityProjection)
            mipView.SetBackgroundColor(1.0, 1.0, 1.0)
            mipView.SetBackgroundColor2(1.0, 1.0, 1.0)
            mipView.SetBoxVisible(False)
            mipView.SetAxisLabelsVisible(False)
            self._updateMip()
            lookFromAnterior(mipView)
        try:
            excludeForeignRenderings([v.GetID() for v in (segmentsView, mipView) if v is not None],
                                     [segmentation])
        except Exception as error:
            logging.warning(f"Could not isolate the LSF 3D views: {error}")

    def _observeSpectDisplay(self, displayNode):
        if self._observedSpectDisplay is not None:
            self.removeObserver(self._observedSpectDisplay, vtk.vtkCommand.ModifiedEvent, self._onSpectDisplayModified)
        self._observedSpectDisplay = displayNode
        if displayNode is not None:
            self.addObserver(displayNode, vtk.vtkCommand.ModifiedEvent, self._onSpectDisplayModified)

    def _onSpectDisplayModified(self, caller, event):
        self._updateMip()

    def _spectDisplayNode(self, spect):
        if spect is None:
            return None
        if spect.GetDisplayNode() is None:
            spect.CreateDefaultDisplayNodes()
        return spect.GetDisplayNode()

    def setSpectWindowPercent(self, percent):
        """SPECT window 0 .. percent% of the image maximum. The MIP follows through the display node."""
        spect = self.spectSelector.currentNode()
        displayNode = self._spectDisplayNode(spect)
        if displayNode is None or spect.GetImageData() is None:
            return
        maximum = spect.GetImageData().GetScalarRange()[1]
        upper = max(maximum * percent / 100.0, 1e-6)
        displayNode.SetAutoWindowLevel(False)
        displayNode.SetWindowLevelMinMax(0.0, upper)

    def setCtWindow(self, window, level):
        displayNode = self._spectDisplayNode(self.ctSelector.currentNode())
        if displayNode is not None:
            displayNode.SetAutoWindowLevel(False)
            displayNode.SetWindowLevel(window, level)

    def _updateMip(self):
        """MIP transfer functions from the SPECT slice-view window, so fusion and MIP always match (also when
        the window is changed by dragging in a slice view)."""
        spect = self.spectSelector.currentNode()
        mipView = slicer.mrmlScene.GetSingletonNode(MIP_VIEW_TAG, "vtkMRMLViewNode")
        sliceDisplay = self._spectDisplayNode(spect)
        if spect is None or mipView is None or sliceDisplay is None:
            return
        if sliceDisplay is not self._observedSpectDisplay:
            self._observeSpectDisplay(sliceDisplay)
        displayNode = mipDisplayNode(spect)
        displayNode.SetViewNodeIDs([mipView.GetID()])
        displayNode.SetVisibility(True)
        lower = sliceDisplay.GetWindowLevelMin()
        upper = max(sliceDisplay.GetWindowLevelMax(), lower + 1e-6)
        # Inverted grey on white: below the window transparent, top of the window black
        color = vtk.vtkColorTransferFunction()
        color.AddRGBPoint(lower, 1.0, 1.0, 1.0)
        color.AddRGBPoint(upper, 0.0, 0.0, 0.0)
        opacity = vtk.vtkPiecewiseFunction()
        opacity.AddPoint(lower, 0.0)
        opacity.AddPoint(upper, 1.0)
        propertyNode = displayNode.GetVolumePropertyNode()
        propertyNode.SetColor(color)
        propertyNode.SetScalarOpacity(opacity)

    def _setFusionOpacity(self, value):
        layoutManager = slicer.app.layoutManager()
        for name in ("Red", "Green"):
            sliceWidget = layoutManager.sliceWidget(name)
            if sliceWidget is not None:
                sliceWidget.mrmlSliceCompositeNode().SetForegroundOpacity(value)

    def onOpenSegmentEditor(self):
        segmentation = self.segmentationSelector.currentNode()
        if segmentation is None:
            segmentation = self._createSegmentation()
        slicer.util.selectModule("SegmentEditor")
        try:
            editor = slicer.modules.segmenteditor.widgetRepresentation().self().editor
            editor.setSegmentationNode(segmentation)
            source = self.ctSelector.currentNode() or self.spectSelector.currentNode()
            if source is not None:
                if hasattr(editor, "setSourceVolumeNode"):
                    editor.setSourceVolumeNode(source)
                else:
                    editor.setMasterVolumeNode(source)
        except Exception as error:
            logging.warning(f"Could not preselect the segmentation in the Segment Editor: {error}")

    def _createSegmentation(self):
        segmentation = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "LSF segmentation")
        segmentation.CreateDefaultDisplayNodes()
        reference = self.ctSelector.currentNode() or self.spectSelector.currentNode()
        if reference is not None:
            segmentation.SetReferenceImageGeometryParameterFromVolumeNode(reference)
        self.segmentationSelector.setCurrentNode(segmentation)
        return segmentation

    # ---- AI segmentation ----

    def _checkAiPackages(self):
        missing = missingAiPackage()
        if missing is None:
            return True
        if missing == "torch":
            hasPyTorchUtils = hasattr(slicer.modules, "pytorchutils")
            text = ("AI segmentation needs PyTorch, which is not installed.\n\n"
                    "PyTorch must be installed with the PyTorch Utils module (SlicerPyTorch extension), which "
                    "selects the right version for your computer (GPU or CPU). ")
            if hasPyTorchUtils:
                if slicer.util.confirmOkCancelDisplay(text + "Open PyTorch Utils now?\n\nAfter installing, "
                                                      "restart Slicer.", windowTitle="PyTorch required"):
                    slicer.util.selectModule("PyTorchUtils")
            else:
                slicer.util.infoDisplay(text + "Install the 'PyTorch' extension from the Extensions Manager, "
                                        "restart Slicer, then install PyTorch in the PyTorch Utils module.",
                                        windowTitle="PyTorch required")
            return False
        if not slicer.util.confirmOkCancelDisplay(
                "AI segmentation needs the MONAI and einops Python packages, which are not installed.\n\n"
                "Install them now? PyTorch is not changed (installed without dependencies).",
                windowTitle="MONAI required"):
            return False
        try:
            with slicer.util.WaitCursor():
                slicer.util.pip_install(["monai", "einops", "--no-deps"])
        except Exception as error:
            slicer.util.errorDisplay(f"Installation of MONAI failed: {error}")
            return False
        return missingAiPackage() is None

    def onSegmentOrgan(self, organ):
        ct = self.ctSelector.currentNode()
        roiNode = (self.lungRoiSelector if organ == "lung" else self.liverRoiSelector).currentNode()
        name = ORGANS[organ][0]
        if ct is None:
            slicer.util.errorDisplay("Select the CT of the SPECT/CT for AI segmentation.")
            return
        if not roiIsPlaced(roiNode):
            slicer.util.errorDisplay(f"An ROI is required for AI {name.lower()} segmentation. Create the "
                                     f"{name.lower()} ROI and make it cover the whole organ.")
            return
        if not self._checkAiPackages():
            return
        segmentation = self.segmentationSelector.currentNode() or self._createSegmentation()

        self.statusLabel.text = f"Running AI {name.lower()} segmentation..."
        self._setAiBusy(True)
        slicer.app.processEvents()
        try:
            with slicer.util.WaitCursor():
                segmentID, message = self.logic.segmentOrgan(ct, roiNode, organ, segmentation,
                                                             forceCPU=self.forceCpuCheckBox.checked)
        except (ValueError, RuntimeError) as error:
            self.statusLabel.text = f"AI {name.lower()} segmentation failed."
            slicer.util.errorDisplay(str(error))
            return
        except Exception as error:
            logging.exception("AI segmentation failed")
            self.statusLabel.text = f"AI {name.lower()} segmentation failed."
            slicer.util.errorDisplay(f"AI segmentation failed: {error}")
            return
        finally:
            self._setAiBusy(False)

        self._candidate = {"organ": organ, "segmentID": segmentID, "segmentation": segmentation}
        self._showRoi(organ, False)  # the ROI box would hide the result; shown again after the decision
        self.candidateLabel.text = (f"Evaluate the AI {name.lower()} segment (yellow) in all views, then Accept, "
                                    "Try again (adjust the ROI first if needed) or Discard.")
        self.statusLabel.text = message
        self.showLsfLayout()
        self.updateButtonStates()

    def _setAiBusy(self, busy):
        for button in (self.segmentLungsButton, self.segmentLiverButton, self.calculateButton):
            button.enabled = not busy
        if not busy:
            self.updateButtonStates()

    def _removeCandidateSegment(self):
        candidate, self._candidate = self._candidate, None
        self.candidateLabel.text = ""
        if candidate is None:
            return
        segmentation = candidate["segmentation"]
        if segmentation is not None and segmentation.GetScene() is not None:
            segmentation.GetSegmentation().RemoveSegment(candidate["segmentID"])

    def _showRoi(self, organ, visible=True):
        roi = (self.lungRoiSelector if organ == "lung" else self.liverRoiSelector).currentNode()
        if roi is not None and roi.GetDisplayNode():
            roi.GetDisplayNode().SetVisibility(visible)

    def _removeRoi(self, organ):
        roi = (self.lungRoiSelector if organ == "lung" else self.liverRoiSelector).currentNode()
        if roi is None:
            return
        if self.hasObserver(roi, slicer.vtkMRMLMarkupsNode.PointModifiedEvent, self._onRoiModified):
            self.removeObserver(roi, slicer.vtkMRMLMarkupsNode.PointModifiedEvent, self._onRoiModified)
        slicer.mrmlScene.RemoveNode(roi)

    def onDiscardCandidate(self):
        organ = self._candidate["organ"] if self._candidate else None
        self._removeCandidateSegment()
        if organ:
            self._showRoi(organ)
        self.statusLabel.text = "AI result discarded."
        self.updateButtonStates()

    def onRetryCandidate(self):
        if self._candidate is None:
            return
        organ = self._candidate["organ"]
        self._removeCandidateSegment()
        self._showRoi(organ)
        self.onSegmentOrgan(organ)

    def onAcceptCandidate(self):
        candidate = self._candidate
        if candidate is None:
            return
        organ, segmentation = candidate["organ"], candidate["segmentation"]
        name, _, color, _ = ORGANS[organ]
        seg = segmentation.GetSegmentation()
        existing = [seg.GetNthSegmentID(i) for i in range(seg.GetNumberOfSegments())
                    if seg.GetNthSegmentID(i) != candidate["segmentID"]
                    and seg.GetSegment(seg.GetNthSegmentID(i)).GetName() == name]
        if existing:
            if not slicer.util.confirmOkCancelDisplay(f"A segment named '{name}' already exists. Replace it with "
                                                      "the AI result?", windowTitle="Replace segment"):
                return
            for segmentID in existing:
                seg.RemoveSegment(segmentID)

        segment = seg.GetSegment(candidate["segmentID"])
        segment.RemoveTag(CANDIDATE_TAG)
        segment.SetName(name)
        segment.SetColor(*color)
        self._candidate = None
        self.candidateLabel.text = ""
        self._removeRoi(organ)
        self._refreshSegmentCombos()
        (self.lungSegmentCombo if organ == "lung" else self.liverSegmentCombo).setCurrentSegmentID(
            candidate["segmentID"])

        # No overlap: the lungs give up voxels shared with the liver
        message = f"AI {name.lower()} segment accepted."
        lungID, liverID = self.lungSegmentCombo.currentSegmentID(), self.liverSegmentCombo.currentSegmentID()
        reference = self.ctSelector.currentNode() or self.spectSelector.currentNode()
        if lungID and liverID and lungID != liverID and reference is not None:
            with slicer.util.WaitCursor():
                removed = removeOverlap(segmentation, liverID, lungID, reference)
            if removed:
                message += f" {removed} voxels shared with the liver were removed from the lung segment."
        self.statusLabel.text = message
        self.updateParameterNodeFromGUI()
        self.showLsfLayout()

    # ---- Quantification ----

    def onCalculate(self):
        spect = self.spectSelector.currentNode()
        segmentation = self.segmentationSelector.currentNode()
        lungID = self.lungSegmentCombo.currentSegmentID()
        liverID = self.liverSegmentCombo.currentSegmentID()
        if spect is None or segmentation is None:
            slicer.util.errorDisplay("Select the SPECT and the segmentation.")
            return
        if not lungID or not liverID:
            slicer.util.errorDisplay("Select both the lung segment and the liver segment.")
            return
        if lungID == liverID:
            slicer.util.errorDisplay("The lung and liver segments must be different segments.")
            return

        # Overlap is not allowed: checked at the resolution of the segments (CT grid when available)
        reference = self.ctSelector.currentNode() or spect
        with slicer.util.WaitCursor():
            overlapVoxels, overlapML = segmentOverlap(segmentation, lungID, liverID, reference)
        if overlapVoxels:
            if not slicer.util.confirmOkCancelDisplay(
                    f"The lung and liver segments overlap ({overlapVoxels} voxels, {overlapML:.1f} mL). Overlap is "
                    "not allowed, because those voxels would be counted for both organs.\n\n"
                    "Remove the overlapping voxels from the lung segment (the liver keeps them)?",
                    windowTitle="Overlapping segments"):
                self.statusLabel.text = "Calculation cancelled: remove the lung-liver overlap first."
                return
            with slicer.util.WaitCursor():
                removeOverlap(segmentation, liverID, lungID, reference)

        try:
            with slicer.util.WaitCursor():
                result = self.logic.calculate(spect, segmentation, lungID, liverID,
                                              clipNegativeValues=self.clipNegativeCheckBox.checked)
        except (ValueError, RuntimeError) as error:
            slicer.util.errorDisplay(str(error))
            return

        self._result = result
        self._accepted = False
        self._showResults(result)
        self.statusLabel.text = ("Evaluate the segments in the views and the results, then press "
                                 "'Accept segmentation and results'.")
        self.showLsfLayout()
        self.updateButtonStates()

    def _showResults(self, result):
        rows = [
            ("Lung Shunt Fraction", f"{result['lsfPercent']:.2f} %"),
            ("Lung counts", f"{result['lungCounts']:.6g}"),
            ("Liver counts", f"{result['liverCounts']:.6g}"),
            ("Lung volume (SPECT grid)", f"{result['lungML']:.1f} mL"),
            ("Liver volume (SPECT grid)", f"{result['liverML']:.1f} mL"),
        ]
        self.resultsTable.setRowCount(len(rows))
        for row, (label, value) in enumerate(rows):
            self.resultsTable.setItem(row, 0, qt.QTableWidgetItem(label))
            self.resultsTable.setItem(row, 1, qt.QTableWidgetItem(value))
        self.resultsTable.resizeColumnToContents(0)
        notes = []
        if result["negativeVoxels"]:
            action = "set to 0" if result["clipped"] else "used as is"
            notes.append(f"{result['negativeVoxels']} negative voxels inside the segments ({action}).")
        if result["sharedVoxels"]:
            notes.append(f"{result['sharedVoxels']} boundary voxels claimed by both segments on the coarser SPECT "
                         "grid were counted once, for the liver.")
        self.resultNotesLabel.text = "\n".join(notes)

    def onAcceptResults(self):
        if self._result is None:
            return
        self._accepted = True
        self.statusLabel.text = (f"Accepted: LSF {self._result['lsfPercent']:.2f} %. It can now be sent to the "
                                 "relative dosimetry module.")
        self.updateButtonStates()

    def onSendLsf(self):
        if self._result is None or not self._accepted:
            return
        if not hasattr(slicer.modules, RELATIVE_MODULE.lower()):
            slicer.util.errorDisplay("The Relative Dosimetry module is not loaded.")
            return
        lsf = self._result["lsfPercent"]
        movedTo = None
        if self.moveLungCheckBox.checked:
            try:
                movedTo = self._moveLungSegmentOut(lsf)
            except Exception as error:
                slicer.util.errorDisplay(f"Could not move the lung segment: {error}")
                return
        self._sent = True
        self.updateButtonStates()
        slicer.util.selectModule(RELATIVE_MODULE)
        try:
            widget = slicer.util.getModuleWidget(RELATIVE_MODULE)
            previous = widget.lungShuntSlider.value
            widget.lungShuntSlider.value = lsf
            applied = widget.lungShuntSlider.value
        except Exception as error:
            slicer.util.errorDisplay(f"Could not set the lung shunt fraction in {RELATIVE_MODULE}: {error}")
            return
        text = f"Lung shunt fraction set to {applied:.2f} % (was {previous:.2f} %)."
        if abs(applied - lsf) > 0.005:
            text += f"\nThe calculated value was {lsf:.2f} %; the slider rounded it."
        if movedTo is not None:
            text += (f"\n\nThe lung segment was moved to the hidden segmentation '{movedTo.GetName()}', so it is "
                     "not part of the dosimetry segmentation.")
        slicer.util.infoDisplay(text, windowTitle="LSF sent")

    def _moveLungSegmentOut(self, lsfPercent):
        """Move the lung segment into a separate segmentation that is hidden from views and node selectors
        (saved with the scene for traceability). Returns that segmentation node."""
        segmentation = self.segmentationSelector.currentNode()
        lungID = self.lungSegmentCombo.currentSegmentID()
        if segmentation is None or not lungID:
            return None
        lungNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode",
                                                      f"{segmentation.GetName()} - lungs (LSF)")
        reference = self.ctSelector.currentNode() or self.spectSelector.currentNode()
        if reference is not None:
            lungNode.SetReferenceImageGeometryParameterFromVolumeNode(reference)
        lungNode.CreateDefaultDisplayNodes()
        lungNode.GetDisplayNode().SetVisibility(False)
        lungNode.SetHideFromEditors(True)
        lungNode.SetAttribute(LUNG_SEGMENTATION_ATTRIBUTE, f"{lsfPercent:.4f}")
        self._keepResults = True
        self._suppressSegmentationEvents = True
        try:
            if not lungNode.GetSegmentation().CopySegmentFromSegmentation(segmentation.GetSegmentation(), lungID,
                                                                          True):
                slicer.mrmlScene.RemoveNode(lungNode)
                raise RuntimeError("the segment could not be copied")
            self.lungSegmentCombo.refresh()
            self.liverSegmentCombo.refresh()
        finally:
            self._suppressSegmentationEvents = False
            self._keepResults = False
        return lungNode


# ---------------------------------------------------------------------------------------------------
# Logic
# ---------------------------------------------------------------------------------------------------

class LSFcalcLogic(ScriptedLoadableModuleLogic):

    def calculate(self, spectVolumeNode, segmentationNode, lungSegmentID, liverSegmentID, clipNegativeValues=True):
        """Lung shunt fraction on the SPECT voxel grid. Temporary nodes are always removed."""
        if spectVolumeNode is None or segmentationNode is None:
            raise ValueError("Select the SPECT and the segmentation.")
        if not lungSegmentID or not liverSegmentID or lungSegmentID == liverSegmentID:
            raise ValueError("Select two different segments for the lungs and the liver.")
        spectArray = slicer.util.arrayFromVolume(spectVolumeNode)
        lungMask = segmentMaskOnVolumeGrid(segmentationNode, lungSegmentID, spectVolumeNode)
        liverMask = segmentMaskOnVolumeGrid(segmentationNode, liverSegmentID, spectVolumeNode)
        result = computeLungShunt(spectArray, lungMask, liverMask, clipNegativeValues)
        sx, sy, sz = spectVolumeNode.GetSpacing()
        voxelML = sx * sy * sz / 1000.0
        result.update({"lungML": result["lungVoxels"] * voxelML, "liverML": result["liverVoxels"] * voxelML,
                       "clipped": clipNegativeValues})
        logging.info(f"LSF: lung {result['lungCounts']:.6g}, liver {result['liverCounts']:.6g}, "
                     f"LSF {result['lsfPercent']:.2f} %")
        return result

    def segmentOrgan(self, ctVolumeNode, roiNode, organ, segmentationNode, forceCPU=False):
        """Run the organ model inside the ROI and add the result to the segmentation as an (unaccepted)
        candidate segment. Returns (segmentID, status message)."""
        import torch
        from monai.inferers import sliding_window_inference

        name, fileName, _, _ = ORGANS[organ]
        if ctVolumeNode.GetParentTransformNode() is not None:
            raise ValueError(f"'{ctVolumeNode.GetName()}' is under a transform. Harden the transform first.")
        if not roiIsPlaced(roiNode):
            raise ValueError(f"An ROI around the {name.lower()} is required.")
        roiBounds, ctBounds = [0.0] * 6, [0.0] * 6
        roiNode.GetRASBounds(roiBounds)
        ctVolumeNode.GetRASBounds(ctBounds)
        if not boundsOverlap(roiBounds, ctBounds):
            raise ValueError(f"The {name.lower()} ROI does not overlap the CT. Move the ROI onto the organ.")
        modelPath = findModelFile(fileName)
        if modelPath is None:
            raise ValueError(f"Model file '{fileName}' not found. Place it next to LSFcalc.py:\n"
                             + "\n".join(modelSearchFolders()))
        descriptor = readModelDescriptor(modelPath)
        if descriptor["dual_channel"]:
            raise ValueError(f"{fileName} is a dual-channel model; this module runs CT-only models.")

        temporaryNodes = []
        model = None
        try:
            # 1. Resample to the model's voxel spacing, 2. crop to the ROI (same order as Aether)
            resampled = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "LSF AI resampled")
            temporaryNodes.append(resampled)
            spacing = ",".join(str(float(v)) for v in descriptor["voxel_spacing"])
            cliNode = slicer.cli.runSync(slicer.modules.resamplescalarvolume, None, {
                "InputVolume": ctVolumeNode.GetID(), "OutputVolume": resampled.GetID(),
                "outputPixelSpacing": spacing, "interpolationType": "linear"})
            temporaryNodes.append(cliNode)   # each run would otherwise leave a CLI node in the scene
            cropped = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "LSF AI cropped")
            temporaryNodes.append(cropped)
            slicer.modules.cropvolume.logic().CropVoxelBased(roiNode, resampled, cropped, False, CROP_FILL_VALUE)
            imageData = cropped.GetImageData()
            if imageData is None or min(imageData.GetDimensions()) < 2:
                raise ValueError(f"The {name.lower()} ROI does not overlap the CT. Move the ROI onto the organ.")

            image = slicer.util.arrayFromVolume(cropped).astype(np.float32)
            if not descriptor["no_rescale_vol1"]:
                image = rescaleIntensity(image, descriptor["input_intensity_vol1"],
                                         descriptor["output_intensity_vol1"])

            device, deviceText = self._device(torch, forceCPU)
            model = buildModel(descriptor, device)
            model.load_state_dict(torch.load(modelPath, map_location=device))
            model.eval()
            start = time.time()
            with torch.no_grad():
                tensor = torch.from_numpy(np.ascontiguousarray(image))[None, None].to(device)
                output = sliding_window_inference(inputs=tensor, roi_size=INFERENCE_ROI_SIZE, sw_batch_size=1,
                                                  predictor=model, overlap=0.25, mode="gaussian")
                outputArray = output.squeeze().cpu().numpy()
            elapsed = time.time() - start
            if outputArray.ndim != 3:
                raise ValueError(f"{fileName} returned {output.shape[1]} channels; a single-channel model is "
                                 "expected.")
            mask = (np.clip(outputArray, 0, None) > float(descriptor["threshold"])).astype(np.uint8)
            if not mask.any():
                raise ValueError(f"The model found no {name.lower()} inside the ROI. Check the ROI position.")

            labelmap = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", f"{name} (AI)")
            temporaryNodes.append(labelmap)
            slicer.util.updateVolumeFromArray(labelmap, mask)
            ijkToRas = vtk.vtkMatrix4x4()
            cropped.GetIJKToRASMatrix(ijkToRas)
            labelmap.SetIJKToRASMatrix(ijkToRas)

            segmentation = segmentationNode.GetSegmentation()
            before = {segmentation.GetNthSegmentID(i) for i in range(segmentation.GetNumberOfSegments())}
            if not slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelmap, segmentationNode):
                raise RuntimeError("Could not import the AI result into the segmentation.")
            newIDs = [segmentation.GetNthSegmentID(i) for i in range(segmentation.GetNumberOfSegments())
                      if segmentation.GetNthSegmentID(i) not in before]
            if not newIDs:
                raise RuntimeError("The AI result could not be added to the segmentation.")
            segmentID = newIDs[0]
            segment = segmentation.GetSegment(segmentID)
            segment.SetName(f"{name} (AI - evaluate)")
            segment.SetColor(*CANDIDATE_COLOR)
            segment.SetTag(CANDIDATE_TAG, "1")
            return segmentID, f"AI {name.lower()} segmentation finished in {elapsed:.1f} s on {deviceText}."
        finally:
            for node in temporaryNodes:
                if node.GetScene() is not None:
                    slicer.mrmlScene.RemoveNode(node)
            del model
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @staticmethod
    def _device(torch, forceCPU):
        if not forceCPU and torch.cuda.is_available():
            try:
                memoryGB = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
                if memoryGB >= MIN_CUDA_MEMORY_GB:
                    return torch.device("cuda"), f"GPU ({memoryGB:.1f} GB)"
                logging.info(f"CUDA GPU has only {memoryGB:.1f} GB; using the CPU.")
            except Exception as error:
                logging.warning(f"Could not check the GPU memory ({error}); using the CPU.")
        return torch.device("cpu"), "CPU"


# ---------------------------------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------------------------------

class LSFcalcTest(ScriptedLoadableModuleTest):
    """Synthetic-data checks ("Reload and Test" in developer mode). AI inference is not tested here."""

    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_countMath()
        self.test_descriptorParsing()
        self.setUp()
        self.test_calculateOnScene()

    def test_countMath(self):
        spect = np.zeros((10, 10, 10))
        lung = np.zeros_like(spect, dtype=bool)
        liver = np.zeros_like(spect, dtype=bool)
        lung[0:3], liver[5:10] = True, True
        spect[0:3], spect[5:10] = 1.0, 9.0 * 3 / 5  # lung 300 counts, liver 2700 counts
        result = computeLungShunt(spect, lung, liver)
        assert abs(result["lsfPercent"] - 10.0) < 1e-9
        # A voxel claimed by both is counted once, for the liver
        lung[5, 0, 0] = True
        assert computeLungShunt(spect, lung, liver)["sharedVoxels"] == 1
        # Negative values: clipped by default
        spect[0, 0, 0] = -50.0
        assert computeLungShunt(spect, lung, liver)["negativeVoxels"] == 1
        assert computeLungShunt(spect, lung, liver, True)["lungCounts"] > \
            computeLungShunt(spect, lung, liver, False)["lungCounts"]
        try:
            computeLungShunt(spect, np.zeros_like(lung), liver)
            raise AssertionError("an empty lung mask must be rejected")
        except ValueError:
            pass
        assert np.allclose(rescaleIntensity([-135, 40, 215, 1000], (-135, 215), (0, 10)), [0, 5, 10, 10])
        self.delayDisplay("Count math OK")

    def test_descriptorParsing(self):
        assert parseDescriptorValue("(2,2,2,2)") == (2, 2, 2, 2)
        assert parseDescriptorValue("-135,215") == (-135, 215)
        assert parseDescriptorValue("[2,2,2]") == [2, 2, 2]
        assert parseDescriptorValue("true") is True
        assert parseDescriptorValue("SwinUNETR") == "SwinUNETR"
        assert parseDescriptorValue("__import__('os')") == "__import__('os')"  # never executed
        self.delayDisplay("Descriptor parsing OK")

    def test_calculateOnScene(self):
        spectArray = np.zeros((20, 20, 20), dtype=np.float32)
        spectArray[2:8] = 1.0     # "lung" slab: 6 x 400 voxels x 1
        spectArray[10:18] = 4.0   # "liver" slab: 8 x 400 voxels x 4
        spect = slicer.util.addVolumeFromArray(spectArray, name="spect")
        labels = np.zeros_like(spectArray, dtype=np.uint8)
        labels[2:8], labels[10:18] = 1, 2
        labelmap = slicer.util.addVolumeFromArray(labels, name="labels",
                                                  nodeClassName="vtkMRMLLabelMapVolumeNode")
        segmentation = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelmap, segmentation)
        seg = segmentation.GetSegmentation()
        lungID, liverID = seg.GetNthSegmentID(0), seg.GetNthSegmentID(1)
        labelmapCount = len(slicer.util.getNodesByClass("vtkMRMLLabelMapVolumeNode"))
        result = LSFcalcLogic().calculate(spect, segmentation, lungID, liverID)
        expected = 100.0 * (6 * 400) / (6 * 400 + 8 * 400 * 4)
        assert abs(result["lsfPercent"] - expected) < 1e-6, result
        assert len(slicer.util.getNodesByClass("vtkMRMLLabelMapVolumeNode")) == labelmapCount  # no leftovers
        assert segmentOverlap(segmentation, lungID, liverID, spect)[0] == 0
        self.delayDisplay("Calculation on scene OK")
