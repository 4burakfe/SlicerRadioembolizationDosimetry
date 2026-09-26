"""AI segmentation engine of the Taranis suite.

Models are PyTorch/MONAI networks trained with Aether; each .pth file has a same-name .txt descriptor
("key: value" lines). Preprocessing follows Aether exactly: resample to the model voxel spacing (linear), set the
voxels outside the mask segment to fill_outside_roi_value (mask_before), crop to the region of interest
(crop_before), rescale intensities (unless no_rescale_vol1), sliding-window inference, output > threshold.

Model files are searched in: the folder chosen in the Taranis settings, the Taranis download folder, and the
LSF calculator's folder. Missing models can be downloaded from the GitHub release (with SHA256 checks).
"""

import ast
import dataclasses
import logging
import os
import re
import time

import numpy as np
import qt
import vtk
import slicer

from . import workflow as W
from .case import settingText, SETTING_MODELS_FOLDER

# Release that holds the model files (upload the .pth and .txt files as release assets with these names)
MODEL_RELEASE_URL = "https://github.com/4burakfe/SlicerAether/releases/download/Trained_Models/"

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
    "block_size": (96, 96, 96),
    "input_intensity_vol1": (-135.0, 215.0),
    "output_intensity_vol1": (0.0, 10.0),
    "no_rescale_vol1": False,
    "dual_channel": False,
    "mask_before": False,
    "crop_before": True,
    "fill_outside_roi_value": 0.0,
    "threshold": 0.5,
}
MIN_CUDA_MEMORY_GB = 1.9
CANDIDATE_TAG = "Taranis.Candidate"   # AI result that is not accepted yet (same as controller.TARANIS_CANDIDATE_TAG)
CANDIDATE_COLOR = (1.0, 0.9, 0.0)
CROP_MARGIN_MM = 10.0

INPUT_CT = "CT"
INPUT_PET = "PET"


@dataclasses.dataclass
class ModelSpec:
    key: str
    label: str
    fileName: str
    inputKind: str
    segmentName: str
    role: str
    maskWithLiver: bool = False
    description: str = ""
    roiSizeMM: tuple = None     # the model was trained on crops around the organ: an ROI of about this size is needed
    sha256: str = ""            # of the .pth file; "" = not verified
    descriptorSha256: str = ""


MODELS = [
    ModelSpec("liver_ct", "Whole liver (CT)", "Liver_SwinUNETR24.pth", INPUT_CT, "Whole liver", W.SEGMENT_LIVER,
              description="Whole liver on a CT (also non-contrast or low-dose CT of a SPECT/CT), inside an ROI "
                          "around the liver.", roiSizeMM=(260.0, 240.0, 240.0),
              sha256="ef0d0bbb576f9ce363b4f841c877996fdd5b1c78b5d084f7f4217680208c793c",
              descriptorSha256="a958d586573dc041bcc62fa2724517916da4b35a6bbda84dbb63fb5f1fa34909"),
    ModelSpec("lungs_ct", "Lungs (CT)", "CTLUNGswin12_segmenter.pth", INPUT_CT, "Lungs", W.SEGMENT_LUNGS,
              description="Both lungs on a CT, inside an ROI covering both lungs; used for the lung shunt fraction "
                          "and lung mass.", roiSizeMM=(380.0, 260.0, 320.0),
              sha256="96aba3607a0d21bf15e7a98f163dd12e4e0c20ef5e7dce1a37233115cc7a8b39",
              descriptorSha256="88a42620435735d0f5fc3c9951e9a21777ea39734ffa9a4a76560a3bf05f4041"),
    ModelSpec("tumor_fdg", "Viable tumours (FDG PET)", "Tumor_SwinUNETR24.pth", INPUT_PET, "Viable tumors",
              W.SEGMENT_VIABLE,
              maskWithLiver=True,
              description="FDG-avid (viable) liver tumours, inside the whole-liver segment (the PET must be "
                          "registered and in SUV). Accepted as 'Viable tumour': calculated like tumours, reported "
                          "separately.",
              sha256="6171ef4094dcff8db4dee543d2392f1f0aa74be60d9c17cf724290d057d651b4",
              descriptorSha256="04710af9bfabac8b541d98015a0ec883a00f0c869c49e4778d255e90ce5bdc93"),
]
MODELS_BY_KEY = {spec.key: spec for spec in MODELS}


# -- Descriptors (pure) -------------------------------------------------------------------------------------------

def parseDescriptorValue(text):
    value = text.strip()
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    try:
        return ast.literal_eval(value)  # numbers, tuples, lists; never executes code
    except (ValueError, SyntaxError):
        return value


def parseDescriptor(text):
    descriptor = dict(DESCRIPTOR_DEFAULTS)
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        descriptor[key.strip().lower()] = parseDescriptorValue(value)
    return descriptor


def readDescriptor(modelPath):
    descriptorPath = os.path.splitext(modelPath)[0] + ".txt"
    if not os.path.isfile(descriptorPath):
        raise ValueError(f"Descriptor file not found: {descriptorPath}")
    with open(descriptorPath, "r", encoding="utf-8") as f:
        return parseDescriptor(f.read())


def rescaleIntensity(array, inputRange, outputRange):
    """Linear rescale with clipping (as MONAI ScaleIntensityRange with clip=True)."""
    (a, b), (c, d) = inputRange, outputRange
    scaled = (np.asarray(array, dtype=np.float32) - a) / float(b - a) * (d - c) + c
    return np.clip(scaled, min(c, d), max(c, d))


# -- Model files ------------------------------------------------------------------------------------------------------

def downloadFolder():
    base = qt.QStandardPaths.writableLocation(qt.QStandardPaths.GenericDataLocation)
    return os.path.join(base, "slicer.org", "Taranis", "Models")


def modelFolders():
    folders = []
    custom = settingText(SETTING_MODELS_FOLDER, "")
    if custom:
        folders.append(custom)
    folders.append(downloadFolder())
    lsf = getattr(slicer.modules, "lsfcalc", None)
    if lsf is not None:
        moduleDir = os.path.dirname(lsf.path)
        folders += [moduleDir, os.path.join(moduleDir, "Resources", "Models")]
    return folders


def findModel(spec):
    """Path of the model file (with its descriptor next to it), or None."""
    for folder in modelFolders():
        path = os.path.join(folder, spec.fileName)
        if os.path.isfile(path) and os.path.isfile(os.path.splitext(path)[0] + ".txt"):
            return path
    return None


def downloadModel(spec):
    """Download the model and its descriptor from the release into the download folder. Returns the path."""
    import SampleData
    folder = downloadFolder()
    os.makedirs(folder, exist_ok=True)
    logic = SampleData.SampleDataLogic()
    descriptorName = os.path.splitext(spec.fileName)[0] + ".txt"
    for name, checksum in ((descriptorName, spec.descriptorSha256), (spec.fileName, spec.sha256)):
        path = logic.downloadFile(MODEL_RELEASE_URL + name, folder, name,
                                  f"SHA256:{checksum}" if checksum else None)
        if not path or not os.path.isfile(path):
            raise RuntimeError(f"Could not download {name} from {MODEL_RELEASE_URL}.")
    return os.path.join(folder, spec.fileName)


def missingPackage():
    """None if PyTorch, MONAI and einops can be imported, otherwise the missing package name."""
    import importlib.util
    for package in ("torch", "monai", "einops"):
        if importlib.util.find_spec(package) is None:
            return package
    return None


# -- Network (as in Aether; the wrapper classes define the state_dict key names) ----------------------------------

def buildModel(descriptor, device):
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


def chooseDevice(torch, forceCPU=False):
    if not forceCPU and torch.cuda.is_available():
        try:
            memoryGB = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            if memoryGB >= MIN_CUDA_MEMORY_GB:
                return torch.device("cuda"), f"GPU ({memoryGB:.1f} GB)"
        except Exception as e:
            logging.warning(f"Taranis: could not check the GPU ({e}); using the CPU.")
    return torch.device("cpu"), "CPU"


# -- Geometry helpers ---------------------------------------------------------------------------------------------

def _ijkToRas(volumeNode):
    matrix = vtk.vtkMatrix4x4()
    volumeNode.GetIJKToRASMatrix(matrix)
    return np.array([[matrix.GetElement(r, c) for c in range(4)] for r in range(4)])


def _setIjkToRas(volumeNode, array):
    matrix = vtk.vtkMatrix4x4()
    for r in range(4):
        for c in range(4):
            matrix.SetElement(r, c, float(array[r][c]))
    volumeNode.SetIJKToRASMatrix(matrix)


ROI_ATTRIBUTE = "Taranis.ModelROI"   # on the ROI node: model key


def modelRoi(spec, create=False, volumeNode=None):
    """The ROI node of a model (one per model and scene). A new ROI is centred on the slice views' centre (or on
    the volume) with the model's default size."""
    for node in slicer.util.getNodesByClass("vtkMRMLMarkupsROINode"):
        if node.GetAttribute(ROI_ATTRIBUTE) == spec.key:
            return node
    if not create:
        return None
    roi = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode", f"ROI {spec.segmentName}")
    roi.SetAttribute(ROI_ATTRIBUTE, spec.key)
    roi.CreateDefaultDisplayNodes()
    color = W.SEGMENT_ROLES[spec.role][2] if spec.role in W.SEGMENT_ROLES else (1.0, 0.9, 0.0)
    roi.GetDisplayNode().SetSelectedColor(*color)
    roi.GetDisplayNode().SetColor(*color)
    center = None
    layoutManager = slicer.app.layoutManager()
    red = layoutManager.sliceWidget("Red") if layoutManager else None
    if red is not None:
        matrix = red.mrmlSliceNode().GetSliceToRAS()
        center = [matrix.GetElement(r, 3) for r in range(3)]
    size = list(spec.roiSizeMM or (200.0, 200.0, 200.0))
    if volumeNode is not None:
        bounds = [0.0] * 6
        volumeNode.GetRASBounds(bounds)
        extent = [bounds[2 * i + 1] - bounds[2 * i] for i in range(3)]
        size = [min(s, e) if e > 0 else s for s, e in zip(size, extent)]
        inside = center is not None and all(bounds[2 * i] <= center[i] <= bounds[2 * i + 1] for i in range(3))
        if not inside:
            center = [(bounds[2 * i] + bounds[2 * i + 1]) / 2.0 for i in range(3)]
    roi.SetSize(*size)
    roi.SetCenter(*(center or [0.0, 0.0, 0.0]))
    return roi


def cropBox(mask, marginVoxels):
    """(lower [k, j, i], upper exclusive) of the mask's bounding box grown by marginVoxels (clipped)."""
    k, j, i = np.nonzero(mask)
    lower = np.maximum(np.array([k.min(), j.min(), i.min()]) - marginVoxels, 0)
    upper = np.minimum(np.array([k.max(), j.max(), i.max()]) + 1 + marginVoxels, mask.shape)
    return lower, upper


def boxFromRoi(roiNode, volumeNode, shape):
    bounds = [0.0] * 6
    roiNode.GetRASBounds(bounds)
    corners = np.array([[x, y, z, 1.0] for x in bounds[0:2] for y in bounds[2:4] for z in bounds[4:6]])
    ijk = (np.linalg.inv(_ijkToRas(volumeNode)) @ corners.T)[:3].T   # i, j, k
    lower = np.maximum(np.floor(ijk.min(axis=0))[::-1].astype(int), 0)
    upper = np.minimum(np.ceil(ijk.max(axis=0))[::-1].astype(int) + 1, shape)
    if np.any(upper - lower < 2):
        raise ValueError("The ROI does not overlap the image.")
    return lower, upper


# -- Inference ------------------------------------------------------------------------------------------------------

def runModel(spec, inputVolume, liverSegmentation=None, liverSegmentID=None, roiNode=None, scale=1.0,
             forceCPU=False, modelPath=None):
    """Run a model on inputVolume. Returns (label map node with the binary result, message). The label map is a
    temporary node (hidden) in the geometry of the resampled, cropped input; the caller imports and removes it.
    scale multiplies the input values after resampling (e.g. Bq/mL -> SUV)."""
    missing = missingPackage()
    if missing:
        raise RuntimeError(f"The Python package '{missing}' is not installed (needed for AI segmentation). Install "
                           "PyTorch (PyTorch extension) and MONAI, then restart Slicer.")
    import torch
    from monai.inferers import sliding_window_inference

    modelPath = modelPath or findModel(spec)
    if modelPath is None:
        raise ValueError(f"The model {spec.fileName} was not found. Download it or select the AI models folder "
                         "in the Taranis settings.")
    descriptor = readDescriptor(modelPath)
    if descriptor["dual_channel"]:
        raise ValueError(f"{spec.fileName} is a dual-channel model; only single-channel models are supported.")
    if spec.roiSizeMM and roiNode is None:
        raise ValueError(f"{spec.label}: place the ROI around the organ first (the model was trained on crops "
                         "around it; without an ROI it also marks unrelated structures).")
    needsLiver = spec.maskWithLiver or descriptor["mask_before"]
    if needsLiver and (liverSegmentation is None or not liverSegmentID):
        raise ValueError(f"{spec.label}: a whole-liver segment is needed (the model only looks inside the liver).")

    temporary = []
    model = None
    try:
        # 1. Resample to the model's voxel spacing
        resampled = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "Taranis AI resampled")
        resampled.SetHideFromEditors(True)
        temporary.append(resampled)
        spacing = ",".join(str(float(v)) for v in descriptor["voxel_spacing"])
        from .memory import runCliAndRemove
        runCliAndRemove(slicer.modules.resamplescalarvolume, {
            "InputVolume": inputVolume.GetID(), "OutputVolume": resampled.GetID(),
            "outputPixelSpacing": spacing, "interpolationType": "linear"})
        transformNode = inputVolume.GetParentTransformNode()
        if transformNode is not None:
            # the CLI ignores parent transforms: place the resampled copy where the input is shown (a linear
            # transform only changes the image geometry, a deformable one resamples it)
            resampled.SetAndObserveTransformNodeID(transformNode.GetID())
            slicer.vtkSlicerTransformLogic().hardenTransform(resampled)
        image = slicer.util.arrayFromVolume(resampled).astype(np.float32) * float(scale)

        # 2. Mask with the liver (outside value as in training)
        liverMask = None
        if needsLiver:
            labelmap = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
            labelmap.SetHideFromEditors(True)
            temporary.append(labelmap)
            if not slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(
                    liverSegmentation, [liverSegmentID], labelmap, resampled):
                raise RuntimeError("Could not export the liver segment onto the image grid.")
            liverMask = slicer.util.arrayFromVolume(labelmap) > 0
            if not liverMask.any():
                raise ValueError("The whole-liver segment does not overlap the image.")
            if descriptor["mask_before"]:
                image[~liverMask] = float(descriptor["fill_outside_roi_value"])

        # 3. Crop: ROI if given, otherwise the liver (with a margin) when masking, otherwise the whole image
        lower, upper = np.zeros(3, int), np.array(image.shape)
        if descriptor["crop_before"]:
            if roiNode is not None:
                lower, upper = boxFromRoi(roiNode, resampled, image.shape)
            elif liverMask is not None:
                marginVoxels = int(np.ceil(CROP_MARGIN_MM / min(descriptor["voxel_spacing"])))
                lower, upper = cropBox(liverMask, marginVoxels)
        region = tuple(slice(a, b) for a, b in zip(lower, upper))
        image = np.ascontiguousarray(image[region])
        if liverMask is not None:
            liverMask = liverMask[region]

        # 4. Intensities
        if not descriptor["no_rescale_vol1"]:
            image = rescaleIntensity(image, descriptor["input_intensity_vol1"], descriptor["output_intensity_vol1"])

        # 5. Inference
        device, deviceText = chooseDevice(torch, forceCPU)
        model = buildModel(descriptor, device)
        model.load_state_dict(torch.load(modelPath, map_location=device))
        model.eval()
        start = time.time()
        with torch.no_grad():
            tensor = torch.from_numpy(image)[None, None].to(device)
            output = sliding_window_inference(inputs=tensor, roi_size=tuple(descriptor["block_size"]),
                                              sw_batch_size=1, predictor=model, overlap=0.25, mode="gaussian")
            outputArray = output.squeeze().cpu().numpy()
        elapsed = time.time() - start
        if outputArray.ndim != 3:
            raise ValueError(f"{spec.fileName} returned {output.shape[1]} channels; a single-channel model is "
                             "expected.")
        mask = np.clip(outputArray, 0, None) > float(descriptor["threshold"])
        if liverMask is not None and spec.maskWithLiver:
            mask &= liverMask

        # 6. Result as a label map in the cropped geometry
        result = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", f"{spec.segmentName} (AI)")
        result.SetHideFromEditors(True)
        slicer.util.updateVolumeFromArray(result, mask.astype(np.uint8))
        offset = np.eye(4)
        offset[:3, 3] = lower[::-1]  # (i, j, k) of the first voxel
        _setIjkToRas(result, _ijkToRas(resampled) @ offset)
        return result, (f"{spec.label}: {int(mask.sum())} voxels found in {elapsed:.1f} s on {deviceText}."
                        if mask.any() else f"{spec.label}: nothing found.")
    finally:
        from .memory import removeTemporaryLabelmap
        for node in temporary:
            if node.GetScene() is not None:
                if node.IsA("vtkMRMLLabelMapVolumeNode"):
                    removeTemporaryLabelmap(node)
                else:
                    slicer.mrmlScene.RemoveNode(node)
        del model
        import gc
        gc.collect()
        from .memory import trimHeap
        trimHeap()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def importCandidate(segmentationNode, labelmapNode, name, role=""):
    """Import the AI result as a candidate segment (yellow, tagged with the role it will get when accepted) and
    remove the label map. Returns the segment ID, or None if the result is empty."""
    try:
        if not np.any(slicer.util.arrayFromVolume(labelmapNode)):
            return None
        segmentation = segmentationNode.GetSegmentation()
        before = {segmentation.GetNthSegmentID(i) for i in range(segmentation.GetNumberOfSegments())}
        if not slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelmapNode, segmentationNode):
            raise RuntimeError("Could not import the AI result into the segmentation.")
        newIDs = [segmentation.GetNthSegmentID(i) for i in range(segmentation.GetNumberOfSegments())
                  if segmentation.GetNthSegmentID(i) not in before]
        if not newIDs:
            return None
        from .segtools import ownLayer
        ownLayer(segmentationNode, newIDs[0])   # so that the candidate never takes voxels from other segments
        segment = segmentation.GetSegment(newIDs[0])
        segment.SetName(f"{name} (AI - evaluate)")
        segment.SetColor(*CANDIDATE_COLOR)
        segment.SetTag(CANDIDATE_TAG, role or "1")
        return newIDs[0]
    finally:
        from .memory import removeTemporaryLabelmap
        removeTemporaryLabelmap(labelmapNode)


def candidateRole(segment):
    """Role an AI candidate gets when accepted ("" if unknown), None if the segment is not a candidate."""
    from .controller import segmentTag
    if not segment.HasTag(CANDIDATE_TAG):
        return None
    value = segmentTag(segment, CANDIDATE_TAG)
    return "" if value in ("", "1") else value


def acceptCandidate(segmentationNode, segmentID, name, role):
    """Turn a candidate into a regular segment with a role (and the role's colour)."""
    from .controller import setSegmentRole
    segment = segmentationNode.GetSegmentation().GetSegment(segmentID)
    segment.RemoveTag(CANDIDATE_TAG)
    segment.RemoveTag("LSFcalc.Candidate")
    segment.SetName(name)
    if role:
        setSegmentRole(segmentationNode, segmentID, role)


def mergedLabelmap(segmentationNode, segmentIDs, name="Taranis merged"):
    """Hidden label map node with 1 inside any of the segments."""
    labelmap = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", name)
    labelmap.SetHideFromEditors(True)
    ids = vtk.vtkStringArray()
    for segmentID in segmentIDs:
        ids.InsertNextValue(segmentID)
    if not slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(segmentationNode, ids, labelmap):
        from .memory import removeTemporaryLabelmap
        removeTemporaryLabelmap(labelmap)
        raise RuntimeError("Could not export the segments.")
    array = slicer.util.arrayFromVolume(labelmap)
    array[:] = array > 0
    slicer.util.arrayFromVolumeModified(labelmap)
    return labelmap


# -- TotalSegmentator (liver on MRI, liver/lungs on CT when the Taranis models are not available) -----------------

TOTALSEG_STRUCTURES = {
    # structure: {"CT"/"MR": (task, [TotalSegmentator class names], licence needed)}
    W.SEGMENT_LIVER: {"CT": ("total", ["liver"], False), "MR": ("total_mr", ["liver"], False)},
    W.SEGMENT_LUNGS: {"CT": ("total", ["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
                                       "lung_middle_lobe_right", "lung_lower_lobe_right"], False),
                      "MR": ("total_mr", ["lung_left", "lung_right"], False)},
    W.SEGMENT_TUMOR: {"CT": ("liver_lesions", ["liver_lesions"], True),
                      "MR": ("liver_lesions_mr", ["liver_lesions"], True)},
}
TOTALSEG_LICENSE_NOTE = ("The TotalSegmentator liver-lesion tasks need a TotalSegmentator licence (free for "
                         "non-commercial use): request it on the TotalSegmentator website and enter it in the "
                         "TotalSegmentator module (Advanced section).")


def totalSegmentatorAvailable():
    return hasattr(slicer.modules, "totalsegmentator")


def _normalName(text):
    return str(text).lower().replace("_", " ").replace("-", " ").strip()


def runTotalSegmentator(inputVolume, structure, isMRI, forceCPU=False):
    """Run TotalSegmentator for one structure (liver, lungs or liver tumours on CT or MRI). Returns (label map
    node, message) like runModel."""
    if not totalSegmentatorAvailable():
        raise RuntimeError("The TotalSegmentator extension is not installed (Extensions Manager: TotalSegmentator).")
    import inspect
    from TotalSegmentator import TotalSegmentatorLogic
    task, classes, licensed = TOTALSEG_STRUCTURES[structure]["MR" if isMRI else "CT"]
    logic = TotalSegmentatorLogic()
    parameters = inspect.signature(logic.process).parameters
    kwargs = {"task": task}
    if task in ("total", "total_mr") and "subset" in parameters:
        kwargs["subset"] = classes
    if "cpu" in parameters:
        kwargs["cpu"] = bool(forceCPU)
    if "fast" in parameters:
        kwargs["fast"] = False
    output = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "Taranis TotalSegmentator")
    try:
        start = time.time()
        try:
            logic.process(inputVolume, output, **kwargs)
        except Exception as e:
            if licensed:
                raise RuntimeError(f"TotalSegmentator task '{task}' failed: {e}\n\n{TOTALSEG_LICENSE_NOTE}")
            raise
        elapsed = time.time() - start
        wanted = {_normalName(c) for c in classes}
        segmentation = output.GetSegmentation()
        found = []
        for index in range(segmentation.GetNumberOfSegments()):
            segmentID = segmentation.GetNthSegmentID(index)
            names = {_normalName(segmentID), _normalName(segmentation.GetSegment(segmentID).GetName())}
            keywords = {W.SEGMENT_LUNGS: ("lung",), W.SEGMENT_TUMOR: ("lesion", "tumor", "tumour")}.get(structure, ())
            if names & wanted or any(k in n for k in keywords for n in names):
                found.append(segmentID)
        if not found:
            if structure == W.SEGMENT_TUMOR:
                return None, f"TotalSegmentator ({task}): no liver lesion found ({elapsed:.0f} s)."
            raise ValueError(f"TotalSegmentator did not return {', '.join(classes)}.")
        labelmap = mergedLabelmap(output, found, "TotalSegmentator result")
        return labelmap, f"TotalSegmentator ({task}): done in {elapsed:.0f} s."
    finally:
        if output.GetScene() is not None:
            slicer.mrmlScene.RemoveNode(output)


# -- PET units ------------------------------------------------------------------------------------------------------

SUV_QUANTITY_CODE = "126400"     # DCM "Standardized Uptake Value"
SUV_PLAUSIBLE_MAX = 100.0        # 99.9th percentile above this: not SUV
SUV_IMPLAUSIBLY_LOW = 0.5        # 99.9th percentile of a converted PET below this: the conversion was wrong


def voxelSuvType(volumeNode):
    """"SUVbw", "SUVlbm", ... when the volume node says its voxels are SUV (Slicer's PET/DICOM real-world value
    mapping sets the quantity and units, e.g. "{SUVbw}g/ml"), otherwise ""."""
    texts = []
    for getter in ("GetVoxelValueQuantity", "GetVoxelValueUnits"):
        try:
            code = getattr(volumeNode, getter)()
        except Exception:
            code = None
        if code is not None:
            texts += [code.GetCodeValue() or "", code.GetCodeMeaning() or ""]
    text = " ".join(texts)
    match = re.search(r"SUV(bw|lbm|bsa|ibw)", text, re.IGNORECASE)
    if match:
        return "SUV" + match.group(1).lower()
    if SUV_QUANTITY_CODE in texts or re.search(r"SUV|standardized uptake", text, re.IGNORECASE):
        return "SUV"
    return ""


def suvScale(volumeNode):
    """(factor converting the PET values to SUVbw, explanation). Factor 1 when the image is already in SUV.
    The volume's own units come first: a PET loaded as SUV keeps "Bq/mL" in its DICOM header."""
    from .case import _dicomHeader, volumeInfo
    from . import segmentops
    suvType = voxelSuvType(volumeNode)
    if suvType:
        if suvType in ("SUV", "SUVbw"):
            return 1.0, f"PET already in {suvType}."
        return 1.0, (f"PET in {suvType}: used as it is, but the model was trained on SUVbw (load the PET as SUVbw "
                     "for the intended thresholds).")
    info = volumeInfo(volumeNode)
    if info.units == "GML":
        return 1.0, "PET in SUV (g/mL)."
    array = slicer.util.arrayFromVolume(volumeNode)
    robustMax = float(np.percentile(array, 99.9))
    if info.units == "BQML":
        dataset = _dicomHeader(volumeNode)
        if dataset is None:
            raise ValueError("The PET is in Bq/mL but its DICOM header is not available for SUV conversion.")
        sequence = dataset.get("RadiopharmaceuticalInformationSequence")
        item = sequence[0] if sequence else {}

        def dicomTime(date, timeText):
            import datetime
            from .roles import dicomDateTime
            text = dicomDateTime(date, timeText)
            return datetime.datetime.fromisoformat(text) if text else None

        injectionDateTime = str(item.get("RadiopharmaceuticalStartDateTime", "") or "")
        injection = (dicomTime(injectionDateTime[:8], injectionDateTime[8:]) if injectionDateTime else
                     dicomTime(str(dataset.get("SeriesDate", "")), str(item.get("RadiopharmaceuticalStartTime", ""))))
        series = dicomTime(str(dataset.get("SeriesDate", "")), str(dataset.get("SeriesTime", "")))
        factor = segmentops.suvFactor(float(dataset.get("PatientWeight", 0) or 0),
                                      float(item.get("RadionuclideTotalDose", 0) or 0), injection, series,
                                      float(item.get("RadionuclideHalfLife", 0) or 0),
                                      str(dataset.get("DecayCorrection", "START") or "START"))
        if robustMax * factor < SUV_IMPLAUSIBLY_LOW and robustMax <= SUV_PLAUSIBLE_MAX:
            # the header says Bq/mL but the voxels are already SUV-like (converted when loaded)
            return 1.0, (f"PET values look like SUV already (99.9th percentile {robustMax:.1f}) although the DICOM "
                         "header says Bq/mL: used as they are.")
        return factor, "PET converted from Bq/mL to SUVbw with the DICOM header."
    if robustMax > SUV_PLAUSIBLE_MAX:
        raise ValueError("The PET units are unknown and the values do not look like SUV (99.9th percentile "
                         f"{robustMax:.0f}). Load the PET as SUV (e.g. PET DICOM extension) or from DICOM.")
    return 1.0, "PET units unknown; the values look like SUV and are used as they are."
