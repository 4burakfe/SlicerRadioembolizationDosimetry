"""Lung shunt fraction of the case: counts of the dosimetry image (MAA SPECT) in the lungs and the whole liver, lung
mass from the CT and the estimated lung dose.

The pure functions (top) are unit-tested outside Slicer; calculateFromCase / lungMassFromCase run in Slicer.
"""

import json
import math

import numpy as np

from . import doseguard as DG
from . import workflow as W

Y90_GY_KG_PER_GBQ = W.Y90_GY_KG_PER_GBQ
LUNG_DOSE_SESSION_LIMIT_GY = W.LUNG_DOSE_SESSION_LIMIT_GY
LUNG_DOSE_CUMULATIVE_LIMIT_GY = W.LUNG_DOSE_CUMULATIVE_LIMIT_GY
DEFAULT_LUNG_MASS_G = W.DEFAULT_LUNG_MASS_G
LUNG_COVERAGE_WARNING = 0.85           # part of the lung segment inside the SPECT field of view
LUNG_DENSITY_RANGE = (0.0, 1.1)        # g/mL clipped from HU (air .. soft tissue)
LUNG_MIN_ML_FOR_MASS = 1500.0          # smaller lung segments are probably cut by the CT field of view
SOURCE_IMAGE = "SPECT counts (Taranis)"


# -- Pure --------------------------------------------------------------------------------------------------------

def computeLungShunt(values, lungMask, liverMask, clipNegativeValues=True):
    """Lung shunt from the image values inside the two masks (same grid). A voxel claimed by both masks (partial
    volume at the lung-liver boundary) is counted once, for the liver. Raises ValueError for empty masks or
    non-positive totals. Same method as the LSF calculator."""
    lungMask = np.asarray(lungMask, dtype=bool)
    liverMask = np.asarray(liverMask, dtype=bool)
    if not lungMask.any():
        raise ValueError("The lung segment does not cover any voxel of the dosimetry image (outside its field of "
                         "view?).")
    if not liverMask.any():
        raise ValueError("The whole-liver segment does not cover any voxel of the dosimetry image.")
    shared = lungMask & liverMask
    lungOnly = lungMask & ~shared
    values = np.asarray(values, dtype=np.float64)
    negativeVoxels = int(np.count_nonzero(values[lungOnly | liverMask] < 0))
    if clipNegativeValues:
        values = np.where(values < 0, 0.0, values)
    lungCounts = float(values[lungOnly].sum())
    liverCounts = float(values[liverMask].sum())
    if lungCounts < 0 or liverCounts <= 0:
        raise ValueError("Lung or liver counts are not positive (negative voxel values?).")
    positive = np.clip(values, 0, None)
    total = float(positive.sum())
    extraFraction = float(positive[~(lungMask | liverMask)].sum()) / total if total > 0 else None
    return {"lungCounts": lungCounts, "liverCounts": liverCounts,
            "lsfPercent": 100.0 * lungCounts / (lungCounts + liverCounts),
            "extraFraction": extraFraction,   # image counts outside the lungs and the whole liver
            "lungVoxels": int(np.count_nonzero(lungOnly)), "liverVoxels": int(np.count_nonzero(liverMask)),
            "sharedVoxels": int(np.count_nonzero(shared)), "negativeVoxels": negativeVoxels}


LUNG_MASS_REFERENCE = ("Kao YH, Magsombol BM, Toh Y, et al. Personalized predictive lung dosimetry by technetium-99m "
                       "macroaggregated albumin SPECT/CT for yttrium-90 radioembolization. EJNMMI Res. 2014;4:33. "
                       "doi:10.1186/s13550-014-0033-7")
LUNG_MASS_METHOD_HTML = ("Lung mass from CT (CT densitovolumetry): physical density = (CT number + 1000) / 1000 g/mL, "
                         "mass = lung volume × density, summed voxel by voxel here (density clipped to 0–1.1 g/mL). "
                         "Method: Kao YH et al., <i>EJNMMI Res</i> 2014;4:33, "
                         "<a href='https://doi.org/10.1186/s13550-014-0033-7'>doi:10.1186/s13550-014-0033-7</a>.")


def lungMassFromHU(huValues, voxelML):
    """Lung mass (g) from the CT numbers inside the lungs (CT densitovolumetry): physical density (g/mL) =
    (CT number + 1000) / 1000 and mass = volume x density (see LUNG_MASS_REFERENCE; Kao et al. used the lung volume
    and the mean CT number, which equals this voxel-wise sum without clipping). Density clipped to
    LUNG_DENSITY_RANGE (air .. soft tissue)."""
    hu = np.asarray(huValues, dtype=np.float64)
    density = np.clip((hu + 1000.0) / 1000.0, *LUNG_DENSITY_RANGE)
    return float(density.sum() * voxelML)


def lungDoseGy(activityGBq, lsfPercent, lungMassG):
    """Estimated mean lung dose (Gy) of a Y-90 administration: A x LSF x 49.67 / lung mass (kg)."""
    if not activityGBq or activityGBq <= 0 or lsfPercent is None or not lungMassG or lungMassG <= 0:
        return None
    return Y90_GY_KG_PER_GBQ * activityGBq * (lsfPercent / 100.0) / (lungMassG / 1000.0)


def lungDosePerGBq(lsfPercent, lungMassG):
    return lungDoseGy(1.0, lsfPercent, lungMassG)


def lsfIssues(result, lungSegmentML=None, lungMassNote=""):
    """[(severity, text)] of a calculation, for the toolbar (severity names as in workflow)."""
    issues = []
    coverage = result.get("lungCoverage")
    if coverage is not None and coverage < LUNG_COVERAGE_WARNING:
        issues.append((W.SEVERITY_WARNING,
                       f"Only {100 * coverage:.0f}% of the lung segment is inside the field of view of the dosimetry "
                       "image: the lung counts are too low and the LSF is underestimated."))
    if lungSegmentML is not None and lungSegmentML < LUNG_MIN_ML_FOR_MASS:
        issues.append((W.SEVERITY_INFO,
                       f"The lung segment is {lungSegmentML:.0f} mL: the lungs may be cut by the field of view."))
    extra = result.get("extraFraction")
    if extra is not None and extra > DG.EXTRA_UPTAKE_FRACTION:
        issues.append((W.SEVERITY_WARNING,
                       f"Significant uptake outside the whole liver and the lungs: {100 * extra:.1f}% of the image "
                       "counts. It may be related to free Tc-99m pertechnetate (stomach, thyroid, kidneys), the "
                       "reconstruction method and noise, or segmentation errors: double-check the image and the "
                       "segments."))
    if result.get("negativeVoxels"):
        issues.append((W.SEVERITY_INFO,
                       f"{result['negativeVoxels']} negative voxel values in the segments were set to 0."))
    if lungMassNote:
        issues.append((W.SEVERITY_INFO, lungMassNote))
    return issues


def inputsKey(segments, dosimetryImageID):
    """Identifies the inputs of an image-based LSF: dosimetry image and the lung / whole-liver segment contents."""
    parts = [f"{s.segmentID}:{s.voxels}" for s in sorted(segments, key=lambda s: s.segmentID)
             if s.role in (W.SEGMENT_LUNGS, W.SEGMENT_LIVER) and not s.candidate]
    return f"{dosimetryImageID}|" + ";".join(parts)


# -- Slicer ----------------------------------------------------------------------------------------------------

def _maskOnGrid(segmentationNode, segmentIDs, volumeNode):
    """Union of segments as a boolean array on the voxel grid of volumeNode (transforms applied)."""
    import slicer
    import vtk
    labelmap = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "Taranis LSF mask")
    labelmap.SetHideFromEditors(True)
    try:
        ids = vtk.vtkStringArray()
        for segmentID in segmentIDs:
            ids.InsertNextValue(segmentID)
        if not slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(segmentationNode, ids, labelmap,
                                                                                volumeNode):
            raise RuntimeError("Could not export the segments onto the image grid.")
        return slicer.util.arrayFromVolume(labelmap) > 0
    finally:
        from .memory import removeTemporaryLabelmap
        removeTemporaryLabelmap(labelmap)


def _voxelML(volumeNode):
    spacing = volumeNode.GetSpacing()
    return spacing[0] * spacing[1] * spacing[2] / 1000.0


def caseInputs(case, segments):
    """(dosimetry image, segmentation, [lung segment IDs], liver segment ID or None, message or "")."""
    from . import roles as R
    image = case.roleNode(R.ROLE_DOSIMETRY)
    segmentationNode = case.roleNode(R.ROLE_SEGMENTATION)
    lungs = [s.segmentID for s in segments if s.role == W.SEGMENT_LUNGS and not s.candidate]
    livers = [s.segmentID for s in segments if s.role == W.SEGMENT_LIVER and not s.candidate]
    message = ""
    if image is None:
        message = "Assign the dosimetry image in the Data step."
    elif segmentationNode is None:
        message = "Create the segmentation (Segmentation step)."
    elif not lungs:
        message = "No lung segment: segment the lungs in the Segmentation step (AI on the CT or TotalSegmentator)."
    elif not livers:
        message = "No whole-liver segment."
    elif len(livers) > 1:
        message = "Several whole-liver segments: keep one."
    return image, segmentationNode, lungs, (livers[0] if len(livers) == 1 else None), message


def calculateFromCase(case, segments, clipNegativeValues=True):
    """LSF from the dosimetry image counts. Returns the result dict (see computeLungShunt) with lungML, liverML
    (on the image grid), lungSegmentML and lungCoverage (part of the lung segment inside the image)."""
    image, segmentationNode, lungs, liver, message = caseInputs(case, segments)
    if message:
        raise ValueError(message)
    import slicer
    values = slicer.util.arrayFromVolume(image)
    lungMask = _maskOnGrid(segmentationNode, lungs, image)
    liverMask = _maskOnGrid(segmentationNode, [liver], image)
    result = computeLungShunt(values, lungMask, liverMask, clipNegativeValues)
    voxelML = _voxelML(image)
    lungSegmentML = sum((s.volumeML or 0.0) for s in segments if s.segmentID in lungs)
    lungOnImageML = float(np.count_nonzero(lungMask)) * voxelML
    result.update({"lungML": result["lungVoxels"] * voxelML, "liverML": result["liverVoxels"] * voxelML,
                   "lungSegmentML": lungSegmentML,
                   "lungCoverage": min(1.0, lungOnImageML / lungSegmentML) if lungSegmentML else None,
                   "imageID": image.GetID(), "imageName": image.GetName(), "clipped": clipNegativeValues})
    return result


def lungMassFromCase(case, segments):
    """(lung mass g, CT name, note) from the CT numbers inside the lung segments. The CT is the dosimetry image's CT
    or the reference image, if it is a CT."""
    from . import roles as R
    ct = None
    for role in (R.ROLE_DOSIMETRY_ANATOMY, R.ROLE_REFERENCE, R.ROLE_METABOLIC_ANATOMY):
        node = case.roleNode(role)
        if node is not None and case.roleType(role) == R.TYPE_CT:
            ct = node
            break
    if ct is None:
        raise ValueError("No CT among the case images (lung mass needs CT numbers).")
    lungs = [s.segmentID for s in segments if s.role == W.SEGMENT_LUNGS and not s.candidate]
    if not lungs:
        raise ValueError("No lung segment.")
    import slicer
    mask = _maskOnGrid(case.roleNode(R.ROLE_SEGMENTATION), lungs, ct)
    if not mask.any():
        raise ValueError(f"The lung segment is outside '{ct.GetName()}'.")
    voxelML = _voxelML(ct)
    mass = lungMassFromHU(slicer.util.arrayFromVolume(ct)[mask], voxelML)
    segmentML = sum((s.volumeML or 0.0) for s in segments if s.segmentID in lungs)
    onCTML = float(np.count_nonzero(mask)) * voxelML
    note = ""
    if segmentML and onCTML < 0.95 * segmentML:
        note = (f"Only {100 * onCTML / segmentML:.0f}% of the lung segment is inside '{ct.GetName()}': the lung mass "
                "is underestimated.")
    elif segmentML and segmentML < LUNG_MIN_ML_FOR_MASS:
        note = f"Lung segment of {segmentML:.0f} mL: the lungs may be cut by the CT field of view (mass too low)."
    return mass, ct.GetName(), note


def detailsJson(result, key):
    keep = ("lungCounts", "liverCounts", "lsfPercent", "extraFraction", "lungML", "liverML", "lungSegmentML", "lungCoverage",
            "sharedVoxels", "negativeVoxels", "imageName")
    details = {k: result.get(k) for k in keep}
    details["key"] = key
    return json.dumps(details)


def isFinite(value):
    return value is not None and math.isfinite(value)
