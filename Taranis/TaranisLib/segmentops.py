"""Mask operations of the segmentation step (pure numpy / scipy, testable outside Slicer).

Masks are boolean arrays [k, j, i] on one voxel grid; voxelML is the volume of one voxel in mL.
"""

import math

import numpy as np


def volumeML(mask, voxelML):
    return float(np.count_nonzero(mask)) * voxelML


def splitComponents(mask, minVoxels=1, connectivity=1):
    """Connected components of a mask, largest first, each at least minVoxels voxels."""
    from scipy import ndimage
    structure = ndimage.generate_binary_structure(3, connectivity)
    labels, count = ndimage.label(mask, structure=structure)
    if count == 0:
        return []
    sizes = np.bincount(labels.ravel())[1:]
    order = np.argsort(sizes)[::-1]
    return [labels == (index + 1) for index in order if sizes[index] >= minVoxels]


def fillAxialHoles(mask):
    from scipy import ndimage
    filled = np.zeros_like(mask, dtype=bool)
    for k in range(mask.shape[0]):
        filled[k] = ndimage.binary_fill_holes(mask[k])
    return filled


def normalTissue(liver, tumours):
    """Liver minus the union of the tumours."""
    result = liver.astype(bool).copy()
    for tumour in tumours:
        result &= ~tumour.astype(bool)
    return result


def clipToContainer(mask, container):
    """(mask inside the container, number of voxels removed)."""
    inside = mask.astype(bool) & container.astype(bool)
    return inside, int(np.count_nonzero(mask) - np.count_nonzero(inside))


def perfusedFromUptake(values, liver, percent, minVoxels=1, fillHoles=True):
    """Perfused territory from a MAA (or Y-90) image resampled onto the liver grid: liver voxels with at least
    percent of the robust maximum inside the liver (99.9th percentile), holes filled per axial slice, components
    smaller than minVoxels dropped. Returns a list of components (largest first); several components usually
    mean several territories or noise."""
    liver = liver.astype(bool)
    inside = np.asarray(values, dtype=float)[liver]
    inside = inside[np.isfinite(inside)]
    if inside.size == 0 or inside.max() <= 0:
        raise ValueError("No uptake inside the liver.")
    threshold = float(np.percentile(inside, 99.9)) * percent / 100.0
    mask = liver & (np.asarray(values, dtype=float) >= threshold)
    if fillHoles:
        mask = fillAxialHoles(mask) & liver
    return splitComponents(mask, minVoxels)


def overlapReport(namedMasks, voxelML):
    """[(nameA, nameB, overlap mL)] for every overlapping pair."""
    result = []
    names = list(namedMasks)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            overlap = np.count_nonzero(namedMasks[names[a]] & namedMasks[names[b]])
            if overlap:
                result.append((names[a], names[b], overlap * voxelML))
    return result


def outsideReport(namedMasks, container, voxelML):
    """[(name, outside mL, outside fraction)] for masks that are partly outside the container."""
    result = []
    for name, mask in namedMasks.items():
        total = np.count_nonzero(mask)
        outside = np.count_nonzero(mask & ~container)
        if outside:
            result.append((name, outside * voxelML, outside / total if total else 0.0))
    return result


def perfusionReport(namedMasks, perfusedUnion, voxelML):
    """[(name, total mL, unperfused mL, unperfused fraction)] of every non-empty mask (e.g. tumours) with respect
    to the union of the perfused volumes. Fraction 1.0: the mask does not intersect any perfused volume."""
    result = []
    for name, mask in namedMasks.items():
        total = int(np.count_nonzero(mask))
        if not total:
            continue
        outside = int(np.count_nonzero(mask & ~perfusedUnion))
        result.append((name, total * voxelML, outside * voxelML, outside / total))
    return result


def mismatchFraction(mask, expected):
    """Voxels in one mask but not the other, relative to the expected mask (0: identical)."""
    size = int(np.count_nonzero(expected))
    return int(np.count_nonzero(mask ^ expected)) / size if size else (0.0 if not mask.any() else 1.0)


# -- SUV -------------------------------------------------------------------------------------------------------

def suvFactor(weightKg, injectedDoseBq, injectionTime, seriesTime, halfLifeSeconds, decayCorrection="START"):
    """Factor converting Bq/mL into body-weight SUV (g/mL): SUV = value * factor.
    With decay correction START the image is decay corrected to the series (scan) start, so the injected dose is
    decayed to that time; with ADMIN the image is already corrected to the injection time."""
    if not weightKg or weightKg <= 0:
        raise ValueError("Patient weight is missing in the DICOM header.")
    if not injectedDoseBq or injectedDoseBq <= 0:
        raise ValueError("Injected dose is missing in the DICOM header.")
    if not halfLifeSeconds or halfLifeSeconds <= 0:
        raise ValueError("Radionuclide half-life is missing in the DICOM header.")
    dose = float(injectedDoseBq)
    if str(decayCorrection).upper() == "START":
        if injectionTime is None or seriesTime is None:
            raise ValueError("Injection or scan time is missing in the DICOM header.")
        elapsed = (seriesTime - injectionTime).total_seconds()
        if elapsed < 0:
            elapsed += 24 * 3600  # injection before midnight, scan after
        dose *= math.exp(-math.log(2.0) * elapsed / float(halfLifeSeconds))
    return float(weightKg) * 1000.0 / dose
