"""Geometry for functional-only registration (a SPECT or PET without its own CT, registered to a CT or MRI).

Pure numpy / scipy (no Slicer imports) so it can be unit tested outside Slicer. Conventions:
- volume arrays are indexed [k, j, i] (as slicer.util.arrayFromVolume returns them);
- ijkToWorld is a 4x4 numpy matrix mapping (i, j, k, 1) to world RAS (mm), including any parent transform;
- rigid transforms are 4x4 numpy matrices mapping moving-image world points onto the fixed image.
"""

import numpy as np

MIN_LANDMARK_PAIRS = 3


# -- Rigid transforms --------------------------------------------------------------------------------------------

def rigidFromLandmarks(movingPoints, fixedPoints):
    """Least-squares rigid transform (rotation + translation, no scaling) mapping movingPoints onto fixedPoints
    (Kabsch). Returns (4x4 matrix, RMS error in mm, per-pair residuals in mm). Points are paired by order."""
    moving = np.asarray(movingPoints, dtype=float).reshape(-1, 3)
    fixed = np.asarray(fixedPoints, dtype=float).reshape(-1, 3)
    if len(moving) != len(fixed):
        raise ValueError(f"The number of landmarks differs: {len(moving)} on the moving image, {len(fixed)} on the "
                         "reference.")
    if len(moving) < MIN_LANDMARK_PAIRS:
        raise ValueError(f"At least {MIN_LANDMARK_PAIRS} landmark pairs are needed ({len(moving)} placed).")
    movingCentre, fixedCentre = moving.mean(axis=0), fixed.mean(axis=0)
    a, b = moving - movingCentre, fixed - fixedCentre
    if np.linalg.matrix_rank(a, tol=1.0) < 2 or np.linalg.matrix_rank(b, tol=1.0) < 2:
        raise ValueError("The landmarks are (almost) on a line: spread them out (e.g. liver dome, hilum, spleen "
                         "tip, kidneys).")
    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = fixedCentre - rotation @ movingCentre
    residuals = np.linalg.norm((moving @ rotation.T + matrix[:3, 3]) - fixed, axis=1)
    return matrix, float(np.sqrt(np.mean(residuals ** 2))), residuals


def translationMatrix(offset):
    matrix = np.eye(4)
    matrix[:3, 3] = offset
    return matrix


def rigidChange(matrix):
    """(translation in mm, rotation angle in degrees) of a rigid 4x4 matrix."""
    matrix = np.asarray(matrix, dtype=float)
    cosine = np.clip((np.trace(matrix[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.linalg.norm(matrix[:3, 3])), float(np.degrees(np.arccos(cosine)))


def changeBetween(before, after, point):
    """How far `after` moves the image compared with `before`: (displacement in mm of `point`, e.g. the liver
    centre, and rotation angle in degrees). Used as a plausibility check of an automatic refinement."""
    before, after = np.asarray(before, dtype=float), np.asarray(after, dtype=float)
    p = np.append(np.asarray(point, dtype=float), 1.0)
    source = np.linalg.solve(before, p)  # the moving-image point that `before` places at `point`
    displacement = float(np.linalg.norm((after @ source)[:3] - p[:3]))
    _, angle = rigidChange(after @ np.linalg.inv(before))
    return displacement, angle


# -- Masks ---------------------------------------------------------------------------------------------------

def otsuThreshold(values, bins=256):
    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("No image values.")
    histogram, edges = np.histogram(values, bins=bins)
    centres = (edges[:-1] + edges[1:]) / 2.0
    weight0 = np.cumsum(histogram)
    weight1 = weight0[-1] - weight0
    sum0 = np.cumsum(histogram * centres)
    mean0 = np.divide(sum0, weight0, out=np.zeros_like(sum0, dtype=float), where=weight0 > 0)
    mean1 = np.divide(sum0[-1] - sum0, weight1, out=np.zeros_like(sum0, dtype=float), where=weight1 > 0)
    between = weight0 * weight1 * (mean0 - mean1) ** 2
    best = np.flatnonzero(between >= between.max() * (1.0 - 1e-6))
    return float(centres[best].mean())  # middle of a flat optimum (well separated classes)


def largestComponent(mask):
    from scipy import ndimage
    labels, count = ndimage.label(mask)
    if count == 0:
        return mask.astype(bool)
    sizes = ndimage.sum(mask, labels, index=np.arange(1, count + 1))
    return labels == (int(np.argmax(sizes)) + 1)


def fillAxialHoles(mask):
    """Fill holes slice by slice (axial slices = first array axis), so the lungs and bowel gas of a CT or MRI and
    the photopenic regions of a SPECT count as inside the body."""
    from scipy import ndimage
    filled = np.zeros_like(mask, dtype=bool)
    for k in range(mask.shape[0]):
        filled[k] = ndimage.binary_fill_holes(mask[k])
    return filled


def bodyMask(array, threshold, smoothingVoxels=0.0):
    """Body outline: voxels above threshold (after optional Gaussian smoothing), holes filled, largest
    connected component. Raises ValueError if nothing plausible is found."""
    from scipy import ndimage
    data = np.asarray(array, dtype=np.float32)
    if smoothingVoxels > 0:
        data = ndimage.gaussian_filter(data, smoothingVoxels)
    mask = data > threshold
    if not mask.any():
        raise ValueError("No voxel is above the body-outline threshold.")
    mask = largestComponent(fillAxialHoles(mask))
    fraction = mask.mean()
    if fraction < 0.002:
        raise ValueError("The body outline found is too small: lower the threshold.")
    if fraction > 0.97:
        raise ValueError("The body outline fills the whole image: raise the threshold (or the image shows no "
                         "outline).")
    return mask


def functionalOutlineThreshold(array, percent):
    """Threshold for the body outline of a SPECT/PET: a percentage of a robust maximum (99.9th percentile)."""
    values = np.asarray(array, dtype=float)
    positive = values[values > 0]
    if positive.size == 0:
        raise ValueError("The functional image has no positive values.")
    return float(np.percentile(positive, 99.9)) * percent / 100.0


def anatomicalOutlineThreshold(array, modality):
    """Body threshold of a CT (HU) or MRI (Otsu on the non-background intensities)."""
    if modality == "CT":
        return -400.0
    values = np.asarray(array, dtype=float)[::2, ::2, ::2]
    values = values[values > np.percentile(values, 1)]
    return otsuThreshold(values) * 0.5  # below the Otsu level: MRI body signal is inhomogeneous


def dilateMask(mask, spacingKJI, marginMm):
    """Mask grown by marginMm (exact Euclidean distance, anisotropic voxels)."""
    from scipy import ndimage
    if marginMm <= 0:
        return mask.astype(bool)
    distance = ndimage.distance_transform_edt(~mask.astype(bool), sampling=spacingKJI)
    return distance <= marginMm


def maskWorldPoints(mask, ijkToWorld, maxPoints=200000, seed=0):
    """World coordinates (N x 3) of the mask voxels, randomly subsampled to at most maxPoints."""
    k, j, i = np.nonzero(mask)
    if len(i) == 0:
        return np.zeros((0, 3))
    if len(i) > maxPoints:
        choice = np.random.default_rng(seed).choice(len(i), maxPoints, replace=False)
        i, j, k = i[choice], j[choice], k[choice]
    ijk = np.stack([i, j, k, np.ones(len(i))], axis=0).astype(float)
    return (np.asarray(ijkToWorld, dtype=float) @ ijk)[:3].T


def weightedCentroid(array, mask, ijkToWorld):
    """Intensity-weighted world centroid of the voxels in mask."""
    k, j, i = np.nonzero(mask)
    weights = np.asarray(array, dtype=float)[k, j, i]
    weights = np.clip(weights, 0, None)
    if weights.sum() <= 0:
        raise ValueError("No uptake inside the region.")
    ijk = np.stack([i, j, k, np.ones(len(i))], axis=0).astype(float)
    world = (np.asarray(ijkToWorld, dtype=float) @ ijk)[:3]
    return world @ weights / weights.sum()


# -- Body outline alignment ------------------------------------------------------------------------------------------

def outlineOffset(movingPoints, fixedPoints, minPoints=500):
    """Left-right and anterior-posterior offset (mm) that centres the moving body outline on the fixed one,
    using only the head-feet range both outlines cover. Returns (offset xyz with z = 0, overlap in mm)."""
    moving = np.asarray(movingPoints, dtype=float)
    fixed = np.asarray(fixedPoints, dtype=float)
    low = max(moving[:, 2].min(), fixed[:, 2].min())
    high = min(moving[:, 2].max(), fixed[:, 2].max())
    overlap = high - low
    if overlap > 20.0:
        movingIn = moving[(moving[:, 2] >= low) & (moving[:, 2] <= high)]
        fixedIn = fixed[(fixed[:, 2] >= low) & (fixed[:, 2] <= high)]
        if len(movingIn) >= minPoints and len(fixedIn) >= minPoints:
            moving, fixed = movingIn, fixedIn
    offset = fixed.mean(axis=0) - moving.mean(axis=0)
    offset[2] = 0.0
    return offset, max(overlap, 0.0)


OUTLINE_AREA_RATIO = 0.6          # a functional-image outline is accepted as a body outline above this ratio
FALLBACK_OUTLINE_PERCENTS = (1.5, 0.75, 0.4)


def sliceAreas(mask, ijkToWorld):
    """(world z of each axial slice centre, area of the mask in that slice in mm^2) for an axial image."""
    matrix = np.asarray(ijkToWorld, dtype=float)
    voxelArea = float(np.linalg.norm(np.cross(matrix[:3, 0], matrix[:3, 1])))
    ks = np.arange(mask.shape[0])
    ci, cj = (mask.shape[2] - 1) / 2.0, (mask.shape[1] - 1) / 2.0
    z = (matrix @ np.stack([np.full(len(ks), ci), np.full(len(ks), cj), ks, np.ones(len(ks))]))[2]
    return z, mask.reshape(mask.shape[0], -1).sum(axis=1) * voxelArea


def outlineAreaRatio(movingMask, movingIjkToWorld, fixedMask, fixedIjkToWorld):
    """Mean axial cross-section of the moving outline divided by that of the fixed outline, over the head-feet
    range both cover (over all slices if they do not overlap). A SPECT/PET that shows no body outline (only the
    uptake) gives a small ratio."""
    movingZ, movingArea = sliceAreas(movingMask, movingIjkToWorld)
    fixedZ, fixedArea = sliceAreas(fixedMask, fixedIjkToWorld)
    movingZ, movingArea = movingZ[movingArea > 0], movingArea[movingArea > 0]
    fixedZ, fixedArea = fixedZ[fixedArea > 0], fixedArea[fixedArea > 0]
    if len(movingArea) == 0 or len(fixedArea) == 0:
        return 0.0
    low, high = max(movingZ.min(), fixedZ.min()), min(movingZ.max(), fixedZ.max())
    if high - low > 20.0:
        inMoving = (movingZ >= low) & (movingZ <= high)
        inFixed = (fixedZ >= low) & (fixedZ <= high)
        if inMoving.any() and inFixed.any():
            movingArea, fixedArea = movingArea[inMoving], fixedArea[inFixed]
    return float(movingArea.mean() / fixedArea.mean())


def hotRegionMask(array, bodyMaskArray, percent=30.0):
    """Voxels above percent of the robust maximum inside the body (the liver / perfused territory on MAA or Y-90)."""
    values = np.asarray(array, dtype=float)
    inside = values[bodyMaskArray] if bodyMaskArray is not None else values.ravel()
    inside = inside[inside > 0]
    if inside.size == 0:
        raise ValueError("No uptake found.")
    threshold = float(np.percentile(inside, 99.9)) * percent / 100.0
    mask = values > threshold
    if bodyMaskArray is not None:
        mask &= bodyMaskArray
    return largestComponent(mask)
