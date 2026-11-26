"""Multi-resolution rigid / affine registration for EasyReg (numpy + scipy + SimpleITK; no Slicer scene).
The same engine as DICOMGatedDynamicPETLib/sitkreg.py of the PET4D extension (same author, MIT): keep the two copies in
step when the engine changes (synced 2026-11-27: bounded B-spline passes, mask-only folding check, optional
levels / grid scaling / percent sampling / per-iteration callback used by the PET4D Neuro normalization).

Pipeline (register()):
  1. Both images are brought onto an (almost) isotropic working grid with the final resolution chosen by the user
     (finalSpacingMm): block averaging first (anti-aliasing), then linear resampling along the image's own axes.
     CT values are clipped to -1000..2000 HU and other anatomical images to their 0.5-99.5 percentiles, so metal,
     dense contrast or CBCT streaks do not stretch the joint histogram. Functional images (SPECT/PET) lose their
     negative noise and are smoothed lightly.
  2. Optional starting position: image centres aligned, and optionally a coarse search of the head-feet position
     (then left-right / anterior-posterior) on 8 mm images, for scans of different lengths (e.g. abdominal SPECT/CT
     vs chest-abdomen-pelvis CT).
  3. Mattes mutual information, regular-step gradient descent, coarse to fine: levels of about 16, 8, 4 ... mm down
     to the final resolution. Rigid (Euler 3D), then optionally affine starting from the rigid result. A fixed-image
     mask (e.g. the dilated liver) restricts the comparison. The number of samples per level is fixed by the quality
     preset (not a percentage of the voxels), so the run time depends little on the final resolution.
  4. The result is kept only if it improves the metric over the starting position (same sample points: fixed seed).
  5. Optional iterative deformable stage (deformable=True), starting from the linear result: B-spline free-form
     deformations in passes, coarse to fine control-point grid (e.g. 200, 100, 50 mm). Each pass registers the
     remaining local differences with a new B-spline on top of the passes kept so far, and is kept only if it improves
     the metric (a pass that does not is dropped; the next, finer one still runs). Only the first pass starts on a
     coarse level. The control points cover the mask, or else the body (the air around it is left out), plus one
     control-point spacing on each side; beyond that the deformation fades to the linear result. The result is a
     displacement field (world RAS) on a coarse grid, with the minimum Jacobian determinant (folding check).

Conventions (as in functional.py): arrays are [k, j, i]; ijkToWorld maps (i, j, k, 1) to world RAS in mm. The
registration matrix M maps moving-image points (native RAS, no parent transform) to world RAS, i.e. it is the
matrix to parent of the Slicer transform the moving image goes under.

Methods: Mattes et al., IEEE TMI 2003 (mutual information); multi-resolution registration as in ITK (Yoo et al.,
Insight Toolkit) and SimpleITK (Lowekamp et al., Front Neuroinform 2013); B-spline free-form deformation (Rueckert et
al., IEEE TMI 1999) with coarse-to-fine control-point refinement.
"""

import dataclasses
import math
import time

import numpy as np

RIGID = "rigid"
AFFINE = "affine"

KIND_CT = "ct"                   # Hounsfield units (CT, CBCT reconstructed in HU)
KIND_ANATOMICAL = "anatomical"   # MRI, CBCT not in HU, anything else anatomical
KIND_FUNCTIONAL = "functional"   # SPECT / PET

INIT_CURRENT = "current"         # start from the current position (initial matrix)
INIT_CENTRES = "centres"         # centres of the (cropped) images on top of each other
INIT_SEARCH = "search"           # centres, then a coarse search of the head-feet (and R/A) position
INIT_SEARCH_CURRENT = "searchCurrent"   # the same search around the current position (no centring)

# Final resolution choices (label, mm). The last level of the pyramid runs at this spacing.
FINAL_SPACING_CHOICES = [
    ("4 mm (fastest; whole body, PET/CT)", 4.0),
    ("3 mm", 3.0),
    ("2 mm (recommended; SPECT/CT to CT/MRI)", 2.0),
    ("1.5 mm", 1.5),
    ("1 mm (fine; CBCT, small ROIs)", 1.0),
]
DEFAULT_FINAL_SPACING_MM = 2.0
MIN_FINAL_SPACING_MM = 0.5
COARSEST_SPACING_MM = 16.0        # first pyramid level (capture range of a few centimetres)
MIN_LEVEL_VOXELS = 8              # a level is dropped when the image would have fewer voxels along an axis
MAX_WORKING_VOXELS = 40_000_000   # per image; above this the final resolution is coarsened (memory)

# Quality: number of metric samples per level
QUALITY_SAMPLES = {"Fastest": 8_000, "Fast": 20_000, "Standard": 60_000, "Thorough": 200_000}
# Quality as a fraction of the voxels of each level (register(samplingMode=SAMPLING_PERCENT), e.g. the Neuro
# normalization): coarse levels get fewer samples than fine ones, at least MIN_LEVEL_SAMPLES (or all voxels)
SAMPLING_COUNT = "count"
SAMPLING_PERCENT = "percent"
QUALITY_PERCENT = {"Fastest": 0.003, "Fast": 0.01, "Standard": 0.03, "Thorough": 0.10}
MIN_LEVEL_SAMPLES = 5_000
DEFAULT_QUALITY = "Standard"
EVALUATION_SAMPLES = 200_000      # samples for the before / after comparison
HISTOGRAM_BINS = 32
SAMPLING_SEED = 20260930          # fixed seed: the same images and settings always give the same result
MAX_ITERATIONS = 200
RELAXATION_FACTOR = 0.5
MAX_STEP_MM = 8.0                 # first optimiser step (as in the Epona time-point registration)
MIN_STEP_MM = 2.0
AFFINE_START_SPACING_MM = 8.0     # the affine stage starts from the rigid result: no need for the coarsest level

CT_RANGE = (-1000.0, 2000.0)
ANATOMICAL_PERCENTILES = (0.5, 99.5)
FUNCTIONAL_UPPER_PERCENTILE = 99.9
FUNCTIONAL_SMOOTHING_VOXELS = 1.0
MAX_PERCENTILE_SAMPLES = 2_000_000

SEARCH_SPACING_MM = 8.0
DEFAULT_SEARCH_RANGE_MM = 200.0   # head-feet offsets tried: -range .. +range
SEARCH_STEP_MM = 10.0
SEARCH_LATERAL_OFFSETS_MM = (-30.0, -15.0, 0.0, 15.0, 30.0)
SEARCH_FINE_STEP_MM = 4.0
SEARCH_MIN_OVERLAP = 0.6          # head-feet overlap of the two fields of view, fraction of the shorter one
SEARCH_SAMPLES = 20_000

# Iterative deformable stage (B-spline free-form deformation)
# (label, final control-point spacing in mm). Coarser grids are stiffer.
GRID_SPACING_CHOICES = [
    ("80 mm (stiff; whole organs)", 80.0),
    ("50 mm (recommended; liver)", 50.0),
    ("30 mm (flexible; local differences)", 30.0),
    ("20 mm (very flexible; check for distortion)", 20.0),
    ("10 mm (most flexible; check for distortion)", 10.0),
]
DEFAULT_GRID_SPACING_MM = 50.0
DEFAULT_DEFORMABLE_PASSES = 3     # control-point spacing halves from pass to pass, ending at the chosen spacing
MAX_DEFORMABLE_PASSES = 4
DEFORMABLE_MIN_SPACING_MM = 2.0   # image resolution of a pass: grid spacing / 8, between these limits
DEFORMABLE_MAX_SPACING_MM = 8.0
DEFORMABLE_ITERATIONS = 25        # most L-BFGS-B iterations per pass and level (a pass usually converges earlier)
MIN_DEFORMABLE_ITERATIONS = 10
DEFORMABLE_CORRECTIONS = 5        # L-BFGS-B memory (corrections kept for the Hessian approximation)
MAX_DEFORMABLE_ITERATIONS = 200
FOREGROUND_FRACTION = 0.1         # body (no mask): voxels above 10 % of the reference's intensity range
# Each control point of a pass moves at most this fraction of the control-point spacing (L-BFGS-B bounds): below
# about 0.4 a cubic B-spline cannot fold (Choi & Lee, 2000; Rueckert et al., 2006), so a pass is no longer fitted
# to a folding deformation and then dropped as a whole (as happened without a bound)
MAX_CONTROL_POINT_SHIFT = 0.4
MIN_JACOBIAN = 0.1                # a pass that compresses any region below this volume ratio (or folds it) is dropped
# A pass that folds is not dropped at once: its control-point shifts are scaled down by these factors in turn and the
# first scaled pass that no longer folds and still improves the metric is kept (the bounded runs of a scaled grid can
# fold together although each alone cannot)
FOLD_BACKOFF_SCALES = (0.5, 0.25)
FIELD_SPACING_MM = 4.0            # displacement field grid (coarsened when larger than MAX_FIELD_VOXELS)
MAX_FIELD_VOXELS = 4_000_000

_LPS = np.diag([-1.0, -1.0, 1.0, 1.0])   # RAS <-> LPS (its own inverse)


class RegistrationCancelled(Exception):
    pass


@dataclasses.dataclass
class ImageInput:
    """An image for the engine: voxels [k, j, i], IJK-to-world RAS matrix (4x4) and kind (KIND_*)."""
    voxels: np.ndarray
    ijkToWorld: np.ndarray
    kind: str = KIND_ANATOMICAL


@dataclasses.dataclass
class Working:
    """An image prepared on its working grid."""
    voxels: np.ndarray
    ijkToWorld: np.ndarray

    @property
    def spacing(self):
        return np.linalg.norm(self.ijkToWorld[:3, :3], axis=0)

    @property
    def shape(self):
        return self.voxels.shape


# ---------------------------------------------------------------------------------------------------------------
# Pure helpers (numpy / scipy)
# ---------------------------------------------------------------------------------------------------------------

def imageKind(voxels, functional=False):
    """KIND_FUNCTIONAL when requested, KIND_CT for images with air at about -1000 HU, else KIND_ANATOMICAL."""
    if functional:
        return KIND_FUNCTIONAL
    array = np.asarray(voxels)
    if array.size == 0:
        return KIND_ANATOMICAL
    minimum = float(np.nanmin(array)) if array.dtype.kind == "f" else float(array.min())
    return KIND_CT if minimum <= -500.0 else KIND_ANATOMICAL


def _sample(values, maxCount=MAX_PERCENTILE_SAMPLES):
    flat = np.asarray(values).ravel()
    if flat.size > maxCount:
        flat = flat[::int(math.ceil(flat.size / maxCount))]
    return flat[np.isfinite(flat)] if flat.dtype.kind == "f" else flat


def _clipInPlace(array, low=None, high=None):
    """np.clip in place; for integer arrays the bounds are rounded outwards and kept within the type's range."""
    if array.dtype.kind in "iu":
        info = np.iinfo(array.dtype)
        low = None if low is None else array.dtype.type(min(max(math.floor(low), info.min), info.max))
        high = None if high is None else array.dtype.type(min(max(math.ceil(high), info.min), info.max))
    if low is None and high is None:
        return array
    return np.clip(array, low, high, out=array)


def normalizeIntensities(voxels, kind):
    """Copy with the clipping of the image kind (see the module doc). Integer images keep their type (a CT is not
    turned into a full-resolution float copy before it is downsampled); float images become float32."""
    source = np.asarray(voxels)
    if source.dtype.kind == "f":
        array = np.array(source, dtype=np.float32)
        array[~np.isfinite(array)] = 0.0
    elif source.dtype.kind in "iu":
        array = np.array(source)
    else:
        array = np.array(source, dtype=np.float32)
    if kind == KIND_CT:
        _clipInPlace(array, *CT_RANGE)
    elif kind == KIND_FUNCTIONAL:
        _clipInPlace(array, 0, None)
        sample = _sample(array)
        if sample.size:
            upper = float(np.percentile(sample, FUNCTIONAL_UPPER_PERCENTILE))
            if upper > 0:
                _clipInPlace(array, None, upper)
    else:
        sample = _sample(array)
        if sample.size:
            low, high = np.percentile(sample, ANATOMICAL_PERCENTILES)
            if high > low:
                _clipInPlace(array, float(low), float(high))
    return array


def blockMean(voxels, ijkToWorld, factors):
    """Average blocks of factors (i, j, k) voxels. Returns (voxels, IJK-to-world of the block centres). The last
    incomplete block along an axis is dropped."""
    fi, fj, fk = (max(1, int(f)) for f in factors)
    array = np.asarray(voxels)
    if (fi, fj, fk) == (1, 1, 1):
        return array.astype(np.float32, copy=False), np.asarray(ijkToWorld, dtype=float)
    nk, nj, ni = (array.shape[0] // fk), (array.shape[1] // fj), (array.shape[2] // fi)
    if min(nk, nj, ni) < 1:
        return array.astype(np.float32, copy=False), np.asarray(ijkToWorld, dtype=float)
    trimmed = array[:nk * fk, :nj * fj, :ni * fi]
    pooled = trimmed.reshape(nk, fk, nj, fj, ni, fi).mean(axis=(1, 3, 5), dtype=np.float32)
    scale = np.eye(4)
    scale[:3, :3] = np.diag([fi, fj, fk])
    scale[:3, 3] = [(fi - 1) / 2.0, (fj - 1) / 2.0, (fk - 1) / 2.0]
    return pooled, np.asarray(ijkToWorld, dtype=float) @ scale


def gridFor(shape, ijkToWorld, spacingMm):
    """(shape [k, j, i], IJK-to-world) of a grid with the axis directions of the image, voxel size spacingMm (or the
    original voxel size where that is coarser), covering the same voxel centres."""
    matrix = np.asarray(ijkToWorld, dtype=float)
    spacing = np.linalg.norm(matrix[:3, :3], axis=0)          # (i, j, k)
    counts = np.array(shape[::-1], dtype=int)                  # (i, j, k)
    newSpacing = np.maximum(spacing, float(spacingMm))
    lengths = (counts - 1) * spacing
    newCounts = np.maximum(1, np.floor(lengths / newSpacing + 1e-6).astype(int) + 1)
    grid = np.array(matrix)
    grid[:3, :3] = matrix[:3, :3] / spacing[None, :] * newSpacing[None, :]
    return tuple(int(v) for v in newCounts[::-1]), grid


def resampleOnto(voxels, ijkToWorld, shape, gridIjkToWorld, order=1, fill=0.0, mode="constant"):
    """Values of an image on another grid (both placed by their IJK-to-world matrices). mode "constant": points
    outside the image get fill; "nearest": the nearest edge value (for grids inside the image, where rounding would
    otherwise put the edge points just outside)."""
    from scipy import ndimage
    ijkMap = np.linalg.inv(np.asarray(ijkToWorld, dtype=float)) @ np.asarray(gridIjkToWorld, dtype=float)
    reverse = np.eye(3)[::-1]                                  # (i, j, k) <-> (k, j, i)
    matrix = reverse @ ijkMap[:3, :3] @ reverse
    offset = reverse @ ijkMap[:3, 3]
    source = np.asarray(voxels)
    if order > 0:
        source = source.astype(np.float32, copy=False)
    return ndimage.affine_transform(source, matrix, offset=offset, output_shape=tuple(shape), order=order,
                                    mode=mode, cval=fill, prefilter=False)


def prepareImage(image, spacingMm):
    """ImageInput -> Working image on a grid of about spacingMm (see the module doc)."""
    voxels = normalizeIntensities(image.voxels, image.kind)
    matrix = np.asarray(image.ijkToWorld, dtype=float)
    spacing = np.linalg.norm(matrix[:3, :3], axis=0)
    factors = [max(1, int(math.floor(spacingMm / s + 1e-6))) if s > 0 else 1 for s in spacing]
    pooled, pooledMatrix = blockMean(voxels, matrix, factors)
    shape, grid = gridFor(pooled.shape, pooledMatrix, spacingMm)
    fill = CT_RANGE[0] if image.kind == KIND_CT else float(np.min(pooled)) if pooled.size else 0.0
    if shape == pooled.shape and np.allclose(grid, pooledMatrix):
        result = np.array(pooled, dtype=np.float32)
    else:
        result = resampleOnto(pooled, pooledMatrix, shape, grid, order=1, fill=fill, mode="nearest")
    if image.kind == KIND_FUNCTIONAL and FUNCTIONAL_SMOOTHING_VOXELS > 0:
        from scipy import ndimage
        result = ndimage.gaussian_filter(result, FUNCTIONAL_SMOOTHING_VOXELS)
    return Working(np.ascontiguousarray(result, dtype=np.float32), grid)


def prepareMask(mask, maskIjkToWorld, working):
    """Mask (bool [k, j, i] on its own grid) on the grid of a working image (nearest neighbour)."""
    resampled = resampleOnto(np.asarray(mask, dtype=np.uint8), maskIjkToWorld, working.shape, working.ijkToWorld,
                             order=0, fill=0)
    return resampled > 0


def workingSpacing(images, spacingMm, maxVoxels=MAX_WORKING_VOXELS):
    """(spacing, note): spacingMm, coarsened in 0.5 mm steps until every image fits in maxVoxels."""
    spacing = max(MIN_FINAL_SPACING_MM, float(spacingMm))
    while True:
        largest = max(int(np.prod(gridFor(np.shape(image.voxels), image.ijkToWorld, spacing)[0]))
                      for image in images)
        if largest <= maxVoxels or spacing >= COARSEST_SPACING_MM:
            break
        spacing += 0.5
    note = ""
    if spacing > spacingMm + 1e-6:
        note = (f"Final resolution coarsened from {spacingMm:g} to {spacing:g} mm (images too large at "
                f"{spacingMm:g} mm; use ROIs for a finer resolution).")
    return spacing, note


def pyramid(spacingMm, shape, coarsestMm=COARSEST_SPACING_MM, minVoxels=MIN_LEVEL_VOXELS):
    """Shrink factors (powers of two, coarse to fine, ending with 1) for a working image of spacingMm: the first
    level is at most coarsestMm, and no level has fewer than minVoxels voxels along an axis."""
    factors = [1]
    while spacingMm * factors[-1] * 2 <= coarsestMm + 1e-6 and min(shape) / (factors[-1] * 2) >= minVoxels:
        factors.append(factors[-1] * 2)
    return factors[::-1]


def smoothingSigmas(factors, spacingMm):
    """Gaussian sigma (mm) for each level: half the level's voxel size, none at full resolution."""
    return [0.5 * spacingMm * f if f > 1 else 0.0 for f in factors]


class SampleFraction(float):
    """A quality given as a fraction of each level's voxels (SAMPLING_PERCENT) instead of a number of samples."""


def samplingPercentages(factors, voxelCount, samples):
    """Sampling fraction per level giving about `samples` metric samples (voxelCount: voxels of the working image,
    or of the mask, at full resolution). samples a SampleFraction: that fraction of each level's voxels, but at
    least MIN_LEVEL_SAMPLES samples."""
    result = []
    for f in factors:
        count = max(1.0, voxelCount / float(f ** 3))
        if isinstance(samples, SampleFraction):
            wanted = max(float(samples), MIN_LEVEL_SAMPLES / count)
        else:
            wanted = samples / count
        result.append(float(min(1.0, max(1e-4, wanted))))
    return result


def worldBounds(shape, ijkToWorld, matrix=None):
    """(min xyz, max xyz) of the voxel centres of an image, optionally moved by a 4x4 matrix."""
    k, j, i = (np.array([0, n - 1], dtype=float) for n in shape)
    corners = np.array([[a, b, c, 1.0] for a in i for b in j for c in k])
    world = (np.asarray(ijkToWorld, dtype=float) @ corners.T)
    if matrix is not None:
        world = np.asarray(matrix, dtype=float) @ world
    return world[:3].min(axis=1), world[:3].max(axis=1)


def centre(shape, ijkToWorld, matrix=None):
    low, high = worldBounds(shape, ijkToWorld, matrix)
    return (low + high) / 2.0


def translationMatrix(offset):
    matrix = np.eye(4)
    matrix[:3, 3] = offset
    return matrix


def centresMatrix(fixed, moving, initialMatrix):
    """Initial matrix with the moving image's centre (under initialMatrix) moved onto the fixed image's centre."""
    shift = centre(fixed.shape, fixed.ijkToWorld) - centre(moving.shape, moving.ijkToWorld, initialMatrix)
    return translationMatrix(shift) @ np.asarray(initialMatrix, dtype=float)


def headFeetOverlap(fixed, moving, matrix):
    """Head-feet overlap of the two fields of view, as a fraction of the shorter one."""
    fixedLow, fixedHigh = worldBounds(fixed.shape, fixed.ijkToWorld)
    movingLow, movingHigh = worldBounds(moving.shape, moving.ijkToWorld, matrix)
    overlap = min(fixedHigh[2], movingHigh[2]) - max(fixedLow[2], movingLow[2])
    shorter = min(fixedHigh[2] - fixedLow[2], movingHigh[2] - movingLow[2])
    return max(0.0, overlap) / shorter if shorter > 0 else 0.0


def nearestRotation(matrix3):
    """Closest rotation (polar decomposition) to a 3x3 matrix."""
    u, _, vt = np.linalg.svd(np.asarray(matrix3, dtype=float))
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation


def rigidPart(matrix):
    """The matrix with its 3x3 part replaced by the nearest rotation (translation of the centre kept)."""
    result = np.array(matrix, dtype=float)
    result[:3, :3] = nearestRotation(result[:3, :3])
    return result


def toSitkAffine(matrix):
    """Moving-to-world RAS matrix -> (A, o) of the SimpleITK mapping fixed LPS point x -> moving LPS point A x + o."""
    lps = _LPS @ np.linalg.inv(np.asarray(matrix, dtype=float)) @ _LPS
    return lps[:3, :3], lps[:3, 3]


def fromSitkAffine(a, o):
    """Inverse of toSitkAffine."""
    lps = np.eye(4)
    lps[:3, :3] = a
    lps[:3, 3] = o
    return np.linalg.inv(_LPS @ lps @ _LPS)


def centredParameters(a, o, centreLps):
    """Translation t of a centred transform y = A (x - c) + c + t equal to y = A x + o."""
    c = np.asarray(centreLps, dtype=float)
    return np.asarray(o, dtype=float) - c + np.asarray(a, dtype=float) @ c


def matrixChange(before, after, point):
    """(displacement of point in mm, rotation angle in degrees) between two matrices."""
    before, after = np.asarray(before, dtype=float), np.asarray(after, dtype=float)
    p = np.append(np.asarray(point, dtype=float), 1.0)
    source = np.linalg.solve(before, p)
    displacement = float(np.linalg.norm((after @ source)[:3] - p[:3]))
    relative = after @ np.linalg.inv(before)
    rotation = nearestRotation(relative[:3, :3])
    angle = float(np.degrees(np.arccos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))))
    return displacement, angle


# ---------------------------------------------------------------------------------------------------------------
# SimpleITK
# ---------------------------------------------------------------------------------------------------------------

def sitkAvailable():
    try:
        import SimpleITK  # noqa: F401
        return True
    except ImportError:
        return False


def sitkImage(working):
    """SimpleITK float32 image (LPS physical space) of a working image."""
    import SimpleITK as sitk
    image = sitk.GetImageFromArray(np.ascontiguousarray(working.voxels, dtype=np.float32))
    _place(image, working.ijkToWorld)
    return image


def sitkMask(mask, working):
    import SimpleITK as sitk
    image = sitk.GetImageFromArray(np.ascontiguousarray(mask, dtype=np.uint8))
    _place(image, working.ijkToWorld)
    return image


def _place(image, ijkToWorld):
    matrix = np.asarray(ijkToWorld, dtype=float)
    spacing = np.linalg.norm(matrix[:3, :3], axis=0)
    direction = _LPS[:3, :3] @ (matrix[:3, :3] / spacing[None, :])
    image.SetSpacing([float(v) for v in spacing])
    image.SetOrigin([float(v) for v in (_LPS[:3, :3] @ matrix[:3, 3])])
    image.SetDirection([float(v) for v in direction.flatten()])


def sitkTransform(matrix, kind, centreLps):
    """Euler3D (rigid) or affine SimpleITK transform equal to a moving-to-world RAS matrix, centred on centreLps."""
    import SimpleITK as sitk
    a, o = toSitkAffine(rigidPart(matrix) if kind == RIGID else matrix)
    c = [float(v) for v in centreLps]
    if kind == RIGID:
        transform = sitk.Euler3DTransform()
        transform.SetCenter(c)
        transform.SetMatrix([float(v) for v in a.flatten()], 1e-6)
    else:
        transform = sitk.AffineTransform(3)
        transform.SetCenter(c)
        transform.SetMatrix([float(v) for v in a.flatten()])
    transform.SetTranslation([float(v) for v in centredParameters(a, o, centreLps)])
    return transform


def matrixFromSitk(transform):
    """Moving-to-world RAS matrix of an Euler3D / affine SimpleITK transform."""
    a = np.array(transform.GetMatrix(), dtype=float).reshape(3, 3)
    c = np.array(transform.GetCenter(), dtype=float)
    t = np.array(transform.GetTranslation(), dtype=float)
    return fromSitkAffine(a, t + c - a @ c)


def _method(samples, voxelCount, factors=(1,), fixedMask=None):
    import SimpleITK as sitk
    method = sitk.ImageRegistrationMethod()
    method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=HISTOGRAM_BINS)
    method.SetMetricSamplingStrategy(method.RANDOM)
    percentages = samplingPercentages(factors, voxelCount, samples)
    if len(factors) > 1 and hasattr(method, "SetMetricSamplingPercentagePerLevel"):
        method.SetMetricSamplingPercentagePerLevel(percentages, SAMPLING_SEED)
    else:
        method.SetMetricSamplingPercentage(max(percentages) if len(factors) > 1 else percentages[0], SAMPLING_SEED)
    method.SetInterpolator(sitk.sitkLinear)
    if fixedMask is not None:
        method.SetMetricFixedMask(fixedMask)
    return method


def evaluateMetric(fixedImage, movingImage, matrix, kind, centreLps, voxelCount, fixedMask=None,
                   samples=EVALUATION_SAMPLES):
    """Mattes mutual information (lower is better) of the moving image placed by matrix; inf when it cannot be
    evaluated (e.g. the images do not overlap)."""
    method = _method(samples, voxelCount, fixedMask=fixedMask)
    method.SetInitialTransform(sitkTransform(matrix, kind, centreLps), False)
    try:
        value = float(method.MetricEvaluate(fixedImage, movingImage))
    except RuntimeError:
        return math.inf
    return value if math.isfinite(value) else math.inf


def _stop(method):
    for name in ("StopRegistration", "Abort"):
        if hasattr(method, name):
            try:
                getattr(method, name)()
                return
            except Exception:
                pass


def runStage(fixedImage, movingImage, matrix, kind, centreLps, spacingMm, factors, voxelCount, samples,
             fixedMask=None, progress=None, isCancelled=None, label="", step=None):
    """One optimisation stage (rigid or affine) over the pyramid levels. Returns (matrix, stop condition).
    step(level index, iteration, metric), when given, is called at each iteration instead of the progress text."""
    import SimpleITK as sitk
    method = _method(samples, voxelCount, factors, fixedMask)
    # initial step of each level (mm); halved whenever the direction changes, down to minStep
    learningRate = min(MAX_STEP_MM, max(MIN_STEP_MM, 4.0 * spacingMm))
    method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=float(learningRate), minStep=float(0.05 * spacingMm), numberOfIterations=MAX_ITERATIONS,
        relaxationFactor=RELAXATION_FACTOR, gradientMagnitudeTolerance=1e-8)
    method.SetOptimizerScalesFromPhysicalShift()
    method.SetShrinkFactorsPerLevel([int(f) for f in factors])
    method.SetSmoothingSigmasPerLevel(smoothingSigmas(factors, spacingMm))
    method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    transform = sitkTransform(matrix, kind, centreLps)
    method.SetInitialTransform(transform, True)
    state = {"cancelled": False}

    def onIteration():
        if isCancelled is not None and isCancelled():
            state["cancelled"] = True
            _stop(method)
            return
        if step is not None:
            step(method.GetCurrentLevel(), method.GetOptimizerIteration(), method.GetMetricValue())
        elif progress is not None:
            level = method.GetCurrentLevel()
            levelMm = spacingMm * factors[min(level, len(factors) - 1)]
            progress(f"{label}level {level + 1} of {len(factors)} ({levelMm:g} mm), iteration "
                     f"{method.GetOptimizerIteration()}, metric {method.GetMetricValue():.4f}")

    method.AddCommand(sitk.sitkIterationEvent, onIteration)
    try:
        method.Execute(fixedImage, movingImage)
    except RuntimeError:
        if state["cancelled"]:
            raise RegistrationCancelled()
        raise
    finally:
        # onIteration holds the method and the method holds onIteration through a C++ command, a cycle the
        # garbage collector cannot see: without this every run leaks the method (and its images and masks)
        method.RemoveAllCommands()
    if state["cancelled"]:
        raise RegistrationCancelled()
    return matrixFromSitk(transform), method.GetOptimizerStopConditionDescription()


def searchStart(fixedImage, movingImage, fixedCoarse, movingCoarse, matrix, centreLps, voxelCount,
                rangeMm=DEFAULT_SEARCH_RANGE_MM, fixedMask=None, isCancelled=None, progress=None):
    """Coarse search of the starting position on 8 mm images: head-feet offsets over +-rangeMm, then left-right /
    anterior-posterior offsets, then a finer head-feet step. Only offsets where the head-feet fields of view overlap
    by at least SEARCH_MIN_OVERLAP are tried. Returns (matrix, metric, number of positions tried)."""
    tried = [0]

    def score(offset):
        if isCancelled is not None and isCancelled():
            raise RegistrationCancelled()
        candidate = translationMatrix(offset) @ matrix
        if headFeetOverlap(fixedCoarse, movingCoarse, candidate) < SEARCH_MIN_OVERLAP:
            return math.inf
        tried[0] += 1
        return evaluateMetric(fixedImage, movingImage, candidate, RIGID, centreLps, voxelCount, fixedMask,
                              samples=SEARCH_SAMPLES)

    def best(offsets):
        values = [(score(o), tuple(o)) for o in offsets]
        return min(values, key=lambda v: v[0])

    steps = int(round(rangeMm / SEARCH_STEP_MM))
    if progress is not None:
        progress("Searching the head-feet position")
    value, offset = best([(0.0, 0.0, s * SEARCH_STEP_MM) for s in range(-steps, steps + 1)])
    if progress is not None:
        progress("Searching the left-right / anterior-posterior position")
    value, offset = min((value, offset), best([(r, a, offset[2]) for r in SEARCH_LATERAL_OFFSETS_MM
                                               for a in SEARCH_LATERAL_OFFSETS_MM]), key=lambda v: v[0])
    fine = [(offset[0], offset[1], offset[2] + d) for d in np.arange(-SEARCH_STEP_MM, SEARCH_STEP_MM + 1e-6,
                                                                     SEARCH_FINE_STEP_MM)]
    value, offset = min((value, offset), best(fine), key=lambda v: v[0])
    if not math.isfinite(value):
        return matrix, math.inf, tried[0]
    return translationMatrix(offset) @ matrix, value, tried[0]


def deformablePasses(gridSpacingMm, passes):
    """Control-point spacings (mm) of the passes, coarse to fine, ending with gridSpacingMm."""
    count = int(min(MAX_DEFORMABLE_PASSES, max(1, passes)))
    return [float(gridSpacingMm) * 2 ** (count - 1 - index) for index in range(count)]


def deformableShrink(gridSpacingMm, spacingMm, shape):
    """Shrink factor of the working images for a pass with this control-point spacing (images of about 1/8 of it,
    between DEFORMABLE_MIN_SPACING_MM and DEFORMABLE_MAX_SPACING_MM, at least MIN_LEVEL_VOXELS along each axis)."""
    target = min(DEFORMABLE_MAX_SPACING_MM, max(DEFORMABLE_MIN_SPACING_MM, gridSpacingMm / 8.0))
    factor = max(1, int(round(target / spacingMm)))
    while factor > 1 and min(shape) / factor < MIN_LEVEL_VOXELS:
        factor -= 1
    return factor


def deformableLevels(levelsMm, spacingMm, shape):
    """Shrink factors (coarse to fine) of the working images for the image sizes levelsMm (e.g. 8, 4, 2 mm): each
    rounded to a whole factor of spacingMm (at least 1), levels with fewer than MIN_LEVEL_VOXELS along an axis and
    repeats dropped. The last level may be coarser than the working images (e.g. 2 mm on 1 mm images)."""
    factors = []
    for mm in sorted((float(v) for v in levelsMm), reverse=True):
        factor = max(1, int(round(mm / spacingMm)))
        if min(shape) / factor < MIN_LEVEL_VOXELS or (factors and factor >= factors[-1]):
            continue
        factors.append(factor)
    return factors or [1]


def deformableFactors(gridSpacingMm, spacingMm, shape, coarse=True, levelsMm=None):
    """Shrink factors (coarse to fine) of the levels of a deformable pass: levelsMm (deformableLevels) when given,
    else a coarse level (deformableShrink) then the working images when coarse, else the working images only."""
    if levelsMm:
        return deformableLevels(levelsMm, spacingMm, shape)
    shrink = deformableShrink(gridSpacingMm, spacingMm, shape)
    return [shrink, 1] if shrink > 1 and coarse else [1]


def gridScales(factors):
    """With the control-point grid scaled together with the images (register(deformableGridScaling=True)): the
    control-point spacing of each level as a multiple of the last one (e.g. images 8, 4, 2 mm -> 4, 2, 1), and the
    mesh refinement of each level relative to the first (1, 2, 4). Whole numbers only (the factors halve)."""
    last = max(1, int(factors[-1]))
    ratios = [max(1, int(round(f / float(last)))) for f in factors]
    return ratios, [max(1, ratios[0] // r) for r in ratios]


def paddedDomain(working, padMm):
    """(size i j k, origin LPS, spacing, direction) of the working image's grid extended by padMm on each side: the
    domain of a B-spline, so that its control points reach beyond the compared region and the deformation fades to
    nothing outside it."""
    matrix = np.asarray(working.ijkToWorld, dtype=float)
    spacing = np.linalg.norm(matrix[:3, :3], axis=0)
    pad = np.ceil(padMm / spacing).astype(int)
    size = np.array(working.shape[::-1], dtype=int) + 2 * pad
    shifted = matrix @ np.append(-pad.astype(float), 1.0)
    direction = _LPS[:3, :3] @ (matrix[:3, :3] / spacing[None, :])
    return ([int(v) for v in size], [float(v) for v in _LPS[:3, :3] @ shifted[:3]], [float(v) for v in spacing],
            [float(v) for v in direction.flatten()])


def deformableRegion(working, mask=None, fraction=FOREGROUND_FRACTION):
    """Working view of the box the deformation is fitted in: the mask's bounding box, or without a mask the
    foreground's (the body: voxels above `fraction` of the image's 1-99 percentile intensity range, so the air
    around the patient gets no control points). Stray voxels are ignored (0.5-99.5 percentiles of the coordinates).
    The whole image when the box would be empty."""
    if mask is not None:
        inside = np.asarray(mask, dtype=bool)
    else:
        sample = _sample(working.voxels)
        if sample.size == 0:
            return working
        low, high = (float(v) for v in np.percentile(sample, (1.0, 99.0)))
        if not high > low:
            return working
        inside = working.voxels > low + fraction * (high - low)
    indices = np.nonzero(inside)
    if indices[0].size == 0:
        return working
    start = [int(np.floor(np.percentile(axis, 0.5))) for axis in indices]
    stop = [int(np.ceil(np.percentile(axis, 99.5))) + 1 for axis in indices]
    box = tuple(slice(a, b) for a, b in zip(start, stop))
    matrix = np.asarray(working.ijkToWorld, dtype=float) @ translationMatrix(start[::-1])
    return Working(working.voxels[box], matrix)


def fieldGrid(shape, ijkToWorld, spacingMm=FIELD_SPACING_MM, maxVoxels=MAX_FIELD_VOXELS):
    """(shape [k, j, i], IJK-to-world) of the displacement field grid over an image: spacingMm, coarsened until it has
    at most maxVoxels voxels."""
    spacing = float(spacingMm)
    while True:
        fieldShape, matrix = gridFor(shape, ijkToWorld, spacing)
        if int(np.prod(fieldShape)) <= maxVoxels:
            return fieldShape, matrix
        spacing *= 1.25


def _evaluateTransform(fixedImage, movingImage, transform, voxelCount, fixedMask=None, samples=EVALUATION_SAMPLES):
    """Mattes mutual information (lower is better) of the moving image mapped by a SimpleITK transform (fixed to
    moving); inf when it cannot be evaluated."""
    method = _method(samples, voxelCount, fixedMask=fixedMask)
    method.SetInitialTransform(transform, False)
    try:
        value = float(method.MetricEvaluate(fixedImage, movingImage))
    except RuntimeError:
        return math.inf
    return value if math.isfinite(value) else math.inf


def _composite(transforms):
    """SimpleITK composite transform: transforms[0](transforms[1](... x)) (the last one is applied first)."""
    import SimpleITK as sitk
    composite = sitk.CompositeTransform(3)
    for transform in transforms:
        composite.AddTransform(transform)
    return composite


def runDeformablePass(fixedImage, movingImage, fixedWorking, initialTransform, gridSpacingMm, spacingMm, voxelCount,
                      samples, fixedMask=None, progress=None, isCancelled=None, label="", coarse=True, domain=None,
                      iterations=DEFORMABLE_ITERATIONS, corrections=DEFORMABLE_CORRECTIONS, levelsMm=None,
                      step=None, bounded=True, maxShift=MAX_CONTROL_POINT_SHIFT):
    """One B-spline pass: a new free-form deformation (control points gridSpacingMm apart) registered on top of
    initialTransform (fixed to moving; linear, or linear and the displacement field of the previous passes). The
    B-spline domain is domain (a Working box inside the fixed image, see deformableRegion; default the whole working
    fixed image) extended by one control-point spacing, so the deformation fades out beyond the compared region.
    With coarse, a coarse level first gives the capture range (needed in the first pass only: later passes start
    close); the pass ends on the working images, the images the result is judged on. The run time grows with the
    number of control points (SimpleITK's metrics handle the B-spline derivative as a dense vector), hence the
    region. iterations: most optimiser iterations per level. Returns (B-spline transform, stop condition)."""
    import SimpleITK as sitk
    size, origin, spacing, direction = paddedDomain(domain if domain is not None else fixedWorking, gridSpacingMm)
    domain = sitk.Image(size, sitk.sitkUInt8)
    domain.SetOrigin(origin)
    domain.SetSpacing(spacing)
    domain.SetDirection(direction)
    extent = [n * s for n, s in zip(size, spacing)]
    factors = deformableFactors(gridSpacingMm, spacingMm, fixedWorking.shape, coarse, levelsMm)
    mesh = [max(1, int(round(length / gridSpacingMm))) for length in extent]
    bspline = sitk.BSplineTransformInitializer(domain, mesh, 3)
    method = _method(samples, voxelCount, factors, fixedMask)
    iterations = int(min(MAX_DEFORMABLE_ITERATIONS, max(MIN_DEFORMABLE_ITERATIONS, iterations)))
    # the coefficients are the control points' shifts (mm): bounded so that the pass cannot fold
    bound = float(maxShift) * float(gridSpacingMm)
    method.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=iterations,
                                maximumNumberOfCorrections=int(max(1, corrections)),
                                maximumNumberOfFunctionEvaluations=4 * iterations,
                                costFunctionConvergenceFactor=1e7,
                                **({"lowerBound": -bound, "upperBound": bound} if bounded else {}))
    method.SetShrinkFactorsPerLevel([int(f) for f in factors])
    method.SetSmoothingSigmasPerLevel(smoothingSigmas(factors, spacingMm))
    method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    # (ITK maps image gradients through the moving initial transform: it must not contain a B-spline)
    method.SetMovingInitialTransform(initialTransform)
    method.SetInitialTransform(bspline, True)
    state = {"cancelled": False}

    def onIteration():
        if isCancelled is not None and isCancelled():
            state["cancelled"] = True
            _stop(method)
            return
        if step is not None:
            step(method.GetCurrentLevel(), method.GetOptimizerIteration(), method.GetMetricValue())
        elif progress is not None:
            level = factors[min(method.GetCurrentLevel(), len(factors) - 1)]
            progress(f"{label}control points {gridSpacingMm:g} mm apart, images {spacingMm * level:g} mm, "
                     f"iteration {method.GetOptimizerIteration()}, metric {method.GetMetricValue():.4f}")

    method.AddCommand(sitk.sitkIterationEvent, onIteration)
    try:
        method.Execute(fixedImage, movingImage)
    except RuntimeError:
        if state["cancelled"]:
            raise RegistrationCancelled()
        raise
    finally:
        # onIteration holds the method and the method holds onIteration through a C++ command, a cycle the
        # garbage collector cannot see: without this every run leaks the method (and its images and masks)
        method.RemoveAllCommands()
    if state["cancelled"]:
        raise RegistrationCancelled()
    return bspline, method.GetOptimizerStopConditionDescription()


def maskOnGrid(maskWorking, working, shape, ijkToWorld, dilateVoxels=1):
    """The registration mask (bool, on the working image's grid) on another grid (e.g. the displacement field's),
    dilated by dilateVoxels; None without a mask."""
    if maskWorking is None:
        return None
    region = resampleOnto(np.asarray(maskWorking, dtype=np.uint8), working.ijkToWorld, shape, ijkToWorld, order=0,
                          fill=0) > 0
    if dilateVoxels > 0 and region.any():
        from scipy import ndimage
        region = ndimage.binary_dilation(region, iterations=int(dilateVoxels))
    return region


def _field(transform, shape, ijkToWorld, region=None):
    """(SimpleITK displacement field image (LPS) of a transform on a grid, minimum Jacobian determinant; within
    region (bool [k, j, i] on that grid) when given: outside the mask the control points have no image to fit and
    may fold where it does not matter)."""
    import SimpleITK as sitk
    reference = sitk.Image([int(v) for v in shape[::-1]], sitk.sitkUInt8)
    _place(reference, ijkToWorld)
    converter = sitk.TransformToDisplacementFieldFilter()
    converter.SetReferenceImage(reference)
    converter.SetOutputPixelType(sitk.sitkVectorFloat64)
    field = converter.Execute(transform)
    minimumJacobian = math.nan
    if min(shape) >= 2:
        try:
            jacobian = sitk.DisplacementFieldJacobianDeterminant(field)   # kept alive while its array is read
            values = sitk.GetArrayViewFromImage(jacobian)
            if region is not None and np.shape(region) == values.shape and np.any(region):
                values = values[np.asarray(region, dtype=bool)]
            minimumJacobian = float(np.min(values))
        except RuntimeError:
            pass
    return field, minimumJacobian


def fieldTransform(transform, shape, ijkToWorld, region=None):
    """(SimpleITK displacement field transform equal to a transform (fixed to moving) on a grid, minimum Jacobian
    determinant (within region when given))."""
    import SimpleITK as sitk
    field, minimumJacobian = _field(transform, shape, ijkToWorld, region)
    return sitk.DisplacementFieldTransform(field), minimumJacobian


def displacementField(transform, shape, ijkToWorld, region=None):
    """(displacement [k, j, i, 3] in world RAS mm, minimum Jacobian determinant) of a SimpleITK transform (fixed to
    moving) on a grid: the moving-image point of world point x is x + displacement(x). This is the transform from
    parent of a Slicer grid transform."""
    import SimpleITK as sitk
    field, minimumJacobian = _field(transform, shape, ijkToWorld, region)
    displacement = np.array(sitk.GetArrayFromImage(field), dtype=np.float64)
    displacement[..., :2] *= -1.0   # LPS -> RAS
    return displacement, minimumJacobian


def _deformableStage(fixedImage, movingImage, fixedWorking, matrix, kind, centreLps, spacing, voxelCount, samples,
                     maskImage, maskWorking, gridSpacingMm, passes, fieldGeometry, progress, isCancelled, notes,
                     iterations=DEFORMABLE_ITERATIONS, coarse=True, corrections=DEFORMABLE_CORRECTIONS,
                     levelsMm=None, step=None, gridScaling=False, foldGuard=True,
                     maxShift=MAX_CONTROL_POINT_SHIFT):
    """Iterative B-spline passes on top of the linear matrix (see the module doc). A pass that does not improve the
    metric is dropped and the next, finer pass starts from the previous result. The deformation of the kept passes
    is carried between passes as a displacement field on the working grid. Returns a dict: displacement (None when
    no pass was kept), fieldIjkToWorld, minimumJacobian, passes [(grid mm, metric before, after, kept)], metric."""
    import SimpleITK as sitk
    linear = sitkTransform(matrix, kind, centreLps)
    metric = _evaluateTransform(fixedImage, movingImage, linear, voxelCount, maskImage)
    carried = None          # displacement field transform of the kept passes (fixed to fixed), or None
    # (the runs are listed first: the carried field must be fine enough for the finest grid, or its interpolation
    # shows folds that the B-splines do not have)
    runs = []
    for index, gridMm in enumerate(deformablePasses(gridSpacingMm, passes)):
        passFactors = deformableFactors(gridMm, spacing, fixedWorking.shape, coarse and index == 0, levelsMm)
        if gridScaling and len(passFactors) > 1:
            ratios = gridScales(passFactors)[0]
            runs += [(index, j, gridMm * r, [spacing * f], False) for j, (f, r) in enumerate(zip(passFactors, ratios))]
        else:
            runs.append((index, None, gridMm, levelsMm, coarse and index == 0))
    fieldSpacing = max(float(spacing), min(FIELD_SPACING_MM, min(run[2] for run in runs) / 4.0))
    workShape, workMatrix = fieldGrid(fixedWorking.shape, fixedWorking.ijkToWorld, fieldSpacing)
    # the folding check of a pass only inside the (dilated) mask, when there is one
    workRegion = maskOnGrid(maskWorking, fixedWorking, workShape, workMatrix)
    region = deformableRegion(fixedWorking, maskWorking)
    history = []
    # runs: (pass index, level index within the pass, grid mm, image levels mm, coarse). With gridScaling each image
    # level of a pass is its own run with the grid scaled as the images (e.g. 40, 20, 10 mm on 8, 4, 2 mm), each on
    # top of the previous one (a bounded L-BFGS-B cannot refine its mesh between levels)
    for number, (index, sub, gridMm, runLevels, runCoarse) in enumerate(runs):
        start = _composite([linear, carried]) if carried is not None else linear
        if step is None:
            runStep = None
        elif sub is None:
            runStep = (lambda *a, i=index: step(i, *a))
        else:
            runStep = (lambda level, *a, i=index, j=sub: step(i, j, *a))
        bspline, _ = runDeformablePass(fixedImage, movingImage, fixedWorking, start, gridMm, spacing, voxelCount,
                                       samples, maskImage, progress, isCancelled,
                                       f"Deformable pass {number + 1} of {len(runs)}: ",
                                       coarse=runCoarse, domain=region, iterations=iterations,
                                       corrections=corrections, levelsMm=runLevels, step=runStep,
                                       bounded=foldGuard, maxShift=maxShift)
        full = np.asarray(bspline.GetParameters(), dtype=np.float64)
        kept, value, candidate = False, math.nan, None
        for scale in (1.0,) + (tuple(FOLD_BACKOFF_SCALES) if foldGuard else ()):
            if scale != 1.0:
                bspline = sitk.BSplineTransform(bspline)        # a copy: the composites above keep the old one
                bspline.SetParameters(tuple(full * scale))
            local = _composite([carried, bspline]) if carried is not None else bspline
            value = _evaluateTransform(fixedImage, movingImage, _composite([linear, local]), voxelCount, maskImage)
            if not (math.isfinite(value) and value < metric):
                break                                           # a smaller step only gives back less of the gain
            candidate, jacobian = fieldTransform(local, workShape, workMatrix, workRegion)
            if foldGuard and math.isfinite(jacobian) and jacobian <= MIN_JACOBIAN:
                notes.append(f"Deformable pass {number + 1} ({gridMm:g} mm grid) folds the image at "
                             f"{scale:g} of its shifts (minimum Jacobian determinant {jacobian:.2f}).")
                continue
            kept = True
            if scale != 1.0:
                notes.append(f"Deformable pass {number + 1} ({gridMm:g} mm grid) kept at {scale:g} of its shifts.")
            break
        if not kept and foldGuard:
            notes.append(f"Deformable pass {number + 1} ({gridMm:g} mm grid) was dropped.")
        history.append((gridMm, metric, value, kept))
        if kept:
            metric = value
            carried = candidate
    result = {"displacement": None, "fieldIjkToWorld": None, "minimumJacobian": math.nan, "passes": history,
              "metric": metric}
    keptGrids = [f"{grid:g}" for grid, _, _, ok in history if ok]
    if carried is None:
        notes.append("The deformable passes (control points " + ", ".join(f"{run[2]:g}" for run in runs)
                     + " mm apart) did not improve the match: the linear result was kept.")
        return result
    if progress is not None:
        progress("Computing the displacement field")
    if fieldGeometry is None:
        fieldGeometry = (fixedWorking.shape, fixedWorking.ijkToWorld)
    fieldShape, fieldMatrix = fieldGrid(*fieldGeometry, spacingMm=fieldSpacing)
    displacement, minimumJacobian = displacementField(_composite([linear, carried]), fieldShape, fieldMatrix,
                                                      maskOnGrid(maskWorking, fixedWorking, fieldShape, fieldMatrix))
    result.update(displacement=displacement, fieldIjkToWorld=fieldMatrix, minimumJacobian=minimumJacobian)
    notes.append(f"Deformable: {len(keptGrids)} of {len(runs)} passes kept (control points "
                 f"{', '.join(keptGrids)} mm apart); mutual information {history[0][1]:.4f} \u2192 {metric:.4f}.")
    if math.isfinite(minimumJacobian) and minimumJacobian <= 0:
        notes.append(f"Warning: the deformation folds (minimum Jacobian determinant {minimumJacobian:.2f}). Use a "
                     "coarser grid or fewer passes.")
    return result


def register(fixed, moving, initialMatrix=None, transformType=RIGID, finalSpacingMm=DEFAULT_FINAL_SPACING_MM,
             quality=DEFAULT_QUALITY, fixedMask=None, initialization=INIT_CURRENT,
             searchRangeMm=DEFAULT_SEARCH_RANGE_MM, progress=None, isCancelled=None, deformable=False,
             gridSpacingMm=DEFAULT_GRID_SPACING_MM, passes=DEFAULT_DEFORMABLE_PASSES, fieldGeometry=None,
             deformableIterations=DEFORMABLE_ITERATIONS, coarsestMm=COARSEST_SPACING_MM, deformableCoarse=True,
             deformableCorrections=DEFORMABLE_CORRECTIONS, deformableLevelsMm=None, samplingMode=SAMPLING_COUNT,
             onIteration=None, deformableGridScaling=False, deformableFoldGuard=True,
             deformableMaxShift=MAX_CONTROL_POINT_SHIFT):
    """
    Register moving (ImageInput, native coordinates) to fixed (ImageInput). fixedMask: (bool [k, j, i], ijkToWorld)
    restricting the comparison, or None. initialMatrix: current moving-to-world matrix (identity if None).
    progress(text) is called during the run; isCancelled() -> True stops it (RegistrationCancelled).
    deformable: after the linear stage (transformType), iterative B-spline passes (control points gridSpacingMm apart
    in the last pass, at most deformableIterations optimiser iterations per pass and level); fieldGeometry: (shape
    [k, j, i], ijkToWorld) of the image the displacement field must cover (default: the fixed image given).
    deformableCoarse: False runs the first deformable pass on the working images only (no coarser level first; the
    linear stage already brought the images close, e.g. a brain to a template after an affine).
    deformableCorrections: L-BFGS-B corrections (memory) of the deformable passes.
    deformableLevelsMm: image sizes (mm) of the levels of every deformable pass, e.g. (8, 4, 2); None: the default
    (a coarse level in the first pass, see deformableCoarse, then the working images).
    deformableFoldGuard: False lets the B-spline passes fold (no bound on the control points, no pass dropped for
    folding; a warning note is still written), as BRAINSFit does. With it a folding pass is scaled down
    (FOLD_BACKOFF_SCALES) before it is dropped.
    deformableMaxShift: the bound of each control point's shift as a fraction of its grid spacing (with the fold
    guard; 0.4 at most for a single cubic B-spline not to fold, smaller values give a smoother, gentler deformation).
    deformableGridScaling: the control-point spacing scaled with the images of the deformable levels (e.g. grid 10
    mm on 8, 4, 2 mm images: 40, 20, 10 mm), the mesh refined between the levels.
    samplingMode: SAMPLING_COUNT (quality = number of samples per level, QUALITY_SAMPLES) or SAMPLING_PERCENT
    (quality = fraction of each level's voxels, QUALITY_PERCENT).
    onIteration(info), when given, is called at every optimiser iteration (the progress text then only tells the
    phases): info is a dict with plan (the levels of the whole run: [{"stage", "levelMm", "maxIterations"}]), index
    (current entry of plan), stage, level / levels (within the stage), levelMm, iteration, maxIterations, metric
    (Mattes mutual information, lower is better), seconds.
    coarsestMm: voxel size of the first pyramid level (smaller for images that are already close, e.g. the frames of
    one dynamic acquisition in Motion Correction).

    Returns a dict: matrix (moving-to-world RAS, to set on the transform), initialMatrix (after the starting
    position), refined (False: the result did not improve the match and the starting position is returned),
    initialMetric, finalMetric, spacing (final resolution used), levels (mm), notes [text], stop, seconds.
    With deformable, also: displacement ([k, j, i, 3] world RAS, the transform from parent of the moving image; None
    when no deformable pass improved the match), fieldIjkToWorld, minimumJacobian, deformablePasses
    [(grid mm, metric before, after, kept)]; finalMetric is then the metric with the deformation.
    """
    start = time.time()
    notes = []
    report = progress or (lambda text: None)
    matrix0 = np.eye(4) if initialMatrix is None else np.array(initialMatrix, dtype=float)
    if transformType not in (RIGID, AFFINE):
        raise ValueError(f"Unknown transform type '{transformType}'.")
    if samplingMode == SAMPLING_PERCENT:
        samples = SampleFraction(QUALITY_PERCENT.get(quality, QUALITY_PERCENT[DEFAULT_QUALITY]))
    else:
        samples = QUALITY_SAMPLES.get(quality, QUALITY_SAMPLES[DEFAULT_QUALITY])

    report("Preparing the images")
    spacing, note = workingSpacing([fixed, moving], finalSpacingMm)
    if note:
        notes.append(note)
    fixedWorking = prepareImage(fixed, spacing)
    movingWorking = prepareImage(moving, spacing)
    maskWorking = None
    if fixedMask is not None:
        maskWorking = prepareMask(fixedMask[0], fixedMask[1], fixedWorking)
        if not maskWorking.any():
            raise ValueError("The registration mask does not overlap the reference image.")
    voxelCount = int(np.count_nonzero(maskWorking)) if maskWorking is not None else int(fixedWorking.voxels.size)

    fixedImage, movingImage = sitkImage(fixedWorking), sitkImage(movingWorking)
    maskImage = sitkMask(maskWorking, fixedWorking) if maskWorking is not None else None
    if maskWorking is not None:
        k, j, i = (np.mean(axis) for axis in np.nonzero(maskWorking))
        centreRas = (fixedWorking.ijkToWorld @ np.array([i, j, k, 1.0]))[:3]
    else:
        centreRas = centre(fixedWorking.shape, fixedWorking.ijkToWorld)
    centreLps = _LPS[:3, :3] @ centreRas

    if initialization in (INIT_CENTRES, INIT_SEARCH):
        matrix0 = centresMatrix(fixedWorking, movingWorking, matrix0)
    if initialization in (INIT_SEARCH, INIT_SEARCH_CURRENT):
        coarseSpacing = max(SEARCH_SPACING_MM, spacing)
        fixedCoarse = prepareImage(fixed, coarseSpacing)
        movingCoarse = prepareImage(moving, coarseSpacing)
        maskCoarse = prepareMask(fixedMask[0], fixedMask[1], fixedCoarse) if fixedMask is not None else None
        coarseCount = int(np.count_nonzero(maskCoarse)) if maskCoarse is not None else int(fixedCoarse.voxels.size)
        found, value, tried = searchStart(
            sitkImage(fixedCoarse), sitkImage(movingCoarse), fixedCoarse, movingCoarse, matrix0, centreLps,
            coarseCount, searchRangeMm, sitkMask(maskCoarse, fixedCoarse) if maskCoarse is not None else None,
            isCancelled, report)
        if math.isfinite(value):
            shift = found[:3, 3] - matrix0[:3, 3]
            origin = "the image centres" if initialization == INIT_SEARCH else "the current position"
            notes.append(f"Start position found by search ({tried} positions): shifted R {shift[0]:+.0f}, "
                         f"A {shift[1]:+.0f}, S {shift[2]:+.0f} mm from {origin}.")
            matrix0 = found
        else:
            notes.append("The start position search found no position where the images overlap enough; the "
                         "search was ignored.")

    initialMetric = evaluateMetric(fixedImage, movingImage, matrix0, transformType, centreLps, voxelCount, maskImage)

    factors = pyramid(spacing, fixedWorking.shape, coarsestMm)
    levels = [spacing * f for f in factors]
    affineFactors = [f for f in factors if spacing * f <= AFFINE_START_SPACING_MM + 1e-6] or [1]
    # the levels of the whole run, for onIteration
    plan = [{"stage": "Rigid", "levelMm": spacing * f, "maxIterations": MAX_ITERATIONS} for f in factors]
    offsets = {"Rigid": 0}
    if transformType == AFFINE:
        offsets["Affine"] = len(plan)
        plan += [{"stage": "Affine", "levelMm": spacing * f, "maxIterations": MAX_ITERATIONS} for f in affineFactors]
    if deformable:
        bsplineIterations = int(min(MAX_DEFORMABLE_ITERATIONS, max(MIN_DEFORMABLE_ITERATIONS, deformableIterations)))
        gridList = deformablePasses(gridSpacingMm, passes)
        for passIndex, gridMm in enumerate(gridList):
            offsets[f"B-spline {passIndex}"] = len(plan)
            name = f"B-spline, grid {gridMm:g} mm" + (f" (pass {passIndex + 1} of {len(gridList)})"
                                                     if len(gridList) > 1 else "")
            passFactors = deformableFactors(gridMm, spacing, fixedWorking.shape,
                                            deformableCoarse and passIndex == 0, deformableLevelsMm)
            ratios = gridScales(passFactors)[0] if deformableGridScaling and len(passFactors) > 1 \
                else [1] * len(passFactors)
            plan += [{"stage": name, "levelMm": spacing * f, "maxIterations": bsplineIterations,
                      "gridMm": gridMm * r} for f, r in zip(passFactors, ratios)]

    def stepper(key):
        """step callback of a stage: maps (level, iteration, metric) to onIteration(info)."""
        if onIteration is None:
            return None
        offset = offsets[key]
        count = sum(1 for entry in plan[offset:] if entry["stage"] == plan[offset]["stage"])

        def step(level, iteration, metric):
            index = min(offset + int(level), len(plan) - 1)
            entry = plan[index]
            onIteration({"plan": plan, "index": index, "stage": entry["stage"], "level": int(level),
                         "gridMm": entry.get("gridMm"),
                         "levels": count, "levelMm": entry["levelMm"], "iteration": int(iteration),
                         "maxIterations": entry["maxIterations"], "metric": float(metric),
                         "seconds": time.time() - start})
        return step

    matrix, stop = runStage(fixedImage, movingImage, matrix0, RIGID, centreLps, spacing, factors, voxelCount,
                            samples, maskImage, report, isCancelled, "Rigid: ", step=stepper("Rigid"))
    if transformType == AFFINE:
        matrix, stop = runStage(fixedImage, movingImage, matrix, AFFINE, centreLps, spacing, affineFactors,
                                voxelCount, samples, maskImage, report, isCancelled, "Affine: ",
                                step=stepper("Affine"))

    finalMetric = evaluateMetric(fixedImage, movingImage, matrix, transformType, centreLps, voxelCount, maskImage)
    refined = math.isfinite(finalMetric) and finalMetric < initialMetric
    if not refined:
        notes.append("The registration did not improve the match (mutual information "
                     f"{initialMetric:.4f} -> {finalMetric:.4f}): the starting position was kept.")
        matrix = matrix0
    result = {"matrix": matrix, "initialMatrix": matrix0, "refined": bool(refined),
              "initialMetric": float(initialMetric), "finalMetric": float(finalMetric), "spacing": float(spacing),
              "levels": levels, "notes": notes, "stop": stop, "seconds": time.time() - start,
              "centreRas": centreRas}
    if deformable:
        stage = _deformableStage(fixedImage, movingImage, fixedWorking, matrix, transformType, centreLps, spacing,
                                 voxelCount, samples, maskImage, maskWorking, gridSpacingMm, passes, fieldGeometry,
                                 report, isCancelled, notes, deformableIterations, deformableCoarse,
                                 deformableCorrections, deformableLevelsMm,
                                 step=(lambda passIndex, *a: stepper(f"B-spline {passIndex}")(*a))
                                 if onIteration is not None else None, gridScaling=deformableGridScaling,
                                 foldGuard=deformableFoldGuard, maxShift=deformableMaxShift)
        result.update(displacement=stage["displacement"], fieldIjkToWorld=stage["fieldIjkToWorld"],
                      minimumJacobian=stage["minimumJacobian"], deformablePasses=stage["passes"])
        if stage["displacement"] is not None:
            result["finalMetric"] = float(stage["metric"])
            result["refined"] = True
        result["seconds"] = time.time() - start
    return result
