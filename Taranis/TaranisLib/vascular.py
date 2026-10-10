"""CBCT territory planning: arterial tree, planned catheter tip, predicted downstream territory.

Pure numpy / scipy (no Slicer imports), so every step can be unit tested on synthetic vessel trees. Arrays are
indexed (k, j, i) like slicer.util.arrayFromVolume; spacing is given in the same order, in mm. The Slicer side
(cbct.py) resamples the CBCT and the case segments onto an axis-aligned working grid and calls these functions.

Methods (each from the published literature; the combination and the Taranis integration are ours):
  - Vesselness (optional shape filter): multiscale Hessian filter of Frangi AF, Niessen WJ, Vincken KL, Viergever MA. Multiscale vessel
    enhancement filtering. MICCAI 1998, LNCS 1496:130-137.
  - Tree segment: hysteresis thresholding and the connected component containing the injection point (standard
    region growing) inside a search mask (liver, a ball around the injection point, no air / lung / metal), on the local contrast
    against the surrounding liver (local mean and SD by normalised convolution over the liver voxels only, Knutsson H,
    Westin CF. Normalized and differential convolution. CVPR 1993:515-523), which removes the slow shading of CBCT;
    vessel wall at half maximum.
  - Centerlines: 3D medial-axis thinning of Lee TC, Kashyap RL, Chu CN. Building skeleton models via 3-D medial
    surface/axis thinning algorithms. CVGIP: Graphical Models and Image Processing 1994;56(6):462-478
    (scikit-image skeletonize, method "lee").
  - Tree orientation: shortest-path tree (Dijkstra 1959) from the centerline point nearest the injection point.
  - Territories: nearest-branch assignment of each liver voxel, as for the portal tree in Selle D, Preim B,
    Schenk A, Peitgen HO. Analysis of vasculature for liver surgical planning. IEEE Trans Med Imaging
    2002;21(11):1344-1357; the geodesic option uses minimum-cost (shortest) paths inside the liver as in the
    minimum-cost-path perfusion territories of the Molloi group (e.g. Malkasian S et al., 2022).
  - Feeder finder: tumour-feeding vessels as the tree branches reaching the (dilated) tumour, cf. Deschamps F
    et al. Computed analysis of three-dimensional cone-beam computed tomography angiography for determination
    of tumor-feeding vessels during chemoembolization of liver tumor: a pilot study. Cardiovasc Intervent Radiol
    2010;33(6):1235-1242.
  - Enhancement territory: perfused volume from contrast enhancement on selective / parenchymal CBCT, the CBCT
    analogue of the perfused volume from MAA uptake (e.g. van den Hoven AF et al., 2016; Pellerin O et al., 2013).
"""

import dataclasses

import numpy as np
from scipy import ndimage
from scipy import sparse
from scipy.sparse import csgraph
from scipy.spatial import cKDTree

# -- Defaults (mm unless stated otherwise) ------------------------------------------------------------------------

VESSEL_SCALES_MM = (0.5, 1.0, 2.0, 3.0)  # scales of the explicit vesselness calls (tests); see vesselScales
MAX_VESSEL_RADIUS_MM = 5.2            # tube check: largest Hessian scale radius (sigma 3 mm); thicker lumens below
SPUR_RADIUS_FACTOR = 1.5              # a side branch shorter than 1.5 x the radius at its junction is a surface bump
SEARCH_MARGIN_MM = 5.0                # arteries are searched in the liver dilated by this margin
INJECTION_REGION_MM = 40.0            # ... and in a ball around the injection point (vessels before the hilum)
AIR_CONTRAST_SD = 15.0                # air / lung is excluded only when the darkest voxels are this far below the
AIR_LEVEL = 0.35                      # liver; then below darkest + 0.35 x (liver mean - darkest), never fat
AIR_MARGIN_MM = 4.0                   # (the lung-liver interface)
VESSEL_SD = 4.0                       # seeds: brighter than the local liver background + 4 local SD
GROW_SD = 2.0                         # ... grown into connected voxels above background + 2 local SD
DENOISE_ITERATIONS = 5.0              # curvature anisotropic diffusion: iterations
DENOISE_CONDUCTANCE = 1.0             # ... conductance (lower: edges kept more strictly)
BILATERAL_DOMAIN_MM = 1.0             # bilateral: spatial sigma (mm)
BILATERAL_RANGE = 50.0                # ... intensity sigma (HU): differences larger than this are edges
SMOOTHING_MM = 0.6                    # Gaussian smoothing of the CBCT (keeps 1-2 mm arteries)
LOCAL_WINDOW_MM = 10.0                # local background: liver voxels within this cube around each voxel
LOCAL_OUTLIER_SD = 2.0                # second pass: voxels above background + 2 SD left out (vessels)
LOCAL_HALO_MM = 1.5                   # ... with this halo around them
LOCAL_MIN_COVERAGE = 0.2              # fewer liver voxels in the window (hilum, outside): global liver statistics
LOCAL_MIN_SD_FRACTION = 0.5           # the local SD is never below half of the liver's typical local SD
LIVER_SHELL_MM = 2.0                  # no seeds in the outer 2 mm of the liver (partial volume with ribs, lung)
JUNCTION_MM = 3.0                     # within this distance of the tree the shape is not checked (bifurcations)
JUNCTION_ROUNDS = 1                   # (more rounds let flat streaks creep along the tree)
TUBE_MIN = 0.3                        # tree voxels must be at least this tube-shaped (shape-only vesselness)
METAL_FACTOR = 2.5                    # metal: more than 2.5 x the brightest arteries' contrast above the liver
METAL_MARGIN_MM = 8.0                 # ... removed with this margin (blooming, the start of the streaks)
METAL_CORE_MM = 1.5                   # metal blur excluded from the tree (the margin beyond only blocks seeds)
METAL_MIN_VALUE = 2500.0              # metal is at least this bright (HU; 0 = no floor, e.g. for uncalibrated CBCT)
METAL_SPARED_MM = 10.0                # ... except around the injection point (catheter, undiluted contrast)
HALF_MAXIMUM_WINDOW_MM = 3.0          # vessel wall: half way between the liver and the brightest lumen within 3 mm
ROOT_MAX_MM = 10.0                    # injection point farther than this from the tree: not on the tree
SPUR_MM = 3.0                         # terminal centerline branches shorter than this are pruned
SNAP_MM = 3.0                         # a planned tip farther than this from any centerline is not used
LOOP_MIN_MM = 10.0                    # cycles shorter than this are skeleton noise, not loops
MAX_LOOPS = 3                         # more loops after orientation: veins or bone in the tree segment?
CANDIDATE_MIN_STEP = 0.10            # feeder finder: an upstream position must add at least 10 % territory
FEEDER_MARGIN_MM = 4.0                # tumour dilated by this margin to find the feeding branches
EXTRAHEPATIC_MM = 5.0                 # downstream branch this far outside the liver: possible non-target vessel
GEODESIC_STEP_MM = 2.0                # grid step of the geodesic territory (minimum-cost path inside the liver)
ENHANCEMENT_MIN_ML = 5.0              # enhancement territory: smaller components are dropped
FOV_EROSION_MM = 2.0                  # edge of the CBCT field of view (reconstruction artefacts) is not trusted


# Parameters that can be tuned from the module (Advanced settings): (name, label, minimum, maximum, step, tooltip).
# setParameters changes them for the whole library; DEFAULT_PARAMETERS keeps the values above.
TUNABLE = [
    ("SMOOTHING_MM", "Gaussian smoothing σ (mm)", 0.0, 3.0, 0.1,
     "Gaussian / discrete Gaussian denoising: larger, less noise but thin arteries fade."),
    ("DENOISE_ITERATIONS", "Anisotropic diffusion: iterations", 1.0, 30.0, 1.0,
     "Curvature anisotropic diffusion (denoising method): more iterations, smoother parenchyma."),
    ("DENOISE_CONDUCTANCE", "Anisotropic diffusion: conductance", 0.25, 5.0, 0.25,
     "Curvature anisotropic diffusion: lower keeps edges more strictly, higher smooths more like a Gaussian."),
    ("BILATERAL_DOMAIN_MM", "Bilateral: spatial σ (mm)", 0.25, 5.0, 0.25, "Bilateral filter: size of the smoothing."),
    ("BILATERAL_RANGE", "Bilateral: intensity σ (HU)", 5.0, 500.0, 5.0,
     "Bilateral filter: grey-value differences larger than this are treated as edges and kept."),
    ("LOCAL_WINDOW_MM", "Local background window (mm)", 4.0, 40.0, 1.0,
     "Side of the cube of liver voxels the local mean and SD are taken from."),
    ("LOCAL_OUTLIER_SD", "Background outlier (SD)", 1.0, 5.0, 0.25,
     "Voxels this far above the local background are left out of it (vessels)."),
    ("LOCAL_HALO_MM", "Background halo (mm)", 0.0, 5.0, 0.5, "Margin left out around those bright voxels."),
    ("LOCAL_MIN_COVERAGE", "Minimum liver in window", 0.05, 0.9, 0.05,
     "Fewer liver voxels in the window: the global liver statistics are used."),
    ("LOCAL_MIN_SD_FRACTION", "Local SD floor (× typical)", 0.1, 1.0, 0.05,
     "The local SD is never below this fraction of the liver's typical local SD."),
    ("SEARCH_MARGIN_MM", "Search margin around liver (mm)", 0.0, 30.0, 1.0, "Search mask: liver dilated by this."),
    ("INJECTION_REGION_MM", "Search ball at injection (mm)", 0.0, 100.0, 5.0,
     "Search mask: ball around the injection point."),
    ("AIR_MARGIN_MM", "Air / lung margin (mm)", 0.0, 15.0, 0.5, "Removed around air and lung."),
    ("AIR_CONTRAST_SD", "Air detection (SD below liver)", 5.0, 50.0, 1.0,
     "Air / lung is excluded only when the darkest voxels are this far below the liver."),
    ("LIVER_SHELL_MM", "No seeds in outer liver (mm)", 0.0, 10.0, 0.5, "Partial volume with ribs and lung."),
    ("TUBE_MIN", "Tube-shape minimum", 0.0, 1.0, 0.05, "Shape-only vesselness the tree voxels need (tube check)."),
    ("JUNCTION_MM", "Bifurcation tolerance (mm)", 0.0, 10.0, 0.5,
     "Within this distance of the tree the tube shape is not checked."),
    ("HALF_MAXIMUM_WINDOW_MM", "Vessel wall window (mm)", 1.0, 10.0, 0.5,
     "Wall at half maximum of the brightest lumen within this distance."),
    ("METAL_FACTOR", "Metal: × brightest arteries", 1.0, 10.0, 0.25,
     "Brighter above the liver than this times the brightest arteries: metal."),
    ("METAL_MIN_VALUE", "Metal minimum value (HU)", 0.0, 10000.0, 100.0,
     "Metal is at least this bright; dense contrast in an artery stays below it. 0: no floor (uncalibrated CBCT)."),
    ("METAL_MARGIN_MM", "Metal margin, no seeds (mm)", 0.0, 20.0, 1.0,
     "No seeds this close to metal (blooming); arteries passing by still grow through."),
    ("METAL_SPARED_MM", "Metal guard off near injection (mm)", 0.0, 40.0, 1.0,
     "No metal exclusion this close to the injection point."),
    ("SPUR_MM", "Prune spurs shorter than (mm)", 0.0, 10.0, 0.5,
     "Short terminal centerline branches (in thick vessels: shorter than 1.5 x the vessel radius)."),
    ("MAX_VESSEL_RADIUS_MM", "Tube check: largest radius (mm)", 3.0, 25.0, 0.5,
     "Largest Hessian scale of the tube check (sigma = radius / sqrt 3). Larger also recognises thick tubes by "
     "shape, but thick flat streaks and blobs start to look like tubes too."),
    ("FOV_EROSION_MM", "FOV edge removed (mm)", 0.0, 10.0, 0.5, "Edge of the CBCT field of view not trusted."),
]
DEFAULT_PARAMETERS = {name: globals()[name] for name, *_ in TUNABLE}


def setParameters(values):
    """Set tunable parameters (unknown names ignored); missing ones go back to their defaults. The denoising method
    is the text value DENOISE_METHOD."""
    global DENOISE_METHOD
    values = values or {}
    for name, default in DEFAULT_PARAMETERS.items():
        globals()[name] = float(values.get(name, default))
    method = values.get("DENOISE_METHOD", DEFAULT_DENOISE_METHOD)
    DENOISE_METHOD = method if method in dict(DENOISE_METHODS) else DEFAULT_DENOISE_METHOD


# -- Vesselness --------------------------------------------------------------------------------------------------

def _hessianEigenvalues(image, sigmaVoxels, region):
    """Eigenvalues (n, 3) of the scale-normalised Hessian at the region voxels, sorted by absolute value."""
    image = np.asarray(image, dtype=np.float32)
    orders = [(2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (1, 0, 1), (0, 1, 1)]
    components = [ndimage.gaussian_filter(image, sigmaVoxels, order=order, mode="nearest")[region]
                  for order in orders]
    hkk, hjj, hii, hkj, hki, hji = components
    count = hkk.shape[0]
    eigen = np.empty((count, 3), dtype=np.float32)
    block = 500000
    for start in range(0, count, block):
        stop = min(count, start + block)
        matrix = np.empty((stop - start, 3, 3), dtype=np.float32)
        matrix[:, 0, 0], matrix[:, 1, 1], matrix[:, 2, 2] = hkk[start:stop], hjj[start:stop], hii[start:stop]
        matrix[:, 0, 1] = matrix[:, 1, 0] = hkj[start:stop]
        matrix[:, 0, 2] = matrix[:, 2, 0] = hki[start:stop]
        matrix[:, 1, 2] = matrix[:, 2, 1] = hji[start:stop]
        values = np.linalg.eigvalsh(matrix)
        order = np.argsort(np.abs(values), axis=1)
        eigen[start:stop] = np.take_along_axis(values, order, axis=1)
    return eigen


def vesselScales(maxRadiusMM=None):
    """Hessian scales (Gaussian sigma, mm) for tubes from 0.87 mm radius up to maxRadiusMM: a bright tube of
    radius r responds best at sigma = r / sqrt(3) (Frangi et al. 1998); radii doubling."""
    maxRadiusMM = MAX_VESSEL_RADIUS_MM if maxRadiusMM is None else maxRadiusMM
    radii, radius = [], 0.5 * np.sqrt(3.0)
    while radius < maxRadiusMM:
        radii.append(radius)
        radius *= 2.0
    radii.append(maxRadiusMM)
    return tuple(round(r / np.sqrt(3.0), 3) for r in radii)


def vesselness(image, spacing, region=None, scalesMM=VESSEL_SCALES_MM, alpha=0.5, beta=0.5, bright=True,
               progress=None, shapeOnly=False):
    """Frangi vesselness (max over scales) inside region (bool, default everywhere); float32 volume, 0 outside.

    bright: contrast-filled vessels are brighter than their surroundings (CBCT arteriography). shapeOnly: without
    Frangi's structure-strength term (which is scaled by the strongest structure, e.g. a bright trunk, and would
    make every small branch look weak): 1 for an ideal tube, ~0 for blobs (speckle) and plates (streaks)."""
    image = np.asarray(image, dtype=np.float32)
    if region is None:
        region = np.ones(image.shape, bool)
    spacing = np.asarray(spacing, dtype=float)
    result = np.zeros(int(np.count_nonzero(region)), dtype=np.float32)
    for number, scale in enumerate(scalesMM):
        if progress is not None:
            progress(f"Vesselness at {scale:g} mm ({number + 1} of {len(scalesMM)})")
        sigma = np.maximum(scale / spacing, 0.5)
        eigen = _hessianEigenvalues(image, sigma, region) * float(scale) ** 2
        l1, l2, l3 = eigen[:, 0], eigen[:, 1], eigen[:, 2]
        a2, a3 = np.abs(l2), np.abs(l3)
        with np.errstate(divide="ignore", invalid="ignore"):
            ra = np.where(a3 > 0, a2 / a3, 0.0)
            rb = np.where(a2 * a3 > 0, np.abs(l1) / np.sqrt(a2 * a3), 0.0)
        s = np.sqrt(l1 ** 2 + l2 ** 2 + l3 ** 2)
        c = 0.5 * float(s.max()) if s.size and s.max() > 0 else 1.0
        value = (1.0 - np.exp(-ra ** 2 / (2 * alpha ** 2))) * np.exp(-rb ** 2 / (2 * beta ** 2))
        if not shapeOnly:
            value = value * (1.0 - np.exp(-s ** 2 / (2 * c ** 2)))
        tubular = (l2 < 0) & (l3 < 0) if bright else (l2 > 0) & (l3 > 0)
        value = np.where(tubular, value, 0.0).astype(np.float32)
        np.maximum(result, value, out=result)
    volume = np.zeros(image.shape, dtype=np.float32)
    volume[region] = result
    return volume


def tubeShape(image, spacing, region=None, scalesMM=None, progress=None):
    """Shape-only vesselness: how much each voxel looks like a bright tube, whatever its brightness (0..1). Scales
    up to MAX_VESSEL_RADIUS_MM (vesselScales): a vessel thicker than the largest scale is not tube-shaped inside, its
    core falls out of the tree and its skeleton becomes a tangle."""
    scalesMM = vesselScales() if scalesMM is None else scalesMM
    return vesselness(image, spacing, region, scalesMM, progress=progress, shapeOnly=True)


def ballMask(shape, centreKJI, radiusMM, spacing):
    """Ball around a voxel position (k, j, i), radius in mm."""
    spacing = np.asarray(spacing, dtype=float)
    centre = np.asarray(centreKJI, dtype=float)
    lower = np.maximum(np.floor(centre - radiusMM / spacing).astype(int), 0)
    upper = np.minimum(np.ceil(centre + radiusMM / spacing).astype(int) + 1, np.array(shape))
    mask = np.zeros(shape, bool)
    if np.any(upper <= lower):
        return mask
    k, j, i = np.ogrid[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]]
    distance2 = (((k - centre[0]) * spacing[0]) ** 2 + ((j - centre[1]) * spacing[1]) ** 2
                 + ((i - centre[2]) * spacing[2]) ** 2)
    mask[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]] = distance2 <= radiusMM ** 2
    return mask


def dilateMM(mask, marginMM, spacing):
    """Dilation by a margin in mm (Euclidean, anisotropic spacing)."""
    if marginMM <= 0:
        return np.asarray(mask, bool).copy()
    distance = ndimage.distance_transform_edt(~np.asarray(mask, bool), sampling=spacing)
    return distance <= marginMM


def erodeMM(mask, marginMM, spacing):
    if marginMM <= 0:
        return np.asarray(mask, bool).copy()
    distance = ndimage.distance_transform_edt(np.asarray(mask, bool), sampling=spacing)
    return distance > marginMM


@dataclasses.dataclass
class LiverStatistics:
    """Robust grey-value statistics of the liver parenchyma (median and MAD-based SD: vessels and tumours inside the
    liver do not inflate them). CBCT values are not calibrated HU; every threshold is relative to these."""
    mean: float
    sd: float

    def above(self, k):
        return self.mean + k * self.sd


def liverStatistics(values, liver, fov=None):
    region = np.asarray(liver, bool) if fov is None else (np.asarray(liver, bool) & fov)
    inside = np.asarray(values, dtype=float)[region]
    if inside.size == 0:
        raise ValueError("No liver voxels inside the CBCT field of view: check the registration of the CBCT.")
    median = float(np.median(inside))
    sd = 1.4826 * float(np.median(np.abs(inside - median)))
    if sd <= 0:
        sd = float(inside.std()) or 1.0
    return LiverStatistics(median, sd)


def airThreshold(values, statistics, region=None, contrastSD=None, level=None):
    """Grey value below which a voxel is air or lung, or None when there is no air in the image. The darkest voxels
    (0.1 percentile) must be at least contrastSD liver SDs below the liver; the threshold then lies at level of the
    way from them up to the liver mean, well below fat (fat around the hilum must stay: the arteries run in it)."""
    contrastSD = AIR_CONTRAST_SD if contrastSD is None else contrastSD
    level = AIR_LEVEL if level is None else level
    sample = np.asarray(values) if region is None else np.asarray(values)[region]
    if sample.size == 0:
        return None
    darkest = float(np.percentile(sample, 0.1))
    if statistics.mean - darkest < contrastSD * statistics.sd:
        return None
    return darkest + level * (statistics.mean - darkest)


def airMask(values, statistics, region=None):
    threshold = airThreshold(values, statistics, region)
    if threshold is None:
        return np.zeros(np.asarray(values).shape, bool)
    return np.asarray(values) < threshold


def searchRegion(values, liver, spacing, statistics, injectionKJI=None, fov=None, marginMM=None,
                 injectionMM=None, airMarginMM=None):
    """Where arteries are searched: the liver plus a small margin and a ball around the injection point (the
    arteries between the catheter tip and the hilum), inside the CBCT field of view, without air / lung and a margin
    around it (the lung-liver interface). Bone and kidneys beside the liver are outside this region."""
    marginMM = SEARCH_MARGIN_MM if marginMM is None else marginMM
    injectionMM = INJECTION_REGION_MM if injectionMM is None else injectionMM
    airMarginMM = AIR_MARGIN_MM if airMarginMM is None else airMarginMM
    liver = np.asarray(liver, bool)
    region = dilateMM(liver, marginMM, spacing)
    if injectionKJI is not None:
        region |= ballMask(liver.shape, injectionKJI, injectionMM, spacing)
    if fov is not None:
        region &= fov
    air = airMask(values, statistics, fov)
    if air.any():
        region &= ~dilateMM(air, airMarginMM, spacing)
    return region


# -- Arterial tree segment ------------------------------------------------------------------------------------

def nearestTrue(mask, positionKJI, spacing, maxMM):
    """Index (k, j, i) of the mask voxel nearest to a position (mm metric), or None if farther than maxMM."""
    points = np.argwhere(mask)
    if points.size == 0:
        return None
    spacing = np.asarray(spacing, dtype=float)
    distance = np.sqrt((((points - np.asarray(positionKJI, dtype=float)) * spacing) ** 2).sum(axis=1))
    best = int(np.argmin(distance))
    return tuple(int(v) for v in points[best]) if distance[best] <= maxMM else None


DENOISE_GAUSSIAN = "gaussian"
DENOISE_DISCRETE_GAUSSIAN = "discreteGaussian"
DENOISE_CURVATURE = "curvatureAnisotropicDiffusion"
DENOISE_BILATERAL = "bilateral"
DENOISE_METHODS = [
    (DENOISE_CURVATURE, "Curvature anisotropic diffusion (edge-preserving, default)"),
    (DENOISE_GAUSSIAN, "Gaussian (sampled kernel)"),
    (DENOISE_DISCRETE_GAUSSIAN, "Discrete Gaussian (Lindeberg kernel)"),
    (DENOISE_BILATERAL, "Bilateral (edge-preserving)"),
]
DEFAULT_DENOISE_METHOD = DENOISE_CURVATURE
DENOISE_METHOD = DEFAULT_DENOISE_METHOD


def denoise(values, spacing, sigmaMM=None, method=None):
    """CBCT noise. method: DENOISE_METHOD (curvature anisotropic diffusion unless changed).
    - Gaussian: scipy, a sampled Gaussian kernel of sigmaMM (SMOOTHING_MM).
    - Discrete Gaussian: SimpleITK, Lindeberg's discrete Gaussian kernel (more correct for a sigma below a voxel).
    - Curvature anisotropic diffusion: SimpleITK, edge-preserving (Whitaker & Xue 2001, after Perona & Malik 1990):
      the parenchyma is smoothed, vessel walls are not; DENOISE_ITERATIONS, DENOISE_CONDUCTANCE.
    - Bilateral: SimpleITK, edge-preserving (Tomasi & Manduchi 1998); BILATERAL_DOMAIN_MM, BILATERAL_RANGE.
    No median filter: it erases arteries only 1-3 voxels wide; speckle is rejected by the local SD and the tube
    check instead."""
    sigmaMM = SMOOTHING_MM if sigmaMM is None else sigmaMM
    method = method or DENOISE_METHOD
    values = np.asarray(values, dtype=np.float32)
    if method == DENOISE_GAUSSIAN or sigmaMM <= 0 and method == DENOISE_DISCRETE_GAUSSIAN:
        sigma = np.asarray([sigmaMM / float(s) for s in spacing])
        return ndimage.gaussian_filter(values, sigma, mode="nearest")
    try:
        import SimpleITK as sitk
    except ImportError:   # outside Slicer without SimpleITK: Gaussian
        sigma = np.asarray([sigmaMM / float(s) for s in spacing])
        return ndimage.gaussian_filter(values, sigma, mode="nearest")
    image = sitk.GetImageFromArray(values)
    image.SetSpacing([float(s) for s in spacing[::-1]])    # arrays are (k, j, i), SimpleITK (x, y, z)
    if method == DENOISE_DISCRETE_GAUSSIAN:
        result = sitk.DiscreteGaussian(image, variance=float(sigmaMM) ** 2, maximumKernelWidth=64,
                                       maximumError=0.01, useImageSpacing=True)
    elif method == DENOISE_CURVATURE:
        timeStep = min(float(s) for s in spacing) / 2.0 ** (3 + 1)   # stable step for 3D
        result = sitk.CurvatureAnisotropicDiffusion(image, timeStep=timeStep,
                                                    conductanceParameter=float(DENOISE_CONDUCTANCE),
                                                    numberOfIterations=int(round(DENOISE_ITERATIONS)))
    elif method == DENOISE_BILATERAL:
        result = sitk.Bilateral(image, domainSigma=float(BILATERAL_DOMAIN_MM), rangeSigma=float(BILATERAL_RANGE))
    else:
        raise ValueError(f"Unknown denoising method '{method}'.")
    return sitk.GetArrayFromImage(result).astype(np.float32)


@dataclasses.dataclass
class LocalContrast:
    """Contrast of every voxel against the liver around it: z = (value - local background) / local SD. The background
    and SD come from the liver voxels within a window (normalised convolution: voxels outside the liver are not
    used, so the liver edge is measured against the liver only); bright voxels (vessels) are left out of the
    background in later passes. Slow shading of the CBCT (scatter, cupping) cancels out, and the local SD rises in
    streaks and noisy regions, which lowers their contrast. Away from the liver (hilum) the global liver
    statistics are used."""
    z: np.ndarray
    background: np.ndarray
    scale: np.ndarray
    statistics: LiverStatistics       # of the denoised liver (global fallback)

    def at(self, kji):
        index = tuple(int(round(v)) for v in kji)
        return float(self.z[index])


def _boxMean(array, window):
    return ndimage.uniform_filter(array, size=window, mode="constant")


def localContrast(denoised, liver, spacing, windowMM=None, outlierSD=None,
                  minCoverage=None, minSDFraction=None, fov=None):
    windowMM = LOCAL_WINDOW_MM if windowMM is None else windowMM
    outlierSD = LOCAL_OUTLIER_SD if outlierSD is None else outlierSD
    minCoverage = LOCAL_MIN_COVERAGE if minCoverage is None else minCoverage
    minSDFraction = LOCAL_MIN_SD_FRACTION if minSDFraction is None else minSDFraction
    denoised = np.asarray(denoised, dtype=np.float32)
    liver = np.asarray(liver, bool)
    mask = liver if fov is None else (liver & fov)
    statistics = liverStatistics(denoised, mask)
    window = [max(3, 2 * int(round(windowMM / (2 * float(s)))) + 1) for s in spacing]
    weights = mask.astype(np.float32)
    background = np.full(denoised.shape, statistics.mean, dtype=np.float32)
    variance = np.full(denoised.shape, statistics.sd ** 2, dtype=np.float32)
    for _ in range(3):   # later passes: vessels and other bright voxels left out of the background
        coverage = _boxMean(weights, window)
        valid = coverage >= minCoverage
        with np.errstate(divide="ignore", invalid="ignore"):
            mean = _boxMean(denoised * weights, window) / coverage
            square = _boxMean(denoised * denoised * weights, window) / coverage
        background = np.where(valid, mean, statistics.mean).astype(np.float32)
        variance = np.where(valid, np.maximum(square - mean * mean, 0.0), statistics.sd ** 2).astype(np.float32)
        bright = denoised >= background + outlierSD * np.sqrt(variance)
        # vessels and their blur halo (the reconstruction spreads a bright lumen over 1-2 mm) are left out
        bright = dilateMM(bright, LOCAL_HALO_MM, spacing) if bright.any() else bright
        weights = (mask & ~bright).astype(np.float32)
    del weights
    scale = np.sqrt(variance)
    inside = mask & valid
    # the typical local SD of the liver (noise, without the shading that inflates the global SD): used away from the
    # liver, so z has one scale inside and outside it, and as the floor of the local SD in smooth regions
    typical = float(np.median(scale[inside])) if inside.any() else statistics.sd
    scale = np.where(valid, scale, typical)
    scale = np.maximum(scale, minSDFraction * typical).astype(np.float32)
    z = ((denoised - background) / scale).astype(np.float32)
    return LocalContrast(z, background, scale, statistics)


def metalMask(denoised, contrast, seeds, spacing, injectionKJI=None, factor=None, marginMM=None,
              sparedMM=None):
    """(metal, zone): metal (coils, clips) = brighter above the liver than METAL_FACTOR times the brightest arteries
    (99th percentile of the seeds), or at the saturation value of the image (metal is often clipped at the top of
    the range) when that is well above the arteries, and in any case at least METAL_MIN_VALUE (HU: dense iodine in
    an artery stays below it); zone = metal plus the margin (blooming, the start of the streaks). Spared around the
    injection point (catheter tip, undiluted contrast)."""
    factor = METAL_FACTOR if factor is None else factor
    marginMM = METAL_MARGIN_MM if marginMM is None else marginMM
    sparedMM = METAL_SPARED_MM if sparedMM is None else sparedMM
    denoised = np.asarray(denoised)
    none = np.zeros(denoised.shape, bool)
    if not np.any(seeds):
        return none, none
    level = contrast.statistics.mean
    brightest = float(np.percentile(denoised[seeds], 99)) - level
    if brightest <= 0:
        return none, none
    metal = denoised > level + factor * brightest
    top = float(denoised.max())
    if top > level + 1.5 * brightest:
        metal |= denoised >= level + 0.95 * (top - level)
    if METAL_MIN_VALUE > 0:
        metal &= denoised >= METAL_MIN_VALUE
    if injectionKJI is not None:
        metal &= ~ballMask(metal.shape, injectionKJI, sparedMM, spacing)
    if not metal.any():
        return metal, metal
    # the metal's own blur (blooming) belongs to it; the rest of the margin keeps only the seeds out
    return dilateMM(metal, min(METAL_CORE_MM, marginMM), spacing), dilateMM(metal, marginMM, spacing)


@dataclasses.dataclass
class TreeSegment:
    mask: np.ndarray
    metalML: float = 0.0       # metal + margin removed from the search region
    seedML: float = 0.0
    seeds: np.ndarray = None   # seed voxels (bright and tube-shaped) of the tree: twigs without one are pruned


def _injectionComponent(candidate, strong, injectionKJI, spacing, rootMaxMM, growSD):
    """Hysteresis: the connected component of the candidates that contains a seed and is nearest to the injection
    point."""
    labels, count = ndimage.label(candidate, structure=np.ones((3, 3, 3)))
    if count == 0:
        raise ValueError(f"Nothing brighter than the local liver background + {growSD:g} SD in the search region: "
                         "lower the thresholds or check the CBCT registration.")
    seeded = np.unique(labels[strong])
    keep = np.zeros(count + 1, bool)
    keep[seeded[seeded > 0]] = True
    start = nearestTrue(keep[labels], injectionKJI, spacing, rootMaxMM)
    if start is None:
        raise ValueError(f"No contrast-filled vessel within {rootMaxMM:g} mm of the injection point: place it inside "
                         "the artery at the catheter tip, or lower the vessel threshold.")
    return labels == labels[start]


def arterialTree(denoised, contrast, region, injectionKJI, spacing, liver=None, vesselSD=VESSEL_SD, growSD=GROW_SD,
                 tube=None, rootMaxMM=ROOT_MAX_MM, voxelML=None, trimWall=True):
    """Arterial tree segment by hysteresis on the local contrast z: seeds above vesselSD local SD (not in the outer
    LIVER_SHELL_MM of the liver: partial volume with ribs), grown into connected voxels above growSD, inside the
    search region without metal; the connected component containing (or nearest to) the injection point, trimmed to
    the vessel wall at half maximum (half way between the local background and the brightest lumen within
    HALF_MAXIMUM_WINDOW_MM), keeping the centerlines so faint branches stay attached. tube (optional shape-only
    vesselness, tubeShape): seeds and grown voxels must also be tube-shaped (3 x 3 x 3 mean score at least
    TUBE_MIN), except within JUNCTION_MM of the tree (bifurcations); speckle and streaks are not tubes."""
    z = contrast.z
    strong = region & (z >= vesselSD)
    if liver is not None:
        strong &= ~(np.asarray(liver, bool) & ~erodeMM(liver, LIVER_SHELL_MM, spacing))
    # metal itself is never part of the tree; its margin only keeps seeds out (blooming, streak starts): an artery
    # running past a coil is still grown through it (the tube check stops the streaks)
    metalCore, metal = metalMask(denoised, contrast, strong, spacing, injectionKJI)
    region = region & ~metalCore
    strong &= ~metal
    candidate = region & (z >= min(growSD, vesselSD))
    if tube is None:
        tree = _injectionComponent(candidate, strong, injectionKJI, spacing, rootMaxMM, growSD)
    else:
        # seeds and grown voxels must be tube-shaped (bright streaks are flat, speckle is blob-like); the score is
        # averaged over the candidate voxels of the 3 x 3 x 3 neighbourhood (stable against noise)
        weights = candidate.astype(np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            average = (ndimage.uniform_filter(np.where(candidate, tube, 0.0).astype(np.float32), size=3)
                       / ndimage.uniform_filter(weights, size=3))
        tubular = candidate & (np.nan_to_num(average) >= TUBE_MIN)
        del weights, average
        strong &= tubular
        tree = _injectionComponent(tubular, strong, injectionKJI, spacing, rootMaxMM, growSD)
        # bifurcations: where a branch leaves its parent the Hessian is not tube-like for a few mm, so the shape is
        # not checked within JUNCTION_MM of the tree; repeated for the branches this adds
        for _ in range(JUNCTION_ROUNDS):
            allowed = tubular | (dilateMM(tree, JUNCTION_MM, spacing) & candidate)
            grown = _injectionComponent(allowed, strong, injectionKJI, spacing, rootMaxMM, growSD)
            if np.count_nonzero(grown) == np.count_nonzero(tree):
                break
            tree = grown
    if trimWall:
        # the wall at half maximum: a bright lumen blurred by the reconstruction would otherwise grow by its blur.
        # The centerlines (and one voxel around them) are kept, so a faint branch is not cut off where it leaves a
        # much brighter parent.
        window = [max(3, 2 * int(round(HALF_MAXIMUM_WINDOW_MM / (2 * float(s)))) + 1) for s in spacing]
        above = np.asarray(denoised, dtype=np.float32) - contrast.background   # grey values: same scale everywhere
        local = ndimage.maximum_filter(np.where(tree, above, 0.0).astype(np.float32), size=window)
        trimmed = tree & (above >= 0.5 * local)
        del above, local
        try:
            core = ndimage.binary_dilation(skeletonize(tree), structure=ndimage.generate_binary_structure(3, 1))
            trimmed |= core & tree
        except ImportError:
            pass   # without scikit-image: plain half-maximum wall
        tree = trimmed
        labels, _ = ndimage.label(tree, structure=np.ones((3, 3, 3)))
        start = nearestTrue(labels > 0, injectionKJI, spacing, rootMaxMM)
        if start is not None:
            tree = labels == labels[start]
    tree = ndimage.binary_closing(tree, structure=ndimage.generate_binary_structure(3, 1)) | tree
    tree = ndimage.binary_fill_holes(tree)
    voxelML = voxelML if voxelML is not None else float(np.prod(spacing)) / 1000.0
    return TreeSegment(tree, float(np.count_nonzero(metal)) * voxelML, float(np.count_nonzero(strong)) * voxelML,
                       strong & tree)


# -- Centerlines and tree graph -------------------------------------------------------------------------------

def skeletonize(mask):
    """3D medial-axis thinning (Lee et al. 1994) via scikit-image. Raises ImportError if it is missing."""
    from skimage.morphology import skeletonize as _skeletonize
    try:
        skeleton = _skeletonize(np.asarray(mask, bool), method="lee")
    except TypeError:   # older scikit-image: skeletonize_3d
        from skimage.morphology import skeletonize_3d
        skeleton = skeletonize_3d(np.asarray(mask, bool))
    return np.asarray(skeleton) > 0


_FORWARD_OFFSETS = np.array([(dk, dj, di) for dk in (-1, 0, 1) for dj in (-1, 0, 1) for di in (-1, 0, 1)
                             if (dk, dj, di) > (0, 0, 0)])


def _skeletonGraph(points, shape, spacing):
    """Symmetric sparse graph of 26-connected skeleton voxels, edge weights in mm."""
    count = len(points)
    flat = np.ravel_multi_index(points.T, shape)
    order = np.argsort(flat)
    sortedFlat = flat[order]
    rows, cols, weights = [], [], []
    spacing = np.asarray(spacing, dtype=float)
    upper = np.array(shape)
    for offset in _FORWARD_OFFSETS:
        neighbours = points + offset
        valid = np.all((neighbours >= 0) & (neighbours < upper), axis=1)
        if not valid.any():
            continue
        source = np.flatnonzero(valid)
        target = np.ravel_multi_index(neighbours[valid].T, shape)
        position = np.searchsorted(sortedFlat, target)
        position[position >= count] = 0
        found = sortedFlat[position] == target
        rows.append(source[found])
        cols.append(order[position[found]])
        weights.append(np.full(int(found.sum()), float(np.linalg.norm(offset * spacing))))
    if rows:
        rows, cols, weights = np.concatenate(rows), np.concatenate(cols), np.concatenate(weights)
    else:
        rows = cols = np.zeros(0, int)
        weights = np.zeros(0)
    graph = sparse.coo_matrix((weights, (rows, cols)), shape=(count, count)).tocsr()
    return (graph + graph.T).tocsr()


def _pruneSpurs(graph, keepIndex, spurMM, rounds=3, radius=None):
    """Alive mask after removing terminal branches shorter than spurMM, or than SPUR_RADIUS_FACTOR x the vessel
    radius at their junction (radius (n,), optional: bumps of a thick vessel's surface give long thinning spurs);
    never the branch of keepIndex."""
    count = graph.shape[0]
    alive = np.ones(count, bool)
    indptr, indices, data = graph.indptr, graph.indices, graph.data
    longest = spurMM if radius is None or not len(radius) else max(spurMM, SPUR_RADIUS_FACTOR * float(radius.max()))
    for _ in range(rounds):
        degree = np.array([int(alive[indices[indptr[n]:indptr[n + 1]]].sum()) for n in range(count)])
        degree[~alive] = 0
        remove = set()
        for end in np.flatnonzero(alive & (degree == 1)):
            path, length, previous, current = [int(end)], 0.0, -1, int(end)
            spur = False
            while True:
                neighbours = [(int(indices[p]), float(data[p])) for p in range(indptr[current], indptr[current + 1])
                              if alive[indices[p]] and indices[p] != previous]
                if not neighbours:
                    break   # an isolated segment: not a spur
                nextNode, step = neighbours[0]
                length += step
                limit = spurMM if radius is None else max(spurMM, SPUR_RADIUS_FACTOR * float(radius[nextNode]))
                if length >= longest:
                    break
                if degree[nextNode] >= 3:
                    spur = length < limit   # reached a junction: a spur if shorter than its limit
                    break
                previous, current = current, nextNode
                path.append(current)
            if spur and keepIndex not in path:
                remove.update(path)
        if not remove:
            break
        alive[list(remove)] = False
    return alive


@dataclasses.dataclass
class VesselTree:
    """Centerline points of the arterial tree, oriented from the injection point (root)."""
    points: np.ndarray        # (n, 3) voxel indices (k, j, i)
    spacing: tuple            # (sk, sj, si) mm
    shape: tuple              # grid shape
    parent: np.ndarray        # (n,) index of the upstream point, -1 at the root
    distance: np.ndarray      # (n,) path length from the root (mm)
    radius: np.ndarray        # (n,) vessel radius from the distance map (mm)
    root: int
    tin: np.ndarray           # Euler tour: point m is downstream of p iff tin[p] <= tin[m] < tout[p]
    tout: np.ndarray
    loops: int = 0            # independent cycles longer than LOOP_MIN_MM

    @property
    def count(self):
        return len(self.points)

    def positionsMM(self):
        return self.points * np.asarray(self.spacing, dtype=float)

    def children(self):
        result = [[] for _ in range(self.count)]
        for child, parent in enumerate(self.parent):
            if parent >= 0:
                result[parent].append(child)
        return result

    def branchPoints(self):
        counts = np.bincount(self.parent[self.parent >= 0], minlength=self.count)
        return np.flatnonzero(counts >= 2)

    def downstream(self, index):
        """Bool (n,): the point and every point downstream of it."""
        return (self.tin >= self.tin[index]) & (self.tin < self.tout[index])

    def isAncestor(self, ancestor, index):
        return self.tin[ancestor] <= self.tin[index] < self.tout[ancestor]

    def pathToRoot(self, index):
        path = [int(index)]
        while self.parent[path[-1]] >= 0:
            path.append(int(self.parent[path[-1]]))
        return path

    def nearest(self, positionKJI, maxMM=None):
        """(index, distance mm) of the centerline point nearest to a voxel position; index None if beyond maxMM."""
        spacing = np.asarray(self.spacing, dtype=float)
        distance = np.sqrt((((self.points - np.asarray(positionKJI, dtype=float)) * spacing) ** 2).sum(axis=1))
        best = int(np.argmin(distance))
        if maxMM is not None and distance[best] > maxMM:
            return None, float(distance[best])
        return best, float(distance[best])

    def edgesMM(self):
        """(n - 1, 2, 3) segment end points in mm (grid frame), child -> parent."""
        child = np.flatnonzero(self.parent >= 0)
        positions = self.positionsMM()
        return np.stack([positions[child], positions[self.parent[child]]], axis=1)

    def lengthMM(self):
        child = np.flatnonzero(self.parent >= 0)
        return float(np.sum(self.distance[child] - self.distance[self.parent[child]]))


def _eulerTour(parent, root):
    count = len(parent)
    children = [[] for _ in range(count)]
    for child, up in enumerate(parent):
        if up >= 0:
            children[up].append(child)
    tin = np.zeros(count, dtype=np.int64)
    tout = np.zeros(count, dtype=np.int64)
    clock = 0
    stack = [(root, False)]
    while stack:
        node, done = stack.pop()
        if done:
            tout[node] = clock
            continue
        tin[node] = clock
        clock += 1
        stack.append((node, True))
        for child in children[node]:
            stack.append((child, False))
    return tin, tout


def _countLoops(graph, parent, distance, positionsMM, loopMinMM):
    """Independent cycles longer than loopMinMM: non-tree edges whose tree path is long, grouped by position."""
    coo = sparse.triu(graph).tocoo()
    flagged = []
    for u, v, weight in zip(coo.row, coo.col, coo.data):
        if parent[u] == v or parent[v] == u:
            continue
        # tree path between u and v, walked up at most loopMinMM from each end
        seen = {}
        node, walked = int(u), 0.0
        while node >= 0 and walked <= loopMinMM:
            seen[node] = walked
            up = parent[node]
            if up >= 0:
                walked += distance[node] - distance[up]
            node = up
        node, walked, short = int(v), 0.0, False
        while node >= 0 and walked <= loopMinMM:
            if node in seen and seen[node] + walked + weight < loopMinMM:
                short = True
                break
            up = parent[node]
            if up >= 0:
                walked += distance[node] - distance[up]
            node = up
        if not short:
            flagged.append((u, v))
    if not flagged:
        return 0
    centres = np.array([(positionsMM[u] + positionsMM[v]) / 2.0 for u, v in flagged])
    pairs = cKDTree(centres).query_pairs(r=loopMinMM / 2.0, output_type="ndarray")
    groups = sparse.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])) if len(pairs) else
                               (np.zeros(0), (np.zeros(0, int), np.zeros(0, int))), shape=(len(flagged),) * 2)
    return int(csgraph.connected_components(groups, directed=False)[0])


def buildTree(skeleton, spacing, injectionKJI, vesselMask=None, spurMM=None, rootMaxMM=ROOT_MAX_MM,
              loopMinMM=LOOP_MIN_MM):
    """Oriented centerline tree from a skeleton: spurs pruned, the component of the root kept, shortest-path tree
    from the point nearest the injection point. vesselMask (optional) gives the radius at each point."""
    spurMM = SPUR_MM if spurMM is None else spurMM
    skeleton = np.asarray(skeleton, bool)
    points = np.argwhere(skeleton)
    if len(points) < 2:
        raise ValueError("The arterial tree has no centerline (too small or empty).")
    spacing = tuple(float(v) for v in spacing)
    graph = _skeletonGraph(points, skeleton.shape, spacing)
    rootIndex, rootDistance = _nearestPoint(points, injectionKJI, spacing)
    if rootDistance > rootMaxMM:
        raise ValueError(f"The injection point is {rootDistance:.0f} mm from the arterial tree (more than "
                         f"{rootMaxMM:g} mm): place it inside the artery at the catheter tip.")
    pointRadius = None
    if vesselMask is not None:
        radiusMap = ndimage.distance_transform_edt(np.asarray(vesselMask, bool), sampling=spacing)
        pointRadius = radiusMap[tuple(points.T)].astype(float)
    alive = _pruneSpurs(graph, rootIndex, spurMM, radius=pointRadius)
    _, labels = csgraph.connected_components(graph[alive][:, alive], directed=False)
    aliveIndex = np.flatnonzero(alive)
    rootLabel = labels[np.searchsorted(aliveIndex, rootIndex)]
    keep = aliveIndex[labels == rootLabel]
    points = points[keep]
    graph = graph[keep][:, keep].tocsr()
    root = int(np.searchsorted(keep, rootIndex))
    distance, predecessors = csgraph.dijkstra(graph, directed=False, indices=root, return_predecessors=True)
    parent = np.where(predecessors < 0, -1, predecessors).astype(np.int64)
    parent[root] = -1
    tin, tout = _eulerTour(parent, root)
    if pointRadius is not None:
        radius = pointRadius[keep]
    else:
        radius = np.zeros(len(points))
    positions = points * np.asarray(spacing)
    loops = _countLoops(graph, parent, distance, positions, loopMinMM)
    return VesselTree(points, spacing, tuple(skeleton.shape), parent, distance, radius, root, tin, tout, loops)


def _nearestPoint(points, positionKJI, spacing):
    distance = np.sqrt((((points - np.asarray(positionKJI, dtype=float)) * np.asarray(spacing)) ** 2).sum(axis=1))
    best = int(np.argmin(distance))
    return best, float(distance[best])


def pruneTwigs(tree, seeded):
    """Bool (n,) of the centerline points to keep: terminal branches (from a leaf up to the next branch point) that
    contain no seed point are removed, repeatedly, until every remaining leaf branch reaches a seed. Faint segments
    leading to a bright vessel further downstream are kept; twigs grown into noise are not (hysteresis on the tree)."""
    seeded = np.asarray(seeded, bool)
    alive = np.ones(tree.count, bool)
    while True:
        children = np.zeros(tree.count, dtype=np.int64)
        living = alive & (tree.parent >= 0)
        np.add.at(children, tree.parent[living], 1)
        leaves = np.flatnonzero(alive & (children == 0))
        removed = False
        for leaf in leaves:
            if leaf == tree.root:
                continue
            path, node = [], int(leaf)
            while node >= 0 and node != tree.root and children[node] <= 1:
                path.append(node)
                node = int(tree.parent[node])
            if path and not seeded[path].any():
                alive[path] = False
                removed = True
        if not removed:
            return alive


def maskOfKept(mask, tree, keep):
    """The tree segment without the voxels nearest to pruned centerline points."""
    mask = np.asarray(mask, bool)
    if keep.all():
        return mask.copy()
    voxels = np.argwhere(mask)
    spacing = np.asarray(tree.spacing, dtype=float)
    _, nearest = cKDTree(tree.points * spacing).query(voxels * spacing, k=1)
    result = np.zeros(mask.shape, bool)
    kept = voxels[keep[nearest]]
    result[tuple(kept.T)] = True
    return result


def snapTip(tree, positionKJI, snapMM=SNAP_MM):
    """(centerline index, distance mm) of a planned tip; index None when farther than snapMM from the tree."""
    return tree.nearest(positionKJI, snapMM)


# -- Territories -------------------------------------------------------------------------------------------------

@dataclasses.dataclass
class TerritoryMap:
    """For every liver voxel, the centerline point that supplies it (-1: none)."""
    shape: tuple
    liverFlat: np.ndarray     # flat indices of the liver voxels
    nearest: np.ndarray       # (len(liverFlat),) centerline index
    rule: str = "euclidean"
    distance: np.ndarray = None   # (len(liverFlat),) mm from the voxel to its supplying point (path length, geodesic)
    limitMM: float = None         # territories only contain liver at most this far from their own branches

    def mask(self, pointSelection):
        """Bool volume of the liver voxels supplied by the selected points (bool (n,) over the centerline). With
        limitMM, only those at most limitMM from the point that supplies them: after a selective injection the
        liver without visible branches is not claimed by the nearest visible ones."""
        selected = np.zeros(self.liverFlat.shape, bool)
        valid = self.nearest >= 0
        if self.limitMM is not None and self.distance is not None:
            valid &= self.distance <= self.limitMM
        selected[valid] = np.asarray(pointSelection, bool)[self.nearest[valid]]
        volume = np.zeros(int(np.prod(self.shape)), bool)
        volume[self.liverFlat[selected]] = True
        return volume.reshape(self.shape)

    def territory(self, tree, tipIndex):
        return self.mask(tree.downstream(tipIndex))


def territoryEuclidean(tree, liver):
    """Nearest-branch rule (Selle et al. 2002): each liver voxel is supplied by its nearest centerline point."""
    liverFlat = np.flatnonzero(np.asarray(liver, bool).ravel())
    coordinates = np.column_stack(np.unravel_index(liverFlat, tree.shape)) * np.asarray(tree.spacing)
    distance, nearest = cKDTree(tree.positionsMM()).query(coordinates, k=1)
    return TerritoryMap(tuple(tree.shape), liverFlat, nearest.astype(np.int64), "euclidean",
                        distance.astype(np.float32))


def territoryGeodesic(tree, liver, stepMM=GEODESIC_STEP_MM):
    """Minimum-cost-path rule: each liver voxel is supplied by the centerline point with the shortest path inside
    the liver (fissures and gaps between lobes are not crossed). Computed on a grid of about stepMM; liver voxels
    that no path reaches fall back to the nearest-branch rule."""
    liver = np.asarray(liver, bool)
    spacing = np.asarray(tree.spacing, dtype=float)
    factor = np.maximum(1, np.round(stepMM / spacing).astype(int))
    coarse = liver[::factor[0], ::factor[1], ::factor[2]]
    coarseSpacing = spacing * factor
    euclidean = territoryEuclidean(tree, liver)
    voxelIndex = -np.ones(coarse.shape, dtype=np.int64)
    coarseFlat = np.flatnonzero(coarse.ravel())
    voxelIndex.ravel()[coarseFlat] = np.arange(len(coarseFlat))
    if len(coarseFlat) == 0:
        return euclidean
    # sources: centerline points inside the coarse liver (the most downstream point per coarse voxel)
    coarsePoints = tree.points // factor
    inside = np.all(coarsePoints < np.array(coarse.shape), axis=1)
    inside[inside] = coarse[tuple(coarsePoints[inside].T)]
    if not inside.any():
        return euclidean
    pointIDs = np.flatnonzero(inside)
    nodes = voxelIndex[tuple(coarsePoints[pointIDs].T)]
    order = np.lexsort((-tree.distance[pointIDs], nodes))
    nodes, pointIDs = nodes[order], pointIDs[order]
    first = np.concatenate([[True], nodes[1:] != nodes[:-1]])
    sourceNodes, sourcePoints = nodes[first], pointIDs[first]
    # 6-connected graph of the coarse liver voxels
    rows, cols, weights = [], [], []
    for axis in range(3):
        lower = [slice(None)] * 3
        upper = [slice(None)] * 3
        lower[axis] = slice(0, -1)
        upper[axis] = slice(1, None)
        a = voxelIndex[tuple(lower)]
        b = voxelIndex[tuple(upper)]
        both = (a >= 0) & (b >= 0)
        rows.append(a[both])
        cols.append(b[both])
        weights.append(np.full(int(both.sum()), coarseSpacing[axis]))
    graph = sparse.coo_matrix((np.concatenate(weights), (np.concatenate(rows), np.concatenate(cols))),
                              shape=(len(coarseFlat),) * 2).tocsr()
    pathLength, _, sources = csgraph.dijkstra(graph, directed=False, indices=sourceNodes, min_only=True,
                                              return_predecessors=True)
    pointOfNode = -np.ones(len(coarseFlat), dtype=np.int64)
    reached = sources >= 0
    lookup = dict(zip(sourceNodes.tolist(), sourcePoints.tolist()))
    pointOfNode[reached] = [lookup[int(s)] for s in sources[reached]]
    # full-resolution liver voxels take the point of their coarse voxel
    full = np.column_stack(np.unravel_index(euclidean.liverFlat, tree.shape)) // factor
    full = np.minimum(full, np.array(coarse.shape) - 1)
    node = voxelIndex[tuple(full.T)]
    nearest = euclidean.nearest.copy()
    distance = euclidean.distance.copy()
    usable = node >= 0
    usable[usable] = pointOfNode[node[usable]] >= 0
    nearest[usable] = pointOfNode[node[usable]]
    distance[usable] = pathLength[node[usable]]
    return TerritoryMap(tuple(tree.shape), euclidean.liverFlat, nearest, "geodesic", distance)


def subtractNested(territories, tipIndices, tree):
    """Nested tips on the same branch (e.g. lobar and segmental): the upstream territory loses the downstream one.
    territories: list of bool volumes in the order of tipIndices. Returns new masks."""
    result = [mask.copy() for mask in territories]
    for a, tipA in enumerate(tipIndices):
        for b, tipB in enumerate(tipIndices):
            if a != b and tipA is not None and tipB is not None and tipA != tipB and tree.isAncestor(tipA, tipB):
                result[a] &= ~territories[b]
    return result


# -- Safety flags ------------------------------------------------------------------------------------------------

def _subtreeHas(tree, flags):
    """Bool (n,): the point or a point downstream of it has the flag."""
    has = np.asarray(flags, bool).copy()
    order = np.argsort(-tree.distance)
    for node in order:
        up = tree.parent[node]
        if up >= 0 and has[node]:
            has[up] = True
    return has


def extrahepaticBranches(tree, liver, tipIndex, thresholdMM=EXTRAHEPATIC_MM):
    """Branches downstream of the tip that leave the liver by more than thresholdMM and do not come back
    (possible non-target vessels: gastric, falciform, cystic). Returns [{start, deepest, distanceMM, lengthMM}]
    with centerline indices, deepest = the point farthest from the liver."""
    liver = np.asarray(liver, bool)
    outside = ndimage.distance_transform_edt(~liver, sampling=tree.spacing)[tuple(tree.points.T)]
    inLiver = outside <= 0
    reachesLiver = _subtreeHas(tree, inLiver)
    downstream = tree.downstream(tipIndex)
    candidate = downstream & ~reachesLiver & (outside > 0)
    flagged = candidate & (outside > thresholdMM)
    if not flagged.any():
        return []
    # one finding per branch: the first candidate point of each run (its parent is not a candidate)
    starts = [n for n in np.flatnonzero(candidate) if tree.parent[n] < 0 or not candidate[tree.parent[n]]]
    findings = []
    for start in starts:
        members = tree.downstream(start) & candidate
        if not (members & flagged).any():
            continue
        memberIDs = np.flatnonzero(members)
        deepest = int(memberIDs[np.argmax(outside[memberIDs])])
        child = memberIDs[memberIDs != start]
        length = float(np.sum(tree.distance[child] - tree.distance[tree.parent[child]]))
        findings.append(dict(start=int(start), deepest=deepest, distanceMM=float(outside[deepest]),
                             lengthMM=length))
    return sorted(findings, key=lambda f: -f["distanceMM"])


# -- Feeder finder -----------------------------------------------------------------------------------------------

def lowestCommonAncestor(tree, indices):
    indices = [int(i) for i in indices]
    if not indices:
        return None
    for node in tree.pathToRoot(indices[0]):
        if all(tree.isAncestor(node, other) for other in indices[1:]):
            return node
    return tree.root


def tumourFeeders(tree, tumour, marginMM=FEEDER_MARGIN_MM):
    """Feeding branches of a tumour: centerline points inside the tumour dilated by marginMM. Returns
    {"points": indices inside, "entries": first point of each path entering the tumour, "ancestor": most selective
    common ancestor of all of them, "branches": [(branch ancestor, entries)] one per branch leaving the ancestor}
    or None when no branch reaches the tumour."""
    zone = dilateMM(tumour, marginMM, tree.spacing)
    inside = zone[tuple(tree.points.T)]
    if not inside.any():
        return None
    pointIDs = np.flatnonzero(inside)
    entries = [int(n) for n in pointIDs if tree.parent[n] < 0 or not inside[tree.parent[n]]]
    ancestor = lowestCommonAncestor(tree, entries)
    groups = {}
    for entry in entries:
        path = tree.pathToRoot(entry)
        position = path.index(ancestor) if ancestor in path else len(path) - 1
        branch = path[position - 1] if position > 0 else entry
        groups.setdefault(branch, []).append(entry)
    branches = [(lowestCommonAncestor(tree, members), members) for members in groups.values()]
    return dict(points=pointIDs, entries=entries, ancestor=ancestor, branches=branches)


def candidatePositions(tree, territoryMap, tumour, liverMask, voxelML, others=(), feeders=None,
                       marginMM=FEEDER_MARGIN_MM, minStep=CANDIDATE_MIN_STEP):
    """Candidate catheter positions for a tumour, most selective first: the ancestor of each feeding branch, the
    common ancestor of all of them, then the branch points upstream of it, each placed midway along the unbranched
    segment leading to it (not at the bifurcation), whose territory is at least minStep (10 %)
    larger than that of the previous position shown (small side branches change almost nothing). Each candidate: {index, kind,
    territoryML, coveragePercent, normalML, upstreamMM}. others: other tumour masks (not counted as normal
    liver)."""
    if feeders is None:
        feeders = tumourFeeders(tree, tumour, marginMM)
    if feeders is None:
        return []
    tumour = np.asarray(tumour, bool)
    tumourVoxels = int(np.count_nonzero(tumour))
    allTumours = tumour.copy()
    for other in others:
        allTumours |= other
    branchPoints = set(tree.branchPoints().tolist())
    ancestor = feeders["ancestor"]
    candidates, seen = [], set()

    children = np.bincount(tree.parent[tree.parent >= 0], minlength=tree.count)

    def midway(node):
        """Catheters are not placed at a bifurcation: the middle of the unbranched segment leading to the node (from
        the previous bifurcation): the same branches downstream, plus the liver next to that segment."""
        path = [int(node)]
        while True:
            up = int(tree.parent[path[-1]])
            if up < 0 or up == tree.root or children[up] >= 2:
                break
            path.append(up)
        middle = tree.distance[path[0]] - (tree.distance[path[0]] - tree.distance[path[-1]]) / 2.0
        return min(path, key=lambda n: abs(tree.distance[n] - middle))

    def add(index, kind):
        if index is None:
            return
        original = index
        index = midway(index)
        if index in seen:
            return
        seen.add(index)
        territory = territoryMap.territory(tree, index)
        covered = int(np.count_nonzero(territory & tumour))
        normal = int(np.count_nonzero(territory & liverMask & ~allTumours))
        candidates.append(dict(index=int(index), branchPoint=int(original), kind=kind, territoryML=float(np.count_nonzero(territory) * voxelML),
                               coveragePercent=100.0 * covered / tumourVoxels if tumourVoxels else 0.0,
                               normalML=normal * voxelML, upstreamMM=float(tree.distance[index])))

    if len(feeders["branches"]) > 1:
        for branchAncestor, _ in feeders["branches"]:
            add(branchAncestor, "branch")
    add(ancestor, "all feeders")
    reference = candidates[-1]["territoryML"] if candidates else 0.0
    for node in tree.pathToRoot(ancestor)[1:]:
        if node in branchPoints:
            add(node, "upstream")
            if candidates and candidates[-1]["kind"] == "upstream" and candidates[-1].get("branchPoint") == node:
                if candidates[-1]["territoryML"] < reference * (1.0 + minStep):
                    candidates.pop()
                else:
                    reference = candidates[-1]["territoryML"]
    return candidates


# -- Enhancement territory -------------------------------------------------------------------------------------

def otsuThreshold(values, bins=256):
    values = np.asarray(values, dtype=float).ravel()
    if values.size == 0:
        raise ValueError("No values.")
    histogram, edges = np.histogram(values, bins=bins)
    centres = (edges[:-1] + edges[1:]) / 2.0
    weight = np.cumsum(histogram)
    total = weight[-1]
    mean = np.cumsum(histogram * centres)
    with np.errstate(divide="ignore", invalid="ignore"):
        meanLow = mean / weight
        meanHigh = (mean[-1] - mean) / (total - weight)
        between = weight * (total - weight) * (meanLow - meanHigh) ** 2
    between = np.nan_to_num(between)
    return float(centres[int(np.argmax(between))])


def enhancementRange(values, liver, fov=None):
    """(unenhanced, enhanced) liver grey values: 10th and 99th percentile inside the liver (and the FOV)."""
    region = np.asarray(liver, bool) if fov is None else (np.asarray(liver, bool) & fov)
    inside = np.asarray(values)[region]
    if inside.size == 0:
        raise ValueError("No liver voxels inside the CBCT field of view.")
    return float(np.percentile(inside, 10.0)), float(np.percentile(inside, 99.0))


def thresholdFromPercent(values, liver, percent, fov=None):
    low, high = enhancementRange(values, liver, fov)
    return low + percent / 100.0 * (high - low)


def autoEnhancementPercent(values, liver, fov=None):
    """Otsu threshold of the liver grey values, as % between the unenhanced and the enhanced liver."""
    region = np.asarray(liver, bool) if fov is None else (np.asarray(liver, bool) & fov)
    low, high = enhancementRange(values, liver, fov)
    threshold = otsuThreshold(np.clip(np.asarray(values)[region], low, high))
    return float(np.clip(100.0 * (threshold - low) / (high - low), 0.0, 100.0)) if high > low else 50.0


def enhancementTerritory(values, liver, threshold, voxelML, fov=None, minML=ENHANCEMENT_MIN_ML, vessels=None):
    """Perfused volume from contrast enhancement on a selective or parenchymal CBCT: liver voxels above the
    threshold (median-filtered against noise), holes filled, components of at least minML. vessels (optional):
    large vessels are not parenchyma and are removed."""
    liver = np.asarray(liver, bool)
    region = liver if fov is None else (liver & fov)
    smoothed = ndimage.median_filter(np.asarray(values, dtype=np.float32), size=3)
    mask = region & (smoothed >= threshold)
    if vessels is not None:
        mask &= ~vessels
    mask = ndimage.binary_opening(mask, structure=ndimage.generate_binary_structure(3, 1))
    mask = ndimage.binary_fill_holes(mask) & region
    labels, count = ndimage.label(mask)
    if count == 0:
        return mask
    sizes = np.bincount(labels.ravel())
    minVoxels = max(1, int(round(minML / voxelML)))
    keep = sizes >= minVoxels
    keep[0] = False
    return keep[labels]


# -- Field of view and comparison ------------------------------------------------------------------------------

def fieldOfViewMask(array, minFillFraction=0.5):
    """Voxels inside the reconstructed CBCT field of view. Outside the reconstruction cylinder (and the cone ends)
    scanners write one constant fill value; it is the dominant value on the border of the volume, and the fill
    region is the part of it connected to the border. Without a dominant border value the whole volume is valid."""
    array = np.asarray(array)
    border = np.concatenate([array[0].ravel(), array[-1].ravel(), array[:, 0].ravel(), array[:, -1].ravel(),
                             array[:, :, 0].ravel(), array[:, :, -1].ravel()])
    values, counts = np.unique(border, return_counts=True)
    fill = values[int(np.argmax(counts))]
    if counts.max() < minFillFraction * border.size:
        return np.ones(array.shape, bool)
    outside = array == fill
    labels, count = ndimage.label(outside)
    if count == 0:
        return np.ones(array.shape, bool)
    edge = np.zeros(array.shape, bool)
    edge[0] = edge[-1] = True
    edge[:, 0] = edge[:, -1] = True
    edge[:, :, 0] = edge[:, :, -1] = True
    touching = np.unique(labels[edge & outside])
    keep = np.zeros(count + 1, bool)
    keep[touching[touching > 0]] = True
    return ~keep[labels]


def coverageFraction(mask, fov):
    total = int(np.count_nonzero(mask))
    return float(np.count_nonzero(np.asarray(mask, bool) & fov)) / total if total else 0.0


def dice(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    total = int(np.count_nonzero(a)) + int(np.count_nonzero(b))
    return 2.0 * int(np.count_nonzero(a & b)) / total if total else 0.0


def countsInside(values, liver, mask):
    """Fraction of the (positive) counts of the liver that lie inside the mask (e.g. MAA counts in territories)."""
    values = np.clip(np.asarray(values, dtype=float), 0.0, None)
    liver = np.asarray(liver, bool)
    total = float(values[liver].sum())
    return float(values[liver & np.asarray(mask, bool)].sum()) / total if total > 0 else 0.0


# -- C-arm angulation of a 3D view -------------------------------------------------------------------------------

def carmAngles(viewDirectionRAS):
    """(primary, secondary) C-arm angles in degrees for a 3D view, as on an angiography system (DICOM Positioner
    Primary / Secondary Angle): primary > 0 LAO, < 0 RAO; secondary > 0 cranial, < 0 caudal. viewDirectionRAS: from
    the patient towards the detector (the 3D camera), in RAS. A view from anterior (AP) is 0 / 0."""
    d = np.asarray(viewDirectionRAS, dtype=float)
    norm = float(np.linalg.norm(d))
    if norm == 0:
        return 0.0, 0.0
    d = d / norm
    primary = float(np.degrees(np.arctan2(-d[0], d[1])))    # towards the patient's left (-R) is LAO
    secondary = float(np.degrees(np.arcsin(np.clip(d[2], -1.0, 1.0))))
    return primary, secondary


def carmLabel(primary, secondary):
    side = "LAO" if primary > 0.5 else ("RAO" if primary < -0.5 else "AP")
    tilt = "CRA" if secondary > 0.5 else ("CAU" if secondary < -0.5 else "")
    first = f"{side} {abs(primary):.0f}°" if side != "AP" else "AP 0°"
    return first + (f"  {tilt} {abs(secondary):.0f}°" if tilt else "  CRA/CAU 0°")


# -- Liver reached by the imaged tree (selective injections) ------------------------------------------

SUPPLY_DISTANCE_MM = 15.0             # territory limit: liver at most this far from its own (downstream) branches
SUPPLY_TREE_MARGIN_MM = 3.0           # centerline points used for the distance: inside the liver dilated by this


def treeDistance(tree, liver):
    """Distance (mm) of every liver voxel to the nearest centerline point lying in (or just at) the liver; float32
    volume, +inf outside the liver."""
    liver = np.asarray(liver, bool)
    spacing = np.asarray(tree.spacing, dtype=float)
    near = dilateMM(liver, SUPPLY_TREE_MARGIN_MM, tree.spacing)[tuple(tree.points.T)]
    points = tree.points[near] if near.any() else tree.points
    distance = np.full(liver.shape, np.inf, dtype=np.float32)
    voxels = np.argwhere(liver)
    if len(voxels):
        distance[liver] = cKDTree(points * spacing).query(voxels * spacing, k=1)[0].astype(np.float32)
    return distance


def suggestSupplyDistance(distance, liver, default=SUPPLY_DISTANCE_MM, low=8.0, high=40.0):
    """A supply distance from the gap in the liver's distances to the tree: a lobe the injection did not reach is
    far from every branch, so the distances form a near and a far group (Otsu). default when there is no far
    group (the whole liver is close to the tree)."""
    values = np.asarray(distance)[np.asarray(liver, bool)]
    values = values[np.isfinite(values)]
    if values.size == 0 or np.percentile(values, 95) <= high * 0.5:
        return float(default)
    threshold = otsuThreshold(np.clip(values, 0.0, 100.0))
    return float(np.clip(threshold, low, high))
