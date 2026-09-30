"""Voxel S value (dose point kernel) dosimetry - EXPERIMENTAL.

The local deposition method (LDM) puts the whole decay energy of a voxel into that voxel. The voxel S value method
spreads it to the neighbouring voxels as the beta particles do: the LDM dose map is convolved with a 3D kernel K,
where K[offset] is the fraction of the energy emitted in a voxel that is absorbed in the voxel at that offset.

The kernel is built for the image's own voxel size (anisotropic voxels are fine, the image is never resampled) from a
radionuclide dose point kernel in water, by Monte Carlo sampling:
  1. a random decay position, uniform inside the source voxel,
  2. a distance drawn from the energy deposited in each spherical shell of the dose point kernel (uniform in volume
     inside the shell),
  3. an isotropic random direction,
  4. the voxel the energy lands in is counted.
The counts are averaged over the 8 mirror images of the kernel (the physics is symmetric, this halves the noise),
truncated at KERNEL_RADIUS_MM and normalised to 1: the total energy equals the LDM energy, so the conversion factor,
the decay correction and the patient-relative scaling stay exactly as they are for LDM. A fixed random seed makes
every calculation with the same voxel size give the same kernel.

Pure numpy (no Slicer), so it can be unit-tested outside Slicer.
"""

import os
import re

import numpy as np

from . import doseguard as DG

# -- Methods ----------------------------------------------------------------------------------------------------------

METHOD_LDM = "ldm"
METHOD_VOXEL_S = "voxelS"
METHODS = (METHOD_LDM, METHOD_VOXEL_S)
METHOD_LABELS = {METHOD_LDM: "Local deposition (LDM)",
                 METHOD_VOXEL_S: "Voxel S value / dose point kernel (experimental)"}
METHOD_SHORT_LABELS = {METHOD_LDM: "LDM", METHOD_VOXEL_S: "Voxel S (experimental)"}

# -- Radionuclides with a kernel ----------------------------------------------------------------------------------------

NUCLIDE_Y90 = "Y-90"
KERNEL_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Resources", "DoseKernels")
# label, physics locked while the voxel S method is used (the Y-90 defaults of the modules), kernel file and emission
NUCLIDES = {
    NUCLIDE_Y90: {
        "label": "Y-90",
        "conversionFactor": DG.Y90_CONVERSION_FACTOR,   # J/GBq
        "halfLifeH": DG.Y90_HALF_LIFE_H,
        "kernelFile": "90Y_64.10H_beta.io_processed.csv",
        "emission": "beta emission of the Y-90 ground state (T1/2 64.1 h)",
    },
}
DEFAULT_NUCLIDE = NUCLIDE_Y90

KERNEL_REFERENCE = ("Graves SA, Flynn RT, Hyer DE. Dose point kernels for 2,174 radionuclides. Med Phys. "
                    "2019;46(11):5284-5293. doi:10.1002/mp.13789. PMCID: PMC7685392.")
KERNEL_REFERENCE_SHORT = "Graves et al., Med Phys 2019 (doi:10.1002/mp.13789)"
# Density scaling / density correction of dose kernels and the lung in Y-90 radioembolization
DENSITY_REFERENCES = [
    "Woo MK, et al. The validity of the density scaling method in primary electron transport for photon and electron "
    "beams. Med Phys. 1990;17(2):187.",
    "Dieudonne A, et al. Study of the impact of tissue density heterogeneities on 3-dimensional abdominal dosimetry: "
    "comparison between dose kernel convolution and direct Monte Carlo methods. J Nucl Med. 2013;54(2):236.",
    "Mikell JK, et al. Comparing voxel-based absorbed dosimetry methods in tumors, liver, lung, and at the liver-lung "
    "interface for 90Y microsphere selective internal radiation therapy. EJNMMI Phys. 2015;2:16.",
    "Tiwari A, et al. The impact of tissue type and density on dose point kernels for patient-specific voxel-wise "
    "dosimetry: a Monte Carlo investigation. Radiat Res. 2020;193(6):531.",
]

# -- Kernel construction ------------------------------------------------------------------------------------------------

KERNEL_SAMPLES = 4_000_000     # simulated decays per kernel: largest weight error about 1e-4 (0.01 % of the energy)
KERNEL_SEED = 5284             # fixed: the same voxel size always gives the same kernel (reproducible reports)
KERNEL_RADIUS_MM = 11.0        # Y-90 beta: 99.6 % of the energy within 11 mm, the rest is a bremsstrahlung tail
KERNEL_BATCH = 500_000         # samples per batch (memory)


def kernelPath(nuclide=DEFAULT_NUCLIDE):
    if nuclide not in NUCLIDES:
        raise ValueError(f"No dose point kernel for '{nuclide}'. Available: {', '.join(NUCLIDES)}.")
    return os.path.join(KERNEL_FOLDER, NUCLIDES[nuclide]["kernelFile"])


def loadKernelShells(path, maxRadiusMM=KERNEL_RADIUS_MM):
    """Spherical shells of a dose point kernel file (Graves et al. format):
        line 1: '<nuclide>,<emission> per decay = <yield>'
        line 2: column names; then rows: outer radius of the shell (cm), energy deposited per emission (MeV), ...
    Returns {"rInnerMM", "rOuterMM", "energyMeV" (per decay, shells up to maxRadiusMM), "totalEnergyMeV" (per decay,
    whole file), "yieldPerDecay", "name"}."""
    with open(path, "r", encoding="utf-8-sig") as f:
        first = f.readline().strip()
        f.readline()  # column names
        rows = [line.split(",") for line in f if line.strip()]
    name = first.split(",")[0].strip()
    match = re.search(r"=\s*([0-9.eE+-]+)\s*$", first)
    if not match:
        raise ValueError(f"Unexpected first line in the kernel file {os.path.basename(path)}: '{first}'.")
    yieldPerDecay = float(match.group(1))
    data = np.array([[float(r[0]), float(r[1])] for r in rows], dtype=np.float64)
    rOuterMM = data[:, 0] * 10.0
    energy = data[:, 1] * yieldPerDecay
    if rOuterMM.size == 0 or np.any(np.diff(rOuterMM) <= 0) or rOuterMM[0] <= 0 or np.any(energy < 0):
        raise ValueError(f"The kernel file {os.path.basename(path)} has invalid shells.")
    rInnerMM = np.concatenate([[0.0], rOuterMM[:-1]])
    keep = rOuterMM <= maxRadiusMM + 1e-9
    return {"rInnerMM": rInnerMM[keep], "rOuterMM": rOuterMM[keep], "energyMeV": energy[keep],
            "totalEnergyMeV": float(energy.sum()), "yieldPerDecay": yieldPerDecay, "name": name}


def buildVoxelKernel(spacingZYX, shells, samples=KERNEL_SAMPLES, seed=KERNEL_SEED, batch=KERNEL_BATCH):
    """Energy-fraction kernel on a numpy (z, y, x) voxel grid (sums to 1) and a dict describing it.

    spacingZYX: voxel size in mm in numpy axis order, i.e. reversed vtkMRMLVolumeNode.GetSpacing()."""
    spacing = np.asarray(spacingZYX, dtype=np.float64)
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError(f"Invalid voxel spacing {tuple(spacingZYX)}.")
    energy = np.asarray(shells["energyMeV"], dtype=np.float64)
    if energy.sum() <= 0:
        raise ValueError("The dose point kernel has no energy.")
    cdf = np.cumsum(energy) / energy.sum()
    rInner, rOuter = np.asarray(shells["rInnerMM"]), np.asarray(shells["rOuterMM"])
    radius = float(rOuter[-1])
    half = np.ceil(radius / spacing + 0.5).astype(int)   # a decay at the voxel edge reaches radius + half a voxel
    counts = np.zeros(tuple(2 * half + 1), dtype=np.float64)
    rng = np.random.default_rng(seed)
    done = 0
    while done < samples:
        n = min(batch, samples - done)
        start = (rng.random((n, 3)) - 0.5) * spacing                       # 1. decay position in the voxel
        shell = np.minimum(np.searchsorted(cdf, rng.random(n)), cdf.size - 1)  # 2. shell (by energy)
        r = np.cbrt(rInner[shell] ** 3 + rng.random(n) * (rOuter[shell] ** 3 - rInner[shell] ** 3))
        direction = rng.normal(size=(n, 3))                                  # 3. isotropic direction
        direction /= np.linalg.norm(direction, axis=1)[:, None]
        index = np.rint((start + direction * r[:, None]) / spacing).astype(np.int64) + half   # 4. target voxel
        flat = np.ravel_multi_index(index.T, counts.shape)
        counts += np.bincount(flat, minlength=counts.size).reshape(counts.shape)
        done += n
    symmetric = sum(np.flip(counts, axis=axes) if axes else counts
                    for axes in [(), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]) / 8.0
    kernel = symmetric / symmetric.sum()
    centre = tuple(half)
    info = {
        "samples": int(samples), "seed": int(seed), "radiusMM": radius, "spacingZYX": tuple(float(s) for s in spacing),
        "shape": tuple(int(s) for s in kernel.shape), "selfFraction": float(kernel[centre]),
        "energyWithinRadius": float(energy.sum() / shells["totalEnergyMeV"]) if shells.get("totalEnergyMeV") else 1.0,
    }
    return kernel, info


def kernelForSpacing(spacingXYZ, nuclide=DEFAULT_NUCLIDE, densityGPerML=1.0, samples=KERNEL_SAMPLES,
                     seed=KERNEL_SEED):
    """Kernel for a volume node's GetSpacing() (x, y, z) in a medium of the given density: the spacing is reversed to
    the numpy (z, y, x) axis order of slicer.util.arrayFromVolume. Built anew at every call (about 1 s).

    Density scaling (same composition as water, different density rho): all distances scale as 1/rho (the ranges are
    the same in g/cm2), so the water point kernel becomes D_rho(r) = rho^2 * D_water(rho * r) (one rho from the
    shorter distances, one from the mass). For the energy fractions of the voxel kernel this is exactly the water
    kernel of voxels rho times larger: the kernel is built for spacing x rho; the energy absorbed in a voxel is then
    divided by that voxel's own mass (rho x volume) when the dose is calculated."""
    if not (densityGPerML > 0 and np.isfinite(densityGPerML)):
        raise ValueError(f"Invalid tissue density {densityGPerML} g/mL.")
    shells = loadKernelShells(kernelPath(nuclide))
    scaledZYX = tuple(float(s) * densityGPerML for s in spacingXYZ)[::-1]
    kernel, info = buildVoxelKernel(scaledZYX, shells, samples=samples, seed=seed)
    info["nuclide"] = nuclide
    info["densityGPerML"] = float(densityGPerML)
    info["waterSpacingZYX"] = info["spacingZYX"]   # the voxel size in water-equivalent mm (spacing x density)
    info["spacingZYX"] = tuple(float(s) for s in spacingXYZ)[::-1]
    info["spacingXYZ"] = tuple(float(s) for s in spacingXYZ)
    info["waterRadiusMM"] = info["radiusMM"]
    info["radiusMM"] = info["radiusMM"] / densityGPerML   # physical range in this tissue
    return kernel, info


def _fastLength(n):
    """Smallest 2-3-5-smooth integer >= n (fast FFT length)."""
    m = max(int(n), 1)
    while True:
        rest = m
        for prime in (2, 3, 5):
            while rest % prime == 0:
                rest //= prime
        if rest == 1:
            return m
        m += 1


def convolveSame(values, kernel):
    """values convolved with an odd-sized kernel, same shape as values (zero outside the image), by FFT (numpy only).
    FFT round-off (|x| below 1e-12 of the largest value) is set to exactly 0."""
    values = np.asarray(values, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    if values.ndim != kernel.ndim or any(k % 2 == 0 for k in kernel.shape):
        raise ValueError("The kernel must be odd-sized and have the dimensions of the image.")
    full = [v + k - 1 for v, k in zip(values.shape, kernel.shape)]
    fftShape = [_fastLength(n) for n in full]
    axes = list(range(values.ndim))
    spectrum = np.fft.rfftn(values, fftShape, axes=axes)
    spectrum *= np.fft.rfftn(kernel, fftShape, axes=axes)
    result = np.fft.irfftn(spectrum, fftShape, axes=axes)
    del spectrum
    start = [(k - 1) // 2 for k in kernel.shape]
    result = np.ascontiguousarray(result[tuple(slice(s, s + v) for s, v in zip(start, values.shape))])
    largest = float(np.max(np.abs(values))) if values.size else 0.0
    if largest > 0:
        result[np.abs(result) < 1e-12 * largest] = 0.0
    return result


def voxelSDoseMap(ldmDose, spacingXYZ, nuclide=DEFAULT_NUCLIDE, densityGPerML=1.0, localMask=None,
                  samples=KERNEL_SAMPLES, seed=KERNEL_SEED):
    """(voxel S dose map, kernel info) from an LDM dose map (numpy z, y, x) and the volume's GetSpacing(). The kernel
    is scaled to densityGPerML, the density the LDM map was calculated with (energy / (density x volume)).

    localMask: voxels kept at local deposition, e.g. the lungs in absolute mode (their own density already applied in
    ldmDose): their decays are not spread and their dose is the LDM dose. Energy of the other decays that would
    reach them is left out (a few mm at the interface)."""
    ldmDose = np.asarray(ldmDose, dtype=np.float64)
    kernel, info = kernelForSpacing(spacingXYZ, nuclide, densityGPerML, samples=samples, seed=seed)
    if localMask is None or not np.any(localMask):
        return convolveSame(ldmDose, kernel), info
    if localMask.shape != ldmDose.shape:
        raise ValueError("The local deposition mask does not have the shape of the dose map.")
    dose = convolveSame(np.where(localMask, 0.0, ldmDose), kernel)
    dose[localMask] = ldmDose[localMask]
    info["localDepositionVoxels"] = int(np.count_nonzero(localMask))
    return dose, info


def applyDoseMethod(ldmDose, spacingXYZ, method, nuclide=DEFAULT_NUCLIDE, densityGPerML=1.0, localMask=None):
    """(dose map, kernel info or None): the LDM map itself for LDM, voxelSDoseMap for voxel S. The input map is never
    modified."""
    if method == METHOD_VOXEL_S:
        return voxelSDoseMap(ldmDose, spacingXYZ, nuclide, densityGPerML, localMask)
    if method != METHOD_LDM:
        raise ValueError(f"Unknown dose calculation method '{method}'.")
    return ldmDose, None


def energyOutsideFraction(doseArray, mask):
    """Part of the (positive) dose sum outside mask: the energy fraction when all voxels have the same density."""
    positive = np.clip(np.asarray(doseArray, dtype=np.float64), 0, None)
    total = float(positive.sum(dtype=np.float64))
    if total <= 0:
        return float("nan")
    return float(positive[~mask].sum(dtype=np.float64)) / total


def meanInMask(doseArray, mask):
    return float(np.mean(doseArray[mask], dtype=np.float64)) if np.any(mask) else float("nan")


# -- Texts: module, report ------------------------------------------------------------------------------------------------

LDM_DESCRIPTION = (
    "Local deposition method (LDM): all the decay energy of a voxel is absorbed in that same voxel, "
    "dose = activity concentration x conversion factor / density. Standard in clinical practice and the basis of "
    "the published dose thresholds; the beta range (Y-90: mean about 2.5 mm, maximum about 11 mm) is ignored, "
    "which is reasonable because the SPECT/PET blur is usually wider.")


def voxelSDescription(nuclide=DEFAULT_NUCLIDE, samples=KERNEL_SAMPLES, radiusMM=KERNEL_RADIUS_MM):
    label = NUCLIDES.get(nuclide, {}).get("label", nuclide)
    return (
        f"Voxel S value / dose point kernel method (EXPERIMENTAL): the LDM dose map is convolved with a 3D kernel "
        f"that spreads the energy of each voxel to its neighbours, as the {label} beta particles do. The kernel is "
        f"built at every calculation for the image's own voxel size (anisotropic voxels supported, the image is not "
        f"resampled) from the {label} dose point kernel in water: {samples:,} simulated decays at random positions "
        f"in one voxel, the distance drawn from the energy deposited in each 0.1 mm shell, a random direction; fixed "
        f"random seed (reproducible), mirror-averaged, truncated at {radiusMM:g} mm (in water) and normalised to 1 "
        f"(total energy as in LDM). Density: electron ranges are inversely proportional to the density, so the kernel "
        f"is scaled to the liver density (built for voxels 'voxel size x density' large) and the energy absorbed in "
        f"a voxel is divided by its mass. Lungs (absolute mode, lung segments): local deposition with the lung "
        f"density, as in LDM. The conversion factor and the half-life are locked to the {label} values.")


VOXEL_S_WARNINGS = [
    "Experimental: not validated for clinical use. Published dose thresholds were mostly derived with LDM or the "
    "partition model: compare with an LDM calculation.",
    "Lungs are not calculated with the voxel S method: local deposition (LDM) with the lung density in absolute mode "
    "(lung segments), lung dose from the lung shunt in patient-relative mode. A single kernel cannot follow electrons "
    "across the liver-lung interface, and lung uptake on post-treatment images is usually diffuse and dominated by "
    "noise, breathing and spill-over from the liver dome.",
    "Patient-relative mode: energy spreads up to about 11 mm beyond the perfused volumes, so their mean doses are "
    "lower than with LDM / the partition model, most in small (e.g. segmental) volumes.",
    "Voxel S does not correct the SPECT/PET blur (partial volume effect), which is usually wider than the beta range.",
]


def _html(text):
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def methodDescriptionHtml(method, nuclide=DEFAULT_NUCLIDE):
    """Rich text shown under the method selector of the dosimetry modules."""
    if method != METHOD_VOXEL_S:
        return _html(LDM_DESCRIPTION)
    items = "".join(f"<li>{_html(w)}</li>" for w in VOXEL_S_WARNINGS)
    density = "".join(f"<br>&bull; {_html(r)}" for r in DENSITY_REFERENCES)
    return (f"{_html(voxelSDescription(nuclide))}"
            f"<ul style='margin-top:4px; margin-bottom:4px; color:#d97706;'>{items}</ul>"
            f"<i>Kernel data: {_html(KERNEL_REFERENCE)}<br>Density scaling and correction:{density}</i>")


def reportParameters(method, nuclide=DEFAULT_NUCLIDE, info=None):
    """(label, value) rows of the report's parameter table."""
    rows = [("Dose calculation method", METHOD_LABELS.get(method, method))]
    if method == METHOD_VOXEL_S:
        label = NUCLIDES.get(nuclide, {}).get("label", nuclide)
        rows.append(("Dose point kernel", f"{label} beta, water ({KERNEL_REFERENCE_SHORT})"))
        if info:
            rows.append(("Voxel S kernel", _kernelText(info)))
            if info.get("localDepositionVoxels"):
                rows.append(("Lungs", "local deposition with the lung density (voxel S not applied)"))
    return rows


def _kernelText(info):
    z, y, x = info["shape"]
    sx, sy, sz = info.get("spacingXYZ", info["spacingZYX"][::-1])
    return (f"{x} x {y} x {z} voxels (x, y, z) for {sx:.2f} x {sy:.2f} x {sz:.2f} mm voxels, density "
            f"{info.get('densityGPerML', 1.0):.2f} g/mL, radius {info['radiusMM']:.1f} mm "
            f"({100 * info['energyWithinRadius']:.1f}% of the energy, normalised to 1), {info['samples']:,} Monte Carlo "
            f"samples, seed {info['seed']}, {100 * info['selfFraction']:.1f}% of the energy kept in the source voxel")


def reportLines(method, nuclide=DEFAULT_NUCLIDE):
    """Text lines of the report's 'Dose calculation method' section."""
    if method != METHOD_VOXEL_S:
        return [LDM_DESCRIPTION]
    return ([voxelSDescription(nuclide)] + [f"Caution: {w}" for w in VOXEL_S_WARNINGS]
            + [f"Kernel data: {KERNEL_REFERENCE}"]
            + [f"Density scaling and correction: {r}" for r in DENSITY_REFERENCES])
