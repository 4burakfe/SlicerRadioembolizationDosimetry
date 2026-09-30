"""Unit tests of TaranisLib.dosekernel (experimental voxel S / dose point kernel dosimetry) and of its dose checks."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from TaranisLib import dosekernel as DK  # noqa: E402
from TaranisLib import doseguard as DG  # noqa: E402
from TaranisLib import workflow as W  # noqa: E402

FAST = 400_000   # samples for tests that do not check the noise level


def directConvolveSame(values, kernel):
    """Reference 'same' convolution by explicit shifts (zero outside the image)."""
    out = np.zeros_like(values, dtype=np.float64)
    half = [(k - 1) // 2 for k in kernel.shape]
    padded = np.pad(values, [(h, h) for h in half])
    for index in np.ndindex(kernel.shape):
        # out[i] += values[i - (index - half)] * kernel[index]
        shifted = [slice(2 * h - j, 2 * h - j + n) for j, h, n in zip(index, half, values.shape)]
        out += kernel[index] * padded[tuple(shifted)]
    return out


class KernelDataTest(unittest.TestCase):

    def test_y90_kernel_file(self):
        shells = DK.loadKernelShells(DK.kernelPath(DK.NUCLIDE_Y90))
        self.assertEqual(shells["name"], "90Y_64.10H")
        self.assertEqual(shells["yieldPerDecay"], 1.0)
        self.assertAlmostEqual(shells["totalEnergyMeV"], 0.9336, places=3)   # mean Y-90 beta energy
        self.assertEqual(len(shells["energyMeV"]), 110)                      # 0.1 mm shells up to 11 mm
        self.assertTrue(np.allclose(shells["rOuterMM"][:3] - shells["rInnerMM"][:3], 0.1))
        within = shells["energyMeV"].sum() / shells["totalEnergyMeV"]
        self.assertGreater(within, 0.995)
        self.assertLess(within, 0.997)

    def test_energy_matches_ldm_conversion_factor(self):
        """1 GBq fully decayed with the kernel's mean energy is the LDM conversion factor (within 0.5 %)."""
        shells = DK.loadKernelShells(DK.kernelPath())
        decays = 1e9 * 64.1 * 3600 / np.log(2)
        joules = decays * shells["totalEnergyMeV"] * 1.602176634e-13
        self.assertLess(abs(joules - DG.Y90_CONVERSION_FACTOR) / DG.Y90_CONVERSION_FACTOR, 0.005)

    def test_unknown_nuclide(self):
        with self.assertRaises(ValueError):
            DK.kernelPath("Ho-166")


class KernelTest(unittest.TestCase):

    def test_isotropic_kernel(self):
        kernel, info = DK.kernelForSpacing((4.42, 4.42, 4.42))   # default: 4 million samples
        self.assertEqual(kernel.shape, (7, 7, 7))
        self.assertAlmostEqual(kernel.sum(), 1.0, places=12)
        self.assertTrue(np.all(kernel >= 0))
        c = (3, 3, 3)
        self.assertAlmostEqual(info["selfFraction"], kernel[c])
        self.assertAlmostEqual(kernel[c], 0.40, delta=0.01)
        neighbours = [kernel[4, 3, 3], kernel[3, 4, 3], kernel[3, 3, 4]]
        self.assertAlmostEqual(max(neighbours), min(neighbours), delta=5e-4)   # isotropic within the MC noise
        for axes in [(0,), (1,), (2,)]:
            self.assertTrue(np.array_equal(kernel, np.flip(kernel, axis=axes)))   # mirror-symmetric
        self.assertEqual(info["samples"], DK.KERNEL_SAMPLES)

    def test_anisotropic_axis_order(self):
        """GetSpacing() is (x, y, z); numpy arrays are (z, y, x). Thin slices (z = 3 mm) -> the kernel is longest in
        numpy axis 0 and the z neighbour gets more energy than the x neighbour."""
        kernel, info = DK.kernelForSpacing((4.0, 4.0, 3.0), samples=FAST)
        self.assertEqual(kernel.shape, (11, 9, 9))
        self.assertEqual(info["spacingZYX"], (3.0, 4.0, 4.0))
        c = tuple(s // 2 for s in kernel.shape)
        zNeighbour = kernel[c[0] + 1, c[1], c[2]]
        xNeighbour = kernel[c[0], c[1], c[2] + 1]
        self.assertAlmostEqual(zNeighbour, 0.068, delta=0.003)
        self.assertAlmostEqual(xNeighbour, 0.049, delta=0.003)
        self.assertAlmostEqual(kernel[c], 0.337, delta=0.005)

    def test_reproducible(self):
        a, _ = DK.kernelForSpacing((4.0, 4.0, 4.0), samples=FAST)
        b, _ = DK.kernelForSpacing((4.0, 4.0, 4.0), samples=FAST)
        self.assertTrue(np.array_equal(a, b))
        c, _ = DK.kernelForSpacing((4.0, 4.0, 4.0), samples=FAST, seed=1)
        self.assertFalse(np.array_equal(a, c))

    def test_smaller_voxels_keep_less_energy(self):
        small, _ = DK.kernelForSpacing((2.0, 2.0, 2.0), samples=FAST)
        large, _ = DK.kernelForSpacing((4.42, 4.42, 4.42), samples=FAST)
        self.assertLess(small[tuple(s // 2 for s in small.shape)], large[3, 3, 3])

    def test_invalid_spacing(self):
        shells = DK.loadKernelShells(DK.kernelPath())
        for spacing in [(0.0, 4.0, 4.0), (4.0, -1.0, 4.0), (4.0, 4.0)]:
            with self.assertRaises(ValueError):
                DK.buildVoxelKernel(spacing, shells, samples=1000)


class DensityScalingTest(unittest.TestCase):
    """Same composition, density rho: distances scale as 1/rho (ranges equal in g/cm2)."""

    def test_point_kernel_scales_with_rho_squared(self):
        """Shells rho times closer, same energy, mass of a shell rho^-2 times the water shell's mass:
        D_rho(r / rho) = rho^2 * D_water(r) (the dose point kernel scaling theorem)."""
        shells = DK.loadKernelShells(DK.kernelPath())
        volume = 4.0 / 3.0 * np.pi * (shells["rOuterMM"] ** 3 - shells["rInnerMM"] ** 3) / 1000.0   # mL
        doseWater = shells["energyMeV"] / (1.0 * volume)
        for rho in (0.3, 1.05, 1.9):
            scaledVolume = volume / rho ** 3                     # radii / rho
            doseRho = shells["energyMeV"] / (rho * scaledVolume)
            self.assertTrue(np.allclose(doseRho, rho ** 2 * doseWater))

    def test_kernel_equals_water_kernel_of_larger_voxels(self):
        """kernelForSpacing(spacing, density) is the water kernel for spacing x density (exactly), and the same as
        moving every shell 1/density closer with the real spacing (same random numbers)."""
        shells = DK.loadKernelShells(DK.kernelPath())
        for rho in (0.3, 1.05):
            kernel, info = DK.kernelForSpacing((4.0, 4.0, 3.0), densityGPerML=rho, samples=FAST)
            water, _ = DK.buildVoxelKernel((3.0 * rho, 4.0 * rho, 4.0 * rho), shells, samples=FAST)
            self.assertTrue(np.array_equal(kernel, water))
            closer = dict(shells, rInnerMM=shells["rInnerMM"] / rho, rOuterMM=shells["rOuterMM"] / rho)
            moved, _ = DK.buildVoxelKernel((3.0, 4.0, 4.0), closer, samples=FAST)
            self.assertEqual(moved.shape, kernel.shape)
            self.assertTrue(np.allclose(moved, kernel, atol=2e-5))   # only float rounding at voxel borders differs
            self.assertAlmostEqual(info["radiusMM"], DK.KERNEL_RADIUS_MM / rho)
            self.assertEqual(info["spacingXYZ"], (4.0, 4.0, 3.0))

    def test_lower_density_wider_kernel(self):
        liver, liverInfo = DK.kernelForSpacing((4.42,) * 3, densityGPerML=1.05, samples=FAST)
        low, lowInfo = DK.kernelForSpacing((4.42,) * 3, densityGPerML=0.3, samples=FAST)
        self.assertEqual(low.shape, (19, 19, 19))                    # about 37 mm reach at 0.3 g/mL
        self.assertAlmostEqual(liverInfo["selfFraction"], 0.416, delta=0.01)
        self.assertLess(lowInfo["selfFraction"], liverInfo["selfFraction"])
        with self.assertRaises(ValueError):
            DK.kernelForSpacing((4.0, 4.0, 4.0), densityGPerML=0.0)

    def test_lungs_local_deposition(self):
        """Absolute mode with lungs: lung voxels keep the LDM dose (lung density), lung decays are not spread, the
        rest is convolved with the liver kernel."""
        ldm = np.zeros((50, 30, 30))
        ldm[5:45, 5:25, 5:25] = 100.0 / 1.05
        lungs = np.zeros(ldm.shape, bool)
        lungs[25:] = True
        ldm[lungs] *= 1.05 / 0.3                              # the caller applies the lung density (as for LDM)
        before = ldm.copy()
        dose, info = DK.voxelSDoseMap(ldm, (4.0, 4.0, 4.0), densityGPerML=1.05, localMask=lungs, samples=FAST)
        self.assertTrue(np.array_equal(dose[lungs], ldm[lungs]))
        self.assertAlmostEqual(dose[12, 15, 15], 100.0 / 1.05, places=6)   # liver, > 11 mm from the edges
        self.assertLess(dose[24, 15, 15], 100.0 / 1.05)                  # nothing comes back from the lungs
        self.assertEqual(info["localDepositionVoxels"], int(lungs.sum()))
        self.assertTrue(np.array_equal(ldm, before))                     # input unchanged
        rows = dict(DK.reportParameters(DK.METHOD_VOXEL_S, DK.NUCLIDE_Y90, info))
        self.assertIn("local deposition", rows["Lungs"])
        self.assertIn("density 1.05 g/mL", rows["Voxel S kernel"])
        same, _ = DK.applyDoseMethod(ldm, (4.0,) * 3, DK.METHOD_LDM, densityGPerML=1.05, localMask=lungs)
        self.assertIs(same, ldm)
        with self.assertRaises(ValueError):
            DK.voxelSDoseMap(ldm, (4.0,) * 3, localMask=np.ones((2, 2, 2), bool), samples=1000)


class ConvolutionTest(unittest.TestCase):

    def test_matches_direct_convolution(self):
        rng = np.random.default_rng(3)
        values = rng.uniform(0, 10, (9, 12, 7))
        kernel = rng.uniform(0, 1, (5, 3, 7))
        self.assertTrue(np.allclose(DK.convolveSame(values, kernel), directConvolveSame(values, kernel), atol=1e-9))

    def test_single_hot_voxel_reproduces_kernel(self):
        kernel, _ = DK.kernelForSpacing((4.0, 4.0, 3.0), samples=FAST)
        values = np.zeros((21, 21, 21))
        values[10, 10, 10] = 1000.0
        dose = DK.convolveSame(values, kernel)
        hz, hy, hx = (s // 2 for s in kernel.shape)
        self.assertTrue(np.allclose(dose[10 - hz:11 + hz, 10 - hy:11 + hy, 10 - hx:11 + hx], 1000.0 * kernel))
        self.assertAlmostEqual(dose.sum(), 1000.0, places=6)
        self.assertEqual(np.count_nonzero(dose), np.count_nonzero(kernel))   # FFT round-off removed

    def test_uniform_activity_equals_ldm(self):
        ldm = np.zeros((40, 40, 40))
        ldm[5:35, 5:35, 5:35] = 100.0
        dose, info = DK.voxelSDoseMap(ldm, (4.0, 4.0, 3.0), samples=FAST)
        self.assertAlmostEqual(dose[20, 20, 20], 100.0, places=6)      # interior: same as LDM
        self.assertAlmostEqual(dose.sum(), ldm.sum(), delta=1e-6 * ldm.sum())   # energy conserved (not at edges)
        self.assertLess(dose[5, 20, 20], 100.0)                        # edge voxels lose energy outwards
        self.assertGreater(dose[4, 20, 20], 0.0)                       # ... to the voxels outside

    def test_ldm_method_is_unchanged(self):
        ldm = np.ones((3, 3, 3))
        dose, info = DK.applyDoseMethod(ldm, (4.0, 4.0, 4.0), DK.METHOD_LDM)
        self.assertIs(dose, ldm)
        self.assertIsNone(info)
        with self.assertRaises(ValueError):
            DK.applyDoseMethod(ldm, (4.0, 4.0, 4.0), "monteCarlo")

    def test_fast_lengths(self):
        self.assertEqual([DK._fastLength(n) for n in (1, 7, 11, 13, 49, 97)], [1, 8, 12, 15, 50, 100])

    def test_energy_outside_mask(self):
        """Patient-relative voxel S: a 300 mL sphere of 4.42 mm voxels loses about 6 % of its energy outwards."""
        spacing = 4.42
        radius = (3 * 300000 / (4 * np.pi)) ** (1 / 3)
        n = int(2 * (radius + 20) / spacing) + 2
        z, y, x = (np.indices((n, n, n)) + 0.5 - n / 2) * spacing
        mask = x ** 2 + y ** 2 + z ** 2 <= radius ** 2
        dose, _ = DK.voxelSDoseMap(mask * 100.0, (spacing,) * 3, samples=FAST)
        outside = DK.energyOutsideFraction(dose, mask)
        self.assertAlmostEqual(outside, 0.059, delta=0.005)
        self.assertAlmostEqual(DK.meanInMask(dose, mask), 100.0 * (1 - outside), places=6)
        self.assertTrue(np.isnan(DK.energyOutsideFraction(np.zeros((2, 2, 2)), np.ones((2, 2, 2), bool))))


class TextsTest(unittest.TestCase):

    def test_report_parameters(self):
        self.assertEqual(DK.reportParameters(DK.METHOD_LDM), [("Dose calculation method", "Local deposition (LDM)")])
        _, info = DK.kernelForSpacing((4.0, 4.0, 3.0), samples=FAST)
        rows = dict(DK.reportParameters(DK.METHOD_VOXEL_S, DK.NUCLIDE_Y90, info))
        self.assertIn("experimental", rows["Dose calculation method"])
        self.assertIn("Y-90", rows["Dose point kernel"])
        self.assertIn("9 x 9 x 11 voxels (x, y, z) for 4.00 x 4.00 x 3.00 mm", rows["Voxel S kernel"])
        self.assertIn("400,000 Monte Carlo samples", rows["Voxel S kernel"])

    def test_descriptions(self):
        lines = DK.reportLines(DK.METHOD_VOXEL_S)
        self.assertIn("4,000,000", lines[0])
        self.assertTrue(any("10.1002/mp.13789" in line for line in lines))
        self.assertEqual(sum(line.startswith("Density scaling and correction:") for line in lines), 4)
        self.assertIn("Mikell", DK.methodDescriptionHtml(DK.METHOD_VOXEL_S))
        self.assertEqual(DK.reportLines(DK.METHOD_LDM), [DK.LDM_DESCRIPTION])
        self.assertIn("EXPERIMENTAL", DK.methodDescriptionHtml(DK.METHOD_VOXEL_S))
        self.assertNotIn("<li>", DK.methodDescriptionHtml(DK.METHOD_LDM))
        for text in DK.reportLines(DK.METHOD_VOXEL_S):
            text.encode("ascii")   # the RTF report escapes non-ASCII, but keep the method texts plain


class VoxelSChecksTest(unittest.TestCase):

    def test_no_checks_for_ldm(self):
        self.assertEqual(DG.doseChecks([], DG.GLASS), [])

    def test_experimental_warning(self):
        issues = DG.doseChecks([], DG.GLASS, voxelS=True)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0][0], W.SEVERITY_WARNING)
        self.assertIn("Experimental voxel S", issues[0][1])

    def test_energy_outside_perfused_volumes(self):
        info = DG.doseChecks([], DG.GLASS, relative=True, voxelS=True, energyOutsidePerfusedFraction=0.059)
        self.assertIn((W.SEVERITY_INFO,), [(s,) for s, t in info if "5.9% of the delivered" in t])
        warn = DG.doseChecks([], DG.GLASS, relative=True, voxelS=True, energyOutsidePerfusedFraction=0.125)
        self.assertTrue(any(s == W.SEVERITY_WARNING and "12.5% of the delivered" in t for s, t in warn))

    def test_lung_note_in_absolute_mode(self):
        issues = DG.doseChecks([], DG.GLASS, lungDosesGy=[("'Lungs'", 5.0)], voxelS=True)
        self.assertTrue(any(s == W.SEVERITY_INFO and "local deposition (LDM) and the lung density" in t
                            for s, t in issues))
        relative = DG.doseChecks([], DG.GLASS, lungDosesGy=[("Estimated lung dose", 5.0)], relative=True,
                                 voxelS=True)
        self.assertFalse(any("lung segments" in t for _, t in relative))   # lungs not calculated


if __name__ == "__main__":
    unittest.main()
