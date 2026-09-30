"""Unit tests of EasyRegLib.sitkreg (outside Slicer: python -m unittest discover -s easy_reg/Testing/Python).

The geometry and preprocessing tests need numpy and scipy only. The registration tests need SimpleITK and are skipped
without it; run them with Slicer's Python, e.g.
    PythonSlicer -m unittest discover -s easy_reg/Testing/Python
"""

import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from EasyRegLib import sitkreg as S  # noqa: E402


def rotation(axis, degrees):
    a = math.radians(degrees)
    c, s = math.cos(a), math.sin(a)
    m = np.eye(4)
    i, j = {"x": (1, 2), "y": (2, 0), "z": (0, 1)}[axis]
    m[i, i], m[i, j], m[j, i], m[j, j] = c, -s, s, c
    return m


def grid(spacing, origin, directions=None):
    m = np.eye(4)
    d = np.eye(3) if directions is None else np.asarray(directions, dtype=float)
    m[:3, :3] = d * np.asarray(spacing, dtype=float)[None, :]
    m[:3, 3] = origin
    return m


def phantomCt(shape=(60, 80, 100), spacing=(4.0, 4.0, 4.0), origin=(-200.0, -160.0, -120.0)):
    """Asymmetric trunk (CT values): body, liver, spine, kidneys, spleen, on a (k, j, i) grid; world RAS."""
    matrix = grid(spacing, origin)
    k, j, i = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]].astype(float)
    x = matrix[0, 3] + i * spacing[0]
    y = matrix[1, 3] + j * spacing[1]
    z = matrix[2, 3] + k * spacing[2]
    array = np.full(shape, -1000.0, dtype=np.float32)
    body = (x / 160) ** 2 + ((y + 10) / 110) ** 2 <= 1
    array[body] = 30.0
    array[body & (((x + 60) / 75) ** 2 + ((y + 10) / 55) ** 2 + ((z - 20) / 70) ** 2 <= 1)] = 110.0   # liver
    array[body & (((x - 80) / 30) ** 2 + ((y - 20) / 25) ** 2 + ((z - 40) / 45) ** 2 <= 1)] = 90.0    # spleen
    array[body & (((x) / 18) ** 2 + ((y - 70) / 18) ** 2 <= 1)] = 700.0                              # spine
    for side in (-1, 1):
        array[body & (((x - side * 55) / 25) ** 2 + ((y - 45) / 22) ** 2 + ((z + 40) / 45) ** 2 <= 1)] = 150.0
    array[body & ((x / 140) ** 2 + ((y + 10) / 95) ** 2 > 1)] = -100.0                               # fat rim
    return array, matrix


def movingFrom(fixedVoxels, fixedMatrix, truth, shape, movingMatrix):
    """Moving image whose native point p shows the fixed image at world truth @ p."""
    return S.resampleOnto(fixedVoxels, fixedMatrix, shape, np.asarray(truth) @ movingMatrix, order=1,
                          fill=-1000.0).astype(np.float32)


class GeometryTest(unittest.TestCase):

    def test_block_mean_keeps_positions(self):
        array = np.zeros((8, 8, 8), dtype=np.float32)
        array[2:4, 4:6, 0:2] = 8.0        # one 2x2x2 block
        matrix = grid((1.0, 2.0, 3.0), (10.0, 20.0, 30.0))
        pooled, pooledMatrix = S.blockMean(array, matrix, (2, 2, 2))
        self.assertEqual(pooled.shape, (4, 4, 4))
        k, j, i = np.argwhere(pooled == pooled.max())[0]
        world = pooledMatrix @ [i, j, k, 1.0]
        original = matrix @ [0.5, 4.5, 2.5, 1.0]  # centre of the block in the original voxels
        np.testing.assert_allclose(world, original)

    def test_resample_linear_function_exactly(self):
        # a linear function of world position is reproduced exactly by linear interpolation, on any grid
        directions = rotation("z", 20.0)[:3, :3] @ np.diag([-1.0, -1.0, 1.0])
        source = grid((0.8, 0.8, 3.0), (5.0, -7.0, 11.0), directions)
        k, j, i = np.mgrid[0:30, 0:60, 0:60].astype(float)
        points = source @ np.vstack([i.ravel(), j.ravel(), k.ravel(), np.ones(i.size)])
        values = (2.0 * points[0] - 1.5 * points[1] + 0.5 * points[2]).reshape(i.shape).astype(np.float32)
        shape, target = S.gridFor(values.shape, source, 2.0)
        resampled = S.resampleOnto(values, source, shape, target, mode="nearest")
        kk, jj, ii = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]].astype(float)
        world = target @ np.vstack([ii.ravel(), jj.ravel(), kk.ravel(), np.ones(ii.size)])
        expected = (2.0 * world[0] - 1.5 * world[1] + 0.5 * world[2]).reshape(ii.shape)
        np.testing.assert_allclose(resampled, expected, atol=1e-3)
        np.testing.assert_allclose(np.linalg.norm(target[:3, :3], axis=0), (2.0, 2.0, 3.0))  # no upsampling

    def test_prepare_image_grid(self):
        voxels, matrix = phantomCt(shape=(40, 60, 70), spacing=(0.8, 0.8, 2.5))
        working = S.prepareImage(S.ImageInput(voxels, matrix, S.KIND_CT), 2.0)
        np.testing.assert_allclose(working.spacing, (2.0, 2.0, 2.5), atol=0.05)
        low, high = S.worldBounds(voxels.shape, matrix)
        workingLow, workingHigh = S.worldBounds(working.shape, working.ijkToWorld)
        self.assertTrue(np.all(workingLow >= low - 1.0) and np.all(workingHigh <= high + 1e-6))
        self.assertTrue(np.all(high - workingHigh < 2.6))   # covers the image up to less than one working voxel
        self.assertGreaterEqual(working.voxels.min(), S.CT_RANGE[0])
        self.assertEqual(working.voxels.dtype, np.float32)

    def test_prepare_mask(self):
        voxels, matrix = phantomCt()
        mask = voxels == 110.0
        working = S.prepareImage(S.ImageInput(voxels, matrix, S.KIND_CT), 8.0)
        resampled = S.prepareMask(mask, matrix, working)
        self.assertEqual(resampled.shape, working.shape)
        voxelML = abs(np.linalg.det(matrix[:3, :3])) / 1000.0
        workingML = abs(np.linalg.det(working.ijkToWorld[:3, :3])) / 1000.0
        self.assertAlmostEqual(resampled.sum() * workingML / (mask.sum() * voxelML), 1.0, delta=0.15)

    def test_intensities(self):
        ct = np.array([[[-3000.0, 0.0, 5000.0, np.nan]]], dtype=np.float32)
        np.testing.assert_allclose(S.normalizeIntensities(ct, S.KIND_CT), [[[-1000.0, 0.0, 2000.0, 0.0]]])
        spect = np.array([[[-5.0, 0.0, 10.0]]], dtype=np.float32)
        self.assertGreaterEqual(S.normalizeIntensities(spect, S.KIND_FUNCTIONAL).min(), 0.0)
        mr = np.linspace(0, 1000, 10001, dtype=np.float32).reshape(1, 1, -1)
        mr[0, 0, -1] = 1e6                                         # one extreme voxel (e.g. a metal artefact)
        self.assertLess(S.normalizeIntensities(mr, S.KIND_ANATOMICAL).max(), 1001.0)
        ctInt = np.array([[[-3000, 0, 5000, 32767]]], dtype=np.int16)
        clipped = S.normalizeIntensities(ctInt, S.KIND_CT)
        self.assertEqual(clipped.dtype, np.int16)                  # no full-resolution float copy of a CT
        np.testing.assert_array_equal(clipped, [[[-1000, 0, 2000, 2000]]])
        self.assertEqual(S.imageKind(ct), S.KIND_CT)
        self.assertEqual(S.imageKind(mr), S.KIND_ANATOMICAL)
        self.assertEqual(S.imageKind(mr, functional=True), S.KIND_FUNCTIONAL)

    def test_pyramid_follows_final_resolution(self):
        shape = (200, 200, 200)
        self.assertEqual(S.pyramid(4.0, shape), [4, 2, 1])            # 16, 8, 4 mm
        self.assertEqual(S.pyramid(3.0, shape), [4, 2, 1])            # 12, 6, 3 mm
        self.assertEqual(S.pyramid(2.0, shape), [8, 4, 2, 1])         # 16 ... 2 mm
        self.assertEqual(S.pyramid(1.0, shape), [16, 8, 4, 2, 1])     # 16 ... 1 mm
        self.assertEqual(S.pyramid(1.0, (40, 200, 200)), [4, 2, 1])   # thin image: no level under 8 voxels
        self.assertEqual(S.smoothingSigmas([4, 2, 1], 2.0), [4.0, 2.0, 0.0])
        percentages = S.samplingPercentages([4, 2, 1], 1_000_000, 50_000)
        self.assertEqual(percentages[0], 1.0)                         # 15 625 voxels: all of them
        self.assertAlmostEqual(percentages[-1], 0.05)

    def test_working_spacing_is_coarsened_for_huge_images(self):
        image = S.ImageInput(np.zeros((400, 512, 512), dtype=np.int16), grid((0.5, 0.5, 0.5), (0, 0, 0)))
        spacing, note = S.workingSpacing([image], 1.0, maxVoxels=5_000_000)
        self.assertGreater(spacing, 1.0)
        self.assertIn("coarsened", note)
        spacing, note = S.workingSpacing([image], 2.0, maxVoxels=5_000_000)
        self.assertEqual((spacing, note), (2.0, ""))

    def test_sitk_matrix_conversions(self):
        truth = S.translationMatrix((12.0, -7.0, 30.0)) @ rotation("z", 9.0) @ rotation("x", -4.0)
        a, o = S.toSitkAffine(truth)
        np.testing.assert_allclose(S.fromSitkAffine(a, o), truth, atol=1e-9)
        centre = np.array([10.0, 20.0, -30.0])
        t = S.centredParameters(a, o, centre)
        np.testing.assert_allclose(a @ (centre + 5.0 - centre) + centre + t, a @ (centre + 5.0) + o)
        # LPS: a moving point p (RAS) is placed at world truth @ p, i.e. the fixed LPS point of it maps back to p
        p = np.array([3.0, 4.0, 5.0, 1.0])
        world = truth @ p
        fixedLps = np.array([-world[0], -world[1], world[2]])
        movingLps = a @ fixedLps + o
        np.testing.assert_allclose(movingLps, [-p[0], -p[1], p[2]], atol=1e-9)

    def test_rigid_part_and_change(self):
        scaled = rotation("y", 10.0) @ np.diag([1.1, 0.95, 1.0, 1.0])
        rigid = S.rigidPart(scaled)
        np.testing.assert_allclose(rigid[:3, :3] @ rigid[:3, :3].T, np.eye(3), atol=1e-9)
        displacement, angle = S.matrixChange(np.eye(4), S.translationMatrix((3.0, 4.0, 0.0)) @ rotation("z", 5.0),
                                             (0.0, 0.0, 0.0))
        self.assertAlmostEqual(displacement, 5.0)
        self.assertAlmostEqual(angle, 5.0)

    def test_start_position_geometry(self):
        fixed = S.Working(np.zeros((100, 10, 10), np.float32), grid((4, 4, 4), (0, 0, -200)))    # S -200 .. 196
        moving = S.Working(np.zeros((30, 10, 10), np.float32), grid((4, 4, 4), (50, 0, 400)))    # S 400 .. 516
        matrix = S.centresMatrix(fixed, moving, np.eye(4))
        np.testing.assert_allclose(S.centre(moving.shape, moving.ijkToWorld, matrix),
                                   S.centre(fixed.shape, fixed.ijkToWorld))
        self.assertEqual(S.headFeetOverlap(fixed, moving, np.eye(4)), 0.0)
        self.assertAlmostEqual(S.headFeetOverlap(fixed, moving, matrix), 1.0)


@unittest.skipUnless(S.sitkAvailable(), "SimpleITK is not available (run with Slicer's Python)")
class RegistrationTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.fixedVoxels, cls.fixedMatrix = phantomCt()
        cls.fixed = S.ImageInput(cls.fixedVoxels, cls.fixedMatrix, S.KIND_CT)

    def _moving(self, truth, shape=(50, 70, 90), spacing=(3.0, 3.5, 4.5), origin=(-180.0, -140.0, -100.0)):
        movingMatrix = grid(spacing, origin)
        voxels = movingFrom(self.fixedVoxels, self.fixedMatrix, truth, shape, movingMatrix)
        return S.ImageInput(voxels, movingMatrix, S.KIND_CT)

    def assertMatrixClose(self, found, truth, mm=2.0, degrees=1.5, point=(-60.0, -10.0, 20.0)):
        displacement, angle = S.matrixChange(truth, found, point)
        self.assertLess(displacement, mm, f"liver centre off by {displacement:.2f} mm")
        self.assertLess(angle, degrees, f"rotation off by {angle:.2f} degrees")

    def test_rigid_recovers_rotation_and_shift(self):
        truth = S.translationMatrix((10.0, -6.0, 12.0)) @ rotation("z", 7.0) @ rotation("x", 3.0)
        result = S.register(self.fixed, self._moving(truth), finalSpacingMm=4.0)
        self.assertTrue(result["refined"], result["notes"])
        self.assertAlmostEqual(result["spacing"], 4.0)
        self.assertEqual(result["levels"], [16.0, 8.0, 4.0])
        self.assertMatrixClose(result["matrix"], truth)

    def test_finer_final_level_is_used(self):
        truth = S.translationMatrix((6.0, 4.0, -8.0)) @ rotation("y", 4.0)
        result = S.register(self.fixed, self._moving(truth), finalSpacingMm=2.0)
        self.assertEqual(result["levels"][-1], 2.0)
        self.assertMatrixClose(result["matrix"], truth, mm=1.5)

    def test_search_finds_far_head_feet_start(self):
        # a short moving scan (upper abdomen), 90 mm off in S: out of reach of a start from the image centres
        truth = S.translationMatrix((5.0, 0.0, 90.0))
        moving = self._moving(truth, shape=(25, 70, 90), origin=(-180.0, -140.0, -110.0))
        result = S.register(self.fixed, moving, initialization=S.INIT_SEARCH, finalSpacingMm=4.0)
        self.assertTrue(any("search" in note for note in result["notes"]), result["notes"])
        self.assertMatrixClose(result["matrix"], truth, mm=3.0)

    def test_mask_restricts_comparison(self):
        truth = S.translationMatrix((4.0, -3.0, 5.0))
        mask = self.fixedVoxels == 110.0                  # liver
        from scipy import ndimage
        mask = ndimage.binary_dilation(mask, iterations=4)
        result = S.register(self.fixed, self._moving(truth), finalSpacingMm=4.0,
                            fixedMask=(mask, self.fixedMatrix))
        self.assertMatrixClose(result["matrix"], truth)
        np.testing.assert_allclose(result["centreRas"], (-60.0, -10.0, 20.0), atol=8.0)   # rotation about the liver

    def test_worse_result_is_rejected(self):
        # a moving image of pure noise: whatever the optimiser finds is not kept unless the metric improves
        rng = np.random.default_rng(1)
        noise = S.ImageInput(rng.normal(0, 100, (40, 60, 70)).astype(np.float32), grid((4, 4, 4), (-140, -120, -80)),
                             S.KIND_ANATOMICAL)
        start = S.translationMatrix((3.0, 2.0, 1.0))
        result = S.register(self.fixed, noise, initialMatrix=start, finalSpacingMm=4.0)
        if not result["refined"]:
            np.testing.assert_allclose(result["matrix"], start)
        self.assertLessEqual(result["finalMetric"] if result["refined"] else result["initialMetric"],
                             result["initialMetric"])

    def test_affine_recovers_scale(self):
        truth = S.translationMatrix((3.0, -2.0, 4.0)) @ np.diag([1.06, 1.0, 0.95, 1.0])
        result = S.register(self.fixed, self._moving(truth), transformType=S.AFFINE, finalSpacingMm=4.0)
        self.assertTrue(result["refined"], result["notes"])
        self.assertMatrixClose(result["matrix"], truth, mm=3.0, degrees=2.0)

    def test_cancel(self):
        truth = S.translationMatrix((10.0, 0.0, 0.0))
        with self.assertRaises(S.RegistrationCancelled):
            S.register(self.fixed, self._moving(truth), finalSpacingMm=4.0, isCancelled=lambda: True)

    def test_deterministic(self):
        truth = S.translationMatrix((7.0, -5.0, 9.0)) @ rotation("z", 3.0)
        moving = self._moving(truth)
        first = S.register(self.fixed, moving, finalSpacingMm=4.0)
        second = S.register(self.fixed, moving, finalSpacingMm=4.0)
        np.testing.assert_allclose(first["matrix"], second["matrix"])


if __name__ == "__main__":
    unittest.main()
