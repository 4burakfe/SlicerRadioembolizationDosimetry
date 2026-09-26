"""Unit tests of EasyRegLib.functional (run outside Slicer: python -m unittest discover -s easy_reg/Testing/Python)."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from EasyRegLib import functional as F  # noqa: E402


def rotationZ(degrees):
    a = np.radians(degrees)
    m = np.eye(4)
    m[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    return m


def phantom(shape=(40, 60, 60), liverValue=100.0, bodyValue=5.0, background=0.0, centre=(20, 30, 30)):
    k, j, i = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]].astype(float)
    array = np.full(shape, background, dtype=np.float32)
    body = ((i - centre[2]) / 24) ** 2 + ((j - centre[1]) / 17) ** 2 <= 1
    array[body] = bodyValue
    liver = ((i - centre[2] + 8) / 9) ** 2 + ((j - centre[1] + 3) / 7) ** 2 + ((k - centre[0]) / 8) ** 2 <= 1
    array[liver] = liverValue
    return array, body, liver


class LandmarkTest(unittest.TestCase):

    def test_recoversRigidTransform(self):
        truth = rotationZ(12) @ np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])
        truth[:3, 3] = [15.0, -8.0, 30.0]
        moving = np.array([[0, 0, 0], [100, 10, 5], [20, 90, -10], [-40, 30, 60], [10, -50, 25]], dtype=float)
        fixed = (np.c_[moving, np.ones(len(moving))] @ truth.T)[:, :3]
        matrix, rms, residuals = F.rigidFromLandmarks(moving, fixed)
        np.testing.assert_allclose(matrix, truth, atol=1e-6)
        self.assertLess(rms, 1e-6)
        self.assertEqual(len(residuals), 5)

    def test_noiseGivesRms(self):
        moving = np.array([[0, 0, 0], [100, 0, 0], [0, 100, 0], [0, 0, 100]], dtype=float)
        fixed = moving + np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])
        _, rms, _ = F.rigidFromLandmarks(moving, fixed)
        self.assertGreater(rms, 0.5)
        self.assertLess(rms, 1.5)

    def test_errors(self):
        with self.assertRaises(ValueError):
            F.rigidFromLandmarks([[0, 0, 0], [1, 1, 1]], [[0, 0, 0], [1, 1, 1]])
        with self.assertRaises(ValueError):
            F.rigidFromLandmarks([[0, 0, 0], [1, 0, 0], [2, 0, 0]], [[0, 0, 0], [1, 0, 0]])
        with self.assertRaises(ValueError):  # collinear
            F.rigidFromLandmarks([[0, 0, 0], [50, 0, 0], [100, 0, 0]], [[0, 0, 0], [50, 0, 0], [100, 0, 0]])

    def test_rigidChange(self):
        m = rotationZ(30)
        m[:3, 3] = [3, 4, 0]
        translation, angle = F.rigidChange(m)
        self.assertAlmostEqual(translation, 5.0)
        self.assertAlmostEqual(angle, 30.0)
        displacement, angle = F.changeBetween(np.eye(4), F.translationMatrix([0, 0, 7]), [10, 20, 30])
        self.assertAlmostEqual(displacement, 7.0)
        self.assertAlmostEqual(angle, 0.0)


class MaskTest(unittest.TestCase):

    def test_otsu(self):
        values = np.r_[np.random.default_rng(1).normal(10, 2, 5000), np.random.default_rng(2).normal(100, 5, 5000)]
        self.assertTrue(30 < F.otsuThreshold(values) < 80)

    def test_bodyMaskFillsHolesAndDropsNoise(self):
        array, body, liver = phantom()
        array[20, 30, 20:24] = 0.0            # a photopenic hole inside the body
        array[5, 2, 2] = 50.0                 # an isolated hot noise voxel outside the body
        mask = F.bodyMask(array, 2.0)
        self.assertTrue(mask[20, 30, 21])
        self.assertFalse(mask[5, 2, 2])
        self.assertTrue(np.array_equal(mask[10], body[10]))

    def test_bodyMaskErrors(self):
        array, _, _ = phantom()
        with self.assertRaises(ValueError):
            F.bodyMask(array, 1000.0)
        with self.assertRaises(ValueError):
            F.bodyMask(array + 1.0, 0.5)   # everything above threshold: no outline

    def test_functionalThreshold(self):
        array, _, _ = phantom()
        self.assertAlmostEqual(F.functionalOutlineThreshold(array, 3.0), 3.0, places=3)

    def test_dilate(self):
        mask = np.zeros((20, 20, 20), dtype=bool)
        mask[10, 10, 10] = True
        grown = F.dilateMask(mask, (2.0, 1.0, 1.0), 4.0)
        self.assertTrue(grown[10, 10, 14])       # 4 voxels of 1 mm
        self.assertFalse(grown[10, 10, 15])
        self.assertTrue(grown[12, 10, 10])       # 2 voxels of 2 mm
        self.assertFalse(grown[13, 10, 10])


class OutlineAlignmentTest(unittest.TestCase):

    def test_offsetRecovered(self):
        ijkToWorld = np.diag([3.0, 3.0, 3.0, 1.0])
        array, _, _ = phantom()
        fixedMask = F.bodyMask(array, 2.0)
        moved = np.roll(np.roll(array, 4, axis=2), -3, axis=1)   # 12 mm in x, -9 mm in y
        movingMask = F.bodyMask(moved, 2.0)
        offset, overlap = F.outlineOffset(F.maskWorldPoints(movingMask, ijkToWorld),
                                          F.maskWorldPoints(fixedMask, ijkToWorld))
        np.testing.assert_allclose(offset, [-12.0, 9.0, 0.0], atol=0.5)
        self.assertGreater(overlap, 100)

    def test_outlineAreaRatio(self):
        ijkToWorld = np.diag([3.0, 3.0, 3.0, 1.0])
        array, body, liver = phantom()
        self.assertAlmostEqual(F.outlineAreaRatio(body, ijkToWorld, body, ijkToWorld), 1.0)
        ratio = F.outlineAreaRatio(liver, ijkToWorld, body, ijkToWorld)   # uptake only: much smaller
        self.assertLess(ratio, F.OUTLINE_AREA_RATIO)
        shifted = ijkToWorld.copy()
        shifted[2, 3] = 1000.0                                              # no head-feet overlap
        self.assertAlmostEqual(F.outlineAreaRatio(body, shifted, body, ijkToWorld), 1.0)

    def test_hotRegionAndCentroid(self):
        ijkToWorld = np.diag([2.0, 2.0, 2.0, 1.0])
        array, body, liver = phantom()
        hot = F.hotRegionMask(array, F.bodyMask(array, 2.0))
        self.assertTrue(np.array_equal(hot, liver))
        centroid = F.weightedCentroid(array, hot, ijkToWorld)
        np.testing.assert_allclose(centroid, [(30 - 8) * 2, (30 - 3) * 2, 20 * 2], atol=0.5)


if __name__ == "__main__":
    unittest.main()
