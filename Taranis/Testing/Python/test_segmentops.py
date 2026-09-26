"""Unit tests of TaranisLib.segmentops (outside Slicer: python -m unittest discover -s Taranis/Testing/Python)."""

import datetime
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from TaranisLib import segmentops as S  # noqa: E402


def ball(shape, centre, radius):
    k, j, i = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    return (k - centre[0]) ** 2 + (j - centre[1]) ** 2 + (i - centre[2]) ** 2 <= radius ** 2


class SegmentOpsTest(unittest.TestCase):

    def setUp(self):
        self.shape = (40, 40, 40)
        self.liver = ball(self.shape, (20, 20, 20), 15)
        self.tumourA = ball(self.shape, (15, 15, 15), 4)
        self.tumourB = ball(self.shape, (25, 25, 25), 3)

    def test_split(self):
        parts = S.splitComponents(self.tumourA | self.tumourB)
        self.assertEqual(len(parts), 2)
        self.assertGreater(parts[0].sum(), parts[1].sum())   # largest first
        self.assertEqual(len(S.splitComponents(self.tumourA | self.tumourB, minVoxels=200)), 1)
        self.assertEqual(S.splitComponents(np.zeros(self.shape, bool)), [])

    def test_normalTissue(self):
        normal = S.normalTissue(self.liver, [self.tumourA, self.tumourB])
        self.assertEqual(normal.sum(), self.liver.sum() - self.tumourA.sum() - self.tumourB.sum())
        self.assertFalse((normal & self.tumourA).any())

    def test_clip(self):
        outside = ball(self.shape, (20, 20, 36), 4)   # sticks out of the liver
        inside, removed = S.clipToContainer(outside, self.liver)
        self.assertGreater(removed, 0)
        self.assertFalse((inside & ~self.liver).any())

    def test_perfusedFromUptake(self):
        values = np.zeros(self.shape, np.float32)
        right = self.liver & (np.mgrid[0:40, 0:40, 0:40][2] < 20)
        values[right] = 100.0
        values[self.liver & ~right] = 5.0
        values[20, 10, 10] = 0.0                          # a cold voxel inside the territory (hole)
        parts = S.perfusedFromUptake(values, self.liver, 20.0)
        self.assertEqual(len(parts), 1)
        self.assertTrue(parts[0][20, 10, 10])             # hole filled
        self.assertFalse((parts[0] & ~self.liver).any())
        self.assertFalse(parts[0][20, 20, 30])            # non-perfused liver excluded
        with self.assertRaises(ValueError):
            S.perfusedFromUptake(np.zeros(self.shape), self.liver, 20.0)

    def test_reports(self):
        masks = {"A": self.tumourA, "B": self.tumourA | self.tumourB}
        overlaps = S.overlapReport(masks, 0.001)
        self.assertEqual(len(overlaps), 1)
        self.assertAlmostEqual(overlaps[0][2], self.tumourA.sum() * 0.001)
        outside = S.outsideReport({"C": ball(self.shape, (20, 20, 36), 4)}, self.liver, 1.0)
        self.assertEqual(outside[0][0], "C")
        self.assertTrue(0 < outside[0][2] < 1)

    def test_suvFactor(self):
        injection = datetime.datetime(2026, 3, 1, 9, 0, 0)
        scan = injection + datetime.timedelta(minutes=110)     # one F-18 half-life
        factor = S.suvFactor(70, 350e6, injection, scan, 110 * 60, "START")
        self.assertAlmostEqual(factor, 70000 / 175e6, places=12)
        self.assertAlmostEqual(S.suvFactor(70, 350e6, injection, scan, 110 * 60, "ADMIN"), 70000 / 350e6)
        overnight = S.suvFactor(70, 350e6, datetime.datetime(2026, 3, 1, 23, 30),
                                datetime.datetime(2026, 3, 1, 1, 20), 110 * 60)
        self.assertAlmostEqual(overnight, 70000 / (350e6 * math.exp(-math.log(2) * 110 / 110)), places=12)
        with self.assertRaises(ValueError):
            S.suvFactor(0, 350e6, injection, scan, 6600)



class PerfusionTest(unittest.TestCase):

    def test_perfusionReport(self):
        shape = (20, 20, 20)
        perfused = np.zeros(shape, bool)
        perfused[:, :, :10] = True
        inside = np.zeros(shape, bool)
        inside[5:8, 5:8, 2:5] = True
        partial = np.zeros(shape, bool)
        partial[5:8, 5:8, 8:11] = True          # i = 10 outside: one third
        outside = np.zeros(shape, bool)
        outside[5:8, 5:8, 14:17] = True
        report = {name: rest for name, *rest in S.perfusionReport(
            {"in": inside, "partial": partial, "out": outside, "empty": np.zeros(shape, bool)}, perfused, 0.5)}
        self.assertNotIn("empty", report)
        self.assertEqual(report["in"][2], 0.0)
        self.assertAlmostEqual(report["partial"][2], 1 / 3)
        self.assertAlmostEqual(report["partial"][1], 9 * 0.5)
        self.assertEqual(report["out"][2], 1.0)

    def test_mismatchFraction(self):
        a = np.zeros((10, 10, 10), bool)
        a[2:6, 2:6, 2:6] = True
        self.assertEqual(S.mismatchFraction(a, a), 0.0)
        b = a.copy()
        b[2, 2:6, 2:6] = False                 # 16 of 64 voxels removed
        self.assertAlmostEqual(S.mismatchFraction(b, a), 0.25)
        self.assertEqual(S.mismatchFraction(np.zeros_like(a), np.zeros_like(a)), 0.0)


if __name__ == "__main__":
    unittest.main()
