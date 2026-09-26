"""Unit test of TaranisLib.segtools.geometryCheck with a fake segment grid (outside Slicer)."""

import os
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import types  # noqa: E402

for name in ("vtk", "slicer", "qt", "ctk"):
    sys.modules.setdefault(name, mock.MagicMock(name=name))   # segtools imports Slicer's modules
if "slicer.util" not in sys.modules:
    util = types.ModuleType("slicer.util")
    util.VTKObservationMixin = type("VTKObservationMixin", (), {})
    sys.modules["slicer.util"] = util

from TaranisLib import segtools  # noqa: E402
from TaranisLib import workflow as W  # noqa: E402

SHAPE = (30, 30, 30)


def box(k, j, i):
    mask = np.zeros(SHAPE, bool)
    mask[k[0]:k[1], j[0]:j[1], i[0]:i[1]] = True
    return mask


class FakeGrid:
    voxelML = 1.0

    def __init__(self, segments):
        self.segments = segments   # {id: (name, role, mask)}

    def ids(self, role=None):
        return [i for i, (_, r, _) in self.segments.items() if role is None or r == role]

    def name(self, segmentID):
        return self.segments[segmentID][0]

    def mask(self, segmentID):
        return self.segments[segmentID][2]

    def union(self, segmentIDs):
        result = np.zeros(SHAPE, bool)
        for segmentID in segmentIDs:
            result |= self.mask(segmentID)
        return result


class EditableGrid(FakeGrid):
    """FakeGrid with the writing side of SegmentGrid (tags, write, add, liver)."""
    node = None

    def __init__(self, segments):
        super().__init__(segments)
        self.tags = {}

    def tag(self, segmentID, tag):
        return self.tags.get((segmentID, tag), "")

    def setTag(self, segmentID, tag, value):
        self.tags[(segmentID, tag)] = value

    def write(self, segmentID, mask):
        name, role, _ = self.segments[segmentID]
        self.segments[segmentID] = (name, role, mask.astype(bool))

    def add(self, name, role, mask, candidate=False):
        segmentID = f"S{len(self.segments)}"
        self.segments[segmentID] = (name, role, mask.astype(bool))
        return segmentID

    def liver(self):
        livers = self.ids(W.SEGMENT_LIVER)
        return livers[0], self.mask(livers[0])


class NormalTissueToolsTest(unittest.TestCase):

    def setUp(self):
        self.liver = box((0, 30), (0, 30), (0, 30))
        self.tumour = box((5, 10), (5, 10), (5, 10))
        patcher = mock.patch.object(segtools, "uniqueName", lambda node, base: base)
        patcher.start()
        self.addCleanup(patcher.stop)

    def normals(self, grid):
        return {grid.name(i): i for i in grid.ids(W.SEGMENT_NORMAL)}

    def test_perfused_normal_one_per_perfused_volume(self):
        grid = EditableGrid({"L": ("Whole liver", W.SEGMENT_LIVER, self.liver),
                             "P1": ("Right", W.SEGMENT_PERFUSED, box((0, 30), (0, 30), (0, 15))),
                             "P2": ("Seg 4", W.SEGMENT_PERFUSED, box((0, 30), (0, 30), (20, 32))),
                             "T": ("Tumor 1", W.SEGMENT_TUMOR, self.tumour)})
        message = segtools.makePerfusedNormal(grid)
        normals = self.normals(grid)
        self.assertEqual(set(normals), {"Perfused normal liver (Right)", "Perfused normal liver (Seg 4)"})
        right = normals["Perfused normal liver (Right)"]
        self.assertEqual(grid.mask(right).sum(), 30 * 30 * 15 - 125)          # minus the tumour
        self.assertEqual(grid.mask(normals["Perfused normal liver (Seg 4)"]).sum(), 30 * 30 * 10)   # inside the liver
        self.assertEqual(grid.tag(right, W.NORMAL_SCOPE_TAG), W.NORMAL_SCOPE_PERFUSED)
        self.assertEqual(grid.tag(right, W.NORMAL_SOURCE_TAG), "P1")
        self.assertIn("created", message)
        # again: updated, not duplicated; the whole normal liver is separate
        self.assertIn("updated", segtools.makePerfusedNormal(grid))
        segtools.makeNormalLiver(grid)
        normals = self.normals(grid)
        self.assertEqual(len(normals), 3)
        self.assertEqual(grid.mask(normals["Normal liver"]).sum(), 27000 - 125)
        self.assertEqual(grid.tag(normals["Normal liver"], W.NORMAL_SCOPE_TAG), W.NORMAL_SCOPE_WHOLE)
        self.assertIn("updated", segtools.makeNormalLiver(grid))
        self.assertEqual(len(self.normals(grid)), 3)
        # up to date: no 'differs' note in the geometry check
        self.assertFalse(any("differs" in text for _, text in segtools.geometryCheck(grid, "relative")))

    def test_perfused_normal_needs_a_perfused_volume(self):
        grid = EditableGrid({"L": ("Whole liver", W.SEGMENT_LIVER, self.liver)})
        with self.assertRaises(ValueError):
            segtools.makePerfusedNormal(grid)

    def test_hand_made_perfused_normal_is_reused(self):
        grid = EditableGrid({"L": ("Whole liver", W.SEGMENT_LIVER, self.liver),
                             "P1": ("Perfused", W.SEGMENT_PERFUSED, box((0, 30), (0, 30), (0, 15))),
                             "N": ("perfused normal", W.SEGMENT_NORMAL, box((0, 2), (0, 2), (0, 2)))})
        segtools.makePerfusedNormal(grid)
        self.assertEqual(list(self.normals(grid)), ["perfused normal"])
        self.assertEqual(grid.mask("N").sum(), 30 * 30 * 15)
        # the whole normal liver does not overwrite it
        segtools.makeNormalLiver(grid)
        self.assertEqual(len(self.normals(grid)), 2)


class TightGridTest(unittest.TestCase):
    """segtools.gridExtent: the tight grid of the hub tools covers only the given bounds (plus a margin)."""

    def setUp(self):
        matrix = np.diag([2.0, 2.0, 3.0, 1.0])            # 2 x 2 x 3 mm voxels, origin 0
        patcher = mock.patch.object(segtools, "_matrix", lambda volume: matrix)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.reference = mock.MagicMock()
        self.reference.GetImageData.return_value.GetDimensions.return_value = (100, 100, 50)

    def test_extended_and_tight(self):
        bounds = [19.0, 41.0, 9.0, 31.0, 28.5, 61.5]       # voxel corners of i 10-20, j 5-15, k 10-20
        lower, dims = segtools.gridExtent(self.reference, [bounds])
        self.assertEqual((list(lower), list(dims)), ([0, 0, 0], [100, 100, 50]))   # whole reference image
        lower, dims = segtools.gridExtent(self.reference, [bounds], includeReference=False)
        self.assertEqual((list(lower), list(dims)), ([10, 5, 10], [11, 11, 11]))
        lower, dims = segtools.gridExtent(self.reference, [bounds], includeReference=False, marginVoxels=2)
        self.assertEqual((list(lower), list(dims)), ([8, 3, 8], [15, 15, 15]))
        # beyond the reference image: extended, never cropped
        lower, dims = segtools.gridExtent(self.reference, [[-11.0, 1.0, 9.0, 31.0, 28.5, 61.5]],
                                          includeReference=False)
        self.assertEqual(lower[0], -5)
        # no bounds: the whole reference image
        lower, dims = segtools.gridExtent(self.reference, [], includeReference=False)
        self.assertEqual(list(dims), [100, 100, 50])


class GeometryCheckTest(unittest.TestCase):

    def setUp(self):
        self.liver = box((0, 30), (0, 30), (0, 30))
        self.perfused = box((0, 30), (0, 30), (0, 15))

    def check(self, segments, mode="relative"):
        return [(severity, text) for severity, text in segtools.geometryCheck(FakeGrid(segments), mode)]

    def test_tumours_and_perfused_volumes(self):
        segments = {"L": ("Whole liver", W.SEGMENT_LIVER, self.liver),
                    "P": ("Perfused 1", W.SEGMENT_PERFUSED, self.perfused),
                    "T1": ("Tumor 1", W.SEGMENT_TUMOR, box((5, 10), (5, 10), (5, 10))),        # perfused
                    "T2": ("Tumor 2", W.SEGMENT_TUMOR, box((5, 10), (5, 10), (12, 17))),       # 40 % outside
                    "T3": ("Tumor 3", W.SEGMENT_VIABLE, box((20, 25), (20, 25), (20, 25)))}    # not perfused
        issues = self.check(segments)
        texts = " | ".join(text for _, text in issues)
        self.assertNotIn("'Tumor 1'", texts)
        self.assertIn("40% of 'Tumor 2'", texts)
        self.assertIn("gets 0 Gy in patient-relative", texts)
        self.assertIn("'Tumor 3' does not intersect any perfused volume (0 Gy", texts)
        self.assertTrue(all(severity == W.SEVERITY_WARNING for severity, text in issues if "perfused" in text))
        # absolute mode: same findings without the patient-relative 0 Gy remark
        texts = " | ".join(text for _, text in self.check(segments, "absolute"))
        self.assertIn("'Tumor 3' does not intersect any perfused volume:", texts)
        self.assertNotIn("0 Gy", texts)
        # no perfused volume: nothing to compare
        del segments["P"]
        self.assertFalse(any("perfused" in text for _, text in self.check(segments)))

    def test_small_unperfused_fraction_is_accepted(self):
        segments = {"L": ("Whole liver", W.SEGMENT_LIVER, self.liver),
                    "P": ("Perfused 1", W.SEGMENT_PERFUSED, self.perfused),
                    "T": ("Tumor 1", W.SEGMENT_TUMOR, box((0, 10), (0, 10), (5, 16)))}    # 1 of 11 columns: 9 %
        self.assertFalse(any("Tumor 1" in text for _, text in self.check(segments)))

    def test_tumour_burden_and_outdated_normal_tissue(self):
        tumour = box((0, 30), (0, 20), (0, 30))                     # 67 % of the liver
        normal = self.liver & ~box((0, 30), (0, 10), (0, 30))       # made for a smaller tumour
        segments = {"L": ("Whole liver", W.SEGMENT_LIVER, self.liver),
                    "T": ("Tumor 1", W.SEGMENT_TUMOR, tumour),
                    "N": ("Normal liver", W.SEGMENT_NORMAL, normal)}
        issues = self.check(segments)
        self.assertTrue(any(s == W.SEVERITY_WARNING and "Tumour burden 67%" in t for s, t in issues))
        self.assertTrue(any(s == W.SEVERITY_INFO and "'Normal liver' differs" in t for s, t in issues))
        # up to date normal tissue (liver - tumours): no note
        segments["N"] = ("Normal liver", W.SEGMENT_NORMAL, self.liver & ~tumour)
        self.assertFalse(any("differs" in t for _, t in self.check(segments)))


if __name__ == "__main__":
    unittest.main()
