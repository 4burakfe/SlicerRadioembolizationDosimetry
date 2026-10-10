"""Unit tests of TaranisLib.vascular on synthetic vessel trees in an ellipsoid liver (outside Slicer:
python -m unittest discover -s Taranis/Testing/Python; the skeleton test needs scikit-image, e.g. PythonSlicer)."""

import importlib.util
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from TaranisLib import vascular as V  # noqa: E402

SPACING = (1.0, 1.0, 1.0)
SHAPE = (60, 90, 100)


def segmentDistance(shape, a, b):
    """Distance (voxels) of every voxel to the segment a-b (k, j, i)."""
    k, j, i = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
    p = np.stack([k, j, i], axis=-1).astype(float)
    a, b = np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    t = np.clip(((p - a) @ ab) / float(ab @ ab), 0.0, 1.0)
    closest = a + t[..., None] * ab
    return np.sqrt(((p - closest) ** 2).sum(axis=-1))


def drawLine(volume, a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    steps = int(np.ceil(np.abs(b - a).max())) + 1
    for t in np.linspace(0.0, 1.0, steps):
        volume[tuple(np.round(a + t * (b - a)).astype(int))] = True


def ellipsoid(shape, centre, radii):
    k, j, i = np.ogrid[0:shape[0], 0:shape[1], 0:shape[2]]
    return (((k - centre[0]) / radii[0]) ** 2 + ((j - centre[1]) / radii[1]) ** 2
            + ((i - centre[2]) / radii[2]) ** 2) <= 1.0


class Phantom:
    """Y-tree: a trunk from the injection point to a bifurcation, a left and a right branch, each with a short side
    branch; ellipsoid liver around the branches."""
    root = (30, 5, 50)
    fork = (30, 30, 50)
    left = (30, 75, 20)
    right = (30, 75, 80)
    leftSide = ((30, 52, 35), (45, 60, 30))
    rightSide = ((30, 52, 65), (15, 60, 70))
    radius = 2.0

    def __init__(self):
        self.segments = [(self.root, self.fork), (self.fork, self.left), (self.fork, self.right),
                         self.leftSide, self.rightSide]
        self.labels = ["trunk", "left", "right", "left", "right"]   # territory label of each segment
        distances = np.stack([segmentDistance(SHAPE, a, b) for a, b in self.segments])
        self.vessels = distances.min(axis=0) <= self.radius
        self.liver = ellipsoid(SHAPE, (30, 52, 50), (25, 35, 45))
        nearest = distances.argmin(axis=0)
        leftSegments = [n for n, label in enumerate(self.labels) if label == "left"]
        self.leftTruth = np.isin(nearest, leftSegments) & self.liver
        self.skeleton = np.zeros(SHAPE, bool)
        for a, b in self.segments:
            drawLine(self.skeleton, a, b)
        rng = np.random.default_rng(1)
        self.image = np.where(self.liver, 100.0, 20.0) + rng.normal(0, 5, SHAPE)
        self.image[self.vessels] = 600.0 + rng.normal(0, 10, int(self.vessels.sum()))

    def tree(self):
        return V.buildTree(self.skeleton, SPACING, self.root, self.vessels)


@unittest.skipIf(importlib.util.find_spec("skimage") is None or importlib.util.find_spec("SimpleITK") is None,
                 "the tree segment keeps its centerlines through the wall trim (scikit-image) and is denoised with "
                 "SimpleITK: run with Slicer's Python")
class TreeSegmentTest(unittest.TestCase):
    """Search mask and local-contrast hysteresis. The phantom: liver in fat with CBCT-like shading (brighter towards
    one side, as with scatter / cupping), lung above the dome with a bright rim at the interface, a rib beside the
    liver touching a branch through a bright bridge, a metal coil with a streak plate touching a branch."""

    @classmethod
    def setUpClass(cls):
        p = cls.phantom = Phantom()
        rng = np.random.default_rng(5)
        k, j, i = np.meshgrid(*[np.arange(n) for n in SHAPE], indexing="ij")
        cls.shading = 60.0 * i / SHAPE[2]                              # up to 12 SD brighter at one side
        image = np.where(p.liver, 100.0 + cls.shading, 60.0) + rng.normal(0, 5, SHAPE)
        image[p.vessels] = 600.0 + rng.normal(0, 10, int(p.vessels.sum()))
        cls.lung = np.zeros(SHAPE, bool)
        cls.lung[52:] = True
        cls.lung &= ~V.dilateMM(p.liver, 2.0, SPACING)
        image[cls.lung] = -800.0
        image[50:52][~p.liver[50:52]] = 400.0                         # bright rim artefact at the lung-liver interface
        cls.bone = np.zeros(SHAPE, bool)
        cls.bone[15:45, 70:85, 97:100] = True                         # rib beside the liver
        image[cls.bone] = 900.0
        bridge = np.zeros(SHAPE, bool)
        drawLine(bridge, (30, 75, 80), (30, 75, 98))
        cls.bridge = V.dilateMM(bridge, 1.0, SPACING)
        image[cls.bridge] = 600.0
        cls.metal = V.ballMask(SHAPE, (30, 45, 92), 1.5, SPACING)    # coil
        cls.streak = np.zeros(SHAPE, bool)
        cls.streak[24:37, 44:47, 62:90] = True                        # flat streak from the coil to the right branch
        cls.streak &= ~p.vessels
        image[cls.streak] += 25.0                                     # about 3 SD: only grown into, not a seed
        image[cls.metal] = 8000.0
        cls.faint = segmentDistance(SHAPE, (30, 40, 57), (48, 40, 72)) <= 1.0   # thin, faint branch off the right one
        cls.faint &= ~p.vessels
        image[cls.faint] = 100.0 + cls.shading[cls.faint] + 40.0 + rng.normal(0, 5, int(cls.faint.sum()))
        cls.image = image
        cls.statistics = V.liverStatistics(image, p.liver & ~p.vessels)
        cls.region = V.searchRegion(image, p.liver, SPACING, cls.statistics, p.root)
        cls.denoised = V.denoise(image, SPACING)
        cls.contrast = V.localContrast(cls.denoised, p.liver, SPACING)

    def test_searchRegion(self):
        p = self.phantom
        self.assertFalse((self.region & self.lung).any())
        self.assertFalse((self.region & self.bone).any())
        self.assertTrue(self.region[tuple(np.array(p.root))])        # injection point (outside the liver)
        self.assertTrue(self.region[p.liver & ~V.dilateMM(self.lung, 6.0, SPACING)].all())
        self.assertIsNone(V.airThreshold(np.where(p.liver, 100.0, 60.0), self.statistics))   # fat only: no air

    def test_localContrastRemovesShading(self):
        p = self.phantom
        parenchyma = p.liver & ~V.dilateMM(p.vessels | self.metal | self.streak, 4.0, SPACING)
        z = self.contrast.z[parenchyma]
        self.assertLess(abs(float(np.median(z))), 0.5)
        lateral = parenchyma.copy()
        lateral[:, :, :75] = False
        medial = parenchyma.copy()
        medial[:, :, 25:] = False
        self.assertLess(abs(float(np.median(self.contrast.z[lateral]) - np.median(self.contrast.z[medial]))), 0.5)
        # the background follows the shading, measured on the liver only (also at the edge)
        error = self.contrast.background[parenchyma] - (100.0 + self.shading[parenchyma])
        self.assertLess(float(np.percentile(np.abs(error), 95)), 4.0)
        self.assertGreater(float(np.median(self.contrast.z[p.vessels & p.liver])), 20.0)
        # robust global statistics: the vessels inside the liver do not change them much
        stats = V.liverStatistics(self.image, p.liver)
        self.assertLess(stats.sd, 20.0)

    def _tree(self, tube=None, region=None):
        p = self.phantom
        return V.arterialTree(self.denoised, self.contrast, self.region if region is None else region, p.root,
                              SPACING, p.liver, tube=tube)

    def test_arterialTree(self):
        p = self.phantom
        segment = self._tree()
        tree = segment.mask
        self.assertGreater(V.dice(tree & ~self.streak & ~self.faint, p.vessels & self.region), 0.9)
        self.assertGreater(segment.metalML, 0.0)
        for name, structure in (("bone", self.bone), ("lung", self.lung), ("metal", self.metal)):
            self.assertFalse((tree & structure).any(), name)
        self.assertFalse(tree[50:52][~p.liver[50:52]].any())       # the interface rim is outside the search mask
        shaded = p.liver & ~V.dilateMM(p.vessels | self.streak | self.faint, 3.0, SPACING)
        shaded[:, :, :70] = False
        self.assertLess(np.count_nonzero(tree & shaded), 20)          # no speckle in the bright (shaded) side
        with self.assertRaises(ValueError):
            V.arterialTree(self.denoised, self.contrast, self.region, (5, 85, 5), SPACING, rootMaxMM=3.0)

    def test_coilNextToArtery(self):
        """A coil 4 mm beside the right branch: the metal margin does not cut the branch, the coil is not tree."""
        p = self.phantom
        image = self.image.copy()
        coil = V.ballMask(SHAPE, (30, 52, 72), 1.5, SPACING)        # the right branch passes about (30, 52, 65)
        image[coil] = 8000.0
        denoised = V.denoise(image, SPACING)
        contrast = V.localContrast(denoised, p.liver, SPACING)
        segment = V.arterialTree(denoised, contrast, self.region, p.root, SPACING, p.liver)
        self.assertGreater(segment.metalML, 0.0)
        self.assertFalse((segment.mask & coil).any())
        end = np.zeros(SHAPE, bool)
        end[tuple(np.array(p.right))] = True
        self.assertTrue((segment.mask & V.dilateMM(end, 3.0, SPACING)).any())   # still reaches the right branch end

    def test_denseContrastIsNotMetal(self):
        """Undiluted contrast (2000 HU) in a branch far from the injection point is below the metal floor."""
        p = self.phantom
        image = self.image.copy()
        dense = p.vessels & (segmentDistance(SHAPE, p.fork, p.right) <= p.radius)
        dense[:, :45] = False
        image[dense] = 2000.0
        denoised = V.denoise(image, SPACING)
        contrast = V.localContrast(denoised, p.liver, SPACING)
        segment = V.arterialTree(denoised, contrast, self.region, p.root, SPACING, p.liver)
        self.assertGreater(np.count_nonzero(segment.mask & dense), 0.8 * np.count_nonzero(dense & self.region))

    def test_tubeShape(self):
        p = self.phantom
        tube = V.tubeShape(self.contrast.z, SPACING, self.region & (self.contrast.z >= V.GROW_SD))
        loose = self._tree().mask
        tubular = self._tree(tube=tube).mask
        # a flat streak touching a branch: brightness alone follows it, the tube check less (with the default
        # edge-preserving denoising the streak edges stay sharp too, so the gain is smaller than with a Gaussian)
        self.assertGreater(np.count_nonzero(loose & self.streak), 0.5 * np.count_nonzero(self.streak))
        self.assertLessEqual(np.count_nonzero(tubular & self.streak), np.count_nonzero(loose & self.streak))
        gaussian = V.denoise(self.image, SPACING, method=V.DENOISE_GAUSSIAN)
        contrast = V.localContrast(gaussian, p.liver, SPACING)
        gaussianTube = V.tubeShape(contrast.z, SPACING, self.region & (contrast.z >= V.GROW_SD))
        gaussianLoose = V.arterialTree(gaussian, contrast, self.region, p.root, SPACING, p.liver).mask
        gaussianTubular = V.arterialTree(gaussian, contrast, self.region, p.root, SPACING, p.liver,
                                         tube=gaussianTube).mask
        self.assertLess(np.count_nonzero(gaussianTubular & self.streak),
                        0.5 * np.count_nonzero(gaussianLoose & self.streak))
        self.assertGreater(V.dice(tubular & ~self.streak & ~self.faint, p.vessels & self.region), 0.9)
        # the faint intrahepatic branch (thin, 8 SD in the raw image) still grows from the bright trunk: with the
        # Gaussian, and with anisotropic diffusion at conductance 0.5 (at 1.0 the diffusion flattens it as noise)
        self.assertGreater(np.count_nonzero(gaussianTubular & self.faint), 0.5 * np.count_nonzero(self.faint))
        saved = V.DENOISE_CONDUCTANCE
        try:
            V.DENOISE_CONDUCTANCE = 0.5
            diffused = V.denoise(self.image, SPACING, method=V.DENOISE_CURVATURE)
        finally:
            V.DENOISE_CONDUCTANCE = saved
        contrast = V.localContrast(diffused, p.liver, SPACING)
        tube = V.tubeShape(contrast.z, SPACING, self.region & (contrast.z >= V.GROW_SD))
        diffusedTree = V.arterialTree(diffused, contrast, self.region, p.root, SPACING, p.liver, tube=tube).mask
        self.assertGreater(np.count_nonzero(diffusedTree & self.faint), 0.5 * np.count_nonzero(self.faint))

    def test_vesselness(self):
        p = self.phantom
        vessel = V.tubeShape(self.contrast.z, SPACING, self.region, scalesMM=(1.0, 2.0))
        centre = np.zeros(SHAPE, bool)
        for a, b in p.segments:
            drawLine(centre, a, b)
        onVessel = np.median(vessel[centre & self.region])
        parenchyma = np.median(vessel[p.liver & ~V.dilateMM(p.vessels, 3.0, SPACING)])
        self.assertGreater(onVessel, 2 * max(parenchyma, 1e-6))
        self.assertGreater(onVessel, 0.5)
        faint = np.median(vessel[self.faint & self.region])
        self.assertGreater(faint, V.TUBE_MIN)                    # shape only: a faint tube is as tubular as a bright one
        plate = np.median(vessel[self.streak & ~V.dilateMM(p.vessels | self.metal, 3.0, SPACING)])
        self.assertLess(plate, 0.2 * onVessel)


class TreeTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.phantom = Phantom()
        cls.tree = cls.phantom.tree()

    def test_orientation(self):
        tree, p = self.tree, self.phantom
        self.assertEqual(tuple(tree.points[tree.root]), p.root)
        self.assertEqual(tree.loops, 0)
        self.assertTrue(np.all(tree.parent[np.arange(tree.count) != tree.root] >= 0))
        leftEnd, _ = tree.nearest(p.left)
        self.assertIn(tree.nearest(p.fork)[0], tree.pathToRoot(leftEnd))
        self.assertAlmostEqual(tree.distance[leftEnd], 25 + np.hypot(45, 30), delta=4.0)
        self.assertAlmostEqual(tree.radius[tree.nearest(p.fork)[0]], p.radius + 1, delta=1.5)
        self.assertGreaterEqual(len(tree.branchPoints()), 3)

    def test_spursPruned(self):
        p = self.phantom
        skeleton = p.skeleton.copy()
        skeleton[30, 40, 51] = skeleton[30, 40, 52] = True   # 2 mm spur off the trunk
        tree = V.buildTree(skeleton, SPACING, p.root)
        self.assertNotIn((30, 40, 52), {tuple(x) for x in tree.points})
        self.assertEqual(tree.count, self.tree.count)

    def test_loops(self):
        p = self.phantom
        skeleton = p.skeleton.copy()
        drawLine(skeleton, (30, 70, 25), (30, 70, 75))   # connects the left and right branches
        self.assertEqual(V.buildTree(skeleton, SPACING, p.root).loops, 1)

    def test_snapping(self):
        index, distance = V.snapTip(self.tree, (30, 60, 32))
        self.assertIsNotNone(index)
        self.assertLessEqual(distance, V.SNAP_MM)
        index, distance = V.snapTip(self.tree, (30, 60, 50))
        self.assertIsNone(index)
        self.assertGreater(distance, V.SNAP_MM)

    def test_rootTooFar(self):
        with self.assertRaises(ValueError):
            V.buildTree(self.phantom.skeleton, SPACING, (55, 85, 95))


class TerritoryTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.phantom = Phantom()
        cls.tree = cls.phantom.tree()
        cls.leftTip = cls.tree.nearest((30, 33, 47))[0]   # just after the fork, on the left branch

    def test_euclidean(self):
        p = self.phantom
        territory = V.territoryEuclidean(self.tree, p.liver).territory(self.tree, self.leftTip)
        self.assertGreater(V.dice(territory, p.leftTruth), 0.95)
        self.assertFalse((territory & ~p.liver).any())

    def test_wholeTreeIsWholeLiver(self):
        territory = V.territoryEuclidean(self.tree, self.phantom.liver).territory(self.tree, self.tree.root)
        self.assertTrue(np.array_equal(territory, self.phantom.liver))

    def test_geodesicInConvexLiver(self):
        p = self.phantom
        geodesic = V.territoryGeodesic(self.tree, p.liver).territory(self.tree, self.leftTip)
        self.assertGreater(V.dice(geodesic, p.leftTruth), 0.9)

    def test_geodesicRespectsFissure(self):
        """Two lobes separated by a fissure, joined only at the far end. The left branch runs close to the fissure,
        the right branch far from it: the right lobe next to the fissure is nearest (Euclidean) to the left branch,
        but inside the liver it is reached by the right branch."""
        skeleton = np.zeros(SHAPE, bool)
        for a, b in (((30, 5, 64), (30, 20, 64)), ((30, 20, 64), (30, 20, 44)), ((30, 20, 44), (30, 60, 44)),
                     ((30, 20, 64), (30, 20, 85)), ((30, 20, 85), (30, 80, 85))):
            drawLine(skeleton, a, b)
        liver = np.zeros(SHAPE, bool)
        liver[10:50, 15:90, 20:96] = True
        liver[:, :86, 48:51] = False
        tree = V.buildTree(skeleton, SPACING, (30, 5, 64))
        tip = tree.nearest((30, 40, 44))[0]
        euclidean = V.territoryEuclidean(tree, liver).territory(tree, tip)
        geodesic = V.territoryGeodesic(tree, liver, stepMM=1.0).territory(tree, tip)
        nearFissure = np.zeros(SHAPE, bool)
        nearFissure[:, 40:60, 51:61] = True
        nearFissure &= liver
        self.assertGreater(np.count_nonzero(euclidean & nearFissure), 0.9 * np.count_nonzero(nearFissure))
        self.assertEqual(np.count_nonzero(geodesic & nearFissure), 0)
        self.assertGreater(np.count_nonzero(geodesic), 0)

    def test_nestedSubtraction(self):
        tmap = V.territoryEuclidean(self.tree, self.phantom.liver)
        lobar = self.leftTip
        segmental = self.tree.nearest((30, 70, 24))[0]
        masks = [tmap.territory(self.tree, lobar), tmap.territory(self.tree, segmental)]
        self.assertTrue((masks[0] & masks[1]).any())
        result = V.subtractNested(masks, [lobar, segmental], self.tree)
        self.assertFalse((result[0] & result[1]).any())
        self.assertTrue(np.array_equal(result[1], masks[1]))
        self.assertTrue(np.array_equal(result[0] | result[1], masks[0]))


class SafetyAndFeederTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.phantom = Phantom()

    def test_extrahepatic(self):
        p = self.phantom
        skeleton = p.skeleton.copy()
        drawLine(skeleton, (30, 52, 35), (59, 52, 35))     # leaves the liver downwards from the left branch
        tree = V.buildTree(skeleton, SPACING, p.root)
        leftTip = tree.nearest((30, 33, 47))[0]
        rightTip = tree.nearest((30, 33, 53))[0]
        findings = V.extrahepaticBranches(tree, p.liver, leftTip)
        self.assertEqual(len(findings), 1)
        self.assertGreater(findings[0]["distanceMM"], V.EXTRAHEPATIC_MM)
        self.assertEqual(V.extrahepaticBranches(tree, p.liver, rightTip), [])
        # the trunk outside the liver leads into the liver: not a non-target branch
        self.assertEqual(len(V.extrahepaticBranches(tree, p.liver, tree.root)), 1)

    def test_feeders(self):
        p = self.phantom
        tree = p.tree()
        tumour = V.ballMask(SHAPE, (30, 72, 24), 6.0, SPACING)
        feeders = V.tumourFeeders(tree, tumour)
        self.assertIsNotNone(feeders)
        ancestor = feeders["ancestor"]
        self.assertTrue(tree.isAncestor(tree.nearest((30, 52, 35))[0], ancestor))
        tmap = V.territoryEuclidean(tree, p.liver)
        candidates = V.candidatePositions(tree, tmap, tumour & p.liver, p.liver, 0.001, feeders=feeders)
        self.assertGreater(len(candidates), 1)
        self.assertEqual(candidates[0]["kind"], "all feeders")
        self.assertGreater(candidates[0]["coveragePercent"], 90.0)
        children = np.bincount(tree.parent[tree.parent >= 0], minlength=tree.count)
        self.assertTrue(all(children[c["index"]] <= 1 for c in candidates))      # never at a bifurcation
        for c in candidates:   # midway: the territory of the branch point it stands for (plus the segment's own)
            mid, branch = tmap.territory(tree, c["index"]), tmap.territory(tree, c["branchPoint"])
            self.assertFalse((branch & ~mid).any())
            self.assertTrue(tree.isAncestor(c["index"], c["branchPoint"]))
        volumes = [c["territoryML"] for c in candidates]
        self.assertEqual(volumes, sorted(volumes))   # more upstream: larger territory
        self.assertIsNone(V.tumourFeeders(tree, V.ballMask(SHAPE, (50, 50, 50), 2.0, SPACING)))

    def test_twoFeedingBranches(self):
        p = self.phantom
        tree = p.tree()
        tumour = V.ballMask(SHAPE, (30, 58, 50), 8.0, SPACING)   # between the left and the right side branches
        feeders = V.tumourFeeders(tree, tumour, marginMM=10.0)
        self.assertEqual(len(feeders["branches"]), 2)
        self.assertEqual(feeders["ancestor"], tree.nearest(p.fork)[0])


class EnhancementAndFovTest(unittest.TestCase):

    def test_enhancementTerritory(self):
        liver = ellipsoid(SHAPE, (30, 45, 50), (25, 35, 45))
        rng = np.random.default_rng(2)
        values = np.where(liver, 100.0, 0.0) + rng.normal(0, 8, SHAPE)
        enhanced = liver.copy()
        enhanced[:, :, :50] = False
        values[enhanced] += 80.0
        percent = V.autoEnhancementPercent(values, liver)
        threshold = V.thresholdFromPercent(values, liver, percent)
        territory = V.enhancementTerritory(values, liver, threshold, 0.001)
        self.assertGreater(V.dice(territory, enhanced), 0.95)

    def test_otsu(self):
        values = np.concatenate([np.full(1000, 10.0), np.full(1000, 50.0)])
        self.assertTrue(10.0 < V.otsuThreshold(values) < 50.0)

    def test_fieldOfView(self):
        shape = (20, 50, 50)
        k, j, i = np.ogrid[0:shape[0], 0:shape[1], 0:shape[2]]
        cylinder = np.broadcast_to((j - 25) ** 2 + (i - 25) ** 2 <= 20 ** 2, shape)
        rng = np.random.default_rng(3)
        array = np.where(cylinder, rng.integers(-900, 900, shape), -1000)
        fov = V.fieldOfViewMask(array)
        self.assertTrue(np.array_equal(fov, cylinder))
        self.assertTrue(V.fieldOfViewMask(rng.integers(0, 1000, shape)).all())
        self.assertAlmostEqual(V.coverageFraction(cylinder, fov), 1.0)

    def test_comparisons(self):
        a = np.zeros((10, 10, 10), bool)
        a[:5] = True
        self.assertAlmostEqual(V.dice(a, a), 1.0)
        self.assertAlmostEqual(V.dice(a, ~a), 0.0)
        values = np.ones((10, 10, 10))
        self.assertAlmostEqual(V.countsInside(values, np.ones_like(a), a), 0.5)


class ThickVesselTest(unittest.TestCase):
    """An aorta-sized tube (radius 10 mm): tube-shaped with the default scales, and one clean centerline."""

    def setUp(self):
        self.shape = (40, 120, 60)
        self.tube = segmentDistance(self.shape, (20, 5, 30), (20, 115, 30)) <= 10.0

    def test_scales(self):
        scales = V.vesselScales(12.0)
        self.assertAlmostEqual(scales[0], 0.5, places=2)
        self.assertAlmostEqual(scales[-1], 12.0 / np.sqrt(3.0), places=2)
        self.assertEqual(list(scales), sorted(scales))

    def test_largeScaleTube(self):
        rng = np.random.default_rng(7)
        image = np.where(self.tube, 30.0, 0.0) + rng.normal(0, 1, self.shape)
        large = V.tubeShape(image, SPACING, self.tube, scalesMM=V.vesselScales(12.0))
        core = segmentDistance(self.shape, (20, 5, 30), (20, 115, 30)) <= 4.0
        core[:, :20] = core[:, 100:] = False
        self.assertGreater(np.median(large[core]), V.TUBE_MIN)

    @unittest.skipIf(importlib.util.find_spec("skimage") is None, "scikit-image not installed")
    def test_coreKeptByTubeCheck(self):
        """With the tube check, the aorta-sized lumen keeps its core (no hollow shell, no tangled centerline)."""
        rng = np.random.default_rng(9)
        liver = np.zeros(self.shape, bool)
        liver[:, :, 46:] = True                                     # beside the aorta, as in a patient
        image = np.where(liver, 100.0, 60.0) + rng.normal(0, 5, self.shape)
        image[self.tube] = 600.0
        region = np.ones(self.shape, bool)
        denoised = V.denoise(image, SPACING, method=V.DENOISE_GAUSSIAN)
        contrast = V.localContrast(denoised, liver, SPACING)
        tube = V.tubeShape(contrast.z, SPACING, region & (contrast.z >= V.GROW_SD))
        tree = V.arterialTree(denoised, contrast, region, (20, 5, 30), SPACING, liver, tube=tube).mask
        core = segmentDistance(self.shape, (20, 5, 30), (20, 115, 30)) <= 5.0
        self.assertGreater(np.count_nonzero(tree & core), 0.9 * np.count_nonzero(core))
        centerline = V.buildTree(V.skeletonize(tree), SPACING, (20, 5, 30), tree)
        self.assertLessEqual(len(centerline.branchPoints()), 6)

    @unittest.skipIf(importlib.util.find_spec("skimage") is None, "scikit-image not installed")
    def test_noTangle(self):
        rng = np.random.default_rng(8)
        bumpy = self.tube | (rng.random(self.shape) < 0.02) & V.dilateMM(self.tube, 1.5, SPACING)
        tree = V.buildTree(V.skeletonize(bumpy), SPACING, (20, 5, 30), bumpy)
        # radius-aware pruning removes the surface bumps that plain pruning keeps
        withoutRadius = V.buildTree(V.skeletonize(bumpy), SPACING, (20, 5, 30))
        self.assertLess(len(tree.branchPoints()), len(withoutRadius.branchPoints()))
        self.assertLessEqual(len(tree.branchPoints()), 6)


class LimitedTerritoryTest(unittest.TestCase):
    """A selective injection: only the trunk and the left branch are opacified, the right lobe is not."""

    def setUp(self):
        self.phantom = Phantom()
        skeleton = np.zeros(SHAPE, bool)
        for a, b in (self.phantom.segments[0], self.phantom.segments[1], self.phantom.segments[3]):
            drawLine(skeleton, a, b)
        self.tree = V.buildTree(skeleton, SPACING, self.phantom.root)

    def _check(self, tmap):
        p = self.phantom
        tip = self.tree.nearest((30, 33, 47))[0]                    # left branch, just after the fork
        whole = tmap.territory(self.tree, tip)
        tmap.limitMM = 15.0
        limited = tmap.territory(self.tree, tip)
        right = p.liver.copy()
        right[:, :, :85] = False                                    # the right lobe: no visible branch
        self.assertGreater(np.count_nonzero(whole & right), 0)       # unlimited: claimed by the left branch
        self.assertEqual(np.count_nonzero(limited & right), 0)
        self.assertFalse((limited & ~whole).any())
        coordinates = self.tree.points[self.tree.downstream(tip)]
        close = np.zeros(SHAPE, bool)
        close[tuple(coordinates.T)] = True
        close = V.dilateMM(close, 7.0, SPACING) & whole   # geodesic: 6-connected path length is longer
        self.assertTrue(np.array_equal(limited & close, close))      # liver near its own branches is kept

    def test_euclidean(self):
        self._check(V.territoryEuclidean(self.tree, self.phantom.liver))

    def test_geodesic(self):
        self._check(V.territoryGeodesic(self.tree, self.phantom.liver))

    def test_suggestDistance(self):
        p = self.phantom
        distance = V.treeDistance(self.tree, p.liver)
        self.assertTrue(8.0 <= V.suggestSupplyDistance(distance, p.liver) <= 40.0)
        close = V.dilateMM(p.vessels, 8.0, SPACING) & p.liver
        full = V.buildTree(p.skeleton, SPACING, p.root)
        self.assertEqual(V.suggestSupplyDistance(V.treeDistance(full, close), close), V.SUPPLY_DISTANCE_MM)


@unittest.skipIf(importlib.util.find_spec("SimpleITK") is None, "SimpleITK not installed")
class DenoiseTest(unittest.TestCase):

    def test_methods(self):
        rng = np.random.default_rng(6)
        shape = (30, 30, 30)
        clean = np.full(shape, 100.0)
        clean[:, :, 15:] = 400.0                                   # an edge (vessel wall)
        noisy = clean + rng.normal(0, 20, shape)
        for method, _ in V.DENOISE_METHODS:
            result = V.denoise(noisy, (1.0, 1.0, 1.0), 1.0, method)
            self.assertEqual(result.shape, shape, method)
            flat = result[5:25, 5:25, 3:10]
            self.assertLess(flat.std(), 0.8 * noisy[5:25, 5:25, 3:10].std(), method)   # noise reduced
        # edge-preserving: the step stays sharper than with the Gaussian
        gaussian = V.denoise(noisy, (1.0, 1.0, 1.0), 1.5, V.DENOISE_GAUSSIAN)
        diffusion = V.denoise(noisy, (1.0, 1.0, 1.0), 1.5, V.DENOISE_CURVATURE)
        step = lambda a: float(np.mean(a[:, :, 15] - a[:, :, 14]))
        self.assertGreater(step(diffusion), step(gaussian))

    def test_parameterMethod(self):
        V.setParameters({"DENOISE_METHOD": V.DENOISE_BILATERAL})
        self.assertEqual(V.DENOISE_METHOD, V.DENOISE_BILATERAL)
        V.setParameters({})
        self.assertEqual(V.DENOISE_METHOD, V.DENOISE_CURVATURE)                 # the default


class CarmAngleTest(unittest.TestCase):

    def test_angles(self):
        self.assertEqual(V.carmAngles((0, 1, 0)), (0.0, 0.0))                       # AP
        primary, secondary = V.carmAngles((-np.sin(np.radians(30)), np.cos(np.radians(30)), 0))
        self.assertAlmostEqual(primary, 30.0)                                      # detector to the left: LAO 30
        self.assertAlmostEqual(V.carmAngles((1, 1, 0))[0], -45.0)                   # RAO 45
        self.assertAlmostEqual(V.carmAngles((0, 1, 1))[1], 45.0)                    # CRA 45
        self.assertAlmostEqual(V.carmAngles((0, 1, -np.tan(np.radians(20))))[1], -20.0)   # CAU 20
        self.assertAlmostEqual(V.carmAngles((-1, 0, 0))[0], 90.0)                   # left lateral: LAO 90
        self.assertEqual(V.carmLabel(30.2, -15.0), "LAO 30°  CAU 15°")
        self.assertEqual(V.carmLabel(-20.0, 10.0), "RAO 20°  CRA 10°")
        self.assertEqual(V.carmLabel(0.0, 0.0), "AP 0°  CRA/CAU 0°")


@unittest.skipIf(importlib.util.find_spec("skimage") is None, "scikit-image not installed")
class SkeletonTest(unittest.TestCase):

    def test_skeletonOfPhantom(self):
        p = Phantom()
        skeleton = V.skeletonize(p.vessels)
        tree = V.buildTree(skeleton, SPACING, p.root, p.vessels)
        self.assertLessEqual(tree.nearest(p.root)[1], 3.0)
        self.assertLessEqual(tree.loops, V.MAX_LOOPS)
        for end in (p.left, p.right):
            self.assertLessEqual(tree.nearest(end)[1], 4.0)
        tip = tree.nearest((30, 33, 47))[0]
        territory = V.territoryEuclidean(tree, p.liver).territory(tree, tip)
        self.assertGreater(V.dice(territory, p.leftTruth), 0.9)


if __name__ == "__main__":
    unittest.main()
