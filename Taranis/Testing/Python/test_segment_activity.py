"""Segment activity / mass / mean dose with voxel densities (TaranisLib.dosimetry.segmentDoseStatistics), outside
Slicer: the Qt / VTK / Slicer imports of the module are replaced by empty stand-ins."""

import os
import sys
import types
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def _stubSlicerModules():
    """Empty qt, ctk, vtk and slicer modules so TaranisLib.dosimetry imports without Slicer."""
    if "slicer" in sys.modules and not getattr(sys.modules["slicer"], "_taranisStub", False):
        return  # inside Slicer
    class _Any:
        def __init__(self, *args, **kwargs):
            pass

        def __getattr__(self, name):
            return _Any()

        def __call__(self, *args, **kwargs):
            return _Any()

    class _Module(types.ModuleType):
        def __getattr__(self, name):
            if name.startswith("__"):
                raise AttributeError(name)
            return type(name, (_Any,), {})

    for name in ("qt", "ctk", "vtk", "slicer", "slicer.util"):
        sys.modules.setdefault(name, _Module(name))
    sys.modules["slicer"]._taranisStub = True
    scripted = types.ModuleType("slicer.ScriptedLoadableModule")
    for cls in ("ScriptedLoadableModule", "ScriptedLoadableModuleWidget", "ScriptedLoadableModuleLogic",
                "ScriptedLoadableModuleTest"):
        setattr(scripted, cls, type(cls, (object,), {}))
    scripted.__all__ = [n for n in dir(scripted) if n.startswith("Scripted")]
    sys.modules["slicer.ScriptedLoadableModule"] = scripted


_stubSlicerModules()
from TaranisLib import dosimetry as D  # noqa: E402

CF = 49.67          # J/GBq (Y-90)
LIVER = 1.05
LUNG = 0.30
VOXEL_ML = 0.05


class SegmentActivityTest(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(1)
        self.shape = (10, 12, 14)
        self.activity = rng.uniform(0.0, 2.0, self.shape)          # MBq per voxel
        self.lung = np.zeros(self.shape, bool)
        self.lung[:4] = True
        self.density = np.full(self.shape, LIVER)
        self.density[self.lung] = LUNG
        # Local deposition dose of every voxel with its own mass (as the absolute module builds it)
        self.dose = self.activity * CF / (self.density * VOXEL_ML)

    def test_field_of_view_segment_overlapping_the_lungs(self):
        fov = np.ones(self.shape, bool)
        dose, volume, activity = D.segmentDoseStatistics(self.dose, fov, VOXEL_ML, CF, LIVER,
                                                         densityArray=self.density)
        self.assertAlmostEqual(activity, self.activity.sum(), places=6)
        mass = D.segmentMassG(fov, VOXEL_ML, LIVER, self.density)
        self.assertAlmostEqual(mass, self.density.sum() * VOXEL_ML, places=6)
        self.assertAlmostEqual(dose, self.activity.sum() * CF / mass, places=6)   # absorbed energy / mass
        # Without the density map the lung activity is counted LIVER / LUNG times (the reported bug)
        _, _, wrong = D.segmentDoseStatistics(self.dose, fov, VOXEL_ML, CF, LIVER)
        expectedWrong = self.activity[~self.lung].sum() + self.activity[self.lung].sum() * LIVER / LUNG
        self.assertAlmostEqual(wrong, expectedWrong, places=6)

    def test_uniform_density_segments_unchanged(self):
        liver = ~self.lung
        old = D.segmentDoseStatistics(self.dose, liver, VOXEL_ML, CF, LIVER)
        new = D.segmentDoseStatistics(self.dose, liver, VOXEL_ML, CF, LIVER, densityArray=self.density)
        for a, b in zip(old, new):
            self.assertAlmostEqual(a, b, places=6)
        _, _, lungActivity = D.segmentDoseStatistics(self.dose, self.lung, VOXEL_ML, CF, LIVER,
                                                     densityArray=self.density)
        self.assertAlmostEqual(lungActivity, self.activity[self.lung].sum(), places=6)

    def test_voxel_s_activity_from_ldm_map(self):
        fov = np.ones(self.shape, bool)
        spread = np.full(self.shape, self.dose.mean())   # any other dose map: activities still come from LDM
        _, _, activity = D.segmentDoseStatistics(spread, fov, VOXEL_ML, CF, LIVER, activityDoseArray=self.dose,
                                                 densityArray=self.density)
        self.assertAlmostEqual(activity, self.activity.sum(), places=6)

    def test_empty_mask(self):
        empty = np.zeros(self.shape, bool)
        dose, volume, activity = D.segmentDoseStatistics(self.dose, empty, VOXEL_ML, CF, LIVER,
                                                         densityArray=self.density)
        self.assertTrue(np.isnan(dose) and np.isnan(activity))
        self.assertEqual(volume, 0.0)
        self.assertEqual(D.segmentMassG(empty, VOXEL_ML, LIVER, self.density), 0.0)


if __name__ == "__main__":
    unittest.main()
