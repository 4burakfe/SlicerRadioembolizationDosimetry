"""Unit tests of TaranisLib.lsf and the LSF validator (outside Slicer)."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from TaranisLib import lsf as L  # noqa: E402
from TaranisLib import roles as R  # noqa: E402
from TaranisLib import workflow as W  # noqa: E402


class LsfTest(unittest.TestCase):

    def test_counts(self):
        values = np.zeros((4, 4, 4))
        lungs = np.zeros(values.shape, bool)
        liver = np.zeros(values.shape, bool)
        lungs[0, :2, :2] = True     # 4 voxels
        liver[2:, :, :] = True      # 32 voxels
        values[lungs] = 10.0
        values[liver] = 22.5
        result = L.computeLungShunt(values, lungs, liver)
        self.assertAlmostEqual(result["lsfPercent"], 100 * 40 / (40 + 720))
        # a boundary voxel claimed by both counts for the liver only
        lungs[2, 0, 0] = True
        shared = L.computeLungShunt(values, lungs, liver)
        self.assertEqual(shared["sharedVoxels"], 1)
        self.assertAlmostEqual(shared["lsfPercent"], result["lsfPercent"])
        with self.assertRaises(ValueError):
            L.computeLungShunt(values, np.zeros(values.shape, bool), liver)

    def test_extrahepaticUptake(self):
        values = np.zeros((4, 4, 4))
        lungs = np.zeros(values.shape, bool)
        liver = np.zeros(values.shape, bool)
        lungs[0, 0, :] = True
        liver[2:, :, :] = True
        values[lungs] = 1.0          # 4
        values[liver] = 2.0          # 64
        values[1, 0, 0] = 32.0       # stomach: 32 of 100
        result = L.computeLungShunt(values, lungs, liver)
        self.assertAlmostEqual(result["extraFraction"], 0.32)
        issues = L.lsfIssues(result)
        self.assertTrue(any(s == W.SEVERITY_WARNING and "free Tc-99m" in t for s, t in issues))
        values[1, 0, 0] = 10.0       # 10 of 78: below 20 %
        self.assertFalse(any("free Tc-99m" in t for _, t in L.lsfIssues(L.computeLungShunt(values, lungs, liver))))

    def test_massAndDose(self):
        # 1000 mL of lung at -700 HU -> 300 g; air and bone clipped
        self.assertAlmostEqual(L.lungMassFromHU(np.full(1000, -700.0), 1.0), 300.0)
        self.assertAlmostEqual(L.lungMassFromHU(np.array([-1100.0, 500.0]), 1.0), 1.1)
        # 2 GBq, LSF 10 %, 1 kg -> 9.934 Gy
        self.assertAlmostEqual(L.lungDoseGy(2.0, 10.0, 1000.0), 49.67 * 0.2)
        self.assertIsNone(L.lungDoseGy(0, 10.0, 1000.0))
        self.assertAlmostEqual(L.lungDosePerGBq(10.0, 500.0), 49.67 * 0.2)

    def test_issues(self):
        issues = L.lsfIssues({"lungCoverage": 0.5, "negativeVoxels": 0}, lungSegmentML=900)
        severities = [severity for severity, _ in issues]
        self.assertEqual(severities, [W.SEVERITY_WARNING, W.SEVERITY_INFO])
        self.assertEqual(L.lsfIssues({"lungCoverage": 0.95}, lungSegmentML=3000), [])

    def test_inputsKey(self):
        segments = [W.SegmentInfo("a", "Lungs", W.SEGMENT_LUNGS, voxels=10),
                    W.SegmentInfo("b", "Liver", W.SEGMENT_LIVER, voxels=20),
                    W.SegmentInfo("c", "Tumor", W.SEGMENT_TUMOR, voxels=5)]
        key = L.inputsKey(segments, "spect")
        segments[2].voxels = 6                  # tumours do not matter
        self.assertEqual(L.inputsKey(segments, "spect"), key)
        segments[0].voxels = 11
        self.assertNotEqual(L.inputsKey(segments, "spect"), key)

    def test_validator(self):
        s = W.CaseSnapshot(mode=R.MODE_RELATIVE, roleTypes={R.ROLE_DOSIMETRY: R.TYPE_MAA_SPECT},
                           lsfValue=8.0, lsfSource=L.SOURCE_IMAGE, lsfLungMassG=1000.0)
        self.assertEqual(W.evaluateLsf(s).state, W.STATE_DONE)
        s.plannedActivityGBq = 8.0              # 8 GBq x 8 % x 49.67 / 1 kg = 31.8 Gy
        status = W.evaluateLsf(s)
        self.assertEqual(status.state, W.STATE_WARNING)
        self.assertTrue(any("30 Gy" in issue.text for issue in status.issues))
        s.plannedActivityGBq = 2.0
        self.assertEqual(W.evaluateLsf(s).state, W.STATE_DONE)
        s.lsfFromImage, s.lsfOutdated = True, True
        self.assertEqual(W.evaluateLsf(s).state, W.STATE_OUTDATED)
        s.lsfOutdated = False
        s.lsfIssues = [(W.SEVERITY_WARNING, "lungs outside the field of view")]
        self.assertEqual(W.evaluateLsf(s).state, W.STATE_WARNING)


if __name__ == "__main__":
    unittest.main()
