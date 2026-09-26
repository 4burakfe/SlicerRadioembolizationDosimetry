"""Unit tests of TaranisLib.doseguard (dose checks after a dosimetry calculation)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from TaranisLib import doseguard as DG  # noqa: E402
from TaranisLib import workflow as W  # noqa: E402


def seg(name, role, dose, volume=100.0, scope=None):
    entry = {"name": name, "role": role, "dose": dose, "volume": volume}
    if scope:
        entry["scope"] = scope
    return entry


def texts(issues, severity=W.SEVERITY_WARNING):
    return " | ".join(text for s, text in issues if s == severity)


class DoseGuardTest(unittest.TestCase):

    def test_microspheres(self):
        self.assertEqual(DG.microspheresFromText("Glass microspheres"), DG.GLASS)
        self.assertEqual(DG.microspheresFromText("Resin microspheres"), DG.RESIN)
        self.assertIsNone(DG.microspheresFromText("Custom"))

    def test_quiet_when_everything_is_fine(self):
        segments = [seg("Tumor 1", "tumor", 200.0), seg("Perfused normal liver", "normal", 50.0, 900,
                                                         W.NORMAL_SCOPE_PERFUSED)]
        issues = DG.doseChecks(segments, DG.GLASS, lsfPercent=5.0, lungDosesGy=[("Lung", 8.0)],
                               extraUptakeFraction=0.05, outsidePerfusedFraction=0.02, relative=True)
        self.assertEqual(texts(issues), "")

    def test_normal_tissue_limits(self):
        normal = [seg("Normal liver", "normal", 50.0)]
        self.assertIn("above 40 Gy for resin", texts(DG.doseChecks(normal, DG.RESIN)))
        self.assertIn("segmentectomy or lobectomy", texts(DG.doseChecks(normal, DG.RESIN)))
        self.assertEqual(texts(DG.doseChecks(normal, DG.GLASS)), "")
        self.assertIn("above 90 Gy for glass", texts(DG.doseChecks([seg("N", "normal", 95.0)], DG.GLASS)))
        self.assertEqual(texts(DG.doseChecks(normal, None)), "")   # unknown device: no device thresholds

    def test_tumour_dose(self):
        tumours = [seg("Tumor 1", "tumor", 100.0), seg("Viable 1", "viable", 60.0), seg("Out", "tumor", float("nan"))]
        text = texts(DG.doseChecks(tumours, DG.RESIN))
        self.assertIn("Low tumour dose: 'Viable 1' 60.0 Gy (below 80 Gy for resin", text)
        self.assertNotIn("Tumor 1", text.split("Low tumour dose")[1].split("|")[0])
        text = texts(DG.doseChecks(tumours, DG.GLASS))
        self.assertIn("'Tumor 1' 100.0 Gy, 'Viable 1' 60.0 Gy (below 140 Gy", text)
        self.assertNotIn("'Out'", text)

    def test_tumour_to_normal_ratio(self):
        segments = [seg("T high", "tumor", 300.0), seg("T low", "tumor", 120.0), seg("T very low", "tumor", 80.0),
                    seg("Normal liver", "normal", 20.0, 1500),
                    seg("Perfused normal liver", "normal", 100.0, 500, W.NORMAL_SCOPE_PERFUSED)]
        issues = DG.doseChecks(segments, DG.GLASS, relative=True)
        text = texts(issues)
        self.assertIn("too low (< 1): 'T very low' 0.80 (perfused normal liver 'Perfused normal liver': 100.0 Gy)",
                      text)
        self.assertIn("low (< 1.5): 'T low' 1.20", text)
        self.assertIn("check the segmentation and the registration", text.lower())
        self.assertNotIn("T high", text.split("Tumour-to-normal")[1])
        # only the whole normal liver: used, with a note in patient-relative mode
        issues = DG.doseChecks(segments[:4], DG.GLASS, relative=True)
        self.assertNotIn("Tumour-to-normal", texts(issues))   # 80 / 20 = 4
        self.assertIn("includes unperfused liver", texts(issues, W.SEVERITY_INFO))
        # no normal tissue
        self.assertIn("no normal tissue segment", texts(DG.doseChecks(segments[:3], DG.GLASS), W.SEVERITY_INFO))

    def test_lungs(self):
        issues = DG.doseChecks([], DG.GLASS, lsfPercent=22.0, lungDosesGy=[("Estimated lung dose", 35.0)])
        text = texts(issues)
        self.assertIn("Lung shunt fraction 22.0 % is above 20 %", text)
        self.assertIn("High lung dose: Estimated lung dose 35.0 Gy", text)
        self.assertEqual(texts(DG.doseChecks([], DG.GLASS, lsfPercent=20.0, lungDosesGy=[("L", 30.0)])), "")

    def test_uptake_outside(self):
        text = texts(DG.doseChecks([], None, extraUptakeFraction=0.25))
        self.assertIn("outside the whole liver and the lungs: 25.0%", text)
        self.assertIn("free Tc-99m", text)
        self.assertIn("no lung segment", texts(DG.doseChecks([], None, extraUptakeFraction=0.25,
                                                            lungsSegmented=False)))
        text = texts(DG.doseChecks([], None, outsidePerfusedFraction=0.3, relative=True))
        self.assertIn("activity outside the perfused volumes: 30.0%", text)
        self.assertIn("0 Gy", text)
        self.assertEqual(texts(DG.doseChecks([], None, extraUptakeFraction=0.2, outsidePerfusedFraction=0.2)), "")

    def test_hours_after_treatment(self):
        self.assertIn("Hours after treatment is 0", texts(DG.doseChecks([], None, hoursAfterTreatment=0.0)))
        self.assertEqual(texts(DG.doseChecks([], None, hoursAfterTreatment=20.0)), "")
        self.assertEqual(texts(DG.doseChecks([], None)), "")   # patient-relative: not applicable

    def test_long_lists_are_shortened(self):
        tumours = [seg(f"T{i}", "tumor", 10.0) for i in range(10)]
        self.assertIn("and 4 more", texts(DG.doseChecks(tumours, DG.RESIN)))


if __name__ == "__main__":
    unittest.main()
