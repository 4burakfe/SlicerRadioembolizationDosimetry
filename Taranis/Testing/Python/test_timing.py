"""Unit tests of TaranisLib.timing (hours after treatment from the DICOM header, outside Slicer)."""

import datetime
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from TaranisLib import timing as T  # noqa: E402

ADMIN = datetime.datetime(2026, 3, 2, 11, 0)


def header(**fields):
    base = {"Modality": "PT", "SeriesDate": "20260303", "SeriesTime": "090000", "AcquisitionDate": "20260303",
            "AcquisitionTime": "091500", "DecayCorrection": "START"}
    base.update(fields)
    return base


class TimingTest(unittest.TestCase):

    def test_start_uses_series_time(self):
        result = T.hoursAfterTreatment(ADMIN, header())
        self.assertAlmostEqual(result["hours"], 22.0)
        self.assertIn("series start", result["basis"])

    def test_series_reconstructed_later(self):
        result = T.hoursAfterTreatment(ADMIN, header(SeriesTime="140000"))
        self.assertAlmostEqual(result["hours"], 22.25)   # acquisition 09:15
        self.assertTrue(any("reconstructed later" in n for n in result["notes"]))

    def test_admin_and_none(self):
        self.assertEqual(T.hoursAfterTreatment(ADMIN, header(DecayCorrection="ADMIN"))["hours"], 0.0)
        result = T.hoursAfterTreatment(ADMIN, header(DecayCorrection="NONE"))
        self.assertAlmostEqual(result["hours"], 22.25)
        self.assertTrue(any("bed positions" in n for n in result["notes"]))

    def test_unknown_decay_correction(self):
        result = T.hoursAfterTreatment(ADMIN, header(DecayCorrection="", Modality="NM"))
        self.assertAlmostEqual(result["hours"], 22.25)
        self.assertIn("does not state", result["basis"])

    def test_acquisition_datetime_with_timezone(self):
        result = T.hoursAfterTreatment(ADMIN, header(DecayCorrection="NONE", AcquisitionDateTime="20260303083000.000000+0300"))
        self.assertAlmostEqual(result["hours"], 21.5)

    def test_implausible(self):
        before = T.hoursAfterTreatment(ADMIN, header(SeriesDate="20260301", AcquisitionDate="20260301"))
        self.assertIsNone(before["hours"])
        self.assertTrue(any("before the administration" in n for n in before["notes"]))
        late = T.hoursAfterTreatment(ADMIN, header(SeriesDate="20260315", AcquisitionDate="20260315"))
        self.assertIsNone(late["hours"])
        self.assertIsNone(T.hoursAfterTreatment(ADMIN, {"DecayCorrection": "START"})["hours"])

    def test_scanner_injection_time(self):
        same = T.hoursAfterTreatment(ADMIN, header(RadiopharmaceuticalStartDateTime="20260302110500"))
        self.assertFalse(any("scanner recorded" in n for n in same["notes"]))
        other = T.hoursAfterTreatment(ADMIN, header(RadiopharmaceuticalStartDateTime="20260302124000"))
        self.assertTrue(any("100 min" in n for n in other["notes"]))

    def test_header_times_from_mapping(self):
        class Item(dict):
            pass
        dataset = {"DecayCorrection": "START", "SeriesDate": "20260303",
                   "RadiopharmaceuticalInformationSequence": [Item(RadiopharmaceuticalStartTime="110000")]}
        fields = T.headerTimes(dataset)
        self.assertEqual(fields["DecayCorrection"], "START")
        self.assertEqual(fields["RadiopharmaceuticalStartTime"], "110000")
        self.assertEqual(fields["AcquisitionTime"], "")


if __name__ == "__main__":
    unittest.main()
