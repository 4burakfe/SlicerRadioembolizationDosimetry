"""Unit tests of the absolute dosimetry lock (TaranisLib.roles.absoluteDosimetryProblem), outside Slicer."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from TaranisLib import roles as R  # noqa: E402


def problem(**kwargs):
    args = dict(inputName="PET", hasCase=False, caseImageName="", isCaseImage=False, caseImageType=None,
                voxelUnits="", dicomUnits="")
    args.update(kwargs)
    return R.absoluteDosimetryProblem(**args)


class AbsoluteLockTest(unittest.TestCase):

    def test_units(self):
        for text in ("Bq/ml", "kBq/mL", "MBq/ml", "BQML", " bq/ml "):
            self.assertTrue(R.unitsAreActivityConcentration(text), text)
        for text in ("", None, "{SUVbw}g/ml", "counts", "g/ml"):
            self.assertFalse(R.unitsAreActivityConcentration(text), text)

    def test_without_case(self):
        self.assertEqual(problem(voxelUnits="Bq/ml"), "")
        self.assertEqual(problem(dicomUnits="BQML"), "")
        self.assertIn("SUV", problem(voxelUnits="{SUVbw}g/ml", dicomUnits="GML"))
        self.assertIn("counts", problem(dicomUnits="CNTS"))
        self.assertIn("not recognised", problem())

    def test_with_case(self):
        base = dict(hasCase=True, caseImageName="PET", isCaseImage=True, caseImageType=R.TYPE_Y90_PET)
        self.assertEqual(problem(**base, dicomUnits="BQML"), "")
        self.assertEqual(problem(**base), "")   # no metadata (e.g. NRRD) but assigned as Y-90 PET in the case
        self.assertIn("no dosimetry image", problem(**dict(base, caseImageName="")))
        self.assertIn("not the dosimetry image", problem(**dict(base, isCaseImage=False)))
        self.assertIn("type of the dosimetry image", problem(**dict(base, caseImageType=None)))
        self.assertIn("MAA activity", problem(**dict(base, caseImageType=R.TYPE_MAA_SPECT), dicomUnits="BQML"))
        self.assertIn("counts", problem(**dict(base, caseImageType=R.TYPE_Y90_SPECT), dicomUnits="CNTS"))
        self.assertEqual(problem(**dict(base, caseImageType=R.TYPE_Y90_SPECT), voxelUnits="Bq/ml"), "")
        self.assertIn("SUV", problem(**base, voxelUnits="{SUVbw}g/ml"))


if __name__ == "__main__":
    unittest.main()
