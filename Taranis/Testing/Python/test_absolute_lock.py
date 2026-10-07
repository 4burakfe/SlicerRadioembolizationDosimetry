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

    def test_voxel_units_win_over_dicom_header(self):
        # A PET loaded as SUV keeps the BQML Units of its series: the SUV array must still lock absolute mode
        for suv in ("{SUVbw}g/ml", "{SUVlbm}g/ml", "{SUVbsa}cm2/ml"):
            self.assertIn("SUV", problem(voxelUnits=suv, dicomUnits="BQML"), suv)
        self.assertEqual(problem(voxelUnits="Bq/ml", dicomUnits="GML"), "")
        self.assertEqual(problem(voxelUnits="Bq/ml", dicomUnits="CNTS"), "")

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
        self.assertIn("SUV", problem(**base, voxelUnits="{SUVbw}g/ml", dicomUnits="BQML"))


    def test_radionuclide(self):
        self.assertEqual(R.radionuclideProblem("PT", 230400.0, ""), "")      # scanners round Y-90 differently
        self.assertEqual(R.radionuclideProblem("PT", R.Y90_HALF_LIFE_S, "Fluorine-18"), "")   # half-life decides
        self.assertIn("F-18", R.radionuclideProblem("PT", 6586.2, ""))
        self.assertIn("109.8 min", R.radionuclideProblem("PT", 6586.2, ""))
        self.assertIn("Zr-89", R.radionuclideProblem("PT", 282276.0, ""))
        self.assertIn("half-life of", R.radionuclideProblem("PT", 1000000.0, ""))
        # Without a half-life the names are used, and an unknown or missing name does not lock
        self.assertEqual(R.radionuclideProblem("PT", None, "^90^Yttrium"), "")
        self.assertIn("Fluorine", R.radionuclideProblem("PT", None, "^18^Fluorine Fludeoxyglucose"))
        self.assertEqual(R.radionuclideProblem("PT", None, ""), "")
        self.assertEqual(R.radionuclideProblem("PT", None, "Unknown"), "")
        # Bremsstrahlung SPECT is often acquired with a Tc-99m protocol: not checked
        self.assertEqual(R.radionuclideProblem("NM", 21624.0, "Technetium-99m"), "")

    def test_wrong_radionuclide_locks_absolute(self):
        fdg = dict(modality="PT", halfLifeSeconds=6586.2, nuclideText="^18^Fluorine FDG")
        # FDG PET loaded as a scalar volume: BQML header, no voxel units
        self.assertIn("F-18", problem(dicomUnits="BQML", **fdg))
        self.assertIn("F-18", problem(voxelUnits="Bq/ml", **fdg))
        base = dict(hasCase=True, caseImageName="PET", isCaseImage=True, caseImageType=R.TYPE_Y90_PET)
        self.assertIn("F-18", problem(**base, dicomUnits="BQML", **fdg))
        self.assertEqual(problem(dicomUnits="BQML", modality="PT", halfLifeSeconds=230580.0), "")
        # Without DICOM metadata (e.g. a hand-corrected NRRD) nothing is known: not locked by the nuclide
        self.assertEqual(problem(**base), "")


if __name__ == "__main__":
    unittest.main()
