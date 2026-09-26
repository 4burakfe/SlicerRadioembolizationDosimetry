"""Unit tests of the Taranis rules that do not need Slicer: roles, workflow validators, toolbar visibility.

Run outside Slicer:  python -m unittest discover -s Taranis/Testing/Python
Inside Slicer they run as part of the Taranis module test.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from TaranisLib import roles as R            # noqa: E402
from TaranisLib import workflow as W         # noqa: E402
from TaranisLib.visibility import ToolbarVisibility  # noqa: E402

BOX = (-200.0, 200.0, -200.0, 200.0, -150.0, 150.0)


def vol(nodeID, name="", **kwargs):
    kwargs.setdefault("bounds", BOX)
    return R.VolumeInfo(nodeID=nodeID, name=name or nodeID, **kwargs)


class ClassificationTest(unittest.TestCase):

    def test_modalityFromDicom(self):
        self.assertEqual(R.guessModality(vol("a", modality="PT")), "PT")

    def test_modalityFromName(self):
        self.assertEqual(R.guessModality(vol("a", "Patient 2 SPECT")), "NM")
        self.assertEqual(R.guessModality(vol("a", "Patient 2 CT", minValue=-1024)), "CT")
        self.assertEqual(R.guessModality(vol("a", "T1 VIBE arterial")), "MR")
        self.assertEqual(R.guessModality(vol("a", "Y90 PET/CT", minValue=0.0)), "PT")
        self.assertEqual(R.guessModality(vol("a", "Y90 PET/CT", minValue=-1000.0)), "CT")
        self.assertEqual(R.guessModality(vol("a", "Volume 3", minValue=-1000.0)), "CT")
        self.assertEqual(R.guessModality(vol("a", "Volume 3", minValue=0.0)), "")

    def test_types(self):
        self.assertEqual(R.guessType(vol("a", "SPECT MAA liver")), R.TYPE_MAA_SPECT)
        self.assertEqual(R.guessType(vol("a", "SPECT bremsstrahlung")), R.TYPE_Y90_SPECT)
        self.assertEqual(R.guessType(vol("a", "", modality="PT", radionuclide="^90^Yttrium")), R.TYPE_Y90_PET)
        self.assertEqual(R.guessType(vol("a", "", modality="PT", radiopharmaceutical="Fludeoxyglucose")),
                         R.TYPE_FDG_PET)
        self.assertEqual(R.guessType(vol("a", "Ga68 DOTATATE PET")), R.TYPE_DOTATATE_PET)
        self.assertIsNone(R.guessType(vol("a", "PET WB")))
        self.assertEqual(R.guessType(vol("a", "", modality="MR")), R.TYPE_MRI)

    def test_modes(self):
        self.assertFalse(R.modeAllowed(R.TYPE_MAA_SPECT, R.MODE_ABSOLUTE))
        self.assertTrue(R.modeAllowed(R.TYPE_MAA_SPECT, R.MODE_RELATIVE))
        self.assertTrue(R.modeAllowed(R.TYPE_Y90_SPECT, R.MODE_ABSOLUTE))
        self.assertEqual(R.defaultMode(R.TYPE_Y90_PET), R.MODE_ABSOLUTE)
        self.assertEqual(R.defaultMode(R.TYPE_Y90_SPECT), R.MODE_RELATIVE)
        self.assertEqual(R.scenarioForType(R.TYPE_MAA_SPECT), R.SCENARIO_PRE)
        self.assertEqual(R.scenarioForType(R.TYPE_Y90_PET), R.SCENARIO_POST)

    def test_dicomDateTime(self):
        self.assertEqual(R.dicomDateTime("20260301", "101502.123"), "2026-03-01T10:15:02")
        self.assertEqual(R.dicomDateTime("2026", "10"), "")
        self.assertEqual(R.dicomDateTime("20260301", ""), "2026-03-01T00:00:00")


class SuggestionTest(unittest.TestCase):

    def test_sampleDataWithoutDicom(self):
        """Sample data 2: CT + SPECT as NRRD files -> hybrid pair guessed from geometry."""
        s = R.suggestAssignments([vol("ct", "Patient 2 CT", minValue=-1024), vol("spect", "Patient 2 SPECT",
                                                                               minValue=0)])
        self.assertEqual(s.assignments[R.ROLE_DOSIMETRY], "spect")
        self.assertEqual(s.assignments[R.ROLE_DOSIMETRY_ANATOMY], "ct")
        self.assertNotIn(R.ROLE_REFERENCE, s.assignments)
        self.assertIsNone(s.types[R.ROLE_DOSIMETRY])  # the SPECT type cannot be told from the name
        self.assertTrue(any("type" in n for n in s.notes))

    def test_mriIsReference(self):
        s = R.suggestAssignments([vol("mri", "Liver MRI"), vol("pet", "Y90 PET")])
        self.assertEqual(s.assignments[R.ROLE_DOSIMETRY], "pet")
        self.assertEqual(s.types[R.ROLE_DOSIMETRY], R.TYPE_Y90_PET)
        self.assertEqual(s.assignments[R.ROLE_REFERENCE], "mri")

    def test_fullDicomCase(self):
        infos = [
            vol("maa", "NM", modality="NM", radiopharmaceutical="Tc99m MAA", frameOfReferenceUID="1.1", fromDicom=True,
                acquisitionDateTime="2026-03-01T10:00:00", studyUID="s1"),
            vol("lowct", "CT", modality="CT", frameOfReferenceUID="1.1", fromDicom=True,
                acquisitionDateTime="2026-03-01T10:20:00", studyUID="s1"),
            vol("dx", "CT portal venous", modality="CT", frameOfReferenceUID="2.2", fromDicom=True,
                acquisitionDateTime="2026-02-10T09:00:00"),
            vol("fdg", "PET", modality="PT", radiopharmaceutical="FDG", frameOfReferenceUID="3.3", fromDicom=True),
            vol("fdgct", "CT", modality="CT", frameOfReferenceUID="3.3", fromDicom=True),
        ]
        s = R.suggestAssignments(infos)
        self.assertEqual(s.assignments, {R.ROLE_DOSIMETRY: "maa", R.ROLE_DOSIMETRY_ANATOMY: "lowct",
                                         R.ROLE_REFERENCE: "dx", R.ROLE_METABOLIC: "fdg",
                                         R.ROLE_METABOLIC_ANATOMY: "fdgct"})
        self.assertEqual(s.types[R.ROLE_DOSIMETRY], R.TYPE_MAA_SPECT)
        self.assertEqual(s.types[R.ROLE_METABOLIC], R.TYPE_FDG_PET)

    def test_differentFrameIsNotPaired(self):
        s = R.suggestAssignments([
            vol("spect", "SPECT MAA", modality="NM", frameOfReferenceUID="1", fromDicom=True),
            vol("ct", "CT", modality="CT", frameOfReferenceUID="2", fromDicom=True)])
        self.assertNotIn(R.ROLE_DOSIMETRY_ANATOMY, s.assignments)
        self.assertEqual(s.assignments[R.ROLE_REFERENCE], "ct")

    def test_caseIdentity(self):
        self.assertEqual(R.suggestCaseIdentity([vol("a"), vol("b", patientName="DOE^JOHN", patientID="123")]),
                         ("DOE JOHN", "123"))
        self.assertEqual(R.suggestCaseIdentity([vol("a")]), ("", ""))


def snapshot(**kwargs):
    s = W.CaseSnapshot()
    for key, value in kwargs.items():
        setattr(s, key, value)
    if s.roles and not s.jobs:
        s.primaryRole, s.jobs = W.planRegistration(s.roles)
    return s


class DataStepTest(unittest.TestCase):

    def test_empty(self):
        self.assertEqual(W.evaluateData(snapshot()).state, W.STATE_NOT_STARTED)

    def test_requiredImages(self):
        status = W.evaluateData(snapshot(roles={R.ROLE_DOSIMETRY: vol("s")},
                                         roleTypes={R.ROLE_DOSIMETRY: R.TYPE_MAA_SPECT}))
        self.assertEqual(status.state, W.STATE_ERROR)
        self.assertTrue(any("anatomical image is needed" in i.text for i in status.issues))

    def test_minimalValidCase(self):
        status = W.evaluateData(snapshot(
            roles={R.ROLE_DOSIMETRY: vol("s"), R.ROLE_DOSIMETRY_ANATOMY: vol("c")},
            roleTypes={R.ROLE_DOSIMETRY: R.TYPE_Y90_PET, R.ROLE_DOSIMETRY_ANATOMY: R.TYPE_CT},
            mode=R.MODE_RELATIVE))
        self.assertEqual(status.state, W.STATE_DONE)

    def test_absoluteNeedsQuantitativeImage(self):
        roles = {R.ROLE_DOSIMETRY: vol("s", units="CNTS"), R.ROLE_REFERENCE: vol("c")}
        types = {R.ROLE_DOSIMETRY: R.TYPE_Y90_PET, R.ROLE_REFERENCE: R.TYPE_CT}
        status = W.evaluateData(snapshot(roles=roles, roleTypes=types, mode=R.MODE_ABSOLUTE))
        self.assertEqual(status.state, W.STATE_ERROR)
        roles[R.ROLE_DOSIMETRY] = vol("s", units="BQML")
        status = W.evaluateData(snapshot(roles=roles, roleTypes=types, mode=R.MODE_ABSOLUTE))
        self.assertTrue(all(i.severity != W.SEVERITY_ERROR for i in status.issues))

    def test_maaAbsoluteIsError(self):
        status = W.evaluateData(snapshot(
            roles={R.ROLE_DOSIMETRY: vol("s"), R.ROLE_DOSIMETRY_ANATOMY: vol("c")},
            roleTypes={R.ROLE_DOSIMETRY: R.TYPE_MAA_SPECT, R.ROLE_DOSIMETRY_ANATOMY: R.TYPE_CT},
            mode=R.MODE_ABSOLUTE))
        self.assertEqual(status.state, W.STATE_ERROR)

    def test_warnings(self):
        status = W.evaluateData(snapshot(
            roles={R.ROLE_DOSIMETRY: vol("s", acquisitionDateTime="2026-06-01T10:00:00"),
                   R.ROLE_REFERENCE: vol("r", acquisitionDateTime="2026-01-01T10:00:00"),
                   R.ROLE_METABOLIC: vol("m")},
            roleTypes={R.ROLE_DOSIMETRY: R.TYPE_MAA_SPECT, R.ROLE_REFERENCE: R.TYPE_MRI,
                       R.ROLE_METABOLIC: R.TYPE_FDG_PET},
            mode=R.MODE_RELATIVE))
        self.assertEqual(status.state, W.STATE_WARNING)
        texts = " ".join(i.text for i in status.issues)
        self.assertIn("functional-only", texts)
        self.assertIn("151 days before", texts)

    def test_duplicates(self):
        status = W.evaluateData(snapshot(roles={R.ROLE_DOSIMETRY: vol("s"), R.ROLE_REFERENCE: vol("c")},
                                         roleTypes={R.ROLE_DOSIMETRY: R.TYPE_MAA_SPECT},
                                         duplicateNodes=[("c", ["A", "B"])]))
        self.assertEqual(status.state, W.STATE_ERROR)


class RegistrationStepTest(unittest.TestCase):

    def test_planHybridOnly(self):
        primary, jobs = W.planRegistration({R.ROLE_DOSIMETRY: vol("s"), R.ROLE_DOSIMETRY_ANATOMY: vol("c")})
        self.assertEqual(primary, R.ROLE_DOSIMETRY_ANATOMY)
        self.assertEqual(jobs, [])

    def test_planWithReferenceAndMetabolic(self):
        primary, jobs = W.planRegistration({
            R.ROLE_DOSIMETRY: vol("s"), R.ROLE_DOSIMETRY_ANATOMY: vol("c"), R.ROLE_REFERENCE: vol("r"),
            R.ROLE_METABOLIC: vol("m")})
        self.assertEqual(primary, R.ROLE_REFERENCE)
        self.assertEqual([(j.key, j.path, j.movingRole) for j in jobs],
                         [(W.JOB_DOSIMETRY, W.PATH_HYBRID, R.ROLE_DOSIMETRY_ANATOMY),
                          (W.JOB_METABOLIC, W.PATH_FUNCTIONAL, R.ROLE_METABOLIC)])

    def test_metabolicToDosimetryCt(self):
        primary, jobs = W.planRegistration({R.ROLE_DOSIMETRY: vol("s"), R.ROLE_DOSIMETRY_ANATOMY: vol("c"),
                                            R.ROLE_METABOLIC: vol("m"), R.ROLE_METABOLIC_ANATOMY: vol("mc")})
        self.assertEqual(primary, R.ROLE_DOSIMETRY_ANATOMY)
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].fixedRole, R.ROLE_DOSIMETRY_ANATOMY)
        self.assertEqual(jobs[0].followerRole, R.ROLE_METABOLIC)

    def test_sameFrameIsAligned(self):
        _, jobs = W.planRegistration({R.ROLE_DOSIMETRY: vol("s", frameOfReferenceUID="1"),
                                      R.ROLE_REFERENCE: vol("r", frameOfReferenceUID="1")})
        self.assertTrue(jobs[0].alignedByAcquisition)

    def test_states(self):
        roles = {R.ROLE_DOSIMETRY: vol("s"), R.ROLE_REFERENCE: vol("r")}
        s = snapshot(roles=roles)
        self.assertEqual(W.evaluateRegistration(s).state, W.STATE_NOT_STARTED)
        s.jobs[0].registered = True
        status = W.evaluateRegistration(s)
        self.assertEqual(status.state, W.STATE_WARNING)  # functional-only
        s = snapshot(roles=roles, registrationSkipped=True)
        status = W.evaluateRegistration(s)
        self.assertEqual(status.state, W.STATE_SKIPPED)
        self.assertEqual(status.count(W.SEVERITY_WARNING), 1)
        s = snapshot(roles={R.ROLE_DOSIMETRY: vol("s"), R.ROLE_DOSIMETRY_ANATOMY: vol("c")})
        self.assertEqual(W.evaluateRegistration(s).state, W.STATE_NOT_APPLICABLE)

    def test_followerWarning(self):
        s = snapshot(roles={R.ROLE_DOSIMETRY: vol("s"), R.ROLE_DOSIMETRY_ANATOMY: vol("c"),
                            R.ROLE_REFERENCE: vol("r")})
        s.jobs[0].registered = True
        self.assertEqual(W.evaluateRegistration(s).state, W.STATE_DONE)
        s.jobs[0].followerFollows = False
        self.assertEqual(W.evaluateRegistration(s).state, W.STATE_WARNING)


class OtherStepsTest(unittest.TestCase):

    def test_segmentRoleGuess(self):
        self.assertEqual(W.guessSegmentRole("Liver"), W.SEGMENT_LIVER)
        self.assertEqual(W.guessSegmentRole("Normal liver"), W.SEGMENT_NORMAL)
        self.assertEqual(W.guessSegmentRole("Tumor 3"), W.SEGMENT_TUMOR)
        self.assertEqual(W.guessSegmentRole("Liver lesion 2"), W.SEGMENT_TUMOR)
        self.assertEqual(W.guessSegmentRole("Perfused volume 1"), W.SEGMENT_PERFUSED)
        self.assertEqual(W.guessSegmentRole("Lungs"), W.SEGMENT_LUNGS)
        self.assertEqual(W.guessSegmentRole("Segment_1"), "")
        self.assertEqual(W.guessSegmentRole("perfused normal"), W.SEGMENT_NORMAL)
        self.assertEqual(W.guessSegmentRole("whole liver normal"), W.SEGMENT_NORMAL)
        self.assertEqual(W.guessSegmentRole("Non-tumoral liver"), W.SEGMENT_NORMAL)
        self.assertEqual(W.guessSegmentRole("Perfused territory 2"), W.SEGMENT_PERFUSED)
        self.assertEqual(W.guessSegmentRole("Viable tumors"), W.SEGMENT_VIABLE)
        self.assertEqual(W.guessSegmentRole("Tumor FDG"), W.SEGMENT_VIABLE)
        self.assertEqual(W.SEGMENT_ROLES[W.SEGMENT_VIABLE][2], (195 / 255.0, 33 / 255.0, 72 / 255.0))

    def test_segmentation(self):
        self.assertEqual(W.evaluateSegmentation(snapshot()).state, W.STATE_NOT_STARTED)
        s = snapshot(segmentationPresent=True, mode=R.MODE_RELATIVE,
                     segments=[W.SegmentInfo("1", "Tumor", W.SEGMENT_TUMOR)])
        self.assertEqual(W.evaluateSegmentation(s).state, W.STATE_ERROR)
        s.segments = [W.SegmentInfo("1", "Liver", W.SEGMENT_LIVER), W.SegmentInfo("2", "Tumor", W.SEGMENT_TUMOR),
                      W.SegmentInfo("4", "Normal liver", W.SEGMENT_NORMAL), W.SegmentInfo("3", "Perfused", W.SEGMENT_PERFUSED)]
        self.assertEqual(W.evaluateSegmentation(s).state, W.STATE_DONE)
        s.mode = R.MODE_ABSOLUTE
        s.segments = s.segments[:3]
        self.assertEqual(W.evaluateSegmentation(s).state, W.STATE_DONE)
        # viable tumours alone count as tumours (no "no tumour" warning)
        viableOnly = W.CaseSnapshot(segmentationPresent=True, mode=R.MODE_ABSOLUTE,
                                    segments=[W.SegmentInfo("1", "Liver", W.SEGMENT_LIVER),
                                              W.SegmentInfo("2", "Viable tumors", W.SEGMENT_VIABLE),
                                              W.SegmentInfo("3", "Normal liver", W.SEGMENT_NORMAL)])
        self.assertEqual(W.evaluateSegmentation(viableOnly).state, W.STATE_DONE)
        # geometry check findings are shown; a stale check is only a note
        s.geometryIssues = [(W.SEVERITY_WARNING, "'Tumor' outside the liver")]
        self.assertEqual(W.evaluateSegmentation(s).state, W.STATE_WARNING)
        s.geometryIssues, s.geometryStale = None, True
        self.assertEqual(W.evaluateSegmentation(s).state, W.STATE_DONE)
        # an AI candidate without a role is a warning (not accepted), not a "no role" note
        s.segments.append(W.SegmentInfo("9", "Whole liver (AI - evaluate)", "", candidate=True))
        issues = W.evaluateSegmentation(s).issues
        self.assertTrue(any("not accepted" in i.text for i in issues))
        self.assertFalse(any("without a role" in i.text for i in issues))

    def test_segmentationGuardrails(self):
        def seg(id_, name, role, ml, empty=False):
            return W.SegmentInfo(id_, name, role, empty=empty, voxels=0 if empty else 100, volumeML=ml)

        def texts(segments, mode=R.MODE_RELATIVE):
            status = W.evaluateSegmentation(W.CaseSnapshot(segmentationPresent=True, mode=mode, segments=segments))
            return status, [(i.severity, i.text) for i in status.issues]

        good = [seg("1", "Whole liver", W.SEGMENT_LIVER, 1600.0), seg("2", "Perfused 1", W.SEGMENT_PERFUSED, 700.0),
                seg("3", "Tumor 1", W.SEGMENT_TUMOR, 40.0), seg("8", "Perfused normal liver", W.SEGMENT_NORMAL, 600.0)]
        status, issues = texts(good)
        self.assertEqual(status.state, W.STATE_DONE, issues)
        # no normal tissue: warning naming the tools
        status, issues = texts(good[:3])
        self.assertEqual(status.state, W.STATE_WARNING)
        self.assertTrue(any(sev == W.SEVERITY_WARNING and "No normal tissue segment" in t and "Perfused normal" in t
                            for sev, t in issues))
        # liver too small / too large
        _, issues = texts([seg("1", "Whole liver", W.SEGMENT_LIVER, 420.0)] + good[1:])
        self.assertTrue(any(sev == W.SEVERITY_WARNING and "less than 500 mL" in t for sev, t in issues))
        _, issues = texts([seg("1", "Whole liver", W.SEGMENT_LIVER, 5200.0)] + good[1:])
        self.assertTrue(any("more than 4500 mL" in t for _, t in issues))
        # small perfused volume
        _, issues = texts([good[0], seg("2", "Perfused 1", W.SEGMENT_PERFUSED, 80.0), good[2], good[3]])
        self.assertTrue(any(sev == W.SEVERITY_WARNING and "'Perfused 1' is 80 mL" in t for sev, t in issues))
        # no tumour: warning that says the other segments can still be calculated
        _, issues = texts(good[:2] + good[3:])
        self.assertTrue(any(sev == W.SEVERITY_WARNING and "can still be calculated" in t for sev, t in issues))
        # empty segment, segment without a role (now a warning), duplicate names
        status, issues = texts(good + [seg("4", "Tumor 2", W.SEGMENT_TUMOR, None, empty=True),
                                       seg("5", "Segment_5", "", 12.0), seg("6", "Tumor 1", W.SEGMENT_TUMOR, 5.0)])
        self.assertEqual(status.state, W.STATE_WARNING)
        self.assertTrue(any(sev == W.SEVERITY_WARNING and "Empty segment" in t for sev, t in issues))
        self.assertTrue(any(sev == W.SEVERITY_WARNING and "without a role" in t for sev, t in issues))
        self.assertTrue(any(sev == W.SEVERITY_WARNING and "same name: 'Tumor 1'" in t for sev, t in issues))
        # small tumours: a note only
        status, issues = texts(good + [seg("7", "Tumor 3", W.SEGMENT_TUMOR, 1.2)])
        self.assertEqual(status.state, W.STATE_DONE)
        self.assertTrue(any(sev == W.SEVERITY_INFO and "partial volume" in t for sev, t in issues))
        # unknown volumes (no binary labelmap): no volume warnings
        status, issues = texts([seg("1", "Whole liver", W.SEGMENT_LIVER, None), seg("2", "Perfused 1", W.SEGMENT_PERFUSED, None),
                                seg("3", "Tumor 1", W.SEGMENT_TUMOR, None), seg("8", "Normal", W.SEGMENT_NORMAL, None)])
        self.assertEqual(status.state, W.STATE_DONE, issues)

    def test_segmentsKey(self):
        a = [W.SegmentInfo("b", "B", voxels=5), W.SegmentInfo("a", "A", voxels=3)]
        self.assertEqual(W.segmentsKey(a), "a:3;b:5")
        self.assertNotEqual(W.segmentsKey(a), W.segmentsKey([W.SegmentInfo("a", "A", voxels=4), a[0]]))

    def test_lsf(self):
        pre = {R.ROLE_DOSIMETRY: R.TYPE_MAA_SPECT}
        self.assertEqual(W.evaluateLsf(snapshot(roleTypes=pre, mode=R.MODE_RELATIVE)).state, W.STATE_NOT_STARTED)
        self.assertEqual(W.evaluateLsf(snapshot(roleTypes=pre, mode=R.MODE_RELATIVE, lsfValue=6.0)).state,
                         W.STATE_DONE)
        self.assertEqual(W.evaluateLsf(snapshot(roleTypes=pre, mode=R.MODE_RELATIVE, lsfValue=12.0)).state,
                         W.STATE_WARNING)
        skipped = W.evaluateLsf(snapshot(roleTypes=pre, mode=R.MODE_RELATIVE, lsfSkipped=True))
        self.assertEqual(skipped.state, W.STATE_SKIPPED)
        self.assertEqual(skipped.count(W.SEVERITY_WARNING), 1)
        post = {R.ROLE_DOSIMETRY: R.TYPE_Y90_PET}
        self.assertEqual(W.evaluateLsf(snapshot(roleTypes=post, mode=R.MODE_ABSOLUTE)).state, W.STATE_NOT_APPLICABLE)
        self.assertEqual(W.evaluateLsf(snapshot(roleTypes=post, mode=R.MODE_RELATIVE, lsfSkipped=True)).count(
            W.SEVERITY_WARNING), 0)

    def test_dosimetryAndReport(self):
        module = R.MODE_MODULES[R.MODE_RELATIVE]
        self.assertEqual(W.evaluateDosimetry(snapshot(mode=R.MODE_RELATIVE)).state, W.STATE_NOT_STARTED)
        self.assertEqual(W.evaluateDosimetry(snapshot(mode=R.MODE_RELATIVE, dosimetryResultsModule=module)).state,
                         W.STATE_DONE)
        self.assertEqual(W.evaluateDosimetry(snapshot(mode=R.MODE_RELATIVE, dosimetryResultsModule=module,
                                                      dosimetryOutdated=True)).state, W.STATE_OUTDATED)
        self.assertEqual(W.evaluateDosimetry(snapshot(mode=R.MODE_ABSOLUTE, dosimetryResultsModule=module)).state,
                         W.STATE_NOT_STARTED)
        self.assertEqual(W.evaluateReport(snapshot(reportSaved=True, reportOutdated=True)).state, W.STATE_OUTDATED)
        # dose checks stored with the calculation: warnings on the toolbar
        checks = [(W.SEVERITY_WARNING, "High lung dose"), (W.SEVERITY_INFO, "Tumour-to-normal ratio not checked")]
        status = W.evaluateDosimetry(snapshot(mode=R.MODE_RELATIVE, dosimetryResultsModule=module,
                                              dosimetryChecks=checks))
        self.assertEqual(status.state, W.STATE_WARNING)
        self.assertEqual((status.count(W.SEVERITY_WARNING), status.count(W.SEVERITY_INFO)), (1, 1))
        self.assertIn("1 dose check(s)", status.summary)
        info = W.evaluateDosimetry(snapshot(mode=R.MODE_RELATIVE, dosimetryResultsModule=module,
                                            dosimetryChecks=checks[1:]))
        self.assertEqual(info.state, W.STATE_DONE)
        # outdated results: the old checks are not shown
        outdated = W.evaluateDosimetry(snapshot(mode=R.MODE_RELATIVE, dosimetryResultsModule=module,
                                                dosimetryOutdated=True, dosimetryChecks=checks))
        self.assertFalse(any("lung" in issue.text for issue in outdated.issues))

    def test_locksAndNextStep(self):
        s = snapshot(roles={R.ROLE_DOSIMETRY: vol("s")}, roleTypes={R.ROLE_DOSIMETRY: R.TYPE_MAA_SPECT},
                     mode=R.MODE_RELATIVE)
        statuses = W.evaluateAll(s)
        self.assertEqual(statuses[W.STEP_DATA].state, W.STATE_ERROR)
        self.assertEqual(statuses[W.STEP_SEGMENTATION].state, W.STATE_LOCKED)
        self.assertEqual(statuses[W.STEP_REPORT].state, W.STATE_LOCKED)
        self.assertEqual(W.nextStep(statuses), W.STEP_DATA)
        self.assertTrue(all(issue.step for key in W.STEP_KEYS for issue in statuses[key].issues))
        self.assertEqual(W.allIssues(statuses)[0].severity, W.SEVERITY_ERROR)


class VisibilityTest(unittest.TestCase):

    def test_firstUse(self):
        v = ToolbarVisibility(initialized=False, showAtStartup=True)
        self.assertFalse(v.visible)          # never used: nothing at startup
        self.assertTrue(v.onHubOpened())     # first use adds it
        self.assertTrue(v.visible)
        self.assertFalse(v.onHubOpened())

    def test_startupAndClose(self):
        v = ToolbarVisibility(initialized=True, showAtStartup=True)
        self.assertTrue(v.visible)
        v.onClosePressed()
        self.assertFalse(v.visible)
        v.onHubOpened()
        self.assertFalse(v.visible)          # closed by the user: opening the hub does not force it back
        v.onCaseActivated()
        self.assertTrue(v.visible)           # starting a case does
        v.onCaseDeactivated()
        self.assertTrue(v.visible)           # shown at startup: stays, in the idle state

    def test_disabledAtStartup(self):
        v = ToolbarVisibility(initialized=True, showAtStartup=False)
        self.assertFalse(v.visible)
        v.onCaseActivated()
        self.assertTrue(v.visible)
        v.onCaseDeactivated()
        self.assertFalse(v.visible)          # vanishes when the case / scene is closed


if __name__ == "__main__":
    unittest.main()
