"""Workflow steps, step states and the validators that turn a case snapshot into step statuses.

Pure Python (no Slicer imports): the Slicer side (controller.py) builds a CaseSnapshot from the scene and
these functions decide what the toolbar shows. Gating is soft: a step is only "locked" for display, it can
always be opened. Errors are reserved for situations in which a calculation would be impossible or wrong.
"""

import dataclasses

from . import roles as R

# -- Steps ---------------------------------------------------------------------------------------------------

STEP_DATA = "data"
STEP_REGISTRATION = "registration"
STEP_SEGMENTATION = "segmentation"
STEP_PLANNING = "planning"
STEP_LSF = "lsf"
STEP_DOSIMETRY = "dosimetry"
STEP_REPORT = "report"

STEPS = [
    (STEP_DATA, "Data"),
    (STEP_REGISTRATION, "Registration"),
    (STEP_SEGMENTATION, "Segmentation"),
    (STEP_PLANNING, "Planning"),
    (STEP_LSF, "LSF"),
    (STEP_DOSIMETRY, "Dosimetry"),
    (STEP_REPORT, "Report"),
]
STEP_KEYS = [key for key, _ in STEPS]
STEP_LABELS = dict(STEPS)

# -- States --------------------------------------------------------------------------------------------------

STATE_LOCKED = "locked"
STATE_NOT_STARTED = "notStarted"
STATE_IN_PROGRESS = "inProgress"
STATE_DONE = "done"
STATE_WARNING = "warning"            # done, with warnings
STATE_ERROR = "error"
STATE_SKIPPED = "skipped"            # skipped by the user
STATE_NOT_APPLICABLE = "notApplicable"
STATE_OUTDATED = "outdated"          # an input changed after the step was completed

# state -> (label, badge colour, badge glyph)
STATE_STYLE = {
    STATE_LOCKED: ("Waiting for an earlier step", "#9aa0a6", "–"),
    STATE_NOT_STARTED: ("Not started", "#9aa0a6", ""),
    STATE_IN_PROGRESS: ("In progress", "#2f7ed8", "…"),
    STATE_DONE: ("Done", "#2e9e44", "✓"),
    STATE_WARNING: ("Done with warnings", "#e0a100", "!"),
    STATE_ERROR: ("Error", "#d63b3b", "✕"),
    STATE_SKIPPED: ("Skipped", "#7b8794", "»"),
    STATE_NOT_APPLICABLE: ("Not needed", "#c3c7cb", "–"),
    STATE_OUTDATED: ("Outdated: inputs changed", "#e67e22", "↻"),
}

# States in which the step does not hold up the workflow
FINISHED_STATES = (STATE_DONE, STATE_WARNING, STATE_SKIPPED, STATE_NOT_APPLICABLE)

SEVERITY_ERROR = "error"
SEVERITY_WARNING = "warning"
SEVERITY_INFO = "info"
SEVERITY_ORDER = {SEVERITY_ERROR: 0, SEVERITY_WARNING: 1, SEVERITY_INFO: 2}


@dataclasses.dataclass
class Issue:
    severity: str
    text: str
    step: str = ""


@dataclasses.dataclass
class StepStatus:
    state: str
    summary: str = ""
    issues: list = dataclasses.field(default_factory=list)

    def count(self, severity):
        return sum(1 for issue in self.issues if issue.severity == severity)


def _stateFromIssues(issues, okState=STATE_DONE):
    if any(i.severity == SEVERITY_ERROR for i in issues):
        return STATE_ERROR
    if any(i.severity == SEVERITY_WARNING for i in issues):
        return STATE_WARNING
    return okState


# -- Registration plan ----------------------------------------------------------------------------------------

PATH_HYBRID = "hybrid"          # anatomy -> anatomy, the functional image follows
PATH_FUNCTIONAL = "functional"  # functional image registered directly (no anatomical image of its own)

JOB_DOSIMETRY = "dosimetry"
JOB_METABOLIC = "metabolic"
JOB_CBCT = "cbct"
JOB_CBCT_PARENCHYMAL = "cbctParenchymal"
CBCT_JOBS = (JOB_CBCT, JOB_CBCT_PARENCHYMAL)
JOB_AUXILIARY = "auxiliary"   # + number (1, 2, 3)


@dataclasses.dataclass
class RegistrationJob:
    key: str
    label: str
    movingRole: str
    fixedRole: str
    followerRole: str = None
    path: str = PATH_HYBRID
    alignedByAcquisition: bool = False  # same DICOM frame of reference: nothing to do
    registered: bool = False            # an EasyReg / Taranis alignment transform exists for the moving image
    method: str = ""
    followerFollows: bool = True        # the functional image moves with its anatomical image
    checked: bool = False               # the user confirmed the alignment
    optional: bool = False              # auxiliary volume: does not hold up the workflow

    @property
    def complete(self):
        return self.alignedByAcquisition or self.registered or self.checked


def primaryRole(roleInfos):
    """Role of the image that defines the primary space (segmentations, display)."""
    if roleInfos.get(R.ROLE_REFERENCE):
        return R.ROLE_REFERENCE
    if roleInfos.get(R.ROLE_DOSIMETRY_ANATOMY):
        return R.ROLE_DOSIMETRY_ANATOMY
    if roleInfos.get(R.ROLE_DOSIMETRY):
        return R.ROLE_DOSIMETRY
    return None


def _sameFrame(a, b):
    return bool(a and b and a.frameOfReferenceUID and a.frameOfReferenceUID == b.frameOfReferenceUID)


def planRegistration(roleInfos):
    """(primary role, [RegistrationJob]) from the assigned roles {role: VolumeInfo}."""
    primary = primaryRole(roleInfos)
    jobs = []
    if primary is None:
        return None, jobs
    fixed = roleInfos.get(primary)
    pairs = [(JOB_DOSIMETRY, "Dosimetry image", R.ROLE_DOSIMETRY, R.ROLE_DOSIMETRY_ANATOMY),
             (JOB_METABOLIC, "Metabolic image", R.ROLE_METABOLIC, R.ROLE_METABOLIC_ANATOMY)]
    for key, label, functionalRole, anatomyRole in pairs:
        functional = roleInfos.get(functionalRole)
        anatomy = roleInfos.get(anatomyRole)
        if functional is None and anatomy is None:
            continue
        if functional is None:
            continue  # an anatomical image without its functional image is not used
        if primary in (functionalRole, anatomyRole):
            continue  # this pair defines the primary space
        if anatomy is not None:
            job = RegistrationJob(key, label, anatomyRole, primary, functionalRole, PATH_HYBRID)
            job.alignedByAcquisition = _sameFrame(anatomy, fixed)
        else:
            job = RegistrationJob(key, label, functionalRole, primary, None, PATH_FUNCTIONAL)
            job.alignedByAcquisition = _sameFrame(functional, fixed)
        jobs.append(job)
    # CBCT: anatomy to anatomy (rigid, inside the liver). A parenchymal phase acquired in the same run (same frame
    # of reference) follows the arterial CBCT; otherwise it is registered on its own.
    arterial = roleInfos.get(R.ROLE_CBCT)
    parenchymal = roleInfos.get(R.ROLE_CBCT_PARENCHYMAL)
    if arterial is not None:
        follows = parenchymal is not None and _sameFrame(arterial, parenchymal)
        job = RegistrationJob(JOB_CBCT, "CBCT", R.ROLE_CBCT, primary,
                              R.ROLE_CBCT_PARENCHYMAL if follows else None, PATH_HYBRID)
        job.alignedByAcquisition = _sameFrame(arterial, fixed)
        jobs.append(job)
        if follows:
            parenchymal = None
    if parenchymal is not None:
        job = RegistrationJob(JOB_CBCT_PARENCHYMAL, "Parenchymal CBCT", R.ROLE_CBCT_PARENCHYMAL, primary, None,
                              PATH_HYBRID)
        job.alignedByAcquisition = _sameFrame(parenchymal, fixed)
        jobs.append(job)
    # Auxiliary volumes: optional, anatomy to anatomy, or functional-only for a SPECT / PET
    for index, role in enumerate(R.AUXILIARY_ROLES):
        info = roleInfos.get(role)
        if info is None:
            continue
        path = PATH_FUNCTIONAL if R.isFunctional(info) else PATH_HYBRID
        job = RegistrationJob(f"{JOB_AUXILIARY}{index + 1}", f"Auxiliary volume {index + 1}", role, primary, None, path,
                              optional=True)
        job.alignedByAcquisition = _sameFrame(info, fixed)
        jobs.append(job)
    return primary, jobs


# -- Segments ---------------------------------------------------------------------------------------------------

SEGMENT_LIVER = "liver"
SEGMENT_PERFUSED = "perfused"
SEGMENT_TUMOR = "tumor"
SEGMENT_VIABLE = "viable"   # viable (e.g. FDG-avid) tumour: dosimetry like tumours, reported separately, may overlap them
SEGMENT_NORMAL = "normal"
SEGMENT_LUNGS = "lungs"
SEGMENT_VESSELS = "vessels"   # arterial tree from the CBCT: ignored by dosimetry
SEGMENT_OTHER = "other"

# role -> (label, standard name for new segments, colour)
SEGMENT_ROLES = {
    SEGMENT_LIVER: ("Whole liver", "Whole liver", (1.0, 1.0, 1.0)),
    SEGMENT_PERFUSED: ("Perfused volume", "Perfused volume", (1.0, 0.0, 0.0)),
    SEGMENT_TUMOR: ("Tumour", "Tumor", (0.70, 0.55, 1.00)),
    SEGMENT_VIABLE: ("Viable tumour", "Viable tumor", (195 / 255.0, 33 / 255.0, 72 / 255.0)),   # #c32148
    SEGMENT_NORMAL: ("Normal tissue", "Normal liver", (0.25, 0.88, 0.82)),
    SEGMENT_LUNGS: ("Lungs", "Lungs", (0.45, 0.65, 1.00)),
    SEGMENT_VESSELS: ("Vessels", "Arterial tree", (0.86, 0.16, 0.16)),
    SEGMENT_OTHER: ("Other", "Other", (0.6, 0.6, 0.6)),
}
SEGMENT_ROLE_KEYS = list(SEGMENT_ROLES)


def perfusedColor(key):
    """A red of its own for every perfused volume: full red, plus a little green and / or blue (together at most
    0.3), chosen from the segment ID so it stays the same."""
    import zlib
    value = zlib.crc32(str(key).encode("utf-8"))
    total = 0.08 + 0.22 * ((value % 1000) / 999.0)          # green + blue: 0.08 .. 0.30
    share = ((value // 1000) % 1000) / 999.0                 # how it is split between green and blue
    return (1.0, round(total * share, 4), round(total * (1.0 - share), 4))

# Normal tissue segments: whole normal liver (liver − tumours) or perfused normal liver (perfused volume − tumours).
# Tag on the segment, read by the dose checks (tumour-to-normal ratio against the perfused normal liver).
NORMAL_SCOPE_TAG = "Taranis.NormalScope"
NORMAL_SOURCE_TAG = "Taranis.NormalSource"     # perfused normal: ID of its perfused volume segment
NORMAL_SCOPE_WHOLE = "whole"
NORMAL_SCOPE_PERFUSED = "perfused"


def guessSegmentRole(name):
    """Segment role from its name ("" if unclear)."""
    text = (name or "").lower()
    if "lung" in text:
        return SEGMENT_LUNGS
    if any(word in text for word in ("arter", "vessel", "vascul")):
        return SEGMENT_VESSELS
    if "viable" in text or "fdg" in text or "metabolic" in text:
        return SEGMENT_VIABLE
    if any(word in text for word in ("tumor", "tumour", "lesion", "hcc", "metasta", "nodule")) \
            and not any(word in text for word in ("non-tumo", "nontumo")):
        return SEGMENT_TUMOR
    # before "perfused": "perfused normal" is normal tissue inside a perfused volume
    if any(word in text for word in ("normal", "healthy", "non-tumo", "nontumo", "parenchyma")):
        return SEGMENT_NORMAL
    if "perfus" in text or "territor" in text:
        return SEGMENT_PERFUSED
    if "liver" in text:
        return SEGMENT_LIVER
    return ""


@dataclasses.dataclass
class SegmentInfo:
    segmentID: str
    name: str
    role: str = ""
    empty: bool = False
    candidate: bool = False   # AI result not accepted yet
    voxels: int = None        # None: unknown (no binary labelmap)
    volumeML: float = None


def segmentsKey(segments):
    """Identifies the segment contents (IDs and voxel counts); a stored geometry check is valid for this key."""
    return ";".join(f"{x.segmentID}:{x.voxels}" for x in sorted(segments, key=lambda x: x.segmentID))


# -- Snapshot ---------------------------------------------------------------------------------------------------

@dataclasses.dataclass
class CaseSnapshot:
    """Everything the validators need, collected from the scene by the controller."""
    caseName: str = ""
    caseID: str = ""
    roles: dict = dataclasses.field(default_factory=dict)       # role -> VolumeInfo
    roleTypes: dict = dataclasses.field(default_factory=dict)   # role -> image type
    mode: str = ""
    duplicateNodes: list = dataclasses.field(default_factory=list)  # [(name, [role labels])]
    registrationSkipped: bool = False
    primaryRole: str = None
    jobs: list = dataclasses.field(default_factory=list)
    segmentationPresent: bool = False
    segments: list = dataclasses.field(default_factory=list)
    geometryIssues: list = None     # [(severity, text)] of a geometry check valid for the current segments
    geometryStale: bool = False     # a check exists but the segments changed since
    lsfValue: float = None
    lsfSource: str = ""
    lsfSkipped: bool = False
    lsfLungMassG: float = None
    lsfFromImage: bool = False          # calculated by the hub from the dosimetry image
    lsfOutdated: bool = False           # its inputs (image, lung / liver segments) changed since
    lsfIssues: list = dataclasses.field(default_factory=list)   # [(severity, text)] stored with the calculation
    plannedActivityGBq: float = None
    dosimetryResultsModule: str = ""
    dosimetryOutdated: bool = False
    dosimetryChecks: list = dataclasses.field(default_factory=list)   # [(severity, text)] of the last calculation
    reportSaved: bool = False
    reportOutdated: bool = False
    studyIntervalLimitDays: int = 60
    planningSkipped: bool = False
    planningHasInjection: bool = False      # injection point placed
    planningHasTree: bool = False           # arterial tree extracted
    planningTips: int = 0                   # planned catheter positions
    planningResults: dict = None            # stored result of the last territory prediction (see planningFindings)
    planningOutdated: bool = False          # tree, markers or liver / tumour segments changed since
    planningReviewed: bool = False          # extrahepatic findings of these results marked as reviewed

    @property
    def dosimetryType(self):
        return self.roleTypes.get(R.ROLE_DOSIMETRY)

    @property
    def scenario(self):
        return R.scenarioForType(self.dosimetryType)


# -- Validators ---------------------------------------------------------------------------------------------------

def _roleLabel(role):
    return R.ROLE_INFO[role][0]


WRONG_RADIONUCLIDE = "Wrong radionuclide:"


def evaluateData(s):
    issues = []
    dosimetry = s.roles.get(R.ROLE_DOSIMETRY)
    if not any(s.roles.values()):
        return StepStatus(STATE_NOT_STARTED, "No images assigned yet.",
                          [Issue(SEVERITY_ERROR, "Assign the dosimetry image and an anatomical image.")])
    if dosimetry is None:
        issues.append(Issue(SEVERITY_ERROR, "No dosimetry image (MAA SPECT, Y-90 SPECT or Y-90 PET)."))
    else:
        imageType = s.dosimetryType
        if not imageType:
            issues.append(Issue(SEVERITY_ERROR, f"Select the type of the dosimetry image '{dosimetry.name}'."))
        elif s.mode and not R.modeAllowed(imageType, s.mode):
            issues.append(Issue(SEVERITY_ERROR, f"{R.MODE_LABELS[s.mode]} dosimetry is not possible with "
                                                f"{R.TYPE_LABELS[imageType]}."))
        nuclide = R.radionuclideProblem(dosimetry.modality, dosimetry.radionuclideHalfLife,
                                        f"{dosimetry.radionuclide} {dosimetry.radiopharmaceutical}")
        if nuclide and s.mode == R.MODE_ABSOLUTE:
            issues.append(Issue(SEVERITY_ERROR, f"{WRONG_RADIONUCLIDE} '{dosimetry.name}' was acquired as {nuclide}, "
                                                "not Y-90: its Bq/mL values are not Y-90 activity, absolute dosimetry "
                                                "is not possible. Use patient-relative dosimetry, or reconstruct the "
                                                "PET with Y-90 as the radionuclide."))
        elif nuclide:
            issues.append(Issue(SEVERITY_WARNING, f"{WRONG_RADIONUCLIDE} '{dosimetry.name}' was acquired as "
                                                  f"{nuclide}, not Y-90. Patient-relative dosimetry is still valid "
                                                  "for a single-bed acquisition (multi-bed: the beds are "
                                                  "decay-corrected with the wrong half-life)."))
        if s.mode == R.MODE_ABSOLUTE:
            if R.looksLikeSuv(dosimetry):
                issues.append(Issue(SEVERITY_ERROR, "The dosimetry image is in SUV: absolute dosimetry needs an "
                                                    "activity-concentration image (Bq/mL)."))
            elif dosimetry.units == R.UNITS_COUNTS:
                issues.append(Issue(SEVERITY_ERROR, "The dosimetry image is in counts: absolute dosimetry needs "
                                                    "an activity-concentration image (Bq/mL)."))
            elif dosimetry.units != R.UNITS_QUANTITATIVE:
                issues.append(Issue(SEVERITY_WARNING, "Image units unknown: make sure the dosimetry image is a "
                                                      "quantitative activity-concentration image (Bq/mL)."))
            if imageType == R.TYPE_Y90_SPECT:
                issues.append(Issue(SEVERITY_WARNING, "Absolute dosimetry from Y-90 SPECT is only valid with a "
                                                      "quantitatively calibrated reconstruction."))
    if not s.roles.get(R.ROLE_DOSIMETRY_ANATOMY) and not s.roles.get(R.ROLE_REFERENCE):
        issues.append(Issue(SEVERITY_ERROR, "An anatomical image is needed: the CT/MRI of the dosimetry image or a "
                                            "reference CT/MRI."))
    for name, roleLabels in s.duplicateNodes:
        issues.append(Issue(SEVERITY_ERROR, f"'{name}' is assigned to several roles: {', '.join(roleLabels)}."))

    for role in (R.ROLE_DOSIMETRY_ANATOMY, R.ROLE_REFERENCE, R.ROLE_METABOLIC_ANATOMY):
        if s.roles.get(role) and not s.roleTypes.get(role):
            issues.append(Issue(SEVERITY_WARNING, f"Select CT or MRI for the {_roleLabel(role).lower()}."))

    anatomy = s.roles.get(R.ROLE_DOSIMETRY_ANATOMY)
    if dosimetry and anatomy and dosimetry.frameOfReferenceUID and anatomy.frameOfReferenceUID \
            and dosimetry.frameOfReferenceUID != anatomy.frameOfReferenceUID:
        issues.append(Issue(SEVERITY_WARNING, f"'{dosimetry.name}' and '{anatomy.name}' have different DICOM frames "
                                              "of reference: they may not be a hybrid pair. Check their alignment."))
    if dosimetry and not anatomy and s.roles.get(R.ROLE_REFERENCE):
        issues.append(Issue(SEVERITY_WARNING, "No anatomical image acquired with the dosimetry image: registration to "
                                              "the reference will be functional-only (manual / landmarks)."))
    if s.roles.get(R.ROLE_METABOLIC) and not s.roles.get(R.ROLE_METABOLIC_ANATOMY):
        issues.append(Issue(SEVERITY_WARNING, "Metabolic image without its anatomical image: its registration will be "
                                              "functional-only (less robust)."))
    if s.roles.get(R.ROLE_METABOLIC_ANATOMY) and not s.roles.get(R.ROLE_METABOLIC):
        issues.append(Issue(SEVERITY_INFO, "An anatomical image is assigned for the metabolic image, but no metabolic "
                                           "image: it is not used."))
    if s.roles.get(R.ROLE_METABOLIC) and s.roleTypes.get(R.ROLE_METABOLIC) is None:
        issues.append(Issue(SEVERITY_WARNING, "Select the type of the metabolic image (FDG, DOTATATE, other)."))

    reference = dosimetry.dateTime() if dosimetry else None
    if reference:
        for role in (R.ROLE_REFERENCE, R.ROLE_METABOLIC):
            info = s.roles.get(role)
            when = info.dateTime() if info else None
            if when:
                days = (when - reference).total_seconds() / 86400.0
                if abs(days) > s.studyIntervalLimitDays:
                    relation = "before" if days < 0 else "after"
                    issues.append(Issue(SEVERITY_WARNING, f"The {_roleLabel(role).lower()} was acquired "
                                                          f"{abs(days):.0f} days {relation} the dosimetry image."))

    summaryParts = []
    if s.dosimetryType:
        summaryParts.append(R.TYPE_LABELS[s.dosimetryType])
    if s.mode:
        summaryParts.append(R.MODE_LABELS[s.mode])
    return StepStatus(_stateFromIssues(issues), " · ".join(summaryParts) or "Roles assigned.", issues)


def evaluateRegistration(s):
    issues = []
    jobs = s.jobs
    required = [job for job in jobs if not job.optional]
    pending = [job for job in required if not job.complete]
    optionalPending = [job for job in jobs if job.optional and not job.complete]
    if s.registrationSkipped:
        if pending:
            names = ", ".join(job.label.lower() for job in pending)
            issues.append(Issue(SEVERITY_WARNING, f"Registration skipped, but the {names} is not in the space of the "
                                                  f"{_roleLabel(s.primaryRole).lower()}. Check the alignment."))
        return StepStatus(STATE_SKIPPED, "Skipped by the user.", issues)
    if not jobs:
        return StepStatus(STATE_NOT_APPLICABLE, "Not needed: all images share the space of the dosimetry image.")
    for job in jobs:
        if job.registered and not job.followerFollows and job.followerRole:
            issues.append(Issue(SEVERITY_WARNING, f"The {_roleLabel(job.followerRole).lower()} does not follow the "
                                                  f"registration of its anatomical image."))
        if job.complete and job.path == PATH_FUNCTIONAL and not job.alignedByAcquisition:
            issues.append(Issue(SEVERITY_WARNING, f"{job.label}: functional-only registration. Verify the alignment "
                                                  "visually."))
        if job.method == "Deformable":
            issues.append(Issue(SEVERITY_INFO, f"{job.label}: deformable registration (kept live, not hardened)."))
    if optionalPending:
        issues.append(Issue(SEVERITY_INFO, "Optional, not registered yet: "
                            + ", ".join(job.label.lower() for job in optionalPending) + "."))
    done = len(required) - len(pending)
    if pending:
        state = STATE_IN_PROGRESS if done else STATE_NOT_STARTED
        issues.append(Issue(SEVERITY_INFO, "Waiting: " + ", ".join(job.label.lower() for job in pending) + "."))
        return StepStatus(state, f"{done} of {len(required)} registrations done.", issues)
    if not required:
        return StepStatus(_stateFromIssues(issues), "Only optional registrations (auxiliary volumes).", issues)
    return StepStatus(_stateFromIssues(issues), f"{len(required)} of {len(required)} registrations done.", issues)


# Plausibility limits of the segmentation (warnings only; the user decides)
LIVER_MIN_ML = 500.0            # smaller: incomplete whole-liver segment?
LIVER_MAX_ML = 4500.0           # larger: other organs included?
PERFUSED_MIN_ML = 100.0         # smaller: very selective territory, or an incomplete segment
SMALL_TUMOUR_ML = 2.0           # about the size of the SPECT/PET resolution: dose affected by partial volume


def _names(segments):
    return ", ".join(f"'{x.name}'" for x in segments)


def segmentsWithRole(s, role):
    return [segment for segment in s.segments if segment.role == role]


def evaluateSegmentation(s):
    issues = []
    if not s.segmentationPresent:
        return StepStatus(STATE_NOT_STARTED, "No segmentation yet.")
    liver = segmentsWithRole(s, SEGMENT_LIVER)
    if not liver:
        issues.append(Issue(SEVERITY_ERROR, "No whole-liver segment."))
    elif len(liver) > 1:
        issues.append(Issue(SEVERITY_WARNING, "Several segments are marked as whole liver: " +
                            ", ".join(f"'{x.name}'" for x in liver) + "."))
    if s.mode == R.MODE_RELATIVE and not segmentsWithRole(s, SEGMENT_PERFUSED):
        issues.append(Issue(SEVERITY_WARNING, "Patient-relative dosimetry needs at least one perfused-volume segment "
                                              "(for a whole-liver treatment it can cover the whole liver)."))
    for segment in liver:
        if segment.volumeML and not segment.empty:
            if segment.volumeML < LIVER_MIN_ML:
                issues.append(Issue(SEVERITY_WARNING, f"Whole liver '{segment.name}' is {segment.volumeML:.0f} mL "
                                                      f"(less than {LIVER_MIN_ML:.0f} mL): is the segment complete?"))
            elif segment.volumeML > LIVER_MAX_ML:
                issues.append(Issue(SEVERITY_WARNING, f"Whole liver '{segment.name}' is {segment.volumeML:.0f} mL "
                                                      f"(more than {LIVER_MAX_ML:.0f} mL): are other organs "
                                                      "included?"))
    smallPerfused = [x for x in segmentsWithRole(s, SEGMENT_PERFUSED)
                     if x.volumeML and not x.empty and x.volumeML < PERFUSED_MIN_ML]
    for segment in smallPerfused:
        issues.append(Issue(SEVERITY_WARNING, f"Perfused volume '{segment.name}' is {segment.volumeML:.0f} mL (less "
                                              f"than {PERFUSED_MIN_ML:.0f} mL): check that it covers the whole "
                                              "territory of the injection."))
    tumours = segmentsWithRole(s, SEGMENT_TUMOR) + segmentsWithRole(s, SEGMENT_VIABLE)
    if not tumours:
        issues.append(Issue(SEVERITY_WARNING, "No tumour segment: tumour doses will not be reported. Dosimetry can "
                                              "still be calculated for the other segments (whole liver, perfused "
                                              "volumes, normal tissue)."))
    if not segmentsWithRole(s, SEGMENT_NORMAL):
        issues.append(Issue(SEVERITY_WARNING, "No normal tissue segment: the normal tissue dose and the "
                                              "tumour-to-normal ratio will not be reported or checked. Create it with "
                                              "'Perfused normal = perfused − tumours' or 'Normal liver = liver − "
                                              "tumours' (Tools)."))
    small = [x for x in tumours if x.volumeML and not x.empty and x.volumeML < SMALL_TUMOUR_ML]
    if small:
        issues.append(Issue(SEVERITY_INFO, f"Small tumour(s) (< {SMALL_TUMOUR_ML:g} mL): {_names(small)}. Their dose "
                                           "is underestimated by the limited resolution of the SPECT/PET "
                                           "(partial volume)."))
    lsfFromImage = s.lsfValue is None and not s.lsfSkipped and s.mode != R.MODE_ABSOLUTE
    if lsfFromImage and s.scenario == R.SCENARIO_PRE and not segmentsWithRole(s, SEGMENT_LUNGS):
        issues.append(Issue(SEVERITY_INFO, "A lung segment is needed to calculate the LSF from the image "
                                           "(not needed if the LSF is entered manually)."))
    empty = [x.name for x in s.segments if x.empty]
    if empty:
        issues.append(Issue(SEVERITY_WARNING, "Empty segment(s): " + ", ".join(f"'{n}'" for n in empty) + "."))
    candidates = [x.name for x in s.segments if x.candidate]
    if candidates:
        issues.append(Issue(SEVERITY_WARNING, "AI result(s) not accepted yet: " +
                            ", ".join(f"'{n}'" for n in candidates) + "."))
    noRole = [x.name for x in s.segments if not x.role and not x.candidate]
    if noRole:
        issues.append(Issue(SEVERITY_WARNING, "Segment(s) without a role: " + ", ".join(f"'{n}'" for n in noRole) +
                            ". Set their role in the segment table (the dosimetry modules list them as "
                            "uncategorized)."))
    seen, duplicates = set(), []
    for segment in s.segments:
        if segment.name in seen and segment.name not in duplicates:
            duplicates.append(segment.name)
        seen.add(segment.name)
    if duplicates:
        issues.append(Issue(SEVERITY_WARNING, "Several segments have the same name: " +
                            ", ".join(f"'{n}'" for n in duplicates) + ". Rename them: the report tables cannot "
                            "tell them apart."))
    if s.geometryIssues:
        issues += [Issue(severity, text) for severity, text in s.geometryIssues]
    elif s.geometryStale:
        issues.append(Issue(SEVERITY_INFO, "The segments changed since the last geometry check."))
    counts = []
    for role in (SEGMENT_LIVER, SEGMENT_PERFUSED, SEGMENT_TUMOR, SEGMENT_LUNGS):
        number = len(segmentsWithRole(s, role))
        if number:
            counts.append(f"{number} {SEGMENT_ROLES[role][0].lower()}")
    summary = ", ".join(counts) if counts else f"{len(s.segments)} segment(s), no roles yet."
    return StepStatus(_stateFromIssues(issues), summary, issues)


# Planning (CBCT territories) guardrails
CBCT_LIVER_COVERAGE = 0.90         # less of the whole liver inside the CBCT field of view: territory beyond is unknown
MAX_LOOPS = 3                      # more loops in the oriented tree: veins or bone in the tree segment?
MAA_AGREEMENT_LOW = 0.6            # Dice of the territories with the MAA perfused volume
MAA_COUNTS_LOW = 0.80              # fraction of the liver MAA counts inside the territories
UNSUPPLIED_PERCENT = 10.0          # more of the liver not reached by the imaged tree: territories need the supplied
TUMOUR_COVERAGE_LOW = 90.0         # % of a tumour inside the predicted territories
EXTRAHEPATIC_FLAG = "Possible non-target vessel:"


def planningFindings(results, reviewed=False):
    """[Issue] from the stored result of a territory prediction (dict written by the CBCT planning module):
    {"liverCoverage": 0-1, "loops": n, "positions": [{"label", "snapped", "snapDistanceMM", "territoryML",
    "extrahepatic": [{"distanceMM", "lengthMM"}]}], "overlaps": [[a, b, mL]], "uncovered": [[tumour, coverage %]],
    "maa": {"dice", "countsFraction"} or None}. reviewed: the extrahepatic findings were reviewed by the user."""
    issues = []
    coverage = results.get("liverCoverage")
    if coverage is not None and coverage < CBCT_LIVER_COVERAGE:
        issues.append(Issue(SEVERITY_WARNING, f"Only {100 * coverage:.0f}% of the whole liver is inside the CBCT field "
                                              "of view: the territory beyond it is unknown."))
    unsupplied = results.get("unsuppliedPercent")
    if unsupplied is not None and unsupplied > UNSUPPLIED_PERCENT and not results.get("territoryLimitMM"):
        issues.append(Issue(SEVERITY_WARNING, f"{unsupplied:.0f}% of the whole liver is far from every branch of the "
                                              "imaged arterial tree (selective injection?): the territories claim "
                                              "liver it may not feed. Turn on 'Limit each territory to liver near "
                                              "its branches'."))
    loops = results.get("loops") or 0
    if loops > MAX_LOOPS:
        issues.append(Issue(SEVERITY_WARNING, f"The arterial tree has {loops} loops: veins, bone or the catheter may "
                                              "be included. Edit the tree segment and recompute."))
    for position in results.get("positions", []):
        label = position.get("label", "?")
        if not position.get("snapped", True):
            distance = position.get("snapDistanceMM")
            where = f"is {distance:.0f} mm from the arterial tree" if distance is not None else "is not on the tree"
            issues.append(Issue(SEVERITY_ERROR, f"Tip {label} {where}: move it onto an artery (no territory for this "
                                                "position)."))
            continue
        volume = position.get("territoryML")
        if volume is not None and volume < PERFUSED_MIN_ML:
            issues.append(Issue(SEVERITY_WARNING, f"Territory of {label} is {volume:.0f} mL (less than "
                                                  f"{PERFUSED_MIN_ML:.0f} mL): very selective, or the tree is "
                                                  "incomplete."))
        for finding in position.get("extrahepatic", []):
            severity = SEVERITY_INFO if reviewed else SEVERITY_WARNING
            issues.append(Issue(severity, f"{EXTRAHEPATIC_FLAG} a branch downstream of {label} runs "
                                          f"{finding['distanceMM']:.0f} mm outside the liver "
                                          f"({finding['lengthMM']:.0f} mm long; gastric, falciform, cystic?)."
                                          + (" Marked as reviewed." if reviewed else " Review it on the CBCT.")))
    for a, b, volume in results.get("overlaps", []):
        issues.append(Issue(SEVERITY_WARNING, f"Territories {a} and {b} overlap by {volume:.0f} mL (nested tips): "
                                              "subtract the downstream territory, or plan them as separate "
                                              "sessions."))
    for name, percent in results.get("uncovered", []):
        text = (f"Tumour '{name}' is not in any predicted territory." if percent <= 0 else
                f"Only {percent:.0f}% of tumour '{name}' is in the predicted territories.")
        issues.append(Issue(SEVERITY_WARNING, text))
    maa = results.get("maa")
    if maa:
        if maa.get("dice") is not None and maa["dice"] < MAA_AGREEMENT_LOW:
            issues.append(Issue(SEVERITY_WARNING, f"The predicted territories agree poorly with the MAA perfused volume "
                                                  f"(Dice {maa['dice']:.2f}): check the tip positions and the "
                                                  "registration."))
        if maa.get("countsFraction") is not None and maa["countsFraction"] < MAA_COUNTS_LOW:
            issues.append(Issue(SEVERITY_WARNING, f"Only {100 * maa['countsFraction']:.0f}% of the liver MAA counts lie "
                                                  "inside the predicted territories."))
    return issues


def evaluatePlanning(s):
    cbct = s.roles.get(R.ROLE_CBCT) or s.roles.get(R.ROLE_CBCT_PARENCHYMAL)
    if cbct is None:
        return StepStatus(STATE_NOT_APPLICABLE, "Not needed: no CBCT assigned.")
    if s.planningSkipped:
        return StepStatus(STATE_SKIPPED, "Skipped by the user.")
    issues = []
    if not s.segmentationPresent:
        issues.append(Issue(SEVERITY_ERROR, "No segmentation: segment the whole liver first (Segmentation step)."))
    elif not segmentsWithRole(s, SEGMENT_LIVER):
        issues.append(Issue(SEVERITY_ERROR, "No whole-liver segment: the territories need it (Segmentation step)."))
    for job in s.jobs:
        if job.key in CBCT_JOBS and not job.complete:
            issues.append(Issue(SEVERITY_WARNING, f"The {job.label} is not registered to the primary image: "
                                                  "register it in the Registration step (or confirm its alignment)."))
    results = s.planningResults
    if not results:
        if not s.planningHasTree:
            issues.append(Issue(SEVERITY_INFO, "Place the injection point and extract the arterial tree, or use the "
                                               "perfused volume from CBCT enhancement (CBCT Planning)."))
            state = STATE_ERROR if any(i.severity == SEVERITY_ERROR for i in issues) else STATE_NOT_STARTED
            return StepStatus(state, "No territory predicted yet.", issues)
        if not s.planningTips:
            issues.append(Issue(SEVERITY_INFO, "Place the planned catheter tip(s) on the arterial tree."))
        state = STATE_ERROR if any(i.severity == SEVERITY_ERROR for i in issues) else STATE_IN_PROGRESS
        return StepStatus(state, "Arterial tree extracted; predict the territories.", issues)
    issues += planningFindings(results, s.planningReviewed)
    territoryIDs = set(results.get("segmentIDs", []))
    pending = [x.name for x in s.segments if x.candidate and x.segmentID in territoryIDs]
    if pending:
        issues.append(Issue(SEVERITY_WARNING, "Predicted territory not accepted yet: " +
                            ", ".join(f"'{n}'" for n in pending) + " (accept or discard it in the Segmentation "
                            "step)."))
    parts = [f"{p.get('label', '?')} {p['territoryML']:.0f} mL" for p in results.get("positions", [])
             if p.get("territoryML") is not None]
    parts += [f"enhancement {item.get('volumeML', 0):.0f} mL" for item in results.get("enhancement", [])]
    summary = ("Territories: " + ", ".join(parts)) if parts else "No territory."
    if s.planningOutdated:
        issues.append(Issue(SEVERITY_WARNING, "The arterial tree, the catheter positions or the liver / tumour "
                                              "segments changed: predict the territories again."))
        return StepStatus(STATE_OUTDATED, summary, issues)
    return StepStatus(_stateFromIssues(issues), summary, issues)


LSF_WARNING_PERCENT = 10.0
LSF_HIGH_PERCENT = 20.0
Y90_GY_KG_PER_GBQ = 49.67                # MIRD: Gy per GBq fully absorbed in 1 kg
LUNG_DOSE_SESSION_LIMIT_GY = 30.0
LUNG_DOSE_CUMULATIVE_LIMIT_GY = 50.0
DEFAULT_LUNG_MASS_G = 1000.0


def lungDose(s):
    """Estimated lung dose (Gy) from the planned activity, LSF and lung mass (default 1000 g), or None."""
    if s.plannedActivityGBq is None or s.lsfValue is None:
        return None
    mass = s.lsfLungMassG or DEFAULT_LUNG_MASS_G
    return Y90_GY_KG_PER_GBQ * s.plannedActivityGBq * (s.lsfValue / 100.0) / (mass / 1000.0)


def evaluateLsf(s):
    issues = []
    if s.mode == R.MODE_ABSOLUTE:
        return StepStatus(STATE_NOT_APPLICABLE, "Not used in absolute mode: lung activity is taken from the image.")
    if s.lsfValue is not None:
        value = s.lsfValue
        if value > LSF_HIGH_PERCENT:
            issues.append(Issue(SEVERITY_WARNING, f"LSF {value:.1f} % is above {LSF_HIGH_PERCENT:g} %: commonly "
                                                  "considered a contraindication. Check the lung dose."))
        elif value > LSF_WARNING_PERCENT:
            issues.append(Issue(SEVERITY_WARNING, f"LSF {value:.1f} % is above {LSF_WARNING_PERCENT:g} %: check the "
                                                  "lung dose and the activity reduction rules of the device."))
        issues += [Issue(severity, text) for severity, text in s.lsfIssues]
        if s.lsfLungMassG is None:
            issues.append(Issue(SEVERITY_INFO, "No lung mass: the default of 1000 g will be used."))
        dose = lungDose(s)
        if dose is not None:
            if dose > LUNG_DOSE_CUMULATIVE_LIMIT_GY:
                issues.append(Issue(SEVERITY_WARNING, f"Estimated lung dose {dose:.1f} Gy is above "
                                                      f"{LUNG_DOSE_CUMULATIVE_LIMIT_GY:g} Gy (commonly used cumulative "
                                                      "limit)."))
            elif dose > LUNG_DOSE_SESSION_LIMIT_GY:
                issues.append(Issue(SEVERITY_WARNING, f"Estimated lung dose {dose:.1f} Gy is above "
                                                      f"{LUNG_DOSE_SESSION_LIMIT_GY:g} Gy (commonly used single-session "
                                                      "limit)."))
            else:
                issues.append(Issue(SEVERITY_INFO, f"Estimated lung dose {dose:.1f} Gy "
                                                   f"({s.plannedActivityGBq:g} GBq)."))
        source = f" ({s.lsfSource})" if s.lsfSource else ""
        if s.lsfFromImage and s.lsfOutdated:
            issues.append(Issue(SEVERITY_WARNING, "The dosimetry image or the lung / whole-liver segments changed "
                                                  "after the LSF was calculated: calculate it again."))
            return StepStatus(STATE_OUTDATED, f"LSF {value:.2f} %{source} is outdated.", issues)
        return StepStatus(_stateFromIssues(issues), f"LSF {value:.2f} %{source}", issues)
    if s.lsfSkipped:
        if s.scenario == R.SCENARIO_PRE:
            issues.append(Issue(SEVERITY_WARNING, "LSF not set: set the lung shunt fraction in the dosimetry module."))
        return StepStatus(STATE_SKIPPED, "Skipped by the user.", issues)
    if s.scenario == R.SCENARIO_POST:
        return StepStatus(STATE_NOT_STARTED, "Optional after therapy: enter the pre-therapy LSF or skip.")
    return StepStatus(STATE_NOT_STARTED, "Calculate the LSF or enter it manually.")


def evaluateDosimetry(s):
    issues = []
    if not s.mode:
        return StepStatus(STATE_NOT_STARTED, "Choose the processing mode in the Data step.")
    expectedModule = R.MODE_MODULES[s.mode]
    if s.dosimetryResultsModule == expectedModule:
        if s.dosimetryOutdated:
            issues.append(Issue(SEVERITY_WARNING, "Images, registration or segments changed after the calculation: "
                                                  "calculate again."))
            return StepStatus(STATE_OUTDATED, f"{R.MODE_LABELS[s.mode]} results are outdated.", issues)
        issues += [Issue(severity, text) for severity, text in s.dosimetryChecks]
        warnings = sum(1 for issue in issues if issue.severity == SEVERITY_WARNING)
        summary = f"{R.MODE_LABELS[s.mode]} dosimetry calculated."
        if warnings:
            summary += f" {warnings} dose check(s) need attention."
        return StepStatus(_stateFromIssues(issues), summary, issues)
    if s.dosimetryResultsModule:
        issues.append(Issue(SEVERITY_INFO, "Results exist from the other processing mode; calculate in the "
                                           f"{R.MODE_LABELS[s.mode].lower()} module."))
    return StepStatus(STATE_NOT_STARTED, f"{R.MODE_LABELS[s.mode]} dosimetry not calculated yet.", issues)


def evaluateReport(s):
    if s.reportSaved:
        if s.reportOutdated:
            return StepStatus(STATE_OUTDATED, "The report was saved before the last calculation.",
                              [Issue(SEVERITY_WARNING, "Save the report again.")])
        return StepStatus(STATE_DONE, "Report saved.")
    return StepStatus(STATE_NOT_STARTED, "Report not saved yet.")


EVALUATORS = {
    STEP_DATA: evaluateData,
    STEP_REGISTRATION: evaluateRegistration,
    STEP_SEGMENTATION: evaluateSegmentation,
    STEP_PLANNING: evaluatePlanning,
    STEP_LSF: evaluateLsf,
    STEP_DOSIMETRY: evaluateDosimetry,
    STEP_REPORT: evaluateReport,
}

# step -> steps whose errors make it wait (display only: the step can still be opened)
PREREQUISITES = {
    STEP_REGISTRATION: [STEP_DATA],
    STEP_SEGMENTATION: [STEP_DATA],
    STEP_PLANNING: [STEP_DATA, STEP_SEGMENTATION],
    STEP_LSF: [STEP_DATA],
    STEP_DOSIMETRY: [STEP_DATA, STEP_SEGMENTATION],
    STEP_REPORT: [STEP_DOSIMETRY],
}


def evaluateAll(s):
    """{step: StepStatus} with soft locks and the step key filled in every issue."""
    statuses = {}
    for key in STEP_KEYS:
        status = EVALUATORS[key](s)
        for issue in status.issues:
            issue.step = key
        statuses[key] = status
    for key, prerequisites in PREREQUISITES.items():
        status = statuses[key]
        if status.state != STATE_NOT_STARTED:
            continue
        waiting = [p for p in prerequisites
                   if statuses[p].state == STATE_ERROR
                   or (key == STEP_REPORT and statuses[p].state not in (STATE_DONE, STATE_WARNING))]
        if waiting:
            status.state = STATE_LOCKED
            status.issues.append(Issue(SEVERITY_INFO, "Waiting for: " +
                                       ", ".join(STEP_LABELS[p] for p in waiting) + ".", key))
    return statuses


def nextStep(statuses):
    """First step that still needs attention (None if all are finished)."""
    for key in STEP_KEYS:
        if statuses[key].state not in FINISHED_STATES:
            return key
    return None


def allIssues(statuses, severities=(SEVERITY_ERROR, SEVERITY_WARNING)):
    issues = [issue for key in STEP_KEYS for issue in statuses[key].issues if issue.severity in severities]
    return sorted(issues, key=lambda i: (SEVERITY_ORDER[i.severity], STEP_KEYS.index(i.step)))
