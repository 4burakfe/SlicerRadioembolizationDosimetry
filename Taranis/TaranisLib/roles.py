"""Imaging roles of a Taranis case, image types, processing modes and automatic role suggestion.

Pure Python (no Slicer imports), so the rules can be unit tested outside Slicer.
"""

import dataclasses
import datetime
import re

# -- Roles -------------------------------------------------------------------------------------------------

ROLE_DOSIMETRY = "DosimetryImage"
ROLE_DOSIMETRY_ANATOMY = "DosimetryAnatomy"
ROLE_REFERENCE = "ReferenceImage"
ROLE_METABOLIC = "MetabolicImage"
ROLE_METABOLIC_ANATOMY = "MetabolicAnatomy"
ROLE_SEGMENTATION = "Segmentation"

VOLUME_ROLES = [ROLE_DOSIMETRY, ROLE_DOSIMETRY_ANATOMY, ROLE_REFERENCE, ROLE_METABOLIC, ROLE_METABOLIC_ANATOMY]

# -- Image types -------------------------------------------------------------------------------------------

TYPE_MAA_SPECT = "MAA_SPECT"
TYPE_Y90_SPECT = "Y90_SPECT"
TYPE_Y90_PET = "Y90_PET"
TYPE_FDG_PET = "FDG_PET"
TYPE_DOTATATE_PET = "DOTATATE_PET"
TYPE_OTHER_PET = "OTHER_PET"
TYPE_CT = "CT"
TYPE_MRI = "MRI"

TYPE_LABELS = {
    TYPE_MAA_SPECT: "Tc-99m MAA SPECT",
    TYPE_Y90_SPECT: "Y-90 SPECT (bremsstrahlung)",
    TYPE_Y90_PET: "Y-90 PET",
    TYPE_FDG_PET: "FDG PET",
    TYPE_DOTATATE_PET: "DOTATATE PET (SSTR)",
    TYPE_OTHER_PET: "Other PET",
    TYPE_CT: "CT",
    TYPE_MRI: "MRI",
}

DOSIMETRY_TYPES = [TYPE_MAA_SPECT, TYPE_Y90_SPECT, TYPE_Y90_PET]
METABOLIC_TYPES = [TYPE_FDG_PET, TYPE_DOTATATE_PET, TYPE_OTHER_PET]
ANATOMY_TYPES = [TYPE_CT, TYPE_MRI]

ROLE_TYPES = {
    ROLE_DOSIMETRY: DOSIMETRY_TYPES,
    ROLE_DOSIMETRY_ANATOMY: ANATOMY_TYPES,
    ROLE_REFERENCE: ANATOMY_TYPES,
    ROLE_METABOLIC: METABOLIC_TYPES,
    ROLE_METABOLIC_ANATOMY: ANATOMY_TYPES,
}

# role -> (label, requirement text, tooltip)
ROLE_INFO = {
    ROLE_DOSIMETRY: (
        "Dosimetry image", "Required",
        "Tc-99m MAA SPECT (pre-therapy), Y-90 SPECT or Y-90 PET (post-therapy). The dose is calculated on this "
        "image."),
    ROLE_DOSIMETRY_ANATOMY: (
        "Its anatomical image", "Required: this or the reference",
        "The CT (or MRI) acquired together with the dosimetry image, e.g. the CT of the SPECT/CT or PET/CT."),
    ROLE_REFERENCE: (
        "Reference image", "Required: this or the anatomical image above",
        "Diagnostic CT or MRI. When present, segmentations are drawn on it and the other images are registered "
        "to it."),
    ROLE_METABOLIC: (
        "Metabolic image", "Optional",
        "FDG or DOTATATE PET, used to delineate tumours."),
    ROLE_METABOLIC_ANATOMY: (
        "Its anatomical image", "Optional (recommended with a metabolic image)",
        "The CT (or MRI) of the metabolic PET/CT. Makes registration of the metabolic image robust."),
    ROLE_SEGMENTATION: (
        "Segmentation", "Required for dosimetry",
        "The master segmentation of the case (whole liver, perfused volumes, tumours, lungs, ...)."),
}

# -- Scenario and processing mode ------------------------------------------------------------------------

SCENARIO_PRE = "pre"
SCENARIO_POST = "post"
SCENARIO_LABELS = {SCENARIO_PRE: "Pre-therapy planning", SCENARIO_POST: "Post-therapy verification"}

MODE_RELATIVE = "relative"
MODE_ABSOLUTE = "absolute"
MODE_LABELS = {MODE_RELATIVE: "Patient-relative", MODE_ABSOLUTE: "Absolute"}
MODE_MODULES = {MODE_RELATIVE: "RadioembolizationDosimetryRelative",
                MODE_ABSOLUTE: "RadioembolizationDosimetryAbsolute"}

MICROSPHERES_GLASS = "glass"
MICROSPHERES_RESIN = "resin"
MICROSPHERE_LABELS = {MICROSPHERES_GLASS: "Glass microspheres", MICROSPHERES_RESIN: "Resin microspheres"}


def scenarioForType(imageType):
    if imageType == TYPE_MAA_SPECT:
        return SCENARIO_PRE
    if imageType in (TYPE_Y90_SPECT, TYPE_Y90_PET):
        return SCENARIO_POST
    return None


def modeOptions(imageType):
    """[(mode, allowed, note)] for a dosimetry image type."""
    if imageType == TYPE_MAA_SPECT:
        return [(MODE_RELATIVE, True, ""),
                (MODE_ABSOLUTE, False, "Not possible: MAA activity is not the Y-90 activity.")]
    if imageType == TYPE_Y90_SPECT:
        return [(MODE_RELATIVE, True, "Recommended for bremsstrahlung SPECT."),
                (MODE_ABSOLUTE, True, "Only for a quantitatively calibrated reconstruction (Bq/mL).")]
    if imageType == TYPE_Y90_PET:
        return [(MODE_RELATIVE, True, ""),
                (MODE_ABSOLUTE, True, "Recommended for quantitative Y-90 PET.")]
    return [(MODE_RELATIVE, True, ""), (MODE_ABSOLUTE, False, "Select the dosimetry image type first.")]


def unitsAreActivityConcentration(unitText):
    """True for voxel units such as Bq/mL, kBq/ml or MBq/mL (DICOM BQML)."""
    text = (unitText or "").strip().lower().replace(" ", "")
    return text == "bqml" or text.endswith("bq/ml")


def absoluteDosimetryProblem(inputName, hasCase, caseImageName, isCaseImage, caseImageType, voxelUnits, dicomUnits):
    """Reason why absolute dosimetry must not run on the selected input image, or "" when it may.

    With a Taranis case in the scene, the input must be the case's dosimetry image and its type must allow absolute
    dosimetry (as on the hub's Dosimetry step). Without a case, the image must carry activity-concentration units
    (Bq/mL) in its metadata. voxelUnits: the volume's voxel value units (e.g. "Bq/ml", "{SUVbw}g/ml");
    dicomUnits: DICOM (0054,1001) Units of its series (BQML, CNTS, GML...)."""
    if hasCase:
        if not caseImageName:
            return ("Absolute dosimetry is locked: no dosimetry image is assigned in the Taranis case. Assign the "
                    "quantitative Y-90 PET or calibrated Y-90 SPECT in the Taranis Data step.")
        if not isCaseImage:
            return (f"Absolute dosimetry is locked: the input '{inputName}' is not the dosimetry image of the Taranis "
                    f"case ('{caseImageName}'). Select it, or change the assignment in the Taranis Data step.")
        if not caseImageType:
            return ("Absolute dosimetry is locked: the type of the dosimetry image is not set. Set it (Y-90 PET or "
                    "Y-90 SPECT) in the Taranis Data step.")
        if not modeAllowed(caseImageType, MODE_ABSOLUTE):
            note = next((n for m, _, n in modeOptions(caseImageType) if m == MODE_ABSOLUTE), "")
            return (f"Absolute dosimetry is locked for a {TYPE_LABELS.get(caseImageType, caseImageType)}"
                    + (f": {note}" if note else ".") + " Use patient-relative dosimetry.")
    if unitsAreActivityConcentration(voxelUnits) or (dicomUnits or "").upper() == UNITS_QUANTITATIVE:
        return ""
    if hasCase and not voxelUnits and not dicomUnits:
        return ""   # e.g. loaded from a file without metadata: the user assigned it as quantitative in the case
    if "suv" in (voxelUnits or "").lower() or (dicomUnits or "").upper() == UNITS_SUV:
        return (f"Absolute dosimetry is locked: '{inputName}' is in SUV, not in activity concentration (Bq/mL). "
                "Load the PET in Bq/mL.")
    if (dicomUnits or "").upper() == UNITS_COUNTS:
        return (f"Absolute dosimetry is locked: '{inputName}' is in counts, not in activity concentration (Bq/mL). "
                "Use patient-relative dosimetry, or a quantitatively reconstructed image.")
    return (f"Absolute dosimetry is locked: '{inputName}' is not recognised as a quantitative image (no Bq/mL "
            "units in its metadata). Load the quantitative Y-90 PET / SPECT from DICOM, or start a Taranis case and "
            "assign it as the dosimetry image (Data step).")


def defaultMode(imageType):
    return MODE_ABSOLUTE if imageType == TYPE_Y90_PET else MODE_RELATIVE


def modeAllowed(imageType, mode):
    return any(m == mode and allowed for m, allowed, _ in modeOptions(imageType))


# -- Volume information -------------------------------------------------------------------------------------

UNITS_QUANTITATIVE = "BQML"
UNITS_COUNTS = "CNTS"
UNITS_SUV = "GML"


@dataclasses.dataclass
class VolumeInfo:
    """What Taranis knows about a loaded volume (from DICOM tags when available, otherwise from its name)."""
    nodeID: str
    name: str = ""
    modality: str = ""               # DICOM modality: NM, PT, CT, MR ("" if unknown)
    seriesDescription: str = ""
    units: str = ""                  # DICOM (0054,1001): BQML, CNTS, GML, ...
    radiopharmaceutical: str = ""
    radionuclide: str = ""
    frameOfReferenceUID: str = ""
    studyUID: str = ""
    acquisitionDateTime: str = ""    # ISO format, "" if unknown
    patientName: str = ""
    patientID: str = ""
    minValue: float = None
    bounds: tuple = None             # world RAS bounds (xmin, xmax, ymin, ymax, zmin, zmax)
    fromDicom: bool = False

    def dateTime(self):
        return parseDateTime(self.acquisitionDateTime)


def parseDateTime(text):
    if not text:
        return None
    try:
        return datetime.datetime.fromisoformat(text)
    except ValueError:
        return None


def dicomDateTime(date, time=""):
    """ISO date-time from DICOM DA and TM strings ("" if the date is invalid)."""
    date = (date or "").strip()
    time = (time or "").strip().split(".")[0]
    if len(date) != 8 or not date.isdigit():
        return ""
    time = (time + "000000")[:6] if time.isdigit() else "000000"
    try:
        value = datetime.datetime(int(date[:4]), int(date[4:6]), int(date[6:8]),
                                  int(time[:2]), int(time[2:4]), int(time[4:6]))
    except ValueError:
        return ""
    return value.isoformat()


# -- Classification ---------------------------------------------------------------------------------------

_Y90_WORDS = ["y90", "y-90", "90y", "yttrium", "bremss", "brems", "sirt", "tare", "theraspheres", "therasphere",
              "sirspheres", "sir-spheres"]
_MAA_WORDS = ["maa", "technetium", "tc99", "tc-99", "99mtc", "tc99m", "shunt", "lsf", "macroaggregated"]
_FDG_WORDS = ["fdg", "fluorodeoxyglucose", "fludeoxyglucose"]
_DOTA_WORDS = ["dota", "tate", "dotatoc", "dotanoc", "gallium", "ga68", "ga-68", "68ga", "cu64", "64cu",
               "copper", "somatostatin", "sstr"]
_MR_TOKENS = {"mr", "mri", "t1", "t2", "t1w", "t2w", "dixon", "vibe", "lava", "thrive", "flair", "dwi", "adc",
              "hbp", "eob", "primovist", "gadoxetic"}
_CT_TOKENS = {"ct", "cect", "ldct", "ctac", "ac_ct", "ctpv", "cta"}
_REFERENCE_WORDS = ["diagnostic", "contrast", "arterial", "portal", "venous", "cect", "triphasic", "late",
                    "delayed", "hbp", "eob", "dynamic"]
_SUV_WORDS = ["suv"]


def _text(info):
    return " ".join([info.name or "", info.seriesDescription or "", info.radiopharmaceutical or "",
                     info.radionuclide or ""]).lower()


def _tokens(text):
    return set(t for t in re.split(r"[^a-z0-9]+", text) if t)


def _contains(text, words):
    return any(word in text for word in words)


def guessModality(info):
    """NM, PT, CT, MR or "" (DICOM modality wins; otherwise the name and the value range are used)."""
    if info.modality in ("NM", "PT", "CT", "MR"):
        return info.modality
    text = _text(info)
    tokens = _tokens(text)
    functional = ""
    if "spect" in text or "nm" in tokens or _contains(text, ["maa", "bremss"]):
        functional = "NM"
    elif "pet" in tokens or re.search(r"pet(?![a-z])", text):
        functional = "PT"
    anatomical = ""
    if tokens & _MR_TOKENS:
        anatomical = "MR"
    elif tokens & _CT_TOKENS:
        anatomical = "CT"
    lowValues = info.minValue is not None and info.minValue <= -500
    if functional and anatomical:
        # e.g. "PET/CT": decide from the values (CT has air at about -1000 HU)
        return anatomical if (anatomical == "CT" and lowValues) else functional
    if functional:
        return functional
    if anatomical:
        return anatomical
    return "CT" if lowValues else ""


def guessType(info):
    """Suggested image type (one of TYPE_*) or None."""
    modality = guessModality(info)
    text = _text(info)
    if modality == "NM":
        if _contains(text, _MAA_WORDS):
            return TYPE_MAA_SPECT
        if _contains(text, _Y90_WORDS):
            return TYPE_Y90_SPECT
        return None
    if modality == "PT":
        if _contains(text, _Y90_WORDS):
            return TYPE_Y90_PET
        if _contains(text, _FDG_WORDS):
            return TYPE_FDG_PET
        if _contains(text, _DOTA_WORDS):
            return TYPE_DOTATATE_PET
        return None
    if modality == "CT":
        return TYPE_CT
    if modality == "MR":
        return TYPE_MRI
    return None


def isFunctional(info):
    return guessModality(info) in ("NM", "PT")


def isAnatomical(info):
    return guessModality(info) in ("CT", "MR")


def looksLikeReference(info):
    return guessModality(info) == "MR" or _contains(_text(info), _REFERENCE_WORDS)


def looksLikeSuv(info):
    return info.units == UNITS_SUV or _contains(_text(info), _SUV_WORDS)


# -- Pairing -----------------------------------------------------------------------------------------------

HYBRID_TIME_WINDOW_HOURS = 2.0


def boundsOverlapFraction(boundsA, boundsB):
    """Intersection volume divided by the smaller box volume (0 if either is missing)."""
    if not boundsA or not boundsB:
        return 0.0
    intersection, volumeA, volumeB = 1.0, 1.0, 1.0
    for axis in range(3):
        lowA, highA = boundsA[2 * axis], boundsA[2 * axis + 1]
        lowB, highB = boundsB[2 * axis], boundsB[2 * axis + 1]
        volumeA *= max(0.0, highA - lowA)
        volumeB *= max(0.0, highB - lowB)
        intersection *= max(0.0, min(highA, highB) - max(lowA, lowB))
    smaller = min(volumeA, volumeB)
    return intersection / smaller if smaller > 0 else 0.0


def pairingEvidence(functional, anatomical):
    """(is hybrid pair, reason) for a functional image and a candidate anatomical image."""
    if functional.frameOfReferenceUID and anatomical.frameOfReferenceUID:
        if functional.frameOfReferenceUID == anatomical.frameOfReferenceUID:
            return True, "same DICOM frame of reference"
        return False, ""
    if functional.studyUID and anatomical.studyUID and functional.studyUID == anatomical.studyUID:
        timeA, timeB = functional.dateTime(), anatomical.dateTime()
        if timeA and timeB and abs((timeA - timeB).total_seconds()) <= HYBRID_TIME_WINDOW_HOURS * 3600:
            return True, "same study, acquired together"
    if (not functional.fromDicom and not anatomical.fromDicom and guessModality(anatomical) == "CT"
            and boundsOverlapFraction(functional.bounds, anatomical.bounds) >= 0.7):
        return True, "guessed from image position (no DICOM information)"
    return False, ""


@dataclasses.dataclass
class Suggestion:
    assignments: dict   # role -> nodeID
    types: dict         # role -> image type
    notes: list         # human readable explanations


def suggestAssignments(infos):
    """Suggest roles for a list of VolumeInfo. Never assigns one volume to two roles."""
    assignments, types, notes = {}, {}, []
    used = set()

    functional = [info for info in infos if isFunctional(info)]
    anatomical = [info for info in infos if isAnatomical(info)]

    def newest(candidates):
        dated = [c for c in candidates if c.dateTime()]
        if len(dated) == len(candidates) and dated:
            return max(dated, key=lambda c: c.dateTime())
        return candidates[0]

    # 1. Dosimetry image
    dosimetryCandidates = [f for f in functional if guessType(f) in DOSIMETRY_TYPES]
    if not dosimetryCandidates:
        dosimetryCandidates = [f for f in functional if guessType(f) is None and guessModality(f) == "NM"]
    if not dosimetryCandidates:
        dosimetryCandidates = [f for f in functional if guessType(f) is None]
    dosimetry = None
    if dosimetryCandidates:
        dosimetry = newest(dosimetryCandidates)
        assignments[ROLE_DOSIMETRY] = dosimetry.nodeID
        types[ROLE_DOSIMETRY] = guessType(dosimetry)
        used.add(dosimetry.nodeID)
        if len(dosimetryCandidates) > 1:
            notes.append(f"Several possible dosimetry images; '{dosimetry.name}' was chosen (most recent).")
        if types[ROLE_DOSIMETRY] is None:
            notes.append(f"Select the type of the dosimetry image '{dosimetry.name}'.")

    # 2. Metabolic image
    metabolicCandidates = [f for f in functional if f.nodeID not in used and guessType(f) in METABOLIC_TYPES]
    metabolic = None
    if metabolicCandidates:
        metabolic = newest(metabolicCandidates)
        assignments[ROLE_METABOLIC] = metabolic.nodeID
        types[ROLE_METABOLIC] = guessType(metabolic)
        used.add(metabolic.nodeID)

    # 3. Anatomical images acquired with the functional images
    for functionalInfo, role in ((dosimetry, ROLE_DOSIMETRY_ANATOMY), (metabolic, ROLE_METABOLIC_ANATOMY)):
        if functionalInfo is None:
            continue
        for candidate in anatomical:
            if candidate.nodeID in used:
                continue
            paired, reason = pairingEvidence(functionalInfo, candidate)
            if paired:
                assignments[role] = candidate.nodeID
                types[role] = guessType(candidate)
                used.add(candidate.nodeID)
                notes.append(f"'{candidate.name}' paired with '{functionalInfo.name}': {reason}.")
                break

    # 4. Reference image: remaining anatomical image, preferring diagnostic-looking ones, then the newest
    remaining = [a for a in anatomical if a.nodeID not in used]
    if remaining:
        preferred = [a for a in remaining if looksLikeReference(a)] or remaining
        reference = newest(preferred)
        assignments[ROLE_REFERENCE] = reference.nodeID
        types[ROLE_REFERENCE] = guessType(reference)
        used.add(reference.nodeID)

    unassigned = [info.name for info in infos if info.nodeID not in used]
    if unassigned:
        notes.append("Not assigned: " + ", ".join(f"'{n}'" for n in unassigned) + ".")
    return Suggestion(assignments, types, notes)


def suggestCaseIdentity(infos):
    """(case name, case ID) from the DICOM patient of the loaded volumes ("" when unknown)."""
    for info in infos:
        if info.patientName or info.patientID:
            name = (info.patientName or "").replace("^", " ").strip()
            return name, (info.patientID or "").strip()
    return "", ""
