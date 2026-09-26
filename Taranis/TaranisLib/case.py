"""The Taranis case: one scene-saved parameter node holding the case identity, image roles, processing choices
and workflow flags. Also reads DICOM metadata of volumes and the suite's application settings."""

import datetime
import logging
import os

import qt
import slicer

from . import roles as R

CASE_ATTRIBUTE = "Taranis.Case"
CASE_VERSION = "1"

# Case node parameters
P_VERSION = "Version"
P_NAME = "CaseName"
P_ID = "CaseID"
P_CREATED = "Created"
P_MODE = "Mode"
P_MICROSPHERES = "Microspheres"
P_TREATMENT_DATETIME = "TreatmentDateTime"
P_CURRENT_STEP = "CurrentStep"
P_LSF_VALUE = "LSF.Value"
P_LSF_SOURCE = "LSF.Source"
P_LSF_LUNG_MASS = "LSF.LungMassG"
P_LSF_SKIPPED = "LSF.Skipped"
P_LSF_DETAILS = "LSF.Details"                        # JSON of an image-based calculation (+ inputs key)
P_LSF_ISSUES = "LSF.Issues"                          # JSON [[severity, text]] of that calculation
P_PLANNED_ACTIVITY = "Activity.GBq"                  # planned / administered Y-90 activity (lung dose)
P_REGISTRATION_SKIPPED = "Registration.Skipped"
P_REGISTRATION_CHECKED = "Registration.Checked."     # + job key
P_SEGMENT_GEOMETRY = "Segmentation.GeometryCheck"   # JSON {"key": segments key, "issues": [[severity, text]]}
P_DOSIMETRY_RESULT_KEY = "Dosimetry.ResultKey"
P_DOSIMETRY_FINGERPRINT = "Dosimetry.Fingerprint"
P_ROLE_TYPE = "RoleType."                            # + role

# Application settings
SETTING_TOOLBAR_INITIALIZED = "Taranis/ToolbarInitialized"
SETTING_SHOW_AT_STARTUP = "Taranis/ShowToolbarAtStartup"
SETTING_MODELS_FOLDER = "Taranis/ModelsFolder"
SETTING_DISCLAIMER_ACCEPTED = "Taranis/DisclaimerAccepted"


def settingBool(key, default):
    return slicer.util.settingsValue(key, default, converter=slicer.util.toBool)


def settingText(key, default=""):
    value = slicer.util.settingsValue(key, default)
    return value if isinstance(value, str) else default


def setSetting(key, value):
    settings = qt.QSettings()
    settings.setValue(key, value)
    settings.sync()


# -- Case node ---------------------------------------------------------------------------------------------------

class TaranisCase:
    """Wrapper around the case parameter node (vtkMRMLScriptedModuleNode). One case per scene."""

    def __init__(self, node):
        self.node = node

    # -- Finding and creating --

    @staticmethod
    def findNode():
        for node in slicer.util.getNodesByClass("vtkMRMLScriptedModuleNode"):
            if node.GetAttribute(CASE_ATTRIBUTE) == "1":
                return node
        return None

    @classmethod
    def find(cls):
        node = cls.findNode()
        return cls(node) if node is not None else None

    @classmethod
    def create(cls, name, caseID):
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScriptedModuleNode", "Taranis case")
        node.SetAttribute(CASE_ATTRIBUTE, "1")
        node.SetHideFromEditors(True)
        case = cls(node)
        case.setParameters({P_VERSION: CASE_VERSION, P_NAME: name, P_ID: caseID,
                            P_CREATED: datetime.datetime.now().isoformat(timespec="seconds")})
        return case

    def isValid(self):
        return self.node is not None and self.node.GetScene() is not None

    # -- Parameters --

    def param(self, key, default=""):
        value = self.node.GetParameter(key)
        return value if value else default

    def setParam(self, key, value):
        value = "" if value is None else str(value)
        if self.node.GetParameter(key) != value:
            self.node.SetParameter(key, value)

    def setParameters(self, values):
        wasModifying = self.node.StartModify()
        try:
            for key, value in values.items():
                self.setParam(key, value)
        finally:
            self.node.EndModify(wasModifying)

    def flag(self, key):
        return self.param(key) == "true"

    def setFlag(self, key, value):
        self.setParam(key, "true" if value else "")

    @property
    def name(self):
        return self.param(P_NAME)

    @property
    def caseID(self):
        return self.param(P_ID)

    def title(self):
        return f"{self.name} ({self.caseID})" if self.caseID else self.name

    # -- Roles --

    def roleNode(self, role):
        return self.node.GetNodeReference(role)

    def setRoleNode(self, role, node):
        newID = node.GetID() if node is not None else None
        oldNode = self.roleNode(role)
        if (oldNode.GetID() if oldNode else None) != newID:
            self.node.SetNodeReferenceID(role, newID)

    def roleType(self, role):
        return self.param(P_ROLE_TYPE + role) or None

    def setRoleType(self, role, imageType):
        self.setParam(P_ROLE_TYPE + role, imageType or "")

    def roleNodes(self):
        return {role: self.roleNode(role) for role in R.VOLUME_ROLES}

    # -- Processing choices --

    @property
    def mode(self):
        return self.param(P_MODE)

    def setMode(self, mode):
        self.setParam(P_MODE, mode)

    @property
    def microspheres(self):
        return self.param(P_MICROSPHERES, R.MICROSPHERES_GLASS)

    def lsfValue(self):
        text = self.param(P_LSF_VALUE)
        try:
            return float(text) if text else None
        except ValueError:
            return None

    def setLsf(self, value, source, lungMassG=None, details="", issues=""):
        self.setParameters({P_LSF_VALUE: "" if value is None else f"{float(value):.4f}", P_LSF_SOURCE: source or "",
                            P_LSF_LUNG_MASS: "" if lungMassG is None else f"{float(lungMassG):.1f}",
                            P_LSF_SKIPPED: "", P_LSF_DETAILS: details or "", P_LSF_ISSUES: issues or ""})

    def plannedActivityGBq(self):
        try:
            value = float(self.param(P_PLANNED_ACTIVITY))
        except ValueError:
            return None
        return value if value > 0 else None

    def lungMassG(self):
        try:
            return float(self.param(P_LSF_LUNG_MASS))
        except ValueError:
            return None

    def primaryVolume(self):
        for role in (R.ROLE_REFERENCE, R.ROLE_DOSIMETRY_ANATOMY, R.ROLE_DOSIMETRY):
            node = self.roleNode(role)
            if node is not None:
                return node
        return None


# -- Volume metadata -----------------------------------------------------------------------------------------------

_dicomCache = {}


def _dicomHeader(volumeNode):
    """pydicom dataset of the first instance of a DICOM-loaded volume (None if not from DICOM)."""
    uids = (volumeNode.GetAttribute("DICOM.instanceUIDs") or "").split()
    if not uids or slicer.dicomDatabase is None:
        return None
    try:
        fileName = slicer.dicomDatabase.fileForInstance(uids[0])
    except Exception:
        fileName = ""
    if not fileName or not os.path.exists(fileName):
        return None
    try:
        import pydicom
        return pydicom.dcmread(fileName, stop_before_pixels=True, force=True)
    except Exception as e:
        logging.warning(f"Taranis: could not read DICOM header of '{volumeNode.GetName()}': {e}")
        return None


def _dicomFields(volumeNode):
    key = (volumeNode.GetID(), volumeNode.GetAttribute("DICOM.instanceUIDs") or "")
    if key in _dicomCache:
        return _dicomCache[key]
    fields = {}
    dataset = _dicomHeader(volumeNode)
    if dataset is not None:
        def text(name):
            value = dataset.get(name, "")
            return str(value).strip() if value is not None else ""
        fields = {
            "modality": text("Modality"),
            "seriesDescription": text("SeriesDescription"),
            "units": text("Units"),
            "frameOfReferenceUID": text("FrameOfReferenceUID"),
            "studyUID": text("StudyInstanceUID"),
            "patientName": text("PatientName"),
            "patientID": text("PatientID"),
            "acquisitionDateTime": (R.dicomDateTime(text("AcquisitionDate"), text("AcquisitionTime"))
                                    or R.dicomDateTime(text("SeriesDate"), text("SeriesTime"))
                                    or R.dicomDateTime(text("StudyDate"), text("StudyTime"))),
            "fromDicom": True,
        }
        try:
            sequence = dataset.get("RadiopharmaceuticalInformationSequence")
            if sequence:
                item = sequence[0]
                fields["radiopharmaceutical"] = str(item.get("Radiopharmaceutical", "") or "")
                codes = item.get("RadionuclideCodeSequence")
                if codes:
                    fields["radionuclide"] = str(codes[0].get("CodeMeaning", "") or "")
        except Exception:
            pass
    _dicomCache[key] = fields
    return fields


def volumeInfo(volumeNode):
    """roles.VolumeInfo for a scalar volume node."""
    info = R.VolumeInfo(nodeID=volumeNode.GetID(), name=volumeNode.GetName() or "")
    for key, value in _dicomFields(volumeNode).items():
        setattr(info, key, value)
    imageData = volumeNode.GetImageData()
    if imageData is not None and imageData.GetNumberOfPoints() > 0:
        info.minValue = float(imageData.GetScalarRange()[0])
        bounds = [0.0] * 6
        volumeNode.GetRASBounds(bounds)
        info.bounds = tuple(bounds)
    return info


def candidateVolumes():
    """Scalar volumes the user can assign to a role (no label maps, no hidden helper volumes)."""
    volumes = []
    for node in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode"):
        if node.IsA("vtkMRMLLabelMapVolumeNode") or node.GetHideFromEditors():
            continue
        if node.GetAttribute("Taranis.Role") or node.GetAttribute("EasyReg.OverlayOf"):
            continue  # dose maps and display copies made by the suite
        volumes.append(node)
    return volumes
