"""Save the case to DICOM, filed under the patient and study of the case's images, for storage and transport:

- Scene: the whole Slicer scene (all images, segments, markups, transforms, results) zipped into a private tag of a
  secondary capture with a screenshot (3D Slicer's "Slicer Data Bundle"). Slicer's DICOM browser recognises it,
  extracts it to a temporary folder and opens the scene. Only Slicer reads it; some PACS, routers and anonymisers
  drop private tags or refuse large secondary captures.
- Segments: DICOM Segmentation (SEG, dcmqi): readable by most DICOM viewers.
- Dose map: RT Dose (Gy), on the frame of reference of the dosimetry image, so viewers can overlay it.

Every object is written to a chosen folder and added to Slicer's DICOM database (from there it can be sent to a PACS
with the DICOM browser). Research use only: the series descriptions say so."""

import datetime
import logging
import os
import shutil
import tempfile

import numpy as np
import qt
import slicer

from . import roles as R

RESEARCH_SUFFIX = "research use only"
SETTING_FOLDER = "Taranis/DicomExportFolder"
RT_DOSE_STORAGE = "1.2.840.10008.5.1.4.1.1.481.2"
MANUFACTURER = "Taranis (3D Slicer)"
DOSE_SERIES_NUMBER = 901
SEG_SERIES_NUMBER = 902
SCENE_SERIES_NUMBER = 903

# patient / study attributes copied from the reference image (keyword: DICOM tag)
STUDY_TAGS = ["PatientName", "PatientID", "PatientBirthDate", "PatientSex", "StudyInstanceUID", "StudyID",
              "StudyDate", "StudyTime", "StudyDescription", "AccessionNumber", "ReferringPhysicianName"]

# which case image the objects are filed under: the dosimetry image first (the dose map lies on its grid)
REFERENCE_ROLES = [R.ROLE_DOSIMETRY, R.ROLE_REFERENCE, R.ROLE_DOSIMETRY_ANATOMY, R.ROLE_METABOLIC,
                   R.ROLE_METABOLIC_ANATOMY, R.ROLE_CBCT]
# segments: the anatomical images first (the segments are usually drawn on them)
SEGMENT_REFERENCE_ROLES = [R.ROLE_DOSIMETRY_ANATOMY, R.ROLE_REFERENCE, R.ROLE_DOSIMETRY, R.ROLE_METABOLIC_ANATOMY,
                           R.ROLE_METABOLIC, R.ROLE_CBCT]

TOOLTIPS = {
    "scene": (
        "<b>Save scene as DICOM</b><br>"
        "Packs the <b>whole Slicer scene</b> – every image, the segments, markups (catheter tips, injection points), "
        "transforms, the case settings and all results – into one DICOM file (3D Slicer's <i>Slicer Data Bundle</i>: "
        "a secondary capture with a screenshot, the scene zipped in a private tag).<br>"
        "It is filed under the <b>patient and study of the dosimetry image</b> (or the reference image), written to a "
        "folder you choose and added to Slicer's DICOM database, from where it can be sent to a PACS.<br>"
        "To open it: load it from the DICOM browser; Slicer unpacks it and opens the scene, also on a computer that "
        "has none of the images.<br>"
        "<i>Only 3D Slicer can read it. Some PACS, routers and anonymisers drop private tags or refuse large secondary "
        "captures: check once that yours keeps it. Research use only.</i>"),
    "segments": (
        "<b>Save segments as DICOM</b><br>"
        "Writes the case segmentation (whole liver, tumours, perfused volumes, normal tissue, lungs …) as a standard "
        "<b>DICOM Segmentation object (SEG)</b> that most DICOM viewers and PACS display.<br>"
        "It references the anatomical image the segments belong to (dosimetry CT/MRI, else the reference or the "
        "dosimetry image), which must have been loaded from DICOM. Empty segments are skipped.<br>"
        "Written to a folder you choose and added to Slicer's DICOM database. Research use only."),
    "dose": (
        "<b>Save dose map as DICOM</b><br>"
        "Writes the absorbed dose map (Gy) of the last dosimetry calculation as a standard <b>RT Dose</b> object, on "
        "the frame of reference of the dosimetry image, so DICOM viewers and treatment planning systems can overlay "
        "it on the images.<br>"
        "Written to a folder you choose and added to Slicer's DICOM database. Research use only: the dose map is a "
        "model result to be reviewed, not a measured dose."),
}


class ExportError(Exception):
    pass


# -- Reference image and its patient / study ------------------------------------------------------------------

def dicomFiles(node):
    """The DICOM files a volume was loaded from (found in Slicer's DICOM database), in instance order."""
    if node is None or slicer.dicomDatabase is None:
        return []
    uids = (node.GetAttribute("DICOM.instanceUIDs") or "").split()
    files = [slicer.dicomDatabase.fileForInstance(uid) for uid in uids]
    return [f for f in files if f and os.path.exists(f)]


def referenceVolume(case, roles=REFERENCE_ROLES, preferred=None):
    """The first case image (preferred node first) that was loaded from DICOM, or None."""
    candidates = [preferred] if preferred is not None else []
    if case is not None:
        candidates += [case.roleNode(role) for role in roles]
    for node in candidates:
        if node is not None and node.IsA("vtkMRMLScalarVolumeNode") and dicomFiles(node):
            return node
    return None


def studyTags(referenceFile):
    """Patient and study attributes of a DICOM file (keyword: value, as text)."""
    import pydicom
    dataset = pydicom.dcmread(referenceFile, stop_before_pixels=True)
    tags = {}
    for keyword in STUDY_TAGS + ["FrameOfReferenceUID"]:
        value = dataset.get(keyword)
        if value is not None and str(value) != "":
            tags[keyword] = str(value)
    return tags


def _caseTags(case):
    """Without a DICOM image: the case name and ID as patient (a new study is created)."""
    tags = {}
    if case is not None:
        if case.name:
            tags["PatientName"] = case.name
        if case.caseID:
            tags["PatientID"] = case.caseID
    return tags


def _stamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe(text):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in (text or "case"))[:40]


def _addToDatabase(path):
    try:
        if slicer.dicomDatabase is not None and slicer.dicomDatabase.isOpen:
            slicer.dicomDatabase.insert(path)
            return True
    except Exception as e:
        logging.warning(f"Taranis: could not add {path} to the DICOM database: {e}")
    return False


# -- Scene (Slicer data bundle) -----------------------------------------------------------------------------------

def exportScene(folder, case=None):
    """The whole scene as a Slicer Data Bundle DICOM file in folder, filed under the study of the case's reference
    image. Returns (path, reference node or None, added to the database)."""
    import DICOMLib
    reference = referenceVolume(case)
    files = dicomFiles(reference)
    referenceFile = files[0] if files else None
    tags = studyTags(referenceFile) if referenceFile else _caseTags(case)
    tags.pop("FrameOfReferenceUID", None)
    tags["SeriesNumber"] = str(SCENE_SERIES_NUMBER)
    tags["Modality"] = "OT"   # not the modality of the reference image
    tags["Manufacturer"] = MANUFACTURER
    work = tempfile.mkdtemp("", "TaranisScene", slicer.app.temporaryPath)
    try:
        exporter = DICOMLib.DICOMExportScene(referenceFile, work)
        exporter.optionalTags = tags
        exporter.seriesDescription = f"Taranis scene ({RESEARCH_SUFFIX})"
        if not exporter.export() or not exporter.sdbFile or not os.path.exists(exporter.sdbFile):
            raise ExportError("Slicer could not write the scene bundle (see the log).")
        name = f"Taranis_scene_{_safe(case.caseID if case else '')}_{_stamp()}.dcm"
        path = os.path.join(folder, name)
        shutil.copyfile(exporter.sdbFile, path)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return path, reference, _addToDatabase(path)


# -- Segments (DICOM SEG) ---------------------------------------------------------------------------------------

def exportSegments(folder, segmentationNode, case=None):
    """The segmentation as a DICOM SEG object referencing a DICOM image of the case. Returns (path, reference
    node, added to the database)."""
    if segmentationNode is None:
        raise ExportError("No segmentation in the case.")
    import DICOMSegmentationPlugin
    role = slicer.vtkMRMLSegmentationNode.GetReferenceImageGeometryReferenceRole()
    current = segmentationNode.GetNodeReference(role)
    reference = referenceVolume(case, SEGMENT_REFERENCE_ROLES, preferred=current)
    if reference is None:
        raise ExportError("The segments need an image of the case loaded from DICOM (the dosimetry CT/MRI, the "
                          "reference image or the dosimetry image) to refer to.")
    work = tempfile.mkdtemp("", "TaranisSeg", slicer.app.temporaryPath)
    previousID = current.GetID() if current is not None else None
    exporter = None
    try:
        segmentationNode.SetNodeReferenceID(role, reference.GetID())   # what the SEG refers to (restored below)
        exporter = DICOMSegmentationPlugin.DICOMSegmentationExporter(segmentationNode)
        name = f"Taranis_segments_{_safe(case.caseID if case else '')}_{_stamp()}.dcm"
        metadata = {"SeriesDescription": f"Taranis segments ({RESEARCH_SUFFIX})",
                    "SeriesNumber": str(SEG_SERIES_NUMBER), "ContentCreatorName": "Taranis",
                    "ClinicalTrialCoordinatingCenterName": "Taranis"}
        try:
            exporter.export(work, name, metadata, skipEmpty=True)
        except DICOMSegmentationPlugin.DICOMSegmentationExporter.NoNonEmptySegmentsFoundError:
            raise ExportError("All segments are empty.")
        written = os.path.join(work, name)
        if not os.path.exists(written):
            raise ExportError("The DICOM SEG writer (dcmqi) did not write the file (see the log).")
        path = os.path.join(folder, name)
        shutil.copyfile(written, path)
    finally:
        segmentationNode.SetNodeReferenceID(role, previousID)
        if exporter is not None:
            try:
                exporter.cleanup()
            except Exception:
                pass
        shutil.rmtree(work, ignore_errors=True)
    return path, reference, _addToDatabase(path)


# -- Dose map (RT Dose) -----------------------------------------------------------------------------------------

def doseNodes():
    """Dose maps of the scene: the results of the last calculation of each dosimetry module first, then volumes
    named 'Dose Map …'."""
    from .dosimetry import REF_RESULTS_DOSE
    found = []
    for parameterNode in slicer.util.getNodesByClass("vtkMRMLScriptedModuleNode"):
        node = parameterNode.GetNodeReference(REF_RESULTS_DOSE)
        if node is not None and node not in found:
            found.append(node)
    for node in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode"):
        if node.GetName().lower().startswith("dose map") and node not in found:
            found.append(node)
    return found


def _worldMatrix(node):
    """IJK to world (RAS) of a volume, with its linear transforms; non-linear transforms are refused."""
    import vtk
    ijkToRas = vtk.vtkMatrix4x4()
    node.GetIJKToRASMatrix(ijkToRas)
    matrix = np.array([[ijkToRas.GetElement(r, c) for c in range(4)] for r in range(4)])
    parent = node.GetParentTransformNode()
    if parent is not None:
        if not parent.IsTransformToWorldLinear():
            raise ExportError("The dose map is under a non-linear transform: harden it first.")
        toWorld = vtk.vtkMatrix4x4()
        parent.GetMatrixTransformToWorld(toWorld)
        matrix = np.array([[toWorld.GetElement(r, c) for c in range(4)] for r in range(4)]) @ matrix
    return matrix


def rtDoseDataset(doseGy, ijkToRas, tags, seriesDescription=None):
    """An RT Dose dataset (pydicom) of a dose array (k, j, i) in Gy with its IJK-to-RAS matrix. tags: patient /
    study attributes and the FrameOfReferenceUID (new ones are generated when missing)."""
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    dose = np.asarray(doseGy, dtype=np.float64)
    if dose.ndim != 3:
        raise ExportError("The dose map is not a 3D volume.")
    lps = np.diag([-1.0, -1.0, 1.0, 1.0]) @ np.asarray(ijkToRas, dtype=float)
    stepI, stepJ, stepK = lps[:3, 0], lps[:3, 1], lps[:3, 2]
    spacingI, spacingJ = float(np.linalg.norm(stepI)), float(np.linalg.norm(stepJ))
    rowCosines, columnCosines = stepI / spacingI, stepJ / spacingJ
    normal = np.cross(rowCosines, columnCosines)
    along = float(stepK @ normal)
    if abs(along) < 0.999 * float(np.linalg.norm(stepK)):
        raise ExportError("The dose map grid is sheared; resample it to a rectangular grid first.")
    first = 0
    if along < 0:   # frames must follow the slice normal
        dose = dose[::-1]
        first = dose.shape[0] - 1
    origin = (lps @ np.array([0.0, 0.0, float(first), 1.0]))[:3]
    dose = np.clip(np.nan_to_num(dose), 0.0, None)
    top = float(dose.max())
    scaling = top / 4.0e9 if top > 0 else 1.0
    pixels = np.round(dose / scaling).astype(np.uint32)

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = RT_DOSE_STORAGE
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.preamble = b"\0" * 128
    now = datetime.datetime.now()
    ds.SOPClassUID = RT_DOSE_STORAGE
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.SpecificCharacterSet = "ISO_IR 192"
    for keyword in STUDY_TAGS:
        if tags.get(keyword):
            setattr(ds, keyword, tags[keyword])
    for keyword in ("PatientName", "PatientID", "PatientBirthDate", "PatientSex", "StudyDate", "StudyTime",
                    "StudyID", "AccessionNumber", "ReferringPhysicianName"):
        if keyword not in ds:
            setattr(ds, keyword, "")
    if "StudyInstanceUID" not in ds:
        ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.SeriesNumber = DOSE_SERIES_NUMBER
    ds.InstanceNumber = 1
    ds.Modality = "RTDOSE"
    ds.SeriesDescription = seriesDescription or f"Taranis dose map Gy ({RESEARCH_SUFFIX})"
    ds.Manufacturer = MANUFACTURER
    ds.ContentDate = now.strftime("%Y%m%d")
    ds.ContentTime = now.strftime("%H%M%S")
    ds.InstanceCreationDate, ds.InstanceCreationTime = ds.ContentDate, ds.ContentTime
    ds.FrameOfReferenceUID = tags.get("FrameOfReferenceUID") or generate_uid()
    ds.PositionReferenceIndicator = ""
    ds.ImagePositionPatient = [f"{v:.4f}" for v in origin]
    ds.ImageOrientationPatient = [f"{v:.6f}" for v in list(rowCosines) + list(columnCosines)]
    ds.PixelSpacing = [f"{spacingJ:.4f}", f"{spacingI:.4f}"]          # between rows, between columns
    ds.SliceThickness = f"{abs(along):.4f}"
    ds.NumberOfFrames = int(pixels.shape[0])
    ds.FrameIncrementPointer = pydicom.tag.Tag(0x3004, 0x000C)
    ds.GridFrameOffsetVector = [f"{n * abs(along):.4f}" for n in range(pixels.shape[0])]
    ds.Rows, ds.Columns = int(pixels.shape[1]), int(pixels.shape[2])
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 32
    ds.BitsStored = 32
    ds.HighBit = 31
    ds.PixelRepresentation = 0
    ds.DoseUnits = "GY"
    ds.DoseType = "PHYSICAL"
    ds.DoseSummationType = "PLAN"
    ds.DoseGridScaling = f"{scaling:.10g}"
    ds.DoseComment = f"Radioembolization dose map, Taranis ({RESEARCH_SUFFIX})"
    ds.PixelData = np.ascontiguousarray(pixels).tobytes()
    return ds


def exportDose(folder, doseNode, case=None):
    """The dose map as RT Dose, filed under the study of the dosimetry image (same frame of reference). Returns
    (path, reference node or None, added to the database)."""
    if doseNode is None or doseNode.GetImageData() is None:
        raise ExportError("No dose map: run the dosimetry first.")
    reference = referenceVolume(case)
    files = dicomFiles(reference)
    tags = studyTags(files[0]) if files else _caseTags(case)
    dataset = rtDoseDataset(slicer.util.arrayFromVolume(doseNode), _worldMatrix(doseNode), tags,
                            f"{doseNode.GetName()} ({RESEARCH_SUFFIX})"[:64])
    name = f"Taranis_dose_{_safe(case.caseID if case else '')}_{_stamp()}.dcm"
    path = os.path.join(folder, name)
    dataset.save_as(path, enforce_file_format=True)
    return path, reference, _addToDatabase(path)


# -- Interactive (toolbar buttons) -----------------------------------------------------------------------------

def _askFolder(title):
    settings = qt.QSettings()
    start = settings.value(SETTING_FOLDER) or qt.QDir.homePath()
    folder = qt.QFileDialog.getExistingDirectory(slicer.util.mainWindow(), title, start)
    if folder:
        settings.setValue(SETTING_FOLDER, folder)
    return folder


def _report(what, path, reference, added):
    lines = [f"{what} saved:", path]
    if reference is not None:
        lines.append(f"Filed under the patient and study of '{reference.GetName()}'.")
    else:
        lines.append("No case image was loaded from DICOM: a new study was created (patient = case name and ID).")
    lines.append("Added to the DICOM database (DICOM browser: send it to a PACS from there)." if added
                 else "Not added to the DICOM database.")
    slicer.util.infoDisplay("\n".join(lines), windowTitle="Saved as DICOM")


def _case():
    from .controller import WorkflowController
    controller = WorkflowController.instance()
    return controller.case if controller.isActive else None


def _run(what, function):
    folder = _askFolder(f"Folder for the {what.lower()} (DICOM)")
    if not folder:
        return None
    slicer.app.setOverrideCursor(qt.Qt.WaitCursor)
    try:
        result = function(folder)
    except ExportError as e:
        slicer.app.restoreOverrideCursor()
        slicer.util.errorDisplay(str(e), windowTitle=f"Save {what.lower()} as DICOM")
        return None
    except Exception as e:
        slicer.app.restoreOverrideCursor()
        logging.exception(f"Taranis: DICOM export of the {what.lower()} failed")
        slicer.util.errorDisplay(f"Could not save the {what.lower()} as DICOM:\n{e}",
                                 windowTitle=f"Save {what.lower()} as DICOM")
        return None
    slicer.app.restoreOverrideCursor()
    _report(what, *result)
    return result


def saveSceneAsDicom():
    case = _case()
    return _run("Scene", lambda folder: exportScene(folder, case))


def saveSegmentsAsDicom():
    case = _case()
    segmentation = case.roleNode(R.ROLE_SEGMENTATION) if case is not None else None
    if segmentation is None:
        segmentations = slicer.util.getNodesByClass("vtkMRMLSegmentationNode")
        segmentation = _choose("Segmentation to save", segmentations) if len(segmentations) > 1 else \
            (segmentations[0] if segmentations else None)
        if segmentation is None:
            if not segmentations:
                slicer.util.errorDisplay("There is no segmentation in the scene.", windowTitle="Save segments")
            return None
    return _run("Segments", lambda folder: exportSegments(folder, segmentation, case))


def saveDoseAsDicom():
    case = _case()
    nodes = doseNodes()
    if not nodes:
        slicer.util.errorDisplay("There is no dose map in the scene: run the dosimetry first.",
                                 windowTitle="Save dose map")
        return None
    node = nodes[0] if len(nodes) == 1 else _choose("Dose map to save", nodes)
    if node is None:
        return None
    return _run("Dose map", lambda folder: exportDose(folder, node, case))


def _choose(title, nodes):
    names = [n.GetName() for n in nodes]
    name = qt.QInputDialog.getItem(slicer.util.mainWindow(), title, title + ":", names, 0, False)
    if not name or name not in names:
        return None
    return nodes[names.index(name)]


# -- Icons -------------------------------------------------------------------------------------------------------

EXPORT_ICON_FILES = {"scene": "SaveDicomScene.png", "segments": "SaveDicomSegments.png", "dose": "SaveDicomDose.png"}


def exportIcon(key, size=(143, 44)):
    """'Save as DICOM: Scene / Segments / Dose map' banner (Resources/Icons) on its own dark rounded background,
    so it reads the same on light and dark Slicer themes."""
    width, height = size
    scale = int(max(1, round(slicer.app.devicePixelRatio()))) if hasattr(slicer.app, 'devicePixelRatio') else 1
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Resources", "Icons", EXPORT_ICON_FILES[key])
    pixmap = qt.QPixmap(path).scaled(width * scale, height * scale, qt.Qt.KeepAspectRatio, qt.Qt.SmoothTransformation)
    if scale != 1:
        pixmap.setDevicePixelRatio(scale)
    return qt.QIcon(pixmap)
