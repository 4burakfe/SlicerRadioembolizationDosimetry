"""Toolbar report and screenshot buttons:

- Report: the dosimetry report of the last calculation as PDF (file dialog, the dosimetry module's own export) or as
  DICOM: the report pages as images (secondary capture, one instance per page), straight into Slicer's DICOM
  database, filed under the patient and study of the case's images (dicomexport.referenceVolume).
- Screenshot: the views on screen to the clipboard, or as a secondary capture into the DICOM database (all the
  screenshots of a study in one series).

Nothing asks for a folder for DICOM: the files are written to a temporary folder and the database keeps its own
copy (from the DICOM browser they can be sent to a PACS). Research use only: the series descriptions say so.
"""

import datetime
import logging
import os
import tempfile

import numpy as np
import qt
import slicer

from . import dicomexport as DX

SECONDARY_CAPTURE = "1.2.840.10008.5.1.4.1.1.7"
REPORT_SERIES = (f"Taranis dosimetry report ({DX.RESEARCH_SUFFIX})", 904)        # (description, number)
SCREENSHOT_SERIES = (f"Taranis screenshots ({DX.RESEARCH_SUFFIX})", 905)

# a study made up for cases without DICOM images, and the screenshot series of each study (this session)
_caseStudies = {}
_screenshotSeries = {}


class CaptureError(Exception):
    pass


# -- DICOM ----------------------------------------------------------------------------------------------------

def _case():
    from .controller import WorkflowController
    controller = WorkflowController.instance()
    return controller.case if controller.isActive else None


def studyTags(case):
    """Patient and study attributes for the case's captures: those of the case's DICOM image, else the case name and
    ID with a study made up once per case (so the captures of a case stay together)."""
    from pydicom.uid import generate_uid
    reference = DX.referenceVolume(case)
    files = DX.dicomFiles(reference)
    tags = DX.studyTags(files[0]) if files else DX._caseTags(case)
    if not tags.get("StudyInstanceUID"):
        key = (case.caseID or case.name) if case is not None else ""
        tags["StudyInstanceUID"] = _caseStudies.setdefault(key, generate_uid())
    return tags


def readPngRgb(path):
    """An 8-bit PNG as a (rows, columns, 3) uint8 array, top row first."""
    import vtk
    from vtk.util import numpy_support
    reader = vtk.vtkPNGReader()
    reader.SetFileName(path)
    reader.Update()
    image = reader.GetOutput()
    columns, rows, _ = image.GetDimensions()
    scalars = image.GetPointData().GetScalars()
    if scalars is None or columns < 1 or rows < 1:
        raise CaptureError(f"Could not read {path}")
    pixels = numpy_support.vtk_to_numpy(scalars).reshape(rows, columns, -1)
    if pixels.shape[2] == 1:
        pixels = np.repeat(pixels, 3, axis=2)
    return np.ascontiguousarray(pixels[::-1, :, :3]).astype(np.uint8)     # VTK's first row is the bottom one


def secondaryCapture(pixels, tags, series, seriesUID, number, comment, pixelSpacingMm=None):
    """A secondary capture dataset (pydicom) of an RGB image (rows, columns, 3) in the study of tags."""
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = SECONDARY_CAPTURE
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.preamble = b"\0" * 128
    now = datetime.datetime.now()
    ds.SOPClassUID = SECONDARY_CAPTURE
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.SpecificCharacterSet = "ISO_IR 192"
    for keyword in DX.STUDY_TAGS:
        if tags.get(keyword):
            setattr(ds, keyword, tags[keyword])
    for keyword in ("PatientName", "PatientID", "PatientBirthDate", "PatientSex", "StudyDate", "StudyTime",
                    "StudyID", "AccessionNumber", "ReferringPhysicianName"):
        if keyword not in ds:
            setattr(ds, keyword, "")
    ds.SeriesInstanceUID = seriesUID
    ds.SeriesDescription, ds.SeriesNumber = series
    ds.InstanceNumber = number
    ds.Modality = "OT"
    ds.ConversionType = "WSD"
    ds.Manufacturer = DX.MANUFACTURER
    ds.SeriesDate = ds.ContentDate = now.strftime("%Y%m%d")
    ds.SeriesTime = ds.ContentTime = now.strftime("%H%M%S")
    ds.ImageType = ["DERIVED", "SECONDARY"]
    ds.ImageComments = comment
    ds.BurnedInAnnotation = "YES"
    ds.PatientOrientation = ""
    ds.SamplesPerPixel = 3
    ds.PhotometricInterpretation = "RGB"
    ds.PlanarConfiguration = 0
    ds.Rows, ds.Columns = int(pixels.shape[0]), int(pixels.shape[1])
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    if pixelSpacingMm:
        ds.PixelSpacing = [f"{pixelSpacingMm:.6f}"] * 2
    ds.PixelData = np.ascontiguousarray(pixels, dtype=np.uint8).tobytes()
    return ds


def _save(dataset, path):
    import pydicom
    if int(pydicom.__version_info__[0]) >= 3:
        dataset.save_as(path, enforce_file_format=True)
    else:   # Slicer versions shipping pydicom 2.x
        dataset.is_little_endian = True
        dataset.is_implicit_VR = False
        dataset.save_as(path, write_like_original=False)


def _database():
    database = slicer.dicomDatabase
    if database is None or not database.isOpen:
        raise CaptureError("Open a Slicer DICOM database first (DICOM module).")
    return database


def importCopies(paths, uids):
    """The files copied into Slicer's DICOM database (they are temporary here)."""
    import ctk
    database = _database()
    indexer = ctk.ctkDICOMIndexer()
    for path in paths:
        indexer.addFile(database, path, True)        # True: the database keeps its own copy
    indexer.waitForImportFinished()
    folders = {os.path.normcase(os.path.abspath(os.path.dirname(p))) for p in paths}
    for uid in uids:
        kept = database.fileForInstance(uid)
        if not kept or not os.path.isfile(kept) or os.path.normcase(os.path.abspath(os.path.dirname(kept))) in folders:
            raise CaptureError("The DICOM database did not keep a copy of the images.")


def storeImages(pngPaths, tags, series, seriesUID, firstNumber, comments, pixelSpacingMm=None):
    """PNG images as secondary captures in the DICOM database; their SOP Instance UIDs."""
    with tempfile.TemporaryDirectory() as work:
        paths, uids = [], []
        for offset, (png, comment) in enumerate(zip(pngPaths, comments)):
            dataset = secondaryCapture(readPngRgb(png), tags, series, seriesUID, firstNumber + offset, comment,
                                       pixelSpacingMm)
            path = os.path.join(work, f"capture{offset + 1}.dcm")
            _save(dataset, path)
            paths.append(path)
            uids.append(str(dataset.SOPInstanceUID))
        importCopies(paths, uids)
    return uids


def _run(title, function):
    slicer.app.setOverrideCursor(qt.Qt.WaitCursor)
    try:
        return function()
    except CaptureError as e:
        slicer.app.restoreOverrideCursor()
        slicer.util.errorDisplay(str(e), windowTitle=title)
    except Exception as e:
        slicer.app.restoreOverrideCursor()
        logging.exception(f"Taranis: {title} failed")
        slicer.util.errorDisplay(f"{title} failed:\n{e}", windowTitle=title)
    finally:
        slicer.app.restoreOverrideCursor()
    return None


# -- Report ---------------------------------------------------------------------------------------------------

def dosimetryWidget(controller):
    """The widget of the module holding the last dosimetry results (its results loaded), or None."""
    snapshot = controller.snapshot
    moduleName = snapshot.dosimetryResultsModule if snapshot is not None else ""
    if not moduleName or not hasattr(slicer.modules, moduleName.lower()):
        return None
    widget = slicer.util.getModuleWidget(moduleName)
    if widget is not None and not getattr(widget, "lastResult", None):
        slicer.util.selectModule(moduleName)      # a widget made just now reads its results when it is entered
        widget = slicer.util.getModuleWidget(moduleName)
    return widget if widget is not None and getattr(widget, "lastResult", None) else None


def reportAvailable(controller):
    """The report buttons work once the dosimetry has results and Qt can write PDF files."""
    from .dosimetry import pdfExportAvailable
    snapshot = controller.snapshot
    return bool(controller.isActive and snapshot is not None and snapshot.dosimetryResultsModule
                and pdfExportAvailable())


def saveReportPdf(controller):
    widget = dosimetryWidget(controller)
    if widget is None:
        slicer.util.warningDisplay("There is no dosimetry calculation yet.", windowTitle="Report")
        return
    widget.onSaveReportClicked("pdf")


def saveReportDicom(controller):
    title = "Save report as DICOM"
    widget = dosimetryWidget(controller)
    if widget is None:
        slicer.util.warningDisplay("There is no dosimetry calculation yet.", windowTitle=title)
        return

    def run():
        from pydicom.uid import generate_uid
        from .dosimetry import buildHtmlReport, reportPageImages, REPORT_IMAGE_DPI
        _database()
        tags = studyTags(controller.case)
        with tempfile.TemporaryDirectory() as work:
            pages = reportPageImages(buildHtmlReport(*widget.reportContent()), work)
            comments = [f"Dosimetry report, page {n} of {len(pages)}" for n in range(1, len(pages) + 1)]
            uids = storeImages(pages, tags, REPORT_SERIES, generate_uid(), 1, comments, 25.4 / REPORT_IMAGE_DPI)
        try:
            widget._recordSavedReport("DICOM database")
        except Exception as e:
            logging.warning(f"Taranis: could not record the saved report: {e}")
        return len(uids)

    pages = _run(title, run)
    if pages:
        slicer.util.showStatusMessage(f"Report ({pages} page{'s' if pages != 1 else ''}) added to the DICOM database "
                                      f"(series {REPORT_SERIES[1]}).", 6000)


# -- Screenshots ----------------------------------------------------------------------------------------------

def screenImage():
    """The views of the main window as a QImage (3D views included), or None."""
    import ctk
    layoutManager = slicer.app.layoutManager()
    if layoutManager is None:
        return None
    image = ctk.ctkWidgetsUtils.grabWidget(layoutManager.viewport())
    return image if image is not None and not image.isNull() else None


def screenshotToClipboard():
    image = screenImage()
    if image is None:
        slicer.util.warningDisplay("No views to capture.", windowTitle="Screenshot")
        return
    qt.QApplication.clipboard().setImage(image)
    slicer.util.showStatusMessage("Screenshot copied to the clipboard.", 3000)


def screenshotToDicom(controller):
    title = "Save screenshot as DICOM"
    image = screenImage()
    if image is None:
        slicer.util.warningDisplay("No views to capture.", windowTitle=title)
        return

    def run():
        from pydicom.uid import generate_uid
        _database()
        tags = studyTags(controller.case if controller.isActive else None)
        study = tags["StudyInstanceUID"]
        seriesUID, count = _screenshotSeries.get(study, (generate_uid(), 0))
        with tempfile.TemporaryDirectory() as work:
            png = os.path.join(work, "screenshot.png")
            if not image.convertToFormat(qt.QImage.Format_RGB888).save(png, "PNG"):
                raise CaptureError("Could not write the screenshot image.")
            storeImages([png], tags, SCREENSHOT_SERIES, seriesUID, count + 1, [f"Screenshot {count + 1}"])
        _screenshotSeries[study] = (seriesUID, count + 1)
        return True

    if _run(title, run):
        slicer.util.showStatusMessage(f"Screenshot added to the DICOM database (series {SCREENSHOT_SERIES[1]}).", 5000)
