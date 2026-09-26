"""Taranis hub: welcomes the user, manages the case and guides the radioembolization dosimetry workflow.

Steps: 1 Data & roles, 2 Registration, 3 Segmentation, 4 Lung shunt fraction, 5 Dosimetry, 6 Report.
The step modules (EasyReg, LSF calculator, patient-relative and absolute dosimetry) stay usable on their own; from
here they are opened with the inputs of the case.
"""

import contextlib
import datetime
import importlib.util
import json
import logging
import os
import re

import qt
import ctk
import vtk
import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

import TaranisLib
from TaranisLib import roles as R
from TaranisLib import workflow as W
from TaranisLib.case import (TaranisCase, candidateVolumes, volumeInfo, settingBool, settingText, setSetting,
                             SETTING_SHOW_AT_STARTUP, SETTING_MODELS_FOLDER, P_MICROSPHERES,
                             P_TREATMENT_DATETIME, P_LSF_SKIPPED, P_REGISTRATION_SKIPPED, P_REGISTRATION_CHECKED,
                             P_NAME, P_ID, P_LSF_LUNG_MASS, P_PLANNED_ACTIVITY)
from TaranisLib.controller import (WorkflowController, setSegmentRole, segmentIDs, segmentRole,
                                   persistSegmentRoles)
from TaranisLib.toolbar import WorkflowToolbar, badgeIcon
from TaranisLib import ai, segtools, memory
from TaranisLib import views as V
from TaranisLib import lsf as L
from TaranisLib import timing
from TaranisLib.case import _dicomHeader

MODULE_VERSION = "0.1"
HOME = "home"

EASYREG_MODULE = "easy_reg"
LSF_MODULE = "LSFcalc"
SAMPLE_DATA = [("RadioembolizationDosimetry2", "CT + Tc-99m MAA SPECT (patient-relative)"),
               ("RadioembolizationDosimetry1", "MRI + Y-90 PET (absolute)")]
SETTING_PERFUSED_PERCENT = "Taranis/PerfusedUptakePercentOfMax"   # new key: default changed from 15 to 5 %
DEFAULT_PERFUSED_PERCENT = 5.0   # empirical starting point (no derivation): adjust and review every result
EDITOR_NODE_TAG = "TaranisSegmentEditor"
TOTALSEG_LIVER = "totalseg_liver"
TOTALSEG_LUNGS = "totalseg_lungs"
TOTALSEG_TUMOR = "totalseg_tumor"
TOTALSEG_KEYS = {TOTALSEG_LIVER: W.SEGMENT_LIVER, TOTALSEG_LUNGS: W.SEGMENT_LUNGS, TOTALSEG_TUMOR: W.SEGMENT_TUMOR}
CANDIDATE_SUFFIX = re.compile(r"\s*\([^()]*evaluate\)\s*$")
STRUCTURE_ROLES = [W.SEGMENT_LIVER, W.SEGMENT_PERFUSED, W.SEGMENT_TUMOR, W.SEGMENT_VIABLE, W.SEGMENT_NORMAL,
                   W.SEGMENT_LUNGS]
STRUCTURE_HINTS = {
    W.SEGMENT_LIVER: "CT model (with ROI), TotalSegmentator (CT or MRI) or draw it.",
    W.SEGMENT_PERFUSED: "'Perfused volume from uptake' (Tools) or draw the territory.",
    W.SEGMENT_TUMOR: "TotalSegmentator (CT or MRI) or draw them.",
    W.SEGMENT_VIABLE: "FDG PET model (optional): reported separately, may overlap the tumours.",
    W.SEGMENT_NORMAL: "'Normal liver = liver − tumours' and 'Perfused normal = perfused − tumours' (Tools; the "
                      "tumour-to-normal ratio uses the perfused normal liver).",
    W.SEGMENT_LUNGS: "needed to calculate the LSF from the image.",
}
LSF_SOURCES = ["Planar scintigraphy", "SPECT/CT (other software)", "Pre-therapy value", "Other"]

STEP_TITLES = {
    HOME: "Case overview",
    W.STEP_DATA: "Data and imaging roles",
    W.STEP_REGISTRATION: "Registration",
    W.STEP_SEGMENTATION: "Segmentation",
    W.STEP_LSF: "Lung shunt fraction",
    W.STEP_DOSIMETRY: "Dosimetry",
    W.STEP_REPORT: "Review and report",
}

WELCOME_HTML = """
<h2>Welcome to Taranis</h2>
<p>Voxel-based dosimetry for liver radioembolization (TARE / SIRT), from pre-therapy planning with
Tc-99m MAA SPECT to post-therapy verification with Y-90 SPECT or Y-90 PET.</p>
<p>A case guides you through six steps. The toolbar at the top shows where you are, which steps are done and
any errors or warnings. Every step can be opened at any time.</p>
<ol>
<li><b>Data</b> &ndash; assign the images: the dosimetry image and at least one anatomical image are required;
a reference CT/MRI and a metabolic PET (FDG, DOTATATE) are optional.</li>
<li><b>Registration</b> &ndash; bring all images into one space (can be skipped when not needed).</li>
<li><b>Segmentation</b> &ndash; whole liver, perfused volumes, tumours and lungs.</li>
<li><b>Lung shunt fraction</b> &ndash; calculate or enter manually.</li>
<li><b>Dosimetry</b> &ndash; patient-relative (all images) or absolute (Y-90, post-therapy only).</li>
<li><b>Report</b> &ndash; review and save.</li>
</ol>
"""

RED_DISCLAIMER = "\u26a0 THIS SOFTWARE IS NOT A CERTIFIED MEDICAL DEVICE. IT IS INTENDED FOR RESEARCH PURPOSES ONLY."
DISCLAIMER = ("Taranis is NOT a medical device. Research use only. "
              "Developed by Burak Demir, MD, FEBNM – 4burakfe@gmail.com")


def _startup():
    try:
        TaranisLib.startup()
    except Exception as e:
        logging.exception(f"Taranis: could not start the workflow toolbar: {e}")


# ---------------------------------------------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------------------------------------------

class Taranis(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "Taranis"
        parent.categories = ["Nuclear Medicine"]
        parent.dependencies = []
        parent.contributors = ["Burak Demir, MD, FEBNM"]
        parent.helpText = (
            "Radioembolization dosimetry workflow: case management, imaging roles, registration, segmentation, "
            "lung shunt fraction, dosimetry and report. The workflow toolbar shows the status of every step.<br>"
            "NOT a medical device. Research use only.")
        parent.acknowledgementText = "Developed by Burak Demir, MD, FEBNM."
        iconPath = os.path.join(os.path.dirname(__file__), "Resources", "Icons", "Taranis.png")
        if os.path.exists(iconPath):
            parent.icon = qt.QIcon(iconPath)
        if not slicer.app.commandOptions().noMainWindow:
            slicer.app.connect("startupCompleted()", _startup)


# ---------------------------------------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------------------------------------

def moduleAvailable(moduleName):
    return hasattr(slicer.modules, moduleName.lower())


def moduleWidget(moduleName):
    """Select a module and return its widget (None with an error message if it is not loaded)."""
    if not moduleAvailable(moduleName):
        slicer.util.errorDisplay(f"The module '{moduleName}' is not loaded.")
        return None
    slicer.util.selectModule(moduleName)
    return slicer.util.getModuleWidget(moduleName)


def settingFloat(key, default):
    try:
        return float(slicer.util.settingsValue(key, default))
    except (TypeError, ValueError):
        return default


def trySet(description, function, *args):
    try:
        function(*args)
        return True
    except Exception as e:
        logging.warning(f"Taranis: could not set {description}: {e}")
        return False


def environmentChecks():
    """[(ok, text)] for the modules and packages the suite relies on."""
    checks = []
    for label, moduleName in (("EasyReg (registration)", EASYREG_MODULE), ("LSF calculator", LSF_MODULE),
                              ("Patient-relative dosimetry", R.MODE_MODULES[R.MODE_RELATIVE]),
                              ("Absolute dosimetry", R.MODE_MODULES[R.MODE_ABSOLUTE]),
                              ("BRAINSFit (General Registration)", "BRAINSFit")):
        ok = moduleAvailable(moduleName)
        checks.append((ok, f"{label}: {'available' if ok else 'not loaded'}"))
    for label, package in (("PyTorch", "torch"), ("MONAI", "monai"), ("pydicom", "pydicom")):
        ok = importlib.util.find_spec(package) is not None
        text = "installed" if ok else "not installed"
        if not ok and package in ("torch", "monai"):
            text += " (needed only for AI segmentation)"
        checks.append((ok, f"{label}: {text}"))
    for spec in ai.MODELS:
        path = ai.findModel(spec)
        checks.append((path is not None, f"AI model {spec.label}: " +
                       (f"found in {os.path.dirname(path)}" if path else "not found (download it in the Segmentation "
                                                                          "step)")))
    ok = ai.totalSegmentatorAvailable()
    checks.append((ok, "TotalSegmentator: " + ("available" if ok else "not installed (optional: liver on MRI)")))
    return checks


def styledLabel(text="", style="", wordWrap=True):
    label = qt.QLabel(text)
    label.setWordWrap(wordWrap)
    if style:
        label.setStyleSheet(style)
    return label


def smallGray(text=""):
    return styledLabel(text, "color: #6b7280;")


def nodeComboBox(nodeTypes, toolTip, noneEnabled=True):
    combo = slicer.qMRMLNodeComboBox()
    combo.nodeTypes = nodeTypes
    combo.selectNodeUponCreation = True
    combo.addEnabled = False
    combo.removeEnabled = False
    combo.renameEnabled = True
    combo.noneEnabled = noneEnabled
    combo.showHidden = False
    combo.showChildNodeTypes = False
    combo.setMRMLScene(slicer.mrmlScene)
    combo.setToolTip(toolTip)
    return combo


def infoLine(info):
    """Short description of a volume: modality, units, date, frame of reference."""
    if info is None:
        return ""
    parts = []
    modality = R.guessModality(info)
    if modality:
        parts.append(modality if info.modality else f"{modality} (guessed)")
    if info.units:
        parts.append(info.units)
    if info.radiopharmaceutical or info.radionuclide:
        parts.append(info.radiopharmaceutical or info.radionuclide)
    when = info.dateTime()
    if when:
        parts.append(when.strftime("%Y-%m-%d %H:%M"))
    if info.frameOfReferenceUID:
        parts.append("FoR …" + info.frameOfReferenceUID[-6:])
    if not info.fromDicom:
        parts.append("no DICOM information")
    return " · ".join(parts)


class PageStack(qt.QWidget):
    """Pages of the hub, one shown at a time. Unlike QStackedWidget its height is that of the shown page only
    (QStackedWidget keeps the height of the tallest page when pages contain word-wrapped labels, which left a large
    gap above Back / Next on the short pages)."""

    def __init__(self, parent=None):
        qt.QWidget.__init__(self, parent)
        self._layout = qt.QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._pages = []
        self._current = None

    def addWidget(self, page):
        self._pages.append(page)
        self._layout.addWidget(page)
        page.visible = self._current is None
        if self._current is None:
            self._current = page

    def setCurrentWidget(self, page):
        if page is self._current:
            return
        for other in self._pages:
            if other is not page:
                other.visible = False
        page.visible = True
        self._current = page

    def currentWidget(self):
        return self._current


class CaseDialog(qt.QDialog):
    """Asks for the case name and ID."""

    def __init__(self, title, name="", caseID="", parent=None):
        qt.QDialog.__init__(self, parent or slicer.util.mainWindow())
        self.setWindowTitle(title)
        layout = qt.QFormLayout(self)
        intro = styledLabel("Name the case and give it an ID (e.g. patient, study or research ID). Both are "
                            "shown on the toolbar and printed in the report.")
        layout.addRow(intro)
        self.nameEdit = qt.QLineEdit(name)
        self.nameEdit.setPlaceholderText("e.g. Patient 12 - MAA planning")
        layout.addRow("Case name:", self.nameEdit)
        self.idEdit = qt.QLineEdit(caseID)
        self.idEdit.setPlaceholderText("e.g. TARE-2026-012")
        layout.addRow("Case ID:", self.idEdit)
        buttons = qt.QDialogButtonBox()
        okButton = buttons.addButton(qt.QDialogButtonBox.Ok)
        okButton.setDefault(True)
        buttons.addButton(qt.QDialogButtonBox.Cancel)
        buttons.connect("accepted()", self.onAccept)
        buttons.connect("rejected()", self.reject)
        layout.addRow(buttons)
        self.setMinimumWidth(420)

    def onAccept(self):
        if not self.nameEdit.text.strip():
            slicer.util.warningDisplay("Enter a case name.", parent=self)
            return
        self.accept()

    def values(self):
        return self.nameEdit.text.strip(), self.idEdit.text.strip()


# ---------------------------------------------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------------------------------------------

class TaranisWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):

    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self._updating = False
        self._step = HOME
        self._jobsKey = None
        self._segmentsKey = None
        self._aiDefaultsKey = None
        self._editorShortcuts = False
        self._segLayoutShown = False
        self._segLayoutSegmentation = None
        self._filledSegments = None
        self._retiredJobWidgets = []

    # -- Setup --

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        _startup()
        self.controller = WorkflowController.instance()

        bannerPath = os.path.join(os.path.dirname(__file__), "Resources", "banner.png")
        if os.path.exists(bannerPath):
            banner = qt.QLabel()
            banner.setPixmap(qt.QPixmap(bannerPath).scaledToWidth(400, qt.Qt.SmoothTransformation))
            banner.setAlignment(qt.Qt.AlignCenter)
            self.layout.addWidget(banner)
            self.banner = banner

        # Step header: title, state badge, summary, issues of the step
        header = qt.QFrame()
        header.setFrameShape(qt.QFrame.StyledPanel)
        headerLayout = qt.QGridLayout(header)
        self.stepBadge = qt.QLabel()
        self.stepTitle = styledLabel("", "font-size: 15px; font-weight: bold;")
        self.stepSummary = smallGray()
        headerLayout.addWidget(self.stepBadge, 0, 0, 2, 1, qt.Qt.AlignTop)
        headerLayout.addWidget(self.stepTitle, 0, 1)
        headerLayout.addWidget(self.stepSummary, 1, 1)
        headerLayout.setColumnStretch(1, 1)
        self.issuesList = qt.QListWidget()
        self.issuesList.setWordWrap(True)
        self.issuesList.setMaximumHeight(110)
        self.issuesList.setSelectionMode(qt.QAbstractItemView.NoSelection)
        headerLayout.addWidget(self.issuesList, 2, 0, 1, 2)
        self.layout.addWidget(header)
        self.header = header

        self.stack = PageStack()
        self.layout.addWidget(self.stack)
        self.pages = {}
        for key, builder in ((HOME, self._buildHomePage), (W.STEP_DATA, self._buildDataPage),
                             (W.STEP_REGISTRATION, self._buildRegistrationPage),
                             (W.STEP_SEGMENTATION, self._buildSegmentationPage), (W.STEP_LSF, self._buildLsfPage),
                             (W.STEP_DOSIMETRY, self._buildDosimetryPage), (W.STEP_REPORT, self._buildReportPage)):
            page = qt.QWidget()
            pageLayout = qt.QVBoxLayout(page)
            pageLayout.setContentsMargins(0, 4, 0, 0)
            builder(pageLayout)
            pageLayout.addStretch(1)
            self.stack.addWidget(page)
            self.pages[key] = page

        navigation = qt.QHBoxLayout()
        self.backButton = qt.QPushButton("◀ Back")
        self.nextButton = qt.QPushButton("Next ▶")
        self.backButton.connect("clicked()", lambda: self._moveStep(-1))
        self.nextButton.connect("clicked()", lambda: self._moveStep(+1))
        navigation.addWidget(self.backButton)
        navigation.addStretch(1)
        navigation.addWidget(self.nextButton)
        self.layout.addLayout(navigation)

        self.layout.addWidget(smallGray(DISCLAIMER + f" – Taranis hub {MODULE_VERSION}"))
        self.layout.addStretch(1)

        self.controller.addListener(self.onControllerUpdated)
        self.onControllerUpdated(self.controller)

    def cleanup(self):
        try:
            self.controller.removeListener(self.onControllerUpdated)
        except Exception:
            pass
        self._entered = False
        self._updateEditorShortcuts()
        self.removeObservers()

    def enter(self):
        self._entered = True
        toolbar = WorkflowToolbar.instance(create=False)
        if toolbar is not None:
            toolbar.onHubOpened()
        self._refreshEnvironment()
        self.controller.update()
        self._updateEditorShortcuts()

    def exit(self):
        self._entered = False
        self._updateEditorShortcuts()
        try:
            self._releaseEditor()   # undo history and working images of the embedded Segment Editor
        except Exception as e:
            logging.warning(f"Taranis: {e}")

    def _updateEditorShortcuts(self):
        """Segment Editor keyboard shortcuts only while its page is shown (and no effect left active); the
        segmentation layout while the Segmentation step is shown."""
        entered = getattr(self, "_entered", False)
        onSegmentation = entered and self._step == W.STEP_SEGMENTATION
        self._setEditorActive(onSegmentation)
        self._fitStack()
        # the segmentation layout (fusion, reference + segments, 3D) is also used to check the LSF segments
        active = entered and self._step in (W.STEP_SEGMENTATION, W.STEP_LSF) and self.controller.isActive
        if active and not self._segLayoutShown:
            self._segLayoutShown = True
            qt.QTimer.singleShot(0, self._showSegmentationLayout)
        elif not active and self._segLayoutShown:
            self._segLayoutShown = False
            V.leaveLayout(self._segLayoutSegmentation)
            self._segLayoutSegmentation = None

    def _scrollToTop(self):
        """Show the top of the new page: the module panel keeps its scroll position, so after the long
        Segmentation page a short page appeared empty (scrolled past its end)."""
        widget = self.parent
        while widget is not None and not widget.inherits("QScrollArea"):
            widget = widget.parentWidget()
        if widget is not None:
            widget.verticalScrollBar().setValue(0)

    def _fitStack(self):
        """The page stack takes the height of the current page only (otherwise short pages get the height of the
        tallest one, the Segmentation page, and a large gap above Back / Next)."""
        stack = getattr(self, "stack", None)
        if stack is not None:
            stack.updateGeometry()

    def onSegmentationMonitorMode(self, mode):
        """Single / Dual monitor layout of the Segmentation step (the same choice as in the dosimetry modules)."""
        V.setLayoutMode(mode)
        self._updateMonitorButtons()
        if self._segLayoutShown:
            self._showSegmentationLayout(placeWindow=True)

    def _updateMonitorButtons(self):
        mode = V.layoutMode()
        for button, buttonMode in ((self.segSingleMonitorButton, "single"), (self.segDualMonitorButton, "dual")):
            button.blockSignals(True)
            button.setChecked(mode == buttonMode)
            button.blockSignals(False)

    def _showSegmentationLayout(self, placeWindow=False):
        self._updateMonitorButtons()
        if not self._segLayoutShown or not self.controller.isActive:
            return
        case = self.controller.case
        anatomy = case.primaryVolume()
        functional = case.roleNode(R.ROLE_DOSIMETRY)
        if functional is anatomy:
            functional = None
        segmentationNode = case.roleNode(R.ROLE_SEGMENTATION)
        try:
            if segmentationNode is not None:
                segtools.extendSegmentationGeometry(segmentationNode, anatomy, segtools.caseVolumes(case))
            V.showLayout(anatomy, functional, segmentationNode, fusionOpacity=self.fusionOpacitySlider.value,
                         placeWindow=placeWindow)
            self._segLayoutSegmentation = segmentationNode
            self._style3D(force=True)
        except Exception as e:
            logging.exception(f"Taranis: could not show the segmentation layout: {e}")

    def _style3D(self, force=False):
        """Wireframe segments in the 3D view of the segmentation layout (per role, candidates brighter)."""
        if not self._segLayoutShown:
            return
        case = self.controller.case
        segmentationNode = case.roleNode(R.ROLE_SEGMENTATION)
        if segmentationNode is not self._segLayoutSegmentation:
            V.leaveLayout(self._segLayoutSegmentation)
            self._segLayoutSegmentation = segmentationNode
            if segmentationNode is not None:
                V.showSegmentation(segmentationNode)
        viewNode = V.viewNodes().get("3d")
        snapshot = self.controller.snapshot
        if segmentationNode is None or viewNode is None or snapshot is None:
            return
        roles, candidates = {}, set()
        for segment in snapshot.segments:
            if segment.candidate:
                candidates.add(segment.segmentID)
                roles[segment.segmentID] = self._candidateRole(segmentationNode, segment.segmentID)
            else:
                roles[segment.segmentID] = segment.role
        try:
            V.styleSegmentation3D(segmentationNode, viewNode.GetID(), roles, candidates)
        except Exception as e:
            logging.warning(f"Taranis: could not show the segments in 3D: {e}")
        # a new (non-empty) segment: centre the 3D view on all segments once its surface is there
        filled = {segment.segmentID for segment in snapshot.segments if segment.voxels}
        added = filled - (self._filledSegments or set())
        firstLook = self._filledSegments is None or force
        self._filledSegments = filled
        if added and not firstLook:
            qt.QTimer.singleShot(300, lambda: V.resetThreeDView(viewNode, rotate=False))

    def onReload(self):
        """Developer reload: also reload the shared library (toolbar and controller are rebuilt)."""
        self.cleanup()
        TaranisLib.reloadLibrary()
        ScriptedLoadableModuleWidget.onReload(self)

    # -- Navigation --

    def showStep(self, step):
        if step != HOME and not self.controller.isActive:
            step = HOME
        self._step = step
        self.stack.setCurrentWidget(self.pages[step])
        if step != HOME:
            self.controller.setCurrentStep(step)
        else:
            self.controller.setCurrentStep(HOME)
        self._updateEditorShortcuts()
        self._refresh()
        qt.QTimer.singleShot(0, self._scrollToTop)

    def _moveStep(self, delta):
        order = [HOME] + W.STEP_KEYS
        index = order.index(self._step) + delta
        if delta > 0 and self._step == W.STEP_SEGMENTATION and not self._confirmSegmentation():
            return
        if 0 <= index < len(order):
            self.showStep(order[index])

    def _confirmSegmentation(self):
        """Continue from the Segmentation step: run the geometry check and show every error / warning of the step
        in a pop-up. Returns False if the user stays to fix them (soft gating: continuing is always possible)."""
        if not self.controller.isActive or self.controller.case.roleNode(R.ROLE_SEGMENTATION) is None:
            return True
        slicer.app.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            with self._grid() as grid:
                self._checkGeometry(grid)
            self.controller.update()
        except Exception as e:
            logging.warning(f"Taranis: geometry check before continuing failed: {e}")
        finally:
            slicer.app.restoreOverrideCursor()
        status = self.controller.status(W.STEP_SEGMENTATION)
        findings = [issue for issue in (status.issues if status else [])
                    if issue.severity in (W.SEVERITY_ERROR, W.SEVERITY_WARNING)]
        if not findings:
            return True
        box = qt.QMessageBox(slicer.util.mainWindow())
        box.setIcon(qt.QMessageBox.Warning)
        box.setWindowTitle("Segmentation check")
        box.setText("The segmentation check found:")
        box.setInformativeText("\n".join(f"• {issue.text}" for issue in
                                          sorted(findings, key=lambda i: W.SEVERITY_ORDER[i.severity])))
        stayButton = box.addButton("Stay and fix", qt.QMessageBox.RejectRole)
        box.addButton("Continue anyway", qt.QMessageBox.AcceptRole)
        box.setDefaultButton(stayButton)
        box.exec_()
        return box.clickedButton() is not stayButton

    # -- Controller updates --

    def onControllerUpdated(self, controller):
        if not controller.isActive and self._step != HOME:
            self._step = HOME
            self.stack.setCurrentWidget(self.pages[HOME])
        elif controller.isActive:
            wanted = controller.currentStep
            if wanted and wanted != self._step and wanted in self.pages:
                self._step = wanted
                self.stack.setCurrentWidget(self.pages[wanted])
                qt.QTimer.singleShot(0, self._scrollToTop)
        self._updateEditorShortcuts()
        self._refresh()

    def _refresh(self):
        if self._updating:
            return
        self._updating = True
        try:
            self._refreshHeader()
            self._refreshHome()
            if self.controller.isActive:
                self._refreshData()
                self._refreshRegistration()
                self._refreshSegmentation()
                self._refreshLsf()
                self._refreshDosimetry()
                self._refreshReport()
        except Exception as e:
            logging.exception(f"Taranis: could not refresh the hub: {e}")
        finally:
            self._updating = False

    def _refreshHeader(self):
        active = self.controller.isActive
        step = self._step
        order = [HOME] + W.STEP_KEYS
        self.backButton.enabled = active and order.index(step) > 0
        self.nextButton.enabled = active and order.index(step) < len(order) - 1
        self.backButton.visible = self.nextButton.visible = active
        self.header.visible = active
        if getattr(self, "banner", None) is not None:
            self.banner.visible = not active or step == HOME   # room for the step pages
        self.issuesList.clear()
        if not active:
            return
        if step == HOME:
            self.stepTitle.text = f"Case: {self.controller.case.title()}"
            self.stepSummary.text = "Overview of all steps."
            self.stepBadge.setPixmap(qt.QPixmap())
            self.issuesList.visible = False
            return
        number = W.STEP_KEYS.index(step) + 1
        status = self.controller.status(step)
        self.stepTitle.text = f"{number}. {STEP_TITLES[step]}"
        if status is None:
            return
        self.stepBadge.setPixmap(badgeIcon(status.state, str(number)).pixmap(28, 28))
        self.stepSummary.text = f"{W.STATE_STYLE[status.state][0]} – {status.summary}"
        self.issuesList.visible = bool(status.issues)
        for issue in sorted(status.issues, key=lambda i: W.SEVERITY_ORDER[i.severity]):
            state = {W.SEVERITY_ERROR: W.STATE_ERROR, W.SEVERITY_WARNING: W.STATE_WARNING}.get(
                issue.severity, W.STATE_IN_PROGRESS)
            item = qt.QListWidgetItem(badgeIcon(state), issue.text)
            self.issuesList.addItem(item)

    # =============================================================================================================
    # Home page: welcome (no case) or case overview
    # =============================================================================================================

    def _buildHomePage(self, layout):
        # -- Disclaimer (always on the Home page, as in the dosimetry modules) --
        disclaimer = qt.QLabel(RED_DISCLAIMER)
        disclaimer.setWordWrap(True)
        disclaimer.setStyleSheet("color: #ff0000; font-weight: bold;")
        layout.addWidget(disclaimer)

        # -- Welcome (no case) --
        self.welcomeBox = qt.QWidget()
        welcomeLayout = qt.QVBoxLayout(self.welcomeBox)
        welcomeLayout.setContentsMargins(0, 0, 0, 0)
        welcomeText = styledLabel(WELCOME_HTML)
        welcomeText.setTextFormat(qt.Qt.RichText)
        welcomeLayout.addWidget(welcomeText)

        self.newCaseButton = qt.QPushButton("Start new case")
        self.newCaseButton.setStyleSheet("font-weight: bold; padding: 6px;")
        self.newCaseButton.connect("clicked()", self.onNewCaseClicked)
        welcomeLayout.addWidget(self.newCaseButton)

        loadRow = qt.QHBoxLayout()
        dicomButton = qt.QPushButton("DICOM browser")
        dicomButton.setToolTip("Import and load DICOM studies. DICOM tags are used to suggest the image roles.")
        dicomButton.connect("clicked()", lambda: slicer.util.selectModule("DICOM"))
        addDataButton = qt.QPushButton("Add data...")
        addDataButton.setToolTip("Load image files (NRRD, NIfTI, ...).")
        addDataButton.connect("clicked()", slicer.util.openAddDataDialog)
        sampleButton = qt.QPushButton("Sample data")
        sampleMenu = qt.QMenu(sampleButton)
        for sampleName, description in SAMPLE_DATA:
            action = sampleMenu.addAction(description)
            action.connect("triggered()", lambda name=sampleName: self.onLoadSample(name))
        sampleButton.setMenu(sampleMenu)
        for button in (dicomButton, addDataButton, sampleButton):
            loadRow.addWidget(button)
        welcomeLayout.addLayout(loadRow)
        layout.addWidget(self.welcomeBox)

        # -- Case overview (active case) --
        self.overviewBox = qt.QWidget()
        overviewLayout = qt.QVBoxLayout(self.overviewBox)
        overviewLayout.setContentsMargins(0, 0, 0, 0)
        self.caseInfoLabel = styledLabel()
        self.caseInfoLabel.setTextFormat(qt.Qt.RichText)
        overviewLayout.addWidget(self.caseInfoLabel)
        self.overviewTable = qt.QTableWidget(len(W.STEPS), 3)
        self.overviewTable.setHorizontalHeaderLabels(["Step", "Status", "Summary"])
        self.overviewTable.verticalHeader().visible = False
        self.overviewTable.horizontalHeader().setStretchLastSection(True)
        self.overviewTable.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)
        self.overviewTable.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self.overviewTable.setSelectionMode(qt.QAbstractItemView.SingleSelection)
        self.overviewTable.setMinimumHeight(200)
        self.overviewTable.connect("cellDoubleClicked(int,int)", lambda row, column: self.showStep(W.STEP_KEYS[row]))
        overviewLayout.addWidget(self.overviewTable)
        overviewLayout.addWidget(smallGray("Double-click a step to open it."))
        self.continueButton = qt.QPushButton()
        self.continueButton.setStyleSheet("font-weight: bold; padding: 6px;")
        self.continueButton.connect("clicked()", self.onContinueClicked)
        overviewLayout.addWidget(self.continueButton)
        caseRow = qt.QHBoxLayout()
        editButton = qt.QPushButton("Edit name / ID")
        editButton.connect("clicked()", self.onEditCaseClicked)
        saveButton = qt.QPushButton("Save scene...")
        saveButton.setToolTip("Save the scene, including the case, e.g. as a .mrb bundle.")
        saveButton.connect("clicked()", lambda: slicer.app.ioManager().openSaveDataDialog())
        closeButton = qt.QPushButton("Close case")
        closeButton.setToolTip("Close the case and the scene.")
        closeButton.connect("clicked()", self.onCloseCaseClicked)
        for button in (editButton, saveButton, closeButton):
            caseRow.addWidget(button)
        overviewLayout.addLayout(caseRow)
        layout.addWidget(self.overviewBox)

        # -- Environment and settings (always) --
        environmentBox = ctk.ctkCollapsibleButton()
        environmentBox.text = "Environment check"
        environmentBox.collapsed = True
        environmentLayout = qt.QVBoxLayout(environmentBox)
        self.environmentLabel = styledLabel()
        self.environmentLabel.setTextFormat(qt.Qt.RichText)
        environmentLayout.addWidget(self.environmentLabel)
        recheckButton = qt.QPushButton("Check again")
        recheckButton.connect("clicked()", self._refreshEnvironment)
        environmentLayout.addWidget(recheckButton)
        layout.addWidget(environmentBox)

        settingsBox = ctk.ctkCollapsibleButton()
        settingsBox.text = "Settings"
        settingsBox.collapsed = True
        settingsLayout = qt.QFormLayout(settingsBox)
        self.showAtStartupCheckBox = qt.QCheckBox("Show the workflow toolbar when Slicer starts")
        self.showAtStartupCheckBox.checked = settingBool(SETTING_SHOW_AT_STARTUP, True)
        self.showAtStartupCheckBox.connect("toggled(bool)", self.onShowAtStartupToggled)
        settingsLayout.addRow(self.showAtStartupCheckBox)
        showToolbarButton = qt.QPushButton("Show workflow toolbar")
        showToolbarButton.connect("clicked()", self.onShowToolbarClicked)
        settingsLayout.addRow(showToolbarButton)
        self.modelsFolderEdit = ctk.ctkPathLineEdit()
        self.modelsFolderEdit.filters = ctk.ctkPathLineEdit.Dirs
        self.modelsFolderEdit.currentPath = settingText(SETTING_MODELS_FOLDER, "")
        self.modelsFolderEdit.setToolTip("Folder with the AI model files (.pth + .txt). Empty: the LSF calculator's "
                                         "folder.")
        self.modelsFolderEdit.connect("currentPathChanged(QString)", self.onModelsFolderChanged)
        settingsLayout.addRow("AI models folder:", self.modelsFolderEdit)
        layout.addWidget(settingsBox)

        memoryBox = ctk.ctkCollapsibleButton()
        memoryBox.text = "Memory"
        memoryBox.collapsed = True
        memoryLayout = qt.QVBoxLayout(memoryBox)
        self.memoryLabel = styledLabel("")
        memoryLayout.addWidget(self.memoryLabel)
        memoryButtons = qt.QHBoxLayout()
        reportButton = qt.QPushButton("Memory report")
        reportButton.setToolTip("What uses Slicer's memory: the largest images, segmentations and models of the "
                                "scene, temporary nodes left behind, arrays kept by the modules.")
        reportButton.connect("clicked()", self.onMemoryReport)
        memoryButtons.addWidget(reportButton)
        freeButton = qt.QPushButton("Free memory")
        freeButton.setToolTip("Remove temporary nodes left behind by interrupted operations, clear caches, collect "
                              "Python garbage and return unused memory to the system. Your data is not changed.")
        freeButton.connect("clicked()", self.onFreeMemory)
        memoryButtons.addWidget(freeButton)
        memoryLayout.addLayout(memoryButtons)
        memoryBox.connect("contentsCollapsed(bool)", lambda collapsed: None if collapsed else self._refreshMemoryLabel())
        layout.addWidget(memoryBox)

    def _refreshMemoryLabel(self):
        self.memoryLabel.text = f"Slicer: {memory.processMemoryText()}"

    def onMemoryReport(self):
        slicer.app.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            report = memory.memoryReport()
        finally:
            slicer.app.restoreOverrideCursor()
        logging.info("Taranis memory report:\n" + report)
        box = qt.QMessageBox(slicer.util.mainWindow())
        box.setWindowTitle("Taranis - memory report")
        box.setText("Memory use (also written to the Python console / log):")
        box.setDetailedText(report)
        box.setInformativeText(report.split("\n\n")[0] + "\n\n" + "\n".join(report.split("\n")[2:8]))
        box.exec_()
        self._refreshMemoryLabel()

    def onFreeMemory(self):
        slicer.app.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            if self._step != W.STEP_SEGMENTATION or not getattr(self, "_entered", False):
                self._releaseEditor()
            text = memory.releaseMemory()
        finally:
            slicer.app.restoreOverrideCursor()
        self.memoryLabel.text = f"Slicer: {memory.processMemoryText()}<br>{text}"

    def _refreshHome(self):
        active = self.controller.isActive
        self.welcomeBox.visible = not active
        self.overviewBox.visible = active
        if not active:
            return
        case = self.controller.case
        snapshot = self.controller.snapshot
        lines = [f"<b>{case.name}</b> &nbsp; ID: {case.caseID or '&ndash;'}"]
        if snapshot is not None and snapshot.dosimetryType:
            scenario = R.SCENARIO_LABELS.get(snapshot.scenario, "")
            mode = R.MODE_LABELS.get(snapshot.mode, "mode not chosen")
            lines.append(f"{R.TYPE_LABELS[snapshot.dosimetryType]} &middot; {scenario} &middot; {mode}")
        created = case.param("Created")
        if created:
            lines.append(f"<span style='color:#6b7280'>Created {created.replace('T', ' ')}</span>")
        self.caseInfoLabel.text = "<br>".join(lines)
        for row, (key, label) in enumerate(W.STEPS):
            status = self.controller.status(key)
            stepItem = qt.QTableWidgetItem(badgeIcon(status.state if status else W.STATE_NOT_STARTED, str(row + 1)),
                                           f"{row + 1}. {label}")
            self.overviewTable.setItem(row, 0, stepItem)
            self.overviewTable.setItem(row, 1, qt.QTableWidgetItem(W.STATE_STYLE[status.state][0] if status else ""))
            summaryItem = qt.QTableWidgetItem(status.summary if status else "")
            if status and status.issues:
                summaryItem.setToolTip("\n".join(issue.text for issue in status.issues))
            self.overviewTable.setItem(row, 2, summaryItem)
        self.overviewTable.resizeColumnToContents(0)
        self.overviewTable.resizeColumnToContents(1)
        nextKey = W.nextStep(self.controller.statuses)
        if nextKey:
            self.continueButton.text = f"Continue: {W.STEP_KEYS.index(nextKey) + 1}. {STEP_TITLES[nextKey]}"
            self.continueButton.enabled = True
        else:
            self.continueButton.text = "All steps finished"
            self.continueButton.enabled = False

    def _refreshEnvironment(self):
        rows = []
        for ok, text in environmentChecks():
            color, symbol = ("#2e9e44", "✓") if ok else ("#e0a100", "⚠")
            rows.append(f"<span style='color:{color}'>{symbol}</span> {text}")
        self.environmentLabel.text = "<br>".join(rows)

    # -- Home actions --

    def onNewCaseClicked(self):
        if self.controller.isActive:
            answer = slicer.util.confirmYesNoDisplay(
                f"The scene already contains the case '{self.controller.case.title()}'. One case per scene is "
                "supported.\n\nClose the scene and start a new case?", windowTitle="Taranis")
            if not answer:
                self.showStep(HOME)
                return
            slicer.mrmlScene.Clear(0)
        infos = [volumeInfo(node) for node in candidateVolumes()]
        name, caseID = R.suggestCaseIdentity(infos)
        dialog = CaseDialog("New Taranis case", name, caseID)
        if not dialog.exec_():
            return
        name, caseID = dialog.values()
        self.controller.startCase(name, caseID)
        toolbar = WorkflowToolbar.instance(create=False)
        if toolbar is not None:
            toolbar.show()
        if infos:
            self._autoAssign(silent=True)
        self.showStep(W.STEP_DATA)

    def onEditCaseClicked(self):
        case = self.controller.case
        if case is None:
            return
        dialog = CaseDialog("Edit case", case.name, case.caseID)
        if dialog.exec_():
            name, caseID = dialog.values()
            case.setParameters({P_NAME: name, P_ID: caseID})

    def onCloseCaseClicked(self):
        box = qt.QMessageBox(slicer.util.mainWindow())
        box.setWindowTitle("Close case")
        box.setText("Closing the case closes the scene: all loaded images, segmentations and results are removed.\n"
                    "Save the scene first if you want to keep them.")
        saveButton = box.addButton("Save scene...", qt.QMessageBox.ActionRole)
        closeButton = box.addButton("Close case and scene", qt.QMessageBox.DestructiveRole)
        box.addButton(qt.QMessageBox.Cancel)
        box.exec_()
        clicked = box.clickedButton()
        if clicked == saveButton:
            slicer.app.ioManager().openSaveDataDialog()
        elif clicked == closeButton:
            slicer.mrmlScene.Clear(0)

    def onContinueClicked(self):
        nextKey = W.nextStep(self.controller.statuses)
        if nextKey:
            self.showStep(nextKey)

    def onLoadSample(self, sampleName):
        try:
            import SampleData
            with slicer.util.tryWithErrorDisplay(f"Could not load the sample data '{sampleName}'."):
                SampleData.SampleDataLogic().downloadSamples(sampleName)
        except ImportError:
            slicer.util.errorDisplay("The Sample Data module is not available.")
        if self.controller.isActive:
            self._autoAssign(silent=False)

    def onShowAtStartupToggled(self, checked):
        setSetting(SETTING_SHOW_AT_STARTUP, checked)
        toolbar = WorkflowToolbar.instance(create=False)
        if toolbar is not None:
            toolbar.visibility.setShowAtStartup(checked)
            toolbar.setShowAtStartup(checked)

    def onShowToolbarClicked(self):
        WorkflowToolbar.instance().show()

    def onModelsFolderChanged(self, path):
        setSetting(SETTING_MODELS_FOLDER, path)
        self._refreshEnvironment()

    # =============================================================================================================
    # 1. Data and roles
    # =============================================================================================================

    def _buildDataPage(self, layout):
        loadRow = qt.QHBoxLayout()
        dicomButton = qt.QPushButton("DICOM browser")
        dicomButton.connect("clicked()", lambda: slicer.util.selectModule("DICOM"))
        addDataButton = qt.QPushButton("Add data...")
        addDataButton.connect("clicked()", slicer.util.openAddDataDialog)
        autoButton = qt.QPushButton("Suggest roles")
        autoButton.setToolTip("Assign the loaded images to roles from DICOM tags (modality, radiopharmaceutical, "
                              "units, frame of reference, dates) or, without DICOM, from names and image values.")
        autoButton.connect("clicked()", lambda: self._autoAssign(silent=False))
        for button in (dicomButton, addDataButton, autoButton):
            loadRow.addWidget(button)
        layout.addLayout(loadRow)
        self.suggestionLabel = smallGray()
        layout.addWidget(self.suggestionLabel)

        rolesBox = ctk.ctkCollapsibleButton()
        rolesBox.text = "Imaging roles"
        rolesLayout = qt.QGridLayout(rolesBox)
        rolesLayout.setColumnStretch(1, 1)
        self.roleSelectors = {}
        self.roleTypeCombos = {}
        self.roleInfoLabels = {}
        row = 0
        for role in R.VOLUME_ROLES:
            label, requirement, toolTip = R.ROLE_INFO[role]
            nameLabel = styledLabel(f"<b>{label}</b><br><span style='color:#6b7280'>{requirement}</span>")
            nameLabel.setToolTip(toolTip)
            selector = nodeComboBox(["vtkMRMLScalarVolumeNode"], toolTip)
            selector.connect("currentNodeChanged(vtkMRMLNode*)", lambda node, role=role: self.onRoleNodeChanged(role, node))
            typeCombo = qt.QComboBox()
            typeCombo.addItem("(type)", "")
            for imageType in R.ROLE_TYPES[role]:
                typeCombo.addItem(R.TYPE_LABELS[imageType], imageType)
            typeCombo.connect("currentIndexChanged(int)", lambda index, role=role: self.onRoleTypeChanged(role))
            infoLabel = smallGray()
            rolesLayout.addWidget(nameLabel, row, 0, 2, 1, qt.Qt.AlignTop)
            rolesLayout.addWidget(selector, row, 1)
            rolesLayout.addWidget(typeCombo, row, 2)
            rolesLayout.addWidget(infoLabel, row + 1, 1, 1, 2)
            if role in (R.ROLE_DOSIMETRY_ANATOMY, R.ROLE_METABOLIC_ANATOMY):
                nameLabel.setContentsMargins(14, 0, 0, 0)
            self.roleSelectors[role] = selector
            self.roleTypeCombos[role] = typeCombo
            self.roleInfoLabels[role] = infoLabel
            row += 2
        layout.addWidget(rolesBox)

        treatmentBox = ctk.ctkCollapsibleButton()
        treatmentBox.text = "Treatment and processing mode"
        treatmentLayout = qt.QFormLayout(treatmentBox)
        self.scenarioLabel = styledLabel()
        treatmentLayout.addRow("Scenario:", self.scenarioLabel)
        modeRow = qt.QVBoxLayout()
        self.modeButtons = {}
        self.modeGroup = qt.QButtonGroup()
        for mode in (R.MODE_RELATIVE, R.MODE_ABSOLUTE):
            button = qt.QRadioButton(R.MODE_LABELS[mode])
            self.modeGroup.addButton(button)
            button.connect("toggled(bool)", lambda checked, mode=mode: checked and self.onModeChosen(mode))
            self.modeButtons[mode] = button
            modeRow.addWidget(button)
        self.modeNoteLabel = smallGray()
        modeRow.addWidget(self.modeNoteLabel)
        treatmentLayout.addRow("Processing mode:", modeRow)
        self.microspheresCombo = qt.QComboBox()
        for key, label in R.MICROSPHERE_LABELS.items():
            self.microspheresCombo.addItem(label, key)
        self.microspheresCombo.setToolTip("Selects the isodose set used in the dosimetry modules.")
        self.microspheresCombo.connect("currentIndexChanged(int)", self.onMicrospheresChanged)
        treatmentLayout.addRow("Microspheres:", self.microspheresCombo)
        dateRow = qt.QHBoxLayout()
        self.treatmentKnownCheckBox = qt.QCheckBox("Known")
        self.treatmentDateEdit = qt.QDateTimeEdit()
        self.treatmentDateEdit.calendarPopup = True
        self.treatmentDateEdit.displayFormat = "yyyy-MM-dd HH:mm"
        self.treatmentDateEdit.setDateTime(qt.QDateTime.currentDateTime())
        self.treatmentKnownCheckBox.connect("toggled(bool)", self.onTreatmentDateChanged)
        self.treatmentDateEdit.connect("dateTimeChanged(QDateTime)", self.onTreatmentDateChanged)
        dateRow.addWidget(self.treatmentKnownCheckBox)
        dateRow.addWidget(self.treatmentDateEdit, 1)
        treatmentLayout.addRow("Administration:", dateRow)
        treatmentLayout.addRow(smallGray("The administration time will be used to fill in the time since treatment "
                                         "in absolute dosimetry."))
        layout.addWidget(treatmentBox)

    def _refreshData(self):
        case = self.controller.case
        snapshot = self.controller.snapshot
        for role in R.VOLUME_ROLES:
            node = case.roleNode(role)
            selector = self.roleSelectors[role]
            if selector.currentNode() is not node:
                selector.setCurrentNode(node)
            combo = self.roleTypeCombos[role]
            index = max(0, combo.findData(case.roleType(role) or ""))
            if combo.currentIndex != index:
                combo.setCurrentIndex(index)
            combo.enabled = node is not None
            info = snapshot.roles.get(role) if snapshot else None
            self.roleInfoLabels[role].text = infoLine(info)

        imageType = case.roleType(R.ROLE_DOSIMETRY)
        scenario = R.scenarioForType(imageType)
        self.scenarioLabel.text = R.SCENARIO_LABELS.get(scenario, "Select the dosimetry image and its type.")
        notes = []
        self.modeGroup.setExclusive(False)  # allows showing "no mode chosen"
        for mode, allowed, note in R.modeOptions(imageType):
            button = self.modeButtons[mode]
            button.enabled = allowed
            button.checked = (case.mode == mode)
            if note:
                notes.append(f"{R.MODE_LABELS[mode]}: {note}")
        self.modeGroup.setExclusive(True)
        self.modeNoteLabel.text = "\n".join(notes)
        index = max(0, self.microspheresCombo.findData(case.microspheres))
        if self.microspheresCombo.currentIndex != index:
            self.microspheresCombo.setCurrentIndex(index)
        treatment = case.param(P_TREATMENT_DATETIME)
        self.treatmentKnownCheckBox.checked = bool(treatment)
        self.treatmentDateEdit.enabled = bool(treatment)
        if treatment:
            value = qt.QDateTime.fromString(treatment, qt.Qt.ISODate)
            if value.isValid() and value != self.treatmentDateEdit.dateTime:
                self.treatmentDateEdit.setDateTime(value)

    def onRoleNodeChanged(self, role, node):
        if self._updating or not self.controller.isActive:
            return
        case = self.controller.case
        wasModifying = case.node.StartModify()
        try:
            case.setRoleNode(role, node)
            if node is None:
                case.setRoleType(role, "")
            else:
                guessed = R.guessType(volumeInfo(node))
                if guessed in R.ROLE_TYPES[role]:
                    case.setRoleType(role, guessed)
                elif case.roleType(role) not in R.ROLE_TYPES[role]:
                    case.setRoleType(role, "")
            if role == R.ROLE_DOSIMETRY:
                self._ensureValidMode()
        finally:
            case.node.EndModify(wasModifying)

    def onRoleTypeChanged(self, role):
        if self._updating or not self.controller.isActive:
            return
        case = self.controller.case
        case.setRoleType(role, self.roleTypeCombos[role].currentData or "")
        if role == R.ROLE_DOSIMETRY:
            self._ensureValidMode()

    def _ensureValidMode(self):
        case = self.controller.case
        imageType = case.roleType(R.ROLE_DOSIMETRY)
        if imageType and not R.modeAllowed(imageType, case.mode):
            case.setMode(R.defaultMode(imageType))
        elif imageType and not case.mode:
            case.setMode(R.defaultMode(imageType))

    def onModeChosen(self, mode):
        if not self._updating and self.controller.isActive:
            self.controller.case.setMode(mode)

    def onMicrospheresChanged(self, index):
        if not self._updating and self.controller.isActive:
            self.controller.case.setParam(P_MICROSPHERES, self.microspheresCombo.currentData)

    def onTreatmentDateChanged(self, *args):
        if self._updating or not self.controller.isActive:
            return
        value = ""
        if self.treatmentKnownCheckBox.checked:
            value = self.treatmentDateEdit.dateTime.toString(qt.Qt.ISODate)
        self.controller.case.setParam(P_TREATMENT_DATETIME, value)

    def _autoAssign(self, silent):
        """Suggest roles for the loaded volumes. Roles the user already assigned are kept."""
        if not self.controller.isActive:
            return
        case = self.controller.case
        infos = [volumeInfo(node) for node in candidateVolumes()]
        suggestion = R.suggestAssignments(infos)
        assignedIDs = {node.GetID() for node in case.roleNodes().values() if node is not None}
        changes = []
        wasModifying = case.node.StartModify()
        try:
            for role in R.VOLUME_ROLES:
                nodeID = suggestion.assignments.get(role)
                if not nodeID or case.roleNode(role) is not None or nodeID in assignedIDs:
                    continue
                node = slicer.mrmlScene.GetNodeByID(nodeID)
                case.setRoleNode(role, node)
                case.setRoleType(role, suggestion.types.get(role) or "")
                assignedIDs.add(nodeID)
                changes.append(f"{R.ROLE_INFO[role][0]}: {node.GetName()}")
            self._ensureValidMode()
            segmentations = [node for node in slicer.util.getNodesByClass("vtkMRMLSegmentationNode")
                             if not node.GetHideFromEditors()]
            if case.roleNode(R.ROLE_SEGMENTATION) is None and len(segmentations) == 1:
                case.setRoleNode(R.ROLE_SEGMENTATION, segmentations[0])
                changes.append(f"Segmentation: {segmentations[0].GetName()}")
        finally:
            case.node.EndModify(wasModifying)
        if case.roleNode(R.ROLE_SEGMENTATION) is not None:
            persistSegmentRoles(case.roleNode(R.ROLE_SEGMENTATION))
        text = "\n".join(changes + suggestion.notes) if (changes or suggestion.notes) else \
            "No new suggestions: all roles that could be recognized are assigned."
        self.suggestionLabel.text = text
        if not silent and not infos:
            slicer.util.infoDisplay("No images are loaded yet. Use the DICOM browser or Add data.")

    # =============================================================================================================
    # 2. Registration
    # =============================================================================================================

    def _buildRegistrationPage(self, layout):
        self.primarySpaceLabel = styledLabel()
        layout.addWidget(self.primarySpaceLabel)
        self.jobsBox = qt.QWidget()
        self.jobsLayout = qt.QVBoxLayout(self.jobsBox)
        self.jobsLayout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.jobsBox)
        self.jobWidgets = {}
        self.skipRegistrationCheckBox = qt.QCheckBox("Skip registration (images are already aligned, or not needed)")
        self.skipRegistrationCheckBox.setToolTip("You are warned if an image still needs registration, but you can "
                                                 "continue.")
        self.skipRegistrationCheckBox.connect("toggled(bool)", self.onSkipRegistrationToggled)
        layout.addWidget(self.skipRegistrationCheckBox)
        layout.addWidget(smallGray(
            "Hybrid pairs (SPECT/CT, PET/CT) are registered anatomy-to-anatomy with EasyReg; the functional image "
            "follows its CT. A functional image without its own CT (SPECT only) is registered functional-only: "
            "body outline, landmarks, by hand, then optionally a rigid refinement inside the liver. Segment the "
            "whole liver on the reference first to use the liver-based steps."))

    def _refreshRegistration(self):
        case = self.controller.case
        snapshot = self.controller.snapshot
        self.skipRegistrationCheckBox.checked = case.flag(P_REGISTRATION_SKIPPED)
        if snapshot is None or snapshot.primaryRole is None:
            self.primarySpaceLabel.text = "Assign the images in the Data step first."
            jobs = []
        else:
            primaryNode = case.roleNode(snapshot.primaryRole)
            self.primarySpaceLabel.text = (f"Primary space: <b>{primaryNode.GetName()}</b> "
                                           f"({R.ROLE_INFO[snapshot.primaryRole][0].lower()}). Segmentations are "
                                           "drawn in this space.")
            jobs = snapshot.jobs
        key = tuple((job.key, job.path, job.movingRole, job.fixedRole) for job in jobs)
        if key != self._jobsKey:
            self._jobsKey = key
            self._rebuildJobWidgets(jobs)
        nodes = case.roleNodes()
        for job in jobs:
            widgets = self.jobWidgets.get(job.key)
            if widgets is None:
                continue
            moving, fixed = nodes.get(job.movingRole), nodes.get(job.fixedRole)
            follower = nodes.get(job.followerRole) if job.followerRole else None
            text = f"Moving: <b>{moving.GetName() if moving else '?'}</b>"
            if follower is not None:
                text += f" (followed by {follower.GetName()})"
            text += f"<br>Fixed: <b>{fixed.GetName() if fixed else '?'}</b>"
            text += ("<br>Path: anatomy to anatomy (EasyReg)" if job.path == W.PATH_HYBRID
                     else "<br>Path: functional-only (no anatomical image of its own)")
            widgets["description"].text = text
            if job.alignedByAcquisition:
                state, status = W.STATE_DONE, "Aligned by acquisition (same DICOM frame of reference)."
            elif job.registered:
                state = W.STATE_WARNING if (job.path == W.PATH_FUNCTIONAL or not job.followerFollows) else W.STATE_DONE
                status = f"Registered ({job.method})."
                if not job.followerFollows:
                    status += " The functional image does not follow!"
            elif job.checked:
                state, status = W.STATE_WARNING, "Marked as aligned by the user."
            else:
                state, status = W.STATE_NOT_STARTED, "Not registered yet."
            widgets["badge"].setPixmap(badgeIcon(state).pixmap(18, 18))
            widgets["status"].text = status
            widgets["check"].checked = job.checked
            widgets["check"].enabled = not job.alignedByAcquisition

    def _rebuildJobWidgets(self, jobs):
        # Old cards are hidden and deleted by Qt; their Python wrappers are kept until the next rebuild, because
        # freeing them before Qt processes deleteLater crashes Slicer.
        self._retiredJobWidgets = list(self.jobWidgets.values())
        for widgets in self._retiredJobWidgets:
            widgets["box"].hide()
            self.jobsLayout.removeWidget(widgets["box"])
            widgets["box"].deleteLater()
        self.jobWidgets = {}
        if not jobs:
            box = qt.QGroupBox("Nothing to register")
            boxLayout = qt.QVBoxLayout(box)
            boxLayout.addWidget(styledLabel("All assigned images are in the space of the dosimetry image."))
            self.jobsLayout.addWidget(box)
            self.jobWidgets["_none"] = {"box": box}
            return
        for job in jobs:
            box = qt.QGroupBox(f"{job.label} → {R.ROLE_INFO[job.fixedRole][0].lower()}")
            boxLayout = qt.QGridLayout(box)
            badge = qt.QLabel()
            description = styledLabel()
            description.setTextFormat(qt.Qt.RichText)
            status = styledLabel()
            boxLayout.addWidget(badge, 0, 0, qt.Qt.AlignTop)
            boxLayout.addWidget(description, 0, 1)
            boxLayout.addWidget(status, 1, 1)
            buttons = qt.QHBoxLayout()
            if job.path == W.PATH_HYBRID:
                openButton = qt.QPushButton("Register with EasyReg")
            else:
                openButton = qt.QPushButton("Register with EasyReg (functional-only)")
                openButton.setToolTip("Body outline, landmarks, manual adjustment and a rigid refinement inside the "
                                      "liver. The case segmentation and its whole-liver segment are filled in.")
            openButton.connect("clicked()", lambda job=job: self.onOpenEasyReg(job))
            buttons.addWidget(openButton)
            check = qt.QCheckBox("Alignment checked")
            check.setToolTip("Confirm that the images are aligned (e.g. registered elsewhere).")
            check.connect("toggled(bool)", lambda checked, job=job: self.onJobChecked(job, checked))
            buttons.addWidget(check)
            boxLayout.addLayout(buttons, 2, 1)
            boxLayout.setColumnStretch(1, 1)
            self.jobsLayout.addWidget(box)
            self.jobWidgets[job.key] = {"box": box, "badge": badge, "description": description, "status": status,
                                        "check": check}

    def onSkipRegistrationToggled(self, checked):
        if not self._updating and self.controller.isActive:
            self.controller.case.setFlag(P_REGISTRATION_SKIPPED, checked)

    def onJobChecked(self, job, checked):
        if self._updating or not self.controller.isActive:
            return
        case = self.controller.case
        moving = case.roleNode(job.movingRole)
        case.setParam(P_REGISTRATION_CHECKED + job.key, moving.GetID() if (checked and moving) else "")

    def onOpenEasyReg(self, job):
        case = self.controller.case
        nodes = case.roleNodes()
        widget = moduleWidget(EASYREG_MODULE)
        if widget is None:
            return
        functional = job.path == W.PATH_FUNCTIONAL
        if hasattr(widget, "setPath"):
            trySet("the registration path", widget.setPath, "functional" if functional else "hybrid")
        elif functional:
            slicer.util.warningDisplay("This version of EasyReg has no functional-only path.")
        if functional:
            trySet("the SPECT/PET", widget.spectSelector.setCurrentNode, nodes.get(job.movingRole))
        else:
            trySet("the moving image", widget.spectCTSelector.setCurrentNode, nodes.get(job.movingRole))
            trySet("the following image", widget.spectSelector.setCurrentNode,
                   nodes.get(job.followerRole) if job.followerRole else None)
        trySet("the reference image", widget.referenceSelector.setCurrentNode, nodes.get(job.fixedRole))
        segmentationNode = case.roleNode(R.ROLE_SEGMENTATION)
        if functional and segmentationNode is not None and hasattr(widget, "liverSegmentationSelector"):
            trySet("the liver segmentation", widget.liverSegmentationSelector.setCurrentNode, segmentationNode)
            liver = [i for i in segmentIDs(segmentationNode) if segmentRole(segmentationNode, i) == W.SEGMENT_LIVER]
            if liver:
                index = widget.liverSegmentComboBox.findData(liver[0])
                if index >= 0:
                    widget.liverSegmentComboBox.setCurrentIndex(index)
        if hasattr(widget, "showFusionLayout"):
            trySet("the registration layout", widget.showFusionLayout)   # show the images in EasyReg's layout

    # =============================================================================================================
    # 3. Segmentation
    # =============================================================================================================

    def _buildSegmentationPage(self, layout):
        row = qt.QHBoxLayout()
        self.segmentationSelector = nodeComboBox(["vtkMRMLSegmentationNode"], "Master segmentation of the case.")
        self.segmentationSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSegmentationChanged)
        createButton = qt.QPushButton("Create")
        createButton.setToolTip("Create an empty segmentation on the primary image.")
        createButton.connect("clicked()", self.onCreateSegmentation)
        row.addWidget(qt.QLabel("Segmentation:"))
        row.addWidget(self.segmentationSelector, 1)
        row.addWidget(createButton)
        layout.addLayout(row)

        # -- Display windowing (views of the segmentation layout) --
        windowBox = ctk.ctkCollapsibleButton()
        windowBox.text = "Display windowing"
        windowLayout = qt.QGridLayout(windowBox)
        windowLayout.addWidget(qt.QLabel("CT/MRI:"), 0, 0)
        for index, preset in enumerate(V.ANATOMY_WINDOW_PRESETS):
            button = qt.QPushButton(preset[0])
            button.setToolTip(preset[4] + " (the anatomical image of the views)")
            button.connect("clicked()", lambda preset=preset: self.onAnatomyWindow(preset))
            windowLayout.addWidget(button, index // 3, 1 + index % 3)
        spectRow = (len(V.ANATOMY_WINDOW_PRESETS) + 2) // 3
        windowLayout.addWidget(qt.QLabel("SPECT/PET:"), spectRow, 0)
        spectButtons = qt.QHBoxLayout()
        for percent in V.FUNCTIONAL_WINDOW_PERCENTS:
            button = qt.QPushButton(f"0-{percent}%")
            button.setToolTip(f"SPECT/PET window from 0 to {percent}% of its maximum")
            button.connect("clicked()", lambda percent=percent: self.onFunctionalWindow(percent))
            spectButtons.addWidget(button)
        windowLayout.addLayout(spectButtons, spectRow, 1, 1, 3)
        windowLayout.addWidget(qt.QLabel("Fusion:"), spectRow + 1, 0)
        self.fusionOpacitySlider = ctk.ctkSliderWidget()
        self.fusionOpacitySlider.minimum = 0.0
        self.fusionOpacitySlider.maximum = 1.0
        self.fusionOpacitySlider.singleStep = 0.05
        self.fusionOpacitySlider.decimals = 2
        self.fusionOpacitySlider.value = V.FUSION_OPACITY
        self.fusionOpacitySlider.setToolTip("Opacity of the SPECT/PET over the anatomy in the fused views")
        self.fusionOpacitySlider.connect("valueChanged(double)", V.setFusionOpacity)
        windowLayout.addWidget(self.fusionOpacitySlider, spectRow + 1, 1, 1, 3)
        windowLayout.addWidget(qt.QLabel("Layout:"), spectRow + 2, 0)
        monitorRow = qt.QHBoxLayout()
        self.segSingleMonitorButton = qt.QPushButton("Single monitor")
        self.segSingleMonitorButton.setToolTip("All views in the main window: 2 x 2 slice views and the 3D view.")
        self.segDualMonitorButton = qt.QPushButton("Dual monitor")
        self.segDualMonitorButton.setToolTip("2 x 2 slice views in the main window; the 3D view of the segments in "
                                             "its own window, maximized on the second screen.")
        for button, mode in ((self.segSingleMonitorButton, "single"), (self.segDualMonitorButton, "dual")):
            button.setCheckable(True)
            button.connect("clicked()", lambda mode=mode: self.onSegmentationMonitorMode(mode))
            monitorRow.addWidget(button)
        windowLayout.addLayout(monitorRow, spectRow + 2, 1, 1, 3)
        layout.addWidget(windowBox)

        # -- Structures checklist --
        checklistBox = ctk.ctkCollapsibleButton()
        checklistBox.text = "Structures"
        checklistLayout = qt.QGridLayout(checklistBox)
        checklistLayout.setColumnStretch(1, 1)
        self.structureLabels = {}
        for index, role in enumerate(STRUCTURE_ROLES):
            nameLabel = styledLabel(f"<b>{W.SEGMENT_ROLES[role][0]}</b>", wordWrap=False)
            statusLabel = smallGray()
            addButton = qt.QPushButton("+ Empty")
            addButton.setToolTip("Add an empty segment with this role and select it in the Segment Editor below.")
            addButton.connect("clicked()", lambda role=role: self.onAddSegment(role))
            checklistLayout.addWidget(nameLabel, index, 0)
            checklistLayout.addWidget(statusLabel, index, 1)
            checklistLayout.addWidget(addButton, index, 2)
            self.structureLabels[role] = statusLabel
        layout.addWidget(checklistBox)

        # -- Segments and roles --
        segmentsBox = ctk.ctkCollapsibleButton()
        segmentsBox.text = "Segments and roles"
        segmentsLayout = qt.QVBoxLayout(segmentsBox)
        self.segmentsTable = qt.QTableWidget(0, 4)
        self.segmentsTable.setHorizontalHeaderLabels(["Segment", "Role", "Volume", ""])
        self.segmentsTable.verticalHeader().visible = False
        self.segmentsTable.horizontalHeader().setStretchLastSection(True)
        self.segmentsTable.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)
        self.segmentsTable.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self.segmentsTable.setSelectionMode(qt.QAbstractItemView.SingleSelection)
        self.segmentsTable.setMinimumHeight(200)
        self.segmentsTable.connect("itemSelectionChanged()", self.onSegmentsTableSelection)
        segmentsLayout.addWidget(self.segmentsTable)
        segmentsLayout.addWidget(smallGray("Roles are stored with the segmentation and used by the dosimetry "
                                           "modules. AI results appear in yellow until you accept or discard them."))
        acceptHint = qt.QLabel("After accepting, an AI segment is an ordinary segment: edit it freely in the "
                               "<b>Segment Editor below</b> (paint, erase, scissors, islands, smoothing...).")
        acceptHint.setWordWrap(True)
        acceptHint.setTextFormat(qt.Qt.RichText)
        segmentsLayout.addWidget(acceptHint)
        layout.addWidget(segmentsBox)

        # -- AI segmentation --
        aiBox = ctk.ctkCollapsibleButton()
        aiBox.text = "AI segmentation"
        aiLayout = qt.QGridLayout(aiBox)
        aiLayout.setColumnStretch(1, 1)
        self.aiRows = {}
        rows = [(spec.key, spec.label, spec.description) for spec in ai.MODELS]
        rows += [(TOTALSEG_LIVER, "Whole liver (TotalSegmentator)",
                  "TotalSegmentator on a CT (task 'total') or an MRI (task 'total_mr'); needs the TotalSegmentator "
                  "extension."),
                 (TOTALSEG_LUNGS, "Lungs (TotalSegmentator)",
                  "TotalSegmentator lungs (CT: the five lobes merged; MRI: left and right lung)."),
                 (TOTALSEG_TUMOR, "Liver tumours (TotalSegmentator)",
                  "TotalSegmentator liver lesions on a CT ('liver_lesions') or an MRI ('liver_lesions_mr'). "
                  "These tasks need a TotalSegmentator licence (free for non-commercial use).")]
        for index, (key, label, description) in enumerate(rows):
            nameLabel = qt.QLabel(label)
            nameLabel.setToolTip(description)
            inputCombo = nodeComboBox(["vtkMRMLScalarVolumeNode"], f"Input image of '{label}'.")
            inputCombo.renameEnabled = False
            runButton = qt.QPushButton("Run")
            runButton.setToolTip(description)
            runButton.connect("clicked()", lambda key=key: self.onRunAi(key))
            downloadButton = qt.QPushButton("Download")
            downloadButton.setToolTip("Download the model from the Taranis GitHub release.")
            downloadButton.connect("clicked()", lambda key=key: self.onDownloadModel(key))
            stateLabel = smallGray()
            aiLayout.addWidget(nameLabel, 2 * index, 0)
            aiLayout.addWidget(inputCombo, 2 * index, 1)
            spec = ai.MODELS_BY_KEY.get(key)
            roiButton = None
            if spec is not None and spec.roiSizeMM:
                roiButton = qt.QPushButton("ROI")
                roiButton.setToolTip("Show the region of interest of this model (created on first use, centred on "
                                     "the slice views). Drag its handles so that it covers the whole organ, then "
                                     "Run.")
                roiButton.connect("clicked()", lambda key=key: self.onShowModelRoi(key))
                aiLayout.addWidget(roiButton, 2 * index, 2)
            aiLayout.addWidget(downloadButton, 2 * index, 3)
            aiLayout.addWidget(runButton, 2 * index, 4)
            aiLayout.addWidget(stateLabel, 2 * index + 1, 1, 1, 4)
            self.aiRows[key] = dict(input=inputCombo, run=runButton, download=downloadButton, state=stateLabel,
                                    roi=roiButton)
        base = 2 * len(rows)
        self.aiCpuCheck = qt.QCheckBox("Run on the CPU (slower; use when the GPU memory is too small)")
        aiLayout.addWidget(self.aiCpuCheck, base, 0, 1, 5)
        aiLayout.addWidget(smallGray("Results are added as candidates (yellow, '… (AI - evaluate)'). Check them in "
                                     "the Segment Editor below, correct if needed, then Accept (or Discard) them in "
                                     "the segment table. The tumour model needs the whole liver and a registered "
                                     "FDG PET."), base + 1, 0, 1, 5)
        self.aiStatus = styledLabel("")
        aiLayout.addWidget(self.aiStatus, base + 2, 0, 1, 5)
        # never squeezed by the boxes below (e.g. when Tools is expanded)
        aiBox.setSizePolicy(qt.QSizePolicy.Preferred, qt.QSizePolicy.Minimum)
        aiBox.setMinimumHeight(470)
        layout.addWidget(aiBox)

        # -- Tools --
        toolsBox = ctk.ctkCollapsibleButton()
        toolsBox.text = "Tools"
        toolsLayout = qt.QGridLayout(toolsBox)
        toolsLayout.setColumnStretch(1, 1)
        normalButton = qt.QPushButton("Normal liver = liver − tumours")
        normalButton.setToolTip("Create or update the normal-liver segment: the whole liver without any tumour.")
        normalButton.connect("clicked()", lambda: self.runSegmentTool(segtools.makeNormalLiver))
        toolsLayout.addWidget(normalButton, 0, 0, 1, 2)
        perfusedNormalButton = qt.QPushButton("Perfused normal = perfused − tumours")
        perfusedNormalButton.setToolTip("Create or update the perfused normal liver: each perfused volume (inside the "
                                        "whole liver) without any tumour, one segment per perfused volume. The dose "
                                        "checks use it for the tumour-to-normal ratio and the normal tissue dose.")
        perfusedNormalButton.connect("clicked()", lambda: self.runSegmentTool(segtools.makePerfusedNormal))
        toolsLayout.addWidget(perfusedNormalButton, 0, 2, 1, 2)
        toolsLayout.addWidget(smallGray("Clipping to the liver (Logical operators / masking) and splitting into "
                                        "lesions (Islands) are in the Segment Editor below."), 1, 0, 1, 4)

        perfusedButton = qt.QPushButton("Perfused volume from uptake")
        perfusedButton.setToolTip("Candidate perfused volume: liver voxels of the dosimetry image (MAA / Y-90) "
                                  "with at least this percentage of the maximum uptake in the liver; holes filled, "
                                  "regions smaller than 5 mL removed.")
        perfusedButton.connect("clicked()", self.onPerfusedFromUptake)
        self.perfusedPercentSpin = qt.QDoubleSpinBox()
        self.perfusedPercentSpin.setRange(1.0, 90.0)
        self.perfusedPercentSpin.setDecimals(0)
        self.perfusedPercentSpin.setValue(settingFloat(SETTING_PERFUSED_PERCENT, DEFAULT_PERFUSED_PERCENT))
        self.perfusedPercentSpin.setToolTip("Empirical threshold (default 5 %, not derived from a calculation): "
                                            "percentage of the robust maximum (99.9th percentile) of the uptake "
                                            "inside the whole liver. Review the candidate in the views.")
        self.perfusedPercentSpin.setSuffix(" % of max")
        toolsLayout.addWidget(perfusedButton, 2, 0, 1, 2)
        toolsLayout.addWidget(qt.QLabel("threshold"), 2, 2, qt.Qt.AlignRight)
        toolsLayout.addWidget(self.perfusedPercentSpin, 2, 3)

        geometryButton = qt.QPushButton("Check geometry")
        geometryButton.setToolTip("Look for overlaps, segments outside the liver and unusual volumes. The result "
                                  "is shown on the toolbar until the segments change.")
        geometryButton.connect("clicked()", self.onCheckGeometry)
        toolsLayout.addWidget(geometryButton, 3, 0, 1, 2)
        self.toolStatus = styledLabel("")
        toolsLayout.addWidget(self.toolStatus, 4, 0, 1, 4)
        layout.addWidget(toolsBox)

        # -- Embedded Segment Editor --
        editorBox = ctk.ctkCollapsibleButton()
        editorBox.text = "Segment Editor"
        editorLayout = qt.QVBoxLayout(editorBox)
        self.segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
        self.segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
        for propertyName, value in (("segmentationNodeSelectorVisible", False), ("maximumNumberOfUndoStates", 10),
                                    ("switchToSegmentationsButtonVisible", False)):
            try:
                setattr(self.segmentEditorWidget, propertyName, value)
            except Exception:
                pass
        editorLayout.addWidget(self.segmentEditorWidget)
        layout.addWidget(editorBox)
        self.editorBox = editorBox
        self._editorShortcuts = False

    # -- Segment Editor (embedded) --

    def _editorNode(self):
        node = slicer.mrmlScene.GetSingletonNode(EDITOR_NODE_TAG, "vtkMRMLSegmentEditorNode")
        if node is None:
            node = slicer.vtkMRMLSegmentEditorNode()
            node.SetSingletonTag(EDITOR_NODE_TAG)
            node = slicer.mrmlScene.AddNode(node)
        return node

    def _syncEditor(self):
        editor = self.segmentEditorWidget
        node = self._editorNode()
        if editor.mrmlSegmentEditorNode() is not node:
            editor.setMRMLSegmentEditorNode(node)
        if not self.hasObserver(node, vtk.vtkCommand.ModifiedEvent, self._enforceAllowOverlap):
            self.addObserver(node, vtk.vtkCommand.ModifiedEvent, self._enforceAllowOverlap)
        self._enforceAllowOverlap(node)
        case = self.controller.case
        segmentationNode = case.roleNode(R.ROLE_SEGMENTATION) if case is not None else None
        self._watchSegmentLayers(segmentationNode)
        if not (getattr(self, "_entered", False) and self._step == W.STEP_SEGMENTATION):
            self._releaseEditor()
            return
        if editor.segmentationNode() is not segmentationNode:
            editor.setSegmentationNode(segmentationNode)
            restored = getattr(self, "_releasedEditorState", None)
            self._releasedEditorState = None
            if restored and segmentationNode is not None and restored[0] == segmentationNode.GetID():
                source = slicer.mrmlScene.GetNodeByID(restored[1]) if restored[1] else None
                if source is not None and self._editorSourceVolume() is None:
                    self._setEditorSourceVolume(source)
                if restored[2] and segmentationNode.GetSegmentation().GetSegment(restored[2]) is not None:
                    editor.setCurrentSegmentID(restored[2])
        if segmentationNode is not None and self._editorSourceVolume() is None:
            primary = case.primaryVolume()
            if primary is not None:
                self._setEditorSourceVolume(primary)

    def _releaseEditor(self):
        """Outside the Segmentation step the embedded Segment Editor lets go of the segmentation and the source
        image: its undo history (copies of the edited segments) and working images (source image resampled to the
        segmentation, modifier and mask labelmaps) are freed. Segmentation, source image and selected segment are
        restored when the step is shown again."""
        editor = self.segmentEditorWidget
        segmentationNode = editor.segmentationNode()
        if segmentationNode is None:
            return
        source = self._editorSourceVolume()
        self._releasedEditorState = (segmentationNode.GetID(), source.GetID() if source is not None else "",
                                     editor.currentSegmentID())
        try:
            editor.setActiveEffect(None)
            editor.setSegmentationNode(None)
            self._setEditorSourceVolume(None)
        except Exception as e:
            logging.warning(f"Taranis: could not release the Segment Editor: {e}")
        memory.collectSoon()

    def _watchSegmentLayers(self, segmentationNode):
        """Every segment of the case segmentation on its own labelmap layer: now, and for each segment added later
        (the Segment Editor's Add button puts a new segment on the first segment's layer, where painting it takes
        voxels from that segment even with "Allow overlap")."""
        previous = getattr(self, "_layerWatched", None)
        if previous is not None and previous is not segmentationNode:
            self.removeObserver(previous, slicer.vtkSegmentation.SegmentAdded, self._onSegmentAddedLayer)
        self._layerWatched = segmentationNode
        if segmentationNode is None:
            return
        try:
            moved = segtools.separateSharedLayers(segmentationNode)
            if moved:
                logging.info(f"Taranis: {moved} segment(s) moved to their own labelmap layer.")
        except Exception as e:
            logging.warning(f"Taranis: could not separate the segment layers: {e}")
        if not self.hasObserver(segmentationNode, slicer.vtkSegmentation.SegmentAdded, self._onSegmentAddedLayer):
            self.addObserver(segmentationNode, slicer.vtkSegmentation.SegmentAdded, self._onSegmentAddedLayer)

    def _onSegmentAddedLayer(self, caller, event):
        # after the event: the segment is fully added (and before the user can paint it)
        qt.QTimer.singleShot(0, lambda node=caller: self._separateLater(node))

    def _separateLater(self, segmentationNode):
        if segmentationNode is None or segmentationNode.GetScene() is None:
            return
        try:
            segtools.separateSharedLayers(segmentationNode)
        except Exception as e:
            logging.warning(f"Taranis: could not separate the segment layers: {e}")

    def _enforceAllowOverlap(self, node, event=None):
        """Segments of the case may overlap (tumours and perfused volumes inside the whole liver, normal tissue):
        the editor never removes painted voxels from other segments ("Modify other segments: Allow overlap")."""
        allow = slicer.vtkMRMLSegmentEditorNode.OverwriteNone
        if node.GetOverwriteMode() != allow:
            node.SetOverwriteMode(allow)
        self._isolateForFillBetweenSlices(node)

    FILL_BETWEEN_SLICES = "Fill between slices"

    def _isolateForFillBetweenSlices(self, editorNode):
        """Slicer's Fill between slices interpolates all visible segments together in one label image, so
        overlapping segments (a tumour inside the whole liver) cannot be filled: the effect initialises but does
        nothing. While it is the active effect only the selected segment is shown; the others come back when
        another effect (or none) is selected."""
        try:
            active = editorNode.GetActiveEffectName() if hasattr(editorNode, "GetActiveEffectName") else ""
        except Exception:
            active = ""
        if active != self.FILL_BETWEEN_SLICES:
            self._restoreFillIsolation()
            return
        segmentationNode = editorNode.GetSegmentationNode()
        selectedID = editorNode.GetSelectedSegmentID()
        if segmentationNode is None or not selectedID:
            return
        state = getattr(self, "_fillIsolation", None)
        if state is not None and (state["node"] is not segmentationNode):
            self._restoreFillIsolation()
            state = None
        if state is not None and state["selected"] == selectedID:
            return
        displayNode = segmentationNode.GetDisplayNode()
        if displayNode is None:
            return
        hidden = list(state["hidden"]) if state is not None else []
        if selectedID in hidden:   # another segment selected while the effect is active: show it
            hidden.remove(selectedID)
        displayNode.SetSegmentVisibility(selectedID, True)
        for segmentID in segmentIDs(segmentationNode):
            if segmentID != selectedID and displayNode.GetSegmentVisibility(segmentID):
                displayNode.SetSegmentVisibility(segmentID, False)
                hidden.append(segmentID)
        self._fillIsolation = {"node": segmentationNode, "selected": selectedID, "hidden": hidden}
        if hidden:
            slicer.util.showStatusMessage(
                "Fill between slices works on the visible segments and cannot handle overlapping ones: only the "
                "selected segment is shown; the others come back when you leave the effect.", 8000)

    def _restoreFillIsolation(self):
        state = getattr(self, "_fillIsolation", None)
        self._fillIsolation = None
        if state is None or state["node"].GetScene() is None:
            return
        displayNode = state["node"].GetDisplayNode()
        segmentation = state["node"].GetSegmentation()
        if displayNode is None:
            return
        for segmentID in state["hidden"]:
            if segmentation.GetSegment(segmentID) is not None:
                displayNode.SetSegmentVisibility(segmentID, True)

    def _editorSourceVolume(self):
        editor = self.segmentEditorWidget
        return editor.sourceVolumeNode() if hasattr(editor, "sourceVolumeNode") else editor.masterVolumeNode()

    def _setEditorSourceVolume(self, volume):
        editor = self.segmentEditorWidget
        if hasattr(editor, "setSourceVolumeNode"):
            editor.setSourceVolumeNode(volume)
        else:
            editor.setMasterVolumeNode(volume)

    def _setEditorActive(self, active):
        editor = getattr(self, "segmentEditorWidget", None)
        if editor is None or active == self._editorShortcuts:
            return
        try:
            if active:
                editor.installKeyboardShortcuts()
            else:
                editor.setActiveEffect(None)
                editor.uninstallKeyboardShortcuts()
        except Exception as e:
            logging.warning(f"Taranis: Segment Editor shortcuts: {e}")
        self._editorShortcuts = active

    def openSegmentEditor(self, segmentID=None):
        """Select a segment in the embedded Segment Editor (Segmentation step)."""
        if self._step != W.STEP_SEGMENTATION:
            self.showStep(W.STEP_SEGMENTATION)
        self._syncEditor()
        self.editorBox.collapsed = False
        if segmentID:
            self.segmentEditorWidget.setCurrentSegmentID(segmentID)

    # -- Refresh --

    def _refreshSegmentation(self):
        case = self.controller.case
        segmentationNode = case.roleNode(R.ROLE_SEGMENTATION)
        if self.segmentationSelector.currentNode() is not segmentationNode:
            self.segmentationSelector.setCurrentNode(segmentationNode)
        try:
            self._syncEditor()
        except Exception as e:
            logging.warning(f"Taranis: could not update the Segment Editor: {e}")
        snapshot = self.controller.snapshot
        segments = snapshot.segments if snapshot else []
        self._refreshStructures(snapshot, segments)
        self._refreshAiRows(case)
        key = tuple((s.segmentID, s.name, s.role, s.empty, s.candidate, s.voxels) for s in segments)
        if key == self._segmentsKey and (not self._segLayoutShown or segmentationNode is self._segLayoutSegmentation):
            return
        self._segmentsKey = key
        self._style3D()
        selected = self._selectedSegmentID()
        self.segmentsTable.setRowCount(len(segments))
        segmentationNode = case.roleNode(R.ROLE_SEGMENTATION)
        for row, segment in enumerate(segments):
            nameItem = qt.QTableWidgetItem(segment.name)
            nameItem.setData(qt.Qt.UserRole, segment.segmentID)
            self.segmentsTable.setItem(row, 0, nameItem)
            combo = qt.QComboBox()
            combo.addItem("(no role)", "")
            for role in W.SEGMENT_ROLE_KEYS:
                combo.addItem(W.SEGMENT_ROLES[role][0], role)
            if segment.candidate:
                intended = self._candidateRole(segmentationNode, segment.segmentID)
                combo.setCurrentIndex(max(0, combo.findData(intended)))
                combo.setToolTip("Role the AI result gets when accepted.")
                combo.connect("currentIndexChanged(int)",
                              lambda index, segmentID=segment.segmentID, combo=combo: self.onCandidateRoleChanged(
                                  segmentID, combo.itemData(index)))
            else:
                combo.setCurrentIndex(max(0, combo.findData(segment.role)))
                combo.connect("currentIndexChanged(int)",
                              lambda index, segmentID=segment.segmentID, combo=combo: self.onSegmentRoleChanged(
                                  segmentID, combo.itemData(index)))
            self.segmentsTable.setCellWidget(row, 1, combo)
            volume = "empty" if segment.empty else (f"{segment.volumeML:.1f} mL" if segment.volumeML else "")
            self.segmentsTable.setItem(row, 2, qt.QTableWidgetItem(volume))
            if segment.candidate:
                actions = qt.QWidget()
                actionsLayout = qt.QHBoxLayout(actions)
                actionsLayout.setContentsMargins(2, 0, 2, 0)
                acceptButton = qt.QPushButton("Accept")
                acceptButton.connect("clicked()", lambda segmentID=segment.segmentID: self.onAcceptCandidate(segmentID))
                discardButton = qt.QPushButton("Discard")
                discardButton.connect("clicked()",
                                      lambda segmentID=segment.segmentID: self.onDiscardCandidate(segmentID))
                actionsLayout.addWidget(acceptButton)
                actionsLayout.addWidget(discardButton)
                self.segmentsTable.setCellWidget(row, 3, actions)
            else:
                self.segmentsTable.removeCellWidget(row, 3)
                self.segmentsTable.setItem(row, 3, qt.QTableWidgetItem(""))
            if segment.segmentID == selected:
                self.segmentsTable.selectRow(row)
        self.segmentsTable.resizeColumnToContents(0)
        self.segmentsTable.resizeColumnToContents(2)

    def _refreshStructures(self, snapshot, segments):
        relative = snapshot is not None and snapshot.mode == R.MODE_RELATIVE
        for role, label in self.structureLabels.items():
            found = [s for s in segments if s.role == role]
            candidates = [s for s in segments if s.candidate and
                          self._candidateRole(self.controller.case.roleNode(R.ROLE_SEGMENTATION), s.segmentID) == role]
            if found:
                total = sum(s.volumeML or 0.0 for s in found)
                text = f"<span style='color:#16a34a'>✔</span> {len(found)} segment(s)"
                if total:
                    text += f", {total:.0f} mL" if total >= 10 else f", {total:.1f} mL"
                if any(s.empty for s in found):
                    text += " <span style='color:#d97706'>(empty segment)</span>"
            else:
                required = role == W.SEGMENT_LIVER or (role == W.SEGMENT_PERFUSED and relative)
                color = "#dc2626" if role == W.SEGMENT_LIVER else ("#d97706" if required else "#6b7280")
                text = f"<span style='color:{color}'>{'missing' if required else 'none'}</span>"
                text += f" – {STRUCTURE_HINTS[role]}"
            if candidates:
                text += f" · <span style='color:#b45309'>{len(candidates)} AI candidate(s) to review</span>"
            label.setText(text)

    def _refreshAiRows(self, case):
        rolesKey = tuple((role, node.GetID() if node else "") for role, node in sorted(case.roleNodes().items()))
        if rolesKey != self._aiDefaultsKey:
            self._aiDefaultsKey = rolesKey
            for key, row in self.aiRows.items():
                default = self._defaultAiInput(case, key)
                if default is not None:
                    row["input"].setCurrentNode(default)
        missingPackage = ai.missingPackage()
        for key, row in self.aiRows.items():
            if key in TOTALSEG_KEYS:
                available = ai.totalSegmentatorAvailable()
                row["download"].visible = False
                row["run"].enabled = available
                text = ("TotalSegmentator installed (the first run of a task downloads its weights)." if available
                        else "Install the TotalSegmentator extension (Extensions Manager).")
                if available and key == TOTALSEG_TUMOR:
                    text += " Needs a TotalSegmentator licence (free for non-commercial use)."
                row["state"].text = text
                continue
            spec = ai.MODELS_BY_KEY[key]
            path = ai.findModel(spec)
            row["download"].visible = path is None
            row["run"].enabled = path is not None and missingPackage is None
            if missingPackage:
                row["state"].text = f"Python package '{missingPackage}' missing (PyTorch extension / MONAI)."
            elif path is None:
                row["state"].text = f"{spec.fileName} not found: download it or set the AI models folder (Home)."
            else:
                roi = ai.modelRoi(spec) if spec.roiSizeMM else None
                text = f"Model {spec.fileName} found."
                if spec.roiSizeMM:
                    text += " ROI placed." if roi is not None else " Place the ROI first (button 'ROI')."
                row["state"].text = text
                row["state"].setToolTip(path)

    def _defaultAiInput(self, case, key):
        def ofType(roles, wanted):
            for role in roles:
                node = case.roleNode(role)
                if node is not None and case.roleType(role) in wanted:
                    return node
            return None
        anatomical = [R.ROLE_DOSIMETRY_ANATOMY, R.ROLE_REFERENCE, R.ROLE_METABOLIC_ANATOMY]
        if key == "tumor_fdg":
            return case.roleNode(R.ROLE_METABOLIC)
        if key in TOTALSEG_KEYS:
            return ofType([R.ROLE_REFERENCE] + anatomical, (R.TYPE_CT, R.TYPE_MRI))
        return ofType(anatomical, (R.TYPE_CT,))

    def _candidateRole(self, segmentationNode, segmentID):
        if segmentationNode is None:
            return ""
        segment = segmentationNode.GetSegmentation().GetSegment(segmentID)
        if segment is None:
            return ""
        role = ai.candidateRole(segment)
        return role or W.guessSegmentRole(CANDIDATE_SUFFIX.sub("", segment.GetName()))

    def _selectedSegmentID(self):
        rows = self.segmentsTable.selectionModel().selectedRows()
        if not rows:
            return None
        item = self.segmentsTable.item(rows[0].row(), 0)
        return item.data(qt.Qt.UserRole) if item is not None else None

    # -- Segment actions --

    def onSegmentsTableSelection(self):
        segmentID = self._selectedSegmentID()
        if segmentID and not self._updating:
            try:
                self.segmentEditorWidget.setCurrentSegmentID(segmentID)
            except Exception:
                pass

    def onSegmentationChanged(self, node):
        if not self._updating and self.controller.isActive:
            self.controller.case.setRoleNode(R.ROLE_SEGMENTATION, node)
            persistSegmentRoles(node)

    def onCreateSegmentation(self):
        case = self.controller.case
        primary = case.primaryVolume()
        if primary is None:
            slicer.util.errorDisplay("Assign the images in the Data step first.")
            return None
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", f"{case.name} segmentation")
        node.CreateDefaultDisplayNodes()
        node.SetReferenceImageGeometryParameterFromVolumeNode(primary)
        # voxel size of the primary image, extent of all case images (e.g. lungs on a SPECT/CT above a liver MRI)
        segtools.extendSegmentationGeometry(node, primary, segtools.caseVolumes(case))
        case.setRoleNode(R.ROLE_SEGMENTATION, node)
        return node

    def _caseSegmentation(self, create=True):
        node = self.controller.case.roleNode(R.ROLE_SEGMENTATION)
        if node is None and create:
            node = self.onCreateSegmentation()
        return node

    def onAddSegment(self, role):
        segmentationNode = self._caseSegmentation()
        if segmentationNode is None:
            return
        baseName = W.SEGMENT_ROLES[role][1]
        if role in (W.SEGMENT_PERFUSED, W.SEGMENT_TUMOR, W.SEGMENT_OTHER):
            name = segtools.numberedName(segmentationNode, baseName)
        else:
            name = segtools.uniqueName(segmentationNode, baseName)
        segmentID = segmentationNode.GetSegmentation().AddEmptySegment("", name, W.SEGMENT_ROLES[role][2])
        segtools.ownLayer(segmentationNode, segmentID)   # else the first stroke carves it out of the whole liver
        setSegmentRole(segmentationNode, segmentID, role)
        self.openSegmentEditor(segmentID)

    def onSegmentRoleChanged(self, segmentID, role):
        if self._updating:
            return
        segmentationNode = self.controller.case.roleNode(R.ROLE_SEGMENTATION)
        if segmentationNode is not None:
            setSegmentRole(segmentationNode, segmentID, role or "")

    def onCandidateRoleChanged(self, segmentID, role):
        if self._updating:
            return
        segmentationNode = self.controller.case.roleNode(R.ROLE_SEGMENTATION)
        segment = segmentationNode.GetSegmentation().GetSegment(segmentID) if segmentationNode else None
        if segment is not None:
            segment.SetTag(ai.CANDIDATE_TAG, role or "1")

    def onAcceptCandidate(self, segmentID):
        segmentationNode = self.controller.case.roleNode(R.ROLE_SEGMENTATION)
        segment = segmentationNode.GetSegmentation().GetSegment(segmentID) if segmentationNode else None
        if segment is None:
            return
        role = self._candidateRole(segmentationNode, segmentID)
        existing = [i for i in segmentIDs(segmentationNode)
                    if i != segmentID and role in (W.SEGMENT_LIVER, W.SEGMENT_LUNGS)
                    and segmentRole(segmentationNode, i) == role]
        if existing:
            names = ", ".join(f"'{segmentationNode.GetSegmentation().GetSegment(i).GetName()}'" for i in existing)
            box = qt.QMessageBox(slicer.util.mainWindow())
            box.setWindowTitle("Accept AI result")
            box.setText(f"There is already a {W.SEGMENT_ROLES[role][0].lower()} segment: {names}.\n\n"
                        "Replace it with the AI result, or keep both?")
            replaceButton = box.addButton("Replace", qt.QMessageBox.AcceptRole)
            keepButton = box.addButton("Keep both", qt.QMessageBox.NoRole)
            box.addButton(qt.QMessageBox.Cancel)
            box.exec_()
            clicked = box.clickedButton()
            if clicked is replaceButton:
                for i in existing:
                    segmentationNode.GetSegmentation().RemoveSegment(i)
            elif clicked is not keepButton:
                return
        baseName = CANDIDATE_SUFFIX.sub("", segment.GetName()).strip() or (
            W.SEGMENT_ROLES[role][1] if role in W.SEGMENT_ROLES else "Segment")
        segment.SetName(baseName + "__accepting")  # so the new name can be the base name itself
        name = segtools.uniqueName(segmentationNode, baseName)
        ai.acceptCandidate(segmentationNode, segmentID, name, role)
        if role in (W.SEGMENT_LUNGS, W.SEGMENT_LIVER):
            try:
                with self._grid() as grid:
                    removed = segtools.removeLiverFromLungs(grid)
                if removed:
                    self.toolStatus.text = (f"{removed:.1f} mL shared by the lungs and the whole liver removed from "
                                            "the lungs.")
            except Exception as e:
                logging.warning(f"Taranis: could not separate the lungs from the liver: {e}")
        if role in (W.SEGMENT_TUMOR, W.SEGMENT_VIABLE):
            self.toolStatus.text = (f"'{name}' accepted. Use the Islands effect of the Segment Editor to report "
                                    "every lesion separately, and 'Normal liver = liver − tumours'.")
        self.aiStatus.text = (f"'{name}' accepted. It is now an ordinary segment: edit it freely in the Segment "
                              "Editor below.")
        self._showReferenceBackground()
        try:
            self.segmentEditorWidget.setCurrentSegmentID(segmentID)   # ready to be edited
        except Exception:
            pass

    def onDiscardCandidate(self, segmentID):
        segmentationNode = self.controller.case.roleNode(R.ROLE_SEGMENTATION)
        if segmentationNode is not None:
            segmentationNode.GetSegmentation().RemoveSegment(segmentID)
        self._showReferenceBackground()

    def _showReferenceBackground(self):
        """Back to the case's reference (primary) image in the segmentation layout after an AI result."""
        if self._segLayoutShown:
            V.showBackground(self.controller.case.primaryVolume())

    # -- AI --

    def onDownloadModel(self, key):
        spec = ai.MODELS_BY_KEY[key]
        with self._busy(self.aiStatus, f"Downloading {spec.fileName} …"):
            path = ai.downloadModel(spec)
            self.aiStatus.text = f"Downloaded to {path}."
        self._aiDefaultsKey = None
        self._refresh()

    def onShowModelRoi(self, key):
        spec = ai.MODELS_BY_KEY[key]
        roi = ai.modelRoi(spec, create=True, volumeNode=self.aiRows[key]["input"].currentNode())
        roi.GetDisplayNode().SetVisibility(True)
        V.showBackground(self.aiRows[key]["input"].currentNode())   # place the ROI on the model's input image
        center = [0.0] * 3
        roi.GetCenter(center)
        slicer.modules.markups.logic().JumpSlicesToLocation(center[0], center[1], center[2], True)
        self.aiStatus.text = (f"Adjust the ROI '{roi.GetName()}' so that it covers the whole "
                              f"{spec.segmentName.lower()} in all three views, then Run.")
        self._refresh()

    def onRunAi(self, key):
        case = self.controller.case
        row = self.aiRows[key]
        volume = row["input"].currentNode()
        if volume is None:
            slicer.util.errorDisplay("Select the input image.")
            return
        segmentationNode = self._caseSegmentation()
        if segmentationNode is None:
            return
        forceCPU = self.aiCpuCheck.checked
        views = self._saveViews()
        with self._busy(self.aiStatus, f"Running on '{volume.GetName()}' … (this can take a few minutes)"):
            if key in TOTALSEG_KEYS:
                structure = TOTALSEG_KEYS[key]
                isMRI = self._isMRI(volume)
                labelmap, message = ai.runTotalSegmentator(volume, structure, isMRI, forceCPU)
                name = "Tumors" if structure == W.SEGMENT_TUMOR else W.SEGMENT_ROLES[structure][1]
                role = structure
                if labelmap is None:
                    self._restoreViews(views)
                    self.aiStatus.text = message
                    return
            else:
                spec = ai.MODELS_BY_KEY[key]
                liverID = None
                if spec.maskWithLiver:
                    livers = [i for i in segmentIDs(segmentationNode)
                              if segmentRole(segmentationNode, i) == W.SEGMENT_LIVER]
                    if len(livers) != 1:
                        raise ValueError("The tumour model needs exactly one accepted whole-liver segment.")
                    liverID = livers[0]
                scale, unitsNote = 1.0, ""
                if spec.inputKind == ai.INPUT_PET:
                    scale, unitsNote = ai.suvScale(volume)
                roi = ai.modelRoi(spec) if spec.roiSizeMM else None
                if spec.roiSizeMM and roi is None:
                    self.onShowModelRoi(key)
                    return
                labelmap, message = ai.runModel(spec, volume, segmentationNode, liverID, roi, scale, forceCPU)
                if roi is not None and roi.GetDisplayNode():
                    roi.GetDisplayNode().SetVisibility(False)   # the box would hide the result
                if unitsNote:
                    message += " " + unitsNote
                name, role = spec.segmentName, spec.role
            self._restoreViews(views)   # temporary volumes may have replaced the displayed images
            # the segmentation's geometry must cover the result, or the import crops it (e.g. lungs vs liver MRI)
            segtools.extendSegmentationGeometry(segmentationNode, case.primaryVolume(),
                                                segtools.caseVolumes(case) + [labelmap])
            segmentID = ai.importCandidate(segmentationNode, labelmap, name, role)
            if segmentID is None:
                self.aiStatus.text = message + " No candidate added."
            else:
                self.aiStatus.text = (message + " Review the yellow candidate, then accept or discard it in the "
                                      "segment table; once accepted you can edit it freely in the Segment Editor "
                                      "below.")
                self.openSegmentEditor(segmentID)

    def onAnatomyWindow(self, preset):
        volume = self.controller.case.primaryVolume()
        if volume is None:
            slicer.util.errorDisplay("Assign the images in the Data step first.")
            return
        try:
            V.applyAnatomyWindow(volume, preset)
        except Exception as e:
            slicer.util.errorDisplay(f"Could not set the window: {e}")

    def onFunctionalWindow(self, percent):
        volume = self.controller.case.roleNode(R.ROLE_DOSIMETRY)
        if volume is None:
            slicer.util.errorDisplay("Assign the dosimetry image in the Data step first.")
            return
        try:
            V.applyFunctionalWindow(volume, percent)
        except Exception as e:
            slicer.util.errorDisplay(f"Could not set the window: {e}")

    def _isMRI(self, volume):
        case = self.controller.case
        for role, node in case.roleNodes().items():
            if node is volume and case.roleType(role) in (R.TYPE_CT, R.TYPE_MRI):
                return case.roleType(role) == R.TYPE_MRI
        return R.guessModality(volumeInfo(volume)) == "MR"

    @staticmethod
    def _saveViews():
        return {node.GetID(): (node.GetBackgroundVolumeID(), node.GetForegroundVolumeID(), node.GetForegroundOpacity())
                for node in slicer.util.getNodesByClass("vtkMRMLSliceCompositeNode")}

    @staticmethod
    def _restoreViews(views):
        for nodeID, (background, foreground, opacity) in views.items():
            node = slicer.mrmlScene.GetNodeByID(nodeID)
            if node is None:
                continue
            node.SetBackgroundVolumeID(background if background and slicer.mrmlScene.GetNodeByID(background) else None)
            node.SetForegroundVolumeID(foreground if foreground and slicer.mrmlScene.GetNodeByID(foreground) else None)
            node.SetForegroundOpacity(opacity)

    # -- Tools --

    def _grid(self):
        """Primary-image voxel grid covering the segmentation (tight: every tool works inside existing segments;
        falls back to the grid extended to all case images and segments). Use with 'with'."""
        case = self.controller.case
        return segtools.SegmentGrid(case.roleNode(R.ROLE_SEGMENTATION), case.primaryVolume(),
                                    segtools.caseVolumes(case), tight=True)

    def runSegmentTool(self, function, *args, check=True):
        with self._busy(self.toolStatus, "Working …"), self._grid() as grid:
            result = function(grid, *args)
            message = result[1] if isinstance(result, tuple) else result
            if check:
                issues = self._checkGeometry(grid)
                message += f" Geometry check: {len(issues)} finding(s)." if issues else " Geometry check: no findings."
            self.toolStatus.text = message
            return result

    def onPerfusedFromUptake(self):
        case = self.controller.case
        uptake = segtools.uptakeVolume(case)
        if uptake is None:
            slicer.util.errorDisplay("Assign the dosimetry image in the Data step first.")
            return
        percent = self.perfusedPercentSpin.value
        setSetting(SETTING_PERFUSED_PERCENT, percent)
        result = self.runSegmentTool(segtools.perfusedFromUptake, uptake, percent, check=False)
        if result:
            self.openSegmentEditor(result[0])

    def onCheckGeometry(self):
        with self._busy(self.toolStatus, "Checking …"):
            with self._grid() as grid:
                issues = self._checkGeometry(grid)
            if not issues:
                self.toolStatus.text = "Geometry check: no overlaps, nothing outside the liver, volumes plausible."
            else:
                self.toolStatus.text = "Geometry check:<br>" + "<br>".join(
                    f"• {text}" for severity, text in issues)

    def _checkGeometry(self, grid):
        issues = segtools.geometryCheck(grid, self.controller.case.mode)
        segtools.storeGeometryCheck(self.controller.case, grid.node, issues)
        return issues

    # -- Busy helper --

    @contextlib.contextmanager
    def _busy(self, statusLabel, text):
        statusLabel.text = text
        slicer.app.setOverrideCursor(qt.Qt.WaitCursor)
        slicer.app.processEvents()
        try:
            yield
        except Exception as e:
            logging.exception(f"Taranis: {e}")
            statusLabel.text = f"<span style='color:#dc2626'>{e}</span>"
            slicer.app.restoreOverrideCursor()
            slicer.util.errorDisplay(str(e))
            return
        slicer.app.restoreOverrideCursor()

    # =============================================================================================================
    # 4. Lung shunt fraction
    # =============================================================================================================

    def _buildLsfPage(self, layout):
        self.lsfValueLabel = styledLabel("", "font-size: 14px;")
        self.lsfValueLabel.setTextFormat(qt.Qt.RichText)
        layout.addWidget(self.lsfValueLabel)

        # -- Calculate from the case --
        imageBox = ctk.ctkCollapsibleButton()
        imageBox.text = "Calculate from the dosimetry image"
        imageLayout = qt.QVBoxLayout(imageBox)
        imageLayout.addWidget(smallGray("Counts of the dosimetry image (MAA SPECT) in the lung segment(s) and the "
                                        "whole liver of the case segmentation: LSF = lungs / (lungs + liver). "
                                        "Voxels claimed by both are counted for the liver."))
        self.lsfInputsLabel = styledLabel()
        self.lsfInputsLabel.setTextFormat(qt.Qt.RichText)
        imageLayout.addWidget(self.lsfInputsLabel)
        row = qt.QHBoxLayout()
        self.lsfCalculateButton = qt.QPushButton("Calculate LSF")
        self.lsfCalculateButton.setToolTip("Count the dosimetry image in the lungs and the whole liver.")
        self.lsfCalculateButton.connect("clicked()", self.onCalculateLsf)
        self.lsfClipCheck = qt.QCheckBox("Negative voxel values = 0")
        self.lsfClipCheck.checked = True
        row.addWidget(self.lsfCalculateButton)
        row.addWidget(self.lsfClipCheck)
        row.addStretch(1)
        imageLayout.addLayout(row)
        self.lsfResultsTable = qt.QTableWidget(0, 2)
        self.lsfResultsTable.horizontalHeader().visible = False
        self.lsfResultsTable.verticalHeader().visible = False
        self.lsfResultsTable.horizontalHeader().setStretchLastSection(True)
        self.lsfResultsTable.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)
        self.lsfResultsTable.visible = False
        imageLayout.addWidget(self.lsfResultsTable)
        self.lsfResultNotes = styledLabel()
        self.lsfResultNotes.setTextFormat(qt.Qt.RichText)
        imageLayout.addWidget(self.lsfResultNotes)
        self.lsfUseResultButton = qt.QPushButton("Use this LSF")
        self.lsfUseResultButton.setToolTip("Store the calculated LSF (with the lung mass below) in the case.")
        self.lsfUseResultButton.enabled = False
        self.lsfUseResultButton.connect("clicked()", self.onUseCalculatedLsf)
        imageLayout.addWidget(self.lsfUseResultButton)
        layout.addWidget(imageBox)

        # -- Lung mass and lung dose --
        doseBox = ctk.ctkCollapsibleButton()
        doseBox.text = "Lung mass and lung dose"
        doseLayout = qt.QFormLayout(doseBox)
        massRow = qt.QHBoxLayout()
        self.lungMassSpinBox = qt.QDoubleSpinBox()
        self.lungMassSpinBox.setRange(100.0, 5000.0)
        self.lungMassSpinBox.setDecimals(0)
        self.lungMassSpinBox.setSuffix(" g")
        self.lungMassSpinBox.value = L.DEFAULT_LUNG_MASS_G
        self.lungMassSpinBox.setToolTip("Lung mass used for the lung dose (default 1000 g).")
        self.lungMassSpinBox.connect("editingFinished()", self.onLungMassEdited)
        estimateButton = qt.QPushButton("Estimate from CT")
        estimateButton.setToolTip("Lung mass from the CT numbers inside the lung segment: density = (HU + 1000) / "
                                  "1000 g/mL (CT densitovolumetry, Kao et al., EJNMMI Res 2014). Needs the whole "
                                  "lungs inside the CT.")
        estimateButton.connect("clicked()", self.onEstimateLungMass)
        massRow.addWidget(self.lungMassSpinBox, 1)
        massRow.addWidget(estimateButton)
        doseLayout.addRow("Lung mass:", massRow)
        self.lungMassNote = smallGray()
        doseLayout.addRow(self.lungMassNote)
        citation = smallGray(L.LUNG_MASS_METHOD_HTML)
        citation.setTextFormat(qt.Qt.RichText)
        citation.setOpenExternalLinks(True)
        doseLayout.addRow(citation)
        self.activitySpinBox = qt.QDoubleSpinBox()
        self.activitySpinBox.setRange(0.0, 20.0)
        self.activitySpinBox.setDecimals(2)
        self.activitySpinBox.setSingleStep(0.1)
        self.activitySpinBox.setSuffix(" GBq")
        self.activitySpinBox.setSpecialValueText("not known")
        self.activitySpinBox.setToolTip("Planned (pre-therapy) or administered Y-90 activity, for the lung dose.")
        self.activitySpinBox.connect("editingFinished()", self.onActivityEdited)
        doseLayout.addRow("Y-90 activity:", self.activitySpinBox)
        self.lungDoseLabel = styledLabel()
        self.lungDoseLabel.setTextFormat(qt.Qt.RichText)
        doseLayout.addRow("Lung dose:", self.lungDoseLabel)
        layout.addWidget(doseBox)

        # -- Manual --
        manualBox = ctk.ctkCollapsibleButton()
        manualBox.text = "Enter manually (completes this step)"
        manualBox.collapsed = True
        manualLayout = qt.QFormLayout(manualBox)
        self.lsfSpinBox = qt.QDoubleSpinBox()
        self.lsfSpinBox.setRange(0.0, 100.0)
        self.lsfSpinBox.setDecimals(2)
        self.lsfSpinBox.setSuffix(" %")
        manualLayout.addRow("Lung shunt fraction:", self.lsfSpinBox)
        self.lsfSourceCombo = qt.QComboBox()
        for source in LSF_SOURCES:
            self.lsfSourceCombo.addItem(source)
        manualLayout.addRow("Source:", self.lsfSourceCombo)
        applyButton = qt.QPushButton("Use this LSF")
        applyButton.setToolTip("Store the value with the lung mass above.")
        applyButton.connect("clicked()", self.onApplyManualLsf)
        manualLayout.addRow(applyButton)
        layout.addWidget(manualBox)

        # -- LSF calculator module (advanced) --
        calculatorBox = ctk.ctkCollapsibleButton()
        calculatorBox.text = "LSF calculator module"
        calculatorBox.collapsed = True
        calculatorLayout = qt.QVBoxLayout(calculatorBox)
        calculatorLayout.addWidget(smallGray("The stand-alone LSF calculator (its own layout with a SPECT MIP), "
                                             "filled with the case images and segmentation."))
        row = qt.QHBoxLayout()
        openButton = qt.QPushButton("Open LSF calculator")
        openButton.connect("clicked()", self.onOpenLsfCalculator)
        takeButton = qt.QPushButton("Use the calculator's result")
        takeButton.setToolTip("Take the accepted LSF of the LSF calculator into the case.")
        takeButton.connect("clicked()", self.onTakeLsfFromCalculator)
        row.addWidget(openButton)
        row.addWidget(takeButton)
        calculatorLayout.addLayout(row)
        layout.addWidget(calculatorBox)

        row = qt.QHBoxLayout()
        self.skipLsfButton = qt.QPushButton("Skip this step")
        self.skipLsfButton.connect("clicked()", self.onSkipLsf)
        clearButton = qt.QPushButton("Clear LSF")
        clearButton.connect("clicked()", self.onClearLsf)
        row.addWidget(self.skipLsfButton)
        row.addWidget(clearButton)
        layout.addLayout(row)
        self._lsfResult = None
        self._lsfResultKey = None

    def _refreshLsf(self):
        case = self.controller.case
        snapshot = self.controller.snapshot
        segments = snapshot.segments if snapshot else []
        value = case.lsfValue()
        if case.mode == R.MODE_ABSOLUTE:
            self.lsfValueLabel.text = "Not used in absolute mode (the lung activity is measured in the image)."
        elif value is not None:
            mass = case.lungMassG()
            massText = f", lung mass {mass:.0f} g" if mass else ""
            outdated = " <span style='color:#b45309'>(outdated: calculate again)</span>" if (
                snapshot is not None and snapshot.lsfOutdated) else ""
            self.lsfValueLabel.text = (f"LSF: <b>{value:.2f} %</b> ({case.param('LSF.Source') or 'unknown source'}"
                                       f"{massText}){outdated}")
        elif case.flag(P_LSF_SKIPPED):
            self.lsfValueLabel.text = "Skipped."
        else:
            self.lsfValueLabel.text = "No LSF yet."
        self.skipLsfButton.enabled = value is None and not case.flag(P_LSF_SKIPPED)

        image, segmentationNode, lungs, liver, message = L.caseInputs(case, segments)
        byID = {s.segmentID: s for s in segments}
        lines = []
        if image is not None:
            imageType = R.TYPE_LABELS.get(case.roleType(R.ROLE_DOSIMETRY), "type not set")
            lines.append(f"Dosimetry image: <b>{image.GetName()}</b> ({imageType})")
        if lungs:
            volume = sum(byID[i].volumeML or 0.0 for i in lungs)
            lines.append("Lungs: " + ", ".join(f"'{byID[i].name}'" for i in lungs) + f" ({volume:.0f} mL)")
        if liver:
            lines.append(f"Whole liver: '{byID[liver].name}' ({(byID[liver].volumeML or 0.0):.0f} mL)")
        if message:
            lines.append(f"<span style='color:#b45309'>{message}</span>")
        self.lsfInputsLabel.text = "<br>".join(lines)
        self.lsfCalculateButton.enabled = not message and case.mode != R.MODE_ABSOLUTE
        # a calculation shown but not used becomes stale when the inputs change
        if self._lsfResult is not None and self._lsfResultKey != L.inputsKey(segments, image.GetID() if image else ""):
            self._showLsfResult(None)
            self.lsfResultNotes.text = "The inputs changed: calculate again."

        if not self.lungMassSpinBox.hasFocus():
            self.lungMassSpinBox.value = case.lungMassG() or self.lungMassSpinBox.value
        if not self.activitySpinBox.hasFocus():
            self.activitySpinBox.value = case.plannedActivityGBq() or 0.0
        self._refreshLungDose()

    def _refreshLungDose(self):
        case = self.controller.case
        lsfValue = case.lsfValue()
        if lsfValue is None and self._lsfResult is not None:
            lsfValue = self._lsfResult["lsfPercent"]
        mass = self.lungMassSpinBox.value
        if lsfValue is None:
            self.lungDoseLabel.text = "needs the LSF"
            return
        perGBq = L.lungDosePerGBq(lsfValue, mass)
        activity = self.activitySpinBox.value
        if activity <= 0:
            self.lungDoseLabel.text = f"{perGBq:.1f} Gy per GBq (LSF {lsfValue:.2f} %, {mass:.0f} g)"
            return
        dose = L.lungDoseGy(activity, lsfValue, mass)
        color = ("#dc2626" if dose > L.LUNG_DOSE_CUMULATIVE_LIMIT_GY else
                 "#d97706" if dose > L.LUNG_DOSE_SESSION_LIMIT_GY else "#16a34a")
        self.lungDoseLabel.text = (f"<b style='color:{color}'>{dose:.1f} Gy</b> ({perGBq:.1f} Gy/GBq; limits commonly "
                                   f"{L.LUNG_DOSE_SESSION_LIMIT_GY:g} Gy per session, "
                                   f"{L.LUNG_DOSE_CUMULATIVE_LIMIT_GY:g} Gy cumulative)")

    def _showLsfResult(self, result):
        self._lsfResult = result
        self.lsfUseResultButton.enabled = result is not None
        self.lsfResultsTable.visible = result is not None
        if result is None:
            self.lsfResultsTable.setRowCount(0)
            self.lsfResultNotes.text = ""
            self._refreshLungDose()
            return
        rows = [("Lung shunt fraction", f"{result['lsfPercent']:.2f} %"),
                ("Lung counts", f"{result['lungCounts']:.6g}"),
                ("Liver counts", f"{result['liverCounts']:.6g}"),
                ("Lungs on the image grid", f"{result['lungML']:.0f} mL"),
                ("Liver on the image grid", f"{result['liverML']:.0f} mL")]
        if result.get("lungCoverage") is not None:
            rows.append(("Lung segment inside the image", f"{100 * result['lungCoverage']:.0f} %"))
        self.lsfResultsTable.setRowCount(len(rows))
        for index, (label, text) in enumerate(rows):
            self.lsfResultsTable.setItem(index, 0, qt.QTableWidgetItem(label))
            self.lsfResultsTable.setItem(index, 1, qt.QTableWidgetItem(text))
        self.lsfResultsTable.resizeColumnToContents(0)
        self.lsfResultsTable.setFixedHeight(self.lsfResultsTable.verticalHeader().length() + 6)
        notes = [f"<span style='color:{'#b45309' if severity == W.SEVERITY_WARNING else '#6b7280'}'>{text}</span>"
                 for severity, text in result.get("issues", [])]
        if result.get("sharedVoxels"):
            notes.append(f"{result['sharedVoxels']} boundary voxels claimed by both segments were counted for the "
                         "liver.")
        notes.append("Check the lung and liver outlines on the fused views, then press 'Use this LSF'.")
        self.lsfResultNotes.text = "<br>".join(notes)
        self._refreshLungDose()

    def onCalculateLsf(self):
        case = self.controller.case
        snapshot = self.controller.snapshot
        segments = snapshot.segments if snapshot else []
        with self._busy(self.lsfResultNotes, "Counting …"):
            result = L.calculateFromCase(case, segments, self.lsfClipCheck.checked)
            issues = L.lsfIssues(result, result.get("lungSegmentML"))
            if case.roleType(R.ROLE_DOSIMETRY) != R.TYPE_MAA_SPECT:
                issues.append((W.SEVERITY_INFO, "The LSF is usually calculated on the Tc-99m MAA SPECT; this dosimetry "
                                                "image is " + R.TYPE_LABELS.get(case.roleType(R.ROLE_DOSIMETRY),
                                                                                "of unknown type") + "."))
            result["issues"] = issues
            image = case.roleNode(R.ROLE_DOSIMETRY)
            self._lsfResultKey = L.inputsKey(segments, image.GetID())
            self._showLsfResult(result)

    def onUseCalculatedLsf(self):
        result = self._lsfResult
        if result is None:
            return
        issues = list(result.get("issues", []))
        if self.lungMassNote.text:
            issues.append((W.SEVERITY_INFO, self.lungMassNote.text))
        self.controller.case.setLsf(result["lsfPercent"], L.SOURCE_IMAGE, self.lungMassSpinBox.value,
                                    L.detailsJson(result, self._lsfResultKey), json.dumps(issues))
        self._showLsfResult(None)
        self.lsfResultNotes.text = f"LSF {result['lsfPercent']:.2f} % stored in the case."

    def onEstimateLungMass(self):
        snapshot = self.controller.snapshot
        with self._busy(self.lungMassNote, "Estimating …"):
            mass, ctName, note = L.lungMassFromCase(self.controller.case, snapshot.segments if snapshot else [])
            self.lungMassSpinBox.value = round(mass)
            self.lungMassNote.text = note or f"From the CT numbers of '{ctName}' inside the lung segment."
            self.onLungMassEdited()

    def onLungMassEdited(self):
        case = self.controller.case
        if case is not None and case.lsfValue() is not None:
            case.setParam(P_LSF_LUNG_MASS, f"{self.lungMassSpinBox.value:.1f}")
        self._refreshLungDose()

    def onActivityEdited(self):
        case = self.controller.case
        if case is not None:
            value = self.activitySpinBox.value
            case.setParam(P_PLANNED_ACTIVITY, f"{value:.3f}" if value > 0 else "")
        self._refreshLungDose()

    def onOpenLsfCalculator(self):
        case = self.controller.case
        widget = moduleWidget(LSF_MODULE)
        if widget is None or case is None:
            return
        trySet("the SPECT", widget.spectSelector.setCurrentNode, case.roleNode(R.ROLE_DOSIMETRY))
        ct = case.roleNode(R.ROLE_DOSIMETRY_ANATOMY) or case.roleNode(R.ROLE_REFERENCE)
        trySet("the CT", widget.ctSelector.setCurrentNode, ct)
        segmentationNode = case.roleNode(R.ROLE_SEGMENTATION)
        if segmentationNode is not None:
            trySet("the segmentation", widget.segmentationSelector.setCurrentNode, segmentationNode)

    def onTakeLsfFromCalculator(self):
        if not moduleAvailable(LSF_MODULE):
            slicer.util.errorDisplay("The LSF calculator is not loaded.")
            return
        widget = slicer.util.getModuleWidget(LSF_MODULE)
        result = getattr(widget, "_result", None)
        if not result or not getattr(widget, "_accepted", False):
            slicer.util.warningDisplay("The LSF calculator has no accepted result yet. Calculate and accept it there.")
            return
        self.controller.case.setLsf(result["lsfPercent"], "LSF calculator", self.lungMassSpinBox.value)

    def onApplyManualLsf(self):
        self.controller.case.setLsf(self.lsfSpinBox.value, self.lsfSourceCombo.currentText, self.lungMassSpinBox.value)

    def onSkipLsf(self):
        case = self.controller.case
        case.setLsf(None, "")
        case.setFlag(P_LSF_SKIPPED, True)

    def onClearLsf(self):
        self.controller.case.setLsf(None, "")

    # =============================================================================================================
    # 5. Dosimetry
    # =============================================================================================================

    def _buildDosimetryPage(self, layout):
        self.dosimetryModeLabel = styledLabel()
        self.dosimetryModeLabel.setTextFormat(qt.Qt.RichText)
        layout.addWidget(self.dosimetryModeLabel)
        self.dosimetryButtons = {}
        for mode in (R.MODE_RELATIVE, R.MODE_ABSOLUTE):
            button = qt.QPushButton(f"Open {R.MODE_LABELS[mode].lower()} dosimetry")
            button.setToolTip("Opens the module with the images, segmentation, liver, perfused volumes, LSF and "
                              "isodose set of the case filled in.")
            button.connect("clicked()", lambda mode=mode: self.onOpenDosimetry(mode))
            layout.addWidget(button)
            self.dosimetryButtons[mode] = button
        layout.addWidget(smallGray("The inputs are filled in each time you open the module from here; you can "
                                   "still change them in the module. The step becomes 'Outdated' when images, "
                                   "registration or segments change after the calculation."))

    def _refreshDosimetry(self):
        case = self.controller.case
        mode = case.mode
        imageType = case.roleType(R.ROLE_DOSIMETRY)
        if mode:
            self.dosimetryModeLabel.text = (f"Mode: <b>{R.MODE_LABELS[mode]}</b>"
                                            + (f" &middot; {R.TYPE_LABELS[imageType]}" if imageType else ""))
        else:
            self.dosimetryModeLabel.text = "Choose the processing mode in the Data step."
        for buttonMode, button in self.dosimetryButtons.items():
            button.enabled = (not imageType or R.modeAllowed(imageType, buttonMode))
            font = button.font
            font.setBold(buttonMode == mode)
            button.setFont(font)

    def onOpenDosimetry(self, mode):
        case = self.controller.case
        moduleName = R.MODE_MODULES[mode]
        widget = moduleWidget(moduleName)
        if widget is None:
            return
        if case.mode != mode:
            case.setMode(mode)
        segmentationNode = case.roleNode(R.ROLE_SEGMENTATION)
        trySet("the dosimetry image", widget.spectSelector.setCurrentNode, case.roleNode(R.ROLE_DOSIMETRY))
        trySet("the reference image", widget.referenceSelector.setCurrentNode, case.primaryVolume())
        if segmentationNode is None:
            return
        persistSegmentRoles(segmentationNode)
        trySet("the segmentation", widget.segmentationSelector.setCurrentNode, segmentationNode)
        if hasattr(widget, "onSegmentationNodeChanged"):
            # reloads the segment categories even if the segmentation was already selected
            trySet("the segment categories", widget.onSegmentationNodeChanged, segmentationNode)
        roleOf = {segmentID: segmentRole(segmentationNode, segmentID) for segmentID in segmentIDs(segmentationNode)}
        liver = [s for s, role in roleOf.items() if role == W.SEGMENT_LIVER]
        if liver:
            trySet("the whole liver", widget.liverSegmentSelector.setCurrentSegmentID, liver[0])
        if mode == R.MODE_RELATIVE:
            perfused = [s for s, role in roleOf.items() if role == W.SEGMENT_PERFUSED]
            if perfused and hasattr(widget, "perfusedRows"):
                while len(widget.perfusedRows) < len(perfused):
                    widget.addPerfusedVolumeRow()
                while len(widget.perfusedRows) > len(perfused) and hasattr(widget, "removePerfusedVolumeRow"):
                    widget.removePerfusedVolumeRow(widget.perfusedRows[-1])
                for row, segmentID in zip(widget.perfusedRows, perfused):
                    trySet("a perfused volume", row["selector"].setCurrentSegmentID, segmentID)
            lsf = case.lsfValue()
            if lsf is not None:
                trySet("the LSF", setattr, widget.lungShuntSlider, "value", lsf)
            mass = case.lungMassG()
            if mass:
                trySet("the lung mass", setattr, widget.lungMassSpinBox, "value", mass)
        preset = R.MICROSPHERE_LABELS.get(case.microspheres)
        if preset and hasattr(widget, "isodosePresetComboBox"):
            index = widget.isodosePresetComboBox.findText(preset)
            if index >= 0:
                widget.isodosePresetComboBox.setCurrentIndex(index)
        self._showDosimetryNotice(widget, mode, roleOf)

    def _showDosimetryNotice(self, widget, mode, roleOf):
        """Fill in what the case knows beyond the inputs (hours after treatment, activity) and list everything that
        was filled in, in a box at the top of the dosimetry module: the user reviews every value."""
        if not hasattr(widget, "showCaseNotice"):
            return
        case = self.controller.case
        filled = ["dosimetry and reference images", "segmentation and whole liver", "segment categories"]
        if preset := R.MICROSPHERE_LABELS.get(case.microspheres):
            filled.append(f"isodose set ({preset})")
        check, fields = [], []
        if mode == R.MODE_RELATIVE:
            perfused = [s for s, role in roleOf.items() if role == W.SEGMENT_PERFUSED]
            if perfused:
                filled.append(f"{len(perfused)} perfused volume{'s' if len(perfused) > 1 else ''}")
            if case.lsfValue() is not None:
                filled.append(f"LSF {case.lsfValue():.1f} %")
            if case.lungMassG():
                filled.append(f"lung mass {case.lungMassG():.0f} g")
            activity = case.plannedActivityGBq()
            rows = getattr(widget, "perfusedRows", [])
            if activity and len(perfused) == 1 and len(rows) == 1:
                trySet("the activity", setattr, rows[0]["activitySlider"], "value", round(activity * 1000.0))
                check.append(f"<b>Activity {activity * 1000.0:.0f} MBq</b>: the activity entered in the LSF step "
                             "(planned). Use the activity actually prescribed or administered to this perfused "
                             "volume.")
                fields.append((rows[0]["activitySlider"], rows[0].get("activityLabel")))
            elif activity and len(perfused) > 1:
                check.append(f"Activity: {activity * 1000.0:.0f} MBq was entered in the LSF step, but there are "
                             f"{len(perfused)} perfused volumes: enter the activity of each one.")
        elif mode == R.MODE_ABSOLUTE and hasattr(widget, "hourSlider"):
            text = self._hoursAfterTreatmentText(widget)
            check.append(text)
            fields.append((widget.hourSlider, None))
        html = ["<b>Filled in from the Taranis case</b> – review every input before calculating:<br>"
                + ", ".join(filled) + "."]
        if check:
            html.append("<ul style='margin-top:4px; margin-bottom:0px'>"
                        + "".join(f"<li>{item}</li>" for item in check) + "</ul>")
        try:
            widget.showCaseNotice("".join(html), fields)
        except Exception as e:
            logging.warning(f"Taranis: could not show the review box in the dosimetry module: {e}")

    def _hoursAfterTreatmentText(self, widget):
        """Fill the absolute module's hours after treatment from the administration time and the image header;
        returns the explanation for the review box."""
        case = self.controller.case
        caveat = ("Check it: the scanner clock may differ from the time of the administration record, the series may "
                  "have been reconstructed later, and the decay-correction reference depends on the scanner and the "
                  "reconstruction.")
        text = case.param(P_TREATMENT_DATETIME)
        if not text:
            return ("<b>Hours after treatment</b> not filled in: the administration time is not set (Data step). "
                    "Enter the time from the administration to the time the image is decay-corrected to.")
        try:
            administration = datetime.datetime.fromisoformat(text.split("+")[0].rstrip("Z"))
        except ValueError:
            return f"<b>Hours after treatment</b> not filled in: administration time '{text}' not understood."
        image = case.roleNode(R.ROLE_DOSIMETRY)
        dataset = _dicomHeader(image) if image is not None else None
        if dataset is None:
            return ("<b>Hours after treatment</b> not filled in: the dosimetry image has no DICOM header (not loaded "
                    "from the DICOM database), so its acquisition time is unknown.")
        result = timing.hoursAfterTreatment(administration, timing.headerTimes(dataset))
        notes = "".join(f"<br>⚠ {note}" for note in result["notes"])
        if result["hours"] is None:
            return f"<b>Hours after treatment</b> not filled in.{notes}"
        hours = round(result["hours"], 2)
        trySet("the hours after treatment", setattr, widget.hourSlider, "value", hours)
        return (f"<b>Hours after treatment {hours:.2f} h</b>: administration {administration:%Y-%m-%d %H:%M} → "
                f"{result['reference']:%Y-%m-%d %H:%M}, {result['basis']}. {caveat}{notes}")

    # =============================================================================================================
    # 6. Report
    # =============================================================================================================

    def _buildReportPage(self, layout):
        self.summaryBrowser = qt.QTextBrowser()
        self.summaryBrowser.setMinimumHeight(320)
        layout.addWidget(self.summaryBrowser)
        row = qt.QHBoxLayout()
        reportButton = qt.QPushButton("Save dosimetry report (RTF)...")
        reportButton.setToolTip("Saves the report of the last calculation from the dosimetry module.")
        reportButton.connect("clicked()", self.onSaveReport)
        pdfReportButton = qt.QPushButton("Save dosimetry report (PDF)...")
        pdfReportButton.setToolTip("Saves the report of the last calculation from the dosimetry module as a PDF.")
        pdfReportButton.connect("clicked()", lambda: self.onSaveReport("pdf"))
        sceneButton = qt.QPushButton("Save scene...")
        sceneButton.connect("clicked()", lambda: slicer.app.ioManager().openSaveDataDialog())
        copyButton = qt.QPushButton("Copy summary")
        copyButton.connect("clicked()", lambda: qt.QApplication.clipboard().setText(self.summaryBrowser.toPlainText()))
        row.addWidget(reportButton)
        if hasattr(qt, "QPdfWriter") or hasattr(qt, "QPrinter"):   # Qt only, no extra library
            row.addWidget(pdfReportButton)
        row.addWidget(sceneButton)
        row.addWidget(copyButton)
        layout.addLayout(row)
        layout.addWidget(smallGray("A consolidated case report (registration, segment provenance, LSF, dosimetry and "
                                   "all warnings) is planned; for now the dosimetry module's report is saved."))

    def _refreshReport(self):
        if self._step != W.STEP_REPORT:
            return  # built only when visible
        self.summaryBrowser.setHtml(self.summaryHtml())

    def summaryHtml(self):
        controller = self.controller
        case, snapshot = controller.case, controller.snapshot
        if case is None or snapshot is None:
            return ""
        html = [f"<h3>{case.name}</h3><p>ID: {case.caseID or '&ndash;'}<br>"
                f"Created: {case.param('Created').replace('T', ' ')}<br>"
                f"Summary made: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}</p>"]
        html.append("<h4>Images</h4><ul>")
        for role in R.VOLUME_ROLES:
            info = snapshot.roles.get(role)
            if info is not None:
                imageType = R.TYPE_LABELS.get(snapshot.roleTypes.get(role), "type not set")
                html.append(f"<li><b>{R.ROLE_INFO[role][0]}</b>: {info.name} ({imageType})</li>")
        html.append("</ul>")
        if snapshot.mode:
            html.append(f"<p>Mode: {R.MODE_LABELS[snapshot.mode]} &middot; "
                        f"{R.MICROSPHERE_LABELS.get(case.microspheres, '')}</p>")
        html.append("<h4>Steps</h4><table cellspacing='4'>")
        for number, (key, label) in enumerate(W.STEPS, start=1):
            status = controller.status(key)
            if status is None:
                continue
            color = W.STATE_STYLE[status.state][1]
            html.append(f"<tr><td>{number}. {label}</td><td style='color:{color}'><b>"
                        f"{W.STATE_STYLE[status.state][0]}</b></td><td>{status.summary}</td></tr>")
        html.append("</table>")
        if snapshot.segments:
            html.append("<h4>Segments</h4><ul>")
            for segment in snapshot.segments:
                role = W.SEGMENT_ROLES[segment.role][0] if segment.role else "no role"
                html.append(f"<li>{segment.name} &ndash; {role}</li>")
            html.append("</ul>")
        issues = W.allIssues(controller.statuses)
        if issues:
            html.append("<h4>Errors and warnings</h4><ul>")
            for issue in issues:
                html.append(f"<li><b>{W.STEP_LABELS[issue.step]}</b> ({issue.severity}): {issue.text}</li>")
            html.append("</ul>")
        return "".join(html)

    def onSaveReport(self, fileFormat="rtf"):
        snapshot = self.controller.snapshot
        moduleName = snapshot.dosimetryResultsModule if snapshot else ""
        if not moduleName:
            slicer.util.warningDisplay("There is no dosimetry calculation yet.")
            return
        widget = moduleWidget(moduleName)
        if widget is not None:
            widget.onSaveReportClicked(fileFormat)


# ---------------------------------------------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------------------------------------------

class TaranisTest(ScriptedLoadableModuleTest):

    def setUp(self):
        slicer.mrmlScene.Clear(0)

    def runTest(self):
        self.setUp()
        self.test_pureRules()
        self.setUp()
        self.test_caseAndStatuses()
        self.setUp()
        self.test_segmentRoles()
        self.setUp()
        self.test_segmentTools()
        self.setUp()
        self.test_segmentLayers()
        self.setUp()
        self.test_segmentationLayout()
        self.setUp()
        self.test_lsfFromCase()
        self.setUp()
        self.test_toolbarVisibility()
        self.delayDisplay("Taranis tests passed")

    def test_segmentLayers(self):
        """A new segment shares the whole liver's labelmap layer; writing it then took voxels from the liver."""
        import numpy as np
        ct = self._volume("CT", 0.0)
        segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(ct)
        segmentation = segmentationNode.GetSegmentation()
        liverID = segmentation.AddEmptySegment("", "Whole liver")
        liver = np.zeros((20, 20, 20), np.uint8)
        liver[5:15, 5:15, 5:15] = 1
        slicer.util.updateSegmentBinaryLabelmapFromArray(liver, segmentationNode, liverID, ct)
        tumorID = segmentation.AddEmptySegment("", "Tumor 1")
        if segmentation.GetLayerIndex(tumorID) == segmentation.GetLayerIndex(liverID):   # Slicer's behaviour
            self.assertTrue(segtools.ownLayer(segmentationNode, tumorID))
        self.assertNotEqual(segmentation.GetLayerIndex(tumorID), segmentation.GetLayerIndex(liverID))
        tumor = np.zeros((20, 20, 20), np.uint8)
        tumor[8:11, 8:11, 8:11] = 1
        slicer.util.updateSegmentBinaryLabelmapFromArray(tumor, segmentationNode, tumorID, ct)
        liverAfter = slicer.util.arrayFromSegmentBinaryLabelmap(segmentationNode, liverID, ct)
        self.assertEqual(int(liverAfter.sum()), int(liver.sum()))   # the liver keeps the tumour voxels
        # shared layers of an existing scene are separated without changing any voxel
        otherID = segmentation.AddEmptySegment("", "Normal")
        segtools.separateSharedLayers(segmentationNode)
        layers = [segmentation.GetLayerIndex(i) for i in (liverID, tumorID, otherID)]
        self.assertEqual(len(set(layers)), 3)
        self.assertEqual(int(slicer.util.arrayFromSegmentBinaryLabelmap(segmentationNode, liverID, ct).sum()),
                         int(liver.sum()))
        self.delayDisplay("Every segment on its own layer; a new segment no longer carves the liver")

    def _volume(self, name, value, origin=(0.0, 0.0, 0.0)):
        import numpy as np
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", name)
        array = np.full((20, 20, 20), value, dtype=np.float32)
        slicer.util.updateVolumeFromArray(node, array)
        node.SetOrigin(*origin)
        return node

    def test_pureRules(self):
        """The unit tests of the rules (roles, workflow, visibility) also run inside Slicer."""
        import unittest
        testsPath = os.path.join(os.path.dirname(__file__), "Testing", "Python")
        if not os.path.isdir(testsPath):
            return  # installed extension: the unit test files are not installed
        suite = unittest.defaultTestLoader.discover(testsPath, pattern="test_*.py", top_level_dir=testsPath)
        result = unittest.TextTestRunner(verbosity=1).run(suite)
        self.assertTrue(result.wasSuccessful())

    def test_caseAndStatuses(self):
        controller = WorkflowController.instance()
        controller.update()
        self.assertFalse(controller.isActive)
        spect = self._volume("Patient SPECT MAA", 5.0)
        ct = self._volume("Patient CT", -1000.0)
        controller.startCase("Test case", "T-1")
        self.assertTrue(controller.isActive)
        case = controller.case
        self.assertEqual(controller.status(W.STEP_DATA).state, W.STATE_NOT_STARTED)

        suggestion = R.suggestAssignments([volumeInfo(spect), volumeInfo(ct)])
        self.assertEqual(suggestion.assignments[R.ROLE_DOSIMETRY], spect.GetID())
        self.assertEqual(suggestion.assignments[R.ROLE_DOSIMETRY_ANATOMY], ct.GetID())
        case.setRoleNode(R.ROLE_DOSIMETRY, spect)
        case.setRoleType(R.ROLE_DOSIMETRY, R.TYPE_MAA_SPECT)
        case.setRoleNode(R.ROLE_DOSIMETRY_ANATOMY, ct)
        case.setRoleType(R.ROLE_DOSIMETRY_ANATOMY, R.TYPE_CT)
        case.setMode(R.MODE_RELATIVE)
        controller.update()
        self.assertIn(controller.status(W.STEP_DATA).state, (W.STATE_DONE, W.STATE_WARNING))
        self.assertEqual(controller.status(W.STEP_REGISTRATION).state, W.STATE_NOT_APPLICABLE)
        self.assertEqual(controller.status(W.STEP_SEGMENTATION).state, W.STATE_NOT_STARTED)

        # A reference image makes registration necessary
        mri = self._volume("Liver MRI", 100.0, origin=(5.0, 0.0, 0.0))
        case.setRoleNode(R.ROLE_REFERENCE, mri)
        case.setRoleType(R.ROLE_REFERENCE, R.TYPE_MRI)
        controller.update()
        self.assertEqual(controller.status(W.STEP_REGISTRATION).state, W.STATE_NOT_STARTED)
        case.setFlag(P_REGISTRATION_SKIPPED, True)
        controller.update()
        registration = controller.status(W.STEP_REGISTRATION)
        self.assertEqual(registration.state, W.STATE_SKIPPED)
        self.assertEqual(registration.count(W.SEVERITY_WARNING), 1)

        # Manual LSF completes the step
        case.setLsf(7.5, "Planar scintigraphy", 1000)
        controller.update()
        self.assertEqual(controller.status(W.STEP_LSF).state, W.STATE_DONE)
        case.setMode(R.MODE_ABSOLUTE)  # not allowed for MAA
        controller.update()
        self.assertEqual(controller.status(W.STEP_DATA).state, W.STATE_ERROR)
        self.assertEqual(controller.status(W.STEP_LSF).state, W.STATE_NOT_APPLICABLE)

        # Closing the scene ends the case
        slicer.mrmlScene.Clear(0)
        controller.update()
        self.assertFalse(controller.isActive)

    def test_segmentRoles(self):
        controller = WorkflowController.instance()
        ct = self._volume("CT", -1000.0)
        case = controller.startCase("Segments", "S-1")
        case.setRoleNode(R.ROLE_DOSIMETRY_ANATOMY, ct)
        segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(ct)
        segmentation = segmentationNode.GetSegmentation()
        liverID = segmentation.AddEmptySegment("", "Liver")
        tumorID = segmentation.AddEmptySegment("", "Segment_2")
        case.setRoleNode(R.ROLE_SEGMENTATION, segmentationNode)
        self.assertEqual(segmentRole(segmentationNode, liverID), W.SEGMENT_LIVER)
        self.assertEqual(segmentRole(segmentationNode, tumorID), "")
        setSegmentRole(segmentationNode, tumorID, W.SEGMENT_TUMOR)
        self.assertEqual(segmentRole(segmentationNode, tumorID), W.SEGMENT_TUMOR)
        import json
        self.assertEqual(json.loads(segmentationNode.GetAttribute("Taranis.SegmentCategories")), {tumorID: "tumor"})
        controller.update()
        status = controller.status(W.STEP_SEGMENTATION)
        self.assertTrue(any("Empty" in issue.text for issue in status.issues))

    def test_segmentTools(self):
        """Normal liver, clip, split, perfused volume from uptake, candidates and the geometry check."""
        import numpy as np
        from TaranisLib import segmentops as S
        ct = self._volume("CT", 0.0)
        spect = self._volume("SPECT MAA", 0.0)
        for node in (ct, spect):
            node.SetSpacing(5.0, 5.0, 5.0)   # 0.125 mL voxels: the small test parts exceed the reporting limit
        k, j, i = np.mgrid[0:20, 0:20, 0:20]
        liver = (k - 10) ** 2 + (j - 10) ** 2 + (i - 10) ** 2 <= 49
        uptake = np.where(liver & (i < 10), 100.0, 2.0).astype(np.float32)
        slicer.util.updateVolumeFromArray(spect, uptake)
        segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(ct)
        grid = segtools.SegmentGrid(segmentationNode, ct)
        liverID = grid.add("Whole liver", W.SEGMENT_LIVER, liver)
        tumours = np.zeros_like(liver)
        tumours[8:11, 8:11, 8:11] = True
        tumours[13:15, 13:15, 15:19] = True           # second lesion, partly outside the liver
        tumourID = grid.add("Tumors (AI)", W.SEGMENT_TUMOR, tumours)
        self.assertEqual(segmentRole(segmentationNode, liverID), W.SEGMENT_LIVER)

        issues = segtools.geometryCheck(grid)
        self.assertTrue(any("outside the whole liver" in text for _, text in issues))
        self.assertIn("−", segtools.clipToLiver(grid))
        self.assertFalse((grid.mask(tumourID) & ~liver).any())
        segtools.makeNormalLiver(grid)
        normals = grid.ids(W.SEGMENT_NORMAL)
        self.assertEqual(len(normals), 1)
        self.assertFalse((grid.mask(normals[0]) & grid.mask(tumourID)).any())
        segtools.splitSegment(grid, tumourID, minML=0.0)
        self.assertEqual(len(grid.ids(W.SEGMENT_TUMOR)), 2)
        self.assertTrue(all(grid.name(x).startswith("Tumor ") for x in grid.ids(W.SEGMENT_TUMOR)))

        candidateID, _ = segtools.perfusedFromUptake(grid, spect, 20.0, minML=0.0)
        segment = segmentationNode.GetSegmentation().GetSegment(candidateID)
        self.assertEqual(segmentRole(segmentationNode, candidateID), "")   # candidates have no role yet
        self.assertEqual(ai.candidateRole(segment), W.SEGMENT_PERFUSED)
        ai.acceptCandidate(segmentationNode, candidateID, "Perfused volume 1", W.SEGMENT_PERFUSED)
        self.assertEqual(segmentRole(segmentationNode, candidateID), W.SEGMENT_PERFUSED)
        perfused = grid.mask(candidateID)
        self.assertGreater(S.volumeML(perfused, grid.voxelML), 0)
        self.assertFalse((perfused & ~liver).any())
        self.assertFalse(perfused[10, 10, 15])       # low-uptake half of the liver excluded

        # the perfused candidate is the high-uptake half (i < 10): the first lesion (i 8-10) is partly outside it,
        # the second (i 15-18) not at all
        issues = segtools.geometryCheck(grid, R.MODE_RELATIVE)
        texts = [text for _, text in issues]
        self.assertTrue(any("does not intersect any perfused volume (0 Gy" in t for t in texts), texts)
        self.assertTrue(any("is outside the perfused volume(s)" in t for t in texts), texts)
        self.assertFalse(any("unusual" in t for t in texts))   # liver size: checked by the workflow validator

        # perfused normal liver: perfused volume minus the tumours, tagged; the whole normal liver stays separate
        segtools.makePerfusedNormal(grid)
        normals = grid.ids(W.SEGMENT_NORMAL)
        self.assertEqual(len(normals), 2)
        perfusedNormal = [n for n in normals if grid.tag(n, W.NORMAL_SCOPE_TAG) == W.NORMAL_SCOPE_PERFUSED]
        self.assertEqual(len(perfusedNormal), 1)
        self.assertEqual(grid.tag(perfusedNormal[0], W.NORMAL_SOURCE_TAG), candidateID)
        normalMask = grid.mask(perfusedNormal[0])
        self.assertFalse((normalMask & ~perfused).any())
        self.assertFalse(any((normalMask & grid.mask(t)).any() for t in grid.ids(W.SEGMENT_TUMOR)))
        segtools.makePerfusedNormal(grid)                      # updated, not duplicated
        self.assertEqual(len(grid.ids(W.SEGMENT_NORMAL)), 2)
        for segmentID in grid.ids(W.SEGMENT_NORMAL):
            grid.remove(segmentID)                           # keep the rest of the test as before
        grid.close()

        # tight grid (hub tools): only the bounds of the segmentation, same masks
        with segtools.SegmentGrid(segmentationNode, ct, tight=True) as tightGrid:
            self.assertLessEqual(max(tightGrid.shape), 20 + 2 * segtools.TIGHT_MARGIN_VOXELS)
            self.assertEqual(int(tightGrid.mask(liverID).sum()), int(liver.sum()))
            self.assertEqual(int(tightGrid.mask(candidateID).sum()), int(perfused.sum()))
            segtools.makeNormalLiver(tightGrid)
            normalID = tightGrid.ids(W.SEGMENT_NORMAL)[0]
        with segtools.SegmentGrid(segmentationNode, ct) as fullGrid:   # written on the tight grid, read on the full
            self.assertEqual(int(fullGrid.mask(normalID).sum()),
                             int((liver & ~fullGrid.union(fullGrid.ids(W.SEGMENT_TUMOR))).sum()))
            self.assertEqual(int(fullGrid.mask(liverID).sum()), int(liver.sum()))   # other segments untouched
            fullGrid.remove(normalID)

        # Segments beyond the primary image (lungs of a SPECT/CT above a liver MRI) are not cropped
        chestCT = self._volume("Chest CT", 0.0, origin=(0.0, 0.0, 60.0))
        chestCT.SetSpacing(5.0, 5.0, 5.0)
        with segtools.SegmentGrid(segmentationNode, ct, [chestCT]) as extended:
            self.assertGreater(extended.shape[0], 20)
            lungs = np.zeros(extended.shape, bool)
            lungs[15:30, 5:15, 5:15] = True   # from the liver (overlap) up into the chest CT
            lungsID = extended.add("Lungs", W.SEGMENT_LUNGS, lungs)
            before = int(extended.mask(lungsID).sum())
            removed = segtools.removeLiverFromLungs(extended)
            self.assertGreater(removed, 0)
            after = int(extended.mask(lungsID).sum())
            self.assertEqual(before - after, round(removed / extended.voxelML))
        with segtools.SegmentGrid(segmentationNode, ct, [chestCT]) as reread:
            self.assertEqual(int(reread.mask(lungsID).sum()), after)   # nothing lost outside the liver image
        # the hub's tight grid covers the lungs beyond the primary image (accepting AI lungs removes the liver from
        # them on this grid): writing does not crop them
        with segtools.SegmentGrid(segmentationNode, ct, tight=True) as tightGrid:
            self.assertEqual(int(tightGrid.mask(lungsID).sum()), after)
            tightGrid.write(lungsID, tightGrid.mask(lungsID))
        with segtools.SegmentGrid(segmentationNode, ct, [chestCT]) as reread:
            self.assertEqual(int(reread.mask(lungsID).sum()), after)
        # a grid that does not cover a segment refuses to write it (the part outside would be lost)
        with segtools.SegmentGrid(segmentationNode, ct) as primaryOnly:
            slicer.mrmlScene.RemoveNode(primaryOnly.reference)   # shrink the grid to the liver image
            primaryOnly.reference = segtools._gridNode(ct, np.zeros(3, int),
                                                       np.array(ct.GetImageData().GetDimensions()), "Taranis grid")
            primaryOnly.shape = tuple(slicer.util.arrayFromVolume(primaryOnly.reference).shape)
            primaryOnly._cache = {}
            with self.assertRaises(RuntimeError):
                primaryOnly.write(lungsID, np.zeros(primaryOnly.shape, bool))
        with segtools.SegmentGrid(segmentationNode, ct, [chestCT]) as reread:
            self.assertEqual(int(reread.mask(lungsID).sum()), after)   # the refused write changed nothing
        self.assertTrue(segtools.extendSegmentationGeometry(segmentationNode, ct, [ct, chestCT]))

    def test_segmentationLayout(self):
        """Own views; segments in the reference views and as wireframe in 3D; foreign renderings kept out;
        the segmentation's view list is restored when leaving."""
        import numpy as np
        ct = self._volume("CT", 0.0)
        spect = self._volume("SPECT", 5.0)
        segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        segmentationNode.CreateDefaultDisplayNodes()
        with segtools.SegmentGrid(segmentationNode, ct) as grid:
            mask = np.zeros(grid.shape, bool)
            mask[5:15, 5:15, 5:15] = True
            liverID = grid.add("Whole liver", W.SEGMENT_LIVER, mask)
        foreign = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "Other module model")
        foreign.CreateDefaultDisplayNodes()
        layoutManager = slicer.app.layoutManager()
        previousLayout = layoutManager.layout
        nodes = V.showLayout(ct, spect, segmentationNode)
        try:
            self.assertIn(layoutManager.layout, V.SEGMENTATION_LAYOUT_IDS)   # single or dual monitor
            self.assertTrue(all(nodes[key] is not None for key in list(V.SLICE_VIEWS) + ["3d"]))
            main = segmentationNode.GetDisplayNode()
            self.assertEqual(sorted(V._viewIDs(main)),
                             sorted([nodes["axialReference"].GetID(), nodes["coronalReference"].GetID()]))
            V.styleSegmentation3D(segmentationNode, nodes["3d"].GetID(), {liverID: W.SEGMENT_LIVER}, set())
            extra = V.segmentation3DDisplayNode(segmentationNode, create=False)
            self.assertEqual(V._viewIDs(extra), [nodes["3d"].GetID()])
            self.assertEqual(extra.GetRepresentation(), slicer.vtkMRMLDisplayNode.WireframeRepresentation)
            foreignViews = V._viewIDs(foreign.GetDisplayNode())
            self.assertTrue(foreignViews)   # explicit list now, without the layout's views
            self.assertNotIn(nodes["3d"].GetID(), foreignViews)
            composite = layoutManager.sliceWidget("TaranisSegAxialFused").mrmlSliceCompositeNode()
            self.assertEqual(composite.GetForegroundVolumeID(), spect.GetID())
        finally:
            V.leaveLayout(segmentationNode)
            layoutManager.setLayout(previousLayout)
        self.assertEqual(V._viewIDs(segmentationNode.GetDisplayNode()), [])   # all views again
        self.assertFalse(V.segmentation3DDisplayNode(segmentationNode, create=False).GetVisibility())

    def test_lsfFromCase(self):
        """Image-based LSF of the case, lung mass from the CT, outdated when the segments change."""
        import numpy as np
        controller = WorkflowController.instance()
        spect = self._volume("SPECT MAA", 0.0)
        ct = self._volume("CT", 0.0)
        for node in (spect, ct):
            node.SetSpacing(10.0, 10.0, 10.0)            # 1 mL voxels
        liver = np.zeros((20, 20, 20), bool)
        liver[2:8, 5:15, 5:15] = True                    # 600 mL
        lungs = np.zeros_like(liver)
        lungs[10:18, 5:15, 5:15] = True                  # 800 mL
        values = np.zeros(liver.shape, np.float32)
        values[liver] = 9.0
        values[lungs] = 0.25
        slicer.util.updateVolumeFromArray(spect, values)
        slicer.util.updateVolumeFromArray(ct, np.where(lungs, -800.0, 40.0).astype(np.float32))
        controller.startCase("LSF test", "L-1")
        case = controller.case
        case.setRoleNode(R.ROLE_DOSIMETRY, spect)
        case.setRoleType(R.ROLE_DOSIMETRY, R.TYPE_MAA_SPECT)
        case.setRoleNode(R.ROLE_DOSIMETRY_ANATOMY, ct)
        case.setRoleType(R.ROLE_DOSIMETRY_ANATOMY, R.TYPE_CT)
        case.setMode(R.MODE_RELATIVE)
        segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(ct)
        with segtools.SegmentGrid(segmentationNode, ct) as grid:
            grid.add("Whole liver", W.SEGMENT_LIVER, liver)
            lungsID = grid.add("Lungs", W.SEGMENT_LUNGS, lungs)
        case.setRoleNode(R.ROLE_SEGMENTATION, segmentationNode)
        controller.update()
        segments = controller.snapshot.segments
        result = L.calculateFromCase(case, segments)
        self.assertAlmostEqual(result["lsfPercent"], 100 * 200 / (200 + 5400), places=6)
        self.assertAlmostEqual(result["lungCoverage"], 1.0, places=6)
        mass, _, note = L.lungMassFromCase(case, segments)
        self.assertAlmostEqual(mass, 800 * 0.2, places=3)
        self.assertIn("cut", note)                        # 800 mL of lungs: probably cut by the field of view
        key = L.inputsKey(segments, spect.GetID())
        case.setLsf(result["lsfPercent"], L.SOURCE_IMAGE, mass, L.detailsJson(result, key), "[]")
        controller.update()
        self.assertEqual(controller.status(W.STEP_LSF).state, W.STATE_DONE)
        with segtools.SegmentGrid(segmentationNode, ct) as grid:
            smaller = lungs.copy()
            smaller[17] = False
            grid.write(lungsID, smaller)
        controller.update()
        self.assertEqual(controller.status(W.STEP_LSF).state, W.STATE_OUTDATED)

    def test_toolbarVisibility(self):
        toolbar = WorkflowToolbar.instance()
        visibility = toolbar.visibility
        saved = (visibility.initialized, visibility.showAtStartup, visibility.closedByUser, visibility.visible)
        try:
            visibility.showAtStartup = False
            visibility.visible = True
            controller = WorkflowController.instance()
            controller.startCase("Toolbar", "")
            self.assertTrue(toolbar.toolbar.visible or not slicer.util.mainWindow().visible)
            slicer.mrmlScene.Clear(0)
            controller.update()
            self.assertFalse(visibility.visible)
        finally:
            (visibility.initialized, visibility.showAtStartup, visibility.closedByUser, visibility.visible) = saved
            toolbar.applyVisibility()
