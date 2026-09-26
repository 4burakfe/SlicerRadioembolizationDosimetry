import os
import json
import math
import datetime
import logging

import numpy as np
import qt
import ctk
import vtk
import slicer
from slicer.ScriptedLoadableModule import *

try:
    from TaranisLib.dosimetry import *  # code shared with the other dosimetry module
except ImportError:  # developer layout (module folders side by side) before the Taranis folder is on sys.path
    import sys
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Taranis"))
    from TaranisLib.dosimetry import *


# Image value unit -> factor to convert to MBq/mL
CONCENTRATION_UNITS = [("Bq/mL", 1e-6), ("kBq/mL", 1e-3), ("MBq/mL", 1.0)]


ABSOLUTE_MODE_INFO = (
    "Absolute mode: dose is calculated for every voxel of the quantitative image (local deposition), "
    "including extrahepatic uptake. The whole-liver segment is used for display only. The liver density is "
    "applied to all voxels except those of the lung segments, which use the lung density; doses in other "
    "low-density tissue are underestimated.")


# ---------------------------------------------------------------------------
# Pure computation helpers (numpy only, no Slicer dependency -> unit-testable)
# ---------------------------------------------------------------------------

LUNG_DENSITY_TYPICAL = (0.2, 0.4)   # g/mL, aerated lung
LUNG_DENSITY_DEFAULT = 0.30         # g/mL, voxels of the Lungs segments (adjustable; estimate it from a CT)


def volumeLooksLikeCT(volumeNode):
    """CT numbers: air at about -1000 HU."""
    if volumeNode is None or volumeNode.GetImageData() is None:
        return False
    return float(volumeNode.GetImageData().GetScalarRange()[0]) <= -500.0


def meanDensityFromHU(huValues):
    """Mean physical density (g/mL) from CT numbers: (HU + 1000) / 1000, clipped to 0.001-1.1 (air .. soft tissue)."""
    density = np.clip((np.asarray(huValues, dtype=np.float64) + 1000.0) / 1000.0, 0.001, 1.1)
    return float(density.mean())


def computeAbsoluteDoseMap(concentration, toMBqPerML, decayFactor, conversionFactor, densityGPerML):
    """Local energy deposition from an activity-concentration image.

    dose_i [Gy] = C_i [MBq/mL] * decayFactor * CF [Gy*g/MBq] / rho [g/mL]
    """
    if densityGPerML <= 0:
        raise ValueError("Tissue density must be positive.")
    return concentration * (toMBqPerML * decayFactor * conversionFactor / densityGPerML)


# ---------------------------------------------------------------------------
# Slicer helpers
# ---------------------------------------------------------------------------


def detectConcentrationUnit(volumeNode):
    """Unit string stored on the volume (e.g. 'Bq/ml', '{SUVbw}g/ml'), or None if unknown."""
    try:
        coded = volumeNode.GetVoxelValueUnits()
        if coded and coded.GetCodeValue():
            return coded.GetCodeValue()
    except AttributeError:
        pass
    return None


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------

class RadioembolizationDosimetryAbsolute(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "Taranis - Radioembolization Dosimetry - Absolute"
        parent.categories = ["Nuclear Medicine"]
        parent.dependencies = []
        parent.contributors = ["Burak Demir, MD, FEBNM"]
        parent.helpText = """
        Post-treatment radioembolization dosimetry from a quantitative (Bq/mL) PET or SPECT image
        (local energy deposition, every voxel of the image).<br>
        For predictive (patient-relative) dosimetry use the "Taranis - Patient Relative Dosimetry" module.<br>
        After calculation the view layout shows segment models, isodose surfaces, the DVH and the reference
        image with isodose lines.
        """
        parent.acknowledgementText = """
        This file was developed by Burak Demir.
        """
        iconPath = os.path.join(os.path.dirname(__file__), "Resources", "taranis_logo.png")
        self.parent.icon = qt.QIcon(iconPath)

        slicer.app.connect("startupCompleted()", registerSampleData)
        # The results layout must be known before a scene that uses it is loaded (see registerResultsLayout)
        slicer.app.connect("startupCompleted()", installResultsLayoutRegistration)
        installResultsLayoutRegistration()


def registerSampleData():
    """Add data sets to Sample Data module."""
    import SampleData

    iconsPath = os.path.join(os.path.dirname(__file__), "Resources", "Icons")

    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        category="RadioembolizationDosimetry",
        sampleName="RadioembolizationDosimetry1",
        thumbnailFileName=os.path.join(iconsPath, "RadioembolizationDosimetry1.png"),
        uris=["https://github.com/4burakfe/SlicerRadioembolizationDosimetry_SampleImages/releases/download/TestImages/patient_1_MRI.nrrd",
              "https://github.com/4burakfe/SlicerRadioembolizationDosimetry_SampleImages/releases/download/TestImages/patient_1_Y90_PET.nrrd",
              "https://github.com/4burakfe/SlicerRadioembolizationDosimetry_SampleImages/releases/download/TestImages/patient_1_Segmentation.seg.nrrd"],
        fileNames=["patient_1_MRI.nrrd", "patient_1_Y90_PET.nrrd", "patient_1_Segmentation.seg.nrrd"],
        checksums=["SHA256:e2c598ae76d85e0b2cc0ebfd643d4f5ebda1d6f3df632c9172696878b858dfbe",
                   "SHA256:4f1f195ccb0dcd3c4c9fc967ed0c2e4bf9ac5985db02d951d64772d69979e55b",
                   "SHA256:550ceea296c7eab81f1dc7fb4ccf2f647fe223ba310eb2bd2dd0d0050aca739b"],
        nodeNames=["Patient 1 MRI", "Patient 1 Y90 PET", "Patient 1 Segmentation"],
    )


def reopenModuleSavedAsActive(*args):
    """After a scene is loaded: open this module if it was the active module when the scene was saved.
    Opening it creates the module GUI (if needed), which then restores its settings from the scene."""
    node = slicer.mrmlScene.GetSingletonNode(__name__, "vtkMRMLScriptedModuleNode")
    if node is None or node.GetParameter(PARAM_ACTIVE_MODULE) != "true":
        return

    def select():
        try:
            slicer.util.selectModule(__name__)
        except Exception as e:
            logging.warning(f"Could not open module {__name__}: {e}")

    qt.QTimer.singleShot(0, select)  # after the import has been fully processed


def installResultsLayoutRegistration(*args):
    """Register the results layout now and before every scene import, and reopen this module after a scene
    import if it was active when the scene was saved. Safe to call repeatedly: the scene observers of a
    previous call (e.g. before a module reload) are replaced."""
    registerResultsLayout()
    scene = getattr(slicer, "mrmlScene", None)
    if scene is None:
        return
    key = "_taranisSceneObservers_" + __name__
    for observedScene, tag in getattr(slicer, key, None) or []:
        try:
            observedScene.RemoveObserver(tag)
        except Exception:
            pass
    setattr(slicer, key, [(scene, scene.AddObserver(scene.StartImportEvent, registerResultsLayout)),
                          (scene, scene.AddObserver(scene.EndImportEvent, reopenModuleSavedAsActive)),
                          (scene, scene.AddObserver(scene.EndImportEvent, placeSecondaryViewWindowAfterLoad))])


class RadioembolizationDosimetryAbsoluteWidget(DosimetryWidgetBase):

    WIDGET_SETTINGS = [
        ("ClipNegativeValues", "clipNegativeCheckBox", "bool"),
        ("ImageUnit", "unitComboBox", "text"),  # after the input volume, which may preselect a unit
        ("HoursAfterTreatment", "hourSlider", "number"),
        ("HalfLifeHours", "halfLifeSpinBox", "number"),
        ("TotalActivityText", "totalActivityTextBox", "lineedit"),
        ("DecayCorrectedActivityText", "dectotalActivityTextBox", "lineedit"),
        ("ConversionFactor", "conversionFactorSpinBox", "number"),
        ("LiverDensity", "liverDensitySpinBox", "number"),
        ("LungDensity", "lungDensitySpinBox", "number"),
        ("IsodosePreset", "isodosePresetComboBox", "text"),
        ("IsodoseInSliceViews", "isodoseSliceToggleButton", "bool"),
        ("SegmentLabelsVisible", "segmentLabelToggleButton", "bool"),
        ("SegmentOutlineThickness", "segmentOutlineThicknessSpinBox", "number"),
        ("IsodoseLineThickness", "isodoseLineThicknessSpinBox", "number"),
        ("CustomMetric", "customMetricComboBox", "text"),  # before CustomValue: changing it resets the value
        ("CustomValue", "customValueSpinBox", "number"),
    ]

    # -- UI construction ---------------------------------------------------

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = RadioembolizationDosimetryAbsoluteLogic()
        self._restoring = True  # no parameter node writes until the GUI is built and filled from the scene
        self._sceneObservations = []
        self._dvhSources = []           # [{"label", "segmentIDs", "voxels"}] to rebuild dvhData after loading a scene
        self._dvhVoxelVolumeML = None
        self._doseChecksum = None
        self._annotationLabels = []     # [(segmentID, text)] of the slice view labels
        self._resultsDoseNode = None    # dose map and segmentation of the last calculation
        self._resultsSegmentationNode = None
        installResultsLayoutRegistration()  # also after a module reload in developer mode
        self._entered = False                 # module is shown (between enter() and exit())
        self._calculating = False
        self._suppressSegmentationEvents = False  # own segment colour/representation changes are not user edits
        self._pendingPreview = {"slices": False, "segments": False, "recenter": False}
        self._previewCameraKey = None         # (segmentation, whole liver) the 3D cameras were last reset for
        self._previewTimer = qt.QTimer()      # coalesces selection changes before the preview is updated
        self._previewTimer.setSingleShot(True)
        self._previewTimer.setInterval(400)
        self._previewTimer.connect("timeout()", self._runPreview)
        self.lastResult = None          # parameters/rows/sections of the last calculation (report)
        self.dvhData = []               # [(label, sortedDosesAsc, voxelVolumeML)] of the last calculation
        self.lastVisualization = None   # {"dose": node, "views": {...}} for isodose regeneration
        self._segmentationObservations = []
        self._segmentEditTimer = qt.QTimer()  # coalesces segment edits before the segment lists are refreshed
        self._segmentEditTimer.setSingleShot(True)
        self._segmentEditTimer.setInterval(300)
        self._segmentEditTimer.connect("timeout()", self._onSegmentEditsSettled)

        # ---- Layout: single / dual monitor (automatic from the number of screens until a button is used) ----
        layoutRow = qt.QHBoxLayout()
        self.singleMonitorButton = qt.QPushButton("Single monitor layout")
        self.singleMonitorButton.setToolTip("All six views in the main window.")
        self.dualMonitorButton = qt.QPushButton("Dual monitor layout")
        self.dualMonitorButton.setToolTip("Left column (segment models, fusion) in the main window; the right 2x2 views "
                                          "(isodose 3D, DVH, isodose and reference slices) in a separate window on the "
                                          "second screen.")
        for button in (self.singleMonitorButton, self.dualMonitorButton):
            button.setCheckable(True)
            layoutRow.addWidget(button)
        if not dualMonitorLayoutSupported():
            self.dualMonitorButton.setEnabled(False)
            self.dualMonitorButton.setToolTip("The dual monitor layout requires 3D Slicer 5.0 or later.")
        self.singleMonitorButton.connect("clicked()", lambda: self.onLayoutModeClicked("single"))
        self.dualMonitorButton.connect("clicked()", lambda: self.onLayoutModeClicked("dual"))
        self.layout.addLayout(layoutRow)

        bannerPath = os.path.join(os.path.dirname(__file__), "Resources", "banner.png")
        if os.path.exists(bannerPath):
            bannerLabel = qt.QLabel()
            bannerLabel.setPixmap(qt.QPixmap(bannerPath).scaledToWidth(500, qt.Qt.SmoothTransformation))
            bannerLabel.setAlignment(qt.Qt.AlignCenter)
            self.layout.addWidget(bannerLabel)
        else:
            logging.warning(f"Banner file not found at {bannerPath}")

        # ---- Method information ----
        modeBox = qt.QGroupBox("Absolute quantification (post-treatment quantitative PET/SPECT)")
        modeLayout = qt.QVBoxLayout(modeBox)
        self.modeInfoLabel = qt.QLabel(ABSOLUTE_MODE_INFO)
        self.modeInfoLabel.setWordWrap(True)
        self.modeInfoLabel2 = qt.QLabel(DISCLAIMER)
        self.modeInfoLabel2.setWordWrap(True)
        self.modeInfoLabel2.setStyleSheet("color: #ff0000; font-weight: bold;")

        modeLayout.addWidget(self.modeInfoLabel)
        modeLayout.addWidget(self.modeInfoLabel2)
        self.layout.addWidget(modeBox)

        parametersCollapsibleButton = ctk.ctkCollapsibleButton()
        parametersCollapsibleButton.text = "Parameters"
        self.layout.addWidget(parametersCollapsibleButton)
        parametersLayout = qt.QVBoxLayout(parametersCollapsibleButton)

        # ---- Images and segments ----
        generalBox = qt.QGroupBox("Images and segments")
        generalLayout = qt.QFormLayout(generalBox)
        parametersLayout.addWidget(generalBox)

        self.spectSelector = self._makeNodeSelector("vtkMRMLScalarVolumeNode", "Select the input SPECT/PET volume.")
        self.inputVolumeLabel = qt.QLabel("Input quantitative PET/SPECT: ")
        generalLayout.addRow(self.inputVolumeLabel, self.spectSelector)
        self.clipNegativeCheckBox = qt.QCheckBox("Set negative voxel values to 0")
        self.clipNegativeCheckBox.setChecked(True)
        self.clipNegativeCheckBox.setToolTip(
            "Set negative voxel values (reconstruction noise) to 0 when the image is read,\n"
            "so the dose map, totals, QC fractions and statistics are all calculated from the same values.\n"
            "Unticked: negative values are used as is and produce negative voxel doses.\n"
            "The number of negative voxels is reported in the results either way.")
        generalLayout.addRow("", self.clipNegativeCheckBox)

        self.referenceSelector = self._makeNodeSelector(
            "vtkMRMLScalarVolumeNode", "Anatomical reference (CT or MRI) shown under the isodose lines.")
        generalLayout.addRow("Reference Volume (CT/MRI): ", self.referenceSelector)

        self.segmentationSelector = self._makeNodeSelector(
            "vtkMRMLSegmentationNode",
            "Master segmentation for dosimetric calculations. All segment selections below use it.")
        generalLayout.addRow("Master Segmentation: ", self.segmentationSelector)

        self.liverSegmentSelector = SegmentComboBox(
            "Segment representing the whole liver. Used for display and QC only; "
            "the dose is calculated for the whole image.", onChanged=self._refreshCategorizer)
        generalLayout.addRow("Whole Liver Segment: ", self.liverSegmentSelector.widget)

        # ---- Display windowing of the slice views ----
        windowBox = qt.QGroupBox("Display windowing")
        windowLayout = qt.QFormLayout(windowBox)
        parametersLayout.addWidget(windowBox)
        referenceGrid = qt.QGridLayout()
        self.referenceWindowButtons = []
        for index, preset in enumerate(REFERENCE_WINDOW_PRESETS):
            button = qt.QPushButton(preset[0])
            button.setToolTip(preset[4])
            button.connect("clicked()", lambda *args, p=preset: self.onReferenceWindowPreset(p))
            referenceGrid.addWidget(button, index // 3, index % 3)
            self.referenceWindowButtons.append(button)
        windowLayout.addRow("Reference:", referenceGrid)
        spectRow = qt.QHBoxLayout()
        self.spectWindowButtons = []
        for percent in SPECT_WINDOW_PERCENTS:
            button = qt.QPushButton(f"0-{percent}%")
            button.setToolTip(f"PET/SPECT window from 0 to {percent} % of the maximum voxel value")
            button.connect("clicked()", lambda *args, p=percent: self.onSpectWindowPercent(p))
            spectRow.addWidget(button)
            self.spectWindowButtons.append(button)
        windowLayout.addRow("SPECT/PET:", spectRow)
        for selector in (self.referenceSelector, self.spectSelector):
            selector.connect("currentNodeChanged(vtkMRMLNode*)", self._updateWindowButtons)
        self._updateWindowButtons()

        # ---- Segment categories ----
        categoryBox = qt.QGroupBox("Segment categories")
        categoryBoxLayout = qt.QVBoxLayout(categoryBox)
        parametersLayout.addWidget(categoryBox)
        categoryInfoLabel = qt.QLabel(
            "Select segments on the left and move them into a category with > (back with <). "
            "Colours after calculation: whole liver white, tumors pink, normal tissue turquoise, "
            "others and uncategorized gray. Categories are saved with the segmentation.")
        categoryInfoLabel.setWordWrap(True)
        categoryBoxLayout.addWidget(categoryInfoLabel)
        self.categorizer = SegmentCategorizer(onChanged=self._onCategoriesChanged, categories=SEGMENT_CATEGORIES)
        categoryBoxLayout.addWidget(self.categorizer.widget)

        # ---- Calculation and display settings ----
        settingsBox = qt.QGroupBox("Calculation and display settings")
        generalLayout = qt.QFormLayout(settingsBox)
        parametersLayout.addWidget(settingsBox)

        self.conversionFactorSpinBox = qt.QDoubleSpinBox()
        self.conversionFactorSpinBox.setRange(0.01, 100.0)
        self.conversionFactorSpinBox.setValue(49.67)
        self.conversionFactorSpinBox.setSingleStep(0.1)
        self.conversionFactorSpinBox.setToolTip("Energy deposited per unit activity, J/GBq (= Gy*kg/GBq = Gy*g/MBq).")
        generalLayout.addRow("Conversion Factor (J/GBq):", self.conversionFactorSpinBox)
        self.physicsNoteLabel = qt.QLabel("")
        self.physicsNoteLabel.setWordWrap(True)
        self.physicsNoteLabel.setStyleSheet("color: #d97706; font-weight: bold;")
        self.physicsNoteLabel.setVisible(False)
        generalLayout.addRow(self.physicsNoteLabel)

        self.liverDensitySpinBox = qt.QDoubleSpinBox()
        self.liverDensitySpinBox.setRange(0.01, 10.0)
        self.liverDensitySpinBox.setValue(1.05)
        self.liverDensitySpinBox.setSingleStep(0.01)
        generalLayout.addRow("Liver Density (g/mL):", self.liverDensitySpinBox)
        self.lungDensitySpinBox = qt.QDoubleSpinBox()
        self.lungDensitySpinBox.setRange(0.05, 1.2)
        self.lungDensitySpinBox.setDecimals(2)
        self.lungDensitySpinBox.setSingleStep(0.01)
        self.lungDensitySpinBox.setValue(LUNG_DENSITY_DEFAULT)
        self.lungDensitySpinBox.setToolTip(
            "Density used for the voxels of the Lungs segment(s) (default 0.30 g/mL; aerated lung is typically "
            "0.2-0.4 g/mL). All other voxels use the liver density. After a calculation the lung density can be "
            "estimated from a CT reference image.")
        generalLayout.addRow("Lung Density (g/mL):", self.lungDensitySpinBox)

        self.isodosePresetComboBox = qt.QComboBox()
        for name, _ in ISODOSE_PRESETS:
            self.isodosePresetComboBox.addItem(name)
        self.isodosePresetComboBox.setToolTip("Isodose levels and colours. Changing it updates existing isodose surfaces.")
        generalLayout.addRow("Isodose set: ", self.isodosePresetComboBox)
        self.isodoseLegendLabel = qt.QLabel()
        self.isodoseLegendLabel.setTextFormat(qt.Qt.RichText)
        self.isodoseLegendLabel.setWordWrap(True)
        generalLayout.addRow("", self.isodoseLegendLabel)
        self.isodoseSliceToggleButton = qt.QPushButton()
        self.isodoseSliceToggleButton.setCheckable(True)
        self.isodoseSliceToggleButton.setChecked(True)
        self.isodoseSliceToggleButton.setToolTip("Show or hide the isodose lines (and legend) in the bottom-middle view.")
        generalLayout.addRow("", self.isodoseSliceToggleButton)
        self.segmentLabelToggleButton = qt.QPushButton()
        self.segmentLabelToggleButton.setCheckable(True)
        self.segmentLabelToggleButton.setChecked(True)
        self.segmentLabelToggleButton.setToolTip(
            "Show or hide segment names with leader lines in the slice views (bottom row).")
        generalLayout.addRow("", self.segmentLabelToggleButton)

        self.segmentOutlineThicknessSpinBox = qt.QSpinBox()
        self.segmentOutlineThicknessSpinBox.setRange(1, 10)
        self.segmentOutlineThicknessSpinBox.setValue(DEFAULT_SEGMENT_OUTLINE_THICKNESS)
        self.segmentOutlineThicknessSpinBox.setSuffix(" px")
        self.segmentOutlineThicknessSpinBox.setToolTip("Thickness of the segment outlines in the slice views. "
                                                       "Applied immediately.")
        generalLayout.addRow("Segment line thickness:", self.segmentOutlineThicknessSpinBox)

        self.isodoseLineThicknessSpinBox = qt.QSpinBox()
        self.isodoseLineThicknessSpinBox.setRange(1, 10)
        self.isodoseLineThicknessSpinBox.setValue(DEFAULT_ISODOSE_LINE_THICKNESS)
        self.isodoseLineThicknessSpinBox.setSuffix(" px")
        self.isodoseLineThicknessSpinBox.setToolTip("Thickness of the isodose lines in the slice view. "
                                                    "Applied immediately.")
        generalLayout.addRow("Isodose line thickness:", self.isodoseLineThicknessSpinBox)

        self.outputVolumeSelector = self._makeNodeSelector(
            "vtkMRMLScalarVolumeNode", "Output volume for the Gy map. If none is selected, a new one is created.",
            allowCreate=True)
        generalLayout.addRow("Output Volume: ", self.outputVolumeSelector)

        # ---- Absolute-quantification settings ----
        self.absoluteGroupBox = qt.QGroupBox("Absolute quantification settings")
        absLayout = qt.QFormLayout(self.absoluteGroupBox)
        parametersLayout.addWidget(self.absoluteGroupBox)

        self.unitComboBox = qt.QComboBox()
        for label, _ in CONCENTRATION_UNITS:
            self.unitComboBox.addItem(label)
        self.unitComboBox.setToolTip("Unit of the voxel values. SUV images are not supported.")
        absLayout.addRow("Image unit: ", self.unitComboBox)

        self.hourSlider = ctk.ctkSliderWidget()
        self.hourSlider.singleStep = 1
        self.hourSlider.minimum = 0
        self.hourSlider.maximum = 200
        self.hourSlider.value = 0
        self.hourSlider.setToolTip(
            "Hours between administration and the time the image is decay-corrected to.\n"
            "Most PET scanners decay-correct to scan start, so use administration-to-scan-start time.\n"
            "If the image is already decay-corrected to administration time, set 0.")
        absLayout.addRow("Hours after treatment: ", self.hourSlider)

        self.halfLifeSpinBox = qt.QDoubleSpinBox()
        self.halfLifeSpinBox.setRange(0.1, 200.0)
        self.halfLifeSpinBox.setValue(64.2)
        self.halfLifeSpinBox.setSingleStep(0.1)
        self.halfLifeSpinBox.setToolTip("Physical half-life in hours (Y-90: 64.2 h, default; Ho-166: 26.8 h; "
                                        "Re-188: 17.0 h).")
        absLayout.addRow("Half-Life (hours):", self.halfLifeSpinBox)
        self.conversionFactorSpinBox.connect("valueChanged(double)", self._updatePhysicsNote)
        self.halfLifeSpinBox.connect("valueChanged(double)", self._updatePhysicsNote)

        self.totalActivityTextBox = qt.QLineEdit()
        self.totalActivityTextBox.setReadOnly(True)
        self.totalActivityTextBox.setToolTip("Total activity in the whole field of view (incl. background/noise), MBq.")
        absLayout.addRow("Total Activity\nin image FOV (MBq): ", self.totalActivityTextBox)

        self.dectotalActivityTextBox = qt.QLineEdit()
        self.dectotalActivityTextBox.setReadOnly(True)
        self.dectotalActivityTextBox.setToolTip("Total FOV activity decay-corrected to administration, MBq.")
        absLayout.addRow("Total Decay\nCorr Act (MBq): ", self.dectotalActivityTextBox)

        # Absolute dosimetry only on a quantitative image (see _absoluteInputProblem)
        self.absoluteLockLabel = qt.QLabel()
        self.absoluteLockLabel.setWordWrap(True)
        self.absoluteLockLabel.setStyleSheet("color: #d32f2f; font-weight: bold;")
        self.absoluteLockLabel.hide()
        parametersLayout.addWidget(self.absoluteLockLabel)

        self.calculateButton = qt.QPushButton("Calculate")
        self.calculateButton.toolTip = "Perform dosimetric calculations and show the results layout."
        parametersLayout.addWidget(self.calculateButton)

        # ---- Results ----
        resultsCollapsibleButton = ctk.ctkCollapsibleButton()
        resultsCollapsibleButton.text = "Results"
        self.layout.addWidget(resultsCollapsibleButton)
        resultsLayout = qt.QVBoxLayout(resultsCollapsibleButton)

        resultsLayout.addWidget(qt.QLabel("Segment doses:"))
        self.segmentDoseTable = self._makeTable(["Segment", "Category", "Dose (Gy)", "Volume (mL)", "Mass (g)",
                                                "Activity (MBq)"], 220)
        resultsLayout.addWidget(self.segmentDoseTable)

        self.resultNoteLabel = qt.QLabel()
        self.resultNoteLabel.setWordWrap(True)
        resultsLayout.addWidget(self.resultNoteLabel)

        resultsLayout.addWidget(qt.QLabel("D values (Gy) - minimum dose to the hottest x % of the volume:"))
        self.dValueTable = self._makeTable(["Segment"] + [f"D{x}" for x in D_METRICS], 170)
        resultsLayout.addWidget(self.dValueTable)

        resultsLayout.addWidget(qt.QLabel("V values (%) - percentage of the volume receiving at least x Gy:"))
        self.vValueTable = self._makeTable(["Segment"] + [f"V{x}" for x in V_METRICS], 170)
        resultsLayout.addWidget(self.vValueTable)

        self._addReportButtons(resultsLayout)

        # ---- Custom DVH metrics ----
        self.customCollapsibleButton = ctk.ctkCollapsibleButton()
        self.customCollapsibleButton.text = "Custom DVH metrics"
        self.layout.addWidget(self.customCollapsibleButton)
        customLayout = qt.QFormLayout(self.customCollapsibleButton)

        self.customSegmentList = qt.QListWidget()
        self.customSegmentList.setMinimumHeight(100)
        self.customSegmentList.setToolTip("Tick the segments to evaluate (available after a calculation).")
        customLayout.addRow("Segments: ", self.customSegmentList)

        self.customMetricComboBox = qt.QComboBox()
        self.customMetricComboBox.addItem("D - dose (Gy) covering x % of the volume")
        self.customMetricComboBox.addItem("V - volume (%) receiving at least x Gy")
        customLayout.addRow("Metric: ", self.customMetricComboBox)

        self.customValueSpinBox = qt.QDoubleSpinBox()
        self.customValueSpinBox.setDecimals(1)
        customLayout.addRow("x: ", self.customValueSpinBox)

        customButtons = qt.QHBoxLayout()
        self.customComputeButton = qt.QPushButton("Compute")
        self.customClearButton = qt.QPushButton("Clear results")
        customButtons.addWidget(self.customComputeButton)
        customButtons.addWidget(self.customClearButton)
        customLayout.addRow(customButtons)

        self.customResultTable = self._makeTable(["Segment", "Metric", "Value"], 140)
        customLayout.addRow(self.customResultTable)

        # ---- Connections ----
        self.segmentationSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSegmentationNodeChanged)
        self.spectSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onInputVolumeChanged)
        self.isodosePresetComboBox.connect("currentIndexChanged(int)", self.onIsodosePresetChanged)
        self.isodoseSliceToggleButton.connect("toggled(bool)", self.onIsodoseSliceToggled)
        self.segmentLabelToggleButton.connect("toggled(bool)", self.onSegmentLabelsToggled)
        self.segmentOutlineThicknessSpinBox.connect("valueChanged(int)", self.logic.setSegmentOutlineThickness)
        self.isodoseLineThicknessSpinBox.connect("valueChanged(int)", self.logic.setIsodoseLineThickness)
        self.calculateButton.connect("clicked(bool)", self.onCalculateButton)
        self.customMetricComboBox.connect("currentIndexChanged(int)", self.onCustomMetricChanged)
        self.customComputeButton.connect("clicked(bool)", self.onCustomComputeClicked)
        self.customClearButton.connect("clicked(bool)", self.onCustomClearClicked)

        self.onSegmentationNodeChanged(self.segmentationSelector.currentNode())
        self.onInputVolumeChanged(self.spectSelector.currentNode())
        self.onCustomMetricChanged(0)
        self._updateIsodoseLegend()
        self.onIsodoseSliceToggled(self.isodoseSliceToggleButton.checked)
        self.onSegmentLabelsToggled(self.segmentLabelToggleButton.checked)
        self.logic.setSegmentOutlineThickness(self.segmentOutlineThicknessSpinBox.value)
        self.logic.setIsodoseLineThickness(self.isodoseLineThicknessSpinBox.value)
        self.customCollapsibleButton.setEnabled(False)

        self.layout.addStretch(1)
        infoTextBox = qt.QTextEdit()
        infoTextBox.setReadOnly(True)
        infoTextBox.setMaximumHeight(250)
        infoTextBox.setPlainText(
            "This module enables post-treatment dosimetry with quantitative SPECT and PET images.\n"
            "This module is NOT a medical device. It is for research purposes only.\n"
            "Default conversion factor and half-life are for Y-90: 49.67 J/GBq, 64.2 h.\n"
            "For reference (local deposition): Ho-166 15.87 J/GBq, half-life 26.8 h; Re-188 about 10.8 J/GBq, "
            "half-life 17.0 h. Other values are flagged in the dose checks: the Y-90 dose thresholds may not apply.\n"
            "Written by: Burak Demir, MD, FEBNM \n"
            "This module is provided open-source for the nuclear medicine community. If you find it helpful for your research, please consider citing:\n"
            "- Demir B, Soydal C, Mesci I, Celebioglu EC, Bilgic MS, Kuru Oz D, Kucuk NO. Utility of respiratory motion correction and effects on dosimetry in imaging with integrated Y-90 PET/MRI after radioembolization of liver tumors. Phys Med. 2026 Feb;142:105717. doi: 10.1016/j.ejmp.2026.105717. Epub 2026 Jan 5. PMID: 41494332.\n"
            "- Soydal C, Demir B, Araz M, Mesci I, Çelebioğlu EC, Kucuk NO. Safety of repeated trans-arterial radioembolization with multi-compartment dosimetry. Ann Nucl Med. 2025 Dec;39(12):1306-1318. doi: 10.1007/s12149-025-02094-9. Epub 2025 Aug 20. PMID: 40833652.\n"
            "For support, feedback, and suggestions: 4burakfe@gmail.com\n"
        )
        infoTextBox.setToolTip("Module information and instructions.")
        self.layout.addWidget(infoTextBox)

        # ---- Settings are kept in the module's parameter node, which is saved with the scene ----
        self._connectSettingsSignals()
        self._connectPreviewSignals()
        self._updateLayoutButtons()
        scene = slicer.mrmlScene
        for event, callback in ((scene.StartCloseEvent, self.onSceneStartClose),
                                (scene.EndCloseEvent, self.onSceneEndClose),
                                (scene.StartImportEvent, self.onSceneStartImport),
                                (scene.EndImportEvent, self.onSceneEndImport),
                                (scene.StartSaveEvent, self.onSceneStartSave)):
            self._sceneObservations.append(scene.AddObserver(event, callback))
        self._restoreFromParameterNode()  # the scene may have been loaded before the module was opened

    # -- selector handling --------------------------------------------------


    def onSegmentationNodeChanged(self, node):
        self.liverSegmentSelector.setSegmentation(node)
        self._observeSegmentation(node)
        self.categorizer.categories = loadSegmentCategories(node)
        self._refreshCategorizer()

    # -- segment categories ------------------------------------------------

    def _categoryExclusions(self):
        return {self.liverSegmentSelector.currentSegmentID()} - {""}

    # -- segmentation edits (keep the segment lists current) ---------------

    def _observeSegmentation(self, segmentationNode):
        for observedObject, tag in self._segmentationObservations:
            observedObject.RemoveObserver(tag)
        self._segmentationObservations = []
        if segmentationNode is None:
            return
        segmentation = segmentationNode.GetSegmentation()
        eventNames = ("SegmentAdded", "SegmentRemoved", "SegmentModified", "SegmentsOrderModified")
        events = {getattr(slicer.vtkSegmentation, n) for n in eventNames if hasattr(slicer.vtkSegmentation, n)}
        for event in events:
            tag = segmentation.AddObserver(event, lambda caller, eventId: self._onSegmentationEvent())
            self._segmentationObservations.append((segmentation, tag))

    def _onSegmentEditsSettled(self):
        node = self.segmentationSelector.currentNode()
        self.liverSegmentSelector.refresh()
        existing = {sid for sid, _ in segmentList(node)}
        self.categorizer.categories = {sid: c for sid, c in self.categorizer.categories.items() if sid in existing}
        self._refreshCategorizer()
        self._saveSettings()  # the selected whole-liver segment may have been removed
        self._schedulePreview(segments=True)

    # -- layout (single / dual monitor) --------------------------------------


    def _showPreviewLabels(self, segmentationNode):
        """Segment name and volume in the slice views. The labels of a calculation (with the mean dose) are kept
        while its segmentation is selected."""
        if segmentationNode is None:
            self.logic.removeSegmentAnnotations()
            return
        results = self._resultsSegmentationNode
        if self._hasResults() and results is not None and results.GetID() == segmentationNode.GetID():
            return
        labels = []
        ignored = {sid for sid, category in loadSegmentCategories(segmentationNode).items()
                   if category == CATEGORY_IGNORED}
        for segmentID, segmentName in segmentList(segmentationNode):
            if segmentID in ignored:
                continue  # e.g. the lungs: not part of the dosimetry
            try:
                volumeML = self._previewSegmentVolumeML(segmentationNode, segmentID)
            except Exception:
                volumeML = None
            labels.append((segmentID, segmentLabelText(segmentName, volumeML)))
        self.logic.showSegmentAnnotations(segmentationNode, labels)

    # -- live preview --------------------------------------------------------


    def _absoluteInputProblem(self, node=None):
        """Why absolute dosimetry is locked for the selected input, or "" (see roles.absoluteDosimetryProblem):
        with a Taranis case the input must be the case's quantitative dosimetry image, without a case it must carry
        Bq/mL units."""
        node = node or self.spectSelector.currentNode()
        if node is None:
            return ""   # the usual input checks ask for an input volume
        try:
            from TaranisLib import roles as R
            from TaranisLib.case import TaranisCase, volumeInfo
        except ImportError as e:
            logging.warning(f"Absolute dosimetry: Taranis rules not available ({e}); input not checked.")
            return ""
        case = TaranisCase.find()
        caseImage = case.roleNode(R.ROLE_DOSIMETRY) if case is not None else None
        try:
            dicomUnits = volumeInfo(node).units
        except Exception:
            dicomUnits = ""
        return R.absoluteDosimetryProblem(
            node.GetName(), case is not None, caseImage.GetName() if caseImage is not None else "",
            caseImage is not None and caseImage.GetID() == node.GetID(),
            case.roleType(R.ROLE_DOSIMETRY) if case is not None else None,
            detectConcentrationUnit(node) or "", dicomUnits or "")

    def _updateAbsoluteLock(self):
        try:
            problem = self._absoluteInputProblem()
        except Exception as e:
            logging.warning(f"Absolute dosimetry: could not check the input image: {e}")
            problem = ""
        self.absoluteLockLabel.text = problem
        self.absoluteLockLabel.setVisible(bool(problem))
        self.calculateButton.enabled = not problem
        self.calculateButton.toolTip = problem or "Perform dosimetric calculations and show the results layout."

    def enter(self):
        DosimetryWidgetBase.enter(self)
        self._updateAbsoluteLock()   # the case's image assignment may have changed in the Taranis module

    def onInputVolumeChanged(self, node):
        """Pre-select the image unit if the volume carries unit metadata."""
        self._updateAbsoluteLock()
        unit = detectConcentrationUnit(node) if node else None
        if not unit:
            return
        for index, (label, _) in enumerate(CONCENTRATION_UNITS):
            if unit.lower() == label.lower():
                self.unitComboBox.setCurrentIndex(index)
                return

    def onIsodosePresetChanged(self, index):
        self._updateIsodoseLegend()
        presetName, levels = self.currentIsodosePreset()
        vis = self.lastVisualization
        if not vis or not slicer.mrmlScene.IsNodePresent(vis["dose"]) or vis["dose"].GetImageData() is None:
            return
        with slicer.util.tryWithErrorDisplay("Could not update the isodose surfaces.", waitCursor=True):
            self.logic.createIsodoseModels(vis["dose"], levels, vis["views"])
            self._setReportParameter("Isodose set", presetName)

    # -- input validation --------------------------------------------------

    def _collectInputs(self):
        """Validated inputs, or None after showing an error."""
        spect = self.spectSelector.currentNode()
        reference = self.referenceSelector.currentNode()
        seg = self.segmentationSelector.currentNode()
        output = self.outputVolumeSelector.currentNode()
        liverID = self.liverSegmentSelector.currentSegmentID()

        problems = []
        if not spect:
            problems.append("Select an input volume.")
        if not reference:
            problems.append("Select a reference volume (CT/MRI).")
        if not seg:
            problems.append("Select a segmentation.")
        if not liverID:
            problems.append("Select the whole liver segment.")
        if output and output in (spect, reference):
            problems.append("The output volume must differ from the input and reference volumes "
                            "(otherwise their image data would be overwritten).")
        if spect:
            locked = self._absoluteInputProblem(spect)
            if locked:
                problems.append(locked)

        presetName, levels = self.currentIsodosePreset()
        inputs = {
            "spect": spect, "reference": reference, "segmentation": seg, "output": output,
            "liverID": liverID, "clipNegative": self.clipNegativeCheckBox.checked,
            "categories": {sid: c for sid, c in self.categorizer.categories.items()
                           if sid != liverID and sid in {s for s, _ in segmentList(seg)}},
            "conversionFactor": self.conversionFactorSpinBox.value,
            "density": self.liverDensitySpinBox.value,
            "lungDensity": self.lungDensitySpinBox.value,
            "isodosePreset": presetName, "isodoseLevels": levels,
        }

        unitLabel, toMBqPerML = CONCENTRATION_UNITS[self.unitComboBox.currentIndex]
        inputs.update({
            "unitLabel": unitLabel, "toMBqPerML": toMBqPerML,
            "hours": self.hourSlider.value, "halfLife": self.halfLifeSpinBox.value,
        })

        if problems:
            slicer.util.errorDisplay("\n".join(problems))
            return None

        storedUnit = detectConcentrationUnit(spect)
        if storedUnit and storedUnit.lower() != inputs["unitLabel"].lower():
            if not slicer.util.confirmOkCancelDisplay(
                    f"The volume metadata says its unit is '{storedUnit}', "
                    f"but '{inputs['unitLabel']}' is selected.\nContinue anyway?"):
                return None

        if not output:  # no output selected: create one and select it, so later runs reuse it
            output = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "Dose Map (Gy)")
            self.outputVolumeSelector.setCurrentNode(output)
            inputs["output"] = output
        return inputs

    # -- button handlers ---------------------------------------------------

    def _calculate(self):
        inputs = self._collectInputs()
        if inputs is None:
            return
        with slicer.util.tryWithErrorDisplay("Dose calculation failed.", waitCursor=True):
            self._runAbsolute(inputs)

    # -- calculations ------------------------------------------------------

    def _runAbsolute(self, inputs):
        (doseArray, totalAtScan, totalAtAdmin, decayFactor, voxelVolumeML,
         concentration, negativeVoxels) = self.logic.computeAbsoluteDose(
            inputs["spect"], inputs["toMBqPerML"], inputs["hours"], inputs["halfLife"],
            inputs["conversionFactor"], inputs["density"], clipNegativeValues=inputs["clipNegative"])
        lungIDs = [sid for sid, c in inputs["categories"].items() if c == CATEGORY_LUNGS]
        lungVoxels = 0
        if lungIDs:
            # dose = concentration x factor / density: voxels of the lungs (not also in the whole liver) get the lung
            # density instead of the liver density
            lungMask = np.zeros(doseArray.shape, bool)
            for segmentID in lungIDs:
                lungMask |= segmentMaskOnVolumeGrid(inputs["segmentation"], segmentID, inputs["spect"])
            lungMask &= ~segmentMaskOnVolumeGrid(inputs["segmentation"], inputs["liverID"], inputs["spect"])
            lungVoxels = int(np.count_nonzero(lungMask))
            if lungVoxels:
                doseArray[lungMask] *= inputs["density"] / inputs["lungDensity"]
        self.totalActivityTextBox.setText(f"{totalAtScan:.2f} MBq")
        self.dectotalActivityTextBox.setText(f"{totalAtAdmin:.2f} MBq")
        writeDoseVolume(inputs["output"], inputs["spect"], doseArray)
        inputs["output"].SetName("Dose Map (Gy)")

        # Liver mask is only used for display and QC; the dose covers the whole image
        liverMask = segmentMaskOnVolumeGrid(inputs["segmentation"], inputs["liverID"], inputs["spect"])
        if not liverMask.any():
            raise ValueError("The whole liver segment does not overlap the PET/SPECT volume. "
                             "Check registration and transforms.")
        applySegmentColors(inputs["segmentation"], inputs["liverID"], (), inputs["categories"])
        stats = self.logic.segmentStatistics(doseArray, inputs["segmentation"], inputs["spect"], voxelVolumeML,
                                             inputs["conversionFactor"], inputs["density"],
                                             inputs["liverID"], inputs["categories"])
        for s in stats:  # lung activity from the lung density (dose x mass / conversion factor)
            if s["role"] == CATEGORY_LUNGS and np.isfinite(s["dose"]):
                s["activity"] = s["dose"] * s["volume"] * inputs["lungDensity"] / inputs["conversionFactor"]

        # Same (prepared) values as the dose map, so the QC numbers are consistent with it
        outsideFraction = countsOutsideMaskFraction(concentration, liverMask)
        notes = [ABSOLUTE_MODE_INFO]
        qcLines = []
        if np.isfinite(outsideFraction):
            qcLines.append(f"{100 * outsideFraction:.1f}% of the image activity lies outside the whole-liver segment "
                           "(it is included in the dose map).")
        negativeInLiver = int(np.count_nonzero(slicer.util.arrayFromVolume(inputs["spect"])[liverMask] < 0))
        negativeNote = negativeVoxelNote(negativeVoxels, concentration.size, inputs["clipNegative"],
                                         f"; {negativeInLiver} inside the whole liver")
        if negativeVoxels:
            qcLines.append(negativeNote)
        notes.extend(qcLines)
        if not negativeVoxels:
            notes.append(negativeNote)
        qcNote = "\n".join(qcLines)

        # Dose checks: image activity outside the liver and lungs, whole-liver activity outside the perfused volumes
        # (segments marked as perfused volumes in the Taranis workflow)
        liverOrLungs = liverMask.copy()
        for segmentID in lungIDs:
            liverOrLungs |= segmentMaskOnVolumeGrid(inputs["segmentation"], segmentID, inputs["spect"])
        perfusedIDs = [sid for sid in taggedPerfusedIDs(inputs["segmentation"]) if sid != inputs["liverID"]]
        outsidePerfused = None
        if perfusedIDs:
            perfusedMask = np.zeros(liverMask.shape, bool)
            for segmentID in perfusedIDs:
                perfusedMask |= segmentMaskOnVolumeGrid(inputs["segmentation"], segmentID, inputs["spect"])
            outsidePerfused = countsOutsideMaskFraction(concentration[liverMask], perfusedMask[liverMask])
        lungRows = [s for s in stats if s["role"] == CATEGORY_LUNGS and s["id"] and np.isfinite(s["dose"])]
        checks = DG.doseChecks(
            doseCheckSegments(stats, inputs["segmentation"]),
            microspheres=DG.microspheresFromText(inputs["isodosePreset"]),
            lungDosesGy=[(f"'{row['name']}'", float(row["dose"])) for row in lungRows],
            extraUptakeFraction=countsOutsideMaskFraction(concentration, liverOrLungs),
            lungsSegmented=bool(lungIDs), outsidePerfusedFraction=outsidePerfused,
            hoursAfterTreatment=inputs["hours"], conversionFactor=inputs["conversionFactor"],
            halfLifeHours=inputs["halfLife"])
        notes, qcNote = self._withDoseChecks(notes, qcNote, checks)

        segmentation = inputs["segmentation"].GetSegmentation()
        parameters = [
            ("Mode", "Absolute quantification"),
            ("Input volume", inputs["spect"].GetName()),
            ("Reference volume", inputs["reference"].GetName()),
            ("Segmentation", inputs["segmentation"].GetName()),
            ("Whole liver segment", segmentation.GetSegment(inputs["liverID"]).GetName()),
            ("Image unit", inputs["unitLabel"]),
            ("Negative voxel values", "set to 0" if inputs["clipNegative"] else "used as is"),
            ("Activity During Imaging (whole FOV)", f"{totalAtScan:.2f} MBq"),
            ("Hours after treatment", f"{inputs['hours']:.2f} h"),
            ("Half-life", f"{inputs['halfLife']:.2f} h"),
            ("Decay correction factor", f"{decayFactor:.4f}"),
            ("Decay Corrected Activity (whole FOV)", f"{totalAtAdmin:.2f} MBq"),
            ("Conversion Factor", f"{inputs['conversionFactor']:.2f} J/GBq"),
            ("Liver Density", f"{inputs['density']:.2f} g/mL"),
        ]
        if lungVoxels:
            parameters.append(("Lung Density", f"{inputs['lungDensity']:.2f} g/mL (voxels of the lung segments)"))
        parameters.append(("Isodose set", inputs["isodosePreset"]))
        self._finishCalculation(inputs, stats, voxelVolumeML, liverMask, [], parameters, notes, qcNote,
                                "Taranis - Absolute Quantification")
        # dose check warnings first, then the lung density question
        self._publishDoseChecks(checks, then=(lambda: self._warnLungDose(inputs, lungRows)) if lungRows else None)

    def _warnLungDose(self, inputs, lungRows):
        """Lung doses depend on the lung density (a single assumed value): tell the user, and offer to estimate the
        density from the CT and to calculate again with it."""
        if getattr(self, "_lungWarningAcknowledged", None) == inputs["lungDensity"]:
            return  # just recalculated with the density estimated from the CT
        density = inputs["lungDensity"]
        reference = inputs["reference"]
        referenceIsCT = volumeLooksLikeCT(reference)
        doses = ", ".join(f"{row['name']} {row['dose']:.2f} Gy" for row in lungRows)
        box = qt.QMessageBox(slicer.util.mainWindow())
        box.setIcon(qt.QMessageBox.Warning)
        box.setWindowTitle("Lung dose: check the lung density")
        box.setText("Interpret the lung dose cautiously.")
        box.setInformativeText(
            f"Lung dose ({doses}) was calculated with a lung density of {density:.2f} g/mL (default "
            f"{LUNG_DENSITY_DEFAULT:.2f} g/mL). The real density of aerated lung varies (about "
            f"{LUNG_DENSITY_TYPICAL[0]:.1f}-{LUNG_DENSITY_TYPICAL[1]:.1f} g/mL, higher with fluid, fibrosis or "
            "atelectasis) and the dose is inversely proportional to it. Partial volume at the lung-liver boundary "
            "and breathing motion add further uncertainty.\n\nCheck the lung density."
            + ("" if referenceIsCT else "\n\nThe reference volume does not look like a CT, so the lung density "
                                        "cannot be estimated from it."))
        estimateButton = box.addButton("Estimate lung density from CT", qt.QMessageBox.ActionRole) \
            if referenceIsCT else None
        box.addButton("I understand", qt.QMessageBox.AcceptRole)
        box.exec_()
        if estimateButton is None or box.clickedButton() is not estimateButton:
            return
        values = []
        with slicer.util.tryWithErrorDisplay("Could not estimate the lung density.", waitCursor=True):
            huValues = slicer.util.arrayFromVolume(reference)
            for row in lungRows:
                mask = segmentMaskOnVolumeGrid(inputs["segmentation"], row["id"], reference)
                if mask.any():
                    values.append(huValues[mask])
        if not values:
            slicer.util.errorDisplay("The lung segments are not inside the CT.")
            return
        estimated = round(meanDensityFromHU(np.concatenate(values)), 2)
        if not slicer.util.confirmYesNoDisplay(
                f"Mean lung density from the CT numbers of '{reference.GetName()}': {estimated:.2f} g/mL\n"
                "(density = (CT number + 1000) / 1000, as in Kao YH et al., EJNMMI Res 2014;4:33).\n\n"
                f"Set the lung density to {estimated:.2f} g/mL and calculate again?",
                windowTitle="Lung density from CT"):
            return
        self.lungDensitySpinBox.value = estimated
        self._lungWarningAcknowledged = self.lungDensitySpinBox.value
        qt.QTimer.singleShot(0, self.onCalculateButton)

    def _finishCalculation(self, inputs, stats, voxelVolumeML, liverMask, firstRows, parameters, notes, qcNote, title):
        """Fill all result tables, store report data, then build the results layout."""
        # Segment dose table
        rows = list(firstRows)
        for s in stats:
            density = inputs.get("lungDensity") if (s["role"] == CATEGORY_LUNGS and inputs.get("lungDensity")) \
                else inputs["density"]
            rows.append((s["label"], s["roleLabel"], formatNumber(s["dose"]), formatNumber(s["volume"]),
                         formatNumber(s["volume"] * density),
                         formatNumber(s["activity"])))
        self._fillTable(self.segmentDoseTable, rows)
        self.resultNoteLabel.setText(qcNote)

        # DVH-derived tables (only segments that are actually modelled)
        modelled = [s for s in stats if s["doses"].size > 0]
        self.dvhData = [(s["label"], s["doses"], voxelVolumeML) for s in modelled]
        dRows, vRows, dLines, vLines = [], [], [], []
        for s in modelled:
            dValues = [doseAtVolumePercent(s["doses"], x) for x in D_METRICS]
            vValues = [volumePercentAtDose(s["doses"], x) for x in V_METRICS]
            dRows.append([s["label"]] + [formatNumber(v) for v in dValues])
            vRows.append([s["label"]] + [formatNumber(v) for v in vValues])
            dLines.append(f"{s['label']}: " + ", ".join(f"D{x} = {formatNumber(v)} Gy" for x, v in zip(D_METRICS, dValues)))
            vLines.append(f"{s['label']}: " + ", ".join(f"V{x} = {formatNumber(v)} %" for x, v in zip(V_METRICS, vValues)))
        self._fillTable(self.dValueTable, dRows)
        self._fillTable(self.vValueTable, vRows)

        # Custom metrics panel
        self.customSegmentList.clear()
        for label, _, _ in self.dvhData:
            item = qt.QListWidgetItem(label)
            item.setFlags(item.flags() | qt.Qt.ItemIsUserCheckable)
            item.setCheckState(qt.Qt.Checked)
            self.customSegmentList.addItem(item)
        self.customResultTable.setRowCount(0)
        self.customCollapsibleButton.setEnabled(bool(self.dvhData))

        parameters = list(parameters)
        for key, categoryTitle in SEGMENT_CATEGORIES:
            names = [s["name"] for s in stats if s["role"] == key]
            parameters.append((f"{categoryTitle} segments", ", ".join(names) if names else "none"))
        self.lastResult = {"title": title, "parameters": parameters, "rows": rows, "notes": notes,
                           "sections": {"D values": dLines, "V values": vLines, "Custom DVH metrics": []}}
        try:
            self.lastResult["patient"] = reportPatientInfo(inputs["spect"], inputs["reference"])
        except Exception as e:
            logging.warning(f"Could not read the patient information for the report: {e}")

        # Kept so that the results can be saved with the scene and the DVHs rebuilt after loading it
        combinedIDs = {"tumors": [s["id"] for s in stats if s["role"] == CATEGORY_TUMOR and s["id"]],
                       "viables": [s["id"] for s in stats if s["role"] == CATEGORY_VIABLE and s["id"]]}
        self._dvhSources = [{"label": s["label"],
                             "segmentIDs": [s["id"]] if s["id"] else combinedIDs.get(s["role"], []),
                             "voxels": int(s["doses"].size)} for s in modelled]
        self._dvhVoxelVolumeML = float(voxelVolumeML)
        self._doseChecksum = doseChecksum(slicer.util.arrayFromVolume(inputs["output"]))
        self._resultsDoseNode = inputs["output"]
        self._resultsSegmentationNode = inputs["segmentation"]
        self._annotationLabels = [(s["id"], segmentLabelText(s.get("displayName", s["name"]), s["volume"], s["dose"]))
                                  for s in stats if s["id"]]

        # Visualization: failures here must not discard the numeric results above
        with slicer.util.tryWithErrorDisplay("Dose results were calculated, but building the views failed."):
            views = self.logic.setupResultsLayout()
            self.logic.createSegmentModels(inputs["segmentation"], {s["id"]: s["role"] for s in stats if s["id"]}, views)
            self.logic.createIsodoseModels(inputs["output"], inputs["isodoseLevels"], views)
            # Segment colours (whole liver green); same-colour curves get different line patterns
            colors = [DVH_LIVER_COLOR if s["role"] == "liver" else tuple(s["color"]) for s in modelled]
            styles = dvhCurveStyles(colors, [s["id"] is None for s in modelled])
            curves = [(s.get("displayName", s["name"]), rgb, s["doses"], style, width)
                      for s, (style, width, rgb) in zip(modelled, styles)]
            self.logic.showDvhChart(self.logic.createDvhChart(curves), views)
            self.logic.showSliceViews(inputs["reference"], inputs["output"], inputs["spect"])
            self.logic.linkThreeDViewCameras([views["segments3D"], views["isodose3D"]])
            try:  # labels are cosmetic; never fail the visualization for them
                self.logic.showSegmentAnnotations(inputs["segmentation"], self._annotationLabels)
            except Exception as e:
                logging.warning(f"Could not create the segment labels: {e}")
            self.lastVisualization = {"dose": inputs["output"], "views": views}
        self._saveResults()

    # -- custom DVH metrics ------------------------------------------------


    def _saveModuleSettings(self, parameterNode):
        pass  # all settings of this module are in NODE_SETTINGS / WIDGET_SETTINGS

    def _restoreModuleSettings(self, parameterNode):
        pass

    def _clearModuleResults(self):
        self.totalActivityTextBox.setText("")
        self.dectotalActivityTextBox.setText("")

    def _previewPerfusedIDs(self):
        return set()  # no perfused volumes in absolute mode

    def _previewSegmentVolumeML(self, segmentationNode, segmentID):
        return segmentVolumeML(segmentationNode, segmentID)

class RadioembolizationDosimetryAbsoluteLogic(DosimetryLogicBase):
    """Computation and visualization, separated from the GUI so it can be tested and scripted."""

    # -- dose ----------------------------------------------------------------

    def computeAbsoluteDose(self, petVolumeNode, toMBqPerML, hoursAfterTreatment, halfLifeHours,
                            conversionFactor, densityGPerML, clipNegativeValues=True):
        """Return (doseArray [Gy], FOV activity at scan [MBq], at administration [MBq], decay factor, voxel mL,
        prepared concentration array, number of negative voxels in the original image). With clipNegativeValues,
        negative voxel values are set to 0 when the image is read."""
        if halfLifeHours <= 0:
            raise ValueError("Half-life must be positive.")
        concentration, negativeVoxels = prepareImageValues(slicer.util.arrayFromVolume(petVolumeNode),
                                                           clipNegativeValues)
        if not np.all(np.isfinite(concentration)):
            raise ValueError("The input volume contains NaN or infinite values.")
        voxelVolumeML = voxelVolumeMLFromNode(petVolumeNode)
        totalAtScanMBq = float(concentration.sum()) * voxelVolumeML * toMBqPerML
        decayFactor = 2.0 ** (hoursAfterTreatment / halfLifeHours)
        doseArray = computeAbsoluteDoseMap(concentration, toMBqPerML, decayFactor, conversionFactor, densityGPerML)
        if negativeVoxels and not clipNegativeValues:
            logging.warning(f"{negativeVoxels} voxels have negative values (reconstruction noise); "
                            "they contribute negative dose to segment means.")
        return (doseArray, totalAtScanMBq, totalAtScanMBq * decayFactor, decayFactor, voxelVolumeML,
                concentration, negativeVoxels)

    def segmentStatistics(self, doseArray, segmentationNode, referenceVolumeNode, voxelVolumeML,
                          conversionFactor, densityGPerML, liverSegmentID, categories):
        """One dict per segment, plus 'All tumors (combined)' (union of the tumor segments, id None) when
        tumors are categorized; ordered by role. Keys: id, name, displayName, label, role, roleLabel, color,
        dose, volume, activity and doses (sorted voxel doses, empty if not on the image grid)."""
        segmentation = segmentationNode.GetSegmentation()
        results = []
        tumorMask = None
        viableMask = None
        for segmentID, name in segmentList(segmentationNode):
            if categories.get(segmentID) == CATEGORY_IGNORED and segmentID != liverSegmentID:
                continue  # not calculated (e.g. the lungs)
            mask = segmentMaskOnVolumeGrid(segmentationNode, segmentID, referenceVolumeNode)
            if mask.shape != doseArray.shape:
                raise RuntimeError(f"Mask shape {mask.shape} does not match image shape {doseArray.shape}.")
            role = segmentRole(segmentID, liverSegmentID, (), categories)
            if role == CATEGORY_TUMOR:
                tumorMask = mask.copy() if tumorMask is None else (tumorMask | mask)
            elif role == CATEGORY_VIABLE:
                viableMask = mask.copy() if viableMask is None else (viableMask | mask)
            results.append(self._statisticsEntry(doseArray, mask, voxelVolumeML, conversionFactor, densityGPerML,
                                                 segmentID, name, role, tuple(segmentation.GetSegment(segmentID).GetColor())))
        if tumorMask is not None:
            results.append(self._statisticsEntry(doseArray, tumorMask, voxelVolumeML, conversionFactor, densityGPerML,
                                                 None, COMBINED_TUMORS_NAME, "tumors", COLOR_TUMOR))
        if viableMask is not None:
            results.append(self._statisticsEntry(doseArray, viableMask, voxelVolumeML, conversionFactor, densityGPerML,
                                                 None, COMBINED_VIABLE_NAME, "viables", COLOR_VIABLE))
        return sortResults(results)

    @staticmethod
    def _statisticsEntry(doseArray, mask, voxelVolumeML, conversionFactor, densityGPerML, segmentID, name, role, color):
        dose, volume, activity = segmentDoseStatistics(doseArray, mask, voxelVolumeML, conversionFactor, densityGPerML)
        modelled = volume > 0
        if not modelled:
            logging.warning(f"Segment '{name}' has no voxels on the image grid.")
        doses = np.sort(doseArray[mask]) if modelled else np.zeros(0)
        return {"id": segmentID, "name": name, "displayName": name, "label": name,
                "role": role, "roleLabel": RESULT_ROLES[role][0], "color": color,
                "dose": dose, "volume": volume, "activity": activity, "doses": doses}

class RadioembolizationDosimetryAbsoluteTest(ScriptedLoadableModuleTest):
    """Synthetic-data checks of the dose math ("Reload and Test" in developer mode)."""

    def runTest(self):
        self.test_absoluteDose()
        self.test_labelStacking()
        self.test_negativeVoxelValues()
        self.test_categories()
        self.test_segmentModelRoles()
        self.test_reportScreenshots()
        self.test_dvhMetrics()
        self.test_isodoseOpacity()

    def test_absoluteDose(self):
        conc = np.full((5, 5, 5), 1e6)  # 1 MBq/mL expressed in Bq/mL
        dose = computeAbsoluteDoseMap(conc, 1e-6, 1.0, 49.67, 1.05)
        assert np.allclose(dose, 49.67 / 1.05)
        assert np.allclose(computeAbsoluteDoseMap(conc, 1e-6, 2.0, 49.67, 1.05), 2 * dose)
        self.delayDisplay("Absolute: dose math OK")

    def test_categories(self):
        categories = {"t1": CATEGORY_TUMOR, "n1": CATEGORY_NORMAL, "liver": CATEGORY_TUMOR}
        assert segmentRole("liver", "liver", (), categories) == "liver"   # liver takes precedence
        assert segmentRole("pv", "liver", {"pv"}, categories) == "perfused"
        assert segmentRole("t1", "liver", (), categories) == CATEGORY_TUMOR
        assert segmentRole("x", "liver", (), categories) == "uncategorized"
        rows = [{"role": r} for r in ("uncategorized", CATEGORY_TUMOR, "tumors", "liver", CATEGORY_NORMAL)]
        assert [r["role"] for r in sortResults(rows)] == ["liver", "tumors", CATEGORY_TUMOR, CATEGORY_NORMAL,
                                                          "uncategorized"]
        # DVH: same colour -> different patterns; combined tumors thicker; different colours both solid
        styles = dvhCurveStyles([COLOR_TUMOR, COLOR_TUMOR, COLOR_TUMOR, COLOR_NORMAL],
                                [True, False, False, False])
        assert styles == [("Solid", 3.5, COLOR_TUMOR), ("Dash", 2.0, COLOR_TUMOR), ("DashDot", 2.0, COLOR_TUMOR),
                          ("Solid", 2.0, COLOR_NORMAL)]
        fifth = dvhCurveStyles([COLOR_OTHER] * 6, [False] * 6)[4]  # patterns repeat in another shade
        assert fifth[:2] == ("Solid", 2.0) and fifth[2] == shadeColor(COLOR_OTHER, 1) and fifth[2] != COLOR_OTHER
        self.delayDisplay("Segment categories OK")

    def test_reportScreenshots(self):
        import struct
        import zlib
        # 3 x 2 px PNG
        raw = b"".join(b"\x00" + b"\xff\x00\x00" * 3 for _ in range(2))
        def chunk(kind, data):
            return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
        png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 3, 2, 8, 2, 0, 0, 0))
               + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
        assert pngSize(png) == (3, 2)
        rtf = buildRtfReport(["T"], [], [], screenshots=[("Isodose view", png)])
        assert "\\pngblip\\picw3\\pich2\\picwgoal45\\pichgoal30" in rtf
        assert png.hex()[:64] in rtf.replace("\n", "")
        assert rtf.index("Screenshots") < rtf.index("End of Report")
        rtf.encode("ascii")  # still pure ASCII
        self.delayDisplay("Report screenshots OK")

    def test_segmentModelRoles(self):
        assert SEGMENT_MODEL_OPACITY == {"liver": 0.03, "perfused": 0.05, CATEGORY_TUMOR: 0.10,
                                         CATEGORY_VIABLE: 0.10}
        for role in (CATEGORY_NORMAL, CATEGORY_OTHER, "uncategorized", "tumors", "viables", CATEGORY_IGNORED):
            assert role not in SEGMENT_MODEL_OPACITY  # not rendered in the 3D view
        self.delayDisplay("3D segment model roles OK")

    def test_negativeVoxelValues(self):
        raw = np.array([[-3, 0, 5], [2, -1, 4]], dtype=np.int16)
        values, negatives = prepareImageValues(raw, True)
        assert negatives == 2 and values.dtype == np.float64
        assert values.tolist() == [[0.0, 0.0, 5.0], [2.0, 0.0, 4.0]]
        assert raw[0, 0] == -3                                   # input volume array is never modified
        kept, negatives = prepareImageValues(raw, False)
        assert negatives == 2 and kept.min() == -3.0
        assert "set to 0" in negativeVoxelNote(2, 6, True)
        assert "used as is" in negativeVoxelNote(2, 6, False)
        assert negativeVoxelNote(0, 6, True).startswith("No negative")
        self.delayDisplay("Negative voxel handling OK")

    def test_labelStacking(self):
        labels = [{"anchorY": 100.0, "h": 20.0}, {"anchorY": 105.0, "h": 20.0}, {"anchorY": 5.0, "h": 20.0}]
        stackLabelCentres(labels, 8.0, 300.0, 4.0)
        ys = sorted(label["y"] for label in labels)
        assert all(b - a >= 24.0 - 1e-9 for a, b in zip(ys, ys[1:]))  # no overlap
        assert ys[0] >= 18.0 - 1e-9                                    # inside the bottom margin
        self.delayDisplay("Label stacking OK")

    def test_isodoseOpacity(self):
        assert [isodoseOpacity(i, 8) for i in range(8)] == [0.05, 0.05, 0.1, 0.1, 0.1, 0.15, 0.15, 0.15]
        self.delayDisplay("Isodose opacity OK")

    def test_dvhMetrics(self):
        doses = np.arange(1.0, 101.0)  # 100 voxels, 1..100 Gy
        assert doseAtVolumePercent(doses, 90) == 11.0       # 90 voxels receive >= 11 Gy
        assert doseAtVolumePercent(doses, 50) == 51.0
        assert volumePercentAtDose(doses, 30) == 71.0       # 30..100 Gy -> 71 voxels
        dvh = cumulativeDvh(doses, np.array([0.0, 50.0, 101.0]))
        assert list(dvh) == [100.0, 51.0, 0.0]
        self.delayDisplay("DVH metrics OK")
