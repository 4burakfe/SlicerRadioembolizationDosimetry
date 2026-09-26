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


# The shared segment categories (TaranisLib.dosimetry) with the box titles of this module: lungs are not calculated
# in patient-relative mode (their dose comes from the lung shunt)
SEGMENT_CATEGORIES = [(CATEGORY_TUMOR, "Tumors"), (CATEGORY_VIABLE, "Viable tumors"),
                      (CATEGORY_NORMAL, "Normal tissue"), (CATEGORY_LUNGS, "Lungs (not calculated)"), (CATEGORY_OTHER, "Others"),
                      (CATEGORY_IGNORED, "Ignored (not calculated)")]


RELATIVE_MODE_WARNING = (
    "\u26a0 Patient-relative mode: the activity of each perfused volume (minus lung shunt) is distributed ONLY "
    "inside that perfused volume (which must lie inside the whole-liver segment), in proportion to image counts. "
    "Liver outside the perfused volumes and extrahepatic uptake (e.g. stomach, duodenum, gallbladder) are IGNORED "
    "and set to 0 Gy. Segments outside the perfused volumes are flagged in the tables.")


# The note under the results suggests adjusting the perfused volumes if more of the whole-liver counts than this lie
# outside all perfused volumes; above doseguard.OUTSIDE_PERFUSED_FRACTION (20 %) a dose check warns
OUTSIDE_PERFUSED_WARNING_FRACTION = 0.10

# ---------------------------------------------------------------------------
# Pure computation helpers (numpy only, no Slicer dependency -> unit-testable)
# ---------------------------------------------------------------------------

def computeRelativeDoseMap(counts, regionMask, regionActivityMBq, voxelVolumeML,
                           conversionFactor, densityGPerML, regionName="the segment"):
    """Distribute regionActivityMBq over regionMask in proportion to image counts.

    dose_i [Gy] = A_region * (c_i / sum_region(c)) * CF / (rho * V_voxel)
    A in MBq, CF in J/GBq (= Gy*g/MBq), rho in g/mL, V_voxel in mL. Outside the region: 0 Gy.
    """
    if voxelVolumeML <= 0 or densityGPerML <= 0:
        raise ValueError("Voxel volume and tissue density must be positive.")
    regionCounts = float(np.sum(counts[regionMask], dtype=np.float64))
    if not np.isfinite(regionCounts) or regionCounts <= 0:
        raise ValueError(
            f"Total counts inside {regionName} are zero or negative ({regionCounts:g}).\n"
            "Check that the segment overlaps the SPECT/PET volume.")
    scale = regionActivityMBq * conversionFactor / (densityGPerML * voxelVolumeML * regionCounts)
    dose = np.zeros(counts.shape, dtype=np.float64)
    dose[regionMask] = counts[regionMask] * scale
    return dose


def findMaskOverlaps(masks):
    """[(i, j, voxelCount)] for every pair of masks (i < j) that share at least one voxel."""
    overlaps = []
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            n = int(np.count_nonzero(masks[i] & masks[j]))
            if n:
                overlaps.append((i, j, n))
    return overlaps


def containmentProblems(namedMasks, containerMask, voxelVolumeML):
    """[(name, outside mL, outside fraction)] for every mask with voxels outside containerMask."""
    problems = []
    for name, mask in namedMasks:
        total = int(np.count_nonzero(mask))
        outside = int(np.count_nonzero(mask & ~containerMask))
        if total and outside:
            problems.append((name, outside * voxelVolumeML, outside / total))
    return problems


def computeMultiTerritoryDoseMap(counts, territoryMasks, territoryActivitiesMBq, voxelVolumeML,
                                 conversionFactor, densityGPerML, territoryNames=None):
    """One relative dose map per perfused volume (territory), then summed.

    Each territory's activity is distributed only inside its own mask, in proportion to the counts there,
    so the relative uptake between territories does not matter. Territories must not overlap.
    """
    if len(territoryMasks) == 0:
        raise ValueError("At least one perfused volume is required.")
    if len(territoryMasks) != len(territoryActivitiesMBq):
        raise ValueError("Each perfused volume needs exactly one activity.")
    names = territoryNames or [f"perfused volume {k + 1}" for k in range(len(territoryMasks))]
    overlaps = findMaskOverlaps(territoryMasks)
    if overlaps:
        raise ValueError("Perfused volumes must not intersect: " + "; ".join(
            f"{names[i]} / {names[j]}: {n} voxels" for i, j, n in overlaps))
    total = np.zeros(counts.shape, dtype=np.float64)
    for mask, activityMBq, name in zip(territoryMasks, territoryActivitiesMBq, names):
        total += computeRelativeDoseMap(counts, mask, activityMBq, voxelVolumeML,
                                        conversionFactor, densityGPerML, regionName=name)
    return total


def singleCompartmentDoseGy(activityMBq, lungShuntFraction, conversionFactor, densityGPerML, volumeML):
    """Mean absorbed dose of a perfused volume, single-compartment (MIRD) model:
    D [Gy] = A [MBq] * (1 - LSF) * CF [Gy*g/MBq] / (rho [g/mL] * V [mL]). NaN for a non-positive volume."""
    if volumeML is None or volumeML <= 0 or densityGPerML <= 0:
        return float("nan")
    return activityMBq * (1.0 - lungShuntFraction) * conversionFactor / (densityGPerML * volumeML)


def isodoseColorForDose(doseGy, levels):
    """Colour of the highest isodose level that doseGy reaches; None below the lowest level."""
    color = None
    if doseGy is None or not np.isfinite(doseGy):
        return None
    for level, (_, rgb) in zip(levels, ISODOSE_COLORS):
        if doseGy >= level:
            color = rgb
    return color


def doseEstimateStyleSheet(rgb):
    """Label background in the isodose colour, with black or white text for contrast."""
    if rgb is None:
        return "padding: 3px; border-radius: 3px; border: 1px solid #888888;"
    luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    textColor = "#000000" if luminance > 0.5 else "#ffffff"
    return (f"background-color: {rgbToHex(rgb)}; color: {textColor}; padding: 3px; "
            "border-radius: 3px; font-weight: bold;")


def territoryTag(territoryNumbers):
    """' (Perfused volume 2)' or ' (Perfused volumes 1, 3)'; empty string without numbers."""
    numbers = list(territoryNumbers or ())
    if not numbers:
        return ""
    if len(numbers) == 1:
        return f" (Perfused volume {numbers[0]})"
    return " (Perfused volumes " + ", ".join(str(n) for n in numbers) + ")"


def decorateSegmentName(name, territoryNumbers=(), outsideFraction=None):
    """Segment name + perfused volume number(s) + a flag for the part lying outside all perfused volumes."""
    label = name + territoryTag(territoryNumbers)
    if outsideFraction is None or outsideFraction <= 0:
        return label
    if outsideFraction >= 1.0:
        return f"{label} [outside perfused volumes - not modelled]"
    return f"{label} [{100 * outsideFraction:.1f}% outside perfused volumes]"


# ---------------------------------------------------------------------------
# Slicer helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------

class RadioembolizationDosimetryRelative(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "Taranis - Radioembolization Dosimetry - Patient Relative "
        parent.categories = ["Nuclear Medicine"]
        parent.dependencies = []
        parent.contributors = ["Burak Demir, MD, FEBNM"]
        parent.helpText = """
        Predictive (patient-relative) radioembolization dosimetry, e.g. from Tc-99m MAA SPECT.<br>
        One or more non-intersecting perfused volumes are defined; the activity chosen for each perfused volume
        (minus lung shunt) is distributed inside it in proportion to image counts, and the dose maps of all
        perfused volumes are added (extrahepatic uptake is ignored).<br>
        For post-treatment dosimetry from a quantitative image use the "Taranis - Absolute Dosimetry" module.<br>
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
        sampleName="RadioembolizationDosimetry2",
        thumbnailFileName=os.path.join(iconsPath, "RadioembolizationDosimetry2.png"),
        uris=["https://github.com/4burakfe/SlicerRadioembolizationDosimetry_SampleImages/releases/download/TestImages/patient_2_CT.nrrd",
              "https://github.com/4burakfe/SlicerRadioembolizationDosimetry_SampleImages/releases/download/TestImages/patient_2_SPECT.nrrd",
              "https://github.com/4burakfe/SlicerRadioembolizationDosimetry_SampleImages/releases/download/TestImages/patient_2_Segmentation.seg.nrrd"],
        fileNames=["patient_2_CT.nrrd", "patient_2_SPECT.nrrd", "patient_2_Segmentation.seg.nrrd"],
        checksums=["SHA256:99600e480dc6e5353953377dbe66f232d8eae28bfa50aa7a61a4f83574ccde47",
                   "SHA256:5647f8109babdc8655b217da6952cd5aafde3df2598a0dab52b2ad60208dc86b",
                   "SHA256:49e0eae32ad99464a66ae1fab48a9343d9068b5633e5a6f260640bc9b5666abe"],
        nodeNames=["Patient 2 CT", "Patient 2 SPECT", "Patient 2 Segmentation"],
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


class RadioembolizationDosimetryRelativeWidget(DosimetryWidgetBase):

    WIDGET_SETTINGS = [
        ("ClipNegativeValues", "clipNegativeCheckBox", "bool"),
        ("LungShuntPercent", "lungShuntSlider", "number"),
        ("LungMassG", "lungMassSpinBox", "number"),
        ("ConversionFactor", "conversionFactorSpinBox", "number"),
        ("LiverDensity", "liverDensitySpinBox", "number"),
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
        self.logic = RadioembolizationDosimetryRelativeLogic()
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
        self._segmentVolumeCache = {}   # (segmentation node ID, segment ID) -> volume mL (dose estimates)
        self._segmentationObservations = []
        self._segmentEditTimer = qt.QTimer()  # coalesces segment edits before the estimates are refreshed
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
        modeBox = qt.QGroupBox("Patient relative (predictive, e.g. Tc-99m MAA SPECT)")
        modeLayout = qt.QVBoxLayout(modeBox)
        self.modeInfoLabel = qt.QLabel(RELATIVE_MODE_WARNING)
        self.modeInfoLabel.setWordWrap(True)
        self.modeInfoLabel.setStyleSheet("color: #c46a00; font-weight: bold;")

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

        # ---- Images and whole liver ----
        inputBox = qt.QGroupBox("Images and segments")
        inputLayout = qt.QFormLayout(inputBox)
        parametersLayout.addWidget(inputBox)

        self.spectSelector = self._makeNodeSelector("vtkMRMLScalarVolumeNode", "Select the input SPECT/PET volume.")
        inputLayout.addRow("Input SPECT/PET Volume: ", self.spectSelector)
        self.clipNegativeCheckBox = qt.QCheckBox("Set negative voxel values to 0")
        self.clipNegativeCheckBox.setChecked(True)
        self.clipNegativeCheckBox.setToolTip(
            "Set negative voxel values (reconstruction noise) to 0 when the image is read,\n"
            "so the dose map, totals, QC fractions and statistics are all calculated from the same values.\n"
            "Unticked: negative values are used as is and produce negative voxel doses.\n"
            "The number of negative voxels is reported in the results either way.")
        inputLayout.addRow("", self.clipNegativeCheckBox)

        self.referenceSelector = self._makeNodeSelector(
            "vtkMRMLScalarVolumeNode", "Anatomical reference (CT or MRI) shown under the isodose lines.")
        inputLayout.addRow("Reference Volume (CT/MRI): ", self.referenceSelector)

        self.segmentationSelector = self._makeNodeSelector(
            "vtkMRMLSegmentationNode",
            "Master segmentation for dosimetric calculations. All segment selections below use it.")
        inputLayout.addRow("Master Segmentation: ", self.segmentationSelector)

        self.liverSegmentSelector = SegmentComboBox(
            "Segment representing the whole liver. Perfused volumes and tumors must lie entirely inside it; "
            "voxels outside it receive 0 Gy.", onChanged=self._refreshCategorizer)
        inputLayout.addRow("Whole Liver Segment: ", self.liverSegmentSelector.widget)

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

        # ---- Perfused volumes (one row per territory; rows 2+ can be removed) ----
        perfusedBox = qt.QGroupBox("Perfused volumes")
        perfusedBoxLayout = qt.QVBoxLayout(perfusedBox)
        parametersLayout.addWidget(perfusedBox)
        perfusedInfoLabel = qt.QLabel("Select at least one perfused volume (vascular territory of an injection "
                                      "position). Perfused volumes must not intersect each other and must lie "
                                      "entirely inside the whole liver. Tumors must also lie inside the whole liver.")
        perfusedInfoLabel.setWordWrap(True)
        perfusedBoxLayout.addWidget(perfusedInfoLabel)
        self.perfusedRowsLayout = qt.QVBoxLayout()
        perfusedBoxLayout.addLayout(self.perfusedRowsLayout)
        self.addPerfusedVolumeButton = qt.QPushButton("Add perfused volume")
        self.addPerfusedVolumeButton.toolTip = "Add another perfused volume with its own activity."
        perfusedBoxLayout.addWidget(self.addPerfusedVolumeButton)

        # ---- Segment categories (segments other than the whole liver and the perfused volumes) ----
        categoryBox = qt.QGroupBox("Segment categories")
        categoryBoxLayout = qt.QVBoxLayout(categoryBox)
        parametersLayout.addWidget(categoryBox)
        categoryInfoLabel = qt.QLabel(
            "Select segments on the left and move them into a category with > (back with <). "
            "Colours after calculation: whole liver white, perfused volumes bright red, tumors pink, "
            "normal tissue turquoise, others and uncategorized gray. Categories are saved with the segmentation.")
        categoryInfoLabel.setWordWrap(True)
        categoryBoxLayout.addWidget(categoryInfoLabel)
        self.categorizer = SegmentCategorizer(onChanged=self._onCategoriesChanged, categories=SEGMENT_CATEGORIES)
        categoryBoxLayout.addWidget(self.categorizer.widget)

        # ---- Lung shunt (one set for the whole treatment) ----
        lungBox = qt.QGroupBox("Lung shunt")
        lungLayout = qt.QFormLayout(lungBox)
        parametersLayout.addWidget(lungBox)

        self.lungShuntSlider = ctk.ctkSliderWidget()
        self.lungShuntSlider.singleStep = 1
        self.lungShuntSlider.minimum = 0
        self.lungShuntSlider.maximum = 99
        self.lungShuntSlider.value = 0
        self.lungShuntSlider.setToolTip("Lung shunt fraction as a percentage. Applied to the activity of every "
                                        "perfused volume.")
        lungLayout.addRow("Lung Shunt Fraction (%): ", self.lungShuntSlider)

        self.lungMassSpinBox = qt.QDoubleSpinBox()
        self.lungMassSpinBox.setRange(1.0, 5000.0)
        self.lungMassSpinBox.setValue(1000.0)
        self.lungMassSpinBox.setSingleStep(50.0)
        lungLayout.addRow("Lung Mass (g):", self.lungMassSpinBox)

        # ---- Activities (one slider per perfused volume) ----
        activityBox = qt.QGroupBox("Administered activities")
        activityBoxLayout = qt.QVBoxLayout(activityBox)
        parametersLayout.addWidget(activityBox)
        self.activityRowsLayout = qt.QVBoxLayout()
        activityBoxLayout.addLayout(self.activityRowsLayout)
        self.totalActivityLabel = qt.QLabel()
        activityBoxLayout.addWidget(self.totalActivityLabel)

        # ---- Calculation and display settings ----
        generalBox = qt.QGroupBox("Calculation and display settings")
        generalLayout = qt.QFormLayout(generalBox)
        parametersLayout.addWidget(generalBox)

        self.conversionFactorSpinBox = qt.QDoubleSpinBox()
        self.conversionFactorSpinBox.setRange(0.01, 100.0)
        self.conversionFactorSpinBox.setValue(49.67)
        self.conversionFactorSpinBox.setSingleStep(0.1)
        self.conversionFactorSpinBox.setToolTip("Energy deposited per unit activity, J/GBq (= Gy*kg/GBq = Gy*g/MBq).")
        generalLayout.addRow("Conversion Factor (J/GBq):", self.conversionFactorSpinBox)

        self.liverDensitySpinBox = qt.QDoubleSpinBox()
        self.liverDensitySpinBox.setRange(0.01, 10.0)
        self.liverDensitySpinBox.setValue(1.05)
        self.liverDensitySpinBox.setSingleStep(0.01)
        generalLayout.addRow("Liver Density (g/mL):", self.liverDensitySpinBox)

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

        self.calculateButton = qt.QPushButton("Calculate with the desired activities")
        self.calculateButton.toolTip = "Perform dosimetric calculations and show the results layout."
        parametersLayout.addWidget(self.calculateButton)

        # First perfused volume row is always present (it has no Remove button)
        self.perfusedRows = []  # [{"widget", "numberLabel", "selector", "activityWidget", "activityLabel", "activitySlider"}]
        self.addPerfusedVolumeRow()

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
        self.resultNoteLabel.setStyleSheet("color: #c46a00;")
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
        self.addPerfusedVolumeButton.connect("clicked(bool)", self.addPerfusedVolumeRow)
        self.lungShuntSlider.connect("valueChanged(double)", self._updateDoseEstimates)
        self.conversionFactorSpinBox.connect("valueChanged(double)", self._updateDoseEstimates)
        self.liverDensitySpinBox.connect("valueChanged(double)", self._updateDoseEstimates)
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
            "This module enables predictive (patient-relative) dosimetry with SPECT and PET images.\n"
            "This module is NOT a medical device. It is for research purposes only.\n"
            "Default conversion factor is for Y-90 which equals to 49.67 J/GBq\n"
            "Conversion factor for Ho-166 is 14.85 J/GBq (half-life ~26.8 h)\n"
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

    # -- selector handling ------------------------------------------------


    def onSegmentationNodeChanged(self, node):
        self.liverSegmentSelector.setSegmentation(node)
        for row in self.perfusedRows:
            row["selector"].setSegmentation(node)
        self._observeSegmentation(node)
        self.categorizer.categories = loadSegmentCategories(node)
        self._updatePerfusedRowLabels()  # also refreshes the categorizer

    # -- segment categories ------------------------------------------------

    def _perfusedSegmentIDs(self):
        return {row["selector"].currentSegmentID() for row in self.perfusedRows} - {""}

    def _categoryExclusions(self):
        """Segments not offered for categorization: the whole liver and the perfused volumes."""
        return ({self.liverSegmentSelector.currentSegmentID()} | self._perfusedSegmentIDs()) - {""}

    # -- perfused volume rows ---------------------------------------------

    def addPerfusedVolumeRow(self, *args):
        """Add a perfused volume selector and its activity slider. Rows 2+ get a Remove button."""
        isFirst = not self.perfusedRows
        row = {}

        rowWidget = qt.QWidget()
        rowLayout = qt.QHBoxLayout(rowWidget)
        rowLayout.setContentsMargins(0, 0, 0, 0)
        row["numberLabel"] = qt.QLabel()
        rowLayout.addWidget(row["numberLabel"])
        selector = SegmentComboBox("Segment representing this perfused volume (vascular territory).",
                                   onChanged=self._updatePerfusedRowLabels)
        selector.setSegmentation(self.segmentationSelector.currentNode())
        rowLayout.addWidget(selector.widget, 1)
        row["selector"] = selector
        if not isFirst:
            removeButton = qt.QPushButton("Remove")
            removeButton.setToolTip("Remove this perfused volume and its activity.")
            removeButton.connect("clicked()", lambda *_, r=row: self.removePerfusedVolumeRow(r))
            rowLayout.addWidget(removeButton)
        row["widget"] = rowWidget
        self.perfusedRowsLayout.addWidget(rowWidget)

        activityWidget = qt.QWidget()
        activityLayout = qt.QVBoxLayout(activityWidget)
        activityLayout.setContentsMargins(0, 0, 0, 6)
        row["activityLabel"] = qt.QLabel()
        activityLayout.addWidget(row["activityLabel"])
        slider = ctk.ctkSliderWidget()
        slider.singleStep = 1
        slider.minimum = 0
        slider.maximum = 10000
        slider.value = 0
        slider.setToolTip("Administered activity for this perfused volume in MBq (before lung shunt).")
        slider.connect("valueChanged(double)", self._updateTotalActivity)
        slider.connect("valueChanged(double)", self._updateDoseEstimates)
        slider.connect("valueChanged(double)", self._saveSettings)
        activityLayout.addWidget(slider)
        row["activitySlider"] = slider
        row["doseEstimateLabel"] = qt.QLabel()
        row["doseEstimateLabel"].setWordWrap(True)
        activityLayout.addWidget(row["doseEstimateLabel"])
        row["activityWidget"] = activityWidget
        self.activityRowsLayout.addWidget(activityWidget)

        self.perfusedRows.append(row)
        self._updatePerfusedRowLabels()
        self._updateTotalActivity()

    def removePerfusedVolumeRow(self, row):
        index = next((i for i, r in enumerate(self.perfusedRows) if r is row), None)
        if not index:  # unknown row, or the first row (always kept)
            return
        del self.perfusedRows[index]
        for key, layout in (("widget", self.perfusedRowsLayout), ("activityWidget", self.activityRowsLayout)):
            widget = row[key]
            layout.removeWidget(widget)
            widget.hide()
            widget.deleteLater()
        self._updatePerfusedRowLabels()  # later rows are renumbered
        self._updateTotalActivity()

    @staticmethod
    def _selectedSegmentName(selector):
        node = selector.currentNode()
        segmentID = selector.currentSegmentID()
        if node and segmentID:
            segment = node.GetSegmentation().GetSegment(segmentID)
            if segment:
                return segment.GetName()
        return None

    def _updatePerfusedRowLabels(self, *args):
        for number, row in enumerate(self.perfusedRows, start=1):
            name = self._selectedSegmentName(row["selector"])
            row["numberLabel"].setText(f"Perfused volume {number}: ")
            row["activityLabel"].setText(
                f"Perfused volume {number} ({name if name else 'no segment selected'}) - activity (MBq):")
        self._refreshCategorizer()  # perfused volumes are not offered for categorization
        self._updateDoseEstimates()
        self._saveSettings()  # rows added/removed or segments selected
        self._schedulePreview(segments=True)

    def _cachedSegmentVolumeML(self, segmentationNode, segmentID):
        key = (segmentationNode.GetID(), segmentID)
        if key not in self._segmentVolumeCache:
            try:
                self._segmentVolumeCache[key] = segmentVolumeML(segmentationNode, segmentID)
            except Exception as e:
                logging.warning(f"Could not compute the volume of segment '{segmentID}': {e}")
                self._segmentVolumeCache[key] = None
        return self._segmentVolumeCache[key]

    def _updateDoseEstimates(self, *args):
        """Single-compartment dose estimate under each activity slider, coloured like the isodose level
        the dose reaches (current isodose set)."""
        lungShuntFraction = self.lungShuntSlider.value / 100.0
        conversionFactor = self.conversionFactorSpinBox.value
        density = self.liverDensitySpinBox.value
        _, levels = self.currentIsodosePreset()
        for number, row in enumerate(self.perfusedRows, start=1):
            label = row["doseEstimateLabel"]
            selector = row["selector"]
            segmentationNode = selector.currentNode()
            segmentID = selector.currentSegmentID()
            name = self._selectedSegmentName(selector)
            activityMBq = row["activitySlider"].value
            volumeML = (self._cachedSegmentVolumeML(segmentationNode, segmentID)
                        if segmentationNode and segmentID else None)
            if not volumeML:
                label.setText(f"Select a segment for perfused volume {number} to estimate its absorbed dose."
                              if not segmentID else f"Perfused volume {number} ({name}) is empty.")
                label.setStyleSheet(doseEstimateStyleSheet(None))
                label.setToolTip("")
                continue
            doseGy = singleCompartmentDoseGy(activityMBq, lungShuntFraction, conversionFactor, density, volumeML)
            label.setText(f"{activityMBq:.2f} MBq will result in {doseGy:.1f} Gy absorbed dose "
                          f"for perfused volume {number} ({name}).")
            label.setStyleSheet(doseEstimateStyleSheet(isodoseColorForDose(doseGy, levels)))
            label.setToolTip(
                "Single-compartment estimate: D = A x (1 - LSF) x CF / (density x V)\n"
                f"A = {activityMBq:.2f} MBq, LSF = {100 * lungShuntFraction:.1f} %, CF = {conversionFactor:.2f} J/GBq, "
                f"density = {density:.2f} g/mL, V = {volumeML:.1f} mL (whole segment)\n"
                "Equals the mean dose of the perfused volume in the voxel-based calculation when the segment "
                "lies inside the whole-liver segment. Colour: isodose level reached (current isodose set).")

    # -- segmentation edits (keep the estimates current) --------------------

    def _observeSegmentation(self, segmentationNode):
        for observedObject, tag in self._segmentationObservations:
            observedObject.RemoveObserver(tag)
        self._segmentationObservations = []
        self._segmentVolumeCache = {}
        if segmentationNode is None:
            return
        segmentation = segmentationNode.GetSegmentation()
        eventNames = ("SegmentAdded", "SegmentModified", "SegmentRemoved", "SegmentsOrderModified", "RepresentationModified",
                      "SourceRepresentationModified", "MasterRepresentationModified")  # names differ by Slicer version
        events = {getattr(slicer.vtkSegmentation, n) for n in eventNames if hasattr(slicer.vtkSegmentation, n)}
        for event in events:
            tag = segmentation.AddObserver(event, lambda caller, eventId: self._onSegmentationEvent())
            self._segmentationObservations.append((segmentation, tag))

    def _onSegmentEditsSettled(self):
        self._segmentVolumeCache = {}
        node = self.segmentationSelector.currentNode()
        self.liverSegmentSelector.refresh()
        for row in self.perfusedRows:
            row["selector"].refresh()
        existing = {sid for sid, _ in segmentList(node)}
        self.categorizer.categories = {sid: c for sid, c in self.categorizer.categories.items() if sid in existing}
        self._updatePerfusedRowLabels()  # names, categorizer and estimates

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
                   if category in (CATEGORY_IGNORED, CATEGORY_LUNGS)}
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


    def _updateTotalActivity(self, *args):
        total = sum(row["activitySlider"].value for row in self.perfusedRows)
        self.totalActivityLabel.setText(f"Total administered activity: {total:.2f} MBq")

    def onIsodosePresetChanged(self, index):
        self._updateIsodoseLegend()
        self._updateDoseEstimates()  # colours follow the isodose set
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

        perfused = []
        firstUse = {}  # segment ID -> perfused volume number that selected it first
        for number, row in enumerate(self.perfusedRows, start=1):
            selector = row["selector"]
            segmentID = selector.currentSegmentID()
            activityMBq = row["activitySlider"].value
            if not segmentID:
                problems.append(f"Select a segment for perfused volume {number}.")
                continue
            if segmentID in firstUse:
                problems.append(f"Perfused volume {number} is the same segment as perfused volume "
                                f"{firstUse[segmentID]}. Perfused volumes must not intersect.")
            firstUse.setdefault(segmentID, number)
            if activityMBq <= 0:
                problems.append(f"Enter an activity greater than 0 MBq for perfused volume {number}.")
            perfused.append({"number": number, "segmentID": segmentID,
                             "name": self._selectedSegmentName(selector), "activityMBq": activityMBq})
        if not self.perfusedRows:
            problems.append("Select at least one perfused volume.")

        if problems:
            slicer.util.errorDisplay("\n".join(problems))
            return None

        presetName, levels = self.currentIsodosePreset()
        if not output:  # no output selected: create one and select it, so later runs reuse it
            output = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "Dose Map (Gy)")
            self.outputVolumeSelector.setCurrentNode(output)
        return {
            "spect": spect, "reference": reference, "segmentation": seg, "output": output,
            "liverID": liverID, "perfused": perfused, "clipNegative": self.clipNegativeCheckBox.checked,
            "categories": {sid: c for sid, c in self.categorizer.categories.items()
                           if sid != liverID and sid not in firstUse and sid in {s for s, _ in segmentList(seg)}},
            "lungShuntPercent": self.lungShuntSlider.value,
            "lungMass": self.lungMassSpinBox.value,
            "conversionFactor": self.conversionFactorSpinBox.value,
            "density": self.liverDensitySpinBox.value,
            "isodosePreset": presetName, "isodoseLevels": levels,
        }

    # -- button handlers ---------------------------------------------------

    def _calculate(self):
        self._updatePerfusedRowLabels()  # segment names may have been edited since selection
        inputs = self._collectInputs()
        if inputs is None:
            return
        result = None
        with slicer.util.tryWithErrorDisplay("Dose calculation failed.", waitCursor=True):
            result = self.logic.computeRelativeDose(
                inputs["spect"], inputs["segmentation"], inputs["liverID"], inputs["perfused"],
                inputs["lungShuntPercent"], inputs["conversionFactor"], inputs["density"], inputs["lungMass"],
                clipNegativeValues=inputs["clipNegative"],
                tumorSegmentIDs=[sid for sid, c in inputs["categories"].items()
                                 if c in (CATEGORY_TUMOR, CATEGORY_VIABLE)],
                lungSegmentIDs=[sid for sid, c in inputs["categories"].items() if c == CATEGORY_LUNGS])
        if result is None:
            return
        # Counts outside the perfused volumes: advisory only, among the dose checks shown after the calculation
        with slicer.util.tryWithErrorDisplay("Dose calculation failed.", waitCursor=True):
            self._showRelativeResults(inputs, result)

    # -- calculations ------------------------------------------------------

    def _showRelativeResults(self, inputs, result):
        writeDoseVolume(inputs["output"], inputs["spect"], result["dose"])
        inputs["output"].SetName("Dose Map (Gy)")

        applySegmentColors(inputs["segmentation"], inputs["liverID"], {pv["segmentID"] for pv in inputs["perfused"]},
                           inputs["categories"])
        stats = self.logic.segmentStatistics(
            result["dose"], inputs["segmentation"], inputs["spect"], result["voxelVolumeML"],
            inputs["conversionFactor"], inputs["density"], result["dosedMask"], result["territoryMasks"],
            inputs["liverID"], inputs["categories"])
        firstRows = [("Estimated Lung Dose", "Lung", formatNumber(result["lungDoseGy"]), "",
                      formatNumber(inputs["lungMass"]), "")]

        qcLines = []
        if np.isfinite(result["countsOutsideLiver"]):
            qcLines.append(f"{100 * result['countsOutsideLiver']:.1f}% of the image counts lie outside the "
                           "whole-liver segment and were not included in the dose calculation.")
        outsidePerfused = result["liverCountsOutsidePerfused"]
        if np.isfinite(outsidePerfused) and outsidePerfused > 0:
            line = (f"{100 * outsidePerfused:.1f}% of the counts inside the whole-liver segment lie outside the "
                    "perfused volumes (0 Gy).")
            if outsidePerfused > OUTSIDE_PERFUSED_WARNING_FRACTION:
                line += " Consider adjusting the perfused volume segments if perfused tissue was left out."
            qcLines.append(line)
        negativeNote = negativeVoxelNote(result["negativeVoxels"], result["voxelCount"],
                                         result["clippedNegativeValues"],
                                         f"; {result['negativeVoxelsInLiver']} inside the whole liver")
        if result["negativeVoxels"]:
            qcLines.append(negativeNote)
        notes = [RELATIVE_MODE_WARNING.replace("\u26a0 ", "WARNING - ")] + qcLines
        if not result["negativeVoxels"]:
            notes.append(negativeNote)

        segmentation = inputs["segmentation"].GetSegmentation()
        lsf = inputs["lungShuntPercent"] / 100.0
        parameters = [
            ("Mode", "Patient relative"),
            ("Input volume", inputs["spect"].GetName()),
            ("Reference volume", inputs["reference"].GetName()),
            ("Segmentation", inputs["segmentation"].GetName()),
            ("Whole liver segment", segmentation.GetSegment(inputs["liverID"]).GetName()),
            ("Negative voxel values", "set to 0" if result["clippedNegativeValues"] else "used as is"),
            ("Number of perfused volumes", str(len(result["territories"]))),
        ]
        for t in result["territories"]:
            parameters.append((f"Perfused volume {t['number']}",
                               f"{t['name']}: {t['activityMBq']:.2f} MBq administered, "
                               f"{t['activityMBq'] * (1.0 - lsf):.2f} MBq after lung shunt"))
        parameters += [
            ("Total administered activity", f"{result['totalActivityMBq']:.2f} MBq"),
            ("Lung Shunt", f"{inputs['lungShuntPercent']:.2f}%"),
            ("Conversion Factor", f"{inputs['conversionFactor']:.2f} J/GBq"),
            ("Lung Mass", f"{inputs['lungMass']:.2f} g"),
            ("Liver Density", f"{inputs['density']:.2f} g/mL"),
            ("Isodose set", inputs["isodosePreset"]),
        ]

        checks = DG.doseChecks(
            doseCheckSegments(stats, inputs["segmentation"]),
            microspheres=DG.microspheresFromText(inputs["isodosePreset"]),
            lsfPercent=inputs["lungShuntPercent"], lungDosesGy=[("Estimated lung dose", result["lungDoseGy"])],
            extraUptakeFraction=result["countsOutsideLiverAndLungs"], lungsSegmented=result["lungsSegmented"],
            outsidePerfusedFraction=outsidePerfused, relative=True)
        notes, qcNote = self._withDoseChecks(notes, "\n".join(qcLines), checks)
        self._finishCalculation(inputs, stats, result["voxelVolumeML"], firstRows, parameters, notes,
                                qcNote, "Taranis - Patient Relative Quantification")
        self._publishDoseChecks(checks)

    def _finishCalculation(self, inputs, stats, voxelVolumeML, firstRows, parameters, notes, qcNote, title):
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
        self.resultNoteLabel.setStyleSheet("color: #c46a00;")

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
        parameterNode.SetParameter("PerfusedVolumes", json.dumps(
            [{"segmentID": row["selector"].currentSegmentID(), "activityMBq": float(row["activitySlider"].value)}
             for row in self.perfusedRows]))

    def _restoreModuleSettings(self, parameterNode):
        """Perfused volume rows: same number of rows, segments and activities as saved."""
        text = parameterNode.GetParameter("PerfusedVolumes")
        if not text:
            return
        saved = json.loads(text)
        while len(self.perfusedRows) > max(1, len(saved)):
            self.removePerfusedVolumeRow(self.perfusedRows[-1])
        while len(self.perfusedRows) < len(saved):
            self.addPerfusedVolumeRow()
        for row, values in zip(self.perfusedRows, saved):
            row["selector"].setCurrentSegmentID(values.get("segmentID", ""))
            row["activitySlider"].value = float(values.get("activityMBq", 0.0))
        self._updatePerfusedRowLabels()
        self._updateTotalActivity()

    def _clearModuleResults(self):
        pass

    def _previewPerfusedIDs(self):
        return self._perfusedSegmentIDs()

    def _previewSegmentVolumeML(self, segmentationNode, segmentID):
        return self._cachedSegmentVolumeML(segmentationNode, segmentID)

class RadioembolizationDosimetryRelativeLogic(DosimetryLogicBase):
    """Computation and visualization, separated from the GUI so it can be tested and scripted."""

    # -- dose ----------------------------------------------------------------

    def computeRelativeDose(self, spectVolumeNode, segmentationNode, liverSegmentID, perfusedVolumes,
                            lungShuntPercent, conversionFactor, densityGPerML, lungMassG, clipNegativeValues=True,
                            tumorSegmentIDs=(), lungSegmentIDs=()):
        """Patient-relative dose with one or more perfused volumes.

        perfusedVolumes: [{"number", "segmentID", "name", "activityMBq"}], activityMBq = administered activity
        for that perfused volume before lung shunt. Each perfused volume (inside the whole liver) receives
        its activity * (1 - LSF), distributed in proportion to its counts; the per-volume dose maps are added.

        Returns a dict: dose [Gy], voxelVolumeML, liverMask, dosedMask (union of clipped perfused volumes),
        territoryMasks [(number, segmentID, mask)], territories (per-volume info), lungDoseGy,
        totalActivityMBq, countsOutsideLiver (fraction of image counts), liverCountsOutsidePerfused
        (fraction of whole-liver counts outside all perfused volumes), negativeVoxels / negativeVoxelsInLiver
        (counts in the original image), voxelCount and clippedNegativeValues. With clipNegativeValues, negative
        voxel values are set to 0 when the image is read.
        Perfused volumes and tumors (tumorSegmentIDs) must lie entirely inside the whole-liver segment on the
        SPECT/PET grid; otherwise a ValueError lists the offending segments.
        countsOutsideLiverAndLungs: fraction of the image counts outside the whole liver and the lung segments
        (lungSegmentIDs; lungsSegmented False if there are none).
        """
        if not (0 <= lungShuntPercent < 100):
            raise ValueError("Lung shunt fraction must be between 0 and 100 % (exclusive).")
        if lungMassG <= 0 or conversionFactor <= 0:
            raise ValueError("Lung mass and conversion factor must be positive.")
        if not perfusedVolumes:
            raise ValueError("Select at least one perfused volume.")

        rawCounts = slicer.util.arrayFromVolume(spectVolumeNode)
        counts, negativeVoxels = prepareImageValues(rawCounts, clipNegativeValues)
        voxelVolumeML = voxelVolumeMLFromNode(spectVolumeNode)
        liverMask = segmentMaskOnVolumeGrid(segmentationNode, liverSegmentID, spectVolumeNode)
        if liverMask.shape != counts.shape:
            raise RuntimeError(f"Liver mask shape {liverMask.shape} does not match image shape {counts.shape}.")
        if not liverMask.any():
            raise ValueError("The whole liver segment does not overlap the SPECT/PET volume. "
                             "Check registration and transforms.")

        def title(pv):
            return f"perfused volume {pv['number']} ({pv['name']})"

        # Perfused volume masks on the SPECT/PET grid
        rawMasks = []
        for pv in perfusedVolumes:
            mask = segmentMaskOnVolumeGrid(segmentationNode, pv["segmentID"], spectVolumeNode)
            if mask.shape != counts.shape:
                raise RuntimeError(f"Mask shape {mask.shape} of {title(pv)} does not match image shape {counts.shape}.")
            if not mask.any():
                raise ValueError(f"The segment of {title(pv)} has no voxels on the SPECT/PET grid. "
                                 "Check registration and transforms.")
            rawMasks.append(mask)

        # Perfused volumes must not intersect
        overlaps = findMaskOverlaps(rawMasks)
        if overlaps:
            lines = [f"  P{title(perfusedVolumes[i])[1:]} and {title(perfusedVolumes[j])}: "
                     f"{n} voxels ({n * voxelVolumeML:.2f} mL)" for i, j, n in overlaps]
            raise ValueError("Perfused volumes must not intersect. Overlap on the SPECT/PET grid:\n"
                             + "\n".join(lines) + "\n\nRemove the overlap in the Segment Editor "
                             "(e.g. Logical operators > Subtract, or edit with 'Overwrite other segments').")

        # Perfused volumes and tumors must lie entirely inside the whole liver
        namedMasks = [(title(pv).capitalize()[0] + title(pv)[1:], mask) for pv, mask in zip(perfusedVolumes, rawMasks)]
        segmentation = segmentationNode.GetSegmentation()
        for segmentID in tumorSegmentIDs:
            if segmentID == liverSegmentID or segmentation.GetSegment(segmentID) is None:
                continue
            tumorMask = segmentMaskOnVolumeGrid(segmentationNode, segmentID, spectVolumeNode)
            namedMasks.append((f"Tumor '{segmentation.GetSegment(segmentID).GetName()}'", tumorMask))
        outside = containmentProblems(namedMasks, liverMask, voxelVolumeML)
        if outside:
            lines = [f"  {name}: {volumeML:.2f} mL ({100 * fraction:.1f}%) outside" for name, volumeML, fraction in outside]
            raise ValueError("Perfused volumes and tumors must lie entirely inside the whole-liver segment "
                             "(checked on the SPECT/PET grid):\n" + "\n".join(lines) + "\n\nFix the segments in "
                             "the Segment Editor, e.g. Logical operators > Intersect with the whole liver, or edit "
                             "with Masking > Editable area: inside the whole-liver segment.")

        clippedMasks, territories = [], []
        for pv, mask in zip(perfusedVolumes, rawMasks):
            clippedMasks.append(mask)  # fully inside the liver (checked above)
            territories.append({
                "number": pv["number"], "segmentID": pv["segmentID"], "name": pv["name"],
                "activityMBq": pv["activityMBq"],
                "volumeML": np.count_nonzero(mask) * voxelVolumeML,
            })

        dosedMask = np.zeros(counts.shape, dtype=bool)
        for clipped in clippedMasks:
            dosedMask |= clipped
        negativeVoxelsInLiver = int(np.count_nonzero(rawCounts[liverMask] < 0))
        if negativeVoxels and not clipNegativeValues:
            logging.warning(f"{negativeVoxels} voxels have negative values; they produce negative voxel doses.")

        # One dose map per perfused volume, then summed
        lsf = lungShuntPercent / 100.0
        dose = computeMultiTerritoryDoseMap(
            counts, clippedMasks, [pv["activityMBq"] * (1.0 - lsf) for pv in perfusedVolumes],
            voxelVolumeML, conversionFactor, densityGPerML,
            territoryNames=[title(pv) for pv in perfusedVolumes])

        totalActivityMBq = float(sum(pv["activityMBq"] for pv in perfusedVolumes))
        liverOrLungs = liverMask.copy()
        lungsSegmented = False
        for segmentID in lungSegmentIDs:
            if segmentation.GetSegment(segmentID) is None:
                continue
            lungMask = segmentMaskOnVolumeGrid(segmentationNode, segmentID, spectVolumeNode)
            if lungMask.shape == counts.shape:
                liverOrLungs |= lungMask
                lungsSegmented = True
        return {
            "dose": dose,
            "voxelVolumeML": voxelVolumeML,
            "liverMask": liverMask,
            "dosedMask": dosedMask,
            "territoryMasks": [(pv["number"], pv["segmentID"], mask) for pv, mask in zip(perfusedVolumes, rawMasks)],
            "territories": territories,
            "lungDoseGy": totalActivityMBq * lsf * conversionFactor / lungMassG,
            "totalActivityMBq": totalActivityMBq,
            "countsOutsideLiver": countsOutsideMaskFraction(counts, liverMask),
            "countsOutsideLiverAndLungs": countsOutsideMaskFraction(counts, liverOrLungs),
            "lungsSegmented": lungsSegmented,
            "liverCountsOutsidePerfused": countsOutsideMaskFraction(counts[liverMask], dosedMask[liverMask]),
            "negativeVoxels": negativeVoxels,
            "negativeVoxelsInLiver": negativeVoxelsInLiver,
            "voxelCount": int(counts.size),
            "clippedNegativeValues": bool(clipNegativeValues),
        }

    def segmentStatistics(self, doseArray, segmentationNode, referenceVolumeNode, voxelVolumeML,
                          conversionFactor, densityGPerML, dosedMask, territoryMasks, liverSegmentID, categories):
        """One dict per segment, plus 'All tumors (combined)' (union of the tumor segments, id None) when tumors
        are categorized; ordered by role. Keys: id, name, displayName (name + perfused volume numbers), label
        (displayName + outside flag), role, roleLabel, color, dose, volume, activity, outside (fraction outside
        all perfused volumes) and doses (sorted voxel doses, empty if the segment is not modelled).
        territoryMasks: [(number, segmentID, mask)]. A segment is tagged with the numbers of the perfused
        volumes it is or overlaps; the whole-liver segment only when it is itself a perfused volume."""
        segmentation = segmentationNode.GetSegmentation()
        perfusedIDs = {segmentID for _, segmentID, _ in territoryMasks}
        results = []
        tumorMask = None
        viableMask = None
        for segmentID, name in segmentList(segmentationNode):
            if categories.get(segmentID) in (CATEGORY_IGNORED, CATEGORY_LUNGS) and segmentID != liverSegmentID:
                continue  # not calculated: ignored segments and the lungs (patient-relative mode)
            mask = segmentMaskOnVolumeGrid(segmentationNode, segmentID, referenceVolumeNode)
            if mask.shape != doseArray.shape:
                raise RuntimeError(f"Mask shape {mask.shape} does not match image shape {doseArray.shape}.")
            role = segmentRole(segmentID, liverSegmentID, perfusedIDs, categories)
            if role == CATEGORY_TUMOR:
                tumorMask = mask.copy() if tumorMask is None else (tumorMask | mask)
            elif role == CATEGORY_VIABLE:
                viableMask = mask.copy() if viableMask is None else (viableMask | mask)
            results.append(self._statisticsEntry(
                doseArray, mask, voxelVolumeML, conversionFactor, densityGPerML, dosedMask, territoryMasks,
                segmentID, name, role, tuple(segmentation.GetSegment(segmentID).GetColor()),
                tagOverlaps=segmentID != liverSegmentID))
        if tumorMask is not None:
            results.append(self._statisticsEntry(
                doseArray, tumorMask, voxelVolumeML, conversionFactor, densityGPerML, dosedMask, territoryMasks,
                None, COMBINED_TUMORS_NAME, "tumors", COLOR_TUMOR, tagOverlaps=True))
        if viableMask is not None:
            results.append(self._statisticsEntry(
                doseArray, viableMask, voxelVolumeML, conversionFactor, densityGPerML, dosedMask, territoryMasks,
                None, COMBINED_VIABLE_NAME, "viables", COLOR_VIABLE, tagOverlaps=True))
        return sortResults(results)

    @staticmethod
    def _statisticsEntry(doseArray, mask, voxelVolumeML, conversionFactor, densityGPerML, dosedMask, territoryMasks,
                         segmentID, name, role, color, tagOverlaps):
        dose, volume, activity = segmentDoseStatistics(doseArray, mask, voxelVolumeML, conversionFactor, densityGPerML)
        outside = None
        numbers = tuple(n for n, sid, _ in territoryMasks if segmentID is not None and sid == segmentID)
        modelled = volume > 0
        if not modelled:
            logging.warning(f"Segment '{name}' has no voxels on the image grid.")
        else:
            if tagOverlaps:
                numbers = tuple(n for n, sid, tMask in territoryMasks
                                if (segmentID is not None and sid == segmentID) or np.any(mask & tMask))
            outside = np.count_nonzero(mask & ~dosedMask) / np.count_nonzero(mask)
            if outside >= 1.0:  # outside every perfused volume: not modelled, dose unknown (not 0 Gy)
                dose, activity, modelled = float("nan"), float("nan"), False
        doses = np.sort(doseArray[mask]) if modelled else np.zeros(0)
        return {"id": segmentID, "name": name, "displayName": name + territoryTag(numbers),
                "label": decorateSegmentName(name, numbers, outside),
                "role": role, "roleLabel": RESULT_ROLES[role][0], "color": color,
                "dose": dose, "volume": volume, "activity": activity, "outside": outside, "doses": doses}

class RadioembolizationDosimetryRelativeTest(ScriptedLoadableModuleTest):
    """Synthetic-data checks of the dose math ("Reload and Test" in developer mode)."""

    def runTest(self):
        self.test_uniformLiverDose()
        self.test_perfusedVolumesAreAdded()
        self.test_overlappingPerfusedVolumesRejected()
        self.test_segmentLabels()
        self.test_singleCompartmentEstimate()
        self.test_labelStacking()
        self.test_containment()
        self.test_negativeVoxelValues()
        self.test_categories()
        self.test_segmentModelRoles()
        self.test_reportScreenshots()
        self.test_dvhMetrics()
        self.test_isodoseOpacity()

    def test_uniformLiverDose(self):
        """One perfused volume = whole liver reproduces the classic single-compartment result."""
        counts = np.ones((10, 10, 10))
        liver = np.zeros(counts.shape, bool)
        liver[2:8, 2:8, 2:8] = True
        vox, A, CF, rho = 0.1, 1000.0, 49.67, 1.05
        dose = computeMultiTerritoryDoseMap(counts, [liver], [A], vox, CF, rho)
        expected = A * CF / (liver.sum() * vox * rho)
        assert abs(dose[liver].mean() - expected) < 1e-9 * expected
        assert np.all(dose[~liver] == 0)
        assert abs(countsOutsideMaskFraction(counts, liver) - (1 - liver.mean())) < 1e-12
        self.delayDisplay("Relative: uniform liver dose OK")

    def test_perfusedVolumesAreAdded(self):
        """Each territory gets only its own activity; the total map is the sum of the per-territory maps."""
        rng = np.random.default_rng(1)
        counts = rng.uniform(1, 100, (12, 12, 12))
        right = np.zeros(counts.shape, bool)
        right[:6] = True
        left = np.zeros(counts.shape, bool)
        left[6:10] = True
        vox, CF, rho = 0.2, 49.67, 1.05
        total = computeMultiTerritoryDoseMap(counts, [right, left], [1500.0, 500.0], vox, CF, rho)
        separate = (computeRelativeDoseMap(counts, right, 1500.0, vox, CF, rho)
                    + computeRelativeDoseMap(counts, left, 500.0, vox, CF, rho))
        assert np.allclose(total, separate)
        # Delivered activity per territory is conserved: sum(dose * rho * V) / CF = A
        for mask, A in ((right, 1500.0), (left, 500.0)):
            assert abs(total[mask].sum() * rho * vox / CF - A) < 1e-9 * A
        assert np.all(total[10:] == 0)  # not in any perfused volume
        # Changing one territory's activity leaves the other untouched
        other = computeMultiTerritoryDoseMap(counts, [right, left], [1500.0, 900.0], vox, CF, rho)
        assert np.array_equal(other[right], total[right])
        self.delayDisplay("Relative: perfused volume dose maps are added OK")

    def test_overlappingPerfusedVolumesRejected(self):
        counts = np.ones((8, 8, 8))
        a = np.zeros(counts.shape, bool)
        a[:5] = True
        b = np.zeros(counts.shape, bool)
        b[4:] = True
        assert findMaskOverlaps([a, b]) == [(0, 1, 64)]
        try:
            computeMultiTerritoryDoseMap(counts, [a, b], [100.0, 100.0], 0.1, 49.67, 1.05)
        except ValueError:
            pass
        else:
            raise AssertionError("Overlapping perfused volumes were accepted.")
        self.delayDisplay("Relative: overlap rejected OK")

    def test_singleCompartmentEstimate(self):
        """The estimate equals the mean voxel dose of a perfused volume (activity is conserved in the mask)."""
        rng = np.random.default_rng(2)
        counts = rng.uniform(1, 100, (10, 10, 10))
        pv = np.zeros(counts.shape, bool)
        pv[2:7] = True
        vox, A, lsf, CF, rho = 0.3, 1200.0, 0.08, 49.67, 1.05
        voxelMean = computeMultiTerritoryDoseMap(counts, [pv], [A * (1 - lsf)], vox, CF, rho)[pv].mean()
        estimate = singleCompartmentDoseGy(A, lsf, CF, rho, pv.sum() * vox)
        assert abs(voxelMean - estimate) < 1e-9 * estimate
        levels = ISODOSE_PRESETS[0][1]  # glass: 10, 20, 50, ...
        assert isodoseColorForDose(5.0, levels) is None
        assert isodoseColorForDose(20.0, levels) == ISODOSE_COLORS[1][1]
        assert isodoseColorForDose(120.0, levels) == ISODOSE_COLORS[4][1]
        assert isodoseColorForDose(5000.0, levels) == ISODOSE_COLORS[7][1]
        self.delayDisplay("Relative: single-compartment estimate OK")

    def test_categories(self):
        categories = {"t1": CATEGORY_TUMOR, "n1": CATEGORY_NORMAL, "pv": CATEGORY_TUMOR}
        assert segmentRole("liver", "liver", {"pv"}, categories) == "liver"
        assert segmentRole("pv", "liver", {"pv"}, categories) == "perfused"  # perfused volume takes precedence
        assert segmentRole("t1", "liver", {"pv"}, categories) == CATEGORY_TUMOR
        assert segmentRole("x", "liver", {"pv"}, categories) == "uncategorized"
        rows = [{"role": r} for r in ("uncategorized", CATEGORY_TUMOR, "perfused", "tumors", "liver", CATEGORY_OTHER)]
        assert [r["role"] for r in sortResults(rows)] == ["liver", "perfused", "tumors", CATEGORY_TUMOR,
                                                          CATEGORY_OTHER, "uncategorized"]
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

    def test_containment(self):
        liver = np.zeros((6, 6, 6), bool)
        liver[1:5] = True
        inside = np.zeros(liver.shape, bool)
        inside[2:4] = True
        protruding = np.zeros(liver.shape, bool)
        protruding[4:6] = True  # half of it outside the liver
        problems = containmentProblems([("PV", inside), ("Tumor 'T1'", protruding)], liver, 0.5)
        assert len(problems) == 1 and problems[0][0] == "Tumor 'T1'"
        assert problems[0][1] == 36 * 0.5 and problems[0][2] == 0.5
        self.delayDisplay("Relative: containment check OK")

    def test_labelStacking(self):
        labels = [{"anchorY": 100.0, "h": 20.0}, {"anchorY": 105.0, "h": 20.0}, {"anchorY": 5.0, "h": 20.0}]
        stackLabelCentres(labels, 8.0, 300.0, 4.0)
        ys = sorted(label["y"] for label in labels)
        assert all(b - a >= 24.0 - 1e-9 for a, b in zip(ys, ys[1:]))  # no overlap
        assert ys[0] >= 18.0 - 1e-9                                    # inside the bottom margin
        self.delayDisplay("Label stacking OK")

    def test_segmentLabels(self):
        assert decorateSegmentName("Right lobe", (1,), 0.0) == "Right lobe (Perfused volume 1)"
        assert decorateSegmentName("Tumor", (1, 2), 0.25) == "Tumor (Perfused volumes 1, 2) [25.0% outside perfused volumes]"
        assert decorateSegmentName("Stomach", (), 1.0) == "Stomach [outside perfused volumes - not modelled]"
        assert decorateSegmentName("Liver", (), None) == "Liver"
        self.delayDisplay("Relative: segment labels OK")

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
