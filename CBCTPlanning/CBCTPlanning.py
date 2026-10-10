"""Taranis - CBCT territory planning.

Arterial tree from a CBCT acquired during catheter angiography -> planned catheter tip (markup) -> predicted
downstream liver territory -> perfused-volume candidate in the case segmentation, used as is by patient-relative
dosimetry. Also: feeder finder for a tumour, flag for branches leaving the liver downstream of a tip, CBCT field of
view coverage, comparison with the MAA uptake, and the perfused volume from contrast enhancement on a selective or
parenchymal CBCT.

Usable alone (choose the CBCT, the segmentation and its whole-liver segment) or opened from the Taranis hub
(Planning step), which fills in the images of the case. Methods and references: TaranisLib/vascular.py.
NOT a medical device. Research use only.
"""

import contextlib
import json
import logging
import os

import numpy as np
import qt
import ctk
import vtk
import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

try:
    import TaranisLib  # noqa: F401
except ImportError:  # developer layout (module folders side by side) before the Taranis folder is on sys.path
    import sys
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Taranis"))
from TaranisLib import roles as R
from TaranisLib import workflow as W
from TaranisLib import vascular as V
from TaranisLib import cbct as C
from TaranisLib.widgets import allowNarrowPanel
from TaranisLib.case import (TaranisCase, P_PLANNING_SKIPPED, P_PLANNING_REVIEWED, REF_PLANNING_INJECTION,
                             REF_PLANNING_TIPS)
from TaranisLib.controller import (controlPoints, planningResults, segmentIDs, segmentRole, isCandidate,
                                   WorkflowController)

DISCLAIMER = ("⚠ NOT A MEDICAL DEVICE. RESEARCH USE ONLY. The predicted territory is a model of where the "
              "microspheres may go, built from the visible arterial tree: it does not replace the angiography, the "
              "MAA scan or the operator's judgement.")


class CBCTPlanning(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "Taranis - CBCT Planning"
        parent.categories = ["Nuclear Medicine"]
        parent.dependencies = []
        parent.contributors = ["Burak Demir, MD, FEBNM"]
        parent.helpText = (
            "CBCT territory planning for radioembolization: arterial tree from the CBCT, planned catheter tips, "
            "predicted territories as perfused volumes, feeder finder, extrahepatic branch flag, perfused volume from "
            "CBCT enhancement. Opened from the Taranis hub (Planning step) or used alone.<br>"
            "NOT a medical device. Research use only.")
        parent.acknowledgementText = (
            "Developed by Burak Demir, MD, FEBNM. Methods: Frangi et al. 1998 (vesselness), Lee et al. 1994 (3D "
            "thinning), Selle et al. 2002 (nearest-branch territories), minimum-cost-path territories, Deschamps "
            "et al. 2010 (feeders on CBCT).")
        iconPath = os.path.join(os.path.dirname(__file__), "Resources", "Icons", "CBCTPlanning.png")
        if os.path.exists(iconPath):
            parent.icon = qt.QIcon(iconPath)


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
    combo.noneEnabled = noneEnabled
    combo.showHidden = False
    combo.showChildNodeTypes = False
    combo.setMRMLScene(slicer.mrmlScene)
    combo.setToolTip(toolTip)
    return combo


def table(headers):
    widget = qt.QTableWidget(0, len(headers))
    widget.setHorizontalHeaderLabels(headers)
    widget.verticalHeader().visible = False
    widget.horizontalHeader().setStretchLastSection(True)
    widget.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)
    widget.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
    widget.setSelectionMode(qt.QAbstractItemView.SingleSelection)
    widget.setMinimumHeight(110)
    return widget


class CBCTPlanningWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):

    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.session = None
        self._feeders = []
        self._updating = False

    # -- Setup --

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        disclaimer = styledLabel(DISCLAIMER, "color: #ff0000; font-weight: bold;")
        self.layout.addWidget(disclaimer)
        self.caseLabel = smallGray()
        self.caseLabel.setTextFormat(qt.Qt.RichText)
        self.layout.addWidget(self.caseLabel)

        # -- Inputs --
        inputsBox = ctk.ctkCollapsibleButton()
        inputsBox.text = "Inputs"
        form = qt.QFormLayout(inputsBox)
        self.cbctSelector = nodeComboBox(["vtkMRMLScalarVolumeNode"], "Arterial-phase CBCT (contrast injected at the "
                                         "catheter tip), registered to the image of the segmentation.")
        form.addRow("CBCT:", self.cbctSelector)
        self.segmentationSelector = nodeComboBox(["vtkMRMLSegmentationNode"], "Segmentation with the whole liver and "
                                                 "the tumours (the case segmentation).")
        form.addRow("Segmentation:", self.segmentationSelector)
        self.liverCombo = qt.QComboBox()
        self.liverCombo.setToolTip("The whole-liver segment: territories are predicted inside it.")
        form.addRow("Whole liver:", self.liverCombo)
        self.spacingCombo = qt.QComboBox()
        for spacing in C.SPACING_CHOICES_MM:
            self.spacingCombo.addItem(f"{spacing:g} mm", spacing)
        self.spacingCombo.setToolTip("Voxel size of the working grid around the liver. 1 mm: fast, recommended. "
                                     "0.5 mm: finer vessels, about 8 times more memory and time.")
        form.addRow("Working resolution:", self.spacingCombo)
        self.uptakeSelector = nodeComboBox(["vtkMRMLScalarVolumeNode"], "Optional: MAA SPECT to compare the predicted "
                                           "territories with (Dice, MAA counts inside).")
        form.addRow("Compare with MAA:", self.uptakeSelector)
        self.inputStatus = smallGray()
        self.inputStatus.setTextFormat(qt.Qt.RichText)
        form.addRow(self.inputStatus)
        row = qt.QHBoxLayout()
        layoutButton = qt.QPushButton("Show planning layout")
        layoutButton.connect("clicked()", self.onShowLayout)
        self.registerButton = qt.QPushButton("Register the CBCT…")
        self.registerButton.setToolTip("Opens the Registration step of the Taranis hub (EasyReg, rigid, 1 mm, inside "
                                       "the liver).")
        self.registerButton.connect("clicked()", self.onRegister)
        row.addWidget(layoutButton)
        row.addWidget(self.registerButton)
        form.addRow(row)
        row = qt.QHBoxLayout()
        self.threeDButtons = {}
        for key, label, toolTip in (
                ("Liver", "Liver", "The whole liver (green wireframe)."),
                ("Tumours", "Tumours", "Tumour and viable tumour segments."),
                ("Perfused", "Perfused", "Perfused volumes and predicted territories (candidates included)."),
                ("Mip", "MIP", "DSA-like maximum intensity projection of the CBCT in the search mask."),
                ("Vessels", "Vessels", "Centerlines of the extracted arterial tree (red lines).")):
            button = qt.QPushButton(label)
            button.checkable = True
            button.checked = True
            button.setToolTip(f"Show / hide in the 3D view: {toolTip}")
            button.connect("toggled(bool)", lambda checked, key=key: self.onThreeDToggled(key, checked))
            row.addWidget(button)
            self.threeDButtons[key] = button
        form.addRow("3D view:", row)
        dsaRow = qt.QHBoxLayout()
        self.dsaStartSpin = qt.QDoubleSpinBox()
        self.dsaStartSpin.setRange(0.0, 99.0)
        self.dsaStartSpin.setSingleStep(1.0)
        self.dsaStartSpin.setDecimals(0)
        self.dsaStartSpin.setPrefix("white up to ")
        self.dsaStartSpin.setSuffix(" percentile")
        self.dsaStartSpin.setToolTip("3D view (DSA-like maximum intensity projection of the CBCT inside the search "
                                     "mask): voxels up to this percentile of the CBCT inside the search mask are "
                                     "white (transparent). Higher: less parenchyma and noise.")
        self.dsaTopSpin = qt.QDoubleSpinBox()
        self.dsaTopSpin.setRange(50.0, 99.99)
        self.dsaTopSpin.setDecimals(2)
        self.dsaTopSpin.setSingleStep(0.25)
        self.dsaTopSpin.setPrefix("black from ")
        self.dsaTopSpin.setSuffix(" percentile")
        self.dsaTopSpin.setToolTip("3D view: voxels from this percentile of the CBCT inside the search mask up are "
                                   "black. Higher: only the densest contrast is black, the image is lighter.")
        dsaRow.addWidget(self.dsaStartSpin)
        dsaRow.addWidget(self.dsaTopSpin)
        form.addRow("MIP windowing:", dsaRow)
        self.shadedSliders = []
        for attribute, label, toolTip in (
                ("shadedStartSpin", "Shaded: opacity 0 at",
                 "Shaded view: transparent up to this percentile of the CBCT inside the search mask (colour #ff003c)."),
                ("shadedMidSpin", "Shaded: opacity 0.5 at", "Shaded view: opacity 0.5 at this percentile (#ff0000)."),
                ("shadedTopSpin", "Shaded: opacity 1.0 at",
                 "Shaded view: fully opaque from this percentile up (white; white already half way from the 0.5 "
                 "point).")):
            slider = ctk.ctkSliderWidget()
            slider.minimum = 0.0
            slider.maximum = 100.0
            slider.decimals = 2
            slider.singleStep = 0.25
            slider.pageStep = 1.0
            slider.suffix = " pp"
            slider.setToolTip(toolTip)
            setattr(self, attribute, slider)
            self.shadedSliders.append(slider)
            form.addRow(label + ":", slider)
        modeRow = qt.QHBoxLayout()
        self.compositeButton = qt.QPushButton("Shaded")
        self.compositeButton.checkable = True
        self.compositeButton.setToolTip("Off: maximum intensity projection (DSA-like). On: composite volume rendering "
                                        "with shading (depth and vessel surfaces), its own windowing above.")
        self.negativeButton = qt.QPushButton("Negative")
        self.negativeButton.checkable = True
        self.negativeButton.setToolTip("Black background. MIP: white vessels; shaded view: same colours. Segments and "
                                       "lines keep their colours.")
        for button in (self.compositeButton, self.negativeButton):
            button.connect("toggled(bool)", self.onDsaChanged)
            modeRow.addWidget(button)
        form.addRow("Rendering:", modeRow)
        for spin in (self.shadedStartSpin, self.shadedMidSpin, self.shadedTopSpin):
            spin.connect("valueChanged(double)", self.onDsaChanged)
        self.layout.addWidget(inputsBox)

        # -- 1. Arterial tree --
        treeBox = ctk.ctkCollapsibleButton()
        treeBox.text = "1. Arterial tree"
        treeLayout = qt.QFormLayout(treeBox)
        treeLayout.addRow(smallGray("Place the injection point at the catheter tip of the CBCT run, inside the "
                                    "contrast-filled artery, then extract the tree. Review it, erase veins, bone or the "
                                    "catheter in the Segment Editor if needed, and recompute the centerlines."))
        row = qt.QHBoxLayout()
        self.injectionButton = qt.QPushButton("Place injection point")
        self.injectionButton.connect("clicked()", self.onPlaceInjection)
        row.addWidget(self.injectionButton)
        self.injectionLabel = smallGray()
        row.addWidget(self.injectionLabel, 1)
        treeLayout.addRow(row)
        self.vesselSdSpin = qt.QDoubleSpinBox()
        self.vesselSdSpin.setRange(0.5, 20.0)
        self.vesselSdSpin.setSingleStep(0.5)
        self.vesselSdSpin.setPrefix("local liver + ")
        self.vesselSdSpin.setSuffix(" SD")
        self.vesselSdSpin.setToolTip(f"Contrast-filled arteries (seeds) are brighter than the liver around them: "
                                     f"mean and SD of the liver voxels within {V.LOCAL_WINDOW_MM:g} mm (bright voxels "
                                     "left out, only liver voxels used, so the shading of the CBCT cancels out). "
                                     "Higher: fewer false vessels, fewer small branches.")
        treeLayout.addRow("Vessels brighter than:", self.vesselSdSpin)
        self.growSdSpin = qt.QDoubleSpinBox()
        self.growSdSpin.setRange(0.0, 20.0)
        self.growSdSpin.setSingleStep(0.5)
        self.growSdSpin.setPrefix("local liver + ")
        self.growSdSpin.setSuffix(" SD")
        self.growSdSpin.setToolTip("The tree grows from the seeds into connected voxels brighter than this (smaller, "
                                   "fainter branches). Lower: more branches, but more leaks into the parenchyma.")
        treeLayout.addRow("Grow into:", self.growSdSpin)
        self.tubularCheck = qt.QCheckBox("Grown part must be tube-shaped (vesselness filter)")
        self.tubularCheck.setToolTip("The voxels between the two thresholds must also look like a tube (Frangi "
                                     "vesselness on the local contrast): noise speckle is blob-like and metal streaks "
                                     "are flat, so neither grows the tree.")
        treeLayout.addRow(self.tubularCheck)
        self.searchMaskCheck = qt.QCheckBox("Show the search mask")
        self.searchMaskCheck.setToolTip(f"Where arteries are searched: the whole liver + {V.SEARCH_MARGIN_MM:g} mm and "
                                        f"{V.INJECTION_REGION_MM:g} mm around the injection point, inside the CBCT "
                                        "field of view, without air / lung (much darker than fat) and "
                                        f"{V.AIR_MARGIN_MM:g} mm around it.")
        treeLayout.addRow(self.searchMaskCheck)
        filteredButton = qt.QPushButton("Show filtered images")
        filteredButton.setToolTip("Adds the images the tree is extracted from: the local contrast (SD above the liver "
                                  "around each voxel: what the thresholds apply to; window 0-9), the denoised CBCT, "
                                  "the local liver background and the tube-shape score (0-1), plus the search mask. "
                                  "Pick them as background or foreground in the slice views.")
        filteredButton.connect("clicked()", self.onShowFiltered)
        treeLayout.addRow(filteredButton)
        self.statisticsLabel = smallGray()
        treeLayout.addRow(self.statisticsLabel)
        row = qt.QHBoxLayout()
        self.extractButton = qt.QPushButton("Extract arterial tree")
        self.extractButton.setStyleSheet("font-weight: bold;")
        self.extractButton.connect("clicked()", self.onExtractTree)
        editButton = qt.QPushButton("Edit tree…")
        editButton.setToolTip("Opens the Segment Editor with the arterial tree segment and the CBCT.")
        editButton.connect("clicked()", self.onEditTree)
        recomputeButton = qt.QPushButton("Recompute centerlines")
        recomputeButton.setToolTip("After editing the tree segment.")
        recomputeButton.connect("clicked()", self.onRecomputeCenterlines)
        for button in (self.extractButton, editButton, recomputeButton):
            row.addWidget(button)
        treeLayout.addRow(row)
        self.treeLabel = smallGray()
        treeLayout.addRow(self.treeLabel)
        self.layout.addWidget(treeBox)

        # -- 2. Planned tips --
        tipsBox = ctk.ctkCollapsibleButton()
        tipsBox.text = "2. Planned catheter tips and territories"
        tipsLayout = qt.QFormLayout(tipsBox)
        row = qt.QHBoxLayout()
        addTipButton = qt.QPushButton("Add planned tip")
        addTipButton.setToolTip("Click on an artery in a view; the tip is snapped to the nearest centerline "
                                f"(within {V.SNAP_MM:g} mm).")
        addTipButton.connect("clicked()", self.onAddTip)
        removeTipButton = qt.QPushButton("Remove selected tip")
        removeTipButton.setToolTip("Removes the tip selected in the table below, with its territory.")
        removeTipButton.connect("clicked()", self.onRemoveTip)
        row.addWidget(addTipButton)
        row.addWidget(removeTipButton)
        tipsLayout.addRow(row)
        limitRow = qt.QHBoxLayout()
        self.limitCheck = qt.QCheckBox("Limit each territory to liver within")
        self.limitCheck.setToolTip("After a selective injection part of the liver has no visible branches; without "
                                   "the limit it is assigned to the nearest visible ones. With it, a territory only "
                                   "contains liver at most this far from its own downstream branches (territory "
                                   "prediction and feeder finder).")
        self.supplyDistanceSlider = ctk.ctkSliderWidget()
        self.supplyDistanceSlider.minimum = 3.0
        self.supplyDistanceSlider.maximum = 60.0
        self.supplyDistanceSlider.singleStep = 1.0
        self.supplyDistanceSlider.decimals = 0
        self.supplyDistanceSlider.suffix = " mm of its branches"
        limitRow.addWidget(self.limitCheck)
        limitRow.addWidget(self.supplyDistanceSlider, 1)
        tipsLayout.addRow(limitRow)
        self.limitCheck.connect("toggled(bool)", self._onSettingsChanged)
        self.supplyDistanceSlider.connect("valueChanged(double)", self._onSettingsChanged)
        self.ruleCombo = qt.QComboBox()
        for rule, label in C.RULE_LABELS.items():
            self.ruleCombo.addItem(label, rule)
        self.ruleCombo.setToolTip("Nearest branch: each liver voxel is supplied by the nearest centerline (Selle et "
                                  "al. 2002). Shortest path inside the liver: the path may not cross fissures or "
                                  "leave the liver (slower).")
        tipsLayout.addRow("Territory rule:", self.ruleCombo)
        self.subtractCheck = qt.QCheckBox("Nested tips: the upstream territory excludes the downstream one")
        self.subtractCheck.setToolTip("Two tips on the same branch (e.g. lobar and segmental): without this the "
                                      "territories overlap, which patient-relative dosimetry refuses.")
        tipsLayout.addRow(self.subtractCheck)
        self.predictButton = qt.QPushButton("Predict territories")
        self.predictButton.setStyleSheet("font-weight: bold;")
        self.predictButton.setToolTip("Creates one perfused-volume candidate per tip in the segmentation. Review and "
                                      "accept them in the Taranis Segmentation step.")
        self.predictButton.connect("clicked()", self.onPredict)
        self.resultsTable = table(["Tip", "Territory", "Tumours covered", "Outside liver", "Note"])
        tipsLayout.addRow(self.resultsTable)
        tipsLayout.addRow(self.predictButton)
        self.resultsLabel = smallGray()
        self.resultsLabel.setTextFormat(qt.Qt.RichText)
        tipsLayout.addRow(self.resultsLabel)
        acceptRow = qt.QHBoxLayout()
        acceptButton = qt.QPushButton("Accept selected territory")
        acceptButton.setToolTip("The territory of the tip selected in the table becomes a perfused volume (used by "
                                "patient-relative dosimetry), as with Accept in the Segmentation step.")
        acceptButton.connect("clicked()", self.onAcceptTerritory)
        acceptAllButton = qt.QPushButton("Accept all territories")
        acceptAllButton.setToolTip("Every territory still waiting to be evaluated (and every perfused volume from CBCT "
                                   "enhancement) becomes a perfused volume.")
        acceptAllButton.connect("clicked()", lambda: self.onAcceptTerritory(allTerritories=True))
        acceptRow.addWidget(acceptButton)
        acceptRow.addWidget(acceptAllButton)
        tipsLayout.addRow(acceptRow)
        self.reviewedButton = qt.QPushButton("Mark extrahepatic findings as reviewed")
        self.reviewedButton.setToolTip("You checked the branches leaving the liver on the CBCT / angiography (e.g. "
                                       "coil-embolised or not reached at this tip).")
        self.reviewedButton.connect("clicked()", self.onMarkReviewed)
        tipsLayout.addRow(self.reviewedButton)
        self.layout.addWidget(tipsBox)

        # -- 3. Feeder finder --
        feedersBox = ctk.ctkCollapsibleButton()
        feedersBox.text = "3. Feeder finder"
        feedersBox.collapsed = True
        feedersLayout = qt.QFormLayout(feedersBox)
        self.tumourCombo = qt.QComboBox()
        feedersLayout.addRow("Tumour:", self.tumourCombo)
        findButton = qt.QPushButton("Find feeding branches")
        findButton.connect("clicked()", self.onFindFeeders)
        feedersLayout.addRow(findButton)
        self.feedersTable = table(["Position", "From injection", "Territory", "Tumour covered", "Normal liver"])
        feedersLayout.addRow(self.feedersTable)
        useButton = qt.QPushButton("Add selected position as planned tip")
        useButton.connect("clicked()", self.onUseFeeder)
        feedersLayout.addRow(useButton)
        self.feedersLabel = smallGray()
        feedersLayout.addRow(self.feedersLabel)
        self.layout.addWidget(feedersBox)

        # -- 4. Enhancement --
        enhancementBox = ctk.ctkCollapsibleButton()
        enhancementBox.text = "4. Perfused volume from CBCT enhancement"
        enhancementBox.collapsed = True
        enhancementLayout = qt.QFormLayout(enhancementBox)
        enhancementLayout.addRow(smallGray("For a selective CBCT from the planned position, or the parenchymal "
                                           "phase: the enhanced liver is the perfused volume (the CBCT analogue of "
                                           "'Perfused volume from uptake')."))
        self.enhancementSelector = nodeComboBox(["vtkMRMLScalarVolumeNode"], "Selective or parenchymal-phase CBCT.")
        enhancementLayout.addRow("Image:", self.enhancementSelector)
        self.enhancementSlider = ctk.ctkSliderWidget()
        self.enhancementSlider.minimum = 0
        self.enhancementSlider.maximum = 100
        self.enhancementSlider.singleStep = 1
        self.enhancementSlider.suffix = " %"
        self.enhancementSlider.setToolTip("Threshold between the unenhanced liver (0 %, 10th percentile) and the "
                                          "most enhanced liver (100 %, 99th percentile).")
        enhancementLayout.addRow("Threshold:", self.enhancementSlider)
        row = qt.QHBoxLayout()
        autoButton = qt.QPushButton("Automatic threshold")
        autoButton.setToolTip("Otsu threshold of the liver grey values.")
        autoButton.connect("clicked()", self.onAutoEnhancement)
        createButton = qt.QPushButton("Create perfused volume")
        createButton.connect("clicked()", self.onEnhancement)
        row.addWidget(autoButton)
        row.addWidget(createButton)
        enhancementLayout.addRow(row)
        self.layout.addWidget(enhancementBox)

        # -- Advanced settings --
        advancedBox = ctk.ctkCollapsibleButton()
        advancedBox.text = "Advanced settings (preprocessing and extraction)"
        advancedBox.collapsed = True
        advancedLayout = qt.QFormLayout(advancedBox)
        advancedLayout.addRow(smallGray("For tuning by trial and error: saved with the case, used from the next "
                                        "'Extract arterial tree' (the working grid is rebuilt)."))
        self.denoiseCombo = qt.QComboBox()
        for key, label in V.DENOISE_METHODS:
            self.denoiseCombo.addItem(label, key)
        self.denoiseCombo.setToolTip("Gaussian: blurs everything alike. Edge-preserving filters smooth the parenchyma "
                                     "but not across vessel walls (slower).")
        self.denoiseCombo.connect("currentIndexChanged(int)", self._onAdvancedChanged)
        advancedLayout.addRow("Denoising method:", self.denoiseCombo)
        self.advancedSpins = {}
        for name, label, minimum, maximum, step, toolTip in V.TUNABLE:
            spin = qt.QDoubleSpinBox()
            spin.setRange(minimum, maximum)
            spin.setSingleStep(step)
            spin.setDecimals(2)
            spin.setToolTip(f"{toolTip} Default {V.DEFAULT_PARAMETERS[name]:g}. ({name})")
            spin.connect("valueChanged(double)", self._onAdvancedChanged)
            advancedLayout.addRow(label + ":", spin)
            self.advancedSpins[name] = spin
        resetButton = qt.QPushButton("Reset to defaults")
        resetButton.connect("clicked()", self.onResetAdvanced)
        advancedLayout.addRow(resetButton)
        self.layout.addWidget(advancedBox)

        # -- Workflow --
        row = qt.QHBoxLayout()
        self.skipCheck = qt.QCheckBox("Skip planning for this case")
        self.skipCheck.connect("toggled(bool)", self.onSkipToggled)
        hubButton = qt.QPushButton("Back to Taranis")
        hubButton.connect("clicked()", self.onBackToHub)
        row.addWidget(self.skipCheck)
        row.addStretch(1)
        row.addWidget(hubButton)
        self.layout.addLayout(row)
        self.statusLabel = styledLabel()
        self.statusLabel.setTextFormat(qt.Qt.RichText)
        self.layout.addWidget(self.statusLabel)
        self.layout.addStretch(1)

        self.cbctSelector.connect("currentNodeChanged(vtkMRMLNode*)", self._onInputsChanged)
        self.segmentationSelector.connect("currentNodeChanged(vtkMRMLNode*)", self._onSegmentationChanged)
        self.spacingCombo.connect("currentIndexChanged(int)", self._onSettingsChanged)
        self.vesselSdSpin.connect("valueChanged(double)", self._onSettingsChanged)
        self.growSdSpin.connect("valueChanged(double)", self._onSettingsChanged)
        self.tubularCheck.connect("toggled(bool)", self._onSettingsChanged)
        self.searchMaskCheck.connect("toggled(bool)", self.onShowSearchMask)
        self.dsaStartSpin.connect("valueChanged(double)", self.onDsaChanged)
        self.dsaTopSpin.connect("valueChanged(double)", self.onDsaChanged)
        self.ruleCombo.connect("currentIndexChanged(int)", self._onSettingsChanged)
        self.subtractCheck.connect("toggled(bool)", self._onSettingsChanged)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self._onSceneClose)
        self.feedersTable.connect("itemSelectionChanged()", self.onFeederSelected)
        self._autoPredictTimer = qt.QTimer()
        self._autoPredictTimer.setSingleShot(True)
        self._autoPredictTimer.setInterval(400)
        self._autoPredictTimer.connect("timeout()", self._autoPredict)
        self._observedTips = None

    def cleanup(self):
        self.removeObservers()
        self._closeSession()

    def enter(self):
        allowNarrowPanel(self.parent)   # long names must not widen the panel
        self._fillFromCase()
        self._observeTips()
        self._refresh()
        qt.QTimer.singleShot(0, lambda: self._showLayout(resetView=True))

    def _showLayout(self, resetView=False):
        """Planning layout; 3D view white, orthographic, centred and seen from anterior when resetView."""
        try:
            C.showLayout(self.cbctSelector.currentNode(), C.treeNode(self.store), self.segmentationSelector.currentNode(),
                         self.session, self._dsa(), resetView)
        except Exception as e:
            logging.warning(f"CBCT planning: could not show the planning layout: {e}")

    def exit(self):
        interaction = slicer.app.applicationLogic().GetInteractionNode()
        if interaction is not None and interaction.GetCurrentInteractionMode() == interaction.Place:
            interaction.SetCurrentInteractionMode(interaction.ViewTransform)

    def _onSceneClose(self, caller=None, event=None):
        self._closeSession()
        self._observedTips = None

    def _observeTips(self):
        """Tips added, moved or removed: refresh the table and predict the territories again."""
        node = C.markupsNode(self.store, REF_PLANNING_TIPS)
        if node is None or node is self._observedTips:
            return
        if self._observedTips is not None:
            self.removeObserver(self._observedTips, vtk.vtkCommand.ModifiedEvent, self._onTipsChanged)
            for name in ("PointPositionDefinedEvent", "PointEndInteractionEvent", "PointRemovedEvent"):
                self.removeObserver(self._observedTips, getattr(slicer.vtkMRMLMarkupsNode, name), self._onTipsChanged)
        for name in ("PointPositionDefinedEvent", "PointEndInteractionEvent", "PointRemovedEvent"):
            self.addObserver(node, getattr(slicer.vtkMRMLMarkupsNode, name), self._onTipsChanged)
        self._observedTips = node

    def _onTipsChanged(self, caller=None, event=None):
        """Only the tip table follows the markers (no automatic prediction: it made placing tips lag)."""
        self._refreshResults()
        self.statusLabel.text = "Tips changed: click 'Predict territories' to update the territories."

    def _autoPredict(self):
        """Predict again after a tip change, quietly (no dialogs), once the tree exists."""
        if self.session is None or self.session.tree is None:
            self.statusLabel.text = "Tip changed: click 'Predict territories' (after extracting the tree)."
            return
        node = C.markupsNode(self.store, REF_PLANNING_TIPS)
        if node is None or node.GetNumberOfControlPoints() == 0:
            C.clearTerritories(self.store, self.segmentationSelector.currentNode())
            self._refresh()
            self._updateController()
            self._showLayout(resetView=False)
            self.statusLabel.text = "No planned tip: territory candidates removed."
            return
        self._predict(quiet=True)

    def _closeSession(self):
        if self.session is not None:
            try:
                self.session.close()
            except Exception:
                pass
        self.session = None

    # -- Inputs --

    @property
    def store(self):
        return C.storeNode()

    def _fillFromCase(self):
        """Fill the inputs from the Taranis case (if any) and the stored settings."""
        self._updating = True
        try:
            case = TaranisCase.find()
            store = self.store
            values = C.settings(store)
            index = self.spacingCombo.findData(values["spacingMM"])
            self.spacingCombo.setCurrentIndex(index if index >= 0 else self.spacingCombo.findData(C.DEFAULT_SPACING_MM))
            self.vesselSdSpin.value = float(values["vesselSD"])
            self.growSdSpin.value = float(values["growSD"])
            self.tubularCheck.checked = bool(values["tubularOnly"])
            self.searchMaskCheck.checked = bool(values["showSearchMask"])
            self.dsaStartSpin.value = float(values["dsaStartPercentile"])
            self.compositeButton.checked = values.get("dsaMode") == C.DSA_MODE_COMPOSITE
            self.negativeButton.checked = bool(values.get("negative", False))
            self.shadedStartSpin.value = float(values["shadedStartPercentile"])
            self.shadedTopSpin.value = float(values["shadedTopPercentile"])
            self.shadedMidSpin.value = float(values["shadedMidPercentile"])
            advanced = C.applyAdvanced(store)
            for name, spin in self.advancedSpins.items():
                spin.value = float(advanced.get(name, V.DEFAULT_PARAMETERS[name]))
            self.denoiseCombo.setCurrentIndex(max(0, self.denoiseCombo.findData(
                advanced.get("DENOISE_METHOD", V.DEFAULT_DENOISE_METHOD))))
            for key, button in self.threeDButtons.items():
                button.checked = bool(values[f"show3D{key}"])
            self.dsaTopSpin.value = float(values["dsaTopPercentile"])
            self.ruleCombo.setCurrentIndex(max(0, self.ruleCombo.findData(values["rule"])))
            self.subtractCheck.checked = bool(values["subtractNested"])
            self.enhancementSlider.value = float(values["enhancementPercent"])
            self.supplyDistanceSlider.value = float(values["supplyDistanceMM"])
            self.limitCheck.checked = bool(values["limitTerritories"])
            if case is None:
                self.caseLabel.text = ("No Taranis case in the scene: choose the inputs below. Start a case in "
                                       "Taranis to use the planning in the workflow.")
                self.skipCheck.visible = False
                return
            self.skipCheck.visible = True
            self.skipCheck.checked = case.flag(P_PLANNING_SKIPPED)
            self.caseLabel.text = f"Case: <b>{case.title()}</b> &ndash; inputs filled in from the case."
            cbct = case.roleNode(R.ROLE_CBCT) or case.roleNode(R.ROLE_CBCT_PARENCHYMAL)
            if cbct is not None:
                self.cbctSelector.setCurrentNode(cbct)
            self.enhancementSelector.setCurrentNode(case.roleNode(R.ROLE_CBCT_PARENCHYMAL) or cbct)
            segmentation = case.roleNode(R.ROLE_SEGMENTATION)
            if segmentation is not None:
                self.segmentationSelector.setCurrentNode(segmentation)
            if case.roleType(R.ROLE_DOSIMETRY) == R.TYPE_MAA_SPECT:
                self.uptakeSelector.setCurrentNode(case.roleNode(R.ROLE_DOSIMETRY))
        finally:
            self._updating = False
        self._onSegmentationChanged()

    def _onSegmentationChanged(self, *args):
        node = self.segmentationSelector.currentNode()
        previous = self.liverCombo.currentData
        self.liverCombo.clear()
        self.tumourCombo.clear()
        if node is not None:
            segmentation = node.GetSegmentation()
            liverGuess = None
            for segmentID in segmentIDs(node):
                segment = segmentation.GetSegment(segmentID)
                if isCandidate(segment):
                    continue
                role = segmentRole(node, segmentID)
                if role in (W.SEGMENT_TUMOR, W.SEGMENT_VIABLE):
                    self.tumourCombo.addItem(segment.GetName(), segmentID)
                self.liverCombo.addItem(segment.GetName(), segmentID)
                if role == W.SEGMENT_LIVER and liverGuess is None:
                    liverGuess = segmentID
            index = self.liverCombo.findData(previous) if previous else -1
            if index < 0 and liverGuess is not None:
                index = self.liverCombo.findData(liverGuess)
            if index >= 0:
                self.liverCombo.setCurrentIndex(index)
        self._onInputsChanged()

    def _onInputsChanged(self, *args):
        if not self._updating:
            self._refresh()

    def _onSettingsChanged(self, *args):
        if self._updating:
            return
        C.setSettings(self.store, spacingMM=float(self.spacingCombo.currentData or C.DEFAULT_SPACING_MM),
                      vesselSD=float(self.vesselSdSpin.value), growSD=float(self.growSdSpin.value),
                      tubularOnly=bool(self.tubularCheck.checked), showSearchMask=bool(self.searchMaskCheck.checked),
                      rule=self.ruleCombo.currentData, subtractNested=bool(self.subtractCheck.checked),
                      enhancementPercent=float(self.enhancementSlider.value),
                      supplyDistanceMM=float(self.supplyDistanceSlider.value),
                      limitTerritories=bool(self.limitCheck.checked),
                      **self._dsa())

    def _inputs(self):
        cbct = self.cbctSelector.currentNode()
        segmentation = self.segmentationSelector.currentNode()
        liverID = self.liverCombo.currentData
        if cbct is None:
            raise ValueError("Select the CBCT.")
        if segmentation is None:
            raise ValueError("Select the segmentation with the whole liver.")
        if not liverID:
            raise ValueError("Select the whole-liver segment.")
        return cbct, segmentation, liverID

    def _getSession(self):
        """The working session for the current inputs (rebuilt when an input changed)."""
        cbct, segmentation, liverID = self._inputs()
        spacing = float(self.spacingCombo.currentData or C.DEFAULT_SPACING_MM)
        key = C.sessionKey(cbct, segmentation, liverID, spacing, self.store)
        if self.session is None or self.session.key != key:
            self._closeSession()
            self.session = C.PlanningSession(cbct, segmentation, liverID, spacing, self.store, progress=self._progress)
        return self.session

    def _progress(self, text):
        self.statusLabel.text = f"{text} …"
        slicer.app.processEvents()

    @contextlib.contextmanager
    def _busy(self, text):
        self._progress(text)
        slicer.app.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            yield
        except ImportError:
            slicer.app.restoreOverrideCursor()
            self.statusLabel.text = ""
            if self._offerSkimage():
                self.statusLabel.text = "scikit-image installed: click the button again."
            return
        except Exception as e:
            logging.exception(f"CBCT planning: {e}")
            slicer.app.restoreOverrideCursor()
            self.statusLabel.text = f"<span style='color:#dc2626'>{e}</span>"
            slicer.util.errorDisplay(str(e), windowTitle="CBCT planning")
            return
        slicer.app.restoreOverrideCursor()
        self._refresh()

    def _offerSkimage(self):
        if not slicer.util.confirmOkCancelDisplay(
                "The centerlines need the Python package scikit-image (3D thinning), which is not installed.\n\n"
                "Install it now (download from the Python package index)?", windowTitle="CBCT planning"):
            return False
        slicer.app.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            slicer.util.pip_install("scikit-image")
        finally:
            slicer.app.restoreOverrideCursor()
        return True

    # -- Refresh --

    def _refresh(self):
        store = self.store
        case = TaranisCase.find()
        cbct = self.cbctSelector.currentNode()
        lines = []
        if cbct is not None:
            from TaranisLib.controller import alignmentTransformFor
            transform = alignmentTransformFor(cbct)
            if transform is not None:
                lines.append(f"✓ CBCT registered ({transform.GetAttribute('EasyReg.Method') or 'manual'}).")
            else:
                lines.append("<span style='color:#d97706'>⚠ The CBCT has no registration transform: register it to "
                             "the image of the segmentation (or confirm the alignment in the Registration step).</span>")
        if self.session is not None:
            lines.append(f"Working grid {self.session.grid.spacingMM:g} mm, {self.session.grid.shape[2]} × "
                         f"{self.session.grid.shape[1]} × {self.session.grid.shape[0]} voxels. Whole liver inside the "
                         f"CBCT field of view: {100 * self.session.liverCoverage:.0f}%.")
        self.inputStatus.text = "<br>".join(lines)
        if self.session is not None:
            stats = self.session.statistics
            contrast = self.session.contrast
            parts = [f"Liver on this CBCT: mean {stats.mean:.0f}, SD {stats.sd:.0f} (robust)"]
            if contrast is not None:
                inside = self.session.liver
                local = contrast.background[inside]
                parts.append(f"local background {np.percentile(local, 5):.0f}–{np.percentile(local, 95):.0f} across "
                             f"the liver (shading removed), denoised SD {contrast.statistics.sd:.0f}")
            parts.append(f"air / lung below {self.session.airThreshold:.0f}" if self.session.airThreshold is not None
                         else "no air / lung in the field of view")
            if self.session.metalML:
                parts.append(f"metal + {V.METAL_MARGIN_MM:g} mm margin excluded: {self.session.metalML:.1f} mL")
            self.statisticsLabel.text = "; ".join(parts) + "."
        else:
            self.statisticsLabel.text = ""
        self.registerButton.enabled = case is not None
        injection = controlPoints(store.GetNodeReference(REF_PLANNING_INJECTION))
        self.injectionLabel.text = ("placed" if injection else "not placed")
        tree = self.session.tree if self.session is not None else None
        if tree is not None:
            self.treeLabel.text = (f"Tree: {tree.lengthMM() / 10.0:.0f} cm of centerline, {len(tree.branchPoints())} "
                                   f"branch points, {tree.loops} loop(s).")
        elif C.treeNode(store) is not None:
            self.treeLabel.text = "Arterial tree segment present (centerlines are computed when needed)."
        else:
            self.treeLabel.text = "No arterial tree yet."
        self._refreshResults()

    def _refreshResults(self):
        results = planningResults(self.store) or {}
        tipsNode = C.markupsNode(self.store, REF_PLANNING_TIPS)
        labels = C.tipLabels(tipsNode)
        byLabel = {p.get("label"): p for p in results.get("positions", [])}
        self.resultsTable.setRowCount(len(labels))
        for row, label in enumerate(labels):
            position = byLabel.get(label)
            if position is None:
                cells = [label, "…", "", "", "not predicted yet"]
            elif not position.get("snapped", True):
                cells = [position["label"], "–", "–", "–", f"{position.get('snapDistanceMM', 0):.0f} mm from the tree"]
            else:
                tumours = ", ".join(f"{name} {percent:.0f}%" for name, percent in position.get("tumours", []))
                outside = position.get("extrahepatic", [])
                cells = [position["label"], f"{position.get('territoryML', 0):.0f} mL", tumours or "–",
                         ("⚠ " + ", ".join(f"{f['distanceMM']:.0f} mm" for f in outside)) if outside else "–",
                         (f"{position['outsideFovPercent']:.0f}% outside FOV"
                          if position.get("outsideFovPercent", 0) >= 5 else "")]
            for column, text in enumerate(cells):
                self.resultsTable.setItem(row, column, qt.QTableWidgetItem(text))
        self.resultsTable.resizeColumnsToContents()
        if not results:
            self.resultsLabel.text = ""
            return
        issues = W.planningFindings(results, self.store.GetParameter(P_PLANNING_REVIEWED) == results.get("key"))
        lines = [f"Rule: {C.RULE_LABELS.get(results.get('rule'), '')} · liver in FOV "
                 f"{100 * (results.get('liverCoverage') or 0):.0f}%"]
        maa = results.get("maa")
        if maa:
            lines.append(f"MAA ({maa['image']}, ≥ {maa['percent']:g}%): Dice {maa['dice']:.2f}, "
                         f"{100 * maa['countsFraction']:.0f}% of the liver counts inside the territories.")
        for item in results.get("enhancement", []):
            lines.append(f"Enhancement ({item['image']}, {item['percent']:g}%): {item['volumeML']:.0f} mL")
        symbols = {W.SEVERITY_ERROR: "✕", W.SEVERITY_WARNING: "⚠", W.SEVERITY_INFO: "ℹ"}
        lines += [f"{symbols[i.severity]} {i.text}" for i in issues]
        self.resultsLabel.text = "<br>".join(lines)

    # -- Actions --

    def _dsa(self):
        """The 3D rendering choices (stored in the settings as they are)."""
        return dict(dsaStartPercentile=float(self.dsaStartSpin.value), dsaTopPercentile=float(self.dsaTopSpin.value),
                    dsaMode=C.DSA_MODE_COMPOSITE if self.compositeButton.checked else C.DSA_MODE_MIP,
                    shadedStartPercentile=float(self.shadedStartSpin.value),
                    shadedMidPercentile=float(self.shadedMidSpin.value),
                    shadedTopPercentile=float(self.shadedTopSpin.value),
                    negative=bool(self.negativeButton.checked))

    def onDsaChanged(self, *args):
        """Grey ramp of the 3D DSA-like view changed: store it and re-render (the masked CBCT is kept)."""
        self._onSettingsChanged()
        if self._updating or self.session is None:
            return
        try:
            C.renderMaskedContrast(self.session, C.threeDViewNode(), self._dsa())
            C.updateCarmAnnotation()
            C.overlayCenterlines(self.store)   # line colour follows the rendering mode
        except Exception as e:
            logging.warning(f"CBCT planning: could not update the 3D view: {e}")

    def _onAdvancedChanged(self, *args):
        if self._updating:
            return
        values = {name: float(spin.value) for name, spin in self.advancedSpins.items()
                  if abs(spin.value - V.DEFAULT_PARAMETERS[name]) > 1e-9}
        if self.denoiseCombo.currentData != V.DEFAULT_DENOISE_METHOD:
            values["DENOISE_METHOD"] = self.denoiseCombo.currentData
        C.setSettings(self.store, advanced=values)
        C.applyAdvanced(self.store)
        self.statusLabel.text = "Advanced settings changed: click 'Extract arterial tree' to use them."

    def onResetAdvanced(self):
        self._updating = True
        try:
            for name, spin in self.advancedSpins.items():
                spin.value = float(V.DEFAULT_PARAMETERS[name])
            self.denoiseCombo.setCurrentIndex(self.denoiseCombo.findData(V.DEFAULT_DENOISE_METHOD))
        finally:
            self._updating = False
        self._onAdvancedChanged()

    def onThreeDToggled(self, key, checked):
        if self._updating:
            return
        try:
            C.setThreeDVisible(self.store, self.segmentationSelector.currentNode(), key, checked)
        except Exception as e:
            logging.warning(f"CBCT planning: could not change the 3D view: {e}")

    def onShowLayout(self):
        self._showLayout(resetView=True)

    def onRegister(self):
        from TaranisLib.toolbar import openHub
        openHub(W.STEP_REGISTRATION)

    def onPlaceInjection(self):
        node = C.markupsNode(self.store, REF_PLANNING_INJECTION, create=True)
        node.RemoveAllControlPoints()
        C.startPlacing(node)
        self.statusLabel.text = "Click the catheter tip (inside the contrast-filled artery) in a slice view."

    def onAddTip(self):
        node = C.markupsNode(self.store, REF_PLANNING_TIPS, create=True)
        self._observeTips()
        C.startPlacing(node)
        self.statusLabel.text = "Click on an artery of the tree for the planned catheter tip."

    def onRemoveTip(self):
        """Remove the tip selected in the table, with its territory segment (candidate or accepted) and its
        downstream path in the 3D view."""
        node = C.markupsNode(self.store, REF_PLANNING_TIPS)
        row = self.resultsTable.currentRow()
        if node is None or row < 0 or row >= node.GetNumberOfControlPoints():
            slicer.util.warningDisplay("Select a tip in the table first.")
            return
        label = node.GetNthControlPointLabel(row)
        removed = C.removeTipResults(self.store, self.segmentationSelector.currentNode(), label)
        node.RemoveNthControlPoint(row)
        self._refresh()
        self._updateController()
        self.statusLabel.text = f"Tip {label} removed" + (f" with its territory '{removed}'." if removed else ".")

    def onExtractTree(self):
        with self._busy("Preparing the working grid"):
            self._onSettingsChanged()
            session = self._getSession()
            tree = session.extractTree(float(self.vesselSdSpin.value), float(self.growSdSpin.value),
                                       self.tubularCheck.checked, self.searchMaskCheck.checked)
            self._showLayout(resetView=True)
            self.statusLabel.text = (f"Arterial tree extracted: {tree.lengthMM() / 10.0:.0f} cm of centerline, "
                                     f"{len(tree.branchPoints())} branch points. Review it in the views; edit the "
                                     "segment if veins, bone or the catheter are included.")

    def onShowFiltered(self):
        with self._busy("Filtering the CBCT"):
            session = self._getSession()
            node = C.exportFilteredImages(session)
            self._showLayout(resetView=False)
            if node is not None:
                slicer.util.setSliceViewerLayers(background=node, fit=True)
            self.statusLabel.text = ("Filtered images added: 'CBCT local contrast' is shown (window 0-9: white = 9 SD "
                                     "or more above the liver around). Vessels should stand out against a uniform "
                                     "grey liver. The tube-shape image appears after 'Extract arterial tree' with "
                                     "the tube check on.")

    def onShowSearchMask(self, checked):
        self._onSettingsChanged()
        if not C.setSearchMaskVisible(self.store, checked) and checked:
            self.statusLabel.text = "The search mask appears with the next 'Extract arterial tree'."

    def onRecomputeCenterlines(self):
        with self._busy("Reading the arterial tree"):
            tree = self._getSession().buildCenterlines()
            self.statusLabel.text = f"Centerlines recomputed: {len(tree.branchPoints())} branch points, " \
                                    f"{tree.loops} loop(s)."

    def onEditTree(self):
        node = C.treeNode(self.store)
        if node is None:
            slicer.util.warningDisplay("Extract the arterial tree first.")
            return
        slicer.util.selectModule("SegmentEditor")
        editor = slicer.modules.segmenteditor.widgetRepresentation().self().editor
        editor.setSegmentationNode(node)
        cbct = self.cbctSelector.currentNode()
        if cbct is not None:
            if hasattr(editor, "setSourceVolumeNode"):
                editor.setSourceVolumeNode(cbct)
            else:
                editor.setMasterVolumeNode(cbct)

    def onPredict(self):
        self._predict(quiet=False)

    def _predict(self, quiet):
        if quiet:
            try:
                self._predictNow()
            except Exception as e:
                logging.warning(f"CBCT planning: automatic prediction failed: {e}")
                self.statusLabel.text = f"<span style='color:#dc2626'>{e}</span>"
            self._refresh()
            return
        with self._busy("Predicting the territories"):
            self._predictNow()

    def _predictNow(self):
        self._onSettingsChanged()
        session = self._getSession()
        results = C.predictTerritories(session, self.ruleCombo.currentData, self.subtractCheck.checked,
                                       uptakeNode=self.uptakeSelector.currentNode())
        territories = [p for p in results["positions"] if p.get("territoryML") is not None]
        self.statusLabel.text = (f"{len(territories)} territory candidate(s) added to the segmentation. Review "
                                 "them, then accept them in the Taranis Segmentation step (they become perfused "
                                 "volumes for dosimetry).")
        self._updateController()
        self._showLayout(resetView=False)   # the territory candidates in 3D

    def onAcceptTerritory(self, allTerritories=False):
        segmentationNode = self.segmentationSelector.currentNode()
        results = planningResults(self.store) or {}
        if segmentationNode is None or not results:
            slicer.util.warningDisplay("Predict the territories first.")
            return
        if allTerritories:
            segmentIDList = list(results.get("segmentIDs", []))
        else:
            row = self.resultsTable.currentRow()
            item = self.resultsTable.item(row, 0) if row >= 0 else None
            label = item.text() if item is not None else None
            segmentIDList = [p.get("segmentID") for p in results.get("positions", []) if p.get("label") == label]
            if not label or not any(segmentIDList):
                slicer.util.warningDisplay("Select a tip with a predicted territory in the table.")
                return
        accepted = C.acceptTerritories(segmentationNode, [i for i in segmentIDList if i])
        self._updateController()
        self._showLayout(resetView=False)
        self.statusLabel.text = (("Accepted as perfused volume(s): " + ", ".join(f"'{n}'" for n in accepted) + ".")
                                 if accepted else "Nothing to accept: the territories are already accepted.")

    def onMarkReviewed(self):
        C.markReviewed(self.store)
        self._refresh()
        self._updateController()

    def onFindFeeders(self):
        tumourID = self.tumourCombo.currentData
        if not tumourID:
            slicer.util.warningDisplay("The segmentation has no tumour segment.")
            return
        with self._busy("Finding the feeding branches"):
            name, candidates = self._getSession().feeders(tumourID, self.ruleCombo.currentData)
            self._feeders = candidates
            self.feedersTable.setRowCount(len(candidates))
            labels = {"branch": "Feeding branch", "all feeders": "All feeders (most selective)",
                      "upstream": "Upstream branch"}
            for row, c in enumerate(candidates):
                cells = [labels.get(c["kind"], c["kind"]), f"{c['upstreamMM']:.0f} mm", f"{c['territoryML']:.0f} mL",
                         f"{c['coveragePercent']:.0f}%", f"{c['normalML']:.0f} mL"]
                for column, text in enumerate(cells):
                    self.feedersTable.setItem(row, column, qt.QTableWidgetItem(text))
            self.feedersTable.resizeColumnsToContents()
            self.feedersLabel.text = (f"'{name}': no branch of the tree reaches it (within {V.FEEDER_MARGIN_MM:g} "
                                      "mm)." if not candidates else
                                      f"'{name}': {len(candidates)} candidate position(s), most selective first.")

    def onFeederSelected(self):
        row = self.feedersTable.currentRow()
        try:
            C.showCandidatePoint(self._feeders[row]["ras"] if 0 <= row < len(self._feeders) else None)
        except Exception as e:
            logging.warning(f"CBCT planning: could not show the position: {e}")

    def onUseFeeder(self):
        row = self.feedersTable.currentRow()
        if row < 0 or row >= len(self._feeders):
            slicer.util.warningDisplay("Select a position in the table.")
            return
        node = C.markupsNode(self.store, REF_PLANNING_TIPS, create=True)
        self._observeTips()
        node.AddControlPointWorld(vtk.vtkVector3d(*self._feeders[row]["ras"]))
        self.statusLabel.text = "Tip added: click 'Predict territories'."

    def _enhancementImage(self):
        node = self.enhancementSelector.currentNode() or self.cbctSelector.currentNode()
        if node is None:
            raise ValueError("Select the CBCT image.")
        return node

    def onAutoEnhancement(self):
        with self._busy("Automatic threshold"):
            session = self._getSession()
            self.enhancementSlider.value = round(C.enhancementPercentAuto(session, self._enhancementImage()))
            self._onSettingsChanged()

    def onEnhancement(self):
        with self._busy("Perfused volume from enhancement"):
            self._onSettingsChanged()
            _, message = C.enhancementTerritory(self._getSession(), self._enhancementImage(),
                                                float(self.enhancementSlider.value))
            self.statusLabel.text = message
            self._updateController()

    def onSkipToggled(self, checked):
        case = TaranisCase.find()
        if case is not None and not self._updating:
            case.setFlag(P_PLANNING_SKIPPED, checked)

    def onBackToHub(self):
        from TaranisLib.toolbar import openHub
        openHub(W.STEP_PLANNING if TaranisCase.find() is not None else None)

    def _updateController(self):
        try:
            WorkflowController.instance().scheduleUpdate()
        except Exception:
            pass


# ---------------------------------------------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------------------------------------------

class CBCTPlanningTest(ScriptedLoadableModuleTest):
    """Synthetic CBCT with a Y-shaped arterial tree in an ellipsoid liver: tree, tip, territory candidate."""

    def setUp(self):
        slicer.mrmlScene.Clear(0)

    def runTest(self):
        self.setUp()
        self.test_pipeline()

    def _phantom(self):
        shape = (60, 90, 100)
        k, j, i = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
        p = np.stack([k, j, i], axis=-1).astype(float)

        def distance(a, b):
            a, b = np.asarray(a, float), np.asarray(b, float)
            t = np.clip(((p - a) @ (b - a)) / float((b - a) @ (b - a)), 0.0, 1.0)
            return np.sqrt(((p - (a + t[..., None] * (b - a))) ** 2).sum(axis=-1))

        segments = [((30, 5, 50), (30, 30, 50)), ((30, 30, 50), (30, 75, 20)), ((30, 30, 50), (30, 75, 80))]
        vessels = np.min([distance(a, b) for a, b in segments], axis=0) <= 2.0
        liver = (((k - 30) / 25.0) ** 2 + ((j - 52) / 35.0) ** 2 + ((i - 50) / 45.0) ** 2) <= 1.0
        rng = np.random.default_rng(0)
        values = np.where(liver, 100.0, 60.0) + rng.normal(0, 5, shape)   # liver in fat
        values[vessels] = 600.0
        return values.astype(np.int16), liver

    def test_pipeline(self):
        values, liver = self._phantom()
        cbct = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "CBCT test")
        slicer.util.updateVolumeFromArray(cbct, values)
        labelmap = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "liver")
        slicer.util.updateVolumeFromArray(labelmap, liver.astype(np.uint8))
        segmentation = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "Segmentation")
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelmap, segmentation)
        liverID = segmentIDs(segmentation)[0]
        segmentation.GetSegmentation().GetSegment(liverID).SetName("Whole liver")
        store = C.storeNode()
        injection = C.markupsNode(store, REF_PLANNING_INJECTION, create=True)
        injection.AddControlPointWorld(vtk.vtkVector3d(50.0, 5.0, 30.0))   # (i, j, k) with unit spacing
        tips = C.markupsNode(store, REF_PLANNING_TIPS, create=True)
        tips.AddControlPointWorld(vtk.vtkVector3d(40.0, 47.0, 30.0))      # left branch
        session = C.PlanningSession(cbct, segmentation, liverID, 1.0, store)
        try:
            tree = session.extractTree()
            self.assertGreater(tree.count, 50)
            results = C.predictTerritories(session)
            position = results["positions"][0]
            self.assertTrue(position["snapped"])
            liverML = float(liver.sum()) / 1000.0
            self.assertGreater(position["territoryML"], 0.2 * liverML)
            self.assertLess(position["territoryML"], 0.7 * liverML)
            candidate = segmentation.GetSegmentation().GetSegment(position["segmentID"])
            self.assertIsNotNone(candidate)
            self.assertTrue(isCandidate(candidate))
        finally:
            session.close()
        self.delayDisplay("CBCT planning test passed")
