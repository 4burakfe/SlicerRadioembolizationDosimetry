import os
import numpy as np
import slicer
from slicer.ScriptedLoadableModule import *
import logging
import qt
import ctk
import vtk

class LSFcalc(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "Taranis - LSF Calculator"
        parent.categories = ["Nuclear Medicine"]
        parent.dependencies = []
        parent.contributors = ["Burak Demir, MD, FEBNM"]
        parent.helpText = """
        This module calculates lung shunt fraction before radioembolization treatment.
        """
        parent.acknowledgementText = """
        This file was developed by Burak Demir.
        """
        # **✅ Set the module icon**
        # **✅ Set the module icon**
        iconPath = os.path.join(os.path.dirname(__file__), "Resources\\taranis_logo.png")
        self.parent.icon = qt.QIcon(iconPath)  # Assign icon to the module
        self.parent = parent

class LSFcalcWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        # Create a collapsible button for parameters
        parametersCollapsibleButton = ctk.ctkCollapsibleButton()
        parametersCollapsibleButton.text = "Parameters"
        self.layout.addWidget(parametersCollapsibleButton)

        # Form layout within the collapsible button
        formLayout = qt.QFormLayout(parametersCollapsibleButton)
        
        # **✅ Load the banner image**
        moduleDir = os.path.dirname(__file__)  # Get module directory
        bannerPath = os.path.join(moduleDir, "Resources\\banner.png")  # Change to your banner file

        if os.path.exists(bannerPath):
            bannerLabel = qt.QLabel()
            bannerPixmap = qt.QPixmap(bannerPath)  # Load image
            bannerLabel.setPixmap(bannerPixmap.scaledToWidth(400, qt.Qt.SmoothTransformation))  # Adjust width

            # **Center the image**
            bannerLabel.setAlignment(qt.Qt.AlignCenter)

            # **Add to layout**
            self.layout.addWidget(bannerLabel)
        else:
            print(f"❌ WARNING: Banner file not found at {bannerPath}")

        
        # Input SPECT Volume Selector
        self.spectSelector = slicer.qMRMLNodeComboBox()
        self.spectSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.spectSelector.selectNodeUponCreation = True
        self.spectSelector.addEnabled = False
        self.spectSelector.removeEnabled = False
        self.spectSelector.noneEnabled = False
        self.spectSelector.showHidden = False
        self.spectSelector.showChildNodeTypes = False
        self.spectSelector.setMRMLScene(slicer.mrmlScene)
        self.spectSelector.setToolTip("Select the input SPECT/PET volume.")
        formLayout.addRow("Input SPECT/PET Volume: ", self.spectSelector)

        # Segmentation Selector
        self.segmentationSelector = slicer.qMRMLNodeComboBox()
        self.segmentationSelector.nodeTypes = ["vtkMRMLSegmentationNode"]
        self.segmentationSelector.selectNodeUponCreation = True
        self.segmentationSelector.addEnabled = False
        self.segmentationSelector.removeEnabled = False
        self.segmentationSelector.noneEnabled = False
        self.segmentationSelector.showHidden = False
        self.segmentationSelector.showChildNodeTypes = False
        self.segmentationSelector.setMRMLScene(slicer.mrmlScene)
        self.segmentationSelector.setToolTip("Select the lung segment.")
        formLayout.addRow("Segmentation: ", self.segmentationSelector)
        # Lung Segment Selector
        self.lungSegmentSelector = slicer.qMRMLSegmentSelectorWidget()
        self.lungSegmentSelector.setMRMLScene(slicer.mrmlScene)
        self.lungSegmentSelector.setToolTip("Select the segment representing the Lungs.")
        formLayout.addRow("Lung Segment: ", self.lungSegmentSelector)

        # Connect segmentation selector to update lung segment selector
        self.segmentationSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSegmentationNodeChanged)

        # Whole Liver Segment Selector
        self.liverSegmentSelector = slicer.qMRMLSegmentSelectorWidget()
        self.liverSegmentSelector.setMRMLScene(slicer.mrmlScene)
        self.liverSegmentSelector.setToolTip("Select the segment representing the whole liver.")
        formLayout.addRow("Whole Liver Segment: ", self.liverSegmentSelector)

        # Connect segmentation selector to update liver segment selector
        self.segmentationSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSegmentationNodeChanged2)
        
        # Calculate Button
        self.calculateButton = qt.QPushButton("Calculate")
        self.calculateButton.toolTip = "Perform dosimetric calculations."
        formLayout.addRow(self.calculateButton)

        # Total Activity Text Box
        self.lungTextBox = qt.QLineEdit()
        self.lungTextBox.setReadOnly(True)
        self.lungTextBox.setToolTip("Lung counts")
        formLayout.addRow("Lung counts: ", self.lungTextBox)

        # Total Activity Text Box
        self.liverTextBox = qt.QLineEdit()
        self.liverTextBox.setReadOnly(True)
        self.liverTextBox.setToolTip("Liver counts")
        formLayout.addRow("Liver counts: ", self.liverTextBox)

        # Total Activity Text Box
        self.lsfTextBox = qt.QLineEdit()
        self.lsfTextBox.setReadOnly(True)
        self.lsfTextBox.setToolTip("LSF")
        formLayout.addRow("Lung Shunt Fraction: ", self.lsfTextBox)

        # Connections
        self.calculateButton.connect('clicked(bool)', self.onCalculateButton)

        # Add vertical spacer
        self.layout.addStretch(1)
        infoTextBox = qt.QTextEdit()
        infoTextBox.setReadOnly(True)  # Make the text box read-only
        infoTextBox.setPlainText(
            "This module enables predictive lung shunt fraction calculation with SPECT images.\n"
            "This module is NOT a medical device. It is for research purposes only.\n"
            "Prepared by: Burak Demir, MD, FEBNM \n"
            "For support, feedback, and suggestions: 4burakfe@gmail.com\n"
            "Version: alpha v1.1"
        )
        infoTextBox.setToolTip("Module information and instructions.")  # Add a tooltip for additional help
        self.layout.addWidget(infoTextBox)

    def onSegmentationNodeChanged(self, node):
        self.lungSegmentSelector.setSegmentationNode(node)
    def onSegmentationNodeChanged2(self, node):
        self.liverSegmentSelector.setSegmentationNode(node)
        

    def onCalculateButton(self):
        spectVolumeNode = self.spectSelector.currentNode()
        segmentationNode = self.segmentationSelector.currentNode()
        liverSegmentID = self.liverSegmentSelector.currentSegmentID()  # Retrieve the selected liver segment ID
        lungSegmentID = self.lungSegmentSelector.currentSegmentID()  # Retrieve the selected liver segment ID

        if not spectVolumeNode or not segmentationNode:
            slicer.util.errorDisplay("Please select valid input nodes.")
            return

        # Perform dosimetric calculations
        logic = LSFcalcLogic()
        logic.calculateDose(spectVolumeNode, segmentationNode, lungSegmentID, liverSegmentID, self.lungTextBox,self.liverTextBox,self.lsfTextBox)

class LSFcalcLogic(ScriptedLoadableModuleLogic):
    def calculateDose(self, spectVolumeNode, segmentationNode, lungSegmentID, liverSegmentID, lungTextBox,liverTextBox,lsfTextBox):
        """
        Perform dosimetric calculations using the given inputs.
        """
        logging.info("Starting dosimetric calculations.")

        # Validate inputs
        if not spectVolumeNode or not segmentationNode :
            raise ValueError("Invalid inputs. Please select valid nodes.")

        # Get input volume array
        spectArray = slicer.util.arrayFromVolume(spectVolumeNode)
        if spectArray is None:
            raise ValueError("Unable to access data from the input SPECT volume.")

        
        # Calculate mean dose for each segment
        liverlabelMapVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
        slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(
            segmentationNode, [liverSegmentID], liverlabelMapVolumeNode, spectVolumeNode
        )
 
        # Mask dose array to calculate liver
        liverlabelMapArray = slicer.util.arrayFromVolume(liverlabelMapVolumeNode)
        maskedDoseArray = spectArray[liverlabelMapArray == 1]
        livercounts = np.sum(maskedDoseArray)



        lunglabelMapVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
        slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(
            segmentationNode, [lungSegmentID], lunglabelMapVolumeNode, spectVolumeNode
        )

        # Mask dose array to calculate lung
        lunglabelMapArray = slicer.util.arrayFromVolume(lunglabelMapVolumeNode)
        maskedDoseArray2 = spectArray[lunglabelMapArray == 1]
        lungcounts = np.sum(maskedDoseArray2)        

        lsf = (lungcounts/(lungcounts+livercounts))*100

        lungTextBox.setText(f"{lungcounts:.2f}")
        liverTextBox.setText(f"{livercounts:.2f}")
        lsfTextBox.setText(f"{lsf:.2f}%")

        logging.info("Dosimetric calculations completed.")
        return 1