import os
import numpy as np
import slicer
from slicer.ScriptedLoadableModule import *
import logging
import qt
import ctk
import vtk

class RadioembolizationDosimetry(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "Taranis - Dosimetry (Patient Relative)"
        parent.categories = ["Nuclear Medicine"]
        parent.dependencies = []
        parent.contributors = ["Burak Demir, MD, FEBNM"]
        parent.helpText = """
        This module calculates a dosimetry model for radioembolization using SPECT and PET images.
        """
        parent.acknowledgementText = """
        This file was developed by Burak Demir.
        """
        # **✅ Set the module icon**
        iconPath = os.path.join(os.path.dirname(__file__), "Resources\\Icons\\RadioembolizationDosimetry.png")
        self.parent.icon = qt.QIcon(iconPath)  # Assign icon to the module
        self.parent = parent

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)

#
# Register sample data sets in Sample Data module
#


def registerSampleData():
    """Add data sets to Sample Data module."""
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData

    iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # RadioembolizationDosimetry1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="RadioembolizationDosimetry",
        sampleName="RadioembolizationDosimetry1",
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, "RadioembolizationDosimetry1.png"),
        # Download URL and target file name
        uris=["https://github.com/4burakfe/SlicerRadioembolizationDosimetry_SampleImages/releases/download/TestImages/patient_1_MRI.nrrd",
              "https://github.com/4burakfe/SlicerRadioembolizationDosimetry_SampleImages/releases/download/TestImages/patient_1_Y90_PET.nrrd",
              "https://github.com/4burakfe/SlicerRadioembolizationDosimetry_SampleImages/releases/download/TestImages/patient_1_Segmentation.seg.nrrd"],
        fileNames=["patient_1_MRI.nrrd", "patient_1_Y90_PET.nrrd", "patient_1_Segmentation.seg.nrrd"],
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums=["SHA256:e2c598ae76d85e0b2cc0ebfd643d4f5ebda1d6f3df632c9172696878b858dfbe",
                   "SHA256:4f1f195ccb0dcd3c4c9fc967ed0c2e4bf9ac5985db02d951d64772d69979e55b",
                   "SHA256:550ceea296c7eab81f1dc7fb4ccf2f647fe223ba310eb2bd2dd0d0050aca739b"],
        # This node name will be used when the data set is loaded
        nodeNames=["Patient 1 MRI",
                   "Patient 1 Y90 PET",
                   "Patient 1 Segmentation"],
    )

    # RadioembolizationDosimetry2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="RadioembolizationDosimetry",
        sampleName="RadioembolizationDosimetry2",
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, "RadioembolizationDosimetry2.png"),
        # Download URL and target file name
        uris=["https://github.com/4burakfe/SlicerRadioembolizationDosimetry_SampleImages/releases/download/TestImages/patient_2_CT.nrrd",
              "https://github.com/4burakfe/SlicerRadioembolizationDosimetry_SampleImages/releases/download/TestImages/patient_2_SPECT.nrrd",
              "https://github.com/4burakfe/SlicerRadioembolizationDosimetry_SampleImages/releases/download/TestImages/patient_2_Segmentation.seg.nrrd"],
        fileNames=["patient_2_CT.nrrd", "patient_2_SPECT.nrrd", "patient_2_Segmentation.seg.nrrd"],
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums=["SHA256:99600e480dc6e5353953377dbe66f232d8eae28bfa50aa7a61a4f83574ccde47",
                   "SHA256:5647f8109babdc8655b217da6952cd5aafde3df2598a0dab52b2ad60208dc86b",
                   "SHA256:49e0eae32ad99464a66ae1fab48a9343d9068b5633e5a6f260640bc9b5666abe"],
        # This node name will be used when the data set is loaded
        nodeNames=["Patient 2 CT",
                   "Patient 2 SPECT",
                   "Patient 2 Segmentation"],
    )


class RadioembolizationDosimetryWidget(ScriptedLoadableModuleWidget):
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
        bannerPath = os.path.join(moduleDir, "Resources\\Icons\\RadioembolizationDosimetryabs.png")  # Change to your banner file

        if os.path.exists(bannerPath):
            bannerLabel = qt.QLabel()
            bannerPixmap = qt.QPixmap(bannerPath)  # Load image
            bannerLabel.setPixmap(bannerPixmap.scaledToWidth(500, qt.Qt.SmoothTransformation))  # Adjust width

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
        self.segmentationSelector.setToolTip("Select the segmentation for dosimetric calculations.")
        formLayout.addRow("Segmentation: ", self.segmentationSelector)

        # Whole Liver Segment Selector
        self.liverSegmentSelector = slicer.qMRMLSegmentSelectorWidget()
        self.liverSegmentSelector.setMRMLScene(slicer.mrmlScene)
        self.liverSegmentSelector.setToolTip("Select the segment representing the whole liver.")
        formLayout.addRow("Whole Liver Segment: ", self.liverSegmentSelector)

        # Connect segmentation selector to update liver segment selector
        self.segmentationSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSegmentationNodeChanged)

        # Activity Slider and Input Box
        self.activitySlider = ctk.ctkSliderWidget()
        self.activitySlider.singleStep = 1
        self.activitySlider.minimum = 0
        self.activitySlider.maximum = 10000
        self.activitySlider.value = 0
        self.activitySlider.setToolTip("Specify the desired activity in MBq.")
        formLayout.addRow("Y-90 Activity (MBq): ", self.activitySlider)

        # Lung Shunt Fraction Slider
        self.lungShuntSlider = ctk.ctkSliderWidget()
        self.lungShuntSlider.singleStep = 1
        self.lungShuntSlider.minimum = 0
        self.lungShuntSlider.maximum = 100
        self.lungShuntSlider.value = 0
        self.lungShuntSlider.setToolTip("Specify the lung shunt fraction as a percentage.")
        formLayout.addRow("Lung Shunt Fraction (%): ", self.lungShuntSlider)

        # Conversion Factor (Gy/MBq/g)
        self.conversionFactorSpinBox = qt.QDoubleSpinBox()
        self.conversionFactorSpinBox.setRange(0.0, 100.0)
        self.conversionFactorSpinBox.setValue(49.67)
        self.conversionFactorSpinBox.setSingleStep(0.1)
        formLayout.addRow("Conversion Factor (Gy/MBq/g):", self.conversionFactorSpinBox)

        # Lung Mass (g)
        self.lungMassSpinBox = qt.QDoubleSpinBox()
        self.lungMassSpinBox.setRange(0.0, 5000.0)
        self.lungMassSpinBox.setValue(1000.0)
        self.lungMassSpinBox.setSingleStep(50.0)
        formLayout.addRow("Lung Mass (g):", self.lungMassSpinBox)

        # Liver Tissue Density (g/mL)
        self.liverDensitySpinBox = qt.QDoubleSpinBox()
        self.liverDensitySpinBox.setRange(0.0, 10.0)
        self.liverDensitySpinBox.setValue(1.05)
        self.liverDensitySpinBox.setSingleStep(0.01)
        formLayout.addRow("Liver Density (g/mL):", self.liverDensitySpinBox)

        # Output Volume Selector
        self.outputVolumeSelector = slicer.qMRMLNodeComboBox()
        self.outputVolumeSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.outputVolumeSelector.selectNodeUponCreation = True
        self.outputVolumeSelector.addEnabled = True
        self.outputVolumeSelector.removeEnabled = True
        self.outputVolumeSelector.noneEnabled = True
        self.outputVolumeSelector.showHidden = False
        self.outputVolumeSelector.showChildNodeTypes = False
        self.outputVolumeSelector.setMRMLScene(slicer.mrmlScene)
        self.outputVolumeSelector.setToolTip("Select the output volume for the Gy maps.")
        formLayout.addRow("Output Volume: ", self.outputVolumeSelector)

        # Calculate Button
        self.calculateButton = qt.QPushButton("Calculate with the desired activity")
        self.calculateButton.toolTip = "Perform dosimetric calculations."
        formLayout.addRow(self.calculateButton)

        self.targetdoseSlider = ctk.ctkSliderWidget()
        self.targetdoseSlider.singleStep = 1
        self.targetdoseSlider.minimum = 0
        self.targetdoseSlider.maximum = 1000
        self.targetdoseSlider.value = 0
        formLayout.addRow("Target dose for segment(Gy):", self.targetdoseSlider)

        self.targetSegmentSelector = slicer.qMRMLSegmentSelectorWidget()
        self.targetSegmentSelector.setMRMLScene(slicer.mrmlScene)
        self.targetSegmentSelector.setToolTip("Select the segment representing the tumor.")
        formLayout.addRow("Target Tissue: ", self.targetSegmentSelector)
       
        # Calculate Button
        self.calculateButtonlim = qt.QPushButton("Calculate the activity for target dose")
        self.calculateButtonlim.toolTip = "Perform dosimetric calculations."
        formLayout.addRow(self.calculateButtonlim)

        # Segment Dose Table
        self.segmentDoseTable = qt.QTableWidget()
        self.segmentDoseTable.setColumnCount(4)
        self.segmentDoseTable.setHorizontalHeaderLabels(["Segment", "Dose (Gy)","Volume (mL)","Activity (MBq)"])
        self.segmentDoseTable.setFixedSize(420,350)
        formLayout.addRow("Segment Doses: ", self.segmentDoseTable)

        # Save Report Button
        self.saveReportButton = qt.QPushButton("Save Report as RTF")
        self.saveReportButton.toolTip = "Export dosimetry results to a report file."
        formLayout.addRow(self.saveReportButton)
        self.saveReportButton.connect('clicked(bool)', self.onSaveReportClicked)

        # Connections
        self.calculateButton.connect('clicked(bool)', self.onCalculateButton)
        self.calculateButtonlim.connect('clicked(bool)', self.limonCalculateButton)

        # Add vertical spacer
        self.layout.addStretch(1)
        infoTextBox = qt.QTextEdit()
        infoTextBox.setReadOnly(True)  # Make the text box read-only
        infoTextBox.setPlainText(
            "This module enables predictive dosimetry with SPECT and PET images.\n"
            "This module is NOT a medical device. It is for research purposes only.\n"
            "Default conversion factor is for Y-90 which equals to 49.67 J/GBq\n"
            "Conversion factor for Ho-166 is 14.85 J/GBq\n"         
            "Written by: Burak Demir, MD, FEBNM \n"
            "For support, feedback, and suggestions: 4burakfe@gmail.com\n"
        )
        infoTextBox.setToolTip("Module information and instructions.")  # Add a tooltip for additional help
        self.layout.addWidget(infoTextBox)




        
    def onSegmentationNodeChanged(self, node):
        self.liverSegmentSelector.setCurrentNode(node)

    def onCalculateButton(self):
        spectVolumeNode = self.spectSelector.currentNode()
        segmentationNode = self.segmentationSelector.currentNode()
        liverSegmentID = self.liverSegmentSelector.currentSegmentID()  # Retrieve the selected liver segment ID
        activityMBq = self.activitySlider.value
        outputVolumeNode = self.outputVolumeSelector.currentNode()
        lungShuntFractionPercent = self.lungShuntSlider.value

        sliceWidget = slicer.app.layoutManager().sliceWidget("Red")
        sliceCompositeNode = sliceWidget.mrmlSliceCompositeNode()
        backgroundVolumeid = slicer.app.layoutManager().sliceWidget("Red").mrmlSliceCompositeNode().GetBackgroundVolumeID()

        missingInputs = []
        if not spectVolumeNode:
            missingInputs.append("Input SPECT/PET Volume")
        if not segmentationNode:
            missingInputs.append("Segmentation")
        if not liverSegmentID:
            missingInputs.append("Whole Liver Segment")
        if not outputVolumeNode:
            missingInputs.append("Output Volume")
        if missingInputs:
            slicer.util.errorDisplay("Please select " + ", ".join(missingInputs) + ".")
            return

        # Perform dosimetric calculations
        self.calculateDose(spectVolumeNode, segmentationNode, liverSegmentID, activityMBq, outputVolumeNode, lungShuntFractionPercent)
        self.outputVolumeSelector.currentNode().SetName("Dose Map (Gy)")
        foregroundVolumeid = self.outputVolumeSelector.currentNode().GetID()
        sliceCompositeNode.SetBackgroundVolumeID(backgroundVolumeid)
        sliceCompositeNode.SetForegroundVolumeID(foregroundVolumeid)
        sliceCompositeNode.SetForegroundOpacity(0.5)
            
    def calculateDose(self, spectVolumeNode, segmentationNode, liverSegmentID, activityMBq, outputVolumeNode, lungShuntFractionPercent):
        """
        Perform dosimetric calculations using the given inputs.
        """
        logging.info("Starting dosimetric calculations.")

        # Validate inputs
        if not spectVolumeNode or not segmentationNode or not liverSegmentID or not outputVolumeNode:
            raise ValueError("Invalid inputs. Please select valid nodes and ensure the liver segment is specified.")

        # Export the liver segment to a label map aligned with the input volume
        labelMapVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
        slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(
            segmentationNode, [liverSegmentID], labelMapVolumeNode, spectVolumeNode
        )

        # Apply the label map to mask the input volume
        maskedVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "MaskedVolume")
        parameters = {
            "InputVolume": spectVolumeNode.GetID(),
            "MaskVolume": labelMapVolumeNode.GetID(),
            "OutputVolume": maskedVolumeNode.GetID(),
            "Label": 1,  # Label value corresponding to the liver segment
            "Replace": 0  # Set voxels outside the segment to 0
        }
        slicer.cli.runSync(slicer.modules.maskscalarvolume, None, parameters)

        # Clone the masked volume to the output volume to inherit spatial properties
        volumesLogic = slicer.modules.volumes.logic()
        clonedOutputVolumeNode = volumesLogic.CloneVolume(slicer.mrmlScene, maskedVolumeNode, outputVolumeNode.GetName())
        outputVolumeNode.Copy(clonedOutputVolumeNode)
        outputVolumeNode.SetAttribute("DicomRtImport.DoseVolume", "1")
        slicer.mrmlScene.RemoveNode(clonedOutputVolumeNode)

        # Get the masked volume array
        maskedArray = slicer.util.arrayFromVolume(maskedVolumeNode)
        if maskedArray is None:
            raise ValueError("Unable to access data from the masked volume.")

        # Calculate total volume in mL
        spacing = spectVolumeNode.GetSpacing()  # spacing is in mm
        voxelVolumeML = (spacing[0] * spacing[1] * spacing[2]) / 1000.0  # convert mm^3 to mL
        totalVolumeML = maskedArray.size * voxelVolumeML

        # Adjust activity based on lung shunt fraction
        lungShuntFraction = lungShuntFractionPercent / 100.0
        ncorr_activityMBq = activityMBq
        activityMBq = activityMBq * (1 - lungShuntFraction)
        if totalVolumeML == 0:
            raise ValueError("Total volume is zero. Ensure the liver segment is correctly defined.")

        # Calculate mean input value within the liver segment
        meanInputValue = np.sum(maskedArray) / maskedArray.size

        # Constants for Y-90 dosimetry
        conversionFactor = self.conversionFactorSpinBox.value
        lungMassg = self.lungMassSpinBox.value
        densityGPerML = self.liverDensitySpinBox.value

        # Calculate mean output dose
        meanOutputDoseGy = (activityMBq / (totalVolumeML * densityGPerML)) * conversionFactor

        # Rescale factor to normalize dose
        rescaleFactor = meanOutputDoseGy / meanInputValue

        # Write rescaled dose values to output volume
        doseArray = maskedArray * rescaleFactor
        slicer.util.updateVolumeFromArray(outputVolumeNode, doseArray)

        # Set window/level for the output volume display
        displayNode = outputVolumeNode.GetDisplayNode()
        if displayNode:
            maxDoseGy = np.max(doseArray)
            window = 250
            level = 125
            displayNode.SetAutoWindowLevel(False)
            displayNode.SetWindow(window)
            displayNode.SetLevel(level)
            colorNode = slicer.util.getNode('PET-Rainbow2')
            displayNode.SetAndObserveColorNodeID(colorNode.GetID())
            
        # Calculate mean dose for each segment
        segmentation = segmentationNode.GetSegmentation()
        segmentIDs = vtk.vtkStringArray()
        segmentation.GetSegmentIDs(segmentIDs)

        segmentDoses = {}
        segmentVolumes = {}
        segmentActivity = {}
        for i in range(segmentIDs.GetNumberOfValues()):
            segmentID = segmentIDs.GetValue(i)
            segmentName = segmentation.GetSegment(segmentID).GetName()

            # Create label map for segment
            segmentLabelMapNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
            slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(
                segmentationNode, [segmentID], segmentLabelMapNode, spectVolumeNode
            )

            # Mask dose array to calculate mean dose for the segment
            segmentLabelMapArray = slicer.util.arrayFromVolume(segmentLabelMapNode)
            maskedDoseArray = doseArray[segmentLabelMapArray == 1]
            segmentDoses[segmentName] = np.mean(maskedDoseArray)
            segmentVolumes[segmentName] = voxelVolumeML*maskedDoseArray.size
            segmentActivity[segmentName] = (((voxelVolumeML*maskedDoseArray.size)*segmentDoses[segmentName])/conversionFactor)*densityGPerML
            # Remove temporary label map node
            slicer.mrmlScene.RemoveNode(segmentLabelMapNode)

        # Clean up temporary nodes
        slicer.mrmlScene.RemoveNode(labelMapVolumeNode)
        slicer.mrmlScene.RemoveNode(maskedVolumeNode)

        logging.info("Dosimetric calculations completed.")
         # Estimate lung absorbed dose
        lungDoseGy = (ncorr_activityMBq * lungShuntFractionPercent * 0.01 * conversionFactor) / lungMassg
   
        # Update segment dose table
        # Clear existing table contents
        self.segmentDoseTable.setRowCount(0)

        # Insert lung dose at the top
        self.segmentDoseTable.insertRow(0)
        self.segmentDoseTable.setItem(0, 0, qt.QTableWidgetItem("Estimated Lung Dose"))
        self.segmentDoseTable.setItem(0, 1, qt.QTableWidgetItem(f"{lungDoseGy:.2f}"))

        # Populate table with segment doses
        for segmentName, dose in segmentDoses.items():
            rowPosition = self.segmentDoseTable.rowCount
            self.segmentDoseTable.insertRow(rowPosition)
            self.segmentDoseTable.setItem(rowPosition, 0, qt.QTableWidgetItem(segmentName))
            self.segmentDoseTable.setItem(rowPosition, 1, qt.QTableWidgetItem(f"{dose:.2f}"))
            self.segmentDoseTable.setItem(rowPosition, 2, qt.QTableWidgetItem(f"{segmentVolumes[segmentName]:.2f}"))
            self.segmentDoseTable.setItem(rowPosition, 3, qt.QTableWidgetItem(f"{segmentActivity[segmentName]:.2f}"))     
       
        
        return segmentDoses





    def limonCalculateButton(self):
        spectVolumeNode = self.spectSelector.currentNode()
        segmentationNode = self.segmentationSelector.currentNode()
        liverSegmentID = self.liverSegmentSelector.currentSegmentID()  # Retrieve the selected liver segment ID
        activityMBq = self.activitySlider.value
        outputVolumeNode = self.outputVolumeSelector.currentNode()
        lungShuntFractionPercent = self.lungShuntSlider.value

        sliceWidget = slicer.app.layoutManager().sliceWidget("Red")
        sliceCompositeNode = sliceWidget.mrmlSliceCompositeNode()
        backgroundVolumeid = slicer.app.layoutManager().sliceWidget("Red").mrmlSliceCompositeNode().GetBackgroundVolumeID()

        if not spectVolumeNode or not segmentationNode or not liverSegmentID or not outputVolumeNode:
            slicer.util.errorDisplay("Please select valid input and output nodes, and ensure the liver segment is specified.")
            return


        # Perform dosimetric calculations
        self.limcalculateDose(spectVolumeNode, segmentationNode, liverSegmentID, activityMBq, outputVolumeNode, lungShuntFractionPercent)
        self.outputVolumeSelector.currentNode().SetName("Dose Map (Gy)")
        foregroundVolumeid = self.outputVolumeSelector.currentNode().GetID()
        sliceCompositeNode.SetBackgroundVolumeID(backgroundVolumeid)
        sliceCompositeNode.SetForegroundVolumeID(foregroundVolumeid)
        sliceCompositeNode.SetForegroundOpacity(0.5)

    def limcalculateDose(self, spectVolumeNode, segmentationNode, liverSegmentID, activityMBq, outputVolumeNode, lungShuntFractionPercent):
        """
        Perform dosimetric calculations using the given inputs.
        """
        logging.info("Starting dosimetric calculations.")

        # Validate inputs
        if not spectVolumeNode or not segmentationNode or not liverSegmentID or not outputVolumeNode:
            raise ValueError("Invalid inputs. Please select valid nodes and ensure the liver segment is specified.")

        # Export the liver segment to a label map aligned with the input volume
        labelMapVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
        slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(
            segmentationNode, [liverSegmentID], labelMapVolumeNode, spectVolumeNode
        )

        # Apply the label map to mask the input volume
        maskedVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "MaskedVolume")
        parameters = {
            "InputVolume": spectVolumeNode.GetID(),
            "MaskVolume": labelMapVolumeNode.GetID(),
            "OutputVolume": maskedVolumeNode.GetID(),
            "Label": 1,  # Label value corresponding to the liver segment
            "Replace": 0  # Set voxels outside the segment to 0
        }
        slicer.cli.runSync(slicer.modules.maskscalarvolume, None, parameters)

        # Clone the masked volume to the output volume to inherit spatial properties
        volumesLogic = slicer.modules.volumes.logic()
        clonedOutputVolumeNode = volumesLogic.CloneVolume(slicer.mrmlScene, maskedVolumeNode, outputVolumeNode.GetName())
        outputVolumeNode.Copy(clonedOutputVolumeNode)
        outputVolumeNode.SetAttribute("DicomRtImport.DoseVolume", "1")
        slicer.mrmlScene.RemoveNode(clonedOutputVolumeNode)

        # Get the masked volume array
        maskedArray = slicer.util.arrayFromVolume(maskedVolumeNode)
        if maskedArray is None:
            raise ValueError("Unable to access data from the masked volume.")

        # Calculate total volume in mL
        spacing = spectVolumeNode.GetSpacing()  # spacing is in mm
        voxelVolumeML = (spacing[0] * spacing[1] * spacing[2]) / 1000.0  # convert mm^3 to mL
        totalVolumeML = maskedArray.size * voxelVolumeML

        #SET ACTIVITY TO 1000 MBQ
        activityMBq=1000

        # Constants for Y-90 dosimetry
        conversionFactor = self.conversionFactorSpinBox.value
        lungMassg = self.lungMassSpinBox.value
        densityGPerML = self.liverDensitySpinBox.value

        NsegmentID = self.targetSegmentSelector.currentSegmentID()
        
        
        # Adjust activity based on lung shunt fraction
        lungShuntFraction = lungShuntFractionPercent / 100.0
        activityMBq = activityMBq * (1 - lungShuntFraction)
        if totalVolumeML == 0:
            raise ValueError("Total volume is zero. Ensure the liver segment is correctly defined.")

        # Calculate mean input value within the liver segment
        meanInputValue = np.sum(maskedArray) / maskedArray.size



        # Calculate mean output dose
        meanOutputDoseGy = (activityMBq / (totalVolumeML * densityGPerML)) * conversionFactor

        # Rescale factor to normalize dose
        rescaleFactor = meanOutputDoseGy / meanInputValue

        # Write rescaled dose values to output volume
        doseArray = maskedArray * rescaleFactor
        slicer.util.updateVolumeFromArray(outputVolumeNode, doseArray)

        # Set window/level for the output volume display
        displayNode = outputVolumeNode.GetDisplayNode()
        if displayNode:
            maxDoseGy = np.max(doseArray)
            window = 250
            level = 125
            displayNode.SetAutoWindowLevel(False)
            displayNode.SetWindow(window)
            displayNode.SetLevel(level)
            colorNode = slicer.util.getNode('PET-Rainbow2')
            displayNode.SetAndObserveColorNodeID(colorNode.GetID())
            
        segmentation = segmentationNode.GetSegmentation()

        # Calculate NDOSE for LSF corrected 1000mbq
        NsegmentName = segmentation.GetSegment(NsegmentID).GetName()
        segmentLabelMapNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
        slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(
        segmentationNode, [NsegmentID], segmentLabelMapNode, spectVolumeNode
        )
        segmentLabelMapArray = slicer.util.arrayFromVolume(segmentLabelMapNode)
        maskedDoseArray = doseArray[segmentLabelMapArray == 1]
        NDOSE = np.mean(maskedDoseArray)
        # Remove temporary label map node
        slicer.mrmlScene.RemoveNode(segmentLabelMapNode)
        slicer.mrmlScene.RemoveNode(maskedVolumeNode)
        slicer.mrmlScene.RemoveNode(labelMapVolumeNode)


        permittedMBq = 1000
        if self.targetdoseSlider.value>0:
            permittedMBq = ((self.targetdoseSlider.value/NDOSE)*1000)
                
        self.activitySlider.value = permittedMBq


        self.calculateDose(spectVolumeNode, segmentationNode, liverSegmentID, permittedMBq, outputVolumeNode, lungShuntFractionPercent)
        
 
       
        
        return 0

    def onSaveReportClicked(self):
        # Open file dialog to select save path
        fileDialog = qt.QFileDialog()
        fileName = fileDialog.getSaveFileName(None, "Save Dosimetry Report", "", "RTF Files (*.rtf)")
        if not fileName:
            return
        if not fileName.lower().endswith(".rtf"):
            fileName += ".rtf"

        # Build RTF content
        import datetime
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conversionFactor = self.conversionFactorSpinBox.value
        lungMass = self.lungMassSpinBox.value
        liverDensity = self.liverDensitySpinBox.value
        activity = self.activitySlider.value
        lungShunt = self.lungShuntSlider.value

        rtf = r"""{\rtf1\ansi\deff0
{\b Taranis - Patient Relative Quantification\line
\b Radioembolization Dosimetry Report}\line
Generated: """ + now + r"""\line
\line
{\b Parameters}\line
Activity: """ + f"{activity:.2f} MBq" + r"""\line
Lung Shunt: """ + f"{lungShunt:.2f}%" + r"""\line
Conversion Factor: """ + f"{conversionFactor:.2f} Gy/MBq/g" + r"""\line
Lung Mass: """ + f"{lungMass:.2f} g" + r"""\line
Liver Density: """ + f"{liverDensity:.2f} g/mL" + r"""\line
\line
{\b Segment Doses}\line
"""

        rowCount = self.segmentDoseTable.rowCount
        colCount = self.segmentDoseTable.columnCount
        for row in range(rowCount):
            segment = self.segmentDoseTable.item(row, 0)
            dose = self.segmentDoseTable.item(row, 1)
            vol = self.segmentDoseTable.item(row, 2) if colCount > 2 else None
            act = self.segmentDoseTable.item(row, 3) if colCount > 3 else None
            
            rtf += f"Segment: {segment.text()}, "

            if dose:
                rtf += f"Dose = {dose.text()} Gy"
            if vol:
                rtf += f", Volume = {vol.text()} mL"
            if act:
                rtf += f", Activity = {act.text()} MBq"
            rtf += r"\line\n "

        rtf += r"\line\n End of Report}"
        
        # Write to file
        with open(fileName, "w") as file:
            file.write(rtf)

        slicer.util.infoDisplay("RTF report saved successfully.")