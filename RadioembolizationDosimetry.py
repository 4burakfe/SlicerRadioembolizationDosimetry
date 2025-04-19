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
        iconPath = os.path.join(os.path.dirname(__file__), "taranis_logo.png")
        self.parent.icon = qt.QIcon(iconPath)  # Assign icon to the module
        self.parent = parent

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
        bannerPath = os.path.join(moduleDir, "banner.png")  # Change to your banner file

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
        self.activitySlider.maximum = 6000
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

        # Limit normal Dose Slider
        self.ndoseSlider = ctk.ctkSliderWidget()
        self.ndoseSlider.singleStep = 1
        self.ndoseSlider.minimum = 0
        self.ndoseSlider.maximum = 400
        self.ndoseSlider.value = 0
        self.ndoseSlider.setToolTip("Specify the dose limit for normal tissue")
#        formLayout.addRow("Limit dose for segment(Gy):", self.ndoseSlider)

        # normal
        self.normalSegmentSelector = slicer.qMRMLSegmentSelectorWidget()
        self.normalSegmentSelector.setMRMLScene(slicer.mrmlScene)
        self.normalSegmentSelector.setToolTip("Select the segment representing the tumor.")
#        formLayout.addRow("Normal Tissue: ", self.normalSegmentSelector)

        # Limit lung Dose Slider
        self.ldoseSlider = ctk.ctkSliderWidget()
        self.ldoseSlider.singleStep = 1
        self.ldoseSlider.minimum = 0
        self.ldoseSlider.maximum = 400
        self.ldoseSlider.value = 0
        self.ldoseSlider.setToolTip("Specify the dose limit for lung tissue")
#        formLayout.addRow("Limit dose for lungs(Gy):", self.ldoseSlider)

        limTextBox = qt.QTextEdit()
        limTextBox.setReadOnly(True)  # Make the text box read-only
        limTextBox.append(
            "Set maximum limit doses for segments. Tumor dose will be maximized within the limits.\n"
            "If you do not desire to use one or more limits or targets, set value to zero.\n"
        )
#        formLayout.addRow("Processing Log:", limTextBox)
       
        # Calculate Button
        self.calculateButtonlim = qt.QPushButton("Calculate with the maximal activity within limit doses")
        self.calculateButtonlim.toolTip = "Perform dosimetric calculations."
#        formLayout.addRow(self.calculateButtonlim)

        # Segment Dose Table
        self.segmentDoseTable = qt.QTableWidget()
        self.segmentDoseTable.setColumnCount(4)
        self.segmentDoseTable.setHorizontalHeaderLabels(["Segment", "Dose (Gy)","Volume (mL)","Activity (MBq)"])
        self.segmentDoseTable.setFixedSize(420,350)
        formLayout.addRow("Segment Doses: ", self.segmentDoseTable)

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
            "Prepared by: Burak Demir, MD, FEBNM \n"
            "For support, feedback, and suggestions: 4burakfe@gmail.com\n"
            "Version: alpha v1.1"
        )
        infoTextBox.setToolTip("Module information and instructions.")  # Add a tooltip for additional help
        self.layout.addWidget(infoTextBox)
        
    def onSegmentationNodeChanged(self, node):
        self.liverSegmentSelector.setSegmentationNode(node)

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

        if not spectVolumeNode or not segmentationNode or not liverSegmentID or not outputVolumeNode:
            slicer.util.errorDisplay("Please select valid input and output nodes, and ensure the liver segment is specified.")
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
        densityGPerML = 1.05  # g/mL for liver tissue
        conversionFactor = 50  # Gy per MBq per g of tissue

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
        lungMassg = 1000  # Assumed lung mass g
        conversionFactor = 50  # Gy per MBq per g of tissue
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



        NsegmentID = self.normalSegmentSelector.currentSegmentID()
        
        
        # Adjust activity based on lung shunt fraction
        lungShuntFraction = lungShuntFractionPercent / 100.0
        activityMBq = activityMBq * (1 - lungShuntFraction)
        if totalVolumeML == 0:
            raise ValueError("Total volume is zero. Ensure the liver segment is correctly defined.")

        # Calculate mean input value within the liver segment
        meanInputValue = np.sum(maskedArray) / maskedArray.size

        # Constants for Y-90 dosimetry
        densityGPerML = 1.05  # g/mL for liver tissue
        conversionFactor = 50  # Gy per MBq per g of tissue

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


        permittedMBq = 1000
        if self.ndoseSlider.value>0:
            permittedMBq = ((self.ndoseSlider.value/NDOSE)*1000)
        lungMassg = 1000  # Assumed lung mass g
        conversionFactor = 50  # Gy per MBq per g of tissue
        LDOSE = (1000 * lungShuntFractionPercent * 0.01 * conversionFactor) / lungMassg
        if self.ldoseSlider.value>0:
            if permittedMBq > (self.ldoseSlider.value/LDOSE)*1000:
                permittedMBq = (self.ldoseSlider.value/LDOSE)*1000
            else:
                permittedMBq = permittedMBq
                
        self.activitySlider.value = permittedMBq
        self.calculateDose(spectVolumeNode, segmentationNode, liverSegmentID, permittedMBq, outputVolumeNode, lungShuntFractionPercent)
        
 
       
        
        return 0
