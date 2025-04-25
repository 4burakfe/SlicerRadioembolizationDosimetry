import os
import numpy as np
import slicer
from slicer.ScriptedLoadableModule import *
import logging
import qt
import ctk
import vtk

class RadioembolizationDosimetryabs(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "Taranis - Dosimetry (Absolute Quantification)"
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

class RadioembolizationDosimetryabsWidget(ScriptedLoadableModuleWidget):
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
            bannerLabel.setPixmap(bannerPixmap.scaledToWidth(450, qt.Qt.SmoothTransformation))  # Adjust width

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
        formLayout.addRow("Input PET Volume: ", self.spectSelector)

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

        # Hour Slider and Input Box
        self.hourSlider = ctk.ctkSliderWidget()
        self.hourSlider.singleStep = 1
        self.hourSlider.minimum = 0
        self.hourSlider.maximum = 200
        self.hourSlider.value = 0
        self.hourSlider.setToolTip("Hours after treatment")
        formLayout.addRow("Hours after treatment: ", self.hourSlider)


        # Half-Life SpinBox
        self.halfLifeSpinBox = qt.QDoubleSpinBox()
        self.halfLifeSpinBox.setRange(0.1, 200.0)  # Allow reasonable half-lives
        self.halfLifeSpinBox.setValue(64.2)         # Default for Y-90
        self.halfLifeSpinBox.setSingleStep(0.1)
        self.halfLifeSpinBox.setToolTip("Specify the radionuclide's physical half-life in hours.")
        formLayout.addRow("Half-Life (hours):", self.halfLifeSpinBox)


        # Conversion Factor (Gy/MBq/g)
        self.conversionFactorSpinBox = qt.QDoubleSpinBox()
        self.conversionFactorSpinBox.setRange(0.0, 100.0)
        self.conversionFactorSpinBox.setValue(49.67)
        self.conversionFactorSpinBox.setSingleStep(0.1)
        formLayout.addRow("Conversion Factor (Gy/MBq/g):", self.conversionFactorSpinBox)

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
        self.calculateButton = qt.QPushButton("Calculate")
        self.calculateButton.toolTip = "Perform dosimetric calculations."
        formLayout.addRow(self.calculateButton)

        # Segment Dose Table
        self.segmentDoseTable = qt.QTableWidget()
        self.segmentDoseTable.setColumnCount(4)
        self.segmentDoseTable.setHorizontalHeaderLabels(["Segment", "Dose (Gy)","Volume (mL)","Activity (MBq)"])
        self.segmentDoseTable.setFixedSize(400,350)
        formLayout.addRow("Segment Doses: ", self.segmentDoseTable)


        # Save Report Button
        self.saveReportButton = qt.QPushButton("Save Report as RTF")
        self.saveReportButton.toolTip = "Export dosimetry results to a report file."
        formLayout.addRow(self.saveReportButton)
        self.saveReportButton.connect('clicked(bool)', self.onSaveReportClicked)
        
        
        # Total Activity Text Box
        self.totalActivityTextBox = qt.QLineEdit()
        self.totalActivityTextBox.setReadOnly(True)
        self.totalActivityTextBox.setToolTip("Displays the total calculated activity from PET/SPECT images in MBq.")
        formLayout.addRow("Total Activity\nfrom PET(MBq): ", self.totalActivityTextBox)

        # Total Activity Text Box
        self.dectotalActivityTextBox = qt.QLineEdit()
        self.dectotalActivityTextBox.setReadOnly(True)
        self.dectotalActivityTextBox.setToolTip("Displays the total calculated decay corrected activity in MBq.")
        formLayout.addRow("Total Decay\nCorr Act (MBq): ", self.dectotalActivityTextBox)
        
        # Connections
        self.calculateButton.connect('clicked(bool)', self.onCalculateButton)

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

    def onCalculateButton(self):
        spectVolumeNode = self.spectSelector.currentNode()
        segmentationNode = self.segmentationSelector.currentNode()
        hourelapsed = self.hourSlider.value
        outputVolumeNode = self.outputVolumeSelector.currentNode()

        if not spectVolumeNode or not segmentationNode or not outputVolumeNode:
            slicer.util.errorDisplay("Please select valid input and output nodes.")
            return

        sliceWidget = slicer.app.layoutManager().sliceWidget(slicer.app.layoutManager().sliceViewNames()[0])
        sliceCompositeNode = sliceWidget.mrmlSliceCompositeNode()
        backgroundVolumeid = slicer.app.layoutManager().sliceWidget("Red").mrmlSliceCompositeNode().GetBackgroundVolumeID()

        # Perform dosimetric calculations
        self.calculateDose(spectVolumeNode, segmentationNode, hourelapsed, outputVolumeNode,self.totalActivityTextBox,self.dectotalActivityTextBox,self.segmentDoseTable)
        self.outputVolumeSelector.currentNode().SetName("Dose Map (Gy)")
        foregroundVolumeid = self.outputVolumeSelector.currentNode().GetID()
        sliceCompositeNode.SetBackgroundVolumeID(backgroundVolumeid)
        sliceCompositeNode.SetForegroundVolumeID(foregroundVolumeid)
        sliceCompositeNode.SetForegroundOpacity(0.5)
        segmentationNode
        
        
    def calculateDose(self, spectVolumeNode, segmentationNode, hourelapsed, outputVolumeNode, totalActivityTextBox,dectotalActivityTextBox,segmentDoseTable):
        """
        Perform dosimetric calculations using the given inputs.
        """
        logging.info("Starting dosimetric calculations.")

        # Validate inputs
        if not spectVolumeNode or not segmentationNode or not outputVolumeNode:
            raise ValueError("Invalid inputs. Please select valid nodes.")

        # Clone the input volume to the output volume to inherit spatial properties
        volumesLogic = slicer.modules.volumes.logic()
        clonedOutputVolumeNode = volumesLogic.CloneVolume(slicer.mrmlScene, spectVolumeNode, outputVolumeNode.GetName())
        outputVolumeNode.Copy(clonedOutputVolumeNode)
        outputVolumeNode.SetAttribute("DicomRtImport.DoseVolume", "1")
        slicer.mrmlScene.RemoveNode(clonedOutputVolumeNode)

        # Get input volume array
        spectArray = slicer.util.arrayFromVolume(spectVolumeNode)
        if spectArray is None:
            raise ValueError("Unable to access data from the input SPECT volume.")

        # Calculate total volume in mL
        spacing = spectVolumeNode.GetSpacing()  # spacing is in mm
        voxelVolumeML = (spacing[0] * spacing[1] * spacing[2]) / 1000.0  # convert mm^3 to mL
        totalVolumeML = spectArray.size * voxelVolumeML
        if totalVolumeML == 0:
            raise ValueError("Total volume is zero. Ensure the SPECT volume contains valid data.")

        # Calculate mean input value per voxel
        meanInputValue = np.sum(spectArray) / spectArray.size
        activityMBq = totalVolumeML * meanInputValue /1000000
        totalActivityTextBox.setText(f"{activityMBq:.2f} MBq")

        # Decay correction
        activityMBq = activityMBq * (2.0 ** (hourelapsed / self.halfLifeSpinBox.value))
        # Update total activity text box
        dectotalActivityTextBox.setText(f"{activityMBq:.2f} MBq")

        # Constants for Y-90 dosimetry
        conversionFactor = self.conversionFactorSpinBox.value
        densityGPerML = self.liverDensitySpinBox.value

        # Calculate mean output dose
        meanOutputDoseGy = (activityMBq / (totalVolumeML * densityGPerML)) * conversionFactor

        # Rescale factor to normalize dose
        rescaleFactor = meanOutputDoseGy / meanInputValue

        # Write rescaled dose values to output volume
        doseArray = spectArray * rescaleFactor
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
            labelMapVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
            slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(
                segmentationNode, [segmentID], labelMapVolumeNode, spectVolumeNode
            )

            # Mask dose array to calculate mean dose for the segment
            labelMapArray = slicer.util.arrayFromVolume(labelMapVolumeNode)
            maskedDoseArray = doseArray[labelMapArray == 1]
            segmentDoses[segmentName] = np.mean(maskedDoseArray)
            segmentVolumes[segmentName] = voxelVolumeML*maskedDoseArray.size
            segmentActivity[segmentName] = (((voxelVolumeML*maskedDoseArray.size)*segmentDoses[segmentName])/conversionFactor)*densityGPerML
            # Remove temporary label map node
            slicer.mrmlScene.RemoveNode(labelMapVolumeNode)

        # Populate table with segment doses
        segmentDoseTable.setRowCount(0)
        for segmentName, dose in segmentDoses.items():
            rowPosition = segmentDoseTable.rowCount
            segmentDoseTable.insertRow(rowPosition)
            segmentDoseTable.setItem(rowPosition, 0, qt.QTableWidgetItem(segmentName))
            segmentDoseTable.setItem(rowPosition, 1, qt.QTableWidgetItem(f"{dose:.2f}"))
            segmentDoseTable.setItem(rowPosition, 2, qt.QTableWidgetItem(f"{segmentVolumes[segmentName]:.2f}"))
            segmentDoseTable.setItem(rowPosition, 3, qt.QTableWidgetItem(f"{segmentActivity[segmentName]:.2f}"))     

        logging.info("Dosimetric calculations completed.")
        return segmentDoses
        
        
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
        liverDensity = self.liverDensitySpinBox.value
        rtf = r"""{\rtf1\ansi\deff0
{\b Taranis - Absolute Quantification\line
\b Radioembolization Dosimetry Report}\line
Generated: """ + now + r"""\line
\line
{\b Parameters}\line
Activity During Imaging: """ + f"{self.totalActivityTextBox.text}" + r"""\line
Decay Corrected Activity: """ + f"{self.dectotalActivityTextBox.text}" + r"""\line
Conversion Factor: """ + f"{conversionFactor:.2f} Gy/MBq/g" + r"""\line
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