import os
import numpy as np
import slicer
from slicer.ScriptedLoadableModule import *
import logging
import qt
import ctk
import vtk
  

class easy_reg(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "EasyReg"
        parent.categories = ["Nuclear Medicine"]
        parent.dependencies = []
        parent.contributors = ["Burak Demir, MD, FEBNM"]
        parent.helpText = """
        This module provides easy workflow for registration of SPECT/CT images to diagnostic CT/MR images.
        """
        parent.acknowledgementText = """
        This file was developed by Burak Demir.
        """
        # **✅ Set the module icon**
        iconPath = os.path.join(os.path.dirname(__file__), "Resources\\Icons\\taranis_logo.png")
        self.parent.icon = qt.QIcon(iconPath)  # Assign icon to the module
        self.parent = parent

class easy_regWidget(ScriptedLoadableModuleWidget):

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)




        # Create collapsible section
        parametersCollapsibleButton = ctk.ctkCollapsibleButton()
        parametersCollapsibleButton.text = "Parameters"
        self.layout.addWidget(parametersCollapsibleButton)
        formLayout = qt.QFormLayout(parametersCollapsibleButton)
        
        
       
        # 1️⃣ Input Volume Selector (CT Image)
        self.inputVolumeSelector = slicer.qMRMLNodeComboBox()
        self.inputVolumeSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.inputVolumeSelector.selectNodeUponCreation = True
        self.inputVolumeSelector.addEnabled = False
        self.inputVolumeSelector.removeEnabled = False
        self.inputVolumeSelector.noneEnabled = False
        self.inputVolumeSelector.showHidden = False
        self.inputVolumeSelector.showChildNodeTypes = False
        self.inputVolumeSelector.setMRMLScene(slicer.mrmlScene)
        self.inputVolumeSelector.setToolTip("Select the SPECT image for segmentation.")
        formLayout.addRow("SPECT: ", self.inputVolumeSelector)

        # 1️⃣ Input Volume Selector (CT Image)
        self.inputVolumeSelectorCT = slicer.qMRMLNodeComboBox()
        self.inputVolumeSelectorCT.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.inputVolumeSelectorCT.selectNodeUponCreation = True
        self.inputVolumeSelectorCT.addEnabled = False
        self.inputVolumeSelectorCT.removeEnabled = False
        self.inputVolumeSelectorCT.noneEnabled = False
        self.inputVolumeSelectorCT.showHidden = False
        self.inputVolumeSelectorCT.showChildNodeTypes = False
        self.inputVolumeSelectorCT.setMRMLScene(slicer.mrmlScene)
        self.inputVolumeSelectorCT.setToolTip("Select the CT image SPECT/CT for segmentation.")
        formLayout.addRow("CT of SPECT: ", self.inputVolumeSelectorCT)

        # 2️⃣ ROI Selector (Select or Create New Markups ROI)
        self.roiSelector = slicer.qMRMLNodeComboBox()
        self.roiSelector.nodeTypes = ["vtkMRMLMarkupsROINode"]  # Use Markups ROI
        self.roiSelector.selectNodeUponCreation = True
        self.roiSelector.addEnabled = True  # Allow creation of new ROI
        self.roiSelector.removeEnabled = True  # Allow deletion of ROI
        self.roiSelector.noneEnabled = False
        self.roiSelector.showHidden = False
        self.roiSelector.showChildNodeTypes = False
        self.roiSelector.setMRMLScene(slicer.mrmlScene)
        self.roiSelector.setToolTip("Select or create a new Markups ROI.")
        formLayout.addRow("ROI for SPECT/CT: ", self.roiSelector)

        # 6️⃣ Calculate Button
        self.cropSPECTButton = qt.QPushButton("Perform Cropping for SPECT/CT")
        self.cropSPECTButton.enabled = True
        formLayout.addRow(self.cropSPECTButton)


        # 1️⃣ Input Volume Selector (CT Image)
        self.refVolumeSelector = slicer.qMRMLNodeComboBox()
        self.refVolumeSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.refVolumeSelector.selectNodeUponCreation = True
        self.refVolumeSelector.addEnabled = False
        self.refVolumeSelector.removeEnabled = False
        self.refVolumeSelector.noneEnabled = False
        self.refVolumeSelector.showHidden = False
        self.refVolumeSelector.showChildNodeTypes = False
        self.refVolumeSelector.setMRMLScene(slicer.mrmlScene)
        self.refVolumeSelector.setToolTip("Select the CT/MRI for segmentation.")
        formLayout.addRow("Reference CT/MRI: ", self.refVolumeSelector)

         # 2️⃣ ROI Selector (Select or Create New Markups ROI)
        self.refroiSelector = slicer.qMRMLNodeComboBox()
        self.refroiSelector.nodeTypes = ["vtkMRMLMarkupsROINode"]  # Use Markups ROI
        self.refroiSelector.selectNodeUponCreation = True
        self.refroiSelector.addEnabled = True  # Allow creation of new ROI
        self.refroiSelector.removeEnabled = True  # Allow deletion of ROI
        self.refroiSelector.noneEnabled = False
        self.refroiSelector.showHidden = False
        self.refroiSelector.showChildNodeTypes = False
        self.refroiSelector.setMRMLScene(slicer.mrmlScene)
        self.refroiSelector.setToolTip("Select or create a new Markups ROI.")
        formLayout.addRow("ROI for Ref: ", self.refroiSelector)

        # 6️⃣ Calculate Button
        self.cropREFButton = qt.QPushButton("Perform Cropping for Reference Image")
        self.cropREFButton.enabled = True
        formLayout.addRow(self.cropREFButton)

        # **✅ Create a Group for Radio Buttons**
        self.registrationMethodGroup = qt.QButtonGroup()  # Ensures only one selection at a time

        # **✅ Create Radio Buttons**
        self.rigidRegistrationButton = qt.QRadioButton("Rigid Registration")
        self.affineRegistrationButton = qt.QRadioButton("Affine Registration")
        self.deformableRegistrationButton = qt.QRadioButton("Deformable Registration")

        # **✅ Add Buttons to Group**
        self.registrationMethodGroup.addButton(self.rigidRegistrationButton)
        self.registrationMethodGroup.addButton(self.affineRegistrationButton)
        self.registrationMethodGroup.addButton(self.deformableRegistrationButton)

        # **✅ Set Default Selection**
        self.rigidRegistrationButton.setChecked(True)  # Default selection

        # **✅ Create a Layout for the Radio Buttons**
        self.registrationLayout = qt.QVBoxLayout()
        self.registrationLayout.addWidget(self.rigidRegistrationButton)
        self.registrationLayout.addWidget(self.affineRegistrationButton)
        self.registrationLayout.addWidget(self.deformableRegistrationButton)

        # **✅ Add Layout to UI**
        self.layout.addLayout(self.registrationLayout)

        # **✅ Create "Register Images" button**
        self.registerButton = qt.QPushButton("Register Images")
        self.registerButton.setToolTip("Registers the CT of the SPECT to the Reference Image.")
        self.registerButton.clicked.connect(self.registerImages)
        self.layout.addWidget(self.registerButton)


        # Connect ROI creation event to set default size
        self.roiSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.setDefaultROISize)
        self.refroiSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.setDefaultROISizeREF)

        # Connect Calculate button to function
        self.cropSPECTButton.connect("clicked(bool)", self.CropSPECTImage)
        self.cropREFButton.connect("clicked(bool)", self.CropREFImage)
        

        self.layout.addStretch(1)

        # **✅ Load the banner image**
        moduleDir = os.path.dirname(__file__)  # Get module directory
        bannerPath = os.path.join(moduleDir, "Resources\\Icons\\banner.png")  # Change to your banner file

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
            


        # 5️⃣ Info Text Box
        infoTextBox = qt.QTextEdit()
        infoTextBox.setReadOnly(True)  # Make the text box read-only
        infoTextBox.setPlainText(
            "This module provides automatic liver segmentation on non-contrast CT images.\n"
            "Select the CT volume to segment.\n"
            "Define a Region of Interest (ROI) that contains the liver. Maximum ROI size: (416, 352, 288) mm.\n"
            "The module will resample and crop the image to isotropic 2x2x2 mm voxels.\n"
            "This module is NOT a medical device. Research use only.\n"
            "Developed by: Burak Demir, MD, FEBNM \n"
            "For support and feedback: 4burakfe@gmail.com\n"
            "Version: alpha v1.0"
        )
        infoTextBox.setToolTip("Module information and instructions.")  # Add a tooltip for additional help
        self.layout.addWidget(infoTextBox)



    def getSelectedRegistrationMethod(self):
        """
        Returns the selected registration method from the radio buttons.
        """
        if self.rigidRegistrationButton.isChecked():
            return "Rigid"
        elif self.affineRegistrationButton.isChecked():
            return "Affine"
        elif self.deformableRegistrationButton.isChecked():
            return "BSpline"
        else:
            return None

    def setDefaultROISize(self):




        roinode = self.roiSelector.currentNode()

        if roinode is not None:
            
            sliceWidget = slicer.app.layoutManager().sliceWidget("Red")
            slicercontroller = slicer.app.layoutManager().sliceWidget("Red").sliceController()
            sliceCompositeNode = sliceWidget.mrmlSliceCompositeNode()

            sliceCompositeNode.SetBackgroundVolumeID(self.inputVolumeSelectorCT.currentNode().GetID())
            slicercontroller.fitSliceToBackground()

            # **✅ Set SPECT as Foreground in Rainbow**
            sliceCompositeNode.SetForegroundVolumeID(self.inputVolumeSelector.currentNode().GetID())
            slicercontroller.setForegroundHidden(False)  # ✅ Make sure foreground is visible
            slicercontroller.setForegroundOpacity(0.5)  # ✅ Ensure opacity is 50%

            sliceWidget = slicer.app.layoutManager().sliceWidget("Yellow")
            slicercontroller = slicer.app.layoutManager().sliceWidget("Yellow").sliceController()
            sliceCompositeNode = sliceWidget.mrmlSliceCompositeNode()

            sliceCompositeNode.SetBackgroundVolumeID(self.inputVolumeSelectorCT.currentNode().GetID())
            slicercontroller.fitSliceToBackground()

            # **✅ Set SPECT as Foreground in Rainbow**
            sliceCompositeNode.SetForegroundVolumeID(self.inputVolumeSelector.currentNode().GetID())
            slicercontroller.setForegroundHidden(False)  # ✅ Make sure foreground is visible
            slicercontroller.setForegroundOpacity(0.5)  # ✅ Ensure opacity is 50%

            sliceWidget = slicer.app.layoutManager().sliceWidget("Green")
            slicercontroller = slicer.app.layoutManager().sliceWidget("Green").sliceController()
            sliceCompositeNode = sliceWidget.mrmlSliceCompositeNode()

            sliceCompositeNode.SetBackgroundVolumeID(self.inputVolumeSelectorCT.currentNode().GetID())
            slicercontroller.fitSliceToBackground()

            # **✅ Set SPECT as Foreground in Rainbow**
            sliceCompositeNode.SetForegroundVolumeID(self.inputVolumeSelector.currentNode().GetID())
            slicercontroller.setForegroundHidden(False)  # ✅ Make sure foreground is visible
            slicercontroller.setForegroundOpacity(0.5)  # ✅ Ensure opacity is 50%


            # **✅ Set Correct Color Maps**
            # **Set SPECT to PET-Rainbow2**
            spectDisplayNode = self.inputVolumeSelector.currentNode().GetDisplayNode()
            spectColorNode = slicer.util.getNode('PET-Rainbow2')
            spectDisplayNode.SetAndObserveColorNodeID(spectColorNode.GetID())

            # **Set CT to Grayscale**
            ctDisplayNode = self.inputVolumeSelectorCT.currentNode().GetDisplayNode()
            ctColorNode = slicer.util.getNode('Grey')  # ✅ Use "Grey" instead of "Gray"
            ctDisplayNode.SetAndObserveColorNodeID(ctColorNode.GetID())  # Fix applied here!
        
            """
            Sets the default size and centers the ROI in the geometric center of the selected volume upon creation.
            Also ensures that the ROI size updates dynamically when changed.
            """
                # Set default size
            roinode.SetSize(400, 300, 250)

            # Get the selected volume
            inputVolumeNode = self.inputVolumeSelectorCT.currentNode()

            # Compute the volume's geometric center in RAS coordinates
            bounds = [0] * 6
            inputVolumeNode.GetRASBounds(bounds)
            centerX = (bounds[0] + bounds[1]) / 2
            centerY = (bounds[2] + bounds[3]) / 2
            centerZ = (bounds[4] + bounds[5]) / 2

                # Set ROI center
            roinode.SetCenter(centerX, centerY, centerZ)

    def setDefaultROISizeREF(self):



        roinode = self.refroiSelector.currentNode()

        if roinode is not None:
            sliceWidget = slicer.app.layoutManager().sliceWidget("Red")
            slicercontroller = slicer.app.layoutManager().sliceWidget("Red").sliceController()
            sliceCompositeNode = sliceWidget.mrmlSliceCompositeNode()

            slicercontroller.setForegroundHidden(True)  
            sliceCompositeNode.SetForegroundOpacity(0.0)
            sliceCompositeNode.SetBackgroundVolumeID(self.refVolumeSelector.currentNode().GetID())
            slicercontroller.fitSliceToBackground()



            sliceWidget = slicer.app.layoutManager().sliceWidget("Yellow")
            slicercontroller = slicer.app.layoutManager().sliceWidget("Yellow").sliceController()
            sliceCompositeNode = sliceWidget.mrmlSliceCompositeNode()

            slicercontroller.setForegroundHidden(True)  
            sliceCompositeNode.SetForegroundOpacity(0.0)
            sliceCompositeNode.SetBackgroundVolumeID(self.refVolumeSelector.currentNode().GetID())
            slicercontroller.fitSliceToBackground()




            sliceWidget = slicer.app.layoutManager().sliceWidget("Green")
            slicercontroller = slicer.app.layoutManager().sliceWidget("Green").sliceController()
            sliceCompositeNode = sliceWidget.mrmlSliceCompositeNode()

            slicercontroller.setForegroundHidden(True)  
            sliceCompositeNode.SetForegroundOpacity(0.0)
            sliceCompositeNode.SetBackgroundVolumeID(self.refVolumeSelector.currentNode().GetID())
            slicercontroller.fitSliceToBackground()
        
        
            """
            Sets the default size and centers the ROI in the geometric center of the selected volume upon creation.
            Also ensures that the ROI size updates dynamically when changed.
            """
                # Set default size
            roinode.SetSize(400, 300, 250)

            # Get the selected volume
            inputVolumeNode = self.refVolumeSelector.currentNode()

            # Compute the volume's geometric center in RAS coordinates
            bounds = [0] * 6
            inputVolumeNode.GetRASBounds(bounds)
            centerX = (bounds[0] + bounds[1]) / 2
            centerY = (bounds[2] + bounds[3]) / 2
            centerZ = (bounds[4] + bounds[5]) / 2

                # Set ROI center
            roinode.SetCenter(centerX, centerY, centerZ)



    def CropSPECTImage(self):
        # Set parameters
        cropVolumeLogic = slicer.modules.cropvolume.logic()
        cropVolumeLogic.CropVoxelBased(self.roiSelector.currentNode(), self.inputVolumeSelectorCT.currentNode(), self.inputVolumeSelectorCT.currentNode(), True)
        self.roiSelector.removeCurrentNode ()




    def CropREFImage(self):


        # Set parameters
        cropVolumeLogic = slicer.modules.cropvolume.logic()
        cropVolumeLogic.CropVoxelBased(self.refroiSelector.currentNode(), self.refVolumeSelector.currentNode(), self.refVolumeSelector.currentNode(), True)
        self.refroiSelector.removeCurrentNode ()
   


    def registerImages(self):
        """
        Registers the CT of the SPECT to the Reference Image using rigid registration.
        Applies the resulting transform to both the SPECT CT and SPECT image.
        """
        spectCT = self.inputVolumeSelectorCT.currentNode()
        referenceCT = self.refVolumeSelector.currentNode()

        if not spectCT or not referenceCT:
            slicer.util.errorDisplay("❌ Error: Please select both SPECT CT and Reference CT volumes before registration.")
            return

        print("🚀 Starting  registration...")



        if(self.getSelectedRegistrationMethod()=="Rigid"):

            # **✅ Step 1: Create Transform Node**
            transformNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", "SPECT-CT Registration Transform")

            # **✅ Step 2: Geometry Align Initialization**
            #slicer.modules.registration.logic().ComputeRigidTransform(referenceCT, spectCT, transformNode)

            # **✅ Step 3: Run Rigid Registration**
            parameters = {
                "fixedVolume": referenceCT.GetID(),
                "movingVolume": spectCT.GetID(),
                "initializeTransformMode": "useGeometryAlign",
                "transformType": "Rigid",
                "outputTransform": transformNode.GetID(),
                "samplingPercentage": 0.002,
                "useInitialTransform": True,
            }
            slicer.cli.runSync(slicer.modules.brainsfit, None, parameters)

            print("✅ Registration completed!")
            slicer.util.infoDisplay("✅ Registration completed successfully!")

            # **✅ Step 4: Apply the Transform to Both SPECT CT & SPECT**
            spectCT.SetAndObserveTransformNodeID(transformNode.GetID())
            spect = self.inputVolumeSelector.currentNode()
            if spect:
                spect.SetAndObserveTransformNodeID(transformNode.GetID())

            # **✅ Step 5: Harden the Transform**
            slicer.vtkSlicerTransformLogic().hardenTransform(spectCT)
            if spect:
                slicer.vtkSlicerTransformLogic().hardenTransform(spect)

            print("✅ Transform applied and hardened to both SPECT CT and SPECT.")
            slicer.util.infoDisplay("✅ Transform successfully applied and hardened.")

        elif(self.getSelectedRegistrationMethod()=="Affine"):
            # **✅ Step 1: Create Transform Node**
            transformNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", "SPECT-CT Registration Transform")

            # **✅ Step 2: Geometry Align Initialization**
            #slicer.modules.registration.logic().ComputeRigidTransform(referenceCT, spectCT, transformNode)

            # **✅ Step 3: Run Rigid Registration**
            parameters = {
                "fixedVolume": referenceCT.GetID(),
                "movingVolume": spectCT.GetID(),
                "initializeTransformMode": "useGeometryAlign",
                "transformType": "Affine",
                "outputTransform": transformNode.GetID(),
                "samplingPercentage": 0.002,
                "useInitialTransform": True,
            }
            slicer.cli.runSync(slicer.modules.brainsfit, None, parameters)

            print("✅ Registration completed!")
            slicer.util.infoDisplay("✅ Registration completed successfully!")

            # **✅ Step 4: Apply the Transform to Both SPECT CT & SPECT**
            spectCT.SetAndObserveTransformNodeID(transformNode.GetID())
            spect = self.inputVolumeSelector.currentNode()
            if spect:
                spect.SetAndObserveTransformNodeID(transformNode.GetID())

            # **✅ Step 5: Harden the Transform**
            slicer.vtkSlicerTransformLogic().hardenTransform(spectCT)
            if spect:
                slicer.vtkSlicerTransformLogic().hardenTransform(spect)

            print("✅ Transform applied and hardened to both SPECT CT and SPECT.")
            slicer.util.infoDisplay("✅ Transform successfully applied and hardened.")



        elif(self.getSelectedRegistrationMethod()=="BSpline"):

            # **✅ Step 1: Create Transform Node**
            transformNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLBSplineTransformNode", "SPECT-CT Registration Transform")

            # **✅ Step 2: Geometry Align Initialization**
            #slicer.modules.registration.logic().ComputeRigidTransform(referenceCT, spectCT, transformNode)

            # **✅ Step 3: Run Rigid Registration**
            parameters = {
                "fixedVolume": referenceCT.GetID(),
                "movingVolume": spectCT.GetID(),
                "initializeTransformMode": "useGeometryAlign",
                "transformType": "BSpline",
                "outputTransform": transformNode.GetID(),
                "samplingPercentage": 0.002,
                "useInitialTransform": True,
            }
            slicer.cli.runSync(slicer.modules.brainsfit, None, parameters)

            print("✅ Registration completed!")
            slicer.util.infoDisplay("✅ Registration completed successfully!")

            # **✅ Step 4: Apply the Transform to Both SPECT CT & SPECT**
            spectCT.SetAndObserveTransformNodeID(transformNode.GetID())
            spect = self.inputVolumeSelector.currentNode()
            if spect:
                spect.SetAndObserveTransformNodeID(transformNode.GetID())

            # **✅ Step 5: Harden the Transform**
            slicer.vtkSlicerTransformLogic().hardenTransform(spectCT)
            if spect:
                slicer.vtkSlicerTransformLogic().hardenTransform(spect)

            print("✅ Transform applied and hardened to both SPECT CT and SPECT.")
            slicer.util.infoDisplay("✅ Transform successfully applied and hardened.")









        else:
            return void
        # **✅ Step 6: Visualize Registration**
        self.visualizeRegistration()


    def visualizeRegistration(self):
        """
        Updates the Slicer view to show the reference CT in grayscale and the registered CT in red.
        """
        referenceCT = self.refVolumeSelector.currentNode()
        registeredCT = self.inputVolumeSelectorCT.currentNode()

        if not referenceCT or not registeredCT:
            slicer.util.errorDisplay("❌ Error: Registration visualization failed. Missing volumes.")
            return

        print("🔹 Setting visualization: Reference CT in grayscale, Registered CT in red...")

        # **✅ Set Background to Reference CT (Grayscale)**
        sliceCompositeNode = slicer.app.layoutManager().sliceWidget("Red").mrmlSliceCompositeNode()
        sliceCompositeNode.SetBackgroundVolumeID(referenceCT.GetID())

        # **✅ Set Foreground to Registered CT (Red scale)**
        sliceCompositeNode.SetForegroundVolumeID(registeredCT.GetID())
        sliceCompositeNode.SetForegroundOpacity(0.5)

        # **✅ Set Correct Color Maps**
        referenceCT.GetDisplayNode().SetAndObserveColorNodeID(slicer.util.getNode("Grey").GetID())
        registeredCT.GetDisplayNode().SetAndObserveColorNodeID(slicer.util.getNode("Red").GetID())

                # **✅ Set Background to Reference CT (Grayscale)**
        sliceCompositeNode = slicer.app.layoutManager().sliceWidget("Green").mrmlSliceCompositeNode()
        sliceCompositeNode.SetBackgroundVolumeID(referenceCT.GetID())

        # **✅ Set Foreground to Registered CT (Red scale)**
        sliceCompositeNode.SetForegroundVolumeID(registeredCT.GetID())
        sliceCompositeNode.SetForegroundOpacity(0.5)

        # **✅ Set Background to Reference CT (Grayscale)**
        sliceCompositeNode = slicer.app.layoutManager().sliceWidget("Yellow").mrmlSliceCompositeNode()
        sliceCompositeNode.SetBackgroundVolumeID(referenceCT.GetID())

        # **✅ Set Foreground to Registered CT (Red scale)**
        sliceCompositeNode.SetForegroundVolumeID(registeredCT.GetID())
        sliceCompositeNode.SetForegroundOpacity(0.5)



        print("✅ Registration visualization updated: Reference CT (Grey), Registered CT (Red).")