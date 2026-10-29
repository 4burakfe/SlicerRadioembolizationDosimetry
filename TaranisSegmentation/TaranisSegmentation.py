"""Taranis - Segmentation: the Segmentation step of the Taranis hub, listed on its own in the module finder.

Entering the module opens the hub on this step (see TaranisLib.shortcut): case, status and page are the hub's.
"""

import os

from slicer.ScriptedLoadableModule import *

from TaranisLib import workflow as W
from TaranisLib.shortcut import StepShortcut, setupShortcutModule

STEP = W.STEP_SEGMENTATION
STEP_LABEL = "Segmentation"


class TaranisSegmentation(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        setupShortcutModule(parent, "Taranis - Segmentation", STEP_LABEL, "Segmentation of the case: whole liver, perfused volumes, tumours and lungs.",
                            os.path.join(os.path.dirname(__file__), "Resources", "Icons", "TaranisSegmentation.png"))


class TaranisSegmentationWidget(ScriptedLoadableModuleWidget):

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.shortcut = StepShortcut(STEP, STEP_LABEL)
        self.shortcut.setup(self.layout)

    def enter(self):
        self.shortcut.enter()
