"""Taranis - Data Roles: the Data and imaging roles step of the Taranis hub, listed on its own in the module finder.

Entering the module opens the hub on this step (see TaranisLib.shortcut): case, status and page are the hub's.
"""

import os

from slicer.ScriptedLoadableModule import *

from TaranisLib import workflow as W
from TaranisLib.shortcut import StepShortcut, setupShortcutModule

STEP = W.STEP_DATA
STEP_LABEL = "Data and imaging roles"


class TaranisDataRoles(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        setupShortcutModule(parent, "Taranis - Data Roles", STEP_LABEL, "Data and imaging roles of the case: which volume is the anatomy, the dosimetry image, the segmentation.",
                            os.path.join(os.path.dirname(__file__), "Resources", "Icons", "TaranisDataRoles.png"))


class TaranisDataRolesWidget(ScriptedLoadableModuleWidget):

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.shortcut = StepShortcut(STEP, STEP_LABEL)
        self.shortcut.setup(self.layout)

    def enter(self):
        self.shortcut.enter()
