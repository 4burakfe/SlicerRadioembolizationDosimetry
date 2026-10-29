"""Step shortcut modules: Taranis steps listed on their own in the module finder.

A shortcut module has no page of its own: entering it opens the Taranis hub on its step, so the case, the
workflow status and the step page are the hub's (nothing is duplicated and the hub workflow is unchanged).
"""

import logging
import os

import qt
import slicer

from .controller import WorkflowController
from .toolbar import HUB_MODULE, openHub

SUBMODULES_CATEGORY = "Nuclear Medicine.Taranis - Submodules"


def setupShortcutModule(parent, title, stepLabel, helpText, iconPath=None):
    """Module meta-information of a step shortcut."""
    parent.title = title
    parent.categories = [SUBMODULES_CATEGORY]
    parent.dependencies = [HUB_MODULE]
    parent.contributors = ["Burak Demir, MD, FEBNM"]
    parent.helpText = (f"{helpText}<br>Opens the <b>{stepLabel}</b> step of the Taranis hub, on the open "
                       "case.<br>NOT a medical device. Research use only.")
    parent.acknowledgementText = "Developed by Burak Demir, MD, FEBNM."
    if iconPath and os.path.exists(iconPath):
        parent.icon = qt.QIcon(iconPath)


class StepShortcut:
    """Widget logic of a shortcut module: on enter, switch to the hub on the step (deferred, because a module
    must not be selected while Slicer is still entering another one)."""

    def __init__(self, step, stepLabel):
        self.step = step
        self.stepLabel = stepLabel

    def setup(self, layout):
        label = qt.QLabel(f"This module opens the <b>{self.stepLabel}</b> step of the Taranis hub.")
        label.setWordWrap(True)
        layout.addWidget(label)
        button = qt.QPushButton(f"Open Taranis: {self.stepLabel}")
        button.connect("clicked()", self.open)
        layout.addWidget(button)
        layout.addStretch(1)

    def enter(self):
        qt.QTimer.singleShot(0, self.open)

    def open(self):
        if not hasattr(slicer.modules, HUB_MODULE.lower()):
            slicer.util.errorDisplay(f"The module '{HUB_MODULE}' is not loaded.")
            return
        try:
            active = WorkflowController.instance().isActive
        except Exception as e:
            logging.warning(f"Taranis: {e}")
            active = False
        if not active:
            openHub("home")
            slicer.util.showStatusMessage(
                f"Taranis: create or open a case first, then continue with {self.stepLabel}.", 6000)
            return
        openHub(self.step)
