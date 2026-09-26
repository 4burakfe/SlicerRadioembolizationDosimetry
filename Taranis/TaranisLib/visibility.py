"""Rules for showing the workflow toolbar. Pure Python, so the rules can be tested outside Slicer.

- The first time Taranis is opened the toolbar is added, and from then on it is shown at every Slicer start.
- "Close" hides it for this session. "Disable at startup" stops showing it at Slicer start.
- Starting or resuming a case always shows it.
- With "Disable at startup" on, closing the scene / case hides it again.
"""


class ToolbarVisibility:

    def __init__(self, initialized=False, showAtStartup=True):
        self.initialized = initialized
        self.showAtStartup = showAtStartup
        self.closedByUser = False
        self.visible = initialized and showAtStartup

    def onHubOpened(self):
        """The Taranis module was opened. Returns True if 'initialized' changed (to be saved in settings)."""
        firstTime = not self.initialized
        self.initialized = True
        if firstTime or not self.closedByUser:
            self.visible = True
        return firstTime

    def onCaseActivated(self):
        self.closedByUser = False
        self.visible = True

    def onCaseDeactivated(self):
        if not self.showAtStartup:
            self.visible = False

    def onClosePressed(self):
        self.closedByUser = True
        self.visible = False

    def onShowRequested(self):
        self.closedByUser = False
        self.visible = True

    def setShowAtStartup(self, show):
        self.showAtStartup = bool(show)
