"""Rules for showing the workflow toolbar. Pure Python, so the rules can be tested outside Slicer.

- The first time Taranis is opened the toolbar is added. It is not shown at Slicer start unless
  "Show the workflow toolbar when Slicer starts" is on (off by default).
- "Close" hides it for this session. "Disable at startup" stops showing it at Slicer start.
- Opening Taranis shows it (unless it was closed in this session).
- Starting or resuming a case always shows it.
- When it is not shown at startup, closing the scene / case hides it again.
"""


class ToolbarVisibility:

    def __init__(self, initialized=False, showAtStartup=False):
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
