"""Workflow toolbar at the top of the Slicer main window.

Idle (no case in the scene): one button "Start TARE dosimetry workflow".
Active: case name and ID, the six steps with a status badge, an issue counter with a menu, and on the right
"Disable at startup", "Close" and Epona. Visibility rules are in visibility.py.

The toolbar adapts to the width of the window (COMPACT_LEVELS): on small screens the explanation lines are dropped,
the step buttons get narrower, "Disable at startup" and "Close" move into a "⋯" menu, and finally the steps show
their badge only (details in the tooltips), so that the whole toolbar, Epona included, stays visible.
"""

import logging
import os

import qt
import slicer

from . import workflow as W
from .case import (settingBool, setSetting, SETTING_TOOLBAR_INITIALIZED, SETTING_SHOW_AT_STARTUP)
from .controller import WorkflowController
from .visibility import ToolbarVisibility

TOOLBAR_OBJECT_NAME = "TaranisWorkflowToolbar"
HUB_MODULE = "Taranis"
BADGE_SIZE = 24                # status badge (px)
BUTTON_HEIGHT = 54             # two-line toolbar buttons (px)
TITLE_PIXEL_SIZE = 14
SUBTITLE_PIXEL_SIZE = 11
STEP_TEXT_WIDTH = 165          # text column of a step button (px); the explanation wraps onto two lines
SUBTITLE_COLOR = "#9aa3ab"
LOGO_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Resources", "Icons", "Taranis.png")
EPONA_LOGO_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Resources", "Icons", "Epona.png")
EPONA_MODULE = "Easy_fusion"
EPONA_LOGO_SIZE = 40
CASE_TEXT_WIDTH_MAX = 240      # the case name is elided beyond this width
WIDTH_SLACK = 24               # px kept free when choosing the compaction level
WIDTH_CHECK_INTERVAL_MS = 1000  # fallback polling of the window width (resize events are the main trigger)

# Compaction levels, tried in order until the toolbar fits in the width available to it.
#   step: (text width or None = title width, explanation lines, show texts)  case: (max text width, lines)
#   issues: (lines, show texts)  epona: (text width, lines, show texts)  fold: "Disable at startup" and "Close"
#   in the "⋯" menu  arrows: "›" between the steps  logo: Taranis logo
COMPACT_LEVELS = [
    dict(step=(STEP_TEXT_WIDTH, 2, True), case=(CASE_TEXT_WIDTH_MAX, 1), issues=(1, True), epona=(150, 2, True),
         fold=False, arrows=True, logo=True),
    dict(step=(STEP_TEXT_WIDTH, 2, True), case=(200, 1), issues=(1, True), epona=(None, 0, True),
         fold=True, arrows=True, logo=True),
    dict(step=(125, 2, True), case=(160, 1), issues=(1, True), epona=(None, 0, True),
         fold=True, arrows=True, logo=True),
    dict(step=(None, 0, True), case=(140, 1), issues=(0, True), epona=(None, 0, False),
         fold=True, arrows=True, logo=True),
    dict(step=(None, 0, True), case=(110, 0), issues=(0, False), epona=(None, 0, False),
         fold=True, arrows=False, logo=False),
    dict(step=(None, 0, False), case=(90, 0), issues=(0, False), epona=(None, 0, False),
         fold=True, arrows=False, logo=False),
]

_iconCache = {}


def badgeIcon(state, text=""):
    """Round status badge: filled circle with a glyph, or an outlined circle with the step number."""
    key = (state, text)
    if key in _iconCache:
        return _iconCache[key]
    _, color, glyph = W.STATE_STYLE[state]
    scale = 2  # draw at twice the size for high-DPI screens
    size = BADGE_SIZE * scale
    pixmap = qt.QPixmap(size, size)
    pixmap.fill(qt.QColor(0, 0, 0, 0))
    painter = qt.QPainter(pixmap)
    try:
        painter.setRenderHint(qt.QPainter.Antialiasing)
        outlined = state in (W.STATE_NOT_STARTED, W.STATE_LOCKED, W.STATE_NOT_APPLICABLE)
        pen = qt.QPen(qt.QColor(color))
        pen.setWidth(2 * scale)
        if outlined:
            painter.setPen(pen)
            painter.setBrush(qt.QBrush(qt.QColor(0, 0, 0, 0)))
        else:
            painter.setPen(qt.Qt.NoPen)
            painter.setBrush(qt.QBrush(qt.QColor(color)))
        margin = 2 * scale
        painter.drawEllipse(margin, margin, size - 2 * margin, size - 2 * margin)
        label = text if (outlined and state == W.STATE_NOT_STARTED) else glyph
        if label:
            font = qt.QFont()
            font.setBold(True)
            font.setPixelSize(int(size * 0.55))
            painter.setFont(font)
            painter.setPen(qt.QColor(color) if outlined else qt.QColor("white"))
            painter.drawText(qt.QRect(0, 0, size, size), qt.Qt.AlignCenter, label)
    finally:
        painter.end()
    icon = qt.QIcon(pixmap)
    _iconCache[key] = icon
    return icon


def openHub(step=None):
    """Open the Taranis module, optionally on a step ("home" for the case overview)."""
    slicer.util.selectModule(HUB_MODULE)
    try:
        widget = slicer.util.getModuleWidget(HUB_MODULE)
        if step:
            widget.showStep(step)
        return widget
    except Exception as e:
        logging.warning(f"Taranis: could not open the hub on step {step}: {e}")
        return None


def _textWidth(fontMetrics, text):
    try:
        return fontMetrics.horizontalAdvance(text)
    except AttributeError:  # Qt < 5.11
        return fontMetrics.width(text)


class TwoLineButton:
    """Tool button with an optional status badge, a title and a small explanation line underneath.

    The look can be compacted (see setMode): fewer or no explanation lines, a narrower text column, or the badge
    alone (the texts then live in the tooltip)."""

    def __init__(self, textWidth=None, badge=True, checkable=False, subtitleLines=2, badgeSize=BADGE_SIZE,
                 maxTextWidth=None):
        self.textWidth = textWidth
        self.maxTextWidth = maxTextWidth
        self.badgeSize = badgeSize
        self.subtitleLines = subtitleLines
        self.showText = True
        self._title, self._subtitle, self._subtitleColor = "", "", SUBTITLE_COLOR
        self.button = qt.QToolButton()
        self.button.setAutoRaise(True)
        self.button.checkable = checkable
        self.button.setFixedHeight(BUTTON_HEIGHT)
        self.button.setStyleSheet(
            "QToolButton { border-radius: 5px; padding: 0px; }"
            "QToolButton:checked { background-color: rgba(47, 126, 216, 0.28); border: 1px solid #2f7ed8; }"
            "QToolButton::menu-indicator { image: none; width: 0px; }")
        self.layout = qt.QHBoxLayout(self.button)
        self.layout.setContentsMargins(7, 3, 9, 3)
        self.layout.setSpacing(7)
        self.badge = None
        if badge:
            self.badge = qt.QLabel()
            self.badge.setFixedSize(badgeSize, badgeSize)
            self.layout.addWidget(self.badge, 0, qt.Qt.AlignVCenter)
        column = qt.QVBoxLayout()
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        self.title = qt.QLabel()
        titleFont = qt.QFont(self.title.font)
        titleFont.setPixelSize(TITLE_PIXEL_SIZE)
        titleFont.setBold(True)
        self.title.setFont(titleFont)
        self.subtitle = qt.QLabel()
        subtitleFont = qt.QFont(self.subtitle.font)
        subtitleFont.setPixelSize(SUBTITLE_PIXEL_SIZE)
        self.subtitle.setFont(subtitleFont)
        self.subtitle.setAlignment(qt.Qt.AlignLeft | qt.Qt.AlignTop)
        column.addStretch(1)
        column.addWidget(self.title)
        column.addWidget(self.subtitle)
        column.addStretch(1)
        self.layout.addLayout(column)
        for widget in (self.badge, self.title, self.subtitle):
            if widget is not None:
                widget.setAttribute(qt.Qt.WA_TransparentForMouseEvents)
        self._render()

    def setMode(self, textWidth=None, subtitleLines=None, showText=True, maxTextWidth=None):
        """Compact or restore the button. textWidth None sizes the text column to the texts (up to maxTextWidth)."""
        mode = (textWidth, self.subtitleLines if subtitleLines is None else subtitleLines, showText, maxTextWidth)
        if mode == (self.textWidth, self.subtitleLines, self.showText, self.maxTextWidth):
            return
        self.textWidth, self.subtitleLines, self.showText, self.maxTextWidth = mode
        self._render()

    def setBadge(self, icon):
        if self.badge is not None:
            self.badge.setPixmap(icon.pixmap(self.badgeSize, self.badgeSize))

    def setTexts(self, title, subtitle, subtitleColor=SUBTITLE_COLOR):
        self._title, self._subtitle, self._subtitleColor = title, subtitle, subtitleColor
        self._render()

    def _render(self):
        showText = self.showText or self.badge is None
        lines = self.subtitleLines if showText else 0
        self.title.setVisible(showText)
        self.subtitle.setVisible(lines > 0)
        badgeWidth = (self.badgeSize + 7) if self.badge is not None else 0
        if not showText:  # badge only, centred
            self.layout.setContentsMargins(6, 3, 6, 3)
            self.button.setFixedWidth(self.badgeSize + 12)
            return
        self.layout.setContentsMargins(7, 3, 9, 3)
        titleMetrics = qt.QFontMetrics(self.title.font)
        subtitleMetrics = qt.QFontMetrics(self.subtitle.font)
        width = self.textWidth
        if not width:  # size to the texts
            width = _textWidth(titleMetrics, self._title)
            if lines > 0:
                width = max(width, _textWidth(subtitleMetrics, self._subtitle))
            width += 4
            if self.maxTextWidth:
                width = min(width, self.maxTextWidth)
        self.title.setFixedWidth(width)
        self.title.text = titleMetrics.elidedText(self._title, qt.Qt.ElideRight, width)
        if lines > 0:
            self.subtitle.setWordWrap(lines > 1)
            self.subtitle.setFixedHeight(subtitleMetrics.lineSpacing() * lines)
            self.subtitle.setFixedWidth(width)
            self.subtitle.setStyleSheet(f"color: {self._subtitleColor};")
            available = width * lines - (12 if lines > 1 else 0)
            self.subtitle.text = subtitleMetrics.elidedText(self._subtitle, qt.Qt.ElideRight, available)
        self.button.setFixedWidth(width + badgeWidth + 7 + 9 + 2)


class _ResizeFilter(qt.QObject):
    """Calls back when the watched widget (the main window) is resized."""

    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def eventFilter(self, obj, event):
        try:
            if event.type() == qt.QEvent.Resize:
                self.callback()
        except Exception:
            pass
        return False


class WorkflowToolbar:

    _instance = None

    @classmethod
    def instance(cls, create=True):
        if cls._instance is None and create:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def destroyInstance(cls):
        if cls._instance is not None:
            cls._instance.destroy()
            cls._instance = None

    def __init__(self):
        self.controller = WorkflowController.instance()
        self.visibility = ToolbarVisibility(settingBool(SETTING_TOOLBAR_INITIALIZED, False),
                                            settingBool(SETTING_SHOW_AT_STARTUP, True))
        self._wasActive = self.controller.isActive
        if self._wasActive:
            self.visibility.onCaseActivated()
        self.compactLevel = None
        self._layingOut = False
        self._lastAvailableWidth = None
        self._build()
        self._watchWidth()
        self.controller.addListener(self.onControllerUpdated)
        self.onControllerUpdated(self.controller)

    # -- Construction --

    def _build(self):
        mainWindow = slicer.util.mainWindow()
        # A developer reload never deletes the previous toolbar (deleting Qt widgets that Python still wraps is
        # unsafe): it is hidden and renamed, and the new toolbar takes its place.
        previous = [w for w in mainWindow.children()
                    if isinstance(w, qt.QToolBar) and w.objectName == TOOLBAR_OBJECT_NAME]
        self.toolbar = qt.QToolBar("Taranis workflow", mainWindow)
        self.toolbar.objectName = TOOLBAR_OBJECT_NAME
        if previous:
            old = previous[-1]
            mainWindow.insertToolBar(old, self.toolbar)
            old.setVisible(False)
            old.objectName = TOOLBAR_OBJECT_NAME + "Retired"
            old.toggleViewAction().setVisible(False)
        else:
            mainWindow.addToolBarBreak()
            mainWindow.addToolBar(self.toolbar)
        self.toolbar.setMovable(True)

        logo = qt.QLabel()
        if os.path.exists(LOGO_PATH):
            logo.setPixmap(qt.QPixmap(LOGO_PATH).scaledToHeight(BUTTON_HEIGHT - 10, qt.Qt.SmoothTransformation))
        logo.setToolTip("Taranis - radioembolization dosimetry suite")
        logo.setContentsMargins(4, 0, 6, 0)
        self.logoAction = self.toolbar.addWidget(logo)

        # Idle
        self.startButton = TwoLineButton(badge=False, subtitleLines=1)
        self.startButton.setTexts("Start TARE dosimetry workflow", "New case, or resume the case of this scene")
        self.startButton.button.setToolTip("Open Taranis and start (or resume) a radioembolization dosimetry case.")
        self.startButton.button.connect("clicked()", self.onStartClicked)
        self.idleActions = [self.toolbar.addWidget(self.startButton.button)]

        # Active case
        self.activeActions = []
        self.caseButton = TwoLineButton(badge=False, subtitleLines=1, maxTextWidth=CASE_TEXT_WIDTH_MAX)
        self.caseButton.button.connect("clicked()", lambda: openHub("home"))
        self.activeActions.append(self.toolbar.addWidget(self.caseButton.button))
        self.activeActions.append(self.toolbar.addSeparator())

        self.stepButtons = {}
        self.arrowActions = []  # shown by _applyLevel
        for number, (key, label) in enumerate(W.STEPS, start=1):
            if number > 1:
                arrow = qt.QLabel("\u203a")
                arrow.setStyleSheet("color: #9aa0a6; font-size: 20px; padding: 0 1px;")
                self.arrowActions.append(self.toolbar.addWidget(arrow))
            step = TwoLineButton(textWidth=STEP_TEXT_WIDTH, checkable=True)
            step.setTexts(f"{number}. {label}", "")
            step.button.connect("clicked()", lambda key=key: self.onStepClicked(key))
            self.stepButtons[key] = step
            self.activeActions.append(self.toolbar.addWidget(step.button))

        self.activeActions.append(self.toolbar.addSeparator())
        self.issuesButton = TwoLineButton(subtitleLines=1)
        self.issuesButton.button.setPopupMode(qt.QToolButton.InstantPopup)
        self.issuesMenu = qt.QMenu(self.issuesButton.button)
        self.issuesButton.button.setMenu(self.issuesMenu)
        self.activeActions.append(self.toolbar.addWidget(self.issuesButton.button))

        spacer = qt.QWidget()
        spacer.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Preferred)
        self.toolbar.addWidget(spacer)

        self.disableAtStartupButton = qt.QToolButton()
        self.disableAtStartupButton.text = "Disable at startup"
        self.disableAtStartupButton.checkable = True
        self.disableAtStartupButton.checked = not self.visibility.showAtStartup
        self.disableAtStartupButton.setToolTip(
            "Checked: the toolbar is not shown when Slicer starts; it appears only when a case is started or "
            "resumed, and disappears when the case is closed.")
        self.disableAtStartupButton.connect("toggled(bool)", self.onDisableAtStartupToggled)
        self.disableAction = self.toolbar.addWidget(self.disableAtStartupButton)

        self.closeButton = qt.QToolButton()
        self.closeButton.text = "Close"
        self.closeButton.setToolTip("Hide the workflow toolbar for this session. It comes back when a case is "
                                    "started, or with 'Show workflow toolbar' in the Taranis module.")
        self.closeButton.connect("clicked()", self.onCloseClicked)
        self.closeAction = self.toolbar.addWidget(self.closeButton)

        # Narrow windows: "Disable at startup" and "Close" folded into a menu
        self.optionsButton = qt.QToolButton()
        self.optionsButton.text = "\u22ef"
        self.optionsButton.setStyleSheet("QToolButton { font-size: 18px; padding: 0 6px; }"
                                         "QToolButton::menu-indicator { image: none; width: 0px; }")
        self.optionsButton.setToolTip("Toolbar options")
        self.optionsButton.setPopupMode(qt.QToolButton.InstantPopup)
        self.optionsMenu = qt.QMenu(self.optionsButton)
        self.disableMenuAction = self.optionsMenu.addAction("Disable at startup")
        self.disableMenuAction.checkable = True
        self.disableMenuAction.checked = self.disableAtStartupButton.checked
        self.disableMenuAction.setToolTip(self.disableAtStartupButton.toolTip)
        self.disableMenuAction.connect("toggled(bool)", self._onDisableMenuToggled)
        closeMenuAction = self.optionsMenu.addAction("Close toolbar")
        closeMenuAction.connect("triggered()", self.onCloseClicked)
        self.optionsButton.setMenu(self.optionsMenu)
        self.optionsAction = self.toolbar.addWidget(self.optionsButton)
        self.optionsAction.setVisible(False)

        # Epona - SPECT/PET review (SlicerPETDenoise extension), always available at the far right
        self.toolbar.addSeparator()
        self.eponaButton = TwoLineButton(badge=True, subtitleLines=2, badgeSize=EPONA_LOGO_SIZE, textWidth=150)
        if os.path.exists(EPONA_LOGO_PATH):
            self.eponaButton.setBadge(qt.QIcon(EPONA_LOGO_PATH))
        self.eponaButton.setTexts("Epona", "SPECT/PET review: fusion, MIP, SUV and count ROIs")
        self.eponaButton.button.setToolTip(
            "Epona - SPECT/PET Review: easy fusion of SPECT/PET with CT/MRI, MIP, spherical ROIs with max, mean, "
            "MTV and TLG, window presets and review layouts. Useful to explore the images of a case.")
        self.eponaButton.button.connect("clicked()", self.onEponaClicked)
        self.toolbar.addWidget(self.eponaButton.button)

    def destroy(self):
        """Disconnect; the toolbar is hidden and replaced by the next instance (see _build)."""
        self.controller.removeListener(self.onControllerUpdated)
        try:
            self._widthTimer.stop()
            self._layoutTimer.stop()
            if self._resizeFilter is not None:
                slicer.util.mainWindow().removeEventFilter(self._resizeFilter)
        except Exception as e:
            logging.warning(f"Taranis: could not stop the toolbar width watch: {e}")
        try:
            self.toolbar.setVisible(False)
        except Exception as e:
            logging.warning(f"Taranis: could not hide the toolbar: {e}")

    # -- Visibility --

    def applyVisibility(self):
        self.toolbar.setVisible(self.visibility.visible)

    def onHubOpened(self):
        if self.visibility.onHubOpened():
            setSetting(SETTING_TOOLBAR_INITIALIZED, True)
        self.applyVisibility()

    def show(self):
        self.visibility.onShowRequested()
        if not self.visibility.initialized:
            self.visibility.initialized = True
            setSetting(SETTING_TOOLBAR_INITIALIZED, True)
        self.applyVisibility()

    def onCloseClicked(self):
        self.visibility.onClosePressed()
        self.applyVisibility()
        slicer.util.showStatusMessage("Taranis toolbar closed. Reopen it from the Taranis module.", 5000)

    def onDisableAtStartupToggled(self, checked):
        self.visibility.setShowAtStartup(not checked)
        setSetting(SETTING_SHOW_AT_STARTUP, not checked)
        if self.disableMenuAction.checked != checked:
            self.disableMenuAction.checked = checked

    def _onDisableMenuToggled(self, checked):
        if self.disableAtStartupButton.checked != checked:
            self.disableAtStartupButton.checked = checked  # -> onDisableAtStartupToggled

    def setShowAtStartup(self, show):
        """Called by the hub's settings: keeps the toolbar button in sync."""
        self.disableAtStartupButton.checked = not show

    # -- Actions --

    def onStartClicked(self):
        widget = openHub("home")
        if widget is not None and not self.controller.isActive:
            qt.QTimer.singleShot(0, widget.onNewCaseClicked)

    def onEponaClicked(self):
        if hasattr(slicer.modules, EPONA_MODULE.lower()):
            slicer.util.selectModule(EPONA_MODULE)
        else:
            slicer.util.infoDisplay(
                "Epona - SPECT/PET Review is part of the SlicerPETDenoise extension (PETDenoise in the Extensions "
                "Manager). Install it and restart Slicer to open Epona from here.", windowTitle="Epona")

    def onStepClicked(self, key):
        openHub(key)
        self._refreshChecked()

    # -- Refresh --

    def onControllerUpdated(self, controller):
        active = controller.isActive
        if active != self._wasActive:
            if active:
                self.visibility.onCaseActivated()
            else:
                self.visibility.onCaseDeactivated()
            self._wasActive = active
        for action in self.idleActions:
            action.setVisible(not active)
        for action in self.activeActions:
            action.setVisible(active)
        if active:
            self._refreshActive(controller)
        self.updateLayout()
        self.applyVisibility()

    def _refreshChecked(self):
        current = self.controller.currentStep
        for key, step in self.stepButtons.items():
            step.button.checked = (key == current)

    def _refreshActive(self, controller):
        case = controller.case
        self.caseButton.setTexts(case.name or "Unnamed case", f"ID: {case.caseID}" if case.caseID else "No ID")
        self.caseButton.button.setToolTip(f"Case: {case.name}\nID: {case.caseID or '-'}\n"
                                          "Click for the case overview.")
        symbols = {W.SEVERITY_ERROR: "\u2715", W.SEVERITY_WARNING: "\u26a0", W.SEVERITY_INFO: "\u2139"}
        for number, (key, label) in enumerate(W.STEPS, start=1):
            status = controller.status(key)
            step = self.stepButtons[key]
            if status is None:
                step.setBadge(badgeIcon(W.STATE_NOT_STARTED, str(number)))
                step.setTexts(f"{number}. {label}", "")
                step.button.setToolTip(f"<b>{number}. {label}</b>")
                continue
            step.setBadge(badgeIcon(status.state, str(number)))
            stateLabel, stateColor, _ = W.STATE_STYLE[status.state]
            # the explanation line: the step summary, or the first error/warning when there is one
            problems = [i for i in status.issues if i.severity in (W.SEVERITY_ERROR, W.SEVERITY_WARNING)]
            if status.state in (W.STATE_ERROR, W.STATE_WARNING, W.STATE_OUTDATED) and problems:
                explanation, color = problems[0].text, stateColor
            else:
                explanation, color = status.summary or stateLabel, SUBTITLE_COLOR
            step.setTexts(f"{number}. {label}", explanation, color)
            lines = [f"<b>{number}. {label}</b> \u2013 {stateLabel}"]
            if status.summary:
                lines.append(status.summary)
            for issue in status.issues:
                lines.append(f"{symbols[issue.severity]} {issue.text}")
            step.button.setToolTip("<br>".join(lines))
        self._refreshChecked()

        issues = W.allIssues(controller.statuses)
        errors = sum(1 for i in issues if i.severity == W.SEVERITY_ERROR)
        warnings = len(issues) - errors
        self.issuesMenu.clear()
        if not issues:
            self.issuesButton.setTexts("No issues", "No errors or warnings")
            self.issuesButton.setBadge(badgeIcon(W.STATE_DONE))
            action = self.issuesMenu.addAction("No errors or warnings.")
            action.enabled = False
        else:
            parts = []
            if errors:
                parts.append(f"{errors} error{'s' if errors > 1 else ''}")
            if warnings:
                parts.append(f"{warnings} warning{'s' if warnings > 1 else ''}")
            self.issuesButton.setTexts(", ".join(parts), "Click to see them")
            self.issuesButton.setBadge(badgeIcon(W.STATE_ERROR if errors else W.STATE_WARNING))
            for issue in issues:
                state = W.STATE_ERROR if issue.severity == W.SEVERITY_ERROR else W.STATE_WARNING
                action = self.issuesMenu.addAction(badgeIcon(state), f"{W.STEP_LABELS[issue.step]}: {issue.text}")
                action.connect("triggered()", lambda step=issue.step: self.onStepClicked(step))
        self.issuesButton.button.setToolTip(f"<b>{self.issuesButton._title}</b><br>Errors and warnings of all "
                                            "steps. Click an item to go to its step.")

    # -- Adaptive layout --

    def _watchWidth(self):
        self._layoutTimer = qt.QTimer()
        self._layoutTimer.setSingleShot(True)
        self._layoutTimer.setInterval(60)
        self._layoutTimer.connect("timeout()", self.updateLayout)
        self._resizeFilter = None
        try:
            self._resizeFilter = _ResizeFilter(self._layoutTimer.start)
            slicer.util.mainWindow().installEventFilter(self._resizeFilter)
        except Exception as e:
            self._resizeFilter = None
            logging.warning(f"Taranis: no resize events for the toolbar ({e}); the window width is polled.")
        # fallback: cheap check of the width (docking/undocking, screen changes, missed events)
        self._widthTimer = qt.QTimer()
        self._widthTimer.setInterval(WIDTH_CHECK_INTERVAL_MS)
        self._widthTimer.connect("timeout()", self._checkWidth)
        self._widthTimer.start()

    def _checkWidth(self):
        try:
            if self.toolbar.visible and self._availableWidth() != self._lastAvailableWidth:
                self.updateLayout()
        except Exception:
            pass

    def _availableWidth(self):
        """Width the toolbar can use: the main window width, minus the toolbars sharing its row."""
        mainWindow = slicer.util.mainWindow()
        width = mainWindow.width
        if self.toolbar.floating:
            return width
        if self.toolbar.orientation == qt.Qt.Vertical:
            return 1 << 20  # height-limited, not width-limited
        if not self.toolbar.visible:
            return width
        area = mainWindow.toolBarArea(self.toolbar)
        row = self.toolbar.geometry.y()
        for other in mainWindow.children():  # the main window toolbars are its direct children
            if not isinstance(other, qt.QToolBar) or other.objectName.startswith(TOOLBAR_OBJECT_NAME):
                continue
            if not other.visible or other.floating or mainWindow.toolBarArea(other) != area:
                continue
            if other.geometry.y() == row:
                width -= other.sizeHint.width()
        return width

    def _requiredWidth(self):
        """Width the visible toolbar items need (as the toolbar layout computes it, approximately)."""
        style = self.toolbar.style()
        metric = lambda m: style.pixelMetric(m, None, self.toolbar)
        total = 2 * (metric(qt.QStyle.PM_ToolBarItemMargin) + metric(qt.QStyle.PM_ToolBarFrameWidth))
        if self.toolbar.movable:
            total += metric(qt.QStyle.PM_ToolBarHandleExtent)
        count = 0
        for action in self.toolbar.actions():
            if not action.visible:
                continue
            widget = self.toolbar.widgetForAction(action)
            if widget is None:
                continue
            width = min(max(widget.sizeHint.width(), widget.minimumWidth), widget.maximumWidth)
            total += max(width, 0)
            count += 1
        return total + metric(qt.QStyle.PM_ToolBarItemSpacing) * max(count - 1, 0) + WIDTH_SLACK

    def _applyLevel(self, level):
        spec = COMPACT_LEVELS[level]
        self.compactLevel = level
        active = self.controller.isActive
        textWidth, lines, showText = spec["step"]
        for step in self.stepButtons.values():
            step.setMode(textWidth, lines, showText)
        maxWidth, lines = spec["case"]
        self.caseButton.setMode(None, lines, True, maxWidth)
        lines, showText = spec["issues"]
        self.issuesButton.setMode(None, lines, showText)
        textWidth, lines, showText = spec["epona"]
        self.eponaButton.setMode(textWidth, lines, showText)
        self.startButton.setMode(None, spec["issues"][0])
        for action in self.arrowActions:
            action.setVisible(active and spec["arrows"])
        self.logoAction.setVisible(spec["logo"])
        self.disableAction.setVisible(not spec["fold"])
        self.closeAction.setVisible(not spec["fold"])
        self.optionsAction.setVisible(spec["fold"])

    def updateLayout(self):
        """Choose the least compact level with which the toolbar fits in the window."""
        if self._layingOut:
            return
        self._layingOut = True
        self.toolbar.setUpdatesEnabled(False)
        try:
            available = self._availableWidth()
            self._lastAvailableWidth = available
            for level in range(len(COMPACT_LEVELS)):
                self._applyLevel(level)
                if self._requiredWidth() <= available:
                    break
        except Exception as e:
            logging.warning(f"Taranis: toolbar layout failed: {e}")
        finally:
            self.toolbar.setUpdatesEnabled(True)
            self._layingOut = False
