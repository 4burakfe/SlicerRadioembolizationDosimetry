"""Views of the Taranis hub: the Segmentation step layout and keeping other modules' renderings out of it.

The layout uses its own view nodes (not Slicer's default "1", Red, Green, Yellow), so volume renderings and models
of other modules (Epona MIP, dosimetry isodose / segment models) do not appear in it, and nothing shown here
appears in their views.

    +-------------------+-------------------+-----------+
    | axial fused       | axial reference   |           |
    | (anatomy + SPECT) | + segments        |    3D     |
    +-------------------+-------------------+ segments  |
    | coronal fused     | coronal reference | wireframe |
    |                   | + segments        |           |
    +-------------------+-------------------+-----------+

Dual monitor (SEGMENTATION_DUAL_LAYOUT_ID): the four slice views (2 x 2) stay in the main window and the 3D view
opens in its own window, maximized on the second screen. Same view nodes as the single monitor layout. The layout
is chosen like in the dosimetry modules: automatically from the number of screens, or with the Single / Dual
monitor buttons (one choice shared by the Taranis modules).
"""

import json
import logging

import qt
import slicer

from . import workflow as W

SEGMENTATION_LAYOUT_ID = 50401   # EasyReg 50201/50202, dosimetry 50101/50102, LSF 50301
SEGMENTATION_DUAL_LAYOUT_ID = 50402
SEGMENTATION_LAYOUT_IDS = (SEGMENTATION_LAYOUT_ID, SEGMENTATION_DUAL_LAYOUT_ID)
DUAL_WINDOW_NAME = "Taranis segments 3D"
SLICE_VIEWS = {
    # key: (singleton tag, orientation, label, colour, fused)
    "axialFused": ("TaranisSegAxialFused", "Axial", "AF", "#F34A33", True),
    "axialReference": ("TaranisSegAxialRef", "Axial", "A", "#F34A33", False),
    "coronalFused": ("TaranisSegCoronalFused", "Coronal", "CF", "#6EB04B", True),
    "coronalReference": ("TaranisSegCoronalRef", "Coronal", "C", "#6EB04B", False),
}
THREED_VIEW_TAG = "TaranisSeg3D"
FUSION_OPACITY = 0.5
FUNCTIONAL_COLORMAP = "Inferno"

SEGMENTATION_3D_ATTRIBUTE = "Taranis.Segmentation3D"       # on the extra display node of the 3D view
SAVED_VIEWS_ATTRIBUTE = "Taranis.SavedViewNodeIDs"         # on the main display node while the layout is shown
SAVED_FILL_ATTRIBUTE = "Taranis.SavedFillOpacity"          # idem: 2D fill opacity before the layout
FILL_OPACITY_2D = 0.2                                      # segment fill in the reference slice views
# Wireframe opacity per role, as in the dosimetry modules (dense meshes: low opacity)
SEGMENT_3D_OPACITY = {W.SEGMENT_LIVER: 0.03, W.SEGMENT_PERFUSED: 0.05, W.SEGMENT_TUMOR: 0.10, W.SEGMENT_VIABLE: 0.10,
                      W.SEGMENT_LUNGS: 0.03, W.SEGMENT_OTHER: 0.05, "": 0.05}
CANDIDATE_3D_OPACITY = 0.15
FOREIGN_DISPLAY_CLASSES = ("vtkMRMLVolumeRenderingDisplayNode", "vtkMRMLModelDisplayNode",
                           "vtkMRMLSegmentationDisplayNode")


def _sliceItem(key):
    tag, orientation, label, color, _ = SLICE_VIEWS[key]
    return (f'<item splitSize="500"><view class="vtkMRMLSliceNode" singletontag="{tag}">'
            f'<property name="orientation" action="default">{orientation}</property>'
            f'<property name="viewlabel" action="default">{label}</property>'
            f'<property name="viewcolor" action="default">{color}</property></view></item>')


LAYOUT_XML = (
    '<layout type="horizontal" split="true">'
    ' <item splitSize="680"><layout type="vertical" split="true">'
    '  <item splitSize="500"><layout type="horizontal" split="true">'
    + _sliceItem("axialFused") + _sliceItem("axialReference") +
    '  </layout></item>'
    '  <item splitSize="500"><layout type="horizontal" split="true">'
    + _sliceItem("coronalFused") + _sliceItem("coronalReference") +
    '  </layout></item>'
    ' </layout></item>'
    f' <item splitSize="320"><view class="vtkMRMLViewNode" singletontag="{THREED_VIEW_TAG}">'
    '<property name="viewlabel" action="default">S</property></view></item>'
    '</layout>')


MAIN_VIEWPORT_MARKER = "MAIN_VIEWPORT_ATTRIBUTES"   # replaced by the main-window viewport attributes of Slicer
DUAL_LAYOUT_XML = (
    '<viewports>'
    ' <layout type="vertical"' + MAIN_VIEWPORT_MARKER + ' split="true">'
    '  <item splitSize="500"><layout type="horizontal" split="true">'
    + _sliceItem("axialFused") + _sliceItem("axialReference") +
    '  </layout></item>'
    '  <item splitSize="500"><layout type="horizontal" split="true">'
    + _sliceItem("coronalFused") + _sliceItem("coronalReference") +
    '  </layout></item>'
    ' </layout>'
    f' <layout type="vertical" name="{DUAL_WINDOW_NAME}" dockable="true" dockPosition="floating">'
    f'  <item><view class="vtkMRMLViewNode" singletontag="{THREED_VIEW_TAG}">'
    '<property name="viewlabel" action="default">S</property></view></item>'
    ' </layout>'
    '</viewports>')


# Windowing presets (same as the dosimetry modules). Anatomy: (button, kind, a, b, tooltip); kind "hu": window width
# a / level b in HU, "range": fraction a..b of the intensity range, "percentile": percentiles a..b of the tissue.
ANATOMY_WINDOW_PRESETS = [
    ("CT lung", "hu", 1500.0, -600.0, "CT lung window: W 1500 / L -600 HU"),
    ("CT soft tissue", "hu", 400.0, 40.0, "CT soft tissue (abdomen) window: W 400 / L 40 HU"),
    ("CT liver", "hu", 150.0, 60.0, "CT liver window: W 150 / L 60 HU"),
    ("MRI 0-100%", "range", 0.0, 1.0, "MRI: full intensity range (minimum to maximum)"),
    ("MRI contrast", "percentile", 2.0, 98.0, "MRI with more contrast: 2nd to 98th percentile of the tissue voxels"),
]
FUNCTIONAL_WINDOW_PERCENTS = [10, 25, 50, 75, 100]   # SPECT/PET window 0 .. % of the maximum
MAX_SAMPLE_VOXELS = 2000000


def _setWindowMinMax(volumeNode, minimum, maximum):
    if maximum <= minimum:
        maximum = minimum + 1.0
    if volumeNode.GetDisplayNode() is None:
        volumeNode.CreateDefaultDisplayNodes()
    displayNode = volumeNode.GetDisplayNode()
    displayNode.SetAutoWindowLevel(False)
    displayNode.SetWindowLevelMinMax(float(minimum), float(maximum))


def applyAnatomyWindow(volumeNode, preset):
    """Apply an ANATOMY_WINDOW_PRESETS entry to the CT/MRI."""
    import numpy as np
    _, kind, a, b = preset[:4]
    if kind == "hu":
        _setWindowMinMax(volumeNode, b - a / 2.0, b + a / 2.0)
        return
    values = slicer.util.arrayFromVolume(volumeNode).ravel()
    values = values[::max(1, values.size // MAX_SAMPLE_VOXELS)].astype(np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("The image has no finite voxel values.")
    low, high = float(values.min()), float(values.max())
    if kind == "range":
        _setWindowMinMax(volumeNode, low + a * (high - low), low + b * (high - low))
        return
    tissue = values[values > low]   # without the background (air, outside the field of view)
    minimum, maximum = np.percentile(tissue if tissue.size else values, [a, b])
    _setWindowMinMax(volumeNode, minimum, maximum)


def applyFunctionalWindow(volumeNode, percent):
    """SPECT/PET window from 0 to percent % of the maximum voxel value."""
    import numpy as np
    maximum = float(np.nanmax(slicer.util.arrayFromVolume(volumeNode)))
    if not np.isfinite(maximum) or maximum <= 0:
        raise ValueError("The SPECT/PET image has no positive voxel values.")
    _setWindowMinMax(volumeNode, 0.0, maximum * percent / 100.0)


def setFusionOpacity(opacity):
    """Opacity of the SPECT/PET in the fused views of the layout."""
    layoutManager = slicer.app.layoutManager()
    for tag, _, _, _, fused in SLICE_VIEWS.values():
        sliceWidget = layoutManager.sliceWidget(tag) if (layoutManager and fused) else None
        if sliceWidget is not None:
            sliceWidget.mrmlSliceCompositeNode().SetForegroundOpacity(opacity)


def dualLayoutXml():
    from . import dosimetry as D
    return DUAL_LAYOUT_XML.replace(MAIN_VIEWPORT_MARKER, D.mainViewportAttributes())


def registerLayout(*args):
    """Register the single and dual monitor segmentation layouts (at startup, before every scene import and before
    they are shown: a scene saved in one of them stores only the layout ID)."""
    try:
        layoutManager = slicer.app.layoutManager()
    except Exception:
        layoutManager = None
    if layoutManager is None:
        return
    layoutNode = layoutManager.layoutLogic().GetLayoutNode()
    if layoutNode is None:
        return
    from . import dosimetry as D
    layouts = [(SEGMENTATION_LAYOUT_ID, LAYOUT_XML)]
    if D.dualMonitorLayoutSupported():
        layouts.append((SEGMENTATION_DUAL_LAYOUT_ID, dualLayoutXml()))
    for layoutID, layoutXML in layouts:
        if layoutNode.IsLayoutDescription(layoutID):
            if layoutNode.GetLayoutDescription(layoutID) != layoutXML:
                layoutNode.SetLayoutDescription(layoutID, layoutXML)
        else:
            layoutNode.AddLayoutDescription(layoutID, layoutXML)


def installLayoutRegistration():
    """Register the layouts now and before every scene import (replaces the observer of a previous call)."""
    registerLayout()
    scene = getattr(slicer, "mrmlScene", None)
    if scene is None:
        return
    key = "_taranisSegmentationLayoutObserver"
    previous = getattr(slicer, key, None)
    if previous is not None:
        try:
            previous[0].RemoveObserver(previous[1])
        except Exception:
            pass
    setattr(slicer, key, (scene, scene.AddObserver(scene.StartImportEvent, registerLayout)))


def layoutMode():
    """'single' or 'dual' monitor: the choice of the Single / Dual monitor buttons (shared with the dosimetry
    modules), else automatic from the number of screens."""
    from . import dosimetry as D
    mode = getattr(slicer, D.LAYOUT_MODE_ATTRIBUTE, None)
    if mode not in ("single", "dual"):
        mode = "dual" if D.screenCount() >= 2 else "single"
    if mode == "dual" and not D.dualMonitorLayoutSupported():
        mode = "single"
    return mode


def setLayoutMode(mode):
    from . import dosimetry as D
    setattr(slicer, D.LAYOUT_MODE_ATTRIBUTE, mode if mode in ("single", "dual") else None)


def threeDWindow():
    """The separate window holding the 3D view in the dual monitor layout, or None."""
    layoutManager = slicer.app.layoutManager()
    viewNode = slicer.mrmlScene.GetSingletonNode(THREED_VIEW_TAG, "vtkMRMLViewNode")
    if layoutManager is None or viewNode is None:
        return None
    for index in range(layoutManager.threeDViewCount):
        widget = layoutManager.threeDWidget(index)
        if widget is not None and widget.mrmlViewNode() is not None and widget.mrmlViewNode().GetID() == viewNode.GetID():
            window = widget.window()
            if window is not None and not window.inherits("qSlicerMainWindow"):
                return window
    return None


def placeThreeDWindow():
    """Dual monitor layout: the 3D window with normal window buttons, maximized on the second screen."""
    from . import dosimetry as D
    window = threeDWindow()
    if window is None:
        return
    D.placeWindowOnSecondaryScreen(window)
    try:
        window.setWindowTitle("Taranis - segments 3D")
    except Exception:
        pass
    if D.secondaryScreen() is None and not (window.inherits("QDockWidget") and not getattr(window, "floating", True)):
        # one screen (dual layout chosen with the button): a usable window on the right, not a tiny one
        try:
            screen = slicer.util.mainWindow().windowHandle().screen().availableGeometry
            width, height = int(screen.width() * 0.4), int(screen.height() * 0.6)
            window.setGeometry(screen.x() + screen.width() - width - 40, screen.y() + 80, width, height)
            window.show()
            window.raise_()
        except Exception as e:
            logging.debug(f"Taranis: 3D window size: {e}")


def _viewIDs(displayNode):
    return [displayNode.GetNthViewNodeID(i) for i in range(displayNode.GetNumberOfViewNodeIDs())]


def excludeForeignRenderings(viewNodeIDs, keepNodes=()):
    """Keep volume renderings, models and segmentations of other modules out of the given views. A display node
    without view IDs is shown in every view, so it gets the explicit list of all other views (it stays visible
    where it was). keepNodes (and nodes hidden from editors) are not changed."""
    targets = {viewID for viewID in viewNodeIDs if viewID}
    if not targets:
        return
    keepIDs = {node.GetID() for node in keepNodes if node is not None}
    others = [node.GetID() for className in ("vtkMRMLViewNode", "vtkMRMLSliceNode")
              for node in slicer.util.getNodesByClass(className) if node.GetID() not in targets]
    for className in FOREIGN_DISPLAY_CLASSES:
        for displayNode in slicer.util.getNodesByClass(className):
            displayable = displayNode.GetDisplayableNode()
            if displayable is None or displayable.GetHideFromEditors() or displayable.GetID() in keepIDs:
                continue
            current = _viewIDs(displayNode)
            if current and not targets.intersection(current):
                continue
            remaining = [viewID for viewID in (current or others) if viewID not in targets]
            if remaining:
                displayNode.SetViewNodeIDs(remaining)
            else:
                displayNode.SetVisibility(False)


def _colorNode(name):
    for node in slicer.util.getNodesByClass("vtkMRMLColorNode"):
        if node.GetName().lower() == name.lower():
            return node
    return None


def viewNodes():
    """{key: slice or 3D view node} of the layout (None if it was not created yet)."""
    nodes = {key: slicer.mrmlScene.GetSingletonNode(spec[0], "vtkMRMLSliceNode") for key, spec in SLICE_VIEWS.items()}
    nodes["3d"] = slicer.mrmlScene.GetSingletonNode(THREED_VIEW_TAG, "vtkMRMLViewNode")
    return nodes


def segmentation3DDisplayNode(segmentationNode, create=True):
    """Extra display node of the segmentation for the 3D view of the layout (wireframe, per-role opacity), so the
    segmentation's own display settings (used by the dosimetry modules and the LSF calculator) stay untouched."""
    for index in range(segmentationNode.GetNumberOfDisplayNodes()):
        displayNode = segmentationNode.GetNthDisplayNode(index)
        if displayNode is not None and displayNode.GetAttribute(SEGMENTATION_3D_ATTRIBUTE):
            return displayNode
    if not create:
        return None
    displayNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationDisplayNode", "Taranis segments 3D")
    displayNode.SetAttribute(SEGMENTATION_3D_ATTRIBUTE, "1")
    segmentationNode.AddAndObserveDisplayNodeID(displayNode.GetID())
    return displayNode


def styleSegmentation3D(segmentationNode, viewNodeID, roles, candidates):
    """Wireframe, flat (ambient only), per-role opacity, only in the given 3D view. roles: {segmentID: role};
    candidates: set of candidate segment IDs. Normal tissue is not drawn (it overlaps the liver and tumours)."""
    displayNode = segmentation3DDisplayNode(segmentationNode)
    wasModifying = displayNode.StartModify()
    displayNode.SetViewNodeIDs([viewNodeID])
    displayNode.SetVisibility(True)
    displayNode.SetVisibility2D(False)
    displayNode.SetVisibility3D(True)
    displayNode.SetRepresentation(slicer.vtkMRMLDisplayNode.WireframeRepresentation)
    displayNode.SetAmbient(1.0)
    displayNode.SetDiffuse(0.0)
    displayNode.SetSpecular(0.0)
    displayNode.SetBackfaceCulling(False)
    displayNode.SetOpacity3D(1.0)
    for segmentID, role in roles.items():
        visible = role != W.SEGMENT_NORMAL or segmentID in candidates
        displayNode.SetSegmentVisibility(segmentID, True)
        displayNode.SetSegmentVisibility3D(segmentID, visible)
        opacity = CANDIDATE_3D_OPACITY if segmentID in candidates else SEGMENT_3D_OPACITY.get(role, 0.05)
        displayNode.SetSegmentOpacity3D(segmentID, opacity)
    displayNode.EndModify(wasModifying)
    segmentationNode.CreateClosedSurfaceRepresentation()


def _mainDisplayNode(segmentationNode):
    segmentationNode.CreateDefaultDisplayNodes()
    for index in range(segmentationNode.GetNumberOfDisplayNodes()):
        displayNode = segmentationNode.GetNthDisplayNode(index)
        if displayNode is not None and not displayNode.GetAttribute(SEGMENTATION_3D_ATTRIBUTE):
            return displayNode
    return None


def showLayout(anatomy, functional, segmentationNode, fit=True, fusionOpacity=FUSION_OPACITY, placeWindow=False):
    """Switch to the segmentation layout (single or dual monitor, see layoutMode) and fill it. In the dual monitor
    layout the 3D window is moved to the second screen when the layout is switched on (or placeWindow is set).
    Returns the view nodes."""
    layoutManager = slicer.app.layoutManager()
    if layoutManager is None:
        return None
    registerLayout()
    layoutID = SEGMENTATION_DUAL_LAYOUT_ID if layoutMode() == "dual" else SEGMENTATION_LAYOUT_ID
    switched = layoutManager.layout != layoutID
    if switched:
        layoutManager.setLayout(layoutID)
    slicer.app.processEvents()
    if layoutID == SEGMENTATION_DUAL_LAYOUT_ID and (switched or placeWindow):
        try:
            placeThreeDWindow()
            slicer.app.processEvents()
        except Exception as e:
            logging.warning(f"Taranis: could not move the 3D window to the second screen: {e}")
    nodes = viewNodes()

    if functional is not None:
        if functional.GetDisplayNode() is None:
            functional.CreateDefaultDisplayNodes()
        displayNode = functional.GetDisplayNode()
        colorNode = _colorNode(FUNCTIONAL_COLORMAP)
        if colorNode is not None and displayNode.GetColorNodeID() in (None, "", "vtkMRMLColorTableNodeGrey"):
            displayNode.SetAndObserveColorNodeID(colorNode.GetID())   # grey on grey would hide the uptake

    for key, (tag, orientation, _, _, fused) in SLICE_VIEWS.items():
        sliceWidget = layoutManager.sliceWidget(tag)
        if sliceWidget is None:
            continue
        sliceWidget.mrmlSliceNode().SetOrientation(orientation)
        compositeNode = sliceWidget.mrmlSliceCompositeNode()
        wasModifying = compositeNode.StartModify()
        compositeNode.SetBackgroundVolumeID(anatomy.GetID() if anatomy is not None else
                                            (functional.GetID() if functional is not None else None))
        foreground = functional if (fused and anatomy is not None) else None
        compositeNode.SetForegroundVolumeID(foreground.GetID() if foreground is not None else None)
        compositeNode.SetForegroundOpacity(fusionOpacity)
        compositeNode.SetLabelVolumeID(None)
        compositeNode.EndModify(wasModifying)
        if fit:
            sliceWidget.sliceController().fitSliceToBackground()

    viewNode = nodes["3d"]
    if viewNode is not None:
        wasModifying = viewNode.StartModify()
        viewNode.SetBackgroundColor(0.0, 0.0, 0.0)
        viewNode.SetBackgroundColor2(0.0, 0.0, 0.0)
        viewNode.SetRenderMode(slicer.vtkMRMLViewNode.Orthographic)
        viewNode.SetBoxVisible(False)
        viewNode.SetAxisLabelsVisible(False)
        viewNode.EndModify(wasModifying)

    if segmentationNode is not None:
        showSegmentation(segmentationNode, nodes)
    ownViews = [node.GetID() for node in nodes.values() if node is not None]
    excludeForeignRenderings(ownViews, [segmentationNode, anatomy, functional])
    if fit and viewNode is not None:
        resetThreeDView(viewNode)
    return nodes


def showSegmentation(segmentationNode, nodes=None):
    """Segments in the reference slice views (outline + fill, the segmentation's own settings) and as wireframe in
    the 3D view. The previous view list of the segmentation is kept and restored by leaveLayout."""
    nodes = nodes or viewNodes()
    displayNode = _mainDisplayNode(segmentationNode)
    if displayNode is None:
        return
    if not displayNode.GetAttribute(SAVED_VIEWS_ATTRIBUTE):
        displayNode.SetAttribute(SAVED_VIEWS_ATTRIBUTE, json.dumps(_viewIDs(displayNode)))
    if not displayNode.GetAttribute(SAVED_FILL_ATTRIBUTE):
        displayNode.SetAttribute(SAVED_FILL_ATTRIBUTE, repr(displayNode.GetOpacity2DFill()))
    displayNode.SetOpacity2DFill(FILL_OPACITY_2D)
    referenceViews = [nodes[key].GetID() for key in ("axialReference", "coronalReference") if nodes.get(key)]
    if referenceViews:
        displayNode.SetViewNodeIDs(referenceViews)
    displayNode.SetVisibility(True)
    displayNode.SetVisibility2D(True)


def showBackground(volumeNode):
    """Show volumeNode as the background of the four slice views of the layout (e.g. the input image of an AI model
    while its ROI is placed); the fused views keep their SPECT/PET overlay."""
    layoutManager = slicer.app.layoutManager()
    if layoutManager is None or volumeNode is None or layoutManager.layout not in SEGMENTATION_LAYOUT_IDS:
        return
    for tag, _, _, _, _ in SLICE_VIEWS.values():
        sliceWidget = layoutManager.sliceWidget(tag)
        if sliceWidget is not None and sliceWidget.mrmlSliceCompositeNode().GetBackgroundVolumeID() != volumeNode.GetID():
            sliceWidget.mrmlSliceCompositeNode().SetBackgroundVolumeID(volumeNode.GetID())


def resetThreeDView(viewNode, rotate=True):
    """Centre the 3D view on what it shows; rotate: also look from anterior."""
    layoutManager = slicer.app.layoutManager()
    if layoutManager is None or viewNode is None or viewNode.GetScene() is None:
        return
    for index in range(layoutManager.threeDViewCount):
        widget = layoutManager.threeDWidget(index)
        if widget is not None and widget.mrmlViewNode() is not None and widget.mrmlViewNode().GetID() == viewNode.GetID():
            view = widget.threeDView()
            if rotate:
                view.rotateToViewAxis(3)   # from anterior
            view.resetFocalPoint()
            view.resetCamera()
            return


def leaveLayout(segmentationNode):
    """Restore the segmentation's view list and hide the extra 3D display node (the layout stays as it is)."""
    if segmentationNode is None or segmentationNode.GetScene() is None:
        return
    try:
        displayNode = _mainDisplayNode(segmentationNode)
        saved = displayNode.GetAttribute(SAVED_VIEWS_ATTRIBUTE) if displayNode is not None else None
        if saved is not None:
            displayNode.SetViewNodeIDs(json.loads(saved or "[]"))
            displayNode.RemoveAttribute(SAVED_VIEWS_ATTRIBUTE)
        savedFill = displayNode.GetAttribute(SAVED_FILL_ATTRIBUTE) if displayNode is not None else None
        if savedFill:
            displayNode.SetOpacity2DFill(float(savedFill))
            displayNode.RemoveAttribute(SAVED_FILL_ATTRIBUTE)
        extra = segmentation3DDisplayNode(segmentationNode, create=False)
        if extra is not None:
            extra.SetVisibility(False)
    except Exception as e:
        logging.warning(f"Taranis: could not restore the segmentation display: {e}")
