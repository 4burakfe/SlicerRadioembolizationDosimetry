"""Workflow controller: watches the scene, builds a CaseSnapshot of the active case, evaluates the step statuses
and notifies the toolbar and the hub. One instance per Slicer session."""

import hashlib
import json
import logging
import zlib

import numpy as np
import qt
import vtk
import slicer
from slicer.util import VTKObservationMixin

from . import roles as R
from . import workflow as W
from .case import (TaranisCase, volumeInfo, P_CURRENT_STEP, P_REGISTRATION_SKIPPED, P_REGISTRATION_CHECKED,
                   P_LSF_SOURCE, P_LSF_SKIPPED, P_DOSIMETRY_RESULT_KEY, P_DOSIMETRY_FINGERPRINT,
                   P_SEGMENT_GEOMETRY, P_LSF_DETAILS, P_LSF_ISSUES)

# Attributes written by EasyReg on its registration transforms
EASYREG_TRANSFORM = "EasyReg.Transform"
EASYREG_METHOD = "EasyReg.Method"
EASYREG_REGISTERED_NODES = "EasyReg.RegisteredNodeIDs"
EASYREG_HARDENED_NODES = "EasyReg.HardenedNodeIDs"
EASYREG_CENTRE_METHOD = "Centred on reference"   # EasyReg's 'Centre moving image on reference' (not a registration)
# Manual alignment transforms made from the Taranis hub
ALIGNMENT_ATTRIBUTE = "Taranis.Alignment"
ALIGNMENT_NODES_ATTRIBUTE = "Taranis.AlignedNodeIDs"

# Segment roles: tag on the segment; categories shared with the dosimetry modules (JSON on the segmentation)
SEGMENT_ROLE_TAG = "Taranis.SegmentRole"
CATEGORY_ATTRIBUTE = "Taranis.SegmentCategories"
# "lungs": not calculated in patient-relative dosimetry (only used for the LSF); calculated in absolute dosimetry
# with a warning (one tissue density for all voxels)
CATEGORY_TO_ROLE = {"tumor": W.SEGMENT_TUMOR, "viable": W.SEGMENT_VIABLE, "normal": W.SEGMENT_NORMAL,
                    "lungs": W.SEGMENT_LUNGS, "other": W.SEGMENT_OTHER}
ROLE_TO_CATEGORY = {role: category for category, role in CATEGORY_TO_ROLE.items()}
LSF_CANDIDATE_TAG = "LSFcalc.Candidate"
TARANIS_CANDIDATE_TAG = "Taranis.Candidate"   # AI result of the hub (value: role it gets when accepted)

# Parameter-node keys of the dosimetry modules
DOSIMETRY_RESULTS = "Results"
DOSIMETRY_DOSE_REFERENCE = "ResultsDoseVolume"
DOSIMETRY_REPORT_CHECKSUM = "LastReportDoseChecksum"

UPDATE_DELAY_MS = 300


# -- Segment helpers --------------------------------------------------------------------------------------------

def _mutableString():
    factory = getattr(vtk, "reference", None) or getattr(vtk, "mutable")
    return factory("")


def segmentTag(segment, tag):
    value = _mutableString()
    try:
        if segment.GetTag(tag, value):
            return str(value)
    except TypeError:
        pass
    return ""


def segmentCategories(segmentationNode):
    try:
        return json.loads(segmentationNode.GetAttribute(CATEGORY_ATTRIBUTE) or "{}")
    except ValueError:
        return {}


def segmentRole(segmentationNode, segmentID, categories=None):
    segment = segmentationNode.GetSegmentation().GetSegment(segmentID)
    if segment is None or isCandidate(segment):
        return ""
    tagged = segmentTag(segment, SEGMENT_ROLE_TAG)
    if tagged in W.SEGMENT_ROLES:
        return tagged
    if categories is None:
        categories = segmentCategories(segmentationNode)
    if categories.get(segmentID) in CATEGORY_TO_ROLE:
        return CATEGORY_TO_ROLE[categories[segmentID]]
    return W.guessSegmentRole(segment.GetName())


def isCandidate(segment):
    return segment.HasTag(TARANIS_CANDIDATE_TAG) or segment.HasTag(LSF_CANDIDATE_TAG)


def setSegmentRole(segmentationNode, segmentID, role, recolor=True):
    """Store the role on the segment and keep the dosimetry modules' categories in sync."""
    segment = segmentationNode.GetSegmentation().GetSegment(segmentID)
    if segment is None:
        return
    if role:
        segment.SetTag(SEGMENT_ROLE_TAG, role)
    else:
        segment.RemoveTag(SEGMENT_ROLE_TAG)
    categories = segmentCategories(segmentationNode)
    if role in ROLE_TO_CATEGORY:
        categories[segmentID] = ROLE_TO_CATEGORY[role]
    else:
        categories.pop(segmentID, None)
    segmentationNode.SetAttribute(CATEGORY_ATTRIBUTE, json.dumps(dict(sorted(categories.items()))))
    if recolor and role in W.SEGMENT_ROLES:
        segment.SetColor(*W.SEGMENT_ROLES[role][2])


def persistSegmentRoles(segmentationNode):
    """Store the guessed role of every segment that has no stored role yet (tag and dosimetry categories), so
    the dosimetry modules see the same roles as the hub. Colours are not changed."""
    if segmentationNode is None:
        return
    categories = segmentCategories(segmentationNode)
    segmentation = segmentationNode.GetSegmentation()
    wasModifying = segmentationNode.StartModify()
    try:
        for segmentID in segmentIDs(segmentationNode):
            segment = segmentation.GetSegment(segmentID)
            if isCandidate(segment):
                continue
            tagged = segmentTag(segment, SEGMENT_ROLE_TAG)
            role = segmentRole(segmentationNode, segmentID, categories)
            # untagged: store the guessed role; tagged: keep the dosimetry category in sync (e.g. lungs -> ignored)
            if role and (not tagged or categories.get(segmentID) != ROLE_TO_CATEGORY.get(role, categories.get(segmentID))):
                setSegmentRole(segmentationNode, segmentID, role, recolor=False)
                categories = segmentCategories(segmentationNode)
    finally:
        segmentationNode.EndModify(wasModifying)


def segmentIDs(segmentationNode):
    if segmentationNode is None:
        return []
    ids = vtk.vtkStringArray()
    segmentationNode.GetSegmentation().GetSegmentIDs(ids)
    return [ids.GetValue(i) for i in range(ids.GetNumberOfValues())]


_labelStatsCache = {}


def _labelmapStats(segment):
    """(voxel count, world centroid rounded to 0.1 mm) of a segment's binary labelmap; cached per layer MTime.
    Independent of the labelmap extent, so it survives saving and loading the scene."""
    from vtk.util.numpy_support import vtk_to_numpy
    name = slicer.vtkSegmentationConverter.GetBinaryLabelmapRepresentationName()
    image = segment.GetRepresentation(name)
    if image is None:
        return None
    label = segment.GetLabelValue() if hasattr(segment, "GetLabelValue") else 1
    key = (id(image), image.GetMTime(), label)
    if key in _labelStatsCache:
        return _labelStatsCache[key]
    extent = image.GetExtent()
    if extent[0] > extent[1] or extent[2] > extent[3] or extent[4] > extent[5] or image.GetPointData().GetScalars() is None:
        result = (0, None)
    else:
        shape = (extent[5] - extent[4] + 1, extent[3] - extent[2] + 1, extent[1] - extent[0] + 1)
        array = vtk_to_numpy(image.GetPointData().GetScalars()).reshape(shape)
        inside = array == label   # one byte per voxel (np.nonzero would need 24 bytes per segment voxel)
        count = int(np.count_nonzero(inside))
        if count == 0:
            result = (0, None)
        else:
            # centroid from the projections on the three axes
            perSlice = inside.sum(axis=(1, 2), dtype=np.int64)
            perRow = inside.sum(axis=(0, 2), dtype=np.int64)
            perColumn = inside.sum(axis=(0, 1), dtype=np.int64)
            k = float(np.dot(perSlice, np.arange(shape[0]))) / count
            j = float(np.dot(perRow, np.arange(shape[1]))) / count
            i = float(np.dot(perColumn, np.arange(shape[2]))) / count
            del inside
            matrix = vtk.vtkMatrix4x4()
            image.GetImageToWorldMatrix(matrix)
            world = matrix.MultiplyPoint([i + extent[0], j + extent[2], k + extent[4], 1.0])
            result = (count, tuple(round(v, 1) for v in world[:3]))
    if result[0] and len(result) == 2:
        voxelMM3 = abs(np.linalg.det(np.array([[matrix.GetElement(r, c) for c in range(3)] for r in range(3)])))
        result = result + (result[0] * voxelMM3 / 1000.0,)
    if len(_labelStatsCache) > 512:
        _labelStatsCache.clear()
    _labelStatsCache[key] = result
    return result


def segmentInfos(segmentationNode):
    if segmentationNode is None:
        return []
    categories = segmentCategories(segmentationNode)
    infos = []
    segmentation = segmentationNode.GetSegmentation()
    for segmentID in segmentIDs(segmentationNode):
        segment = segmentation.GetSegment(segmentID)
        stats = _labelmapStats(segment)
        infos.append(W.SegmentInfo(segmentID, segment.GetName(), segmentRole(segmentationNode, segmentID, categories),
                                   empty=(stats is not None and stats[0] == 0), candidate=isCandidate(segment),
                                   voxels=stats[0] if stats is not None else None,
                                   volumeML=stats[2] if stats is not None and len(stats) > 2 else None))
    return infos


# -- Registration helpers -------------------------------------------------------------------------------------

def isAlignmentTransform(transformNode):
    """An EasyReg transform that holds a registration (an EasyReg transform without a method is only prepared,
    e.g. while landmarks are being placed), or a Taranis manual alignment transform."""
    if transformNode is None:
        return False
    if transformNode.GetAttribute(EASYREG_TRANSFORM) == "1":
        method = transformNode.GetAttribute(EASYREG_METHOD)
        return bool(method) and method != EASYREG_CENTRE_METHOD   # only centred on the reference: not registered
    return transformNode.GetAttribute(ALIGNMENT_ATTRIBUTE) == "1"


def alignedNodeIDs(transformNode):
    ids = []
    for attribute in (EASYREG_REGISTERED_NODES, EASYREG_HARDENED_NODES, ALIGNMENT_NODES_ATTRIBUTE):
        ids += [i for i in (transformNode.GetAttribute(attribute) or "").split(",") if i]
    return set(ids)


def alignmentTransformFor(volumeNode):
    """The EasyReg or Taranis alignment transform that moved this volume (live or hardened), or None."""
    parent = volumeNode.GetParentTransformNode()
    while parent is not None:
        if isAlignmentTransform(parent):
            return parent
        parent = parent.GetParentTransformNode()
    for transformNode in slicer.util.getNodesByClass("vtkMRMLTransformNode"):
        if isAlignmentTransform(transformNode) and volumeNode.GetID() in alignedNodeIDs(transformNode):
            return transformNode
    return None


def followsTransform(node, transformNode):
    if node.GetID() in alignedNodeIDs(transformNode):
        return True
    parent = node.GetParentTransformNode()
    while parent is not None:
        if parent is transformNode:
            return True
        parent = parent.GetParentTransformNode()
    return False


# -- Fingerprint of the dosimetry inputs -------------------------------------------------------------------

_volumeHashCache = {}


def _volumeHash(volumeNode):
    imageData = volumeNode.GetImageData()
    if imageData is None or imageData.GetPointData().GetScalars() is None:
        return "empty"
    key = (volumeNode.GetID(), imageData.GetMTime())
    if key not in _volumeHashCache:
        from vtk.util.numpy_support import vtk_to_numpy
        array = np.ascontiguousarray(vtk_to_numpy(imageData.GetPointData().GetScalars()))
        if len(_volumeHashCache) > 64:
            _volumeHashCache.clear()
        _volumeHashCache[key] = f"{zlib.crc32(array.view(np.uint8)):08x}"
    return _volumeHashCache[key]


def _placement(node):
    """Position of a node in the world: linear matrix (rounded) or the chain of non-linear transforms."""
    parent = node.GetParentTransformNode()
    if parent is None:
        return "identity"
    matrix = vtk.vtkMatrix4x4()
    if slicer.vtkMRMLTransformNode.GetMatrixTransformBetweenNodes(parent, None, matrix):
        return [round(matrix.GetElement(r, c), 4) for r in range(3) for c in range(4)]
    chain = []
    while parent is not None:
        chain.append(parent.GetID())
        parent = parent.GetParentTransformNode()
    return chain


def inputsFingerprint(case):
    parts = {}
    for role in R.VOLUME_ROLES:
        node = case.roleNode(role)
        if node is None:
            parts[role] = None
            continue
        spacing = [round(v, 4) for v in node.GetSpacing()]
        origin = [round(v, 3) for v in node.GetOrigin()]
        parts[role] = [node.GetID(), spacing, origin, _volumeHash(node), _placement(node)]
    segmentationNode = case.roleNode(R.ROLE_SEGMENTATION)
    if segmentationNode is not None:
        segmentation = segmentationNode.GetSegmentation()
        segments = []
        for segmentID in segmentIDs(segmentationNode):
            segment = segmentation.GetSegment(segmentID)
            segments.append([segmentID, segment.GetName(), (_labelmapStats(segment) or ())[:2] or None])
        parts[R.ROLE_SEGMENTATION] = [segmentationNode.GetID(), _placement(segmentationNode), segments]
    text = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


# -- Dosimetry module results --------------------------------------------------------------------------------

_resultKeyCache = {}


def dosimetryParameterNode(moduleName):
    return slicer.mrmlScene.GetSingletonNode(moduleName, "vtkMRMLScriptedModuleNode")


def _dosimetryResult(moduleName):
    """(dose checksum, [(severity, text)] dose checks) of the last calculation stored by a dosimetry module
    (("", []) if none)."""
    node = dosimetryParameterNode(moduleName)
    if node is None or node.GetNodeReference(DOSIMETRY_DOSE_REFERENCE) is None:
        return "", []
    cacheKey = (node.GetID(), node.GetMTime())
    if cacheKey not in _resultKeyCache:
        key, checks = "", []
        text = node.GetParameter(DOSIMETRY_RESULTS)
        if text:
            try:
                results = json.loads(text)
                key = str(results.get("doseChecksum") or "")
                checks = [(str(severity), str(message))
                          for severity, message in (results.get("lastResult") or {}).get("checks", [])]
            except (ValueError, AttributeError, TypeError):
                key, checks = "", []
        if len(_resultKeyCache) > 16:
            _resultKeyCache.clear()
        _resultKeyCache[cacheKey] = (key, checks)
    return _resultKeyCache[cacheKey]


def dosimetryResultKey(moduleName):
    """Dose checksum of the last calculation stored by a dosimetry module ("" if none)."""
    return _dosimetryResult(moduleName)[0]


def dosimetryResultChecks(moduleName):
    """Dose checks [(severity, text)] stored with the last calculation of a dosimetry module."""
    return list(_dosimetryResult(moduleName)[1])


def dosimetryReportKey(moduleName):
    node = dosimetryParameterNode(moduleName)
    return node.GetParameter(DOSIMETRY_REPORT_CHECKSUM) if node is not None else ""


# -- Snapshot ---------------------------------------------------------------------------------------------------

def buildSnapshot(case, updateBookkeeping=True):
    s = W.CaseSnapshot(caseName=case.name, caseID=case.caseID)
    nodes = case.roleNodes()
    byID = {}
    for role, node in nodes.items():
        if node is None:
            continue
        s.roles[role] = volumeInfo(node)
        s.roleTypes[role] = case.roleType(role)
        byID.setdefault(node.GetID(), (node.GetName(), []))[1].append(R.ROLE_INFO[role][0])
    s.duplicateNodes = [(name, labels) for name, labels in byID.values() if len(labels) > 1]
    s.mode = case.mode

    s.registrationSkipped = case.flag(P_REGISTRATION_SKIPPED)
    s.primaryRole, s.jobs = W.planRegistration(s.roles)
    for job in s.jobs:
        moving = nodes.get(job.movingRole)
        job.checked = moving is not None and case.param(P_REGISTRATION_CHECKED + job.key) == moving.GetID()
        transformNode = alignmentTransformFor(moving) if moving is not None else None
        if transformNode is not None:
            job.registered = True
            job.method = transformNode.GetAttribute(EASYREG_METHOD) or "Manual"
            follower = nodes.get(job.followerRole) if job.followerRole else None
            job.followerFollows = follower is None or followsTransform(follower, transformNode)

    segmentationNode = case.roleNode(R.ROLE_SEGMENTATION)
    s.segmentationPresent = segmentationNode is not None
    s.segments = segmentInfos(segmentationNode)
    try:
        stored = json.loads(case.param(P_SEGMENT_GEOMETRY) or "{}")
    except ValueError:
        stored = {}
    if stored:
        if s.segmentationPresent and stored.get("key") == W.segmentsKey(s.segments):
            s.geometryIssues = [tuple(issue) for issue in stored.get("issues", [])]
        else:
            s.geometryStale = True

    s.lsfValue = case.lsfValue()
    s.lsfSource = case.param(P_LSF_SOURCE)
    s.lsfSkipped = case.flag(P_LSF_SKIPPED)
    s.lsfLungMassG = case.lungMassG()
    s.plannedActivityGBq = case.plannedActivityGBq()
    if s.lsfValue is not None and case.param(P_LSF_DETAILS):
        from .lsf import inputsKey
        try:
            details = json.loads(case.param(P_LSF_DETAILS))
            s.lsfIssues = [tuple(issue) for issue in json.loads(case.param(P_LSF_ISSUES) or "[]")]
        except ValueError:
            details = {}
        s.lsfFromImage = bool(details.get("key"))
        image = case.roleNode(R.ROLE_DOSIMETRY)
        s.lsfOutdated = s.lsfFromImage and details.get("key") != inputsKey(s.segments, image.GetID() if image else "")

    # Dosimetry results: prefer the module of the chosen mode
    modules = [R.MODE_MODULES[s.mode]] if s.mode else []
    modules += [m for m in R.MODE_MODULES.values() if m not in modules]
    resultModule, resultKey = "", ""
    for moduleName in modules:
        key = dosimetryResultKey(moduleName)
        if key:
            resultModule, resultKey = moduleName, key
            break
    s.dosimetryResultsModule = resultModule
    if resultKey:
        s.dosimetryChecks = dosimetryResultChecks(resultModule)
        combinedKey = f"{resultModule}:{resultKey}"
        fingerprint = inputsFingerprint(case)
        if case.param(P_DOSIMETRY_RESULT_KEY) != combinedKey:
            # a new calculation: remember the inputs it was made from
            if updateBookkeeping:
                case.setParameters({P_DOSIMETRY_RESULT_KEY: combinedKey, P_DOSIMETRY_FINGERPRINT: fingerprint})
            s.dosimetryOutdated = False
        else:
            s.dosimetryOutdated = case.param(P_DOSIMETRY_FINGERPRINT) != fingerprint
        reportKey = dosimetryReportKey(resultModule)
        s.reportSaved = bool(reportKey)
        s.reportOutdated = bool(reportKey) and (reportKey != resultKey or s.dosimetryOutdated)
    return s


# -- Controller --------------------------------------------------------------------------------------------------

class WorkflowController(VTKObservationMixin):
    """Singleton. Listeners are called with the controller after every evaluation."""

    _instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def shutdownInstance(cls):
        if cls._instance is not None:
            cls._instance.shutdown()
            cls._instance = None

    def __init__(self):
        VTKObservationMixin.__init__(self)
        self.case = None
        self.snapshot = None
        self.statuses = {}
        self.listeners = []
        self._observedNodes = {}
        self._sceneClosing = False
        self._timer = qt.QTimer()
        self._timer.setSingleShot(True)
        self._timer.setInterval(UPDATE_DELAY_MS)
        self._timer.connect("timeout()", self.update)
        scene = slicer.mrmlScene
        for event in (scene.NodeAddedEvent, scene.NodeRemovedEvent, scene.EndImportEvent, scene.EndRestoreEvent):
            self.addObserver(scene, event, self._onSceneEvent)
        self.addObserver(scene, scene.StartCloseEvent, self._onSceneStartClose)
        self.addObserver(scene, scene.EndCloseEvent, self._onSceneEndClose)
        self.update()

    def shutdown(self):
        self._timer.stop()
        self.removeObservers()
        self.listeners = []

    # -- Listeners --

    def addListener(self, callback):
        if callback not in self.listeners:
            self.listeners.append(callback)

    def removeListener(self, callback):
        if callback in self.listeners:
            self.listeners.remove(callback)

    # -- State --

    @property
    def isActive(self):
        return self.case is not None and self.case.isValid()

    @property
    def currentStep(self):
        if not self.isActive:
            return None
        step = self.case.param(P_CURRENT_STEP, W.STEP_DATA)
        return step if step in W.STEP_KEYS or step == "home" else W.STEP_DATA

    def setCurrentStep(self, step):
        if self.isActive:
            self.case.setParam(P_CURRENT_STEP, step)

    def status(self, step):
        return self.statuses.get(step)

    # -- Updating --

    def scheduleUpdate(self, *args):
        if not self._sceneClosing:
            self._timer.start()

    def _onSceneEvent(self, caller, event):
        self.scheduleUpdate()

    def _onSceneStartClose(self, caller, event):
        self._sceneClosing = True
        self._timer.stop()

    def _onSceneEndClose(self, caller, event):
        self._sceneClosing = False
        self.update()

    def _onNodeEvent(self, caller, event):
        self.scheduleUpdate()

    def update(self):
        if self._sceneClosing:
            return
        self._timer.stop()
        node = TaranisCase.findNode()
        self.case = TaranisCase(node) if node is not None else None
        if self.case is not None:
            try:
                self.snapshot = buildSnapshot(self.case)
                self.statuses = W.evaluateAll(self.snapshot)
            except Exception as e:
                logging.exception(f"Taranis: could not evaluate the workflow: {e}")
                self.snapshot, self.statuses = None, {}
        else:
            self.snapshot, self.statuses = None, {}
        self._updateObservations()
        for callback in list(self.listeners):
            try:
                callback(self)
            except Exception as e:
                logging.exception(f"Taranis: workflow listener failed: {e}")

    def _nodesToObserve(self):
        """{node: [events]} that can change a step status."""
        wanted = {}
        if self.case is None:
            return wanted
        wanted[self.case.node] = [vtk.vtkCommand.ModifiedEvent]
        volumeEvents = [vtk.vtkCommand.ModifiedEvent, slicer.vtkMRMLVolumeNode.ImageDataModifiedEvent,
                        slicer.vtkMRMLTransformableNode.TransformModifiedEvent]
        for node in self.case.roleNodes().values():
            if node is not None:
                wanted[node] = volumeEvents
        segmentationNode = self.case.roleNode(R.ROLE_SEGMENTATION)
        if segmentationNode is not None:
            events = [vtk.vtkCommand.ModifiedEvent, slicer.vtkMRMLTransformableNode.TransformModifiedEvent]
            for name in ("SegmentAdded", "SegmentRemoved", "SegmentModified", "SourceRepresentationModified",
                         "MasterRepresentationModified"):
                if hasattr(slicer.vtkSegmentation, name):
                    events.append(getattr(slicer.vtkSegmentation, name))
            wanted[segmentationNode] = events
        for moduleName in R.MODE_MODULES.values():
            parameterNode = dosimetryParameterNode(moduleName)
            if parameterNode is not None:
                wanted[parameterNode] = [vtk.vtkCommand.ModifiedEvent]
        return wanted

    def _updateObservations(self):
        wanted = self._nodesToObserve()
        for node in list(self._observedNodes):
            if node not in wanted or self._observedNodes[node] != wanted[node]:
                for event in self._observedNodes.pop(node):
                    self.removeObserver(node, event, self._onNodeEvent)
        for node, events in wanted.items():
            if node not in self._observedNodes:
                for event in events:
                    self.addObserver(node, event, self._onNodeEvent)
                self._observedNodes[node] = events

    # -- Case actions --

    def startCase(self, name, caseID):
        case = TaranisCase.create(name, caseID)
        case.setParam(P_CURRENT_STEP, W.STEP_DATA)
        self.update()
        return case
