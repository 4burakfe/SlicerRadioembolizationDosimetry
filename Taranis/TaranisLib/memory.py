"""Memory: what uses Slicer's RAM, and releasing what can be rebuilt.

- processMemoryMB(): working set / private memory of the Slicer process.
- memoryReport(): process memory, the biggest data of the scene (volumes, segmentations, models, transforms,
  tables), temporary nodes left behind and Python objects.
- releaseMemory(): removes temporary nodes left behind by an interrupted operation, clears the Taranis caches,
  collects Python garbage and returns freed heap memory to the operating system. Nothing the user made is removed.
- collectSoon(): cheap garbage collection after heavy operations (deferred, at most once per event loop cycle).

Only in Slicer (no pure part).
"""

import ctypes
import gc
import logging
import os
import re
import sys

import slicer

# Temporary nodes of Taranis, EasyReg and the LSF calculator (removed at the end of each operation; left behind only
# when an operation was interrupted). Matched on the node name.
TEMPORARY_NAME_PARTS = ("Taranis grid", "Taranis LSF mask", "Taranis AI resampled", "Taranis merged",
                        "Taranis TotalSegmentator", "TotalSegmentator result", "(temporary)", "LSF AI resampled",
                        "LSF AI cropped")
TEMPORARY_ATTRIBUTE = "Taranis.Temporary"
COLOR_TABLE_SUFFIX = "_ColorTable"   # colour tables Slicer creates when segments are exported to a label map
EXPORT_COLOR_TABLE = re.compile(r"_ColorTable(_\d+)?$")   # "..._ColorTable", "..._ColorTable_1" (name taken)


def isExportColorTableName(name):
    return bool(EXPORT_COLOR_TABLE.search(name or ""))


# -- Process memory -----------------------------------------------------------------------------------------------

class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t)]


def processMemoryMB():
    """{"working": MB in RAM, "private": MB committed, "peak": peak working set MB} (values None if unknown)."""
    result = {"working": None, "private": None, "peak": None}
    try:
        if sys.platform.startswith("win"):
            from ctypes import wintypes
            counters = _ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE   # pseudo handle -1: must stay 64-bit
            getInfo = getattr(kernel32, "K32GetProcessMemoryInfo", None)
            if getInfo is None:
                getInfo = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
            getInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ProcessMemoryCounters), wintypes.DWORD]
            getInfo.restype = wintypes.BOOL
            if getInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
                result.update(working=counters.WorkingSetSize / 2 ** 20, private=counters.PrivateUsage / 2 ** 20,
                              peak=counters.PeakWorkingSetSize / 2 ** 20)
        elif os.path.exists("/proc/self/status"):
            with open("/proc/self/status") as f:
                fields = dict(line.split(":", 1) for line in f if ":" in line)
            kb = {key: float(fields[key].split()[0]) for key in ("VmRSS", "VmHWM", "VmData") if key in fields}
            result.update(working=kb.get("VmRSS", 0) / 1024, peak=kb.get("VmHWM", 0) / 1024,
                          private=kb.get("VmData", 0) / 1024)
        else:
            import resource
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            result["peak"] = peak / 2 ** 20 if sys.platform == "darwin" else peak / 1024
    except Exception as e:
        logging.debug(f"Taranis: process memory not available: {e}")
    return result


def processMemoryText():
    memory = processMemoryMB()
    if memory["working"] is None:
        return "not available"
    text = f"{memory['working'] / 1024:.2f} GB in RAM"
    if memory["private"]:
        text += f", {memory['private'] / 1024:.2f} GB committed"
    if memory["peak"]:
        text += f" (peak {memory['peak'] / 1024:.2f} GB)"
    return text


# -- Scene data -----------------------------------------------------------------------------------------------------

def _dataMB(dataObject):
    try:
        return dataObject.GetActualMemorySize() / 1024.0 if dataObject is not None else 0.0
    except Exception:
        return 0.0


def _segmentationMB(segmentationNode):
    """(MB, text): binary labelmap layers (each counted once) and closed surfaces of a segmentation."""
    segmentation = segmentationNode.GetSegmentation()
<<<<<<< Updated upstream
    seen, labelmapMB, surfaceMB = set(), 0.0, 0.0
=======
    seen, labelmapMB, surfaceMB = {}, 0.0, 0.0
>>>>>>> Stashed changes
    labelmapName = slicer.vtkSegmentationConverter.GetBinaryLabelmapRepresentationName()
    surfaceName = slicer.vtkSegmentationConverter.GetClosedSurfaceRepresentationName()
    for index in range(segmentation.GetNumberOfSegments()):
        segment = segmentation.GetNthSegment(index)
        for name in (labelmapName, surfaceName):
            data = segment.GetRepresentation(name)
            if data is None or id(data) in seen:
                continue
<<<<<<< Updated upstream
            seen.add(id(data))
=======
            seen[id(data)] = data   # keep the wrapper alive: a freed wrapper's id can be reused by another layer
>>>>>>> Stashed changes
            if name == labelmapName:
                labelmapMB += _dataMB(data)
            else:
                surfaceMB += _dataMB(data)
    return labelmapMB + surfaceMB, (f"{segmentation.GetNumberOfSegments()} segments, labelmaps {labelmapMB:.0f} MB, "
                                    f"surfaces {surfaceMB:.0f} MB")


def nodeMemoryMB(node):
    """(MB, detail) of the bulk data held by a node."""
    try:
        if node.IsA("vtkMRMLSegmentationNode"):
            return _segmentationMB(node)
        if node.IsA("vtkMRMLVolumeNode"):
            image = node.GetImageData()
            if image is None:
                return 0.0, ""
            dims = "x".join(str(d) for d in image.GetDimensions())
            return _dataMB(image), f"{dims} {image.GetScalarTypeAsString()}"
        if node.IsA("vtkMRMLModelNode"):
            mesh = node.GetMesh()
            return _dataMB(mesh), f"{mesh.GetNumberOfPoints() if mesh else 0} points"
        if node.IsA("vtkMRMLTableNode"):
            return _dataMB(node.GetTable()), ""
        if node.IsA("vtkMRMLTransformNode"):
            total = 0.0
            for getter in ("GetTransformToParent", "GetTransformFromParent"):
                transform = getattr(node, getter)()
                if transform is not None and hasattr(transform, "GetDisplacementGrid"):
                    total += _dataMB(transform.GetDisplacementGrid())
            return total, "displacement field" if total else ""
    except Exception as e:
        logging.debug(f"Taranis: memory of '{node.GetName()}': {e}")
    return 0.0, ""


def isTemporaryNode(node):
    """A temporary node of Taranis / EasyReg / the LSF calculator. Colour tables are never counted here: an export
    colour table is removed only when nothing uses it (orphanExportColorTables)."""
    if node.IsA("vtkMRMLColorNode") or node.IsA("vtkMRMLStorageNode"):
        return False
    name = node.GetName() or ""
    return node.GetAttribute(TEMPORARY_ATTRIBUTE) == "1" or any(part in name for part in TEMPORARY_NAME_PARTS)


def _referencedColorNodeIDs(exclude=()):
    ids = set()
    for displayNode in slicer.util.getNodesByClass("vtkMRMLDisplayNode"):
        if displayNode in exclude:
            continue
        colorID = displayNode.GetColorNodeID() if hasattr(displayNode, "GetColorNodeID") else None
        if colorID:
            ids.add(colorID)
    return ids


def orphanExportColorTables():
    """Colour tables made by label map exports that no display node uses any more."""
    used = _referencedColorNodeIDs()
    return [node for node in slicer.util.getNodesByClass("vtkMRMLColorTableNode")
            if isExportColorTableName(node.GetName()) and not node.GetSingletonTag()
            and node.GetID() not in used]


def finishedCliNodes():
    nodes = []
    for node in slicer.util.getNodesByClass("vtkMRMLCommandLineModuleNode"):
        try:
            if node.GetAttribute(TEMPORARY_ATTRIBUTE) == "1" and not node.IsBusy():
                nodes.append(node)
        except Exception:
            pass
    return nodes


def removeTemporaryLabelmap(labelmapNode):
    """Remove a temporary label map node and the colour table an export created for it (if nothing else uses it)."""
    if labelmapNode is None or labelmapNode.GetScene() is None:
        return
    colorNode = None
    displayNode = labelmapNode.GetDisplayNode()
    if displayNode is not None and hasattr(displayNode, "GetColorNode"):
        colorNode = displayNode.GetColorNode()
    slicer.mrmlScene.RemoveNode(labelmapNode)
    if colorNode is not None and colorNode.GetScene() is not None and not colorNode.GetSingletonTag() \
            and isExportColorTableName(colorNode.GetName()) \
            and colorNode.GetID() not in _referencedColorNodeIDs():
        slicer.mrmlScene.RemoveNode(colorNode)


def runCliAndRemove(module, parameters):
    """slicer.cli.runSync, then remove the CLI node (each run otherwise leaves a node in the scene)."""
    cliNode = slicer.cli.runSync(module, None, parameters)
    try:
        cliNode.SetAttribute(TEMPORARY_ATTRIBUTE, "1")
        if not cliNode.IsBusy():
            slicer.mrmlScene.RemoveNode(cliNode)
    except Exception as e:
        logging.debug(f"Taranis: CLI node not removed: {e}")
    return cliNode


# -- Python objects -------------------------------------------------------------------------------------------------

def _arraysOf(value, depth=0, seen=None):
    """Total bytes of numpy arrays in an object tree (attributes, lists, dicts), a few levels deep."""
    import numpy as np
    seen = set() if seen is None else seen
    if id(value) in seen or depth > 4:
        return 0
    seen.add(id(value))
    if isinstance(value, np.ndarray):
        return int(value.nbytes) if value.base is None else 0
    total = 0
    if isinstance(value, dict):
        items = list(value.values())
    elif isinstance(value, (list, tuple, set)):
        items = list(value)
    elif hasattr(value, "__dict__") and not isinstance(value, type) \
            and not str(type(value).__module__).startswith(("qt", "vtk", "PythonQt", "ctk", "slicer")):
        try:
            items = list(vars(value).values())
        except TypeError:
            return 0
    else:
        return 0
    for item in items[:10000]:
        total += _arraysOf(item, depth + 1, seen)
    return total


def pythonArraysMB():
    """{module name: MB of numpy arrays held by its widget and logic}."""
    result = {}
    for name in ("Taranis", "RadioembolizationDosimetryRelative", "RadioembolizationDosimetryAbsolute", "easy_reg",
                 "LSFcalc"):
        # the widget object registered by Slicer when the module GUI was created (never created here)
        owner = getattr(slicer.modules, name + "Widget", None)
        if owner is None:
            continue
        total = 0
        try:
            total += _arraysOf(owner)
            total += _arraysOf(getattr(owner, "logic", None))
        except Exception:
            pass
        if total:
            result[name] = total / 2 ** 20
    return result


# -- Report ---------------------------------------------------------------------------------------------------------

def memoryReport(top=15):
    """Plain-text report of what uses memory."""
    gc.collect()
    lines = [f"Slicer process: {processMemoryText()}", ""]
    rows, byClass, temporary = [], {}, []
    for node in slicer.util.getNodesByClass("vtkMRMLNode"):
        megabytes, detail = nodeMemoryMB(node)
        className = node.GetClassName()
        count, total = byClass.get(className, (0, 0.0))
        byClass[className] = (count + 1, total + megabytes)
        if megabytes >= 0.5:
            rows.append((megabytes, node.GetName(), className.replace("vtkMRML", ""), detail))
        if isTemporaryNode(node):
            temporary.append(node.GetName())
    sceneMB = sum(r[0] for r in rows)
    lines.append(f"Scene data: {sceneMB / 1024:.2f} GB in {len(rows)} nodes with bulk data")
    working = processMemoryMB()["working"]
    if working:
        lines.append(f"Outside the scene data (Slicer, Qt, VTK views, Python, loaded libraries): "
                     f"{max(0.0, working - sceneMB) / 1024:.2f} GB")
    libraries = [label for module, label in (("torch", "PyTorch"), ("monai", "MONAI"),
                                             ("totalsegmentator", "TotalSegmentator"))
                 if module in sys.modules]
    if libraries:
        lines.append(f"AI libraries loaded: {', '.join(libraries)} (they stay in memory, often about 1 GB, until "
                     "Slicer is restarted)")
    for megabytes, name, className, detail in sorted(rows, reverse=True)[:top]:
        lines.append(f"  {megabytes:8.1f} MB  {name} ({className}{', ' + detail if detail else ''})")
    lines.append("")
    lines.append("Nodes by type (count, MB):")
    for className, (count, megabytes) in sorted(byClass.items(), key=lambda item: (-item[1][1], -item[1][0]))[:12]:
        lines.append(f"  {className.replace('vtkMRML', '')}: {count}" + (f", {megabytes:.0f} MB" if megabytes >= 1 else ""))
    lines.append("")
    orphans = orphanExportColorTables()
    cli = finishedCliNodes()
    lines.append(f"Left-over temporary nodes: {len(temporary)}" + (f" ({', '.join(temporary[:5])})" if temporary else ""))
    lines.append(f"Unused export colour tables: {len(orphans)}; finished Taranis CLI nodes: {len(cli)}")
    arrays = pythonArraysMB()
    if arrays:
        lines.append("Arrays held by modules: " + ", ".join(f"{k} {v:.0f} MB" for k, v in arrays.items()))
    lines.append(f"Python objects tracked: {len(gc.get_objects())}")
    return "\n".join(lines)


# -- Releasing ------------------------------------------------------------------------------------------------------

def trimHeap():
    """Return freed heap memory to the operating system (Windows CRT / glibc); harmless if not available."""
    try:
        if sys.platform.startswith("win"):
            for library in ("ucrtbase", "msvcrt"):
                try:
                    getattr(ctypes.cdll, library)._heapmin()
                except Exception:
                    pass
        elif sys.platform.startswith("linux"):
            ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception as e:
        logging.debug(f"Taranis: heap trim: {e}")


def clearCaches():
    """Caches of the Taranis library (rebuilt on demand)."""
    for moduleName, attribute in (("TaranisLib.controller", "_labelStatsCache"),
                                  ("TaranisLib.controller", "_volumeHashCache"),
                                  ("TaranisLib.controller", "_resultKeyCache"),
                                  ("TaranisLib.case", "_dicomCache")):
        module = sys.modules.get(moduleName)
        cache = getattr(module, attribute, None) if module is not None else None
        if isinstance(cache, dict):
            cache.clear()


def processingRunning():
    """A CLI module (e.g. an EasyReg registration) is running: its temporary nodes are still in use."""
    for node in slicer.util.getNodesByClass("vtkMRMLCommandLineModuleNode"):
        try:
            if node.IsBusy():
                return True
        except Exception:
            pass
    return False


def releaseMemory():
    """Remove left-over temporary nodes, clear caches, collect garbage, trim the heap. Returns a summary text."""
    before = processMemoryMB()
    removed = 0
    running = processingRunning()
    for node in [n for n in slicer.util.getNodesByClass("vtkMRMLNode") if isTemporaryNode(n) and not running]:
        if node.GetScene() is not None and not node.IsA("vtkMRMLCommandLineModuleNode"):
            slicer.mrmlScene.RemoveNode(node)
            removed += 1
    for node in orphanExportColorTables() + finishedCliNodes():
        if node.GetScene() is not None:
            slicer.mrmlScene.RemoveNode(node)
            removed += 1
    clearCaches()
    collected = gc.collect()
    trimHeap()
    after = processMemoryMB()
    text = f"{removed} temporary node(s) removed, {collected} Python objects collected."
    if running:
        text += " A registration or other processing is running: its temporary nodes were kept."
    if before["working"] is not None and after["working"] is not None:
        text += f" RAM {before['working'] / 1024:.2f} → {after['working'] / 1024:.2f} GB."
    return text


_collectPending = False


def collectSoon():
    """Garbage collection and heap trim on the next event loop cycle (after heavy operations)."""
    global _collectPending
    if _collectPending:
        return
    _collectPending = True

    def run():
        global _collectPending
        _collectPending = False
        gc.collect()
        trimHeap()

    try:
        import qt
        qt.QTimer.singleShot(0, run)
    except Exception:
        run()
