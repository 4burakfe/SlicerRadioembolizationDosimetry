"""Code shared by the Taranis dosimetry modules (patient-relative and absolute).

Moved here unchanged from RadioembolizationDosimetryRelative / RadioembolizationDosimetryAbsolute, where it was
duplicated: constants, segment and isodose helpers, DVH, labels, layouts, report building, and the widget / logic
methods that were identical in both modules (DosimetryWidgetBase, DosimetryLogicBase). What differs between the
modules stays in the modules; they import everything listed in __all__.
"""

import os
import re
import json
import math
import datetime
import logging

import numpy as np
import qt
import ctk
import vtk
import slicer
from slicer.ScriptedLoadableModule import *

from . import doseguard as DG
from . import workflow as _W

# Isodose presets share one colour sequence (lowest -> highest level)
ISODOSE_COLORS = [
    ("blue", (0.00, 0.35, 1.00)),
    ("green", (0.00, 0.75, 0.00)),
    ("green-gold", (0.70, 0.78, 0.00)),
    ("yellow", (1.00, 1.00, 0.00)),
    ("orange", (1.00, 0.55, 0.00)),
    ("red", (1.00, 0.00, 0.00)),
    ("dark red", (0.60, 0.00, 0.00)),
    ("magenta", (1.00, 0.00, 1.00)),
]
ISODOSE_PRESETS = [
    ("Glass microspheres", [10, 20, 50, 75, 120, 200, 300, 400]),
    ("Resin microspheres", [5, 10, 25, 40, 75, 100, 150, 250]),
]

D_METRICS = [50, 60, 70, 80, 90, 95, 99]  # % of volume -> dose (Gy)
V_METRICS = [30, 40, 50, 60, 70, 80]      # dose (Gy) -> % of volume

# Nodes created by this module are tagged so they can be replaced on the next run
ROLE_ATTRIBUTE = "Taranis.Role"
ROLE_SEGMENT_MODEL = "SegmentModel"
ROLE_ISODOSE = "IsodoseModel"
ROLE_DVH = "DVH"
SEGMENT_MODEL_FOLDER = "Taranis segment models"
ISODOSE_FOLDER = "Taranis isodose surfaces"
DVH_FOLDER = "Taranis DVH"

RESULTS_LAYOUT_ID = 50101
RESULTS_LAYOUT_XML = """
<layout type="vertical" split="true">
 <item splitSize="500">
  <layout type="horizontal" split="true">
   <item splitSize="1000"><view class="vtkMRMLViewNode" singletontag="TaranisSegments3D"><property name="viewlabel" action="default">1</property></view></item>
   <item splitSize="1000"><view class="vtkMRMLViewNode" singletontag="TaranisIsodose3D"><property name="viewlabel" action="default">2</property></view></item>
   <item splitSize="1000"><view class="vtkMRMLPlotViewNode" singletontag="PlotView1"><property name="viewlabel" action="default">P</property></view></item>
  </layout>
 </item>
 <item splitSize="500">
  <layout type="horizontal" split="true">
   <item splitSize="1000"><view class="vtkMRMLSliceNode" singletontag="TaranisFusion">
    <property name="orientation" action="default">Axial</property>
    <property name="viewlabel" action="default">G</property>
    <property name="viewcolor" action="default">#6EB04B</property></view></item>
   <item splitSize="1000"><view class="vtkMRMLSliceNode" singletontag="TaranisIsodose">
    <property name="orientation" action="default">Axial</property>
    <property name="viewlabel" action="default">R</property>
    <property name="viewcolor" action="default">#F34A33</property></view></item>
   <item splitSize="1000"><view class="vtkMRMLSliceNode" singletontag="TaranisReference">
    <property name="orientation" action="default">Axial</property>
    <property name="viewlabel" action="default">Y</property>
    <property name="viewcolor" action="default">#EDD54C</property></view></item>
  </layout>
 </item>
</layout>
"""
# Dual monitor: the left column (segment models 3D view, fusion slice view) stays in the main window and the
# right 2x2 views (isodose 3D view, DVH, isodose and reference slice views) open in a separate window, which
# is moved to the second screen. Same views as the single monitor layout, so the display carries over.
RESULTS_DUAL_LAYOUT_ID = 50102
RESULTS_DUAL_LAYOUT_XML = """
<viewports>
 <layout type="vertical"MAIN_VIEWPORT_ATTRIBUTES split="true">
  <item splitSize="500"><view class="vtkMRMLViewNode" singletontag="TaranisSegments3D"><property name="viewlabel" action="default">1</property></view></item>
  <item splitSize="500"><view class="vtkMRMLSliceNode" singletontag="TaranisFusion">
   <property name="orientation" action="default">Axial</property>
   <property name="viewlabel" action="default">G</property>
   <property name="viewcolor" action="default">#6EB04B</property></view></item>
 </layout>
 <layout type="vertical" name="Taranis second screen" dockable="true" dockPosition="floating" split="true">
  <item splitSize="500">
   <layout type="horizontal" split="true">
    <item splitSize="1000"><view class="vtkMRMLViewNode" singletontag="TaranisIsodose3D"><property name="viewlabel" action="default">2</property></view></item>
    <item splitSize="1000"><view class="vtkMRMLPlotViewNode" singletontag="PlotView1"><property name="viewlabel" action="default">P</property></view></item>
   </layout>
  </item>
  <item splitSize="500">
   <layout type="horizontal" split="true">
    <item splitSize="1000"><view class="vtkMRMLSliceNode" singletontag="TaranisIsodose">
     <property name="orientation" action="default">Axial</property>
     <property name="viewlabel" action="default">R</property>
     <property name="viewcolor" action="default">#F34A33</property></view></item>
    <item splitSize="1000"><view class="vtkMRMLSliceNode" singletontag="TaranisReference">
     <property name="orientation" action="default">Axial</property>
     <property name="viewlabel" action="default">Y</property>
     <property name="viewcolor" action="default">#EDD54C</property></view></item>
   </layout>
  </item>
 </layout>
</viewports>
"""
# Replaced by the main-window viewport attributes of the running Slicer version (see resultsDualLayoutXml)
MAIN_VIEWPORT_MARKER = "MAIN_VIEWPORT_ATTRIBUTES"
RESULTS_LAYOUT_IDS = (RESULTS_LAYOUT_ID, RESULTS_DUAL_LAYOUT_ID)
# Layout chosen with the buttons ("single"/"dual"), shared by both Taranis modules; None = automatic (screens)
LAYOUT_MODE_ATTRIBUTE = "_taranisLayoutMode"
# Bottom row, all axial: (slice view name, role)
#   fusion:    reference + PET/SPECT (inferno) overlay, segment outlines
#   isodose:   reference + isodose lines + segment outlines (dose kept as invisible layer for the Data Probe)
#   reference: reference volume + segment outlines
SLICE_VIEWS = [("TaranisFusion", "fusion"), ("TaranisIsodose", "isodose"), ("TaranisReference", "reference")]
# The results layout uses its own view nodes (not Slicer's default "1", "2", Red, Green, Yellow), so renderings of
# other modules (e.g. the Epona MIP) never appear in the results views and the segment / isodose models never
# appear in the views of other modules.
SEGMENTS_VIEW_TAG = "TaranisSegments3D"
ISODOSE_VIEW_TAG = "TaranisIsodose3D"
THREED_VIEWS = (("segments3D", SEGMENTS_VIEW_TAG), ("isodose3D", ISODOSE_VIEW_TAG))
# View nodes used by earlier versions: models of saved scenes are moved to the new views
LEGACY_THREED_TAGS = {"1": "segments3D", "2": "isodose3D"}
LEGACY_SLICE_NAMES = {"Green": "fusion", "Red": "isodose", "Yellow": "reference"}
# Display nodes of other modules that are kept out of the results views
FOREIGN_DISPLAY_CLASSES = ("vtkMRMLVolumeRenderingDisplayNode", "vtkMRMLModelDisplayNode",
                           "vtkMRMLSegmentationDisplayNode", "vtkMRMLMarkupsDisplayNode")
# Attribute on the slicer module holding the active slice-view segment labels, so that a calculation in
# either Taranis module (absolute / patient relative) replaces the labels of the other one
ANNOTATIONS_ATTRIBUTE = "_taranisSegmentAnnotations"

# The results of the last calculation are marked with the name of the module that made them (attribute of
# the DVH chart node), so that after loading a scene only that module rebuilds its legend, labels and linked
# cameras (both Taranis modules share the results layout)
OWNER_ATTRIBUTE = "Taranis.Owner"

# Settings and the last results of each module are kept in the module's parameter node
# (vtkMRMLScriptedModuleNode), which is saved with the scene. Keys used by both modules:
SETTINGS_VERSION = "1"
PARAM_VERSION = "SettingsVersion"
PARAM_LIVER_SEGMENT = "LiverSegmentID"
PARAM_RESULTS = "Results"                          # JSON: tables, report data and display state
PARAM_ACTIVE_MODULE = "WasActiveModule"            # "true": module was open when the scene was saved
PARAM_REPORT_FILE = "LastReportFile"                # last saved RTF report (read by the Taranis workflow)
PARAM_REPORT_DOSE_CHECKSUM = "LastReportDoseChecksum"  # dose checksum of the calculation in that report
REF_RESULTS_DOSE = "ResultsDoseVolume"             # node references of the last calculation
REF_RESULTS_SEGMENTATION = "ResultsSegmentation"

# Segment categories. Stored as a JSON attribute on the segmentation node, so they are saved with the scene
# and shared by both Taranis modules.
CATEGORY_ATTRIBUTE = "Taranis.SegmentCategories"
CATEGORY_TUMOR = "tumor"
CATEGORY_NORMAL = "normal"
CATEGORY_OTHER = "other"
CATEGORY_VIABLE = "viable"     # viable (metabolically active, e.g. FDG-avid) tumour: like tumours, reported separately;
                               # may overlap the tumour segments
CATEGORY_LUNGS = "lungs"       # patient-relative: not calculated (only used for the lung shunt); absolute: calculated,
                               # with a warning (one tissue density for all voxels underestimates the lung dose)
CATEGORY_IGNORED = "ignored"   # not calculated and not shown
SEGMENT_CATEGORIES = [(CATEGORY_TUMOR, "Tumors"), (CATEGORY_VIABLE, "Viable tumors"),
                      (CATEGORY_NORMAL, "Normal tissue"), (CATEGORY_LUNGS, "Lungs"), (CATEGORY_OTHER, "Others"),
                      (CATEGORY_IGNORED, "Ignored (not calculated)")]

# Standard segment colours
COLOR_LIVER = (1.0, 1.0, 1.0)       # white
COLOR_PERFUSED = (1.0, 0.0, 0.0)    # bright red
COLOR_TUMOR = (0.70, 0.55, 1.00)      # pink
COLOR_NORMAL = (0.25, 0.88, 0.82)   # turquoise
COLOR_OTHER = (0.6, 0.6, 0.6)       # gray: others and uncategorized
COLOR_VIABLE = (195 / 255.0, 33 / 255.0, 72 / 255.0)   # bright maroon (#c32148)
COLOR_LUNGS = (0.45, 0.65, 1.00)    # light blue

# Role of a result row -> (text in the Category column, sort order, colour)
RESULT_ROLES = {
    "liver": ("Whole liver", 0, COLOR_LIVER),
    "perfused": ("Perfused volume", 1, COLOR_PERFUSED),
    "tumors": ("Tumors (combined)", 2, COLOR_TUMOR),
    CATEGORY_TUMOR: ("Tumor", 3, COLOR_TUMOR),
    "viables": ("Viable tumors (combined)", 4, COLOR_VIABLE),
    CATEGORY_VIABLE: ("Viable tumor", 5, COLOR_VIABLE),
    CATEGORY_NORMAL: ("Normal tissue", 6, COLOR_NORMAL),
    CATEGORY_LUNGS: ("Lung", 7, COLOR_LUNGS),
    CATEGORY_OTHER: ("Other", 8, COLOR_OTHER),
    "uncategorized": ("Uncategorized", 9, COLOR_OTHER),
    CATEGORY_IGNORED: ("Ignored", 10, COLOR_OTHER),
}
COMBINED_TUMORS_NAME = "All tumors (combined)"
COMBINED_VIABLE_NAME = "All viable tumors (combined)"

# Top-left 3D view: only these roles get a wireframe model, with this opacity
SEGMENT_MODEL_OPACITY = {"liver": 0.03, "perfused": 0.05, CATEGORY_TUMOR: 0.10, CATEGORY_VIABLE: 0.10}
FUSION_OPACITY = 0.5

# Display window presets of the slice views ("Display windowing" box).
# Reference volume: (button, kind, a, b, tooltip) - kind "hu": window width a / level b in Hounsfield units,
# "range": fraction a..b of the intensity range, "percentile": percentiles a..b of the non-background voxels
REFERENCE_WINDOW_PRESETS = [
    ("CT lung", "hu", 1500.0, -600.0, "CT lung window: W 1500 / L -600 HU"),
    ("CT soft tissue", "hu", 400.0, 40.0, "CT soft tissue (abdomen) window: W 400 / L 40 HU"),
    ("CT head", "hu", 80.0, 40.0, "CT head (brain) window: W 80 / L 40 HU"),
    ("MRI 0-100%", "range", 0.0, 1.0, "MRI: full intensity range (minimum to maximum)"),
    ("MRI high contrast", "percentile", 2.0, 98.0,
     "MRI with more contrast: 2nd to 98th percentile of the non-background voxels"),
]
# PET/SPECT: window from 0 to this percentage of the maximum voxel value
SPECT_WINDOW_PERCENTS = [10, 25, 50, 75, 100]
DEFAULT_SEGMENT_OUTLINE_THICKNESS = 3  # px, segment outlines in the slice views
DEFAULT_ISODOSE_LINE_THICKNESS = 2     # px, isodose lines in the isodose slice view

DISCLAIMER = "\u26a0 THIS SOFTWARE IS NOT A CERTIFIED MEDICAL DEVICE. IT IS INTENDED FOR RESEARCH PURPOSES ONLY."


def prepareImageValues(volumeArray, clipNegativeValues):
    """Input voxel values as a float64 copy (no integer overflow, the volume itself is never modified) and
    the number of negative voxels. With clipNegativeValues, negative values (reconstruction noise) are set
    to 0. This is done once, at input, so every later calculation (dose map, totals,
    QC fractions, statistics) uses the same values."""
    values = np.array(volumeArray, dtype=np.float64)
    negativeCount = int(np.count_nonzero(values < 0))
    if clipNegativeValues and negativeCount:
        np.maximum(values, 0.0, out=values)
    return values, negativeCount


def negativeVoxelNote(negativeCount, voxelCount, clippedNegativeValues, regionText=""):
    """QC sentence about negative input voxels."""
    if negativeCount == 0:
        return "No negative voxel values in the input image."
    share = 100.0 * negativeCount / max(voxelCount, 1)
    sentence = f"{negativeCount} voxels ({share:.2f}% of the image{regionText}) had negative values (reconstruction noise); "
    if clippedNegativeValues:
        return sentence + "they were set to 0 in all calculations."
    return sentence + "they were used as is and produce negative voxel doses."


def countsOutsideMaskFraction(counts, mask):
    """Fraction of (positive) image counts lying outside the mask (QC number)."""
    positive = np.clip(counts, 0, None)
    total = float(positive.sum(dtype=np.float64))
    if total <= 0:
        return float("nan")
    return float(positive[~mask].sum(dtype=np.float64)) / total


def segmentDoseStatistics(doseArray, segmentMask, voxelVolumeML, conversionFactor, densityGPerML):
    """Return (mean dose Gy, volume mL, activity MBq). Dose/activity are NaN for empty masks."""
    nVoxels = int(np.count_nonzero(segmentMask))
    volumeML = nVoxels * voxelVolumeML
    if nVoxels == 0:
        return float("nan"), 0.0, float("nan")
    meanDoseGy = float(np.mean(doseArray[segmentMask], dtype=np.float64))
    activityMBq = meanDoseGy * volumeML * densityGPerML / conversionFactor
    return meanDoseGy, volumeML, activityMBq


def doseAtVolumePercent(sortedDosesAsc, volumePercent):
    """D_x: minimum dose received by the hottest x % of the volume (voxel-exact, no interpolation)."""
    n = len(sortedDosesAsc)
    if n == 0:
        return float("nan")
    hottest = int(math.ceil(round(volumePercent / 100.0 * n, 9)))
    hottest = min(max(hottest, 1), n)
    return float(sortedDosesAsc[n - hottest])


def volumePercentAtDose(sortedDosesAsc, doseGy):
    """V_x: percentage of the volume receiving at least doseGy."""
    n = len(sortedDosesAsc)
    if n == 0:
        return float("nan")
    return 100.0 * (n - int(np.searchsorted(sortedDosesAsc, doseGy, side="left"))) / n


def cumulativeDvh(sortedDosesAsc, doseAxis):
    """Cumulative DVH: % volume receiving >= each dose in doseAxis."""
    n = len(sortedDosesAsc)
    return 100.0 * (n - np.searchsorted(sortedDosesAsc, doseAxis, side="left")) / n


def formatNumber(value):
    return "n/a" if value is None or not np.isfinite(value) else f"{value:.2f}"


def rgbToHex(rgb):
    return "#%02x%02x%02x" % tuple(int(round(max(0.0, min(1.0, c)) * 255)) for c in rgb)


def isodoseLegendHtml(levels):
    parts = []
    for level, (_, rgb) in zip(levels, ISODOSE_COLORS):
        parts.append(f'<span style="color:{rgbToHex(rgb)}; font-size:15px;">&#9632;</span>&nbsp;{level:g}&nbsp;Gy')
    return "&nbsp;&nbsp; ".join(parts)


def rtfEscape(text):
    """Escape text for RTF. Non-ASCII (e.g. Turkish ç, ğ, ı, ş) becomes \\uN? so the file stays pure ASCII."""
    out = []
    for ch in str(text):
        if ch in "\\{}":
            out.append("\\" + ch)
        elif ch == "\n":
            out.append("\\line ")
        elif ord(ch) < 128:
            out.append(ch)
        else:
            utf16 = ch.encode("utf-16-le")
            for i in range(0, len(utf16), 2):
                code = int.from_bytes(utf16[i:i + 2], "little")
                out.append(f"\\u{code - 65536 if code > 32767 else code}?")
    return "".join(out)


def pngSize(pngBytes):
    """(width, height) in pixels from a PNG header."""
    if len(pngBytes) < 24 or pngBytes[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Not a PNG image.")
    return int.from_bytes(pngBytes[16:20], "big"), int.from_bytes(pngBytes[20:24], "big")


def rtfPicture(pngBytes, maxWidthTwips=8600):
    """RTF picture group for a PNG, scaled to at most maxWidthTwips wide (the default RTF text width is
    8640 twips = 6 inch)."""
    width, height = pngSize(pngBytes)
    goalWidth = min(maxWidthTwips, width * 15)  # 15 twips per pixel at 96 dpi
    goalHeight = int(round(goalWidth * height / max(width, 1)))
    hexData = pngBytes.hex()
    lines = "\n".join(hexData[i:i + 128] for i in range(0, len(hexData), 128))
    return (r"{\pict\pngblip\picw%d\pich%d\picwgoal%d\pichgoal%d" % (width, height, goalWidth, goalHeight)
            + "\n" + lines + "}")


def dicomValue(volumeNode, tag):
    """DICOM value (string) of the first instance of a DICOM-loaded volume, "" if unknown."""
    if volumeNode is None:
        return ""
    uids = (volumeNode.GetAttribute("DICOM.instanceUIDs") or "").split()
    if not uids or getattr(slicer, "dicomDatabase", None) is None:
        return ""
    try:
        return (slicer.dicomDatabase.instanceValue(uids[0], tag) or "").strip()
    except Exception:
        return ""


def formatDicomName(name):
    """'DOE^JOHN^^^' -> 'DOE JOHN'."""
    return " ".join(part.strip() for part in (name or "").split("^") if part.strip())


def formatDicomDateTime(date, time=""):
    """('20220203', '100213.00') -> '2022-02-03 10:02'; '' if the date is not a DICOM date."""
    date = (date or "").strip()
    if len(date) != 8 or not date.isdigit():
        return ""
    text = f"{date[:4]}-{date[4:6]}-{date[6:]}"
    time = (time or "").strip().split(".")[0]
    if len(time) >= 4 and time[:4].isdigit():
        text += f" {time[:2]}:{time[2:4]}"
    return text


def imagingDate(volumeNode):
    """Acquisition (or series, or study) date and time of a DICOM-loaded volume, '' if unknown."""
    for dateTag, timeTag in (("0008,0022", "0008,0032"), ("0008,0021", "0008,0031"), ("0008,0020", "0008,0030")):
        text = formatDicomDateTime(dicomValue(volumeNode, dateTag), dicomValue(volumeNode, timeTag))
        if text:
            return text
    return ""


def reportPatientInfo(imageNode, referenceNode=None):
    """[(label, value)] for the top of the report: patient name and ID (DICOM header of the dosimetry image, else
    of the reference image), the Taranis case, and the imaging dates, when available."""
    rows = []
    name = patientID = ""
    for node in (imageNode, referenceNode):
        name = name or formatDicomName(dicomValue(node, "0010,0010"))
        patientID = patientID or dicomValue(node, "0010,0020")
    try:
        from TaranisLib.case import TaranisCase
        case = TaranisCase.find()
    except Exception:
        case = None
    if case is not None:
        name = name or (case.name or "")
        patientID = patientID or (case.caseID or "")
    rows.append(("Patient name", name or "not available"))
    rows.append(("Patient ID", patientID or "not available"))
    if case is not None and (case.name or case.caseID):
        rows.append(("Taranis case", " / ".join(t for t in (case.name, case.caseID) if t)))
    date = imagingDate(imageNode)
    if date:
        rows.append(("Imaging date (dosimetry image)", date))
    referenceDate = imagingDate(referenceNode)
    if referenceDate:
        rows.append(("Imaging date (reference image)", referenceDate))
    return rows


SEGMENT_DOSE_HEADER = ["Segment", "Category", "Dose (Gy)", "Volume (mL)", "Mass (g)", "Activity (MBq)"]
REPORT_DISCLAIMER = "This software is NOT a medical device. For research purposes only."
REPORT_WARNING = DISCLAIMER.replace("\u26a0 ", "")   # red, under the report title
_METRIC_LINE = re.compile(r"^(?P<label>.*): (?P<items>[DV]\d+(?:\.\d+)? = .*)$")
_NUMBER_WITH_UNIT = re.compile(r"^(?P<number>-?[\d.,]+|n/a)\s*(?P<unit>Gy|%)?(?P<rest>.*)$")


def segmentDoseTable(rows):
    """(header, rows) of the segment dose table; older results without mass (5 values) get an empty mass, and the
    mass column is left out when no row has a mass."""
    table = []
    for row in rows:
        row = [str(v) for v in row]
        if len(row) == 5:
            row.insert(4, "")
        table.append(row[:6] + [""] * (6 - len(row[:6])))
    header = list(SEGMENT_DOSE_HEADER)
    if table and not any(r[4] for r in table):
        header.pop(4)
        table = [r[:4] + r[5:] for r in table]
    return header, table


def parseMetricLines(lines):
    """[(label, [(metric, value text)])] of 'label: D50 = 1.00 Gy, D60 = ...' lines, or None if a line differs."""
    parsed = []
    for line in lines:
        match = _METRIC_LINE.match(line)
        if not match:
            return None
        items = []
        for item in match.group("items").split(", "):
            metric, _, value = item.partition(" = ")
            if not value:
                return None
            items.append((metric.strip(), value.strip()))
        parsed.append((match.group("label"), items))
    return parsed


def metricTable(lines):
    """(header, rows) of a D / V / custom metric section, or None if its lines cannot be read as metrics.
    Same metrics on every line (D and V values): one column per metric, unit in the header, as in the module.
    Otherwise (custom metrics): Segment | Metric | Value."""
    parsed = parseMetricLines(lines)
    if not parsed:
        return None
    names = [[metric for metric, _ in items] for _, items in parsed]
    if len(names[0]) > 1 and all(n == names[0] for n in names):
        units = set()
        table = []
        for label, items in parsed:
            row = [label]
            for _, value in items:
                match = _NUMBER_WITH_UNIT.match(value)
                if match and not match.group("rest").strip():
                    row.append(match.group("number"))
                    units.add(match.group("unit") or "")
                else:
                    row.append(value)
                    units.add(None)
            table.append(row)
        unit = units.pop() if len(units) == 1 else None
        header = ["Segment"] + [f"{n} ({unit})" if unit else n for n in names[0]]
        return header, table
    return ["Segment", "Metric", "Value"], [[label, metric, value] for label, items in parsed for metric, value in items]


def reportBlocks(parameters, rows, notes=(), sections=(), patient=()):
    """The report content as blocks shared by the RTF and PDF writers:
    ("table", heading, header or None, rows) and ("text", heading, lines)."""
    blocks = []
    if patient:
        blocks.append(("table", "Patient", None, [[str(a), str(b)] for a, b in patient]))
    blocks.append(("table", "Parameters", ["Parameter", "Value"], [[str(a), str(b)] for a, b in parameters]))
    header, table = segmentDoseTable(rows)
    blocks.append(("table", "Segment Doses", header, table))
    for section in sections:
        heading, lines = section[0], list(section[1])
        if not lines:
            continue
        parsed = metricTable(lines)
        if parsed:
            blocks.append(("table", heading, parsed[0], parsed[1]))
        else:
            blocks.append(("text", heading, lines))
    if notes:
        blocks.append(("text", "Notes", list(notes)))
    return blocks


def buildTsvTable(rows, sections=()):
    """Tab-separated table: one line per segment of the segment dose table, with its D and V values side by side
    (Segment, Category, Dose, Volume, Mass, Activity, D50 ... D99, V30 ... V80). sections: (heading, [lines]) as in
    the report; only the D and V value sections are used (matched by segment label, in order for repeated labels)."""
    header, table = segmentDoseTable(rows)
    header, table = list(header), [list(r) for r in table]
    for heading, lines in sections:
        if not (heading.startswith("D values") or heading.startswith("V values")):
            continue
        parsed = metricTable(list(lines)) if lines else None
        if not parsed or parsed[0][:1] != ["Segment"] or parsed[0][1:2] == ["Metric"]:
            continue
        metricHeader, metricRows = parsed
        byLabel = {}
        for row in metricRows:
            byLabel.setdefault(row[0], []).append(row[1:])
        used = {}
        for row in table:
            values = byLabel.get(row[0], [])
            index = used.get(row[0], 0)
            used[row[0]] = index + 1
            row.extend(values[index] if index < len(values) else [""] * (len(metricHeader) - 1))
        header.extend(metricHeader[1:])
    clean = lambda text: str(text).replace("\t", " ").replace("\r", " ").replace("\n", " ")
    return "\n".join("\t".join(clean(c) for c in line) for line in [header] + table) + "\n"


def _isNumeric(text):
    return bool(re.match(r"^-?[\d.,]+( ?%| ?Gy| ?mL| ?MBq)?( \(.*\))?$|^n/a$", str(text).strip()))


RTF_PAGE_WIDTH_TWIPS = 11906 - 2 * 1134   # A4, 2 cm margins


def rtfColumnWidths(header, rows, total=RTF_PAGE_WIDTH_TWIPS):
    """Column widths (twips) proportional to the longest text of each column, first column favoured."""
    lengths = []
    for col in range(len(header)):
        texts = [str(header[col])] + [str(r[col]) for r in rows if col < len(r)]
        longest = max(len(t) for t in texts)
        lengths.append(max(6, min(longest, 42 if col == 0 else 22)))
    scale = total / float(sum(lengths))
    widths = [int(n * scale) for n in lengths]
    widths[-1] += total - sum(widths)
    return widths


def rtfTable(header, rows):
    """RTF table with a shaded, bold, repeated header row; numbers right-aligned."""
    columns = header or (rows[0] if rows else [""])
    widths = rtfColumnWidths(columns, rows)
    border = r"\clbrdrt\brdrs\brdrw6\brdrcf3\clbrdrl\brdrs\brdrw6\brdrcf3\clbrdrb\brdrs\brdrw6\brdrcf3" \
             r"\clbrdrr\brdrs\brdrw6\brdrcf3"
    out = []
    allRows = ([header] if header else []) + list(rows)
    for index, cells in enumerate(allRows):
        isHeader = bool(header) and index == 0
        definition = r"\trowd\trgaph70\trleft0\trkeep" + (r"\trhdr" if isHeader else "")
        right = 0
        for width in widths:
            right += width
            definition += border + (r"\clcbpat2" if isHeader else "") + r"\cellx%d" % right
        out.append(definition + "\n")
        texts = []
        for col in range(len(widths)):
            text = str(cells[col]) if col < len(cells) else ""
            align = r"\qr" if (header and col > 0 and not isHeader and _isNumeric(text)) else (r"\qc" if isHeader and col > 0
                                                                                   else r"\ql")
            bold = isHeader or (not header and col == 0)   # header-less (label | value) tables: bold labels
            content = (r"{\b " + rtfEscape(text) + "}") if bold else rtfEscape(text)
            texts.append(r"\pard\intbl" + align + r"\fs18 " + content + r"\cell")
        out.append("".join(texts) + r"\row" + "\n")
    out.append(r"\pard\fs20" + "\n")
    return "".join(out)


def buildRtfReport(titleLines, parameters, rows, notes=(), sections=(), screenshots=(), patient=()):
    """parameters: (label, value) pairs; rows: (segment, category, dose, volume, mass, activity) strings (older
    results without mass: 5 values); notes: text lines; sections: (heading, [lines]) pairs after the segment
    doses (D / V / custom metric lines become tables as in the module); screenshots: (caption, pngBytes) pairs,
    one per view, appended at the end of the report."""
    parts = [r"{\rtf1\ansi\deff0{\fonttbl{\f0 Arial;}}"
             r"{\colortbl;\red0\green0\blue0;\red221\green228\blue237;\red140\green140\blue140;\red220\green0\blue0;}"
             r"\paperw11906\paperh16838\margl1134\margr1134\margt1134\margb1134\f0\fs20" + "\n"]
    parts.append(r"\pard\sa60{\b\fs28 " + r"\line ".join(rtfEscape(t) for t in titleLines) + r"}\par" + "\n")
    parts.append(r"\pard\sa120{\b\cf4 " + rtfEscape(REPORT_WARNING) + r"}\par" + "\n")   # colour 4: red
    parts.append(r"\pard\sa200 Generated: " + rtfEscape(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                 + r"\par" + "\n")
    for block in reportBlocks(parameters, rows, notes, sections, patient):
        kind, heading = block[0], block[1]
        parts.append(r"\pard\keepn\sb240\sa80{\b\fs22 " + rtfEscape(heading) + r"}\par" + "\n")
        if kind == "table":
            parts.append(rtfTable(block[2], block[3]))
        else:
            for line in block[2]:
                parts.append(r"\pard\sa40 " + rtfEscape(line) + r"\par" + "\n")
    parts.append(r"\pard\sb240{\i " + rtfEscape(REPORT_DISCLAIMER) + r"}\par" + "\n")
    if screenshots:
        # \par before \page: a page break inside a paragraph makes some readers (LibreOffice) drop the pictures
        parts.append(r"\par\page{\b " + rtfEscape("Screenshots") + r"}\par" + "\n")
        for caption, pngBytes in screenshots:
            # \keepn keeps each caption on the same page as its picture
            parts.append(r"\pard\keepn\sb200{\b " + rtfEscape(caption) + r"}\par\pard" + "\n")
            parts.append(rtfPicture(pngBytes) + r"\par" + "\n")
    parts.append(r"\pard\sb200 End of Report\par}" + "\n")
    return "".join(parts)


def _html(text):
    return (str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace("\n", "<br>"))


PDF_RESOLUTION = 96                # layout in screen pixels: image widths and table sizes as on screen
PDF_MARGINS_MM = (12, 12, 12, 10)  # left, top, right, bottom of the A4 page
PDF_FOOTER_MM = 7                  # page number line at the bottom of the printable area
# screenshots: the A4 text width (a few pixels less, so rounding never pushes them past the right margin)
PDF_IMAGE_WIDTH_PX = int((210 - PDF_MARGINS_MM[0] - PDF_MARGINS_MM[2]) / 25.4 * PDF_RESOLUTION) - 4


def buildHtmlReport(titleLines, parameters, rows, notes=(), sections=(), screenshots=(), patient=()):
    """The report as HTML for Qt's rich text engine (PDF export): same content and tables as the RTF report."""
    import base64
    out = ["<html><body style='font-family: Arial; font-size: 9pt;'>",
           "<p style='font-size: 14pt; font-weight: bold; margin-bottom: 2px;'>"
           + "<br>".join(_html(t) for t in titleLines) + "</p>",
           f"<p style='color: #dc0000; font-weight: bold;'>{_html(REPORT_WARNING)}</p>",
           f"<p>Generated: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}</p>"]
    for block in reportBlocks(parameters, rows, notes, sections, patient):
        kind, heading = block[0], block[1]
        out.append(f"<p style='font-size: 11pt; font-weight: bold; margin-top: 12px; margin-bottom: 4px;'>"
                   f"{_html(heading)}</p>")
        if kind == "table":
            header, table = block[2], block[3]
            out.append("<table width='100%' border='1' cellspacing='0' cellpadding='3' "
                       "style='border-collapse: collapse; border-color: #8c8c8c;'>")
            if header:
                out.append("<thead><tr>" + "".join(
                    f"<th bgcolor='#dde4ed' align='{'left' if i == 0 else 'center'}'>{_html(h)}</th>"
                    for i, h in enumerate(header)) + "</tr></thead>")
            for cells in table:
                out.append("<tr>" + "".join(
                    f"<td align='{'right' if header and i > 0 and _isNumeric(c) else 'left'}'>"
                    + (f"<b>{_html(c)}</b>" if (not header and i == 0) else _html(c)) + "</td>"
                    for i, c in enumerate(cells)) + "</tr>")
            out.append("</table>")
        else:
            out.append("".join(f"<p style='margin: 0px;'>{_html(line)}</p>" for line in block[2]))
    out.append(f"<p style='margin-top: 12px;'><i>{_html(REPORT_DISCLAIMER)}</i></p>")
    if screenshots:
        out.append("<p style='page-break-before: always; font-size: 11pt; font-weight: bold;'>Screenshots</p>")
        for caption, pngBytes in screenshots:
            width, height = pngSize(pngBytes)
            shownWidth = min(PDF_IMAGE_WIDTH_PX, width)
            shownHeight = int(round(shownWidth * height / max(width, 1)))
            data = base64.b64encode(pngBytes).decode("ascii")
            out.append(f"<p style='margin-top: 10px;'><b>{_html(caption)}</b><br>"
                       f"<img src='data:image/png;base64,{data}' width='{shownWidth}' height='{shownHeight}'></p>")
    out.append("<p>End of Report</p></body></html>")
    return "".join(out)


def pdfExportAvailable():
    """PDF export uses Qt only (QPdfWriter, or QPrinter in PDF mode), both part of Slicer."""
    return hasattr(qt, "QPdfWriter") or hasattr(qt, "QPrinter")


def _pdfDevice(fileName):
    """A4 PDF paint device (QPdfWriter, or QPrinter in PDF mode on older Qt) with the report's margins."""
    if hasattr(qt, "QPdfWriter"):
        device = qt.QPdfWriter(fileName)
        device.setResolution(PDF_RESOLUTION)
    else:
        device = qt.QPrinter(qt.QPrinter.ScreenResolution)
        device.setOutputFormat(qt.QPrinter.PdfFormat)
        device.setOutputFileName(fileName)
    try:
        device.setPageSize(qt.QPageSize(qt.QPageSize.A4))
    except Exception as e:
        logging.debug(f"PDF page size: {e}")
    try:
        device.setPageMargins(qt.QMarginsF(*PDF_MARGINS_MM), qt.QPageLayout.Millimeter)
    except Exception as e:
        logging.debug(f"PDF margins: {e}")
    return device


def _paintPdfPages(document, device):
    """Lay the document out on the printable area of the pages and paint it page by page, with the page number
    in a footer. (QTextDocument.print adds its own 2 cm margins on every side: the text column became narrower
    than the screenshots, which were cut off at the right.)"""
    paintRect = device.pageLayout().paintRectPixels(PDF_RESOLUTION)
    width = float(paintRect.width())
    footer = PDF_FOOTER_MM / 25.4 * PDF_RESOLUTION
    bodyHeight = float(paintRect.height()) - footer
    if width <= 0 or bodyHeight <= 0:
        raise RuntimeError("Empty PDF page area.")
    document.documentLayout().setPaintDevice(device)
    document.setPageSize(qt.QSizeF(width, bodyHeight))   # paginated: lines and table rows are not split
    pageCount = max(1, document.pageCount())
    painter = qt.QPainter()
    if not painter.begin(device):
        raise RuntimeError("Could not start writing the PDF.")
    try:
        font = qt.QFont("Arial", 8)
        for page in range(pageCount):
            if page:
                device.newPage()
            painter.save()
            painter.translate(0.0, -page * bodyHeight)
            document.drawContents(painter, qt.QRectF(0.0, page * bodyHeight, width, bodyHeight))
            painter.restore()
            painter.setFont(font)
            painter.drawText(qt.QRectF(0.0, bodyHeight, width, footer),
                             int(qt.Qt.AlignRight | qt.Qt.AlignVCenter), f"{page + 1} / {pageCount}")
    finally:
        painter.end()


def writePdfReport(html, fileName):
    """Render the HTML report to an A4 PDF with Qt's rich text engine: 12 mm side margins, screenshots scaled to
    the text width (PDF_IMAGE_WIDTH_PX)."""
    def newDocument():
        document = qt.QTextDocument()
        document.setDefaultFont(qt.QFont("Arial", 9))
        document.setDocumentMargin(0)   # the page margins are the only margins
        # black text: painting the pages ourselves uses the application palette (white text in Slicer's dark theme)
        document.setDefaultStyleSheet("body, p, table, tr, th, td, li, span, b, i { color: #000000; }")
        document.setHtml(html)
        return document

    device = _pdfDevice(fileName)
    try:
        _paintPdfPages(newDocument(), device)
    except Exception as e:
        # fallback: Qt's own printing (adds 2 cm margins; wide screenshots may be cut at the right)
        logging.warning(f"PDF page layout failed ({e}); using Qt's default printing.")
        del device
        device = _pdfDevice(fileName)
        document = newDocument()
        printDocument = getattr(document, "print_", None) or getattr(document, "print")
        printDocument(device)
    del device  # the PDF file is completed when the writer is destroyed
    if not os.path.exists(fileName) or os.path.getsize(fileName) == 0:
        raise RuntimeError("The PDF file was not written.")


def segmentVolumeML(segmentationNode, segmentID):
    """Volume of one segment in mL, counted on its binary labelmap (the segmentation's own resolution)."""
    from vtk.util.numpy_support import vtk_to_numpy
    segmentation = segmentationNode.GetSegmentation()
    segment = segmentation.GetSegment(segmentID)
    if segment is None:
        return None
    labelmapName = slicer.vtkSegmentationConverter.GetSegmentationBinaryLabelmapRepresentationName()
    if not segmentation.ContainsRepresentation(labelmapName):
        segmentationNode.CreateBinaryLabelmapRepresentation()
    image = segment.GetRepresentation(labelmapName)
    if image is None or image.GetPointData() is None or image.GetPointData().GetScalars() is None:
        return None
    values = vtk_to_numpy(image.GetPointData().GetScalars())
    labelValue = segment.GetLabelValue() if hasattr(segment, "GetLabelValue") else 1  # shared labelmap layers
    sx, sy, sz = image.GetSpacing()
    return int(np.count_nonzero(values == labelValue)) * sx * sy * sz / 1000.0  # mm^3 -> mL


def voxelVolumeMLFromNode(volumeNode):
    sx, sy, sz = volumeNode.GetSpacing()
    return sx * sy * sz / 1000.0  # mm^3 -> mL


from .memory import removeTemporaryLabelmap, collectSoon   # noqa: E402  (Slicer-only helpers)


def segmentMaskOnVolumeGrid(segmentationNode, segmentID, referenceVolumeNode):
    """Boolean numpy mask of one segment resampled onto the reference volume grid.
    The temporary labelmap node is always removed, even if export fails."""
    labelmapNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
    labelmapNode.SetHideFromEditors(True)
    try:
        success = slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(
            segmentationNode, [segmentID], labelmapNode, referenceVolumeNode)
        if not success or labelmapNode.GetImageData() is None:
            raise RuntimeError(f"Could not export segment '{segmentID}' onto the image grid.")
        mask = slicer.util.arrayFromVolume(labelmapNode) > 0  # copy; valid after node removal
    finally:
        removeTemporaryLabelmap(labelmapNode)
    return mask


def segmentLabelText(name, volumeML=None, meanDoseGy=None):
    """Slice view label of a segment: name, volume (mL) below it and, after a calculation, the mean dose (Gy)."""
    lines = [name]
    if volumeML is not None and np.isfinite(volumeML):
        lines.append(f"{volumeML:.1f} mL")
    if meanDoseGy is not None:
        lines.append(f"Mean {meanDoseGy:.1f} Gy" if np.isfinite(meanDoseGy) else "Mean dose n/a")
    return "\n".join(lines)


def _volumeDisplayNode(volumeNode):
    volumeNode.CreateDefaultDisplayNodes()
    return volumeNode.GetDisplayNode()


def _setWindowMinMax(volumeNode, minimum, maximum):
    """Fixed display window (automatic window/level off)."""
    if not np.isfinite(minimum) or not np.isfinite(maximum):
        raise ValueError("The image has no finite voxel values.")
    if maximum <= minimum:
        maximum = minimum + 1.0
    displayNode = _volumeDisplayNode(volumeNode)
    displayNode.SetAutoWindowLevel(False)
    displayNode.SetWindowLevelMinMax(float(minimum), float(maximum))


def applyReferenceWindowPreset(volumeNode, preset, maxSampleVoxels=2000000):
    """Apply a REFERENCE_WINDOW_PRESETS entry to the reference (CT/MRI) volume."""
    _, kind, a, b = preset[:4]
    if kind == "hu":
        _setWindowMinMax(volumeNode, b - a / 2.0, b + a / 2.0)
        return
    values = slicer.util.arrayFromVolume(volumeNode).ravel()
    values = values[::max(1, values.size // maxSampleVoxels)].astype(np.float64)  # sample of large volumes
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("The image has no finite voxel values.")
    low, high = float(values.min()), float(values.max())
    if kind == "range":
        _setWindowMinMax(volumeNode, low + a * (high - low), low + b * (high - low))
    else:
        tissue = values[values > low]  # ignore the background (air / outside the field of view)
        if tissue.size == 0:
            tissue = values
        minimum, maximum = np.percentile(tissue, [a, b])
        _setWindowMinMax(volumeNode, minimum, maximum)


def applySpectWindowPercent(volumeNode, percent):
    """PET/SPECT window from 0 to `percent` % of the maximum voxel value."""
    maximum = float(np.nanmax(slicer.util.arrayFromVolume(volumeNode)))
    if not np.isfinite(maximum) or maximum <= 0:
        raise ValueError("The PET/SPECT image has no positive voxel values.")
    _setWindowMinMax(volumeNode, 0.0, maximum * percent / 100.0)


def tableRows(table):
    """Cell texts of a QTableWidget, row by row ('' for empty cells)."""
    rows = []
    for row in range(table.rowCount):
        cells = []
        for col in range(table.columnCount):
            item = table.item(row, col)
            cells.append(item.text() if item is not None else "")
        rows.append(cells)
    return rows


def doseChecksum(doseArray):
    """[sum, max] of a dose map: detects a dose map that was overwritten after the calculation."""
    if doseArray.size == 0:
        return [0.0, 0.0]
    return [float(np.sum(doseArray, dtype=np.float64)), float(np.max(doseArray))]


def widgetSettingValue(widget, kind):
    """Value of a settings widget as a parameter node string. kind: bool, number, text (combo box), lineedit."""
    if kind == "bool":
        return "true" if widget.checked else "false"
    if kind == "number":
        return repr(float(widget.value))
    if kind == "text":
        return widget.currentText
    if kind == "lineedit":
        return widget.text
    raise ValueError(f"Unknown setting kind '{kind}'")


def applyWidgetSetting(widget, kind, value):
    """Inverse of widgetSettingValue. Combo box texts that no longer exist are ignored."""
    if kind == "bool":
        widget.setChecked(value == "true")
    elif kind == "number":
        number = float(value)
        if isinstance(widget, qt.QSpinBox):
            widget.setValue(int(round(number)))
        else:
            widget.setValue(number)
    elif kind == "text":
        index = widget.findText(value)
        if index >= 0:
            widget.setCurrentIndex(index)
    elif kind == "lineedit":
        widget.setText(value)
    else:
        raise ValueError(f"Unknown setting kind '{kind}'")


def writeDoseVolume(outputVolumeNode, referenceVolumeNode, doseArray):
    """Write dose array into output volume with the reference volume's geometry and transform."""
    slicer.util.updateVolumeFromArray(outputVolumeNode, doseArray.astype(np.float32))
    ijkToRas = vtk.vtkMatrix4x4()
    referenceVolumeNode.GetIJKToRASMatrix(ijkToRas)
    outputVolumeNode.SetIJKToRASMatrix(ijkToRas)
    outputVolumeNode.SetAndObserveTransformNodeID(referenceVolumeNode.GetTransformNodeID())
    outputVolumeNode.SetAttribute("DicomRtImport.DoseVolume", "1")
    outputVolumeNode.CreateDefaultDisplayNodes()
    displayNode = outputVolumeNode.GetDisplayNode()
    if displayNode:
        displayNode.SetAutoWindowLevel(False)
        displayNode.SetWindowLevel(250, 125)
        colorNode = slicer.mrmlScene.GetFirstNodeByName("PET-Rainbow2")
        if colorNode:
            displayNode.SetAndObserveColorNodeID(colorNode.GetID())


def makeIsodoseLegendActor(levelColors):
    """2D legend box (highest level on top) for [(levelGy, rgb)], anchored at the lower right of a view."""
    entries = sorted(levelColors, key=lambda lc: lc[0], reverse=True)
    square = vtk.vtkPlaneSource()
    square.Update()
    legend = vtk.vtkLegendBoxActor()
    legend.SetNumberOfEntries(len(entries))
    for index, (level, rgb) in enumerate(entries):
        legend.SetEntry(index, square.GetOutput(), f"{level:g} Gy", list(rgb))
    legend.UseBackgroundOn()
    legend.SetBackgroundColor(0.0, 0.0, 0.0)
    legend.SetBackgroundOpacity(0.6)
    legend.BorderOff()
    textProperty = legend.GetEntryTextProperty()
    textProperty.SetColor(1.0, 1.0, 1.0)
    textProperty.BoldOff()
    textProperty.ShadowOff()
    legend.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    legend.GetPositionCoordinate().SetValue(0.84, 0.10)
    # Position2 is relative to Position: width, height in normalized viewport units
    legend.GetPosition2Coordinate().SetValue(0.14, min(0.5, 0.05 * len(entries) + 0.02))
    return legend


def removeLegendActorsFromRenderer(renderer):
    """Remove every isodose legend box from a renderer. The Taranis absolute and patient-relative modules
    share the results layout, so a legend created by the other module must not stay on top of this one."""
    props = renderer.GetViewProps()
    stale = [props.GetItemAsObject(i) for i in range(props.GetNumberOfItems())]
    for prop in stale:
        if prop is not None and prop.IsA("vtkLegendBoxActor"):
            renderer.RemoveViewProp(prop)


def equalizeLayoutColumns(extraRoots=()):
    """Give the views in each row of the results layout the same width (main window and, in the dual monitor
    layout, the second-screen window)."""
    roots = [slicer.app.layoutManager().viewport()] + list(extraRoots)
    for root in roots:
        for splitter in slicer.util.findChildren(widget=root, className="*Splitter"):
            if splitter.orientation == qt.Qt.Horizontal and splitter.count() >= 2:
                splitter.setSizes([1000] * splitter.count())


def dualMonitorLayoutSupported():
    """Layouts with a separate view window (<viewports>) are available since Slicer 5.0."""
    try:
        return int(slicer.app.majorVersion) >= 5
    except Exception:
        return False


def screenCount():
    """Number of connected screens (1 if it cannot be determined)."""
    try:
        return max(1, len(qt.QGuiApplication.screens()))
    except Exception:
        pass
    try:
        return max(1, int(slicer.app.desktop().screenCount))
    except Exception:
        return 1


def secondaryScreen():
    """A screen other than the one showing the Slicer main window, or None with a single screen."""
    try:
        screens = list(qt.QGuiApplication.screens())
    except Exception:
        return None
    if len(screens) < 2:
        return None
    mainScreen = None
    mainWindow = slicer.util.mainWindow()
    try:
        mainScreen = qt.QGuiApplication.screenAt(mainWindow.geometry.center())
    except Exception:
        pass
    if mainScreen is None:
        try:
            mainScreen = mainWindow.windowHandle().screen()
        except Exception:
            pass
    if mainScreen is None:
        return screens[1]
    for screen in screens:
        if screen.name != mainScreen.name:
            return screen
    return None


def findColorNode(name):
    """Colour node by name (case-insensitive), e.g. 'Inferno'. None if not available."""
    node = slicer.mrmlScene.GetFirstNodeByName(name)
    if node and node.IsA("vtkMRMLColorNode"):
        return node
    for candidate in slicer.util.getNodesByClass("vtkMRMLColorNode"):
        if (candidate.GetName() or "").lower() == name.lower():
            return candidate
    return None


def isodoseOpacity(levelIndex, levelCount):
    """3D opacity of an isodose surface: 2 lowest levels 0.05, 3 highest 0.15, the rest 0.1."""
    if levelIndex < 2:
        return 0.05
    if levelIndex >= levelCount - 3:
        return 0.15
    return 0.1


def taggedNodes(role):
    scene = slicer.mrmlScene
    nodes = []
    for index in range(scene.GetNumberOfNodes()):
        node = scene.GetNthNode(index)
        if node and node.GetAttribute(ROLE_ATTRIBUTE) == role:
            nodes.append(node)
    return nodes


def _viewIDs(displayNode):
    return [displayNode.GetNthViewNodeID(i) for i in range(displayNode.GetNumberOfViewNodeIDs())]


def excludeForeignRenderings(viewNodeIDs, keepNodes=()):
    """Keep volume renderings, models, segmentations and markups of other modules out of the given views. A display
    node without view IDs is shown in every view, so it gets the explicit list of all other views (it stays visible
    where it was). Nodes of the Taranis modules (ROLE_ATTRIBUTE) and keepNodes are not changed."""
    targets = {viewID for viewID in viewNodeIDs if viewID}
    if not targets:
        return
    keepIDs = {node.GetID() for node in keepNodes if node is not None}
    others = [node.GetID() for className in ("vtkMRMLViewNode", "vtkMRMLSliceNode")
              for node in slicer.util.getNodesByClass(className) if node.GetID() not in targets]
    for className in FOREIGN_DISPLAY_CLASSES:
        for displayNode in slicer.util.getNodesByClass(className):
            displayable = displayNode.GetDisplayableNode()
            if (displayable is None or displayable.GetHideFromEditors() or displayable.GetID() in keepIDs
                    or displayNode.GetAttribute(ROLE_ATTRIBUTE) or displayable.GetAttribute(ROLE_ATTRIBUTE)):
                continue
            current = _viewIDs(displayNode)
            if current and not targets.intersection(current):
                continue
            remaining = [viewID for viewID in (current or others) if viewID not in targets]
            wasModifying = displayNode.StartModify()
            if remaining:
                displayNode.SetViewNodeIDs(remaining)
            else:
                displayNode.SetVisibility(False)  # it was shown only in the results views
            displayNode.EndModify(wasModifying)


def migrateLegacyViewIDs(views):
    """Models of scenes saved by earlier versions are shown in Slicer's default views: move them to the results
    views."""
    mapping = {}
    for tag, key in LEGACY_THREED_TAGS.items():
        node = slicer.mrmlScene.GetSingletonNode(tag, "vtkMRMLViewNode")
        if node is not None and views.get(key):
            mapping[node.GetID()] = views[key]
    for name, role in LEGACY_SLICE_NAMES.items():
        node = slicer.mrmlScene.GetSingletonNode(name, "vtkMRMLSliceNode")
        if node is not None and views.get(role + "Slice"):
            mapping[node.GetID()] = views[role + "Slice"]
    if not mapping:
        return
    for displayNode in slicer.util.getNodesByClass("vtkMRMLDisplayNode"):
        displayable = displayNode.GetDisplayableNode() if hasattr(displayNode, "GetDisplayableNode") else None
        if not (displayNode.GetAttribute(ROLE_ATTRIBUTE) or (displayable and displayable.GetAttribute(ROLE_ATTRIBUTE))):
            continue
        current = _viewIDs(displayNode)
        if not any(viewID in mapping for viewID in current):
            continue
        updated = []
        for viewID in current:
            viewID = mapping.get(viewID, viewID)
            if viewID not in updated:
                updated.append(viewID)
        displayNode.SetViewNodeIDs(updated)


def removeTaggedNodes(role):
    """Remove all nodes (and their display nodes) created by a previous run for this role."""
    scene = slicer.mrmlScene
    tagged = []
    for index in range(scene.GetNumberOfNodes()):
        node = scene.GetNthNode(index)
        if node and node.GetAttribute(ROLE_ATTRIBUTE) == role:
            tagged.append(node)
    for node in tagged:
        if not scene.IsNodePresent(node):
            continue
        displayNodes = []
        if node.IsA("vtkMRMLDisplayableNode"):
            displayNodes = [node.GetNthDisplayNode(i) for i in range(node.GetNumberOfDisplayNodes())]
        scene.RemoveNode(node)
        for displayNode in displayNodes:
            if displayNode and scene.IsNodePresent(displayNode):
                scene.RemoveNode(displayNode)


def recreateFolder(name):
    shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
    sceneItemID = shNode.GetSceneItemID()
    oldFolder = shNode.GetItemChildWithName(sceneItemID, name)
    if oldFolder:
        shNode.RemoveItem(oldFolder)
    return shNode.CreateFolderItem(sceneItemID, name)


def tagAndFile(node, role, folderItemID):
    node.SetAttribute(ROLE_ATTRIBUTE, role)
    shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
    itemID = shNode.GetItemByDataNode(node)
    if itemID and folderItemID:
        shNode.SetItemParent(itemID, folderItemID)


def addModel(name, polyData, transformNodeID, role, folderItemID):
    modelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", name)
    modelNode.SetAndObservePolyData(polyData)
    modelNode.SetAndObserveTransformNodeID(transformNodeID)
    modelNode.CreateDefaultDisplayNodes()
    modelNode.GetDisplayNode().SetAttribute(ROLE_ATTRIBUTE, role)
    tagAndFile(modelNode, role, folderItemID)
    return modelNode


def styleModelDisplay(displayNode, color, opacity, wireframe, viewNodeIDs, sliceIntersectionThickness=None):
    """Flat (ambient-only) rendering, restricted to the given views."""
    displayNode.SetColor(*color)
    displayNode.SetOpacity(opacity)
    displayNode.SetAmbient(1.0)
    displayNode.SetDiffuse(0.0)
    displayNode.SetSpecular(0.0)
    displayNode.SetRepresentation(slicer.vtkMRMLDisplayNode.WireframeRepresentation if wireframe
                                  else slicer.vtkMRMLDisplayNode.SurfaceRepresentation)
    displayNode.SetScalarVisibility(False)
    displayNode.SetBackfaceCulling(False)
    if hasattr(displayNode, "SetBackfaceColorHSVOffset"):
        displayNode.SetBackfaceColorHSVOffset(0.0, 0.0, 0.0)  # backfaces in exactly the same colour (pure white liver)
    displayNode.SetVisibility(True)
    displayNode.SetVisibility3D(True)
    if sliceIntersectionThickness:
        displayNode.SetVisibility2D(True)
        displayNode.SetSliceIntersectionThickness(sliceIntersectionThickness)
        if hasattr(displayNode, "SetSliceIntersectionOpacity"):
            displayNode.SetSliceIntersectionOpacity(1.0)
    else:
        displayNode.SetVisibility2D(False)
    # An empty view list means "all views", so always add the IDs explicitly
    displayNode.RemoveAllViewNodeIDs()
    for viewNodeID in viewNodeIDs:
        displayNode.AddViewNodeID(viewNodeID)


def isodoseSurface(doseVolumeNode, levelGy):
    """Iso-surface of the dose volume at levelGy, in the dose volume's (untransformed) RAS space."""
    image = vtk.vtkImageData()
    image.ShallowCopy(doseVolumeNode.GetImageData())
    image.SetOrigin(0, 0, 0)
    image.SetSpacing(1, 1, 1)
    e = image.GetExtent()
    # Pad with zeros so surfaces touching the image border are closed
    pad = vtk.vtkImageConstantPad()
    pad.SetInputData(image)
    pad.SetOutputWholeExtent(e[0] - 1, e[1] + 1, e[2] - 1, e[3] + 1, e[4] - 1, e[5] + 1)
    pad.SetConstant(0.0)
    contour = vtk.vtkFlyingEdges3D()
    contour.SetInputConnection(pad.GetOutputPort())
    contour.SetValue(0, levelGy)
    contour.ComputeNormalsOff()
    contour.ComputeGradientsOff()
    contour.ComputeScalarsOff()
    ijkToRas = vtk.vtkMatrix4x4()
    doseVolumeNode.GetIJKToRASMatrix(ijkToRas)
    transform = vtk.vtkTransform()
    transform.SetMatrix(ijkToRas)
    toRas = vtk.vtkTransformPolyDataFilter()
    toRas.SetTransform(transform)
    toRas.SetInputConnection(contour.GetOutputPort())
    toRas.Update()
    result = vtk.vtkPolyData()
    result.DeepCopy(toRas.GetOutput())
    return result


def segmentList(segmentationNode):
    """[(segmentID, name)] in segmentation order; empty without a segmentation."""
    if segmentationNode is None:
        return []
    segmentation = segmentationNode.GetSegmentation()
    segmentIDs = vtk.vtkStringArray()
    segmentation.GetSegmentIDs(segmentIDs)
    result = []
    for i in range(segmentIDs.GetNumberOfValues()):
        segmentID = segmentIDs.GetValue(i)
        result.append((segmentID, segmentation.GetSegment(segmentID).GetName()))
    return result


def loadSegmentCategories(segmentationNode):
    """{segmentID: category} stored on the segmentation node (unknown segments/categories dropped)."""
    if segmentationNode is None:
        return {}
    try:
        stored = json.loads(segmentationNode.GetAttribute(CATEGORY_ATTRIBUTE) or "{}")
    except ValueError:
        stored = {}
    existing = {segmentID for segmentID, _ in segmentList(segmentationNode)}
    valid = {key for key, _ in SEGMENT_CATEGORIES}
    return {sid: cat for sid, cat in stored.items() if sid in existing and cat in valid}


def saveSegmentCategories(segmentationNode, categories):
    if segmentationNode is None:
        return
    existing = {segmentID for segmentID, _ in segmentList(segmentationNode)}
    segmentationNode.SetAttribute(CATEGORY_ATTRIBUTE, json.dumps(
        {sid: cat for sid, cat in sorted(categories.items()) if sid in existing}))


def segmentRole(segmentID, liverSegmentID, perfusedIDs, categories):
    """'liver', 'perfused', a category key, or 'uncategorized' (liver and perfused volumes take precedence)."""
    if segmentID == liverSegmentID:
        return "liver"
    if segmentID in perfusedIDs:
        return "perfused"
    return categories.get(segmentID, "uncategorized")


def applySegmentColors(segmentationNode, liverSegmentID, perfusedIDs, categories, recolorUncategorized=True):
    """Standard colours: whole liver white, perfused volumes bright red, tumors pink, normal tissue
    turquoise, others and uncategorized segments gray. recolorUncategorized=False (live preview) keeps the
    colours of segments that have no role yet."""
    segmentation = segmentationNode.GetSegmentation()
    for segmentID, _ in segmentList(segmentationNode):
        role = segmentRole(segmentID, liverSegmentID, perfusedIDs, categories)
        if role == CATEGORY_IGNORED or (role == "uncategorized" and not recolorUncategorized):
            continue  # ignored segments (e.g. the lungs) keep their colour
        rgb = RESULT_ROLES[role][2]
        segment = segmentation.GetSegment(segmentID)
        if not np.allclose(segment.GetColor(), rgb, atol=1e-3):  # avoid needless SegmentModified events
            segment.SetColor(*rgb)


def sortResults(results):
    """Result rows ordered by role; segmentation order is kept within a role."""
    return sorted(results, key=lambda s: RESULT_ROLES[s["role"]][1])


# -- Dose checks (thresholds and texts in TaranisLib.doseguard) -------------------------------------------------

SEGMENT_ROLE_TAG = "Taranis.SegmentRole"   # role given by the Taranis workflow (e.g. perfused volumes)


def segmentTagValue(segmentationNode, segmentID, tag):
    """Value of a segment tag, or ""."""
    segment = segmentationNode.GetSegmentation().GetSegment(segmentID) if segmentationNode else None
    if segment is None:
        return ""
    factory = getattr(vtk, "reference", None) or getattr(vtk, "mutable")
    value = factory("")
    try:
        return str(value) if segment.GetTag(tag, value) else ""
    except TypeError:
        return ""


def normalScope(segmentationNode, segmentID, name=""):
    """Perfused or whole normal liver: the tag set by the Taranis tools, otherwise guessed from the name."""
    scope = segmentTagValue(segmentationNode, segmentID, _W.NORMAL_SCOPE_TAG)
    if scope:
        return scope
    return _W.NORMAL_SCOPE_PERFUSED if "perfus" in (name or "").lower() else _W.NORMAL_SCOPE_WHOLE


def taggedPerfusedIDs(segmentationNode):
    """Segments marked as perfused volumes by the Taranis workflow."""
    return [segmentID for segmentID, _ in segmentList(segmentationNode)
            if segmentTagValue(segmentationNode, segmentID, SEGMENT_ROLE_TAG) == _W.SEGMENT_PERFUSED]


def doseCheckSegments(stats, segmentationNode):
    """The individual segments of a calculation (not the combined rows), as doseguard.doseChecks expects them."""
    segments = []
    for s in stats:
        if not s.get("id"):
            continue
        entry = {"name": s["name"], "role": s["role"], "dose": float(s["dose"]), "volume": float(s["volume"])}
        if s["role"] == CATEGORY_NORMAL:
            entry["scope"] = normalScope(segmentationNode, s["id"], s["name"])
        segments.append(entry)
    return segments


def doseCheckLines(checks, severities=(_W.SEVERITY_WARNING,)):
    """Report / note lines of the dose checks."""
    prefix = {_W.SEVERITY_ERROR: "ERROR", _W.SEVERITY_WARNING: "CHECK", _W.SEVERITY_INFO: "Note"}
    return [f"{prefix.get(severity, 'Note')} - {text}" for severity, text in checks if severity in severities]


DVH_LIVER_COLOR = (0.0, 0.6, 0.0)  # green: the white whole-liver colour is invisible on the plot
DVH_LINE_STYLES = ["Solid", "Dash", "DashDot", "DashDotDot"]  # no dotted line: too hard to see


def shadeColor(rgb, step):
    """Shade number `step` of a colour: 0 = the colour itself, then alternately darker and lighter,
    a little more each time (20 % steps, at most 60 %)."""
    rgb = tuple(float(c) for c in rgb)
    if step <= 0:
        return rgb
    amount = min(0.2 * ((step + 1) // 2), 0.6)
    if step % 2:
        return tuple(c * (1.0 - amount) for c in rgb)            # darker
    return tuple(c + (1.0 - c) * amount for c in rgb)             # lighter


def dvhCurveStyles(colors, emphasized):
    """(line style, width, rgb) per DVH curve. Curves sharing a colour (e.g. several tumors) get different
    patterns: solid, dash, dash-dot, dash-dot-dot; from the 5th curve of a colour on, the patterns repeat in
    slightly darker / lighter shades of that colour. Emphasized curves (combined tumors) are drawn thicker."""
    used, styles = {}, []
    for rgb, emphasize in zip(colors, emphasized):
        key = tuple(round(float(c), 3) for c in rgb)
        index = used.get(key, 0)
        used[key] = index + 1
        styles.append((DVH_LINE_STYLES[index % len(DVH_LINE_STYLES)], 3.5 if emphasize else 2.0,
                       shadeColor(rgb, index // len(DVH_LINE_STYLES))))
    return styles


class SegmentComboBox:
    """Segment picker for the master segmentation (the 'Master Segmentation' selector at the top):
    only the segment is chosen here. Item data = segment ID."""
    NONE_TEXT = "(select a segment)"

    def __init__(self, toolTip="", onChanged=None):
        self.segmentationNode = None
        self.onChanged = onChanged
        self.widget = qt.QComboBox()
        self.widget.setToolTip(toolTip)
        self.widget.connect("currentIndexChanged(int)", self._onIndexChanged)
        self.refresh()

    def setSegmentation(self, segmentationNode):
        self.segmentationNode = segmentationNode
        self.refresh()

    def currentNode(self):
        return self.segmentationNode

    def currentSegmentID(self):
        index = self.widget.currentIndex
        if index <= 0:
            return ""
        return self.widget.itemData(index) or ""

    def setCurrentSegmentID(self, segmentID):
        """Select a segment by ID ('' or an ID that does not exist selects nothing). True if it was found."""
        index = self.widget.findData(segmentID) if segmentID else 0
        self.widget.setCurrentIndex(max(index, 0))
        return index >= 0

    def refresh(self):
        """Refill from the segmentation, keeping the selected segment if it still exists."""
        previous = self.currentSegmentID()
        self.widget.blockSignals(True)
        self.widget.clear()
        self.widget.addItem(self.NONE_TEXT, "")
        for segmentID, name in segmentList(self.segmentationNode):
            self.widget.addItem(name, segmentID)
        index = self.widget.findData(previous) if previous else 0
        self.widget.setCurrentIndex(max(index, 0))
        self.widget.blockSignals(False)
        if self.currentSegmentID() != previous:
            self._onIndexChanged()

    def _onIndexChanged(self, *args):
        if self.onChanged:
            self.onChanged()


class SegmentCategorizer:
    """Uncategorized segments on the left; Tumors / Normal tissue / Others on the right.
    Selected segments are moved with the > and < buttons next to each category."""

    def __init__(self, onChanged=None, categories=None):
        """categories: [(key, box title)], default SEGMENT_CATEGORIES (the Relative module passes its own titles)."""
        categories = categories or SEGMENT_CATEGORIES
        self.onChanged = onChanged
        self.categories = {}  # segmentID -> category (also for segments currently not shown)
        self.available = []   # [(segmentID, name)] currently shown
        self.widget = qt.QWidget()
        grid = qt.QGridLayout(self.widget)
        grid.setContentsMargins(0, 0, 0, 0)

        leftBox = qt.QGroupBox("Uncategorized")
        leftLayout = qt.QVBoxLayout(leftBox)
        self.uncategorizedList = self._makeList()
        leftLayout.addWidget(self.uncategorizedList)
        grid.addWidget(leftBox, 0, 0, len(categories), 1)

        self.categoryLists = {}
        for row, (key, title) in enumerate(categories):
            buttonLayout = qt.QVBoxLayout()
            toButton = qt.QPushButton(">")
            fromButton = qt.QPushButton("<")
            toButton.setToolTip(f"Move the selected uncategorized segments to {title.lower()}.")
            fromButton.setToolTip(f"Move the selected {title.lower()} segments back to uncategorized.")
            for button in (toButton, fromButton):
                button.setFixedWidth(32)
            toButton.connect("clicked()", lambda *_, k=key: self._move(k, True))
            fromButton.connect("clicked()", lambda *_, k=key: self._move(k, False))
            buttonLayout.addStretch(1)
            buttonLayout.addWidget(toButton)
            buttonLayout.addWidget(fromButton)
            buttonLayout.addStretch(1)
            grid.addLayout(buttonLayout, row, 1)

            box = qt.QGroupBox(title)
            boxLayout = qt.QVBoxLayout(box)
            categoryList = self._makeList()
            categoryList.setMaximumHeight(90)
            boxLayout.addWidget(categoryList)
            grid.addWidget(box, row, 2)
            self.categoryLists[key] = categoryList
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(2, 1)

    @staticmethod
    def _makeList():
        listWidget = qt.QListWidget()
        listWidget.setSelectionMode(qt.QAbstractItemView.ExtendedSelection)
        listWidget.setMinimumHeight(60)
        return listWidget

    def setSegments(self, available):
        """available: [(segmentID, name)] to show (whole liver / perfused volumes excluded by the caller)."""
        self.available = list(available)
        self._rebuild()

    def shownCategories(self):
        """{segmentID: category} for the segments currently shown."""
        shown = {segmentID for segmentID, _ in self.available}
        return {sid: cat for sid, cat in self.categories.items() if sid in shown}

    def _rebuild(self):
        lists = [self.uncategorizedList] + list(self.categoryLists.values())
        for listWidget in lists:
            listWidget.clear()
        for segmentID, name in self.available:
            category = self.categories.get(segmentID)
            target = self.categoryLists.get(category, self.uncategorizedList)
            item = qt.QListWidgetItem(name)
            item.setData(qt.Qt.UserRole, segmentID)
            target.addItem(item)

    def _move(self, category, toCategory):
        source = self.uncategorizedList if toCategory else self.categoryLists[category]
        segmentIDs = [item.data(qt.Qt.UserRole) for item in source.selectedItems()]
        if not segmentIDs:
            return
        for segmentID in segmentIDs:
            if toCategory:
                self.categories[segmentID] = category
            else:
                self.categories.pop(segmentID, None)
        self._rebuild()
        if self.onChanged:
            self.onChanged()


def qImageToPngBytes(image):
    """PNG file content of a QImage (via a temporary file), or None if it cannot be written."""
    import tempfile
    handle, path = tempfile.mkstemp(suffix=".png")
    os.close(handle)
    try:
        if image is None or image.isNull() or not image.save(path, "PNG"):
            return None
        with open(path, "rb") as f:
            return f.read()
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def segmentSurfacesInWorld(segmentationNode, segmentIDsWanted=None):
    """{segmentID: closed-surface vtkPolyData in world (RAS) coordinates, parent transforms applied}, for all segments
    or those in segmentIDsWanted. Without a transform the segmentation's own surfaces are returned (read-only, not
    copied: a copy of every surface doubled their memory)."""
    segmentationNode.CreateClosedSurfaceRepresentation()
    surfaceName = slicer.vtkSegmentationConverter.GetClosedSurfaceRepresentationName()
    segmentation = segmentationNode.GetSegmentation()
    toWorld = None
    transformNode = segmentationNode.GetParentTransformNode()
    if transformNode:
        toWorld = vtk.vtkGeneralTransform()
        transformNode.GetTransformToWorld(toWorld)
    surfaces = {}
    segmentIDs = vtk.vtkStringArray()
    segmentationNode.GetSegmentation().GetSegmentIDs(segmentIDs)
    for i in range(segmentIDs.GetNumberOfValues()):
        segmentID = segmentIDs.GetValue(i)
        if segmentIDsWanted is not None and segmentID not in segmentIDsWanted:
            continue
        polyData = segmentation.GetSegment(segmentID).GetRepresentation(surfaceName)
        if polyData is None or polyData.GetNumberOfPoints() == 0:
            continue
        if toWorld:
            transformFilter = vtk.vtkTransformPolyDataFilter()
            transformFilter.SetTransform(toWorld)
            transformFilter.SetInputData(polyData)
            transformFilter.Update()
            polyData = vtk.vtkPolyData()
            polyData.DeepCopy(transformFilter.GetOutput())
        surfaces[segmentID] = polyData
    return surfaces


def stackLabelCentres(labels, bottom, top, spacing):
    """Give each label a vertical centre 'y' as close as possible to its segment ('anchorY'), inside
    [bottom, top] and without overlap. labels: dicts with 'anchorY' and 'h' (height); sorted top-down in place."""
    labels.sort(key=lambda label: -label["anchorY"])
    limit = top
    for label in labels:                    # top-down: push labels down below their upper neighbour
        label["y"] = min(label["anchorY"], limit - label["h"] / 2.0)
        limit = label["y"] - label["h"] / 2.0 - spacing
    limit = bottom
    for label in reversed(labels):          # bottom-up: push labels back up above the bottom limit
        label["y"] = max(label["y"], limit + label["h"] / 2.0)
        limit = label["y"] + label["h"] / 2.0 + spacing
    return labels


class SliceSegmentAnnotations:
    """Segment name labels with leader lines in slice views.

    In each slice view the labels are stacked in a column at the left or right edge (the side nearer to
    the segment), and a line with a dot runs from each label to the closest point of the segment's outline
    on the current slice. Segments not cut by the current slice are not labelled. Positions are recomputed
    (throttled) whenever a view is scrolled, zoomed, panned or resized.
    Slice view renderers work in XY coordinates (= viewport pixels), so all actors use viewport coordinates.
    """
    FONT_SIZE = 13
    MARGIN_PX = 10
    SPACING_PX = 14
    DOT_RADIUS_PX = 2.5
    UPDATE_INTERVAL_MS = 40

    def __init__(self, entries, sliceNames):
        """entries: [(label text, rgb, closed surface vtkPolyData in world coordinates)]."""
        self.visible = True
        self.reservedRightBottom = {}  # slice view name -> fraction of the height kept free at the bottom right
        self.segments = []
        for text, rgb, polyData in entries:
            plane = vtk.vtkPlane()
            cutter = vtk.vtkCutter()
            cutter.SetCutFunction(plane)
            cutter.SetInputData(polyData)
            self.segments.append({"text": text, "rgb": rgb, "plane": plane, "cutter": cutter})
        self.cutCache = {}   # (segment index, slice plane) -> outline points (RAS) or None
        self.views = []
        self.observations = []
        layoutManager = slicer.app.layoutManager()
        for sliceName in sliceNames:
            sliceWidget = layoutManager.sliceWidget(sliceName)
            if sliceWidget is None:
                continue
            sliceView = sliceWidget.sliceView()
            renderer = sliceView.renderWindow().GetRenderers().GetFirstRenderer()
            sliceNode = sliceWidget.mrmlSliceNode()
            items = [self._createItem(renderer, segment) for segment in self.segments]
            self.views.append({"name": sliceName, "sliceNode": sliceNode, "sliceView": sliceView,
                               "renderer": renderer, "items": items})
            self.observations.append((sliceNode, sliceNode.AddObserver(vtk.vtkCommand.ModifiedEvent,
                                                                       self._onViewChanged)))
        self.observations.append((slicer.mrmlScene, slicer.mrmlScene.AddObserver(slicer.mrmlScene.EndCloseEvent,
                                                                                 self._onSceneClosed)))
        self.timer = qt.QTimer()
        self.timer.setSingleShot(True)
        self.timer.setInterval(self.UPDATE_INTERVAL_MS)
        self.timer.connect("timeout()", self.update)

    def _createItem(self, renderer, segment):
        textActor = vtk.vtkTextActor()
        textActor.SetInput(segment["text"])
        textProperty = textActor.GetTextProperty()
        textProperty.SetFontSize(self.FONT_SIZE)
        textProperty.BoldOn()
        textProperty.ShadowOff()
        textProperty.SetColor(*segment["rgb"])
        textProperty.SetBackgroundColor(0.0, 0.0, 0.0)
        textProperty.SetBackgroundOpacity(0.6)
        textProperty.SetJustificationToLeft()
        textProperty.SetVerticalJustificationToBottom()
        textActor.GetPositionCoordinate().SetCoordinateSystemToViewport()
        textActor.SetVisibility(False)

        leaderPolyData = vtk.vtkPolyData()
        coordinate = vtk.vtkCoordinate()
        coordinate.SetCoordinateSystemToViewport()
        mapper = vtk.vtkPolyDataMapper2D()
        mapper.SetInputData(leaderPolyData)
        mapper.SetTransformCoordinate(coordinate)
        leaderActor = vtk.vtkActor2D()
        leaderActor.SetMapper(mapper)
        leaderActor.GetProperty().SetColor(*segment["rgb"])
        leaderActor.GetProperty().SetLineWidth(1.5)
        leaderActor.SetVisibility(False)

        renderer.AddViewProp(leaderActor)
        renderer.AddViewProp(textActor)  # added last: drawn above the leader lines
        return {"text": textActor, "leader": leaderActor, "polyData": leaderPolyData}

    # -- public --------------------------------------------------------------

    def setVisible(self, visible):
        self.visible = bool(visible)
        self.update()

    def scheduleUpdate(self):
        if not self.timer.isActive():  # throttle: at most one update per interval while scrolling
            self.timer.start()

    def update(self):
        for view in self.views:
            try:
                self._updateView(view)
            except Exception as e:  # labels are cosmetic; never break the views for them
                logging.warning(f"Could not update the segment labels in the {view['name']} view: {e}")
            view["sliceView"].scheduleRender()

    def clear(self):
        """Remove all labels and observers."""
        try:
            self.timer.stop()
        except Exception:
            pass
        for observedObject, tag in self.observations:
            try:
                observedObject.RemoveObserver(tag)
            except Exception:
                pass
        self.observations = []
        for view in self.views:
            for item in view["items"]:
                try:
                    view["renderer"].RemoveViewProp(item["text"])
                    view["renderer"].RemoveViewProp(item["leader"])
                except Exception:
                    pass  # view was destroyed
            try:
                view["sliceView"].scheduleRender()
            except Exception:
                pass
        self.views = []
        self.segments = []
        self.cutCache = {}

    # -- internals -----------------------------------------------------------

    def _onViewChanged(self, caller, event):
        self.scheduleUpdate()

    def _onSceneClosed(self, caller, event):
        self.clear()

    def _outline(self, index, origin, normal):
        """Outline points (N x 3, RAS) of segment `index` on the plane, or None if the plane misses it."""
        key = (index,) + tuple(np.round(origin, 3)) + tuple(np.round(normal, 5))
        if key in self.cutCache:
            return self.cutCache[key]
        if len(self.cutCache) > 512:
            self.cutCache.clear()
        segment = self.segments[index]
        segment["plane"].SetOrigin(*origin)
        segment["plane"].SetNormal(*normal)
        segment["cutter"].Update()
        points = segment["cutter"].GetOutput().GetPoints()
        outline = None
        if points is not None and points.GetNumberOfPoints() > 0:
            from vtk.util.numpy_support import vtk_to_numpy
            outline = np.array(vtk_to_numpy(points.GetData()), dtype=np.float64)
        self.cutCache[key] = outline
        return outline

    def _updateView(self, view):
        for item in view["items"]:
            item["text"].SetVisibility(False)
            item["leader"].SetVisibility(False)
        if not self.visible or not self.segments:
            return
        sliceNode = view["sliceNode"]
        sliceToRas = slicer.util.arrayFromVTKMatrix(sliceNode.GetSliceToRAS())
        origin, normal = sliceToRas[:3, 3], sliceToRas[:3, 2]
        rasToXy = np.linalg.inv(slicer.util.arrayFromVTKMatrix(sliceNode.GetXYToRAS()))
        width, height = [float(d) for d in sliceNode.GetDimensions()[:2]]

        columns = {"left": [], "right": []}
        for index, (segment, item) in enumerate(zip(self.segments, view["items"])):
            outlineRas = self._outline(index, origin, normal)
            if outlineRas is None:
                continue
            xy = (outlineRas @ rasToXy[:3, :3].T + rasToXy[:3, 3])[:, :2]
            inView = (xy[:, 0] >= 0) & (xy[:, 0] <= width) & (xy[:, 1] >= 0) & (xy[:, 1] <= height)
            if not inView.any():
                continue
            xy = xy[inView]
            centre = xy.mean(axis=0)
            bbox = [0.0, 0.0, 0.0, 0.0]
            item["text"].GetBoundingBox(view["renderer"], bbox)
            textWidth, textHeight = bbox[1] - bbox[0], bbox[3] - bbox[2]
            if textWidth <= 0 or textHeight <= 0:  # not measurable yet: estimate from the font size
                lines = segment["text"].split("\n")
                textWidth = 0.62 * self.FONT_SIZE * max(len(line) for line in lines)
                textHeight = 1.4 * self.FONT_SIZE * len(lines)
            side = "left" if centre[0] < width / 2.0 else "right"
            columns[side].append({"item": item, "outline": xy, "anchorY": centre[1],
                                  "w": textWidth, "h": textHeight})

        for side, labels in columns.items():
            if not labels:
                continue
            bottom = self.MARGIN_PX
            if side == "right":
                bottom = max(bottom, self.reservedRightBottom.get(view["name"], 0.0) * height)
            stackLabelCentres(labels, bottom, height - self.MARGIN_PX, self.SPACING_PX)
            for label in labels:
                self._placeLabel(label, side, width)

    def _placeLabel(self, label, side, width):
        item = label["item"]
        x0 = self.MARGIN_PX if side == "left" else width - self.MARGIN_PX - label["w"]
        item["text"].GetPositionCoordinate().SetValue(x0, label["y"] - label["h"] / 2.0)
        # Leader starts at the inner edge of the label and ends at the nearest outline point
        start = np.array([x0 + label["w"] + 3.0 if side == "left" else x0 - 3.0, label["y"]])
        outline = label["outline"]
        end = outline[int(np.argmin(((outline - start) ** 2).sum(axis=1)))]

        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        polys = vtk.vtkCellArray()
        points.InsertNextPoint(start[0], start[1], 0.0)
        points.InsertNextPoint(end[0], end[1], 0.0)
        lines.InsertNextCell(2)
        lines.InsertCellPoint(0)
        lines.InsertCellPoint(1)
        dotPoints = 12
        polys.InsertNextCell(dotPoints)
        for k in range(dotPoints):  # small filled dot at the segment end of the leader
            angle = 2.0 * math.pi * k / dotPoints
            points.InsertNextPoint(end[0] + self.DOT_RADIUS_PX * math.cos(angle),
                                   end[1] + self.DOT_RADIUS_PX * math.sin(angle), 0.0)
            polys.InsertCellPoint(2 + k)
        polyData = item["polyData"]
        polyData.SetPoints(points)
        polyData.SetLines(lines)
        polyData.SetPolys(polys)
        polyData.Modified()
        item["text"].SetVisibility(True)
        item["leader"].SetVisibility(True)


def secondaryViewWindow():
    """The separate window of the dual monitor layout (the window holding 3D view 2), or None."""
    layoutManager = slicer.app.layoutManager()
    viewNode = slicer.mrmlScene.GetSingletonNode(ISODOSE_VIEW_TAG, "vtkMRMLViewNode")
    if layoutManager is None or viewNode is None:
        return None
    for index in range(layoutManager.threeDViewCount):
        threeDWidget = layoutManager.threeDWidget(index)
        if threeDWidget.mrmlViewNode().GetID() == viewNode.GetID():
            window = threeDWidget.window()
            if window is not None and not window.inherits("qSlicerMainWindow"):
                return window
    return None


def placeSecondaryViewWindow():
    """Give the separate view window of the dual monitor layout normal window buttons (minimize, maximize,
    close) and show it maximized on the second screen."""
    window = secondaryViewWindow()
    if window is None:
        return
    placeWindowOnSecondaryScreen(window)


def placeWindowOnSecondaryScreen(window):
    """A separate view window of a dual monitor layout: normal window buttons, maximized on the second screen
    (also used by the Taranis Segmentation step)."""
    if window.inherits("QDockWidget") and not getattr(window, "floating", True):
        return  # docked into the main window by the user
    # A floating view window is a tool window (close button only); make it a normal window
    window.setWindowFlags(qt.Qt.Window | qt.Qt.CustomizeWindowHint | qt.Qt.WindowTitleHint
                          | qt.Qt.WindowSystemMenuHint | qt.Qt.WindowMinMaxButtonsHint | qt.Qt.WindowCloseButtonHint)
    screen = secondaryScreen()
    if screen is None:
        window.show()  # single screen (dual layout chosen manually): do not cover the main window
        return
    window.setGeometry(screen.availableGeometry)  # maximize on the screen the window is on
    window.showMaximized()


def placeSecondaryViewWindowAfterLoad(*args):
    """After a scene is loaded in the dual monitor layout: move its view window to the second screen, maximized."""
    key = "_taranisPlaceWindowPending"
    if getattr(slicer, key, False):
        return  # already scheduled by the other Taranis module

    def place():
        setattr(slicer, key, False)
        layoutManager = slicer.app.layoutManager()
        if layoutManager is None or layoutManager.layout != RESULTS_DUAL_LAYOUT_ID:
            return
        try:
            placeSecondaryViewWindow()
        except Exception as e:
            logging.warning(f"Could not move the view window to the second screen: {e}")

    setattr(slicer, key, True)
    qt.QTimer.singleShot(500, place)  # after the views of the loaded layout have been created


def mainViewportAttributes():
    """XML attributes of the main-window viewport, taken from Slicer's own dual monitor layouts, so that the
    left column of RESULTS_DUAL_LAYOUT_XML is shown in the main window (and not in a separate window)."""
    import xml.etree.ElementTree as ElementTree
    try:
        layoutNode = slicer.app.layoutManager().layoutLogic().GetLayoutNode()
        for attributeName in dir(slicer.vtkMRMLLayoutNode):
            if "DualMonitor" not in attributeName:
                continue
            layoutID = getattr(slicer.vtkMRMLLayoutNode, attributeName)
            if not isinstance(layoutID, int) or not layoutNode.IsLayoutDescription(layoutID):
                continue
            root = ElementTree.fromstring(layoutNode.GetLayoutDescription(layoutID))
            if root.tag != "viewports":
                continue
            for element in root.findall("layout"):
                if (element.get("dockable") or "").lower() != "true":
                    name = element.get("name")
                    return f' name="{name}"' if name is not None else ""
    except Exception as e:
        logging.debug(f"Could not read Slicer's dual monitor layouts: {e}")
    return ""  # no name: the main window viewport


def resultsDualLayoutXml():
    return RESULTS_DUAL_LAYOUT_XML.replace(MAIN_VIEWPORT_MARKER, mainViewportAttributes())


def registerResultsLayout(*args):
    """Add the results layout (RESULTS_LAYOUT_ID) to the layout node.

    A scene saved while the results layout is shown stores only the layout ID. If that ID is not registered
    when the scene is loaded, Slicer reports "Can't find layout:50101" and the saved views cannot be shown.
    The layout is therefore registered when Slicer starts and again before every scene import (both Taranis
    modules register the same description)."""
    try:
        layoutManager = slicer.app.layoutManager()
    except Exception:
        layoutManager = None
    if layoutManager is None:
        return  # no main window (e.g. batch mode)
    layoutNode = layoutManager.layoutLogic().GetLayoutNode()
    if layoutNode is None:
        return
    layouts = [(RESULTS_LAYOUT_ID, RESULTS_LAYOUT_XML)]
    if dualMonitorLayoutSupported():
        layouts.append((RESULTS_DUAL_LAYOUT_ID, resultsDualLayoutXml()))
    for layoutID, layoutXML in layouts:
        if not layoutNode.IsLayoutDescription(layoutID):
            layoutNode.AddLayoutDescription(layoutID, layoutXML)
        elif layoutNode.GetLayoutDescription(layoutID) != layoutXML:
            layoutNode.SetLayoutDescription(layoutID, layoutXML)


class DosimetryWidgetBase(ScriptedLoadableModuleWidget):
    """Widget methods identical in both dosimetry modules; each module's widget derives from it."""

    def onReload(self):
        """Developer reload: reload this shared code too, then the module (Slicer reloads only the module file)."""
        import importlib
        import sys
        try:
            importlib.reload(DG)
            importlib.reload(sys.modules[__name__])
        except Exception as e:
            logging.warning(f"Could not reload the shared dosimetry code: {e}")
        ScriptedLoadableModuleWidget.onReload(self)

    # -- values filled in by the Taranis workflow -----------------------------------------------------------

    CASE_NOTICE_COLOR = "#d98a00"

    def showCaseNotice(self, html, fields=()):
        """Box at the top of the module listing what the Taranis workflow filled in, for the user to review.
        fields: [(widget, label or None)] of values that must be checked: the label (by default the widget's form
        label) is highlighted until the value is changed or the box is confirmed with "Checked"."""
        if getattr(self, "_caseNoticeFrame", None) is None:
            frame = qt.QFrame()
            frame.objectName = "taranisCaseNotice"
            frame.setStyleSheet("QFrame#taranisCaseNotice { background-color: rgba(217, 138, 0, 0.16); "
                                f"border: 1px solid {self.CASE_NOTICE_COLOR}; border-radius: 4px; }}")
            row = qt.QHBoxLayout(frame)
            row.setContentsMargins(8, 6, 8, 6)
            self._caseNoticeLabel = qt.QLabel()
            self._caseNoticeLabel.setWordWrap(True)
            self._caseNoticeLabel.setTextFormat(qt.Qt.RichText)
            self._caseNoticeLabel.setStyleSheet("background: transparent; border: none;")
            row.addWidget(self._caseNoticeLabel, 1)
            button = qt.QPushButton("Checked")
            button.setToolTip("I have reviewed the values filled in from the case: hide this box.")
            button.connect("clicked()", self.clearCaseNotice)
            row.addWidget(button, 0, qt.Qt.AlignTop)
            self.layout.insertWidget(0, frame)
            self._caseNoticeFrame = frame
            self._caseHighlights = []
            try:
                self._caseNoticeObserver = slicer.mrmlScene.AddObserver(
                    slicer.mrmlScene.EndCloseEvent, self._onSceneClosedCaseNotice)
            except Exception:
                self._caseNoticeObserver = None
        self._clearCaseHighlights()
        self._caseNoticeLabel.text = html
        self._caseNoticeFrame.show()
        for widget, label in fields:
            self._highlightCaseField(widget, label)

    def _onSceneClosedCaseNotice(self, caller=None, event=None):
        try:
            self.clearCaseNotice()   # the values belonged to the closed case
        except Exception:
            pass  # widget already deleted (module reloaded)

    def clearCaseNotice(self):
        self._clearCaseHighlights()
        frame = getattr(self, "_caseNoticeFrame", None)
        if frame is not None:
            frame.hide()

    @staticmethod
    def _formLabel(widget):
        try:
            layout = widget.parentWidget().layout()
            return layout.labelForField(widget) if hasattr(layout, "labelForField") else None
        except Exception:
            return None

    def _highlightCaseField(self, widget, label=None):
        label = label or self._formLabel(widget)
        if label is None:
            return
        entry = {"label": label, "style": label.styleSheet, "toolTip": label.toolTip, "widget": widget,
                 "callback": None}
        label.setStyleSheet(f"color: {self.CASE_NOTICE_COLOR}; font-weight: bold;")
        label.setToolTip("Filled in from the Taranis case: check this value (see the box at the top).")
        callback = lambda *args, entry=entry: self._unhighlightCaseField(entry)
        try:
            widget.connect("valueChanged(double)", callback)
            entry["callback"] = callback
        except Exception:
            pass
        self._caseHighlights.append(entry)

    def _unhighlightCaseField(self, entry):
        try:
            entry["label"].setStyleSheet(entry["style"])
            entry["label"].setToolTip(entry["toolTip"])
            if entry["callback"] is not None:
                entry["widget"].disconnect("valueChanged(double)", entry["callback"])
                entry["callback"] = None
        except Exception as e:
            logging.debug(f"Could not restore a highlighted label: {e}")
        if entry in getattr(self, "_caseHighlights", []):
            self._caseHighlights.remove(entry)

    def _clearCaseHighlights(self):
        for entry in list(getattr(self, "_caseHighlights", [])):
            self._unhighlightCaseField(entry)

    # Settings saved in the module's parameter node (in restore order): node selectors, and
    # (parameter name, widget attribute, kind) for plain widgets - kind: bool, number, text (combo box), lineedit
    NODE_SETTINGS = [("InputVolume", "spectSelector"), ("ReferenceVolume", "referenceSelector"),
                     ("Segmentation", "segmentationSelector"), ("OutputVolume", "outputVolumeSelector")]

    def _makeNodeSelector(self, nodeType, toolTip, allowCreate=False):
        selector = slicer.qMRMLNodeComboBox()
        selector.nodeTypes = [nodeType]
        selector.selectNodeUponCreation = True
        selector.addEnabled = allowCreate
        selector.removeEnabled = allowCreate
        selector.noneEnabled = allowCreate
        selector.showHidden = False
        selector.showChildNodeTypes = False
        selector.setMRMLScene(slicer.mrmlScene)
        selector.setToolTip(toolTip)
        return selector

    def _makeTable(self, headers, minimumHeight):
        table = qt.QTableWidget()
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)
        table.setMinimumHeight(minimumHeight)
        return table

    def currentIsodosePreset(self):
        return ISODOSE_PRESETS[max(0, self.isodosePresetComboBox.currentIndex)]

    def _refreshCategorizer(self, *args):
        if not hasattr(self, "categorizer"):
            return  # still building the UI
        excluded = self._categoryExclusions()
        self.categorizer.setSegments([(sid, name) for sid, name in segmentList(self.segmentationSelector.currentNode())
                                      if sid not in excluded])

    def _onCategoriesChanged(self):
        saveSegmentCategories(self.segmentationSelector.currentNode(), self.categorizer.categories)
        self._schedulePreview(segments=True)

    def _onSegmentationEvent(self):
        if not getattr(self, "_suppressSegmentationEvents", False):
            self._segmentEditTimer.start()

    def cleanup(self):
        self._observeSegmentation(None)
        for tag in self._sceneObservations:
            slicer.mrmlScene.RemoveObserver(tag)
        self._sceneObservations = []
        self._previewTimer.stop()
        for signal in ("screenAdded(QScreen*)", "screenRemoved(QScreen*)"):
            try:
                slicer.app.disconnect(signal, self._onScreensChanged)
            except Exception:
                pass

    def enter(self):
        """Opening the module shows the results layout (single or dual monitor)."""
        self._entered = True
        self._writeActiveModuleFlag(True)
        self._applyLayout()
        self._isolateViews()

    def _isolateViews(self):
        """Keep the renderings of other modules (e.g. the Epona MIP) out of the results views."""
        views = self.logic.resultsViews()
        if views is None:
            return
        keep = [self.segmentationSelector.currentNode(), getattr(self, "_resultsSegmentationNode", None)]
        try:
            self.logic.isolateResultsViews(views, keep)
        except Exception as e:
            logging.warning(f"Could not isolate the results views: {e}")

    def exit(self):
        self._entered = False
        self._writeActiveModuleFlag(False)
        self._previewTimer.stop()

    def _writeActiveModuleFlag(self, active):
        """Remember in the scene whether this module is open, so that it is reopened when the scene is loaded."""
        scene = slicer.mrmlScene
        if scene.IsImporting() or scene.IsClosing():
            return
        parameterNode = self.logic.getParameterNode() if active else self.logic.existingParameterNode()
        if parameterNode is not None:
            parameterNode.SetParameter(PARAM_ACTIVE_MODULE, "true" if active else "false")

    def _layoutMode(self):
        """'single' or 'dual': chosen with the buttons, else automatic from the number of screens."""
        mode = getattr(slicer, LAYOUT_MODE_ATTRIBUTE, None)
        if mode not in ("single", "dual"):
            mode = "dual" if screenCount() >= 2 else "single"
        if mode == "dual" and not dualMonitorLayoutSupported():
            mode = "single"
        return mode

    def _updateLayoutButtons(self):
        mode = self._layoutMode()
        for button, buttonMode in ((self.singleMonitorButton, "single"), (self.dualMonitorButton, "dual")):
            button.blockSignals(True)
            button.setChecked(mode == buttonMode)
            button.blockSignals(False)

    def onLayoutModeClicked(self, mode):
        setattr(slicer, LAYOUT_MODE_ATTRIBUTE, mode)  # shared with the other Taranis module
        self._applyLayout(force=True)

    def _onScreensChanged(self, *args):
        """A screen was connected or disconnected: back to the automatic layout choice."""
        setattr(slicer, LAYOUT_MODE_ATTRIBUTE, None)
        if self._entered:
            qt.QTimer.singleShot(1000, self._applyLayout)  # let the system settle the screen geometry first
        else:
            self._updateLayoutButtons()

    def _applyLayout(self, force=False, rebuildOverlays=True):
        """Show the results layout for the current mode. After a switch the results overlays (legend, labels,
        linked cameras) are rebuilt, or, before any calculation, the selected images and segments are shown."""
        self._updateLayoutButtons()
        layoutManager = slicer.app.layoutManager()
        if layoutManager is None:
            return
        layoutID = RESULTS_DUAL_LAYOUT_ID if self._layoutMode() == "dual" else RESULTS_LAYOUT_ID
        self.logic.preferredLayoutID = layoutID
        switched = layoutManager.layout != layoutID
        if not switched and not force:
            return
        try:
            views = self.logic.setupResultsLayout(layoutID, placeSecondaryWindow=force)
        except Exception as e:
            logging.warning(f"Could not show the results layout: {e}")
            return
        if self._hasResults():
            if rebuildOverlays:
                self._rebuildResultsOverlays()
        else:
            try:
                self.logic.linkThreeDViewCameras([views["segments3D"], views["isodose3D"]])
            except Exception as e:
                logging.warning(f"Could not link the 3D view cameras: {e}")
            self._schedulePreview(slices=True, segments=True, recenter=switched)

    def _hasResults(self):
        node = self._resultsDoseNode
        return bool(self.lastResult) and node is not None and slicer.mrmlScene.IsNodePresent(node)

    def _connectPreviewSignals(self):
        self.referenceSelector.connect("currentNodeChanged(vtkMRMLNode*)",
                                       lambda *args: self._schedulePreview(slices=True, recenter=True))
        self.spectSelector.connect("currentNodeChanged(vtkMRMLNode*)", lambda *args: self._schedulePreview(
            slices=True, recenter=self.referenceSelector.currentNode() is None))
        self.segmentationSelector.connect("currentNodeChanged(vtkMRMLNode*)",
                                          lambda *args: self._schedulePreview(segments=True))
        self.liverSegmentSelector.widget.connect("currentIndexChanged(int)",
                                                 lambda *args: self._schedulePreview(segments=True))
        for signal in ("screenAdded(QScreen*)", "screenRemoved(QScreen*)"):
            try:
                slicer.app.connect(signal, self._onScreensChanged)
            except Exception as e:
                logging.debug(f"Screen changes are not detected: {e}")

    def _schedulePreview(self, slices=False, segments=False, recenter=False):
        """Show the selected images and segments in the results layout right away, before any calculation
        (no isodose lines). Changes within 400 ms are combined into one update."""
        scene = slicer.mrmlScene
        if (not getattr(self, "_entered", False) or getattr(self, "_restoring", True)
                or getattr(self, "_calculating", False) or scene.IsImporting() or scene.IsClosing()):
            return
        pending = self._pendingPreview
        pending["slices"] = pending["slices"] or slices
        pending["segments"] = pending["segments"] or segments
        pending["recenter"] = pending["recenter"] or recenter
        self._previewTimer.start()

    def _runPreview(self):
        pending = self._pendingPreview
        self._pendingPreview = {"slices": False, "segments": False, "recenter": False}
        layoutManager = slicer.app.layoutManager()
        if not self._entered or layoutManager is None or layoutManager.layout not in RESULTS_LAYOUT_IDS:
            return
        views = self.logic.resultsViews()
        if views is None:
            return
        try:
            self.logic.configureThreeDViews()  # e.g. selecting the whole liver after a scene was closed
        except Exception as e:
            logging.warning(f"Could not set up the 3D views: {e}")
        self._isolateViews()
        self._suppressSegmentationEvents = True
        try:
            if pending["slices"]:
                try:
                    self.logic.showPreviewSlices(self.referenceSelector.currentNode(), self.spectSelector.currentNode(),
                                                 self._resultsDoseNode if self._hasResults() else None,
                                                 pending["recenter"])
                except Exception as e:
                    logging.warning(f"Could not update the slice views: {e}")
            if pending["segments"]:
                segmentationNode = self.segmentationSelector.currentNode()
                liverID = self.liverSegmentSelector.currentSegmentID()
                cameraKey = (segmentationNode.GetID() if segmentationNode else None, liverID)
                slicer.app.setOverrideCursor(qt.Qt.WaitCursor)
                try:
                    self.logic.showPreviewSegments(segmentationNode, liverID, self._previewPerfusedIDs(),
                                                   dict(self.categorizer.categories), views,
                                                   resetCamera=cameraKey != self._previewCameraKey)
                    self._previewCameraKey = cameraKey
                except Exception as e:
                    logging.warning(f"Could not update the segment display: {e}")
                try:
                    self._showPreviewLabels(segmentationNode)
                except Exception as e:
                    logging.warning(f"Could not create the segment labels: {e}")
                finally:
                    slicer.app.restoreOverrideCursor()
        finally:
            self._suppressSegmentationEvents = False

    def _updateIsodoseLegend(self):
        _, levels = self.currentIsodosePreset()
        self.isodoseLegendLabel.setText(isodoseLegendHtml(levels))

    def onIsodoseSliceToggled(self, checked):
        self.isodoseSliceToggleButton.text = ("Isodose lines on slice views: ON" if checked
                                              else "Isodose lines on slice views: OFF")
        self.logic.setIsodoseInSliceViews(checked)

    def onSegmentLabelsToggled(self, checked):
        self.segmentLabelToggleButton.text = ("Segment labels on slice views: ON" if checked
                                              else "Segment labels on slice views: OFF")
        self.logic.setSegmentAnnotationsVisible(checked)

    def onCalculateButton(self):
        self._calculating = True  # no live preview while the results are built
        self._suppressSegmentationEvents = True
        try:
            self._calculate()
        finally:
            self._calculating = False
            self._suppressSegmentationEvents = False
            self._previewTimer.stop()
            self._pendingPreview = {"slices": False, "segments": False, "recenter": False}

    def _fillTable(self, table, rows):
        table.setRowCount(0)
        for rowValues in rows:
            row = table.rowCount
            table.insertRow(row)
            for col, text in enumerate(rowValues):
                if text != "":
                    item = qt.QTableWidgetItem(text)
                    item.setToolTip(text)
                    table.setItem(row, col, item)
        table.resizeColumnsToContents()

    def onCustomMetricChanged(self, index):
        if index == 0:  # D: x is a volume percentage
            self.customValueSpinBox.setRange(0.1, 100.0)
            self.customValueSpinBox.setSuffix(" %")
            self.customValueSpinBox.setValue(90.0)
        else:           # V: x is a dose in Gy
            self.customValueSpinBox.setRange(0.0, 2000.0)
            self.customValueSpinBox.setSuffix(" Gy")
            self.customValueSpinBox.setValue(100.0)

    def onCustomComputeClicked(self):
        if not self._ensureDvhData():
            return
        x = self.customValueSpinBox.value
        isD = self.customMetricComboBox.currentIndex == 0
        metric = f"D{x:g}" if isD else f"V{x:g}"
        computed = 0
        for row in range(self.customSegmentList.count):
            if self.customSegmentList.item(row).checkState() != qt.Qt.Checked:
                continue
            label, doses, voxelVolumeML = self.dvhData[row]
            if isD:
                valueText = f"{formatNumber(doseAtVolumePercent(doses, x))} Gy"
            else:
                percent = volumePercentAtDose(doses, x)
                valueText = f"{formatNumber(percent)} % ({formatNumber(percent / 100.0 * doses.size * voxelVolumeML)} mL)"
            tableRow = self.customResultTable.rowCount
            self.customResultTable.insertRow(tableRow)
            for col, text in enumerate([label, metric, valueText]):
                self.customResultTable.setItem(tableRow, col, qt.QTableWidgetItem(text))
            if self.lastResult:
                self.lastResult["sections"]["Custom DVH metrics"].append(f"{label}: {metric} = {valueText}")
            computed += 1
        self.customResultTable.resizeColumnsToContents()
        if computed == 0:
            slicer.util.errorDisplay("Tick at least one segment.")
        self._saveResults()

    def onCustomClearClicked(self):
        self.customResultTable.setRowCount(0)
        if self.lastResult:
            self.lastResult["sections"]["Custom DVH metrics"] = []
        self._saveResults()

    def _updateWindowButtons(self, *args):
        for buttons, selector in ((self.referenceWindowButtons, self.referenceSelector),
                                  (self.spectWindowButtons, self.spectSelector)):
            for button in buttons:
                button.setEnabled(selector.currentNode() is not None)

    def onReferenceWindowPreset(self, preset):
        volumeNode = self.referenceSelector.currentNode()
        if volumeNode is None:
            return
        with slicer.util.tryWithErrorDisplay("Could not set the reference window.", waitCursor=True):
            applyReferenceWindowPreset(volumeNode, preset)

    def onSpectWindowPercent(self, percent):
        volumeNode = self.spectSelector.currentNode()
        if volumeNode is None:
            return
        with slicer.util.tryWithErrorDisplay("Could not set the PET/SPECT window.", waitCursor=True):
            applySpectWindowPercent(volumeNode, percent)

    def _canSaveSettings(self):
        """Settings are written only while the user works in the GUI: not while the GUI is being filled from
        a loaded scene and not while a scene is closed or imported (selectors change on their own then)."""
        scene = slicer.mrmlScene
        return not getattr(self, "_restoring", True) and not scene.IsClosing() and not scene.IsImporting()

    def _connectSettingsSignals(self):
        for _, attributeName in self.NODE_SETTINGS:
            getattr(self, attributeName).connect("currentNodeChanged(vtkMRMLNode*)", self._saveSettings)
        self.liverSegmentSelector.widget.connect("currentIndexChanged(int)", self._saveSettings)
        for _, attributeName, kind in self.WIDGET_SETTINGS:
            widget = getattr(self, attributeName)
            if kind == "bool":
                widget.connect("toggled(bool)", self._saveSettings)
            elif kind == "text":
                widget.connect("currentIndexChanged(int)", self._saveSettings)
            elif kind == "lineedit":
                widget.connect("textChanged(QString)", self._saveSettings)
            elif isinstance(widget, qt.QSpinBox):
                widget.connect("valueChanged(int)", self._saveSettings)
            else:
                widget.connect("valueChanged(double)", self._saveSettings)

    def _saveSettings(self, *args):
        """Write all settings of the GUI into the module's parameter node."""
        if not self._canSaveSettings():
            return
        parameterNode = self.logic.getParameterNode()
        wasModifying = parameterNode.StartModify()
        try:
            parameterNode.SetParameter(PARAM_VERSION, SETTINGS_VERSION)
            for key, attributeName in self.NODE_SETTINGS:
                node = getattr(self, attributeName).currentNode()
                parameterNode.SetNodeReferenceID(key, node.GetID() if node else None)
            parameterNode.SetParameter(PARAM_LIVER_SEGMENT, self.liverSegmentSelector.currentSegmentID())
            for key, attributeName, kind in self.WIDGET_SETTINGS:
                parameterNode.SetParameter(key, widgetSettingValue(getattr(self, attributeName), kind))
            self._saveModuleSettings(parameterNode)
        finally:
            parameterNode.EndModify(wasModifying)

    def _saveResults(self):
        """Store the tables, report data and display state of the last calculation in the parameter node.
        The dose map, models and DVH chart are MRML nodes and are saved with the scene anyway."""
        if not self._canSaveSettings():
            return
        parameterNode = self.logic.getParameterNode()
        wasModifying = parameterNode.StartModify()
        try:
            if not self.lastResult:
                parameterNode.SetParameter(PARAM_RESULTS, "")
                parameterNode.SetNodeReferenceID(REF_RESULTS_DOSE, None)
                parameterNode.SetNodeReferenceID(REF_RESULTS_SEGMENTATION, None)
                return
            results = {
                "lastResult": self.lastResult,
                "tables": {"segmentDose": tableRows(self.segmentDoseTable), "d": tableRows(self.dValueTable),
                           "v": tableRows(self.vValueTable), "custom": tableRows(self.customResultTable)},
                "note": self.resultNoteLabel.text,
                "dvh": self._dvhSources,
                "voxelVolumeML": self._dvhVoxelVolumeML,
                "doseChecksum": self._doseChecksum,
                "annotations": self._annotationLabels,
                "isodoseLegend": self.logic.lastLegendLevels,
            }
            parameterNode.SetParameter(PARAM_RESULTS, json.dumps(results, default=float))
            for key, node in ((REF_RESULTS_DOSE, self._resultsDoseNode),
                              (REF_RESULTS_SEGMENTATION, self._resultsSegmentationNode)):
                present = node is not None and slicer.mrmlScene.IsNodePresent(node)
                parameterNode.SetNodeReferenceID(key, node.GetID() if present else None)
        except Exception as e:
            logging.warning(f"Could not store the results in the scene: {e}")
        finally:
            parameterNode.EndModify(wasModifying)

    def _restoreFromParameterNode(self):
        """Fill the GUI from the parameter node of the scene: settings and the last calculation's results."""
        self._restoring = True
        restoreDisplay = False
        try:
            parameterNode = self.logic.existingParameterNode()
            if parameterNode is None or not parameterNode.GetParameter(PARAM_VERSION):
                return  # nothing stored (new scene, or a scene saved by an older version of the module)
            self._clearResults()
            for key, attributeName in self.NODE_SETTINGS:
                selector = getattr(self, attributeName)
                node = parameterNode.GetNodeReference(key)
                if node is not None or selector.noneEnabled:
                    selector.setCurrentNode(node)
            self.onSegmentationNodeChanged(self.segmentationSelector.currentNode())
            self.liverSegmentSelector.setCurrentSegmentID(parameterNode.GetParameter(PARAM_LIVER_SEGMENT))
            self._restoreModuleSettings(parameterNode)
            for key, attributeName, kind in self.WIDGET_SETTINGS:
                value = parameterNode.GetParameter(key)
                if value == "":
                    continue
                try:
                    applyWidgetSetting(getattr(self, attributeName), kind, value)
                except (ValueError, TypeError) as e:
                    logging.warning(f"Could not restore setting '{key}' = '{value}': {e}")
            restoreDisplay = self._restoreResults(parameterNode)
        except Exception as e:
            logging.warning(f"Could not restore the settings saved in the scene: {e}")
        finally:
            self._restoring = False
        if restoreDisplay:
            qt.QTimer.singleShot(0, self._restoreResultsDisplay)  # after the views of the layout exist
        elif getattr(self, "_entered", False):
            qt.QTimer.singleShot(0, self._applyLayout)

    def _restoreResults(self, parameterNode):
        """Tables and report data of the last calculation. True if results were restored."""
        text = parameterNode.GetParameter(PARAM_RESULTS)
        if not text:
            return False
        results = json.loads(text)
        lastResult = results["lastResult"]
        lastResult["parameters"] = [tuple(p) for p in lastResult.get("parameters", [])]
        lastResult["rows"] = [tuple(r) for r in lastResult.get("rows", [])]
        tables = results.get("tables", {})
        self._fillTable(self.segmentDoseTable, [list(r[:4]) + [""] + list(r[4:]) if len(r) == 5 else r
                                                for r in tables.get("segmentDose", [])])
        self._fillTable(self.dValueTable, tables.get("d", []))
        self._fillTable(self.vValueTable, tables.get("v", []))
        self._fillTable(self.customResultTable, tables.get("custom", []))
        self.resultNoteLabel.setText(results.get("note", ""))
        self.lastResult = lastResult
        self._dvhSources = results.get("dvh", [])
        self._dvhVoxelVolumeML = results.get("voxelVolumeML")
        self._doseChecksum = results.get("doseChecksum")
        self._annotationLabels = [tuple(a) for a in results.get("annotations", [])]
        self._resultsDoseNode = parameterNode.GetNodeReference(REF_RESULTS_DOSE)
        self._resultsSegmentationNode = parameterNode.GetNodeReference(REF_RESULTS_SEGMENTATION)
        self.logic.lastLegendLevels = [(level, tuple(rgb)) for level, rgb in results.get("isodoseLegend", [])]
        self.customSegmentList.clear()
        for source in self._dvhSources:
            item = qt.QListWidgetItem(source["label"])
            item.setFlags(item.flags() | qt.Qt.ItemIsUserCheckable)
            item.setCheckState(qt.Qt.Checked)
            self.customSegmentList.addItem(item)
        self.customCollapsibleButton.setEnabled(bool(self._dvhSources))
        return True

    def _restoreResultsDisplay(self):
        """After a scene is loaded: if the module is open, show the results layout suited to this computer's
        screens; then rebuild the isodose legend, the slice view segment labels and the linked 3D cameras
        (they are drawn directly into the views, they are not MRML nodes)."""
        if getattr(self, "_entered", False):
            self._applyLayout(rebuildOverlays=False)
        self._rebuildResultsOverlays()

    def _rebuildResultsOverlays(self):
        if not self.lastResult:
            return
        try:
            self.lastVisualization = self.logic.restoreResultsDisplay(
                self._resultsSegmentationNode, self._resultsDoseNode, self._annotationLabels)
        except Exception as e:
            logging.warning(f"Could not restore the results display: {e}")

    def _clearResults(self):
        """Forget the last calculation: tables, report data and display state."""
        for table in (self.segmentDoseTable, self.dValueTable, self.vValueTable, self.customResultTable):
            table.setRowCount(0)
        self.resultNoteLabel.setText("")
        self.customSegmentList.clear()
        self.customCollapsibleButton.setEnabled(False)
        self.lastResult = None
        self.dvhData = []
        self._dvhSources = []
        self._dvhVoxelVolumeML = None
        self._doseChecksum = None
        self._annotationLabels = []
        self._resultsDoseNode = None
        self._resultsSegmentationNode = None
        self.lastVisualization = None
        self.logic.clearResultsOverlays()
        self._clearModuleResults()

    def _ensureDvhData(self):
        """Sorted voxel doses for the custom DVH metrics. After a scene is loaded they are recomputed once from
        the saved dose map and segmentation (checked against the values stored at calculation time)."""
        if self.dvhData:
            return True
        if not self._dvhSources:
            slicer.util.errorDisplay("Run a calculation first.")
            return False
        doseNode, segmentationNode = self._resultsDoseNode, self._resultsSegmentationNode
        scene = slicer.mrmlScene
        if (doseNode is None or segmentationNode is None or not scene.IsNodePresent(doseNode)
                or not scene.IsNodePresent(segmentationNode) or doseNode.GetImageData() is None):
            slicer.util.errorDisplay("The dose map or the segmentation of the saved calculation is not in the "
                                     "scene. Press Calculate again to compute custom DVH metrics.")
            return False
        problem = None
        data = []
        with slicer.util.tryWithErrorDisplay("Could not read the saved dose map.", waitCursor=True):
            doseArray = slicer.util.arrayFromVolume(doseNode)
            if self._doseChecksum and not np.allclose(doseChecksum(doseArray), self._doseChecksum,
                                                      rtol=1e-6, atol=1e-9):
                problem = "The dose map was changed after the calculation."
            existing = {sid for sid, _ in segmentList(segmentationNode)}
            for source in self._dvhSources:
                if problem:
                    break
                segmentIDs = source["segmentIDs"]
                if not segmentIDs or any(sid not in existing for sid in segmentIDs):
                    problem = "Segments were removed after the calculation."
                    break
                mask = None
                for segmentID in segmentIDs:  # dose map has the image grid used by the calculation
                    segmentMask = segmentMaskOnVolumeGrid(segmentationNode, segmentID, doseNode)
                    mask = segmentMask if mask is None else (mask | segmentMask)
                doses = np.sort(doseArray[mask])
                if doses.size != source["voxels"]:
                    problem = "Segments were edited after the calculation."
                    break
                data.append((source["label"], doses, self._dvhVoxelVolumeML))
        if problem:
            slicer.util.errorDisplay(problem + " Press Calculate again to compute custom DVH metrics.")
            return False
        if len(data) != len(self._dvhSources):
            return False  # error already shown
        self.dvhData = data
        return True

    def onSceneStartClose(self, caller=None, event=None):
        self._restoring = True

    def onSceneEndClose(self, caller=None, event=None):
        self._clearResults()
        self._restoring = False
        if self._entered:
            qt.QTimer.singleShot(0, self._afterSceneClosed)

    def _afterSceneClosed(self):
        """Closing a scene resets the views: set up the results layout and 3D views again (unless a scene is
        being loaded, which restores its own views)."""
        if self._restoring or slicer.mrmlScene.IsImporting() or not self._entered:
            return
        self._applyLayout()
        try:
            self.logic.configureThreeDViews()
        except Exception as e:
            logging.warning(f"Could not set up the 3D views: {e}")
        self._isolateViews()

    def onSceneStartImport(self, caller=None, event=None):
        self._restoring = True

    def onSceneEndImport(self, caller=None, event=None):
        # Node selectors update themselves after the import; fill the GUI from the loaded parameter node after
        # that (nothing is written to the parameter node until then)
        self._restoring = True
        qt.QTimer.singleShot(0, self._restoreFromParameterNode)

    def onSceneStartSave(self, caller=None, event=None):
        self._saveSettings()
        self._saveResults()
        self._saveActiveModuleFlag()

    def _saveActiveModuleFlag(self):
        """Record whether this module is the one shown, so that it is reopened when the scene is loaded."""
        self._writeActiveModuleFlag(getattr(self, "_entered", False))

    def _setReportParameter(self, label, value):
        if not self.lastResult:
            return
        params = self.lastResult["parameters"]
        for i, (existingLabel, _) in enumerate(params):
            if existingLabel == label:
                params[i] = (label, value)
                break
        else:
            params.append((label, value))
        self._saveResults()

    def onSaveReportClicked(self, fileFormat=None):
        """Save the report of the last calculation. fileFormat: "rtf" (default; the clicked(bool) argument of the
        RTF button is ignored) or "pdf"."""
        fileFormat = fileFormat if fileFormat in ("rtf", "pdf") else "rtf"
        if not self.lastResult:
            slicer.util.errorDisplay("Run a calculation before saving a report.")
            return
        if fileFormat == "pdf" and not pdfExportAvailable():
            slicer.util.errorDisplay("PDF export is not available in this Slicer version. Save the report as RTF.")
            return
        extension = "." + fileFormat
        fileName = qt.QFileDialog.getSaveFileName(None, "Save Dosimetry Report", "",
                                                  "PDF Files (*.pdf)" if fileFormat == "pdf" else "RTF Files (*.rtf)")
        if not fileName:
            return
        if not fileName.lower().endswith(extension):
            fileName += extension
        sections = [
            ("D values (minimum dose to the hottest x % of the volume)", self.lastResult["sections"]["D values"]),
            ("V values (% of the volume receiving at least x Gy)", self.lastResult["sections"]["V values"]),
            ("Custom DVH metrics", self.lastResult["sections"]["Custom DVH metrics"]),
        ]
        screenshots = []
        try:  # the report is still saved if the screenshots fail
            screenshots = self.logic.captureViewScreenshots()
        except Exception as e:
            logging.warning(f"Could not capture the view screenshots: {e}")
        patient = self.lastResult.get("patient")
        if not patient:   # results calculated before patient information was stored
            try:
                patient = reportPatientInfo(self.spectSelector.currentNode(), self.referenceSelector.currentNode())
            except Exception as e:
                logging.warning(f"Could not read the patient information: {e}")
                patient = []
        content = ([self.lastResult["title"], "Radioembolization Dosimetry Report"], self.lastResult["parameters"],
                   self.lastResult["rows"], self.lastResult["notes"], sections, screenshots,
                   [tuple(p) for p in patient])
        with slicer.util.tryWithErrorDisplay("Could not save the report."):
            if fileFormat == "pdf":
                writePdfReport(buildHtmlReport(*content), fileName)
            else:
                with open(fileName, "w", encoding="ascii", newline="") as f:  # pure ASCII after escaping
                    f.write(buildRtfReport(*content))
            self._recordSavedReport(fileName)
            slicer.util.infoDisplay(f"{fileFormat.upper()} report saved successfully.")

    def onExportTsvClicked(self):
        """Segment doses with their D and V values side by side, as a tab-separated file (last calculation)."""
        if not self.lastResult:
            slicer.util.errorDisplay("Run a calculation before exporting the results.")
            return
        fileName = qt.QFileDialog.getSaveFileName(None, "Export Results Table", "", "Tab-separated values (*.tsv)")
        if not fileName:
            return
        if not fileName.lower().endswith(".tsv"):
            fileName += ".tsv"
        sections = [("D values", self.lastResult["sections"]["D values"]),
                    ("V values", self.lastResult["sections"]["V values"])]
        with slicer.util.tryWithErrorDisplay("Could not export the results table."):
            with open(fileName, "w", encoding="utf-8-sig", newline="") as f:   # BOM: non-ASCII names open in Excel
                f.write(buildTsvTable(self.lastResult["rows"], sections))
            slicer.util.showStatusMessage(f"Results table exported to {fileName}", 5000)

    # -- dose checks ---------------------------------------------------------------------------------------

    def _publishDoseChecks(self, checks, then=None):
        """Store the dose checks with the results (the Taranis toolbar reads them from the parameter node), then,
        after the results layout is built, show the warnings in a dialog and call then()."""
        collectSoon()   # image, mask and dose arrays of the calculation
        if self.lastResult:
            self.lastResult["checks"] = [[severity, text] for severity, text in checks]
            self._saveResults()
        warnings = [text for severity, text in checks if severity in (_W.SEVERITY_ERROR, _W.SEVERITY_WARNING)]

        def show():
            if warnings:
                box = qt.QMessageBox(slicer.util.mainWindow())
                box.setIcon(qt.QMessageBox.Warning)
                box.setWindowTitle("Dose checks")
                box.setText(f"Review the results: {len(warnings)} dose check(s) need attention.")
                box.setInformativeText("\n\n".join("• " + text for text in warnings)
                                       + "\n\nThese are warnings only; they are also listed under the results "
                                         "and in the report.")
                box.addButton("I understand", qt.QMessageBox.AcceptRole)
                box.exec_()
            if then is not None:
                then()

        if warnings or then is not None:
            qt.QTimer.singleShot(0, show)

    @staticmethod
    def _withDoseChecks(notes, qcNote, checks):
        """Report notes and the note under the results, with the dose check lines."""
        lines = doseCheckLines(checks, (_W.SEVERITY_ERROR, _W.SEVERITY_WARNING, _W.SEVERITY_INFO))
        warningLines = doseCheckLines(checks)
        qcNote = "\n".join([line for line in [qcNote] if line] + warningLines)
        return list(notes) + lines, qcNote

    def _addReportButtons(self, layout):
        """Save report buttons: RTF, and PDF when Qt can write PDF files (no extra library needed)."""
        row = qt.QHBoxLayout()
        self.saveReportButton = qt.QPushButton("Save Report as RTF")
        self.saveReportButton.toolTip = "Export the dosimetry results of the last calculation to an RTF report."
        self.saveReportButton.connect("clicked(bool)", self.onSaveReportClicked)
        row.addWidget(self.saveReportButton)
        self.savePdfReportButton = None
        if pdfExportAvailable():
            self.savePdfReportButton = qt.QPushButton("Save Report as PDF")
            self.savePdfReportButton.toolTip = "Export the dosimetry results of the last calculation to a PDF report."
            self.savePdfReportButton.connect("clicked(bool)", lambda *args: self.onSaveReportClicked("pdf"))
            row.addWidget(self.savePdfReportButton)
        self.exportTsvButton = qt.QPushButton("Export Table (TSV)")
        self.exportTsvButton.toolTip = ("Export the segment doses of the last calculation with their D and V values "
                                        "side by side, one line per segment, as a tab-separated file.")
        self.exportTsvButton.connect("clicked(bool)", lambda *args: self.onExportTsvClicked())
        row.addWidget(self.exportTsvButton)
        layout.addLayout(row)

    def _recordSavedReport(self, fileName):
        """Remember which calculation the last saved report belongs to (read by the Taranis workflow)."""
        try:
            parameterNode = self.logic.getParameterNode()
            parameterNode.SetParameter(PARAM_REPORT_FILE, fileName)
            parameterNode.SetParameter(PARAM_REPORT_DOSE_CHECKSUM, str(self._doseChecksum))
        except Exception as e:
            logging.warning(f"Could not record the saved report: {e}")


class DosimetryLogicBase(ScriptedLoadableModuleLogic):
    """Logic methods identical in both dosimetry modules; each module's logic derives from it."""

    SLICE_FIELD_OF_VIEW_MM = 300.0

    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)
        self.legendActors = []  # [(view widget, renderer, actor, isSliceView)] of the isodose colour legend
        self.isodoseInSliceViews = True
        self.cameraObservers = []  # [(vtkCamera, observer tag)] used to keep the 3D views in sync
        self._syncingCameras = False
        self.segmentAnnotations = None  # SliceSegmentAnnotations of the last calculation
        self.segmentAnnotationsVisible = True
        self.segmentOutlineThickness = DEFAULT_SEGMENT_OUTLINE_THICKNESS
        self.isodoseLineThickness = DEFAULT_ISODOSE_LINE_THICKNESS
        self.outlinedSegmentationNode = None  # segmentation shown by the last calculation
        self.lastLegendLevels = []  # [(level Gy, rgb)] of the isodose legend (saved with the results)
        self.preferredLayoutID = RESULTS_LAYOUT_ID  # results layout used when none is shown (set by the GUI)

    def setupResultsLayout(self, layoutID=None, placeSecondaryWindow=False):
        """Switch to a results layout: single monitor (3x2) or dual monitor. Default: the results layout already
        shown, else the preferred one. In the dual monitor layout the separate view window is moved to the second
        screen when the layout is switched on (or placeSecondaryWindow is set).
        Returns the view node IDs used by the display functions."""
        layoutManager = slicer.app.layoutManager()
        registerResultsLayout()
        if layoutID is None:
            layoutID = layoutManager.layout if layoutManager.layout in RESULTS_LAYOUT_IDS else self.preferredLayoutID
        if layoutID == RESULTS_DUAL_LAYOUT_ID and not dualMonitorLayoutSupported():
            layoutID = RESULTS_LAYOUT_ID
        switched = layoutManager.layout != layoutID
        if switched:
            layoutManager.setLayout(layoutID)
        slicer.app.processEvents()
        if layoutID == RESULTS_DUAL_LAYOUT_ID and (switched or placeSecondaryWindow):
            try:
                self.placeSecondaryWindow()
                slicer.app.processEvents()
            except Exception as e:
                logging.warning(f"Could not move the view window to the second screen: {e}")
        try:
            equalizeLayoutColumns(self._secondaryWindowRoots())
            slicer.app.processEvents()
        except Exception as e:
            logging.warning(f"Could not equalize view column widths: {e}")

        views = self.configureThreeDViews()
        for key, tag in THREED_VIEWS:
            if key not in views:
                raise RuntimeError(f"3D view '{tag}' was not created by the layout.")

        views["slices"] = []
        for sliceName, role in SLICE_VIEWS:
            sliceWidget = layoutManager.sliceWidget(sliceName)
            sliceNode = sliceWidget.mrmlSliceNode()
            sliceNode.SetOrientation("Axial")
            views["slices"].append(sliceNode.GetID())
            views[role + "Slice"] = sliceNode.GetID()
        return views

    def isolateResultsViews(self, views, keepNodes=()):
        """Results views show only this module's renderings (and the kept nodes, e.g. the selected segmentation)."""
        migrateLegacyViewIDs(views)
        viewIDs = [views.get("segments3D"), views.get("isodose3D")] + list(views.get("slices", []))
        excludeForeignRenderings(viewIDs, list(keepNodes) + [self.outlinedSegmentationNode])

    def configureThreeDViews(self):
        """Black background, orthographic projection, no box and axis labels in the two results 3D views
        (closing a scene resets the views to Slicer's defaults). Returns {"segments3D"/"isodose3D": view ID}."""
        views = {}
        for key, tag in THREED_VIEWS:
            viewNode = slicer.mrmlScene.GetSingletonNode(tag, "vtkMRMLViewNode")
            if viewNode is None:
                continue
            viewNode.SetBackgroundColor(0.0, 0.0, 0.0)
            viewNode.SetBackgroundColor2(0.0, 0.0, 0.0)
            viewNode.SetRenderMode(slicer.vtkMRMLViewNode.Orthographic)
            viewNode.SetBoxVisible(False)
            viewNode.SetAxisLabelsVisible(False)
            views[key] = viewNode.GetID()
        return views

    def createSegmentModels(self, segmentationNode, segmentRoles, views):
        """Wireframe models in the top-left 3D view for the whole liver (white, 0.03; also shown in the isodose
        3D view), perfused volumes (0.05) and tumors (0.10), in their segment colours. Other segments get no
        3D model (they are still outlined in the slice views). segmentRoles: {segmentID: role}."""
        removeTaggedNodes(ROLE_SEGMENT_MODEL)
        folderID = recreateFolder(SEGMENT_MODEL_FOLDER)
        segmentationNode.CreateClosedSurfaceRepresentation()
        # 3D: the exported models replace the segmentation. 2D: 3 px outlines + faint (0.05) fill.
        segmentationNode.CreateDefaultDisplayNodes()
        segDisplayNode = segmentationNode.GetDisplayNode()
        segDisplayNode.SetVisibility(True)
        segDisplayNode.SetVisibility3D(False)
        segDisplayNode.SetVisibility2DFill(True)
        segDisplayNode.SetOpacity2DFill(0.05)
        segDisplayNode.SetVisibility2DOutline(True)
        segDisplayNode.SetOpacity2DOutline(1.0)
        segDisplayNode.SetSliceIntersectionThickness(self.segmentOutlineThickness)
        self.outlinedSegmentationNode = segmentationNode
        segDisplayNode.RemoveAllViewNodeIDs()
        for sliceViewID in views["slices"]:
            segDisplayNode.AddViewNodeID(sliceViewID)

        segmentation = segmentationNode.GetSegmentation()
        for segmentID, name in segmentList(segmentationNode):
            role = segmentRoles.get(segmentID)
            if role not in SEGMENT_MODEL_OPACITY:
                continue  # normal tissue, others, uncategorized: not rendered in 3D
            polyData = vtk.vtkPolyData()
            if not segmentationNode.GetClosedSurfaceRepresentation(segmentID, polyData) \
                    or polyData.GetNumberOfPoints() == 0:
                logging.warning(f"Segment '{name}' has no surface; no model created.")
                continue
            model = addModel(f"{name} model", polyData, segmentationNode.GetTransformNodeID(),
                             ROLE_SEGMENT_MODEL, folderID)
            if role == "liver":
                segmentation.GetSegment(segmentID).SetColor(*COLOR_LIVER)  # whole liver is always white
                viewIDs = [views["segments3D"], views["isodose3D"]]
            else:
                viewIDs = [views["segments3D"]]
            styleModelDisplay(model.GetDisplayNode(), segmentation.GetSegment(segmentID).GetColor(),
                              SEGMENT_MODEL_OPACITY[role], True, viewIDs)

    def createIsodoseModels(self, doseVolumeNode, levels, views):
        """Iso-surfaces at each level: 3D in the middle view (opacity by level, see isodoseOpacity),
        2 px lines in the middle (isodose) slice view."""
        removeTaggedNodes(ROLE_ISODOSE)
        folderID = recreateFolder(ISODOSE_FOLDER)
        maxDose = doseVolumeNode.GetImageData().GetScalarRange()[1]
        created = []
        for levelIndex, (level, (colorName, rgb)) in enumerate(zip(levels, ISODOSE_COLORS)):
            if level > maxDose:
                continue  # level never reached: no surface
            polyData = isodoseSurface(doseVolumeNode, level)
            if polyData.GetNumberOfPoints() == 0:
                continue
            model = addModel(f"Isodose {level:g} Gy", polyData, doseVolumeNode.GetTransformNodeID(),
                             ROLE_ISODOSE, folderID)
            styleModelDisplay(model.GetDisplayNode(), rgb, isodoseOpacity(levelIndex, len(levels)), False,
                              [views["isodose3D"], views["isodoseSlice"]],
                              sliceIntersectionThickness=self.isodoseLineThickness)
            model.GetDisplayNode().SetVisibility2D(self.isodoseInSliceViews)
            created.append((level, rgb))
        self.lastLegendLevels = [(level, tuple(float(c) for c in rgb)) for level, rgb in created]
        logging.info(f"Created {len(created)} isodose surfaces (maximum dose {maxDose:.1f} Gy).")
        try:
            self.showIsodoseLegend(created, views)
        except Exception as e:  # the legend is cosmetic; never fail the calculation for it
            logging.warning(f"Could not show the isodose legend: {e}")
        self._updateAnnotationReservedArea()  # legend size may have changed
        return created

    def setSegmentOutlineThickness(self, thickness):
        """Segment outline thickness (px) in the slice views; updates the last calculation's segmentation."""
        self.segmentOutlineThickness = max(1, int(thickness))
        node = self.outlinedSegmentationNode
        if node is not None and slicer.mrmlScene.IsNodePresent(node) and node.GetDisplayNode():
            node.GetDisplayNode().SetSliceIntersectionThickness(self.segmentOutlineThickness)

    def setIsodoseLineThickness(self, thickness):
        """Isodose line thickness (px) in the isodose slice view; updates the existing isodose surfaces."""
        self.isodoseLineThickness = max(1, int(thickness))
        for node in taggedNodes(ROLE_ISODOSE):
            if node.IsA("vtkMRMLModelNode") and node.GetDisplayNode():
                node.GetDisplayNode().SetSliceIntersectionThickness(self.isodoseLineThickness)

    def setIsodoseInSliceViews(self, visible):
        """Show/hide isodose lines and their legend in the middle slice view (3D view is unaffected)."""
        self.isodoseInSliceViews = bool(visible)
        for node in taggedNodes(ROLE_ISODOSE):
            if node.IsA("vtkMRMLModelNode") and node.GetDisplayNode():
                node.GetDisplayNode().SetVisibility2D(self.isodoseInSliceViews)
        for viewWidget, renderer, actor, isSliceView in self.legendActors:
            if isSliceView:
                actor.SetVisibility(self.isodoseInSliceViews)
                viewWidget.scheduleRender()
        self._updateAnnotationReservedArea()

    def existingParameterNode(self):
        """The module's parameter node if the scene has one (never creates it)."""
        return slicer.mrmlScene.GetSingletonNode(self.moduleName, "vtkMRMLScriptedModuleNode")

    def resultsBelongToThisModule(self):
        """True if the results views of the scene (DVH chart, models) were made by this module."""
        for node in taggedNodes(ROLE_DVH):
            if node.IsA("vtkMRMLPlotChartNode"):
                return node.GetAttribute(OWNER_ATTRIBUTE) == self.moduleName
        return False

    def resultsViews(self):
        """View node IDs of the results layout (same keys as setupResultsLayout) from the view nodes already
        in the scene, without changing the layout. None if a view is missing."""
        views = {}
        for key, tag in THREED_VIEWS:
            viewNode = slicer.mrmlScene.GetSingletonNode(tag, "vtkMRMLViewNode")
            if viewNode is None:
                return None
            views[key] = viewNode.GetID()
        views["slices"] = []
        layoutManager = slicer.app.layoutManager()
        for sliceName, role in SLICE_VIEWS:
            sliceWidget = layoutManager.sliceWidget(sliceName) if layoutManager else None
            sliceNode = (sliceWidget.mrmlSliceNode() if sliceWidget
                         else slicer.mrmlScene.GetSingletonNode(sliceName, "vtkMRMLSliceNode"))
            if sliceNode is None:
                return None
            views["slices"].append(sliceNode.GetID())
            views[role + "Slice"] = sliceNode.GetID()
        return views

    def clearResultsOverlays(self):
        """Remove what this module drew directly into the views (not MRML nodes, not saved with the scene)."""
        self.removeIsodoseLegend()
        self.unlinkThreeDViewCameras()
        self.removeSegmentAnnotations()
        self.outlinedSegmentationNode = None

    def restoreResultsDisplay(self, segmentationNode, doseVolumeNode, annotationLabels):
        """After a scene is loaded: reconnect to the saved results (models, dose map, DVH chart are in the
        scene) and rebuild the isodose legend, the slice view segment labels and the linked 3D cameras.
        Returns the {"dose", "views"} dict used to regenerate the isodose surfaces, or None."""
        scene = slicer.mrmlScene
        if (segmentationNode is None or doseVolumeNode is None or not scene.IsNodePresent(segmentationNode)
                or not scene.IsNodePresent(doseVolumeNode) or doseVolumeNode.GetImageData() is None):
            return None
        if not self.resultsBelongToThisModule():
            return None  # the views show the results of the other Taranis module
        views = self.resultsViews()
        if views is None:
            return None
        self.outlinedSegmentationNode = segmentationNode
        visualization = {"dose": doseVolumeNode, "views": views}
        layoutManager = slicer.app.layoutManager()
        if layoutManager is None or layoutManager.layout not in RESULTS_LAYOUT_IDS:
            return visualization  # overlays need the views of the results layout
        slicer.app.processEvents()
        try:
            equalizeLayoutColumns(self._secondaryWindowRoots())
        except Exception as e:
            logging.warning(f"Could not equalize view column widths: {e}")
        try:
            self.showIsodoseLegend(self.lastLegendLevels, views)
        except Exception as e:
            logging.warning(f"Could not show the isodose legend: {e}")
        try:
            self.linkThreeDViewCameras([views["segments3D"], views["isodose3D"]])
        except Exception as e:
            logging.warning(f"Could not link the 3D view cameras: {e}")
        if annotationLabels:
            try:
                self.showSegmentAnnotations(segmentationNode, annotationLabels)
            except Exception as e:
                logging.warning(f"Could not create the segment labels: {e}")
        return visualization

    def secondaryWindow(self):
        return secondaryViewWindow()

    def _secondaryWindowRoots(self):
        window = self.secondaryWindow()
        return [window] if window is not None else []

    def placeSecondaryWindow(self):
        placeSecondaryViewWindow()

    @staticmethod
    def _samePose(a, b):
        return (np.allclose(a.GetPosition(), b.GetPosition()) and np.allclose(a.GetFocalPoint(), b.GetFocalPoint())
                and np.allclose(a.GetViewUp(), b.GetViewUp())
                and np.isclose(a.GetParallelScale(), b.GetParallelScale())
                and np.isclose(a.GetViewAngle(), b.GetViewAngle()))

    def unlinkThreeDViewCameras(self):
        for camera, tag in self.cameraObservers:
            try:
                camera.RemoveObserver(tag)
            except Exception:
                pass
        self.cameraObservers = []

    def linkThreeDViewCameras(self, viewNodeIDs):
        """Rotate/zoom/pan in one 3D view is copied to the others. The first view's camera is the starting pose."""
        self.unlinkThreeDViewCameras()
        layoutManager = slicer.app.layoutManager()
        linked = []
        for viewNodeID in viewNodeIDs:  # keep the given order: first view defines the initial pose
            for index in range(layoutManager.threeDViewCount):
                threeDWidget = layoutManager.threeDWidget(index)
                if threeDWidget.mrmlViewNode().GetID() == viewNodeID:
                    view = threeDWidget.threeDView()
                    renderer = view.renderWindow().GetRenderers().GetFirstRenderer()
                    linked.append((view, renderer, renderer.GetActiveCamera()))
        if len(linked) < 2:
            logging.warning("Fewer than two 3D views found; cameras not linked.")
            return

        for sourceIndex, (_, _, sourceCamera) in enumerate(linked):
            targets = [entry for i, entry in enumerate(linked) if i != sourceIndex]

            def onCameraModified(caller, event, targets=targets):
                if self._syncingCameras:
                    return
                self._syncingCameras = True  # prevents the target's own Modified event from echoing back
                try:
                    for view, renderer, camera in targets:
                        if self._samePose(caller, camera):
                            continue  # nothing changed (e.g. only the clipping range): avoid render ping-pong
                        camera.SetPosition(caller.GetPosition())
                        camera.SetFocalPoint(caller.GetFocalPoint())
                        camera.SetViewUp(caller.GetViewUp())
                        camera.SetParallelScale(caller.GetParallelScale())  # zoom in orthographic mode
                        camera.SetViewAngle(caller.GetViewAngle())          # zoom in perspective mode
                        renderer.ResetCameraClippingRange()
                        view.scheduleRender()
                finally:
                    self._syncingCameras = False

            tag = sourceCamera.AddObserver(vtk.vtkCommand.ModifiedEvent, onCameraModified)
            self.cameraObservers.append((sourceCamera, tag))

        linked[0][2].Modified()  # align the second view to the first one now

    def removeIsodoseLegend(self):
        for viewWidget, renderer, actor, _ in self.legendActors:
            try:
                renderer.RemoveViewProp(actor)
                viewWidget.scheduleRender()
            except Exception:
                pass  # view was destroyed
        self.legendActors = []

    def showIsodoseLegend(self, levelColors, views):
        """Colour legend of the isodose levels actually present, in the isodose 3D view and the slice views."""
        self.removeIsodoseLegend()
        if not levelColors:
            return
        layoutManager = slicer.app.layoutManager()
        viewWidgets = []
        for index in range(layoutManager.threeDViewCount):
            threeDWidget = layoutManager.threeDWidget(index)
            if threeDWidget.mrmlViewNode().GetID() == views["isodose3D"]:
                viewWidgets.append((threeDWidget.threeDView(), False))
        for sliceName, role in SLICE_VIEWS:
            if role == "isodose":
                viewWidgets.append((layoutManager.sliceWidget(sliceName).sliceView(), True))
        for viewWidget, isSliceView in viewWidgets:
            renderer = viewWidget.renderWindow().GetRenderers().GetFirstRenderer()
            removeLegendActorsFromRenderer(renderer)  # e.g. a legend left by the other Taranis module
            actor = makeIsodoseLegendActor(levelColors)
            if isSliceView:
                actor.SetVisibility(self.isodoseInSliceViews)
            renderer.AddViewProp(actor)
            self.legendActors.append((viewWidget, renderer, actor, isSliceView))
            viewWidget.scheduleRender()

    def showSegmentAnnotations(self, segmentationNode, labels):
        """labels: [(segmentID, text)]. Segment names with leader lines in all slice views of the results
        layout. Replaces the labels of the previous calculation, including one made by the other Taranis module."""
        self.removeSegmentAnnotations()
        previous = getattr(slicer, ANNOTATIONS_ATTRIBUTE, None)
        if previous is not None:
            try:
                previous.clear()
            except Exception:
                pass
        segmentation = segmentationNode.GetSegmentation()
        surfaces = segmentSurfacesInWorld(segmentationNode, {label[0] for label in labels})
        entries = []
        for label in labels:
            segmentID, text = label[0], label[1]
            if len(label) > 2 and label[2] is not None:  # labels saved as (segmentID, name + volume, mean dose)
                meanDose = float(label[2])
                text += "\n" + (f"Mean {meanDose:.1f} Gy" if np.isfinite(meanDose) else "Mean dose n/a")
            segment = segmentation.GetSegment(segmentID)
            if segment is not None and segmentID in surfaces:
                entries.append((text, tuple(segment.GetColor()), surfaces[segmentID]))
        self.segmentAnnotations = SliceSegmentAnnotations(entries, [name for name, _ in SLICE_VIEWS])
        self.segmentAnnotations.visible = self.segmentAnnotationsVisible
        setattr(slicer, ANNOTATIONS_ATTRIBUTE, self.segmentAnnotations)
        self._updateAnnotationReservedArea()

    def removeSegmentAnnotations(self):
        if self.segmentAnnotations is not None:
            self.segmentAnnotations.clear()
            if getattr(slicer, ANNOTATIONS_ATTRIBUTE, None) is self.segmentAnnotations:
                setattr(slicer, ANNOTATIONS_ATTRIBUTE, None)
            self.segmentAnnotations = None

    def setSegmentAnnotationsVisible(self, visible):
        self.segmentAnnotationsVisible = bool(visible)
        if self.segmentAnnotations is not None:
            self.segmentAnnotations.setVisible(visible)

    def _updateAnnotationReservedArea(self):
        """Keep the right-hand labels of the isodose slice view above the isodose legend (when shown)."""
        if self.segmentAnnotations is None:
            return
        reserved = {}
        if self.isodoseInSliceViews:
            for _, _, actor, isSliceView in self.legendActors:
                if isSliceView:  # Position2 is relative to Position (normalized viewport)
                    top = actor.GetPositionCoordinate().GetValue()[1] + actor.GetPosition2Coordinate().GetValue()[1]
                    reserved = {name: top + 0.01 for name, role in SLICE_VIEWS if role == "isodose"}
                    break
        self.segmentAnnotations.reservedRightBottom = reserved
        self.segmentAnnotations.update()

    def captureViewScreenshots(self):
        """[(caption, pngBytes)]: one screenshot per view of the results layout, as currently shown.
        Switches to the results layout if another layout is active and restores it afterwards."""
        layoutManager = slicer.app.layoutManager()
        layoutNode = layoutManager.layoutLogic().GetLayoutNode()
        previousLayout = layoutManager.layout
        targetLayout = previousLayout if previousLayout in RESULTS_LAYOUT_IDS else self.preferredLayoutID
        if not layoutNode.IsLayoutDescription(targetLayout):
            return []
        if previousLayout != targetLayout:
            layoutManager.setLayout(targetLayout)
            slicer.app.processEvents()
        try:
            widgets = []
            # the results layout's own 3D views (not Slicer's default "1" / "2", which other layouts use)
            for tag, caption in ((SEGMENTS_VIEW_TAG, "3D view - segment models"),
                                 (ISODOSE_VIEW_TAG, "3D view - isodose surfaces")):
                viewNode = slicer.mrmlScene.GetSingletonNode(tag, "vtkMRMLViewNode")
                for index in range(layoutManager.threeDViewCount):
                    threeDWidget = layoutManager.threeDWidget(index)
                    if viewNode and threeDWidget.mrmlViewNode().GetID() == viewNode.GetID():
                        widgets.append((caption, threeDWidget.threeDView()))
                        break
            plotViewNode = slicer.mrmlScene.GetSingletonNode("PlotView1", "vtkMRMLPlotViewNode")
            try:
                plotWidgets = [layoutManager.plotWidget(i) for i in range(layoutManager.plotViewCount)]
            except Exception:
                plotWidgets = [layoutManager.plotWidget(0)]
            for plotWidget in plotWidgets:
                if plotWidget is not None and (plotViewNode is None
                                               or plotWidget.mrmlPlotViewNode().GetID() == plotViewNode.GetID()):
                    widgets.append(("Dose-volume histogram", plotWidget.plotView()))
                    break
            sliceCaptions = {"fusion": "Axial - reference with PET/SPECT fusion",
                             "isodose": "Axial - reference with isodose lines",
                             "reference": "Axial - reference image"}
            for sliceName, role in SLICE_VIEWS:
                sliceWidget = layoutManager.sliceWidget(sliceName)
                if sliceWidget is not None:
                    widgets.append((sliceCaptions[role], sliceWidget.sliceView()))

            screenshots = []
            for caption, widget in widgets:
                if hasattr(widget, "forceRender"):
                    widget.forceRender()
                slicer.app.processEvents()
                pngBytes = qImageToPngBytes(ctk.ctkWidgetsUtils.grabWidget(widget))
                if pngBytes:
                    screenshots.append((caption, pngBytes))
                else:
                    logging.warning(f"Screenshot of '{caption}' failed.")
            return screenshots
        finally:
            if previousLayout != targetLayout:
                layoutManager.setLayout(previousLayout)

    def createDvhChart(self, curves):
        """curves: [(name, rgb, sortedDosesAsc[, lineStyle, lineWidth])], lineStyle one of DVH_LINE_STYLES.
        Returns a plot chart node with one cumulative DVH per curve."""
        removeTaggedNodes(ROLE_DVH)
        folderID = recreateFolder(DVH_FOLDER)
        chartNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotChartNode", "Dose-volume histogram")
        tagAndFile(chartNode, ROLE_DVH, folderID)
        chartNode.SetAttribute(OWNER_ATTRIBUTE, self.moduleName)  # see restoreResultsDisplay
        chartNode.SetTitle("Cumulative dose-volume histogram")
        chartNode.SetXAxisTitle("Dose (Gy)")
        chartNode.SetYAxisTitle("Volume (%)")
        chartNode.SetLegendVisibility(True)
        if not curves:
            return chartNode

        maxDose = max(float(curve[2][-1]) for curve in curves)
        doseAxis = np.linspace(0.0, max(maxDose, 1e-6), 501)
        columnNames = ["Dose (Gy)"] + [f"S{i}" for i in range(len(curves))]  # unique even if segment names repeat
        columns = [doseAxis] + [cumulativeDvh(curve[2], doseAxis) for curve in curves]
        tableNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode", "DVH table")
        tagAndFile(tableNode, ROLE_DVH, folderID)
        slicer.util.updateTableFromArray(tableNode, columns, columnNames)

        for curve, columnName in zip(curves, columnNames[1:]):
            name, rgb = curve[0], curve[1]
            lineStyle = curve[3] if len(curve) > 3 else "Solid"
            lineWidth = curve[4] if len(curve) > 4 else 2
            seriesNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotSeriesNode", name)
            tagAndFile(seriesNode, ROLE_DVH, folderID)
            seriesNode.SetAndObserveTableNodeID(tableNode.GetID())
            seriesNode.SetXColumnName("Dose (Gy)")
            seriesNode.SetYColumnName(columnName)
            seriesNode.SetPlotType(slicer.vtkMRMLPlotSeriesNode.PlotTypeScatter)
            seriesNode.SetMarkerStyle(slicer.vtkMRMLPlotSeriesNode.MarkerStyleNone)
            seriesNode.SetLineStyle(getattr(slicer.vtkMRMLPlotSeriesNode, "LineStyle" + lineStyle,
                                            slicer.vtkMRMLPlotSeriesNode.LineStyleSolid))
            seriesNode.SetLineWidth(lineWidth)
            seriesNode.SetColor(*rgb)
            chartNode.AddAndObservePlotSeriesNodeID(seriesNode.GetID())
        return chartNode

    def showDvhChart(self, chartNode, views):
        plotWidget = slicer.app.layoutManager().plotWidget(0)
        if plotWidget is None:
            raise RuntimeError("The results layout has no plot view.")
        plotWidget.mrmlPlotViewNode().SetPlotChartNodeID(chartNode.GetID())

    def setSliceViewsLinked(self, linked):
        """Linked: scrolling/zooming/panning one slice view moves all three. Unlinked while the per-view layers are
        configured; programmatic changes are not broadcast, so the different layers are kept after re-linking."""
        layoutManager = slicer.app.layoutManager()
        for sliceName, _ in SLICE_VIEWS:
            sliceWidget = layoutManager.sliceWidget(sliceName)
            if sliceWidget is None:
                continue
            compositeNode = sliceWidget.mrmlSliceCompositeNode()
            compositeNode.SetLinkedControl(linked)
            compositeNode.SetHotLinkedControl(linked)  # follow continuously while dragging

    def configureSliceLayers(self, referenceVolumeNode, spectVolumeNode, doseVolumeNode):
        """Axial layers: left = reference + PET/SPECT (inferno), middle = reference + dose map as an invisible
        foreground (opacity 0, so the Data Probe still reports the dose under the mouse), right = reference.
        Any node may be None; without a reference volume the PET/SPECT is shown as the background."""
        self.setSliceViewsLinked(False)
        if spectVolumeNode is not None:
            spectVolumeNode.CreateDefaultDisplayNodes()
            infernoNode = findColorNode("Inferno")
            if infernoNode:
                spectVolumeNode.GetDisplayNode().SetAndObserveColorNodeID(infernoNode.GetID())
            else:
                logging.warning("Inferno colour table not found; PET/SPECT keeps its current colours.")
        if referenceVolumeNode is not None and (spectVolumeNode is None
                                                or referenceVolumeNode.GetID() != spectVolumeNode.GetID()):
            referenceVolumeNode.CreateDefaultDisplayNodes()  # reference (CT/MRI) always in grayscale
            greyNode = findColorNode("Grey")
            displayNode = referenceVolumeNode.GetDisplayNode()
            if greyNode and displayNode:
                displayNode.SetAndObserveColorNodeID(greyNode.GetID())
            elif not greyNode:
                logging.warning("Grey colour table not found; the reference volume keeps its current colours.")
        background = referenceVolumeNode if referenceVolumeNode is not None else spectVolumeNode
        fusion = spectVolumeNode if referenceVolumeNode is not None else None
        layers = {
            "fusion": (fusion, FUSION_OPACITY),
            "isodose": (doseVolumeNode, 0.0),
            "reference": (None, 0.0),
        }
        layoutManager = slicer.app.layoutManager()
        for sliceName, role in SLICE_VIEWS:
            sliceWidget = layoutManager.sliceWidget(sliceName)
            if sliceWidget is None:
                continue
            sliceWidget.mrmlSliceNode().SetOrientation("Axial")
            foreground, opacity = layers[role]
            compositeNode = sliceWidget.mrmlSliceCompositeNode()
            compositeNode.SetBackgroundVolumeID(background.GetID() if background is not None else None)
            compositeNode.SetForegroundVolumeID(foreground.GetID() if foreground is not None else None)
            compositeNode.SetForegroundOpacity(opacity)

    def centerSliceViews(self, volumeNode):
        """Centre the slice views on the volume with a 300 mm field of view across the shorter side."""
        bounds = [0.0] * 6
        volumeNode.GetRASBounds(bounds)  # world coordinates, parent transforms included
        center = [(bounds[0] + bounds[1]) / 2.0, (bounds[2] + bounds[3]) / 2.0, (bounds[4] + bounds[5]) / 2.0]
        fov = self.SLICE_FIELD_OF_VIEW_MM
        layoutManager = slicer.app.layoutManager()
        for sliceName, _ in SLICE_VIEWS:
            sliceWidget = layoutManager.sliceWidget(sliceName)
            if sliceWidget is None:
                continue
            sliceNode = sliceWidget.mrmlSliceNode()
            width, height = [max(1, d) for d in sliceNode.GetDimensions()[:2]]
            # The longer side follows the view's aspect ratio (square pixels)
            if width >= height:
                fovX, fovY = fov * width / height, fov
            else:
                fovX, fovY = fov, fov * height / width
            sliceNode.SetXYZOrigin(0.0, 0.0, 0.0)  # remove any previous panning
            sliceNode.SetFieldOfView(fovX, fovY, sliceNode.GetFieldOfView()[2])
        slicer.modules.markups.logic().JumpSlicesToLocation(center[0], center[1], center[2], True)

    def showSliceViews(self, referenceVolumeNode, doseVolumeNode, spectVolumeNode):
        """Slice views of the results (see configureSliceLayers), centred on the reference volume and linked."""
        self.configureSliceLayers(referenceVolumeNode, spectVolumeNode, doseVolumeNode)
        self.centerSliceViews(referenceVolumeNode)
        self.setSliceViewsLinked(True)
        slicer.util.resetThreeDViews()

    def showPreviewSlices(self, referenceVolumeNode, spectVolumeNode, doseVolumeNode, recenter):
        """Selected images in the slice views, without isodose lines. doseVolumeNode: dose map of an existing
        calculation (kept as the invisible Data Probe layer) or None."""
        self.configureSliceLayers(referenceVolumeNode, spectVolumeNode, doseVolumeNode)
        centreOn = referenceVolumeNode if referenceVolumeNode is not None else spectVolumeNode
        if recenter and centreOn is not None:
            self.centerSliceViews(centreOn)
        self.setSliceViewsLinked(True)

    def showPreviewSegments(self, segmentationNode, liverSegmentID, perfusedIDs, categories, views, resetCamera):
        """Segment outlines in the slice views and wireframe models in the 3D views (whole liver, perfused
        volumes, tumors) in the standard colours, for the segments that already have a role."""
        if segmentationNode is None:
            removeTaggedNodes(ROLE_SEGMENT_MODEL)
            return
        applySegmentColors(segmentationNode, liverSegmentID, perfusedIDs, categories, recolorUncategorized=False)
        roles = {segmentID: segmentRole(segmentID, liverSegmentID, perfusedIDs, categories)
                 for segmentID, _ in segmentList(segmentationNode)}
        self.createSegmentModels(segmentationNode, roles, views)
        if resetCamera:
            slicer.util.resetThreeDViews()


__all__ = [
    "ISODOSE_COLORS",
    "ISODOSE_PRESETS",
    "D_METRICS",
    "V_METRICS",
    "ROLE_ATTRIBUTE",
    "ROLE_SEGMENT_MODEL",
    "ROLE_ISODOSE",
    "ROLE_DVH",
    "SEGMENT_MODEL_FOLDER",
    "ISODOSE_FOLDER",
    "DVH_FOLDER",
    "RESULTS_LAYOUT_ID",
    "RESULTS_LAYOUT_XML",
    "RESULTS_DUAL_LAYOUT_ID",
    "RESULTS_DUAL_LAYOUT_XML",
    "MAIN_VIEWPORT_MARKER",
    "RESULTS_LAYOUT_IDS",
    "LAYOUT_MODE_ATTRIBUTE",
    "SLICE_VIEWS",
    "SEGMENTS_VIEW_TAG",
    "ISODOSE_VIEW_TAG",
    "THREED_VIEWS",
    "LEGACY_THREED_TAGS",
    "LEGACY_SLICE_NAMES",
    "FOREIGN_DISPLAY_CLASSES",
    "ANNOTATIONS_ATTRIBUTE",
    "OWNER_ATTRIBUTE",
    "SETTINGS_VERSION",
    "PARAM_VERSION",
    "PARAM_LIVER_SEGMENT",
    "PARAM_RESULTS",
    "PARAM_ACTIVE_MODULE",
    "PARAM_REPORT_FILE",
    "PARAM_REPORT_DOSE_CHECKSUM",
    "REF_RESULTS_DOSE",
    "REF_RESULTS_SEGMENTATION",
    "CATEGORY_ATTRIBUTE",
    "CATEGORY_TUMOR",
    "CATEGORY_NORMAL",
    "CATEGORY_OTHER",
    "CATEGORY_VIABLE",
    "CATEGORY_LUNGS",
    "CATEGORY_IGNORED",
    "SEGMENT_CATEGORIES",
    "COLOR_LIVER",
    "COLOR_PERFUSED",
    "COLOR_TUMOR",
    "COLOR_NORMAL",
    "COLOR_OTHER",
    "COLOR_VIABLE",
    "COLOR_LUNGS",
    "RESULT_ROLES",
    "COMBINED_TUMORS_NAME",
    "COMBINED_VIABLE_NAME",
    "SEGMENT_MODEL_OPACITY",
    "FUSION_OPACITY",
    "REFERENCE_WINDOW_PRESETS",
    "SPECT_WINDOW_PERCENTS",
    "DEFAULT_SEGMENT_OUTLINE_THICKNESS",
    "DEFAULT_ISODOSE_LINE_THICKNESS",
    "DISCLAIMER",
    "prepareImageValues",
    "negativeVoxelNote",
    "countsOutsideMaskFraction",
    "segmentDoseStatistics",
    "doseAtVolumePercent",
    "volumePercentAtDose",
    "cumulativeDvh",
    "formatNumber",
    "rgbToHex",
    "isodoseLegendHtml",
    "rtfEscape",
    "pngSize",
    "rtfPicture",
    "buildRtfReport",
    "buildTsvTable",
    "dicomValue",
    "formatDicomName",
    "formatDicomDateTime",
    "imagingDate",
    "reportPatientInfo",
    "SEGMENT_DOSE_HEADER",
    "REPORT_DISCLAIMER",
    "REPORT_WARNING",
    "segmentDoseTable",
    "parseMetricLines",
    "metricTable",
    "reportBlocks",
    "rtfColumnWidths",
    "rtfTable",
    "buildHtmlReport",
    "pdfExportAvailable",
    "writePdfReport",
    "segmentVolumeML",
    "voxelVolumeMLFromNode",
    "segmentMaskOnVolumeGrid",
    "segmentLabelText",
    "_volumeDisplayNode",
    "_setWindowMinMax",
    "applyReferenceWindowPreset",
    "applySpectWindowPercent",
    "tableRows",
    "doseChecksum",
    "widgetSettingValue",
    "applyWidgetSetting",
    "writeDoseVolume",
    "makeIsodoseLegendActor",
    "removeLegendActorsFromRenderer",
    "equalizeLayoutColumns",
    "dualMonitorLayoutSupported",
    "screenCount",
    "secondaryScreen",
    "findColorNode",
    "isodoseOpacity",
    "taggedNodes",
    "_viewIDs",
    "excludeForeignRenderings",
    "migrateLegacyViewIDs",
    "removeTaggedNodes",
    "recreateFolder",
    "tagAndFile",
    "addModel",
    "styleModelDisplay",
    "isodoseSurface",
    "segmentList",
    "loadSegmentCategories",
    "saveSegmentCategories",
    "segmentRole",
    "applySegmentColors",
    "sortResults",
    "DG",
    "SEGMENT_ROLE_TAG",
    "segmentTagValue",
    "normalScope",
    "taggedPerfusedIDs",
    "doseCheckSegments",
    "doseCheckLines",
    "DVH_LIVER_COLOR",
    "DVH_LINE_STYLES",
    "shadeColor",
    "dvhCurveStyles",
    "SegmentComboBox",
    "SegmentCategorizer",
    "qImageToPngBytes",
    "segmentSurfacesInWorld",
    "stackLabelCentres",
    "SliceSegmentAnnotations",
    "secondaryViewWindow",
    "placeSecondaryViewWindow",
    "placeWindowOnSecondaryScreen",
    "placeSecondaryViewWindowAfterLoad",
    "mainViewportAttributes",
    "resultsDualLayoutXml",
    "registerResultsLayout",
    "DosimetryWidgetBase",
    "DosimetryLogicBase",
]
