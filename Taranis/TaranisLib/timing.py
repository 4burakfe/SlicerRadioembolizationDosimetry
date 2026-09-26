"""Time from the administration to the reference time of the dosimetry image ("Hours after treatment" of the
absolute dosimetry), from the case's administration time and the image's DICOM header.

Pure (no Slicer): the header is given as a dict of DICOM keyword -> string (see headerTimes). The result is a
suggestion only: the user must check it (scanner clock, reconstruction done later, vendor-specific decay
correction...).
"""

import datetime

MAX_HOURS = 200.0                 # range of the absolute module's "Hours after treatment" slider
INJECTION_MISMATCH_MINUTES = 10   # the scanner's injection time differs from the case administration time


def _dt(date, time):
    date = (date or "").strip()
    time = (time or "").strip().split(".")[0]
    if len(date) != 8 or not date.isdigit():
        return None
    time = (time + "000000")[:6] if time.isdigit() else "000000"
    try:
        return datetime.datetime(int(date[:4]), int(date[4:6]), int(date[6:8]),
                                 int(time[:2]), int(time[2:4]), int(time[4:6]))
    except ValueError:
        return None


def _dtCombined(text):
    """DICOM DT (YYYYMMDDHHMMSS.FFFFFF&ZZXX): date and time, time zone ignored."""
    text = (text or "").strip()
    for sign in "+-":
        if sign in text[8:]:
            text = text[:8] + text[8:].split(sign)[0]
    return _dt(text[:8], text[8:]) if len(text) >= 8 else None


def headerTimes(dataset):
    """The DICOM fields used here, as strings, from a pydicom dataset (or any mapping with .get)."""
    def text(source, key):
        try:
            value = source.get(key, "")
        except Exception:
            value = ""
        return str(value or "").strip()
    fields = {key: text(dataset, key) for key in ("DecayCorrection", "SeriesDate", "SeriesTime", "AcquisitionDate",
                                                  "AcquisitionTime", "AcquisitionDateTime", "Modality")}
    item = {}
    try:
        sequence = dataset.get("RadiopharmaceuticalInformationSequence")
        item = sequence[0] if sequence else {}
    except Exception:
        item = {}
    fields["RadiopharmaceuticalStartDateTime"] = text(item, "RadiopharmaceuticalStartDateTime")
    fields["RadiopharmaceuticalStartTime"] = text(item, "RadiopharmaceuticalStartTime")
    return fields


def _fmt(value):
    return value.strftime("%Y-%m-%d %H:%M")


def hoursAfterTreatment(administration, fields):
    """Suggested "hours after treatment" of an image.

    administration: datetime of the Y-90 administration (the case's administration time).
    fields: headerTimes() of the dosimetry image.
    Returns dict(hours (None when it cannot be suggested), reference (datetime or None), basis (text),
    notes [texts the user should read]).
    """
    notes = []
    decay = (fields.get("DecayCorrection") or "").upper()
    series = _dt(fields.get("SeriesDate"), fields.get("SeriesTime"))
    acquisition = (_dtCombined(fields.get("AcquisitionDateTime"))
                   or _dt(fields.get("AcquisitionDate") or fields.get("SeriesDate"), fields.get("AcquisitionTime")))

    if decay == "ADMIN":
        return {"hours": 0.0, "reference": administration, "notes": notes,
                "basis": "the image is decay-corrected to the administration time (DICOM Decay Correction = ADMIN)"}

    if decay == "NONE":
        reference = acquisition or series
        basis = "acquisition start (the image is not decay-corrected: DICOM Decay Correction = NONE)"
    elif decay == "START":
        reference, basis = series, "series start (DICOM Decay Correction = START)"
        if series is None:
            reference, basis = acquisition, "acquisition start (DICOM Decay Correction = START, no series time)"
        elif acquisition is not None and acquisition < series:
            reference = acquisition
            basis = "acquisition start (DICOM Decay Correction = START)"
            notes.append(f"The series time ({_fmt(series)}) is after the acquisition time ({_fmt(acquisition)}), "
                         "e.g. the series was reconstructed later: the acquisition time was used.")
    else:
        reference = acquisition or series
        basis = ("acquisition start (the header does not state the decay-correction reference; check how the "
                 "image was reconstructed)")
    if reference is None:
        return {"hours": None, "reference": None, "basis": basis,
                "notes": notes + ["The image has no acquisition or series time in its DICOM header."]}
    if reference is acquisition and fields.get("Modality", "").upper() == "PT":
        notes.append("Only the first image of the series is read: with several bed positions the acquisition "
                     "times differ.")

    hours = (reference - administration).total_seconds() / 3600.0
    scanner = (_dtCombined(fields.get("RadiopharmaceuticalStartDateTime"))
               or _dt(fields.get("SeriesDate"), fields.get("RadiopharmaceuticalStartTime")))
    if scanner is not None:
        minutes = abs((scanner - administration).total_seconds()) / 60.0
        if minutes > INJECTION_MISMATCH_MINUTES:
            notes.append(f"The scanner recorded the administration at {_fmt(scanner)}, {minutes:.0f} min from the "
                         f"case administration time ({_fmt(administration)}): check which one is right.")
    if hours < 0:
        notes.append(f"The image reference time ({_fmt(reference)}) is before the administration "
                     f"({_fmt(administration)}): check the administration time and the scanner clock.")
        return {"hours": None, "reference": reference, "basis": basis, "notes": notes}
    if hours > MAX_HOURS:
        notes.append(f"{hours:.1f} h between administration and image is more than {MAX_HOURS:.0f} h: check the "
                     "dates.")
        return {"hours": None, "reference": reference, "basis": basis, "notes": notes}
    return {"hours": hours, "reference": reference, "basis": basis, "notes": notes}
