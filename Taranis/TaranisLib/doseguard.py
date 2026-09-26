"""Dose checks of a dosimetry calculation: advisory warnings (never blocking) shown after the calculation, in the
report and on the Taranis toolbar.

Pure Python (no Slicer): the dosimetry modules collect the numbers and call doseChecks(). The thresholds are values
commonly used in radioembolization planning; they do not replace the device instructions or clinical judgement.
"""

import math

from . import workflow as W

GLASS = "glass"
RESIN = "resin"
MICROSPHERE_NAMES = {GLASS: "glass microspheres", RESIN: "resin microspheres"}

NORMAL_LIVER_LIMIT_GY = {RESIN: 40.0, GLASS: 90.0}   # mean dose of the normal (non-tumoral) liver
TUMOUR_TARGET_GY = {RESIN: 80.0, GLASS: 140.0}       # mean tumour dose below this: probably not enough
TN_LOW = 1.5                                         # tumour-to-normal dose ratio
TN_TOO_LOW = 1.0
LSF_HIGH_PERCENT = 20.0
LUNG_DOSE_LIMIT_GY = 30.0                            # single session
EXTRA_UPTAKE_FRACTION = 0.20                         # image counts outside the whole liver and the lungs
OUTSIDE_PERFUSED_FRACTION = 0.20                     # whole-liver counts outside the perfused volumes
MAX_NAMES = 6                                        # segment names listed in one message

TUMOUR_ROLES = ("tumor", "viable")
NORMAL_ROLE = "normal"


def microspheresFromText(text):
    """GLASS / RESIN from a label such as the isodose set name ("Glass microspheres"), or None."""
    text = (text or "").lower()
    if "resin" in text or "sir-sphere" in text or "sirsphere" in text:
        return RESIN
    if "glass" in text or "therasphere" in text:
        return GLASS
    return None


def _finite(value):
    return value is not None and isinstance(value, (int, float)) and math.isfinite(value)


def _names(items, formatter):
    shown = [formatter(item) for item in items[:MAX_NAMES]]
    more = len(items) - len(shown)
    return ", ".join(shown) + (f" and {more} more" if more > 0 else "")


def referenceNormal(segments):
    """(mean dose Gy, description) of the normal liver used for the tumour-to-normal ratio: the perfused normal
    liver segments if there are any, otherwise all normal tissue segments (volume-weighted), or (None, "")."""
    normals = [s for s in segments if s.get("role") == NORMAL_ROLE and _finite(s.get("dose"))]
    perfused = [s for s in normals if s.get("scope") == W.NORMAL_SCOPE_PERFUSED]
    chosen = perfused or normals
    if not chosen:
        return None, ""
    weights = [s.get("volume") if _finite(s.get("volume")) and s.get("volume") > 0 else 1.0 for s in chosen]
    dose = sum(w * s["dose"] for w, s in zip(weights, chosen)) / sum(weights)
    what = "perfused normal liver" if perfused else "normal liver"
    names = ", ".join(f"'{s['name']}'" for s in chosen)
    return dose, f"{what} {names}"


def doseChecks(segments, microspheres=None, lsfPercent=None, lungDosesGy=(), extraUptakeFraction=None,
               lungsSegmented=True, outsidePerfusedFraction=None, relative=False, hoursAfterTreatment=None):
    """[(severity, text)] of the dose checks.

    segments: [{name, role ("tumor", "viable", "normal", ...), dose (mean Gy), volume (mL), scope (normal tissue:
    W.NORMAL_SCOPE_PERFUSED / W.NORMAL_SCOPE_WHOLE)}], individual segments only (not the combined rows).
    microspheres: GLASS / RESIN (None: the device-dependent checks are skipped).
    lsfPercent: patient-relative mode only. lungDosesGy: [(label, Gy)].
    extraUptakeFraction: part of the image counts outside the whole liver and the lungs; lungsSegmented: False if
    there is no lung segment (then the lungs are part of that fraction).
    outsidePerfusedFraction: part of the whole-liver counts outside the perfused volumes (None: no perfused volumes).
    relative: patient-relative mode (that liver gets 0 Gy). hoursAfterTreatment: absolute mode only.
    """
    issues = []
    device = MICROSPHERE_NAMES.get(microspheres, "")

    # Normal liver
    if microspheres in NORMAL_LIVER_LIMIT_GY:
        limit = NORMAL_LIVER_LIMIT_GY[microspheres]
        high = [s for s in segments if s.get("role") == NORMAL_ROLE and _finite(s.get("dose")) and s["dose"] > limit]
        if high:
            issues.append((W.SEVERITY_WARNING,
                           "High normal tissue dose: " + _names(high, lambda s: f"'{s['name']}' {s['dose']:.1f} Gy")
                           + f" (above {limit:g} Gy for {device}). This can be acceptable for a radiation "
                           "segmentectomy or lobectomy, but be cautious: risk of radioembolization-induced liver "
                           "disease, especially with a small remaining liver or impaired liver function."))

    # Tumours: dose and tumour-to-normal ratio
    tumours = [s for s in segments if s.get("role") in TUMOUR_ROLES and _finite(s.get("dose"))]
    if tumours and microspheres in TUMOUR_TARGET_GY:
        target = TUMOUR_TARGET_GY[microspheres]
        low = [s for s in tumours if s["dose"] < target]
        if low:
            issues.append((W.SEVERITY_WARNING,
                           "Low tumour dose: " + _names(low, lambda s: f"'{s['name']}' {s['dose']:.1f} Gy")
                           + f" (below {target:g} Gy for {device})."))
    if tumours:
        normalDose, normalText = referenceNormal(segments)
        if normalDose is None:
            issues.append((W.SEVERITY_INFO, "Tumour-to-normal ratio not checked: no normal tissue segment (create "
                                            "'Perfused normal liver' in the Segmentation step)."))
        elif normalDose <= 0:
            issues.append((W.SEVERITY_INFO, f"Tumour-to-normal ratio not checked: the {normalText} has no dose."))
        else:
            ratios = [(s, s["dose"] / normalDose) for s in tumours]
            tooLow = [(s, r) for s, r in ratios if r < TN_TOO_LOW]
            low = [(s, r) for s, r in ratios if TN_TOO_LOW <= r < TN_LOW]
            reference = f"{normalText}: {normalDose:.1f} Gy"
            if tooLow:
                issues.append((W.SEVERITY_WARNING,
                               f"Tumour-to-normal ratio too low (< {TN_TOO_LOW:g}): "
                               + _names(tooLow, lambda item: f"'{item[0]['name']}' {item[1]:.2f}")
                               + f" ({reference}). The tumour gets less than the normal liver: check the "
                               "segmentation and the registration."))
            if low:
                issues.append((W.SEVERITY_WARNING,
                               f"Tumour-to-normal ratio low (< {TN_LOW:g}): "
                               + _names(low, lambda item: f"'{item[0]['name']}' {item[1]:.2f}")
                               + f" ({reference}). Check the segmentation and the registration."))
            if normalText.startswith("normal liver") and relative:
                issues.append((W.SEVERITY_INFO, "Tumour-to-normal ratio against the whole normal liver, which "
                                                "includes unperfused liver (0 Gy): create 'Perfused normal liver' "
                                                "for the usual ratio."))

    # Lungs
    if _finite(lsfPercent) and lsfPercent > LSF_HIGH_PERCENT:
        issues.append((W.SEVERITY_WARNING, f"Lung shunt fraction {lsfPercent:.1f} % is above {LSF_HIGH_PERCENT:g} %: "
                                           "commonly a contraindication or a reason to reduce the activity. Check "
                                           "the lung dose."))
    high = [(label, dose) for label, dose in lungDosesGy if _finite(dose) and dose > LUNG_DOSE_LIMIT_GY]
    if high:
        issues.append((W.SEVERITY_WARNING, "High lung dose: " + _names(high, lambda item: f"{item[0]} {item[1]:.1f} Gy")
                       + f" (above the commonly used single-session limit of {LUNG_DOSE_LIMIT_GY:g} Gy)."))

    # Counts outside the liver
    if _finite(extraUptakeFraction) and extraUptakeFraction > EXTRA_UPTAKE_FRACTION:
        where = "the whole liver and the lungs" if lungsSegmented else "the whole liver (no lung segment: the lung " \
                                                                       "counts are included)"
        issues.append((W.SEVERITY_WARNING, f"Significant uptake outside {where}: {100 * extraUptakeFraction:.1f}% of "
                                           "the image counts. It may be related to free Tc-99m pertechnetate "
                                           "(stomach, thyroid, kidneys), the reconstruction method and noise, or "
                                           "segmentation errors: double-check the image and the segments."))
    if _finite(outsidePerfusedFraction) and outsidePerfusedFraction > OUTSIDE_PERFUSED_FRACTION:
        effect = (" That liver gets 0 Gy in patient-relative dosimetry." if relative else "")
        issues.append((W.SEVERITY_WARNING, f"Significant activity outside the perfused volumes: "
                                           f"{100 * outsidePerfusedFraction:.1f}% of the whole-liver counts.{effect} "
                                           "Is a perfused volume missing or too small?"))

    # Timing
    if hoursAfterTreatment is not None and _finite(hoursAfterTreatment) and hoursAfterTreatment <= 0:
        issues.append((W.SEVERITY_WARNING, "Hours after treatment is 0: correct only if the image is decay-corrected "
                                           "to the administration time. If it was not set, the doses are "
                                           "underestimated."))
    return issues
