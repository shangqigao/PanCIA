"""MR sequence family of a series, for reporting and stratification only (stage 2 infers appearance within each scan; no prior is
keyed by sequence since 25 Oct). Fixed rules on the dcm2niix sidecar (DICOM header fields),
the series description as a secondary cue (the protocol name only when there is no series description: it names the whole exam). No image is read.

Families: T2W, T1pre, T1post, T1 (contrast status unknown), ADC, DWI, STIR, SSFP, SUB (subtraction), other.
CT series are 'CT'.
Rules (in order):
  ADC     ImageType contains ADC, or the description contains 'adc'
  SUB     ImageType contains SUB, or the description contains 'sub'
  DWI     echo-planar (ScanningSequence EP) or the description contains diff / dwi / trace
  STIR    inversion recovery (IR) with TI < 0.3 s, or the description contains 'stir'
  SSFP    balanced steady state: description contains fiesta / bffe / truefisp / trufi, or (GR, TR < 5 ms, TE < 2 ms and 'SS' only)
  T2W     TE >= 60 ms
  T1*     TE < 30 ms (inversion recovery allowed only with TI >= 0.6 s): T1post if the description names contrast or a phase
          (post, +c, c+, gd, gad, contrast, arterial, portal, venous, delay, dyn, equil, eovist, primovist, '<n> min'),
          T1pre if it contains 'pre', otherwise T1
  other   everything else (including TE 30-60 ms)"""
import re

EXPERT_FAMILIES = ('T2W', 'T1post', 'ADC')
_POST = re.compile(r'(post(?!erior)|\+c|c\+|\bgd\b|gad|contrast|arterial|portal|venous|delay|dyn|equil|eovist|primovist|\d+\s*min)', re.I)


def classify(meta, modality='MR'):
    """meta: dict of the sidecar (may be empty). Returns (family, reason)."""
    if str(modality).upper().startswith('CT'):
        return 'CT', 'CT'
    d = meta or {}
    desc = str(d.get('SeriesDescription') or d.get('ProtocolName') or '').lower()   # protocol names describe the exam (e.g. 'PROSTATE WITH DIFF')
    it = [str(x).upper() for x in (d.get('ImageType') or [])]
    ss = str(d.get('ScanningSequence') or '').upper()
    te = d.get('EchoTime'); tr = d.get('RepetitionTime'); ti = d.get('InversionTime')
    te = float(te) if te is not None else None; tr = float(tr) if tr is not None else None; ti = float(ti) if ti is not None else None
    if 'ADC' in it or 'adc' in desc:
        return 'ADC', 'image type / description'
    if 'SUB' in it or re.search(r'\bsub\b|_sub\b|subtract', desc):
        return 'SUB', 'image type / description'
    if 'EP' in ss.split('\\') or re.search(r'diff|dwi|trace', desc):
        return 'DWI', 'echo planar / description'
    if ('IR' in ss.split('\\') and ti is not None and ti < 0.3) or 'stir' in desc:
        return 'STIR', 'short-TI inversion recovery'
    if re.search(r'fiesta|bffe|truefisp|trufi', desc) or (ss == 'GR' and tr is not None and te is not None and tr < 0.005 and te < 0.002
                                                          and str(d.get('SequenceVariant') or '').upper() == 'SS'):
        return 'SSFP', 'balanced SSFP'
    if te is None:
        return 'other', 'no echo time'
    if te >= 0.06:
        return 'T2W', f'TE {te * 1000:.0f} ms'
    if te < 0.03 and not ('IR' in ss.split('\\') and (ti is None or ti < 0.6)):
        if _POST.search(desc):
            return 'T1post', 'T1 (short TE), contrast in description'
        if re.search(r'\bpre\b|_pre\b|pre[_ ]', desc):
            return 'T1pre', 'T1 (short TE), pre in description'
        return 'T1', 'T1 (short TE), contrast status unknown'
    return 'other', f'TE {te * 1000:.0f} ms'


def r2seg_family(collection, image_name):
    """sequence family of an R2Seg expert image (from datainfo.xlsx: modality and sequence per collection)"""
    if collection == 'Prostate_Tumor_00':
        return 'ADC' if image_name.endswith('_adc') else 'T2W'
    return {'Bladder_Tumor_00': 'T2W', 'Breast_Tumor_00': 'T1post', 'Cervix_Tumor_00': 'T2W', 'Uterus_Tumor_00': 'T2W'}.get(collection, 'CT')
