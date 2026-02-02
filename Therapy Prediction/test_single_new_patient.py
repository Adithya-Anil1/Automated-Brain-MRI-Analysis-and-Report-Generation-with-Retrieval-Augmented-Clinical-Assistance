#!/usr/bin/env python3
"""
test_single_new_patient.py

Test the integrated decision flow on a SINGLE NEW PATIENT
(not necessarily in the training/test set).

This script:
1. Takes a patient directory
2. Extracts volumes from segmentation files
3. Runs through L1 → L2 → L3 decision flow
4. Shows complete reasoning
"""

import os
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import nibabel as nib

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent
IMAGING_DIR = BASE_DIR / "dataset" / "Imaging"


# ============================================================================
# VOLUME EXTRACTION
# ============================================================================

def parse_week_number(week_str: str) -> float:
    """Parse week string to numeric value."""
    match = re.search(r'week-(\d+)', week_str)
    if match:
        week_num = int(match.group(1))
        if '-' in week_str.split('week-')[1]:
            parts = week_str.split('week-')[1].split('-')
            if len(parts) > 1 and parts[1].isdigit():
                week_num += int(parts[1]) * 0.1
        return week_num
    return 0.0


def find_segmentation_file(week_dir: Path) -> Optional[Path]:
    """Find segmentation file in week directory."""
    for seg_type in ['HD-GLIO-AUTO-segmentation', 'DeepBraTumIA-segmentation']:
        seg_dir = week_dir / seg_type / 'native'
        if seg_dir.exists():
            for seq in ['CT1', 'FLAIR', 'T1']:
                seg_file = seg_dir / f'segmentation_{seq}_origspace.nii.gz'
                if seg_file.exists():
                    return seg_file
    return None


def compute_volumes(seg_path: Path) -> Dict[str, float]:
    """Compute WT, TC, ET volumes from segmentation."""
    img = nib.load(str(seg_path))
    data = img.get_fdata()
    
    voxel_dims = img.header.get_zooms()[:3]
    voxel_vol_ml = np.prod(voxel_dims) / 1000.0
    
    label_1 = np.sum(data == 1)  # NCR/NET
    label_2 = np.sum(data == 2)  # Edema
    label_3 = np.sum(data == 3)  # ET
    label_4 = np.sum(data == 4)  # ET (alt)
    
    et_voxels = label_4 if label_4 > 0 else label_3
    wt_voxels = label_1 + label_2 + et_voxels
    tc_voxels = label_1 + et_voxels
    
    return {
        'WT': wt_voxels * voxel_vol_ml,
        'TC': tc_voxels * voxel_vol_ml,
        'ET': et_voxels * voxel_vol_ml
    }


def extract_patient_scans(patient_dir: Path) -> List[Dict]:
    """Extract all scans for a patient."""
    weeks = sorted(
        [d for d in patient_dir.iterdir() if d.is_dir() and d.name.startswith('week-')],
        key=lambda x: parse_week_number(x.name)
    )
    
    scans = []
    for week_dir in weeks:
        seg_file = find_segmentation_file(week_dir)
        if seg_file is None:
            continue
        
        volumes = compute_volumes(seg_file)
        scans.append({
            'week': week_dir.name,
            'week_num': parse_week_number(week_dir.name),
            'V_WT': volumes['WT'],
            'V_TC': volumes['TC'],
            'V_ET': volumes['ET']
        })
    
    return scans


# ============================================================================
# LEVEL 1: VOLUME-BASED
# ============================================================================

def run_level1(baseline: Dict, followup: Dict) -> Dict:
    """Level 1: Simple volume-based prediction."""
    v_base = baseline['V_WT']
    v_follow = followup['V_WT']
    delta_v = v_follow - v_base
    delta_v_pct = delta_v / v_base if v_base > 0 else (1.0 if v_follow > 0 else 0.0)
    
    if delta_v_pct > 0.25 or delta_v > 5.0:
        pred = 1
        conf = min(0.5 + delta_v_pct * 0.5, 0.95)
        expl = f"Volume increased {delta_v:+.1f}ml ({delta_v_pct*100:+.0f}%)"
    elif delta_v_pct < -0.25:
        pred = 0
        conf = min(0.5 + abs(delta_v_pct) * 0.5, 0.95)
        expl = f"Volume decreased {delta_v:.1f}ml ({delta_v_pct*100:.0f}%)"
    else:
        pred = 0
        conf = 0.55
        expl = f"Volume stable ({delta_v_pct*100:+.0f}%)"
    
    return {
        'prediction': pred,
        'confidence': conf,
        'explanation': expl,
        'delta_v': delta_v,
        'delta_v_pct': delta_v_pct
    }


# ============================================================================
# LEVEL 2: REGION-AWARE
# ============================================================================

def run_level2(baseline: Dict, followup: Dict) -> Dict:
    """Level 2: Region-aware with TC/ET composition."""
    wt_base = baseline['V_WT']
    wt_follow = followup['V_WT']
    tc_base = baseline['V_TC']
    tc_follow = followup['V_TC']
    et_base = baseline['V_ET']
    et_follow = followup['V_ET']
    
    delta_wt = wt_follow - wt_base
    delta_tc = tc_follow - tc_base
    delta_et = et_follow - et_base
    
    delta_tc_pct = delta_tc / tc_base if tc_base > 0.05 else (1.0 if tc_follow > 0.05 else 0)
    
    tc_frac_base = tc_base / wt_base if wt_base > 0.05 else 0
    tc_frac_follow = tc_follow / wt_follow if wt_follow > 0.05 else 0
    delta_tc_frac = tc_frac_follow - tc_frac_base
    
    reasons = []
    prog_signals = 0
    stable_signals = 0
    
    # TC changes (most important)
    if delta_tc_pct > 0.25 or delta_tc > 3.0:
        prog_signals += 2
        reasons.append(f"TC increased {delta_tc:+.1f}ml ({delta_tc_pct*100:+.0f}%)")
    elif delta_tc < -3.0:
        stable_signals += 2
        reasons.append(f"TC decreased {abs(delta_tc):.1f}ml")
    
    # New TC appearance
    if tc_base < 0.1 and tc_follow > 0.5:
        prog_signals += 2
        reasons.append(f"TC newly appeared ({tc_follow:.1f}ml)")
    
    # ET changes
    if delta_et > 1.0 or (et_base < 0.1 and et_follow > 0.5):
        prog_signals += 1
        reasons.append(f"ET increased/appeared")
    
    # Composition shift
    if delta_tc_frac > 0.15:
        prog_signals += 1
        reasons.append(f"TC fraction +{delta_tc_frac*100:.0f}pp")
    
    # Decide
    if prog_signals > stable_signals:
        pred = 1
        score = prog_signals / (prog_signals + stable_signals + 1)
        conf = 0.5 + score * 0.4
        expl = "Region evidence: " + "; ".join(reasons[:2])
    elif stable_signals > prog_signals:
        pred = 0
        score = stable_signals / (prog_signals + stable_signals + 1)
        conf = 0.5 + score * 0.4
        expl = "Region evidence: " + "; ".join(reasons[:2]) if reasons else "Stable regions"
    else:
        pred = 0
        conf = 0.50
        expl = "Region evidence inconclusive"
    
    return {
        'prediction': pred,
        'confidence': conf,
        'explanation': expl,
        'prog_signals': prog_signals,
        'stable_signals': stable_signals
    }


# ============================================================================
# LEVEL 3: TEMPORAL
# ============================================================================

def run_level3(scans: List[Dict]) -> Optional[Dict]:
    """Level 3: Temporal consistency across multiple scans."""
    if len(scans) < 3:
        return None
    
    times = np.array([s['week_num'] for s in scans])
    wt_vols = np.array([s['V_WT'] for s in scans])
    tc_vols = np.array([s['V_TC'] for s in scans])
    
    # Calculate slope
    from scipy import stats
    if np.std(times) > 0:
        tc_slope, _, _, _, _ = stats.linregress(times, tc_vols)
        wt_slope, _, _, _, _ = stats.linregress(times, wt_vols)
    else:
        tc_slope = wt_slope = 0
    
    # Intervals
    tc_intervals = np.diff(tc_vols)
    wt_intervals = np.diff(wt_vols)
    
    # Fraction increasing
    tc_frac_inc = np.sum(tc_intervals > 0.05) / len(tc_intervals) if len(tc_intervals) > 0 else 0
    
    # Consecutive increases
    tc_consec = 0
    current_consec = 0
    for interval in tc_intervals:
        if interval > 0.05:
            current_consec += 1
            tc_consec = max(tc_consec, current_consec)
        else:
            current_consec = 0
    
    reasons = []
    prog_ev = 0
    stable_ev = 0
    
    if tc_slope > 0.1:
        prog_ev += 2
        reasons.append(f"TC slope +{tc_slope:.2f} ml/week")
    elif tc_slope < -0.1:
        stable_ev += 2
        reasons.append(f"TC slope {tc_slope:.2f} ml/week")
    
    if tc_frac_inc > 0.5:
        prog_ev += 1
        reasons.append(f"TC increasing {tc_frac_inc*100:.0f}% of intervals")
    elif tc_frac_inc < 0.3:
        stable_ev += 1
        reasons.append(f"TC only increasing {tc_frac_inc*100:.0f}% of intervals")
    
    if tc_consec >= 3:
        prog_ev += 2
        reasons.append(f"{tc_consec} consecutive TC increases")
    
    if prog_ev > stable_ev:
        pred = 1
        score = prog_ev / (prog_ev + stable_ev + 1)
        conf = 0.5 + score * 0.4
        expl = f"Temporal ({len(scans)} scans): " + "; ".join(reasons[:2])
    elif stable_ev > prog_ev:
        pred = 0
        score = stable_ev / (prog_ev + stable_ev + 1)
        conf = 0.5 + score * 0.4
        expl = f"Temporal ({len(scans)} scans): " + "; ".join(reasons[:2])
    else:
        pred = 0
        conf = 0.50
        expl = f"Temporal ({len(scans)} scans) inconclusive"
    
    return {
        'prediction': pred,
        'confidence': conf,
        'explanation': expl,
        'tc_slope': tc_slope,
        'tc_frac_inc': tc_frac_inc,
        'tc_consecutive': tc_consec
    }


# ============================================================================
# INTEGRATED DECISION
# ============================================================================

def integrate_decision(l1: Dict, l2: Dict, l3: Optional[Dict]) -> Dict:
    """Run the integrated decision flow."""
    trace = []
    
    # L1
    l1_pred = l1['prediction']
    l1_conf = l1['confidence']
    trace.append(f"L1 → {'PROG' if l1_pred else 'NON-PROG'} ({l1_conf:.0%}): {l1['explanation']}")
    
    current_pred = l1_pred
    current_conf = l1_conf
    
    # L2
    l2_pred = l2['prediction']
    l2_conf = l2['confidence']
    trace.append(f"L2 → {'PROG' if l2_pred else 'NON-PROG'} ({l2_conf:.0%}): {l2['explanation']}")
    
    l2_override = False
    if l2_pred != l1_pred:
        l2_override = True
        trace.append("    ⚡ L2 OVERRIDES L1!")
    
    current_pred = l2_pred
    current_conf = l2_conf
    
    # L3
    l3_agreement = None
    if l3:
        l3_pred = l3['prediction']
        l3_conf = l3['confidence']
        trace.append(f"L3 → {'PROG' if l3_pred else 'NON-PROG'} ({l3_conf:.0%}): {l3['explanation']}")
        
        if l3_pred == current_pred:
            l3_agreement = "agree"
            boost = 0.12 * l3_conf
            current_conf = min(current_conf + boost, 0.98)
            trace.append(f"    ✓ L3 AGREES → Confidence boosted to {current_conf:.0%}")
        else:
            l3_agreement = "conflict"
            penalty = 0.18 * l3_conf
            current_conf = max(current_conf - penalty, 0.30)
            trace.append(f"    ⚠ L3 CONFLICTS → Confidence reduced to {current_conf:.0%}")
            trace.append(f"    ⚠ Decision UNCHANGED (L2 stands)")
    else:
        trace.append("L3 → Not available (<3 scans)")
    
    is_uncertain = current_conf < 0.40
    
    return {
        'final_prediction': current_pred,
        'final_confidence': current_conf,
        'is_uncertain': is_uncertain,
        'l2_override': l2_override,
        'l3_agreement': l3_agreement,
        'trace': trace
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print("TESTING INTEGRATED DECISION FLOW ON NEW PATIENT")
    print("="*70)
    
    # List available patients
    patients = sorted([d.name for d in IMAGING_DIR.iterdir() if d.is_dir()])
    
    print(f"\nAvailable patients in dataset: {len(patients)}")
    print(f"First 10: {', '.join(patients[:10])}")
    
    # Pick a patient (user can modify this)
    # Let's pick one that's likely not in training set
    import random
    patient_id = random.choice(patients)
    
    print(f"\n🎯 Selected patient: {patient_id}")
    
    patient_dir = IMAGING_DIR / patient_id
    
    # Extract scans
    print(f"\n📊 Extracting scan data...")
    scans = extract_patient_scans(patient_dir)
    
    if len(scans) < 2:
        print(f"❌ Insufficient scans: {len(scans)} (need at least 2)")
        return
    
    print(f"✓ Found {len(scans)} scans")
    for scan in scans:
        print(f"  • {scan['week']}: WT={scan['V_WT']:.1f}ml, TC={scan['V_TC']:.1f}ml, ET={scan['V_ET']:.1f}ml")
    
    # Use last two scans for L1 and L2
    baseline = scans[-2]
    followup = scans[-1]
    
    print("\n" + "━"*70)
    print("RUNNING INTEGRATED DECISION FLOW")
    print("━"*70)
    
    # Level 1
    print("\n🔹 Level 1: Volume-based analysis")
    l1 = run_level1(baseline, followup)
    print(f"   Prediction: {'PROGRESSION' if l1['prediction'] else 'NON-PROGRESSION'}")
    print(f"   Confidence: {l1['confidence']:.1%}")
    print(f"   Reason: {l1['explanation']}")
    
    # Level 2
    print("\n🔹 Level 2: Region-aware analysis")
    l2 = run_level2(baseline, followup)
    print(f"   Prediction: {'PROGRESSION' if l2['prediction'] else 'NON-PROGRESSION'}")
    print(f"   Confidence: {l2['confidence']:.1%}")
    print(f"   Reason: {l2['explanation']}")
    if l2['prediction'] != l1['prediction']:
        print(f"   ⚡ OVERRIDES Level 1!")
    
    # Level 3
    print("\n🔹 Level 3: Temporal consistency analysis")
    l3 = run_level3(scans)
    if l3:
        print(f"   Prediction: {'PROGRESSION' if l3['prediction'] else 'NON-PROGRESSION'}")
        print(f"   Confidence: {l3['confidence']:.1%}")
        print(f"   Reason: {l3['explanation']}")
        print(f"   Temporal features:")
        print(f"     • TC slope: {l3['tc_slope']:.3f} ml/week")
        print(f"     • TC increasing: {l3['tc_frac_inc']*100:.0f}% of intervals")
        print(f"     • Consecutive increases: {l3['tc_consecutive']}")
    else:
        print(f"   Not available (only {len(scans)} scans, need ≥3)")
    
    # Integrate
    print("\n" + "━"*70)
    print("INTEGRATED DECISION")
    print("━"*70)
    
    result = integrate_decision(l1, l2, l3)
    
    print("\n📋 Decision Trace:")
    for line in result['trace']:
        print(f"   {line}")
    
    print("\n" + "═"*70)
    final_label = "PROGRESSION" if result['final_prediction'] else "NON-PROGRESSION"
    if result['is_uncertain']:
        final_label = "UNCERTAIN"
    
    print(f"🔮 FINAL: {final_label}")
    print(f"📊 Confidence: {result['final_confidence']:.1%}")
    print("═"*70)
    
    print(f"""
    Summary:
    ─────────────────────────────────────────
    Patient: {patient_id}
    Scans analyzed: {len(scans)} ({scans[0]['week']} to {scans[-1]['week']})
    
    Decision flow:
    • L1: {'PROG' if l1['prediction'] else 'NON-PROG'} ({l1['confidence']:.0%})
    • L2: {'PROG' if l2['prediction'] else 'NON-PROG'} ({l2['confidence']:.0%}) {'[OVERRIDE]' if result['l2_override'] else ''}
    • L3: {'PROG' if l3 and l3['prediction'] else 'NON-PROG' if l3 else 'N/A'} {'['+result['l3_agreement'].upper()+']' if result['l3_agreement'] else ''}
    
    Final: {final_label} ({result['final_confidence']:.0%})
    """)


if __name__ == "__main__":
    main()
