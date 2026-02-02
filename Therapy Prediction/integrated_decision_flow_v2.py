#!/usr/bin/env python3
"""
integrated_decision_flow_v2.py

INTEGRATED PROGRESSION DETECTION PIPELINE (Using Trained Models)
=================================================================

This version uses the actual trained model outputs from each level,
rather than rule-based heuristics.

Decision Flow Logic:
1. Run Level 1 → Generate preliminary signal (from trained model)
2. Run Level 2 → Replace L1 decision if region evidence contradicts
3. If patient has ≥3 scans:
   - Run Level 3 (from trained model)
   - If trends AGREE with L2 → INCREASE confidence
   - If trends CONFLICT → DECREASE confidence or mark "uncertain"

KEY RULES:
- Level 2 OVERRIDES Level 1
- Level 3 NEVER overrides decisions, only adjusts confidence
"""

import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent

# Results from trained models
L1_SUMMARIES = BASE_DIR / "level1_progression_results" / "clinical_summaries.csv"
L2_SUMMARIES = BASE_DIR / "level2_progression_results" / "clinical_summaries.csv"
L2_FEATURES = BASE_DIR / "level2_progression_results" / "level2_feature_table.csv"
L3_SUMMARIES = BASE_DIR / "level3_progression_results" / "clinical_summaries.csv"
L3_FEATURES = BASE_DIR / "level3_progression_results" / "level3_temporal_features.csv"

OUTPUT_DIR = BASE_DIR / "integrated_results"

# Confidence adjustment parameters
CONFIDENCE_BOOST_AGREE = 0.12       # Increase when L3 agrees with L2
CONFIDENCE_PENALTY_CONFLICT = 0.18  # Decrease when L3 conflicts
UNCERTAINTY_THRESHOLD = 0.40        # Below this, mark as "uncertain"


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class DecisionSource(Enum):
    LEVEL_1 = "Level 1 (Volume)"
    LEVEL_2 = "Level 2 (Region)"
    LEVEL_3_ADJUSTED = "Level 2 + Level 3 Adjustment"


@dataclass
class LevelPrediction:
    """Prediction from a trained model."""
    prediction: int          # 0 or 1
    confidence: float        # Probability
    explanation: str         # Model's explanation


@dataclass
class IntegratedDecision:
    """Final integrated decision."""
    patient_id: str
    true_label: int
    
    # Level predictions
    l1_pred: Optional[int]
    l1_conf: Optional[float]
    l2_pred: Optional[int]
    l2_conf: Optional[float]
    l3_pred: Optional[int]
    l3_conf: Optional[float]
    
    # Final decision
    final_pred: int
    final_label: str
    final_confidence: float
    source: str
    
    # Flow flags
    l2_overrode_l1: bool
    l3_agreement: str  # "agree", "conflict", "N/A"
    is_uncertain: bool
    
    # Trace
    decision_trace: str
    explanation: str


# ============================================================================
# LOAD TRAINED MODEL PREDICTIONS
# ============================================================================

def load_all_predictions() -> Tuple[Dict, Dict, Dict]:
    """Load predictions from all levels' trained models."""
    
    print("\n" + "="*70)
    print("LOADING TRAINED MODEL PREDICTIONS")
    print("="*70)
    
    l1_preds = {}
    l2_preds = {}
    l3_preds = {}
    
    # ─────────────────────────────────────────────────────────────────────
    # Level 1: Load from clinical summaries (if available)
    # ─────────────────────────────────────────────────────────────────────
    
    if L1_SUMMARIES.exists():
        l1_df = pd.read_csv(L1_SUMMARIES)
        for _, row in l1_df.iterrows():
            patient_id = row['patient_id']
            l1_preds[patient_id] = {
                'prediction': 1 if row['prediction'] == 'Progression' else 0,
                'confidence': row['confidence'],
                'true_label': 1 if row['true_label'] == 'Progression' else 0,
                'explanation': row.get('explanation', 'Volume-based prediction')
            }
        print(f"✓ Level 1: Loaded {len(l1_preds)} patient predictions")
    else:
        print(f"⚠ Level 1 summaries not found at {L1_SUMMARIES}")
    
    # ─────────────────────────────────────────────────────────────────────
    # Level 2: Load from clinical summaries (trained model output)
    # ─────────────────────────────────────────────────────────────────────
    
    if L2_SUMMARIES.exists():
        l2_df = pd.read_csv(L2_SUMMARIES)
        
        # L2 summaries are at scan-pair level, aggregate to patient level
        # Use majority vote or any-progression rule
        for patient_id in l2_df['patient_id'].unique():
            patient_rows = l2_df[l2_df['patient_id'] == patient_id]
            
            # Get predictions and labels
            preds = [1 if p == 'Progression' else 0 for p in patient_rows['prediction']]
            labels = [1 if p == 'Progression' else 0 for p in patient_rows['true_label']]
            confs = patient_rows['confidence'].tolist()
            
            # Patient-level: any progression prediction = progression
            # Or use majority vote weighted by confidence
            if sum(preds) > len(preds) / 2:
                patient_pred = 1
                patient_conf = np.mean([c for p, c in zip(preds, confs) if p == 1])
            elif sum(preds) > 0:
                # Some pairs predicted progression
                patient_pred = 1  # Conservative: any progression
                patient_conf = max([c for p, c in zip(preds, confs) if p == 1]) * 0.9
            else:
                patient_pred = 0
                patient_conf = np.mean(confs)
            
            # True label: any PD = progression
            patient_true_label = 1 if any(labels) else 0
            
            # Get explanation from the most recent pair
            last_expl = patient_rows.iloc[-1].get('explanation', 'Region-based prediction')
            
            l2_preds[patient_id] = {
                'prediction': patient_pred,
                'confidence': patient_conf,
                'true_label': patient_true_label,
                'explanation': last_expl
            }
        
        print(f"✓ Level 2: Loaded {len(l2_preds)} patient predictions (from trained model)")
    
    elif L2_FEATURES.exists():
        l2_df = pd.read_csv(L2_FEATURES)
        
        # For Level 2, we need to aggregate scan-pairs to patient level
        # Use the most informative pair (largest absolute change) for prediction
        for patient_id in l2_df['Patient_ID'].unique():
            patient_pairs = l2_df[l2_df['Patient_ID'] == patient_id]
            
            # Get true label (any PD = progression)
            true_label = patient_pairs['Progression_Label'].max()
            
            # Aggregate features for prediction
            # Use total changes and composition shifts
            total_tc_change = patient_pairs['Delta_V_TC'].sum()
            total_wt_change = patient_pairs['Delta_V_WT'].sum()
            max_tc_frac_change = patient_pairs['Delta_TC_fraction'].max()
            any_new_tc = patient_pairs['newly_appeared_TC'].max()
            
            # Simple L2 prediction logic based on region features
            if total_tc_change > 3.0 or any_new_tc:
                pred = 1
                conf = min(0.6 + abs(total_tc_change) * 0.02, 0.95)
                expl = f"TC change: {total_tc_change:+.1f}ml"
            elif total_wt_change > 5.0:
                pred = 1
                conf = min(0.55 + abs(total_wt_change) * 0.01, 0.90)
                expl = f"WT change: {total_wt_change:+.1f}ml"
            elif max_tc_frac_change > 0.1:
                pred = 1
                conf = 0.60
                expl = f"TC fraction increased {max_tc_frac_change*100:+.0f}pp"
            elif total_tc_change < -3.0:
                pred = 0
                conf = min(0.6 + abs(total_tc_change) * 0.02, 0.90)
                expl = f"TC decreased: {total_tc_change:.1f}ml (response)"
            else:
                pred = 0
                conf = 0.55
                expl = "Regions stable"
            
            l2_preds[patient_id] = {
                'prediction': pred,
                'confidence': conf,
                'true_label': true_label,
                'explanation': expl,
                'total_tc_change': total_tc_change,
                'total_wt_change': total_wt_change
            }
        
        print(f"✓ Level 2: Aggregated {len(l2_preds)} patient predictions")
    else:
        print(f"⚠ Level 2 features not found at {L2_FEATURES}")
    
    # ─────────────────────────────────────────────────────────────────────
    # Level 3: Load from clinical summaries (trained model output)
    # ─────────────────────────────────────────────────────────────────────
    
    if L3_SUMMARIES.exists():
        l3_df = pd.read_csv(L3_SUMMARIES)
        for _, row in l3_df.iterrows():
            patient_id = row['patient_id']
            l3_preds[patient_id] = {
                'prediction': 1 if row['prediction'] == 'Progression' else 0,
                'confidence': row['confidence'],
                'true_label': 1 if row['true_label'] == 'Progression' else 0,
                'explanation': row.get('explanation', 'Temporal-based prediction'),
                'n_scans': row.get('n_scans', 3)
            }
        print(f"✓ Level 3: Loaded {len(l3_preds)} patient predictions (≥3 scans)")
    else:
        print(f"⚠ Level 3 summaries not found at {L3_SUMMARIES}")
    
    return l1_preds, l2_preds, l3_preds


# ============================================================================
# INTEGRATED DECISION FLOW
# ============================================================================

def run_decision_flow(
    patient_id: str,
    l1_data: Optional[Dict],
    l2_data: Optional[Dict],
    l3_data: Optional[Dict]
) -> IntegratedDecision:
    """
    Execute the integrated decision flow for one patient.
    
    Flow:
    1. L1 → preliminary signal
    2. L2 → override L1 if contradicts (L2 becomes current decision)
    3. L3 → adjust confidence only (never override)
    """
    
    trace = []
    
    # Get ground truth
    true_label = None
    if l3_data:
        true_label = l3_data['true_label']
    elif l2_data:
        true_label = l2_data['true_label']
    elif l1_data:
        true_label = l1_data['true_label']
    
    # =========================================================================
    # STEP 1: Level 1 - Preliminary Signal
    # =========================================================================
    
    l1_pred = l1_conf = None
    current_pred = None
    current_conf = None
    current_source = None
    
    if l1_data:
        l1_pred = l1_data['prediction']
        l1_conf = l1_data['confidence']
        current_pred = l1_pred
        current_conf = l1_conf
        current_source = "Level 1"
        
        trace.append(f"Step 1: L1 → {'PROG' if l1_pred else 'NON-PROG'} ({l1_conf:.0%})")
    else:
        trace.append("Step 1: L1 → No data")
    
    # =========================================================================
    # STEP 2: Level 2 - Override Check
    # =========================================================================
    
    l2_pred = l2_conf = None
    l2_overrode_l1 = False
    
    if l2_data:
        l2_pred = l2_data['prediction']
        l2_conf = l2_data['confidence']
        
        trace.append(f"Step 2: L2 → {'PROG' if l2_pred else 'NON-PROG'} ({l2_conf:.0%})")
        trace.append(f"        [{l2_data['explanation']}]")
        
        # Check for override
        if l1_pred is not None and l2_pred != l1_pred:
            l2_overrode_l1 = True
            trace.append(f"        ⚡ L2 OVERRIDES L1!")
        
        # L2 becomes current decision
        current_pred = l2_pred
        current_conf = l2_conf
        current_source = "Level 2"
    else:
        trace.append("Step 2: L2 → No data")
        if current_pred is None:
            # No L1 or L2 - insufficient data
            return IntegratedDecision(
                patient_id=patient_id,
                true_label=true_label or 0,
                l1_pred=l1_pred, l1_conf=l1_conf,
                l2_pred=l2_pred, l2_conf=l2_conf,
                l3_pred=None, l3_conf=None,
                final_pred=0,
                final_label="INSUFFICIENT DATA",
                final_confidence=0.0,
                source="None",
                l2_overrode_l1=False,
                l3_agreement="N/A",
                is_uncertain=True,
                decision_trace="\n".join(trace),
                explanation="Insufficient data"
            )
    
    # =========================================================================
    # STEP 3: Level 3 - Confidence Adjustment (NEVER override)
    # =========================================================================
    
    l3_pred = l3_conf = None
    l3_agreement = "N/A"
    
    if l3_data:
        l3_pred = l3_data['prediction']
        l3_conf = l3_data['confidence']
        n_scans = l3_data.get('n_scans', 3)
        
        trace.append(f"Step 3: L3 → {'PROG' if l3_pred else 'NON-PROG'} ({l3_conf:.0%}, {n_scans} scans)")
        
        if l3_pred == current_pred:
            # AGREE - boost confidence
            l3_agreement = "agree"
            boost = CONFIDENCE_BOOST_AGREE * l3_conf
            new_conf = min(current_conf + boost, 0.98)
            trace.append(f"        ✓ L3 AGREES → Confidence: {current_conf:.0%} → {new_conf:.0%}")
            current_conf = new_conf
            current_source = "L2 + L3 (agree)"
        else:
            # CONFLICT - reduce confidence but DO NOT change decision
            l3_agreement = "conflict"
            penalty = CONFIDENCE_PENALTY_CONFLICT * l3_conf
            new_conf = max(current_conf - penalty, 0.30)
            trace.append(f"        ⚠ L3 CONFLICTS → Confidence: {current_conf:.0%} → {new_conf:.0%}")
            trace.append(f"        ⚠ Decision UNCHANGED (L2 stands)")
            current_conf = new_conf
            current_source = "L2 + L3 (conflict)"
    else:
        trace.append("Step 3: L3 → No temporal data (<3 scans)")
    
    # =========================================================================
    # FINAL DECISION
    # =========================================================================
    
    is_uncertain = current_conf < UNCERTAINTY_THRESHOLD
    
    if is_uncertain:
        final_label = "UNCERTAIN"
        trace.append(f"\n⚠ FINAL: UNCERTAIN (conf {current_conf:.0%} < {UNCERTAINTY_THRESHOLD:.0%})")
    else:
        final_label = "PROGRESSION" if current_pred == 1 else "NON-PROGRESSION"
        trace.append(f"\n✓ FINAL: {final_label} ({current_conf:.0%})")
    
    # Build explanation
    expl_parts = [f"Classified as {final_label} ({current_conf:.0%} confidence)."]
    if l2_data:
        expl_parts.append(f"Region evidence: {l2_data['explanation']}.")
    if l2_overrode_l1:
        expl_parts.append("L2 overrode initial L1 signal.")
    if l3_agreement == "agree":
        expl_parts.append("Temporal trends confirm this assessment.")
    elif l3_agreement == "conflict":
        expl_parts.append("NOTE: Temporal trends show different pattern - review recommended.")
    
    return IntegratedDecision(
        patient_id=patient_id,
        true_label=true_label or 0,
        l1_pred=l1_pred, l1_conf=l1_conf,
        l2_pred=l2_pred, l2_conf=l2_conf,
        l3_pred=l3_pred, l3_conf=l3_conf,
        final_pred=current_pred,
        final_label=final_label,
        final_confidence=current_conf,
        source=current_source,
        l2_overrode_l1=l2_overrode_l1,
        l3_agreement=l3_agreement,
        is_uncertain=is_uncertain,
        decision_trace="\n".join(trace),
        explanation=" ".join(expl_parts)
    )


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def process_all_patients(l1_preds, l2_preds, l3_preds) -> List[IntegratedDecision]:
    """Process all patients through the decision flow."""
    
    print("\n" + "="*70)
    print("RUNNING INTEGRATED DECISION FLOW")
    print("="*70)
    
    # Get all unique patients
    all_patients = set(l1_preds.keys()) | set(l2_preds.keys()) | set(l3_preds.keys())
    print(f"\nProcessing {len(all_patients)} patients...")
    
    results = []
    
    for patient_id in sorted(all_patients):
        l1_data = l1_preds.get(patient_id)
        l2_data = l2_preds.get(patient_id)
        l3_data = l3_preds.get(patient_id)
        
        result = run_decision_flow(patient_id, l1_data, l2_data, l3_data)
        results.append(result)
    
    return results


def evaluate_results(results: List[IntegratedDecision]) -> Dict:
    """Evaluate integrated predictions against ground truth."""
    
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    # Filter out uncertain and insufficient data
    evaluated = [r for r in results if not r.is_uncertain and r.final_label != "INSUFFICIENT DATA"]
    uncertain = [r for r in results if r.is_uncertain]
    
    if len(evaluated) == 0:
        print("⚠ No patients to evaluate!")
        return {}
    
    # Calculate accuracy
    correct = sum(1 for r in evaluated if r.final_pred == r.true_label)
    incorrect = len(evaluated) - correct
    accuracy = correct / len(evaluated)
    
    # Count flow behaviors
    l2_overrides = sum(1 for r in results if r.l2_overrode_l1)
    l3_agrees = sum(1 for r in results if r.l3_agreement == "agree")
    l3_conflicts = sum(1 for r in results if r.l3_agreement == "conflict")
    
    # Breakdown by true label
    prog_correct = sum(1 for r in evaluated if r.true_label == 1 and r.final_pred == 1)
    prog_total = sum(1 for r in evaluated if r.true_label == 1)
    nonprog_correct = sum(1 for r in evaluated if r.true_label == 0 and r.final_pred == 0)
    nonprog_total = sum(1 for r in evaluated if r.true_label == 0)
    
    print(f"\n📊 Summary:")
    print(f"  Total patients: {len(results)}")
    print(f"  Evaluated: {len(evaluated)}")
    print(f"  Uncertain: {len(uncertain)}")
    
    print(f"\n✅ Accuracy: {correct}/{len(evaluated)} = {accuracy:.1%}")
    if prog_total > 0:
        print(f"  Progression recall: {prog_correct}/{prog_total} = {100*prog_correct/prog_total:.1f}%")
    if nonprog_total > 0:
        print(f"  Non-progression specificity: {nonprog_correct}/{nonprog_total} = {100*nonprog_correct/nonprog_total:.1f}%")
    
    print(f"\n📈 Decision Flow Behaviors:")
    print(f"  L2 overrode L1: {l2_overrides} cases")
    print(f"  L3 agreed with L2: {l3_agrees} cases")
    print(f"  L3 conflicted with L2: {l3_conflicts} cases")
    
    # Check if override helped
    override_cases = [r for r in evaluated if r.l2_overrode_l1]
    if override_cases:
        override_correct = sum(1 for r in override_cases if r.final_pred == r.true_label)
        print(f"  L2 override accuracy: {override_correct}/{len(override_cases)}")
    
    return {
        'total': len(results),
        'evaluated': len(evaluated),
        'uncertain': len(uncertain),
        'correct': correct,
        'incorrect': incorrect,
        'accuracy': accuracy,
        'l2_overrides': l2_overrides,
        'l3_agrees': l3_agrees,
        'l3_conflicts': l3_conflicts
    }


def print_sample_traces(results: List[IntegratedDecision], n: int = 6):
    """Print sample decision traces showing variety of behaviors."""
    
    print("\n" + "="*70)
    print("SAMPLE DECISION TRACES")
    print("="*70)
    
    samples = []
    
    # Get diverse examples
    # 1. L2 override case
    for r in results:
        if r.l2_overrode_l1 and len(samples) < 1:
            samples.append(("L2 Override Example", r))
    
    # 2. L3 conflict case
    for r in results:
        if r.l3_agreement == "conflict" and len([s for s in samples if s[0] == "L3 Conflict Example"]) < 1:
            samples.append(("L3 Conflict Example", r))
    
    # 3. L3 agree case
    for r in results:
        if r.l3_agreement == "agree" and len([s for s in samples if s[0] == "L3 Agreement Example"]) < 1:
            samples.append(("L3 Agreement Example", r))
    
    # 4. Uncertain case
    for r in results:
        if r.is_uncertain and len([s for s in samples if s[0] == "Uncertain Example"]) < 1:
            samples.append(("Uncertain Example", r))
    
    # 5. Correct prediction
    for r in results:
        if not r.is_uncertain and r.final_pred == r.true_label and len(samples) < n:
            if r not in [s[1] for s in samples]:
                samples.append(("Correct Prediction", r))
    
    # 6. Fill remaining
    for r in results[:n]:
        if len(samples) >= n:
            break
        if r not in [s[1] for s in samples]:
            samples.append(("Standard Case", r))
    
    for label, r in samples[:n]:
        correct_str = "✓" if r.final_pred == r.true_label else "✗"
        true_str = "PROG" if r.true_label == 1 else "NON-PROG"
        
        print(f"\n{'━'*70}")
        print(f"[{label}] Patient: {r.patient_id}  |  True: {true_str}  |  {correct_str}")
        print(f"{'━'*70}")
        print(r.decision_trace)


def save_results(results: List[IntegratedDecision], metrics: Dict):
    """Save all results to files."""
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Convert to DataFrame
    rows = []
    for r in results:
        rows.append({
            'Patient_ID': r.patient_id,
            'True_Label': r.true_label,
            'L1_Pred': r.l1_pred,
            'L1_Conf': r.l1_conf,
            'L2_Pred': r.l2_pred,
            'L2_Conf': r.l2_conf,
            'L3_Pred': r.l3_pred,
            'L3_Conf': r.l3_conf,
            'Final_Pred': r.final_pred,
            'Final_Label': r.final_label,
            'Final_Confidence': r.final_confidence,
            'Source': r.source,
            'L2_Overrode_L1': r.l2_overrode_l1,
            'L3_Agreement': r.l3_agreement,
            'Is_Uncertain': r.is_uncertain,
            'Correct': r.final_pred == r.true_label if not r.is_uncertain else None,
            'Explanation': r.explanation
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "integrated_decisions_v2.csv", index=False)
    print(f"\n✓ Saved: {OUTPUT_DIR / 'integrated_decisions_v2.csv'}")
    
    # Save traces
    traces = [{'Patient_ID': r.patient_id, 'Trace': r.decision_trace} for r in results]
    pd.DataFrame(traces).to_csv(OUTPUT_DIR / "decision_traces_v2.csv", index=False)
    print(f"✓ Saved: {OUTPUT_DIR / 'decision_traces_v2.csv'}")
    
    # Save metrics
    pd.DataFrame([metrics]).to_csv(OUTPUT_DIR / "metrics_v2.csv", index=False)
    print(f"✓ Saved: {OUTPUT_DIR / 'metrics_v2.csv'}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution."""
    
    print("\n" + "="*70)
    print("INTEGRATED PROGRESSION DETECTION PIPELINE v2")
    print("="*70)
    print("""
    Decision Flow (Trained Models):
    ════════════════════════════════════════════════════
    
    ┌─────────────┐         ┌─────────────┐         ┌─────────────┐
    │   LEVEL 1   │  ─────► │   LEVEL 2   │  ─────► │   LEVEL 3   │
    │   Volume    │         │   Region    │         │  Temporal   │
    │  (trained)  │         │ (aggregated)│         │  (trained)  │
    └─────────────┘         └─────────────┘         └─────────────┘
          │                       │                       │
          ▼                       ▼                       ▼
     Preliminary             OVERRIDES L1            ADJUSTS CONF
       Signal              if contradicts          (never overrides)
    
    ════════════════════════════════════════════════════
    """)
    
    # Load predictions
    l1_preds, l2_preds, l3_preds = load_all_predictions()
    
    if not l2_preds:
        print("❌ Level 2 predictions required. Run level2_region_aware_progression.py first.")
        return
    
    # Process all patients
    results = process_all_patients(l1_preds, l2_preds, l3_preds)
    
    # Print sample traces
    print_sample_traces(results)
    
    # Evaluate
    metrics = evaluate_results(results)
    
    # Save
    save_results(results, metrics)
    
    # Final summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print(f"""
    ✅ Processed: {metrics.get('total', 0)} patients
    ✅ Evaluated: {metrics.get('evaluated', 0)} patients  
    ✅ Accuracy:  {metrics.get('accuracy', 0):.1%}
    
    Decision Flow Verified:
    ───────────────────────────────────────
    ✓ L1 provides preliminary signal
    ✓ L2 overrides L1: {metrics.get('l2_overrides', 0)} cases
    ✓ L3 adjusts confidence only: {metrics.get('l3_agrees', 0) + metrics.get('l3_conflicts', 0)} cases
      - Agreed (boosted): {metrics.get('l3_agrees', 0)}
      - Conflicted (reduced): {metrics.get('l3_conflicts', 0)}
    ✓ Uncertain cases flagged: {metrics.get('uncertain', 0)}
    """)


if __name__ == "__main__":
    main()
