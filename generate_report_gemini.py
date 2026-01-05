#!/usr/bin/env python3
"""
Generate MRI Radiology Report using Template-Driven Approach

This script uses a TEMPLATE-DRIVEN method where:
1. A rigid, human-written template defines the report structure
2. Rule-based sentence generators fill each placeholder
3. Gemini API is optionally used ONLY for edge cases or refinements

The template is 100% controlled - the LLM cannot modify structure.

Usage:
    python generate_report_gemini.py <case_folder>
    python generate_report_gemini.py results/BraTS-GLI-00009-000
    python generate_report_gemini.py results/BraTS-GLI-00009-000 --use-llm  # Enable LLM refinement
"""

# Suppress deprecation warnings from google packages before importing them
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="google")
warnings.filterwarnings("ignore", message=".*google.generativeai.*")

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Import template-based report generation
from report_templates import (
    ReportTemplateFiller,
    generate_report_from_summary,
    generate_report_simple,
    MRI_BRAIN_TEMPLATE,
    SLOT_SPECIFICATIONS,
    SlotValidator,
    FactExtractor,
    FactsToSlotMapper,
    ConstrainedLLMFiller,
)

# ============================================================================
# ADD YOUR GEMINI API KEY HERE (only needed if using --use-llm)
# ============================================================================
GEMINI_API_KEY = "AIzaSyDvj-ZfCCJv2tuzydb3Zh9T6Qk5hm97P0s"
# ============================================================================

# Flag for Gemini availability
GEMINI_AVAILABLE = False
try:
    # Suppress the deprecation warning during import
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    pass  # Gemini not required for template-based generation


# ============================================================================
# LLM REFINEMENT PROMPT (only used with --use-llm flag)
# ============================================================================
# This prompt is used ONLY for optional refinement of the template-generated report.
# The LLM is NOT allowed to change the structure - only improve phrasing.

LLM_REFINEMENT_PROMPT = """You are a medical editor reviewing an automatically generated radiology report.

Your task is to REFINE the report for better readability while following these STRICT RULES:

WHAT YOU CAN DO:
- Improve sentence flow and readability
- Fix grammatical issues
- Make phrasing more natural and clinical

WHAT YOU CANNOT DO:
- Add new information not present in the original
- Remove any information from the original
- Change the report structure or section order
- Add new sections or headings
- Change any measurements or values
- Add diagnostic conclusions not present in the original
- Modify the disclaimer

The template structure is FIXED. Your refinements must preserve:
1. All section headings exactly as they appear
2. All measurements and values
3. All clinical findings
4. The exact disclaimer text
5. The overall report structure

Return ONLY the refined report text, nothing else."""

# Clinically significant thresholds (now handled in report_templates.py)
MIDLINE_SHIFT_THRESHOLD_MM = 2.0


def load_summary(case_folder: Path) -> dict:
    """Load the LLM-ready summary JSON file."""
    summary_path = case_folder / "feature_extraction" / "llm_ready_summary.json"
    
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")
    
    with open(summary_path, 'r') as f:
        return json.load(f)


def generate_template_report(summary: dict) -> tuple:
    """
    Generate the radiology report using the 4-STEP TEMPLATE-DRIVEN approach.
    
    Pipeline:
        Step 1: Rigid template (MRI_BRAIN_TEMPLATE) - human-written
        Step 2: Slot specifications with constraints
        Step 3: FactExtractor - model outputs → structured facts (NO LLM)
        Step 4: FactsToSlotMapper - facts → slot values (deterministic)
    
    Returns:
        Tuple of (report_string, validation_log, extracted_facts)
    """
    return generate_report_from_summary(summary, validate=True)


def refine_with_llm(report: str, api_key: str) -> str:
    """
    Optionally refine the template-generated report using Gemini.
    
    This is ONLY for improving readability - not changing content.
    The LLM cannot modify the structure or add new information.
    """
    if not GEMINI_AVAILABLE:
        print("Warning: Gemini not available for refinement. Using template output.")
        return report
    
    # Configure Gemini
    genai.configure(api_key=api_key)
    
    # Create the model
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        system_instruction=LLM_REFINEMENT_PROMPT
    )
    
    prompt = f"""Please refine the following radiology report for better readability.
Remember: DO NOT change the structure, add information, or modify any values.

REPORT TO REFINE:
{report}

Return only the refined report:"""
    
    print("Refining report with Gemini API (optional)...")
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.2,  # Very low temperature for minimal changes
            max_output_tokens=4096,
        )
    )
    
    return response.text


def save_report(report: str, case_folder: Path, case_id: str, method: str = "template"):
    """Save the generated report to files."""
    output_folder = case_folder / "feature_extraction"
    
    # Save as text file
    report_path = output_folder / "radiology_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Report saved to: {report_path}")
    
    # Save as JSON with metadata
    report_json = {
        "case_id": case_id,
        "generated_at": datetime.now().isoformat(),
        "generation_method": method,
        "template_version": "1.0",
        "report": report
    }
    json_path = output_folder / "radiology_report.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report_json, f, indent=2)
    print(f"JSON saved to: {json_path}")
    
    return report_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate MRI radiology report using template-driven approach"
    )
    parser.add_argument(
        "case_folder",
        type=str,
        help="Path to the case results folder (e.g., results/BraTS-GLI-00009-000)"
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Optionally refine report with Gemini LLM (not required)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Gemini API key (only needed with --use-llm)"
    )
    
    args = parser.parse_args()
    
    # Resolve case folder path
    case_folder = Path(args.case_folder)
    if not case_folder.is_absolute():
        case_folder = Path(__file__).parent / case_folder
    
    if not case_folder.exists():
        print(f"Error: Case folder not found: {case_folder}")
        sys.exit(1)
    
    case_id = case_folder.name
    
    print("=" * 70)
    print("TEMPLATE-DRIVEN RADIOLOGY REPORT GENERATOR")
    print("=" * 70)
    print(f"\nCase ID: {case_id}")
    print(f"Case folder: {case_folder}")
    print(f"Method: Template-driven (LLM refinement: {'enabled' if args.use_llm else 'disabled'})")
    
    # Load summary
    print("\nLoading analysis summary...")
    try:
        summary = load_summary(case_folder)
        print(f"Loaded summary for: {summary.get('case_id', 'Unknown')}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nMake sure you have run the full pipeline first:")
        print("  python run_full_pipeline.py <case_folder>")
        sys.exit(1)
    
    # Generate report using 4-step template pipeline
    print("\n" + "-" * 50)
    print("4-STEP TEMPLATE PIPELINE")
    print("-" * 50)
    print("Step 1: Rigid Template (human-written)")
    print("Step 2: Slot Specifications (constraints)")
    print("Step 3: Fact Extraction (deterministic, no LLM)")
    print("Step 4: Facts → Slot Values (deterministic)")
    print("-" * 50)
    
    print(f"\nSlot specifications: {len(SLOT_SPECIFICATIONS)} slots defined")
    report, validation_log, facts = generate_template_report(summary)
    method = "template (4-step pipeline)"
    
    # Show extracted facts summary
    print(f"\nStep 3 - Extracted facts:")
    print(f"  Lesion count: {facts.get('lesion_count', 'N/A')}")
    print(f"  Hemisphere: {facts.get('hemisphere', 'N/A')}")
    print(f"  Size: {facts.get('size_cm', 'N/A')} cm")
    print(f"  Edema degree: {facts.get('edema_degree', 'N/A')}")
    print(f"  Ring-enhancing: {facts.get('is_ring_enhancing', 'N/A')}")
    
    # Report validation results
    if validation_log:
        print(f"\n⚠ Validation found {len(validation_log)} issues (auto-corrected):")
        for entry in validation_log:
            print(f"  - {entry['slot']}: {entry['violations']}")
    else:
        print("\n✓ All slots passed validation")
    
    # Optionally refine with LLM
    if args.use_llm:
        api_key = args.api_key or os.environ.get("GEMINI_API_KEY") or GEMINI_API_KEY
        
        if api_key == "YOUR_API_KEY_HERE" or not api_key:
            print("\nWarning: No API key provided for LLM refinement.")
            print("Using template output without refinement.")
        elif not GEMINI_AVAILABLE:
            print("\nWarning: google-generativeai not installed.")
            print("Install with: pip install google-generativeai")
            print("Using template output without refinement.")
        else:
            try:
                report = refine_with_llm(report, api_key)
                method = "template+llm"
            except Exception as e:
                print(f"\nWarning: LLM refinement failed: {e}")
                print("Using template output without refinement.")
    
    # Save report
    print("\nSaving report...")
    report_path = save_report(report, case_folder, case_id, method)
    
    print("\n" + "=" * 70)
    print("REPORT GENERATED SUCCESSFULLY")
    print("=" * 70)
    print(f"\nMethod: {method}")
    print(f"Slots validated: {len(SLOT_SPECIFICATIONS)}")
    print(f"Output: {report_path}")
    
    # Print preview
    print("\n" + "-" * 70)
    print("REPORT PREVIEW:")
    print("-" * 70)
    preview_lines = report.split('\n')[:35]
    print('\n'.join(preview_lines))
    if len(report.split('\n')) > 35:
        print("\n... [truncated for preview] ...")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
