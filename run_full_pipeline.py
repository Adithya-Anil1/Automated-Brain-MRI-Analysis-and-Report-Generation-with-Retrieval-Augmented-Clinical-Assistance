#!/usr/bin/env python3
"""
Automated Brain MRI Analysis Pipeline

This script automates the complete workflow:
1. Rename BraTS 2025 files to BraTS 2021 naming convention
2. Run segmentation inference (BraTS 2021 KAIST model)
3. Convert labels to match ground truth format
4. Evaluate segmentation against ground truth
5. Run feature extraction pipeline
6. Generate Gemini radiology report
7. Generate professional PDF report
8. Launch RAG Educational Assistant (interactive Q&A)

Usage:
    python run_full_pipeline.py <case_folder>
    python run_full_pipeline.py BraTS-GLI-00003-000
    python run_full_pipeline.py C:\path\to\BraTS-GLI-00003-000

The case folder should contain BraTS 2025 format files:
    BraTS-GLI-XXXXX-XXX-t1n.nii.gz
    BraTS-GLI-XXXXX-XXX-t1c.nii.gz
    BraTS-GLI-XXXXX-XXX-t2w.nii.gz
    BraTS-GLI-XXXXX-XXX-t2f.nii.gz
    BraTS-GLI-XXXXX-XXX-seg.nii.gz (ground truth)

Note: For Gemini report generation, add your API key in generate_report_gemini.py
"""

import os
import sys
import re
import time
import json
import argparse
import subprocess
import gzip
import shutil
from pathlib import Path
from datetime import datetime

# Get the project root directory
SCRIPT_DIR = Path(__file__).parent.absolute()
PYTHON_EXE = SCRIPT_DIR / "venv310" / "Scripts" / "python.exe"

# Environment variables for nnU-Net
NNUNET_ENV = {
    "nnUNet_raw_data_base": str(SCRIPT_DIR / "nnUNet_raw"),
    "nnUNet_preprocessed": str(SCRIPT_DIR / "nnUNet_preprocessed"),
    "RESULTS_FOLDER": str(SCRIPT_DIR / "nnUNet_results"),
}

# BraTS 2025 to 2021 naming conversion
SUFFIX_MAPPING = {
    't1n': 't1',      # T1 native → t1
    't1c': 't1ce',    # T1 contrast → t1ce
    't2w': 't2',      # T2 weighted → t2
    't2f': 'flair',   # T2 FLAIR → flair
    'seg': 'seg',     # Segmentation stays the same
}

# Regex pattern to match BraTS 2025 filenames
BRATS2025_PATTERN = re.compile(
    r'^(BraTS-GLI-\d{5}-\d{3})-(t1n|t1c|t2w|t2f|seg)\.(nii(?:\.gz)?)$'
)


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def print_step(step_num, text):
    """Print a step indicator."""
    print(f"\n{'─' * 70}")
    print(f"STEP {step_num}: {text}")
    print("─" * 70)


def compress_nifti(input_path, output_path):
    """Compress a .nii file to .nii.gz"""
    with open(input_path, 'rb') as f_in:
        with gzip.open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


def rename_brats2025_files(case_folder):
    """
    Rename BraTS 2025 files to BraTS 2021 naming convention.
    
    Returns:
        Tuple of (case_id, files_renamed, already_converted)
    """
    case_folder = Path(case_folder)
    case_id = case_folder.name
    
    files_renamed = 0
    already_converted = 0
    
    for file_path in case_folder.iterdir():
        if not file_path.is_file():
            continue
            
        filename = file_path.name
        match = BRATS2025_PATTERN.match(filename)
        
        if not match:
            # Check if already in BraTS 2021 format
            if re.match(rf'^{re.escape(case_id)}_(t1|t1ce|t2|flair|seg)\.nii\.gz$', filename):
                already_converted += 1
            continue
        
        file_case_id = match.group(1)
        old_suffix = match.group(2)
        extension = match.group(3)
        
        new_suffix = SUFFIX_MAPPING.get(old_suffix)
        if new_suffix is None:
            continue
        
        needs_compression = extension == 'nii'
        new_filename = f"{file_case_id}_{new_suffix}.nii.gz"
        new_path = case_folder / new_filename
        
        if new_path.exists():
            print(f"  ⚠ Target exists, skipping: {new_filename}")
            continue
        
        if needs_compression:
            print(f"  📦 Compressing: {filename} → {new_filename}")
            compress_nifti(file_path, new_path)
            file_path.unlink()
        else:
            print(f"  📝 Renaming: {filename} → {new_filename}")
            file_path.rename(new_path)
        
        files_renamed += 1
    
    return case_id, files_renamed, already_converted


def run_segmentation(case_folder, output_folder):
    """
    Run the BraTS 2021 KAIST segmentation model.
    
    Returns:
        Path to the segmentation output file
    """
    case_folder = Path(case_folder)
    output_folder = Path(output_folder)
    case_id = case_folder.name
    
    # Set up environment
    env = os.environ.copy()
    env.update(NNUNET_ENV)
    
    # Build command
    inference_script = SCRIPT_DIR / "run_brats2021_inference_singlethread.py"
    cmd = [
        str(PYTHON_EXE),
        str(inference_script),
        "--input", str(case_folder),
        "--output", str(output_folder)
    ]
    
    print(f"  🔄 Running inference (this takes 20-30 minutes on CPU)...")
    print(f"  📂 Input: {case_folder}")
    print(f"  📂 Output: {output_folder}")
    
    start_time = time.time()
    
    # Run the inference
    result = subprocess.run(
        cmd,
        env=env,
        capture_output=False,  # Show output in real-time
        cwd=str(SCRIPT_DIR)
    )
    
    elapsed = time.time() - start_time
    print(f"  ⏱ Inference completed in {elapsed/60:.1f} minutes")
    
    if result.returncode != 0:
        raise RuntimeError(f"Segmentation failed with return code {result.returncode}")
    
    # Find the output file
    output_file = output_folder / f"{case_id}.nii.gz"
    if not output_file.exists():
        raise FileNotFoundError(f"Expected output file not found: {output_file}")
    
    return output_file


def convert_labels(input_file, output_file):
    """
    Convert segmentation labels to match ground truth format.
    
    Returns:
        Path to the converted file
    """
    cmd = [
        str(PYTHON_EXE),
        str(SCRIPT_DIR / "convert_labels_to_brats.py"),
        str(input_file),
        str(output_file)
    ]
    
    print(f"  🔄 Converting labels...")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(SCRIPT_DIR))
    
    if result.returncode != 0:
        print(f"  ⚠ Warning: Label conversion returned non-zero: {result.stderr}")
    
    # Print conversion summary
    for line in result.stdout.split('\n'):
        if 'Labels' in line or 'SUCCESS' in line or 'mapping' in line.lower():
            print(f"  {line}")
    
    return Path(output_file)


def evaluate_segmentation(pred_file, gt_file):
    """
    Evaluate segmentation against ground truth.
    
    Returns:
        Dictionary with evaluation metrics
    """
    cmd = [
        str(PYTHON_EXE),
        str(SCRIPT_DIR / "evaluate_segmentation.py"),
        "--pred", str(pred_file),
        "--gt", str(gt_file)
    ]
    
    print(f"  🔄 Evaluating segmentation...")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(SCRIPT_DIR))
    
    if result.returncode != 0:
        print(f"  ⚠ Evaluation error: {result.stderr}")
        return None
    
    # Parse and display key metrics
    output = result.stdout
    print(output)
    
    # Extract metrics for summary
    metrics = {}
    for line in output.split('\n'):
        if 'Mean Dice Score' in line:
            match = re.search(r'(\d+\.\d+)%', line)
            if match:
                metrics['mean_dice'] = float(match.group(1))
        elif 'Whole Tumor' in line and 'Dice' not in line:
            match = re.search(r'(\d+\.\d+)%', line)
            if match:
                metrics['wt_dice'] = float(match.group(1))
        elif 'Tumor Core' in line and 'Dice' not in line:
            match = re.search(r'(\d+\.\d+)%', line)
            if match:
                metrics['tc_dice'] = float(match.group(1))
        elif 'Enhancing Tumor' in line and 'Label' not in line:
            match = re.search(r'(\d+\.\d+)%', line)
            if match:
                metrics['et_dice'] = float(match.group(1))
    
    return metrics


def run_feature_extraction(mri_folder, segmentation_file, output_folder):
    """
    Run the 6-step feature extraction pipeline.
    
    Returns:
        Path to the output folder
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        str(PYTHON_EXE),
        str(SCRIPT_DIR / "feature_extraction" / "run_all.py"),
        "--input", str(mri_folder),
        "--segmentation", str(segmentation_file),
        "--output", str(output_folder)
    ]
    
    print(f"  🔄 Running feature extraction pipeline...")
    result = subprocess.run(cmd, capture_output=False, cwd=str(SCRIPT_DIR))
    
    if result.returncode != 0:
        print(f"  ⚠ Feature extraction completed with warnings")
    
    return output_folder


def run_gemini_report(results_folder):
    """
    Generate radiology report using Gemini API.
    
    Returns:
        Path to the generated report, or None if generation fails
    """
    results_folder = Path(results_folder)
    report_path = results_folder / "feature_extraction" / "radiology_report.txt"
    
    # Record modification time before running (feature extraction may have created one)
    old_mtime = report_path.stat().st_mtime if report_path.exists() else 0
    
    cmd = [
        str(PYTHON_EXE),
        str(SCRIPT_DIR / "generate_report_gemini.py"),
        str(results_folder)
    ]
    
    print(f"  Generating radiology report with template system...")
    # Use encoding='utf-8' and errors='replace' to handle Unicode characters on Windows
    result = subprocess.run(
        cmd, 
        capture_output=True, 
        text=True, 
        encoding='utf-8',
        errors='replace',
        cwd=str(SCRIPT_DIR)
    )
    
    # Check return code first (primary success indicator)
    if result.returncode == 0:
        # Verify file was actually updated
        if report_path.exists():
            new_mtime = report_path.stat().st_mtime
            if new_mtime > old_mtime:
                return report_path
            else:
                print(f"  Warning: Report file was not updated")
        else:
            print(f"  Warning: Report file not found after generation")
    else:
        print(f"  Warning: Report generation failed:")
        # Check for API key error
        if "API key" in result.stdout or "API key" in result.stderr:
            print(f"     API key not configured. Edit generate_report_gemini.py to add your key.")
        else:
            # Filter out FutureWarning messages from error display
            stderr_lines = [l for l in result.stderr.split('\n') if 'FutureWarning' not in l and l.strip()]
            error_msg = '\n'.join(stderr_lines[:5]) if stderr_lines else 'Unknown error'
            print(f"     {error_msg}")
    
    return None


def run_pdf_report(results_folder):
    """
    Generate professional PDF report from the text radiology report.
    
    Returns:
        Path to the generated PDF, or None if generation fails
    """
    results_folder = Path(results_folder)
    
    cmd = [
        str(PYTHON_EXE),
        str(SCRIPT_DIR / "generate_pdf_report.py"),
        str(results_folder)
    ]
    
    print(f"  Generating professional PDF report...")
    result = subprocess.run(
        cmd, 
        capture_output=True, 
        text=True, 
        encoding='utf-8',
        errors='replace',
        cwd=str(SCRIPT_DIR)
    )
    
    if result.returncode != 0:
        print(f"  Warning: PDF generation failed:")
        print(f"     {result.stderr[:200] if result.stderr else 'Unknown error'}")
        return None
    
    pdf_path = results_folder / "feature_extraction" / "radiology_report.pdf"
    if pdf_path.exists():
        return pdf_path
    
    return None


def run_rag_assistant(report_path):
    """
    Launch the interactive RAG Educational Assistant.

    The assistant answers questions using the patient's radiology report
    and the verified medical-definitions knowledge base.

    Parameters
    ----------
    report_path : Path
        Path to the generated radiology_report.txt file.
    """
    report_path = Path(report_path)

    if not report_path.exists():
        print(f"  ⚠ Report not found at {report_path}")
        print(f"    RAG assistant requires a generated radiology report.")
        return

    # Read the patient report
    with open(report_path, "r", encoding="utf-8") as f:
        patient_report = f.read()

    if not patient_report.strip():
        print(f"  ⚠ Report file is empty: {report_path}")
        return

    # Import the RAG module
    try:
        from RAG_Assistant.rag_assistant import answer_query
    except ImportError as e:
        print(f"  ⚠ Could not import RAG assistant: {e}")
        print(f"    Make sure RAG_Assistant/ is in the project root.")
        return

    case_id = report_path.parent.parent.name  # results/<CaseID>/feature_extraction/

    print(f"\n  📂 Report: {report_path.name}")
    print(f"  🧠 Patient: {case_id}")
    print(f"  💡 Ask questions about the patient's MRI findings.")
    print(f"  🚫 Clinical questions (treatment, prognosis) are blocked.")
    print(f"  ⌨  Type 'quit' or 'exit' to finish.")
    print("=" * 70)

    while True:
        print()
        try:
            user_input = input("💬 Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 RAG session ended.")
            break

        if not user_input:
            continue

        if user_input.lower() in ('quit', 'exit', 'q'):
            print("\n👋 RAG session ended.")
            break

        answer = answer_query(
            user_query=user_input,
            patient_report_text=patient_report,
        )
        print(f"\n📚 Answer:\n{answer}")
        print("-" * 70)


def run_pipeline(case_folder):
    """
    Run the complete analysis pipeline on a case folder.
    """
    case_folder = Path(case_folder).absolute()
    
    if not case_folder.exists():
        # Try relative to script directory
        case_folder = SCRIPT_DIR / case_folder
        if not case_folder.exists():
            raise FileNotFoundError(f"Case folder not found: {case_folder}")
    
    case_id = case_folder.name
    results_folder = SCRIPT_DIR / "results" / case_id
    
    print_header(f"BRAIN MRI ANALYSIS PIPELINE")
    print(f"\n📋 Case ID: {case_id}")
    print(f"📂 Input folder: {case_folder}")
    print(f"📂 Results folder: {results_folder}")
    print(f"🕐 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    pipeline_start = time.time()
    
    # =========================================================================
    # STEP 1: Rename files
    # =========================================================================
    print_step(1, "RENAMING FILES (BraTS 2025 → BraTS 2021 format)")
    
    case_id, renamed, already_ok = rename_brats2025_files(case_folder)
    
    if renamed > 0:
        print(f"\n  ✅ Renamed {renamed} files")
    elif already_ok > 0:
        print(f"\n  ✅ Files already in correct format ({already_ok} files)")
    else:
        print(f"\n  ⚠ No files found to rename")
    
    # Verify required files exist
    required_files = ['t1', 't1ce', 't2', 'flair']
    missing = []
    for suffix in required_files:
        if not (case_folder / f"{case_id}_{suffix}.nii.gz").exists():
            missing.append(suffix)
    
    if missing:
        raise FileNotFoundError(f"Missing required MRI files: {missing}")
    
    # Check for ground truth
    gt_file = case_folder / f"{case_id}_seg.nii.gz"
    if not gt_file.exists():
        raise FileNotFoundError(f"Ground truth segmentation not found: {gt_file}")
    
    print(f"  ✅ All required files present")
    print(f"  ✅ Ground truth found: {gt_file.name}")
    
    # =========================================================================
    # STEP 2: Run segmentation
    # =========================================================================
    print_step(2, "RUNNING SEGMENTATION (BraTS 2021 KAIST Model)")
    
    results_folder.mkdir(parents=True, exist_ok=True)
    seg_output = run_segmentation(case_folder, results_folder)
    
    print(f"\n  ✅ Segmentation complete: {seg_output.name}")
    
    # =========================================================================
    # STEP 3: Convert labels
    # =========================================================================
    print_step(3, "CONVERTING LABELS")
    
    converted_file = results_folder / f"{case_id}_brats.nii.gz"
    convert_labels(seg_output, converted_file)
    
    print(f"\n  ✅ Labels converted: {converted_file.name}")
    
    # =========================================================================
    # STEP 4: Evaluate segmentation
    # =========================================================================
    print_step(4, "EVALUATING SEGMENTATION")
    
    metrics = evaluate_segmentation(converted_file, gt_file)
    
    if metrics:
        print(f"\n  📊 Summary:")
        if 'mean_dice' in metrics:
            print(f"     Mean Dice: {metrics['mean_dice']:.2f}%")
        if 'wt_dice' in metrics:
            print(f"     Whole Tumor: {metrics['wt_dice']:.2f}%")
        if 'tc_dice' in metrics:
            print(f"     Tumor Core: {metrics['tc_dice']:.2f}%")
        if 'et_dice' in metrics:
            print(f"     Enhancing Tumor: {metrics['et_dice']:.2f}%")
    
    # =========================================================================
    # STEP 5: Feature extraction
    # =========================================================================
    print_step(5, "RUNNING FEATURE EXTRACTION PIPELINE")
    
    feature_output = results_folder / "feature_extraction"
    run_feature_extraction(case_folder, converted_file, feature_output)
    
    print(f"\n  ✅ Feature extraction complete")
    print(f"  📂 Output folder: {feature_output}")
    
    # =========================================================================
    # STEP 6: Generate Gemini report
    # =========================================================================
    print_step(6, "GENERATING RADIOLOGY REPORT")
    
    gemini_report = run_gemini_report(results_folder)
    
    if gemini_report:
        print(f"\n  ✅ Radiology report generated: {gemini_report.name}")
    else:
        print(f"\n  ⚠ Radiology report not generated (check API key in generate_report_gemini.py)")
    
    # =========================================================================
    # STEP 7: Generate PDF report
    # =========================================================================
    print_step(7, "GENERATING PROFESSIONAL PDF REPORT")
    
    pdf_report = None
    if gemini_report:
        pdf_report = run_pdf_report(results_folder)
        
        if pdf_report:
            print(f"\n  ✅ PDF report generated: {pdf_report.name}")
        else:
            print(f"\n  ⚠ PDF report not generated")
    else:
        print(f"\n  ⚠ Skipped (text report required first)")
    
    # =========================================================================
    # STEP 8: RAG Educational Assistant (interactive Q&A)
    # =========================================================================
    print_step(8, "RAG EDUCATIONAL ASSISTANT")

    report_path = results_folder / "feature_extraction" / "radiology_report.txt"
    if report_path.exists():
        print(f"\n  ✅ Report available — launching interactive RAG assistant")
        print_header("RAG EDUCATIONAL ASSISTANT — Interactive Q&A")
        run_rag_assistant(report_path)
    else:
        print(f"\n  ⚠ Skipped — no radiology report available for RAG")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    pipeline_elapsed = time.time() - pipeline_start
    
    print_header("PIPELINE COMPLETE")
    print(f"\n📋 Case ID: {case_id}")
    print(f"⏱ Total time: {pipeline_elapsed/60:.1f} minutes")
    print(f"\n📂 Output files:")
    print(f"   • Segmentation: {seg_output}")
    print(f"   • Converted: {converted_file}")
    print(f"   • Feature extraction: {feature_output}")
    print(f"   • LLM-ready JSON: {feature_output / 'llm_ready_summary.json'}")
    if gemini_report:
        print(f"   • Radiology report: {gemini_report}")
    if pdf_report:
        print(f"   • PDF report: {pdf_report}")
    
    if metrics and 'mean_dice' in metrics:
        rating = "⭐ Excellent" if metrics['mean_dice'] >= 90 else \
                 "✓ Good" if metrics['mean_dice'] >= 80 else \
                 "~ Moderate" if metrics['mean_dice'] >= 70 else "△ Fair"
        print(f"\n📊 Performance: {metrics['mean_dice']:.2f}% Mean Dice ({rating})")
    
    print(f"\n🕐 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Save pipeline summary
    summary = {
        "case_id": case_id,
        "timestamp": datetime.now().isoformat(),
        "pipeline_duration_minutes": round(pipeline_elapsed / 60, 2),
        "input_folder": str(case_folder),
        "output_folder": str(results_folder),
        "segmentation_file": str(seg_output),
        "converted_file": str(converted_file),
        "ground_truth_file": str(gt_file),
        "feature_extraction_folder": str(feature_output),
        "gemini_report": str(gemini_report) if gemini_report else None,
        "pdf_report": str(pdf_report) if pdf_report else None,
        "metrics": metrics
    }
    
    summary_file = results_folder / "pipeline_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📄 Pipeline summary saved: {summary_file}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Automated Brain MRI Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s BraTS-GLI-00003-000
      Process case folder in current directory
      
  %(prog)s C:\\path\\to\\BraTS-GLI-00003-000
      Process case folder at specified path

Pipeline Steps:
  1. Rename BraTS 2025 files → BraTS 2021 format
  2. Run segmentation (BraTS 2021 KAIST model, ~20-30 min)
  3. Convert labels to match ground truth format
  4. Evaluate segmentation (Dice, IoU, etc.)
  5. Run 6-step feature extraction pipeline
  6. Generate radiology report (requires API key)
  7. Generate professional PDF report
  8. Launch RAG Educational Assistant (interactive Q&A)

Output:
  results/<CaseID>/
  ├── <CaseID>.nii.gz              (raw segmentation)
  ├── <CaseID>_brats.nii.gz        (converted labels)
  ├── pipeline_summary.json        (metrics & paths)
  └── feature_extraction/
      ├── radiology_report.txt     (AI-generated report)
      ├── radiology_report.pdf     (professional PDF)
      ├── llm_ready_summary.json   (for LLM consumption)
      └── step1-6 JSON files       (detailed analysis)
        """
    )
    
    parser.add_argument(
        'case_folder',
        help='Path to the case folder (e.g., BraTS-GLI-00003-000)'
    )
    
    args = parser.parse_args()
    
    try:
        summary = run_pipeline(args.case_folder)
        sys.exit(0)
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    except RuntimeError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(2)
    except KeyboardInterrupt:
        print(f"\n\n⚠ Pipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
