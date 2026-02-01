"""
04_extract_mgmt_features_wholetumor.py

Extract radiomic features from the Whole Tumor (merged labels) on FLAIR images.

Output: Level4_Radiomic_Features_WholeTumor.csv
"""
from pathlib import Path
import sys
import traceback
from typing import Optional, Dict, Any

import pandas as pd
import numpy as np

try:
    import SimpleITK as sitk
    from radiomics import featureextractor
except Exception as e:
    print('ERROR: SimpleITK and pyradiomics are required. Install them first.')
    raise


SCRIPT_DIR = Path(__file__).parent
INPUT_CSV = SCRIPT_DIR / "Level4_MGMT_Dataset.csv"
OUTPUT_CSV = SCRIPT_DIR / "Level4_Radiomic_Features_WholeTumor.csv"


def find_flair_file(scan_folder: Path) -> Optional[Path]:
    # Try registered folders first
    registered = scan_folder / "HD-GLIO-AUTO-segmentation" / "registered"
    if registered.exists():
        for f in registered.glob("*.nii*"):
            if 'flair' in f.name.lower():
                return f

    deep_reg = scan_folder / "DeepBraTumIA-segmentation" / "registered"
    if deep_reg.exists():
        for f in deep_reg.glob("*.nii*"):
            if 'flair' in f.name.lower():
                return f

    # fallback to main folder
    for f in scan_folder.glob("*.nii*"):
        if 'flair' in f.name.lower():
            return f

    return None


def find_mask_file(scan_folder: Path) -> Optional[Path]:
    search_locations = [
        scan_folder / "HD-GLIO-AUTO-segmentation" / "registered",
        scan_folder / "HD-GLIO-AUTO-segmentation" / "native",
        scan_folder / "DeepBraTumIA-segmentation" / "registered",
        scan_folder / "DeepBraTumIA-segmentation" / "native",
        scan_folder,
    ]

    for loc in search_locations:
        if not loc.exists():
            continue
        for f in loc.glob("*.nii*"):
            if 'seg' in f.name.lower() or 'segmentation' in f.name.lower():
                return f
    return None


def create_extractor():
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.enableAllFeatures()
    extractor.settings['binCount'] = 32
    extractor.settings['resampledPixelSpacing'] = [1, 1, 1]
    extractor.settings['interpolator'] = 'sitkBSpline'
    extractor.settings['preCrop'] = True
    extractor.settings['force2D'] = False
    return extractor


def extract_features_for_patient(extractor, flair_path: Path, mask_path: Path) -> Dict[str, Any]:
    image = sitk.ReadImage(str(flair_path))
    mask = sitk.ReadImage(str(mask_path))

    # Create binary whole-tumor mask (merge labels > 0)
    bin_mask = sitk.BinaryThreshold(mask, lowerThreshold=1, upperThreshold=65535, insideValue=1, outsideValue=0)

    extractor.settings['label'] = 1
    result = extractor.execute(image, bin_mask)

    # Filter diagnostics
    features = {k: (v.item() if hasattr(v, 'item') else v) for k, v in result.items() if not k.startswith('diagnostics_')}
    return features


def main():
    print('=' * 60)
    print('Whole Tumor radiomics extraction (FLAIR)')
    print('=' * 60)

    if not INPUT_CSV.exists():
        print('ERROR: Input CSV not found:', INPUT_CSV)
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV)
    print(f'Total patients to process: {len(df)}')

    extractor = create_extractor()

    results = []
    success = 0
    failed = 0
    failed_list = []

    for idx, row in df.iterrows():
        pid = row.get('Patient_ID', f'idx_{idx}')
        scan_path = Path(row['Baseline_Scan_Path'])
        mgmt = row.get('MGMT_Label', None)

        print(f'[{idx+1}/{len(df)}] {pid} ...', end=' ')

        try:
            flair = find_flair_file(scan_path)
            mask = find_mask_file(scan_path)

            if flair is None:
                raise FileNotFoundError('FLAIR not found')
            if mask is None:
                raise FileNotFoundError('Segmentation mask not found')

            feats = extract_features_for_patient(extractor, flair, mask)
            feats['Patient_ID'] = pid
            feats['MGMT_Label'] = mgmt
            results.append(feats)
            success += 1
            print('✓')

        except Exception as e:
            failed += 1
            failed_list.append(pid)
            print('✗', str(e))
            traceback.print_exc(limit=1)
            continue

    if not results:
        print('No features extracted. Exiting.')
        sys.exit(1)

    df_out = pd.DataFrame(results)
    # Ensure Patient_ID and MGMT_Label are first columns
    cols = ['Patient_ID', 'MGMT_Label'] + [c for c in df_out.columns if c not in ('Patient_ID', 'MGMT_Label')]
    df_out = df_out[cols]

    df_out.to_csv(OUTPUT_CSV, index=False)
    print('\nSaved features to:', OUTPUT_CSV)
    print(f'Success: {success}, Failed: {failed}')
    if failed_list:
        print('Failed patients:', failed_list)


if __name__ == '__main__':
    main()
