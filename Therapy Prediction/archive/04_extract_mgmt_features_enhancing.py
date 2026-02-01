"""
04_extract_mgmt_features_enhancing.py

Extract pyradiomics features from Enhancing Tumor (label 4) with fallback
to Necrosis/Core (label 1) per patient. Saves `Level4_Radiomic_Features_Enhancing.csv`.

Folder structure assumptions:
- Patient folders under a root directory (default: `dataset/Imaging`)
- Each patient folder contains one MRI image (contains 't1' or 't1ce' or 'flair') and a segmentation mask with 'seg' in the filename.

The script sets the extractor label per-patient to either 4 or 1 depending on mask contents.
"""
from pathlib import Path
import sys
import traceback
from typing import Optional, Dict, Any

import pandas as pd

try:
    import SimpleITK as sitk
    from radiomics import featureextractor
except Exception as e:
    print('ERROR: SimpleITK and pyradiomics must be installed. Aborting.')
    raise


SCRIPT_DIR = Path(__file__).parent
IMAGING_ROOT = SCRIPT_DIR / 'dataset' / 'Imaging'
OUTPUT_CSV = SCRIPT_DIR / 'Level4_Radiomic_Features_Enhancing.csv'


def find_image_and_mask(patient_folder: Path) -> Optional[tuple]:
    # Search recursively for nii files (handles week subfolders and registered folders)
    all_niis = list(patient_folder.rglob('*.nii*'))
    if not all_niis:
        return None

    # Masks: files with 'seg' or 'segmentation' in name
    mask_files = [f for f in all_niis if ('seg' in f.name.lower() or 'segmentation' in f.name.lower())]
    # Prefer registered masks
    def prefer_registered(files):
        for f in files:
            if 'registered' in str(f).lower() or 'reg' in f.name.lower():
                return f
        return None

    mask = None
    if mask_files:
        mask = prefer_registered(mask_files) or mask_files[0]

    # Images: all nii files excluding masks
    image_files = [f for f in all_niis if f not in mask_files]
    # Prefer T1CE, then T1, then FLAIR, else first available
    priority = ['t1ce', 't1', 'flair']
    image = None
    for token in priority:
        for f in image_files:
            if token in f.name.lower():
                image = f
                break
        if image:
            break

    if image is None and image_files:
        image = image_files[0]

    if image is None or mask is None:
        return None
    return image, mask


def create_extractor():
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.enableAllFeatures()
    # speed/consistency settings
    extractor.settings['binCount'] = 32
    extractor.settings['resampledPixelSpacing'] = [1, 1, 1]
    extractor.settings['interpolator'] = 'sitkBSpline'
    extractor.settings['preCrop'] = True
    extractor.settings['force2D'] = False
    return extractor


def mask_has_label(mask_path: Path, label: int) -> bool:
    m = sitk.ReadImage(str(mask_path))
    arr = sitk.GetArrayFromImage(m)
    return int((arr == label).sum()) > 0


def extract_features(extractor, image_path: Path, mask_path: Path, label: int) -> Dict[str, Any]:
    extractor.settings['label'] = int(label)
    img = sitk.ReadImage(str(image_path))
    msk = sitk.ReadImage(str(mask_path))
    res = extractor.execute(img, msk)
    # filter diagnostics
    features = {k: (v.item() if hasattr(v, 'item') else v) for k, v in res.items() if not k.startswith('diagnostics_')}
    return features


def main():
    if not IMAGING_ROOT.exists():
        print('Imaging root not found:', IMAGING_ROOT)
        sys.exit(1)

    extractor = create_extractor()

    results = []
    processed = 0
    failed = 0

    patients = sorted([p for p in IMAGING_ROOT.iterdir() if p.is_dir()])
    print(f'Found {len(patients)} patient folders in {IMAGING_ROOT}')

    for idx, patient in enumerate(patients, 1):
        pid = patient.name
        print(f'[{idx}/{len(patients)}] {pid} ...', end=' ')
        try:
            pair = find_image_and_mask(patient)
            if pair is None:
                print('✗ missing image or mask')
                failed += 1
                continue
            image_path, mask_path = pair

            # Prefer label 4 (Enhancing Tumor), fallback to label 1
            used_label = None
            if mask_has_label(mask_path, 4):
                used_label = 4
            elif mask_has_label(mask_path, 1):
                used_label = 1
            else:
                print('✗ mask contains neither label 4 nor 1')
                failed += 1
                continue

            feats = extract_features(extractor, image_path, mask_path, used_label)
            feats['Patient_ID'] = pid
            feats['Used_Label'] = used_label
            results.append(feats)
            processed += 1
            print(f'✓ label={used_label}, features={len(feats)-2}')

        except Exception as e:
            failed += 1
            print('✗ ERROR')
            traceback.print_exc(limit=1)
            continue

    if not results:
        print('No features extracted.')
        sys.exit(1)

    df = pd.DataFrame(results)
    cols = ['Patient_ID', 'Used_Label'] + [c for c in df.columns if c not in ('Patient_ID', 'Used_Label')]
    df = df[cols]
    df.to_csv(OUTPUT_CSV, index=False)
    print(f'\nSaved {processed} feature rows to {OUTPUT_CSV} (failed: {failed})')


if __name__ == '__main__':
    main()
