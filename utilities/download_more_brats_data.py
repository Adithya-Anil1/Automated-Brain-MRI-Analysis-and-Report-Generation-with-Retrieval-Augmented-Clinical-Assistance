"""
Download additional BraTS dataset samples for testing and evaluation.
This script helps you get more samples from the BraTS dataset.
"""

import os
from pathlib import Path
import requests
from tqdm import tqdm
import zipfile


def download_file(url, destination):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    with open(destination, 'wb') as f, tqdm(
        desc=destination.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)


def download_brats_samples():
    """
    Download sample BraTS cases for testing.
    
    Note: Full BraTS dataset requires registration at Synapse.
    This downloads publicly available sample/demo data.
    """
    print("=" * 80)
    print("DOWNLOAD BRATS DATASET SAMPLES")
    print("=" * 80)
    
    project_dir = Path(__file__).parent.absolute()
    sample_dir = project_dir / "sample_data"
    sample_dir.mkdir(exist_ok=True)
    
    print("\n📊 BRATS DATASET OPTIONS:")
    print("\n" + "=" * 80)
    
    print("\n1. BraTS 2021 Dataset (Competition)")
    print("-" * 80)
    print("   • 1,251 training cases with ground truth segmentations")
    print("   • 219 validation cases")
    print("   • Requires registration at Synapse")
    print("   • URL: https://www.synapse.org/#!Synapse:syn25829067")
    print("   • Best for: Comprehensive evaluation")
    
    print("\n2. BraTS 2024 Dataset (Latest)")
    print("-" * 80)
    print("   • Latest dataset with more diverse cases")
    print("   • Includes adult, pediatric, and African cases")
    print("   • Requires registration at Synapse")
    print("   • URL: https://www.synapse.org/#!Synapse:syn53708249")
    print("   • Best for: Most up-to-date evaluation")
    
    print("\n3. BraTS Sample Data (Quick Test)")
    print("-" * 80)
    print("   • Small subset for quick testing")
    print("   • No registration required")
    print("   • Best for: Quick validation")
    
    print("\n" + "=" * 80)
    print("RECOMMENDED WORKFLOW")
    print("=" * 80)
    
    print("\nStep 1: Register for BraTS Dataset")
    print("   1. Go to: https://www.synapse.org/")
    print("   2. Create a free account")
    print("   3. Navigate to BraTS 2021: https://www.synapse.org/#!Synapse:syn25829067")
    print("   4. Click 'Files' tab")
    print("   5. Accept terms and download 'TrainingData' folder")
    
    print("\nStep 2: Download the Data")
    print("   • Use Synapse web interface OR")
    print("   • Use Synapse Python client:")
    print("     pip install synapseclient")
    print("     synapse get -r syn25829067")
    
    print("\nStep 3: Organize the Data")
    print("   Place downloaded BraTS cases in:")
    print(f"   {sample_dir}")
    print("   ")
    print("   Expected structure:")
    print("   sample_data/")
    print("   ├── BraTS2021_00000/")
    print("   │   ├── BraTS2021_00000_t1.nii.gz")
    print("   │   ├── BraTS2021_00000_t1ce.nii.gz")
    print("   │   ├── BraTS2021_00000_t2.nii.gz")
    print("   │   ├── BraTS2021_00000_flair.nii.gz")
    print("   │   └── BraTS2021_00000_seg.nii.gz  (ground truth)")
    print("   ├── BraTS2021_00001/")
    print("   │   └── ...")
    print("   └── ...")
    
    print("\n" + "=" * 80)
    print("ALTERNATIVE: USE SYNAPSE CLIENT")
    print("=" * 80)
    
    print("\nQuick download using Synapse client:")
    print("```bash")
    print("# Install Synapse client")
    print("pip install synapseclient")
    print("")
    print("# Login to Synapse")
    print("import synapseclient")
    print("syn = synapseclient.Synapse()")
    print("syn.login('your_username', 'your_password')")
    print("")
    print("# Download BraTS 2021 training data")
    print("syn.get('syn25829067', downloadLocation='./sample_data')")
    print("```")
    
    print("\n" + "=" * 80)
    print("AFTER DOWNLOADING")
    print("=" * 80)
    
    print("\nUse the batch evaluation script:")
    print("   python batch_evaluate_brats.py --data_dir sample_data")
    print("\nThis will:")
    print("   • Run inference on all cases")
    print("   • Calculate accuracy metrics for each")
    print("   • Generate summary statistics")
    print("   • Create comparison visualizations")
    
    print("\n" + "=" * 80)
    print("CURRENT SAMPLE DATA")
    print("=" * 80)
    
    # Check what samples we currently have
    existing_samples = [d for d in sample_dir.iterdir() if d.is_dir()]
    
    if existing_samples:
        print(f"\nFound {len(existing_samples)} sample(s):")
        for sample in existing_samples:
            files = list(sample.glob("*.nii.gz"))
            has_seg = any('seg' in f.name for f in files)
            print(f"   ✓ {sample.name} ({len(files)} files, "
                  f"{'with' if has_seg else 'NO'} ground truth)")
    else:
        print("\nNo samples found yet.")
    
    print("\n" + "=" * 80)
    print("QUICK START (Using Existing Sample)")
    print("=" * 80)
    
    print("\nYou already have BraTS2021_00495 sample.")
    print("Let's verify it was processed correctly:")
    print("\n1. Re-run inference with correct script:")
    print("   python run_brats2021_inference.py \\")
    print("       --input sample_data\\BraTS2021_sample \\")
    print("       --output results\\BraTS2021_00495_corrected")
    print("\n2. Check labels in output:")
    print("   python check_labels.py results\\BraTS2021_00495_corrected\\BraTS2021_00495.nii.gz")
    print("\n3. Evaluate accuracy:")
    print("   python evaluate_segmentation.py \\")
    print("       --pred results\\BraTS2021_00495_corrected\\BraTS2021_00495.nii.gz \\")
    print("       --gt sample_data\\BraTS2021_sample\\BraTS2021_00495_seg.nii.gz")
    
    return existing_samples


def install_synapse_client():
    """Guide to install Synapse client for downloading BraTS data"""
    print("\n" + "=" * 80)
    print("INSTALLING SYNAPSE CLIENT")
    print("=" * 80)
    
    print("\nTo download BraTS dataset programmatically:")
    print("\n1. Install Synapse client:")
    print("   pip install synapseclient")
    
    print("\n2. Create a Synapse account:")
    print("   https://www.synapse.org/")
    
    print("\n3. Use this Python code to download:")
    print("""
import synapseclient
import synapseutils

# Login
syn = synapseclient.Synapse()
syn.login('your_username', 'your_password')

# Download BraTS 2021 training data
entity = syn.get('syn25829067', downloadLocation='./sample_data')

# Or download specific files
files = synapseutils.syncFromSynapse(syn, 'syn25829067', path='./sample_data')
    """)
    
    print("\n4. After download, organize files as shown above")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Download BraTS dataset samples')
    parser.add_argument('--install-synapse', action='store_true',
                        help='Show instructions for installing Synapse client')
    
    args = parser.parse_args()
    
    if args.install_synapse:
        install_synapse_client()
    else:
        samples = download_brats_samples()
        
        print("\n" + "=" * 80)
        print("NEXT STEPS")
        print("=" * 80)
        print("\n1. Register at Synapse and download BraTS data")
        print("2. Place data in sample_data/ folder")
        print("3. Run batch evaluation: python batch_evaluate_brats.py")
        print("\nOr test with your current sample by re-running inference correctly.")
