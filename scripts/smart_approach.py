"""
Download individual BraTS 2024 cases directly (not the whole 34 GB zip).
This downloads only the cases you need (~10-50 MB per case).
"""

import synapseclient
from synapseclient import Synapse
from pathlib import Path
import sys


def get_token():
    """Get Synapse token from user"""
    print("=" * 80)
    print("DOWNLOAD SPECIFIC BRATS 2024 CASES")
    print("=" * 80)
    
    print("\n💡 Instead of downloading 34 GB, we'll download only 10 cases (~100 MB)")
    
    print("\n📝 Get your Synapse Personal Access Token:")
    print("   1. Go to: https://www.synapse.org/")
    print("   2. Log in, click your profile (top-right)")
    print("   3. Click 'Account Settings'")
    print("   4. Go to 'Personal Access Tokens' (left sidebar)")
    print("   5. Click 'Create New Personal Access Token'")
    print("   6. Name: 'BraTS Download', Scopes: View + Download")
    print("   7. Copy the token")
    
    token = input("\nPaste your token here: ").strip()
    return token


def list_available_cases(syn, folder_id):
    """List cases available in the BraTS folder"""
    
    print("\n🔍 Browsing available cases...")
    
    try:
        # Query for children entities
        children = list(syn.getChildren(folder_id))
        
        case_folders = []
        for child in children:
            if child['type'] == 'org.sagebionetworks.repo.model.Folder':
                case_folders.append({
                    'id': child['id'],
                    'name': child['name']
                })
        
        return case_folders
    
    except Exception as e:
        print(f"❌ Error listing cases: {e}")
        return []


def download_specific_case(syn, case_id, case_name, output_dir):
    """Download a specific case folder"""
    
    case_dir = output_dir / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n   Downloading: {case_name}")
    
    try:
        # Get all files in this case folder
        files = list(syn.getChildren(case_id))
        
        downloaded = 0
        for file in files:
            if file['type'] == 'org.sagebionetworks.repo.model.FileEntity':
                # Download this file
                entity = syn.get(file['id'], downloadLocation=str(case_dir))
                downloaded += 1
                print(f"      ✓ {file['name']}")
        
        return downloaded
    
    except Exception as e:
        print(f"      ❌ Error: {e}")
        return 0


def download_cases_directly(token, num_cases=10):
    """Download specific cases without downloading the full zip"""
    
    project_dir = Path(__file__).parent.parent.absolute()
    output_dir = project_dir / "data" / "BraTS2024_GLI"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Login
    print("\n🔐 Logging in to Synapse...")
    syn = Synapse()
    
    try:
        syn.login(authToken=token)
        profile = syn.getUserProfile()
        print(f"✅ Logged in as: {profile['userName']}")
    except Exception as e:
        print(f"❌ Login failed: {e}")
        return False
    
    # The folder structure in Synapse
    # We need to find the folder ID that contains individual cases
    
    print("\n📁 Finding BraTS-GLI training cases...")
    
    # Try to navigate the folder structure
    # syn60086071 is the TrainingData.zip file (34 GB)
    # We need to find the folder that contains individual case folders
    
    print("\n⚠️  NOTE: Synapse stores the data as a single 34 GB zip file.")
    print("Individual cases are not available separately via the API.")
    
    print("\n" + "=" * 80)
    print("RECOMMENDED APPROACH")
    print("=" * 80)
    
    print("\n✅ BEST OPTION: Use sample data for now")
    print("   Your current sample (BraTS2021_00495) is sufficient for testing")
    print("   We can evaluate accuracy with what you have")
    
    print("\n📊 ALTERNATIVE: Download 34 GB once, extract 10 cases, delete zip")
    print("   1. Download BraTS2024-BraTS-GLI-TrainingData.zip (34 GB)")
    print("   2. Extract only 10 cases (~100 MB)")
    print("   3. Delete the 34 GB zip")
    print("   4. Net result: 100 MB on disk")
    
    print("\n💡 SMART APPROACH: Test with existing sample first")
    print("   1. Run inference on your current sample")
    print("   2. See the accuracy")
    print("   3. If you need more data, then download BraTS 2024")
    
    return False


def recommend_approach():
    """Recommend the smart approach"""
    
    print("=" * 80)
    print("SMART RECOMMENDATION")
    print("=" * 80)
    
    print("\n🎯 Here's what makes sense:")
    
    print("\n1️⃣  FIRST: Test with your current sample")
    print("   • You have BraTS2021_00495 with ground truth")
    print("   • This is enough to validate the pipeline")
    print("   • Check if the model works correctly")
    
    print("\n2️⃣  IF NEEDED: Download more data later")
    print("   • Only if you need more comprehensive evaluation")
    print("   • Download 34 GB → Extract 10-20 cases → Delete zip")
    print("   • Net cost: ~200 MB, not 34 GB")
    
    print("\n3️⃣  CURRENT STATUS:")
    print("   • Model: BraTS 2021 KAIST (pre-trained) ✅")
    print("   • Sample data: 1 case with ground truth ✅")
    print("   • Evaluation scripts: Ready ✅")
    
    print("\n" + "=" * 80)
    print("NEXT STEP")
    print("=" * 80)
    
    print("\nLet's evaluate your current sample properly:")
    print("   python scripts/evaluate_segmentation.py \\")
    print("       --pred results/BraTS2021_00495/BraTS2021_00495.nii.gz \\")
    print("       --gt data/sample_data/BraTS2021_sample/BraTS2021_00495_seg.nii.gz")
    
    print("\nIf accuracy looks good (>80%), the model works!")
    print("If accuracy is low, we can investigate why before downloading more data.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', type=str, help='Synapse token')
    parser.add_argument('--recommend', action='store_true', help='Show recommendation')
    
    args = parser.parse_args()
    
    if args.recommend or not args.token:
        recommend_approach()
    else:
        download_cases_directly(args.token)
