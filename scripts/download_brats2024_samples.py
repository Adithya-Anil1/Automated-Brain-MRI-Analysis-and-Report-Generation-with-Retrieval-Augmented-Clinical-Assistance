"""
Download only 10 sample test cases from BraTS 2024 dataset
This avoids downloading the entire 39GB GLI Training folder
"""

import synapseclient
from synapseclient import Synapse
import os
from pathlib import Path


def download_sample_cases(auth_token, num_cases=10, output_dir=None):
    """
    Download a limited number of BraTS 2024 test cases
    
    Args:
        auth_token: Your Synapse authentication token
        num_cases: Number of test cases to download (default: 10)
        output_dir: Where to save the downloaded files
    """
    
    print("=" * 80)
    print(f"DOWNLOADING {num_cases} SAMPLE CASES FROM BRATS 2024 DATASET")
    print("=" * 80)
    
    # Setup output directory
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "data" / "sample_data"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Login to Synapse
    print("\nLogging into Synapse...")
    syn = Synapse()
    
    try:
        syn.login(authToken=auth_token)
        print("✓ Successfully logged in!")
    except Exception as e:
        print(f"✗ Login failed: {e}")
        print("\nPlease check your authentication token.")
        print("You can get your token from: https://www.synapse.org/#!PersonalAccessTokens:")
        return
    
    # BraTS 2024 GLI Training Dataset entity ID
    # syn60086071 is the main training dataset
    print(f"\nFetching BraTS 2024 dataset information...")
    
    try:
        # Get the entity to explore its contents
        entity = syn.get(entity='syn60086071', downloadFile=False)
        print(f"✓ Found dataset: {entity.name}")
        
        # List children (subdirectories/files) of this entity
        print("\nExploring dataset structure...")
        children = list(syn.getChildren(entity='syn60086071'))
        
        print(f"\nFound {len(children)} items in the dataset:")
        for i, child in enumerate(children[:20]):  # Show first 20 items
            print(f"  [{i+1}] {child['name']} (Type: {child['type']}, ID: {child['id']})")
        
        if len(children) > 20:
            print(f"  ... and {len(children) - 20} more items")
        
        # Download only the first N cases
        print(f"\n{'=' * 80}")
        print(f"DOWNLOADING FIRST {num_cases} CASES")
        print(f"{'=' * 80}\n")
        
        downloaded_count = 0
        case_folders = [c for c in children if c['type'] == 'org.sagebionetworks.repo.model.Folder']
        
        # If children are folders (individual cases)
        if case_folders:
            print(f"Found {len(case_folders)} case folders")
            
            for i, case in enumerate(case_folders[:num_cases]):
                print(f"\n[{i+1}/{num_cases}] Downloading: {case['name']}")
                
                # Download this case folder
                try:
                    case_path = output_dir / case['name']
                    case_path.mkdir(exist_ok=True)
                    
                    # Get all files in this case folder
                    case_files = list(syn.getChildren(case['id']))
                    print(f"  Found {len(case_files)} files in this case")
                    
                    for file_info in case_files:
                        print(f"    Downloading: {file_info['name']}", end=" ... ")
                        downloaded_file = syn.get(file_info['id'], downloadLocation=str(case_path))
                        print("✓")
                    
                    downloaded_count += 1
                    print(f"  ✓ Case {case['name']} downloaded successfully")
                    
                except Exception as e:
                    print(f"  ✗ Error downloading {case['name']}: {e}")
        
        # If children are individual files (not folders)
        else:
            print(f"Found {len(children)} files (not organized in case folders)")
            print("Downloading first files as samples...")
            
            for i, file_info in enumerate(children[:num_cases * 4]):  # Assume ~4 files per case
                print(f"\n[{i+1}] Downloading: {file_info['name']}", end=" ... ")
                try:
                    downloaded_file = syn.get(file_info['id'], downloadLocation=str(output_dir))
                    print("✓")
                    downloaded_count += 1
                except Exception as e:
                    print(f"✗ Error: {e}")
        
        print(f"\n{'=' * 80}")
        print(f"DOWNLOAD COMPLETE")
        print(f"{'=' * 80}")
        print(f"\n✓ Downloaded {downloaded_count} items to: {output_dir}")
        print(f"\nYou can now use these sample cases for testing your model.")
        
    except Exception as e:
        print(f"\n✗ Error accessing dataset: {e}")
        print("\nPossible issues:")
        print("  1. You may not have accepted the BraTS 2024 data use terms")
        print("  2. The entity ID might have changed")
        print("  3. Your token might not have the right permissions")
        print("\nPlease visit: https://www.synapse.org/#!Synapse:syn53708249")
        print("And make sure you've accepted the terms of use.")


def main():
    """Main function with interactive prompts"""
    
    print("\n" + "=" * 80)
    print("BRATS 2024 SAMPLE DOWNLOADER")
    print("=" * 80)
    
    print("\nThis script downloads only a few sample cases instead of the full 39GB dataset.")
    
    # Check if synapseclient is installed
    try:
        import synapseclient
        print("✓ Synapse client is installed")
    except ImportError:
        print("\n✗ Synapse client is not installed")
        print("\nPlease run: pip install synapseclient")
        return
    
    # Get authentication token
    print("\n" + "-" * 80)
    print("AUTHENTICATION")
    print("-" * 80)
    print("\nYou need a Synapse Personal Access Token.")
    print("Get it from: https://www.synapse.org/#!PersonalAccessTokens:")
    print("\nOption 1: Enter your token now")
    print("Option 2: Edit this script and replace 'YOUR_TOKEN_HERE' with your token")
    
    auth_token = input("\nEnter your Synapse token (or press Enter to use default): ").strip()
    
    if not auth_token:
        auth_token = "YOUR_TOKEN_HERE"
        print(f"Using default token placeholder: {auth_token}")
        print("⚠️  Please edit this script and replace 'YOUR_TOKEN_HERE' with your actual token")
    
    # Get number of cases
    num_cases_input = input("\nHow many test cases to download? (default: 10): ").strip()
    num_cases = int(num_cases_input) if num_cases_input else 10
    
    # Get output directory
    print("\nWhere should the data be saved?")
    print(f"Press Enter for default: data/sample_data")
    output_dir_input = input("Or enter a custom path: ").strip()
    output_dir = output_dir_input if output_dir_input else None
    
    # Start download
    print("\n" + "=" * 80)
    download_sample_cases(auth_token, num_cases, output_dir)


if __name__ == "__main__":
    # Quick start option: Uncomment and edit the line below with your token
    # download_sample_cases(auth_token="YOUR_TOKEN_HERE", num_cases=10)
    
    # Or run interactively
    main()
