"""
Download BraTS 2024 ZIP and extract only 10 sample cases
This downloads the full ZIP but only extracts 10 cases to save space
"""

import synapseclient
from pathlib import Path
import zipfile
import os

# Initialize and login
syn = synapseclient.Synapse()
syn.login(authToken="eyJ0eXAiOiJKV1QiLCJraWQiOiJXN05OOldMSlQ6SjVSSzpMN1RMOlQ3TDc6M1ZYNjpKRU9VOjY0NFI6VTNJWDo1S1oyOjdaQ0s6RlBUSCIsImFsZyI6IlJTMjU2In0.eyJhY2Nlc3MiOnsic2NvcGUiOlsidmlldyIsImRvd25sb2FkIl0sIm9pZGNfY2xhaW1zIjp7fX0sInRva2VuX3R5cGUiOiJQRVJTT05BTF9BQ0NFU1NfVE9LRU4iLCJpc3MiOiJodHRwczovL3JlcG8tcHJvZC5wcm9kLnNhZ2ViYXNlLm9yZy9hdXRoL3YxIiwiYXVkIjoiMCIsIm5iZiI6MTc1OTU5Mzk1NSwiaWF0IjoxNzU5NTkzOTU1LCJqdGkiOiIyNjczMiIsInN1YiI6IjM1NDk2MzAifQ.e-v-yN-JrSbmoCfr_RuJmB8f1ww8waEyww8rqj21HwqzHDaidnnvFjXnGA1uKB8a15oHbpniMvmSQRh4GykUiZRkzE3vgopMNbg4jpU9cF7OERAHPChSjFCr7oxMHFN6bLLKK9J5yARcSx4bJHDOHLBmOgwjTEyeyFIc6rQwLjzltJLC5HJ0SQR34-5s19Ec785U0dHIqJ8VNY_cS-Lh9QRDmQBmyYeg1fbqcSVEnoozHffKU0dXkbdpj36yOEerTna-9Udoet2RCTfsGaE0aVzVgtwWN0ltw3jUuZgQFb8keZb-WVz-hPN6TNFpEeXdEOuxfKqecGPQOcww1N5gJA")

print("=" * 80)
print("BRATS 2024 - DOWNLOAD 10 SAMPLE CASES")
print("=" * 80)

# Setup directories
download_dir = Path("downloads")
output_dir = Path("data/brats2024_samples")
download_dir.mkdir(exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)

# Entity ID for the ZIP file
entity_id = 'syn60086071'

print(f"\nStep 1: Downloading BraTS 2024 ZIP file...")
print("⚠️  WARNING: This is a large file (~39GB). This will take time!")
print("The file will be saved to: downloads/")

user_input = input("\nDo you want to continue? (yes/no): ").strip().lower()

if user_input not in ['yes', 'y']:
    print("Download cancelled.")
    exit()

print("\nDownloading... This may take 30 minutes to several hours depending on your internet speed.")

try:
    # Download the ZIP file
    downloaded_file = syn.get(entity_id, downloadLocation=str(download_dir))
    zip_path = Path(downloaded_file.path)
    
    print(f"\n✓ Download complete!")
    print(f"ZIP file location: {zip_path}")
    
    # Get ZIP file size
    zip_size_gb = zip_path.stat().st_size / (1024**3)
    print(f"ZIP file size: {zip_size_gb:.2f} GB")
    
    print("\n" + "=" * 80)
    print("Step 2: Extracting 10 sample cases from ZIP")
    print("=" * 80)
    
    # Open the ZIP and list contents
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        all_files = zip_ref.namelist()
        print(f"\nTotal files in ZIP: {len(all_files)}")
        
        # Find case directories (typically BraTS-GLI-XXXXX-XXX format)
        cases = {}
        for file_path in all_files:
            # Extract case name (folder name)
            parts = file_path.split('/')
            if len(parts) > 1:
                case_name = parts[0]  # First directory is usually the case name
                if case_name not in cases:
                    cases[case_name] = []
                cases[case_name].append(file_path)
        
        case_names = list(cases.keys())
        print(f"Found {len(case_names)} cases in the ZIP")
        
        if len(case_names) == 0:
            print("\n⚠️  No case folders found. Listing first 20 files:")
            for i, f in enumerate(all_files[:20]):
                print(f"  {i+1}. {f}")
            print("\nPlease check the ZIP structure manually.")
        else:
            # Extract first 10 cases
            num_cases_to_extract = min(10, len(case_names))
            print(f"\nExtracting first {num_cases_to_extract} cases...")
            
            for i, case_name in enumerate(case_names[:num_cases_to_extract]):
                print(f"\n[{i+1}/{num_cases_to_extract}] Extracting case: {case_name}")
                case_files = cases[case_name]
                print(f"  Files in this case: {len(case_files)}")
                
                # Extract all files for this case
                for file_path in case_files:
                    zip_ref.extract(file_path, output_dir)
                
                print(f"  ✓ Extracted successfully")
            
            print("\n" + "=" * 80)
            print("EXTRACTION COMPLETE!")
            print("=" * 80)
            print(f"\nExtracted {num_cases_to_extract} sample cases to: {output_dir.absolute()}")
            print("\nYou can now delete the ZIP file if you want to save space:")
            print(f"  del \"{zip_path}\"")

except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("DONE!")
print("=" * 80)
