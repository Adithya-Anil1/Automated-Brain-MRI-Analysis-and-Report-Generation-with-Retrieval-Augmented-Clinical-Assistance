"""
Find and download individual BraTS 2024 case files from Synapse
This explores the Synapse project to find individual cases instead of the full ZIP
"""

import synapseclient
from pathlib import Path

# Initialize and login
syn = synapseclient.Synapse()
syn.login(authToken="eyJ0eXAiOiJKV1QiLCJraWQiOiJXN05OOldMSlQ6SjVSSzpMN1RMOlQ3TDc6M1ZYNjpKRU9VOjY0NFI6VTNJWDo1S1oyOjdaQ0s6RlBUSCIsImFsZyI6IlJTMjU2In0.eyJhY2Nlc3MiOnsic2NvcGUiOlsidmlldyIsImRvd25sb2FkIl0sIm9pZGNfY2xhaW1zIjp7fX0sInRva2VuX3R5cGUiOiJQRVJTT05BTF9BQ0NFU1NfVE9LRU4iLCJpc3MiOiJodHRwczovL3JlcG8tcHJvZC5wcm9kLnNhZ2ViYXNlLm9yZy9hdXRoL3YxIiwiYXVkIjoiMCIsIm5iZiI6MTc1OTU5Mzk1NSwiaWF0IjoxNzU5NTkzOTU1LCJqdGkiOiIyNjczMiIsInN1YiI6IjM1NDk2MzAifQ.e-v-yN-JrSbmoCfr_RuJmB8f1ww8waEyww8rqj21HwqzHDaidnnvFjXnGA1uKB8a15oHbpniMvmSQRh4GykUiZRkzE3vgopMNbg4jpU9cF7OERAHPChSjFCr7oxMHFN6bLLKK9J5yARcSx4bJHDOHLBmOgwjTEyeyFIc6rQwLjzltJLC5HJ0SQR34-5s19Ec785U0dHIqJ8VNY_cS-Lh9QRDmQBmyYeg1fbqcSVEnoozHffKU0dXkbdpj36yOEerTna-9Udoet2RCTfsGaE0aVzVgtwWN0ltw3jUuZgQFb8keZb-WVz-hPN6TNFpEeXdEOuxfKqecGPQOcww1N5gJA")

print("=" * 80)
print("EXPLORING BRATS 2024 SYNAPSE PROJECT")
print("=" * 80)

# The main BraTS 2024 Synapse project
project_id = 'syn53708249'  # BraTS 2024 main project

print(f"\nExploring project: {project_id}")

# Get project
try:
    project = syn.get(project_id, downloadFile=False)
    print(f"Project name: {project.name}")
    
    # List all items in the project
    print("\nListing all items in the project...")
    children = list(syn.getChildren(project_id))
    
    print(f"\nFound {len(children)} items:")
    print("-" * 80)
    
    for i, child in enumerate(children):
        print(f"{i+1:2d}. {child['name']:<60} Type: {child['type']:<45} ID: {child['id']}")
    
    # Look for folders that might contain individual cases
    print("\n" + "=" * 80)
    print("LOOKING FOR FOLDERS WITH INDIVIDUAL CASES")
    print("=" * 80)
    
    folders = [c for c in children if c['type'] == 'org.sagebionetworks.repo.model.Folder']
    
    if folders:
        print(f"\nFound {len(folders)} folders. Exploring each one...\n")
        
        for folder in folders:
            print(f"\nFolder: {folder['name']} (ID: {folder['id']})")
            print("-" * 80)
            
            folder_children = list(syn.getChildren(folder['id']))
            print(f"Items inside: {len(folder_children)}")
            
            if len(folder_children) > 0:
                print(f"First 10 items:")
                for i, item in enumerate(folder_children[:10]):
                    print(f"  {i+1:2d}. {item['name']:<50} Type: {item['type']}")
                
                if len(folder_children) > 10:
                    print(f"  ... and {len(folder_children) - 10} more")
                
                # Check if these look like individual case folders
                case_like_folders = [c for c in folder_children 
                                    if c['type'] == 'org.sagebionetworks.repo.model.Folder']
                
                if case_like_folders:
                    print(f"\n  ✓ This folder has {len(case_like_folders)} sub-folders (likely individual cases!)")
                    print(f"  Folder ID to use for downloading: {folder['id']}")
    
    print("\n" + "=" * 80)
    print("EXPLORATION COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Identify which folder ID contains individual cases")
    print("2. Update the download script with that folder ID")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
