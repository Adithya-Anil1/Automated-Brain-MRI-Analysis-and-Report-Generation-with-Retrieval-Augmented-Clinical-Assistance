"""
Debug script to explore BraTS 2024 Synapse structure
"""

import synapseclient

# Initialize and login
syn = synapseclient.Synapse()
syn.login(authToken="eyJ0eXAiOiJKV1QiLCJraWQiOiJXN05OOldMSlQ6SjVSSzpMN1RMOlQ3TDc6M1ZYNjpKRU9VOjY0NFI6VTNJWDo1S1oyOjdaQ0s6RlBUSCIsImFsZyI6IlJTMjU2In0.eyJhY2Nlc3MiOnsic2NvcGUiOlsidmlldyIsImRvd25sb2FkIl0sIm9pZGNfY2xhaW1zIjp7fX0sInRva2VuX3R5cGUiOiJQRVJTT05BTF9BQ0NFU1NfVE9LRU4iLCJpc3MiOiJodHRwczovL3JlcG8tcHJvZC5wcm9kLnNhZ2ViYXNlLm9yZy9hdXRoL3YxIiwiYXVkIjoiMCIsIm5iZiI6MTc1OTU5Mzk1NSwiaWF0IjoxNzU5NTkzOTU1LCJqdGkiOiIyNjczMiIsInN1YiI6IjM1NDk2MzAifQ.e-v-yN-JrSbmoCfr_RuJmB8f1ww8waEyww8rqj21HwqzHDaidnnvFjXnGA1uKB8a15oHbpniMvmSQRh4GykUiZRkzE3vgopMNbg4jpU9cF7OERAHPChSjFCr7oxMHFN6bLLKK9J5yARcSx4bJHDOHLBmOgwjTEyeyFIc6rQwLjzltJLC5HJ0SQR34-5s19Ec785U0dHIqJ8VNY_cS-Lh9QRDmQBmyYeg1fbqcSVEnoozHffKU0dXkbdpj36yOEerTna-9Udoet2RCTfsGaE0aVzVgtwWN0ltw3jUuZgQFb8keZb-WVz-hPN6TNFpEeXdEOuxfKqecGPQOcww1N5gJA")

print("=" * 80)
print("EXPLORING BRATS 2024 SYNAPSE STRUCTURE")
print("=" * 80)

# Entity ID for BraTS 2024 GLI Training dataset
entity_id = 'syn60086071'

print(f"\nGetting entity: {entity_id}")
entity = syn.get(entity=entity_id, downloadFile=False)
print(f"Entity name: {entity.name}")
print(f"Entity type: {type(entity)}")

# Get children
print("\nListing children...")
children = list(syn.getChildren(entity_id))

print(f"\nTotal children found: {len(children)}")

if len(children) == 0:
    print("\n⚠️  No children found! This might mean:")
    print("  - You don't have access yet (check permissions)")
    print("  - This entity is a file, not a folder")
    print("  - The data structure is different")
else:
    print(f"\nFirst 20 items:")
    print("-" * 80)
    for i, child in enumerate(children[:20]):
        print(f"{i+1:2d}. Name: {child['name']:<40} Type: {child['type']:<45} ID: {child['id']}")
    
    if len(children) > 20:
        print(f"\n... and {len(children) - 20} more items")
    
    # Check types
    types = {}
    for child in children:
        t = child['type']
        types[t] = types.get(t, 0) + 1
    
    print("\n" + "=" * 80)
    print("SUMMARY OF ITEM TYPES:")
    print("=" * 80)
    for t, count in types.items():
        print(f"  {t}: {count}")
    
    # If there are folders, explore the first one
    folders = [c for c in children if c['type'] == 'org.sagebionetworks.repo.model.Folder']
    if folders:
        print("\n" + "=" * 80)
        print(f"EXPLORING FIRST FOLDER: {folders[0]['name']}")
        print("=" * 80)
        
        folder_children = list(syn.getChildren(folders[0]['id']))
        print(f"Items in this folder: {len(folder_children)}")
        
        for i, item in enumerate(folder_children[:10]):
            print(f"  {i+1}. {item['name']} (Type: {item['type']})")
    
    # If there are files, show info about the first one
    files = [c for c in children if c['type'] == 'org.sagebionetworks.repo.model.FileEntity']
    if files:
        print("\n" + "=" * 80)
        print(f"EXAMPLE FILE INFO: {files[0]['name']}")
        print("=" * 80)
        
        file_entity = syn.get(files[0]['id'], downloadFile=False)
        print(f"  ID: {file_entity.id}")
        print(f"  Name: {file_entity.name}")
        if hasattr(file_entity, 'contentSize'):
            size_mb = file_entity.contentSize / (1024 * 1024)
            print(f"  Size: {size_mb:.2f} MB")

print("\n" + "=" * 80)
print("EXPLORATION COMPLETE")
print("=" * 80)
