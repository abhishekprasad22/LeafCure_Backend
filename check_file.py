import os

# The folder where your files are
target_dir = "models"
# The file you are trying to load
target_file = "Original_Image_CheckPoint_Model.keras"

print(f"Current Working Directory: {os.getcwd()}")

if os.path.exists(target_dir):
    print(f"\nScanning '{target_dir}' folder:")
    files = os.listdir(target_dir)
    found = False
    for f in files:
        # repr() reveals hidden spaces or special characters
        print(f" - Found file: {repr(f)}")
        
        if f == target_file:
            print("   -> EXACT MATCH! Python can see it.")
            found = True
        elif f.strip().lower() == target_file.strip().lower():
            print("   -> SIMILAR MATCH (Check spelling/spaces!)")
    
    if not found:
        print(f"\n❌ Python could NOT find exact match for: {repr(target_file)}")
else:
    print(f"\n❌ Python cannot find the '{target_dir}' folder itself!")