import os

def rename_files(directory="dresses"):
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' doesn't exist")
        return
    
    renamed_count = 0
    skipped_count = 0
    
    for filename in os.listdir(directory):
        if filename.endswith("_cloth.png"):
            new_filename = filename.replace("_cloth.png", ".png")
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_filename)
            
            if os.path.exists(new_path):
                print(f"Skipping {filename}: Destination file already exists")
                skipped_count += 1
                continue
            
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} → {new_filename}")
            renamed_count += 1
    
    print(f"\nRenaming complete!")
    print(f"Total files renamed: {renamed_count}")
    print(f"Files skipped: {skipped_count}")

if __name__ == "__main__":
    rename_files()
