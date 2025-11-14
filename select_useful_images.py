import pandas as pd
import shutil
import os

#image paths
images_folder = '/Users/tomrose/Google_Drive/Galaxy_classification_pytorch/images_gz2/images'
output_folder = '/Users/tomrose/Google_Drive/Galaxy_classification_pytorch/images_gz2/useful_images'
spirals_folder = os.path.join(output_folder, 'spirals')
ellipticals_folder = os.path.join(output_folder, 'ellipticals')

#Create output subfolders if they don't exist
os.makedirs(spirals_folder, exist_ok=True)
os.makedirs(ellipticals_folder, exist_ok=True)

#Load CSV file
df = pd.read_csv('joined_galaxies_table.csv')

# Extract necessary columns
image_ids = df.iloc[:, 2].astype(str)           #Third column: image ID
spiral_flags = df.iloc[:, -3]                   #Second-last column: spiral flag (0 or 1)
elliptical_flags = df.iloc[:, -2]               #Last column: elliptical flag (0 or 1)

#Loop through and copy each image to the correct subdirectory
for img_id, is_spiral, is_elliptical in zip(image_ids, spiral_flags, elliptical_flags):
    filename = f"{img_id}.jpg"
    src = os.path.join(images_folder, filename)

    if not os.path.exists(src):
        print(f"Warning: {filename} not found.")
        continue

    if is_spiral == 1:
        dst = os.path.join(spirals_folder, filename)
        shutil.copy(src, dst)
    elif is_elliptical == 1:
        dst = os.path.join(ellipticals_folder, filename)
        shutil.copy(src, dst)

print("Finished. Spiral and elliptical images copied to respective folders.")
