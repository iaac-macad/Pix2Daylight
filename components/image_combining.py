from PIL import Image
import os
import natsort

# Step 1

def combine_image(train_number):
    # Path to the folders containing the images
    folder_A = f'train{train_number}/dataset/input'
    folder_B = f'train{train_number}/dataset/groundtruth'

    # Path to the folder where you want to save the combined images
    output_folder = f'train{train_number}/dataset/combined_images'

    # Ensure the output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get list of images in both folders and sort them
    images_A = natsort.natsorted(os.listdir(folder_A))
    images_B = natsort.natsorted(os.listdir(folder_B))

    # Find common image names
    common_images = set(images_A).intersection(images_B)

    # Iterate through the common images and combine them
    for img_name in common_images:
        # Open images
        img_A = Image.open(os.path.join(folder_A, img_name))
        img_B = Image.open(os.path.join(folder_B, img_name))

        # Create a new blank image with dimensions 512x256
        combined_img = Image.new('RGB', (512, 256))

        # Paste images side by side
        combined_img.paste(img_A, (0, 0))
        combined_img.paste(img_B, (256, 0))

        # Save the combined image
        combined_img.save(os.path.join(output_folder, img_name))