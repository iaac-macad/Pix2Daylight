from PIL import Image
import os
import natsort

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
        img_A = Image.open(os.path.join(folder_A, img_name)).convert('RGBA')
        img_B = Image.open(os.path.join(folder_B, img_name)).convert('RGBA')

        # Create a new blank image with dimensions 1024x512
        combined_img = Image.new('RGBA', (1024, 512))

        # Paste images side by side
        combined_img.paste(img_A, (0, 0), img_A)
        combined_img.paste(img_B, (512, 0), img_B)

        # Save the combined image
        combined_img.save(os.path.join(output_folder, img_name))
