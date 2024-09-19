import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.metrics import mean_absolute_error, mean_squared_error

def load_graph(image_file):
    # Read and decode a PNG image file to a uint8 tensor
    image = tf.io.read_file(image_file)
    image = tf.io.decode_png(image, channels=4)

    # Split each image tensor into two tensors:
    # - one with a real building facade image
    # - one with an architecture label image
    w = tf.shape(image)[1]
    w = w // 2
    input_image = image[:, :w, :]
    real_image = image[:, w:, :]

    # Convert both images to float32 tensors
    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    return input_image, real_image

def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return input_image, real_image

def random_crop(input_image, real_image, img_height, img_width):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[2, img_height, img_width, 4])  # Adjusted for 4 channels
    return cropped_image[0], cropped_image[1]

# Normalizing the images to [-1, 1]
def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1
    return input_image, real_image

@tf.function()
def random_jitter(input_image, real_image, img_height, img_width):
    # Resizing to 572x572
    input_image, real_image = resize(input_image, real_image, 572, 572)

    # Random cropping back to 512x512
    input_image, real_image = random_crop(input_image, real_image, img_height, img_width)

    if tf.random.uniform(()) > 0.5:
        # Random mirroring
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image

def load_image_train(image_file, img_height, img_width):
    input_image, real_image = load_graph(image_file)
    input_image, real_image = resize(input_image, real_image, img_height, img_width)
    input_image, real_image = normalize(input_image, real_image)
    return input_image, real_image

def load_image_test(image_file, img_height, img_width):
    input_image, real_image = load_graph(image_file)
    input_image, real_image = resize(input_image, real_image, img_height, img_width)
    input_image, real_image = normalize(input_image, real_image)
    return input_image, real_image

def generate_images(model, test_input, tar):
    prediction = model(test_input, training=False)
    plt.figure(figsize=(15, 15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # Getting the pixel values in the [0, 1] range to plot.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()

def generate_pred(model, test_input, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # Generate predictions
    predictions = model(test_input, training=False)
    predictions_np = predictions.numpy()

    for i, pred in enumerate(predictions_np):
        # Normalize the prediction to 0-255 if necessary
        pred = (pred - pred.min()) / (pred.max() - pred.min()) * 255
        pred = pred.astype(np.uint8)

        # Save the prediction as an RGBA image
        img = Image.fromarray(pred, 'RGBA')

        save_path = os.path.join(save_dir, f'prediction_{i}.png')
        img.save(save_path)

        print(f"Prediction {i} saved to {save_path}")

def save_pred_images(i, model, test_input, output_folder):
    prediction = model(test_input, training=False)
    prediction = tf.squeeze(prediction, axis=0)
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    # Save the predicted image to the output folder
    tf.keras.preprocessing.image.save_img(
        os.path.join(output_folder, f"prediction_{i}.jpg"),
        prediction,
        scale=True
    )

def scale_image(image):
    """
    Scale an image from the range (-1, 1) to (0, 1).
    
    Args:
    - image (numpy.ndarray): Input image array.
    
    Returns:
    - numpy.ndarray: Scaled image array.
    """
    # Scale the image to the range (0, 1)
    scaled_image = (image + 1) / 2
    return scaled_image

def save_combined_images(model, test_input, tar, output_folder, image_name):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # Getting the pixel values in the [0, 1] range to plot.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Save the combined image
    plt.savefig(os.path.join(output_folder, f"{image_name}.jpg"))
    plt.close()

def calculate_metrics(val_dataset, generator_model):
    psnr_scores = []
    ssim_scores = []
    mae_scores = []
    mse_scores = []
    rmse_scores = []

    for input_batch, target_batch in val_dataset:
        # Generate output batch
        gen_output_batch = generator_model(input_batch, training=True)

        if input_batch.shape[0] == 1:
            # Handle batch size of 1 separately
            gen_output = gen_output_batch[0]
            target_image = target_batch[0]

            # Convert to numpy arrays
            gen_output_np = gen_output.numpy()
            target_image_np = target_image.numpy()

            # Flatten the images for sklearn metrics
            gen_output_flat = gen_output_np.flatten()
            target_image_flat = target_image_np.flatten()

            # Calculate metrics
            psnr_scores.append(float(psnr(target_image_np, gen_output_np, data_range=1.0)))
            ssim_scores.append(float(ssim(target_image_np, gen_output_np, win_size=11, channel_axis=-1, data_range=1.0)))
            mae_scores.append(float(mean_absolute_error(target_image_flat, gen_output_flat)))
            mse_scores.append(float(mean_squared_error(target_image_flat, gen_output_flat)))
            rmse_scores.append(float(np.sqrt(mean_squared_error(target_image_flat, gen_output_flat))))
        else:
            for i in range(input_batch.shape[0]):
                gen_output = gen_output_batch[i]
                target_image = target_batch[i]

                # Convert to numpy arrays
                gen_output_np = gen_output.numpy()
                target_image_np = target_image.numpy()

                # Flatten the images for sklearn metrics
                gen_output_flat = gen_output_np.flatten()
                target_image_flat = target_image_np.flatten()

                # Calculate metrics
                psnr_scores.append(float(psnr(target_image_np, gen_output_np, data_range=1.0)))
                ssim_scores.append(float(ssim(target_image_np, gen_output_np, win_size=11, channel_axis=-1, data_range=1.0)))
                mae_scores.append(float(mean_absolute_error(target_image_flat, gen_output_flat)))
                mse_scores.append(float(mean_squared_error(target_image_flat, gen_output_flat)))
                rmse_scores.append(float(np.sqrt(mean_squared_error(target_image_flat, gen_output_flat))))

    # Compute average scores
    avg_psnr = np.mean(psnr_scores)
    avg_ssim = np.mean(ssim_scores)
    avg_mae = np.mean(mae_scores)
    avg_mse = np.mean(mse_scores)
    avg_rmse = np.mean(rmse_scores)

    # Store metrics in a dictionary
    metrics = {
        'Avg_PSNR': avg_psnr,
        'Avg_SSIM': avg_ssim,
        'Avg_MAE': avg_mae,
        'Avg_MSE': avg_mse,
        'Avg_RMSE': avg_rmse
    }

    return metrics

def preprocess_batch(input_batch):
    # Transpose the input batch from [batch_size, height, width, channels] to [batch_size, channels, height, width]
    transposed_batch = tf.transpose(input_batch, perm=[0, 3, 1, 2])
    return transposed_batch
