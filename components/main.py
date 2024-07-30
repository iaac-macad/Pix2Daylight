import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import json
import pathlib
import logging
from components.folder_to_targz import *
from components.image_combining import *
from components.losses import *
from components.models import *
from components.random_split import split_dataset
from components.train_test import *
from components.utils import *
from datetime import datetime

def main(TRAIN_NUMBER, BATCH_SIZE , LEARNING_RATE ,EPOCH , BETA_1  , ALPHA , BETA,DIS ="v1",GEN ="v1"):
    now = datetime.now()
    current_time = now.strftime("%H_%M_%S")

# Create the folder name with the current time
    folder_name = f"train_{TRAIN_NUMBER}_batch_{BATCH_SIZE}_lr_{LEARNING_RATE}_epoch_{EPOCH}_beta_{BETA_1}_{GEN}_{DIS}_{current_time}"
    if not os.path.exists(folder_name):
        os.makedirs(f"train{TRAIN_NUMBER}/{folder_name}", exist_ok=True)
    else:
        print(f"Folder 'train{TRAIN_NUMBER}/{folder_name}' already exists.")

        
    ###############LOSS FUNCTION###########################
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True) 
    ###############LOSS FUNCTION###########################

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Ensure TensorFlow uses the GPU
    logger.info("Checking GPU availability")
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    

    #logger.info("Combining images for train number: %s", TRAIN_NUMBER)
    #combine_image(TRAIN_NUMBER)

    #logger.info("Splitting dataset")
    #split_dataset(source_dir=f"train{TRAIN_NUMBER}/dataset/combined_images", dest_dir=f"train{TRAIN_NUMBER}/dataset/train_test", train_ratio=0.9, seed=42)

    #logger.info("Creating tar file of the dataset")
    #make_tarfile(output_filename=f"train{TRAIN_NUMBER}/dataset/archive.tar.gz", source_dir=f"train{TRAIN_NUMBER}/dataset/train_test")

    local_path = f"train{TRAIN_NUMBER}/dataset/archive.tar.gz"  
    path_to_zip = pathlib.Path(local_path)

    if path_to_zip.suffix == ".gz":
        logger.info("Extracting tar.gz file")
        import tarfile
        with tarfile.open(path_to_zip, 'r:gz') as tar:
            tar.extractall(path=path_to_zip.parent)

    PATH = f"train{TRAIN_NUMBER}/dataset/train_test"

    logger.info("Applying random jitter to images")
    
    logger.info("Creating training dataset pipeline")
    ###################DATAPREP#####################
    train_dataset = tf.data.Dataset.list_files(f'{PATH}/train/*.png')
    train_dataset = train_dataset.map(lambda x: load_image_train(x, 256, 256),
                                        num_parallel_calls=tf.data.AUTOTUNE)

    test_dataset = tf.data.Dataset.list_files(f'{PATH}/test/*.png')
    test_dataset = test_dataset.map(lambda x: load_image_test(x, img_height=256, img_width=256))
    test_dataset = test_dataset.batch(BATCH_SIZE)
    ###################DATAPREP#####################




    #################MODELS########################################
    logger.info("Creating generator model")
    if GEN == "v1":
        generator = Generator_UNET()
    elif GEN == "v2":
        generator = Generator_UNETPLUSPLUS()
 
    logger.info("Creating discriminator model")
 

    if DIS == "v1":
        discriminator = Discriminator()
    elif DIS == "v2":
        discriminator = Discriminator_V2()
###########################################################################




    logger.info("Setting up optimizers")

    alpha = ALPHA
    beta = BETA
    scheduler = GapAwareLearningRateScheduler(LEARNING_RATE, alpha, beta)
    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=BETA_1)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=BETA_1)


    logger.info("Starting training process")



    fit(train_ds=train_dataset, 
        epochs = EPOCH,
        batch_size = BATCH_SIZE, 
        generator=generator, 
        discriminator=discriminator, 
        generator_optimizer=generator_optimizer, 
        discriminator_optimizer=discriminator_optimizer,
        path_to_save=f"train{TRAIN_NUMBER}/{folder_name}/models",
        loss_object=loss_object,
        folder_name=folder_name,
        scheduler = scheduler)
    
    logger.info("Saving the generator model")
    generator.save(f"train{TRAIN_NUMBER}/{folder_name}/models/generator_model")
    parameters = {
        "TRAIN_NUMBER": TRAIN_NUMBER,
        "BATCH_SIZE": BATCH_SIZE,
        "EPOCH": EPOCH,
        "LEARNING_RATE": LEARNING_RATE,
        "DISCRIMINATOR ": DIS,
        "GENERATOR":GEN
    }

    output_folder = f"train{TRAIN_NUMBER}/{folder_name}"
    output_filename = "parameters.json"

    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, output_filename)

    with open(output_path, 'w') as json_file:
        json.dump(parameters, json_file, indent=4)

    logger.info(f"Parameters saved to {output_path}")

    # Restore the latest checkpoint and test the network
    logger.info("Restoring the latest checkpoint")
    logger.info("loading saved model")
    logger.info("predicting and saving images")
    loaded_generator = tf.keras.models.load_model(f"train{TRAIN_NUMBER}/{folder_name}/models/generator_model")
    for i, (img, tar) in enumerate(test_dataset.take(len(test_dataset))):
    # Check the batch size
        batch_size = img.shape[0]
    
        if batch_size == 1:
            img_name = f"image_{i}"
            save_combined_images(
                model=loaded_generator,
                test_input=img,
                tar=tar,
                output_folder=f"train{TRAIN_NUMBER}/{folder_name}/predictions",
                image_name=img_name
            )
        else:
            for j in range(batch_size):
                img_name = f"image_{i}_batch_{j}"
                save_combined_images(
                    model=loaded_generator,
                    test_input=img[j:j+1],  # Slice to keep the dimension
                    tar=tar[j:j+1],          # Slice to keep the dimension
                    output_folder=f"train{TRAIN_NUMBER}/{folder_name}/predictions",
                    image_name=img_name
                )

    metrics = calculate_metrics(test_dataset, loaded_generator)

    # Save metrics as JSON
    with open(f"train{TRAIN_NUMBER}/{folder_name}/metrics.json", 'w') as f:
        json.dump(metrics, f, indent=4)

    logger.info("Metrics saved successfully.")
