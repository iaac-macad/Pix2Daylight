import tensorflow as tf
import time
import datetime
from components.losses import *
from components.utils import preprocess_batch
import os
import logging
from tqdm import tqdm

def fit(train_ds,
        epochs,   
        generator, 
        discriminator, 
        generator_optimizer, 
        discriminator_optimizer,
        path_to_save,
        loss_object,
        folder_name,
        batch_size,
        scheduler):
    


    log_dir = "logs/"
    summary_writer = tf.summary.create_file_writer(log_dir + "fit/" + folder_name)
    
    train_dataset = train_ds.batch(batch_size)
    train_dataset = train_dataset.shuffle(len(train_dataset),reshuffle_each_iteration=True)
    stepss = 0

    for epoch in range(epochs):
        with tqdm(total=len(train_dataset), desc=f'Training Epoch {epoch + 1}/{epochs}', unit='batch') as pbar:
            pbar.set_description(f'Epoch {epoch+1}/{epochs}')

            
            avg_gtl = tf.keras.metrics.Mean()
            avg_ggl = tf.keras.metrics.Mean()
            avg_gll = tf.keras.metrics.Mean()
            avg_gdl = tf.keras.metrics.Mean()

        
            for step, (input_image, target) in train_dataset.repeat().take(len(train_dataset)).enumerate():
                pbar.set_description(f'Epoch {epoch}, Step {step}')

                gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss,generator_gradients,discriminator_gradients = train_step(input_image=input_image, 
                                                                                    target=target, 
                                                                                    generator=generator, 
                                                                                    discriminator=discriminator, 
                                                                                    loss_object=loss_object,
                                                                                    generator_optimizer=generator_optimizer,
                                                                                    discriminator_optimizer=discriminator_optimizer)
                generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
                discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))
                
                
                
                
                
                
                
                avg_gtl.update_state(gen_total_loss)
                avg_ggl.update_state(gen_gan_loss)
                avg_gll.update_state(gen_l1_loss)
                avg_gdl.update_state(disc_loss)    
            
            

        
                
                #scheduler.update_losses(gen_gan_loss, disc_loss)
                
                #lr_gen, lr_disc = scheduler()
                #generator_optimizer.learning_rate.assign(lr_gen)
                #discriminator_optimizer.learning_rate.assign(lr_disc)


                # Apply the updated learning rates
                with summary_writer.as_default():
                    tf.summary.scalar('gen_total_loss', gen_total_loss, step=stepss)
                    tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=stepss)
                    tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=stepss)
                    tf.summary.scalar('disc_loss', disc_loss, step=stepss)
                    #tf.summary.scalar('learning_rate_generator', lr_gen, step=step)
                    #tf.summary.scalar('learning_rate_discriminator', lr_disc, step=step)

            
    
                pbar.set_postfix({
                    'gen_total': f"{avg_gtl.result().numpy():.5g}" , 
                    'gen_gan': f"{avg_ggl.result().numpy():.5g}", 
                    'gen_l1':f"{avg_gll.result().numpy():.5g}", 
                    'disc': f"{avg_gdl.result().numpy():.5g}",
                })
                pbar.update()
                stepss = (stepss +batch_size)
            if (epoch + 1) % 10 == 0:
                new_lr_gen = generator_optimizer.learning_rate.numpy() * 0.5
                generator_optimizer.learning_rate.assign(new_lr_gen)
                logging.info(f"Reduced generator learning rate to {new_lr_gen}")

                new_lr_disc = discriminator_optimizer.learning_rate.numpy() * 0.5
                discriminator_optimizer.learning_rate.assign(new_lr_disc)
                logging.info(f"Reduced discriminator learning rate to {new_lr_disc}")


            avg_gtl.reset_states()
            avg_ggl.reset_states()
            avg_gll.reset_states()
            avg_gdl .reset_states()   
    logging.info('Training completed')
    logging.info(f'Model saved to {path_to_save}')
    # Save the final model
    os.makedirs(path_to_save, exist_ok=True)
    generator.save(os.path.join(path_to_save, 'generator_model.keras'))
    print("Final models saved successfully")

@tf.function
def train_step(
               input_image,
               target,
               generator,
               discriminator,
               loss_object,
               generator_optimizer,
               discriminator_optimizer):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target,loss_object)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output,loss_object)

    generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

    

    return gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss,generator_gradients,discriminator_gradients