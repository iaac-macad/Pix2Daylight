import tensorflow as tf
@tf.function

def generator_loss(disc_generated_output, gen_output, target,loss_object):
  LAMBDA = 100
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  # Mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

  total_gen_loss = gan_loss + (LAMBDA * l1_loss)

  return total_gen_loss, gan_loss, l1_loss


def discriminator_loss(disc_real_output, disc_generated_output,loss_object):
  
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss


def ssim_score(target, gen_output):
    ssim = tf.image.ssim(target, gen_output, max_val=1.0)
    return tf.reduce_mean(ssim)

def psnr_score(target, gen_output):
    psnr = tf.image.psnr(target, gen_output, max_val=1.0)
    return tf.reduce_mean(psnr)

def mse_score(target, gen_output):
    mse = tf.reduce_mean(tf.square(target - gen_output))
    return mse

import tensorflow as tf

class GapAwareLearningRateScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, alpha, beta):
        super(GapAwareLearningRateScheduler, self).__init__()
        self.initial_lr = initial_lr
        self.alpha = alpha  # Scaling factor for increasing learning rate
        self.beta = beta    # Scaling factor for decreasing learning rate
        self.generator_loss = tf.Variable(0.0, trainable=False)
        self.discriminator_loss = tf.Variable(0.0, trainable=False)

    def update_losses(self, generator_loss, discriminator_loss):
        self.generator_loss.assign(generator_loss)
        self.discriminator_loss.assign(discriminator_loss)

    def __call__(self):
        gap = self.generator_loss - self.discriminator_loss
        lr_generator = self.initial_lr * tf.exp(-self.alpha * tf.maximum(gap, 0))
        lr_discriminator = self.initial_lr * tf.exp(self.beta * tf.minimum(gap, 0))
        return lr_generator, lr_discriminator