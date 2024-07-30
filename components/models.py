import tensorflow as tf


def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU(alpha = 0.2))

  return result



def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result

def Generator_UNET():
  inputs = tf.keras.layers.Input(shape=[256, 256, 3])

  down_stack = [
    downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
    downsample(128, 4),  # (batch_size, 64, 64, 128)
    downsample(256, 4),  # (batch_size, 32, 32, 256)
    downsample(512, 4),  # (batch_size, 16, 16, 512)
    downsample(512, 4),  # (batch_size, 8, 8, 512)
    downsample(512, 4),  # (batch_size, 4, 4, 512)
    downsample(512, 4),  # (batch_size, 2, 2, 512)
    downsample(512, 4),  # (batch_size, 1, 1, 512)
  ]
  up_stack = [
    upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
    upsample(512, 4),  # (batch_size, 16, 16, 1024)
    upsample(256, 4),  # (batch_size, 32, 32, 512)
    upsample(128, 4),  # (batch_size, 64, 64, 256)
    upsample(64, 4),  # (batch_size, 128, 128, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(3, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh')  # (batch_size, 256, 256, 3)

  x = inputs


  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  for up, skip in zip(up_stack, skips):
    
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

#####################UNET++########################

def conv_block(inputs, num_filters):

    x = tf.keras.Sequential([
        tf.keras.layers.Conv2D(num_filters, 2, padding='same', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU()
    
    ])(inputs) 
    return x

def Generator_UNETPLUSPLUS():
    inputs = tf.keras.layers.Input(shape=[256, 256, 3])

    encoders = [
        downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
        downsample(128, 4),  # (batch_size, 64, 64, 128)
        downsample(256, 4),  # (batch_size, 32, 32, 256)
        downsample(512, 4),  # (batch_size, 16, 16, 512)
        downsample(512, 4),  # (batch_size, 8, 8, 512)
        downsample(512, 4),  # (batch_size, 4, 4, 512)
        downsample(512, 4),  # (batch_size, 2, 2, 512)
        downsample(512, 4),  # (batch_size, 1, 1, 512)
    ]
    decoders = [
        upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
        upsample(512, 4),  # (batch_size, 16, 16, 1024)
        upsample(256, 4),  # (batch_size, 32, 32, 512)
        upsample(128, 4),  # (batch_size, 64, 64, 256)
        upsample(64, 4),  # (batch_size, 128, 128, 128)
    ]

  
    initializer = tf.random_normal_initializer(0., 0.02)


    last = tf.keras.layers.Conv2DTranspose(3, 4,
                                            strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            activation='tanh')  # (batch_size, 256, 256, 3)


    x = inputs


    skips = []
    for down in encoders:
        x = down(x)
        skips.append(x)

    skips = skips[:-1]  # Skip the last layer

    for i, (up, skip) in enumerate(zip(decoders, reversed(skips))):
        x = up(x)
        # Perform attention mechanism
        g = tf.keras.layers.Conv2D(filters=512, kernel_size=1, strides=1, padding='same')(skip)
        x = tf.keras.layers.Conv2D(filters=512, kernel_size=1, strides=1, padding='same')(x)
        CONC = tf.keras.layers.Add()([g, x])
        CONC = tf.keras.layers.Activation('relu')(CONC)
        CONC = tf.keras.layers.Conv2D(filters=1, kernel_size=1, strides=1, padding='same')(CONC)
        CONC = tf.keras.layers.Activation('sigmoid')(CONC)
        x = tf.keras.layers.Multiply()([CONC, x])

        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


#################################DISCRIMINATOR###################################
def Discriminator_V2():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, 6)

    down1 = downsample(64, 4, False)(x)  # (batch_size, 128, 128, 128)

    down2 = downsample(128, 4)(down1)  # (batch_size, 64, 64, 256)

    down3 = downsample(256, 4)(down2)  # (batch_size, 32, 32, 512)

    down4 = downsample(512, 4)(down3)  # (batch_size, 16, 16, 512)

    down5 = downsample(512, 4)(down4)  # (batch_size, 8, 8, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(down5)  # (batch_size, 8, 8, 512)

    leaky_relu = tf.keras.layers.LeakyReLU(alpha = 0.2)(batchnorm1)  # (batch_size, 8, 8, 512)

    bottleneck = tf.keras.layers.Conv2D(512,2, strides=2,
                                        kernel_initializer=initializer)(leaky_relu)  # (batch_size, 4, 4, 512)
    up5 = upsample(512, 4)(bottleneck)  # (batch_size, 8, 8, 512)
    up4 = upsample(512, 4)(up5)  # (batch_size, 16, 16, 512)
    up3 = upsample(256, 4)(up4)  # (batch_size, 32, 32, 256)
    up2 = upsample(128, 4)(up3)  # (batch_size, 64, 64, 128)
    up1 = upsample(64, 4)(up2)  # (batch_size, 128, 128, 64)
    last = tf.keras.layers.Conv2DTranspose(3, 4, strides=2, padding='same',
                                           kernel_initializer=initializer)  # (batch_size, 256, 256, 3)
    x1 = tf.keras.layers.Concatenate()([up5, down5])
    x1 = tf.keras.layers.Conv2D(512, 4, padding='same')(x1)
    x1 = tf.keras.layers.UpSampling2D()(x1)
    x2 = tf.keras.layers.Concatenate()([up4, down4, x1])
    x2 = tf.keras.layers.Conv2D(512, 4, padding='same')(x2)
    x2 = tf.keras.layers.UpSampling2D()(x2)
    x3 = tf.keras.layers.Concatenate()([up3, down3, x2])
    x3 = tf.keras.layers.Conv2D(256, 4, padding='same')(x3)
    x3 = tf.keras.layers.UpSampling2D()(x3)
    x4 = tf.keras.layers.Concatenate()([up2, down2, x3])
    x4 = tf.keras.layers.Conv2D(128, 4, padding='same')(x4)
    x4 = tf.keras.layers.UpSampling2D()(x4)
    x5 = tf.keras.layers.Concatenate()([up1, down1, x4])
    x5 = tf.keras.layers.Conv2D(64, 4, padding='same')(x5)

    output = last(x5)

    return tf.keras.Model(inputs=[inp, tar], outputs=output)


def Discriminator():
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
  tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')

  x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

  down1 = downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
  down2 = downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
  down3 = downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
  conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

  last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

  return tf.keras.Model(inputs=[inp, tar], outputs=last)