import tensorflow as tf
from tensorflow.keras import models, layers, utils

class BuildModel():
  def __init__(self, img_shape, z_dim):
    self.img_shape = img_shape
    self.z_dim = z_dim

  def build_gene(self, 
                 activation = 'selu',
                 last_activation='sigmoid',
                 kernel_size=5):
    
    h, w, c = self.img_shape
    z = layers.Input(shape=[self.z_dim])
    y = layers.Dense( int(w/4)*int(h/4)*128)(z)
    y = layers.Reshape( [int(w/4),int(h/4),128] )(y)
    
    y = layers.BatchNormalization()(y)
    y = layers.Conv2DTranspose(64, kernel_size=5, padding='same', strides=2, activation=activation)(y)
    y = layers.BatchNormalization()(y)
    y = layers.Conv2DTranspose(c, kernel_size=5, padding='same', strides=2, activation=last_activation)(y)
    
    return models.Model(z, y, name='Generator')

  def build_disc(self,
                 activation='relu',
                 last_activation='sigmoid',
                 kernel_size=5):
    
    x = layers.Input(shape=self.img_shape)
    y = layers.Conv2D(64, kernel_size=kernel_size, strides=2, padding='same', activation=activation)(x)
    y = layers.Dropout(.5)(y)
    y = layers.Conv2D(128, kernel_size=kernel_size, strides=2, padding='same', activation=activation)(y)
    y = layers.Dropout(.5)(y)
    y = layers.Flatten()(y)

    y = layers.Dense(1, activation=last_activation)(y)
    return models.Model(x, y, name='Discriminator')