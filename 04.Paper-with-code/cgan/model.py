import tensorflow as tf
from tensorflow.keras import models, layers, utils
from tensorflow.keras import backend as K

class BuildModel():
  def __init__(self, img_shape, z_dim, label_dim):
    self.img_shape = img_shape
    self.z_dim = z_dim
    self.label_dim = label_dim

  def build_gene(self, 
                 activation = 'selu',
                 last_activation='sigmoid',
                 kernel_size=5):

    h, w, ch = self.img_shape
    
    z = layers.Input(shape=[self.z_dim,], name='noise')
    c = layers.Input(shape=[self.label_dim,], name='condition')
    y = layers.concatenate([z, c])

    y = layers.Dense(int(w/4)*int(h/4)*128)(y)
    y = layers.Reshape( [int(w/4),int(h/4),128] )(y)
    y = layers.BatchNormalization()(y)
    y = layers.Conv2DTranspose(64, kernel_size=5, padding='same', strides=2, activation=activation)(y)
    y = layers.BatchNormalization()(y)
    y = layers.Conv2DTranspose(ch, kernel_size=5, padding='same', strides=2, activation=last_activation)(y)
    
    return models.Model([z, c], y, name='Generator')

  def build_disc(self,
                 activation='relu',
                 last_activation='sigmoid',
                 kernel_size=5):

    h, w, ch = self.img_shape
    def _expand_label_input(x):
      y = K.expand_dims(x, axis=1)
      y = K.expand_dims(y, axis=1)
      y = K.tile(y, [1, h, w, 1])
      return y

    x = layers.Input(shape=self.img_shape, name='image')
    c = layers.Input(shape= self.label_dim, name='condition')
    c = layers.Lambda(_expand_label_input)(c)
    
    y = layers.concatenate([x, c], axis=3)
    y = layers.Conv2D(64, kernel_size=kernel_size, strides=2, padding='same', activation=activation)(y)
    y = layers.Dropout(.5)(y)
    y = layers.Conv2D(128, kernel_size=kernel_size, strides=2, padding='same', activation=activation)(y)
    y = layers.Dropout(.5)(y)
    y = layers.Flatten()(y)

    y = layers.Dense(1, activation=last_activation)(y)
    return models.Model([x,c], y, name='Discriminator')

