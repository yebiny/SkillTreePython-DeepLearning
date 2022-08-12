import tensorflow as tf
from tensorflow.keras import models, layers, utils
from utils import display_sample_img
import numpy as np

class CGAN():
  def __init__(self, gene, disc, img_shape, noise_dims, label_dims):
    self.name = 'CGAN'
    self.gene = gene
    self.disc = disc
    self.img_shape = img_shape
    self.noise_dims = noise_dims
    self.label_dims = label_dims

  def compile(self, 
              loss = 'binary_crossentropy',
              optimizer = 'adam'):
    
    self.disc.compile(loss = loss, optimizer = optimizer)
    self.disc.trainable = False
    
    # 인풋이 추가 (noise, label)
    input_noise = layers.Input(shape=self.noise_dims)
    input_label = layers.Input(shape=self.label_dims)

    fake_img = self.gene([input_noise, input_label])
    logit = self.disc([fake_img, input_label])

    self.cgan = models.Model([input_noise, input_label] , logit, name=self.name)
    self.cgan.compile(loss = loss, optimizer = optimizer)

  def _make_datasets(self, x_data, y_data):
      dataset = tf.data.Dataset.from_tensor_slices( (x_data, y_data ) ).shuffle(1)
      dataset = dataset.batch(self.batch_size, drop_remainder=True).prefetch(1)
      return dataset

  def _make_constants(self):
      zeros = tf.constant([0.], shape=[self.batch_size, 1])
      ones = tf.constant([1.], shape=[self.batch_size, 1] )
      return zeros, ones
  
  def _make_random(self):
      rnd_noises = tf.random.normal(shape=[self.batch_size, self.noise_dims])
      rnd_labels = np.random.randint(0, self.label_dims, self.batch_size )
      rnd_labels = utils.to_categorical(rnd_labels, self.label_dims)
      return rnd_noises, rnd_labels

  def fit(self, 
          x_data, 
          y_data,
          epochs=1,
          batch_size=32,
          standard=False
          ):
    
    # setting
    self.batch_size = batch_size
    train_ds = self._make_datasets(x_data, y_data)
    zeros, ones = self._make_constants()
    # for generator 
    seed_noises = tf.random.normal(shape=[30, self.noise_dims])
    seed_labels = np.arange(10)
    seed_labels = utils.to_categorical(seed_labels, 10)
    seed_labels = np.tile(seed_labels, (3,1))

    # train
    history = {'d_loss':[], 'g_loss':[]}
    for epoch in range(1+epochs):
      if epoch>0: 
        for h in history: history[h].append(0)

        for real_imgs, real_labels in train_ds:           
            # phase 1 - training the discriminator
            rnd_noises, _ = self._make_random()
            fake_imgs = self.gene.predict_on_batch([rnd_noises, real_labels])
            
            self.disc.trainable = True
            d_loss_real = self.disc.train_on_batch([real_imgs, real_labels], ones)
            d_loss_fake = self.disc.train_on_batch([fake_imgs, real_labels], zeros)
            d_loss = (0.5*d_loss_real) + (0.5*d_loss_fake)
            
            # phase 2 - training the generator
            self.disc.trainable = False
            rnd_noises, rnd_labels = self._make_random()
            g_loss = self.cgan.train_on_batch([rnd_noises , rnd_labels ] , ones)
            
            history['d_loss'][-1]+=d_loss
            history['g_loss'][-1]+=g_loss
        
        # end 1 epoch        
        print('* epoch: %i, d_loss: %f, g_loss: %f'%( epoch
                                                    , history['d_loss'][-1]
                                                    , history['g_loss'][-1]))
        
      fake_imgs = self.gene.predict([seed_noises, seed_labels])
      display_sample_img(fake_imgs, (3,10), standard=standard, size=2)