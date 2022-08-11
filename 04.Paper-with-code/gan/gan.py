import tensorflow as tf
from tensorflow.keras import models, layers, utils
from utils import display_sample_img

class GAN():
  def __init__(self, gene, disc, img_shape, noise_dims):
    self.name = 'GAN'
    self.gene = gene
    self.disc = disc
    self.img_shape = img_shape
    self.noise_dims = noise_dims

  def compile(self, 
              loss = 'binary_crossentropy',
              optimizer = 'adam'):
    
    self.disc.compile(loss = loss, optimizer = optimizer)
    self.disc.trainable = False
    
    input_noise = layers.Input(shape=self.noise_dims)
    fake_img = self.gene(input_noise)
    logit = self.disc(fake_img)

    self.gan = models.Model(input_noise, logit, name=self.name)
    self.gan.compile(loss = loss, optimizer = optimizer)

  def _make_datasets(self, x_data):
      dataset = tf.data.Dataset.from_tensor_slices(x_data).shuffle(1)
      dataset = dataset.batch(self.batch_size, drop_remainder=True).prefetch(1)
      return dataset

  def _make_constants(self):
      zeros = tf.constant([0.], shape=[self.batch_size, 1])
      ones = tf.constant([1.], shape=[self.batch_size, 1] )
      return zeros, ones
  
  def _make_random(self):
      return tf.random.normal(shape=[self.batch_size, self.noise_dims])

  def fit(self, 
          x_data, 
          epochs=1,
          batch_size=32,
          standard=False
          ):
    
    # setting
    self.batch_size = batch_size
    train_ds = self._make_datasets(x_data)
    zeros, ones = self._make_constants()
    
    # train
    history = {'d_loss':[], 'g_loss':[]}
    for epoch in range(1, 1+epochs):
      if epoch>1: 
        for h in history: history[h].append(0)

        for real_imgs in train_ds:           
            # phase 1 - training the discriminator
            fake_imgs = self.gene.predict_on_batch(self._make_random())
            
            self.disc.trainable = True
            d_loss_real = self.disc.train_on_batch(real_imgs, ones)
            d_loss_fake = self.disc.train_on_batch(fake_imgs, zeros)
            d_loss = (0.5*d_loss_real) + (0.5*d_loss_fake)
            
            # phase 2 - training the generator
            self.disc.trainable = False
            g_loss = self.gan.train_on_batch(self._make_random() , ones)
            
            history['d_loss'][-1]+=d_loss
            history['g_loss'][-1]+=g_loss
        
        # end 1 epoch        
        print('* epoch: %i, d_loss: %f, g_loss: %f'%( epoch
                                                    , history['d_loss'][-1]
                                                    , history['g_loss'][-1]))
        
      fake_imgs = self.gene.predict(self._make_random())
      display_sample_img(fake_imgs, (2,8), standard=standard, size=2)