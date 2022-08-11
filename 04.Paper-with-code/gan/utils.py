import matplotlib.pyplot as plt
import numpy as np

def display_sample_img(samples, grid, standard=False, size=1):
  h, w = grid
  n = np.cumprod(grid)[-1]
  plt.figure(figsize=(w*size,h*size))
  for i, sample in enumerate(samples[:n]):
    if i==n: break
    if standard:
      sample = ( sample  + 1. ) / 2.
    sample = np.clip(sample, 0, 1)
    plt.subplot(h,w,i+1)
    if sample.shape[-1]==1: plt.imshow(sample[:,:,0], cmap='gray_r')
    else: plt.imshow(sample)
    plt.xticks([]);plt.yticks([])
  plt.show()