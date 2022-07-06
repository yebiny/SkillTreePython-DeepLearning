import matplotlib.pyplot as plt
import numpy as np

def plot_lcurve(hists, titles, colors, figsize=(12,4), grid=True):
  plt.figure(figsize=figsize)
  # loss 
  plt.subplot(121)
  for i, hist in enumerate(hists):
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    epochs = range(1, 1+len(loss))
    plt.plot(epochs, loss, linestyle=':', label=f'{titles[i]} train loss', c=colors[i])
    plt.plot(epochs, val_loss, marker='.', label=f'{titles[i]} valid loss', c=colors[i])
    plt.legend();plt.grid(grid);plt.xticks(epochs)
    plt.xlabel('Epochs');plt.ylabel('Loss')
    x, y = epochs[-1], hist.history['loss'][-1]
    plt.text(x, y, np.round(y,2), c=colors[i])
    x, y = epochs[-1], hist.history['val_loss'][-1]
    plt.text(x, y, np.round(y,2), c=colors[i])
  # acc
  plt.subplot(122)
  for i, hist in enumerate(hists):
    acc = hist.history['acc']
    val_acc = hist.history['val_acc']
    plt.plot(epochs, acc, linestyle=':', label=f'{titles[i]} train acc', c=colors[i])
    plt.plot(epochs, val_acc, marker='.', label=f'{titles[i]} valid acc', c=colors[i])
    plt.legend();plt.grid(grid);plt.xticks(epochs)
    plt.xlabel('Epochs');plt.ylabel('Acc')
    x, y = epochs[-1], hist.history['acc'][-1]
    plt.text(x, y, np.round(y,2), c=colors[i])
    x, y = epochs[-1], hist.history['val_acc'][-1]
    plt.text(x, y, np.round(y,2), c=colors[i])
  plt.show()
