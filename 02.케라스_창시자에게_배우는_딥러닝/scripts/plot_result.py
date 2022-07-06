import matplotlib.pyplot as plt
import numpy as np

def plot_lcurve(hists, titles, colors, size=(12,4), x_itv=1):
  plt.figure(figsize=size)
  # loss 
  plt.subplot(121)
  for i, hist in enumerate(hists):
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    epochs = range(1, 1+len(loss))
    xbins = range(0, len(loss)+1, x_itv)
    plt.plot(epochs, loss, linestyle=':', label=f'{titles[i]} train loss', c=colors[i])
    plt.plot(epochs, val_loss, marker='.', label=f'{titles[i]} valid loss', c=colors[i])
    plt.legend();plt.grid(True);plt.xticks(xbins)
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
    plt.legend();plt.grid(True);plt.xticks(xbins)
    plt.xlabel('Epochs');plt.ylabel('Acc')
    x, y = epochs[-1], hist.history['acc'][-1]
    plt.text(x, y, np.round(y,2), c=colors[i])
    x, y = epochs[-1], hist.history['val_acc'][-1]
    plt.text(x, y, np.round(y,2), c=colors[i])
  plt.show()
