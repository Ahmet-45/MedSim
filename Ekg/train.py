import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from Ekg.model import Model
import Ekg.dataset as dataset 
from Ekg.dataset import X_train, y_train, X_test, y_test, dataMed,seq_length,train_size

modelT = Model()
optimizer = optim.Adam(modelT.parameters()) # Adam optimizasyon algoritması kullanılır.
loss_fn = nn.MSELoss() 
loader = data.DataLoader(data.TensorDataset(X_train, y_train), batch_size=8, shuffle=True)

n_epochs = 2000
for epoch in range(n_epochs):
    modelT.train()
    for X_batch, y_batch in loader:
        y_pred = modelT(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # her 100 epoch'ta bir kaybı yazdır
        if epoch % 100 != 0:
            continue
        modelT.eval()
        with torch.no_grad(): # değerlendirme sırasında gradyan hesaplamalarını devre dışı bırakır
            y_pred_train = modelT(X_batch)
            train_rmse = np.sqrt(loss_fn(y_pred_train, y_train)) # eğitim verisi üzerindeki RMSE hesaplama
            y_pred_test = modelT(X_test)
            test_rmse = np.sqrt(loss_fn(y_pred_test, y_test)) # test verisi üzerindeki RMSE hesaplama
        print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))
    with torch.no_grad(): # değerlendirme sırasında gradyan hesaplamalarını devre dışı bırakır
        
        train_plot = np.ones_like(dataMed) * np.nan
        y_pred_train_2 = modelT(X_train)
        train_plot[seq_length:train_size] = modelT(X_train) # 
        
        test_plot = np.ones_like(dataMed) * np.nan
        test_plot[train_size+seq_length:len(dataMed)] = modelT(X_test)
# plot
plt.plot(dataMed, c='b')
plt.plot(train_plot, c='r')
plt.plot(test_plot, c='g')
plt.show()

torch.save(modelT.state_dict(),'LSTMmodel.pth')
modelT.load_state_dict(torch.load('LSTMmodel.pth',weights_only=True)) 
modelT.eval() # değerlendirme moduna geçirme

# Model checkpointi için kaydetme ve yükleme örneği
"""
PATH = 'model_checkpoint.pth' 
torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            ...
            }, PATH)

model = TheModelClass(*args, **kwargs)
optimizer = TheOptimizerClass(*args, **kwargs)

checkpoint = torch.load(PATH, weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()
# - ya da -
model.train()
"""