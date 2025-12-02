import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
satir_basina_deger = 2000
toplam_deger = 10000
satir_sayisi = int(np.ceil(toplam_deger/satir_basina_deger))
df = pd.read_csv('C:/Users/FURKAN KORKAT/Downloads/106_ekg.csv')
df_cleaned = df.drop(columns=[ 'time' , 'V1' , 'symbol' ])  # fazla verileri kaldır
dataMed = df[["MLII"]].values.astype('float32')
df_cleaned.head(n=10000)
fig, ax = plt.subplots(satir_sayisi,1,figsize=(12,2*satir_sayisi),sharey=True)
plt.subplots_adjust(hspace=0.5)

for i in range(satir_sayisi):
    start_index = i * satir_basina_deger
    end_index = min((i + 1) * satir_basina_deger, toplam_deger)
    ax[i].grid(which = 'major',linestyle = '--',linewidth = 0.2,color ='black')
    ax[i].plot(df_cleaned['MLII'].values[start_index:end_index])
    ax[i].set_title(f'EKG Verisi Bölüm {i+1}')
plt.show()

train_size = int(len(dataMed) * 0.67) # verinin %67 i eğitim için kullanılır 
test_size = len(dataMed) - train_size # kalan %20 si test için kullanılır   #data verisi daha veriler hazır olmadığı için tanımlı değil.
train, test = dataMed[:train_size], dataMed[train_size:] # veriyi eğitim ve test olarak ayırma



class EKG_Dataset(Dataset):
    def __init__(self,X_data,Y_data):
        self.X_data = X_data
        self.Y_data = Y_data
    def __len__(self):
        return self.X_data.shape[0]
    def __getitem__(self, index):
        return self.X_data[index],self.Y_data[index]
def create_sequance(data_array:np.ndarray,seq_length: int):
    """
    This function creates sequences of data for time series analysis.

    args:
        seq_length (int): her bir sekansin uzunluğu.
        data_array (np.ndarray): girilen veri arrayi.
    """
    X , y = [],[]

    for i in range(len(data_array)- seq_length): # range arrayin uzunluğu - seq_length (V0,V1,V2 - V1,V2,V3) kaç işlem
        feature_sequance = data_array[i : i+seq_length]  # yaparak bu tarz gruplandırma yapılır.
        label = data_array[i + seq_length]# bir sonraki elemanlar etiket olarak alınır.  sequance to one  data_array[i+1:i+ seq_length + 1] sequance to sequance

        X.append(feature_sequance)
        y.append(label)
    
    X_tensor = torch.tensor(np.array(X),dtype=torch.float32)
    y_tensor = torch.tensor(np.array(y),dtype=torch.float32)

    return X_tensor,y_tensor
seq_length = 2
X_train , y_train = create_sequance(train,seq_length=seq_length)
X_test , y_test = create_sequance(test,seq_length=seq_length)

