import dataset
import torch.nn as nn
class Model(nn.Module): # model sınıfı nn.Module sınıfından türetilir.
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=3, batch_first=True) # LSTM katmanı tanımlanır. batch: aynı anda paralel olarak işlenecek veri sayısı. 
        self.dropout = nn.Dropout(p=0.2) # Dropout katmanı tanımlanır. p: dropout oranı rastgele nöronların devre dışı bırakılma olasılığı. over fittingi önlemeye yardımcı olur.
        #dropout katmanı kullanılmayadabilir.durumuna göre karar verilecek.
        self.linear = nn.Linear(in_features=50, out_features=1) # Lineer katman tanımlanır. in_features: giriş özellik sayısı, out_features: çıkış özellik sayısı.
    def forward(self, x):
        x, _ = self.lstm(x) # LSTM katmanından geçirme
        
        x = self.dropout(x) # dropout katmanından geçirme

        x = x[: , -1 , :] # sadece lookback(seq_lenght) sonundaki zaman adımının çıktısını al 
        
        x = self.linear(x) # lineer katmandan geçirme

        return x
class EKG_Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=3, batch_first=True)
        self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(in_features=50, out_features=1)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = x[:, -1, :]
        x = self.linear(x)
        return x
