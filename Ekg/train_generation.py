import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import Ekg.model as model
from Ekg.model import Model as LSTMModel
import Ekg.dataset as dataset 
from Ekg.dataset import X_train, y_train, X_test, y_test, dataMed,seq_length,train_size

def generate_autoregressive(model, start_sequence, generation_length ,seq_length):
    model.eval()  # Değerlendirme moduna geçirme
    current_sequence = start_sequence.clone().detach().to(torch.float32).unsqueeze(0) # Başlangıç sekansını klonla ve ayır
    generated_list = current_sequence.squeeze(0).tolist()  # Başlangıç sekansını listeye ekle

    with torch.no_grad():  # Değerlendirme sırasında gradyan hesaplamalarını devre dışı bırakır
        for _ in range(generation_length):
            predicted_output = model(current_sequence)  # Modeli kullanarak bir sonraki değeri tahmin et
            new_value = predicted_output.item()  # Tahmin edilen değeri al

            generated_list.append(new_value)  # Yeni değeri listeye ekle

            current_sequance = current_sequence[: , 1:, :] # Mevcut sekansın ilk değerini çıkar
            current_sequence = torch.cat((current_sequance, predicted_output.item()), dim=1)  # Yeni değeri sekansın sonuna ekle