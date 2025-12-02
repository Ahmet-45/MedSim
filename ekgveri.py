import pandas as pd
import numpy as np
import torch

dfc = pd.read_csv('C:/Users/FURKAN KORKAT/Downloads/100_ekg.csv')

print("ilk 5 veri: ")
print(dfc.head())

print("\nveri seti bilgisi: ")
print(dfc.shape)
