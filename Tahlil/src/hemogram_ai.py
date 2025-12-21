import pandas as pd
import numpy as np

class HemogramGenerator:
    def __init__(self):
        self.stats = {}
        self.columns = []

    def fit(self, dosya_yolu):  
        try:
            df = pd.read_csv(dosya_yolu, sep=';', decimal=',')
            print(f"Veri basariyla okundu: {dosya_yolu}")
        except FileNotFoundError:
            print(f"HATA: {dosya_yolu} bulunamadı.")
            return

        numeric_df = df.select_dtypes(include=[np.number])
        if 'Hasta_ID' in numeric_df.columns:
            numeric_df = numeric_df.drop(columns=['Hasta_ID'])

        self.columns = numeric_df.columns
        for col in self.columns:
            self.stats[col]  = {
                'mean': numeric_df[col].mean(),
                'std': numeric_df[col].std()
            }
        print(f"Model Egitildi! {len(self.columns)} parametre analiz edildi.")

    def generate(self, adet=1, durum='saglikli', gurultu=1.0):
        if not self.stats:
            print("⚠️ Model henüz eğitilmedi, boş veri dönülüyor.")
            return pd.DataFrame()
        
        data = []
        for _ in range(adet):
            row = {}
            for col in self.columns:
                mu = self.stats[col]['mean']
                sigma = self.stats[col]['std']
                val = np.random.normal(mu, sigma * gurultu)

                row[col] = max(0.01, val) # Negatif değerleri önleme
                
            #Hastalık durumlarına göre bazı parametreleri ayarlama
            if durum == 'demir_eksikligi':
                row['HGB'] -= self.stats['HGB']['std'] * np.random.uniform(2, 4)
                row['MCV'] = np.random.uniform(60, 78)
                row['MCH'] = np.random.uniform(20, 26)
                row['RDW-CV'] += np.random.uniform(3, 8)
                row['Teşhis'] = 'Demir Eksikliği Anemisi'

            elif durum == 'b12_eksikligi':
                row['HGB'] -= self.stats['HGB']['std'] * np.random.uniform(1.5, 3.5)
                row['MCV'] = np.random.uniform(105, 125)
                row['WBC'] *= 0.85
                row['PLT'] *= 0.80
                row['Teşhis'] = 'B12 Vitamini Eksikliği Anemisi'

            elif durum == 'saglikli':
                row['Teşhis'] = 'Sağlıklı'

            else:
                row['Teşhis'] = 'Bilinmeyen Durum'

            data.append(row)      

        return pd.DataFrame(data)            
