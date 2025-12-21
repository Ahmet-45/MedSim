import os
import pandas as pd
from src.hemogram_ai import HemogramGenerator


def klasorleri_kontrol_et():
    # Gerekli klasörlerin varlığını kontrol et ve yoksa oluştur
    if not os.path.exists('cikti'):
        os.makedirs('cikti')
        print("'cikti' klasörü oluşturuldu.")

def main():
    klasorleri_kontrol_et()

    # HemogramGenerator sınıfını başlat ve modeli eğit
    hemogram_ai_model = HemogramGenerator()

    #Veriyi okuma
    veri_yolu = os.path.join('data', 'MedSim_100_Hasta.csv')
    hemogram_ai_model.fit(veri_yolu)

    # Veri üretme
    #Demir eksikliği anemisi için 50 örnek
    df_demir_eksikligi = hemogram_ai_model.generate(adet=50, durum='demir_eksikligi')
    #B12 vitamini eksikliği anemisi için 50 örnek
    df_b12_eksikligi = hemogram_ai_model.generate(adet=50, durum='b12_eksikligi')        
    #Sağlıklı bireyler için 100 örnek
    df_saglikli = hemogram_ai_model.generate(adet=100, durum='saglikli')
    #Tüm veriyi birleştirme
    tum_veri = pd.concat([df_demir_eksikligi, df_b12_eksikligi, df_saglikli], ignore_index=True)
    # Üretilen veriyi CSV dosyasına kaydetme
    kayit_yolu = os.path.join('cikti', 'uretilen_hemogram_verisi.csv')

    tum_veri.to_csv(kayit_yolu, sep=';', decimal=',', index=False, encoding='utf-8-sig')
    print(f"Üretilen veri '{kayit_yolu}' dosyasına kaydedildi.")
if __name__ == "__main__":
    main()    