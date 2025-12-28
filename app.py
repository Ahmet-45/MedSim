import os
import pandas as pd
import numpy as np
import random
from flask import Flask, render_template, jsonify, request

# --- 1. İSTATİSTİKSEL MODÜLÜ YÜKLE ---
try:
    from Tahlil.src.hemogram_ai import HemogramGenerator
    print(" İstatistiksel Modül Yüklendi.")
except ImportError:
    print(" HATA: src klasörü bulunamadı.")

# --- 2. GAN YAPAY ZEKA MODELİNİ YÜKLE (.pkl) ---
gan_model = None
try:
    from sdv.single_table import CTGANSynthesizer
    pkl_path = os.path.join('Tahlil', 'hemogram_model_IDA.pkl')
    
    if os.path.exists(pkl_path):
        gan_model = CTGANSynthesizer.load(pkl_path)
        print(" GAN Yapay Zeka Modeli (.pkl) Yüklendi! ")
    else:
        print(f" UYARI: {pkl_path} bulunamadı.")
except Exception as e:
    print(f" GAN hatası: {e}")

app = Flask(__name__)

# --- MODELLERİ BAŞLAT ---
hemogram_stat_model = None
try:
    hemogram_stat_model = HemogramGenerator()
    csv_path = os.path.join('Tahlil', 'data', 'MedSim_100_Hasta.csv')
    if os.path.exists(csv_path):
        hemogram_stat_model.fit(csv_path)
except Exception as e:
    print(f" İstatistik model hatası: {e}")


# --- TABLO OLUŞTURUCU ---
def tahlil_tablosu_olustur(durum):
    if not hemogram_stat_model: return "Model çalışmıyor."
    
    # 1. ADIM: Taslak veri üret
    df_stat = hemogram_stat_model.generate(adet=1, durum=durum)
    if df_stat.empty: return "Veri üretilemedi."
    
    hasta_data = df_stat.iloc[0].to_dict()
    if 'Teşhis' in hasta_data: del hasta_data['Teşhis']

    # 2. ADIM: Kritik değerleri .pkl (GAN) modelinden çek
    if gan_model:
        try:
            gan_data = gan_model.sample(num_rows=1)
            
            hgb_gan = float(gan_data['Hemoglobin'].values[0])
            mcv_gan = float(gan_data['MCV'].values[0])
            
            # RDW Kontrolü
            if 'RDW' in gan_data.columns:
                rdw_gan = float(gan_data['RDW'].values[0])
            else:
                rdw_gan = None

            # Hastalık Durumuna Göre Manipülasyon (Doktor Müdahalesi)
            if durum == 'demir_eksikligi':
                if hgb_gan > 11.5: hgb_gan = hgb_gan * 0.75 
                if mcv_gan > 78: mcv_gan = mcv_gan * 0.80
            elif durum == 'b12_eksikligi':
                if mcv_gan < 100: mcv_gan = mcv_gan * 1.25

            # Değerleri Güncelle
            hasta_data['HGB'] = hgb_gan
            hasta_data['MCV'] = mcv_gan
            if rdw_gan:
                hasta_data['RDW-CV'] = rdw_gan

            # Konsola yazmaya devam etsin 
            print(f" AI ÜRETİMİ: HGB={hgb_gan:.2f}, MCV={mcv_gan:.2f} (Durum: {durum})")

        except Exception as e:
            print(f" GAN hatası: {e}")

    # 3. ADIM: HTML Tablosunu Oluştur 
    html = "<div class='table-responsive'>"
    html += "<table class='table table-hover custom-table'>"
    html += "<thead><tr><th>TETKİK</th><th>SONUÇ</th><th>BİRİM</th><th>DURUM</th></tr></thead><tbody>"

    for key, val in hasta_data.items():
        if key not in hemogram_stat_model.stats: continue
        
        mu = hemogram_stat_model.stats[key]['mean']
        sigma = hemogram_stat_model.stats[key]['std']
        ref_alt = max(0, mu - 2 * sigma)
        ref_ust = mu + 2 * sigma
        
        durum_text, renk_class = "Normal", ""
        
        if val < ref_alt: 
            durum_text, renk_class = "Düşük (L)", "deger-dusuk"
        elif val > ref_ust: 
            durum_text, renk_class = "Yüksek (H)", "deger-yuksek"

        birim = ""
        if key in ["HGB", "MCHC"]: birim = "g/dL"
        elif key in ["MCV", "MPV"]: birim = "fL"
        elif key in ["WBC", "PLT"]: birim = "10^3/uL"
        elif key in ["RBC"]: birim = "10^6/uL"
        elif key in ["HCT"]: birim = "%"
        
        html += f"<tr><td>{key}</td><td class='{renk_class} font-weight-bold'>{val:.2f}</td><td>{birim}</td><td class='{renk_class}'>{durum_text}</td></tr>"

    html += "</tbody></table></div>"
    return html

# --- ROTALAR ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message', '').lower()
    
    # TAHLİL VE HEMOGRAM KOMUTU
    if "tahlil" in user_input or "hemogram" in user_input:
        olasiliklar = ['saglikli', 'demir_eksikligi', 'b12_eksikligi']
        sans = random.choice(olasiliklar)
        return jsonify({"response": tahlil_tablosu_olustur(sans), "type": "html"})

    elif "merhaba" in user_input:
        return jsonify({"response": "Merhaba Doktor. Tahlil butonuna basarak yeni bir vaka alabilirsiniz.", "type": "text"})
    else:
        return jsonify({"response": "Anlaşılmadı. Lütfen butonu kullanın.", "type": "text"})

if __name__ == '__main__':
    app.run(debug=True)