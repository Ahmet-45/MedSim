import os
import pandas as pd
import numpy as np
import random
from flask import Flask, render_template, jsonify, request
import google.generativeai as genai

app = Flask(__name__,
            template_folder='Frontend/templates',
            static_folder='Frontend/static')

# --- API KEY ---
API_KEY = "AIzaSyCvQFhW10eSpTEOb-b0dP7f5-nF4S6U_DI"
genai.configure(api_key=API_KEY)

# Modeli Hazırla
try:
    model = genai.GenerativeModel('gemini-flash-latest')
    print(" Gemini AI Hazır!")
except:
    print(" Gemini Bağlantı Hatası!")

# --- GLOBAL DEĞİŞKENLER (HAFIZA) ---
# Hastanın bilgilerini burada tutacağız ki tahlil isteyince unutmasın
current_patient = {
    "chat_session": None,
    "hastalik": None,
    "yas": None,
    "cinsiyet": None,
    "isim": None # Gemini kendi uyduruyor ama biz hastalığı tutsak yeter
}

# --- MODÜLLERİ YÜKLE ---
# (Senin mevcut istatistik ve GAN kodların aynen kalıyor)
try:
    from Tahlil.src.hemogram_ai import HemogramGenerator
    hemogram_stat_model = HemogramGenerator()
    csv_path = os.path.join('Tahlil', 'data', 'MedSim_100_Hasta.csv')
    if os.path.exists(csv_path): hemogram_stat_model.fit(csv_path)
except: hemogram_stat_model = None

gan_model = None
try:
    from sdv.single_table import CTGANSynthesizer
    pkl_path = os.path.join('Tahlil', 'hemogram_model_IDA.pkl')
    if os.path.exists(pkl_path): gan_model = CTGANSynthesizer.load(pkl_path)
except: pass

# --- YARDIMCI FONKSİYON: GEMINI HASTA MODUNU BAŞLAT ---
def yeni_hasta_olustur():
    global current_patient
    
    # Rastgele seçimler
    olasiliklar = ['saglikli', 'demir_eksikligi', 'b12_eksikligi']
    secilen_hastalik = random.choice(olasiliklar)
    yas = random.randint(18, 75)
    cinsiyet = random.choice(["Kadın", "Erkek"])
    
    # Hafızaya kaydet (ÇOK ÖNEMLİ)
    current_patient["hastalik"] = secilen_hastalik
    current_patient["yas"] = yas
    current_patient["cinsiyet"] = cinsiyet
    
    # Senaryolar
    senaryolar = {
        'saglikli': {'ad': 'Sağlıklı', 'semptom': 'Hafif yorgunluk var ama turp gibiyim, kontrol amaçlı geldim.', 'ruh': 'Rahat'},
        'demir_eksikligi': {'ad': 'Demir Eksikliği Anemisi', 'semptom': 'Kolumu kaldıracak halim yok, saçım dökülüyor, sürekli uyku hali.', 'ruh': 'Bezgin, yorgun'},
        'b12_eksikligi': {'ad': 'B12 Eksikliği', 'semptom': 'Unutkanlık, ellerde uyuşma, dengesizlik.', 'ruh': 'Endişeli'}
    }
    durum = senaryolar[secilen_hastalik]

    system_instruction = f"""
    SEN BİR SİMÜLASYON HASTASISIN.
    KİMLİK: {yas} yaşında, {cinsiyet}.
    GİZLİ HASTALIK: {durum['ad']} (Bunu söyleme).
    ŞİKAYET: {durum['semptom']}
    RUH HALİ: {durum['ruh']}
    
    KURALLAR:
    1. Asla "Ben yapay zekayım" deme.
    2. Doktor tahlil isterse "Tamam hocam vereyim kanı" de.
    3. Tahlil sonuçları çıkınca "Sonuçlar nasıl hocam, kötü bir şey var mı?" diye sor.
    4. Kısa, doğal ve halk ağzıyla konuş.
    """
    
    # Yeni oturum başlat
    current_patient["chat_session"] = model.start_chat(history=[
        {"role": "user", "parts": [system_instruction]}
    ])
    
    return f"Merhaba doktor bey/hanım. Sıram geldi mi? ({durum['ruh']} görünüyor)"

# --- TABLO OLUŞTURUCU (Değişmedi) ---
def tahlil_tablosu_olustur(durum):
    if not hemogram_stat_model: return "Model Yüklenemedi."
    
    # Veri üretimi (Senin yazdığın mantık)
    df_stat = hemogram_stat_model.generate(adet=1, durum=durum)
    if df_stat.empty: return "Veri yok."
    hasta_data = df_stat.iloc[0].to_dict()
    if 'Teşhis' in hasta_data: del hasta_data['Teşhis']

    # GAN Entegrasyonu
    if gan_model:
        try:
            gan_data = gan_model.sample(num_rows=1)
            hgb = float(gan_data['Hemoglobin'].values[0])
            mcv = float(gan_data['MCV'].values[0])
            
            # Manipülasyon
            if durum == 'demir_eksikligi':
                if hgb > 11.5: hgb *= 0.75
                if mcv > 78: mcv *= 0.80
            elif durum == 'b12_eksikligi':
                if mcv < 100: mcv *= 1.25
                
            hasta_data['HGB'] = hgb
            hasta_data['MCV'] = mcv
        except: pass

    # HTML Tablo
    html = "<div class='table-responsive'><table class='table table-sm table-bordered'>"
    html += "<thead class='thead-dark'><tr><th>TETKİK</th><th>SONUÇ</th><th>DURUM</th></tr></thead><tbody>"
    
    for key, val in hasta_data.items():
        if key not in hemogram_stat_model.stats: continue
        mu = hemogram_stat_model.stats[key]['mean']
        sigma = hemogram_stat_model.stats[key]['std']
        
        style = ""
        durum_txt = "Normal"
        if val < (mu - 2*sigma): 
            style = "color:red; font-weight:bold;"
            durum_txt = "Düşük (L)"
        elif val > (mu + 2*sigma):
            style = "color:red; font-weight:bold;"
            durum_txt = "Yüksek (H)"
            
        html += f"<tr><td>{key}</td><td style='{style}'>{val:.2f}</td><td>{durum_txt}</td></tr>"
    
    html += "</tbody></table></div>"
    return html

# --- ROTALAR ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    global current_patient
    user_input = request.json.get('message', '').lower()
    
    # 1. SENARYO: "YENİ HASTA" BUTONUNA BASILDI
    # Veya kullanıcı açıkça "yeni hasta" yazdı
    if "yeni hasta" in user_input:
        ilk_mesaj = yeni_hasta_olustur()
        return jsonify({"response": ilk_mesaj, "type": "text"})

    # 2. SENARYO: DOKTOR TAHLİL İSTEDİ (Ama hasta değişmeyecek!)
    elif "tahlil" in user_input or "hemogram" in user_input or "kan ver" in user_input:
        
        # Eğer ortada hasta yoksa uyar
        if current_patient["chat_session"] is None:
            return jsonify({"response": "Önce yeni bir hasta çağırmalısınız.", "type": "text"})
        
        # Mevcut hastanın hastalığı neyse ona göre tablo üret
        mevcut_hastalik = current_patient["hastalik"]
        tablo_html = tahlil_tablosu_olustur(mevcut_hastalik)
        
        # Yapay zekaya da haber verelim ki tepki versin
        try:
            response = current_patient["chat_session"].send_message(
                "Doktor kan tahlili istedi. Sonuçları sisteme girdim. Şimdi endişeli bir şekilde 'Sonuçlar nasıl doktor bey?' diye sor."
            )
            bot_reply = response.text.replace("\n", " ")
        except:
            bot_reply = "Sonuçlar çıktı hocam, buyurun."

        return jsonify({
            "response": bot_reply, 
            "type": "html", 
            "table_html": tablo_html
        })

    # 3. SENARYO: NORMAL SOHBET
    else:
        if current_patient["chat_session"] is None:
             # Hasta yoksa otomatik oluştur
            ilk_mesaj = yeni_hasta_olustur()
            return jsonify({"response": ilk_mesaj, "type": "text"})
            
        try:
            response = current_patient["chat_session"].send_message(user_input)
            bot_reply = response.text.replace("</blockquote>", "").replace("<blockquote>", "")
            return jsonify({"response": bot_reply, "type": "text"})
        except Exception as e:
            return jsonify({"response": "Hata: " + str(e), "type": "text"})

if __name__ == '__main__':
    app.run(debug=True)