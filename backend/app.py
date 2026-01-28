from datetime import datetime
import os
import pandas as pd
import numpy as np
import random
from flask import Flask, render_template, jsonify, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import google.generativeai as genai
from dotenv import load_dotenv
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FRONTEND_DIR = os.path.join(BASE_DIR, '..', 'Frontend')

app = Flask(__name__,
            template_folder= os.path.join(FRONTEND_DIR, 'templates'),
            static_folder= os.path.join(FRONTEND_DIR, 'static'))

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- AYARLAR ---
app.secret_key = "medsim_secret_key"
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///medsim.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# --- VERİTABANI ŞEMASI ---
class User(UserMixin, db.Model):
    #Kimlik bilgileri
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)

    # Kişisel bilgiler
    first_name = db.Column(db.String(80), nullable=False) # AD
    last_name = db.Column(db.String(80), nullable=False)  # SOYAD
    phone = db.Column(db.String(20))                     # TELEFON
    gender = db.Column(db.String(10))                   # CİNSİYET
    birth_date = db.Column(db.Date, nullable=True)        # DOĞUM TARİHİ
    avatar_url = db.Column(db.String(200), nullable=True) # PROFİL FOTOĞRAFI URL'Sİ

    # İstatistikler (Otomatk olarak sıfırdan başlar)
    total_cases = db.Column(db.Integer, default = 0)   #Görülen toplam vaka
    correct_count = db.Column(db.Integer, default = 0) #Doğru teşhis sayısı
    wrong_count = db.Column(db.Integer, default = 0) #Yanlış teşhis sayısı



# --- API KEY ---
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
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
def yeni_hasta_olustur(doktor_hitap="Bey"):
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

    return f"Merhaba Doktor {doktor_hitap} ({durum['ruh']} görünüyor)"

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
# 1. KURAL: Uygulama açılınca direkt Login'e at
@app.route('/')
def index():
    return redirect(url_for('login'))

# 2. Login Sayfası Ayarları
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username_input = request.form.get('username')
        password_input = request.form.get('password')

        user = User.query.filter_by(username=username_input).first()

        if user and check_password_hash(user.password, password_input):
           login_user(user)
           session['user_id'] = user.id
           session['user_name'] = user.first_name
           flash(f'Hoş geldin Dr. {user.last_name}!', 'success')
           return redirect(url_for('dashboard'))
        else:
            flash('Kullanıcı adı veya şifre yanlış.', 'error')

    return render_template('login.html')

# 3. Register Sayfası Ayarları
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # Formdan gelen tüm verileri alıyoruz
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        first_name = request.form.get('first_name')
        last_name = request.form.get('last_name')
        phone = request.form.get('phone')
        gender = request.form.get('gender')

        # Kontrol: Şifreler uyuşuyor mu?
        if password != confirm_password:
            flash("Girdiğiniz şifreler birbiriyle uyuşmuyor!", "error")
            return redirect(url_for('register'))
        
        # Kontrol: Bilgilerin alındığına dair 
        if User.query.filter((User.username == username)|(User.email == email)).first():
            flash("Bu kullanıcının adı veya emaili zaten kayıtlı.", "warning")
            return redirect(url_for('register'))
        
        # Şifreleme ve Kayıt
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

        new_user = User(
            username=username,
            email=email,
            password=hashed_password,
            first_name=first_name,
            last_name=last_name,
            phone=phone,
            gender=gender
            #Vaka sayılarını yazmıyoruz çünkü otomatik olarak sıfır oluyor
        )

        db.session.add(new_user)
        db.session.commit()

        flash("Doktor kaydı başarıyla oluşturuldu! Giriş yapabilirsiniz.", "success")

        # Kayıt olunca tekrar giriş ekranına yönlendir
        return redirect(url_for('login'))
    
    return render_template('register.html')
# 4. Dashboard Sayfası Ayarları
@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user = User.query.get(session['user_id'])
    hitap = ""
    if user:
        if user.gender == "Erkek":
            hitap = "Bey"
        else:
            hitap = "Hanım"

    return render_template('dashboard.html', user=user, hitap=hitap)

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':
        current_user.first_name = request.form.get('first_name')
        current_user.last_name = request.form.get('last_name')
        current_user.phone = request.form.get('phone')
        current_user.email = request.form.get('email')
        current_user.gender = request.form.get('gender')
        current_user.avatar_url = request.form.get('avatar_url')
        tarih_str = request.form.get('birth_date')
        if tarih_str:
            try:
                # String -> Date dönüşümü
                current_user.birth_date = datetime.strptime(tarih_str, '%Y-%m-%d').date()
            except ValueError:
                pass
        
        try:
            db.session.commit()
            flash('Profil bilgileriniz başarıyla güncellendi! ✅', 'success')
        except:
            db.session.rollback()
            flash('Bir hata oluştu, kaydedilemedi.', 'error')
        
        return redirect(url_for('profile'))
    
    return render_template('profile.html', user=current_user)


@app.route('/chat', methods=['POST'])
def chat():
    global current_patient
    user_input = request.json.get('message', '').lower()

    doktor_hitap = "Bey"
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        if user and user.gender == "Kadın":
            doktor_hitap = "Hanım"
        print(f"DEBUG - User ID: {session.get('user_id')}")
        print(f"DEBUG - User Gender: {user.gender if user else 'User bulunamadı'}")
        print(f"DEBUG - Doktor Hitap: {doktor_hitap}")
    else:
        print("DEBUG - Session'da user_id yok!")        
    
    # 1. SENARYO: "YENİ HASTA" BUTONUNA BASILDI
    # Veya kullanıcı açıkça "yeni hasta" yazdı
    if "yeni hasta" in user_input:
        ilk_mesaj = yeni_hasta_olustur(doktor_hitap)
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
            ilk_mesaj = yeni_hasta_olustur(doktor_hitap)
            return jsonify({"response": ilk_mesaj, "type": "text"})
            
        try:
            response = current_patient["chat_session"].send_message(user_input)
            bot_reply = response.text.replace("</blockquote>", "").replace("<blockquote>", "")
            return jsonify({"response": bot_reply, "type": "text"})
        except Exception as e:
            return jsonify({"response": "Hata: " + str(e), "type": "text"})

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)