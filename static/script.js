// --- 1. HTML ELEMANLARINI SEÇME ---
// Sayfadaki kutuları ve butonları, kimlik (ID) isimleriyle yakalayıp değişkenlere atıyoruz.
const mesajInput = document.getElementById('mesaj-input'); // Yazı yazdığımız kutu
const gonderBtn = document.getElementById('gonder-btn');   // Gönder butonu 
const sohbetKutusu = document.getElementById('sohbet-kutusu'); // Mesajların ve tablonun düştüğü orta alan

// --- 2. HAZIR MESAJ FONKSİYONU ---
// Sol menüdeki "Yeni Vaka İste" butonuna basınca bu çalışır.
function hazirMesajGonder(mesaj) {
    mesajInput.value = mesaj; // Kutunun içine "Yeni Hasta Tahlili Getir" yazar.
    mesajGonder();            // Ve hemen gönderme fonksiyonunu tetikler.
}

// --- 3. EKRANA MESAJ EKLEME FONKSİYONU ---
function mesajEkle(icerik, kimden, tip = 'text') {
    // Yeni bir kutu (div) yaratıyoruz
    const mesajDiv = document.createElement('div');
    
    // Bu kutuya stil ekliyoruz (Mavi mi gri mi olacağına CSS karar versin diye)
    mesajDiv.classList.add('mesaj', kimden);

    // EĞER gelen cevap bir Tablo ise (HTML türündeyse)
    if (tip === 'html') {
        mesajDiv.innerHTML = icerik; // HTML kodlarını (<table>...</table>) yorumlayarak ekle.
    } else {
        // EĞER düz yazıysa 
        mesajDiv.textContent = icerik; // Düz metin olarak ekle.
    }

    // Hazırladığımız bu kutuyu, ana sohbet alanının içine monte et (sona ekle).
    sohbetKutusu.appendChild(mesajDiv);

    // Sohbet kutusunu otomatik olarak en aşağı kaydır (Scroll Down)
    // Böylece hep en son gelen mesajı görürüz.
    setTimeout(() => {
        sohbetKutusu.scrollTop = sohbetKutusu.scrollHeight;
    }, 50);
}

// --- 4. ANA GÖNDERME FONKSİYONU (BEYİN) ---
function mesajGonder() {
    // Yazı kutusundaki metni al, sağındaki solundaki boşlukları temizle (trim)
    const metin = mesajInput.value.trim();
    
    // Eğer kutu boşsa hiçbir şey yapma, dur.
    if (metin === "") return;
    
    // 1. ADIM: Önce kullanıcının mesajını ekrana bas (Mavi balon)
    mesajEkle(metin, 'kullanici');
    
    // Kutunun içini temizle ki yeni mesaj yazılabilsin
    mesajInput.value = '';

    // 2. ADIM: Python (Backend) ile bağlantı (API İsteği)
    fetch('http://127.0.0.1:5000/chat', {
        method: 'POST', //  (Veri yolluyoruz)
        headers: { 'Content-Type': 'application/json' }, //  JSON formatında diyoruz
        body: JSON.stringify({ message: metin }) // Mesajı geri  yolluyoruz
    })
    .then(response => response.json()) // Python'dan cevap geldi, paketi aç (JSON'a çevir)
    .then(data => { 
        // Python'dan gelen cevabı (data.response) ekrana 'doktor' adına bas.
        // data.type: Gelen verinin Tablo mu yoksa Yazı mı olduğunu söyler.
        mesajEkle(data.response, 'doktor', data.type); 
    })
    .catch(error => { console.error('Hata:', error); }); // Bir sorun olursa (Sunucu kapalıysa) konsola hata yaz.
}

// --- 5. OLAY DİNLEYİCİLER  ---
// Gönder butonuna tıklandığında mesajGonder'i çalıştır.
gonderBtn.addEventListener('click', mesajGonder);

// Yazı kutusundayken "Enter" tuşuna basılırsa da mesajGonder'i çalıştır.
mesajInput.addEventListener('keypress', function (e) {
    if (e.key === 'Enter') mesajGonder();
});