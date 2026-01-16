import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import os

try:
    from src.hemogram_ai import HemogramGenerator
except ImportError:
    print(f"HATA: 'src' klasörü bulunamadı. Lütfen 'Tahlil' dizininde çalıştırdığınızdan emin olun.")

class HemogramApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hemogram Analiz")
        self.root.geometry("1000x700")

        #stil ayarları
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview.Heading", font=("Arial", 10, "bold"))
        style.configure("Treeview", font=("Arial", 10), rowheight=25)

        # HemogramGenerator modelini başlat
        self.hemogram_model = HemogramGenerator()

        yol_secenekleri = [
            os.path.join('data', 'MedSim_100_Hasta.csv'),
            'MedSim_100_Hasta.csv',
            'data/MedSim_100_Hasta.csv'
        ]

        veri_yolu = None
        for yol in yol_secenekleri:
            if os.path.exists(yol):
                veri_yolu = yol
                break

        if veri_yolu:
            self.hemogram_model.fit(veri_yolu)
        else:
            messagebox.showerror("Hata", "MedSim_100_Hasta.csv dosyası bulunamadı!\nLütfen 'data' klasörüne koyun.")          


        # ----ARAYÜZ TASARIMI----    

        # 1. Üst Panel (Butonlar)
        ust_panel = tk.Frame(root, bg="#f0f0f0", pady=15)
        ust_panel.pack(fill="x")

        tk.Label(ust_panel, text="Hasta Profili Seç:", bg="#f0f0f0", font=("Arial", 11)).pack(side="left", padx=10)

        # Butonlar
        tk.Button(ust_panel, text="🟢 Sağlıklı", command=lambda: self.veri_uret('saglikli'), 
                  bg="#28a745", fg="white", font=("Arial", 10, "bold"), width=15).pack(side="left", padx=5)
        
        tk.Button(ust_panel, text="🔴 Demir Eksikliği", command=lambda: self.veri_uret('demir_eksikligi'), 
                  bg="#dc3545", fg="white", font=("Arial", 10, "bold"), width=15).pack(side="left", padx=5)
        
        tk.Button(ust_panel, text="🟣 B12 Eksikliği", command=lambda: self.veri_uret('b12_eksikligi'), 
                  bg="#6f42c1", fg="white", font=("Arial", 10, "bold"), width=15).pack(side="left", padx=5)

        # 2. Tablo Alanı
        tablo_cerceve = tk.Frame(root, padx=20, pady=20)
        tablo_cerceve.pack(fill="both", expand=True)

        columns = ("tahlil", "sonuc", "birim", "referans", "durum")
        self.tree = ttk.Treeview(tablo_cerceve, columns=columns, show="headings", height=20)

        # Sütun Başlıkları
        self.tree.heading("tahlil", text="Tetkik Adı")
        self.tree.heading("sonuc", text="Sonuç")
        self.tree.heading("birim", text="Birim")
        self.tree.heading("referans", text="Referans Değeri")
        self.tree.heading("durum", text="Durum")

        # Sütun Genişlikleri
        self.tree.column("tahlil", width=150)
        self.tree.column("sonuc", width=100, anchor="center")
        self.tree.column("birim", width=100, anchor="center")
        self.tree.column("referans", width=150, anchor="center")
        self.tree.column("durum", width=100, anchor="center")

        # Scrollbar
        scrollbar = ttk.Scrollbar(tablo_cerceve, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        
        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Renk Etiketleri (Tag)
        self.tree.tag_configure('normal', background='white')
        self.tree.tag_configure('dusuk', background='#ffcccc', foreground='red') # Kırmızı
        self.tree.tag_configure('yuksek', background='#ffcccc', foreground='red') # Kırmızı

        # Alt Bilgi
        self.lbl_bilgi = tk.Label(root, text="Analiz için yukarıdan bir butona basınız.", font=("Arial", 10, "italic"), fg="gray")
        self.lbl_bilgi.pack(pady=10)

    def veri_uret(self, durum_kodu):
        #Tabloyu temizle
        for i in self.tree.get_children():
            self.tree.delete(i)

        # Veri üret
        df = self.hemogram_model.generate(adet=1, durum=durum_kodu)
        if df.empty:
            return

        hasta_data = df.iloc[0].to_dict()
        teshis = hasta_data.pop('Teşhis')

        #Baslıgı güncelle
        renk = "green" if durum_kodu == 'saglikli' else "red"
        self.lbl_bilgi.config(text=f"Teşhis Sonucu: {teshis}", fg=renk, font=("Arial", 12, "bold"))


        for key, val in hasta_data.items():
            if key not in self.hemogram_model.stats:
                continue

            mu = self.hemogram_model.stats[key]['mean']
            sigma = self.hemogram_model.stats[key]['std']

            ref_alt = max(0, mu - 2 * sigma)
            ref_ust = mu + 2 * sigma

            durum_text = "Normal"
            tag = "normal"            

            if val < ref_alt:
                durum_text = "Düşük (L)"
                tag = "dusuk"
            elif val > ref_ust:
                durum_text = "Yüksek (H)"
                tag = "yuksek"


            birim = ""
            if key in ["HGB", "MCHC"]: birim = "g/dL"
            elif key in ["MCV", "MPV", "RDW-SD"]: birim = "fL"
            elif key in ["BAS#","EOS#","LYM#","MON#","NEU#","WBC","PLT"]: birim = "10^3/uL"
            elif key in ["RBC"]: birim = "10^6/uL"
            elif key in ["BAS%","EOS%","HCT", "LYM%","MON%","NEU%","PCT","PDW","RDW-CV"]: birim = "%" 
            elif key in ["MCH"]: birim = "pg"   

            self.tree.insert("", "end", values=(
                key, 
                f"{val:.2f}", 
                birim, 
                f"{ref_alt:.2f} - {ref_ust:.2f}",
                durum_text
            ), tags=(tag,))
if __name__ == "__main__":
    root = tk.Tk()
    app = HemogramApp(root)
    root.mainloop()            