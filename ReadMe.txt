## Makine Öğrenmesi Projesi: Lojistik Regresyon ile İkili Sınıflandırma

### Proje Açıklaması
Bu proje, lojistik regresyon yöntemi kullanılarak ikili sınıflandırma yapmayı amaçlamaktadır. Eğitim, doğrulama ve test verileri kullanılarak bir kişinin sınav sonuçlarına göre işe kabul edilip edilmeyeceği tahmin edilmektedir. Model, eğitim sürecinde `CrossEntropyLoss` kullanılarak optimize edilmiş ve aşırı öğrenme olup olmadığını gözlemlemek amacıyla eğitim ve doğrulama loss değerleri grafik üzerinde karşılaştırılmıştır.

### Gereksinimler
Projeyi çalıştırabilmek için aşağıdaki kütüphanelerin yüklü olması gerekmektedir:
- `numpy`
- `matplotlib`
- `sklearn`

Gerekli kütüphaneleri yüklemek için:
```bash
pip install -r requirements.txt

### Proje Dosyaları
main.py: Projenin ana dosyasıdır. Eğitim, doğrulama ve test süreçlerini yürütür ve metrikleri hesaplar.
dataset.py: Veri setini yükleyen ve eğitim, doğrulama ve test setlerine bölen yardımcı dosyadır.
model.py: Lojistik regresyon modelini tanımlayan dosyadır. Modelin eğitimi, tahmin ve loss hesaplama fonksiyonları burada bulunur.
metrics.py: accuracy, precision, recall, ve f1_score metriklerini hesaplayan fonksiyonları içerir.
visualization.py: Eğitim verisi dağılımını ve loss grafiğini çizen fonksiyonları içerir.
requirements.txt: Proje için gerekli kütüphaneleri listeler.
calculate.py: eğitim, test ve doğrulama setleri için metrik hesaplamalarının yapıldığı ve konsola yazdırıldığı fonksiyonları içerir.
README.txt: Projenin genel bilgilerini ve kullanım talimatlarını içeren açıklama dosyasıdır.

### Proje Yapısı
proje_klasoru/
├── main.py
├── dataset.py
├── model.py
├── metrics.py
├── visualization.py
├── requirements.txt
├── README.txt
└── dataset.txt

### Proje Çalıştırma Komnutu
```bash
python main.py

### Örnek Çıktılar
training_data_distribution.png: Eğitim verisinin sınav sonuçlarına göre dağılım grafiğini içerir.
loss_curve.png: Eğitim ve doğrulama loss değerlerini gösteren grafiktir. Aşırı öğrenme olup olmadığını gözlemlemek için kullanılır.