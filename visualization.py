import matplotlib.pyplot as plt

def plot_data(X, y,filename=None):
    """
    Veriyi görselleştirir.

    Parametreler:
    X -- Özellikler (sınav notları)
    y -- Hedef değişken (kabul durumu)
    """
    # Sınıf 0 olan örnekleri mavi, sınıf 1 olan örnekleri yeşil ile göster
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='blue', label="Reddedilen (0)")
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='green', label="Kabul edilen (1)")

    # Grafiğin başlık ve etiketlerini ayarla
    plt.xlabel("1. Sınav Notu")
    plt.ylabel("2. Sınav Notu")
    plt.legend()
    plt.title("Veri Seti Sınav Notlarına Göre Dağılım")
    # Eğer filename verilmişse grafiği kaydet
    if filename:
        plt.savefig(filename)
        print(f"Grafik {filename} olarak kaydedildi.")

    plt.close()  # Grafik gösterimi olmadan kapatılır


def plot_loss(losses, filename=None):
    """
    Eğitim sürecinde kayıp değerlerinin grafiğini çizer ve isteğe bağlı olarak kaydeder.

    Parametreler:
    losses -- Kayıp değerleri listesi
    filename -- Kaydedilecek dosyanın adı (None ise kaydetmez)
    """
    plt.plot(range(len(losses)), losses, label="Ortalama Cross-Entropy Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Eğitim Sürecinde Ortalama Cross-Entropy Loss Değişimi")
    plt.legend()

    if filename:
        plt.savefig(filename)
        print(f"Kayıp grafiği {filename} olarak kaydedildi.")

    plt.close()  # Grafik gösterimi olmadan kapatılır