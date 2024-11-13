from dataset import load_and_split_data
from visualization import plot_data,plot_loss
from model import LogisticRegression
from calculate import calculate_and_print_metrics

def main():
    # Veriyi yükle ve eğitim, doğrulama, test setlerine böl
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data("hw1Data.txt")

    # Eğitim verisini görselleştir ve kaydet
    plot_data(X_train, y_train, filename="training_data_distribution.png")
    print("Eğitim verisi dağılım grafiği 'training_data_distribution.png' dosyasına kaydedildi.")

    # Modeli başlat ve eğit
    print("Model eğitimi başlıyor...")
    model = LogisticRegression(lr=0.001)
    train_losses, val_losses = model.fit(X_train, y_train, X_val, y_val, epochs=5000)
    print("Model eğitimi tamamlandı.")

    # Eğitim ve doğrulama loss grafiğini kaydet
    plot_loss(train_losses, val_losses, filename="loss_curve.png")

    # Eğitim, doğrulama ve test setleri için metrikleri hesapla ve yazdır
    calculate_and_print_metrics(model, X_train, y_train, set_name="Eğitim Seti")
    calculate_and_print_metrics(model, X_val, y_val, set_name="Doğrulama Seti")
    calculate_and_print_metrics(model, X_test, y_test, set_name="Test Seti")


if __name__ == "__main__":
    main()
