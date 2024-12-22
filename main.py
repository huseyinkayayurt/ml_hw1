from dataset import load_and_split_data
from visualization import plot_data,plot_loss
from model import LogisticRegression
from calculate import calculate_metrics
from tabulate import tabulate


def main():
    # Veriyi yükle ve eğitim, doğrulama, test setlerine böl
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data("hw1Data.txt")

    # Eğitim verisini görselleştir ve kaydet
    plot_data(X_train, y_train, filename="training_data_distribution.png")
    print("Eğitim verisi dağılım grafiği 'training_data_distribution.png' dosyasına kaydedildi.")

    # Modeli başlat ve eğit
    print("Model eğitimi başlıyor...")
    model = LogisticRegression(lr=0.001)
    train_losses, val_losses = model.fit(X_train, y_train, X_val, y_val, epochs=5500)
    print("Model eğitimi tamamlandı.")

    # Eğitim ve doğrulama loss grafiğini kaydet
    plot_loss(train_losses, val_losses, filename="loss_curve.png")


    # Eğitim, doğrulama ve test setleri için metrikleri hesapla
    train_metrics = calculate_metrics(model, X_train, y_train)
    val_metrics = calculate_metrics(model, X_val, y_val)
    test_metrics = calculate_metrics(model, X_test, y_test)

    # Tabloda metrikleri göster
    headers = ["Veri Seti", "Accuracy", "Precision", "Recall", "F1-Score"]
    table = [
        ["Eğitim Seti", train_metrics["Accuracy"], train_metrics["Precision"], train_metrics["Recall"], train_metrics["F1-Score"]],
        ["Doğrulama Seti", val_metrics["Accuracy"], val_metrics["Precision"], val_metrics["Recall"], val_metrics["F1-Score"]],
        ["Test Seti", test_metrics["Accuracy"], test_metrics["Precision"], test_metrics["Recall"], test_metrics["F1-Score"]],
    ]
    print("\nModel Performansı:")
    print(tabulate(table, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    main()
