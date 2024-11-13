from metrics import accuracy, precision, recall, f1_score

def calculate_and_print_metrics(model, X, y, set_name=""):
    """
    Verilen veri seti için metrikleri hesaplar ve çıktısını verir.

    Parametreler:
    model -- Lojistik regresyon model nesnesi
    X -- Özellikler (bağımsız değişkenler)
    y -- Etiketler (bağımlı değişken)
    set_name -- Veri setinin adı (örn. "Eğitim Seti", "Doğrulama Seti", "Test Seti")
    """
    y_pred = model.predict(X)
    set_accuracy = accuracy(y, y_pred)
    set_precision = precision(y, y_pred)
    set_recall = recall(y, y_pred)
    set_f1_score = f1_score(y, y_pred)

    print(f"\n{set_name} Sonuçları:")
    print(f"Accuracy: {set_accuracy:.4f}")
    print(f"Precision: {set_precision:.4f}")
    print(f"Recall: {set_recall:.4f}")
    print(f"F1-Score: {set_f1_score:.4f}")

