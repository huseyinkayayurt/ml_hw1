from metrics import accuracy, precision, recall, f1_score

def calculate_metrics(model, X, y):
    """
    Verilen veri seti için metrikleri hesaplar ve çıktısını verir.

    Parametreler:
    model -- Lojistik regresyon model nesnesi
    X -- Özellikler (bağımsız değişkenler)
    y -- Etiketler (bağımlı değişken)
    """
    y_pred = model.predict(X)

    result = {
        "Accuracy": accuracy(y, y_pred),
        "Precision": precision(y, y_pred),
        "Recall": recall(y, y_pred),
        "F1-Score": f1_score(y, y_pred)
    }
    return result

