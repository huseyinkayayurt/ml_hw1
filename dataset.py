import numpy as np
from sklearn.model_selection import train_test_split

def load_and_split_data(filename):
    """
    Veriyi yükler ve eğitim, doğrulama, test setlerine böler.

    Parametreler:
    filename -- veri dosyasının adı (string)

    Çıkış:
    X_train, X_val, X_test, y_train, y_val, y_test -- verinin bölümleri
    """
    # Veriyi yükleme
    data = np.loadtxt(filename, delimiter=",")

    X = data[:, :2]  # İlk iki sütun sınav notları (1. ve 2. sınav)
    y = data[:, 2].astype(int)  # Üçüncü sütun kabul durumu (etiket)

    # Veriyi eğitim, doğrulama ve test setlerine ayırma (%60 eğitim, %20 doğrulama, %20 test)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test
