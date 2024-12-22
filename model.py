import numpy as np


class LogisticRegression:
    def __init__(self, lr=0.01):
        self.lr = lr  # Öğrenme oranı
        self.weights = None
        self.bias = 0

    def sigmoid(self, z):
        """
        Sigmoid aktivasyon fonksiyonu.

        Parametre:
        z -- girdi değeri

        Çıkış:
        Sigmoid fonksiyonunun çıktısı
        """
        return 1 / (1 + np.exp(-z))

    def predict_proba(self, X):
        """
        Olasılık tahminleri hesaplar.

        Parametre:
        X -- Özellikler matrisi

        Çıkış:
        Olasılık tahminleri
        """
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)

    def predict(self, X,threshold=0.5):
        """
        İkili sınıflandırma tahmini (0 veya 1).

        Parametre:
        X -- Özellikler matrisi
        threshold -- Eşik değeri

        Çıkış:
        Tahmin edilen sınıflar
        """
        return [1 if i > threshold else 0 for i in self.predict_proba(X)]

    def compute_loss(self, y_true, y_pred):
        """
        Cross-entropy loss fonksiyonu.

        Parametreler:
        y_true -- Gerçek etiketler
        y_pred -- Tahmin edilen olasılıklar

        Çıkış:
        Ortalama cross-entropy loss değeri
        """
        # Olasılıkları çok küçük veya çok büyük değerlere sınırlayarak log(0) hatasını önlüyoruz
        epsilon = 1e-10
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def fit(self, X_train, y_train, X_val, y_val, epochs=1000):
        """
        Modeli stochastic gradient descent ile eğitir.

        Parametreler:
        X -- Özellikler matrisi
        y -- Hedef etiketler
        epochs -- Eğitim epoch sayısı

        Çıkış:
        Eğitim sürecindeki cross-entropy loss değerlerinin listesi
        """
        n_samples, n_features = X_train.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        train_losses = []
        val_losses = []

        # Eğitim döngüsü
        for epoch in range(epochs):
            for idx, x_i in enumerate(X_train):
                linear_model = np.dot(x_i, self.weights) + self.bias
                y_pred = self.sigmoid(linear_model)

                # Ağırlık güncelleme (SGD)
                update = (y_pred - y_train[idx])
                self.weights -= self.lr * update * x_i
                self.bias -= self.lr * update

            # Eğitim seti için tahmin ve loss hesaplama
            y_pred_train = self.predict_proba(X_train)
            train_loss = self.compute_loss(y_train, y_pred_train)
            train_losses.append(train_loss)

            # Doğrulama seti için tahmin ve loss hesaplama
            y_pred_val = self.predict_proba(X_val)
            val_loss = self.compute_loss(y_val, y_pred_val)
            val_losses.append(val_loss)


            # Her 100 epoch'ta loss değerlerini yazdırma
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Eğitim Loss: {train_loss:.4f}, Doğrulama Loss: {val_loss:.4f}")

        return train_losses, val_losses

