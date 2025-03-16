import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

# 📌 1. Veri Ön İşleme Fonksiyonu
def preprocess_data(file_path, is_train=True):
    df = pd.read_csv(file_path)
    print(f"✅ Veri yüklendi: {df.shape}")

    # 📌 Eksik değerleri doldur
    df.fillna(df.median(numeric_only=True), inplace=True)
    df.fillna(df.mode().iloc[0], inplace=True)
    print("✅ Eksik değerler dolduruldu!")

    # 📌 Kategorik değişkenleri encode et
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    print("✅ Kategorik değişkenler encode edildi!")

    # 📌 Train seti için hedef değişkeni ayır
    if is_train:
        y = df["rainfall"]
        X = df.drop(columns=["rainfall", "id"])
        return X, y
    else:
        test_ids = df["id"]
        X = df.drop(columns=["id"])
        return X, test_ids

# 📌 2. Modeli Eğitme Fonksiyonu (Random Forest)
def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=500, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    print("✅ Random Forest Modeli Eğitildi!")
    return model

# 📌 3. Tahmin Yapma ve Sonucu Kaydetme
def predict_and_save(model, test_file, output_file="submission.csv"):
    X_test, test_ids = preprocess_data(test_file, is_train=False)
    y_pred = model.predict_proba(X_test)[:, 1]  # Yağmur olma olasılığını al
    submission = pd.DataFrame({"id": test_ids, "rainfall": y_pred})
    submission.to_csv(output_file, index=False)
    print(f"✅ Tahminler {output_file} dosyasına kaydedildi!")

# 📌 Ana Akış
if __name__ == "__main__":
    train_file = "train.csv"  # Eğitim verisi dosya yolu
    test_file = "test.csv"  # Test verisi dosya yolu

    # 📌 Veriyi yükle ve işle
    X_train, y_train = preprocess_data(train_file, is_train=True)

    # 📌 Modeli eğit
    rf_model = train_random_forest(X_train, y_train)

    # 📌 Test verisi üzerinde tahmin yap ve sonucu kaydet
    predict_and_save(rf_model, test_file)
