import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# 1. Veriyi CSV dosyasından okuma
# 'LifeExpectancy.csv' dosyasının doğru yolunu yazdığınızdan emin olun.
df = pd.read_csv('LifeExpectancy.csv')

# 2. Sütun isimlerini kontrol etme ve temizleme
df.columns = df.columns.str.strip()  # Başlıklardaki boşlukları temizler

# 3. Eğitim ve test setlerine ayırma
# Örneğin 2003, 2008, 2013 verilerini test seti olarak alabilirsiniz.
train_set = df[~df['Year'].isin([2003, 2008, 2013])]
test_set = df[df['Year'].isin([2003, 2008, 2013])]

# Eğitim ve test setlerinin boyutlarını yazdırma
print(f'Eğitim seti: {train_set.shape[0]} kayıt')
print(f'Test seti: {test_set.shape[0]} kayıt')

# 4. Verilerin temel istatistiklerini ve histogramını görselleştirme
# a. Yaşam süresi histogramı
plt.hist(df['LifeExpectancy'], bins=20, color='blue', edgecolor='black')
plt.title('Life Expectancy Distribution')
plt.xlabel('Life Expectancy')
plt.ylabel('Frequency')
plt.show()

# b. Yaşam süresi istatistiksel bilgileri
print(df['LifeExpectancy'].describe())

# c. En yüksek yaşam beklentisine sahip üç ülke
top_3_countries = df[['Country', 'LifeExpectancy']].sort_values(by='LifeExpectancy', ascending=False).head(3)
print("En yüksek yaşam beklentisine sahip üç ülke:")
print(top_3_countries)

# 5. Regresyon modelleri oluşturma ve eğitme
# a. GDP, b. Toplam harcama, c. Alkol kullanımı ile modelleri oluşturacağız.

# Model için bağımsız değişkenler (features) seçme
features = ['GDP', 'TotalExpenditure', 'Alcohol']
print(train_set[features].isnull().sum())
print(test_set[features].isnull().sum())

# Eksik verileri silme (alternatif olarak doldurma da yapılabilir)
train_set = train_set.dropna(subset=features + ['LifeExpectancy'])
test_set = test_set.dropna(subset=features + ['LifeExpectancy'])

# Verileri ayır
X_train = train_set[features]
y_train = train_set['LifeExpectancy']
X_test = test_set[features]
y_test = test_set['LifeExpectancy']

# Standartlaştırma
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Linear Regression modelini eğitme
model_gdp = LinearRegression().fit(X_train_scaled[:, 0].reshape(-1, 1), y_train)  # GDP ile model
model_expenditure = LinearRegression().fit(X_train_scaled[:, 1].reshape(-1, 1), y_train)  # TotalExpenditure ile model
model_alcohol = LinearRegression().fit(X_train_scaled[:, 2].reshape(-1, 1), y_train)  # Alcohol ile model

# 6. Modellerin katsayılarını ve R^2 skorlarını yazdırma
print("GDP Model Katsayıları:", model_gdp.coef_)
print("Total Expenditure Model Katsayıları:", model_expenditure.coef_)
print("Alcohol Model Katsayıları:", model_alcohol.coef_)

print("GDP Model R^2 Skoru:", model_gdp.score(X_train_scaled[:, 0].reshape(-1, 1), y_train))
print("Total Expenditure Model R^2 Skoru:", model_expenditure.score(X_train_scaled[:, 1].reshape(-1, 1), y_train))
print("Alcohol Model R^2 Skoru:", model_alcohol.score(X_train_scaled[:, 2].reshape(-1, 1), y_train))

# 7. Eğitim seti ile tahminler ve hata hesaplama
y_pred_gdp = model_gdp.predict(X_test_scaled[:, 0].reshape(-1, 1))
y_pred_expenditure = model_expenditure.predict(X_test_scaled[:, 1].reshape(-1, 1))
y_pred_alcohol = model_alcohol.predict(X_test_scaled[:, 2].reshape(-1, 1))

# Hata hesaplama (Ortalama Hata ve Standart Sapma)
mae_gdp = mean_absolute_error(y_test, y_pred_gdp)
mae_expenditure = mean_absolute_error(y_test, y_pred_expenditure)
mae_alcohol = mean_absolute_error(y_test, y_pred_alcohol)

print(f"GDP Modeli Ortalama Hatası: {mae_gdp}")
print(f"Total Expenditure Modeli Ortalama Hatası: {mae_expenditure}")
print(f"Alcohol Modeli Ortalama Hatası: {mae_alcohol}")

# 8. İleri düzey model: Çoklu lineer regresyon (4 parametre ile)
# Daha iyi tahmin için 4 parametre seçimi yapılabilir
features_4 = ['GDP', 'TotalExpenditure', 'Alcohol', 'Income']

X_train_4 = train_set[features_4]
y_train_4 = train_set['LifeExpectancy']
X_test_4 = test_set[features_4]
y_test_4 = test_set['LifeExpectancy']

X_train_scaled_4 = scaler.fit_transform(X_train_4)
X_test_scaled_4 = scaler.transform(X_test_4)

# Çoklu lineer regresyon modelini oluşturma
multi_model = LinearRegression().fit(X_train_scaled_4, y_train_4)

# Modelin katsayıları ve R^2 skoru
print("Çoklu Regresyon Model Katsayıları:", multi_model.coef_)
print("Çoklu Regresyon Model R^2 Skoru:", multi_model.score(X_train_scaled_4, y_train_4))

# Test verisi üzerinde tahmin ve hata hesaplama
y_pred_multi = multi_model.predict(X_test_scaled_4)
mae_multi = mean_absolute_error(y_test_4, y_pred_multi)

print(f"Çoklu Regresyon Modeli Ortalama Hatası: {mae_multi}")
