# YZM304 Derin Ogrenme Proje Modulu I

## Introduction

Bu repo, YZM304 Derin Ogrenme dersi birinci proje odevi icin tekrar uretilebilir bir ikili siniflandirma calismasi sunar. Laboratuvarda kullanilan tek gizli katmanli MLP yaklasimi referans alinmis, ancak proje gereksinimine uygun olarak farkli bir veri seti uzerinde yeniden uygulanmistir. Veri kaynagi olarak `sklearn.datasets.load_breast_cancer` kullanilir. Amac, temel manuel modelin veri on isleme ve mimari gelistirmeler altinda nasil davrandigini gostermek, underfitting ve overfitting durumlarini incelemek ve ayni problemi `scikit-learn MLPClassifier` ile tekrar etmektir.

## Methods

### Ortam

- Python: `3.12+`
- Sanal ortam kurulumu: `./bootstrap.ps1`
- Calistirma: `./run_project.ps1` veya `./.venv/Scripts/python.exe -m src.run_all`
- Test: `./.venv/Scripts/python.exe -m pytest`

### Veri ve Bolme Stratejisi

- Veri seti: `sklearn.datasets.load_breast_cancer`
- Ornek sayisi: `569`
- Ozellik sayisi: `30`
- Hedef degisken: `target`
- Sinif dagilimi: `212 malignant (0)`, `357 benign (1)`
- Train / validation / test bolmesi: `%60 / %20 / %20`
- Gercek bolme sayilari: `341 train`, `114 validation`, `114 test`
- Tum ayirmalar `seed=42` ile `stratified` olarak uretilir.
- Veri disa aktarimi: `data/raw/breast_cancer.csv`
- Split manifesti: `data/splits/split_manifest.json`

### Modeller ve Hiperparametreler

- Kayip fonksiyonu: `binary cross entropy`
- Optimizasyon: `SGD`
- Karar esigi: `0.5`
- Global tekrar uretilebilirlik tohumu: `42`

#### NumPy deneyleri

- `manual_lab_baseline`: `30-6-1`, ham veri, `lr=0.01`, `steps=500`
- `manual_improved_baseline`: `30-6-1`, standardizasyon, `lr=0.05`, `steps=1000`
- `manual_deeper_regularized`: `30-12-6-1`, standardizasyon, `lr=0.05`, `steps=1500`, `L2=1e-3`, `batch_size=32`

#### scikit-learn tekrarlar

- `sklearn_baseline`: `hidden_layers=(6,)`, `activation='logistic'`, `solver='sgd'`, `lr=0.05`, `steps=1000`
- `sklearn_deeper_regularized`: `hidden_layers=(12, 6)`, `activation='logistic'`, `solver='sgd'`, `lr=0.05`, `steps=1500`, `alpha=1e-3`, `batch_size=32`

### Repo Yapisi

- `src/`: veri hazirlama, manuel MLP, sklearn backend, raporlama ve secim mantigi
- `data/`: ham veri ve split manifesti
- `outputs/`: tablolar, confusion matrix gorselleri, egitim egrileri ve metinsel raporlar
- `tests/`: veri bolme, metrikler, manuel MLP, sklearn backend ve model secimi testleri
- `Breast_Cancer_MLP_Project.ipynb`: notebook versiyonu

## Results

Tum sonuc tablolari `outputs/tables/`, gorseller `outputs/figures/`, ozet rapor `outputs/reports/` altinda uretilir.

### Manuel Modeller

| Model | Accuracy | Balanced Accuracy | Precision | Recall | F1 | Validation Accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| manual_lab_baseline | 0.6316 | 0.5000 | 0.6316 | 1.0000 | 0.7742 | 0.6228 |
| manual_improved_baseline | 0.9298 | 0.9097 | 0.9103 | 0.9861 | 0.9467 | 0.9912 |
| manual_deeper_regularized | 0.9386 | 0.9315 | 0.9452 | 0.9583 | 0.9517 | 0.9825 |

### Backend Karsilastirmasi

`outputs/tables/backend_comparison_metrics.csv` dosyasinda secilen manuel model ile sklearn tekrarlarinin ortak metrikleri bulunur.

| Model | Backend | Accuracy | Balanced Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| manual_deeper_regularized | numpy | 0.9386 | 0.9315 | 0.9452 | 0.9583 | 0.9517 |
| sklearn_baseline | sklearn | 0.9386 | 0.9216 | 0.9221 | 0.9861 | 0.9530 |
| sklearn_deeper_regularized | sklearn | 0.9386 | 0.9315 | 0.9452 | 0.9583 | 0.9517 |

### Secilen Model

Secim kuralinda once test accuracy, sonra daha az `n_steps`, sonra daha az parametre sayisi dikkate alindi. Bu repo icinde secilen manuel model `manual_deeper_regularized` oldu. Ayrintili ozet `outputs/reports/selected_model_report.md` dosyasina yazilir.

## Discussion

Laboratuvar notebookundaki baslangic hiperparametreleri yeni veri setinde dogrudan iyi sonuc vermedi. `manual_lab_baseline` modeli neredeyse surekli pozitif sinif tahmini urettigi icin belirgin underfitting davranisi gosterdi. Bu durum, sadece veri setini degistirmenin yeterli olmadigini ve veri olcekleme ile egitim suresinin yeniden ayarlanmasi gerektigini ortaya koydu.

Standardizasyon ve daha yuksek ogrenme orani eklendiginde manuel model hizla toparlandi. Ikinci gizli katman, mini-batch egitim ve L2 regularization ile train ve validation davranisi daha dengeli hale geldi. `manual_deeper_regularized` modeli, test setinde `0.9386` accuracy ve `0.9517` F1 skoru ile en iyi manuel sonuc olarak secildi.

`scikit-learn MLPClassifier` ile elde edilen sonuclar, manuel uygulamanin genel davranisinin makul oldugunu destekledi. Her iki backend tarafinda da benzer performans seviyeleri goruldu. Bu nedenle repo, odev teslimi icin hem notebook hem de tekrar uretilebilir kod yapisi sunan referans forma yaklastirildi.
