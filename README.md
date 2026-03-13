# YZM304 Derin Ogrenme Proje Odev 1

## Introduction

Bu calisma, YZM304 Derin Ogrenme dersi birinci proje odevi icin hazirlandi. Laboratuvarda kullanilan tek gizli katmanli MLP yapisi referans alinmis, ancak proje gereksinimine uygun olarak farkli bir veri seti uzerinde yeniden uygulanmistir. Yeni veri seti olarak `sklearn.datasets.load_breast_cancer` secildi. Bu veri seti ikili siniflandirma yapisina sahiptir ve sayisal ozelliklerden olustugu icin laboratuvardaki notebook akisini bozmadan tekrar uretilebilir deneyler kurmaya uygundur.

Calismanin amaci, temel manuel MLP modelinin farkli veri on isleme ve mimari tercihleri altinda nasil davrandigini gostermek, underfitting ve overfitting durumlarini incelemek ve ayni probleme `scikit-learn MLPClassifier` ile karsilik vermektir.

## Methods

### Dataset

- Veri seti: Breast Cancer Wisconsin Diagnostic Dataset
- Ornek sayisi: 569
- Ozellik sayisi: 30
- Hedef degisken: ikili sinif etiketi
- Veri karistirma: `sample(frac=1.0, random_state=42)`

### Veri Bolme Stratejisi

- Train set: %64
- Validation set: %16
- Test set: %20
- Bolme yontemi: `train_test_split(..., stratify=y, random_state=42)`

### Uygulanan Modeller

1. `manual_lab_baseline`
   Laboratuvar notebookundaki akisa en yakin modeldir. Tek gizli katman, sigmoid aktivasyon, binary cross entropy ve tam batch egitim kullanir.
2. `manual_improved_baseline`
   Ayni manuel modelin standardization uygulanmis ve daha uzun egitilmis surumudur.
3. `manual_deeper_regularized`
   Iki gizli katmanli manuel modeldir. Mini-batch egitim ve L2 regularization eklenmistir.
4. `sklearn_baseline`
   `MLPClassifier` ile tek gizli katmanli karsilik modeldir.
5. `sklearn_deeper_regularized`
   `MLPClassifier` ile daha derin ve regularized karsilik modeldir.

### Mimari ve Hiperparametreler

#### Ortak Ayarlar

- Rastgelelik tohumu: `42`
- Cikis aktivasyonu: `sigmoid`
- Loss function: `binary cross entropy`
- Optimizasyon mantigi: `SGD`
- Siniflandirma esigi: `0.5`

#### Manuel Model Hiperparametreleri

| Model | Standardization | Hidden layers | Learning rate | Steps | L2 | Batch size |
| --- | --- | --- | --- | --- | --- | --- |
| manual_lab_baseline | Hayir | (6,) | 0.01 | 500 | 0.0 | Full batch |
| manual_improved_baseline | Evet | (6,) | 0.05 | 1000 | 0.0 | Full batch |
| manual_deeper_regularized | Evet | (12, 6) | 0.05 | 1500 | 0.001 | 32 |

#### Scikit-learn Hiperparametreleri

- Aktivasyon: `logistic`
- Solver: `sgd`
- Momentum: `0.0`
- `shuffle=False`
- `tol=0.0`
- `n_iter_no_change = max_iter + 1`

### Kod Yapisi

- [run_experiments.py](C:\Users\Seyit Kaan\Desktop\dl_odev\run_experiments.py): deneyleri tek komutla calistirir.
- [src/data.py](C:\Users\Seyit Kaan\Desktop\dl_odev\src\data.py): veri yukleme, karistirma, train/validation/test bolme ve standardization.
- [src/models/manual_mlp.py](C:\Users\Seyit Kaan\Desktop\dl_odev\src\models\manual_mlp.py): manuel MLP sinifi.
- [src/models/sklearn_mlp.py](C:\Users\Seyit Kaan\Desktop\dl_odev\src\models\sklearn_mlp.py): `MLPClassifier` deneyi.
- [Breast_Cancer_MLP_Project.ipynb](C:\Users\Seyit Kaan\Desktop\dl_odev\Breast_Cancer_MLP_Project.ipynb): notebook versiyonu.

### Tekrar Uretme

```bash
python run_experiments.py
python scripts/generate_notebook.py
```

Sonuclar ve gorseller `outputs/` klasorune yazilir.

## Results

### Manuel Modeller

| Model | Accuracy | Precision | Recall | F1 | Validation Accuracy |
| --- | --- | --- | --- | --- | --- |
| manual_lab_baseline | 0.6316 | 0.6316 | 1.0000 | 0.7742 | 0.6228 |
| manual_improved_baseline | 0.9298 | 0.9103 | 0.9861 | 0.9467 | 0.9912 |
| manual_deeper_regularized | 0.9386 | 0.9452 | 0.9583 | 0.9517 | 0.9825 |

### Scikit-learn Modelleri

| Model | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| sklearn_baseline | 0.9386 | 0.9221 | 0.9861 | 0.9530 |
| sklearn_deeper_regularized | 0.9386 | 0.9452 | 0.9583 | 0.9517 |

### Secilen Model

Secim kuralinda once yuksek dogruluk, sonra daha uygun egitim davranisi dikkate alindi. Manuel modeller icinde `manual_deeper_regularized` modeli secildi. Bu model test setinde `0.9386` accuracy, `0.9452` precision, `0.9583` recall ve `0.9517` F1 skoru uretti. Ilgili confusion matrix ve egitim egrileri `outputs/` klasorune kaydedildi.

## Discussion

Laboratuvar notebookundaki hiperparametreler yeni veri setinde dogrudan iyi sonuc vermedi. `manual_lab_baseline` modelinin surekli pozitif sinif tahmini uretmesi, bu veri setinde modelin belirgin bicimde underfitting yasadigini gosterdi. Bu nedenle sadece veri setini degistirmek yetmedi; veri olcekleme ve egitim suresi de yeniden ayarlandi.

Standardization ve daha yuksek ogrenme orani eklendiginde manuel model hizla toparlandi ve dogruluk `0.9298` seviyesine cikti. Daha sonra ikinci gizli katman, mini-batch egitim ve L2 regularization eklenince validation ve test performansi daha dengeli hale geldi. Train accuracy ile validation accuracy arasindaki fark sinirli kaldigi icin modelin asiri ezberleme davranisi kontrol altinda tutuldu.

`scikit-learn MLPClassifier` ile elde edilen sonuclar, manuel uygulamanin davranisinin makul oldugunu dogruladi. Her iki tarafta da benzer performans seviyeleri goruldu. Bu, manuel ileri yayilim, geri yayilim ve parametre guncelleme adimlarinin beklenen sekilde calistigini desteklemektedir.

PyTorch karsiligi bu ortamda `torch` paketi kurulu olmadigi icin eklenmedi. Paket kurulduktan sonra ayni veri bolme ve hiperparametreler ile bu bolum de tamamlanabilir.
