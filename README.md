# Breast Cancer Binary Classification with Reproducible MLP Pipelines

Bu proje, `YZM304_Proje_Odevi1_2526.pdf` kapsamında hazırlanmış, **tekrar üretilebilir** bir ikili sınıflandırma çalışmasıdır. Veri kaynağı olarak `sklearn.datasets.load_breast_cancer` kullanılmıştır. Çalışma; temel laboratuvar modelinden başlayarak veri ön işleme, mimari karşılaştırmaları, veri miktarının etkisi, regularization ve farklı backend tekrarlarını kapsayan uçtan uca bir deney akışı sunar.

---

## Project Overview

Bu repoda amaç, meme kanseri veri seti üzerinde çalışan çok katmanlı algılayıcı (MLP) modellerini sistematik biçimde incelemek ve tüm süreci **aynı ayarlar, aynı veri bölmeleri ve aynı başlangıç koşullarıyla** tekrar üretilebilir hale getirmektir.

Çalışma şu sorulara odaklanır:

- Ham veri ile standardize veri arasındaki fark ne kadar belirleyici?
- Daha geniş ya da daha derin mimariler performansı nasıl etkiliyor?
- L2 regularization gerçekten anlamlı bir katkı sağlıyor mu?
- Eğitim verisi miktarı modelin genellemesini nasıl değiştiriyor?
- Aynı deney NumPy, scikit-learn ve PyTorch üzerinde benzer sonuçlar veriyor mu?

---

## Environment Setup

### Requirements

- Python `3.11`

### Virtual Environment

```bash
py -3.11 -m venv .venv
```

### Installation

```bash
.\bootstrap.ps1
```

### Run Full Pipeline

```bash
.\run_project.ps1
```

veya

```bash
.\.venv\Scripts\python.exe -m src.run_all
```

### Run Tests

```bash
.\.venv\Scripts\python.exe -m pytest
```

---

## Dataset and Split Strategy

Projede kullanılan veri seti:

- **Dataset:** `sklearn.datasets.load_breast_cancer`
- **Toplam örnek sayısı:** `569`
- **Özellik sayısı:** `30`
- **Sınıflar:** `malignant (0)`, `benign (1)`

### Class Distribution

- `212` malignant (`%37.3`)
- `357` benign (`%62.7`)

### Data Split

Veri üç parçaya ayrılmıştır:

- **Train:** `%60`
- **Validation:** `%20`
- **Test:** `%20`

Gerçek örnek sayıları:

- `341` train
- `114` validation
- `114` test

Tüm ayrımlar:

- `seed=42`
- `stratified split`

---

## Training Design

Tüm deneyler aynı temel eğitim mantığıyla yürütülmüştür:

- **Loss function:** Binary Cross Entropy
- **Optimizer:** Full-batch SGD
- **Decision threshold:** `0.5`
- **Global reproducibility seed:** `42`

### Weight Initialization

- Sigmoid katmanlar için: `N(0, sqrt(2 / (fan_in + fan_out)))`
- ReLU gizli katmanlar için: `N(0, sqrt(2 / fan_in))`
- Tüm bias değerleri: `0`

### Full-Batch Equivalence

Bu projede bir epoch, doğrudan bir optimizer update’e karşılık gelir.

- Tam train split için `batch_size=341`
- `%75` train varyantı için `batch_size=255`
- `%50` train varyantı için `batch_size=170`

Dolayısıyla:

- `1 epoch = 1 update`
- `n_steps = epochs`

### Model Selection Rule

Sınıf dağılımı dengesiz olduğu için model seçimi şu sıraya göre yapılmıştır:

1. Validation balanced accuracy (azalan)
2. `n_steps` (artan)
3. Parametre sayısı (artan)
4. Validation ROC-AUC (azalan)

---

## Experiments

### NumPy Experiments

Aşağıdaki deneyler NumPy tabanlı referans akış üzerinde çalıştırılmıştır:

- **`baseline_raw`**
  - Mimari: `30-8-1`
  - Aktivasyon: `sigmoid`
  - Veri: ham
  - `lr=0.08`
  - `epochs=1200`

- **`baseline_scaled`**
  - Mimari: `30-8-1`
  - Aktivasyon: `sigmoid`
  - Veri: standardize
  - `lr=0.08`
  - `epochs=1200`

- **`wide_scaled`**
  - Mimari: `30-16-1`
  - Aktivasyon: `sigmoid`
  - Veri: standardize
  - `lr=0.05`
  - `epochs=1400`

- **`deep_scaled_no_l2`**
  - Mimari: `30-32-16-1`
  - Aktivasyon: `relu`
  - Veri: standardize
  - `lr=0.01`
  - `epochs=1400`

- **`deep_scaled_l2`**
  - Mimari: `30-32-16-1`
  - Aktivasyon: `relu`
  - Veri: standardize
  - `lr=0.01`
  - `epochs=1400`
  - `L2=1e-3`

- **`deep_scaled_l2_data50`**
  - Aynı derin model
  - Train verisinin `%50`’si

- **`deep_scaled_l2_data75`**
  - Aynı derin model
  - Train verisinin `%75`’i

- **`deep_scaled_l2_data100`**
  - Aynı derin model
  - Train verisinin `%100`’ü

---

## Backend Reproductions

Aynı deney akışı üç farklı backend üzerinde tekrar edilmiştir:

- **NumPy** referans implementasyonu
- **scikit-learn** `MLPClassifier`
- **PyTorch** `nn.Module`

### Backend-specific Notes

#### NumPy

- Deterministic full-batch SGD
- `shuffle=False`
- `momentum=0`
- L2 ceza terimi doğrudan loss’a eklenir

#### scikit-learn

- `solver='sgd'`
- `batch_size=train_size`
- `shuffle=False`
- `momentum=0.0`
- `nesterovs_momentum=False`
- `max_iter=1`
- `warm_start=True`
- Epoch döngüsü `partial_fit` ile ilerletilir

#### PyTorch

- `torch.optim.SGD(lr=...)`
- `momentum=0`
- `weight_decay=L2`
- `torch.manual_seed(42)`
- `torch.use_deterministic_algorithms(True)`

### Shared Assets

- Başlangıç ağırlıkları: `data/weights/*.npz`
- Optimizer family: `SGD`
- Split manifest: `data/splits/split_manifest.json`

---

## Repository Structure

```text
src/        -> veri hazırlama, NumPy MLP, sklearn adapter, PyTorch model ve raporlama kodları
data/       -> ham veri çıktıları, split manifestleri ve başlangıç ağırlıkları
outputs/    -> tablolar, confusion matrix’ler, eğitim eğrileri ve raporlar
tests/      -> tekrar üretilebilirlik ve eğitim doğrulama testleri
```

Ek olarak:

- `Breast_Cancer_MLP_Project.ipynb` dosyası, pipeline ile uyumlu bir destekleyici notebook olarak yer almaktadır.

---

## Results

Üretilen tüm sonuçlar şu klasörlerde toplanır:

- **Tables:** `outputs/tables/`
- **Figures:** `outputs/figures/`
- **Reports:** `outputs/reports/`

Önemli çıktı dosyaları:

- `outputs/tables/model_selection.csv`
- `outputs/tables/backend_comparison_metrics.csv`
- `outputs/reports/selected_model_report.md`

---

## NumPy Experiment Summary

| Experiment | Val Balanced Acc | Val Accuracy | Test Accuracy | n_steps | Comment |
|---|---:|---:|---:|---:|---|
| `baseline_raw` | `0.5000` | `0.6228` | `0.6316` | `1200` | Ham özelliklerle belirgin underfitting gözlendi. |
| `baseline_scaled` | `0.9581` | `0.9649` | `0.9649` | `1200` | Standardization tek başına ciddi iyileştirme sağladı. |
| `wide_scaled` | `0.9651` | `0.9737` | `0.9737` | `1400` | En iyi balanced accuracy grubunda ve aynı adım sayısında daha düşük parametreli. |
| `deep_scaled_no_l2` | `0.9581` | `0.9649` | `0.9561` | `1400` | Daha derin yapı testte geniş modele göre geride kaldı. |
| `deep_scaled_l2` | `0.9581` | `0.9649` | `0.9561` | `1400` | L2, accuracy’yi değiştirmedi ancak ağırlık normunu azalttı. |
| `deep_scaled_l2_data50` | `0.9651` | `0.9737` | `0.9649` | `1400` | Balanced accuracy’de seçilen modelle eşit fakat daha fazla parametre içeriyor. |
| `deep_scaled_l2_data75` | `0.9651` | `0.9737` | `0.9474` | `1400` | Validation’da güçlü ama testte daha zayıf genelleme. |
| `deep_scaled_l2_data100` | `0.9581` | `0.9649` | `0.9561` | `1400` | Derin modelin tam veri sürümü. |

---

## Selected Model

Bu çalışmada seçilen model:

- **Model:** `wide_scaled`
- **Architecture:** `30-16-1`
- **Hidden activation:** `sigmoid`
- **Preprocessing:** `StandardScaler`

### Final Metrics

- **Validation balanced accuracy:** `0.9651`
- **Validation accuracy:** `0.9737`
- **Test accuracy:** `0.9737`
- **Test F1:** `0.9790`
- **Test ROC-AUC:** `0.9954`

### Why This Model?

`wide_scaled`, validation balanced accuracy açısından en iyi grup içinde yer almıştır. Aynı başarıyı gösteren daha derin alternatifler bulunmasına rağmen, daha az parametre kullanması nedeniyle seçim kriterlerine göre en uygun model olarak belirlenmiştir.

---

## Backend Comparison

| Architecture | NumPy Test Acc | sklearn Test Acc | PyTorch Test Acc | NumPy Test Balanced Acc |
|---|---:|---:|---:|---:|
| `30-8-1` | `0.9649` | `0.9649` | `0.9649` | `0.9673` |
| `30-16-1` | `0.9737` | `0.9737` | `0.9737` | `0.9742` |
| `30-32-16-1` | `0.9561` | `0.9561` | `0.9561` | `0.9554` |

Bu tablo, aynı splitler, aynı başlangıç ağırlıkları ve aynı SGD ayarları altında NumPy, scikit-learn ve PyTorch backend’lerinin eşdeğer çıktılar üretebildiğini göstermektedir.

Confusion matrix görselleri:

- `outputs/figures/confusion_matrix_*`

---

## Discussion

Bu çalışma iki temel sonucu açık biçimde ortaya koymaktadır:

### 1) Veri ön işleme kritik önemdedir

Ham veri ile eğitilen `baseline_raw` modeli, düşük doğruluk seviyelerinde kalmıştır. Aynı mimarinin yalnızca standardization uygulanmış sürümü ise ciddi bir performans sıçraması göstermiştir. Bu da, bu veri seti için en belirleyici iyileştirmelerden birinin veri ön işleme olduğunu göstermektedir.

### 2) Daha derin model her zaman daha iyi değildir

Daha karmaşık olan `30-32-16-1` mimarisi, bu problemde daha geniş fakat daha sade olan `30-16-1` modelini geçememiştir. Sonuç olarak, tek gizli katmanlı ama daha geniş yapı daha dengeli bir genelleme sunmuştur.

### 3) L2 regularization sınırlı ama anlamlı bir etki göstermiştir

L2 regularization bu deney düzeninde accuracy üzerinde belirgin bir artış oluşturmamıştır. Ancak ağırlık normunda küçük bir azalma sağlayarak parametre büyüklüğünü kontrol altında tutmuştur.

### 4) Veri miktarı tek başına her şeyi çözmemektedir

`%50` ve `%75` train alt kümeleri validation tarafında rekabetçi sonuçlar verebilmiştir. Buna rağmen test tarafında daha istikrarlı genelleme yine seçilen geniş modelde görülmüştür.

---

## Reproducibility

Bu repo tek ve kanonik bir deney akışı sunar. Eski sürümlerle geriye dönük uyumluluk katmanları eklenmemiştir. Temiz bir kurulumdan sonra aşağıdaki adımlar izlenerek tüm çıktıların yeniden üretilmesi mümkündür:

```bash
py -3.11 -m venv .venv
.\bootstrap.ps1
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m src.run_all
```

---

## Conclusion

Bu proje, meme kanseri veri seti üzerinde çalışan MLP tabanlı sınıflandırma deneylerini yalnızca performans odaklı değil, aynı zamanda **karşılaştırılabilir**, **yorumlanabilir** ve **tekrar üretilebilir** bir yapıda sunmaktadır. Elde edilen sonuçlar, özellikle veri ön işleme ve mimari sadeliğin bu görevde büyük önem taşıdığını göstermektedir.
