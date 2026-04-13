# Sistema Multimodal de IA para Diagnóstico Médico
## TFG — Grado en Business Analytics 

**Autor:** Alfredo Pérez Navalón  


---

## Descripción

Sistema multimodal de inteligencia artificial que integra **machine learning** (datos tabulares clínicos) y **deep learning** (imágenes dermatoscópicas) para apoyo al diagnóstico de enfermedad cardiovascular y lesiones cutáneas.

El sistema se concibe como herramienta de apoyo al profesional clínico, no como sustituto del juicio médico. El módulo de ML genera una segunda opinión sobre riesgo cardiovascular a partir de datos clínicos rutinarios. El módulo de DL clasifica imágenes dermatoscópicas entre 7 categorías diagnósticas.

### Datasets

| Dataset | Tipo | Registros | Clases | Fuente |
|---------|------|-----------|--------|--------|
| Heart Disease UCI | Tabular (4 hospitales) | 920 → 434 (tras limpieza) | 2 (sano / enfermo) | [UCI Repository](https://archive.ics.uci.edu/dataset/45/heart+disease) |
| HAM10000 | 10.015 imágenes dermatoscópicas | 7 lesiones cutáneas | 7 (nv, mel, bkl, bcc, akiec, vasc, df) | [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) |

### Modelos implementados

**Machine Learning (Heart Disease):** Logistic Regression, Random Forest, XGBoost, Gradient Boosting, AdaBoost, SVM, KNN — evaluados con validación cruzada estratificada (5-fold).

**Deep Learning (HAM10000):**
- CNN from scratch — baseline naive (1.2M parámetros, ~60% accuracy)
- ResNet-50 con transfer learning — línea base (~79% accuracy)
- EfficientNet-B3 con transfer learning — modelo avanzado (~78% accuracy)
- Ensemble — promedio de probabilidades ResNet + EfficientNet

**Estudios complementarios:**
- Estudio de data augmentation y sus consecuencias en el datatset (3 niveles: mínima, suave, agresiva)
- Validación con 5-Fold Cross-Validation estratificado por lesión

---

## Estructura del repositorio

```
TFG-Sistema-Multimodal-IA/
│
├
│
├── data/
│   ├── raw/
│   │   └── heart-disease/
│   │       ├── heart-disease.names            # Documentación original UCI
│   │       ├── processed.cleveland.data       # Cleveland Clinic (303 registros)
│   │       ├── processed.hungarian.data       # Hungarian Inst. Cardiology (294)
│   │       ├── processed.switzerland.data     # University Hospital Zürich (123)
│   │       └── processed.va.data              # VA Medical Center Long Beach (200)
│   │
│   └── processed/
│       └── heart_disease_clean.csv            # Dataset limpio (434 registros)
│
├── notebooks/
│   │
│   │   # --- HEART DISEASE (ejecución local) ---
│   ├── 01_heart_disease_id.ipynb              # Ingeniería del dato
│   ├── 06_analisis_dato_tabular.ipynb         # Análisis del dato (7 modelos ML)
│   │
    │--- HAM10000 (ejecución en Google Colab con GPU) ---
    │   ├── ID_HAM10000_vf.ipynb                   # Ingeniería del dato (metadata + imágenes)
    │   ├── AD1_HAM10000.ipynb        # Análisis del dato (CNN + ResNet + EfficientNet)
    │   ├── AD1_Augmentatiom_resnet50.ipynb        # Estudio de ablación: 3 niveles de augmentation
    │   └── AD1_Augmentatiom_cv_5fold_resnet50_vf.ipynb    # Cross-validation: suave vs. por clase
│
│
├──
│└── reports/
│   ├── figures_heart/                         # Figuras de Heart Disease
│    └── figures.png                         # Memoria completa del TFG
│``
│--- LICENSE
│--- README.md
│---requerimentes.txt

## Requisitos

## Entorno local (Heart Disease: notebooks 01 y 06)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

El `requirements.txt` incluye todas las versiones exactas para garantizar reproducibilidad.

Principales dependencias: pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, shap.

### Google Colab (HAM10000: notebooks 03–06)

Los notebooks de deep learning requieren **GPU** y están diseñados para Google Colab:

| Notebook | Descripción | GPU | Tiempo aprox. |
|----------|-------------|-----|---------------|
| `ID_HAM10000_id.ipynb` | Ingeniería del dato | CPU | ~5 min |
| `AD1_HAM10000_ad.ipynb` | CNN + ResNet + EfficientNet | T4 / A100 | ~2–3 horas |
| `AD2....ipynb` | 3 niveles × ResNet-50 | A100 | ~1.5 horas |
| `AD3.ipynb` | 5-Fold CV × 2 estrategias | A100 | ~2 horas |

Dependencias en Colab: PyTorch 2.x, torchvision, scikit-learn (preinstalados en Colab).

---

## Instrucciones de reproducción

### Paso 1 — Clonar el repositorio

```bash
git clone https://github.com/[tu-usuario]/TFG-Sistema-Multimodal-IA.git
cd TFG-Sistema-Multimodal-IA
```

### Paso 2 — Heart Disease (ejecución local)

Los datos brutos de Heart Disease están incluidos en `data/raw/heart-disease/`. Ejecutar en orden:

```
01_heart_disease_id_v3.ipynb  →  genera data/processed/heart_disease_clean.csv
06_analisis_dato_tabular.ipynb  →  entrena y evalúa 7 modelos ML (lee el CSV limpio)
```

No requiere GPU ni configuración adicional.

### Paso 3 — HAM10000 (ejecución en Google Colab)

#### 3.1 Descargar las imágenes (no incluidas por tamaño)

Descargar desde el Harvard Dataverse:  
https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T

Archivos necesarios:
- `HAM10000_images_part_1.zip` (~2.2 GB)
- `HAM10000_images_part_2.zip` (~0.8 GB)

#### 3.2 Subir a Google Drive

Crear la carpeta `MyDrive/HAM10000/` en Google Drive y subir:
```
Google Drive/
└── MyDrive/
    └── HAM10000/
        ├── HAM10000_images_part_1/    # Descomprimir aquí (~5000 imágenes .jpg)
        ├── HAM10000_images_part_2/    # Descomprimir aquí (~5000 imágenes .jpg)
        └── ham10000_metadata_clean.csv # Generado por notebook 0ID_HAm...
```

#### 3.3 Ejecutar notebooks en orden

1. **`03_HAM10000_id.ipynb`** — Ingeniería del dato. Genera `ham10000_metadata_clean.csv` con los splits, rutas de imágenes y metadata limpio. Este CSV es necesario para todos los notebooks siguientes.

2. **`AD1`** — Análisis principal. Entrena CNN from scratch, ResNet-50 y EfficientNet-B3. Genera las figuras comparativas y los modelos `.pth`.

3. **`AD2`** — Estudio de ablación. Compara 3 niveles de data augmentation (mínima, suave, agresiva) sobre ResNet-50. Genera 7 figuras de comparación.

4. **`AD3`** — Cross-validation. Valida la estrategia suave uniforme vs. augmentation diferenciada por clase con 5-Fold CV estratificado por lesión.

> **Nota:** Los notebooks 04–06 montan Google Drive automáticamente en la primera celda y copian las imágenes a disco local para acelerar el entrenamiento. La ruta esperada es `MyDrive/HAM10000/`.

---

## Resultados principalesen



### HAM10000 — Deep Learning

| Modelo | Accuracy | F1 Weighted | F1 Macro | AUC-ROC | Mel Recall |
|--------|----------|-------------|----------|---------|------------|
| CNN from scratch | 0.5962 | 0.6339 | 0.4144 | 0.8669 | 0.80 |
| ResNet-50 (TL) | 0.7950 | 0.7950 | 0.6600 | 0.9360 | 0.67 |
| EfficientNet-B3 (TL) | 0.7792 | 0.7910 | 0.6603 | 0.9360 | 0.67 |
| Ensemble | 0.7832 | 0.7944 | 0.6731 | — | — |

### Estudio de ablación — Data Augmentation (5-Fold CV)

| Estrategia | Accuracy | F1 Macro | AUC-ROC | Mel Recall |
|------------|----------|----------|---------|------------|
| **Suave (uniforme)** | **0.7647 ± 0.0135** | **0.6229** | **0.9342** | **0.537** |
| Por clase | 0.7231 ± 0.0085 | 0.4191 | 0.8590 | 0.006 |

La estrategia de augmentation suave uniforme supera en todas las métricas a la estrategia diferenciada por clase (hipotesisi del último cuaderno (AD3)), que colapsa catastróficamente en la detección de melanoma (recall 0.006).

---

## Explicabilidad

- **SHAP** (SHapley Additive exPlanations): aplicado a los modelos de ML sobre Heart Disease para identificar qué variables clínicas contribuyen más a cada predicción.
- **Grad-CAM** (Gradient-weighted Class Activation Mapping): aplicado a ResNet-50 y EfficientNet-B3 para visualizar qué regiones de la imagen dermatoscópica activan las decisiones del modelo.

---

## Tecnologías

| Herramienta | Uso |
|-------------|-----|
| Python 3.10+ | Lenguaje principal |
| scikit-learn | Modelos ML, métricas, cross-validation |
| XGBoost | Gradient boosting para datos tabulares |
| PyTorch + torchvision | Modelos de deep learning |
| SHAP | Explicabilidad en modelos tabulares |
| Google Colab (T4/A100) | Entrenamiento de modelos DL |
| Cursor / VS Code | Desarrollo local |
| Git + GitHub | Control de versiones |
| Matplotlib + Seaborn | Visualización |

---

## Bibliografía principal

- Esteva, A., et al. (2017). Dermatologist-level classification of skin cancer with deep neural networks. *Nature*, 542(7639), 115–118.
- Haenssle, H. A., et al. (2018). Man against machine. *Annals of Oncology*, 29(8), 1836–1842.
- Tschandl, P., et al. (2018). The HAM10000 dataset. *Scientific Data*, 5, 180161.
- He, K., et al. (2016). Deep Residual Learning for Image Recognition. *CVPR*.
- Tan, M. & Le, Q. (2019). EfficientNet: Rethinking Model Scaling. *ICML*.
- Salinas, M. P., et al. (2024). AI versus clinicians for skin cancer diagnosis. *npj Digital Medicine*, 7(141).

---

## Licencia

Este proyecto es un Trabajo Fin de Grado académico. Los datasets utilizados son públicos y están sujetos a sus respectivas licencias. Consultar las fuentes originales para condiciones de uso.

---

## Contacto

Alfredo Pérez Navalón — Universidad Francisco de Vitoria  
pereznavalionalfredo@gmail.com