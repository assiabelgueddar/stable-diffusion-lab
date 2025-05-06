# Stable Diffusion Lab — Fine-Tuning avec Dataset Personnalisé 

Ce projet vous guide à travers les étapes nécessaires pour **entraîner un modèle Stable Diffusion** avec vos propres images et légendes à l’aide de Hugging Face `diffusers`.

##  Prérequis

* Python 3.9+
* GPU avec CUDA (recommandé)
* Environnement Google Colab ou local configuré
* `git`, `wget` installés

---

##  Étapes d'exécution

### 1. Cloner le dépôt Git

```bash
!git clone https://github.com/assiabelgueddar/stable-diffusion-lab.git
%cd stable-diffusion-lab
```

### 2. Installer les dépendances

```bash
!pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install git+https://github.com/huggingface/diffusers
!pip install transformers accelerate datasets pandas
```

### 3. Préparer le dataset

Assurez-vous que le fichier `train_data.csv` contient deux colonnes :

* `image` : chemin vers l’image
* `text` : légende de l’image

Ensuite :

```bash
!python prepare_dataset.py
```

### 4. Configurer `accelerate`

```bash
!accelerate config default
```

> Cela crée automatiquement un fichier `default_config.yaml` pour la configuration de l'entraînement (ex. sur une seule machine, un seul GPU).

### 5. Télécharger le script d'entraînement

```bash
!wget https://raw.githubusercontent.com/huggingface/diffusers/main/examples/text_to_image/train_text_to_image.py
```

### 6. Transformer les données en format Hugging Face

```python
from datasets import Dataset, DatasetDict, Image
import pandas as pd

# Charger les données
df = pd.read_csv("train_data.csv")
df = df.rename(columns={"text": "caption"})
ds = Dataset.from_pandas(df)
ds = ds.cast_column("image", Image())
ds_dict = DatasetDict({"train": ds})
ds_dict.save_to_disk("hf_dataset")
```

### 7. (Optionnel) Recharger et vérifier le dataset

```python
from datasets import load_from_disk
ds = load_from_disk("hf_dataset")
print(ds["train"][0])
```

### 8. Lancer l'entraînement

```bash
!accelerate launch train_text_to_image.py \
  --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4 \
  --train_data_dir=hf_dataset \
  --image_column=image \
  --caption_column=caption \
  --output_dir=output \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler=constant \
  --lr_warmup_steps=0 \
  --max_train_steps=1000 \
  --checkpointing_steps=500 \
  --validation_prompt="a fluffy white cat with green eyes" \
  --seed=42
```

---

##  Structure recommandée

```
stable-diffusion-lab/
├── train_data.csv
├── prepare_dataset.py
├── train_text_to_image.py
├── hf_dataset/
├── output/
└── README.md
```

---

##  Remarques

* Le modèle pré-entraîné utilisé est `CompVis/stable-diffusion-v1-4`.
* Le `caption_column` et `image_column` doivent correspondre aux noms de colonnes dans `train_data.csv`.
* L'ajustement fin utilise 1000 étapes avec un batch size de 1.

---

##  Contact

Ce projet a été initialement préparé par **Assia Belgueddar**.
Pour toute question ou collaboration : [GitHub Profile](https://github.com/assiabelgueddar)


