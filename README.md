# 🎨 AI Photo Studio

> **Projet IA Générative** — Mastère Data Scientist — YNOV Campus Lyon  
> Application de suppression et remplacement de fond par IA générative

---

## 📖 Description

**AI Photo Studio** est une application web interactive qui permet de :

1. **Supprimer automatiquement** le fond d'une image grâce au modèle de deep learning **U2Net**
2. **Générer un nouveau fond** à partir d'une description textuelle libre via **Stable Diffusion**
3. **Assembler** le sujet original sur le fond généré pour obtenir l'image finale

L'interface est construite avec **Streamlit** et les modèles proviennent de **HuggingFace**.

---

## ✨ Fonctionnalités

| Fonctionnalité | Détail |
|----------------|--------|
| 🎭 Suppression de fond | U2Net (rembg) — détection automatique du sujet |
| 🎨 Génération de fond | Stable Diffusion (SDXL, SD 2.1, SD 1.5) via HuggingFace |
| 💻 Mode local | Génération hors-ligne avec GPU (optionnel) |
| 🎨 Mode démo | Dégradé coloré sans clé API ni GPU |
| ⬇️ Export | Téléchargement de l'image résultante en PNG |
| 🔁 Pipeline étapes | Visualisation des 3 étapes de traitement |

---

## 🛠️ Stack technique

```
Suppression fond   →  rembg (U2Net)         [HuggingFace model]
Génération fond    →  Stable Diffusion       [HuggingFace Inference API]
Interface          →  Streamlit
Traitement image   →  Pillow (PIL)
Environnement      →  python-dotenv
```

### Modèles HuggingFace utilisés

| Tâche | Modèle | Lien |
|-------|--------|------|
| Segmentation | `danielgatis/rembg` (U2Net) | [🔗](https://huggingface.co/danielgatis/rembg) |
| Génération (défaut) | `stabilityai/stable-diffusion-xl-base-1.0` | [🔗](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) |
| Génération (alt) | `stabilityai/stable-diffusion-2-1` | [🔗](https://huggingface.co/stabilityai/stable-diffusion-2-1) |
| Génération (léger) | `runwayml/stable-diffusion-v1-5` | [🔗](https://huggingface.co/runwayml/stable-diffusion-v1-5) |

---

## 🚀 Installation & lancement

### Prérequis

- Python 3.10+
- pip

### 1. Cloner / télécharger le projet

```bash
cd ai-photo-studio
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

> 💡 Pour la génération en **local avec GPU**, décommentez les lignes optionnelles dans `requirements.txt`

### 3. Configurer le token HuggingFace

```bash
# Copier le fichier d'exemple
cp .env.example .env

# Éditer .env et remplacer par votre token
# Créez votre token sur : https://huggingface.co/settings/tokens
```

### 4. Lancer l'application

```bash
streamlit run app.py
```

L'application s'ouvre automatiquement sur `http://localhost:8501`

---

## 📁 Structure du projet

```
ai-photo-studio/
│
├── app.py                          # Application Streamlit principale
│
├── modules/
│   ├── __init__.py
│   ├── background_remover.py       # Suppression de fond (rembg / U2Net)
│   ├── background_generator.py     # Génération de fond (HF API / diffusers / démo)
│   └── image_compositor.py         # Assemblage alpha composite (Pillow)
│
├── requirements.txt                # Dépendances Python
├── .env.example                    # Exemple de configuration
└── README.md                       # Ce fichier
```

---

## 🎯 Pipeline de traitement

```
┌─────────────┐     ┌──────────────────┐     ┌───────────────────┐     ┌──────────────┐
│  Image      │     │  Suppression     │     │  Génération fond  │     │  Assemblage  │
│  originale  │────▶│  fond (U2Net)    │────▶│  (Stable Diff.)   │────▶│  (composite) │
│  (RGB)      │     │  → RGBA + alpha  │     │  prompt texte     │     │  → RGB final │
└─────────────┘     └──────────────────┘     └───────────────────┘     └──────────────┘
```

---

## 💡 Exemples de prompts

```
une plage tropicale au coucher du soleil avec des palmiers et des eaux turquoises
une ville futuriste de nuit avec des néons et de la pluie
une forêt enchantée avec des rayons de lumière et de la brume matinale
un studio photo professionnel avec fond blanc et éclairage doux
un ciel étoilé avec la Voie Lactée au-dessus de montagnes enneigées
un bureau minimaliste moderne avec vue sur une grande ville
```

---

## 🔧 Configuration avancée

| Paramètre | Valeur recommandée | Description |
|-----------|-------------------|-------------|
| Guidance Scale | 7.5 | Force du guidage par le prompt (1–20) |
| Étapes inférence | 20–30 | Qualité vs vitesse de génération |
| Modèle | SDXL | Meilleure qualité, plus lent |

---

## 📋 Livrables

- ✅ Code source Python (Streamlit + modules)
- ✅ README avec technologies utilisées
- ✅ Fichier de dépendances (`requirements.txt`)
- ✅ Configuration d'environnement (`.env.example`)
- 🎬 Vidéo démo à produire (capture écran de l'application)

---

## 👤 Auteur

Projet réalisé dans le cadre du cours **Deep Learning — IA Générative**  
Mastère Data Scientist · YNOV Campus Lyon  
Encadrant : Antoine Vacavant
