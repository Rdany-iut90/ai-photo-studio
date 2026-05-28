"""
Module de génération de fond.

Propose trois modes :
  1. HuggingFace Inference API  → FLUX.1-schnell via serveurs HF gratuits
  2. Local (diffusers)          → Stable Diffusion en local (GPU requis)
  3. Démo                       → Dégradé coloré sans IA (test rapide)

Architecture d'appel (v3) :
  Après tests, seul provider="hf-inference" fonctionne sans quota payant.
  Le modèle FLUX.1-schnell est le seul modèle image disponible et fiable
  sur ce provider en mai 2025 (FLUX.1-dev → 410, SD 2.1 → 404).
  Anciennes erreurs corrigées :
    - 500 Together AI (provider par défaut, payant)
    - DNS failure api-inference.huggingface.co (endpoint déprécié)
    - 404 SD2.1 sur hf-inference (non routé)
"""

import io
import time
from PIL import Image, ImageDraw
from typing import Optional


# ---------------------------------------------------------------------------
# Modèles VALIDES avec provider="hf-inference" (testés et fonctionnels)
#
# Seul FLUX.1-schnell est supporté par le provider hf-inference gratuit.
# SD 1.5, SD 2.1, SDXL, FLUX.1-dev → "Model not supported by provider"
#
# Trois presets FLUX avec différents nombres d'étapes :
#   - Rapide  : 4 steps  (quelques secondes, très bon)
#   - Equilibre: 8 steps  (meilleure cohérence)
#   - Qualite : 20 steps (meilleur détail, plus lent)
# ---------------------------------------------------------------------------
FREE_TIER_MODELS = {
    "FLUX.1-schnell — Rapide (4 etapes)":   "black-forest-labs/FLUX.1-schnell",
    "FLUX.1-schnell — Equilibre (8 etapes)": "black-forest-labs/FLUX.1-schnell",
    "FLUX.1-schnell — Qualite (20 etapes)":  "black-forest-labs/FLUX.1-schnell",
}

# Nombre d'étapes associé à chaque preset FLUX
FLUX_STEPS_PRESETS = {
    "FLUX.1-schnell — Rapide (4 etapes)":   4,
    "FLUX.1-schnell — Equilibre (8 etapes)": 8,
    "FLUX.1-schnell — Qualite (20 etapes)":  20,
}


# ---------------------------------------------------------------------------
# Mode 1 : HuggingFace Inference API (gratuit — serveurs HF uniquement)
# ---------------------------------------------------------------------------

def generate_background_hf_api(
    prompt: str,
    width: int = 768,
    height: int = 768,
    hf_token: Optional[str] = None,
    model: str = "black-forest-labs/FLUX.1-schnell",
    guidance_scale: float = 3.5,
    num_inference_steps: int = 4,
) -> Image.Image:
    """
    Génère un fond via l'API HuggingFace Inference (serveurs gratuits HF).

    Utilise provider="hf-inference" pour forcer les serveurs HF gratuits
    au lieu du routage automatique vers Together AI / Replicate (payants).

    Modèle par défaut : FLUX.1-schnell
      - Modèle de diffusion rapide (4 étapes suffisent)
      - Disponible sur hf-inference sans quota payant
      - Génère à la résolution demandée (jusqu'à 1024px)
      - guidance_scale recommandé : 3.5 (spécifique à FLUX)

    Args:
        prompt              : Description du fond souhaité.
        width               : Largeur cible (multiple de 8, max 1024).
        height              : Hauteur cible.
        hf_token            : Token HuggingFace (lecture suffisante).
        model               : ID du modèle HF à utiliser.
        guidance_scale      : Force du guidage (3.5 pour FLUX, 7.5 pour SD).
        num_inference_steps : Étapes (4 pour FLUX-schnell, 20+ pour SD).

    Returns:
        Image.Image: Image PIL RGB du fond généré.

    Raises:
        RuntimeError: Si l'API échoue avec un message d'aide contextuel.
    """
    from huggingface_hub import InferenceClient

    # Dimensions compatibles : multiples de 8, entre 256 et 1024
    target_w = _snap_to_multiple(min(max(width, 256), 1024), 8)
    target_h = _snap_to_multiple(min(max(height, 256), 1024), 8)

    # ----------------------------------------------------------------
    # provider="hf-inference" → routage vers les serveurs HF gratuits
    # (evite Together AI, Replicate, fal-ai, tous payants)
    # ----------------------------------------------------------------
    client = InferenceClient(
        provider="hf-inference",
        token=hf_token,
    )

    try:
        image = client.text_to_image(
            prompt=prompt,
            model=model,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            width=target_w,
            height=target_h,
        )

    except Exception as e:
        err = str(e)

        if "401" in err or "unauthorized" in err.lower():
            raise RuntimeError(
                "Token HuggingFace invalide (401). "
                "Verifiez sur huggingface.co/settings/tokens"
            ) from e
        elif "503" in err or "loading" in err.lower():
            raise RuntimeError(
                "Modele en cours de chargement (503). "
                "Reessayez dans 30 secondes."
            ) from e
        elif "404" in err:
            raise RuntimeError(
                f"Modele introuvable sur hf-inference (404) : {model}\n"
                "Selectionnez 'FLUX.1-schnell' qui est le seul modele "
                "valide sur le provider hf-inference gratuit."
            ) from e
        elif "410" in err:
            raise RuntimeError(
                f"Modele deprecie (410) : {model}\n"
                "Utilisez FLUX.1-schnell a la place."
            ) from e
        elif "500" in err:
            raise RuntimeError(
                "Erreur serveur HF (500). "
                "Cela peut etre transitoire — reessayez dans quelques secondes. "
                "Si le probleme persiste, verifiez le status sur status.huggingface.co"
            ) from e
        else:
            raise RuntimeError(f"Erreur API HuggingFace : {err[:300]}") from e

    return image.convert("RGB")


# ---------------------------------------------------------------------------
# Mode 2 : Génération locale avec diffusers
# ---------------------------------------------------------------------------

def generate_background_local(
    prompt: str,
    width: int = 512,
    height: int = 512,
    model: str = "runwayml/stable-diffusion-v1-5",
    guidance_scale: float = 7.5,
    num_inference_steps: int = 20,
) -> Image.Image:
    """
    Génère un fond localement via la bibliothèque diffusers (HuggingFace).

    Télécharge le modèle au 1er lancement (~4 GB).
    Nécessite un GPU NVIDIA avec 6 GB+ de VRAM.
    Fonctionne sur CPU mais très lentement (~10 min).
    """
    import torch
    from diffusers import StableDiffusionPipeline

    width = _snap_to_multiple(min(max(width, 256), 768), 64)
    height = _snap_to_multiple(min(max(height, 256), 768), 64)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        model,
        torch_dtype=dtype,
        safety_checker=None,
    ).to(device)

    if device == "cpu":
        pipe.enable_attention_slicing()

    result = pipe(
        prompt=prompt,
        negative_prompt="blurry, low quality, distorted, ugly",
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
    )

    return result.images[0]


# ---------------------------------------------------------------------------
# Mode 3 : Génération de démonstration (sans IA)
# ---------------------------------------------------------------------------

def generate_background_demo(
    prompt: str,
    width: int = 512,
    height: int = 512,
) -> Image.Image:
    """
    Génère un fond de démonstration (dégradé coloré) sans modèle IA.
    Utile pour tester le pipeline sans clé API ni GPU.
    La couleur est dérivée du prompt de façon déterministe.
    """
    import hashlib

    h = hashlib.sha256(prompt.encode()).hexdigest()
    r1, g1, b1 = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    r2, g2, b2 = int(h[6:8], 16), int(h[8:10], 16), int(h[10:12], 16)

    r1, g1, b1 = max(r1, 80), max(g1, 60), max(b1, 100)
    r2, g2, b2 = min(r2 + 50, 255), min(g2 + 50, 255), min(b2 + 50, 255)

    image = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(image)

    for y in range(height):
        t = y / max(height - 1, 1)
        r = int(r1 + (r2 - r1) * t)
        g = int(g1 + (g2 - g1) * t)
        b = int(b1 + (b2 - b1) * t)
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    return image


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def _snap_to_multiple(value: int, multiple: int) -> int:
    """Arrondit au multiple inférieur le plus proche."""
    return (value // multiple) * multiple
