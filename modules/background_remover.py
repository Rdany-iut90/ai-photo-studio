"""
Module de suppression de fond.

Modeles disponibles (taille ONNX / compatibilite Streamlit Cloud) :
  - isnet-general-use  : IS-Net ~177 MB  — defaut Cloud (bon equilibre)
  - silueta            : Silueta ~44 MB  — le plus leger
  - u2net              : U2Net  ~176 MB  — modele original
  - birefnet-general-lite : BiRefNet ~370 MB  — qualite superieure
  - birefnet-general   : BiRefNet ~973 MB  — HORS QUOTA Cloud (1 GB RAM)

Streamlit Cloud (tier gratuit) = 1 GB RAM.
BiRefNet-general depasse cette limite → crash.
Le defaut est donc isnet-general-use (bonne qualite, ~177 MB).
"""

import streamlit as st
from PIL import Image, ImageFilter
from typing import Optional


@st.cache_resource(show_spinner="Chargement du modele de detourage...")
def _load_session(model_name: str):
    """
    Charge et met en cache la session rembg via st.cache_resource.
    Le modele n'est telecharge qu'une seule fois par session serveur.
    """
    from rembg import new_session
    return new_session(model_name)


def remove_background(
    image: Image.Image,
    model: str = "isnet-general-use",
    alpha_matting: bool = False,
    alpha_matting_fg_threshold: int = 240,
    alpha_matting_bg_threshold: int = 10,
    smooth_edges: bool = True,
    smooth_radius: int = 1,
    max_size: int = 800,
) -> Image.Image:
    """
    Supprime le fond d'une image avec rembg.

    Args:
        image        : Image PIL RGB en entree.
        model        : Modele rembg (voir AVAILABLE_MODELS).
        alpha_matting: Raffiner les bords fins (cheveux, poils).
        smooth_edges : Flou gaussien sur le canal alpha.
        smooth_radius: Rayon du flou (0-3).
        max_size     : Taille max avant traitement (limite RAM Cloud).

    Returns:
        Image RGBA avec fond transparent.
    """
    if image.mode not in ("RGB",):
        image = image.convert("RGB")

    # Redimensionnement pour limiter la RAM utilisee
    original_size = image.size
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        image_proc = image.resize(
            (int(image.width * ratio), int(image.height * ratio)),
            Image.LANCZOS,
        )
    else:
        image_proc = image

    # Charger le modele (mis en cache par Streamlit)
    session = _load_session(model)

    from rembg import remove
    if alpha_matting:
        result = remove(
            image_proc,
            session=session,
            alpha_matting=True,
            alpha_matting_foreground_threshold=alpha_matting_fg_threshold,
            alpha_matting_background_threshold=alpha_matting_bg_threshold,
            alpha_matting_erode_size=10,
        )
    else:
        result = remove(image_proc, session=session)

    # Remettre a la taille originale
    if result.size != original_size:
        result = result.resize(original_size, Image.LANCZOS)

    result = result.convert("RGBA")

    # Lisser les bords du masque alpha
    if smooth_edges and smooth_radius > 0:
        r, g, b, a = result.split()
        a = a.filter(ImageFilter.GaussianBlur(radius=smooth_radius))
        result = Image.merge("RGBA", (r, g, b, a))

    return result


# Modeles disponibles — du plus leger au plus lourd
# (ordre important : defaut = index 0 = isnet-general-use)
AVAILABLE_MODELS = {
    "IS-Net General (recommande Cloud)": "isnet-general-use",
    "Silueta (le plus leger)":           "silueta",
    "U2Net (original)":                  "u2net",
    "BiRefNet Lite (meilleure qualite)": "birefnet-general-lite",
    "BiRefNet General (hors quota Cloud)": "birefnet-general",
}
