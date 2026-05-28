"""
Module de suppression de fond — version améliorée.

Modèles disponibles (par ordre de qualité décroissante) :
  - birefnet-general       : BiRefNet — état de l'art, meilleure qualité générale ★★★★★
  - birefnet-general-lite  : BiRefNet léger — bonne qualité, plus rapide ★★★★☆
  - isnet-general-use      : IS-Net — bon équilibre qualité/vitesse ★★★☆☆
  - silueta                : Bon pour les objets nets ★★★☆☆
  - u2net                  : Modèle original — rapide mais moins précis ★★☆☆☆

Référence : https://github.com/danielgatis/rembg
Modèle BiRefNet : https://huggingface.co/ZhengPeng7/BiRefNet
"""

from PIL import Image, ImageFilter
from rembg import new_session, remove
from typing import Optional

# Cache des sessions pour ne pas recharger le modèle à chaque appel
_session_cache: dict = {}


def get_session(model_name: str):
    """Retourne (et met en cache) une session rembg pour le modèle donné."""
    if model_name not in _session_cache:
        _session_cache[model_name] = new_session(model_name)
    return _session_cache[model_name]


def remove_background(
    image: Image.Image,
    model: str = "birefnet-general",
    alpha_matting: bool = False,
    alpha_matting_fg_threshold: int = 240,
    alpha_matting_bg_threshold: int = 10,
    smooth_edges: bool = True,
    smooth_radius: int = 1,
    max_size: int = 1024,
) -> Image.Image:
    """
    Supprime le fond d'une image avec rembg (BiRefNet par défaut).

    Améliorations v2 :
    - BiRefNet comme modèle par défaut (bien plus précis que U2Net)
    - Cache de session (pas de rechargement entre deux appels)
    - Alpha matting optionnel pour les bords très fins (cheveux, fourrure)
    - Lissage des bords du masque pour une intégration naturelle
    - Redimensionnement intelligent avant traitement

    Args:
        image                       : Image PIL d'entrée (RGB ou RGBA).
        model                       : Identifiant du modèle rembg à utiliser.
        alpha_matting               : Active le raffinement de bords fins.
        alpha_matting_fg_threshold  : Seuil premier plan pour l'alpha matting.
        alpha_matting_bg_threshold  : Seuil arrière plan pour l'alpha matting.
        smooth_edges                : Applique un léger flou gaussien sur l'alpha.
        smooth_radius               : Rayon du flou (1–3 recommandé).
        max_size                    : Taille max avant redimensionnement.

    Returns:
        Image.Image: Image RGBA avec fond transparent.
    """
    # Normaliser le mode d'entrée
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGB")
    elif image.mode == "RGBA":
        image = image.convert("RGB")

    # ----------------------------------------------------------------
    # Redimensionnement adaptatif
    # Le modèle est plus précis sur des images de taille raisonnable.
    # On réduit si nécessaire, puis on remet à l'échelle originale.
    # ----------------------------------------------------------------
    original_size = image.size
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        resized_size = (int(image.width * ratio), int(image.height * ratio))
        image_proc = image.resize(resized_size, Image.LANCZOS)
    else:
        image_proc = image

    # ----------------------------------------------------------------
    # Suppression du fond avec le modèle sélectionné
    # ----------------------------------------------------------------
    session = get_session(model)

    if alpha_matting:
        # Alpha matting : raffinement des contours fins (cheveux, poils…)
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

    # ----------------------------------------------------------------
    # Remise à la taille originale (si on avait redimensionné)
    # ----------------------------------------------------------------
    if result.size != original_size:
        result = result.resize(original_size, Image.LANCZOS)

    result = result.convert("RGBA")

    # ----------------------------------------------------------------
    # Lissage du canal alpha pour des bords naturels
    # Un léger flou gaussien évite les contours crénelés
    # ----------------------------------------------------------------
    if smooth_edges and smooth_radius > 0:
        r, g, b, a = result.split()
        a_smooth = a.filter(ImageFilter.GaussianBlur(radius=smooth_radius))
        result = Image.merge("RGBA", (r, g, b, a_smooth))

    return result


# Mapping nom affiché → identifiant modèle rembg
AVAILABLE_MODELS = {
    "BiRefNet General (meilleur)": "birefnet-general",
    "BiRefNet Lite (rapide)": "birefnet-general-lite",
    "IS-Net General": "isnet-general-use",
    "Silueta": "silueta",
    "U2Net (original)": "u2net",
}
