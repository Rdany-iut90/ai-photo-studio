"""
Module d'assemblage d'images.

Assemble le sujet (premier plan avec transparence) sur le fond généré
en utilisant la composition alpha de Pillow.
"""

from PIL import Image


def composite_images(
    foreground: Image.Image,
    background: Image.Image,
) -> Image.Image:
    """
    Assemble le premier plan (sujet) sur le fond généré.

    Le premier plan doit être une image RGBA (canal alpha = masque du sujet).
    Le fond est redimensionné pour correspondre exactement aux dimensions
    du premier plan afin de garantir une correspondance pixel-à-pixel.

    Args:
        foreground (Image.Image): Image RGBA — sujet avec fond transparent.
        background (Image.Image): Image RGB/RGBA — fond généré par l'IA.

    Returns:
        Image.Image: Image RGB finale composée.
    """
    # Garantir que le premier plan est en RGBA
    foreground = foreground.convert("RGBA")

    # Redimensionner le fond aux dimensions du premier plan
    # (le fond peut avoir été généré à une résolution différente)
    background = background.resize(foreground.size, Image.LANCZOS).convert("RGBA")

    # Composition alpha : fond + premier plan (respect du canal alpha)
    # alpha_composite applique : result = bg * (1 - alpha) + fg * alpha
    result = Image.alpha_composite(background, foreground)

    # Retourner en RGB (PNG final sans transparence)
    return result.convert("RGB")
