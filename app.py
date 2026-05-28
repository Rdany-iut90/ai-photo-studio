"""
AI Photo Studio — Application principale Streamlit

Fonctionnement :
  1. L'utilisateur uploade une photo
  2. Le fond est supprimé automatiquement (U2Net via rembg)
  3. L'utilisateur décrit le nouveau fond souhaité en texte
  4. Un fond est généré par Stable Diffusion (HuggingFace API ou local)
  5. Le sujet est composé sur le nouveau fond
  6. L'image finale peut être téléchargée

Projet IA Générative — Mastère Data Scientist — YNOV Campus Lyon
"""

import io
import os

import streamlit as st
from dotenv import load_dotenv
from PIL import Image

# Charger les variables d'environnement depuis .env (si présent)
load_dotenv()

# Importer les modules métier
from modules.background_remover import remove_background, AVAILABLE_MODELS
from modules.background_generator import (
    generate_background_demo,
    generate_background_hf_api,
    generate_background_local,
    FREE_TIER_MODELS,
    FLUX_STEPS_PRESETS,
)
from modules.image_compositor import composite_images

# =============================================================================
# Configuration de la page Streamlit
# =============================================================================

st.set_page_config(
    page_title="AI Photo Studio",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS personnalisé pour améliorer l'apparence
st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.8rem;
            font-weight: 800;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-align: center;
            margin-bottom: 0.3rem;
        }
        .subtitle {
            text-align: center;
            color: #888;
            font-size: 1.1rem;
            margin-bottom: 1.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# Initialisation du session state
# =============================================================================

for key, default in {
    "result_image": None,
    "fg_image": None,
    "bg_image": None,
    "prompt_value": "",
    "rembg_model": "birefnet-general",
    "alpha_matting": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# =============================================================================
# En-tête
# =============================================================================

st.markdown('<h1 class="main-title">🎨 AI Photo Studio</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Supprimez le fond de votre photo et remplacez-le par ce que vous imaginez</p>',
    unsafe_allow_html=True,
)

# =============================================================================
# Barre latérale — Configuration
# =============================================================================

with st.sidebar:
    st.header("⚙️ Configuration")

    # -------------------------------------------------------
    # Section 1 : Suppression de fond
    # -------------------------------------------------------
    st.subheader("🎭 Suppression de fond")

    rembg_model_label = st.selectbox(
        "Modèle de détourage",
        options=list(AVAILABLE_MODELS.keys()),
        index=0,
        help=(
            "**BiRefNet General** : Meilleur choix pour tout type de scène.\n\n"
            "**BiRefNet Lite** : Plus rapide, qualité proche.\n\n"
            "**IS-Net / Silueta** : Alternatives compactes.\n\n"
            "**U2Net** : Modèle original, moins précis."
        ),
    )
    rembg_model = AVAILABLE_MODELS[rembg_model_label]

    alpha_matting = st.checkbox(
        "Alpha matting (bords fins)",
        value=False,
        help=(
            "Raffine les contours très fins (cheveux, poils, fourrure).\n"
            "Plus lent mais résultat plus propre sur ces cas."
        ),
    )

    smooth_radius = st.slider(
        "Lissage des bords",
        min_value=0,
        max_value=3,
        value=1,
        step=1,
        help="Flou gaussien appliqué au canal alpha. 0 = aucun lissage.",
    )

    st.divider()

    # -------------------------------------------------------
    # Section 2 : Génération du fond
    # -------------------------------------------------------
    st.subheader("🎨 Génération du fond")

    # --- Mode de génération ---
    generation_mode = st.radio(
        "Mode de génération",
        options=["🤗 HuggingFace API", "💻 Local (diffusers)", "🎨 Démo (sans IA)"],
        index=0,
        help=(
            "**HuggingFace API** : Recommandé. Nécessite un token HF gratuit.\n\n"
            "**Local** : Génère en local, nécessite un GPU (6 GB VRAM min).\n\n"
            "**Démo** : Dégradé coloré, aucune clé ni GPU requis."
        ),
    )

    st.divider()

    # --- Paramètres selon le mode ---
    hf_token = ""
    model_choice = ""
    guidance_scale = 7.5
    num_steps = 20

    if generation_mode == "🤗 HuggingFace API":
        hf_token = st.text_input(
            "Token HuggingFace",
            value=os.getenv("HF_TOKEN", ""),
            type="password",
            help="Obtenez un token gratuit sur huggingface.co/settings/tokens",
        )

        model_label = st.selectbox(
            "Qualite / vitesse",
            options=list(FREE_TIER_MODELS.keys()),
            index=0,
            help=(
                "Tous les presets utilisent FLUX.1-schnell — "
                "le seul modele valide sur le provider hf-inference gratuit.\n\n"
                "Rapide = 4 etapes | Equilibre = 8 | Qualite = 20"
            ),
        )
        model_choice  = FREE_TIER_MODELS[model_label]
        num_steps     = FLUX_STEPS_PRESETS[model_label]
        guidance_scale = 3.5   # Valeur optimale pour FLUX (fixe)

        st.info(
            "Modele : **FLUX.1-schnell** (Black Forest Labs)\n\n"
            "Provider : `hf-inference` (gratuit, sans quota payant)"
        )

    elif generation_mode == "💻 Local (diffusers)":
        model_label_local = st.selectbox(
            "Modèle (téléchargé localement)",
            options=[
                "runwayml/stable-diffusion-v1-5",
                "stabilityai/stable-diffusion-2-1",
            ],
        )
        model_choice = model_label_local
        guidance_scale = st.slider("Guidance Scale", 1.0, 20.0, 7.5, 0.5)
        num_steps = st.slider("Étapes d'inférence", 10, 50, 20, 5)
        st.warning("⚠️ Nécessite un GPU NVIDIA avec 6 GB+ de VRAM")

    else:
        st.info("ℹ️ Mode démo : génère un dégradé coloré basé sur le prompt.")

    st.divider()

    # --- Stack technique ---
    st.markdown("**🛠️ Stack technique**")
    st.markdown(
        """
        - 🎭 `rembg` + BiRefNet — Détourage
        - 🤗 HuggingFace Inference API
        - 🌊 Stable Diffusion — Génération
        - 💻 Streamlit — Interface
        - 🖼️ Pillow — Traitement image
        """
    )

    st.divider()
    st.caption("Projet IA Générative · Mastère Data Scientist · YNOV Campus Lyon")

# =============================================================================
# Zone principale — Upload
# =============================================================================

upload_col, _ = st.columns([2, 1])
with upload_col:
    uploaded_file = st.file_uploader(
        "📤 Importez votre image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Formats supportés : JPG, JPEG, PNG, WEBP",
    )

# =============================================================================
# Pipeline principal (si image uploadée)
# =============================================================================

if uploaded_file is not None:
    original_image = Image.open(uploaded_file).convert("RGB")

    st.divider()

    # --- Affichage original / résultat côte à côte ---
    col_orig, col_result = st.columns(2, gap="large")

    with col_orig:
        st.subheader("📷 Image originale")
        st.image(
            original_image,
            use_container_width=True,
            caption=f"{original_image.width} × {original_image.height} px",
        )

    with col_result:
        st.subheader("✨ Résultat")
        if st.session_state.result_image is not None:
            st.image(
                st.session_state.result_image,
                use_container_width=True,
                caption=f"{st.session_state.result_image.width} × {st.session_state.result_image.height} px",
            )
        else:
            st.info("🎨 Le résultat apparaîtra ici après génération.")

    st.divider()

    # --- Saisie du prompt ---
    st.subheader("✍️ Décrivez le fond souhaité")

    # Exemples rapides (boutons cliquables)
    st.markdown("**💡 Exemples rapides :**")
    ex_cols = st.columns(5)
    examples = [
        ("🏖️", "une plage tropicale au coucher du soleil avec des palmiers"),
        ("🌆", "une ville futuriste illuminée de nuit avec des néons"),
        ("🌲", "une forêt enchantée avec des rayons de lumière et de la brume"),
        ("🎨", "un studio photo professionnel avec fond blanc et éclairage doux"),
        ("🌌", "un ciel étoilé avec la Voie Lactée et des montagnes enneigées"),
    ]

    for i, (emoji, ex_text) in enumerate(examples):
        with ex_cols[i]:
            if st.button(f"{emoji}", key=f"ex_{i}", use_container_width=True, help=ex_text):
                st.session_state.prompt_value = ex_text
                st.rerun()

    prompt = st.text_area(
        "Description du fond",
        value=st.session_state.prompt_value,
        placeholder="Ex: une plage tropicale au coucher du soleil, palmiers, eaux turquoises...",
        height=90,
        label_visibility="collapsed",
    )

    # Options avancées
    with st.expander("⚙️ Options avancées"):
        negative_prompt = st.text_input(
            "Prompt négatif",
            value="blurry, low quality, distorted, ugly, deformed",
            help="Éléments à exclure de la génération",
        )

    st.divider()

    # --- Bouton de génération ---
    gen_col = st.columns([1, 2, 1])[1]
    with gen_col:
        generate_btn = st.button(
            "🚀 Générer le nouveau fond",
            use_container_width=True,
            type="primary",
            disabled=(not prompt.strip()),
        )

    # ==========================================================================
    # Exécution du pipeline IA
    # ==========================================================================

    if generate_btn and prompt.strip():

        # Validation du token si mode API
        if generation_mode == "🤗 HuggingFace API" and not hf_token.strip():
            st.error(
                "⚠️ Token HuggingFace manquant. "
                "Ajoutez-le dans la barre latérale ou créez un fichier `.env` "
                "avec `HF_TOKEN=votre_token`."
            )
            st.stop()

        # Mémoriser le prompt
        st.session_state.prompt_value = prompt

        with st.status("🔄 Traitement en cours…", expanded=True) as status:
            try:
                # ----------------------------------------------------------
                # ÉTAPE 1 : Suppression du fond
                # ----------------------------------------------------------
                st.write(f"**Étape 1/3** 🎭 — Suppression du fond avec **{rembg_model_label}**…")
                fg_image = remove_background(
                    original_image,
                    model=rembg_model,
                    alpha_matting=alpha_matting,
                    smooth_edges=(smooth_radius > 0),
                    smooth_radius=smooth_radius,
                )
                st.write("✅ Fond supprimé avec succès !")

                # ----------------------------------------------------------
                # ÉTAPE 2 : Génération du fond
                # ----------------------------------------------------------
                mode_labels = {
                    "🤗 HuggingFace API": f"HuggingFace API — `{model_choice}`",
                    "💻 Local (diffusers)": f"diffusers local — `{model_choice}`",
                    "🎨 Démo (sans IA)": "Dégradé de démonstration",
                }
                st.write(f"**Étape 2/3** 🎨 — Génération via {mode_labels[generation_mode]}…")
                st.write(f'📝 Prompt : *"{prompt}"*')

                if generation_mode == "🤗 HuggingFace API":
                    bg_image = generate_background_hf_api(
                        prompt=prompt,
                        width=original_image.width,
                        height=original_image.height,
                        hf_token=hf_token,
                        model=model_choice,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_steps,
                    )

                elif generation_mode == "💻 Local (diffusers)":
                    bg_image = generate_background_local(
                        prompt=prompt,
                        width=original_image.width,
                        height=original_image.height,
                        model=model_choice,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_steps,
                    )

                else:
                    bg_image = generate_background_demo(
                        prompt=prompt,
                        width=original_image.width,
                        height=original_image.height,
                    )

                st.write("✅ Fond généré avec succès !")

                # ----------------------------------------------------------
                # ÉTAPE 3 : Assemblage
                # ----------------------------------------------------------
                st.write("**Étape 3/3** 🖼️ — Assemblage final (alpha composite)…")
                final_image = composite_images(fg_image, bg_image)
                st.write("✅ Image finale prête !")

                # Sauvegarder dans le session state
                st.session_state.result_image = final_image
                st.session_state.fg_image = fg_image
                st.session_state.bg_image = bg_image

                status.update(label="✅ Traitement terminé avec succès !", state="complete")

            except Exception as e:
                status.update(label="❌ Erreur lors du traitement", state="error")
                error_msg = str(e)
                st.error(f"**Erreur :** {error_msg}")

                # Messages d'aide contextuels
                if "401" in error_msg or "invalide" in error_msg.lower() or "unauthorized" in error_msg.lower():
                    st.info(
                        "💡 **Token invalide (401).** Vérifiez votre token sur "
                        "[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). "
                        "Assurez-vous qu'il a les droits **read**."
                    )
                elif "500" in error_msg or "Internal server" in error_msg:
                    st.info(
                        "💡 **Erreur serveur (500)** : Le modèle sélectionné n'est plus disponible "
                        "via l'API gratuite HF. **Solution** : choisissez *Stable Diffusion 2.1* "
                        "ou *SD 1.5* dans la liste des modèles (barre latérale)."
                    )
                elif "chargement" in error_msg.lower() or "503" in error_msg or "loading" in error_msg.lower():
                    st.info(
                        "💡 **Modèle en cours de chargement (503).** "
                        "Attendez 30 secondes et réessayez."
                    )
                elif "429" in error_msg or "rate" in error_msg.lower():
                    st.info(
                        "💡 **Limite de requêtes atteinte (429).** Attendez quelques minutes."
                    )
                elif "cuda" in error_msg.lower() or "memory" in error_msg.lower():
                    st.info(
                        "💡 **Mémoire GPU insuffisante.** Passez en mode HuggingFace API."
                    )
                elif "Together" in error_msg or "router.huggingface" in error_msg:
                    st.info(
                        "💡 **Provider payant détecté.** Le code utilise maintenant `provider='hf-inference'` "
                        "pour forcer les serveurs gratuits. Rechargez l'application."
                    )
                st.stop()

        # Forcer le rechargement pour afficher le résultat
        st.rerun()

# =============================================================================
# Affichage des résultats détaillés (après génération)
# =============================================================================

if st.session_state.result_image is not None and uploaded_file is not None:

    st.divider()
    st.subheader("📊 Étapes détaillées du pipeline")

    step1, step2, step3 = st.columns(3)

    with step1:
        st.markdown("**1️⃣ Image originale**")
        st.image(original_image, use_container_width=True)
        st.caption("Input utilisateur")

    with step2:
        st.markdown(f"**2️⃣ Fond supprimé ({rembg_model_label})**")
        if st.session_state.fg_image:
            st.image(st.session_state.fg_image, use_container_width=True)
        st.caption(f"rembg · {rembg_model} · HuggingFace")

    with step3:
        st.markdown("**3️⃣ Résultat final**")
        st.image(st.session_state.result_image, use_container_width=True)
        st.caption("Stable Diffusion + alpha composite")

    # --- Boutons d'action ---
    st.divider()
    buf = io.BytesIO()
    st.session_state.result_image.save(buf, format="PNG")

    action_col1, action_col2 = st.columns([2, 1])
    with action_col1:
        st.download_button(
            label="⬇️ Télécharger l'image résultante (PNG)",
            data=buf.getvalue(),
            file_name="ai_photo_studio_result.png",
            mime="image/png",
            use_container_width=True,
            type="primary",
        )
    with action_col2:
        if st.button("🔄 Recommencer", use_container_width=True):
            st.session_state.result_image = None
            st.session_state.fg_image = None
            st.session_state.bg_image = None
            st.session_state.prompt_value = ""
            st.rerun()

# =============================================================================
# Page d'accueil (aucune image uploadée)
# =============================================================================

elif uploaded_file is None:
    st.divider()
    st.markdown(
        """
        ### 🚀 Comment utiliser AI Photo Studio ?

        | Étape | Action | Technologie |
        |-------|--------|-------------|
        | **1** | Uploadez une photo (portrait, objet, animal…) | PIL / Pillow |
        | **2** | Le fond est automatiquement détecté et supprimé | rembg (BiRefNet) · HuggingFace |
        | **3** | Décrivez le nouveau fond en quelques mots | Prompt texte libre |
        | **4** | L'IA génère et applique le fond | Stable Diffusion · HuggingFace API |
        | **5** | Téléchargez votre photo retouchée | PIL alpha_composite |
        """
    )

    st.info(
        "💡 **Pour démarrer** : importez une image dans la zone de dépôt ci-dessus. "
        "Pour la génération IA, ajoutez votre token HuggingFace (gratuit) dans la barre latérale."
    )
