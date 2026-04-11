# app.py
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import tempfile
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="StoneX — Stone Family Search",
    page_icon="🪨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@700;800&family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }
.main-title {
    font-family: 'Syne', sans-serif; font-size: 2.6rem; font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #c9a96e 0%, #f0d5a0 50%, #c9a96e 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.subtitle { color: #888; font-size: 0.95rem; margin-bottom: 1.5rem; letter-spacing: 0.02em; }
.layer-badge {
    display: inline-block; padding: 3px 10px; border-radius: 20px;
    font-size: 0.75rem; font-family: 'DM Mono', monospace; font-weight: 500;
    margin-right: 4px; letter-spacing: 0.05em;
}
.badge-color     { background: #2d1b69; color: #c4a8ff; border: 1px solid #4a2f9e; }
.badge-embedding { background: #0d2d1b; color: #6fffc4; border: 1px solid #1a6040; }
.badge-model     { background: #2d1500; color: #ffa86f; border: 1px solid #8a3e00; }
.family-card {
    background: #1a1a1a; border: 1px solid #2e2e2e; border-radius: 10px;
    padding: 12px 16px; margin-bottom: 8px; display: flex; align-items: center; gap: 12px;
}
.rank-num    { font-family: 'DM Mono', monospace; font-size: 0.85rem; color: #555; min-width: 24px; }
.family-name { font-weight: 600; font-size: 0.95rem; flex: 1; }
.score-num   { font-family: 'DM Mono', monospace; font-size: 0.8rem; color: #888; }
.section-header {
    font-family: 'Syne', sans-serif; font-size: 1.1rem; font-weight: 700;
    margin: 1.2rem 0 0.6rem 0; color: #e0e0e0;
}
.img-label {
    font-family: 'DM Mono', monospace; font-size: 0.7rem; color: #888;
    padding: 4px 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.pipeline-flow {
    background: #111; border: 1px solid #2a2a2a; border-radius: 12px;
    padding: 14px 18px; margin: 10px 0;
    font-family: 'DM Mono', monospace; font-size: 0.85rem; color: #aaa; letter-spacing: 0.03em;
}
.info-box {
    background: #111827; border-left: 3px solid #c9a96e; border-radius: 4px;
    padding: 10px 14px; font-size: 0.85rem; color: #9ca3af; margin-bottom: 12px;
}
div[data-testid="stSidebar"] { background: #0f0f0f; }
.sortable-item {
    background: #1e1e1e !important; border: 1px solid #333 !important;
    border-radius: 8px !important; color: #e0e0e0 !important;
    font-family: 'DM Mono', monospace !important; font-size: 0.88rem !important;
    padding: 10px 14px !important; cursor: grab !important;
    transition: background 0.15s, border-color 0.15s !important;
}
.sortable-item:hover { background: #2a2a2a !important; border-color: #c9a96e !important; }
.sortable-item:active { cursor: grabbing !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
LAYER_LABELS = {
    "color":     ("🎨 Color",     "badge-color"),
    "embedding": ("🔮 Embedding", "badge-embedding"),
    "model":     ("🧠 Model",     "badge-model"),
}
LAYER_DESCRIPTIONS = {
    "color":     "LAB histogram + KMeans + Gabor vein/base separation. Fast, CPU-only, ChromaDB.",
    "embedding": "DINOv2 fine-tuned embeddings in ChromaDB (cosine similarity). Rich visual semantics.",
    "model":     "DINOv2 + FAISS reranker with fine-tuned ViT-small weights. Most precise.",
}
ALL_LAYERS = ["color", "embedding", "model"]
LAYER_ICONS = {"color": "🎨 Color", "embedding": "🔮 Embedding", "model": "🧠 Model"}
PIPELINE_PRESETS = {
    # ── All 6 three-layer permutations ────────────────────────────────────────
    "Color → Embedding → Model":  ["color", "embedding", "model"],
    "Color → Model → Embedding":  ["color", "model", "embedding"],
    "Embedding → Color → Model":  ["embedding", "color", "model"],
    "Embedding → Model → Color":  ["embedding", "model", "color"],
    "Model → Color → Embedding":  ["model", "color", "embedding"],
    "Model → Embedding → Color":  ["model", "embedding", "color"],
    # ── All 6 two-layer permutations ──────────────────────────────────────────
    "Color → Embedding":          ["color", "embedding"],
    "Color → Model":              ["color", "model"],
    "Embedding → Color":          ["embedding", "color"],
    "Embedding → Model":          ["embedding", "model"],
    "Model → Color":              ["model", "color"],
    "Model → Embedding":          ["model", "embedding"],
    # ── All 3 single-layer options ────────────────────────────────────────────
    "Color only":                 ["color"],
    "Embedding only":             ["embedding"],
    "Model only":                 ["model"],
}

def badge(layer):
    label, cls = LAYER_LABELS[layer]
    return f'<span class="layer-badge {cls}">{label}</span>'

def flow_badges(order):
    return " → ".join(
        f'<span class="layer-badge {LAYER_LABELS[l][1]}">{LAYER_LABELS[l][0]}</span>'
        for l in order
    )

# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="main-title">🪨 StoneX</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Multi-layer stone family search</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚙️ Pipeline Configuration")

    # ── Mode toggle ────────────────────────────────────────────────────────────
    config_mode = st.radio(
        "Configuration mode",
        options=["📋 Presets", "✋ Drag & Drop"],
        horizontal=True,
        label_visibility="collapsed",
    )

    layer_order = ["color", "embedding", "model"]  # safe default

    # ── MODE 1: PRESETS ───────────────────────────────────────────────────────
    if config_mode == "📋 Presets":
        st.caption("Choose a predefined pipeline order:")
        preset_choice = st.selectbox(
            "Pipeline preset",
            options=list(PIPELINE_PRESETS.keys()),
            index=0,
            label_visibility="collapsed",
        )
        layer_order = PIPELINE_PRESETS[preset_choice]
        st.markdown(
            f'<div class="pipeline-flow">{flow_badges(layer_order)}</div>',
            unsafe_allow_html=True,
        )

    # ── MODE 2: DRAG AND DROP ─────────────────────────────────────────────────
    else:
        st.caption("Toggle layers, then drag to reorder:")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            use_color = st.checkbox("🎨 Color", value=True, key="cb_color")
        with col_b:
            use_embedding = st.checkbox("🔮 Embed", value=True, key="cb_embed")
        with col_c:
            use_model = st.checkbox("🧠 Model", value=True, key="cb_model")

        enabled_layers = [
            l for l, on in zip(
                ["color", "embedding", "model"],
                [use_color, use_embedding, use_model]
            ) if on
        ]
        if not enabled_layers:
            st.warning("Enable at least one layer.")
            enabled_layers = ["color"]

        from streamlit_sortables import sort_items
        sorted_result = sort_items(
            [LAYER_ICONS[l] for l in enabled_layers],
            direction="vertical",
            key="layer_sort",
        )
        icon_to_key = {v: k for k, v in LAYER_ICONS.items()}
        layer_order = [icon_to_key[lbl] for lbl in sorted_result if lbl in icon_to_key]
        if not layer_order:
            layer_order = enabled_layers

        st.markdown(
            f'<div class="pipeline-flow">{flow_badges(layer_order)}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### 🎛️ Result controls")
    top_k_families = st.slider("Top-K families", 1, 30, 10, 1,
        help="Number of stone families to return.")
    top_k_images = st.slider("Top-K images per layer", 4, 40, 12, 4,
        help="Matched DB images shown per layer tab.")
    first_layer_fetch = st.slider("First-layer candidate pool", 10, 100, 40, 5,
        help="Broad retrieval count before reranking.")

    st.markdown("---")
    st.markdown("### 👁️ Display options")
    show_layer_families = st.checkbox("Show per-layer family scores", value=False)
    show_layer_images   = st.checkbox("Show matched images per layer", value=True)
    img_cols            = st.slider("Image grid columns", 2, 6, 4, 1)

    st.markdown("---")
    st.markdown("### 📚 Layer info")
    for layer in ALL_LAYERS:
        label, cls = LAYER_LABELS[layer]
        st.markdown(
            f'<div class="info-box">'
            f'<span class="layer-badge {cls}">{label}</span><br>'
            f'<span style="font-size:0.82rem">{LAYER_DESCRIPTIONS[layer]}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### 🗄️ Ingest dataset")
    dataset_path    = st.text_input("Dataset folder path", placeholder="/path/to/stone/dataset")
    do_color_ingest = st.checkbox("Ingest colour vectors",    value=True, key="ing_color")
    do_embed_ingest = st.checkbox("Ingest DINOv2 embeddings", value=True, key="ing_embed")

    if st.button("▶ Run ingestion", use_container_width=True):
        if not dataset_path:
            st.error("Please provide a dataset folder path.")
        else:
            from ingestion.ingest_dataset import ingest_dataset
            with st.spinner("Ingesting dataset…"):
                try:
                    ingest_dataset(dataset_path, do_color=do_color_ingest, do_embedding=do_embed_ingest)
                    st.success("✅ Ingestion complete!")
                except Exception as e:
                    st.error(f"Ingestion error: {e}")

# ═════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="main-title">🪨 StoneX</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="subtitle">Pipeline: {" → ".join(badge(l) for l in layer_order)}</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload a stone image", type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed",
)

if uploaded_file is None:
    st.markdown("""
    <div style="background:#111;border:2px dashed #2e2e2e;border-radius:16px;
        padding:60px 40px;text-align:center;margin-top:24px;">
        <div style="font-size:3rem">🪨</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.3rem;color:#555;margin-top:12px;">
            Drop a stone image to begin search
        </div>
        <div style="font-size:0.85rem;color:#333;margin-top:8px;">Supports JPG, PNG, WEBP</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

image = Image.open(uploaded_file).convert("RGB")
with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
    image.save(tmp.name, format="JPEG")
    temp_path = tmp.name

col_img, col_results = st.columns([1, 2], gap="large")

with col_img:
    st.image(image, caption="Query image", use_column_width=True)
    st.markdown(
        f'<div style="font-size:0.8rem;color:#555;font-family:DM Mono,monospace">'
        f'{image.size[0]}×{image.size[1]}px</div>',
        unsafe_allow_html=True,
    )

with col_results:
    st.markdown('<div class="section-header">🔍 Running pipeline…</div>', unsafe_allow_html=True)

    from query.pipeline import run_pipeline
    results = None
    error   = None

    with st.spinner(f"Running {' → '.join(layer_order)}…"):
        try:
            results = run_pipeline(
                temp_path,
                layer_order=layer_order,
                top_k_families=top_k_families,
                top_k_images=top_k_images,
                first_layer_fetch=first_layer_fetch,
            )
        except Exception as e:
            error = e

    try:
        os.remove(temp_path)
    except Exception:
        pass

    if error:
        st.error(f"Pipeline error: {error}")
        st.stop()

    if not results or not results.get("families"):
        st.warning("No results found. Make sure the collections are ingested.")
        st.stop()

    st.markdown(
        f'<div class="section-header">🏆 Top {top_k_families} Stone Families</div>',
        unsafe_allow_html=True,
    )

    for rank, (family, score) in enumerate(results["families"], 1):
        display = family.replace("_", " ")
        bar_val = max(0.0, min(1.0, float(score)))
        st.markdown(
            f'<div class="family-card">'
            f'<span class="rank-num">#{rank}</span>'
            f'<span class="family-name">{display}</span>'
            f'<span class="score-num">{score:.4f}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.progress(bar_val)

    if show_layer_families:
        st.markdown("---")
        st.markdown('<div class="section-header">📊 Per-layer family scores</div>', unsafe_allow_html=True)
        tabs = st.tabs([LAYER_LABELS[l][0] for l in layer_order])
        for tab, layer in zip(tabs, layer_order):
            with tab:
                layer_fams = results["layer_families"].get(layer, [])
                if not layer_fams:
                    st.caption("No results for this layer.")
                    continue
                for rank, (family, score) in enumerate(layer_fams[:top_k_families], 1):
                    bar_val = max(0.0, min(1.0, float(score)))
                    st.progress(bar_val, text=f"#{rank} {family.replace('_',' ')}  ({score:.4f})")

if show_layer_images:
    st.markdown("---")
    st.markdown('<div class="section-header">🖼️ Matched DB Images by Layer</div>', unsafe_allow_html=True)
    tabs = st.tabs([LAYER_LABELS[l][0] for l in layer_order])
    for tab, layer in zip(tabs, layer_order):
        with tab:
            imgs = results["images"].get(layer, [])
            if not imgs:
                st.caption("No images for this layer.")
                continue
            st.caption(f"{len(imgs)} matched images from '{layer}' layer")
            cols = st.columns(img_cols)
            for i, (path, score) in enumerate(imgs):
                col = cols[i % img_cols]
                with col:
                    # ── Resolve family name from path ──────────────────────
                    path_parts = str(path).replace("\\", "/").split("/")
                    family_name = path_parts[-2].replace("_", " ") if len(path_parts) >= 2 else "unknown"
                    file_name   = path_parts[-1] if path_parts else str(path)

                    # ── Try to load image ──────────────────────────────────
                    img_loaded = False
                    try:
                        col.image(Image.open(path).convert("RGB"), use_column_width=True)
                        img_loaded = True
                    except Exception:
                        col.markdown(
                            f'''<div style="
                                background:#1a0a0a;
                                border:1px dashed #5a2020;
                                border-radius:8px;
                                padding:14px 10px;
                                text-align:center;
                                min-height:100px;
                                display:flex;
                                flex-direction:column;
                                justify-content:center;
                                align-items:center;
                                gap:6px;
                            ">
                                <div style="font-size:1.4rem">🚫</div>
                                <div style="
                                    font-family:'DM Mono',monospace;
                                    font-size:0.65rem;
                                    color:#c06060;
                                    font-weight:600;
                                ">Image not found</div>
                                <div style="
                                    font-family:'DM Mono',monospace;
                                    font-size:0.62rem;
                                    color:#ff9966;
                                    font-weight:700;
                                    background:#2a1010;
                                    padding:2px 6px;
                                    border-radius:4px;
                                ">📁 {family_name}</div>
                                <div style="
                                    font-family:'DM Mono',monospace;
                                    font-size:0.58rem;
                                    color:#555;
                                    word-break:break-all;
                                    margin-top:4px;
                                ">{path}</div>
                            </div>''',
                            unsafe_allow_html=True,
                        )

                    # ── Caption below image (always shown) ────────────────
                    col.markdown(
                        f'<div class="img-label">'
                        f'🏷 <b style="color:#c9a96e">{family_name}</b><br>'
                        f'📄 {file_name}<br>'
                        f'<span style="color:#c9a96e">score: {score:.4f}</span><br>'
                        f'<span style="color:#3a3a3a;font-size:0.6rem;word-break:break-all">{path}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )