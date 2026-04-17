import os
import tempfile
from typing import Dict

from fastapi import FastAPI, File, UploadFile, Header, HTTPException
from PIL import Image
from dotenv import load_dotenv

# ─────────────────────────────────────────────
# LOAD YOUR EXISTING PIPELINE + EMBEDDER
# ─────────────────────────────────────────────
from query.pipeline import run_pipeline
import features.dino_embedder as dino_embedder

# preload model once
dino_embedder.get_model()

load_dotenv()

app = FastAPI(title="StoneX API")

# 🔐 Your fixed token
API_TOKEN = os.getenv("API_KEY")

# ─────────────────────────────────────────────
# CMD OFFICE MAPPING
# ─────────────────────────────────────────────
from cmd_mapping import resolve_family_name, is_cmd_class


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def normalize_name(name: str) -> str:
    return name.lower().replace("_", " ").strip()


def get_best_image_from_results(family_name, model_images):
    """
    Find the best-scoring image for a given family from model results.
    Matches by the raw folder name (before CMD mapping) since paths
    on disk still use the original folder names.
    """
    best_path = None
    best_score = -1.0

    for path, score in model_images:
        parts = path.replace("\\", "/").split("/")
        fam = normalize_name(parts[-2]) if len(parts) >= 2 else ""

        # match against raw family name (folder name on disk)
        if fam == normalize_name(family_name):
            if score > best_score:
                best_path = path
                best_score = score

    return best_path, best_score


# ─────────────────────────────────────────────
# API ENDPOINT: /predict
# ─────────────────────────────────────────────

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    authorization: str = Header(None)
) -> Dict:
    temp_path = None

    try:
        # 🔐 Validate Bearer Token
        if authorization is None or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

        token = authorization.split(" ")[1]
        if token != API_TOKEN:
            raise HTTPException(status_code=403, detail="Invalid token")

        # save temp image
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            contents = await file.read()
            tmp.write(contents)
            temp_path = tmp.name

        # run pipeline
        results = run_pipeline(
            temp_path,
            layer_order=["model"],
            top_k_families=5,
            top_k_images=800,
            first_layer_fetch=60,
        )

        families     = results.get("families", [])[:5]
        model_images = results.get("images", {}).get("model", [])

        output = []

        for raw_family, fam_score in families:
            img_path, img_score = get_best_image_from_results(raw_family, model_images)

            # ── Apply CMD Office mapping ──────────────────────────────
            display_name = resolve_family_name(raw_family)
            cmd_flag     = is_cmd_class(raw_family)

            output.append({
                "family":        display_name,        # mapped stone name (or humanised original)
                "raw_class":     raw_family,          # original model class (for debugging)
                "is_cmd_class":  cmd_flag,            # True if this was a CMD Office internal class
                "family_score":  float(fam_score),
                "best_image":    img_path,
                "image_score":   float(img_score) if img_score is not None else None,
            })

        return {
            "status":  "success",
            "results": output,
        }

    except HTTPException as e:
        raise e

    except Exception as e:
        return {"status": "error", "message": str(e)}

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# ─────────────────────────────────────────────
# API ENDPOINT: /embedding
# ─────────────────────────────────────────────

from typing import List

@app.post("/embedding")
async def get_embedding(
    files: List[UploadFile] = File(...),
    authorization: str = Header(None)
) -> Dict:
    temp_paths = []

    try:
        # 🔐 Auth check
        if authorization is None or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

        token = authorization.split(" ")[1]
        if token != API_TOKEN:
            raise HTTPException(status_code=403, detail="Invalid token")

        embeddings = []

        # process multiple images
        for file in files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                contents = await file.read()
                tmp.write(contents)
                temp_path = tmp.name
                temp_paths.append(temp_path)

            emb = dino_embedder.embed_image(temp_path)

            if emb is None:
                continue

            embeddings.append({
                "filename":  file.filename,
                "embedding": emb.tolist(),
                "dimension": len(emb),
            })

        return {
            "status":  "success",
            "count":   len(embeddings),
            "results": embeddings,
        }

    except HTTPException as e:
        raise e

    except Exception as e:
        return {"status": "error", "message": str(e)}

    finally:
        for p in temp_paths:
            if os.path.exists(p):
                os.remove(p)