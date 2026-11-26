"""Populate the local ChromaDB knowledge store from data/nutrition_45_foods.csv."""

import os
from pathlib import Path

import chromadb
import pandas as pd
from google import genai


PROJECT_DIR = Path(__file__).resolve().parent.parent
NUTRITION_CSV = PROJECT_DIR / "data" / "nutrition_45_foods.csv"
CHROMA_DIR = PROJECT_DIR / "data" / "chroma_db"
SECRETS_PATH = PROJECT_DIR / ".streamlit" / "secrets.toml"
COLLECTION_NAME = "nutrivision_food_knowledge"
EMBEDDING_MODEL = "gemini-embedding-001"


def load_api_key() -> str:
    """Resolve GEMINI_API_KEY from environment variables or Streamlit secrets."""
    env_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if env_key:
        return env_key

    if SECRETS_PATH.is_file():
        try:
            import tomllib

            with open(SECRETS_PATH, "rb") as f:
                secrets = tomllib.load(f)
            secret_key = str(secrets.get("GEMINI_API_KEY", "")).strip()
            if secret_key:
                return secret_key
        except ModuleNotFoundError:
            for line in SECRETS_PATH.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped.startswith("GEMINI_API_KEY") and "=" in stripped:
                    _, raw_value = stripped.split("=", 1)
                    secret_key = raw_value.strip().strip('"').strip("'")
                    if secret_key:
                        return secret_key

    raise ValueError(
        "GEMINI_API_KEY not found. Set the GEMINI_API_KEY environment variable "
        f"or define it in {SECRETS_PATH}."
    )


def build_records(nutrition_df: pd.DataFrame) -> tuple[list[str], list[str], list[dict]]:
    """Convert nutrition table rows into ChromaDB IDs, documents, and metadata."""
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []

    for _, row in nutrition_df.iterrows():
        food = str(row["food"])
        document = (
            f"Food: {food.replace('_', ' ')}. "
            f"Description: {row['description']}. "
            f"Stored calories: {row['calories_kcal']} kcal. "
            f"Protein: {row['protein_g']} g. "
            f"Carbohydrates: {row['carbs_g']} g. "
            f"Fat: {row['fat_g']} g. "
            f"Fiber: {row['fiber_g']} g. "
            f"Sugar: {row['sugar_g']} g. "
            f"Typical serving size: {row['typical_serving_g']} g. "
            "All nutritional values are estimates from the project dataset. "
            "The measurement basis is unverified; do not assume values are "
            "per 100 g or scale them by weight."
        )

        ids.append(food)
        documents.append(document)
        metadatas.append({
            "food": food,
            "calories_kcal": float(row["calories_kcal"]),
            "protein_g": float(row["protein_g"]),
            "carbs_g": float(row["carbs_g"]),
            "fat_g": float(row["fat_g"]),
            "fiber_g": float(row["fiber_g"]),
            "sugar_g": float(row["sugar_g"]),
            "typical_serving_g": float(row["typical_serving_g"]),
            "data_confidence": str(row["confidence"]),
            "measurement_basis": "unverified",
        })

    return ids, documents, metadatas


def main() -> None:
    if not NUTRITION_CSV.is_file():
        raise FileNotFoundError(f"Nutrition CSV not found: {NUTRITION_CSV}")

    api_key = load_api_key()
    nutrition_df = pd.read_csv(NUTRITION_CSV)
    ids, documents, metadatas = build_records(nutrition_df)

    client = genai.Client(api_key=api_key)
    embedding_response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=documents,
        config={"task_type": "RETRIEVAL_DOCUMENT"},
    )
    embeddings = [item.values for item in embedding_response.embeddings]

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Food and nutrition knowledge for NutriVision AI"},
    )
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    print(f"Indexed {len(ids)} food records into {CHROMA_DIR}")
    print(f"Collection '{COLLECTION_NAME}' count: {collection.count()}")


if __name__ == "__main__":
    main()
