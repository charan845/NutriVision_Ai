
import os
import json
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import chromadb

from PIL import Image
from torchvision import models, transforms
from google import genai


PROJECT_DIR = Path("/content/drive/MyDrive/NutriVision_AI")
NUTRITION_CSV = Path("/content/drive/MyDrive/nutrition_45_foods.csv")
CHROMA_DIR = PROJECT_DIR / "chroma_db"
CONFIG_PATH = PROJECT_DIR / "ensemble_config.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(
    page_title="NutriVision AI",
    page_icon="🥗",
    layout="wide"
)

st.title("🥗 NutriVision AI")
st.caption(
    "Food recognition, nutrition information, and evidence-grounded "
    "AI assistance."
)


@st.cache_data
def load_project_data():
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    nutrition_df = pd.read_csv(NUTRITION_CSV)
    class_names = config["class_names"]

    return config, nutrition_df, class_names


@st.cache_resource
def load_models():
    config, _, class_names = load_project_data()

    convnext = models.convnext_base(weights=None)
    convnext.classifier[2] = torch.nn.Linear(
        convnext.classifier[2].in_features,
        len(class_names)
    )

    swin = models.swin_b(weights=None)
    swin.head = torch.nn.Linear(
        swin.head.in_features,
        len(class_names)
    )

    convnext_checkpoint = torch.load(
        config["selected_models"]["convnext"]["checkpoint"],
        map_location="cpu",
        weights_only=True
    )
    swin_checkpoint = torch.load(
        config["selected_models"]["swin"]["checkpoint"],
        map_location="cpu",
        weights_only=True
    )

    # Support the checkpoint format used by this project.
    # The trained weights are stored under "model_state_dict".
    if isinstance(convnext_checkpoint, dict):
        if "model_state_dict" in convnext_checkpoint:
            convnext_checkpoint = convnext_checkpoint["model_state_dict"]
        elif "state_dict" in convnext_checkpoint:
            convnext_checkpoint = convnext_checkpoint["state_dict"]

    if isinstance(swin_checkpoint, dict):
        if "model_state_dict" in swin_checkpoint:
            swin_checkpoint = swin_checkpoint["model_state_dict"]
        elif "state_dict" in swin_checkpoint:
            swin_checkpoint = swin_checkpoint["state_dict"]

    convnext.load_state_dict(convnext_checkpoint)
    swin.load_state_dict(swin_checkpoint)

    convnext = convnext.to(DEVICE).eval()
    swin = swin.to(DEVICE).eval()

    return convnext, swin


@st.cache_resource
def load_gemini_and_chroma():
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is missing from Streamlit secrets."
        )

    gemini_client = genai.Client(api_key=api_key)

    chroma_client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )
    collection = chroma_client.get_or_create_collection(
        name="nutrivision_food_knowledge"
    )

    return gemini_client, collection


@st.cache_data
def get_image_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def predict_image(image):
    config, nutrition_df, class_names = load_project_data()
    convnext, swin = load_models()

    image_tensor = get_image_transform()(image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        convnext_probs = torch.softmax(convnext(image_tensor), dim=1)
        swin_probs = torch.softmax(swin(image_tensor), dim=1)

        cw = config["selected_models"]["convnext"].get("weight", 0.5)
        sw = config["selected_models"]["swin"].get("weight", 0.5)
        total = cw + sw
        ensemble_probs = (cw / total) * convnext_probs + (sw / total) * swin_probs

    k = min(5, len(class_names))
    probs, indices = torch.topk(ensemble_probs[0], k=k)

    results = []
    for probability, index in zip(probs.cpu().tolist(), indices.cpu().tolist()):
        results.append({
            "Food": class_names[index],
            "Confidence (%)": round(probability * 100, 2)
        })

    return results


def get_nutrition(food_name):
    _, nutrition_df, _ = load_project_data()
    matches = nutrition_df[
        nutrition_df["food"].astype(str).str.lower() == food_name.lower()
    ]

    if matches.empty:
        return None

    return matches.iloc[0].to_dict()


def retrieve_food_context(question, top_k=5):
    gemini_client, collection = load_gemini_and_chroma()

    if collection.count() == 0:
        return []

    response = gemini_client.models.embed_content(
        model="gemini-embedding-001",
        contents=question,
        config={"task_type": "RETRIEVAL_QUERY"}
    )

    query_embedding = response.embeddings[0].values

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    records = []
    for document, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        records.append({
            "document": document,
            "food": metadata.get("food", "Unknown"),
            "distance": float(distance)
        })

    return records


def ask_nutrivision(question):
    gemini_client, _ = load_gemini_and_chroma()
    records = retrieve_food_context(question)

    if not records:
        return (
            "The food knowledge database is empty or no relevant "
            "information was retrieved."
        )

    context = "\n\n".join(record["document"] for record in records)

    prompt = f"""
You are NutriVision AI, a food recognition and nutrition assistant.

Use the retrieved context to answer the question.
- Treat project nutrition values as estimates.
- The measurement basis is unverified.
- Do not assume values are per 100 g.
- Do not scale values for a different food weight.
- Do not invent missing nutritional values.
- State when the retrieved context is insufficient.

RETRIEVED CONTEXT:
{context}

USER QUESTION:
{question}
"""

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text or "No answer was returned."


def generate_gradcam(model, image, target_class):
    model.eval()
    model.zero_grad(set_to_none=True)

    activations = {}
    gradients = {}
    target_layer = model.features[-1]

    def forward_hook(module, inputs, output):
        activations["value"] = output

    def backward_hook(module, grad_input, grad_output):
        gradients["value"] = grad_output[0]

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        tensor = get_image_transform()(image).unsqueeze(0).to(DEVICE)

        with torch.enable_grad():
            logits = model(tensor)
            logits[0, target_class].backward()

            feature_maps = activations["value"]
            feature_grads = gradients["value"]

            # TorchVision Swin uses NHWC feature layout.
            if isinstance(model, models.SwinTransformer):
                if feature_maps.ndim == 4 and feature_maps.shape[-1] == 1024:
                    feature_maps = feature_maps.permute(0, 3, 1, 2)
                    feature_grads = feature_grads.permute(0, 3, 1, 2)

            weights = feature_grads.mean(dim=(2, 3), keepdim=True)
            cam = F.relu((weights * feature_maps).sum(dim=1))

            cam = F.interpolate(
                cam.unsqueeze(1),
                size=(224, 224),
                mode="bilinear",
                align_corners=False
            ).squeeze()

            cam = cam.detach().cpu().numpy()
            cam_min, cam_max = cam.min(), cam.max()

            if cam_max > cam_min:
                cam = (cam - cam_min) / (cam_max - cam_min)
            else:
                cam = np.zeros_like(cam)

            return cam
    finally:
        forward_handle.remove()
        backward_handle.remove()
        model.zero_grad(set_to_none=True)


tab_predict, tab_assistant, tab_about = st.tabs([
    "🍲 Food Recognition",
    "🤖 Nutrition Assistant",
    "ℹ️ About"
])

with tab_predict:
    st.subheader("Identify a food")
    uploaded_file = st.file_uploader(
        "Upload a food image",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        col_image, col_result = st.columns([1, 1])

        with col_image:
            st.image(image, caption="Uploaded image", use_container_width=True)

        with col_result:
            with st.spinner("Running the ConvNeXt + Swin ensemble..."):
                try:
                    predictions = predict_image(image)
                    best = predictions[0]
                    st.success(f"Prediction: {best['Food']}")
                    st.metric("Ensemble confidence", f"{best['Confidence (%)']:.2f}%")

                    st.subheader("Top-5 predictions")
                    st.dataframe(
                        pd.DataFrame(predictions),
                        use_container_width=True,
                        hide_index=True
                    )

                    nutrition = get_nutrition(best["Food"])
                    if nutrition:
                        st.subheader("Nutrition information")
                        st.warning(
                            "These are estimates from the project dataset. "
                            "The measurement basis is unverified. Values are "
                            "not scaled to the uploaded food's weight."
                        )

                        nutrient_columns = st.columns(3)
                        fields = [
                            ("Calories", "calories_kcal", "kcal"),
                            ("Protein", "protein_g", "g"),
                            ("Carbohydrates", "carbs_g", "g"),
                            ("Fat", "fat_g", "g"),
                            ("Fiber", "fiber_g", "g"),
                            ("Sugar", "sugar_g", "g"),
                        ]

                        for i, (label, key, unit) in enumerate(fields):
                            value = nutrition.get(key)
                            with nutrient_columns[i % 3]:
                                if pd.notna(value):
                                    st.metric(label, f"{value} {unit}")

                        st.write(
                            "Typical serving size:",
                            nutrition.get("typical_serving_g", "Not available"),
                            "g"
                        )
                        st.write(
                            "Description:",
                            nutrition.get("description", "Not available")
                        )

                    st.subheader("Grad-CAM visual explanations")
                    config, _, class_names = load_project_data()
                    convnext, swin = load_models()
                    target_index = class_names.index(best["Food"])

                    with st.spinner("Generating Grad-CAM heatmaps..."):
                        cam1 = generate_gradcam(convnext, image, target_index)
                        cam2 = generate_gradcam(swin, image, target_index)

                    import matplotlib.pyplot as plt

                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    resized = image.resize((224, 224))

                    axes[0].imshow(resized)
                    axes[0].set_title("Original")
                    axes[0].axis("off")

                    axes[1].imshow(resized)
                    axes[1].imshow(cam1, cmap="jet", alpha=0.45)
                    axes[1].set_title("ConvNeXt Base Grad-CAM")
                    axes[1].axis("off")

                    axes[2].imshow(resized)
                    axes[2].imshow(cam2, cmap="jet", alpha=0.45)
                    axes[2].set_title("Swin Base Grad-CAM")
                    axes[2].axis("off")

                    st.pyplot(fig)
                    plt.close(fig)

                except Exception as error:
                    st.error(f"Prediction failed: {error}")
                    st.exception(error)

with tab_assistant:
    st.subheader("Ask the NutriVision AI assistant")
    question = st.text_area(
        "Your food or nutrition question",
        placeholder="Example: What nutrition information is available for bhindi masala?"
    )

    if st.button("Ask NutriVision AI"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Retrieving knowledge and generating an answer..."):
                try:
                    answer = ask_nutrivision(question)
                    st.markdown(answer)
                except Exception as error:
                    st.error(f"The assistant encountered an error: {error}")

with tab_about:
    st.subheader("About NutriVision AI")
    st.write(
        "NutriVision AI combines a ConvNeXt Base and Swin Transformer Base "
        "ensemble with nutrition lookup, Gemini-powered retrieval-augmented "
        "generation (RAG), and Grad-CAM visual explanations."
    )
    st.write("Food classes:", 45)
    st.write("Validation ensemble accuracy: 85.16%")
    st.write("Validation ensemble macro F1: 85.08%")
    st.caption(
        "Performance figures are validation results. Nutrition values are "
        "estimates, and their measurement basis is unverified."
    )
