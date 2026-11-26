# NutriVision AI

**LLM-Powered Food Recognition & Nutritional Analysis**

NutriVision AI is an experimental computer vision and retrieval-augmented generation (RAG) application for classifying food images across 45 categories, displaying dataset-backed nutritional estimates, visualizing model attention with Grad-CAM, and answering food and nutrition questions through a Google Gemini assistant backed by ChromaDB.

> **Project Status:** Research and demonstration prototype. This repository is not intended for production or clinical/medical deployment.

## Features

- **45-Class Food Classification:** Classifies a single uploaded food image (`jpg`, `jpeg`, `png`, `webp`) across 45 food categories using a 50/50 soft-voting ensemble of ConvNeXt Base and Swin Transformer Base.
- **Top-5 Predictions:** Displays the top predicted food class, ensemble confidence score, and a ranked top-5 probability table.
- **Nutritional Information Lookup:** Retrieves estimated calories, protein, carbohydrates, fat, fiber, sugar, typical serving size, and dish description from `data/nutrition_45_foods.csv`.
- **Grad-CAM Visual Explanations:** Generates side-by-side Grad-CAM heatmaps from the final feature blocks of both ConvNeXt Base and Swin Transformer Base for the top predicted class.
- **Gemini + ChromaDB RAG Assistant:** Embeds user questions with `gemini-embedding-001`, retrieves up to 5 relevant documents from a persistent ChromaDB collection (`nutrivision_food_knowledge`), and generates grounded responses using `gemini-2.5-flash`.
- **Streamlit Web Interface:** Provides three interactive tabs (`Food Recognition`, `Nutrition Assistant`, and `About`) with automatic CUDA GPU acceleration when available and CPU fallback.

## Architecture

NutriVision AI combines two pipelines inside `app.py`:

1. **Vision & Explainability Pipeline**
   - Input images are converted to RGB, resized to `224 × 224`, converted to tensors, and normalized using ImageNet channel statistics (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).
   - Both `torchvision.models.convnext_base` and `torchvision.models.swin_b` (configured with 45-class linear heads) produce softmax class probability distributions under `torch.inference_mode()`.
   - Probabilities are combined via weighted soft voting (`0.5` ConvNeXt Base + `0.5` Swin Transformer Base, configured in `ensemble_config.json`).
   - For explainability, forward and backward hooks on `model.features[-1]` compute Grad-CAM maps (with NHWC-to-NCHW permutation applied for TorchVision Swin Transformer feature maps), bilinearly upsampled to `224 × 224` and overlaid on the resized image.

2. **Nutrition Lookup & RAG Pipeline**
   - Exact case-insensitive class lookup against `data/nutrition_45_foods.csv` populates the nutrition metrics panel for the top-1 predicted food.
   - In the `Nutrition Assistant` tab, user queries are embedded via Google Gemini (`gemini-embedding-001`, task type `RETRIEVAL_QUERY`) and matched against the persistent ChromaDB collection `nutrivision_food_knowledge` at `data/chroma_db`.
   - Retrieved context is passed to `gemini-2.5-flash` with prompt constraints instructing the model to treat nutrition values as unverified estimates, avoid weight-based scaling, and state when context is insufficient.

## Dataset

The classification models were trained and evaluated on a 45-class food image dataset (predominantly Indian dishes, mapped in `class_mapping.csv` and `ensemble_config.json`). The raw image files are not included in this repository; split manifests are tracked in `train.csv`, `validation.csv`, and `test.csv`.

| Split | Images | Percentage | Manifest File |
|---|---:|---:|---|
| Training | 1,575 | 70.0% | `train.csv` |
| Validation | 337 | 15.0% | `validation.csv` |
| Test | 338 | 15.0% | `test.csv` |
| **Total** | **2,250** | **100.0%** | — |

## Model Comparison & Evaluation Results

All metrics below are reported from the tracked evaluation artifacts (`final_evaluation_report.json`, `final_test_model_comparison.csv`, `validation_model_comparison.csv`, `ensemble_validation_comparison.csv`, and `ensemble_weight_comparison.csv`).

> **Verification Note:** These metrics reflect recorded evaluation outputs dated **October 27, 2026**. Because this repository does not include a standalone CLI evaluation script or raw test images, these results have not been independently reproduced via an automated test harness.

### Validation Comparison (337 Images)

| Model | Accuracy (%) | Macro Precision (%) | Macro Recall (%) | Macro F1 (%) | Weighted F1 (%) | Inference (ms/image) |
|---|---:|---:|---:|---:|---:|---:|
| ConvNeXt Base | 83.09 | 84.59 | 83.13 | 83.00 | 83.01 | 37.29 |
| Swin Transformer Base | 83.09 | 85.03 | 83.25 | 82.95 | 82.89 | 26.36 |
| **50/50 Soft-Voting Ensemble** | **85.16** | — | — | **85.08** | — | — |

### Ensemble Weight Sweep on Validation (337 Images)

| ConvNeXt Weight (%) | Swin Weight (%) | Validation Accuracy (%) | Validation Macro F1 (%) |
|---:|---:|---:|---:|
| 30 | 70 | 83.68 | 83.73 |
| **50** | **50** | **85.16** | **85.08** |
| 70 | 30 | 83.98 | 83.95 |

### Final Test Results (338 Images, Evaluated October 27, 2026)

| Model | Test Accuracy (%) | Test Macro Precision (%) | Test Macro Recall (%) | Test Macro F1 (%) |
|---|---:|---:|---:|---:|
| ConvNeXt Base | 81.36 | 83.13 | 81.43 | 81.06 |
| Swin Transformer Base | 82.84 | 84.71 | 82.86 | 82.99 |
| **50/50 Soft-Voting Ensemble** | **84.91** | **86.11** | **84.88** | **84.82** |

- **Ensemble correct test predictions:** 287 / 338
- **Additional error analysis artifacts:** Per-class validation metrics (`per_class_validation_metrics.csv`, `ensemble_per_class_validation_metrics.csv`), top misclassification pairs (`convnext_base_top_confusions.csv`, `swin_base_top_confusions.csv`), and training/confusion plots (`plots/`).

## Repository Structure

```text
.
├── app.py                                   # Streamlit web application (inference, Grad-CAM, nutrition lookup, RAG)
├── ensemble_config.json                     # Class list, checkpoint paths, soft-voting weights, image size
├── requirements.txt                         # Python package dependencies
├── data/
│   └── nutrition_45_foods.csv               # 45-class nutritional reference table
├── class_mapping.csv                        # Class name to integer index mapping (45 classes)
├── train.csv                                # Training split manifest (1,575 entries)
├── validation.csv                           # Validation split manifest (337 entries)
├── test.csv                                 # Test split manifest (338 entries)
├── final_evaluation_report.json             # Final summary metrics and evaluation metadata
├── final_test_model_comparison.csv          # Test-set model comparison metrics
├── validation_model_comparison.csv          # Validation single-model metrics and latency
├── ensemble_validation_comparison.csv       # Validation single-model vs. ensemble metrics
├── ensemble_weight_comparison.csv           # Validation soft-voting weight sweep results
├── per_class_validation_metrics.csv         # Per-class validation metrics for ConvNeXt and Swin
├── ensemble_per_class_validation_metrics.csv # Per-class validation metrics for the 50/50 ensemble
├── convnext_base_top_confusions.csv         # Most frequent ConvNeXt Base validation confusions
├── swin_base_top_confusions.csv           # Most frequent Swin Transformer Base validation confusions
└── plots/                                   # Training curves, confusion matrices, and experiment notebook
```

## External Prerequisites

Several large or secret runtime assets are intentionally excluded from Git history (see `.gitignore`) and must be provisioned before running the application:

1. **Model Checkpoints (Required for Food Recognition & Grad-CAM)**
   - Pretrained fine-tuned weights are distributed via the repository's **GitHub Releases (`v1.0.0`)** rather than tracked in Git.
   - Expected default paths (defined in `ensemble_config.json`):
     - `models/best_convnext_base.pth`
     - `models/best_swin_base.pth`
   - `app.py` loads checkpoints using `torch.load(..., map_location="cpu", weights_only=True)` and supports raw `state_dict` files as well as dictionaries containing `"model_state_dict"` or `"state_dict"`.
   - Alternate checkpoint paths can be specified by updating `ensemble_config.json`.

2. **Google Gemini API Key (Required for Nutrition Assistant)**
   - A valid Google Gemini API key is required for `gemini-embedding-001` and `gemini-2.5-flash`.

3. **Populated ChromaDB Vector Store (Required for Meaningful RAG Retrieval)**
   - `app.py` connects to a persistent ChromaDB directory at `data/chroma_db` and opens collection `nutrivision_food_knowledge`.
   - If the directory is empty or absent, ChromaDB initializes an empty collection and the assistant will report that the food knowledge database is empty.
   - Documents stored in `nutrivision_food_knowledge` must be embedded using `gemini-embedding-001` to match the query embedding space used in `app.py`.

4. **Raw Image Dataset (Optional — Only for Retraining or Re-Evaluation)**
   - The 2,250 raw images referenced by `train.csv`, `validation.csv`, and `test.csv` are external to this repository.

## Installation

```bash
git clone https://github.com/charan845/NutriVision_Ai.git
cd NutriVision_Ai

python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

> **PyTorch / CUDA Note:** `requirements.txt` installs standard `torch` and `torchvision`. For GPU acceleration, install the CUDA-enabled PyTorch build matching your system's CUDA runtime if required.

## Configuration

1. **Provision model checkpoints:** Download `best_convnext_base.pth` and `best_swin_base.pth` from the repository's `v1.0.0` GitHub Release and place them at:
   - `models/best_convnext_base.pth`
   - `models/best_swin_base.pth`

2. **Configure Streamlit secrets:** Create `.streamlit/secrets.toml` (excluded by `.gitignore`) with a Google Gemini API key:

```toml
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
```

3. **Verify nutrition data and ChromaDB:** Confirm `data/nutrition_45_foods.csv` exists and populate `data/chroma_db` (`nutrivision_food_knowledge` collection) before using the RAG assistant.

## Usage

Launch the Streamlit application from the repository root:

```bash
streamlit run app.py
```

- **Food Recognition tab:** Upload a food image (`jpg`, `jpeg`, `png`, or `webp`) to view the top-1 ensemble prediction, top-5 confidence breakdown, nutrition estimates, and side-by-side ConvNeXt Base / Swin Transformer Base Grad-CAM overlays.
- **Nutrition Assistant tab:** Submit natural-language food or nutrition questions answered via ChromaDB retrieval and Google Gemini.
- **About tab:** Displays summary project metadata and validation ensemble metrics.

## Known Limitations & Nutrition Disclaimer

- **Single-class classification only:** No YOLO or object-detection model is implemented; the application does not perform multi-food detection, bounding-box localization, or meal tracking.
- **No portion or weight estimation:** The system does not estimate serving size or food weight from images and does not scale nutritional values by weight.
- **Unverified nutritional measurement basis:** Values in `data/nutrition_45_foods.csv` are project estimates whose measurement basis has not been independently verified (values are not assumed to be per 100 g).
- **Not medical advice:** Outputs are for informational and educational exploration only and must not be used for medical, dietary, or clinical decisions.
- **RAG dependency:** The Nutrition Assistant requires both a valid `GEMINI_API_KEY` and an externally populated `nutrivision_food_knowledge` ChromaDB collection.
- **No standalone training/evaluation CLI scripts:** Model training and evaluation workflows are not packaged as standalone `.py` scripts in this repository.

## Contributing

Contributions are welcome for documentation improvements, bug fixes, reproducibility scripts (such as standalone ChromaDB indexing or evaluation scripts), and model or UI enhancements.

1. Fork the repository and create a feature branch.
2. Keep pull requests focused and well-documented.
3. Do not commit API keys, `.streamlit/secrets.toml`, large binary checkpoints (`.pth`, `.pt`, `.ckpt`), or local vector databases (`chroma_db/`).
4. Open a pull request describing the motivation, changes made, and verification steps.

## License

No `LICENSE` file is currently included in this repository. Until a license is added, standard copyright defaults (all rights reserved) apply.
