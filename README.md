# NutriVision AI

**Computer Vision & RAG Prototype for Food Recognition and Nutritional Analysis**

NutriVision AI is an experimental machine learning and retrieval-augmented generation (RAG) web application built with PyTorch, TorchVision, Streamlit, ChromaDB, and Google Gemini. It classifies food images across 45 categories using a ConvNeXt Base + Swin Transformer Base ensemble, displays dataset-backed nutritional estimates, visualizes model attention with Grad-CAM, and answers food and nutrition questions grounded in a local vector store.

> **Project Status:** Research and demonstration prototype. This repository is not intended for production, clinical, medical, or dietary deployment.

## Features

- **45-Class Food Classification:** Classifies an uploaded image (`jpg`, `jpeg`, `png`, `webp`) across 45 food categories via a 50/50 soft-voting ensemble of ConvNeXt Base and Swin Transformer Base.
- **Top-5 Confidence Breakdown:** Reports the top-1 predicted class, ensemble confidence percentage, and a ranked top-5 probability table.
- **Dataset-Backed Nutrition Lookup:** Retrieves estimated calories, macronutrients (protein, carbohydrates, fat, fiber, sugar), typical serving size, and dish description for the top-1 prediction from `data/nutrition_45_foods.csv`.
- **Grad-CAM Visual Explanations:** Computes side-by-side Grad-CAM heatmaps from the final feature blocks of both ConvNeXt Base and Swin Transformer Base for the top-1 predicted class.
- **Gemini + ChromaDB Nutrition Assistant:** Embeds natural-language questions with `gemini-embedding-001`, retrieves up to 5 records from a persistent ChromaDB collection (`nutrivision_food_knowledge`), and generates grounded responses using `gemini-2.5-flash`.
- **Streamlit Web Interface:** Provides three tabs (`Food Recognition`, `Nutrition Assistant`, and `About`) with automatic CUDA GPU acceleration when available and CPU fallback. The `Food Recognition` tab operates independently of the Gemini API key and ChromaDB vector store.

## Architecture

NutriVision AI implements two runtime pipelines in `app.py`:

1. **Vision & Explainability Pipeline**
   - **Preprocessing (`get_image_transform`):** Converts uploaded images to RGB, resizes directly to `224 × 224` (`transforms.Resize((224, 224))`), converts to tensors, and normalizes using ImageNet statistics (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).
   - **Ensemble Inference (`predict_image`):** Evaluates `torchvision.models.convnext_base` and `torchvision.models.swin_b` (each modified with a 45-class linear classification head) under `torch.inference_mode()`. Softmax class probabilities are combined via normalized weighted soft voting (`0.5` ConvNeXt Base + `0.5` Swin Transformer Base, defined in `ensemble_config.json`).
   - **Grad-CAM (`generate_gradcam`):** Registers forward and full backward hooks on `model.features[-1]`. For TorchVision Swin Transformer feature maps in `NHWC` layout (`shape[-1] == 1024`), tensors are permuted to `NCHW` before global-average-pooling gradients, computing the ReLU-activated weighted sum, bilinearly interpolating to `224 × 224`, min-max normalizing, and overlaying with a `jet` colormap (`alpha=0.45`).

2. **Nutrition Lookup & RAG Pipeline**
   - **Tabular Lookup (`get_nutrition`):** Performs an exact case-insensitive match against `data/nutrition_45_foods.csv` for the top-1 predicted food class.
   - **Vector Retrieval (`retrieve_food_context`):** Embeds user questions via `gemini-embedding-001` (`task_type="RETRIEVAL_QUERY"`) and queries the persistent ChromaDB collection `nutrivision_food_knowledge` at `data/chroma_db` (`top_k=5`).
   - **Grounded Generation (`ask_nutrivision`):** Sends retrieved context to `gemini-2.5-flash` with prompt constraints requiring the model to treat nutrition values as estimates with an unverified measurement basis, avoid weight-based scaling or per-100 g assumptions, and state when retrieved context is insufficient.

## Dataset

The classification models were trained and evaluated on a 45-class food image dataset (predominantly Indian dishes, alongside items such as `crispy-chicken`, `fried_rice`, and `momos`), mapped in `data/class_mapping.csv` and `ensemble_config.json`. Raw image files are external to this repository; stratified 70% / 15% / 15% splits (`random_state=42`) are recorded under `data/splits/`.

| Split | Images | Per-Class Count | Percentage | Manifest File |
|---|---:|---:|---:|---|
| Training | 1,575 | 35 | 70.0% | `data/splits/train.csv` |
| Validation | 337 | 7–8 | 15.0% | `data/splits/validation.csv` |
| Test | 338 | 7–8 | 15.0% | `data/splits/test.csv` |
| **Total** | **2,250** | **50** | **100.0%** | — |

- `data/nutrition_45_foods.csv` contains 45 nutrition records matching every class in `data/class_mapping.csv`, with columns `food`, `calories_kcal`, `protein_g`, `carbs_g`, `fat_g`, `fiber_g`, `sugar_g`, `typical_serving_g`, `confidence`, and `description` (`confidence` is stored in the CSV and ChromaDB metadata but is not rendered in the `app.py` UI).
- `data/splits/train.csv`, `data/splits/validation.csv`, and `data/splits/test.csv` store file paths from the original Colab training environment and serve as split records rather than portable local paths.

## Training & Evaluation Methodology

Training and validation analysis were conducted in `notebooks/NutriVision_Ai.ipynb` on Google Colab using an NVIDIA Tesla T4 GPU (`PyTorch 2.11.0+cu130`, `Torchvision 0.26.0+cu130`, `SEED = 42`):

- **Model Initialization:** Fine-tuned ImageNet-pretrained `ConvNeXt_Base_Weights.DEFAULT` and `Swin_B_Weights.DEFAULT` with 45-class linear heads.
- **Training Augmentations & Optimization:** `RandomResizedCrop(224, scale=(0.75, 1.0))` and `RandomHorizontalFlip(p=0.5)`; `AdamW` (`lr=1e-4`, `weight_decay=0.05`), `CrossEntropyLoss`, `ReduceLROnPlateau` (`mode="max"`, `factor=0.2`, `patience=2`, `min_lr=1e-7`), batch size `8`, CUDA automatic mixed precision (`torch.autocast`), and early stopping (`patience=5`; ConvNeXt Base stopped after 11 epochs and Swin Transformer Base after 12 epochs, each reaching `83.09%` best validation accuracy).
- **Evaluation Preprocessing:** Offline validation and test evaluation in `notebooks/NutriVision_Ai.ipynb` used `transforms.Resize(256)` followed by `transforms.CenterCrop(224)` and ImageNet normalization (whereas `app.py` inference resizes uploaded images directly via `transforms.Resize((224, 224))`).
- **Verification Note:** All metrics below are preserved from the tracked repository artifacts (`results/final_evaluation_report.json` dated **October 27, 2026**, `results/final_test_model_comparison.csv`, `results/validation_model_comparison.csv`, `results/ensemble_validation_comparison.csv`, and `results/ensemble_weight_comparison.csv`). Single-model validation metrics, latency benchmarks (CUDA FP16 with GPU warm-up and synchronization), per-class validation metrics, and top confusion counts are verified in `notebooks/NutriVision_Ai.ipynb`. Ensemble validation comparison, weight-sweep, and test-set comparison metrics are recorded in the tracked CSV/JSON artifacts without standalone CLI evaluation scripts or raw test images in the repository, so those results have not been independently reproduced via an automated test harness.

## Model Comparison & Evaluation Results

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
- **Error analysis artifacts:** Per-class validation scores (`results/per_class_validation_metrics.csv`, `results/ensemble_per_class_validation_metrics.csv`), top validation misclassification pairs (`results/convnext_base_top_confusions.csv`, `results/swin_base_top_confusions.csv`), and training curves / confusion matrices (`plots/`).

## Repository Structure

```text
.
├── .gitignore                                    # Excludes secrets, model checkpoints, and local ChromaDB stores
├── LICENSE                                       # MIT License
├── README.md                                     # Project documentation
├── app.py                                        # Streamlit web application (inference, Grad-CAM, nutrition lookup, RAG)
├── ensemble_config.json                          # Class list, checkpoint paths, soft-voting weights, image size
├── requirements.txt                              # Runtime Python dependencies for app.py and scripts/
├── data/
│   ├── class_mapping.csv                         # Class name to integer index mapping (45 classes)
│   ├── nutrition_45_foods.csv                    # 45-class nutritional reference dataset
│   └── splits/
│       ├── train.csv                             # Training split manifest (1,575 entries)
│       ├── validation.csv                        # Validation split manifest (337 entries)
│       └── test.csv                              # Test split manifest (338 entries)
├── models/
│   └── .gitkeep                                  # Target directory for external .pth checkpoints
├── notebooks/
│   └── NutriVision_Ai.ipynb                      # Colab notebook for training, validation, RAG indexing, and repo export
├── plots/
│   ├── convnext_base_accuracy.png                # ConvNeXt Base training/validation accuracy curve
│   ├── convnext_base_loss.png                    # ConvNeXt Base training/validation loss curve
│   ├── convnext_base_confusion_matrix.png        # ConvNeXt Base validation confusion matrix
│   ├── swin_base_accuracy.png                    # Swin Transformer Base training/validation accuracy curve
│   ├── swin_base_loss.png                        # Swin Transformer Base training/validation loss curve
│   ├── swin_base_confusion_matrix.png            # Swin Transformer Base validation confusion matrix
│   ├── convnext_tiny_accuracy.png                # Exploratory ConvNeXt Tiny accuracy curve (not used in ensemble)
│   └── convnext_tiny_loss.png                    # Exploratory ConvNeXt Tiny loss curve (not used in ensemble)
├── results/
│   ├── final_evaluation_report.json              # Final summary metrics and evaluation metadata
│   ├── final_test_model_comparison.csv           # Test-set model comparison metrics
│   ├── validation_model_comparison.csv           # Validation single-model metrics and T4 inference latency
│   ├── ensemble_validation_comparison.csv        # Validation single-model vs. ensemble metrics
│   ├── ensemble_weight_comparison.csv            # Validation soft-voting weight sweep results
│   ├── per_class_validation_metrics.csv          # Per-class validation metrics (ConvNeXt Base and Swin Base)
│   ├── ensemble_per_class_validation_metrics.csv # Per-class validation metrics (50/50 ensemble)
│   ├── convnext_base_top_confusions.csv          # Top 15 ConvNeXt Base validation confusions
│   └── swin_base_top_confusions.csv            # Top 15 Swin Transformer Base validation confusions
└── scripts/
    └── build_chroma_db.py                        # Builds data/chroma_db from data/nutrition_45_foods.csv
```

## External Prerequisites

Large binary weights, secrets, vector indices, and raw training images are excluded from Git history (`.gitignore`) and must be provisioned externally before running the full application:

1. **Fine-Tuned Model Checkpoints (Required for Food Recognition & Grad-CAM)**
   - Pretrained weights are published as release assets (`best_convnext_base.pth` and `best_swin_base.pth`) under the repository's **`v1.0.0`** GitHub Release (`models`).
   - By default, `ensemble_config.json` expects the checkpoints at:
     - `models/best_convnext_base.pth`
     - `models/best_swin_base.pth`
   - `app.py` loads weights via `torch.load(..., map_location="cpu", weights_only=True)` and supports raw `state_dict` objects as well as checkpoint dictionaries containing `"model_state_dict"` or `"state_dict"`.
   - Alternate file locations can be configured in `ensemble_config.json`.

2. **Google Gemini API Key (Required for Nutrition Assistant)**
   - A valid Google Gemini API key is required for query/document embedding (`gemini-embedding-001`) and response generation (`gemini-2.5-flash`).

3. **Populated ChromaDB Vector Store (Required for RAG Retrieval)**
   - `app.py` opens a persistent ChromaDB client at `data/chroma_db` and accesses collection `nutrivision_food_knowledge`.
   - Because `chroma_db/` is excluded by `.gitignore`, `data/chroma_db` is absent from a fresh clone. If absent or empty, `app.py` initializes an empty collection and the Nutrition Assistant reports that the food knowledge database is empty.
   - Run `python scripts/build_chroma_db.py` (with `GEMINI_API_KEY` configured) to index the 45 records from `data/nutrition_45_foods.csv` into `data/chroma_db` using `gemini-embedding-001` (`task_type="RETRIEVAL_DOCUMENT"`).

4. **Raw Image Dataset (Optional — Only for Retraining or Re-Evaluation)**
   - The 2,250 raw images referenced by `data/splits/train.csv`, `data/splits/validation.csv`, and `data/splits/test.csv` are not distributed in this repository.

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

> **Dependencies Note:** `requirements.txt` lists runtime packages required by `app.py` and `scripts/build_chroma_db.py` (`streamlit`, `torch`, `torchvision`, `pandas`, `numpy`, `Pillow`, `chromadb`, `google-genai`, `matplotlib`). If GPU acceleration is desired, install the CUDA-enabled PyTorch and TorchVision builds matching the host CUDA runtime. Training and evaluation cells in `notebooks/NutriVision_Ai.ipynb` additionally use `scikit-learn`, `seaborn`, and `tqdm`.

## Configuration

1. **Provision model checkpoints:** Obtain `best_convnext_base.pth` and `best_swin_base.pth` from the repository's `v1.0.0` GitHub Release and store them at `models/best_convnext_base.pth` and `models/best_swin_base.pth` (or update the paths in `ensemble_config.json`).

2. **Configure Streamlit secrets (for the Nutrition Assistant):** Create `.streamlit/secrets.toml` (excluded by `.gitignore`) containing:

```toml
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
```

3. **Build the ChromaDB vector store (for the Nutrition Assistant):**

```bash
python scripts/build_chroma_db.py
```

`scripts/build_chroma_db.py` reads `GEMINI_API_KEY` from the environment or `.streamlit/secrets.toml` and populates `data/chroma_db`.

## Usage

Run the Streamlit application from the repository root:

```bash
streamlit run app.py
```

- **Food Recognition tab:** Upload an image (`jpg`, `jpeg`, `png`, or `webp`) to inspect the top-1 ensemble prediction, top-5 confidence table, dataset nutrition estimates, and side-by-side ConvNeXt Base / Swin Transformer Base Grad-CAM overlays.
- **Nutrition Assistant tab:** Submit natural-language food or nutrition questions answered via ChromaDB vector retrieval and `gemini-2.5-flash`.
- **About tab:** Displays project summary metadata and recorded ensemble validation metrics (`85.16%` accuracy, `85.08%` macro F1).

## Known Limitations & Nutrition Disclaimer

- **Single-class classification only:** No object-detection model (e.g., YOLO) is implemented; multi-item plates, bounding-box localization, and meal tracking are not supported.
- **No portion or weight estimation:** The system does not estimate serving weight from images and does not scale nutritional values dynamically.
- **Unverified nutritional measurement basis:** Values in `data/nutrition_45_foods.csv` are project estimates whose measurement basis is unverified (values must not be assumed to be per 100 g).
- **Not medical advice:** Outputs are intended strictly for technical evaluation and educational exploration, not medical, clinical, or dietary guidance.
- **Inference vs. evaluation transform difference:** `app.py` resizes uploaded images directly to `224 × 224`, whereas notebook evaluation applied `Resize(256)` followed by `CenterCrop(224)`.
- **No standalone training or evaluation CLI scripts:** Model training and offline evaluation live inside `notebooks/NutriVision_Ai.ipynb` rather than standalone `.py` entrypoints.
- **External RAG dependencies:** The Nutrition Assistant requires both `GEMINI_API_KEY` and a populated `nutrivision_food_knowledge` ChromaDB store at `data/chroma_db`.

## Contributing

Contributions are welcome for documentation improvements, bug fixes, standalone reproducibility scripts (such as CLI evaluation scripts), and model or UI enhancements.

1. Fork the repository and create a feature branch.
2. Keep pull requests focused, well-tested, and accurately documented.
3. Never commit API keys, `.streamlit/secrets.toml`, model checkpoints (`*.pth`, `*.pt`, `*.ckpt`), or local `chroma_db/` directories.
4. Open a pull request describing the motivation, changes made, and verification performed.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
