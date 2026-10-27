# NutriVision AI

AI-powered food recognition and nutritional analysis.

## Features
- Food classification across 45 categories
- ConvNeXt Base and Swin Transformer Base ensemble
- Top-5 predictions
- Nutritional information lookup
- Gemini-powered nutrition assistant
- Retrieval-augmented generation (RAG) with ChromaDB
- Grad-CAM visual explanations
- Streamlit interface

## Evaluation Results

| Model | Test Accuracy | Test Macro F1 |
|---|---:|---:|
| ConvNeXt Base | 81.36% | 81.06% |
| Swin Transformer Base | 82.84% | 82.99% |
| 50/50 Ensemble | 84.91% | 84.82% |

Ensemble validation accuracy: 85.16%.
Ensemble validation macro F1: 85.08%.

## Setup

Install dependencies using `pip install -r requirements.txt`.

Obtain the required model checkpoints and update their paths in
`ensemble_config.json` if needed.

Configure the Gemini API key in `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
```

Never commit your API key or secrets file.

Ensure the required ChromaDB collection is available, then run
`streamlit run app.py`.

## Nutrition Disclaimer

Nutritional values are estimates. Their measurement basis has not
been verified. The app does not assume values are per 100 g and
does not automatically scale them by food weight. Results are not
medical advice.

## Technology Stack

Python, PyTorch, Torchvision, ConvNeXt, Swin Transformer, Streamlit,
Google Gemini, ChromaDB, RAG, and Grad-CAM.

## License

No license has been selected yet.
