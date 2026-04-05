# Deep Learning & NLP Models

Two end-to-end deep learning projects: fine-grained image classification using transfer learning, and an LSTM language model built from scratch.

---

## Task 1 — Fine-Grained Bird Image Classification

**Dataset:** [CUB-200-2011](https://data.caltech.edu/records/65de6-vp158) — 200 visually similar bird species, ~11,788 images with bounding box annotations.

### Approach
- Bounding boxes used to crop images before resizing to **380×380** to reduce background noise
- Data augmentation: random horizontal flip, rotation (±8°), zoom (15%), contrast (15%), Gaussian noise
- **Model 1 (Transfer Learning):** EfficientNet-B4 pretrained on ImageNet
  - Phase 1: Backbone frozen, only classification head trained (lr=1e-3, cosine decay, 8 epochs)
  - Phase 2: Top layers unfrozen for fine-tuning (lr=1e-4, AdamW, weight decay=1e-5, up to 15 epochs)
  - Head: GlobalAveragePooling → BatchNorm → Dropout(0.5) → Dense(200, softmax)
- **Model 2 (Custom CNN):** Designed from scratch (~8M parameters), deeper and wider architecture

### Key Results
| Model | Notes |
|-------|-------|
| EfficientNet-B4 | Best validation accuracy across all experiments |
| Custom CNN (~8M params) | Outperformed smaller 4M-param variant significantly |

---

## Task 2 — LSTM Language Model

**Dataset:** Project Gutenberg text corpus, preprocessed and tokenized.

### Approach
- Text cleaning: removed Gutenberg headers/footers, lowercased, stripped special characters
- Tokenization with Keras Tokenizer; sequence length = 25 tokens
- **Architecture:** Embedding(vocab_size, 256) → LSTM(256) → LSTM(256) → Dense(vocab_size, softmax)
- Training: Adam (lr=1e-3), sparse categorical crossentropy, early stopping + model checkpoint
- Text generation with **top-k sampling** (k=30, temperature=0.85) for diversity

---

## Tech Stack
- Python, TensorFlow / Keras
- Google Colab (GPU)
- NumPy, Pandas, scikit-learn, Matplotlib

## Files
| File | Description |
|------|-------------|
| `Deepleaning_Coursework-Task1.ipynb` | EfficientNet-B4 & custom CNN for bird classification |
| `NLP_languageModel.ipynb` | LSTM language model with text generation |
