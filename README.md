# Deep Learning & NLP Models

Two deep learning projects: fine-grained bird species classification using EfficientNet-B4 transfer learning, and an LSTM language model built from scratch in TensorFlow.

## Project 1 — Bird Species Classification (`efficientnet_bird_classification.ipynb`)

Fine-grained image classification on the CUB-200-2011 dataset (200 bird species, ~11,788 images) using transfer learning with EfficientNet-B4.

### Approach

- **Base model:** EfficientNet-B4 pre-trained on ImageNet
- **Fine-tuning:** Unfreeze top layers with reduced learning rate
- **Data augmentation:** Random flip, rotation, colour jitter
- **Evaluation:** Top-1 and Top-5 accuracy on held-out test split

### Dataset

CUB-200-2011 (Caltech-UCSD Birds): 200 fine-grained bird species with bounding box and part annotations.

---

## Project 2 — LSTM Language Model (`lstm_language_model.ipynb`)

Sequence language model built from scratch using LSTM in TensorFlow, trained without pre-trained embeddings.

### Approach

- **Architecture:** Embedding layer (trained from scratch) + stacked LSTM layers + Dense output
- **Training:** Cross-entropy loss, gradient clipping, learning rate decay
- **Evaluation:** Perplexity on held-out test set
- **Generation:** Temperature-controlled sampling for text generation

---

## Tech Stack

- TensorFlow / Keras
- Python 3
- NumPy, Matplotlib
- Google Colab (GPU)
