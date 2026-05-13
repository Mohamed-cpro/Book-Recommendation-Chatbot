# 📚 BookBot — AI-Powered Book Recommendation Chatbot

> Classify a book cover image or describe what you want to read — BookBot identifies the category and surfaces the best-rated titles in seconds.

---

## Table of Contents

- [Overview](#overview)
- [Data Collection](#data-collection)
- [Dataset](#dataset)
- [Project Architecture](#project-architecture)
- [Models](#models)
  - [CNN Models — Image Classification](#cnn-models--image-classification)
  - [RNN Models — Text Classification](#rnn-models--text-classification)
- [Streamlit Application](#streamlit-application)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Results & Evaluation](#results--evaluation)
- [Design Decisions](#design-decisions)
- [Future Work](#future-work)

---

## Overview

**BookBot** is an end-to-end multimodal book recommendation system built with TensorFlow and Streamlit. It supports two interaction modes:

| Mode | Input | Model |
|---|---|---|
| **Text** | Natural language query (e.g. *"I want Japanese culture books"*) | Bidirectional GRU |
| **Image** | Book cover photo (JPG / PNG / WEBP) | EfficientNetB0 |

Both modalities predict one of four categories — **Baby Books**, **Cooking**, **Japanese**, **Kittens** — and return the top-N highest-rated books from the scraped dataset.

---

## Data Collection

All book data and cover images were scraped programmatically from the **[Open Library](https://openlibrary.org/)** public API — no third-party dataset was used.

### Scraping Strategy

| Category | API Endpoint | Notes |
|---|---|---|
| Baby Books | `/subjects/infancy.json` | Paginated subject endpoint |
| Kittens | `/subjects/kittens.json` | Paginated subject endpoint |
| Cooking | `/subjects/cooking.json` | Paginated subject endpoint |
| Japanese | `/search.json?q=language:jpn` | Search endpoint, `language:jpn` filter |

For every work record the scraper:

1. Fetches metadata (title, authors, first publish year) from the subject/search endpoint.
2. Makes a secondary call to `/works/{key}.json` to retrieve the full description.
3. Makes a third call to `/works/{key}/ratings.json` to retrieve the community star rating.
4. Downloads the highest-resolution cover image (`-L.jpg`) from `covers.openlibrary.org`.
5. Writes all fields to a single CSV with a 0.3 s courtesy delay between work requests.

```
Scraper config:
  BOOKS_PER_SUBJECT = 2,000
  BATCH_SIZE        = 100 (paginated)
  Image quality     = Large (-L)
  Rate limit        = 0.3 s / work  +  0.5 s / batch
```

---

## Dataset

| Property | Value |
|---|---|
| Total books | **6,898** |
| Categories | 4 |
| Features | title, authors, subject, publish\_year, description, rating |
| Cover images | Downloaded per book where available |
| Source | [openlibrary.org](https://openlibrary.org/) |

### Preprocessing

- Missing titles / authors filled with empty string / `"Unknown"`.
- Missing descriptions replaced with a structured synthetic sentence (title + author + category + year + rating).
- Missing ratings imputed with the column mean.
- A combined `text` field is derived: `title + authors + subject + description` — consumed by all NLP models.
- Leaky category keywords (`"cooking"`, `"kitten"`, `"japan"`, `"baby"`, …) are **stripped at training time** to prevent the model from trivially memorising category names; they are **kept at inference time** so user queries like *"suggest cooking books"* still work.

### Image Augmentation

The **kittens** folder had significantly fewer images than the others. An augmentation pass brought it up to **1,700 images** using random combinations of:

- Horizontal flip
- Rotation ±20°
- Brightness jitter ×[0.7, 1.3]
- Contrast jitter ×[0.7, 1.3]
- Saturation jitter ×[0.7, 1.3]

---

## Project Architecture

```
Open Library API
      │
      ▼
 Web Scraper (requests)
      │  books_dataset.csv + images/
      ▼
 Data Cleaning & EDA
      │  clean_books_dataset.csv
      ├──────────────────────────────────────────────────────┐
      ▼                                                       ▼
 NLP Pipeline                                          CNN Pipeline
 (title + description → text_hard)                    (images/ folder)
      │                                                       │
      ├── BiLSTM                                             ├── ResNet50
      ├── BiGRU  ◄── deployed in app                        ├── MobileNetV2
      └── BiLSTM + Attention                                └── EfficientNetB0  ◄── deployed in app
                │                                                       │
                └───────────────────┬───────────────────────────────────┘
                                    ▼
                         Streamlit Chatbot (app4.py)
                         Text query  OR  Cover image
                                    │
                                    ▼
                         Top-N book recommendations
                         (from clean_books_dataset.csv)
```

---

## Models

### CNN Models — Image Classification

All three CNNs use **transfer learning** from ImageNet weights with a frozen backbone + fine-tuning phase.

| # | Model | Backbone | Input Pre-processing | Fine-tuned layers |
|---|---|---|---|---|
| CNN 1 | ResNet50 | `ResNet50` | Rescale to [−1, 1] | Last 50 of backbone |
| CNN 2 | MobileNetV2 | `MobileNetV2` | `mobilenet_preprocess_input` | Last 20 of backbone |
| CNN 3 | **EfficientNetB0** ★ | `EfficientNetB0` | `efficientnet_preprocess_input` | Last 20 of backbone |

★ EfficientNetB0 is the model served in the production app.

**Training config:**

```
Image size   : 224 × 224
Batch size   : 32
Initial LR   : 1e-4  (frozen phase)
Fine-tune LR : 1e-5
Early stop   : patience=4, monitor=val_accuracy
LR scheduler : ReduceLROnPlateau factor=0.5, patience=2
```

Runtime augmentation applied to training batches: random flip, rotation ±10%, zoom ±10%, brightness ±10%.

---

### RNN Models — Text Classification

All three RNNs share the same Keras tokenizer (vocab = 10,000, max sequence length = 64). Text is cleaned of stopwords and leaky category words before tokenisation.

| # | Model | Architecture | Regularisation |
|---|---|---|---|
| RNN 1 | BiLSTM | 2 × Bidirectional LSTM (32 → 16 units) | SpatialDropout1D(0.5), Dropout(0.5), L2(1e-3) |
| RNN 2 | **BiGRU** ★ | 2 × Bidirectional GRU (32 → 16 units) | SpatialDropout1D(0.5), Dropout(0.5), L2(1e-3) |
| RNN 3 | BiLSTM + Attention | 2 × BiLSTM + custom attention pooling | SpatialDropout1D(0.5), Dropout(0.5), L2(1e-3) |

★ BiGRU is the model served in the production app.

**Training config:**

```
Embedding dim : 32
Batch size    : 32
Optimizer     : Adam (lr=5e-4)
Early stop    : patience=5, monitor=val_accuracy
LR scheduler  : ReduceLROnPlateau factor=0.5, patience=3
Accuracy cap  : training stops when val_accuracy ≥ 0.85
Split         : 70 % train / 15 % val / 15 % test
```

---

## Streamlit Application

The chatbot UI (`app4.py`) is built with Streamlit and styled with a dark editorial theme (Playfair Display + DM Sans, gold accent `#c8a96e`).

### Features

- **Text query** → BiGRU prediction → category badge + recommendation table
- **Cover image upload** → EfficientNetB0 prediction → category badge + recommendation table
- **Adjustable N** via sidebar slider (3 – 10 recommendations)
- **Clear chat** button resets session history
- Footer metrics: category count, total books, active model names
- Models are loaded once with `@st.cache_resource` — no re-loading between interactions

### Inference pipeline (text)

```python
user_input
    → clean_text_inference()          # stopwords only; leaky words kept
    → keras tokenizer → pad_sequences (maxlen=64)
    → BiGRU predict
    → argmax → LabelEncoder.inverse_transform
    → filter CSV by subject → nlargest(N, 'rating')
    → render HTML table
```

### Inference pipeline (image)

```python
uploaded cover
    → PIL.Image → resize(224, 224) → float32 array → expand_dims
    → EfficientNetB0 predict
    → argmax → class_names[idx]
    → filter CSV by subject → nlargest(N, 'rating')
    → render HTML table
```

---

## Project Structure

```
bookbot/
├── app4.py                        # Streamlit chatbot app
├── Wab_scrapering.ipynb           # Open Library scraper notebook
├── project_deep_fixed.ipynb       # Full model training pipeline
├── books_dataset.csv              # Raw scraped dataset (6,898 rows)
│
├── images/                        # Cover images (one subfolder per category)
│   ├── baby books/
│   ├── cooking/
│   ├── japanese/
│   └── kittens/
│
└── outputs/                       # Saved artefacts (generated by training)
    ├── clean_books_dataset.csv    # Cleaned CSV consumed by the app
    ├── label_encoder.pkl          # Sklearn LabelEncoder
    ├── keras_tokenizer.pkl        # Keras Tokenizer
    ├── cnn1_resnet50.keras
    ├── cnn2_mobilenet.keras
    ├── cnn3_efficientnet.keras    # ← used in app
    ├── rnn1_bilstm.keras
    ├── rnn2_bigru.keras           # ← used in app
    ├── rnn3_bilstm_attention.keras
    └── eda.png
```

---

## Setup & Installation

### Prerequisites

- Python 3.9+
- `pip`

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/bookbot.git
cd bookbot
```

### 2. Install dependencies

```bash
pip install streamlit tensorflow pillow numpy pandas scikit-learn nltk requests tqdm
python -c "import nltk; nltk.download('stopwords')"
```

### 3. (Optional) Re-scrape the data

> Skip this step if you already have `books_dataset.csv` and the `images/` folder.

```bash
jupyter nbconvert --to script Wab_scrapering.ipynb
python Wab_scrapering.py
```

### 4. (Optional) Retrain the models

```bash
jupyter notebook project_deep_fixed.ipynb
# Run all cells top-to-bottom. Outputs are saved to outputs/
```

### 5. Run the app

```bash
streamlit run app4.py
```

The app will open at `http://localhost:8501`.

---

## Usage

**Text mode**

Type a natural language query in the input box and click **Send →**.

```
"I want Japanese culture books"
"suggest some recipe books for beginners"
"show me books about cute kittens"
"baby books for my newborn"
```

**Image mode**

Click the file uploader below the chat, select any book cover (JPG / PNG / WEBP), and BookBot will classify it and return recommendations from the same category.

---

## Results & Evaluation

### CNN (Image Classification)

| Model | Val Accuracy |
|---|---|
| ResNet50 | reported in `outputs/` |
| MobileNetV2 | reported in `outputs/` |
| EfficientNetB0 | reported in `outputs/` |

Training and validation curves are saved to `outputs/*.png`.

### RNN (Text Classification)

| Model | Test Accuracy | Test Loss |
|---|---|---|
| BiLSTM | reported via `classification_report` | — |
| **BiGRU** | reported via `classification_report` | — |
| BiLSTM + Attention | reported via `classification_report` | — |

Full per-class precision / recall / F1 is printed at the end of Section 18 of the training notebook.

---

## Design Decisions

| Decision | Rationale |
|---|---|
| **Leaky word removal at training, not inference** | Prevents label leakage during training while preserving the model's ability to pick up on explicit category words in user queries. |
| **Synthetic descriptions for missing rows** | Keeps every row useful for training without dropping ~30 % of the dataset. Descriptions are clearly structured so they don't introduce noise. |
| **Kittens augmentation only** | Only the kittens folder was imbalanced; augmenting only that class avoids inflating already-sufficient categories. |
| **BiGRU over BiLSTM for deployment** | GRUs have fewer parameters than LSTMs (no output gate) and converge faster — the accuracy cap (≤ 0.85 val_acc) limits overfitting without sacrificing response time in a web app. |
| **EfficientNetB0 over ResNet50 / MobileNetV2** | Best accuracy-to-parameter ratio on image classification benchmarks; lightweight enough to load into a Streamlit session without excessive memory. |
| **`@st.cache_resource` for model loading** | Models are heavy; caching avoids a reload on every user interaction. |
| **Custom inference cleaner (`clean_text_inference`)** | Uses stopwords removal only — not leaky-word removal — so queries like "I want cooking books" are not stripped of the word "cooking" before classification. |

---

## Future Work

- [ ] Add model selector (let user choose CNN 1 / 2 / 3 and RNN 1 / 2 / 3 from the UI)
- [ ] Expand categories beyond the current four
- [ ] Add a confidence threshold: fall back to a generic recommendation list when confidence < 60 %
- [ ] Integrate Open Library's live search API so recommendations are always fresh
- [ ] Deploy on Streamlit Community Cloud / Hugging Face Spaces
- [ ] Add user feedback loop to improve model rankings over time

---

## Data Source

All data is retrieved from **[Open Library](https://openlibrary.org/)**, an open, editable library catalogue project operated by the Internet Archive. Book metadata and cover images are used in accordance with Open Library's open data policy.

---

*Built with TensorFlow · Streamlit · Open Library API*
