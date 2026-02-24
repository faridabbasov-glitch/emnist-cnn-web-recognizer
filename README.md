# Neural Ink — EMNIST Alphanumeric Recognizer

A full-stack web application that recognizes handwritten **letters and digits** in real time using a CNN trained on the [EMNIST dataset](https://www.kaggle.com/datasets/crawford/emnist).

Draw a character on the canvas, hit **PREDICT**, and instantly get the top-3 predictions with confidence scores.

---

## Project Structure

```
neural-ink/
├── frontend/
│   └── index.html       # Drawing canvas UI (vanilla JS)
├── backend/
│   └── main.py          # FastAPI server + /predict endpoint
├── best_model.h5       
├── label_map.json       # Class index → character mapping
└── requirements.txt
```


---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/faridabbasov-glitch/neural-ink.git
cd neural-ink

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place best_model.h5 in the root directory

# 5. Run the server
uvicorn backend.main:app --reload
```

Open **http://localhost:8000** in your browser.

---

## Model

The CNN was trained on the **EMNIST ByClass** split, covering:
- 10 digits (0–9)
- 26 uppercase letters (A–Z)
- 26 lowercase letters (a–z, with EMNIST merged classes)

**Architecture:**
```
Input (28×28×1)
→ Conv2D(32) + BatchNorm + MaxPool
→ Conv2D(64) + BatchNorm + MaxPool
→ Conv2D(128) + BatchNorm + MaxPool
→ Flatten → Dense(256) → Dropout(0.5)
→ Dense(num_classes, softmax)
```

---

## Image Preprocessing

Before inference, user drawings are processed to match the EMNIST format:

1. Convert to grayscale
2. Invert colors if background is light
3. Crop tightly around the drawn character
4. Resize to 20×20 and center-pad to 28×28
5. Normalize pixel values to [0, 1]
6. Flip horizontally to match EMNIST's mirrored orientation

---

## API

### `POST /predict`

**Request:**
```json
{ "image": "data:image/png;base64,..." }
```

**Response:**
```json
{
  "predictions": [
    { "label": "A", "confidence": 0.9821 },
    { "label": "R", "confidence": 0.0112 },
    { "label": "H", "confidence": 0.0043 }
  ]
}
```

---

## Tech Stack

| Layer    | Technology            |
|----------|-----------------------|
| Frontend | Vanilla HTML / CSS / JS |
| Backend  | Python, FastAPI       |
| Model    | TensorFlow / Keras    |
| Dataset  | EMNIST ByClass        |



