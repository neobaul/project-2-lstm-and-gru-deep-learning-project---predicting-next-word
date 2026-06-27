# LSTM & GRU Deep Learning — Next Word Prediction

A next-word prediction system trained on Shakespeare's *Hamlet*, with two Streamlit apps: a simple predictor and an advanced text generator with GPT-style sampling controls.

---

## What It Does

Type a sequence of words and the model predicts what comes next — trained entirely on the language patterns of Hamlet. The advanced app goes further, letting you generate multi-word sequences and tune the creativity of the output using the same sampling techniques used by GPT-4.

---

## Project Structure

```
├── train.ipynb           # Train the LSTM model on Hamlet
├── app.py                # Simple next-word prediction app
├── adv_app.py            # Advanced text generation app with sampling controls
├── hamlet.txt            # Training corpus (Shakespeare's Hamlet)
├── next_word_lstm.h5     # Saved trained LSTM model
├── tokenizer.pickle      # Saved word tokenizer
└── requirements.txt      # Python dependencies
```

---

## The Two Apps

### Basic App (`app.py`)
- Enter a seed phrase → get the predicted next word
- Uses the pre-trained LSTM model and tokenizer directly
- Run with: `streamlit run app.py`

### Advanced App (`adv_app.py`)
- Upload any `.h5` model and `.pickle` tokenizer via the sidebar
- Generate multiple words in sequence (configurable length)
- Three sampling controls:
  - **Temperature** — controls creativity (higher = more random)
  - **Top-K** — limits predictions to the K most likely words
  - **Top-P (Nucleus)** — dynamically picks words summing to probability P
- Displays a live probability distribution bar chart for the first prediction step
- Run with: `streamlit run adv_app.py`

---

## How It Was Trained (`train.ipynb`)

1. Loads `hamlet.txt` as the training corpus
2. Tokenizes the text and builds n-gram sequences
3. Pads sequences to a fixed length
4. Trains an LSTM network with Early Stopping
5. Saves the model to `next_word_lstm.h5` and the tokenizer to `tokenizer.pickle`

---

## Setup & Installation

### 1. Clone the repo
```bash
git clone https://github.com/neobaul/project-2-lstm-and-gru-deep-learning-project---predicting-next-word.git
cd project-2-lstm-and-gru-deep-learning-project---predicting-next-word
```

### 2. Create a virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
# Simple version
streamlit run app.py

# Advanced version with sampling controls
streamlit run adv_app.py
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Model | LSTM (Keras / TensorFlow) |
| Training Data | Shakespeare's Hamlet |
| Web App | Streamlit |
| Visualization | Plotly |
| Sampling | Temperature, Top-K, Top-P (Nucleus) |
| Language | Python 3.11 |

---

## Example

**Input:** `To be or not to`

**Predicted next word:** `be`

**Generated sequence (5 words, temp=0.8):** `To be or not to be the king of denmark`
