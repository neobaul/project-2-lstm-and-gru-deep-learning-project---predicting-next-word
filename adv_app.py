# Import libraries
import streamlit as st
import numpy as np
import pickle
import os
import tempfile
import plotly.express as px
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- APP CONFIGURATION ---
st.set_page_config(
    page_title="AI World Architect Prediction with LSTM and GRU",
    page_icon="🔮",
    layout="centered",
    initial_sidebar_state="expanded",
)

# --- CSS FOR MODERN UI ---
st.markdown("""
<style>
.main { background-color: #0e1117; color: white; }
    .stButton>button { width: 100%; border-radius: 20px; background-color: #ff4b4b; color: white; font-weight: bold; }
    .prediction-box { padding: 20px; border-radius: 15px; background-color: #262730; border-left: 5px solid #ff4b4b; color: #ffffff; font-size: 18px; }
</style>
""", unsafe_allow_html=True)

# --- LOAD ENGINE (DYNAMIC MODEL LOADING) ---
@st.cache_resource
def load_custom_model(model_file):
    """Saves uploaded file to a temp path and loads it to avoid path errors."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
        tmp.write(model_file.getvalue())
        tmp_path = tmp.name
    model = load_model(tmp_path)
    return model

@st.cache_resource
def load_custom_tokenizer(tokenizer_file):
    return pickle.load(tokenizer_file)

# advanced sampling function logic
def apply_sampling(preds, temperature=1.0, top_k=0, top_p=0.0):
    """
    Implements the same sampling techniques used by GPT-4.
    """
    preds = np.asarray(preds).astype('float64')

    # 1. Temperature Scaling
    preds = np.log(preds + 1e-10) / temperature
    exp_preds = np.exp(preds)
    probs = exp_preds / np.sum(exp_preds)

    # 2. Top-K Filtering
    if top_k > 0:
        top_k_indices = np.argsort(probs)[-top_k:]
        new_probs = np.zeros_like(probs)
        new_probs[top_k_indices] = probs[top_k_indices]
        probs = new_probs / np.sum(new_probs)

    # 3. Top-P (Nucleus) Sampling
    if top_p > 0.0:
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumulative_probs = np.cumsum(sorted_probs)

        # Remove tokens with cumulative probability above the threshold
        indices_to_remove = cumulative_probs > top_p
        # Shift to keep the first token that crosses the threshold
        indices_to_remove[1:] = indices_to_remove[:-1].copy()
        indices_to_remove[0] = False

        probs[sorted_indices[indices_to_remove]] = 0
        probs = probs / np.sum(probs)

    return np.random.choice(len(probs), p=probs)

# --- 3. UI LAYOUT ---
st.title("🚀 AI Word Architect: Next-Gen Prediction")
st.markdown("Upload your trained **LSTM/H5** model and **Pickle** tokenizer to begin.")

# Sidebar for Uploads and Parameters
with st.sidebar:
    st.header("🗂️ Model Assets")
    uploaded_model = st.file_uploader("Upload .h5 Model", type=["h5"])
    uploaded_tokenizer = st.file_uploader("Upload Tokenizer (.pickle)", type=["pickle"])

    st.divider()
    st.header("tuning 🔧")
    temp = st.slider("Temperature (Creativity)", 0.1, 2.0, 1.0, help="Higher = more random, Lower = more focused.")
    top_k = int(st.number_input("Top-K Filtering", 0, 100, 40, help="Only consider the top K most likely words."))
    top_p = st.slider("Top-P (Nucleus)", 0.0, 1.0, 0.9, help="Keeps a dynamic 'nucleus' of words summing to P probability.")
    gen_len = int(st.number_input("Generation Length", 1, 50, 5))

# Main Application Logic
if uploaded_model and uploaded_tokenizer:
    model = load_custom_model(uploaded_model)
    tokenizer = load_custom_tokenizer(uploaded_tokenizer)
    max_len = model.input_shape[1] + 1

    input_text = st.text_area("Initial Seed Phrase:", "To be or not to", height=100)

    if st.button("Generate Text"):
        with st.spinner("Analyzing patterns..."):
            generated_text = input_text
            last_top_predictions = []

            for _ in range(gen_len):
                token_list = tokenizer.texts_to_sequences([generated_text])[0]
                token_list = pad_sequences([token_list], maxlen=max_len-1, padding='pre')

                # Get raw probabilities
                preds = model.predict(token_list, verbose=0)[0]

                # Store the very first prediction's distribution for the chart
                if _ == 0:
                    top_indices = np.argsort(preds)[-10:][::-1]
                    for idx in top_indices:
                        for word, index in tokenizer.word_index.items():
                            if index == idx:
                                last_top_predictions.append({'Word': word, 'Probability': preds[idx] * 100})

                # Apply advanced sampling
                next_index = apply_sampling(preds, temp, top_k, top_p)

                # Convert index to word
                output_word = ""
                for word, index in tokenizer.word_index.items():
                    if index == next_index:
                        output_word = word
                        break

                generated_text += " " + output_word

        # --- DISPLAY RESULTS ---
        st.subheader("Final Generated Sequence")
        st.markdown(f'<div class="prediction-box">{generated_text}</div>', unsafe_allow_html=True)

        # --- VISUALIZATION ---
        st.divider()
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Probability Distribution (Step 1)")
            df = pd.DataFrame(last_top_predictions)
            fig = px.bar(df, x='Probability', y='Word', orientation='h',
                         color='Probability', color_continuous_scale='Reds',
                         template="plotly_dark")
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Model Stats")
            st.metric("Vocab Size", len(tokenizer.word_index))
            st.metric("Context Window", max_len-1)
            st.success("Sampling Active: Top-K/Top-P")

else:
    st.info("👋 Please upload your **.h5 model** and **.pickle tokenizer** in the sidebar to start.")

    # Placeholder Visual to keep the UI clean
    st.image("https://images.unsplash.com/photo-1677442136019-21780ecad995?auto=format&fit=crop&q=80&w=1000", caption="Deep Learning Sequence Model Ready")
    