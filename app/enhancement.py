# app/enhancement.py

import os
import torch
import random
import joblib
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

# Paths for the models
SEAMLESS_DIR = os.path.join("models", "seamless_translator")
ENHANCER_MODEL_PATH = os.path.join("app", "enhancer_model.pkl")
ENHANCER_VECTORIZER_PATH = os.path.join("app", "enhancer_vectorizer.pkl")
ENHANCER_DATA_PATH = os.path.join("app", "_prompt_enhancer_data_backup.csv")

def download_models():
    """
    Download and save models if they don't already exist in the specified directories.
    """
    # SeamlessM4T for Urdu translation
    if not os.path.exists(SEAMLESS_DIR):
        print("Downloading SeamlessM4T for Urdu translation...")
        tokenizer = AutoTokenizer.from_pretrained("facebook/seamless-m4t-v2-large")
        model = AutoModelForSeq2SeqLM.from_pretrained("facebook/seamless-m4t-v2-large")
        tokenizer.save_pretrained(SEAMLESS_DIR)
        model.save_pretrained(SEAMLESS_DIR)

def translate_to_english(text, lang="Urdu"):
    """
    Translate a given text from Urdu to English using the SeamlessM4T model.
    """
    tokenizer = AutoTokenizer.from_pretrained(SEAMLESS_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(SEAMLESS_DIR)

    inputs = tokenizer(text, return_tensors="pt", src_lang="urd", tgt_lang="eng")
    outputs = model.generate(**inputs, max_length=100)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

def enhance_prompt(prompt: str) -> str:
    """
    Enhance the given prompt using trained TF-IDF + Nearest Neighbors model.
    This uses a previously trained custom model stored in app/.
    The placeholder `{prompt}` in the enhanced prompt is replaced with the input prompt.
    """
    if not os.path.exists(ENHANCER_MODEL_PATH) or not os.path.exists(ENHANCER_VECTORIZER_PATH):
        raise FileNotFoundError("Enhancer model or vectorizer not found. Run train_prompt_enhancer.py first.")

    # Load trained components
    model = joblib.load(ENHANCER_MODEL_PATH)
    vectorizer = joblib.load(ENHANCER_VECTORIZER_PATH)
    df = pd.read_csv(ENHANCER_DATA_PATH)

    # Vectorize the input prompt
    vec = vectorizer.transform([prompt])
    dist, idx = model.kneighbors(vec)

    # Fetch the nearest enhanced prompt
    enhanced_prompt = df.iloc[idx[0][0]]["enhanced_prompt"]

    # Replace the placeholder {prompt} with the input prompt
    enhanced_prompt = enhanced_prompt.replace("{prompt}", prompt)

    return enhanced_prompt
