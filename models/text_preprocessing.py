import re
import string
import pandas as pd
from nltk.tokenize import word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import nltk

# Pastikan nltk tokenizer tersedia
nltk.download('punkt')
nltk.download('punkt_tab')

# Mengambil daftar stopword Bahasa Indonesia dari Sastrawi
factory = StopWordRemoverFactory()
stop_words = set(factory.get_stop_words())
stop_words.update(["nya", "yg", "lah", "sih", "amin", "jadi", "iya", "kok", "hehe", "ada", "kalau", "lebih", "sangat", "sebagainya", "semua", "tidak ada"])

# Fungsi untuk membersihkan teks
def remove_special(text):
    text = text.encode('ascii', 'replace').decode('ascii')
    text = re.sub(r"([#@][A-Za-z0-9_]+)|\w+:\/\/\S+", " ", text)
    text = re.sub(r"[^\w\s-]", " ", text)
    return text

def remove_number(text):
    return re.sub(r"\d+", "", text)

def remove_punctuation(text):
    return text.translate(str.maketrans("", "", string.punctuation))

def remove_whitespace(text):
    return re.sub(r'\s+', ' ', text).strip()

def remove_repeated_chars(text):
    return re.sub(r'(.)\1{2,}', r'\1', text)

def cleaning_text(text):
    text = remove_special(text)
    text = remove_number(text)
    text = remove_punctuation(text)
    text = remove_whitespace(text)
    text = remove_repeated_chars(text)
    return text.lower()

def tokenize_text(text):
    return word_tokenize(text)

def load_normalization_dict(file_path):
    try:
        df = pd.read_csv(file_path)
        return dict(zip(df['word'], df['normalized']))
    except Exception as e:
        print(f"Error loading normalization dictionary: {e}")
        return {}

def load_synonym_dict(file_path):
    try:
        df = pd.read_csv(file_path)
        return dict(zip(df['kata'], df['sinonim']))
    except Exception as e:
        print(f"Error loading synonym dictionary: {e}")
        return {}

def normalize_text(tokens, normalization_dict, synonym_dict):
    normalized_tokens = [normalization_dict.get(word, word) for word in tokens]
    final_normalized_tokens = [synonym_dict.get(word, word) for word in normalized_tokens]
    return final_normalized_tokens

def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words]

def process_text(text, normalization_dict={}, synonym_dict={}):
    text = cleaning_text(text)
    tokens = tokenize_text(text)
    tokens = normalize_text(tokens, normalization_dict, synonym_dict)
    tokens = remove_stopwords(tokens)
    return ' '.join(tokens)