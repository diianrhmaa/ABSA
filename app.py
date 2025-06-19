from flask import Flask, render_template, request, send_file, jsonify
import os
import pickle
import numpy as np
import torch
import pandas as pd
from gensim.models import LdaModel, TfidfModel
from gensim import corpora
from transformers import BertTokenizer, AutoModel
# from sklearn.preprocessing import LabelEncoder
from models.text_preprocessing import process_text, load_normalization_dict, load_synonym_dict

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
RESULTS_FOLDER = "results"
if not os.path.exists(RESULTS_FOLDER):
    os.makedirs(RESULTS_FOLDER)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Load models
lda_model = LdaModel.load("models/lda_model.gensim")
dictionary = corpora.Dictionary.load("models/lda_dictionary.gensim")
tfidf = TfidfModel.load("models/tfidf_model.gensim")

normalization_dict = load_normalization_dict("static/kamus.csv")
synonym_dict = load_synonym_dict("static/synonym_dict.csv")

with open("models/svm_model.pkl", "rb") as file:
    svm_model = pickle.load(file)

with open("models/label_encoder.pkl", "rb") as file:
    label_encoder = pickle.load(file)

tokenizer = BertTokenizer.from_pretrained("indobenchmark/indobert-base-p2")
model = AutoModel.from_pretrained("indobenchmark/indobert-base-p2")

topic_labels = {
    0: "Kesuksesan Acara",
    1: "Materi atau Narasumber",
    2: "Harapan Acara",
    3: "Antusiasme atau Keterlibatan Peserta",
    4: "Teknis atau Jaringan",
    5: "Waktu atau Pelaksanaan Acara",
    6: "Kesan atau Pesan",
}

def clean_text(text):
    return process_text(text, normalization_dict, synonym_dict)

def get_bert_embedding(texts):
    embeddings = []
    for text in texts:
        tokens = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            output = model(tokens["input_ids"], attention_mask=tokens["attention_mask"])
        cls_embedding = output.last_hidden_state[:, 0, :].squeeze().numpy()
        embeddings.append(cls_embedding)
    return np.array(embeddings)

def predict_sentiment_and_topic(texts):
    processed_texts = [clean_text(text) for text in texts]
    topic_predictions = [lda_model.get_document_topics(dictionary.doc2bow(text.split())) for text in processed_texts]

    text_embeddings = get_bert_embedding(processed_texts)
    sentiment_predictions = svm_model.predict(text_embeddings)
    sentiment_labels = label_encoder.inverse_transform(sentiment_predictions)

    results = []
    for i, text in enumerate(texts):
        top_topic = max(topic_predictions[i], key=lambda x: x[1], default=(None, 0))
        topic_name = topic_labels.get(top_topic[0], "Tidak Diketahui")
        results.append({
            "text": text,
            "processed_text": processed_texts[i],
            "sentiment": sentiment_labels[i],
            "topic": topic_name,
        })
    return results

@app.route("/", methods=["GET", "POST"])
def index():
    results_text = None
    results_file = None
    uploaded_filename = None
    file_format = None
    error_message = None

    if request.method == "POST":
        if "text_input" in request.form and request.form["text_input"].strip():
            text = request.form["text_input"].strip()
            results_text = predict_sentiment_and_topic([text])

        elif "file" in request.files and request.files["file"].filename != "":
            file = request.files["file"]
            filename = file.filename.lower()

            try:
                if filename.endswith(".csv"):
                    df = pd.read_csv(file)
                    file_format = "csv"
                elif filename.endswith(".xlsx"):
                    df = pd.read_excel(file, engine="openpyxl")
                    file_format = "xlsx"
                else:
                    error_message = "Format file tidak didukung. Harap unggah file dalam format CSV atau Excel."
                    return render_template("index.html", results_text=results_text, results_file=results_file, uploaded_filename=uploaded_filename, file_format=file_format, error_message=error_message)

                texts = df["text"].dropna().tolist() if "text" in df.columns else df.iloc[:, 0].dropna().tolist()
                results_file = predict_sentiment_and_topic(texts)

                results_df = pd.DataFrame(results_file)
                uploaded_filename = f"hasil_analisis_{os.path.splitext(filename)[0]}"

                # Simpan dalam format CSV atau Excel, terlepas dari format unggahan
                csv_path = os.path.join(RESULTS_FOLDER, uploaded_filename + ".csv")
                excel_path = os.path.join(RESULTS_FOLDER, uploaded_filename + ".xlsx")

                results_df.to_csv(csv_path, index=False, encoding="utf-8")
                results_df.to_excel(excel_path, index=False, engine="openpyxl")

            except Exception as e:
                error_message = f"Terjadi kesalahan saat membaca file: {str(e)}"

    return render_template("index.html", results_text=results_text, results_file=results_file, uploaded_filename=uploaded_filename, file_format=file_format, error_message=error_message)

@app.route("/download/<filetype>/<filename>")
def download(filetype, filename):
    filepath = os.path.join(RESULTS_FOLDER, f"{filename}.{filetype}")

    if not os.path.exists(filepath):
        return f"File {filepath} tidak ditemukan!", 404
    
    return send_file(filepath, as_attachment=True)

# @app.route("/download/<filetype>/<filename>")
# def download(filetype, filename):
#     filepath = os.path.join(RESULTS_FOLDER, filename + "." + filetype)
#     return send_file(filepath, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)