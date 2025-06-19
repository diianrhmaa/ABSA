# from flask import Flask, render_template, request, send_file
# import pickle
# import numpy as np
# import torch
# import pandas as pd
# import re
# from gensim.models import LdaModel, TfidfModel
# from gensim import corpora
# from transformers import BertTokenizer, AutoModel
# from sklearn.pipeline import Pipeline

# # Load preprocessing tools
# with open("models/stopwords.pkl", "rb") as file:
#     stop_words = pickle.load(file)

# with open("models/normalization_dict.pkl", "rb") as file:
#     normalization_dict = pickle.load(file)

# with open("models/synonym_dict.pkl", "rb") as file:
#     synonym_dict = pickle.load(file)

# # Load saved models
# lda_model = LdaModel.load("models/lda_model.gensim")
# dictionary = corpora.Dictionary.load("models/lda_dictionary.gensim")
# tfidf = TfidfModel.load("models/tfidf_model.gensim")

# with open("models/svm_model.pkl", "rb") as file:
#     svm_model = pickle.load(file)

# with open("models/label_encoder.pkl", "rb") as file:
#     label_encoder = pickle.load(file)

# # Load IndoBERT tokenizer & model
# tokenizer = BertTokenizer.from_pretrained("indobenchmark/indobert-base-p2")
# model = AutoModel.from_pretrained("indobenchmark/indobert-base-p2")

# # Flask app
# app = Flask(__name__)

# topic_labels = { 
#     0: "Kesuksesan Acara", 
#     1: "Materi atau Narasumber", 
#     2: "Harapan Acara", 
#     3: "Antusiasme atau Keterlibatan Peserta", 
#     4: "Teknis atau Jaringan", 
#     5: "Waktu atau Pelaksanaan Acara", 
#     6: "Kesan atau Pesan" }

# def clean_text(text):
#     text = text.lower()
#     text = re.sub(r'[^a-zA-Z\s]', '', text)  # Hapus angka dan karakter spesial
#     text = re.sub(r'\s+', ' ', text).strip()  # Hapus spasi berlebih
#     words = text.split()
#     words = [normalization_dict.get(word, word) for word in words]  # Normalisasi kata
#     words = [synonym_dict.get(word, word) for word in words]  # Ubah sinonim
#     words = [word for word in words if word not in stop_words]  # Hapus stopwords
#     return ' '.join(words)

# @app.route("/", methods=["GET", "POST"])
# def index():
#     results = None
#     if request.method == "POST":
#         if "text_input" in request.form and request.form["text_input"].strip():
#             text = request.form["text_input"].strip()
#             results = predict_sentiment_and_topic([text])
#         elif "file" in request.files and request.files["file"].filename != "":
#             file = request.files["file"]
#             df = pd.read_csv(file)
#             texts = df.iloc[:, 0].dropna().tolist()
#             results = predict_sentiment_and_topic(texts)
#     return render_template("index.html", results=results)

# def get_bert_embedding(texts):
#     embeddings = []
#     for text in texts:
#         tokens = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
#         with torch.no_grad():
#             output = model(tokens["input_ids"], attention_mask=tokens["attention_mask"])
#         cls_embedding = output.last_hidden_state[:, 0, :].squeeze().numpy()
#         embeddings.append(cls_embedding)
#     return np.array(embeddings)

# def predict_sentiment_and_topic(texts):
#     processed_texts = [clean_text(text) for text in texts]
#     topic_predictions = [lda_model.get_document_topics(dictionary.doc2bow(text.split())) for text in processed_texts]
    
#     # Convert texts to IndoBERT embeddings
#     text_embeddings = get_bert_embedding(processed_texts)
#     sentiment_predictions = svm_model.predict(text_embeddings)
#     sentiment_labels = label_encoder.inverse_transform(sentiment_predictions)
    
#     results = []
#     for i, text in enumerate(texts):
#         top_topic = max(topic_predictions[i], key=lambda x: x[1], default=(None, 0))
#         topic_name = topic_labels.get(top_topic[0], "Tidak Diketahui")
#         results.append({ "text": text, "processed_text": processed_texts[i], "sentiment": sentiment_labels[i], "topic": topic_name })
#     return results

# if __name__ == "__main__":
#     app.run(debug=True)