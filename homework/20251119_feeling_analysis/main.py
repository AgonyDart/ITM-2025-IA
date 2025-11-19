# ==========================================
#   ANALISIS COMPLETO DE SENTIMIENTOS CSV
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# ================================
# 1) Cargar y limpiar el dataset
# ================================
df = pd.read_csv("datasetTexto.csv", engine="python", on_bad_lines="skip")

print("\n--- HEAD ---\n")
print(df.head())

print("\n--- NULLS ---\n")
print(df.isnull().sum())

# Rellenar NaN en comentarios
df["Comentario_Reaccion"] = df["Comentario_Reaccion"].fillna("")


# =========================================
# 2) Sentiment Analysis (TextBlob polarity)
# =========================================
def get_sentiment(x):
    return TextBlob(str(x)).sentiment.polarity


df["sentiment"] = df["Comentario_Reaccion"].apply(get_sentiment)


def label(v):
    if v > 0.1:
        return "positivo"
    if v < -0.1:
        return "negativo"
    return "neutro"


df["sentiment_label"] = df["sentiment"].apply(label)

print("\n--- DISTRIBUCIÓN DE SENTIMIENTOS ---\n")
print(df["sentiment_label"].value_counts())

# =========================================
# 3) Estadísticas descriptivas
# =========================================
print("\n--- ESTADÍSTICAS DE POLARIDAD ---\n")
print(df["sentiment"].describe())


# =========================================
# 4) Nube de palabras por sentimiento
# =========================================
def plot_wordcloud(text, title):
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.title(title, fontsize=16)
    plt.axis("off")
    plt.show()


text_pos = " ".join(df[df["sentiment_label"] == "positivo"]["Comentario_Reaccion"])
text_neg = " ".join(df[df["sentiment_label"] == "negativo"]["Comentario_Reaccion"])
text_neu = " ".join(df[df["sentiment_label"] == "neutro"]["Comentario_Reaccion"])

plot_wordcloud(text_pos, "Nube de Palabras - POSITIVO")
plot_wordcloud(text_neg, "Nube de Palabras - NEGATIVO")
plot_wordcloud(text_neu, "Nube de Palabras - NEUTRO")


# =========================================
# 5) Top palabras más comunes (BoW)
# =========================================
vectorizer = CountVectorizer(stop_words="spanish")
bow = vectorizer.fit_transform(df["Comentario_Reaccion"])

# Sumamos total de cada palabra
sum_words = bow.sum(axis=0)
words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

print("\n--- TOP 20 PALABRAS MÁS FRECUENTES ---\n")
for w, f in words_freq[:20]:
    print(f"{w}: {f}")


# =========================================
# 6) TF-IDF de palabras más importantes
# =========================================
tfidf = TfidfVectorizer(stop_words="spanish")
tfidf_matrix = tfidf.fit_transform(df["Comentario_Reaccion"])

tfidf_scores = tfidf_matrix.sum(axis=0)
tfidf_scores = [(word, tfidf_scores[0, idx]) for word, idx in tfidf.vocabulary_.items()]
tfidf_scores = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)

print("\n--- TOP 20 PALABRAS CON MAYOR TF-IDF ---\n")
for w, f in tfidf_scores[:20]:
    print(f"{w}: {f:.4f}")


# =========================================
# 7) Insights por categoría (ej: Generación Z vs Frankenstein)
# =========================================
print("\n--- SENTIMIENTO PROMEDIO POR CATEGORÍA ---\n")
print(df.groupby("Categoria")["sentiment"].mean())

print("\n--- DISTRIBUCIÓN DE SENTIMIENTOS POR CATEGORÍA ---\n")
print(df.groupby(["Categoria", "sentiment_label"])["ID"].count())

# Gráfico básico
df.groupby("sentiment_label")["ID"].count().plot(kind="bar", figsize=(6, 4))
plt.title("Distribución Global de Sentimientos")
plt.show()
