# train_sentiment_models.py
# Requisitos:
# pip install pandas scikit-learn matplotlib seaborn joblib textblob nltk

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from textblob import TextBlob
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)
import joblib
import nltk
from datetime import datetime

# Si no tienes stopwords de nltk para español, ejecutar una vez:
# nltk.download('stopwords')
from nltk.corpus import stopwords

SPANISH_STOPWORDS = stopwords.words("spanish")

# ------------- Config -------------
CSV_PATH = "datasetTexto.csv"
TEXT_COL = "Comentario_Reaccion"  # columna con texto
DATE_COL = "Fecha"  # columna con fecha (si aplica)
ID_COL = "ID"
OUTPUT_DIR = "output_sentiment"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------- 1) Cargar y limpiar -------------
df = pd.read_csv(CSV_PATH, engine="python", on_bad_lines="skip")

# Asegurarse columnas existan
if TEXT_COL not in df.columns:
    # si el dataset tiene otra columna de texto puedes ajustar aquí
    TEXT_COL = df.columns[0]
    print(f"Usando columna {TEXT_COL} como texto.")

df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)

# Si hay columna Fecha, parsearla (intenta varios formatos)
if DATE_COL in df.columns:
    try:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    except Exception:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], format="%Y-%m-%d", errors="coerce")
else:
    df[DATE_COL] = pd.NaT

# ------------- 2) Etiquetado (si no existe) -------------
# Si ya tienes una columna 'sentiment_label' en el CSV, úsala. Si no, generamos etiquetas con TextBlob.
if "sentiment_label" in df.columns and df["sentiment_label"].notna().sum() > 0:
    labels = df["sentiment_label"].astype(str)
else:
    # generamos polaridad y etiqueta
    def polarity(text):
        try:
            return TextBlob(str(text)).sentiment.polarity
        except Exception:
            return 0.0

    df["polarity"] = df[TEXT_COL].apply(polarity)

    def lab(p):
        if p > 0.1:
            return "positivo"
        if p < -0.1:
            return "negativo"
        return "neutro"

    df["sentiment_label"] = df["polarity"].apply(lab)
    labels = df["sentiment_label"]

print("Distribución de etiquetas:")
print(labels.value_counts())

# ------------- 3) Split train/test -------------
X = df[TEXT_COL].values
y = labels.values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
)

# ------------- 4) Pipelines para modelos -------------
# TF-IDF vectorizer (español stopwords)
tfidf = TfidfVectorizer(
    stop_words=SPANISH_STOPWORDS, max_df=0.95, min_df=2, ngram_range=(1, 2)
)

pipe_nb = Pipeline([("tfidf", tfidf), ("clf", MultinomialNB())])

pipe_svc = Pipeline([("tfidf", tfidf), ("clf", LinearSVC(max_iter=5000))])

# ------------- 5) Entrenamiento (con CV opcional) -------------
print("\nEntrenando MultinomialNB...")
pipe_nb.fit(X_train, y_train)
print("Entrenando LinearSVC...")
pipe_svc.fit(X_train, y_train)

# Opcional: GridSearch sobre C para SVC (comentar si quieres ejecución más rápida)
# params = {'clf__C':[0.01, 0.1, 1, 5]}
# gs = GridSearchCV(pipe_svc, params, cv=3, scoring='f1_macro', n_jobs=-1)
# gs.fit(X_train, y_train)
# pipe_svc = gs.best_estimator_
# print("Best SVC params:", gs.best_params_)


# ------------- 6) Evaluación -------------
def evaluate(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    print(f"\n=== RESULTADOS {name} ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (macro): {prec:.4f}")
    print(f"Recall (macro): {rec:.4f}")
    print(f"F1 (macro): {f1:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, zero_division=0))
    cm = confusion_matrix(y_test, y_pred, labels=["positivo", "neutro", "negativo"])
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["positivo", "neutro", "negativo"],
        yticklabels=["positivo", "neutro", "negativo"],
    )
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.title(f"Matriz de Confusión - {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"confusion_matrix_{name}.png"))
    plt.close()
    # Guardar reporte por clase en csv-friendly form
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(
        os.path.join(OUTPUT_DIR, f"classification_report_{name}.csv")
    )


evaluate(pipe_nb, X_test, y_test, "MultinomialNB")
evaluate(pipe_svc, X_test, y_test, "LinearSVC")

# ------------- 7) Comparativa cross-val (opcional) -------------
print("\nCross-validation (3 folds) - accuracy:")
for name, model in [("MultinomialNB", pipe_nb), ("LinearSVC", pipe_svc)]:
    scores = cross_val_score(
        model, X_train, y_train, cv=3, scoring="accuracy", n_jobs=-1
    )
    print(f"{name}: mean={scores.mean():.4f} std={scores.std():.4f}")

# ------------- 8) Guardar modelos -------------
joblib.dump(pipe_nb, os.path.join(OUTPUT_DIR, "model_multinomialnb.joblib"))
joblib.dump(pipe_svc, os.path.join(OUTPUT_DIR, "model_linearsvc.joblib"))
print(f"Modelos guardados en {OUTPUT_DIR}/")

# ------------- 9) Predicciones y probabilidades (si aplica) -------------
# MultinomialNB soporta predict_proba
if hasattr(pipe_nb.named_steps["clf"], "predict_proba"):
    proba = pipe_nb.predict_proba(X_test)
    proba_df = pd.DataFrame(proba, columns=pipe_nb.named_steps["clf"].classes_)
    proba_df.to_csv(os.path.join(OUTPUT_DIR, "pred_proba_multinb.csv"), index=False)

# Guardar test preds
test_out = pd.DataFrame(
    {
        ID_COL: np.arange(len(X_test)),
        "text": X_test,
        "true_label": y_test,
        "pred_nb": pipe_nb.predict(X_test),
        "pred_svc": pipe_svc.predict(X_test),
    }
)
test_out.to_csv(os.path.join(OUTPUT_DIR, "test_predictions.csv"), index=False)

# ------------- 10) Análisis temporal y visualizaciones -------------
# Añadir columna predicted sentiment global (usar mejor modelo según métrica)
# Aquí usamos SVC como "mejor", pero puedes escoger según F1
df["pred_svc"] = pipe_svc.predict(df[TEXT_COL].astype(str))

# 10a) Sentimiento promedio por fecha (si tienes fechas)
if df[DATE_COL].notna().sum() > 0:
    by_date = (
        df.groupby(df[DATE_COL].dt.date)
        .agg(
            {
                "polarity": "mean",
                "sentiment_label": lambda s: s.value_counts().to_dict(),
                "pred_svc": lambda s: s.value_counts().to_dict(),
            }
        )
        .rename_axis("date")
        .reset_index()
    )
    by_date["date"] = pd.to_datetime(by_date["date"])
    # Guardar CSV
    by_date.to_csv(os.path.join(OUTPUT_DIR, "sentiment_by_date.csv"), index=False)

    # Plot polarity over time (rolling mean)
    ts = (
        df.dropna(subset=[DATE_COL])
        .set_index(DATE_COL)
        .resample("D")
        .agg({"polarity": "mean"})
        .fillna(0)
    )
    ts["polarity_rolling7"] = ts["polarity"].rolling(7, min_periods=1).mean()
    plt.figure(figsize=(12, 5))
    plt.plot(ts.index, ts["polarity"], label="Polarity daily")
    plt.plot(ts.index, ts["polarity_rolling7"], label="7-day rolling mean", linewidth=2)
    plt.legend()
    plt.title("Polaridad promedio diaria y tendencia (rolling 7)")
    plt.xlabel("Fecha")
    plt.ylabel("Polarity (TextBlob)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "polarity_timeseries.png"))
    plt.close()

    # Plot counts per sentiment per day (stacked)
    df_temp = df.copy()
    df_temp["day"] = df_temp[DATE_COL].dt.date
    counts = df_temp.groupby(["day", "pred_svc"]).size().unstack(fill_value=0)
    counts.plot(kind="bar", stacked=True, figsize=(12, 5))
    plt.title("Conteo diario por sentimiento (pred_svc)")
    plt.xlabel("Fecha")
    plt.ylabel("Cuenta")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "daily_sentiment_counts.png"))
    plt.close()
else:
    print("No se encontró columna de fechas válida; se omiten gráficos temporales.")

# ------------- 11) Extras: Top features por clase (interpretabilidad) -------------
# Mostrar palabras más informativas para NB (log prob diff)
try:
    vect = pipe_nb.named_steps["tfidf"]
    clf = pipe_nb.named_steps["clf"]
    if hasattr(clf, "coef_") or hasattr(clf, "feature_log_prob_"):
        # MultinomialNB tiene feature_log_prob_
        feat_names = np.array(vect.get_feature_names_out())
        if hasattr(clf, "feature_log_prob_"):
            # cada fila = clase
            for i, cls in enumerate(clf.classes_):
                topn = np.argsort(clf.feature_log_prob_[i])[-20:]
                top_features = feat_names[topn]
                print(f"\nTop términos en clase {cls}:")
                print(", ".join(top_features[::-1]))
except Exception as e:
    print("No se pudo calcular top features:", e)

print("¡Proceso terminado! Revisa la carpeta:", OUTPUT_DIR)
