import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time
import json
import re

# Opciones
OUTPUT_TXT = "genz_mexico_tendencias.txt"
OUTPUT_JSON = "genz_mexico_tendencias.json"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
SLEEP_BETWEEN_REQUESTS = 0.4
CRAWL_PAGES = 200
MAX_PARAGRAPH_LENGTH = 3000

KEYWORDS = [
    "generación z",
    "generacion z",
    "gen z",
    "jóvenes",
    "jovenes",
    "juventud",
    "adolescentes",
    "tiktok",
    "viral",
    "tendencia",
    "mexico",
    "méxico",
]

session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})


# -------------------------------------------------------
# SCORING AVANZADO
# -------------------------------------------------------
def score_relevance(text):
    KEY_RELEVANT = [
        "gen z",
        "generación z",
        "generacion z",
        "genz",
        "jovenes",
        "jóvenes",
        "juventud",
        "adolescentes",
        "tiktok",
        "viral",
        "influencer",
        "tendencias",
        "mexico",
        "méxico",
        "cdmx",
        "estudiantes",
        "escuela",
        "universidad",
        "protesta",
        "marcha",
        "redes sociales",
    ]
    txt = text.lower()
    return sum(1 for k in KEY_RELEVANT if k in txt)


# -------------------------------------------------------
# FILE WRITERS
# -------------------------------------------------------
def write_txt(source, url, text):
    with open(OUTPUT_TXT, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"FUENTE: {source}\nURL: {url}\n")
        f.write("=" * 60 + "\n\n")
        f.write(text + "\n")


def write_json(obj):
    with open(OUTPUT_JSON, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + ",\n")


# -------------------------------------------------------
# UTILIDADES
# -------------------------------------------------------
def clean(t):
    return re.sub(r"\s+", " ", t).strip() if t else ""


def title_relevant(title):
    t = title.lower()
    return any(k in t for k in KEYWORDS)


# -------------------------------------------------------
# SCRAPEAR ARTÍCULO COMPLETO
# -------------------------------------------------------
def scrape_full_article(url, source):
    print(f"  → Artículo: {url}")
    try:
        time.sleep(SLEEP_BETWEEN_REQUESTS)
        r = session.get(url, timeout=10)
        soup = BeautifulSoup(r.text, "lxml")

        paragraphs = soup.find_all("p")
        text = " ".join(
            p.get_text(" ", strip=True) for p in paragraphs if len(p.get_text()) > 25
        )

        text = clean(text)
        score = score_relevance(text)

        if score < 2:
            print(f"    [X] Irrelevante (score={score})")
            return

        print(f"    ✔ Guardado (score={score})")

        if len(text) > MAX_PARAGRAPH_LENGTH:
            text = text[:MAX_PARAGRAPH_LENGTH] + "..."

        write_txt(source, url, text)
        write_json({"source": source, "url": url, "text": text})

    except Exception as e:
        print("    [ERROR]:", e)


# -------------------------------------------------------
# EXTRAER TITULARES DESDE RESULTADOS DE BÚSQUEDA
# -------------------------------------------------------
def extract_titles_from_search(url, source):
    try:
        time.sleep(SLEEP_BETWEEN_REQUESTS)
        r = session.get(url, timeout=10)
        soup = BeautifulSoup(r.text, "lxml")

        titles = []
        for a in soup.find_all("a", href=True):
            txt = a.get_text(" ", strip=True)
            if not txt:
                continue
            if title_relevant(txt):
                full_url = urljoin(url, a["href"])
                titles.append((txt, full_url))

        print(f"  → {len(titles)} titulares relevantes")
        return titles
    except:
        return []


# -------------------------------------------------------
# BÚSQUEDAS REALES EN CADA PERIÓDICO
# -------------------------------------------------------
def scrape_many_news():
    fuentes = {
        "CNN México": "https://cnnespanol.cnn.com/?s=generacion+z",
        "El País": "https://elpais.com/busca/?q=generación%20z",
        "BBC Mundo": "https://www.bbc.com/mundo/search?q=generación+z",
        "Milenio": "https://www.milenio.com/buscar?q=generacion+z",
        "El Universal": "https://www.eluniversal.com.mx/resultados-busqueda?search=generacion+z",
        "Proceso": "https://www.proceso.com.mx/?s=generacion+z",
        "Infobae": "https://www.infobae.com/buscar/?q=generacion+z",
        "Expansión": "https://expansion.mx/search?search_api_fulltext=generacion+z",
        "Aristegui": "https://aristeguinoticias.com/?s=generacion+z",
        "Animal Político": "https://www.animalpolitico.com/?s=generacion+z",
        "Forbes": "https://www.forbes.com.mx/?s=generacion+z",
        "UnoTV": "https://www.unotv.com/buscar/?q=generacion+z",
        "Publimetro": "https://www.publimetro.com.mx/?s=generacion+z",
        "El Financiero": "https://www.elfinanciero.com.mx/buscar/?q=generacion+z",
    }

    print("\n[+] Scrapeando búsquedas dedicadas…\n")

    for name, url in fuentes.items():
        print(f"\n=== {name} ===")
        titles = extract_titles_from_search(url, name)
        for title, link in titles:
            print(f"  - {title}")
            scrape_full_article(link, name)


# -------------------------------------------------------
# DUCKDUCKGO (MUCHO MÁS CONTENIDO)
# -------------------------------------------------------
def deep_crawl(query="generacion z mexico"):
    print("\n[+] DuckDuckGo profundo…\n")
    base = "https://html.duckduckgo.com/html/"

    for page in range(1, CRAWL_PAGES + 1):
        print(f"--- Página {page}/{CRAWL_PAGES} ---")

        try:
            time.sleep(SLEEP_BETWEEN_REQUESTS)
            payload = {"q": query, "s": str((page - 1) * 30)}
            r = session.post(base, data=payload, timeout=10)
            soup = BeautifulSoup(r.text, "lxml")

            for a in soup.select("a.result__a"):
                title = a.get_text(" ", strip=True)
                link = a.get("href")

                if not link:
                    continue

                if title_relevant(title):
                    print(f"  ✔ {title}")
                    scrape_full_article(link, "Crawl")

        except Exception as e:
            print("  [ERROR DDG]:", e)


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
def run_all():
    with open(OUTPUT_TXT, "w", encoding="utf-8"):
        pass
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        f.write("[\n")

    t0 = time.time()

    scrape_many_news()
    deep_crawl()

    with open(OUTPUT_JSON, "a", encoding="utf-8") as f:
        f.write("]")

    print("\n[✔ COMPLETADO]")
    print("Tiempo total:", round(time.time() - t0, 1), "seg")


if __name__ == "__main__":
    run_all()
