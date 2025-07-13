# -*- coding: utf-8 -*-
import os
import json
import openai
import feedparser
import telegram
import asyncio
from datetime import datetime, timedelta
import random
import nest_asyncio
import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict

nest_asyncio.apply()

# Configuración
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

openai_client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
bot = telegram.Bot(token=TELEGRAM_TOKEN)

HISTORIAL_PATH = "historial_noticias.json"

RSS_FEEDS = {
    "nacional": [
        "https://www.abc.es/rss/feeds/abc_Espana.xml",
        "https://www.elmundo.es/rss/espana.xml",
        "https://www.lavanguardia.com/mvc/feed/rss/politica.xml",
        "https://www.eldiario.es/rss/",
        "https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/section/espana/portada",
        "https://rss.elconfidencial.com/espana/"
    ],
    "internacional": [
        "https://feeds.bbci.co.uk/news/world/rss.xml",
        "https://www.nytimes.com/svc/collections/v1/publish/http://www.nytimes.com/section/world/rss.xml",
        "https://english.aljazeera.net/xml/rss/all.xml",
        "https://rss.dw.com/rdf/rss-en-world",
        "https://www.theguardian.com/world/rss"
    ],
    "economia": [
        "https://www.cnbc.com/id/10001147/device/rss/rss.html",
        "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
        "https://www.economist.com/the-world-this-week/rss.xml",
        "https://www.ft.com/?format=rss",
        "https://www.bloomberg.com/feed/podcast/etf-report.xml"
    ],
    "wtf": [
        "https://www.theonion.com/rss",
        "https://not-the-onion.reddit.com/.rss",
        "https://www.huffpost.com/section/weird-news/feed",
        "https://boingboing.net/feed"
    ]
}

# 🔁 Funciones de historial
def cargar_historial():
    if os.path.exists(HISTORIAL_PATH):
        with open(HISTORIAL_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def guardar_historial(historial):
    with open(HISTORIAL_PATH, "w", encoding="utf-8") as f:
        json.dump(historial, f, ensure_ascii=False, indent=2)

def limpiar_historial(historial, dias=3):
    limite = datetime.utcnow() - timedelta(days=dias)
    return [item for item in historial if datetime.fromisoformat(item["fecha"]) > limite]

def similitud_coseno(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def es_noticia_duplicada(embedding_nuevo, historial, umbral=0.85):
    for item in historial:
        if similitud_coseno(embedding_nuevo, item["embedding"]) > umbral:
            return True
    return False

# 🔎 Lectura de noticias
def obtener_titulares():
    entradas = []
    for categoria, urls in RSS_FEEDS.items():
        for url in urls:
            feed = feedparser.parse(url)
            for entrada in feed.entries:
                if 'title' in entrada:
                    entradas.append({
                        "categoria": categoria,
                        "titulo": entrada.title.strip(),
                        "link": entrada.link
                    })
    return entradas

async def obtener_embeddings(titulares):
    textos = [t["titulo"] for t in titulares]
    response = await openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=textos
    )
    return np.array([r.embedding for r in response.data])

def agrupar_por_similitud(embeddings, eps=0.3):
    clustering = DBSCAN(eps=eps, min_samples=1, metric="cosine").fit(embeddings)
    return clustering.labels_

# ✍️ Resumen con GPT
async def resumir_grupo(grupo):
    prompt = """Eres un redactor sarcástico para un canal de Telegram. Resume las siguientes noticias en menos de 4 líneas, combinando un tono serio con ironía. Elige un titular representativo y proporciona un solo enlace.

Formato:
📌 *Título elegido*
📰 Resumen serio + comentario sarcástico
🔗 Enlace

Noticias:
"""
    for t in grupo:
        prompt += f"- {t['titulo']} ({t['link']})\n"
    response = await openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8
    )
    return response.choices[0].message.content.strip()

def crear_intro_y_cierre(momento):
    intro = {
        "mañana": "¡Buenos días! Aquí tienes la edición matinal del *Defecation Journal*, el diario que lees mientras haces lo que nadie más quiere hablar.",
        "tarde": "¡Hora de la pausa! Llega la edición vespertina del *Defecation Journal*, ideal para esos momentos donde el baño es tu sala de prensa privada.",
        "noche": "¡Bienvenidos a la edición nocturna del *Defecation Journal*! El resumen perfecto para cerrar el día mientras reflexionas... en el trono."
    }
    cierre = random.choice([
        "Recuerda: la vida es como el papel higiénico, a veces estás en el rollo... y otras te lo quitan.",
        "Gracias por leer donde más se piensa. ¡Hasta la próxima sentada!",
        "No olvides tirar de la cadena... y de este canal. 💩"
    ])
    return intro[momento], cierre

# 🚀 Publicación
async def publicar():
    print("📰 Obteniendo titulares...")
    titulares = obtener_titulares()
    historial = limpiar_historial(cargar_historial())

    hora = datetime.now().hour
    momento = "mañana" if hora < 12 else "tarde" if hora < 20 else "noche"
    intro, cierre = crear_intro_y_cierre(momento)

    cabeceras = {
        "mañana": "https://i.postimg.cc/PfvY2z7H/edicion-matinal.png",
        "tarde":  "https://i.postimg.cc/K88YsmYT/edicion-vespertina.png",
        "noche":  "https://i.postimg.cc/TPWfZ1vK/edicion-nocturna.png"
    }

    try:
        await bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=cabeceras[momento], disable_notification=True)
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=intro, parse_mode="Markdown", disable_web_page_preview=True, disable_notification=True)

        categorias = ["nacional", "internacional", "economia", "wtf"]
        secciones = {
            "nacional": "**🇪🇸 NACIONAL**",
            "internacional": "**🌍 INTERNACIONAL**",
            "economia": "**💰 ECONOMÍA**",
            "wtf": "**🫠 WTF**"
        }

        log_texto = ""
        for cat in categorias:
            subtitulares = [t for t in titulares if t['categoria'] == cat]
            if not subtitulares:
                continue

            embeddings = await obtener_embeddings(subtitulares)
            labels = agrupar_por_similitud(embeddings)

            grupos = defaultdict(list)
            for idx, etiqueta in enumerate(labels):
                grupos[etiqueta].append(subtitulares[idx])

            grupos_ordenados = sorted(grupos.values(), key=lambda g: -len(g))[:3]
            if not grupos_ordenados:
                continue

            bloque = f"{secciones[cat]}\n\n"
            for grupo in grupos_ordenados:
                titulo_representativo = grupo[0]["titulo"]
                emb_resp = await openai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=titulo_representativo
                )
                emb_vector = emb_resp.data[0].embedding

                if es_noticia_duplicada(emb_vector, historial):
                    continue

                resumen = await resumir_grupo(grupo)
                bloque += resumen + "\n\n"
                log_texto += f"[{cat.upper()}] Grupo de {len(grupo)} titulares:\n" + "\n".join(f"- {t['titulo']}" for t in grupo) + "\n\n"

                historial.append({
                    "fecha": datetime.utcnow().isoformat(),
                    "titulo": titulo_representativo,
                    "embedding": emb_vector
                })

            if bloque.strip() != secciones[cat]:
                await bot.send_message(
                    chat_id=TELEGRAM_CHAT_ID,
                    text=bloque.strip(),
                    parse_mode="Markdown",
                    disable_web_page_preview=True,
                    disable_notification=True
                )

        await bot.send_message(
            chat_id=TELEGRAM_CHAT_ID,
            text=f"🎭 {cierre}",
            parse_mode="Markdown",
            disable_web_page_preview=True
        )

        guardar_historial(historial)

        with open("publicacion.log", "w", encoding="utf-8") as f:
            f.write(log_texto)

        print("✅ Publicado correctamente.")
    except Exception as e:
        print(f"❌ Error al publicar: {e}")

if __name__ == "__main__":
    asyncio.run(publicar())
