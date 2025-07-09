# -*- coding: utf-8 -*-

import os
import openai
import feedparser
import telegram
import asyncio
from datetime import datetime
import random
import nest_asyncio
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from collections import defaultdict

nest_asyncio.apply()

OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


openai_client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
bot = telegram.Bot(token=TELEGRAM_TOKEN)

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
        "https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/section/internacional/portada",
        "https://www.eldiario.es/internacional/rss/",
        "https://www.europapress.es/rss/rss.aspx?ch=68",
        "https://www.lavanguardia.com/mvc/feed/rss/internacional.xml"
    ],
    "economia": [
        "https://e00-expansion.uecdn.es/rss/economia.xml",
        "https://www.eleconomista.es/rss/rss-economia.php",
        "https://cincodias.elpais.com/seccion/economia/rss.xml"
    ],
    "wtf": [
        "https://verne.elpais.com/rss/lo-mas.xml",
        "https://www.eldiario.es/caballodenietzsche/rss/",
        "https://www.publico.es/rss/agencias/cultura/",
        "https://nmas1.org/feed"
    ]
}

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

async def generar_bloques_por_categoria(titulares):
    categorias = ["nacional", "internacional", "economia", "wtf"]
    secciones = {
        "nacional": "🇪🇸 Nacional",
        "internacional": "🌍 Internacional",
        "economia": "💰 Economía",
        "wtf": "🫠 WTF"
    }
    bloques_finales = []
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

        if grupos_ordenados:
            bloques_finales.append("━━━━━━━━━━━━━━━")
            bloques_finales.append(f"{secciones[cat]}")
        for grupo in grupos_ordenados:
            resumen = await resumir_grupo(grupo)
            bloques_finales.append(resumen)
            log_texto += f"[{cat.upper()}] Grupo de {len(grupo)} titulares:\n" + "\n".join(f"- {t['titulo']}" for t in grupo) + "\n\n"

    return "\n\n".join(bloques_finales), log_texto

def crear_mensaje_final(cuerpo, momento):
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
    return f"{intro[momento]}\n\n{cuerpo}\n\n🎭 {cierre}"

async def publicar():
    print("📰 Obteniendo titulares...")
    titulares = obtener_titulares()

    hora = datetime.now().hour
    momento = "mañana" if hora < 12 else "tarde" if hora < 20 else "noche"

    print("🧠 Agrupando y resumiendo...")
    cuerpo, log_texto = await generar_bloques_por_categoria(titulares)

    mensaje = crear_mensaje_final(cuerpo, momento)

    cabeceras = {
      "mañana": "https://i.imgur.com/tAJ6WfR.jpg",
      "tarde": "https://i.imgur.com/XMEWksd.jpg",
      "noche": "https://i.imgur.com/z3DcnUs.jpg"
    }

    print("📤 Publicando en Telegram...")
    try:
        await bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=cabeceras[momento])

        if len(mensaje) > 4000:
            print(f"⚠️ Mensaje demasiado largo ({len(mensaje)} caracteres). Recortando...")
            mensaje = mensaje[:3990] + "..."

        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=mensaje, parse_mode="Markdown", disable_web_page_preview=True)

        with open("publicacion.log", "w", encoding="utf-8") as f:
            f.write(log_texto)

        print("✅ Publicado correctamente.")
    except Exception as e:
        print(f"❌ Error al publicar: {e}")

if __name__ == "__main__":
    asyncio.run(publicar())
    try:
        from google.colab import files
        files.download("publicacion.log")
    except:
        pass
