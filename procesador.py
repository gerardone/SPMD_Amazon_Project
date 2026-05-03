import dask.dataframe as dd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import re

# Inicializamos VADER (Mejor que TextBlob para reseñas)
analyzer = SentimentIntensityAnalyzer()

def analizar_sentimiento_texto(texto, rating=None):
    """
    Analizador de texto puro. El rating solo se usa como contexto secundario
    si el texto es demasiado ambiguo.
    """
    if pd.isna(texto) or not isinstance(texto, str) or len(texto.strip()) == 0:
        return "Neutral"
    
    # 1. Análisis de VADER (Detecta "worst", "painful", "ridiculous", etc.)
    # VADER ya entiende negaciones ("not good") e intensificadores ("VERY bad")
    scores = analyzer.polarity_scores(texto)
    compound = scores['compound']
    
    # 2. Refuerzo por palabras clave críticas (Diccionario de ingeniería)
    palabras_negativas = ['worst', 'painful', 'ridiculous', 'broken', 'waste', 'sandpaper', 'terrible', 'awful']
    palabras_positivas = ['excellent', 'perfect', 'amazing', 'love', 'best']
    
    texto_lower = texto.lower()
    
    # Si detectamos palabras extremadamente negativas, forzamos caída de puntaje
    for word in palabras_negativas:
        if word in texto_lower:
            compound -= 0.2
            
    for word in palabras_positivas:
        if word in texto_lower:
            compound += 0.1

    # 3. Lógica de clasificación basada en TEXTO
    # El rating ya NO manda, solo inclina la balanza en casos muy cercanos a cero
    if compound >= 0.05:
        return "Positivo"
    elif compound <= -0.05:
        return "Negativo"
    else:
        # Solo aquí usamos el rating como desempate para lo "Neutral"
        if rating is not None:
            if float(rating) <= 2: return "Negativo"
            if float(rating) >= 4: return "Positivo"
        return "Neutral"

def cargar_datos_masivos(ruta_archivo, sample_size=None):
    try:
        ddf = dd.read_csv(
            ruta_archivo, 
            dtype={'Review Text': 'object', 'Review Title': 'object', 'Rating': 'object'},
            engine='python', on_bad_lines='skip', encoding='utf-8'
        )
        ddf['Rating'] = ddf['Rating'].str.extract('(\d+)', expand=False).astype(float)
        
        if sample_size:
            df_pandas = ddf.head(sample_size).copy()
            df_pandas['Sentimiento'] = df_pandas.apply(
                lambda row: analizar_sentimiento_texto(row['Review Text'], row['Rating']), axis=1
            )
            return df_pandas
        return ddf
    except Exception as e:
        return str(e)

def calcular_sentimientos_masivos(ruta_archivo):
    try:
        ddf = dd.read_csv(
            ruta_archivo, 
            dtype={'Rating': 'object', 'Review Text': 'object'},
            engine='python', on_bad_lines='skip'
        )
        ddf['Rating'] = ddf['Rating'].str.extract('(\d+)', expand=False).astype(float)
        
        def apply_sentiment(df):
            return df.apply(lambda row: analizar_sentimiento_texto(row['Review Text'], row['Rating']), axis=1)
        
        sentimientos = ddf.map_partitions(apply_sentiment).value_counts().compute()
        return sentimientos, len(ddf)
    except Exception as e:
        return None, None


def calcular_metricas_calidad(ruta_archivo):
    try:
        ddf = dd.read_csv(ruta_archivo, dtype={'Rating': 'object', 'Review Text': 'object'}, engine='python', on_bad_lines='skip')
        ddf['Rating'] = ddf['Rating'].str.extract('(\d+)', expand=False).astype(float)

        # Definimos la "Verdad Absoluta" basada en estrellas
        # Asumimos que 1-2 estrellas es Negativo (Clase de interés para la empresa)
        def evaluar_calidad(row):
            real_negativo = row['Rating'] <= 2
            pred_negativo = analizar_sentimiento_texto(row['Review Text'], row['Rating']) == "Negativo"
            
            if real_negativo and pred_negativo: return "TP"
            if not real_negativo and pred_negativo: return "FP"
            return "Other"

        resultados = ddf.map_partitions(lambda df: df.apply(evaluar_calidad, axis=1)).value_counts().compute()
        
        tp = resultados.get("TP", 0)
        fp = resultados.get("FP", 0)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        return precision, tp, fp
    except:
        return 0, 0, 0