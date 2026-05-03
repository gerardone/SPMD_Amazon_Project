import streamlit as st
import pandas as pd
import procesador
import os

st.set_page_config(page_title="SPMD - Amazon Sentiment Analysis", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUTA_ARCHIVO = os.path.join(BASE_DIR, "data", "Amazon_Reviews.csv")

st.title("Sistemas de Procesamiento Masivo de Datos")
st.subheader("Análisis de Sentimiento con Arquitectura Dask")

tab1, tab2 = st.tabs(["📊 Análisis del Dataset", "🔍 Probar Reseña Manual"])

with tab1:
    # Dentro de with tab1:
    st.markdown("""
    Esta aplicación utiliza el **Amazon Reviews Dataset**, un conjunto masivo de datos 
    obtenido de [Kaggle](https://www.kaggle.com/datasets/dongrelaxman/amazon-reviews-dataset). 
    El dataset incluye miles de registros con textos y calificaciones originales que permiten 
    poner a prueba arquitecturas de procesamiento distribuido y minería de textos.
    """)
    
    if st.button("🚀 Ejecutar Procesamiento de Datos"):
        with st.spinner("Procesando 21,214 registros de forma distribuida..."):
            if not os.path.exists(RUTA_ARCHIVO):
                st.error("No se encuentra el dataset en /data")
            else:
                # 1. Obtener métricas masivas (Dask)
                conteo_sent, total = procesador.calcular_sentimientos_masivos(RUTA_ARCHIVO)
                
                # 2. Obtener muestra para la tabla
                df_muestra = procesador.cargar_datos_masivos(RUTA_ARCHIVO, sample_size=100)
                
                if conteo_sent is not None:
                    # Layout de métricas
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total Reseñas", f"{total:,}")
                    c2.success(f"Positivas: {conteo_sent.get('Positivo', 0)}")
                    c3.error(f"Negativas: {conteo_sent.get('Negativo', 0)}")

                    # Gráfica de Sentimientos
                    st.write("### Clasificación General de Sentimiento (Dataset Completo)")
                    st.bar_chart(conteo_sent)

                    st.write("### Muestra de Clasificación Individual (Primeros 100)")
                    st.dataframe(df_muestra[['Rating', 'Sentimiento', 'Review Text']], use_container_width=True)

                    st.write("### Evaluación de Efectividad (Métricas)")
                    precision, tp, fp = procesador.calcular_metricas_calidad(RUTA_ARCHIVO)
                    st.metric("Precisión del Modelo (Clase Negativa)",  f"{precision:.2%}")
                    st.caption(f"Verdaderos Positivos: {tp} | Falsos Positivos: {fp}")  

                    st.write("---")
                    st.write("### 🎯 Evaluación de la Precisión (Métricas SPMD)")
                    with st.status("Evaluando precisión contra Ratings reales...", expanded=False):
                        precision, tp, fp = procesador.obtener_precision_modelo(RUTA_ARCHIVO)
    
                    if tp + fp > 0:
                        col_m1, col_m2, col_m3 = st.columns(3)
                        col_m1.metric("Precisión (Sent. Negativo)", f"{precision:.2%}")
                        col_m2.metric("Verdaderos Negativos (TP)", tp)
                        col_m3.metric("Falsos Negativos (FP)", fp)
                        st.caption("Nota: Se considera 'Verdad Absoluta' las reseñas con 1 y 2 estrellas.") 
                else:
                    st.error("Error al procesar el dataset.")

with tab2:
    st.write("### Simulador de Inferencia NLP")
    st.info("El modelo priorizará el lenguaje detectado (palabras como 'worst' o 'painful') sobre las estrellas.")
    
    user_text = st.text_area("Pega el texto de alguna reseña de Amazon en inglés aquí:", height=150)
    # El rating ahora es opcional/informativo
    rating_opcional = st.selectbox("Calificación original (opcional):", [1, 2, 3, 4, 5], index=0)
    
    if st.button("Analizar con Motor VADER"):
        if user_text:
            # Ejecutamos el análisis
            res = procesador.analizar_sentimiento_texto(user_text, rating=rating_opcional)
            
            st.write("---")
            st.write("#### Resultado del Análisis Lingüístico:")
            if res == "Positivo":
                st.success(f"Sentimiento detectado: **{res}**")
            elif res == "Negativo":
                st.error(f"Sentimiento detectado: **{res}**")
            else:
                st.warning(f"Sentimiento detectado: **{res}**")
        else:
            st.error("Debes ingresar un texto para el análisis de lenguaje natural.")