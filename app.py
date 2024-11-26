import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import spacy
import re
import unicodedata
import tensorflow as tf
from tensorflow.keras.models import load_model

# Cargar modelo y vocabulario
@st.cache_resource
def load_model_and_vocab():
    try:
        modelo = load_model("./sinisters_glove 1.h5")
        with open('./vocabulary.txt', 'r') as f:
            vocab = f.read().splitlines()
        text_vectorization_layer = modelo.layers[0]
        text_vectorization_layer.set_vocabulary(vocab)
        return modelo, vocab
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None, None

modelo, vocab = load_model_and_vocab()

# Cargar Spacy
nlp = spacy.load("es_core_news_sm")

# Preprocesamiento
def remove_stopwords(words, stopwords):
    for word in stopwords:
        token = " " + word + " "
        words = re.sub(token, " ", words)
    return words

def preproccesing(words):
    words = words.lower()
    words = re.sub(r"\d+", "", words)
    words = re.sub(r"([\"(),¡!¿?:;'></]|\\s)+", "", words)
    words = unicodedata.normalize("NFKD", words).encode("ascii", "ignore").decode("utf-8", "ignore")
    words = remove_stopwords(words, stopwords=nlp.Defaults.stop_words)
    return words

def spacy_fix(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])

def predecir_causas(dataframe, modelo):
    textos = dataframe.iloc[:, 0].astype(str).tolist()
    textos_preprocesados = [preproccesing(spacy_fix(text)) for text in textos]
    secuencias = tf.convert_to_tensor(textos_preprocesados, dtype=tf.string)
    predicciones = modelo.predict(secuencias)
    causas = predicciones.argmax(axis=1)
    return causas

# Interfaz Streamlit
st.title("Clasificación de Siniestros Viales")

uploaded_file = st.file_uploader("Sube un archivo Excel con los textos en la primera columna", type=["xlsx"])
if uploaded_file:
    data = pd.read_excel(uploaded_file)
    st.write("Vista previa del archivo cargado:")
    st.dataframe(data.head())

    if st.button("Realizar predicciones"):
        if modelo:
            with st.spinner("Procesando..."):
                try:
                    data["Causa"] = predecir_causas(data, modelo)
                    st.success("Predicciones realizadas correctamente.")
                    st.write("Resultados:")
                    st.dataframe(data)

                    # Gráfico de distribución
                    st.subheader("Distribución de Causas")
                    distribucion = data["Causa"].value_counts()
                    fig, ax = plt.subplots()
                    distribucion.plot(kind="bar", ax=ax, color="skyblue", title="Distribución de Causas")
                    ax.set_xlabel("Causa")
                    ax.set_ylabel("Frecuencia")
                    st.pyplot(fig)

                    # Botón para descargar resultados
                    output_filename = "resultados_clasificacion.xlsx"
                    data.to_excel(output_filename, index=False)
                    with open(output_filename, "rb") as file:
                        st.download_button(
                            label="Descargar resultados",
                            data=file,
                            file_name=output_filename,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                except Exception as e:
                    st.error(f"Error durante las predicciones: {e}")
        else:
            st.error("Modelo no cargado. Por favor verifica.")
