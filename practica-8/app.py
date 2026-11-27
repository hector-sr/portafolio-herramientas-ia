import os
import time
import gradio as gr
import spaces
from transformers import pipeline

DISPOSITIVO = "cuda" if os.getenv("SPACES_ZERO_GPU") else "cpu"

# Cargar el pipeline de clasificación de imágenes
MODEL_ID = "google/mobilenet_v2_1.0_224"
clasificador = pipeline("image-classification", model="google/mobilenet_v2_1.0_224", device=DISPOSITIVO)


@spaces.GPU
def clasificar(imagen):
    if imagen is None:
        return {}, ""

    # Iniciar medición de tiempo
    inicio = time.perf_counter()

    # Ejecutar inferencia y obtener las 3 mejores predicciones
    predicciones = clasificador(imagen, top_k=3)

    # Detener medición de tiempo
    fin = time.perf_counter()
    tiempo_total = f"{fin - inicio:.2f} segundos"

    # Convertir la salida del pipeline [{'label': ..., 'score': ...}] al formato de gr.Label
    resultados = {item["label"]: float(item["score"]) for item in predicciones}

    return resultados, tiempo_total


# Obtener ruta absoluta de la carpeta ejemplos/ respecto a app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CARPETA_EJEMPLOS = os.path.join(BASE_DIR, "ejemplos")

# Filtrar solo imágenes con extensiones soportadas
extensiones_validas = (".jpg", ".jpeg", ".png", ".webp")
lista_ejemplos = (
    [
        os.path.join(CARPETA_EJEMPLOS, archivo)
        for archivo in os.listdir(CARPETA_EJEMPLOS)
        if archivo.lower().endswith(extensiones_validas)
    ]
    if os.path.exists(CARPETA_EJEMPLOS)
    else []
)

# Configurar la interfaz de Gradio con dos salidas
demo = gr.Interface(
    fn=clasificar,
    inputs=gr.Image(type="pil", label="Sube una imagen"),
    outputs=[
        gr.Label(num_top_classes=3, label="Predicciones principales"),
        gr.Textbox(label="Tiempo de inferencia"),
    ],
    title="¿Qué hay en la foto?",
    description=f"Clasificador de imágenes impulsado por el modelo {MODEL_ID}.",
    examples=lista_ejemplos,
    cache_examples=False,
)

if __name__ == "__main__":
    demo.launch()