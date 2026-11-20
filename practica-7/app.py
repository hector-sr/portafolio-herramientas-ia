import os
import gradio as gr
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Cargar el procesador de imágenes y el modelo preentrenado
MODEL_ID = "google/mobilenet_v2_1.0_224"
processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
model.eval()


def clasificar(imagen):
    if imagen is None:
        return {}

    inputs = processor(images=imagen, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    probabilidades = torch.nn.functional.softmax(logits, dim=-1)[0]
    top3_prob, top3_indices = torch.topk(probabilidades, 3)

    resultados = {}
    for prob, idx in zip(top3_prob, top3_indices):
        etiqueta = model.config.id2label[idx.item()]
        resultados[etiqueta] = float(prob.item())

    return resultados


# Obtener ruta absoluta de la carpeta ejemplos/ respecto a app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CARPETA_EJEMPLOS = os.path.join(BASE_DIR, "ejemplos")

# Filtrar solo imágenes con extensiones soportadas
extensiones_validas = (".jpg", ".jpeg", ".png", ".webp")
lista_ejemplos = [
    os.path.join(CARPETA_EJEMPLOS, archivo)
    for archivo in os.listdir(CARPETA_EJEMPLOS)
    if archivo.lower().endswith(extensiones_validas)
]

# Configurar la interfaz de Gradio
interfaz = gr.Interface(
    fn=clasificar,
    inputs=gr.Image(type="pil", label="Sube una imagen"),
    outputs=gr.Label(num_top_classes=3, label="Predicciones principales"),
    title="¿Qué hay en la foto?",
    description=f"Clasificador de imágenes impulsado por el modelo {MODEL_ID}.",
    examples=lista_ejemplos,
)

if __name__ == "__main__":
    interfaz.launch()