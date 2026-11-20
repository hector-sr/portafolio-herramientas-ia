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

    # Preprocesar la imagen
    inputs = processor(images=imagen, return_tensors="pt")

    # Inferencia sin calcular gradientes
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Calcular probabilidades con Softmax
    probabilidades = torch.nn.functional.softmax(logits, dim=-1)[0]

    # Obtener los 3 índices con mayor probabilidad
    top3_prob, top3_indices = torch.topk(probabilidades, 3)

    # Formatear el resultado como diccionario {etiqueta: probabilidad_normalizada}
    resultados = {}
    for prob, idx in zip(top3_prob, top3_indices):
        etiqueta = model.config.id2label[idx.item()]
        resultados[etiqueta] = float(prob.item())

    return resultados


# Configurar la interfaz de Gradio
interfaz = gr.Interface(
    fn=clasificar,
    inputs=gr.Image(type="pil", label="Sube una imagen"),
    outputs=gr.Label(num_top_classes=3, label="Predicciones principales"),
    title="¿Qué hay en la foto?",
    description=f"Clasificador de imágenes impulsado por el modelo {MODEL_ID}.",
)

if __name__ == "__main__":
    interfaz.launch()