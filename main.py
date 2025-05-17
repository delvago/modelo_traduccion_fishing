from fastapi import FastAPI, HTTPException # HTTPException para manejar errores
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
from pydantic import BaseModel # Para definir la estructura de los datos de entrada
from typing import List, Dict, Any # Para especificar que esperamos una lista de textos
from torch import argmax, softmax

# --- Carga de los modelos (Se ejecuta una sola vez al iniciar la app) ---

tokenizer_traductor = None
model_traductor = None
tokenizer_phishing = None
model_phishing = None

print("Iniciando carga de modelos...")

# --- Carga modelo de traduccion ---
# Se utiliza el modelo Helsinki-NLP/opus-mt-es-en para traducir de español a ingles
try: 
    print("Cargando modelo de traduccion...")
    tokenizer_traductor = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-es-en")
    model_traductor = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-es-en")
    print("Modelo de traducción cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar el modelo de traducción: {e}")

# --- Carga modelo de deteccion de phishing ---
# Se utiliza el modelo ealvaradob/bert-finetuned-phishing para detectar phishing
try:
    print("Cargando modelo de deteccion de phishing...")
    tokenizer_phishing = AutoTokenizer.from_pretrained("ealvaradob/bert-finetuned-phishing")
    model_phishing = AutoModelForSequenceClassification.from_pretrained("ealvaradob/bert-finetuned-phishing")
    print("Modelo de phishing cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar el modelo de phishing: {e}")

print("Carga de modelos finalizada")

# --- Estructura de la aplicación FastAPI ---
app = FastAPI(
    title="API de deteccion de phishing", 
    version = "0.1.0",
    description = "Una API para detectar phising")

# --- Definición de los datos de entrada (Request Body) ---
# Esto ayuda a FastAPI a validar los datos que recibe la API
class TextosParaTraducir(BaseModel):
    textos: List[str] # Esperamos un JSON con una clave "textos" que sea una lista de strings

class TextosParaPhishing(BaseModel):
    textos: List[str] # Esperamos un JSON con una clave "textos" que sea una lista de strings

# --- Funciones lógicas internas ---
async def traducir_textos_logica(textos_a_traducir: List[str]) -> List[str]:
    if not tokenizer_traductor or not model_traductor:
        print("Error: Modelo de traduccion no disponible en traducir_textos_logica.")
        raise HTTPException(status_code=500, detail="Servicio de traduccion no disponible temporalmente.")
    
    try:
        inputs = tokenizer_traductor(textos_a_traducir, return_tensors="pt", padding=True, truncation=True)
        outputs = model_traductor.generate(**inputs)
        traducciones = tokenizer_traductor.batch_decode(outputs, skip_special_tokens=True)
        return traducciones
    except Exception as e:
        print(f"Error al traducir: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno en el servicio de traducción: {stre(e)}")

async def detectar_phishing_logica(textos_a_analizar: List[str]) -> List[Dict[str, Any]]:
    if not tokenizer_phishing or not model_phishing:
        print("Error: Modelo de phishing no disponible en detectar_phishing_logica.")
        raise HTTPException(status_code=500, detail="Servicio de detección de phishing no disponible temporalmente.")
    
    try:
        inputs = tokenizer_phishing(textos_a_analizar, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = model_phishing(**inputs)
        logits = outputs.logits
        probabilidades = softmax(logits, dim=1)
        predicciones_id = argmax(probabilidades, dim=1)

        lista_resultados = []
        for i in range(len(predicciones_id)):
            pred_id = predicciones_id[i].item()
            # Aseguramos la existencia de id2label
            if not hasattr(model_phishing, 'config') or not hasattr(model_phishing.config, 'id2label'):
                raise HTTPException(status_code=500, detail="Configuración de etiquetas del modelo de phishing no encontradas.")
            
            etiqueta_predicha = model_phishing.config.id2label[pred_id]
            prob_predicha = probabilidades[i, pred_id].item()

            resultado = {
                "texto_traducido": textos_a_analizar[i],
                "prediccion" : "phishing" if etiqueta_predicha.lower() == "phishing" else "benigno",
                "probabilidad_phishing" : round(prob_predicha, 4) if etiqueta_predicha.lower() == "phishing" else round(1 - prob_predicha, 4),
                "etiqueta_modelo" : etiqueta_predicha,
                "confianza_prediccion" : round(prob_predicha, 4)
            }
            lista_resultados.append(resultado)
        return lista_resultados
    except Exception as e:
        print(f"Error al detectar phishing: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno en el servicio de detección de phishing: {str(e)}")

# --- Endpoint Raíz ---
@app.get("/")
async def root():
    return {"message":"Bienvenido a la API de Deteccion de phishing"}

@app.post("/traducir")
async def traducir_endpoint(textos_input: TextosEntrada):
    """
    Traduce una lista de textos del español al inglés.
    """
    if not textos_input.textos:
        raise HTTPException(status_code=400, detail="No se recibieron textos para traducir.")
    
    traducciones = await traducir_textos_logica(textos_input.textos)
    return {"traducciones": traducciones}

@app.post("/detectar-phishing/")
async def detectar_phishing_endpoint(textos_input: TextosParaPhishing):
    """
    Detecta si una lista de textos (en inglés) son phishing.
    """
    if not textos_input.textos:
        raise HTTPException(status_code=400, detail="No se proporcionaron textos para analizar.")

    resultados = await detectar_phishing_logica(textos_input.textos)
    return {"resultados_phishing": resultados}




# @app.post("/traducir")# Usamos POST porque el cliente enviará datos (el texto a traducir)
# async def traducir_textos_endpoint(textos_input: TextosParaTraducir):
#     """
#     Traduce una lista de textos del español al inglés.
#     """
#     if not tokenizer_traductor or not model_traductor:
#         return {"error": "Modelo de traducción no cargado. Revisa los logs del servidor"}
    
#     textos_a_traducir = textos_input.textos # Accedemos a la lista de textos
#     if not textos_a_traducir:
#         return {"error": "No se recibieron textos para traducir."}
    
#     try:
#         print(f"Traduciendo los siguientes textos: {textos_a_traducir}")
#         inputs = tokenizer_traductor(textos_a_traducir, return_tensors="pt", padding=True, truncation=True)#truncation=True para asegurarnos de que no exceda el tamaño máximo
#         outputs = model_traductor.generate(**inputs)
#         traducciones = tokenizer_traductor.batch_decode(outputs, skip_special_tokens=True)
#         print(f"Traducciones: {traducciones}")
#         return {"traducciones": traducciones}
#     except Exception as e:
#         print(f"Error al traducir: {e}")
#         return {"error": f"Ocurrio un error al traducir: {e}"}
    
@app.post("/phishing")
async def phishing_endpoint(textos_input: TextosParaPhishing):
    """
    Detecta si una lista (en ingles) son phishing.
    """
    if not tokenizer_phishing or not model_phishing:
        return {"error": "Modelo de phishing no cargado. Revisa los logs del servidor"}
    
    textos_a_analizar = textos_input.textos
    if not textos_a_analizar:
        return {"error": "No se recibieron textos para detectar phishing."}
    try:
        print(f"Detectando phishing en los siguientes textos: {textos_a_analizar}")
        inputs = tokenizer_phishing(textos_a_analizar, return_tensors="pt", padding=True, truncation=True, max_length=512)#Bert funciona mejor con max_length=512
        outputs = model_phishing(**inputs)
        logits = outputs.logits
        probabilidades = softmax(logits, dim=1) # Aplicamos softmax para obtener probabilidades
        predicciones_id = argmax(probabilidades, dim=1)

        lista_resultados = []
        for i in range(len(predicciones_id)):
            pred_id = predicciones_id[i].item()
            etiqueta_predicha = model_phishing.config.id2label[pred_id]
            prob_predicha = probabilidades[i][pred_id].item()

            resultado = {
                "texto_original": textos_a_analizar[i],
                "prediccion": "phishing" if etiqueta_predicha.lower() == "phishing" else "benigno", # Asegurar minúsculas para la comparación
                "probabilidad": round(prob_predicha, 4)
            }
            lista_resultados.append(resultado)
        
        print(f"Resultados de phishing: {lista_resultados}")
        return {"resultados": lista_resultados}
    
    except Exception as e:
        print(f"Error al detectar phishing: {e}")
        return {"error": f"Ocurrio un error al detectar phishing: {e}"}