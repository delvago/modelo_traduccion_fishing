import streamlit as st
import requests # Para hacer peticiones HTTP a FastAPI
import json # Para manejar el payload y la respuesta

#Titulo de la aplicación
st.set_page_config(page_title="Detección de Phishing", layout="wide", initial_sidebar_state="auto")

#Título principal de la aplicación
st.title("Herramienta de detección de Phishing")

#Descripción de la aplicación
st.markdown("""Bienvenido a la herramienta de detección de phishing. Ingresa el contenido de los correos electrónicos que quieres analizar
            y esta herramienta te indicará si son phishing o no.""")

#URL de la API FastAPI
API_URL = "http://localhost:8000/analizar-textos/"

# --- Seccion de entrada de texto ---
st.subheader("Ingresa el contenido a analizar")

textos_input= st.text_area("Texto a analizar; un texto por línea:", 
                           height=300,
                           placeholder="Ejemplo:\nFelicidades, has ganado un premio. Haz clic aquí para reclamarlo.\nHola, necesitamos verificar tu cuenta. Por favor, ingresa tus datos en este enlace.")

# --- Botón para enviar la solicitud ---
if st.button("Analizar textos"):
    if textos_input:
        textos_originales = [line.strip() for line in textos_input.split("\n") if line.strip()]

        if not textos_originales:
            st.warning("No se ingresaron textos válidos para analizar.")
        else:
            with st.spinner("Analizando... Esto puede tardar unos segundos"):
                payload = {"textos": textos_originales}
                try: 
                    # Solicitud POST a la API FastAPI
                    response = requests.post(API_URL, json=payload, timeout=120) # Debido a lo que pueden demorar los modelos se usa un timeout alto
                    
                    # Verifica si la solicitud a la API fue exitosa.
                    response.raise_for_status()  # Lanza una excepción para errores HTTP

                    # Resultados
                    resultados_api = response.json()

                    # --- Mostrar resultados ---
                    st.subheader("Resultados del análisis: ")

                    if "analisis_completo" in resultados_api and resultados_api["analisis_completo"]:
                        for i, item in enumerate(resultados_api["analisis_completo"]):
                            st.markdown(f"---")
                            st.markdown(f"texto original: {i+1}")
                            st.text_area(f"Original{i+1}", item.get('texto_original_es', 'N/A'), height=70, disabled=True, key=f"orig_es_{i}")#key Es un argumento de palabra clave crucial cuando se crean widgets dentro de un bucle en Streamlit.
                            analisis = item.get("analisis_phishing", {})
                            prediccion = analisis.get("prediccion", "N/A")
                            confianza = analisis.get("confianza", 0) * 100 # Convertir a porcentaje

                            st.markdown(f"-- Analisis de phishing ---")
                            if prediccion.lower() == "phishing":
                                st.error(f"Predicción: Phishing (Confianza : {confianza:.2f}%)")
                            elif prediccion.lower() == "benigno":
                                st.success(f"Predicción: Benigno (Confianza : {confianza:.2f}%)")
                            else:
                                st.info(f"Predicción: {prediccion} (Confianza : {confianza:.2f}%)")
                            
                            # Mostrar el resultado completo del análisis si se desea
                            with st.expander("Ver detalles completos del análisis"):
                                st.json(analisis)
                        st.markdown(f"---")
                    else:
                        st.error("No se recibieron resultados válidos de la API.")
                        st.json(resultados_api) #Respuesta cruda de la API para depuración
                except requests.exceptions.HTTPError as errh:
                    st.error(f"Error de HTTP al contactar la API: {errh}")
                    try:
                        error_detail = response.json()
                        st.json({"detalle_error_api": error_detail})
                    except json.JSONDecodeError:
                        st.text(f"Respuesta cruda del error: {response.text}")
                except requests.exceptions.ConnectionError as errc:
                    st.error(f"Error de Conexión: No se pudo conectar a la API en {API_URL}. Asegúrate de que la API FastAPI esté corriendo.")
                except requests.exceptions.Timeout as errt:
                    st.error(f"Error de Timeout: La solicitud a la API tardó demasiado tiempo. Intenta nuevamente.")
                except requests.exceptions.RequestException as err:
                    st.error(f"Error inesperado: {err}")
                except json.JSONDecodeError:
                    st.error("Error al decodificar la respuesta JSON de la API.")
                    st.text(f"Contenido de la respuesta cruda: {response.text if 'response' in locals() else 'No response object'}")
    elif st.session_state.get('button_clicked_once', False):
        st.warning("Por favor, ingresa algún texto para analizar.")

    if "button_clicked_once" not in st.session_state:
        st.session_state.button_clicked_once = True
    if st.button:
        st.session_state.button_clicked_once = True