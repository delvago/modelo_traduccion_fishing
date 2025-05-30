import streamlit as st
#from agents.cyber_agent.run_agent import extecute
#import requests
#import json

st.set_page_config(page_title="Cyber Agent", layout="wide", initial_sidebar_state="auto", 
                   page_icon="https://cdn-icons-png.flaticon.com/512/8522/8522214.png")

# --- Configuración de la página ---
# Código CSS para ajustar el padding de la página
st.markdown("""
    <style>
           .block-container {
                padding-top: 2rem; /* Ajusta este valor según necesites */
                padding-bottom: 0rem;
                padding-left: 5rem;
                padding-right: 5rem;
            }
    </style>
    """, unsafe_allow_html=True)

col1, col2 = st.columns([1, 10])
#Icono de la aplicación
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/8522/8522214.png", width=80)
#Título principal de la aplicación
with col2:
    st.title("Cyber Agent")
    
st.markdown("""Bienvenido a Cyber Agent, tu asistente para temas de ciberseguridad.""")
st.markdown("""Con esta herramienta puedes generar correos o inspeccionar tus correos electrónicos (de Gmail) en busca de Phishing (por ahora solo en inglés).""")
#st.markdown("---")
st.divider()

# --- Inicialización del historial del chat ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    
# --- Mostrar mensajes del historial ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Interfaz de entrada de chat ---
if prompt := st.chat_input("Ingresa tu solicitud al Cyber Agent. Ejemplo: 'Inspecciona el primer mensaje de mi bandeja de entrada de Gmail en busca de Phishing.'"):
    # Añadir mensaje del usuario al historial y mostrarlo
    st.session_state.messages.append({"role":"user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Llamada al agente y respuesta
    with st.chat_message("assistant"):
        with st.spinner("Procesando tu solicitud..."):
            
            try:
                respuesta_agente = f"He recibido tu solicitud: '{prompt}'. Ejemplo de respuesta del agente: 'El correo electrónico es sospechoso de ser phishing.'"
                #respuesta_agente_raw = extecute(prompt)  # Llamar a la función del agente
                #respuesta_agente = respuesta_agente_raw.get("output")
                if respuesta_agente is not None:
                    st.markdown(respuesta_agente)
                    # Respuesta del agente al historial
                    st.session_state.messages.append({"role":"assistant", "content": respuesta_agente})
                else:
                    error_msg = "El agente no produjo una salida ('output') válida."
                    st.error(error_msg)
                    st.session_state.messages.append({"role":"assistant", "content" : error_msg})
            except Exception as e:
                error_msg = f"Ocurrió un error al procesar tu solicitud: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role":"assistant", "content" : error_msg})
with st.sidebar:
    st.header("Acerca de Cyber Agent")
    st.markdown("Version 0.1")
    st.subheader("Enlaces Útiles")
    st.markdown("[Repositorio de GitHub](https://github.com/jcvillaaa/project_bootcam)")
    st.markdown("<div style='margin-top: 340px;'></div>", unsafe_allow_html=True)# Espacio para el boton de limpiar el historial   
    if st.button("Limpiar Historial de Chat"):
        st.session_state.messages = []
        st.rerun()

# --- Código para text box ---

# st.subheader("¿Qué deseas que haga el agente?:")

# textos_input = st.text_area("Ingresa tu solicitud:",
#                             height=300,
#                            placeholder="Ejemplo:\nRevisa el último correo entrante y verifica si es phishing.\nEnvia un correo al xxxx@gmail.com solicitando información del último incidente de seguridad."
#                            )

# if st.button("Envia tu solicitud"):
#     if textos_input:
#         solicitud = textos_input.strip()
#         if not solicitud:
#             st.warning("No se ingresó una solicitud válida para enviar.")
#         else:
#             with st.spinner("Enviando solicitud al agente... Esto puede tardar unos segundos"):     
#                 # LLAMAR LA FUNCIÓN REAL DEL AGENTE AQUÍ
#                 #respuesta = funcion(globals.solicitud)
#                 respuesta = solicitud + " Ejemplo de respuesta del agente: 'El correo electrónico es sospechoso de ser phishing.'"#Eliminar cuando este la función real del agente
#                 st.subheader("Respuesta del agente:")
#                 if respuesta:
#                     st.markdown(f"**Respuesta:** {respuesta}")
#                 else:
#                     st.error("No se pudo obtener una respuesta del agente. Intenta nuevamente más tarde.")
#     elif st.session_state.get("button_clicked_once", False):
#         st.warning("Por favor, ingresa una solicitud para enviar al agente.")
        
    