import streamlit as st
#import requests
#import json
#import la librería de funciones que contiene la función del agente

st.set_page_config(page_title="Cyber Agent", layout="wide", initial_sidebar_state="auto", 
                   page_icon="https://cdn-icons-png.flaticon.com/512/8522/8522214.png")

# --- Configuración de la página ---
col1, col2 = st.columns([1, 10])
#Icono de la aplicación
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/8522/8522214.png", width=80)

#Título principal de la aplicación
with col2:
    st.title("Cyber Agent")
    
st.markdown("""Bienvenido a Cyber Agent, tu asistente para temas de ciberseguridad.""")
st.markdown("""Con esta herramienta puedes generar correos o inspeccionar tus correos electrónicos (de Gmail) en busca de Phishing (por ahora solo en inglés).""")


st.subheader("¿Qué deseas que haga el agente?:")

textos_input = st.text_area("Ingresa tu solicitud:",
                            height=300,
                           placeholder="Ejemplo:\nRevisa el último correo entrante y verifica si es phishing.\nEnvia un correo al xxxx@gmail.com solicitando información del último incidente de seguridad."
                           )

if st.button("Envia tu solicitud"):
    if textos_input:
        solicitud = textos_input.strip()
        if not solicitud:
            st.warning("No se ingresó una solicitud válida para enviar.")
        else:
            with st.spinner("Enviando solicitud al agente... Esto puede tardar unos segundos"):     
                # LLAMAR LA FUNCIÓN REAL DEL AGENTE AQUÍ
                #respuesta = funcion(globals.solicitud)
                respuesta = solicitud + " Ejemplo de respuesta del agente: 'El correo electrónico es sospechoso de ser phishing.'"#Eliminar cuando este la función real del agente
                st.subheader("Respuesta del agente:")
                if respuesta:
                    st.markdown(f"**Respuesta:** {respuesta}")
                else:
                    st.error("No se pudo obtener una respuesta del agente. Intenta nuevamente más tarde.")
    elif st.session_state.get("button_clicked_once", False):
        st.warning("Por favor, ingresa una solicitud para enviar al agente.")