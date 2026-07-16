# 1. IMPORTS Y CONFIGURACIÓN DE LA PÁGINA (PRIMERO)

import streamlit as st
st.set_page_config(page_title="Asistente TechStore Perú", page_icon="💻")

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 2. FUNCIÓN PARA PREPARAR EL AGENTE (CON CACHÉ)

@st.cache_resource
def preparar_agente(ruta_pdf, key):
    """
    Carga el PDF, genera embeddings, crea el vector store y configura el chain de recuperación.
    """
    try:
       
        if not os.path.exists(ruta_pdf):
            st.error(f"No se encontró el archivo PDF en la ruta: {ruta_pdf}")
            return None

       
        lector = PyPDFLoader(ruta_pdf)
        paginas = lector.load()
        if not paginas:
            st.error("El PDF está vacío o no se pudo cargar correctamente.")
            return None

        separador = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        fragmentos = separador.split_documents(paginas)

       
        embeddings = GoogleGenerativeAIEmbeddings(
            model="text-embedding-004",  
            google_api_key=key
        )

    
        base_de_datos = FAISS.from_documents(fragmentos, embeddings)

       
        buscador = base_de_datos.as_retriever(search_kwargs={"k": 3})

       
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.2,
            google_api_key=key
        )

      
        instrucciones = (
            "Eres un asesor de ventas y soporte técnico amable de la tienda 'TechStore Perú'. "
            "Responde a las preguntas del cliente utilizando ÚNICAMENTE la información del catálogo "
            "que se te proporciona a continuación. Si te preguntan por precios, indícalos siempre en Soles (S/). "
            "Si no encuentras la respuesta en el documento, di amablemente que no posees esa información en el catálogo actual.\n\n"
            "Catálogo:\n{context}"
        )
        plantilla_prompt = ChatPromptTemplate.from_messages([
            ("system", instrucciones),
            ("human", "{input}"),
        ])

       
        cadena_respuestas = create_stuff_documents_chain(llm, plantilla_prompt)
        agente_final = create_retrieval_chain(buscador, cadena_respuestas)

        return agente_final

    except Exception as e:
      
        st.error(f"❌ Error al preparar el agente: {e}")
       
        st.exception(e)
        return None


# 3. FLUJO PRINCIPAL DE LA APLICACIÓN CON MANEJO DE ERRORES

try:
    
    PDF_PATH = "documento.pdf"

   
    api_key = None
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
        os.environ["GOOGLE_API_KEY"] = api_key
    elif "GOOGLE_API_KEY" in os.environ:
        api_key = os.environ["GOOGLE_API_KEY"]

   
    st.title("💻 Asistente Virtual - TechStore Perú")
    st.write(
        "¡Hola! Soy tu asesor tecnológico. Pregúntame sobre especificaciones, "
        "precios en soles (S/), garantías o envíos de nuestros productos."
    )

   
    if not os.path.exists(PDF_PATH):
        st.error(f"❌ No se encontró el archivo '{PDF_PATH}' en el repositorio. Por favor, súbelo a la raíz del proyecto.")
        st.stop() 

    
    if not api_key:
        st.error("❌ No se encontró la clave GOOGLE_API_KEY. Configúrala en los Secrets de Streamlit (Settings → Secrets).")
        st.stop()

   
    with st.spinner("Cargando catálogo y preparando el asistente... Espera unos segundos."):
        mi_agente = preparar_agente(PDF_PATH, api_key)

   
    if mi_agente is None:
        st.error("❌ El asistente no pudo inicializarse. Revisa los mensajes de error anteriores.")
        st.stop()

    
    pregunta = st.text_input("✏️ Escribe tu consulta sobre laptops, PC o accesorios aquí:")

    if pregunta:
        with st.spinner("Consultando el catálogo de TechStore Perú..."):
            try:
                resultado = mi_agente.invoke({"input": pregunta})
                st.write("### 🤖 Respuesta del Asesor:")
                st.write(resultado["answer"])
            except Exception as e:
                st.error(f"❌ Error al procesar tu consulta: {e}")
                st.exception(e)

except Exception as e:
    
    st.error(f"🚨 Error crítico en la aplicación: {e}")
    st.exception(e)
