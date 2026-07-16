import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import pickle
import time

# ========== CONFIGURACIÓN DE PÁGINA ==========
st.set_page_config(page_title="Asistente TechStore Perú", page_icon="💻")

@st.cache_resource
def preparar_agente(ruta_pdf, key):
    try:
        # 1. Validar que el PDF existe
        if not os.path.exists(ruta_pdf):
            st.error("No se encontró el PDF.")
            return None

        # 2. Cargar solo las primeras 15 páginas (ajusta según necesites)
        lector = PyPDFLoader(ruta_pdf)
        paginas = lector.load()[:15]  # <--- LIMITAR PÁGINAS
        st.info(f"Procesando {len(paginas)} páginas del catálogo...")

        # 3. Fragmentar con chunk_size más grande
        separador = RecursiveCharacterTextSplitter(
            chunk_size=2000,  # <--- AUMENTADO
            chunk_overlap=200
        )
        fragmentos = separador.split_documents(paginas)
        st.info(f"Generando {len(fragmentos)} fragmentos de texto...")

        # 4. Configurar embeddings con timeout
        embeddings = GoogleGenerativeAIEmbeddings(
            model="text-embedding-004",  # o "embedding-001" si es más rápido
            google_api_key=key,
            request_options={
                "timeout": 120  # 2 minutos
            }
        )

        # 5. Generar vectorstore en lotes para evitar timeout
        batch_size = 50
        base_de_datos = None
        
        for i in range(0, len(fragmentos), batch_size):
            batch = fragmentos[i:i+batch_size]
            st.info(f"Procesando lote {i//batch_size + 1}/{len(fragmentos)//batch_size + 1}...")
            
            if base_de_datos is None:
                base_de_datos = FAISS.from_documents(batch, embeddings)
            else:
                temp_db = FAISS.from_documents(batch, embeddings)
                base_de_datos.merge_from(temp_db)
            
            # Pequeña pausa para no saturar la API
            time.sleep(1)

        # 6. Crear buscador
        buscador = base_de_datos.as_retriever(search_kwargs={"k": 3})

        # 7. Configurar LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.2,
            google_api_key=key
        )

        # 8. Prompt y cadenas
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

        st.success("✅ Asistente preparado exitosamente!")
        return agente_final

    except Exception as e:
        st.error(f"❌ Error al preparar el agente: {e}")
        return None

# ========== FLUJO PRINCIPAL ==========
try:
    # 1. Configurar variables
    PDF_PATH = "documento.pdf"
    api_key = None

    # 2. Obtener clave API
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
        os.environ["GOOGLE_API_KEY"] = api_key
    elif "GOOGLE_API_KEY" in os.environ:
        api_key = os.environ["GOOGLE_API_KEY"]

    # 3. Título y presentación
    st.title("💻 Asistente Virtual - TechStore Perú")
    st.write("¡Hola! Soy tu asesor tecnológico. Pregúntame sobre especificaciones, precios en soles (S/), garantías o envíos de nuestros productos.")

    # 4. Validaciones
    if not os.path.exists(PDF_PATH):
        st.error(f"❌ No se encontró el archivo '{PDF_PATH}'. Asegúrate de subirlo al repositorio.")
        st.stop()

    if not api_key:
        st.error("❌ No se encontró la clave GOOGLE_API_KEY. Configúrala en los Secrets de Streamlit.")
        st.stop()

    # 5. Preparar agente (con caché)
    with st.spinner("Cargando catálogo y preparando el asistente..."):
        mi_agente = preparar_agente(PDF_PATH, api_key)

    if mi_agente is None:
        st.error("❌ El asistente no pudo inicializarse. Revisa los mensajes de error anteriores.")
        st.stop()

    # 6. Interfaz de usuario
    pregunta = st.text_input("Escribe tu consulta sobre laptops, PC o accesorios aquí:")

    if pregunta:
        with st.spinner("Consultando el catálogo de TechStore Perú..."):
            resultado = mi_agente.invoke({"input": pregunta})
            st.write("### 🤖 Respuesta del Asesor:")
            st.write(resultado["answer"])

except Exception as e:
    st.error(f"🚨 Error crítico en la aplicación: {e}")
    st.exception(e)
