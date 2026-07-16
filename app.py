import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import time

# ========== CONFIGURACIÓN DE PÁGINA ==========
st.set_page_config(page_title="Asistente TechStore Perú", page_icon="💻")

# ========== FUNCIÓN PARA CREAR EMBEDDINGS POR LOTES ==========
def crear_embeddings_por_lotes(textos, embeddings_model, batch_size=10):
    """
    Genera embeddings en lotes para evitar timeouts.
    """
    todas_las_embeddings = []
    total = len(textos)
    
    for i in range(0, total, batch_size):
        lote = textos[i:i + batch_size]
        st.info(f"Procesando lote {i//batch_size + 1}/{(total + batch_size - 1)//batch_size}...")
        
        # Intentar hasta 3 veces por lote (por si hay timeout intermitente)
        for intento in range(3):
            try:
                embeddings_lote = embeddings_model.embed_documents(lote)
                todas_las_embeddings.extend(embeddings_lote)
                break  # Si funciona, salir del bucle de reintentos
            except Exception as e:
                if intento < 2:
                    st.warning(f"Error en lote {i//batch_size + 1}, reintentando ({intento+1}/3)...")
                    time.sleep(2)  # Esperar 2 segundos antes de reintentar
                else:
                    raise e  # Si falla 3 veces, lanzar la excepción
    
    return todas_las_embeddings

# ========== FUNCIÓN PREPARAR AGENTE ==========
@st.cache_resource
def preparar_agente(ruta_pdf, key):
    try:
        # Verificar que el PDF existe
        if not os.path.exists(ruta_pdf):
            st.error("No se encontró el PDF.")
            return None

        # Cargar el PDF y limitar páginas si es necesario
        lector = PyPDFLoader(ruta_pdf)
        paginas = lector.load()
        
        # Si el PDF tiene más de 20 páginas, procesar solo las primeras 20 (para evitar timeout)
        if len(paginas) > 20:
            st.warning(f"El PDF tiene {len(paginas)} páginas. Procesando solo las primeras 20 para evitar demoras...")
            paginas = paginas[:20]
        
        # Fragmentar el documento
        separador = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        fragmentos = separador.split_documents(paginas)
        
        # Mostrar progreso
        st.info(f"Generando embeddings para {len(fragmentos)} fragmentos...")
        
        # Inicializar el modelo de embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="text-embedding-004",
            google_api_key=key
        )
        
        # --- PROCESAMIENTO POR LOTES PARA EVITAR TIMEOUT ---
        # Extraer los textos de los fragmentos
        textos = [doc.page_content for doc in fragmentos]
        
        # Generar embeddings por lotes
        try:
            todas_las_embeddings = crear_embeddings_por_lotes(textos, embeddings, batch_size=5)
        except Exception as e:
            st.error(f"❌ Error al generar embeddings: {e}")
            return None
        
        # Crear la base de datos FAISS manualmente
        # Primero, crear metadatos (opcional, pero necesario para FAISS)
        metadatas = [doc.metadata for doc in fragmentos]
        
        # Crear el vectorstore
        base_de_datos = FAISS.from_embeddings(
            text_embeddings=list(zip(textos, todas_las_embeddings)),
            embedding=embeddings,
            metadatas=metadatas
        )
        
        buscador = base_de_datos.as_retriever(search_kwargs={"k": 3})
        
        # Modelo de lenguaje
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.2,
            google_api_key=key,
            timeout=60  # Intentar aumentar timeout (puede no funcionar en Streamlit Cloud)
        )
        
        # Prompt
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
        st.error(f"❌ Error dentro de preparar_agente: {e}")
        st.exception(e)  # Muestra el traceback completo
        return None

# ========== FLUJO PRINCIPAL ==========
try:
    # Configuración inicial
    PDF_PATH = "documento.pdf"
    api_key = None

    # Obtener clave API
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
        os.environ["GOOGLE_API_KEY"] = api_key
    elif "GOOGLE_API_KEY" in os.environ:
        api_key = os.environ["GOOGLE_API_KEY"]

    # Mostrar título
    st.title("💻 Asistente Virtual - TechStore Perú")
    st.write("¡Hola! Soy tu asesor tecnológico. Pregúntame sobre especificaciones, precios en soles (S/), garantías o envíos de nuestros productos.")

    # Validaciones
    if not os.path.exists(PDF_PATH):
        st.error(f"❌ No se encontró el archivo '{PDF_PATH}'. Asegúrate de subirlo al repositorio.")
        st.stop()

    if not api_key:
        st.error("❌ No se encontró la clave GOOGLE_API_KEY. Configúrala en los Secrets de Streamlit.")
        st.stop()

    # Preparar el agente
    with st.spinner("Cargando catálogo y preparando el asistente..."):
        mi_agente = preparar_agente(PDF_PATH, api_key)

    if mi_agente is None:
        st.error("❌ El asistente no pudo inicializarse. Revisa los mensajes de error anteriores.")
        st.stop()

    # Interfaz de usuario
    pregunta = st.text_input("Escribe tu consulta sobre laptops, PC o accesorios aquí:")

    if pregunta:
        with st.spinner("Consultando el catálogo de TechStore Perú..."):
            try:
                resultado = mi_agente.invoke({"input": pregunta})
                st.write("### 🤖 Respuesta del Asesor:")
                st.write(resultado["answer"])
            except Exception as e:
                st.error(f"Error al consultar: {e}")

except Exception as e:
    st.error(f"🚨 Error crítico en la aplicación: {e}")
    st.exception(e)
