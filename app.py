import streamlit as st
import os
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from google.api_core.exceptions import DeadlineExceeded, ResourceExhausted

# ========== CONFIGURACIÓN ==========
st.set_page_config(page_title="TechStore Perú - Asistente Laptops", page_icon="💻")
st.title("💻 Asistente TechStore Perú")
st.write("¡Hola! Soy tu asesor de laptops. Pregúntame sobre modelos, precios, envíos, garantías o cualquier duda.")

# ========== FUNCIÓN PARA EMBEDDINGS CON REINTENTOS ==========
def crear_vectorstore_con_reintentos(fragmentos, embeddings, batch_size=5, max_retries=3):
    """Crea FAISS procesando los fragmentos en lotes con reintentos."""
    if not fragmentos:
        return None
    total = len(fragmentos)
    progress_bar = st.progress(0)
    status_text = st.empty()
    base = None

    for i in range(0, total, batch_size):
        batch = fragmentos[i:i+batch_size]
        textos = [doc.page_content for doc in batch]
        metadatas = [doc.metadata for doc in batch]

        for intento in range(max_retries):
            try:
                status_text.text(f"Lote {i//batch_size+1}/{(total-1)//batch_size+1} (intento {intento+1})...")
                if base is None:
                    base = FAISS.from_texts(textos, embeddings, metadatas=metadatas)
                else:
                    base.add_texts(textos, embeddings, metadatas=metadatas)
                break
            except (DeadlineExceeded, ResourceExhausted) as e:
                if intento == max_retries - 1:
                    st.error(f"Fallo tras {max_retries} intentos: {e}")
                    raise
                time.sleep(2 ** intento)
            except Exception as e:
                st.error(f"Error: {e}")
                raise
        progress_bar.progress(min((i + batch_size) / total, 1.0))

    progress_bar.empty()
    status_text.empty()
    return base

# ========== FUNCIÓN PRINCIPAL (CACHÉ) ==========
@st.cache_resource
def preparar_agente(ruta_pdf, api_key):
    try:
        # Cargar PDF
        loader = PyPDFLoader(ruta_pdf)
        paginas = loader.load()
        if not paginas:
            st.error("El PDF está vacío o no se pudo leer.")
            return None
        st.info(f"📄 {len(paginas)} páginas cargadas.")

        # Dividir en fragmentos
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )
        fragmentos = splitter.split_documents(paginas)
        st.info(f"✂️ {len(fragmentos)} fragmentos generados.")

        # Embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="text-embedding-004",
            google_api_key=api_key
        )

        # Vectorstore (con reintentos)
        vectorstore = crear_vectorstore_con_reintentos(fragmentos, embeddings, batch_size=5)

        if vectorstore is None:
            st.error("No se pudo crear la base de vectores.")
            return None

        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

        # LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.2,
            google_api_key=api_key
        )

        # Prompt
        prompt_template = (
            "Eres un asesor de ventas de laptops en 'TechStore Perú'. "
            "Responde usando ÚNICAMENTE la información del contexto proporcionado. "
            "Si la pregunta es sobre precios, indícalos en Soles (S/). "
            "Si no encuentras la respuesta, di: 'Lo siento, no tengo esa información en mi catálogo actual.'\n\n"
            "Contexto:\n{context}\n\n"
            "Pregunta: {input}\n"
            "Respuesta:"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_template),
            ("human", "{input}")
        ])

        chain = create_stuff_documents_chain(llm, prompt)
        agente = create_retrieval_chain(retriever, chain)

        st.success("✅ Asistente listo para consultas.")
        return agente

    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.exception(e)
        return None

# ========== FLUJO PRINCIPAL ==========
def main():
    # Ruta CORREGIDA: el PDF está en la raíz del repositorio
    PDF_PATH = "catalogo_laptops.pdf"
    api_key = None

    # Obtener clave API
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
        os.environ["GOOGLE_API_KEY"] = api_key
    elif "GOOGLE_API_KEY" in os.environ:
        api_key = os.environ["GOOGLE_API_KEY"]

    if not api_key:
        st.error("❌ Configura GOOGLE_API_KEY en los Secrets de Streamlit.")
        st.stop()

    if not os.path.exists(PDF_PATH):
        st.error(f"❌ No se encontró el archivo '{PDF_PATH}'. Asegúrate de subirlo a la raíz del repositorio.")
        st.stop()

    # Preparar agente (con caché)
    with st.spinner("🔧 Preparando asistente..."):
        agente = preparar_agente(PDF_PATH, api_key)

    if agente is None:
        st.stop()

    # Input del usuario
    pregunta = st.text_area("✍️ Escribe tu consulta sobre laptops, precios, envíos, etc.:", height=80)

    if st.button("Consultar", type="primary") and pregunta:
        with st.spinner("🔍 Buscando respuesta..."):
            try:
                respuesta = agente.invoke({"input": pregunta})
                st.markdown("### 🤖 Respuesta:")
                st.write(respuesta["answer"])

                # Mostrar fuentes (opcional)
                with st.expander("📚 Fuentes consultadas"):
                    for doc in respuesta.get("context", []):
                        st.write(f"- Página {doc.metadata.get('page', '?')}")
            except Exception as e:
                st.error(f"Error: {e}")
                st.exception(e)

if __name__ == "__main__":
    main()
