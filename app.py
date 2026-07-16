import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI  # Solo si usas Gemini para el LLM
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="TechStore Perú - Asistente Laptops", page_icon="💻")
st.title("💻 Asistente TechStore Perú")
st.write("¡Hola! Soy tu asesor de laptops. Pregúntame sobre modelos, precios, envíos, garantías o cualquier duda.")

@st.cache_resource
def preparar_agente(ruta_pdf, api_key=None):
    try:
        loader = PyPDFLoader(ruta_pdf)
        paginas = loader.load()
        if not paginas:
            st.error("El PDF está vacío o no se pudo leer.")
            return None

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        fragmentos = splitter.split_documents(paginas)

        # ===== EMBEDDINGS LOCALES (sin API) =====
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        # ========================================

        vectorstore = FAISS.from_documents(fragmentos, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

        # ===== Si usas Gemini para el LLM =====
        if api_key:
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.2,
                google_api_key=api_key
            )
        else:
            st.error("No se proporcionó API key para el LLM.")
            return None
        # =====================================

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
        st.error(f"❌ Error al preparar el agente: {e}")
        st.exception(e)
        return None

def main():
    PDF_PATH = "catalogo_laptops.pdf"
    api_key = None

    # Obtener clave de Gemini desde Secrets (si existe)
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
        os.environ["GOOGLE_API_KEY"] = api_key
    elif "GOOGLE_API_KEY" in os.environ:
        api_key = os.environ["GOOGLE_API_KEY"]

    if not api_key:
        st.error("❌ Configura GOOGLE_API_KEY en los Secrets de Streamlit para usar Gemini.")
        st.stop()

    if not os.path.exists(PDF_PATH):
        st.error(f"❌ No se encontró el archivo '{PDF_PATH}'. Asegúrate de subirlo.")
        st.stop()

    with st.spinner("🔧 Preparando asistente (esto puede tomar un minuto la primera vez)..."):
        agente = preparar_agente(PDF_PATH, api_key)

    if agente is None:
        st.stop()

    pregunta = st.text_area("✍️ Escribe tu consulta sobre laptops, precios, envíos, etc.:", height=80)
    if st.button("Consultar", type="primary") and pregunta:
        with st.spinner("🔍 Buscando respuesta..."):
            try:
                respuesta = agente.invoke({"input": pregunta})
                st.markdown("### 🤖 Respuesta:")
                st.write(respuesta["answer"])
            except Exception as e:
                st.error(f"Error al obtener respuesta: {e}")
                st.exception(e)

if __name__ == "__main__":
    main()
