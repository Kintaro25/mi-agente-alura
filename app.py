import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="TechStore Perú - Asistente Laptops", page_icon="💻")
st.title("💻 Asistente TechStore Perú")
st.write("¡Hola! Soy tu asesor de laptops. Pregúntame sobre modelos, precios, envíos, garantías o cualquier duda.")

@st.cache_resource
def preparar_agente(ruta_pdf, api_key):
    try:
        loader = PyPDFLoader(ruta_pdf)
        paginas = loader.load()
        if not paginas:
            st.error("El PDF está vacío o no se pudo leer.")
            return None

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        fragmentos = splitter.split_documents(paginas)
        st.info(f"📄 {len(paginas)} páginas → ✂️ {len(fragmentos)} fragmentos.")

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vectorstore = FAISS.from_documents(fragmentos, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

        # ===== LLM CON GROQ (rápido y gratuito) =====
        llm = ChatGroq(
            model="llama3-70b-8192",  # Modelo potente y rápido
            temperature=0.2,
            groq_api_key=api_key
        )
        # ===========================================

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

        st.success("✅ Asistente listo para consultas (con Groq).")
        return agente

    except Exception as e:
        st.error(f"❌ Error al preparar el agente: {e}")
        st.exception(e)
        return None

def main():
    PDF_PATH = "catalogo_laptops.pdf"
    api_key = None

    # Obtener clave de Groq desde Secrets
    if "GROQ_API_KEY" in st.secrets:
        api_key = st.secrets["GROQ_API_KEY"]
        os.environ["GROQ_API_KEY"] = api_key
    elif "GROQ_API_KEY" in os.environ:
        api_key = os.environ["GROQ_API_KEY"]

    if not api_key:
        st.error("❌ Configura GROQ_API_KEY en los Secrets de Streamlit.")
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
        with st.spinner("🔍 Buscando respuesta (Groq es rápido)..."):
            try:
                respuesta = agente.invoke({"input": pregunta})
                st.markdown("### 🤖 Respuesta:")
                st.write(respuesta["answer"])
            except Exception as e:
                st.error(f"Error al obtener respuesta: {e}")
                st.exception(e)

if __name__ == "__main__":
    main()
