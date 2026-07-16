import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

st.set_page_config(page_title="TechStore Perú - Asistente Laptops", page_icon="💻")
st.title("💻 Asistente TechStore Perú")
st.write("¡Hola! Soy tu asesor de laptops. Pregúntame sobre modelos, precios, envíos, garantías o cualquier duda.")

@st.cache_resource
def preparar_agente(ruta_pdf):
    try:
        # ========== 1. Cargar y dividir el PDF ==========
        loader = PyPDFLoader(ruta_pdf)
        paginas = loader.load()
        if not paginas:
            st.error("El PDF está vacío o no se pudo leer.")
            return None

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )
        fragmentos = splitter.split_documents(paginas)
        st.info(f"📄 {len(paginas)} páginas → ✂️ {len(fragmentos)} fragmentos.")

        # ========== 2. Embeddings locales ==========
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # ========== 3. Vectorstore FAISS ==========
        vectorstore = FAISS.from_documents(fragmentos, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

        # ========== 4. LLM con Hugging Face (modelo pequeño) ==========
        model_name = "distilgpt2"  # Modelo rápido y liviano (~500 MB)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        # Configurar pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,
            temperature=0.2,
            do_sample=True,
            repetition_penalty=1.1,
            device=-1  # CPU
        )

        llm = HuggingFacePipeline(pipeline=pipe)

        # ========== 5. Prompt y cadena ==========
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

        st.success("✅ Asistente listo (modelo Hugging Face).")
        return agente

    except Exception as e:
        st.error(f"❌ Error al preparar el agente: {e}")
        st.exception(e)
        return None

def main():
    PDF_PATH = "catalogo_laptops.pdf"

    if not os.path.exists(PDF_PATH):
        st.error(f"❌ No se encontró el archivo '{PDF_PATH}'. Asegúrate de subirlo.")
        st.stop()

    with st.spinner("🔧 Preparando asistente (descargando modelo, puede tardar unos minutos)..."):
        agente = preparar_agente(PDF_PATH)

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
