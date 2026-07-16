import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Diseñamos la pantalla del E-commerce en español
st.set_page_config(page_title="Asistente TechStore Perú", page_icon="💻")
st.title("💻 Asistente Virtual - TechStore Perú")
st.write("¡Hola! Soy tu asesor tecnológico. Pregúntame sobre especificaciones, precios en soles (S/), garantías o envíos de nuestros productos.")

PDF_PATH = "documento.pdf"

@st.cache_resource
def preparar_agente(ruta_pdf):
    if not os.path.exists(ruta_pdf):
        st.error("No se encontró el catálogo 'documento.pdf'. Por favor súbelo a tu repositorio.")
        return None

    # Lectura y fragmentación del PDF
    lector = PyPDFLoader(ruta_pdf)
    paginas = lector.load()
    separador = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    fragmentos = separador.split_documents(paginas)

    # Embeddings y Base de Datos Vectorial gratuita de Google
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    base_de_datos = FAISS.from_documents(fragmentos, embeddings)
    buscador = base_de_datos.as_retriever(search_kwargs={"k": 3})

    # Modelo de Inteligencia Artificial gratuito: Gemini 1.5 Flash
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
    
    # Personalidad e instrucciones del Agente Virtual
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

# 3. Flujo principal de la aplicación
if os.path.exists(PDF_PATH):
    # Verificamos que tengamos la clave gratuita de Google configurada
    if "GOOGLE_API_KEY" in os.environ or st.secrets.get("GOOGLE_API_KEY"):
        mi_agente = preparar_agente(PDF_PATH)
        pregunta = st.text_input("Escribe tu consulta sobre laptops, PC o accesorios aquí:")
        
        if pregunta:
            with st.spinner("Consultando el catálogo de TechStore Perú..."):
                resultado = mi_agente.invoke({"input": pregunta})
                st.write("### 🤖 Respuesta del Asesor:")
                st.write(resultado["answer"])
    else:
        st.warning("⚠️ Falta configurar la clave GOOGLE_API_KEY en los ajustes de la plataforma.")
else:
    st.info("📂 Sube tu archivo 'documento.pdf' al repositorio para activar el asesor virtual.")
