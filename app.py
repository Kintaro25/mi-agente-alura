import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Diseñamos la pantalla que verá el usuario en español
st.set_page_config(page_title="Mi Agente Alura", page_icon="🤖")
st.title("🤖 Mi Agente Inteligente")
st.write("¡Hola! Pregúntame lo que quieras sobre el documento que tengo guardado.")

# Ruta donde guardaremos tu PDF dentro de GitHub
PDF_PATH = "documento.pdf"

# 2. Esta función procesa el documento y prepara al agente
@st.cache_resource
def preparar_agente(ruta_pdf):
    # Si el archivo no existe, mostramos un aviso amigable
    if not os.path.exists(ruta_pdf):
        st.error("No se encontró el archivo 'documento.pdf'. Por favor súbelo a tu repositorio.")
        return None

    # Lee el PDF
    lector = PyPDFLoader(ruta_pdf)
    paginas = lector.load()

    # Divide el texto en partes pequeñas para que el agente lo lea más fácil
    separador = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    fragmentos = separador.split_documents(paginas)

    # Convierte el texto a un formato que la IA entiende y lo guarda en memoria
    embeddings = OpenAIEmbeddings()
    base_de_datos = FAISS.from_documents(fragmentos, embeddings)
    buscador = base_de_datos.as_retriever(search_kwargs={"k": 3})

    # Configuramos el modelo de Inteligencia Artificial (usamos GPT-4o-mini de OpenAI)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    
    # Le damos instrucciones de personalidad al agente en español
    instrucciones = (
        "Eres un asistente amable. Responde la pregunta del usuario utilizando "
        "únicamente la información del contexto que se te proporciona a continuación. "
        "Si no encuentras la respuesta en el documento, di amablemente que no posees esa información.\n\n"
        "Contexto:\n{context}"
    )
    
    plantilla_prompt = ChatPromptTemplate.from_messages([
        ("system", instrucciones),
        ("human", "{input}"),
    ])

    # Unimos todo para crear el agente funcional
    cadena_respuestas = create_stuff_documents_chain(llm, plantilla_prompt)
    agente_final = create_retrieval_chain(buscador, cadena_respuestas)
    
    return agente_final

# 3. Ejecución de la aplicación en pantalla
if os.path.exists(PDF_PATH):
    # Verificamos que tengamos la llave de acceso de la IA configurada
    if "OPENAI_API_KEY" in os.environ or st.secrets.get("OPENAI_API_KEY"):
        
        # Inicializamos el agente con el PDF
        mi_agente = preparar_agente(PDF_PATH)
        
        # Cuadro de texto para que el usuario escriba su pregunta
        pregunta = st.text_input("Escribe tu pregunta aquí:")
        
        if pregunta:
            with st.spinner("Buscando en el documento..."):
                # El agente busca la respuesta y nos la entrega
                resultado = mi_agente.invoke({"input": pregunta})
                st.write("### Respuesta del Agente:")
                st.write(resultado["answer"])
    else:
        st.warning("Falta configurar la clave API de OpenAI en los ajustes de la plataforma.")
else:
    st.info("Sube tu archivo 'documento.pdf' a tu repositorio de GitHub para comenzar.")
