import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

@st.cache_resource
def preparar_agente(ruta_pdf, key):
    try:
        # Todo el contenido de la función aquí
        if not os.path.exists(ruta_pdf):
            st.error("No se encontró el PDF.")
            return None

        lector = PyPDFLoader(ruta_pdf)
        paginas = lector.load()
        separador = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        fragmentos = separador.split_documents(paginas)

        embeddings = GoogleGenerativeAIEmbeddings(
            model="text-embedding-004",  # sin "models/"
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
        # Esto mostrará el error completo en la app
        st.error(f"❌ Error dentro de preparar_agente: {e}")
        return None
