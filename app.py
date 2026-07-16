@st.cache_resource
def preparar_agente(ruta_pdf, key):
    if not key:
        st.error("La clave API no está configurada.")
        return None

    if not os.path.exists(ruta_pdf):
        st.error("No se encontró el archivo PDF.")
        return None

    try:
        lector = PyPDFLoader(ruta_pdf)
        paginas = lector.load()
        if not paginas:
            st.error("El PDF está vacío o no se pudo cargar.")
            return None
    except Exception as e:
        st.error(f"Error al cargar el PDF: {e}")
        return None

    separador = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    fragmentos = separador.split_documents(paginas)

    # Intentar con modelo más estable
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="text-embedding-004",  # sin 'models/'
            google_api_key=key
        )
        base_de_datos = FAISS.from_documents(fragmentos, embeddings)
    except Exception as e:
        st.error(f"❌ Error al generar embeddings: {e}")
        return None

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
