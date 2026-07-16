1. 🛒 Asistente Virtual E-commerce - Tienda de computo

Este proyecto consiste en un Agente de Inteligencia Artificial diseñado para funcionar como un asesor de ventas y soporte técnico para una tienda de tecnología. El agente responde preguntas precisas sobre especificaciones de hardware, precios en Soles (S/), garantías y políticas de envío basándose exclusivamente en el catálogo oficial de la tienda.

---

2. 📐 Arquitectura de la Solución

La aplicación utiliza una arquitectura RAG (Retrieval-Augmented Generation)** para asegurar que el agente brinde información veraz y no sufra de alucinaciones:

- Ingesta de Datos: El catálogo de productos en formato PDF es procesado dinámicamente mediante `PyPDFLoader`.
- Segmentación (Chunking): El texto se divide en bloques semánticos utilizando `RecursiveCharacterTextSplitter` para retener el contexto de cada producto y política.
- Indexación y Embeddings: Se generan vectores utilizando el modelo gratuito `embedding-001` de Google Generative AI.
- Almacenamiento Vectorial: Se utiliza **FAISS** como base de datos vectorial en memoria para realizar búsquedas de similitud ultrarrápidas.
- Generación de Respuesta: Ante la consulta del usuario, el sistema recupera los fragmentos más relevantes del catálogo y los procesa utilizando el LLM *Gemini 1.5 Flash* para redactar una respuesta natural, amable y en     español.

---

3. 🛠️ Tecnologías y Herramientas Utilizadas

-Lenguaje: Python 3.10+
-Framework Web: Streamlit
-Orquestador IA: LangChain
-Base de Datos Vectorial: FAISS (Facebook AI Similarity Search)
-Modelos de IA (LLM y Embeddings): Google Gemini API (Totalmente gratuito)
-Despliegue (Cloud): Streamlit Community Cloud

---
