Asistente Virtual E-commerce - Tienda de computo

un asistente virtual que hice para el Challenge de Alura Latam. Mi objetivo es ayudar a los clientes de una tienda de laptops llamada TechStore Perú a resolver sus dudas de forma rápida y sencilla.


Puedes preguntarme sobre:
💰 Precios de laptops
🚚 Envíos y tiempos de entrega
🔧 Garantías y devoluciones
💳 Métodos de pago
📦 Especificaciones técnicas

🧠 ¿Cómo funciona?
Este asistente usa Inteligencia Artificial para leer un catálogo en PDF y responder preguntas como si fuera un asesor de ventas.

El proceso es simple:
Lectura del PDF.
División en fragmentos.
Generación de embeddings.
Búsqueda inteligente.
Respuesta con IA.

- Tecnologías usadas
Python, Streamlit(Crear la interfaz web), LangChain, FAISS, sentence-transformers (embeddings), Groq(Modelo de lenguaje para generar respuestas).


¿Cómo ejecutarlo?
 1: Probar la versión en línea (recomendado)
El asistente ya está desplegado en Streamlit Cloud. Puedes probarlo aquí:

🔗 https://mi-agente-alura-isrlpekytdafhgjgsf7jhf.streamlit.app/

No necesitas instalar nada, solo entrar y hacer preguntas.

2: Ejecutar localmente
Si quieres modificar el código o probarlo en tu computadora:

-Clona el repositorio
-Ve a console.groq.com y crea una cuenta gratis.
-Genera una clave API.
-toml GROQ_API_KEY = "tu-clave-api-aqui"

3:Ejecuta la app

