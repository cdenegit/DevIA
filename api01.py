from flask import Flask, request, jsonify
import os
import google.generativeai as genai

app = Flask(__name__)

# Configura tu API Key de Gemini
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "TU_API_KEY_AQUI")
genai.configure(api_key=GEMINI_API_KEY)
MODEL_NAME="models/gemini-pro"
generation_config = {
    "temperature": 0.7,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048, # Ajusta según la longitud esperada de las respuestas
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

# Inicializar el modelo
model = genai.GenerativeModel(model_name=MODEL_NAME,
                              generation_config=generation_config,
                              safety_settings=safety_settings)

@app.route('/')
def home():
    return "API de análisis de perfil con Gemini funcionando."

@app.route('/analizar_perfil', methods=['POST'])
def analizar_perfil():
    data = request.get_json()

    prompt = data.get("prompt")
    preguntas_respuestas = data.get("respuestas", [])

    if not prompt or not preguntas_respuestas:
        return jsonify({"error": "Faltan el prompt o las respuestas"}), 400

    resultados = []
    resumen_general = ""

    try:
        for item in preguntas_respuestas:
            pregunta = item.get("pregunta")
            respuesta = item.get("respuesta")

            if not pregunta or not respuesta:
                continue

            entrada = f"{prompt}\n\nPregunta: {pregunta}\nRespuesta: {respuesta}"
            respuesta_modelo = model.generate_content(entrada)
            interpretacion = respuesta_modelo.text.strip()

            resultados.append({
                "pregunta": pregunta,
                "respuesta": respuesta,
                "interpretacion": interpretacion
            })

        # Generar resumen general del perfil
        resumen_prompt = prompt + "\n\nHaz un resumen general del perfil del usuario basado en sus respuestas anteriores."
        resumen_modelo = model.generate_content(resumen_prompt)
        resumen_general = resumen_modelo.text.strip()

        return jsonify({
            "interpretaciones": resultados,
            "resumen_perfil": resumen_general
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)