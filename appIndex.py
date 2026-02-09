from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from PIL import Image
import google.generativeai as genai
import base64
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import uvicorn
import os
import uuid
import json
import tempfile
import numpy as np
import rasterio
import xarray as xr
import logging

app = FastAPI()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("analisis_index")

# =========================
# 📥 Request schemas
# =========================
class InitRequest(BaseModel):
    index_name: str
    aspctos_inv: str
    nmbre_fnca: str
    
# --- FUNCIONES DE CÁLCULO ESTADÍSTICO (Información Vital para la IA) ---

@app.get("/")
def read_root():
    return {"status": "online", "service": "AgroTech Analyzer"}

def calcular_estadisticas_pro(valores):
    if len(valores) == 0: return None
    return {
        "min": float(np.nanmin(valores)),
        "max": float(np.nanmax(valores)),
        "media": float(np.nanmean(valores)),
        "mediana": float(np.nanmedian(valores)),
        "std": float(np.nanstd(valores)),
        "p10": float(np.nanpercentile(valores, 10)),
        "p25": float(np.nanpercentile(valores, 25)),
        "p75": float(np.nanpercentile(valores, 75)),
        "p90": float(np.nanpercentile(valores, 90)),
        "varianza": float(np.nanvar(valores))
    }
def generar_prompt_experto(index_name, stats, meta, aspectos):
    # 1. Preparamos los valores formateados para evitar comillas anidadas en el f-string
    finca = meta.get('finca', 'N/A')
    sensor = meta.get('sensor', 'No especificado')
    fecha = meta.get('fecha', 'N/A')
    area = f"{meta.get('area_m2', 0):,.2f}"
    resolucion = meta.get('resolucion_m', 'N/A')
    
    # 2. Creamos un diccionario local con los strings ya formateados
    # Esto elimina la necesidad de usar :.4f dentro del bloque de texto
    s = {k: f"{v:.4f}" if isinstance(v, (int, float)) else v for k, v in stats.items()}

    # 3. Usamos triple comilla simple (''') para el bloque de texto. 
    # Es menos propenso a errores cuando el contenido tiene comillas dobles.
    prompt = f'''
    Eres un Agente de IA especializado en Teledetección y Agronomía de Precisión.
    Tu misión es diagnosticar el estado del cultivo en la finca "{finca}".
    
    === CONTEXTO TÉCNICO ===
    - Índice Analizado: {index_name.upper()}
    - Sensor: {sensor} | Fecha: {fecha}
    - Área: {area} m²
    - Resolución: {resolucion} metros/píxel
    
    === RADIOGRAFÍA ESTADÍSTICA DEL ÍNDICE ===
    - Rango: [{s['min']} a {s['max']}]
    - Promedio Central (Media): {s['media']}
    - Robustez (Mediana): {s['mediana']}
    - Dispersión (Desviación Std): {s['std']}
    - Distribución de Vigor:
      * 10% del área (Crítico): Inferior a {s['p10']}
      * 25% del área (Bajo): Inferior a {s['p25']}
      * 75% del área (Bueno): {s['p75']}
      * 90% del área (Óptimo): Superior a {s['p90']}
    
    === OBJETIVO DEL USUARIO ===
    El productor está investigando: "{aspectos}"
    
    === TAREA DE DIAGNÓSTICO ===
    1. Interpretación de Salud: Basado en el {index_name.upper()}, ¿qué indican estos valores para este tipo de sensor?
    2. Análisis de Homogeneidad: Compara la media con los percentiles {s['p10']} y {s['p90']}. ¿Es un cultivo uniforme o fragmentado?
    3. Respuesta a la Investigación: Aborda específicamente los aspectos solicitados por el usuario.
    4. Plan de Acción: Proporciona 3 recomendaciones técnicas (ej: fertilización variable, riego, muestreo foliar).
    
    Responde en formato Markdown, con tono profesional y científico.
    '''
    return prompt
    
# =========================
# 📂 Funciones de Lectura
# =========================

def leer_raster_gdal(path, bandas_solicitadas=None):
    with rasterio.open(path) as src:
        logger.info(f"📖 Rasterio abrió el archivo. Bandas: {src.count}")
        
        # --- FUNCIÓN INTERNA DE NORMALIZACIÓN ---
        def normalizar_banda(arr):
            # Si el valor máximo es alto (ej. 10000), es reflectancia escalada.
            # Si es > 1, dividimos para llevar a rango 0-1.
            max_val = np.max(arr)
            if max_val > 255: 
                return arr.astype('float32') / 10000.0
            elif max_val > 1.0: 
                return arr.astype('float32') / 255.0
            return arr.astype('float32')

        # --- LECTURA SEGURO DE TODAS LAS BANDAS ---
        # Leemos la 1 para tener un fallback consistente
        b_base = src.read(1)
        
        bandas = {
            "B08": normalizar_banda(b_base),
            "B04": normalizar_banda(src.read(2)) if src.count >= 2 else normalizar_banda(b_base),
            "B03": normalizar_banda(src.read(3)) if src.count >= 3 else normalizar_banda(b_base),
            "B02": normalizar_banda(src.read(4)) if src.count >= 4 else normalizar_banda(b_base),
            "B05": normalizar_banda(src.read(5)) if src.count >= 5 else normalizar_banda(b_base)
        }

        # --- EXTRACCIÓN DE METADATOS ---
        tags = src.tags()
        sensor = tags.get('TIFFTAG_SOFTWARE', tags.get('SENSOR_ID', 'Sensor No Identificado'))
        fecha = tags.get('TIFFTAG_DATETIME', tags.get('ACQUISITION_DATE', 'Fecha No Disponible'))

        # Cálculo de resolución y área real
        # Si src.res es (1.0, 1.0) suele ser un error de georreferencia, asumimos 10m (Sentinel)
        res_m = src.res[0] if (src.res and src.res[0] != 1.0) else 10.0
        
        # Área basada en píxeles y resolución para evitar errores de CRS
        area_calculada = float(src.width * src.height * (res_m ** 2))

        meta = {
            "sensor": sensor,
            "fecha": fecha,
            "resolucion_m": res_m,
            "area_m2": area_calculada,
            "crs": str(src.crs) if src.crs else "No Georreferenciado",
            "ancho": src.width,
            "alto": src.height
        }
        
        return bandas, meta
 
def leer_raster_cientifico(path, bandas_solicitadas):
    ds = xr.open_dataset(path)
    bandas = {}
    for b in bandas_solicitadas:
        var_name = [v for v in ds.data_vars if b.lower() in v.lower() or b.upper() in v]
        if var_name:
            bandas[b] = ds[var_name[0]].values.astype('float32')
        else:
            # Si falta una banda, creamos una matriz de ceros basada en una existente
            if bandas:
                ref = next(iter(bandas.values()))
                bandas[b] = np.zeros_like(ref)
    
    meta = {
        "sensor": ds.attrs.get('sensor', 'Dataset Científico'),
        "fecha": ds.attrs.get('time_coverage_start', 'N/A'),
        "resolucion_m": "Proyectada",
        "area_m2": "Calculada",
        "crs": "Variable"
    }
    ds.close()
    return bandas, meta

def leer_imagen_simple(path):
    img = Image.open(path).convert('RGB')
    arr = np.array(img).astype('float32') / 255.0
    bandas = {
        "B04": arr[:, :, 0], # Red
        "B03": arr[:, :, 1], # Green
        "B02": arr[:, :, 2], # Blue
        "B08": arr[:, :, 1] * 1.2 # Simulación NIR
    }
    meta = {
        "sensor": "Cámara RGB Estándar",
        "fecha": "N/A",
        "resolucion_m": 0,
        "area_m2": 0,
        "crs": "No Georreferenciado"
    }
    return bandas, meta

# =========================
# 🧮 Procesamiento y Análisis
# =========================

def ejecutar_calculo_indice(bandas, index_name):
    eps = 1e-10
    
    # Extraemos bandas con fallback: si una no existe, usamos la B08 para evitar que np.array falle
    B8 = bandas.get("B08")
    B4 = bandas.get("B04") if bandas.get("B04") is not None else B8
    B3 = bandas.get("B03") if bandas.get("B03") is not None else B8
    B2 = bandas.get("B02") if bandas.get("B02") is not None else B4
    B5 = bandas.get("B05") if bandas.get("B05") is not None else B4

    formulas = {
        "ndvi":  lambda: (B8 - B4) / (B8 + B4 + eps),
        "evi":   lambda: 2.5 * ((B8 - B4) / (B8 + 6 * B4 - 7.5 * B2 + 1 + eps)),
        "ndwi":  lambda: (B3 - B8) / (B3 + B8 + eps),
        "ndre":  lambda: (B8 - B5) / (B8 + B5 + eps),
        "msavi": lambda: (2 * B8 + 1 - np.sqrt(np.maximum(0, (2 * B8 + 1)**2 - 8 * (B8 - B4)))) / 2,
        "reci":  lambda: (B8 / (B4 + eps)) - 1
    }

    func = formulas.get(index_name.lower())
    if not func:
        raise ValueError(f"Índice {index_name} no implementado.")
    
    try:
        res = func()
        # Limpieza de seguridad para la IA: eliminar NaNs e Infinitos
        res = np.nan_to_num(res, nan=0.0, posinf=1.0, neginf=-1.0)
        return res
    except Exception as e:
        raise ValueError(f"Error matemático al calcular {index_name}: {str(e)}")

def muestrear_indice(arr, meta, resolucion_objetivo_m):
    valores_validos = arr[~np.isnan(arr)]
    if valores_validos.size == 0:
        return {"resolucion_m": 0, "valores": [0], "total_muestras": 0}
    
    # Muestreo representativo para la IA
    muestras = np.random.choice(valores_validos, min(5000, valores_validos.size), replace=False)
    return {
        "resolucion_m": resolucion_objetivo_m,
        "valores": muestras.tolist(),
        "total_muestras": len(muestras)
    }

def detectar_tipo_archivo(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".tif", ".tiff", ".jp2", ".ntf"]: return "raster_gdal"
    if ext in [".hdf", ".h5", ".nc"]: return "raster_cientifico"
    if ext == ".png": return "imagen_simple"
    raise ValueError(f"Formato {ext} no soportado")

def generar_pdf_diagnostico(texto_markdown, nombre_finca):
    """Convierte el diagnóstico de la IA en un PDF binario."""
    path_pdf = tempfile.mktemp(suffix=".pdf")
    doc = SimpleDocTemplate(path_pdf, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Título del Reporte
    story.append(Paragraph(f"Informe de Diagnóstico: {nombre_finca}", styles['Title']))
    story.append(Spacer(1, 12))

    # Limpiamos un poco el Markdown simple para el PDF
    lineas = texto_markdown.replace("###", "").replace("##", "").replace("**", "").split("\n")
    for linea in lineas:
        if linea.strip():
            story.append(Paragraph(linea, styles['Normal']))
            story.append(Spacer(1, 6))

    doc.build(story)
    
    # Leer el PDF y convertirlo a Base64
    with open(path_pdf, "rb") as f:
        pdf_encoded = base64.b64encode(f.read()).decode('utf-8')
    
    os.remove(path_pdf) # Limpieza
    return pdf_encoded
    
# =========================
# 🚪 Endpoints
# =========================

@app.post("/iniciar")
def iniciar(req: InitRequest):
    logger.info(f"🚀 Iniciando análisis: {req.index_name}")
    return {"status": "ok", "Iniciado": True}

@app.post("/analisis_index")
async def analisis_index(
    nmbre_fnca: str = Form(...),
    geojson: str = Form(...),
    index_name: str = Form(...),
    aspctos_inv: str = Form(...),
    gemini_key: str = Form(...),
    modelo_ia: str = Form(...),
    file: UploadFile = File(...)
):
    file_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1].lower()
    temp_path = os.path.join(tempfile.gettempdir(), f"{file_id}_{file.filename}")
    
    try:
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)
        
        # --- SECCIÓN DE LECTURA MODULARIZADA ---
        tipo_archivo = detectar_tipo_archivo(temp_path)
        
        if tipo_archivo == "raster_gdal":
            # REEMPLAZO: Llamada a la función externa
            bandas, meta = leer_raster_gdal(temp_path, ["B08", "B04", "B03", "B02", "B05"])
            
        elif tipo_archivo == "raster_cientifico":
            bandas, meta = leer_raster_cientifico(temp_path, ["B08", "B04", "B03", "B02"])

        else: # imagen_simple (PNG/JPG)
            img = Image.open(temp_path).convert('RGB')
            arr = np.array(img).astype('float32') / 255.0
            bandas = {"B08": arr[:,:,1]*1.2, "B04": arr[:,:,0]} 
            meta = {"sensor": "Cámara Convencional", "area_m2": 0, "resolucion_m": 0, "ancho": arr.shape[1], "alto": arr.shape[0]}

        meta["finca"] = nmbre_fnca

        # 3. CÁLCULO UNIFICADO (Usando tu función de álgebra corregida)
        # Esta función ya maneja NDVI, EVI, MSAVI, etc.
        idx_map = ejecutar_calculo_indice(bandas, index_name)

        # 4. ESTADÍSTICAS Y MUESTREO
        # Limpieza de valores para evitar errores en el JSON final
        valores_limpios = idx_map[~np.isnan(idx_map)]
        if valores_limpios.size == 0:
            raise ValueError("El sensor no retornó datos válidos para este índice.")

        stats = calcular_estadisticas_pro(valores_limpios)
        
        # Muestreo representativo para la gráfica en PHP/JS
        num_muestras = min(2000, len(valores_limpios))
        muestras = np.random.choice(valores_limpios, num_muestras, replace=False).tolist()

        # 5. CONFIGURACIÓN DINÁMICA DE IA
        if not gemini_key:
            raise ValueError("API Key de Gemini no proporcionada por el servidor PHP.")
        
        genai.configure(api_key=gemini_key)
        # model = genai.GenerativeModel, se Usa la variable modelo_ia enviada desde PHP
        model = genai.GenerativeModel(model_name=modelo_ia) 

        # 6. PROMPT Y DIAGNÓSTICO
        prompt = generar_prompt_experto(index_name, stats, meta, aspctos_inv)
        response = model.generate_content(prompt)
        diagnostico_texto = response.text

        # 7. GENERACIÓN DE PDF (Base64)
        pdf_base64 = generar_pdf_diagnostico(diagnostico_texto, nmbre_fnca)

        # 8. RETORNO ESTRUCTURADO FINAL
        return {
            "status": "success",
            "finca": nmbre_fnca,
            "indice": index_name.upper(),
            "estadisticas": stats,
            "metadatos": meta,
            "muestreo_grafica": muestras,
            "Diagnostico_ia": pdf_base64
        }

    except Exception as e:
        logger.error(f"❌ Error crítico en el análisis: {str(e)}", exc_info=True)
        # Devolvemos un 200 con status error para que el JS capture el mensaje
        return {"status": "error", "msg": str(e)}

    finally:
        # Limpieza garantizada del archivo temporal
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
