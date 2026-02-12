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
    
def generar_prompt_experto(index_name, stats, meta, aspectos, indices_multiples=None, stats_bandas=None):
    finca = meta.get('finca', 'N/A')
    sensor = meta.get('sensor', 'No especificado')
    fecha = meta.get('fecha', 'N/A')
    # Limpieza de la consulta para evitar el ""
    objetivo = aspectos if (aspectos and aspectos.strip()) else "Realizar un diagnóstico Agronomico integral del los Datos."
       
    # --- OPTIMIZACIÓN DE UNIDADES ---
    area_ha = meta.get('area_m2', 0) / 10000
    # Convertimos resolución a cm para que la IA entienda que es un DRONE
    gsd_cm = meta.get('resolucion_m', 0) * 100
    
    # Extraemos el NIR específicamente para darle una instrucción de "Alerta" a la IA
    nir_val = stats_bandas.get('NIR', {}).get('media', 0) if stats_bandas else 0
    alerta_biomasa = "ALERTA: NIR muy bajo. Priorizar análisis de densidad foliar." if nir_val < 0.25 else "NIR Normal."
    
    # Formateo de estadísticas principales
    s = {k: f"{v:.4f}" if isinstance(v, (int, float)) else v for k, v in stats.items()}

    # --- BLOQUE DE FIRMA ESPECTRAL (BANDAS PURAS) ---
    bloque_firmas = ""
    if stats_bandas:
        bloque_firmas = "\n=== FIRMA ESPECTRAL PROMEDIO (Reflectancia) ===\n"
        for banda, val in stats_bandas.items():
            bloque_firmas += f"- {banda}: {val['media']:.3f}\n"

    # --- BLOQUE COMPARATIVO (OTROS ÍNDICES) ---
    bloque_comparativo = ""
    if indices_multiples:
        bloque_comparativo = "\n=== CORRELACIÓN DE ÍNDICES ===\n"
        for idx, val in indices_multiples.items():
            if idx.lower() != index_name.lower(): # No repetir el principal
                bloque_comparativo += f"- {idx.upper()}: Media={val['media']:.4f}\n"

    prompt = f'''
    Eres un Agente de IA experto en Agronomía de Precisión y Teledetección. 
    Analiza los datos de la finca "{finca}" capturados por el sensor {sensor}.

    === CONTEXTO GEOPACIAL ===
    - Sensor: {sensor} (GSD: {gsd_cm:.2f} cm/px)
    - Fecha de Captura: {fecha}
    - Superficie Analizada: {area_ha:.2f} Hectáreas
    - Análisis Líder: {index_name.upper()}
    {bloque_firmas}
    {bloque_comparativo}

    === RADIOGRAFÍA ESTADÍSTICA DEL {index_name.upper()} ===
    - Comportamiento: Media de {s['media']} con una desviación de {s['std']}.
    - Rango Dinámico: [{s['min']} a {s['max']}]
    - Segmentación de Vigor:
      * Zonas Críticas (P10): < {s['p10']}
      * Zonas de Alerta (P25): < {s['p25']}
      * Zonas de Vigor Bueno (P75): > {s['p75']}
      * Zonas de Vigor Óptimo (P90): > {s['p90']}

    === CONSULTA DEL PRODUCTOR ===
    "{objetivo}"

    === TAREA DE DIAGNÓSTICO PROFESIONAL ===
    1. Análisis de Firma: Cruce de Datos: Relaciona el {index_name.upper()} con el NIR. confirma pérdida de biomasa. Si el NDWI es bajo, cruza datos con estrés hídrico.?
    2. Evaluación de Vigor: Identifica si la variabilidad (Std) sugiere necesidad de fertilización diferenciada.
    3. Variabilidad: Evalúa la Desviación Estándar. ¿La finca requiere manejo por sitio específico (Mse) o es uniforme?
    3. Acción Agronómica: Proporciona 3 recomendaciones de Alertas o Mejoraa y 3 basadas en el GSD de {gsd_cm:.1f} cm (aprovechando la alta resolución).
    Responde de forma técnica pero comprensible para un agricultor, usando Markdown.
    '''
    return prompt
    
# =========================
# 📂 Funciones de Lectura
# =========================

def leer_raster_gdal(path, bandas_solicitadas=None):
    with rasterio.open(path) as src:
        num_bandas = src.count
        logger.info(f"📖 Rasterio procesando: {num_bandas} bandas detectadas.")
        
        def normalizar_banda(arr):
            max_val = np.max(arr)
            if max_val > 255: 
                return arr.astype('float32') / 10000.0
            elif max_val > 1.0: 
                return arr.astype('float32') / 255.0
            return arr.astype('float32')

        # --- LÓGICA DE ASIGNACIÓN INTELIGENTE ---
        if num_bandas >= 3:
            # Si tiene 3 o más, asumimos orden estándar RGB para las primeras 3
            # Pero si tiene 8 o más (como Sentinel completo), el orden cambia.
            # Para este FIX, mapeamos basado en la realidad de tus archivos RGB:
            r = normalizar_banda(src.read(1))
            g = normalizar_banda(src.read(2))
            b = normalizar_banda(src.read(3))
            
            # Si es una imagen RGB (3 bandas), no hay NIR real. 
            # Usamos el canal Rojo como B04 y el Verde como B08 para "simular" vigor 
            # o simplemente no dar negativos absurdos.
            bandas = {
                "B08": normalizar_banda(src.read(8)) if num_bandas >= 8 else g, # NIR real o Verde
                "B04": r, # Rojo
                "B03": g, # Verde
                "B02": b, # Azul
                "B05": normalizar_banda(src.read(5)) if num_bandas >= 5 else g
            }
        else:
            # Fallback para monobanda (tu código original)
            b1 = normalizar_banda(src.read(1))
            bandas = {
                "B08": b1, 
                "B04": normalizar_banda(src.read(2)) if num_bandas >= 2 else b1,
                "B03": normalizar_banda(src.read(3)) if num_bandas >= 3 else b1,
                "B02": normalizar_banda(src.read(4)) if num_bandas >= 4 else b1,
                "B05": normalizar_banda(src.read(5)) if num_bandas >= 5 else b1
            }

# --- EXTRACCIÓN DE METADATOS OPTIMIZADA ---
        tags = src.tags()
        
        # 1. Búsqueda exhaustiva del Sensor
        sensor = tags.get('TIFFTAG_SOFTWARE', 
                 tags.get('SENSOR_ID', 
                 tags.get('Make', 'Micasense'))) # Micasense suele ir en 'Make'
        
        # 2. Búsqueda exhaustiva de la Fecha
        fecha = tags.get('TIFFTAG_DATETIME', 
                tags.get('ACQUISITION_DATE', 
                tags.get('DateTime', 'Fecha No Disponible')))

        # 3. CORRECCIÓN DE RESOLUCIÓN (Metros vs Grados)
        raw_res = src.res[0]
        res_m = raw_res
        
        # Si la resolución es pequeñísima (ej: 9.02e-05), está en grados.
        # Convertimos grados a metros aproximadamente (1 grado ≈ 111,111 metros)
        if raw_res < 0.01 and raw_res != 0:
            res_m = raw_res * 111111  # Conversión simple a nivel de Ecuador
        elif raw_res == 1.0 or raw_res == 0:
            res_m = 0.05  # Valor fallback por defecto (5cm)

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
    # 1. Extraemos las bandas base para la validación inicial
    # Usamos B8 y B4 como estándares (NIR y RED)
    B8 = bandas.get("B08")
    B4 = bandas.get("B04") if bandas.get("B04") is not None else B8
    
    # --- PROTECCIÓN PARA IMÁGENES DE 1 BANDA (La que te gustó) ---
    # Si NIR y RED son iguales, el NDVI daría 0. Devolvemos la banda original.
    if np.array_equal(B8, B4):
        logger.warning(f"⚠️ Imagen monobanda detectada para {index_name}. Usando reflectancia base.")
        return B8 

    # --- CASO ESPECIAL: NDVI (Tu lógica explícita) ---
    if index_name.upper() == 'NDVI':
        logger.info("🧪 Calculando NDVI con lógica explícita")
        with np.errstate(divide='ignore', invalid='ignore'):
            idx = (B8 - B4) / (B8 + B4 + eps)
            return np.nan_to_num(idx, nan=0.0)

    # --- RESTO DE ÍNDICES (EVI, MSAVI, etc.) ---
    # Si no es NDVI, buscamos en el diccionario de fórmulas
    B3 = bandas.get("B03") if bandas.get("B03") is not None else B8
    B2 = bandas.get("B02") if bandas.get("B02") is not None else B4
    B5 = bandas.get("B05") if bandas.get("B05") is not None else B4

    formulas = {
        "evi":   lambda: 2.5 * ((B8 - B4) / (B8 + 6 * B4 - 7.5 * B2 + 1 + eps)),
        "ndwi":  lambda: (B3 - B8) / (B3 + B8 + eps),
        "ndre":  lambda: (B8 - B5) / (B8 + B5 + eps),
        "msavi": lambda: (2 * B8 + 1 - np.sqrt(np.maximum(0, (2 * B8 + 1)**2 - 8 * (B8 - B4)))) / 2,
        "reci":  lambda: (B8 / (B4 + eps)) - 1
    }

    func = formulas.get(index_name.lower())
    
    if not func:
        logger.warning(f"⚠️ Índice {index_name} no reconocido, usando B8 como fallback.")
        return B8

    try:
        res = func()
        return np.nan_to_num(res, nan=0.0, posinf=1.0, neginf=-1.0)
    except Exception as e:
        logger.error(f"❌ Error en cálculo de {index_name}: {e}")
        return B8


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
    logger.info(f"🚀 Wake-up recibido para finca: {req.nmbre_fnca}")
    
    # ESTRATEGIA DE PRE-CARGA:
    # Ejecutamos una operación matemática mínima de numpy para asegurar que esté en RAM
    test_array = np.array([0.1, 0.5, 0.9])
    test_mean = np.mean(test_array)
    
    # Intentamos acceder a la versión de rasterio para forzar la carga de sus drivers C
    raster_version = rasterio.__version__
    
    return {
        "status": "warm", 
        "libs_ready": True, 
        "raster_version": raster_version,
        "msg": "Servicio caliente y listo"
    }

@app.post("/analisis_index")
async def analisis_index(
    nmbre_fnca: str = Form(...),
    geojson: str = Form(...),
    index_name: str = Form(...),
    aspctos_inv: str = Form(...),
    gemini_key: str = Form(...),
    modelo_ia: str = Form(...),
    file: UploadFile = File(...) ):
        
    # 1. Gestión de archivo
    file_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1].lower()
    temp_path = os.path.join(tempfile.gettempdir(), f"{file_id}_{file.filename}")
    
    try:
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)
        
        # 2. Lectura según tipo
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
        logger.info(f"📥 Info finca: {nmbre_fnca}  bandas {bandas}  meta {meta} ")
        # 3. CÁLCULO UNIFICADO (Usando tu función de álgebra corregida)
        # Esta función ya maneja NDVI, EVI, MSAVI, etc.

        # 3. PROCESAMIENTO DE ÍNDICES
        indices_calculados = None        
        if index_name.upper() == "TODOS":
            indices_calculados = {}
            for idx_key in ["ndvi", "evi", "ndwi", "msavi", "ndre", "reci"]:
                res_idx = ejecutar_calculo_indice(bandas, idx_key)
                indices_calculados[idx_key] = {
                    "media": float(np.mean(res_idx)),
                    "max": float(np.max(res_idx)),
                    "min": float(np.min(res_idx))
                }
            # Asignamos NDVI a idx_map para que las estadísticas generales sigan funcionando
            idx_map = ejecutar_calculo_indice(bandas, "ndvi")
        else:
            idx_map = ejecutar_calculo_indice(bandas, index_name)

        # 4. ESTADÍSTICAS Y MUESTREO (Ahora idx_map siempre existe)
        valores_limpios = idx_map[~np.isnan(idx_map)]
        if valores_limpios.size == 0:
            raise ValueError("El sensor no retornó datos válidos para este índice.")

        # Estas stats se basan en idx_map (el índice elegido o el NDVI si es TODOS)
        stats = calcular_estadisticas_pro(valores_limpios)

        # --- ESTADÍSTICAS DE BANDAS PURAS ---
        stats_bandas = {}
        for nombre_banda, matriz in bandas.items():
            # Solo bandas reales, evitamos procesar índices aquí
            stats_bandas[nombre_banda.upper()] = {
                "media": float(np.nanmean(matriz)),
                "max": float(np.nanmax(matriz))
            }        
        
        num_muestras = min(2000, len(valores_limpios))
        muestras = np.random.choice(valores_limpios, num_muestras, replace=False).tolist()

        # 5. CONFIGURACIÓN DINÁMICA DE IA
        if not gemini_key:
            raise ValueError("API Key de Gemini no proporcionada por el servidor PHP.")
        
        genai.configure(api_key=gemini_key)
        # model = genai.GenerativeModel, se Usa la variable modelo_ia enviada desde PHP
        model = genai.GenerativeModel(model_name=modelo_ia) 

        # 6. PROMPT Y DIAGNÓSTICO
        prompt = generar_prompt_experto( index_name, stats, meta, aspctos_inv, indices_multiples=indices_calculados )
        #response = model.generate_content(prompt)
        #diagnostico_texto = response.text

        # 7. GENERACIÓN DE PDF (Base64)
        # pdf_base64 = generar_pdf_diagnostico(diagnostico_texto, nmbre_fnca)
        pdf_base64 = generar_pdf_diagnostico(prompt, nmbre_fnca)

        # 8. RETORNO ESTRUCTURADO FINAL

        resultado = {
            "status": "success",
            "finca": nmbre_fnca,
            "indice": index_name.upper(),
            "indices_calculados": indices_calculados if index_name.upper() == "TODOS" else None,
            "stats_bandas": stats_bandas, 
            "estadisticas": stats,
            "metadatos": meta,
            "muestreo_grafica": muestras,
            "Diagnostico_ia": pdf_base64
        }
        
        logger.info(f"🚀 Enviando respuesta exitosa para {nmbre_fnca}")
        return resultado

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
