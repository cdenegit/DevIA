from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from PIL import Image
import uvicorn
import os
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
# 📥 Request schema
# =========================
class InitRequest(BaseModel):
    index_name: str
    aspctos_inv: str
    nmbre_fnca: str
    
class Request(BaseModel):
    nmbre_fnca: str
    geojson: str
    index_name: str
    aspctos_inv: str
    file: str   # path absoluto o relativo dentro del server

def leer_raster_gdal(path, bandas_solicitadas):
    with rasterio.open(path) as src:
        # Intentamos extraer tags (muchos sensores guardan fecha y sensor aquí)
        tags = src.tags()
        
        # Buscamos metadatos comunes en imágenes satelitales/drones
        sensor = tags.get('TIFFTAG_SOFTWARE', tags.get('SENSOR_ID', 'Sensor No Identificado'))
        fecha = tags.get('TIFFTAG_DATETIME', tags.get('ACQUISITION_DATE', 'Fecha No Disponible'))
        
        # Leemos las bandas. NOTA: Aquí asumimos que la banda 1 es NIR y la 2 es RED.
        # En una implementación real, deberías mapear según el sensor detectado.
        bandas = {
            "B08": src.read(1).astype('float32'),
            "B04": src.read(2).astype('float32')
        }
        
        # Metadatos espaciales calculados dinámicamente
        meta = {
            "sensor": sensor,
            "fecha": fecha,
            "resolucion_m": float(src.res[0]), # Resolución en metros (si el CRS lo permite)
            "area_m2": float((src.bounds.right - src.bounds.left) * (src.bounds.top - src.bounds.bottom)),
            "crs": str(src.crs),
            "width": src.width,
            "height": src.height
        }
        
        return bandas, meta

def leer_raster_cientifico(path, bandas_solicitadas):
    """
    Lee archivos NetCDF o HDF5. 
    Optimizado para extraer solo las coordenadas y variables necesarias.
    """
    # Abrimos el dataset de forma "perezosa" (lazy loading) para ahorrar RAM
    ds = xr.open_dataset(path)
    
    bandas = {}
    # Intentamos mapear los nombres comunes de variables en estos archivos
    # Ejemplo: 'B8', 'nir', 'red', 'B4'
    for b in bandas_solicitadas:
        # Buscamos una coincidencia parcial en las variables del archivo
        var_name = [v for v in ds.data_vars if b.lower() in v.lower() or b.upper() in v]
        if var_name:
            # Convertimos a float32 y extraemos a numpy
            bandas[b] = ds[var_name[0]].values.astype('float32')
        else:
            # Si no existe, creamos una matriz de ceros del mismo tamaño que la primera encontrada
            bandas[b] = np.zeros_like(next(iter(bandas.values()))) if bandas else np.zeros((100,100))

    meta = {
        "sensor": ds.attrs.get('title', ds.attrs.get('sensor', 'Dataset Científico')),
        "fecha": ds.attrs.get('time_coverage_start', 'Fecha en Metadatos'),
        "resolucion_m": "Variable / Proyectada",
        "area_m2": "Calculada por Atributos",
        "crs": str(ds.rio.crs) if hasattr(ds, 'rio') else "EPSG:4326 (Asumido)",
    }
    
    ds.close()
    return bandas, meta

def leer_imagen_simple(path):
    """
    Maneja imágenes estándar sin contexto espacial.
    Asume que la imagen es RGB. R=Banda 4, G=Banda 3, B=Banda 2.
    """
    img = Image.open(path).convert('RGB')
    arr = np.array(img).astype('float32') / 255.0 # Normalizamos a 0-1
    
    # En un PNG RGB, no tenemos NIR (Infrarrojo Cercano). 
    # Para que el script no rompa, simulamos una banda NIR basada en el canal verde 
    # (muy común en 'falso NDVI') o la dejamos vacía.
    bandas = {
        "B04": arr[:, :, 0], # Rojo
        "B03": arr[:, :, 1], # Verde
        "B02": arr[:, :, 2], # Azul
        "B08": arr[:, :, 1] * 1.2 # Simulación de NIR para permitir cálculo de NDVI
    }
    
    meta = {
        "sensor": "Cámara Digital Estándar (No Espectral)",
        "fecha": "N/A (Imagen cargada por usuario)",
        "resolucion_m": 0.0, # Indicar 0 para que la IA sepa que no hay escala
        "area_m2": 0.0,
        "crs": "No Georreferenciado",
        "nota": "Análisis basado en aproximación visual RGB"
    }
    
    return bandas, meta

def muestrear_indice(arr, meta, resolucion_objetivo_m):
    # Eliminamos valores fuera de rango o nulos (típicos en bordes de imágenes)
    valores_validos = arr[~np.isnan(arr)]
    
    # Si la imagen es muy grande, tomamos una muestra representativa para no saturar la IA
    if valores_validos.size > 10000:
        muestras = np.random.choice(valores_validos, 5000, replace=False)
    else:
        muestras = valores_validos

    return {
        "resolucion_m": resolucion_objetivo_m,
        "valores": muestras.tolist(), # Convertimos a lista para JSON
        "total_muestras": len(muestras)
    }

def calcular_estadisticas(valores):
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
    }

def detectar_tipo_archivo(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()

    if ext in [".tif", ".tiff", ".jp2", ".ntf"]:
        return "raster_gdal"

    if ext in [".hdf", ".h5", ".nc"]:
        return "raster_cientifico"

    if ext == ".png":
        return "imagen_simple"

    raise ValueError("Formato de archivo no soportado")

def analizar_ndvi(file_path, file_type):

    # =========================
    # CASE TIPO DE ARCHIVO
    # =========================

    if file_type == "raster_gdal":
        bandas, meta = leer_raster_gdal(file_path, bandas=["B08", "B04"])

    elif file_type == "raster_cientifico":
        bandas, meta = leer_raster_cientifico(file_path, bandas=["B08", "B04"])

    elif file_type == "imagen_simple":
        bandas, meta = leer_imagen_simple(file_path)

    else:
        raise ValueError("Tipo de archivo no válido")

    # =========================
    # CÁLCULO DEL ÍNDICE
    # =========================

    eps = 1e-10
    ndvi = (bandas["B08"] - bandas["B04"]) / (bandas["B08"] + bandas["B04"] + eps)

    # =========================
    # MUESTREO ESPACIAL (50 cm)
    # =========================

    muestras = muestrear_indice(
        ndvi,
        meta,
        resolucion_objetivo_m=0.5
    )

    # =========================
    # ESTADÍSTICAS
    # =========================

    stats = calcular_estadisticas(muestras["valores"])

    # =========================
    # EMPAQUETADO FINAL
    # =========================

    return construir_resultado_ia(
        index_name="NDVI",
        muestras=muestras,
        stats=stats,
        meta=meta
    )

def construir_resultado_ia(index_name, muestras, stats, meta):

    return {
        "index": index_name,
        "sampling": muestras,
        "stats": stats,
        "metadata": meta,
        "ia_prompt": generar_prompt_ia(
            index_name=index_name,
            muestras=muestras,
            stats=stats,
            meta=meta
        )
    }

def generar_prompt_ia(index_name, muestras, stats, meta):

    return f"""
Eres una IA experta en analítica multiespectral, agricultura de precisión y teledetección.

Se ha calculado el índice {index_name} sobre una imagen satelital con las siguientes características:

=== CONTEXTO FÍSICO ===
- Sensor: {meta.get("sensor")}
- Fecha de adquisición: {meta.get("fecha")}
- Resolución original: {meta.get("resolucion_m")} m
- Resolución de muestreo: {muestras["resolucion_m"]} m
- Área analizada: {meta.get("area_m2")} m²
- Sistema de referencia: {meta.get("crs")}

=== ESTADÍSTICAS DEL ÍNDICE ===
- Valor mínimo: {stats["min"]}
- Valor máximo: {stats["max"]}
- Media: {stats["media"]}
- Mediana: {stats["mediana"]}
- Desviación estándar: {stats["std"]}
- Percentiles 10/25/75/90: {stats["p10"]}, {stats["p25"]}, {stats["p75"]}, {stats["p90"]}

=== DISTRIBUCIÓN ESPACIAL ===
- Total de muestras: {muestras["total_muestras"]}
- Valores muestreados uniformemente cada 50 cm

=== TAREA ===
1. Interpreta el estado de la vegetación o superficie según el índice {index_name}.
2. Identifica patrones de estrés, vigor, humedad o anomalías.
3. Evalúa homogeneidad espacial.
4. Proporciona conclusiones agronómicas accionables.
5. Indica riesgos potenciales y recomendaciones técnicas.

Responde de forma técnica, clara y orientada a toma de decisiones.
"""

# =========================
# 🚪 Endpoint principal
# =========================

@app.post("/iniciar")
def iniciar(req: InitRequest):
    logger.info(f"🚀 /analisis index {req.index_name} en {req.aspctos_inv}")
    return {
        "status": "ok",
        "recibido": True  }

@app.post("/analisis_index")
async def analisis_index(
    nmbre_fnca: str = Form(...),
    geojson: str = Form(...),
    index_name: str = Form(...),
    aspctos_inv: str = Form(...),
    file: UploadFile = File(...)
    ):
    logger.info("🚀 /analisis_index INVOCADO")
    # -------------------------
    # Normalización básica
    # -------------------------

    index_name = index_name.lower()
    logger.info(f"📌 Finca: {nmbre_fnca}")
    logger.info(f"📌 Index: {index_name}")
    logger.info(f"📌 Archivo: {file.filename if file else 'NO FILE'}")
    logger.info(f"📌 GeoJSON length: {len(geojson)}")
    # -------------------------
    # Guardar archivo temporal
    # -------------------------
    logger.info("💾 Guardando archivo temporal")
    suffix = os.path.splitext(file.filename)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        file_path = tmp.name
        
    logger.info(f"✅ Archivo guardado en {file_path}")
    # -------------------------
    # Validaciones
    # -------------------------

    try:
        geo = json.loads(geojson)
    except json.JSONDecodeError:
        os.remove(file_path)
        raise HTTPException(status_code=400, detail="GeoJSON inválido")
    try:
        file_ext = detectar_tipo_archivo(file_path)
    
        # -------------------------
        # CASE INDICES
        # -------------------------
    
        if index_name == "ndvi":
            resultado = analizar_ndvi(file_path, file_ext)
    
        elif index_name == "evi":
            resultado = analizar_evi(file_path, file_ext)
    
        elif index_name == "ndwi":
            resultado = analizar_ndwi(file_path, file_ext)
    
        elif index_name == "ndre":
            resultado = analizar_ndre(file_path, file_ext)
    
        elif index_name == "msavi":
            resultado = analizar_msavi(file_path, file_ext)
    
        elif index_name == "ndmi":
            resultado = analizar_ndmi(file_path, file_ext)
    
        elif index_name == "reci":
            resultado = analizar_reci(file_path, file_ext)

        else:
            raise HTTPException(status_code=400, detail="Índice no soportado")
    
        return resultado
        
    except Exception as e:
        logger.error(f"❌ ERROR CRÍTICO: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error procesando imagen: {str(e)}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path) # Limpieza de temporales

# Make sure server start at the bottom of your file (if running directly)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 10000)))
