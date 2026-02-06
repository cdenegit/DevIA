from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
import os
import json
import tempfile
import numpy as np

app = FastAPI()

# =========================
# 📥 Request schema
# =========================

class AnalisisIndexRequest(BaseModel):
    nmbre_fnca: str
    geojson: str
    index_name: str
    aspctos_inv: str
    file_path: str   # path absoluto o relativo dentro del server

def leer_raster_gdal(path, bandas):
    """
    Usa rasterio / GDAL
    Retorna:
      bandas: dict { "B08": np.array, ... }
      meta: resolución, CRS, fecha, sensor, bbox
    """
    pass
def leer_raster_gdal(path, bandas):
    """
    Usa rasterio / GDAL
    Retorna:
      bandas: dict { "B08": np.array, ... }
      meta: resolución, CRS, fecha, sensor, bbox
    """
    pass


def leer_raster_cientifico(path, bandas):
    """
    Usa h5py / netCDF4 / xarray
    """
    pass


def leer_imagen_simple(path):
    """
    PNG sin georreferencia.
    Requiere asumir resolución o marcar como 'no georreferenciado'
    """
    pass


def muestrear_indice(arr, meta, resolucion_objetivo_m):
    """
    Retorna:
    {
        "resolucion_m": 0.5,
        "valores": [...],
        "coordenadas": [...],
        "total_muestras": N
    }
    """
    pass

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

@app.post("/analisis_index")
async def analisis_index(
    nmbre_fnca: str = Form(...),
    geojson: str = Form(...),
    index_name: str = Form(...),
    aspctos_inv: str = Form(...),
    file: UploadFile = File(...)
):

    # -------------------------
    # Normalización básica
    # -------------------------

    index_name = index_name.lower()

    # -------------------------
    # Guardar archivo temporal
    # -------------------------

    suffix = os.path.splitext(file.filename)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        file_path = tmp.name

    # -------------------------
    # Validaciones
    # -------------------------

    try:
        geo = json.loads(geojson)
    except json.JSONDecodeError:
        os.remove(file_path)
        raise HTTPException(status_code=400, detail="GeoJSON inválido")

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

# Make sure server start at the bottom of your file (if running directly)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 10000)))
