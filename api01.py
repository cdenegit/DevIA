from fastapi import FastAPI
from pydantic import BaseModel
import json
from sentinelhub import SHConfig, SentinelHubRequest, DataCollection, MimeType, bbox_to_dimensions, BBox
from shapely.geometry import shape, mapping
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import tempfile
import requests

app = FastAPI()

# ================================
# MODELO DE ENTRADA DESDE PHP
# ================================
class Req(BaseModel):
    geojson: str
    fecha_ini: str
    fecha_fin: str

# ================================
# CONFIGURACIÓN SENTINELHUB (GRATIS)
# ================================
config = SHConfig()
config.instance_id = "TU_INSTANCE_ID"
config.sh_client_id = "TU_CLIENT_ID"
config.sh_client_secret = "TU_CLIENT_SECRET"

# ================================
# FUNCIÓN: Buscar máximo 3 imágenes multispectrales
# ================================
def buscar_imagenes(geom, fecha_ini, fecha_fin):
    bbox = shape(geom).bounds
    bbox = BBox(bbox, crs=4326)

    evalscript = """
        // Sentinel-2 L2A bandas necesarias
        function setup() {
          return {
            input: ["B02","B03","B04","B08","B8A","B11","B12"],
            output: { bands: 7 }
          };
        }
        function evaluatePixel(s) {
          return [s.B02,s.B03,s.B04,s.B08,s.B8A,s.B11,s.B12];
        }
    """

    request = SentinelHubRequest(
        data_folder=None,
        evalscript=evalscript,
        input_data=[SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L2A,
            time_interval=(fecha_ini, fecha_fin),
            mosaicking_order="mostRecent"
        )],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=bbox_to_dimensions(bbox, 10),
        config=config
    )

    # Limitar a 3 imágenes máximo
    imgs = request.get_data(max_data=3)
    return imgs


# ================================
# CÁLCULO DE ÍNDICES VEGETATIVOS
# ================================
def calc_indices(bandas):
    B02, B03, B04, B08, B8A, B11, B12 = bandas

    eps = 1e-10

    return {
        "NDVI": (B08 - B04) / (B08 + B04 + eps),
        "EVI": 2.5 * (B08 - B04) / (B08 + 6*B04 - 7.5*B02 + 1 + eps),
        "NDWI": (B03 - B08) / (B03 + B08 + eps),
        "NDRE": (B8A - B04) / (B8A + B04 + eps),
        "MSAVI": (2*B08 + 1 - np.sqrt((2*B08 + 1)**2 - 8*(B08 - B04))) / 2,
        "NDMI": (B11 - B08) / (B11 + B08 + eps),
        "RECI": (B08 / (B04 + eps)) - 1
    }


# ================================
# HEATMAP + VALORES SOBRE LA IMAGEN
# ================================
def generar_heatmap(indice, nombre):
    plt.figure(figsize=(6,6))
    plt.imshow(indice, cmap="RdYlGn")
    plt.colorbar()
    plt.title(nombre)

    # Convertir a base64 PNG
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ================================
# DIAGNÓSTICO AUTOMÁTICO POR ÍNDICE
# ================================
def diagnostico_indice(indice, nombre):

    avg = float(np.nanmean(indice))

    if nombre == "NDVI":
        if avg < 0.2: desc = "Vegetación muy estresada o suelo desnudo."
        elif avg < 0.5: desc = "Vegetación moderada, crecimiento limitado."
        else: desc = "Vegetación vigorosa y saludable."

    elif nombre == "NDMI":
        if avg < 0.2: desc = "Humedad baja, posible estrés hídrico."
        elif avg < 0.5: desc = "Humedad moderada."
        else: desc = "Buena retención de humedad."

    else:
        desc = f"Promedio del índice: {avg:.2f}. Patrón típico observado."

    return desc


# ================================
# ENDPOINT PRINCIPAL DESDE TU PHP
# ================================
@app.post("/analizar")
def analizar(req: Req):

    geo = json.loads(req.geojson)
    geom = shape(geo)

    # 1. Buscar máximo 3 imágenes
    imgs = buscar_imagenes(geom, req.fecha_ini, req.fecha_fin)
    if len(imgs) == 0:
        return {"status": "error", "msg": "No hay imágenes disponibles en el rango."}

    # Usar la mejor (última)
    img = imgs[-1][0]  # TIFF → bandas

    # 2. Recorte al polígono
    bandas = img.transpose((2,0,1))  # pasar a bandas separadas

    # 3. Cálculo de índices
    indices = calc_indices(bandas)

    # 4. Generar imágenes para cada índice
    resultados = {}
    for nombre, matriz in indices.items():
        resultados[nombre] = {
            "img_base64": generar_heatmap(matriz, nombre),
            "diagnostico": diagnostico_indice(matriz, nombre)
        }

    return {
        "status": "ok",
        "indices": resultados
    }

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
