from fastapi import FastAPI
from pydantic import BaseModel
import json
import os
from sentinelhub import (
    SHConfig, BBox, CRS, DataCollection, SentinelHubCatalog,
    SentinelHubRequest, MimeType, bbox_to_dimensions
)
from shapely.geometry import shape
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import uvicorn

# ===============================
# FastAPI
# ===============================
app = FastAPI()

class Req(BaseModel):
    geojson: str
    fecha_ini: str
    fecha_fin: str


# ===============================
# SentinelHub Config (OAuth2)
# ===============================
config = SHConfig()

config.sh_client_id = "51f7ce9b-3718-4960-99b6-65f3f963611d"
config.sh_client_secret = "CF7oglmD9yLwefP3Od30Tg8ZBuciiMmF"
config.sh_base_url = "https://services.sentinel-hub.com"

if not config.sh_client_id or not config.sh_client_secret:
    print("⚠ ERROR: Credenciales OAuth2 incompletas.")


# ===============================
# BÚSQUEDA DE IMÁGENES
# ===============================
def buscar_imagenes(geom, fecha_ini, fecha_fin):

    bbox = BBox(bbox=geom.bounds, crs=CRS.WGS84)
    catalog = SentinelHubCatalog(config=config)

    # Filtro CQL2 JSON
    filter_cql2_json = {
        "op": "and",
        "args": [
            {
                "op": "<",
                "args": [
                    {"property": "eo:cloud_cover"},
                    70
                ]
            }
        ]
    }

    search = catalog.search(
        collection=DataCollection.SENTINEL2_L2A,
        bbox=bbox,
        time=(fecha_ini, fecha_fin),
        filter=filter_cql2_json,
        filter_lang="cql2-json",
        limit=30
    )

    items = list(search)

    if not items:
        return []

    # Ordenarlos por fecha
    items_sorted = sorted(items, key=lambda x: x["properties"]["datetime"])
    selected = items_sorted[-3:]  # Las 3 más recientes

    # Evalscript Sentinel-2 bandas necesarias
    evalscript = """
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

    resultados = []
    for item in selected:

        timestamp = item["properties"]["datetime"]

        req = SentinelHubRequest(
            evalscript=evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A,
                    time_interval=(timestamp, timestamp)
                )
            ],
            responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
            bbox=bbox,
            size=bbox_to_dimensions(bbox, 10),
            config=config
        )

        resultados.append(req.get_data())

    return resultados


# ===============================
# Cálculo de índices vegetativos
# ===============================
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


# ===============================
# Heatmap Base64
# ===============================
def generar_heatmap(indice, nombre):
    plt.figure(figsize=(6,6))
    plt.imshow(indice, cmap="RdYlGn")
    plt.colorbar()
    plt.title(nombre)

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ===============================
# Diagnóstico simple
# ===============================
def diagnostico_indice(indice, nombre):
    avg = float(np.nanmean(indice))

    if nombre == "NDVI":
        if avg < 0.2: desc = "Vegetación muy estresada o sin cobertura."
        elif avg < 0.5: desc = "Vegetación moderada."
        else: desc = "Vegetación saludable."

    elif nombre == "NDMI":
        if avg < 0.2: desc = "Baja humedad."
        elif avg < 0.5: desc = "Humedad media."
        else: desc = "Buena humedad."

    else:
        desc = f"Valor medio: {avg:.2f}"

    return desc


# ===============================
# Endpoint principal
# ===============================
@app.post("/analizar")
def analizar(req: Req):
    try:
        geo = json.loads(req.geojson)
        geom = shape(geo)

        imgs = buscar_imagenes(geom, req.fecha_ini, req.fecha_fin)

        if len(imgs) == 0:
            return {"status": "error", "msg": "No se encontraron imágenes compatibles."}

        img = imgs[-1][0]  # TIFF (H, W, 7 bandas)
        bandas = img.transpose((2, 0, 1))  # -> (7, H, W)

        indices = calc_indices(bandas)

        resultados = {}
        for nombre, matriz in indices.items():
            resultados[nombre] = {
                "img_base64": generar_heatmap(matriz, nombre),
                "diagnostico": diagnostico_indice(matriz, nombre)
            }

        return {"status": "ok", "indices": resultados}

    except Exception as e:
        return {"status": "error", "msg": f"Error inesperado: {str(e)}"}


if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
