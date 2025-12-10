from fastapi import FastAPI, Response, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import json
import os
import math
import logging
from sentinelhub import (
    SHConfig, BBox, CRS, DataCollection, SentinelHubCatalog,
    SentinelHubRequest, MimeType, bbox_to_dimensions
)
from shapely.geometry import shape, mapping
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import uvicorn

# ==== PDF (ya en tu requirements) ====
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# =====================================
# Logging básico
# =====================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eo-microservice")

# =====================================
# FastAPI
# =====================================
app = FastAPI()

class Req(BaseModel):
    geojson: str
    fecha_ini: str
    fecha_fin: str

# =====================================
# SentinelHub OAuth2 (tus credenciales)
# =====================================
config = SHConfig()
config.download_timeout_seconds = 150  # ejemplo
config.sh_client_id = os.getenv("SH_CLIENT_ID", "51f7ce9b-3718-4960-99b6-65f3f963611d")
config.sh_client_secret = os.getenv("SH_CLIENT_SECRET", "CF7oglmD9yLwefP3Od30Tg8ZBuciiMmF")
# config.instance_id ya no es imprescindible si usas OAuth2 + Sentinel services
config.sh_base_url = "https://services.sentinel-hub.com"
client = SentinelHubDownloadClient(config=config)

# =====================================
# Parámetros de control (ajustables)
# =====================================
MAX_PIXELS = 600 * 600         # máximo píxeles aceptables para H*W
DEFAULT_RES = 10               # metros/píxel preferido (Sentinel-2 = 10m bandas)
MIN_RES = 20                   # no solicitar resoluciones más finas de lo necesario
MAX_RES = 120                  # si el área es gigante, recortar a esta resolución
HEATMAP_DPI = 100              # dpi más pequeño para reducir memoria
HEATMAP_FIGSIZE = (4, 4)      # figura más pequeña
SENTINEL_TIMEOUT = 60         # segundos de timeout para solicitudes de SentinelHub

# =====================================
# HELPERS: cálculo de área aproximada y resolución
# =====================================
def bbox_area_meters(bbox):
    """Approx area in square meters for bbox tuple (minx, miny, maxx, maxy)."""
    minx, miny, maxx, maxy = bbox
    # Approx meters per degree
    mean_lat = (miny + maxy) / 2.0
    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = 111320.0 * math.cos(math.radians(mean_lat))
    width_m = (maxx - minx) * meters_per_deg_lon
    height_m = (maxy - miny) * meters_per_deg_lat
    if width_m < 0: width_m = abs(width_m)
    if height_m < 0: height_m = abs(height_m)
    return width_m * height_m, width_m, height_m

def choose_resolution(width_m, height_m):
    """
    Decide meters-per-pixel resolution so that width_px * height_px <= MAX_PIXELS
    and resolution is reasonable (between DEFAULT_RES and MAX_RES).
    """
    # Start with default resolution
    res = DEFAULT_RES
    # compute pixel dims at default res
    w_px = max(1, int(math.ceil(width_m / res)))
    h_px = max(1, int(math.ceil(height_m / res)))
    pixels = w_px * h_px
    if pixels <= MAX_PIXELS:
        return res, w_px, h_px

    # increase resolution (coarsen) until under MAX_PIXELS
    # target scale factor = sqrt(pixels / MAX_PIXELS)
    scale_factor = math.sqrt(pixels / MAX_PIXELS)
    res = int(math.ceil(res * scale_factor))
    # clamp
    if res < MIN_RES:
        res = MIN_RES
    if res > MAX_RES:
        res = MAX_RES
    w_px = max(1, int(math.ceil(width_m / res)))
    h_px = max(1, int(math.ceil(height_m / res)))
    return res, w_px, h_px

# =====================================
# BUSCAR IMÁGENES (optimizada)
# =====================================
def buscar_imagenes(geom, fecha_ini, fecha_fin, max_items=3):
    """
    Busca y descarga hasta `max_items` imágenes Sentinel-2 L2A recortadas al bbox (no al polígono).
    Optimizaciones:
      - calcula resolución automática según área para limitar memoria
      - usa timeout en requests (SentinelHub client internamente usa requests)
      - convierte bandas a float32 y normaliza si valores altos
    Retorna lista de arrays (H, W, 7) numpy.
    """

    bbox_vals = geom.bounds  # (minx, miny, maxx, maxy)
    bbox = BBox(bbox=bbox_vals, crs=CRS.WGS84)
    catalog = SentinelHubCatalog(config=config)

    # Filtro CQL2 JSON para baja nubosidad
    filtro = {
        "op": "and",
        "args": [
            {"op": "<", "args": [{"property": "eo:cloud_cover"}, 70]}
        ]
    }

    # Buscar metadatos
    try:
        search = catalog.search(
            collection=DataCollection.SENTINEL2_L2A,
            bbox=bbox,
            time=(fecha_ini, fecha_fin),
            filter=filtro,
            filter_lang="cql2-json",
            limit=20
        )
        items = list(search)
    except Exception as e:
        logger.exception("Error buscando en catalog: %s", e)
        raise HTTPException(status_code=502, detail=f"Error buscando metadatos: {str(e)}")

    if not items:
        return []

    items_sorted = sorted(items, key=lambda x: x["properties"]["datetime"])
    selected = items_sorted[-max_items:]

    # compute bbox size in meters and choose resolution
    area_m2, width_m, height_m = bbox_area_meters(bbox_vals)
    res_m_per_px, w_px, h_px = choose_resolution(width_m, height_m)
    logger.info("BBox area m2=%.2f width_m=%.2f height_m=%.2f -> res=%dm/pix => px=%dx%d",
                area_m2, width_m, height_m, res_m_per_px, w_px, h_px)

    # build evalscript
    evalscript = """
        function setup() {
          return { input:["B02","B03","B04","B08","B8A","B11","B12"], output:{bands:7} };
        }
        function evaluatePixel(s) {
          return [s.B02,s.B03,s.B04,s.B08,s.B8A,s.B11,s.B12];
        }
    """

    resultados = []
    for item in selected:
        timestamp = item["properties"]["datetime"]

        try:
            req = SentinelHubRequest(
                evalscript=evalscript,
                responses=responses,
                bbox=bbox,
                size=size,
                config=config,                
                input_data=[
                    SentinelHubRequest.input_data(
                        data_collection=DataCollection.SENTINEL2_L2A,
                        time_interval=(timestamp, timestamp)
                    )
                ],
                responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
                bbox=bbox,
                size=bbox_to_dimensions(bbox, res_m_per_px),
                config=config
            )

            # get_data may return list of arrays; we use first element
            data = req.get_data(timeout=SENTINEL_TIMEOUT)
            if not data:
                logger.warning("No data returned for timestamp %s", timestamp)
                continue

            # data is list of numpy arrays; often shape (H, W, bands)
            arr = data[0]

            # convert to float32 early to avoid overflow
            arr = arr.astype("float32")

            # If values appear scaled up (e.g., > 10000), normalize to ~0..1
            if np.nanmax(arr) > 2000:
                arr = arr / 10000.0

            # simple nan handling
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

            resultados.append(arr)

        except Exception as e:
            logger.exception("Error descargando imagen %s: %s", timestamp, e)
            # don't fail the whole loop, continue with next
            continue

    return resultados

# =====================================
# Cálculo de índices vegetativos (sin cambios, operando en float32 normalizado)
# =====================================
def calc_indices(bandas):
    # bandas expected shape (7, H, W)
    B02, B03, B04, B08, B8A, B11, B12 = bandas
    eps = 1e-10
    ndvi = (B08 - B04) / (B08 + B04 + eps)
    evi = 2.5 * (B08 - B04) / (B08 + 6*B04 - 7.5*B02 + 1 + eps)
    ndwi = (B03 - B08) / (B03 + B08 + eps)
    ndre = (B8A - B04) / (B8A + B04 + eps)
    msavi = (2*B08 + 1 - np.sqrt((2*B08 + 1)**2 - 8*(B08 - B04))) / 2
    ndmi = (B11 - B08) / (B11 + B08 + eps)
    reci = (B08 / (B04 + eps)) - 1

    return {
        "NDVI": ndvi,
        "EVI": evi,
        "NDWI": ndwi,
        "NDRE": ndre,
        "MSAVI": msavi,
        "NDMI": ndmi,
        "RECI": reci
    }

# =====================================
# Heatmap Base64 (optimizado, menor DPI / figsize)
# =====================================
def generar_heatmap(indice, nombre):
    plt.figure(figsize=HEATMAP_FIGSIZE)
    # clip values to [-1,1] for better color scaling and to avoid extreme outliers
    arr = np.clip(indice, -1.0, 1.0)
    plt.imshow(arr, cmap="RdYlGn", vmin=-1, vmax=1)
    plt.colorbar()
    plt.title(nombre)
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=HEATMAP_DPI, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

# =====================================
# Diagnóstico (igual)
# =====================================
def diagnostico_indice(indice, nombre):
    # use nanmean on float32 arrays (should not overflow now)
    try:
        avg = float(np.nanmean(indice))
    except Exception:
        avg = 0.0
    if nombre == "NDVI":
        if avg < 0.2: return "Vegetación muy estresada o sin cobertura."
        elif avg < 0.5: return "Vegetación moderada."
        else: return "Vegetación saludable."
    elif nombre == "NDMI":
        if avg < 0.2: return "Baja humedad."
        elif avg < 0.5: return "Humedad media."
        else: return "Buena humedad."
    return f"Valor medio: {avg:.2f}"

# =====================================
# Crear PDF (sin cambios funcionales)
# =====================================
def crear_pdf(indices):
    file_path = "/tmp/diagnostico.pdf"
    doc = SimpleDocTemplate(file_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>Reporte de Índices Vegetativos</b>", styles['Title']))
    story.append(Spacer(1, 20))

    for nombre, data in indices.items():
        story.append(Paragraph(f"<b>{nombre}</b>", styles['Heading2']))
        story.append(Paragraph(data["diagnostico"], styles['BodyText']))

        img_bytes = base64.b64decode(data["img_base64"])
        img_path = f"/tmp/{nombre}.png"
        with open(img_path, "wb") as f:
            f.write(img_bytes)

        story.append(Image(img_path, width=300, height=300))
        story.append(Spacer(1, 20))

    doc.build(story)
    return file_path

# =====================================
# ENDPOINT PRINCIPAL optimizado + FIX SHAPELY
# =====================================
@app.post("/analizar")
def analizar(req: Req):
    try:
        # ================================
        # 1) Leer GeoJSON enviado desde PHP
        # ================================
        geo = json.loads(req.geojson)
        geom = shape(geo)

        # ================================
        # 2) Corrección OGC obligatoria (SELF-INTERSECTIONS, BOWTIES, HOLES)
        # ================================
        try:
            if not geom.is_valid:
                geom = geom.buffer(0)   # ← FIX GEOMETRÍA
        except Exception as gerr:
            return {
                "status": "error",
                "msg": f"Error corrigiendo geometría: {str(gerr)}"
            }

        # ================================
        # 3) Buscar imágenes
        # ================================
        imgs = buscar_imagenes(geom, req.fecha_ini, req.fecha_fin)

        if len(imgs) == 0:
            return {"status": "error", "msg": "No se encontraron imágenes."}

        # ================================
        # 4) Tomar la mejor imagen
        # ================================
        try:
            img = imgs[-1]  # Última imagen temporal

            # Adaptar formato si viene en (1, H, W, 7)
            arr = np.array(img)
            if arr.ndim == 4:
                img = arr[0]

            if img.ndim != 3 or img.shape[2] < 7:
                raise ValueError("Formato de imagen inesperado")

        except Exception as e:
            logger.exception("Error preparando imagen: %s", e)
            return {"status": "error", "msg": f"Error preparando la imagen: {str(e)}"}

        # Convertir a (bands, H, W)
        bandas = img.transpose((2, 0, 1)).astype("float32")

        # Normalización automática si vienen en DN
        if np.nanmax(bandas) > 2000:
            bandas = bandas / 10000.0

        # ================================
        # 5) Calcular índices
        # ================================
        indices_raw = calc_indices(bandas)

        # ================================
        # 6) Heatmaps + diagnósticos
        # ================================
        indices = {}
        for nombre, matriz in indices_raw.items():
            try:
                indices[nombre] = {
                    "img_base64": generar_heatmap(matriz, nombre),
                    "diagnostico": diagnostico_indice(matriz, nombre)
                }
            except Exception as e:
                logger.exception("Error generando heatmap para %s: %s", nombre, e)
                indices[nombre] = {
                    "img_base64": None,
                    "diagnostico": f"Error generando índice: {str(e)}"
                }

        # ================================
        # 7) Respuesta final
        # ================================
        return {
            "status": "ok",
            "indices": indices
        }

    except Exception as e:
        logger.exception("Error inesperado en /analizar: %s", e)
        return {
            "status": "error",
            "msg": f"Error inesperado: {str(e)}"
        }


# =====================================
# /pdf and /dashboard endpoints unchanged (kept in your original file)
# =====================================

# Make sure server start at the bottom of your file (if running directly)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 10000)))
