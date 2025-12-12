from fastapi import FastAPI, Response, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import json
import os
import math
from datetime import datetime, timedelta
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
config.sh_base_url = "https://services.sentinel-hub.com"

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

# utilidad para convertir timestamps ISO a rango de un día
def normalizar_timestamp(ts):
    dt = datetime.fromisoformat(ts.replace("Z", ""))
    start = dt.strftime("%Y-%m-%d")
    end = (dt + timedelta(days=1)).strftime("%Y-%m-%d")
    return (start, end)

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
# BUSCAR IMÁGENES (optimizada - versión de producción)
# =====================================
def buscar_imagenes(geom, fecha_ini, fecha_fin, max_items=3):
    """
    Busca y descarga hasta `max_items` imágenes Sentinel-2 L2A recortadas al bbox (no al polígono).
    Devuelve lista con la primera imagen válida encontrada: [arr] donde arr es numpy array (H, W, bands).
    Si no encuentra ninguna, devuelve [].
    """
    # Logs iniciales
    logger.info("fecha_ini: %s", fecha_ini)
    logger.info("fecha_fin: %s", fecha_fin)
    try:
        max_items = int(max_items)
    except Exception:
        max_items = 3
    if max_items <= 0:
        max_items = 1
    logger.info("max_items recibido: %s", max_items)

    # bbox a partir de la geometría
    bbox_vals = geom.bounds
    bbox = BBox(bbox=bbox_vals, crs=CRS.WGS84)
    logger.info("bbox_vals: %s", bbox_vals)

    catalog = SentinelHubCatalog(config=config)

    # filtro CQL2 para nubosidad
    filtro = {
        "op": "and",
        "args": [
            {"op": "<", "args": [{"property": "eo:cloud_cover"}, 70]}
        ]
    }

    # Buscar metadatos en el catálogo
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
        logger.info("items encontrados en catálogo: %d", len(items))
    except Exception as e:
        logger.exception("Error buscando en catalog: %s", e)
        raise HTTPException(status_code=502, detail=f"Error buscando metadatos: {e}")

    if not items:
        logger.warning("No se encontraron items dentro del rango solicitado.")
        return []

    # ordenar y seleccionar últimos max_items
    items_sorted = sorted(items, key=lambda x: x["properties"]["datetime"])
    selected = items_sorted[-max_items:]
    logger.info("items seleccionados: %d", len(selected))

    if not selected:
        logger.warning("selected está vacío aunque hay items. Revisar max_items.")
        return []

    # calcular resolución óptima
    area_m2, width_m, height_m = bbox_area_meters(bbox_vals)
    res_m_per_px, w_px, h_px = choose_resolution(width_m, height_m)
    logger.info("BBox area m2=%.2f width_m=%.2f height_m=%.2f -> res=%dm/pix => px=%dx%d",
                area_m2, width_m, height_m, res_m_per_px, w_px, h_px)

    # evalscript (bandas B02,B03,B04,B08,B8A,B11,B12)
    evalscript = """
        function setup() {
          return { input:["B02","B03","B04","B08","B8A","B11","B12"], output:{bands:7} };
        }
        function evaluatePixel(s) {
          return [s.B02,s.B03,s.B04,s.B08,s.B8A,s.B11,s.B12];
        }
    """

    # utilidad para convertir timestamp ISO a rango de un día (YYYY-MM-DD, YYYY-MM-DD)
    from datetime import datetime, timedelta
    def normalizar_timestamp_a_dia(ts):
        try:
            # remover Z si la hay y parsear
            dt = datetime.fromisoformat(ts.replace("Z", ""))
        except Exception:
            # si no puede parsear, devolver un rango amplio seguro (fallback)
            logger.warning("No se pudo parsear timestamp '%s', usando rango fallback", ts)
            return (fecha_ini, fecha_fin)
        start = dt.strftime("%Y-%m-%d")
        end = (dt + timedelta(days=1)).strftime("%Y-%m-%d")
        return (start, end)

    # LOOP: intentar descargar cada item seleccionado hasta encontrar la primera imagen válida
    for item in selected:
        timestamp = item["properties"]["datetime"]
        logger.info("Procesando timestamp: %s", timestamp)

        try:
            time_interval = normalizar_timestamp_a_dia(timestamp)

            req = SentinelHubRequest(
                evalscript=evalscript,
                responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
                bbox=bbox,
                size=bbox_to_dimensions(bbox, res_m_per_px),
                input_data=[
                    SentinelHubRequest.input_data(
                        data_collection=DataCollection.SENTINEL2_L2A,
                        time_interval=time_interval
                    )
                ],
                config=config
            )

            data = req.get_data()

            # DEBUG CRÍTICO: inspeccionar qué devuelve SH
            logger.info("Respuesta de SH: type=%s len=%s", type(data), len(data) if data else "None")
            if data:
                logger.info("Shape primer elemento: %s", getattr(data[0], "shape", "sin shape"))

            if not data:
                logger.warning("No data returned para timestamp %s (interval=%s)", timestamp, time_interval)
                continue

            # tomar primer elemento
            arr = data[0]

            # asegurar float32
            arr = arr.astype("float32")

            # normalizar si viene en DN alto
            if np.nanmax(arr) > 2000:
                arr = arr / 10000.0

            # manejos de NaN/Inf
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

            # devolver inmediatamente la primera imagen válida
            return [arr]

        except Exception as e:
            logger.exception("Error descargando imagen %s: %s", timestamp, e)
            # continuar con el siguiente item
            continue

    # si ninguna imagen funcionó
    logger.warning("Ninguna imagen válida fue encontrada entre los items seleccionados.")
    return []

# =====================================
# Cálculo de índices vegetativos (operando en float32 normalizado)
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
# Helper: convertir array float32 (0..1) a PNG base64 (RGB)
# =====================================
def array_to_png_base64(arr01):
    """
    arr01: numpy array HxWx3 with floats expected in ~0..1 range.
    Returns base64 string of PNG.
    """
    # clip and scale to 0..255
    a = np.clip(arr01, 0.0, 1.0)
    uint8 = (a * 255.0).astype(np.uint8)
    buf = BytesIO()
    plt.figure(figsize=(6,6))
    plt.axis('off')
    plt.imshow(uint8)
    plt.tight_layout(pad=0)
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

# =====================================
# Crear PDF (adaptada para usar imágenes en base64 y limpieza)
# =====================================
def crear_pdf(indices):
    """
    indices: dict with keys -> { 'img_base64': str, 'diagnostico': str }
    Returns path to generated PDF.
    """
    file_path = "/tmp/diagnostico.pdf"
    doc = SimpleDocTemplate(file_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>Reporte de Índices Vegetativos</b>", styles['Title']))
    story.append(Spacer(1, 20))

    # Guardar imágenes temporales para ReportLab
    temp_files = []
    for nombre, data in indices.items():
        story.append(Paragraph(f"<b>{nombre}</b>", styles['Heading2']))
        story.append(Paragraph(data.get("diagnostico", ""), styles['BodyText']))

        img_b64 = data.get("img_base64")
        if img_b64:
            img_bytes = base64.b64decode(img_b64)
            img_path = f"/tmp/{nombre}.png"
            with open(img_path, "wb") as f:
                f.write(img_bytes)
            temp_files.append(img_path)
            # ajustar tamaño si muy grande
            story.append(Image(img_path, width=300, height=300))
            story.append(Spacer(1, 20))
        else:
            story.append(Paragraph("Imagen no disponible", styles['BodyText']))
            story.append(Spacer(1, 10))

    doc.build(story)

    # limpiar archivos temporales de imagen (no el PDF)
    for p in temp_files:
        try:
            os.remove(p)
        except Exception:
            pass

    return file_path

# =====================================
# ENDPOINT PRINCIPAL optimizado + FIX SHAPELY (ahora retorna productos)
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
                "msg": f"Error corrigiendo geometría: {str(gerr)}",
                "indices": None,
                "imagenes": None,
                "pdf": None,
                "metadata": None
            }

        # ================================
        # 3) Buscar imágenes
        # ================================
        imgs = buscar_imagenes(geom, req.fecha_ini, req.fecha_fin)

        if len(imgs) == 0:
            return {
                "status": "error",
                "msg": "No se encontraron imágenes.",
                "indices": None,
                "imagenes": None,
                "pdf": None,
                "metadata": None
            }

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
            return {
                "status": "error",
                "msg": f"Error preparando la imagen: {str(e)}",
                "indices": None,
                "imagenes": None,
                "pdf": None,
                "metadata": None
            }

        # Convertir a (bands, H, W)
        bandas = img.transpose((2, 0, 1)).astype("float32")  # shape (7, H, W)

        # Normalización automática si vienen en DN
        if np.nanmax(bandas) > 2000:
            bandas = bandas / 10000.0

        H = bandas.shape[1]
        W = bandas.shape[2]

        # ================================
        # 5) Calcular índices
        # ================================
        indices_raw = calc_indices(bandas)

        # ================================
        # 6) Heatmaps + diagnósticos (ya existían)
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
        # Helper local: convertir array HxWx3 float 0..1 a PNG base64
        # ================================
        def array_to_png_base64(arr_rgb):
            try:
                # asegurar 0..1 float
                arr = np.clip(arr_rgb, 0.0, 1.0)
                arr_uint8 = (arr * 255).astype(np.uint8)
                buf = BytesIO()
                # matplotlib.imsave escribe a buffer sin abrir figura
                plt.imsave(buf, arr_uint8, format="png")
                buf.seek(0)
                return base64.b64encode(buf.read()).decode()
            except Exception as e:
                logger.exception("Error en array_to_png_base64: %s", e)
                return None

        # ================================
        # 7) Generar RGB true-color PNG (B04,B03,B02)
        #    bands order: [B02,B03,B04,B08,B8A,B11,B12] -> indices 0,1,2
        # ================================
        try:
            # seleccionar B04,B03,B02 (indices 2,1,0)
            B02 = bandas[0]
            B03 = bandas[1]
            B04 = bandas[2]

            # stack as float 0..1 using min/max stretch per band
            def stretch01(b):
                lo = np.nanpercentile(b, 2)
                hi = np.nanpercentile(b, 98)
                if hi - lo <= 0:
                    s = b - lo
                    s = s - np.nanmin(s)
                    if np.nanmax(s) > 0:
                        s = s / np.nanmax(s)
                    return np.clip(s, 0.0, 1.0)
                s = (b - lo) / (hi - lo)
                return np.clip(s, 0.0, 1.0)

            r = stretch01(B04)
            g = stretch01(B03)
            b = stretch01(B02)
            rgb = np.dstack([r, g, b])  # HxWx3 floats 0..1
            rgb_b64 = array_to_png_base64(rgb)
        except Exception as e:
            logger.exception("Error generando RGB: %s", e)
            rgb_b64 = None

        # ================================
        # 8) Generar PDF con los heatmaps
        # ================================
        try:
            pdf_path = crear_pdf(indices)
            with open(pdf_path, "rb") as f:
                pdf_b64 = base64.b64encode(f.read()).decode()
            # opcional: eliminar pdf temporal
            try:
                os.remove(pdf_path)
            except Exception:
                pass
        except Exception as e:
            logger.exception("Error generando PDF: %s", e)
            pdf_b64 = None

        # ================================
        # 9) Metadata
        #   - calcular resolución localmente para evitar NameError
        # ================================
        try:
            area_m2, width_m, height_m = bbox_area_meters(geom.bounds)
            res_m_per_px, w_px, h_px = choose_resolution(width_m, height_m)
            res_m_per_px_int = int(res_m_per_px)
        except Exception:
            res_m_per_px_int = None

        metadata = {
            "shape": [int(H), int(W), int(bandas.shape[0])],
            "bbox": list(map(float, geom.bounds)),
            "timestamp": None,  # no hay timestamp disponible aquí (buscarlo requeriría cambiar buscar_imagenes)
            "resolution_m_per_px": res_m_per_px_int
        }

        # ================================
        # 10) Respuesta final (estructura consistente)
        # ================================
        return {
            "status": "ok",
            "msg": "Procesamiento completado correctamente",
            "indices": indices,            # cada uno contiene img_base64 + diagnostico
            "imagenes": {
                "rgb": rgb_b64
            },
            "pdf": pdf_b64,
            "metadata": metadata
        }

    except Exception as e:
        logger.exception("Error inesperado en /analizar: %s", e)
        return {
            "status": "error",
            "msg": f"Error inesperado: {str(e)}",
            "indices": None,
            "imagenes": None,
            "pdf": None,
            "metadata": None
        }


# =====================================
# /pdf and /dashboard endpoints unchanged (kept in your original file)
# =====================================

# Make sure server start at the bottom of your file (if running directly)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 10000)))
