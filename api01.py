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
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer, PageBreak
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
    import numpy as np

    indice = np.nan_to_num(indice, nan=0.0)
    avg = float(np.mean(indice))
    std = float(np.std(indice))

    # Clasificación por umbrales
    area_sana = np.sum(indice >= 0.5) / indice.size * 100
    area_media = np.sum((indice >= 0.3) & (indice < 0.5)) / indice.size * 100
    area_estres = np.sum(indice < 0.3) / indice.size * 100

    resumen = ""
    color = "amarillo"

    if nombre in ["NDVI", "MSAVI", "EVI", "NDRE", "RECI"]:
        if avg >= 0.6:
            color = "verde"
            resumen = (
                "La vegetación presenta un estado general saludable, "
                "con buena actividad fotosintética y vigor adecuado."
            )
        elif avg >= 0.35:
            color = "amarillo"
            resumen = (
                "Se observa una condición vegetal moderada. "
                "Existen zonas con buen desarrollo y otras con posible estrés."
            )
        else:
            color = "rojo"
            resumen = (
                "La cobertura vegetal presenta signos claros de estrés, "
                "posiblemente asociados a déficit hídrico, suelo degradado "
                "o manejo inadecuado."
            )

    elif nombre == "NDMI":
        if avg >= 0.4:
            color = "verde"
            resumen = "La humedad vegetal es adecuada y consistente en la mayor parte del área."
        elif avg >= 0.25:
            color = "amarillo"
            resumen = "Humedad media, con posibles zonas de estrés hídrico incipiente."
        else:
            color = "rojo"
            resumen = "Baja humedad detectada. Riesgo de estrés hídrico significativo."

    # Diagnóstico IA-like estructurado
    diagnostico = (
        f"<b>Resumen:</b> {resumen}<br/>"
        f"<b>Valor medio:</b> {avg:.2f}<br/>"
        f"<b>Variabilidad:</b> {std:.2f}<br/>"
        f"<b>Distribución espacial:</b><br/>"
        f"- Área saludable: {area_sana:.1f}%<br/>"
        f"- Área moderada: {area_media:.1f}%<br/>"
        f"- Área estresada: {area_estres:.1f}%"
    )

    resumen_web = (
        f"Estado general: {resumen.split('.')[0]}."
    )

    return {
        "diagnostico_detallado": diagnostico,
        "resumen_web": resumen_web,
        "color": color
    }

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
        story.append(Paragraph(f"<b>{nombre}</b>", styles['Heading1']))
        story.append(Spacer(1, 12))
    
        story.append(Paragraph(data.get("diagnostico", ""), styles['BodyText']))
        story.append(Spacer(1, 12))
    
        img_b64 = data.get("img_base64")
        if img_b64:
            img_bytes = base64.b64decode(img_b64)
            img_path = f"/tmp/{nombre}.png"
            with open(img_path, "wb") as f:
                f.write(img_bytes)
            story.append(Image(img_path, width=350, height=350))
            story.append(Spacer(1, 20))
    
        story.append(PageBreak())

    doc.build(story)

    # limpiar archivos temporales de imagen (no el PDF)
    for p in temp_files:
        try:
            os.remove(p)
        except Exception:
            pass

    return file_path

# ================================        
# 9) RPDF Avanzado
# ================================
def crear_pdf_avanzado(indices, rgb_b64, metadata):
    file_path = "/tmp/reporte_eo.pdf"
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(file_path, pagesize=letter)
    story = []

    # ================================
    # PORTADA
    # ================================
    story.append(Paragraph("Reporte Satelital EO", styles["Title"]))
    story.append(Spacer(1, 20))

    story.append(Paragraph(
        "Análisis de índices espectrales a partir de imágenes Sentinel-2",
        styles["BodyText"]
    ))
    story.append(Spacer(1, 20))

    # Imagen geográfica RGB
    if rgb_b64:
        portada_img = "/tmp/portada_rgb.png"
        with open(portada_img, "wb") as f:
            f.write(base64.b64decode(rgb_b64))

        story.append(Image(portada_img, width=420, height=420))
        story.append(Spacer(1, 20))

    # ================================
    # METADATA
    # ================================
    story.append(Paragraph("Metadata del Análisis", styles["Heading2"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph(
        f"<b>BBOX:</b> {metadata.get('bbox')}", styles["BodyText"]
    ))
    story.append(Paragraph(
        f"<b>Resolución:</b> {metadata.get('resolution_m_per_px')} m/pixel",
        styles["BodyText"]
    ))
    story.append(Paragraph(
        f"<b>Dimensiones:</b> {metadata.get('shape')}",
        styles["BodyText"]
    ))

    story.append(PageBreak())

    # ================================
    # ÍNDICES (2 por página)
    # ================================
    count = 0
    for nombre, data in indices.items():

        story.append(Paragraph(nombre, styles["Heading2"]))
        story.append(Spacer(1, 8))

        # imagen índice
        if data.get("img_base64"):
            img_path = f"/tmp/{nombre}.png"
            with open(img_path, "wb") as f:
                f.write(base64.b64decode(data["img_base64"]))

            story.append(Image(img_path, width=350, height=300))
            story.append(Spacer(1, 8))

        # diagnóstico
        story.append(Paragraph(
            f"<b>Diagnóstico:</b> {data.get('diagnostico','')}",
            styles["BodyText"]
        ))
        story.append(Spacer(1, 20))

        count += 1
        if count % 2 == 0:
            story.append(PageBreak())

    doc.build(story)
    return file_path
    
# =====================================
# ENDPOINT PRINCIPAL optimizado + FIX SHAPELY (ahora retorna productos)
# =====================================
@app.post("/analizar")
def analizar(req: Req):
    try:
        # ================================
        # 1) Leer GeoJSON
        # ================================
        geo = json.loads(req.geojson)
        geom = shape(geo)

        if not geom.is_valid:
            geom = geom.buffer(0)

        # ================================
        # 2) Buscar imágenes
        # ================================
        imgs = buscar_imagenes(geom, req.fecha_ini, req.fecha_fin)

        if not imgs:
            return {
                "status": "error",
                "msg": "No se encontraron imágenes.",
                "indices": [],
                "indices_lista": [],
                "imagenes": {},
                "pdf_base64": None,
                "metadata": {}
            }

        # ================================
        # 3) Preparar imagen
        # ================================
        img = imgs[-1]
        arr = np.array(img)

        if arr.ndim == 4:
            arr = arr[0]

        if arr.ndim != 3 or arr.shape[2] < 7:
            raise ValueError("Formato de imagen inesperado")

        bandas = arr.transpose((2, 0, 1)).astype("float32")

        if np.nanmax(bandas) > 2000:
            bandas /= 10000.0

        H, W = bandas.shape[1], bandas.shape[2]

        # ===========================
        # 4) Calcular índices
        # ===========================

        indices_raw = calc_indices(bandas)
        
        indices = {}
        indices_lista = []
        
        for nombre, matriz in indices_raw.items():
        
            # imagen
            img_b64 = generar_heatmap(matriz, nombre)
        
            # diagnóstico + semáforo
            diagnostico, semaforo = diagnostico_indice(matriz, nombre)
        
            # objeto principal (para frontend)
            indices[nombre] = {
                "img_base64": generar_heatmap(matriz, nombre),
                "diagnostico": diag["diagnostico_detallado"],   # PDF
                "resumen": diag["resumen_web"],                 # Web
                "color": diag["color"]
            }
        
            # opcional (si lo usas en PDF o logs)
            indices_lista.append({
                "nombre": nombre,
                "imagen": img_b64,
                "diagnostico": diagnostico,
                "semaforo": semaforo
            })

        # ================================
        # 5) Imagen RGB (B04,B03,B02)
        # ================================
        def stretch01(b):
            lo = np.nanpercentile(b, 2)
            hi = np.nanpercentile(b, 98)
            if hi <= lo:
                return np.zeros_like(b)
            return np.clip((b - lo) / (hi - lo), 0, 1)

        r = stretch01(bandas[2])
        g = stretch01(bandas[1])
        b = stretch01(bandas[0])
        rgb = np.dstack([r, g, b])

        buf = BytesIO()
        plt.imsave(buf, (rgb * 255).astype(np.uint8), format="png")
        buf.seek(0)
        rgb_b64 = base64.b64encode(buf.read()).decode()
        rgb_overlay_b64 = generar_rgb_con_geojson(rgb, geom)

        # ================================
        # 6) Metadata  (SIEMPRE ANTES DEL PDF)
        # ================================
        area_m2, width_m, height_m = bbox_area_meters(geom.bounds)
        res_m, _, _ = choose_resolution(width_m, height_m)
        
        metadata = {
            "shape": [int(H), int(W), int(bandas.shape[0])],
            "bbox": list(map(float, geom.bounds)),
            "resolution_m_per_px": int(res_m),
            "area_m2": float(area_m2)
        }
        
        # ================================
        # 7) PDF avanzado
        # ================================
        pdf_path = crear_pdf_avanzado(
            indices=indices,
            rgb_b64=rgb_b64,
            metadata=metadata
        )
        
        with open(pdf_path, "rb") as f:
            pdf_b64 = base64.b64encode(f.read()).decode()
        
        try:
            os.remove(pdf_path)
        except Exception:
            pass

        # ================================
        # 8) RESPUESTA FINAL
        # ================================
        return {
            "status": "ok",
            "msg": "Procesamiento completado correctamente",
            "indices": indices,                 # dict técnico
            "indices_lista": indices_lista,     # ARRAY para frontend
            "imagenes": {
                "rgb": rgb_b64
            },
            "pdf_base64": pdf_b64,
            "metadata": metadata
        }

    except Exception as e:
        logger.exception("Error inesperado en /analizar: %s", e)
        return {
            "status": "error",
            "msg": str(e),
            "indices": [],
            "indices_lista": [],
            "imagenes": {},
            "pdf_base64": None,
            "metadata": {}
        }

# ================================
# A) Generar RGB + overlay GeoJSON
# ================================

def generar_rgb_con_geojson(rgb, geom):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(rgb)
    
    if geom.geom_type == "Polygon":
        xs, ys = geom.exterior.xy
        ax.plot(xs, ys, color="red", linewidth=2)

    ax.set_axis_off()

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

# Make sure server start at the bottom of your file (if running directly)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 10000)))
