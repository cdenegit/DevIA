from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse
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

# ==== PDF ====
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer
from reportlab.lib.styles import getSampleStyleSheet


# =====================================
# FastAPI
# =====================================
app = FastAPI()


class Req(BaseModel):
    geojson: str
    fecha_ini: str
    fecha_fin: str


# =====================================
# SentinelHub OAuth2
# =====================================
config = SHConfig()
config.sh_client_id = "51f7ce9b-3718-4960-99b6-65f3f963611d"
config.sh_client_secret = "CF7oglmD9yLwefP3Od30Tg8ZBuciiMmF"
config.sh_base_url = "https://services.sentinel-hub.com"


# =====================================
# Búsqueda de imágenes
# =====================================
def buscar_imagenes(geom, fecha_ini, fecha_fin):

    bbox = BBox(bbox=geom.bounds, crs=CRS.WGS84)
    catalog = SentinelHubCatalog(config=config)

    filtro = {
        "op": "and",
        "args": [
            {"op": "<", "args": [{"property": "eo:cloud_cover"}, 70]}
        ]
    }

    search = catalog.search(
        collection=DataCollection.SENTINEL2_L2A,
        bbox=bbox,
        time=(fecha_ini, fecha_fin),
        filter=filtro,
        filter_lang="cql2-json",
        limit=20
    )

    items = list(search)
    if not items:
        return []

    items_sorted = sorted(items, key=lambda x: x["properties"]["datetime"])
    selected = items_sorted[-3:]

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


# =====================================
# Cálculo de índices vegetativos
# =====================================
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


# =====================================
# Heatmap Base64
# =====================================
def generar_heatmap(indice, nombre):
    plt.figure(figsize=(6, 6))
    plt.imshow(indice, cmap="RdYlGn")
    plt.colorbar()
    plt.title(nombre)

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# =====================================
# Diagnóstico
# =====================================
def diagnostico_indice(indice, nombre):
    avg = float(np.nanmean(indice))
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
# Generar PDF
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

        story.append(Image(img_path, width=400, height=400))
        story.append(Spacer(1, 20))

    doc.build(story)
    return file_path


# =====================================
# ENDPOINT PRINCIPAL
# =====================================
@app.post("/analizar")
def analizar(req: Req):
    try:
        geo = json.loads(req.geojson)
        geom = shape(geo)

        imgs = buscar_imagenes(geom, req.fecha_ini, req.fecha_fin)

        if len(imgs) == 0:
            return {"status": "error", "msg": "No se encontraron imágenes."}

        bandas = imgs[-1][0].transpose((2, 0, 1))
        indices_raw = calc_indices(bandas)

        indices = {}
        for nombre, matriz in indices_raw.items():
            indices[nombre] = {
                "img_base64": generar_heatmap(matriz, nombre),
                "diagnostico": diagnostico_indice(matriz, nombre)
            }

        return {"status": "ok", "indices": indices}

    except Exception as e:
        return {"status": "error", "msg": str(e)}


# =====================================
# Endpoint PDF
# =====================================
@app.post("/pdf")
def pdf(req: Req):
    result = analizar(req)

    if result["status"] != "ok":
        return result

    file_path = crear_pdf(result["indices"])
    with open(file_path, "rb") as f:
        pdf_bytes = f.read()

    return Response(content=pdf_bytes, media_type="application/pdf")


# =====================================
# Endpoint Dashboard
# =====================================
@app.post("/dashboard")
def dashboard(req: Req):
    result = analizar(req)
    if result["status"] != "ok":
        return result

    html = generar_dashboard(
        result["indices"],
        req.geojson,
        req.fecha_ini,
        req.fecha_fin
    )

    return HTMLResponse(content=html)


    
# =====================================
# Dashboard HTML con Bootstrap 5
# =====================================
import json

def generar_dashboard(indices, geojson, fecha_ini, fecha_fin):
    """
    Construye el HTML del dashboard de forma segura sin usar f-strings
    multilínea que contengan llaves y que rompan el archivo en el editor.
    """

    # Serializamos el geojson para que sea seguro en JS
    try:
        geojson_js = json.dumps(json.loads(geojson))
    except Exception:
        # Si ya es dict, convertir directamente
        geojson_js = json.dumps(geojson)

    parts = []

    # Header (no f-string)
    parts.append("""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Dashboard de Índices</title>

  <!-- Bootstrap 5 -->
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">

  <!-- Leaflet -->
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>

  <style>
    #map {
      height: 350px;
      border-radius: 12px;
      margin-bottom: 20px;
    }
  </style>
</head>

<body class="bg-light">
  <div class="container py-4">
    <div class="d-flex justify-content-between align-items-center mb-4">
      <h1>Dashboard de Índices Vegetativos</h1>
      <button class="btn btn-danger btn-lg" onclick="descargarPDF()">📄 Descargar PDF</button>
    </div>

    <!-- MAPA -->
    <div id="map"></div>

    <div class="row g-4">
""")

    # Cards for indices (built by simple concatenation)
    for nombre, data in indices.items():
        img_b64 = data.get("img_base64", "")
        diagnostico = data.get("diagnostico", "").replace("\n", " ")
        # escape single quotes in nombre and diagnostico to avoid breaking HTML attributes
        safe_nombre = str(nombre).replace("'", "&#39;")
        safe_diag = str(diagnostico).replace("'", "&#39;")

        card_html = (
            "<div class=\"col-12 col-md-6 col-lg-4\">"
              "<div class=\"card shadow\">"
                "<img src=\"data:image/png;base64," + img_b64 + "\" "
                      "class=\"card-img-top img-fluid\" alt=\"" + safe_nombre + "\">"
                "<div class=\"card-body\">"
                  "<h5 class=\"card-title\">" + safe_nombre + "</h5>"
                  "<p class=\"card-text\">" + safe_diag + "</p>"
                "</div>"
              "</div>"
            "</div>\n"
        )
        parts.append(card_html)

    # Close the cards container and add script (insert geojson_js and dates via concatenation)
    script_head = (
        "    </div>\n"  # close row
        "  </div>\n\n"  # close container
        "  <script>\n"
        "    var map = L.map('map');\n\n"
        "    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {\n"
        "      maxZoom: 19\n"
        "    }).addTo(map);\n\n"
        "    var geo = "
    )
    parts.append(script_head)
    parts.append(geojson_js)  # already a JSON string (no extra quotes)
    script_mid = ";\n\n" \
                 "    var capa = L.geoJSON(geo, {\n" \
                 "      style: function() {\n" \
                 "        return { color: 'red', weight: 2, fillOpacity: 0.1 };\n" \
                 "      }\n" \
                 "    }).addTo(map);\n\n" \
                 "    map.fitBounds(capa.getBounds());\n\n" \
                 "    function descargarPDF() {\n" \
                 "      fetch('/pdf', {\n" \
                 "        method: 'POST',\n" \
                 "        headers: { 'Content-Type': 'application/json' },\n" \
                 "        body: JSON.stringify({\n" \
                 "          geojson: JSON.stringify(geo),\n"
    parts.append(script_mid)
    # insert fecha_ini and fecha_fin safely (escape quotes)
    safe_fecha_ini = str(fecha_ini).replace('"', '\\"')
    safe_fecha_fin = str(fecha_fin).replace('"', '\\"')
    parts.append("          \"fecha_ini\": \"" + safe_fecha_ini + "\",\n")
    parts.append("          \"fecha_fin\": \"" + safe_fecha_fin + "\"\n")
    script_tail = (
        "        })\n"
        "      })\n"
        "      .then(function(resp) { return resp.blob(); })\n"
        "      .then(function(blob) {\n"
        "        var url = URL.createObjectURL(blob);\n"
        "        var a = document.createElement('a');\n"
        "        a.href = url;\n"
        "        a.download = 'diagnostico.pdf';\n"
        "        document.body.appendChild(a);\n"
        "        a.click();\n"
        "        a.remove();\n"
        "        URL.revokeObjectURL(url);\n"
        "      });\n"
        "    }\n\n"
        "  </script>\n\n"
        "</body>\n"
        "</html>\n"
    )
    parts.append(script_tail)

    # Join and return
    return "".join(parts)


# =====================================
# Server
# =====================================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 10000)))
