import time
import asyncio
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ============================================================================
# 1. MODELOS DE DATOS (CONTRATO JSON CON PHP)
# ============================================================================
class SolicitudAgro(BaseModel):
    finca_id: str = Field(..., example="FINCA_105")
    finca_nombre: str = Field(..., example="La Guaira")
    latitud: float = Field(..., example=4.5981)
    longitud: float = Field(..., example=-73.5594)
    instruccion_usuario: str = Field(
        ..., 
        example="Generar diagnóstico satelital, revisar clima de esta semana y enviar resumen por WhatsApp."
    )
    canal_comunicacion: Optional[str] = Field("whatsapp", example="whatsapp") # whatsapp, email, pdf

class RespuestaAgro(BaseModel):
    status: str
    finca_id: str
    diagnostico_lider: str
    resultado_operativo: Dict[str, Any]
    resultado_investigador: Dict[str, Any]
    resultado_comunicador: Dict[str, Any]
    tiempo_ejecucion_seg: float


# ============================================================================
# 2. PELOTÓN DE AGENTES (LÓGICA INTERNA DEL MICROSERVICIO)
# ============================================================================

# ⚙️ AGENTE OPERATIVO: Procesamiento Satelital (Planet/Sentinel), Raster & DB
class AgenteOperativo:
    async def ejecutar_tarea(self, finca_id: str, finca_nombre: str) -> Dict[str, Any]:
        """Procesa imágenes satelitales, calcula NDVI/NDRE y consulta DB."""
        # Simulación de tarea de cómputo/descarga satelital
        await asyncio.sleep(0.6) 
        
        return {
            "agente": "AgroIA Operator",
            "satelite": "Planet SuperDove 3m / Sentinel-2",
            "ndvi_promedio": 0.74,
            "ndre_promedio": 0.52,
            "zonas_criticas_detectadas": 2,
            "archivo_pdf_generado": f"/tmp/Diagnostico_{finca_id}.pdf"
        }


# 🔍 AGENTE INVESTIGADOR: Clima, Precios & Tendencias Agronómicas
class AgenteInvestigador:
    async def ejecutar_tarea(self, lat: float, lon: float) -> Dict[str, Any]:
        """Consulta APIs de clima (Open-Meteo) y datos de mercado."""
        # Simulación de consulta a API agroclimática
        await asyncio.sleep(0.4)
        
        return {
            "agente": "AgroIA Researcher",
            "pronostico_lluvias_7d": "45mm acumulados (Precipitación moderada)",
            "alerta_agroclimatica": "Riesgo medio de estrés hídrico en 48h",
            "precio_urea_colombia": "$2.150 COP/kg (Estable)"
        }


# 📢 AGENTE COMUNICADOR: Formateo de Reportes y Despacho
class AgenteComunicador:
    async def despachar_mensaje(
        self, 
        finca_nombre: str, 
        datos_op: Dict[str, Any], 
        datos_inv: Dict[str, Any], 
        canal: str
    ) -> Dict[str, Any]:
        """Formatea el reporte técnico y prepara el despacho omnicanal."""
        await asyncio.sleep(0.2)
        
        mensaje_formateado = (
            f"🌾 *AgroIA Reporte Ejecutivo: {finca_nombre}*\n\n"
            f"✅ *Vigor Vegetativo (NDVI):* {datos_op['ndvi_promedio']} (Saludable)\n"
            f"⚠️ *Atención:* {datos_op['zonas_criticas_detectadas']} sectores requieren revisión de nitrógeno.\n"
            f"🌧️ *Pronóstico Clima:* {datos_inv['pronostico_lluvias_7d']}.\n\n"
            f"📄 El reporte en PDF ha sido adjuntado con éxito."
        )
        
        return {
            "agente": "AgroIA Communicator",
            "canal_utilizado": canal,
            "despachado": True,
            "mensaje_saliente": mensaje_formateado
        }


# 🎖️ AGENTE LÍDER: Orquestador y Comandante General
class AgenteLider:
    def __init__(self):
        self.operativo = AgenteOperativo()
        self.investigador = AgenteInvestigador()
        self.comunicador = AgenteComunicador()

    async def orquestar(self, solicitud: SolicitudAgro) -> Dict[str, Any]:
        """
        Coordina la ejecución paralela asíncrona de los agentes operativos
        e investigadores, recopila datos y le da orden al comunicador.
        """
        # 1. Disparo simultáneo de tareas pesadas (Operativo + Investigador)
        res_operativo, res_investigador = await asyncio.gather(
            self.operativo.ejecutar_tarea(solicitud.finca_id, solicitud.finca_nombre),
            self.investigador.ejecutar_tarea(solicitud.latitud, solicitud.longitud)
        )

        # 2. Entrega de resultados al Agente Comunicador para estructurar la salida
        res_comunicador = await self.comunicador.despachar_mensaje(
            solicitud.finca_nombre,
            res_operativo,
            res_investigador,
            solicitud.canal_comunicacion
        )

        # 3. Síntesis ejecutiva final dictada por el Líder
        sintesis = (
            f"Diagnóstico procesado exitosamente para {solicitud.finca_nombre}. "
            f"Lote con vigor NDVI promedio de {res_operativo['ndvi_promedio']}. "
            f"Recomendación: Aplicar fertilización preventiva antes de las lluvias proyectadas "
            f"({res_investigador['pronostico_lluvias_7d']})."
        )

        return {
            "diagnostico_lider": sintesis,
            "resultado_operativo": res_operativo,
            "resultado_investigador": res_investigador,
            "resultado_comunicador": res_comunicador
        }


# ============================================================================
# 3. APLICACIÓN FASTAPI Y ENDPOINT ÚNICO
# ============================================================================
app = FastAPI(
    title="AgroIA Multi-Agent Engine",
    description="Microservicio único unificado para la orquestación del Pelotón de Agentes",
    version="2.0.0"
)

# Configuración CORS para recibir peticiones desde el servidor PHP / Web
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instancia global del Comandante Líder
comandante_lider = AgenteLider()


@app.post("/v1/agentes/procesar-finca", response_model=RespuestaAgro)
async def procesar_finca(solicitud: SolicitudAgro):
    """
    Punto de entrada único para PHP. Recibe los datos de la finca y activa
    al Pelotón de Agentes en paralelo.
    """
    tiempo_inicio = time.time()
    
    try:
        # Ejecutar orquestación de la tropa
        resultado = await comandante_lider.orquestar(solicitud)
        tiempo_ejecucion = round(time.time() - tiempo_inicio, 3)
        
        return RespuestaAgro(
            status="success",
            finca_id=solicitud.finca_id,
            diagnostico_lider=resultado["diagnostico_lider"],
            resultado_operativo=resultado["resultado_operativo"],
            resultado_investigador=resultado["resultado_investigador"],
            resultado_comunicador=resultado["resultado_comunicador"],
            tiempo_ejecucion_seg=tiempo_ejecucion
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error interno en la orquestación del Pelotón: {str(e)}"
        )


# Servidor listo para ejecución directa con: python main.py
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
