# api.py
# API local FastAPI que carga el modelo ensemble y responde predicciones
# Proyecto Midas — Parte 4

import sys
import os
import torch
import numpy as np
import pandas as pd
import ta
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from contextlib import asynccontextmanager

# Agrega el directorio actual al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from modelo_ensemble import MidasEnsemble, CONFIG_ENSEMBLE
from preparar_datos import normalizar_ventana, agregar_indicadores, ACTIVO_ID

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────

RUTA_MODELO = os.path.join('modelos', 'midas_mejor.pt')
DISPOSITIVO = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Variables globales del modelo
modelo     = None
checkpoint = None

# ─────────────────────────────────────────────
# CARGA DEL MODELO AL INICIAR
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo cuando arranca la API"""
    global modelo, checkpoint

    print(f"\nCargando modelo desde {RUTA_MODELO}...")
    print(f"Dispositivo: {DISPOSITIVO}")

    checkpoint = torch.load(
        RUTA_MODELO, map_location=DISPOSITIVO, weights_only=False
    )
    modelo = MidasEnsemble(CONFIG_ENSEMBLE).to(DISPOSITIVO)
    modelo.load_state_dict(checkpoint['model_state_dict'])
    modelo.eval()

    print(f"Modelo cargado! Epoca {checkpoint['epoca']} | "
          f"Val acc: {checkpoint['val_accuracy']:.1f}%")
    print(f"API lista en http://localhost:8000\n")

    yield  # La API corre aquí

    print("Cerrando API...")

# ─────────────────────────────────────────────
# APLICACIÓN FASTAPI
# ─────────────────────────────────────────────

app = FastAPI(
    title       = "Proyecto Midas API",
    description = "API de predicciones para el bot de trading",
    version     = "1.0.0",
    lifespan    = lifespan
)

# ─────────────────────────────────────────────
# ESQUEMAS DE DATOS
# ─────────────────────────────────────────────

class Vela(BaseModel):
    """Una vela OHLCV"""
    fecha_hora : str
    open       : float
    high       : float
    low        : float
    close      : float
    volumen    : float

class SolicitudPrediccion(BaseModel):
    """Datos para hacer una predicción"""
    activo : str
    velas  : List[Vela]

class RespuestaPrediccion(BaseModel):
    """Respuesta con la señal de trading"""
    accion           : str
    stop_loss_pips   : float
    take_profit_pips : float
    confianza        : float
    umbral_dinamico  : float
    activo           : str

# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/")
def raiz():
    """Verifica que la API está activa"""
    return {
        "estado"  : "activo",
        "modelo"  : "MidasEnsemble CNN+Transformer+LSTM",
        "epoca"   : checkpoint['epoca'] if checkpoint else 0,
        "val_acc" : round(checkpoint['val_accuracy'], 2) if checkpoint else 0
    }

@app.post("/predecir", response_model=RespuestaPrediccion)
def predecir(solicitud: SolicitudPrediccion):
    """
    Recibe las últimas velas y devuelve la señal de trading
    """
    # Respuesta por defecto
    hold = RespuestaPrediccion(
        accion='HOLD', stop_loss_pips=0.0,
        take_profit_pips=0.0, confianza=0.0,
        umbral_dinamico=0.5, activo=solicitud.activo
    )

    try:
        # Convierte las velas a DataFrame
        datos = [{
            'activo'    : solicitud.activo,
            'fecha_hora': v.fecha_hora,
            'open'      : v.open,
            'high'      : v.high,
            'low'       : v.low,
            'close'     : v.close,
            'volumen'   : v.volumen
        } for v in solicitud.velas]

        df = pd.DataFrame(datos)
        df['fecha_hora'] = pd.to_datetime(df['fecha_hora'])

        # Necesitamos suficientes velas para los indicadores
        if len(df) < 40:
            return hold

        # Agrega indicadores técnicos
        df = agregar_indicadores(df)

        # Toma las últimas 20 velas
        ventana = df.tail(20)
        if len(ventana) < 20:
            return hold

        # Normaliza
        X = normalizar_ventana(ventana)

        # Convierte a tensor
        X_tensor  = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(DISPOSITIVO)
        id_tensor = torch.tensor(
            [ACTIVO_ID.get(solicitud.activo, 0)], dtype=torch.long
        ).to(DISPOSITIVO)

        # Inferencia
        with torch.no_grad():
            logits, sl, tp, confianza, umbral = modelo(X_tensor, id_tensor)

        # Decodifica
        acciones  = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
        accion    = acciones[logits.argmax(dim=1).item()]

        return RespuestaPrediccion(
            accion           = accion,
            stop_loss_pips   = round(sl.item(), 1),
            take_profit_pips = round(tp.item(), 1),
            confianza        = round(confianza.item(), 4),
            umbral_dinamico  = round(umbral.item(), 4),
            activo           = solicitud.activo
        )

    except Exception as e:
        print(f"Error en prediccion: {e}")
        return hold

# ─────────────────────────────────────────────
# PUNTO DE ENTRADA
# ─────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, log_level="info")