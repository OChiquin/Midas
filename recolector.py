# recolector.py
# Bot recolector de datos históricos y en tiempo real
# Proyecto Midas — Parte 1

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import os
import time
import logging
from datetime import datetime, timezone

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────

# Los 6 activos del proyecto
ACTIVOS = [
    # Pares principales
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD",
    # Cruces del yen
    "EURJPY", "GBPJPY", "AUDJPY",
    # Cruces euro y libra
    "EURGBP", "EURCAD", "GBPAUD", "GBPCAD",
    # Otros
    "NZDUSD", "AUDCAD", "USDNOK"
]

# Temporalidad M15
TEMPORALIDAD = mt5.TIMEFRAME_M15

# Carpetas
CARPETA_DATOS      = "datos"
CARPETA_LOGS       = "logs"

# Archivos de salida
ARCHIVO_COMPLETO        = os.path.join(CARPETA_DATOS, "historico_completo.csv")
ARCHIVO_ENTRENAMIENTO   = os.path.join(CARPETA_DATOS, "datos_entrenamiento.csv")
ARCHIVO_BACKTEST        = os.path.join(CARPETA_DATOS, "datos_backtest.csv")

# Años de historia a descargar
ANIOS_TOTAL       = 4    # XM provee ~4 años de historico en M15
ANIOS_ENTRENAMIENTO = 3  # 3 años para entrenamiento
ANIOS_BACKTEST    = 1    # 1 año reservado para backtest

# ─────────────────────────────────────────────
# CONFIGURACIÓN DE LOGS
# ─────────────────────────────────────────────

import sys
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(CARPETA_LOGS, "recolector.log"), encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
sys.stdout.reconfigure(encoding='utf-8')
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# FUNCIONES DE CONEXIÓN
# ─────────────────────────────────────────────

def conectar_mt5() -> bool:
    """Inicializa la conexión con MetaTrader 5"""
    if not mt5.initialize():
        log.error(f"Error al conectar con MT5: {mt5.last_error()}")
        return False
    log.info(f"Conectado a MT5: {mt5.terminal_info().name}")
    return True

def desconectar_mt5():
    """Cierra la conexión con MetaTrader 5"""
    mt5.shutdown()
    log.info("Desconectado de MT5")

def verificar_simbolo(activo: str) -> str:
    """
    Verifica si el símbolo existe en MT5
    Retorna el nombre correcto del símbolo o None si no existe
    """
    # Primero prueba el símbolo tal cual
    info = mt5.symbol_info(activo)
    if info is not None:
        # Activa el símbolo si no está visible
        if not info.visible:
            mt5.symbol_select(activo, True)
        return activo

    # Si no existe, prueba con sufijo 'm'
    activo_m = activo + "m"
    info = mt5.symbol_info(activo_m)
    if info is not None:
        if not info.visible:
            mt5.symbol_select(activo_m, True)
        log.info(f"Símbolo {activo} encontrado con sufijo: {activo_m}")
        return activo_m

    log.error(f"Símbolo {activo} no encontrado en MT5")
    return None

# ─────────────────────────────────────────────
# DESCARGA DE DATOS HISTÓRICOS
# ─────────────────────────────────────────────

def descargar_historico(activo: str, simbolo_mt5: str) -> pd.DataFrame:
    """
    Descarga los últimos 20 años de datos en M15 para un activo
    Usa copy_rates_from_pos descargando en bloques desde el presente hacia atrás
    """
    log.info(f"Descargando histórico de {activo}...")

    # Activa el símbolo y espera que MT5 lo cargue
    mt5.symbol_select(simbolo_mt5, True)
    time.sleep(2)

    # Calcula cuántas velas M15 hay en 20 años
    # 20 años × 365 días × 24 horas × 4 velas por hora = ~700,800 velas
    VELAS_POR_BLOQUE = 50000
    TOTAL_VELAS      = 700800

    todos_los_bloques = []
    posicion_actual   = 0

    while posicion_actual < TOTAL_VELAS:
        # Descarga un bloque desde la posición actual
        bloque = mt5.copy_rates_from_pos(
            simbolo_mt5,
            TEMPORALIDAD,
            posicion_actual,
            VELAS_POR_BLOQUE
        )

        # Si no hay más datos, termina
        if bloque is None or len(bloque) == 0:
            log.info(f"  {activo}: no hay más datos en posición {posicion_actual:,}")
            break

        todos_los_bloques.append(pd.DataFrame(bloque))
        posicion_actual += len(bloque)

        fecha_bloque = pd.to_datetime(bloque[0]['time'], unit='s')
        log.info(f"  {activo}: {posicion_actual:,} velas — hasta {fecha_bloque.date()}")

        # Si el bloque fue menor al solicitado, llegamos al inicio del histórico
        if len(bloque) < VELAS_POR_BLOQUE:
            break

        time.sleep(0.3)

    if not todos_los_bloques:
        log.error(f"No se obtuvieron datos para {activo}: {mt5.last_error()}")
        return None

    # Combina todos los bloques
    df = pd.concat(todos_los_bloques, ignore_index=True)

    # Elimina duplicados y ordena por fecha
    df = df.drop_duplicates(subset='time').sort_values('time').reset_index(drop=True)

    # Convierte timestamp a fecha legible
    df['fecha_hora'] = pd.to_datetime(df['time'], unit='s')

    # Selecciona columnas necesarias
    df = df[['fecha_hora', 'open', 'high', 'low', 'close', 'tick_volume']]
    df.columns = ['fecha_hora', 'open', 'high', 'low', 'close', 'volumen']

    # Agrega columna del activo
    df.insert(0, 'activo', activo)

    log.info(f"  ✅ {activo}: {len(df):,} velas totales "
             f"({df['fecha_hora'].min().date()} → {df['fecha_hora'].max().date()})")

    return df

def descargar_todos_los_activos() -> pd.DataFrame:
    """
    Descarga histórico de los 6 activos y los combina en un solo DataFrame
    """
    log.info("=" * 55)
    log.info("  INICIANDO DESCARGA DE DATOS HISTÓRICOS")
    log.info("=" * 55)

    dataframes = []

    for activo in ACTIVOS:
        # Verifica que el símbolo existe en MT5
        simbolo = verificar_simbolo(activo)
        if simbolo is None:
            log.warning(f"Saltando {activo} — símbolo no disponible")
            continue

        # Descarga los datos
        df = descargar_historico(activo, simbolo)
        if df is not None:
            dataframes.append(df)

        # Pausa breve entre descargas para no saturar MT5
        time.sleep(1)

    if not dataframes:
        log.error("No se pudo descargar ningún activo")
        return None

    # Combina todos los activos en un solo DataFrame
    df_total = pd.concat(dataframes, ignore_index=True)
    df_total = df_total.sort_values(['activo', 'fecha_hora']).reset_index(drop=True)

    log.info(f"\nTotal de velas descargadas: {len(df_total):,}")
    return df_total

# ─────────────────────────────────────────────
# SEPARACIÓN DE DATOS
# ─────────────────────────────────────────────

def separar_datos(df: pd.DataFrame):
    """
    Separa los datos en dos conjuntos:
    - Entrenamiento: primeros 15 años
    - Backtest: últimos 5 años (el modelo NUNCA verá estos datos)
    """
    log.info("\nSeparando datos entrenamiento / backtest...")

    fecha_min = df['fecha_hora'].min()
    fecha_max = df['fecha_hora'].max()

    # Calcula la fecha de corte (15 años desde el inicio)
    fecha_corte = fecha_max - pd.DateOffset(years=ANIOS_BACKTEST)

    df_entrenamiento = df[df['fecha_hora'] < fecha_corte].copy()
    df_backtest      = df[df['fecha_hora'] >= fecha_corte].copy()

    log.info(f"  Rango total      : {fecha_min.date()} → {fecha_max.date()}")
    log.info(f"  Fecha de corte   : {fecha_corte.date()}")
    log.info(f"  Entrenamiento    : {len(df_entrenamiento):,} velas "
             f"({df_entrenamiento['fecha_hora'].min().date()} → "
             f"{df_entrenamiento['fecha_hora'].max().date()})")
    log.info(f"  Backtest         : {len(df_backtest):,} velas "
             f"({df_backtest['fecha_hora'].min().date()} → "
             f"{df_backtest['fecha_hora'].max().date()})")

    return df_entrenamiento, df_backtest

# ─────────────────────────────────────────────
# GUARDADO DE ARCHIVOS
# ─────────────────────────────────────────────

def guardar_datos(df_completo, df_entrenamiento, df_backtest):
    """Guarda los tres CSV en la carpeta datos/"""
    log.info("\nGuardando archivos CSV...")

    df_completo.to_csv(ARCHIVO_COMPLETO, index=False)
    log.info(f"  ✅ {ARCHIVO_COMPLETO} ({len(df_completo):,} filas)")

    df_entrenamiento.to_csv(ARCHIVO_ENTRENAMIENTO, index=False)
    log.info(f"  ✅ {ARCHIVO_ENTRENAMIENTO} ({len(df_entrenamiento):,} filas)")

    df_backtest.to_csv(ARCHIVO_BACKTEST, index=False)
    log.info(f"  ✅ {ARCHIVO_BACKTEST} ({len(df_backtest):,} filas)")

# ─────────────────────────────────────────────
# RECOLECTOR EN TIEMPO REAL
# ─────────────────────────────────────────────

def obtener_ultima_vela(activo: str, simbolo_mt5: str) -> pd.DataFrame:
    """Obtiene la última vela cerrada M15 de un activo"""
    velas = mt5.copy_rates_from_pos(simbolo_mt5, TEMPORALIDAD, 0, 2)
    if velas is None or len(velas) < 2:
        return None

    # Toma la penúltima (la última ya cerrada)
    vela = velas[-2]
    df = pd.DataFrame([vela])
    df['fecha_hora'] = pd.to_datetime(df['time'], unit='s')
    df = df[['fecha_hora', 'open', 'high', 'low', 'close', 'tick_volume']]
    df.columns = ['fecha_hora', 'open', 'high', 'low', 'close', 'volumen']
    df.insert(0, 'activo', activo)
    return df

def agregar_vela_al_csv(df_vela: pd.DataFrame):
    """Agrega una nueva vela al CSV histórico completo"""
    df_vela.to_csv(ARCHIVO_COMPLETO, mode='a', header=False, index=False)

def iniciar_recolector_tiempo_real():
    """
    Bucle infinito que cada 15 minutos verifica si hay una vela nueva
    y la agrega al CSV histórico
    """
    log.info("\n" + "=" * 55)
    log.info("  RECOLECTOR EN TIEMPO REAL INICIADO")
    log.info("  Guardando nuevas velas cada 15 minutos...")
    log.info("  Presiona Ctrl+C para detener")
    log.info("=" * 55)

    # Mapeo de activos a símbolos MT5
    simbolos = {}
    for activo in ACTIVOS:
        simbolo = verificar_simbolo(activo)
        if simbolo:
            simbolos[activo] = simbolo

    ultima_vela_guardada = {}

    while True:
        try:
            for activo, simbolo in simbolos.items():
                df_vela = obtener_ultima_vela(activo, simbolo)
                if df_vela is None:
                    continue

                fecha_vela = df_vela['fecha_hora'].iloc[0]

                # Solo guarda si es una vela nueva
                if activo not in ultima_vela_guardada or \
                   ultima_vela_guardada[activo] != fecha_vela:
                    agregar_vela_al_csv(df_vela)
                    ultima_vela_guardada[activo] = fecha_vela
                    log.info(f"  Nueva vela guardada: {activo} {fecha_vela}")

            # Espera 60 segundos antes de verificar de nuevo
            time.sleep(60)

        except KeyboardInterrupt:
            log.info("\nRecolector detenido por el usuario")
            break
        except Exception as e:
            log.error(f"Error en recolector: {e}")
            time.sleep(60)

# ─────────────────────────────────────────────
# PUNTO DE ENTRADA
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Recolector de datos Proyecto Midas")
    parser.add_argument('--modo', choices=['historico', 'tiempo-real', 'completo'],
                        default='completo',
                        help='historico=solo descarga | tiempo-real=solo live | completo=ambos')
    args = parser.parse_args()

    # Conecta con MT5
    if not conectar_mt5():
        exit(1)

    try:
        if args.modo in ['historico', 'completo']:
            # Descarga histórico
            df_completo = descargar_todos_los_activos()

            if df_completo is not None:
                # Separa entrenamiento / backtest
                df_entrenamiento, df_backtest = separar_datos(df_completo)

                # Guarda los CSV
                guardar_datos(df_completo, df_entrenamiento, df_backtest)

                log.info("\n✅ Descarga histórica completada")
                log.info(f"   Archivos guardados en: {CARPETA_DATOS}/")

        if args.modo in ['tiempo-real', 'completo']:
            # Inicia recolector en tiempo real
            iniciar_recolector_tiempo_real()

    finally:
        desconectar_mt5()