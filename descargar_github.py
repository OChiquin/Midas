# descargar_github.py
# Descarga datos históricos de GitHub ejtraderLabs (2012-2022)
# y los combina con los datos de XM (2022-2026)
# Proyecto Midas

import pandas as pd
import os
import time

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────

BASE_URL = "https://raw.githubusercontent.com/ejtraderLabs/historical-data/main/"

# Mapeo de nuestros activos a los nombres del repositorio
ACTIVOS_DISPONIBLES = {
    "EURUSD" : "EURUSD/EURUSDm15.csv",
    "GBPUSD" : "GBPUSD/GBPUSDm15.csv",
    "USDJPY" : "USDJPY/USDJPYm15.csv",
    "USDCHF" : "USDCHF/USDCHFm15.csv",
    "AUDUSD" : "AUDUSD/AUDUSDm15.csv",
    "USDCAD" : "USDCAD/USDCADm15.csv",
    "EURJPY" : "EURJPY/EURJPYm15.csv",
    "GBPJPY" : "GBPJPY/GBPJPYm15.csv",
    "AUDJPY" : "AUDJPY/AUDJPYm15.csv",
    "EURGBP" : "EURGBP/EURGBPm15.csv",
}

CARPETA_DATOS = "datos"

# ─────────────────────────────────────────────
# DESCARGA
# ─────────────────────────────────────────────

def descargar_activo(activo, ruta_archivo):
    """Descarga los datos históricos de un activo desde GitHub"""
    url = BASE_URL + ruta_archivo
    print(f"  Descargando {activo}...")

    try:
        df = pd.read_csv(url)

        # Renombra columnas al formato del proyecto
        df.columns = ['fecha_hora', 'open', 'high', 'low', 'close', 'volumen']
        df['fecha_hora'] = pd.to_datetime(df['fecha_hora'])
        df.insert(0, 'activo', activo)

        print(f"  OK {activo}: {len(df):,} velas "
              f"({df['fecha_hora'].min().date()} → {df['fecha_hora'].max().date()})")
        return df

    except Exception as e:
        print(f"  ERROR {activo}: {e}")
        return None

def combinar_con_xm(df_github, df_xm, activo):
    """
    Combina los datos de GitHub (2012-2022) con los de XM (2022-2026)
    Elimina duplicados y ordena cronológicamente
    """
    df_activo_xm = df_xm[df_xm['activo'] == activo].copy()

    if df_github is None:
        return df_activo_xm

    # Combina ambos DataFrames
    df_combinado = pd.concat([df_github, df_activo_xm], ignore_index=True)

    # Elimina duplicados por fecha
    df_combinado = df_combinado.drop_duplicates(subset='fecha_hora')

    # Ordena por fecha
    df_combinado = df_combinado.sort_values('fecha_hora').reset_index(drop=True)

    return df_combinado

# ─────────────────────────────────────────────
# PROGRAMA PRINCIPAL
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("   PROYECTO MIDAS — DESCARGA DATOS GITHUB")
    print("=" * 60)

    # Carga los datos existentes de XM
    print("\nCargando datos existentes de XM...")
    ruta_historico = os.path.join(CARPETA_DATOS, "historico_completo.csv")
    df_xm = pd.read_csv(ruta_historico, parse_dates=['fecha_hora'])
    print(f"  {len(df_xm):,} velas de XM cargadas")

    # Descarga y combina cada activo
    todos_combinados = []

    for activo, ruta in ACTIVOS_DISPONIBLES.items():
        df_github = descargar_activo(activo, ruta)
        df_combinado = combinar_con_xm(df_github, df_xm, activo)
        todos_combinados.append(df_combinado)
        print(f"  Total {activo}: {len(df_combinado):,} velas combinadas")
        time.sleep(1)  # Pausa para no saturar GitHub

    # Activos que no están en GitHub — solo datos de XM
    activos_sin_github = [a for a in df_xm['activo'].unique()
                          if a not in ACTIVOS_DISPONIBLES]

    for activo in activos_sin_github:
        df_activo = df_xm[df_xm['activo'] == activo].copy()
        todos_combinados.append(df_activo)
        print(f"  {activo}: {len(df_activo):,} velas (solo XM)")

    # Combina todo
    df_total = pd.concat(todos_combinados, ignore_index=True)
    df_total = df_total.sort_values(['activo', 'fecha_hora']).reset_index(drop=True)

    print(f"\nTotal combinado: {len(df_total):,} velas")
    print(f"Activos: {df_total['activo'].nunique()}")
    print(f"Rango: {df_total['fecha_hora'].min().date()} → "
          f"{df_total['fecha_hora'].max().date()}")

    # Separa entrenamiento y backtest
    fecha_corte = df_total['fecha_hora'].max() - pd.DateOffset(years=1)
    df_entrenamiento = df_total[df_total['fecha_hora'] < fecha_corte].copy()
    df_backtest      = df_total[df_total['fecha_hora'] >= fecha_corte].copy()

    print(f"\nEntrenamiento: {len(df_entrenamiento):,} velas "
          f"({df_entrenamiento['fecha_hora'].min().date()} → "
          f"{df_entrenamiento['fecha_hora'].max().date()})")
    print(f"Backtest     : {len(df_backtest):,} velas "
          f"({df_backtest['fecha_hora'].min().date()} → "
          f"{df_backtest['fecha_hora'].max().date()})")

    # Guarda los archivos
    print("\nGuardando archivos...")
    df_total.to_csv(os.path.join(CARPETA_DATOS, 'historico_completo.csv'), index=False)
    df_entrenamiento.to_csv(os.path.join(CARPETA_DATOS, 'datos_entrenamiento.csv'), index=False)
    df_backtest.to_csv(os.path.join(CARPETA_DATOS, 'datos_backtest.csv'), index=False)

    print("OK datos guardados!")
    print("=" * 60)