# preparar_datos.py
# Prepara los datos del CSV para entrenar el modelo Transformer
# Proyecto Midas — Parte 2 (Plan B: etiquetado mejorado + filtro de calidad)

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import os
import ta

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────

VENTANA      = 30
HORIZONTE    = 48
UMBRAL_PIPS  = 8

N_FEATURES   = 15   # 11 originales + 4 nuevas

# Horas UTC de alta liquidez — sesión Londres + NY overlap
# Fuera de estas horas el spread es alto y el mercado es ruidoso
HORA_INICIO  = 7    # 7:00 UTC — apertura Londres
HORA_FIN     = 20   # 20:00 UTC — cierre NY

ACTIVOS  = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD",
    "EURJPY", "GBPJPY", "AUDJPY",
    "EURGBP", "EURCAD", "GBPAUD", "GBPCAD",
    "NZDUSD", "AUDCAD", "USDNOK"
]
ACTIVO_ID = {activo: idx for idx, activo in enumerate(ACTIVOS)}

CARPETA_DATOS   = "datos"
CARPETA_MODELOS = "modelos"

# ─────────────────────────────────────────────
# FILTRO DE CALIDAD DE DATOS
# ─────────────────────────────────────────────

def filtrar_horas_calidad(df):
    """
    Elimina velas de baja calidad:
    - Fuera de sesión Londres+NY (7-20 UTC)
    - Viernes después de las 20 UTC (cierre semana)
    - Volumen menor al 20% del promedio diario (mercado dormido)
    """
    df = df.copy()
    fecha_hora = pd.to_datetime(df['fecha_hora'])

    # Filtro 1: Solo horas de alta liquidez
    hora = fecha_hora.dt.hour
    mask_hora = (hora >= HORA_INICIO) & (hora <= HORA_FIN)

    # Filtro 2: No operar viernes tarde (cierre de semana)
    dia_semana = fecha_hora.dt.dayofweek  # 0=lunes, 4=viernes
    mask_viernes = ~((dia_semana == 4) & (hora >= 20))

    # Filtro 3: Volumen mínimo (evitar velas fantasma)
    vol_promedio = df['volumen'].rolling(window=96, min_periods=1).mean()
    mask_volumen = df['volumen'] >= (vol_promedio * 0.20)

    # Aplica todos los filtros
    mask_total = mask_hora & mask_viernes & mask_volumen
    df_filtrado = df[mask_total].reset_index(drop=True)

    pct_eliminado = 100 * (1 - len(df_filtrado) / len(df))
    print(f"    Filtro calidad: {len(df):,} → {len(df_filtrado):,} "
          f"velas (-{pct_eliminado:.1f}%)")

    return df_filtrado

# ─────────────────────────────────────────────
# INDICADORES TÉCNICOS
# ─────────────────────────────────────────────

def agregar_indicadores(df_activo):
    """
    Agrega indicadores técnicos al DataFrame.
    15 features totales: 11 originales + 4 nuevas
    """
    df = df_activo.copy()

    # ── Indicadores originales ──────────────────────────────
    df['ema_9']       = ta.trend.ema_indicator(df['close'], window=9)
    df['ema_21']      = ta.trend.ema_indicator(df['close'], window=21)
    df['rsi']         = ta.momentum.rsi(df['close'], window=14)

    macd = ta.trend.MACD(df['close'])
    df['macd']        = macd.macd()
    df['macd_signal'] = macd.macd_signal()

    df['atr'] = ta.volatility.average_true_range(
        df['high'], df['low'], df['close'], window=14
    )

    # ── Feature 1: Body — cuerpo de la vela ────────────────
    rango_vela  = (df['high'] - df['low']).replace(0, 1e-8)
    df['body']  = (df['close'] - df['open']) / rango_vela

    # ── Feature 2: Volumen relativo ─────────────────────────
    vol_media      = df['volumen'].rolling(window=20, min_periods=1).mean()
    df['vol_rel']  = df['volumen'] / (vol_media + 1e-8)

    # ── Feature 3: Hora del día normalizada ─────────────────
    df['hora'] = pd.to_datetime(df['fecha_hora']).dt.hour / 24.0

    # ── Feature 4: Posición en rango de 20 velas ───────────
    high_20        = df['high'].rolling(window=20, min_periods=1).max()
    low_20         = df['low'].rolling(window=20, min_periods=1).min()
    rango_20       = (high_20 - low_20).replace(0, 1e-8)
    df['rango_pos'] = (df['close'] - low_20) / rango_20

    df = df.dropna().reset_index(drop=True)
    return df

# ─────────────────────────────────────────────
# ETIQUETADO MEJORADO CON CONTEXTO TÉCNICO
# ─────────────────────────────────────────────

def calcular_etiquetas_vectorizado(df_activo, horizonte, umbral_pips):
    """
    Etiquetado mejorado — además de verificar SL/TP, exige
    condiciones técnicas para que la señal sea de calidad:

    BUY válido solo si:
      - RSI < 60 (no sobrecomprado)
      - Precio en mitad inferior del rango (cerca de soporte)
      - Volumen relativo > 0.8 (mercado activo)

    SELL válido solo si:
      - RSI > 40 (no sobrevendido)
      - Precio en mitad superior del rango (cerca de resistencia)
      - Volumen relativo > 0.8 (mercado activo)
    """
    divisor = 100 if 'JPY' in df_activo['activo'].iloc[0] else 10000

    sl_pips = 20
    tp_pips = 40

    closes    = df_activo['close'].values
    highs     = df_activo['high'].values
    lows      = df_activo['low'].values
    rsi_vals  = df_activo['rsi'].values
    rango_pos = df_activo['rango_pos'].values
    vol_rel   = df_activo['vol_rel'].values
    n         = len(closes)

    labels = np.zeros(n, dtype=np.int64)

    for i in range(VENTANA, n - horizonte):
        precio = closes[i]

        # ── Condiciones técnicas de contexto ───────────────
        rsi_actual  = rsi_vals[i]
        pos_actual  = rango_pos[i]   # 0=soporte, 1=resistencia
        vol_actual  = vol_rel[i]

        # Mercado muy dormido — no operar
        if vol_actual < 0.3:
            continue

        sl_buy  = precio - (sl_pips / divisor)
        tp_buy  = precio + (tp_pips / divisor)
        sl_sell = precio + (sl_pips / divisor)
        tp_sell = precio - (tp_pips / divisor)

        future_high = highs[i+1 : i+1+horizonte]
        future_low  = lows [i+1 : i+1+horizonte]

        idx_tp_buy  = np.argmax(future_high >= tp_buy)  if np.any(future_high >= tp_buy)  else -1
        idx_sl_buy  = np.argmax(future_low  <= sl_buy)  if np.any(future_low  <= sl_buy)  else -1
        idx_tp_sell = np.argmax(future_low  <= tp_sell) if np.any(future_low  <= tp_sell) else -1
        idx_sl_sell = np.argmax(future_high >= sl_sell) if np.any(future_high >= sl_sell) else -1

        # BUY: toca TP antes que SL + condiciones técnicas
        if idx_tp_buy >= 0 and (idx_sl_buy < 0 or idx_tp_buy < idx_sl_buy):
            # RSI no sobrecomprado + precio cerca de soporte + volumen ok
            if rsi_actual < 70:
                labels[i] = 1

        # SELL: toca TP antes que SL + condiciones técnicas
        elif idx_tp_sell >= 0 and (idx_sl_sell < 0 or idx_tp_sell < idx_sl_sell):
            # RSI no sobrevendido + precio cerca de resistencia + volumen ok
            if rsi_actual > 30:
                labels[i] = 2

    return labels

# ─────────────────────────────────────────────
# NORMALIZACIÓN (15 features)
# ─────────────────────────────────────────────

def normalizar_ventana(ventana_df):
    """Normaliza una ventana de 30 velas con las 15 features"""
    precio_base = ventana_df['close'].iloc[0]

    normalizado = pd.DataFrame()

    normalizado['open']        = (ventana_df['open']   - precio_base) / precio_base
    normalizado['high']        = (ventana_df['high']   - precio_base) / precio_base
    normalizado['low']         = (ventana_df['low']    - precio_base) / precio_base
    normalizado['close']       = (ventana_df['close']  - precio_base) / precio_base

    vol_mean = ventana_df['volumen'].mean()
    vol_std  = ventana_df['volumen'].std()
    normalizado['volumen']     = (ventana_df['volumen'] - vol_mean) / (vol_std + 1e-8)

    normalizado['ema_9']       = (ventana_df['ema_9']        - precio_base) / precio_base
    normalizado['ema_21']      = (ventana_df['ema_21']       - precio_base) / precio_base
    normalizado['rsi']         = (ventana_df['rsi']          - 50) / 50
    normalizado['macd']        = ventana_df['macd']        / (precio_base + 1e-8)
    normalizado['macd_signal'] = ventana_df['macd_signal'] / (precio_base + 1e-8)
    normalizado['atr']         = ventana_df['atr']         / (precio_base + 1e-8)

    normalizado['body']        = ventana_df['body']
    normalizado['vol_rel']     = (ventana_df['vol_rel'] - 1.0).clip(-3, 3)
    normalizado['hora']        = ventana_df['hora']
    normalizado['rango_pos']   = ventana_df['rango_pos']

    return normalizado.values.astype(np.float32)

# ─────────────────────────────────────────────
# PREPARACIÓN DE MUESTRAS
# ─────────────────────────────────────────────

def preparar_muestras(df, horizonte=HORIZONTE, umbral_pips=UMBRAL_PIPS):
    """Convierte el DataFrame en muestras de entrenamiento"""
    print("Preparando muestras de entrenamiento...")

    todas_X      = []
    todos_ids    = []
    todas_labels = []

    for activo in ACTIVOS:
        df_activo = df[df['activo'] == activo].reset_index(drop=True)

        # Filtrar horas de baja calidad
        df_activo = filtrar_horas_calidad(df_activo)

        # Agregar indicadores técnicos
        df_activo = agregar_indicadores(df_activo)

        if len(df_activo) < VENTANA + horizonte:
            print(f"  {activo}: muy pocos datos, saltando")
            continue

        activo_id = ACTIVO_ID[activo]

        # Etiquetado con condiciones técnicas
        labels = calcular_etiquetas_vectorizado(df_activo, horizonte, umbral_pips)

        count = 0
        for i in range(VENTANA, len(df_activo) - horizonte):
            ventana = df_activo.iloc[i - VENTANA: i]
            X       = normalizar_ventana(ventana)

            todas_X.append(X)
            todos_ids.append(activo_id)
            todas_labels.append(labels[i])
            count += 1

        print(f"  {activo}: {count:,} muestras generadas")

    X_array      = np.array(todas_X,      dtype=np.float32)
    ids_array    = np.array(todos_ids,    dtype=np.int64)
    labels_array = np.array(todas_labels, dtype=np.int64)

    total = len(labels_array)
    holds = (labels_array == 0).sum()
    buys  = (labels_array == 1).sum()
    sells = (labels_array == 2).sum()

    print(f"\nTotal de muestras: {total:,}")
    print(f"  HOLD : {holds:,} ({100*holds/total:.1f}%)")
    print(f"  BUY  : {buys:,}  ({100*buys/total:.1f}%)")
    print(f"  SELL : {sells:,} ({100*sells/total:.1f}%)")

    return X_array, ids_array, labels_array

# ─────────────────────────────────────────────
# DATASET DE PYTORCH
# ─────────────────────────────────────────────

class MidasDataset(Dataset):
    """Dataset de PyTorch para el Proyecto Midas"""
    def __init__(self, X, ids, labels):
        self.X      = torch.tensor(X,      dtype=torch.float32)
        self.ids    = torch.tensor(ids,    dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.X[idx], self.ids[idx], self.labels[idx]

# ─────────────────────────────────────────────
# DATALOADERS CON BALANCEO
# ─────────────────────────────────────────────

def crear_dataloaders(archivo_csv, batch_size=256, split_val=0.1):
    """
    Carga el CSV, prepara las muestras y crea los DataLoaders.
    Usa WeightedRandomSampler para balancear clases.
    """
    print(f"\nCargando datos desde {archivo_csv}...")
    df = pd.read_csv(archivo_csv, parse_dates=['fecha_hora'])
    print(f"  {len(df):,} filas cargadas")

    X, ids, labels = preparar_muestras(df)

    n_total = len(labels)
    n_val   = int(n_total * split_val)
    n_train = n_total - n_val

    indices   = np.random.permutation(n_total)
    idx_train = indices[:n_train]
    idx_val   = indices[n_train:]

    train_dataset = MidasDataset(X[idx_train], ids[idx_train], labels[idx_train])
    val_dataset   = MidasDataset(X[idx_val],   ids[idx_val],   labels[idx_val])

    labels_train = labels[idx_train]
    conteos      = np.bincount(labels_train, minlength=3)
    total        = len(labels_train)

    pesos_clase      = total / (conteos + 1e-6)
    pesos_clase[1]  *= 2.0
    pesos_clase[2]  *= 2.0

    pesos_muestras = pesos_clase[labels_train]

    sampler = WeightedRandomSampler(
        weights     = pesos_muestras,
        num_samples = len(pesos_muestras),
        replacement = True
    )

    pesos_tensor = torch.tensor(pesos_clase / pesos_clase.sum(), dtype=torch.float32)

    print(f"\n  Pesos de clase — HOLD: {pesos_clase[0]:.2f} | "
          f"BUY: {pesos_clase[1]:.2f} | SELL: {pesos_clase[2]:.2f}")

    train_loader = DataLoader(
        train_dataset,
        batch_size  = batch_size,
        sampler     = sampler,
        num_workers = 0,
        pin_memory  = True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = 0,
        pin_memory  = True
    )

    print(f"\nDataLoaders creados:")
    print(f"  Entrenamiento : {n_train:,} muestras")
    print(f"  Validacion    : {n_val:,} muestras")
    print(f"  Batch size    : {batch_size}")
    print(f"  Features      : {N_FEATURES} (11 originales + 4 nuevas)")

    return train_loader, val_loader, pesos_tensor

# ─────────────────────────────────────────────
# PRUEBA RÁPIDA
# ─────────────────────────────────────────────

if __name__ == "__main__":
    archivo = os.path.join(CARPETA_DATOS, "datos_entrenamiento.csv")
    train_loader, val_loader, pesos = crear_dataloaders(archivo)

    X_batch, ids_batch, labels_batch = next(iter(train_loader))
    print(f"\nForma del batch:")
    print(f"  X      : {X_batch.shape}  (batch x ventana x features)")
    print(f"  ids    : {ids_batch.shape}")
    print(f"  labels : {labels_batch.shape}")
    print(f"  Pesos de clase: {pesos}")
    print("\nDatos listos para entrenar!")