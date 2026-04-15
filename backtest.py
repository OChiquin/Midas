# backtest.py
# Motor de backtest MEJORADO del Proyecto Midas
# Procesa los 16 activos SIMULTÁNEAMENTE en cada timestamp
# Proyecto Midas — Parte 5 Mejorado

import sys
import os
import torch
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from modelo_ensemble import MidasEnsemble, CONFIG_ENSEMBLE
from preparar_datos import normalizar_ventana, agregar_indicadores, ACTIVO_ID, VENTANA, ACTIVOS
from gestor_riesgo import GestorRiesgo, Modo

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────

RUTA_MODELO   = os.path.join('modelos', 'midas_mejor.pt')
RUTA_BACKTEST = os.path.join('datos', 'datos_backtest.csv')
DISPOSITIVO   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CONFIG_GESTOR = {
    'balance_inicial'    : 1000.0,
    'meta_pct'           : 2.0,
    'umbral_cauteloso'   : 2.0,
    'techo_pct'          : 5.0,
    'limite_perdida_pct' : 1.0,
    'confianza_cautelosa': 0.85,
}

UMBRAL_MINIMO = 0.90  # Umbral mínimo de confianza
MAX_POSICIONES = 1   # Máximo de posiciones abiertas simultáneas

# ─────────────────────────────────────────────
# CARGA DEL MODELO
# ─────────────────────────────────────────────

def cargar_modelo():
    print(f"Cargando modelo desde {RUTA_MODELO}...")
    checkpoint = torch.load(RUTA_MODELO, map_location=DISPOSITIVO, weights_only=False)
    modelo = MidasEnsemble(CONFIG_ENSEMBLE).to(DISPOSITIVO)
    modelo.load_state_dict(checkpoint['model_state_dict'])
    modelo.eval()
    print(f"Modelo cargado — Epoca {checkpoint['epoca']} | "
          f"Val acc: {checkpoint['val_accuracy']:.1f}%")
    return modelo

# ─────────────────────────────────────────────
# SIMULACIÓN DE TRADE
# ─────────────────────────────────────────────

def simular_trade(accion, precio_entrada, sl_pips, tp_pips,
                  activo, df_futuro, lote):
    divisor = 100 if 'JPY' in activo else 10000
    pip_val = 1 / divisor
    valor_pip_dinero = 10.0 * lote

    if accion == 'BUY':
        sl_precio = precio_entrada - (sl_pips * pip_val)
        tp_precio = precio_entrada + (tp_pips * pip_val)
    else:
        sl_precio = precio_entrada + (sl_pips * pip_val)
        tp_precio = precio_entrada - (tp_pips * pip_val)

    for i, (_, vela) in enumerate(df_futuro.iterrows()):
        if accion == 'BUY':
            if vela['low'] <= sl_precio:
                return -sl_pips * valor_pip_dinero, -sl_pips, i + 1
            if vela['high'] >= tp_precio:
                return tp_pips * valor_pip_dinero, tp_pips, i + 1
        else:
            if vela['high'] >= sl_precio:
                return -sl_pips * valor_pip_dinero, -sl_pips, i + 1
            if vela['low'] <= tp_precio:
                return tp_pips * valor_pip_dinero, tp_pips, i + 1

    ultima_vela = df_futuro.iloc[-1]
    if accion == 'BUY':
        pips = (ultima_vela['close'] - precio_entrada) * divisor
    else:
        pips = (precio_entrada - ultima_vela['close']) * divisor

    return pips * valor_pip_dinero, pips, len(df_futuro)

# ─────────────────────────────────────────────
# PREPARAR DATOS POR ACTIVO
# ─────────────────────────────────────────────

def preparar_datos_activos(df_total):
    """
    Prepara un diccionario con los datos de cada activo
    ya con indicadores técnicos calculados
    """
    print("\nPreparando datos por activo...")
    datos_activos = {}

    for activo in ACTIVOS:
        df_activo = df_total[df_total['activo'] == activo].reset_index(drop=True)
        if len(df_activo) < VENTANA + 50:
            print(f"  {activo}: muy pocos datos, saltando")
            continue
        df_activo = agregar_indicadores(df_activo)
        datos_activos[activo] = df_activo
        print(f"  {activo}: {len(df_activo):,} velas listas")

    return datos_activos

# ─────────────────────────────────────────────
# OBTENER SEÑALES SIMULTÁNEAS
# ─────────────────────────────────────────────

def obtener_senales_simultaneas(modelo, datos_activos, indices_actuales):
    """
    Obtiene señales de TODOS los activos en el mismo momento
    Retorna lista de señales ordenadas por confianza (mayor primero)
    """
    senales = []

    for activo, idx in indices_actuales.items():
        df_activo = datos_activos[activo]

        if idx < VENTANA or idx >= len(df_activo) - 50:
            continue

        # Verifica filtro de sesión (Londres 7-17 UTC)
        hora_utc = df_activo.iloc[idx]['fecha_hora'].hour
        if hora_utc < 7 or hora_utc > 17:
            continue
        
        # Filtro de volatilidad ATR
        # No operar si el mercado está muy caótico o muy dormido
        #atr_actual = df_activo.iloc[idx]['atr']
        #atr_promedio = df_activo['atr'].iloc[max(0, idx-100):idx].mean()
        #if atr_actual > atr_promedio * 2.0:  # Mercado demasiado volátil
        #    continue
        #if atr_actual < atr_promedio * 0.3:  # Mercado demasiado dormido
           # continue

        ventana = df_activo.iloc[idx - VENTANA: idx]
        X = normalizar_ventana(ventana)

        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(DISPOSITIVO)
        id_tensor = torch.tensor([ACTIVO_ID[activo]], dtype=torch.long).to(DISPOSITIVO)

        with torch.no_grad():
            logits, sl, tp, confianza, umbral = modelo(X_tensor, id_tensor)

        accion_idx = logits.argmax(dim=1).item()
        acciones = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
        accion = acciones[accion_idx]
        conf = confianza.item()
        umbral_val = max(umbral.item(), UMBRAL_MINIMO)

        if accion != 'HOLD' and conf >= umbral_val:
            senales.append({
                'activo'   : activo,
                'accion'   : accion,
                'confianza': conf,
                'umbral'   : umbral_val,
                'sl_pips'  : 10.0,    # SL más ajustado
                'tp_pips'  : 45.0,   # TP fijo — ratio 1:4
                'idx'      : idx
            })

    # Ordena por confianza — las mejores señales primero
    senales.sort(key=lambda x: x['confianza'], reverse=True)
    return senales

# ─────────────────────────────────────────────
# MOTOR DE BACKTEST SIMULTÁNEO
# ─────────────────────────────────────────────

def correr_backtest():
    print("\n" + "=" * 60)
    print("   PROYECTO MIDAS — BACKTEST SIMULTÁNEO")
    print(f"   16 activos procesados al mismo tiempo")
    print("=" * 60)

    modelo = cargar_modelo()

    print(f"\nCargando datos de backtest...")
    df_total = pd.read_csv(RUTA_BACKTEST, parse_dates=['fecha_hora'])
    print(f"  {len(df_total):,} velas cargadas")
    print(f"  Periodo: {df_total['fecha_hora'].min().date()} → "
          f"{df_total['fecha_hora'].max().date()}")

    # Prepara datos por activo
    datos_activos = preparar_datos_activos(df_total)

    # Inicializa el gestor de riesgo
    gestor = GestorRiesgo(CONFIG_GESTOR)
    balance = CONFIG_GESTOR['balance_inicial']
    balance_inicial = balance

    # Estadísticas
    historial_balance = [balance]
    trades = []
    total_trades = 0
    trades_ganadores = 0
    trades_perdedores = 0
    confianzas = []

    # Posiciones abiertas: {activo: {accion, precio, sl, tp, lote, idx_entrada}}
    posiciones_abiertas = {}

    # Índice actual por activo
    indices_actuales = {activo: VENTANA for activo in datos_activos}

    print(f"\nIniciando simulación simultánea...")
    print(f"  Activos activos: {len(datos_activos)}")
    print(f"  Umbral mínimo : {UMBRAL_MINIMO}")
    print(f"  Max posiciones: {MAX_POSICIONES}\n")

    ciclo = 0
    max_ciclos = max(len(df) for df in datos_activos.values())

    while ciclo < max_ciclos:
        ciclo += 1

        # Actualiza el gestor con el balance actual
        fecha_actual = datos_activos[list(datos_activos.keys())[0]].iloc[
            min(indices_actuales[list(datos_activos.keys())[0]],
            len(datos_activos[list(datos_activos.keys())[0]]) - 1)
        ]['fecha_hora'].date()

        if not hasattr(correr_backtest, '_fecha_anterior'):
            correr_backtest._fecha_anterior = fecha_actual

        if fecha_actual != correr_backtest._fecha_anterior:
            # Nuevo día — reinicia contadores del gestor
            gestor.ganancia_dia = 0.0
            gestor.posiciones_abiertas = {}
            gestor.modo_actual = Modo.NORMAL
            gestor.balance_referencia = balance
            correr_backtest._fecha_anterior = fecha_actual

        # Actualiza el gestor con el balance actual
        gestor.balance_referencia = balance
        gestor.actualizar_modo(balance)

        if gestor.modo_actual in [Modo.META_ALCANZADA, Modo.LIMITE_PERDIDA]:
            # Avanza todos los índices
            for activo in indices_actuales:
                indices_actuales[activo] += 1
            continue

        # Verifica posiciones cerradas
        posiciones_a_cerrar = []
        for activo, pos in posiciones_abiertas.items():
            df_activo = datos_activos[activo]
            idx_actual = indices_actuales[activo]

            if idx_actual >= len(df_activo):
                posiciones_a_cerrar.append(activo)
                continue

            vela_actual = df_activo.iloc[idx_actual]

            # Verifica si tocó SL o TP
            divisor = 100 if 'JPY' in activo else 10000
            pip_val = 1 / divisor
            valor_pip = 10.0 * pos['lote']
            cerrado = False
            ganancia = 0

            if pos['accion'] == 'BUY':
                sl_precio = pos['precio'] - (pos['sl_pips'] * pip_val)
                tp_precio = pos['precio'] + (pos['tp_pips'] * pip_val)
                if vela_actual['low'] <= sl_precio:
                    ganancia = -pos['sl_pips'] * valor_pip
                    cerrado = True
                elif vela_actual['high'] >= tp_precio:
                    ganancia = pos['tp_pips'] * valor_pip
                    cerrado = True
            else:
                sl_precio = pos['precio'] + (pos['sl_pips'] * pip_val)
                tp_precio = pos['precio'] - (pos['tp_pips'] * pip_val)
                if vela_actual['high'] >= sl_precio:
                    ganancia = -pos['sl_pips'] * valor_pip
                    cerrado = True
                elif vela_actual['low'] <= tp_precio:
                    ganancia = pos['tp_pips'] * valor_pip
                    cerrado = True

            if cerrado:
                balance += ganancia
                balance = max(balance, 0.01)
                total_trades += 1

                if ganancia > 0:
                    trades_ganadores += 1
                else:
                    trades_perdedores += 1

                trades.append({
                    'activo'   : activo,
                    'accion'   : pos['accion'],
                    'resultado': round(ganancia, 2),
                    'confianza': round(pos['confianza'], 4),
                    'balance'  : round(balance, 2)
                })

                historial_balance.append(balance)
                gestor.registrar_cierre(activo, ganancia)
                posiciones_a_cerrar.append(activo)

        for activo in posiciones_a_cerrar:
            posiciones_abiertas.pop(activo, None)

        # Obtiene señales simultáneas de todos los activos
        if len(posiciones_abiertas) < MAX_POSICIONES:
            senales = obtener_senales_simultaneas(modelo, datos_activos, indices_actuales)

            for senal in senales:
                activo = senal['activo']

                # No abrir si ya hay posición en este activo
                if activo in posiciones_abiertas:
                    continue

                # No abrir más del máximo permitido
                if len(posiciones_abiertas) >= MAX_POSICIONES:
                    break

                # Consulta al gestor
                puede, _, lote = gestor.puede_operar(
                    activo, senal['confianza'], senal['umbral']
                )

                if puede:
                    idx = senal['idx']
                    precio = datos_activos[activo].iloc[idx]['close']

                    posiciones_abiertas[activo] = {
                        'accion'   : senal['accion'],
                        'precio'   : precio,
                        'sl_pips'  : senal['sl_pips'],
                        'tp_pips'  : senal['tp_pips'],
                        'lote'     : lote,
                        'confianza': senal['confianza']
                    }

                    confianzas.append(senal['confianza'])
                    gestor.registrar_apertura(activo)

        # Avanza todos los índices
        for activo in indices_actuales:
            indices_actuales[activo] = min(
                indices_actuales[activo] + 1,
                len(datos_activos[activo]) - 1
            )

        # Muestra progreso cada 1000 ciclos
        if ciclo % 5000 == 0:
            pct = ciclo / max_ciclos * 100
            print(f"  Progreso: {pct:.1f}% | Balance: ${balance:.2f} | "
                  f"Trades: {total_trades} | Posiciones: {len(posiciones_abiertas)}")

    # ─────────────────────────────────────────────
    # REPORTE FINAL
    # ─────────────────────────────────────────────

    print("\n" + "=" * 60)
    print("   REPORTE FINAL DEL BACKTEST")
    print("=" * 60)

    ganancia_total = balance - balance_inicial
    pct_ganancia = (ganancia_total / balance_inicial) * 100

    balance_arr = np.array(historial_balance)
    picos = np.maximum.accumulate(balance_arr)
    drawdowns = (picos - balance_arr) / picos * 100
    drawdown_max = drawdowns.max()

    win_rate = (trades_ganadores / total_trades * 100) if total_trades > 0 else 0
    conf_media = np.mean(confianzas) if confianzas else 0

    print(f"\n  Balance inicial    : ${balance_inicial:.2f}")
    print(f"  Balance final      : ${balance:.2f}")
    print(f"  Ganancia total     : ${ganancia_total:+.2f} ({pct_ganancia:+.2f}%)")
    print(f"\n  Total trades       : {total_trades}")
    print(f"  Trades ganadores   : {trades_ganadores} ({win_rate:.1f}%)")
    print(f"  Trades perdedores  : {trades_perdedores}")
    print(f"\n  Drawdown máximo    : {drawdown_max:.2f}%")
    print(f"  Confianza media    : {conf_media:.4f}")
    print("=" * 60)

    # Gráfica
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.patch.set_facecolor('#0d1117')

    ax1 = axes[0]
    ax1.set_facecolor('#161b22')
    ax1.plot(historial_balance, color='#58a6ff', linewidth=1.5, label='Balance')
    ax1.axhline(y=balance_inicial, color='#8b949e', linestyle='--',
                linewidth=1, label=f'Balance inicial ${balance_inicial:.0f}')
    ax1.fill_between(range(len(historial_balance)),
                     balance_inicial, historial_balance,
                     where=[b >= balance_inicial for b in historial_balance],
                     alpha=0.2, color='#3fb950')
    ax1.fill_between(range(len(historial_balance)),
                     balance_inicial, historial_balance,
                     where=[b < balance_inicial for b in historial_balance],
                     alpha=0.2, color='#f85149')
    ax1.set_title('Proyecto Midas — Backtest Simultáneo (16 activos)',
                  color='#e6edf3', fontsize=13, pad=10)
    ax1.set_ylabel('Balance ($)', color='#8b949e')
    ax1.tick_params(colors='#8b949e')
    ax1.spines['bottom'].set_color('#30363d')
    ax1.spines['top'].set_color('#30363d')
    ax1.spines['left'].set_color('#30363d')
    ax1.spines['right'].set_color('#30363d')
    ax1.legend(facecolor='#161b22', labelcolor='#e6edf3')
    ax1.grid(True, alpha=0.1, color='#30363d')

    ax2 = axes[1]
    ax2.set_facecolor('#161b22')
    drawdowns2 = (np.maximum.accumulate(balance_arr) - balance_arr) / np.maximum.accumulate(balance_arr) * 100
    ax2.fill_between(range(len(drawdowns2)), 0, -drawdowns2, alpha=0.6, color='#f85149')
    ax2.plot(range(len(drawdowns2)), -drawdowns2, color='#f85149', linewidth=1)
    ax2.set_title('Drawdown (%)', color='#e6edf3', fontsize=11, pad=10)
    ax2.set_ylabel('Drawdown (%)', color='#8b949e')
    ax2.set_xlabel('Número de trades', color='#8b949e')
    ax2.tick_params(colors='#8b949e')
    ax2.spines['bottom'].set_color('#30363d')
    ax2.spines['top'].set_color('#30363d')
    ax2.spines['left'].set_color('#30363d')
    ax2.spines['right'].set_color('#30363d')
    ax2.grid(True, alpha=0.1, color='#30363d')

    stats_text = (f'Balance final: ${balance:.2f} ({pct_ganancia:+.2f}%)  |  '
                  f'Win rate: {win_rate:.1f}%  |  '
                  f'Trades: {total_trades}  |  '
                  f'Max Drawdown: {drawdown_max:.2f}%  |  '
                  f'Conf. media: {conf_media:.4f}')
    fig.text(0.5, 0.01, stats_text, ha='center', color='#8b949e', fontsize=9)

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    ruta_grafica = os.path.join('datos', 'curva_capital.png')
    plt.savefig(ruta_grafica, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.show()
    print(f"\n  Grafica guardada en: {ruta_grafica}")

    # Guarda reporte
    reporte = {
        'fecha_backtest'   : datetime.now().isoformat(),
        'balance_inicial'  : balance_inicial,
        'balance_final'    : round(balance, 2),
        'ganancia_total'   : round(ganancia_total, 2),
        'pct_ganancia'     : round(pct_ganancia, 2),
        'total_trades'     : total_trades,
        'trades_ganadores' : trades_ganadores,
        'trades_perdedores': trades_perdedores,
        'win_rate'         : round(win_rate, 2),
        'drawdown_max'     : round(drawdown_max, 2),
        'confianza_media'  : round(conf_media, 4),
        'historial_balance': historial_balance,
        'trades'           : trades
    }

    with open(os.path.join('datos', 'reporte_backtest.json'), 'w') as f:
        json.dump(reporte, f, indent=2)

    return reporte

# ─────────────────────────────────────────────
# PUNTO DE ENTRADA
# ─────────────────────────────────────────────

if __name__ == "__main__":
    correr_backtest()