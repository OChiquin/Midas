# entrenar.py
# Script de entrenamiento del modelo Transformer
# Proyecto Midas — Parte 2

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
from datetime import datetime
from sklearn.metrics import f1_score
from preparar_datos import crear_dataloaders
from modelo_ensemble import MidasEnsemble, CONFIG_ENSEMBLE

# ─────────────────────────────────────────────
# CONFIGURACIÓN DE ENTRENAMIENTO
# ─────────────────────────────────────────────

CONFIG_ENTRENAMIENTO = {
    'epochs'          : 80,
    'batch_size'      : 256,
    'lr'              : 3e-4,  # LR más alto para converger más rápido
    'lr_min'          : 1e-6,
    'weight_decay'    : 1e-4,
    'patience'        : 15,
    'grad_clip'       : 1.0,
    'archivo_datos'   : os.path.join('datos', 'datos_entrenamiento.csv'),
    'carpeta_modelos' : 'modelos',
}

# ─────────────────────────────────────────────
# FUNCIÓN DE PÉRDIDA COMBINADA
# ─────────────────────────────────────────────

class PerdidaMidas(nn.Module):
    """
    Pérdida combinada que entrena simultáneamente:
    - Clasificación (BUY/SELL/HOLD) con pesos por clase
    - Regresión de SL y TP
    - Score de confianza
    """
    def __init__(self, pesos_clases):
        super().__init__()
        # Pérdida de clasificación con pesos agresivos para BUY/SELL
        self.perdida_clasificacion = nn.CrossEntropyLoss(weight=pesos_clases)
        # Pérdida de regresión para SL y TP
        self.perdida_regresion = nn.HuberLoss()

    def forward(self, logits, sl_pred, tp_pred, confianza, umbral, labels):
        """Calcula la pérdida total combinada"""

        # Pérdida principal: clasificar correctamente BUY/SELL/HOLD
        loss_clf = self.perdida_clasificacion(logits, labels)

        # Pérdida de SL
        sl_objetivo = torch.where(
            labels > 0,
            torch.full_like(sl_pred.squeeze(), 20.0),
            torch.full_like(sl_pred.squeeze(), 5.0)
        )
        loss_sl = self.perdida_regresion(sl_pred.squeeze(), sl_objetivo)

        # Pérdida de TP
        tp_objetivo = torch.where(
            labels > 0,
            torch.full_like(tp_pred.squeeze(), 40.0),
            torch.full_like(tp_pred.squeeze(), 5.0)
        )
        loss_tp = self.perdida_regresion(tp_pred.squeeze(), tp_objetivo)

        # Pérdida de confianza
        pred_correcta = (logits.argmax(dim=1) == labels).float()
        loss_conf = nn.functional.binary_cross_entropy(
            confianza.squeeze(), pred_correcta
        )

        # Pérdida total combinada
        loss_total = loss_clf + 0.1 * loss_sl + 0.1 * loss_tp + 0.1 * loss_conf

        return loss_total, loss_clf, loss_sl, loss_tp

# ─────────────────────────────────────────────
# FUNCIONES DE ENTRENAMIENTO
# ─────────────────────────────────────────────

def entrenar_epoca(modelo, loader, optimizador, criterio, dispositivo, grad_clip):
    """Entrena el modelo por una época completa"""
    modelo.train()
    loss_total = 0.0
    correctos  = 0
    total      = 0

    for batch_idx, (velas, ids, labels) in enumerate(loader):
        velas  = velas.to(dispositivo)
        ids    = ids.to(dispositivo)
        labels = labels.to(dispositivo)

        optimizador.zero_grad()
        logits, sl, tp, conf, umbral = modelo(velas, ids)

        loss, loss_clf, loss_sl, loss_tp = criterio(
            logits, sl, tp, conf, umbral, labels
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), grad_clip)
        optimizador.step()

        loss_total += loss.item()
        correctos  += (logits.argmax(dim=1) == labels).sum().item()
        total      += labels.size(0)

        if (batch_idx + 1) % 100 == 0:
            acc_parcial = 100 * correctos / total
            print(f"    Batch {batch_idx+1}/{len(loader)} — "
                  f"Loss: {loss_total/(batch_idx+1):.4f} — "
                  f"Acc: {acc_parcial:.1f}%", end='\r')

    loss_promedio = loss_total / len(loader)
    accuracy      = 100 * correctos / total
    return loss_promedio, accuracy


def validar_epoca(modelo, loader, criterio, dispositivo):
    """
    Evalúa el modelo en validación.
    Retorna loss, accuracy y F1 de BUY+SELL (métrica clave).
    """
    modelo.eval()
    loss_total  = 0.0
    correctos   = 0
    total       = 0
    y_real      = []   # Etiquetas reales
    y_pred      = []   # Predicciones del modelo

    with torch.no_grad():
        for velas, ids, labels in loader:
            velas  = velas.to(dispositivo)
            ids    = ids.to(dispositivo)
            labels = labels.to(dispositivo)

            logits, sl, tp, conf, umbral = modelo(velas, ids)
            loss, _, _, _ = criterio(logits, sl, tp, conf, umbral, labels)

            loss_total += loss.item()
            preds       = logits.argmax(dim=1)
            correctos  += (preds == labels).sum().item()
            total      += labels.size(0)

            # Acumula predicciones para calcular F1
            y_real.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    loss_promedio = loss_total / len(loader)
    accuracy      = 100 * correctos / total

    # F1 solo de BUY (1) y SELL (2) — ignoramos HOLD
    # Esta es la métrica real de calidad del modelo
    f1_buy_sell = f1_score(
        y_real, y_pred,
        labels       = [1, 2],
        average      = 'macro',
        zero_division = 0
    )

    # Distribución de predicciones para monitorear
    y_pred_arr = np.array(y_pred)
    n          = len(y_pred_arr)
    pct_hold   = 100 * (y_pred_arr == 0).sum() / n
    pct_buy    = 100 * (y_pred_arr == 1).sum() / n
    pct_sell   = 100 * (y_pred_arr == 2).sum() / n
    print(f"  Predicciones — HOLD: {pct_hold:.1f}% | "
          f"BUY: {pct_buy:.1f}% | SELL: {pct_sell:.1f}%")

    return loss_promedio, accuracy, f1_buy_sell


def guardar_modelo(modelo, config_modelo, config_entrenamiento,
                   epoca, val_acc, f1, nombre):
    """Guarda el modelo y su configuración en la carpeta modelos/"""
    carpeta = config_entrenamiento['carpeta_modelos']
    os.makedirs(carpeta, exist_ok=True)

    ruta_modelo = os.path.join(carpeta, f"{nombre}.pt")
    ruta_config = os.path.join(carpeta, f"{nombre}_config.json")

    torch.save({
        'model_state_dict' : modelo.state_dict(),
        'config_modelo'    : config_modelo,
        'epoca'            : epoca,
        'val_accuracy'     : val_acc,
        'f1_buy_sell'      : f1,
        'fecha'            : datetime.now().isoformat()
    }, ruta_modelo)

    config_guardable = {**config_modelo, **config_entrenamiento}
    with open(ruta_config, 'w') as f:
        json.dump(config_guardable, f, indent=2)

    print(f"\n  Modelo guardado: {ruta_modelo}")
    print(f"  Config guardada: {ruta_config}")

# ─────────────────────────────────────────────
# ENTRENAMIENTO PRINCIPAL
# ─────────────────────────────────────────────

def entrenar():
    """Función principal de entrenamiento"""

    print("=" * 60)
    print("   PROYECTO MIDAS — ENTRENAMIENTO DEL TRANSFORMER")
    print("=" * 60)

    dispositivo = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDispositivo: {dispositivo}")
    if dispositivo.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")

    # Carga los datos con balanceo (Opción B ya integrada en preparar_datos.py)
    cfg = CONFIG_ENTRENAMIENTO
    train_loader, val_loader, pesos_clases = crear_dataloaders(
        cfg['archivo_datos'],
        batch_size=cfg['batch_size']
    )
    pesos_clases = pesos_clases.to(dispositivo)

    # Crea el modelo desde cero
    modelo = MidasEnsemble(CONFIG_ENSEMBLE).to(dispositivo)
    print(f"\nParametros del modelo: {modelo.contar_parametros():,}")

    # Optimizador
    optimizador = optim.AdamW(
        modelo.parameters(),
        lr           = cfg['lr'],
        weight_decay = cfg['weight_decay']
    )

    # Scheduler que baja el LR cuando el F1 se estanca
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizador,
        mode     = 'max',    # Queremos maximizar el F1
        factor   = 0.5,      # Divide el LR a la mitad
        patience = 5,        # Espera 5 épocas sin mejora
        min_lr   = cfg['lr_min'],
        verbose  = True
    )

    # Función de pérdida con pesos de clase agresivos
    criterio = PerdidaMidas(pesos_clases)

    # Variables de control
    mejor_f1          = 0.0   # Guardamos por F1, no por accuracy
    epocas_sin_mejora = 0
    historial         = []

    print(f"\nIniciando entrenamiento por {cfg['epochs']} epocas...")
    print(f"Métrica principal: F1 de BUY+SELL (no accuracy total)")
    print("=" * 60)

    for epoca in range(1, cfg['epochs'] + 1):
        lr_actual = optimizador.param_groups[0]['lr']
        print(f"\nEpoca {epoca}/{cfg['epochs']} — LR: {lr_actual:.2e}")

        # Entrenamiento
        train_loss, train_acc = entrenar_epoca(
            modelo, train_loader, optimizador, criterio,
            dispositivo, cfg['grad_clip']
        )

        # Validación — ahora retorna también el F1
        val_loss, val_acc, f1_buy_sell = validar_epoca(
            modelo, val_loader, criterio, dispositivo
        )

        # Scheduler basado en F1 (no en loss)
        scheduler.step(f1_buy_sell)

        # Guarda historial
        historial.append({
            'epoca'      : epoca,
            'train_loss' : train_loss,
            'train_acc'  : train_acc,
            'val_loss'   : val_loss,
            'val_acc'    : val_acc,
            'f1_buy_sell': f1_buy_sell
        })

        print(f"\n  Train — Loss: {train_loss:.4f} | Acc: {train_acc:.1f}%")
        print(f"  Val   — Loss: {val_loss:.4f}   | Acc: {val_acc:.1f}%")
        print(f"  F1 BUY+SELL : {f1_buy_sell:.4f}  ← métrica principal")

        # Early stopping y guardado basado en F1 de BUY+SELL
        if f1_buy_sell > mejor_f1:
            mejor_f1 = f1_buy_sell
            epocas_sin_mejora = 0
            guardar_modelo(
                modelo, CONFIG_ENSEMBLE, cfg,
                epoca, val_acc, f1_buy_sell,
                nombre="midas_mejor"
            )
            print(f"  ✅ Nuevo mejor modelo! F1: {f1_buy_sell:.4f}")
        else:
            epocas_sin_mejora += 1
            print(f"  Sin mejora en F1: {epocas_sin_mejora}/{cfg['patience']}")

        # Early stopping
        if epocas_sin_mejora >= cfg['patience']:
            print(f"\nEarly stopping en epoca {epoca}")
            break

    # Guarda historial
    ruta_historial = os.path.join(cfg['carpeta_modelos'], 'historial.json')
    with open(ruta_historial, 'w') as f:
        json.dump(historial, f, indent=2)

    print("\n" + "=" * 60)
    print(f"  Entrenamiento completado!")
    print(f"  Mejor F1 BUY+SELL : {mejor_f1:.4f}")
    print(f"  Modelo guardado en : modelos/midas_mejor.pt")
    print("=" * 60)

# ─────────────────────────────────────────────
# PUNTO DE ENTRADA
# ─────────────────────────────────────────────

if __name__ == "__main__":
    entrenar()