# modelo.py
# Arquitectura del modelo Transformer para el Proyecto Midas
# Optimizado para RTX 4060 Ti 16GB
# Proyecto Midas — Parte 2

import torch
import torch.nn as nn
import math

# ─────────────────────────────────────────────
# CONFIGURACIÓN DEL MODELO
# ─────────────────────────────────────────────

CONFIG = {
    # Entrada
    'n_features'   : 11,     # OHLCV + EMA9 + EMA21 + RSI + MACD + MACD_signal + ATR
    'ventana'      : 20,     # velas de contexto
    'n_activos'    : 6,      # EURUSD, GBPUSD, etc.

    # Arquitectura Transformer
    'd_model'      : 128,    # dimensión interna del modelo
    'n_heads'      : 8,      # cabezas de atención (128/8=16 por cabeza)
    'n_layers'     : 4,      # capas del Transformer
    'd_ff'         : 512,    # dimensión de la capa feed-forward
    'dropout'      : 0.1,    # regularización

    # Salidas
    'n_clases'     : 3,      # BUY / SELL / HOLD
}

# ─────────────────────────────────────────────
# CODIFICACIÓN POSICIONAL
# ─────────────────────────────────────────────

class CodificacionPosicional(nn.Module):
    """
    Le dice al modelo en qué posición de la secuencia está cada vela
    Sin esto el Transformer no sabe el orden de las velas
    """
    def __init__(self, d_model, ventana, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Crea la matriz de posiciones
        pe = torch.zeros(ventana, d_model)
        posicion = torch.arange(0, ventana).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(posicion * div_term)
        pe[:, 1::2] = torch.cos(posicion * div_term)

        # Agrega dimensión de batch: (1, ventana, d_model)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Suma la codificación posicional a la entrada
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# ─────────────────────────────────────────────
# MODELO PRINCIPAL
# ─────────────────────────────────────────────

class MidasTransformer(nn.Module):
    """
    Transformer completo para trading algorítmico

    Entrada:
        velas    : (batch, 20, 5) — ventana OHLCV normalizada
        activo_id: (batch,)       — qué par de divisas es

    Salidas:
        logits_accion    : (batch, 3)  — BUY/SELL/HOLD
        sl_pips          : (batch, 1)  — stop loss en pips
        tp_pips          : (batch, 1)  — take profit en pips
        confianza        : (batch, 1)  — score 0.0 a 1.0
        umbral_dinamico  : (batch, 1)  — umbral mínimo aprendido
    """
    def __init__(self, config=CONFIG):
        super().__init__()
        self.config = config

        d_model = config['d_model']

        # ── Proyección de entrada ──
        # Convierte las 5 features OHLCV a d_model dimensiones
        self.proyeccion_entrada = nn.Linear(config['n_features'], d_model)

        # ── Embedding del activo ──
        # Cada par de divisas tiene su propio vector aprendido
        self.embedding_activo = nn.Embedding(config['n_activos'], d_model)

        # ── Codificación posicional ──
        self.cod_posicional = CodificacionPosicional(
            d_model, config['ventana'], config['dropout']
        )

        # ── Capas Transformer ──
        capa_encoder = nn.TransformerEncoderLayer(
            d_model      = d_model,
            nhead        = config['n_heads'],
            dim_feedforward = config['d_ff'],
            dropout      = config['dropout'],
            batch_first  = True,   # (batch, seq, features)
            norm_first   = True    # Pre-norm: más estable en entrenamiento
        )
        self.transformer = nn.TransformerEncoder(
            capa_encoder,
            num_layers = config['n_layers']
        )

        # ── Normalización final ──
        self.norm_final = nn.LayerNorm(d_model)

        # ── Cabezas de salida ──

        # Clasificación: BUY / SELL / HOLD
        self.cabeza_accion = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(config['dropout']),
            nn.Linear(d_model // 2, config['n_clases'])
        )

        # Regresión: stop loss en pips
        self.cabeza_sl = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Softplus()   # Siempre positivo
        )

        # Regresión: take profit en pips
        self.cabeza_tp = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Softplus()   # Siempre positivo
        )

        # Score de confianza (qué tan seguro está el modelo)
        self.cabeza_confianza = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()    # Entre 0.0 y 1.0
        )

        # Umbral dinámico (el modelo aprende cuándo operar)
        self.cabeza_umbral = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()    # Entre 0.0 y 1.0
        )

        # Inicializa los pesos correctamente
        self._inicializar_pesos()

    def _inicializar_pesos(self):
        """Inicialización Xavier para mejor convergencia"""
        for modulo in self.modules():
            if isinstance(modulo, nn.Linear):
                nn.init.xavier_uniform_(modulo.weight)
                if modulo.bias is not None:
                    nn.init.zeros_(modulo.bias)
            elif isinstance(modulo, nn.Embedding):
                nn.init.normal_(modulo.weight, std=0.02)

    def forward(self, velas, activo_id):
        """
        Paso hacia adelante del modelo

        velas     : (batch, 20, 5)
        activo_id : (batch,)
        """
        batch_size = velas.size(0)

        # 1. Proyecta las velas a d_model dimensiones
        x = self.proyeccion_entrada(velas)  # (batch, 20, d_model)

        # 2. Agrega el embedding del activo a cada paso de tiempo
        emb_activo = self.embedding_activo(activo_id)          # (batch, d_model)
        emb_activo = emb_activo.unsqueeze(1).expand(-1, 20, -1) # (batch, 20, d_model)
        x = x + emb_activo

        # 3. Agrega codificación posicional
        x = self.cod_posicional(x)

        # 4. Pasa por el Transformer
        x = self.transformer(x)  # (batch, 20, d_model)

        # 5. Normalización final
        x = self.norm_final(x)

        # 6. Toma solo el último token para las predicciones
        # (el último token resume toda la secuencia)
        x_ultimo = x[:, -1, :]  # (batch, d_model)

        # 7. Calcula todas las salidas
        logits_accion   = self.cabeza_accion(x_ultimo)    # (batch, 3)
        sl_pips         = self.cabeza_sl(x_ultimo)         # (batch, 1)
        tp_pips         = self.cabeza_tp(x_ultimo)         # (batch, 1)
        confianza       = self.cabeza_confianza(x_ultimo)  # (batch, 1)
        umbral_dinamico = self.cabeza_umbral(x_ultimo)     # (batch, 1)

        return logits_accion, sl_pips, tp_pips, confianza, umbral_dinamico

    def contar_parametros(self):
        """Cuenta cuántos parámetros entrenables tiene el modelo"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ─────────────────────────────────────────────
# PRUEBA RÁPIDA
# ─────────────────────────────────────────────

if __name__ == "__main__":
    dispositivo = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo: {dispositivo}")

    # Crea el modelo
    modelo = MidasTransformer().to(dispositivo)
    print(f"Parametros entrenables: {modelo.contar_parametros():,}")

    # Prueba con un batch falso
    batch_size = 8
    velas     = torch.randn(batch_size, 20, 5).to(dispositivo)
    activo_id = torch.randint(0, 6, (batch_size,)).to(dispositivo)

    # Paso hacia adelante
    with torch.no_grad():
        logits, sl, tp, conf, umbral = modelo(velas, activo_id)

    print(f"\nFormas de salida:")
    print(f"  logits_accion   : {logits.shape}")
    print(f"  sl_pips         : {sl.shape}")
    print(f"  tp_pips         : {tp.shape}")
    print(f"  confianza       : {conf.shape}")
    print(f"  umbral_dinamico : {umbral.shape}")
    print(f"\nEjemplo de salida (primera muestra del batch):")
    print(f"  Logits  : {logits[0].cpu().numpy()}")
    print(f"  SL pips : {sl[0].item():.2f}")
    print(f"  TP pips : {tp[0].item():.2f}")
    print(f"  Conf    : {conf[0].item():.4f}")
    print(f"  Umbral  : {umbral[0].item():.4f}")
    print("\nModelo funcionando correctamente!")