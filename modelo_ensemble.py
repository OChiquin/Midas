# modelo_ensemble.py
# Ensemble CNN + Transformer + LSTM para el Proyecto Midas
# Proyecto Midas — Parte 2 (Plan B: modelo agrandado + Feature Attention)

import torch
import torch.nn as nn
import math

# ─────────────────────────────────────────────
# CONFIGURACIÓN — MODELO MÁS GRANDE
# ─────────────────────────────────────────────

CONFIG_ENSEMBLE = {
    'n_features'  : 15,    # 11 originales + 4 nuevas
    'ventana'     : 30,
    'n_activos'   : 16,
    'd_model'     : 256,   # era 128 — el doble de capacidad
    'n_heads'     : 8,     # igual — compatible con d_model=256
    'n_layers'    : 6,     # era 4 — más profundidad
    'd_ff'        : 1024,  # era 512 — el doble
    'dropout'     : 0.2,
    'n_clases'    : 3,
    'lstm_hidden' : 256,   # era 128 — el doble
    'lstm_layers' : 2,
    'cnn_filters' : 128,   # era 64 — el doble
}

# ─────────────────────────────────────────────
# FEATURE ATTENTION
# ─────────────────────────────────────────────

class FeatureAttention(nn.Module):
    """
    Mecanismo de atención sobre las features.
    Aprende automáticamente qué feature importa más
    en cada situación de mercado.
    """
    def __init__(self, n_features, dropout=0.1):
        super().__init__()

        self.attencion = nn.Sequential(
            nn.Linear(n_features, n_features * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_features * 2, n_features),
            nn.Softmax(dim=-1)
        )
        self.norm = nn.LayerNorm(n_features)

    def forward(self, x):
        contexto = x.mean(dim=1)
        pesos    = self.attencion(contexto)
        pesos    = pesos.unsqueeze(1)
        x_pesado = x * pesos
        return self.norm(x + x_pesado)

# ─────────────────────────────────────────────
# MÓDULO CNN
# ─────────────────────────────────────────────

class ModuloCNN(nn.Module):
    """
    CNN 1D con kernels múltiples.
    Kernel 3: patrones cortos | Kernel 5: medios | Kernel 7: largos
    """
    def __init__(self, n_features, n_filtros, dropout):
        super().__init__()

        self.conv_3 = nn.Sequential(
            nn.Conv1d(n_features, n_filtros, kernel_size=3, padding=1),
            nn.BatchNorm1d(n_filtros),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.conv_5 = nn.Sequential(
            nn.Conv1d(n_features, n_filtros, kernel_size=5, padding=2),
            nn.BatchNorm1d(n_filtros),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.conv_7 = nn.Sequential(
            nn.Conv1d(n_features, n_filtros, kernel_size=7, padding=3),
            nn.BatchNorm1d(n_filtros),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.combinar = nn.Sequential(
            nn.Conv1d(n_filtros * 3, n_filtros, kernel_size=1),
            nn.BatchNorm1d(n_filtros),
            nn.GELU()
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x  = x.transpose(1, 2)
        c3 = self.conv_3(x)
        c5 = self.conv_5(x)
        c7 = self.conv_7(x)
        x  = torch.cat([c3, c5, c7], dim=1)
        x  = self.combinar(x)
        x  = self.pool(x).squeeze(-1)
        return x

# ─────────────────────────────────────────────
# MÓDULO LSTM
# ─────────────────────────────────────────────

class ModuloLSTM(nn.Module):
    """LSTM bidireccional que captura tendencias y memoria secuencial"""
    def __init__(self, n_features, hidden_size, n_layers, dropout):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size    = n_features,
            hidden_size   = hidden_size,
            num_layers    = n_layers,
            dropout       = dropout if n_layers > 1 else 0,
            batch_first   = True,
            bidirectional = True
        )
        self.norm    = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h_forward   = h_n[-2]
        h_backward  = h_n[-1]
        x = torch.cat([h_forward, h_backward], dim=1)
        x = self.norm(x)
        x = self.dropout(x)
        return x

# ─────────────────────────────────────────────
# MÓDULO TRANSFORMER
# ─────────────────────────────────────────────

class CodificacionPosicional(nn.Module):
    """Codificación posicional sinusoidal"""
    def __init__(self, d_model, ventana, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe       = torch.zeros(ventana, d_model)
        posicion = torch.arange(0, ventana).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(posicion * div_term)
        pe[:, 1::2] = torch.cos(posicion * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])

class ModuloTransformer(nn.Module):
    """Transformer que captura relaciones globales entre todas las velas"""
    def __init__(self, n_features, d_model, n_heads, n_layers, d_ff,
                 ventana, dropout):
        super().__init__()

        self.proyeccion = nn.Linear(n_features, d_model)
        self.cod_pos    = CodificacionPosicional(d_model, ventana, dropout)

        capa = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = n_heads,
            dim_feedforward = d_ff,
            dropout         = dropout,
            batch_first     = True,
            norm_first      = True
        )
        self.transformer = nn.TransformerEncoder(capa, num_layers=n_layers)
        self.norm        = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.proyeccion(x)
        x = self.cod_pos(x)
        x = self.transformer(x)
        x = self.norm(x)
        return x[:, -1, :]

# ─────────────────────────────────────────────
# ENSEMBLE COMPLETO
# ─────────────────────────────────────────────

class MidasEnsemble(nn.Module):
    """
    Ensemble CNN + Transformer + LSTM con Feature Attention.
    Versión Plan B — modelo más grande y profundo.

    Flujo:
    1. Feature Attention — aprende qué features importan más
    2. CNN              — detecta patrones locales de velas
    3. Transformer      — captura contexto global (6 capas)
    4. LSTM             — memoria secuencial y tendencias
    5. Fusión           — combina todo para la decisión final
    """
    def __init__(self, config=CONFIG_ENSEMBLE):
        super().__init__()
        self.config = config

        d  = config['d_model']
        cf = config['cnn_filters']
        lh = config['lstm_hidden'] * 2  # *2 por bidireccional

        # Feature Attention
        self.feature_attention = FeatureAttention(
            config['n_features'],
            config['dropout']
        )

        # Embedding del activo
        self.embedding_activo = nn.Embedding(config['n_activos'], d)

        # Los tres módulos
        self.cnn = ModuloCNN(
            config['n_features'], cf, config['dropout']
        )
        self.transformer = ModuloTransformer(
            config['n_features'], d, config['n_heads'],
            config['n_layers'], config['d_ff'],
            config['ventana'], config['dropout']
        )
        self.lstm = ModuloLSTM(
            config['n_features'], config['lstm_hidden'],
            config['lstm_layers'], config['dropout']
        )

        # CNN:cf + Transformer:d + LSTM:lh + Embedding:d
        tam_combinado = cf + d + lh + d

        # Capa de fusión más profunda
        self.fusion = nn.Sequential(
            nn.Linear(tam_combinado, d * 2),
            nn.GELU(),
            nn.Dropout(config['dropout']),
            nn.Linear(d * 2, d),
            nn.GELU(),
            nn.Dropout(config['dropout'])
        )

        # Cabezas de salida
        self.cabeza_accion = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Dropout(config['dropout']),
            nn.Linear(d // 2, config['n_clases'])
        )
        self.cabeza_sl = nn.Sequential(
            nn.Linear(d, d // 4),
            nn.GELU(),
            nn.Linear(d // 4, 1),
            nn.Softplus()
        )
        self.cabeza_tp = nn.Sequential(
            nn.Linear(d, d // 4),
            nn.GELU(),
            nn.Linear(d // 4, 1),
            nn.Softplus()
        )
        self.cabeza_confianza = nn.Sequential(
            nn.Linear(d, d // 4),
            nn.GELU(),
            nn.Linear(d // 4, 1),
            nn.Sigmoid()
        )
        self.cabeza_umbral = nn.Sequential(
            nn.Linear(d, d // 4),
            nn.GELU(),
            nn.Linear(d // 4, 1),
            nn.Sigmoid()
        )

        self._inicializar_pesos()

    def _inicializar_pesos(self):
        """Inicialización Xavier para mejor convergencia"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, velas, activo_id):
        """
        velas     : (batch, 30, 15)
        activo_id : (batch,)
        """
        # 1. Feature Attention
        velas_pesadas = self.feature_attention(velas)

        # 2. Embedding del activo
        emb = self.embedding_activo(activo_id)

        # 3. Tres módulos
        repr_cnn         = self.cnn(velas_pesadas)
        repr_transformer = self.transformer(velas_pesadas)
        repr_lstm        = self.lstm(velas_pesadas)

        # 4. Combina todo
        combinado = torch.cat([repr_cnn, repr_transformer, repr_lstm, emb], dim=1)

        # 5. Fusión
        x = self.fusion(combinado)

        # 6. Salidas
        logits_accion   = self.cabeza_accion(x)
        sl_pips         = self.cabeza_sl(x)
        tp_pips         = self.cabeza_tp(x)
        confianza       = self.cabeza_confianza(x)
        umbral_dinamico = self.cabeza_umbral(x)

        return logits_accion, sl_pips, tp_pips, confianza, umbral_dinamico

    def contar_parametros(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────
# PRUEBA RÁPIDA
# ─────────────────────────────────────────────

if __name__ == "__main__":
    dispositivo = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo: {dispositivo}")

    modelo = MidasEnsemble().to(dispositivo)
    print(f"Parametros entrenables: {modelo.contar_parametros():,}")

    velas     = torch.randn(8, 30, 15).to(dispositivo)
    activo_id = torch.randint(0, 16, (8,)).to(dispositivo)

    with torch.no_grad():
        logits, sl, tp, conf, umbral = modelo(velas, activo_id)

    print(f"\nFormas de salida:")
    print(f"  logits_accion   : {logits.shape}")
    print(f"  sl_pips         : {sl.shape}")
    print(f"  tp_pips         : {tp.shape}")
    print(f"  confianza       : {conf.shape}")
    print(f"  umbral_dinamico : {umbral.shape}")
    print("\nModelo Plan B funcionando!")