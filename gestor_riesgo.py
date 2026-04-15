# gestor_riesgo.py
# Gestor de riesgo del Proyecto Midas
# Controla cuándo y cuánto operar según el rendimiento del día
# Proyecto Midas — Parte 3

import json
import os
import logging
import sys
from datetime import datetime, time
from enum import Enum

# ─────────────────────────────────────────────
# CONFIGURACIÓN DE LOGS
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [GESTOR] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'gestor.log'), encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
sys.stdout.reconfigure(encoding='utf-8')
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# MODOS DEL GESTOR
# ─────────────────────────────────────────────

class Modo(Enum):
    """Los 6 modos de comportamiento del gestor de riesgo"""
    NORMAL          = "normal"           # Opera normalmente
    CAUTELOSO       = "cauteloso"        # Solo señales de alta confianza
    META_ALCANZADA  = "meta_alcanzada"   # Bot detenido — meta diaria lograda
    LIMITE_PERDIDA  = "limite_perdida"   # Bot detenido — pérdida máxima
    POSICION_ABIERTA = "posicion_abierta" # Ya hay trade en ese activo
    NUEVO_DIA       = "nuevo_dia"        # Reinicio de contadores

# ─────────────────────────────────────────────
# GESTOR DE RIESGO PRINCIPAL
# ─────────────────────────────────────────────

class GestorRiesgo:
    """
    Gestor de riesgo del Proyecto Midas

    Controla:
    - Cuándo operar (modo activo)
    - Cuánto operar (tamaño de lote)
    - Cuándo detener el bot (meta/límite)
    - Reinicio automático diario
    - Interés compuesto automático
    """

    def __init__(self, config: dict = None):
        # Configuración con valores por defecto
        cfg = config or {}

        # Porcentajes del día (sobre el balance de referencia)
        self.meta_pct          = cfg.get('meta_pct',          2.0)   # +2% meta diaria
        self.umbral_cauteloso  = cfg.get('umbral_cauteloso',  2.0)   # +2% modo cauteloso
        self.techo_pct         = cfg.get('techo_pct',         5.0)   # +5% techo máximo
        self.limite_perdida_pct = cfg.get('limite_perdida_pct', 1.0) # -1% límite pérdida

        # Confianza mínima en modo cauteloso
        self.confianza_cautelosa = cfg.get('confianza_cautelosa', 0.85)

        # Balance de referencia (se actualiza cada día a las 00:00)
        self.balance_referencia = cfg.get('balance_inicial', 1000.0)
        self.balance_inicial    = cfg.get('balance_inicial', 1000.0)

        # Estado del día
        self.ganancia_dia    = 0.0   # Ganancia/pérdida acumulada hoy en dinero
        self.modo_actual     = Modo.NORMAL
        self.posiciones_abiertas = {}  # {activo: True/False}
        self.fecha_actual    = datetime.now().date()

        # Archivo de estado para recuperación tras cortes
        self.archivo_estado = os.path.join('licencia', 'estado_gestor.json')

        # Carga estado previo si existe
        self._cargar_estado()

        log.info(f"Gestor de riesgo iniciado")
        log.info(f"  Balance referencia : ${self.balance_referencia:.2f}")
        log.info(f"  Meta diaria        : +{self.meta_pct}%")
        log.info(f"  Límite pérdida     : -{self.limite_perdida_pct}%")
        log.info(f"  Techo operación    : +{self.techo_pct}%")

    # ─────────────────────────────────────────────
    # CÁLCULOS PRINCIPALES
    # ─────────────────────────────────────────────

    def _pct_ganancia(self) -> float:
        """Calcula el % de ganancia/pérdida del día respecto al balance de referencia"""
        if self.balance_referencia <= 0:
            return 0.0
        return (self.ganancia_dia / self.balance_referencia) * 100

    def _calcular_lote(self, balance_actual: float) -> float:
        """
        Calcula el tamaño de lote dinámico según el balance actual
        Usa el 1% del balance como margen base
        Mínimo: 0.01 lotes | Máximo: 1.0 lote
        """
        # Fórmula: balance * 0.01 / 1000 (un lote estándar = $1000 de margen aprox)
        lote = (balance_actual * 0.01) / 1000
        lote = round(lote, 2)
        lote = max(0.01, min(1.0, lote))
        return lote

    # ─────────────────────────────────────────────
    # ACTUALIZACIÓN DEL MODO
    # ─────────────────────────────────────────────

    def actualizar_modo(self, balance_actual: float):
        """
        Revisa el rendimiento del día y actualiza el modo del gestor
        Se debe llamar antes de cada decisión de trading
        """
        # Verifica si es un nuevo día
        hoy = datetime.now().date()
        if hoy != self.fecha_actual:
            self._nuevo_dia(balance_actual)
            return

        pct = self._pct_ganancia()

        # Determina el modo según el rendimiento
        if pct <= -self.limite_perdida_pct:
            if self.modo_actual != Modo.LIMITE_PERDIDA:
                log.info(f"LIMITE DE PERDIDA alcanzado: {pct:.2f}% — Bot detenido")
            self.modo_actual = Modo.LIMITE_PERDIDA

        elif pct >= self.techo_pct:
            if self.modo_actual != Modo.META_ALCANZADA:
                log.info(f"TECHO DE OPERACION alcanzado: {pct:.2f}% — Bot detenido")
            self.modo_actual = Modo.META_ALCANZADA

        elif pct >= self.umbral_cauteloso:
            if self.modo_actual != Modo.CAUTELOSO:
                log.info(f"Modo CAUTELOSO activado: {pct:.2f}%")
            self.modo_actual = Modo.CAUTELOSO

        else:
            self.modo_actual = Modo.NORMAL

        self._guardar_estado()

    def _nuevo_dia(self, balance_actual: float):
        """
        Reinicia los contadores diarios a las 00:00
        Implementa el interés compuesto: el balance actual es la nueva base
        """
        log.info("=" * 50)
        log.info(f"NUEVO DIA — Reiniciando contadores")
        log.info(f"  Balance cierre ayer : ${self.balance_referencia + self.ganancia_dia:.2f}")
        log.info(f"  Ganancia/pérdida    : ${self.ganancia_dia:.2f} ({self._pct_ganancia():.2f}%)")

        # Interés compuesto: el nuevo balance es la base del día
        self.balance_referencia = balance_actual
        self.ganancia_dia       = 0.0
        self.fecha_actual       = datetime.now().date()
        self.posiciones_abiertas = {}
        self.modo_actual        = Modo.NORMAL

        log.info(f"  Nuevo balance base  : ${self.balance_referencia:.2f}")
        log.info("=" * 50)

        self._guardar_estado()

    # ─────────────────────────────────────────────
    # DECISIÓN DE TRADING
    # ─────────────────────────────────────────────

    def puede_operar(self, activo: str, confianza: float,
                     umbral_dinamico: float) -> tuple:
        """
        Decide si se puede abrir una operación en el activo dado

        Retorna: (puede_operar: bool, razon: str, lote: float)
        """
        # Modo detenido
        if self.modo_actual == Modo.META_ALCANZADA:
            return False, "Meta diaria alcanzada — bot detenido", 0.0

        if self.modo_actual == Modo.LIMITE_PERDIDA:
            return False, "Límite de pérdida alcanzado — bot detenido", 0.0

        # Posición ya abierta en este activo
        if self.posiciones_abiertas.get(activo, False):
            return False, f"Ya hay posición abierta en {activo}", 0.0

        # Verifica el umbral de confianza según el modo
        if self.modo_actual == Modo.CAUTELOSO:
            if confianza < self.confianza_cautelosa:
                return False, (f"Modo cauteloso: confianza {confianza:.2f} "
                               f"< {self.confianza_cautelosa}"), 0.0
        else:
            # Modo normal: usa el umbral dinámico del modelo
            if confianza < umbral_dinamico:
                return False, (f"Confianza {confianza:.2f} "
                               f"< umbral dinámico {umbral_dinamico:.2f}"), 0.0

        # Calcula el lote dinámico
        balance = self.balance_referencia + self.ganancia_dia
        lote    = self._calcular_lote(balance)

        return True, f"Señal válida en modo {self.modo_actual.value}", lote

    # ─────────────────────────────────────────────
    # REGISTRO DE OPERACIONES
    # ─────────────────────────────────────────────

    def registrar_apertura(self, activo: str):
        """Registra que se abrió una posición en el activo"""
        self.posiciones_abiertas[activo] = True
        log.info(f"Posición abierta: {activo}")
        self._guardar_estado()

    def registrar_cierre(self, activo: str, ganancia_dinero: float):
        """
        Registra el cierre de una posición y actualiza la ganancia del día
        ganancia_dinero: positivo = ganancia, negativo = pérdida
        """
        self.posiciones_abiertas[activo] = False
        self.ganancia_dia += ganancia_dinero

        pct = self._pct_ganancia()
        emoji = "+" if ganancia_dinero >= 0 else ""
        log.info(f"Posición cerrada: {activo} | "
                 f"P&L: {emoji}${ganancia_dinero:.2f} | "
                 f"Día: {pct:+.2f}%")

        self._guardar_estado()

    def reiniciar_contadores(self, nuevo_balance: float):
        """
        Reinicio manual de contadores
        Útil cuando se hace un depósito o retiro que cambia el balance
        sin que sea ganancia/pérdida real
        """
        log.info(f"Reinicio manual de contadores")
        log.info(f"  Balance anterior: ${self.balance_referencia:.2f}")
        log.info(f"  Nuevo balance   : ${nuevo_balance:.2f}")

        self.balance_referencia  = nuevo_balance
        self.ganancia_dia        = 0.0
        self.posiciones_abiertas = {}
        self.modo_actual         = Modo.NORMAL

        self._guardar_estado()

    # ─────────────────────────────────────────────
    # ESTADO Y PANEL
    # ─────────────────────────────────────────────

    def obtener_estado(self) -> dict:
        """Retorna el estado actual del gestor para mostrar en el panel"""
        balance_actual = self.balance_referencia + self.ganancia_dia
        pct_dia        = self._pct_ganancia()
        pct_total      = ((balance_actual - self.balance_inicial)
                          / self.balance_inicial * 100)

        # Proyección a 30 días basada en el ritmo actual del día
        hora_actual    = datetime.now().hour + datetime.now().minute / 60
        horas_mercado  = max(hora_actual, 1)
        pct_por_hora   = pct_dia / horas_mercado
        proyeccion_30d = balance_actual * (1 + pct_por_hora * 8 / 100) ** 30

        return {
            'modo'             : self.modo_actual.value,
            'balance_referencia': self.balance_referencia,
            'balance_actual'   : balance_actual,
            'ganancia_dia'     : self.ganancia_dia,
            'pct_dia'          : pct_dia,
            'pct_total'        : pct_total,
            'balance_inicial'  : self.balance_inicial,
            'proyeccion_30d'   : proyeccion_30d,
            'posiciones'       : self.posiciones_abiertas,
            'meta_pct'         : self.meta_pct,
            'limite_pct'       : self.limite_perdida_pct,
            'techo_pct'        : self.techo_pct,
        }

    def mostrar_estado(self):
        """Muestra el estado en consola de forma legible"""
        e = self.obtener_estado()
        print("\n" + "=" * 55)
        print("   PROYECTO MIDAS — GESTOR DE RIESGO")
        print("=" * 55)
        print(f"  Modo actual      : {e['modo'].upper()}")
        print(f"  Balance inicial  : ${e['balance_inicial']:.2f}")
        print(f"  Balance actual   : ${e['balance_actual']:.2f}")
        print(f"  Ganancia hoy     : ${e['ganancia_dia']:+.2f} ({e['pct_dia']:+.2f}%)")
        print(f"  Crecimiento total: {e['pct_total']:+.2f}%")
        print(f"  Proyección 30d   : ${e['proyeccion_30d']:.2f}")
        print(f"  Meta diaria      : +{e['meta_pct']}%")
        print(f"  Límite pérdida   : -{e['limite_pct']}%")
        print(f"  Techo operación  : +{e['techo_pct']}%")
        print(f"  Posiciones abiertas: {sum(e['posiciones'].values())}")
        print("=" * 55)

    # ─────────────────────────────────────────────
    # PERSISTENCIA
    # ─────────────────────────────────────────────

    def _guardar_estado(self):
        """Guarda el estado en disco para recuperación tras cortes"""
        estado = {
            'balance_referencia'  : self.balance_referencia,
            'balance_inicial'     : self.balance_inicial,
            'ganancia_dia'        : self.ganancia_dia,
            'modo_actual'         : self.modo_actual.value,
            'posiciones_abiertas' : self.posiciones_abiertas,
            'fecha_actual'        : str(self.fecha_actual),
            'meta_pct'            : self.meta_pct,
            'umbral_cauteloso'    : self.umbral_cauteloso,
            'techo_pct'           : self.techo_pct,
            'limite_perdida_pct'  : self.limite_perdida_pct,
            'confianza_cautelosa' : self.confianza_cautelosa,
        }
        with open(self.archivo_estado, 'w') as f:
            json.dump(estado, f, indent=2)

    def _cargar_estado(self):
        """Carga el estado previo del disco si existe"""
        if not os.path.exists(self.archivo_estado):
            return

        try:
            with open(self.archivo_estado, 'r') as f:
                estado = json.load(f)

            # Solo restaura si es el mismo día
            fecha_guardada = datetime.fromisoformat(
                estado.get('fecha_actual', '2000-01-01')
            ).date()

            if fecha_guardada == datetime.now().date():
                self.balance_referencia  = estado['balance_referencia']
                self.balance_inicial     = estado['balance_inicial']
                self.ganancia_dia        = estado['ganancia_dia']
                self.posiciones_abiertas = estado['posiciones_abiertas']
                self.modo_actual         = Modo(estado['modo_actual'])
                self.fecha_actual        = fecha_guardada
                log.info(f"Estado restaurado desde disco")
            else:
                log.info("Estado previo de otro día — iniciando fresco")

        except Exception as e:
            log.warning(f"No se pudo cargar estado previo: {e}")

# ─────────────────────────────────────────────
# PRUEBA RÁPIDA
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Crea el gestor con configuración por defecto
    gestor = GestorRiesgo({'balance_inicial': 1000.0})

    print("\n--- PRUEBA 1: Estado inicial ---")
    gestor.mostrar_estado()

    print("\n--- PRUEBA 2: Simular operaciones ---")

    # Simula una señal válida
    puede, razon, lote = gestor.puede_operar('EURUSD', 0.75, 0.60)
    print(f"EURUSD: puede_operar={puede} | lote={lote} | razon={razon}")

    # Registra apertura
    gestor.registrar_apertura('EURUSD')

    # Intenta abrir otra en el mismo activo
    puede, razon, lote = gestor.puede_operar('EURUSD', 0.80, 0.60)
    print(f"EURUSD 2do intento: puede_operar={puede} | razon={razon}")

    # Cierra con ganancia
    gestor.registrar_cierre('EURUSD', 15.50)

    print("\n--- PRUEBA 3: Simular día ganador (supera meta) ---")
    gestor.registrar_cierre('GBPUSD', 35.0)  # Supera el 2% de $1000
    gestor.actualizar_modo(1050.0)
    gestor.mostrar_estado()

    print("\n--- PRUEBA 4: Intentar operar en modo meta alcanzada ---")
    puede, razon, lote = gestor.puede_operar('USDJPY', 0.90, 0.60)
    print(f"USDJPY: puede_operar={puede} | razon={razon}")