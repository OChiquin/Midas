# bot.py
# Bot principal de ejecución del Proyecto Midas
# Cada 60 segundos lee MT5, consulta la API y ejecuta órdenes
# Proyecto Midas — Parte 4

import sys
import os
import time
import json
import logging
import requests
import threading
from datetime import datetime
import MetaTrader5 as mt5

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from gestor_riesgo import GestorRiesgo, Modo

# ─────────────────────────────────────────────
# CONFIGURACIÓN DE LOGS
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [BOT] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'bot.log'), encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
sys.stdout.reconfigure(encoding='utf-8')
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────

ACTIVOS      = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD"]
TEMPORALIDAD = mt5.TIMEFRAME_M15
API_URL      = "http://localhost:8000/predecir"
INTERVALO    = 60       # segundos entre ciclos
VELAS_API    = 60       # velas a enviar a la API (necesita al menos 40)
MAGIC        = 20240101 # número mágico para identificar las órdenes del bot

# Archivo de configuración del bot
ARCHIVO_CONFIG = os.path.join('licencia', 'config_bot.json')

# ─────────────────────────────────────────────
# CONFIGURACIÓN POR DEFECTO
# ─────────────────────────────────────────────

CONFIG_DEFAULT = {
    'balance_inicial'    : 1000.0,
    'meta_pct'           : 2.0,
    'umbral_cauteloso'   : 2.0,
    'techo_pct'          : 5.0,
    'limite_perdida_pct' : 1.0,
    'confianza_cautelosa': 0.85,
    'sl_pips_default'    : 15.0,
    'tp_pips_default'    : 25.0,
}

# ─────────────────────────────────────────────
# CLASE BOT PRINCIPAL
# ─────────────────────────────────────────────

class MidasBot:
    """
    Bot principal del Proyecto Midas

    Flujo cada 60 segundos:
    1. Verifica conexión MT5 e internet
    2. Actualiza modo del gestor de riesgo
    3. Para cada activo: obtiene velas → consulta API → decide → ejecuta
    4. Registra resultados
    """

    def __init__(self, config: dict = None):
        self.config   = config or CONFIG_DEFAULT
        self.gestor   = GestorRiesgo(self.config)
        self.corriendo = False
        self.pausado   = False
        self.modo_juego        = False
        self.modo_mantenimiento = False
        self.ultimo_activo     = None
        self.ultimo_resultado  = None
        self.ciclos            = 0
        self.internet_ok       = True
        self._lock             = threading.Lock()

        log.info("Bot Midas iniciado")

    # ─────────────────────────────────────────────
    # CONEXIÓN
    # ─────────────────────────────────────────────

    def conectar_mt5(self) -> bool:
        """Conecta con MetaTrader 5"""
        if not mt5.initialize():
            log.error(f"Error al conectar MT5: {mt5.last_error()}")
            return False
        log.info(f"Conectado a MT5: {mt5.terminal_info().name}")
        return True

    def verificar_internet(self) -> bool:
        """Verifica si hay conexión a internet verificando la API"""
        try:
            r = requests.get("http://localhost:8000", timeout=5)
            return r.status_code == 200
        except:
            return False

    def verificar_simbolo(self, activo: str) -> str:
        """Verifica que el símbolo existe en MT5"""
        info = mt5.symbol_info(activo)
        if info:
            if not info.visible:
                mt5.symbol_select(activo, True)
            return activo
        return None

    # ─────────────────────────────────────────────
    # OBTENCIÓN DE DATOS
    # ─────────────────────────────────────────────

    def obtener_velas(self, simbolo: str, n: int = VELAS_API) -> list:
        """Obtiene las últimas N velas M15 de un activo"""
        velas = mt5.copy_rates_from_pos(simbolo, TEMPORALIDAD, 0, n)
        if velas is None or len(velas) == 0:
            return None

        resultado = []
        for v in velas:
            resultado.append({
                'fecha_hora': str(datetime.fromtimestamp(v['time'])),
                'open'      : float(v['open']),
                'high'      : float(v['high']),
                'low'       : float(v['low']),
                'close'     : float(v['close']),
                'volumen'   : float(v['tick_volume'])
            })
        return resultado

    def consultar_api(self, activo: str, velas: list) -> dict:
        """Consulta la API de predicciones"""
        try:
            respuesta = requests.post(
                API_URL,
                json={'activo': activo, 'velas': velas},
                timeout=10
            )
            return respuesta.json()
        except Exception as e:
            log.error(f"Error consultando API para {activo}: {e}")
            return None

    # ─────────────────────────────────────────────
    # EJECUCIÓN DE ÓRDENES
    # ─────────────────────────────────────────────

    def ejecutar_orden(self, activo: str, accion: str,
                       lote: float, sl_pips: float, tp_pips: float) -> bool:
        """
        Ejecuta una orden en MT5

        Calcula SL y TP en precios reales a partir de los pips
        """
        info    = mt5.symbol_info(activo)
        tick    = mt5.symbol_info_tick(activo)

        if not info or not tick:
            log.error(f"No se pudo obtener info de {activo}")
            return False

        # Calcula el valor del pip
        divisor = 100 if 'JPY' in activo else 10000
        pip_val = 1 / divisor

        if accion == 'BUY':
            precio = tick.ask
            sl     = round(precio - sl_pips * pip_val, info.digits)
            tp     = round(precio + tp_pips * pip_val, info.digits)
            tipo   = mt5.ORDER_TYPE_BUY
        elif accion == 'SELL':
            precio = tick.bid
            sl     = round(precio + sl_pips * pip_val, info.digits)
            tp     = round(precio - tp_pips * pip_val, info.digits)
            tipo   = mt5.ORDER_TYPE_SELL
        else:
            return False

        # Prepara la orden
        orden = {
            'action'       : mt5.TRADE_ACTION_DEAL,
            'symbol'       : activo,
            'volume'       : lote,
            'type'         : tipo,
            'price'        : precio,
            'sl'           : sl,
            'tp'           : tp,
            'magic'        : MAGIC,
            'comment'      : 'Midas Bot',
            'type_time'    : mt5.ORDER_TIME_GTC,
            'type_filling' : mt5.ORDER_FILLING_IOC,
        }

        # Envía la orden
        resultado = mt5.order_send(orden)

        if resultado.retcode == mt5.TRADE_RETCODE_DONE:
            log.info(f"Orden ejecutada: {accion} {lote} {activo} "
                     f"@ {precio} | SL:{sl} TP:{tp}")
            return True
        else:
            log.error(f"Error al ejecutar orden {activo}: "
                      f"retcode={resultado.retcode} ({resultado.comment})")
            return False

    def verificar_posiciones_cerradas(self):
        """
        Verifica si alguna posición abierta se cerró (por SL o TP)
        y registra el resultado en el gestor
        """
        for activo in ACTIVOS:
            if not self.gestor.posiciones_abiertas.get(activo, False):
                continue

            # Busca si hay posición abierta en MT5
            posiciones = mt5.positions_get(symbol=activo)
            posiciones_bot = [p for p in (posiciones or [])
                              if p.magic == MAGIC]

            if not posiciones_bot:
                # La posición se cerró — obtiene el resultado del historial
                historial = mt5.history_deals_get(
                    datetime.now().replace(hour=0, minute=0, second=0),
                    datetime.now()
                )
                if historial:
                    deals_activo = [d for d in historial
                                    if d.symbol == activo and d.magic == MAGIC]
                    if deals_activo:
                        ganancia = sum(d.profit for d in deals_activo)
                        self.gestor.registrar_cierre(activo, ganancia)
                        self.ultimo_activo   = activo
                        self.ultimo_resultado = ganancia
                else:
                    # No hay historial disponible, registra cierre con 0
                    self.gestor.registrar_cierre(activo, 0.0)

    # ─────────────────────────────────────────────
    # CICLO PRINCIPAL
    # ─────────────────────────────────────────────

    def ciclo(self):
        """Un ciclo completo del bot"""
        self.ciclos += 1

        # Verifica conexión a internet/API
        if not self.verificar_internet():
            if self.internet_ok:
                log.warning("Sin conexión a la API — pausando ejecución")
                self.internet_ok = False
            return

        if not self.internet_ok:
            log.info("Conexión recuperada — reanudando")
            self.internet_ok = True

        # Obtiene el balance actual de MT5
        info_cuenta = mt5.account_info()
        if not info_cuenta:
            log.error("No se pudo obtener info de la cuenta MT5")
            return

        balance_actual = info_cuenta.balance

        # Verifica posiciones que se cerraron
        self.verificar_posiciones_cerradas()

        # Actualiza el modo del gestor
        self.gestor.actualizar_modo(balance_actual)

        # Si el bot está detenido, no opera
        if self.gestor.modo_actual in [Modo.META_ALCANZADA, Modo.LIMITE_PERDIDA]:
            return

        # Procesa cada activo
        for activo in ACTIVOS:
            try:
                simbolo = self.verificar_simbolo(activo)
                if not simbolo:
                    continue

                # Obtiene las velas
                velas = self.obtener_velas(simbolo)
                if not velas:
                    continue

                # Consulta la API
                prediccion = self.consultar_api(activo, velas)
                if not prediccion:
                    continue

                accion          = prediccion['accion']
                confianza       = prediccion['confianza']
                umbral_dinamico = prediccion['umbral_dinamico']
                sl_pips         = prediccion['stop_loss_pips'] or self.config['sl_pips_default']
                tp_pips         = prediccion['take_profit_pips'] or self.config['tp_pips_default']

                # Modo juego: solo confianza muy alta y usa CPU
                if self.modo_juego and confianza < 0.90:
                    continue

                # Consulta al gestor si puede operar
                puede, razon, lote = self.gestor.puede_operar(
                    activo, confianza, umbral_dinamico
                )

                if not puede:
                    log.info(f"{activo}: {accion} | conf={confianza:.2f} | {razon}")
                    continue

                # Ejecuta la orden si no es HOLD
                if accion in ['BUY', 'SELL']:
                    log.info(f"{activo}: {accion} | conf={confianza:.2f} | "
                             f"lote={lote} | SL={sl_pips} TP={tp_pips}")
                    exito = self.ejecutar_orden(activo, accion, lote, sl_pips, tp_pips)
                    if exito:
                        self.gestor.registrar_apertura(activo)

            except Exception as e:
                log.error(f"Error procesando {activo}: {e}")

    # ─────────────────────────────────────────────
    # CONTROL DEL BOT
    # ─────────────────────────────────────────────

    def iniciar(self):
        """Inicia el bot en un hilo separado"""
        if self.corriendo:
            log.warning("El bot ya está corriendo")
            return

        if not self.conectar_mt5():
            return

        self.corriendo = True
        self.pausado   = False

        hilo = threading.Thread(target=self._bucle, daemon=True)
        hilo.start()
        log.info("Bot iniciado")

    def _bucle(self):
        """Bucle principal del bot"""
        while self.corriendo:
            if not self.pausado and not self.modo_mantenimiento:
                with self._lock:
                    try:
                        self.ciclo()
                    except Exception as e:
                        log.error(f"Error en ciclo: {e}")
            time.sleep(INTERVALO)

    def pausar(self):
        """Pausa el bot (no cierra posiciones)"""
        self.pausado = True
        log.info("Bot pausado")

    def reanudar(self):
        """Reanuda el bot"""
        self.pausado = False
        log.info("Bot reanudado")

    def detener(self):
        """Detiene el bot completamente"""
        self.corriendo = False
        mt5.shutdown()
        log.info("Bot detenido")

    def activar_modo_juego(self, activar: bool):
        """Activa/desactiva el modo juego (reduce uso de GPU)"""
        self.modo_juego = activar
        estado = "activado" if activar else "desactivado"
        log.info(f"Modo juego {estado}")

    def activar_mantenimiento(self, activar: bool):
        """Activa/desactiva el modo mantenimiento"""
        self.modo_mantenimiento = activar
        if activar:
            log.info("Modo mantenimiento activado — cerrando posiciones...")
            # Cierra todas las posiciones abiertas
            for activo in ACTIVOS:
                posiciones = mt5.positions_get(symbol=activo)
                for pos in (posiciones or []):
                    if pos.magic == MAGIC:
                        self._cerrar_posicion(pos)
        else:
            log.info("Modo mantenimiento desactivado — reanudando")

    def _cerrar_posicion(self, posicion):
        """Cierra una posición específica"""
        tick = mt5.symbol_info_tick(posicion.symbol)
        if not tick:
            return

        tipo   = mt5.ORDER_TYPE_SELL if posicion.type == 0 else mt5.ORDER_TYPE_BUY
        precio = tick.bid if posicion.type == 0 else tick.ask

        orden = {
            'action'       : mt5.TRADE_ACTION_DEAL,
            'symbol'       : posicion.symbol,
            'volume'       : posicion.volume,
            'type'         : tipo,
            'position'     : posicion.ticket,
            'price'        : precio,
            'magic'        : MAGIC,
            'comment'      : 'Midas cierre',
            'type_time'    : mt5.ORDER_TIME_GTC,
            'type_filling' : mt5.ORDER_FILLING_IOC,
        }
        mt5.order_send(orden)

    def obtener_estado(self) -> dict:
        """Retorna el estado completo del bot"""
        estado_gestor = self.gestor.obtener_estado()
        return {
            **estado_gestor,
            'corriendo'          : self.corriendo,
            'pausado'            : self.pausado,
            'modo_juego'         : self.modo_juego,
            'modo_mantenimiento' : self.modo_mantenimiento,
            'ciclos'             : self.ciclos,
            'ultimo_activo'      : self.ultimo_activo,
            'ultimo_resultado'   : self.ultimo_resultado,
            'internet_ok'        : self.internet_ok,
        }

# ─────────────────────────────────────────────
# CONTROL POR TERMINAL
# ─────────────────────────────────────────────

def control_terminal(bot: MidasBot):
    """Permite controlar el bot desde la terminal"""
    print("\nComandos disponibles:")
    print("  start    → iniciar bot")
    print("  pause    → pausar bot")
    print("  stop     → detener bot")
    print("  status   → ver estado")
    print("  reset    → reiniciar contadores")
    print("  gaming-mode on/off  → modo juego")
    print("  maintenance on/off  → modo mantenimiento")
    print("  exit     → salir\n")

    while True:
        try:
            cmd = input("Midas> ").strip().lower()

            if cmd == 'start':
                bot.iniciar()

            elif cmd == 'pause':
                bot.pausar()

            elif cmd == 'stop':
                bot.detener()
                break

            elif cmd == 'status':
                estado = bot.obtener_estado()
                print(f"\n  Modo          : {estado['modo']}")
                print(f"  Corriendo     : {estado['corriendo']}")
                print(f"  Pausado       : {estado['pausado']}")
                print(f"  Balance       : ${estado['balance_actual']:.2f}")
                print(f"  Ganancia hoy  : ${estado['ganancia_dia']:+.2f} ({estado['pct_dia']:+.2f}%)")
                print(f"  Crecimiento   : {estado['pct_total']:+.2f}%")
                print(f"  Ciclos        : {estado['ciclos']}")
                print(f"  Internet      : {'OK' if estado['internet_ok'] else 'SIN CONEXION'}\n")

            elif cmd == 'reset':
                info = mt5.account_info()
                if info:
                    bot.gestor.reiniciar_contadores(info.balance)

            elif cmd == 'gaming-mode on':
                bot.activar_modo_juego(True)

            elif cmd == 'gaming-mode off':
                bot.activar_modo_juego(False)

            elif cmd == 'maintenance on':
                bot.activar_mantenimiento(True)

            elif cmd == 'maintenance off':
                bot.activar_mantenimiento(False)

            elif cmd == 'exit':
                bot.detener()
                break

            else:
                print(f"Comando no reconocido: {cmd}")

        except KeyboardInterrupt:
            bot.detener()
            break

# ─────────────────────────────────────────────
# PUNTO DE ENTRADA
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Carga configuración si existe
    config = CONFIG_DEFAULT.copy()
    if os.path.exists(ARCHIVO_CONFIG):
        with open(ARCHIVO_CONFIG, 'r') as f:
            config.update(json.load(f))

    print("\n" + "=" * 55)
    print("   PROYECTO MIDAS — BOT DE EJECUCIÓN")
    print("=" * 55)
    print(f"  Balance inicial : ${config['balance_inicial']:.2f}")
    print(f"  Meta diaria     : +{config['meta_pct']}%")
    print(f"  Límite pérdida  : -{config['limite_perdida_pct']}%")
    print("=" * 55 + "\n")

    bot = MidasBot(config)
    control_terminal(bot)