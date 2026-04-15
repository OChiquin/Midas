# interfaz.py
# Panel gráfico del Proyecto Midas con tkinter
# Proyecto Midas — Parte 4

import sys
import os
import json
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from bot import MidasBot, CONFIG_DEFAULT

# ─────────────────────────────────────────────
# CONFIGURACIÓN VISUAL
# ─────────────────────────────────────────────

COLOR_FONDO      = "#0d1117"
COLOR_PANEL      = "#161b22"
COLOR_BORDE      = "#30363d"
COLOR_TEXTO      = "#e6edf3"
COLOR_VERDE      = "#3fb950"
COLOR_ROJO       = "#f85149"
COLOR_AMARILLO   = "#d29922"
COLOR_AZUL       = "#58a6ff"
COLOR_GRIS       = "#8b949e"
FUENTE_TITULO    = ("Consolas", 14, "bold")
FUENTE_GRANDE    = ("Consolas", 20, "bold")
FUENTE_NORMAL    = ("Consolas", 10)
FUENTE_PEQUENA   = ("Consolas", 9)

# ─────────────────────────────────────────────
# VENTANA PRINCIPAL
# ─────────────────────────────────────────────

class InterfazMidas:
    """Panel gráfico del Proyecto Midas"""

    def __init__(self):
        # Carga configuración
        config = CONFIG_DEFAULT.copy()
        archivo_config = os.path.join('licencia', 'config_bot.json')
        if os.path.exists(archivo_config):
            with open(archivo_config, 'r') as f:
                config.update(json.load(f))

        # Crea el bot
        self.bot = MidasBot(config)

        # Ventana principal
        self.ventana = tk.Tk()
        self.ventana.title("Proyecto Midas — Panel de Control")
        self.ventana.configure(bg=COLOR_FONDO)
        self.ventana.geometry("800x620")
        self.ventana.resizable(False, False)

        # Construye la interfaz
        self._construir_ui()

        # Inicia el bucle de actualización
        self._actualizar()

        # Maneja el cierre de la ventana
        self.ventana.protocol("WM_DELETE_WINDOW", self._cerrar)

    def _construir_ui(self):
        """Construye todos los elementos de la interfaz"""

        # ── Título ──
        frame_titulo = tk.Frame(self.ventana, bg=COLOR_FONDO)
        frame_titulo.pack(fill='x', padx=15, pady=(15, 5))

        tk.Label(
            frame_titulo, text="🏆 PROYECTO MIDAS",
            font=FUENTE_TITULO, bg=COLOR_FONDO, fg=COLOR_AZUL
        ).pack(side='left')

        self.lbl_hora = tk.Label(
            frame_titulo, text="",
            font=FUENTE_NORMAL, bg=COLOR_FONDO, fg=COLOR_GRIS
        )
        self.lbl_hora.pack(side='right')

        # ── Panel de estado ──
        frame_estado = tk.Frame(
            self.ventana, bg=COLOR_PANEL,
            highlightbackground=COLOR_BORDE, highlightthickness=1
        )
        frame_estado.pack(fill='x', padx=15, pady=5)

        # Fila 1: Modo y conexiones
        fila1 = tk.Frame(frame_estado, bg=COLOR_PANEL)
        fila1.pack(fill='x', padx=10, pady=(10, 5))

        tk.Label(fila1, text="MODO:", font=FUENTE_NORMAL,
                 bg=COLOR_PANEL, fg=COLOR_GRIS).pack(side='left')
        self.lbl_modo = tk.Label(fila1, text="DETENIDO",
                                  font=FUENTE_NORMAL, bg=COLOR_PANEL, fg=COLOR_ROJO)
        self.lbl_modo.pack(side='left', padx=(5, 20))

        tk.Label(fila1, text="MT5:", font=FUENTE_NORMAL,
                 bg=COLOR_PANEL, fg=COLOR_GRIS).pack(side='left')
        self.lbl_mt5 = tk.Label(fila1, text="●", font=FUENTE_NORMAL,
                                  bg=COLOR_PANEL, fg=COLOR_ROJO)
        self.lbl_mt5.pack(side='left', padx=(5, 20))

        tk.Label(fila1, text="API:", font=FUENTE_NORMAL,
                 bg=COLOR_PANEL, fg=COLOR_GRIS).pack(side='left')
        self.lbl_api = tk.Label(fila1, text="●", font=FUENTE_NORMAL,
                                  bg=COLOR_PANEL, fg=COLOR_ROJO)
        self.lbl_api.pack(side='left', padx=5)

        # Fila 2: Balances
        fila2 = tk.Frame(frame_estado, bg=COLOR_PANEL)
        fila2.pack(fill='x', padx=10, pady=5)

        self._campo(fila2, "Balance inicial:", "lbl_balance_inicial", "$0.00")
        self._campo(fila2, "Balance actual:", "lbl_balance_actual", "$0.00")
        self._campo(fila2, "Crecimiento total:", "lbl_crecimiento", "+0.00%")

        # Fila 3: Día
        fila3 = tk.Frame(frame_estado, bg=COLOR_PANEL)
        fila3.pack(fill='x', padx=10, pady=5)

        self._campo(fila3, "Ganancia hoy:", "lbl_ganancia_dia", "$0.00")
        self._campo(fila3, "% del día:", "lbl_pct_dia", "+0.00%")
        self._campo(fila3, "Proyección 30d:", "lbl_proyeccion", "$0.00")

        # Fila 4: Límites
        fila4 = tk.Frame(frame_estado, bg=COLOR_PANEL)
        fila4.pack(fill='x', padx=10, pady=(5, 10))

        self._campo(fila4, "Meta:", "lbl_meta", "+2.0%")
        self._campo(fila4, "Límite pérdida:", "lbl_limite", "-1.0%")
        self._campo(fila4, "Techo:", "lbl_techo", "+5.0%")

        # ── Último trade ──
        frame_trade = tk.Frame(
            self.ventana, bg=COLOR_PANEL,
            highlightbackground=COLOR_BORDE, highlightthickness=1
        )
        frame_trade.pack(fill='x', padx=15, pady=5)

        fila_trade = tk.Frame(frame_trade, bg=COLOR_PANEL)
        fila_trade.pack(fill='x', padx=10, pady=8)

        tk.Label(fila_trade, text="Último activo:",
                 font=FUENTE_NORMAL, bg=COLOR_PANEL, fg=COLOR_GRIS).pack(side='left')
        self.lbl_ultimo_activo = tk.Label(
            fila_trade, text="—", font=FUENTE_NORMAL, bg=COLOR_PANEL, fg=COLOR_TEXTO
        )
        self.lbl_ultimo_activo.pack(side='left', padx=(5, 20))

        tk.Label(fila_trade, text="Resultado:",
                 font=FUENTE_NORMAL, bg=COLOR_PANEL, fg=COLOR_GRIS).pack(side='left')
        self.lbl_ultimo_resultado = tk.Label(
            fila_trade, text="—", font=FUENTE_NORMAL, bg=COLOR_PANEL, fg=COLOR_TEXTO
        )
        self.lbl_ultimo_resultado.pack(side='left', padx=5)

        tk.Label(fila_trade, text="Posiciones abiertas:",
                 font=FUENTE_NORMAL, bg=COLOR_PANEL, fg=COLOR_GRIS).pack(side='left', padx=(20, 0))
        self.lbl_posiciones = tk.Label(
            fila_trade, text="0", font=FUENTE_NORMAL, bg=COLOR_PANEL, fg=COLOR_TEXTO
        )
        self.lbl_posiciones.pack(side='left', padx=5)

        # ── Log de actividad ──
        frame_log = tk.Frame(
            self.ventana, bg=COLOR_PANEL,
            highlightbackground=COLOR_BORDE, highlightthickness=1
        )
        frame_log.pack(fill='both', expand=True, padx=15, pady=5)

        tk.Label(frame_log, text="Actividad reciente",
                 font=FUENTE_PEQUENA, bg=COLOR_PANEL, fg=COLOR_GRIS).pack(anchor='w', padx=10, pady=(5, 0))

        self.texto_log = tk.Text(
            frame_log, height=8, bg=COLOR_FONDO, fg=COLOR_TEXTO,
            font=FUENTE_PEQUENA, relief='flat', state='disabled',
            wrap='word'
        )
        self.texto_log.pack(fill='both', expand=True, padx=10, pady=5)

        # ── Botones de control ──
        frame_botones = tk.Frame(self.ventana, bg=COLOR_FONDO)
        frame_botones.pack(fill='x', padx=15, pady=(5, 15))

        botones = [
            ("START",        COLOR_VERDE,    self._cmd_start),
            ("PAUSE",        COLOR_AMARILLO, self._cmd_pause),
            ("STOP",         COLOR_ROJO,     self._cmd_stop),
            ("RESET",        COLOR_AZUL,     self._cmd_reset),
            ("GAMING",       COLOR_GRIS,     self._cmd_gaming),
            ("MAINTENANCE",  COLOR_GRIS,     self._cmd_mantenimiento),
        ]

        for texto, color, comando in botones:
            tk.Button(
                frame_botones, text=texto, command=comando,
                bg=color, fg='white', font=FUENTE_PEQUENA,
                relief='flat', padx=10, pady=5, cursor='hand2'
            ).pack(side='left', padx=3)

    def _campo(self, frame, etiqueta, nombre_var, valor_default):
        """Crea un campo de etiqueta + valor"""
        sub = tk.Frame(frame, bg=COLOR_PANEL)
        sub.pack(side='left', padx=10)
        tk.Label(sub, text=etiqueta, font=FUENTE_PEQUENA,
                 bg=COLOR_PANEL, fg=COLOR_GRIS).pack(anchor='w')
        lbl = tk.Label(sub, text=valor_default, font=FUENTE_NORMAL,
                       bg=COLOR_PANEL, fg=COLOR_TEXTO)
        lbl.pack(anchor='w')
        setattr(self, nombre_var, lbl)

    # ─────────────────────────────────────────────
    # ACTUALIZACIÓN DEL PANEL
    # ─────────────────────────────────────────────

    def _actualizar(self):
        """Actualiza el panel cada 2 segundos"""
        try:
            self.lbl_hora.config(
                text=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )

            if self.bot.corriendo:
                estado = self.bot.obtener_estado()

                # Modo
                modo  = estado['modo'].upper()
                color = {
                    'NORMAL'         : COLOR_VERDE,
                    'CAUTELOSO'      : COLOR_AMARILLO,
                    'META_ALCANZADA' : COLOR_AZUL,
                    'LIMITE_PERDIDA' : COLOR_ROJO,
                }.get(modo, COLOR_GRIS)
                self.lbl_modo.config(text=modo, fg=color)

                # Conexiones
                self.lbl_mt5.config(fg=COLOR_VERDE if estado['corriendo'] else COLOR_ROJO)
                self.lbl_api.config(fg=COLOR_VERDE if estado['internet_ok'] else COLOR_ROJO)

                # Balances
                self.lbl_balance_inicial.config(
                    text=f"${estado['balance_inicial']:.2f}"
                )
                self.lbl_balance_actual.config(
                    text=f"${estado['balance_actual']:.2f}"
                )
                pct_total = estado['pct_total']
                self.lbl_crecimiento.config(
                    text=f"{pct_total:+.2f}%",
                    fg=COLOR_VERDE if pct_total >= 0 else COLOR_ROJO
                )

                # Día
                ganancia = estado['ganancia_dia']
                pct_dia  = estado['pct_dia']
                self.lbl_ganancia_dia.config(
                    text=f"${ganancia:+.2f}",
                    fg=COLOR_VERDE if ganancia >= 0 else COLOR_ROJO
                )
                self.lbl_pct_dia.config(
                    text=f"{pct_dia:+.2f}%",
                    fg=COLOR_VERDE if pct_dia >= 0 else COLOR_ROJO
                )
                self.lbl_proyeccion.config(
                    text=f"${estado['proyeccion_30d']:.2f}"
                )

                # Límites
                self.lbl_meta.config(text=f"+{estado['meta_pct']}%")
                self.lbl_limite.config(text=f"-{estado['limite_pct']}%")
                self.lbl_techo.config(text=f"+{estado['techo_pct']}%")

                # Último trade
                if estado['ultimo_activo']:
                    self.lbl_ultimo_activo.config(text=estado['ultimo_activo'])
                if estado['ultimo_resultado'] is not None:
                    res = estado['ultimo_resultado']
                    self.lbl_ultimo_resultado.config(
                        text=f"${res:+.2f}",
                        fg=COLOR_VERDE if res >= 0 else COLOR_ROJO
                    )

                # Posiciones
                posiciones = sum(1 for v in estado['posiciones'].values() if v)
                self.lbl_posiciones.config(text=str(posiciones))

        except Exception as e:
            pass

        # Repite cada 2 segundos
        self.ventana.after(2000, self._actualizar)

    def _log(self, mensaje: str):
        """Agrega un mensaje al log de la interfaz"""
        self.texto_log.config(state='normal')
        hora = datetime.now().strftime("%H:%M:%S")
        self.texto_log.insert('end', f"[{hora}] {mensaje}\n")
        self.texto_log.see('end')
        self.texto_log.config(state='disabled')

    # ─────────────────────────────────────────────
    # COMANDOS DE LOS BOTONES
    # ─────────────────────────────────────────────

    def _cmd_start(self):
        self.bot.iniciar()
        self.lbl_modo.config(text="NORMAL", fg=COLOR_VERDE)
        self._log("Bot iniciado")

    def _cmd_pause(self):
        if self.bot.pausado:
            self.bot.reanudar()
            self._log("Bot reanudado")
        else:
            self.bot.pausar()
            self._log("Bot pausado")

    def _cmd_stop(self):
        if messagebox.askyesno("Detener", "¿Detener el bot?"):
            self.bot.detener()
            self.lbl_modo.config(text="DETENIDO", fg=COLOR_ROJO)
            self._log("Bot detenido")

    def _cmd_reset(self):
        import MetaTrader5 as mt5
        info = mt5.account_info()
        if info:
            self.bot.gestor.reiniciar_contadores(info.balance)
            self._log(f"Contadores reiniciados — nuevo balance: ${info.balance:.2f}")

    def _cmd_gaming(self):
        self.bot.activar_modo_juego(not self.bot.modo_juego)
        estado = "ON" if self.bot.modo_juego else "OFF"
        self._log(f"Modo juego {estado}")

    def _cmd_mantenimiento(self):
        self.bot.activar_mantenimiento(not self.bot.modo_mantenimiento)
        estado = "ON" if self.bot.modo_mantenimiento else "OFF"
        self._log(f"Modo mantenimiento {estado}")

    def _cerrar(self):
        """Cierra la interfaz ordenadamente"""
        if messagebox.askyesno("Salir", "¿Cerrar el panel? El bot se detendrá."):
            self.bot.detener()
            self.ventana.destroy()

    # ─────────────────────────────────────────────
    # ARRANQUE
    # ─────────────────────────────────────────────

    def iniciar(self):
        """Inicia la ventana gráfica"""
        self.ventana.mainloop()

# ─────────────────────────────────────────────
# PUNTO DE ENTRADA
# ─────────────────────────────────────────────

if __name__ == "__main__":
    app = InterfazMidas()
    app.iniciar()