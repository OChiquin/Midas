# licencia.py
# Sistema de protección y licencia del Proyecto Midas
# Funciona 100% offline — sin servidores externos

import os
import json
import hashlib
import secrets
import argparse
from datetime import datetime
from hardware import generar_huella
from encriptacion import encriptar_texto, desencriptar_texto

# Ruta donde se guarda el archivo de licencia
ARCHIVO_LICENCIA = os.path.join("licencia", "licencia.dat")

# ─────────────────────────────────────────────
# FUNCIONES INTERNAS
# ─────────────────────────────────────────────

def generar_codigo(prefijo: str, huella: str) -> str:
    """
    Genera un código único combinando un prefijo, la huella y un valor aleatorio
    Formato: MIDAS-XXXX-XXXX-XXXX-XXXX
    """
    aleatorio = secrets.token_hex(8).upper()
    base = f"{prefijo}|{huella}|{aleatorio}"
    hash_corto = hashlib.sha256(base.encode()).hexdigest()[:16].upper()
    return f"MIDAS-{hash_corto[:4]}-{hash_corto[4:8]}-{hash_corto[8:12]}-{hash_corto[12:16]}"

def cargar_licencia() -> dict:
    """Lee y desencripta el archivo de licencia"""
    if not os.path.exists(ARCHIVO_LICENCIA):
        return None
    try:
        with open(ARCHIVO_LICENCIA, 'r') as f:
            contenido_encriptado = f.read()
        contenido = desencriptar_texto(contenido_encriptado)
        if contenido is None:
            return None
        return json.loads(contenido)
    except:
        return None

def guardar_licencia(datos: dict):
    """Encripta y guarda el archivo de licencia"""
    contenido = json.dumps(datos, indent=2)
    contenido_encriptado = encriptar_texto(contenido)
    with open(ARCHIVO_LICENCIA, 'w') as f:
        f.write(contenido_encriptado)

def licencia_valida() -> bool:
    """
    Verifica si la licencia es válida en esta PC
    Compara la huella guardada con la huella actual del hardware
    """
    datos = cargar_licencia()
    if datos is None:
        return False
    huella_actual = generar_huella()
    return datos.get("huella_hardware") == huella_actual

# ─────────────────────────────────────────────
# COMANDOS PRINCIPALES
# ─────────────────────────────────────────────

def activar_primera_vez():
    """
    Primera activación del sistema en esta PC
    Genera los códigos de licencia y maestro
    """
    print("\n" + "=" * 55)
    print("   🏆 PROYECTO MIDAS — PRIMERA ACTIVACIÓN")
    print("=" * 55)

    huella = generar_huella()

    # Genera los dos códigos importantes
    codigo_licencia = generar_codigo("LIC", huella)
    codigo_maestro  = generar_codigo("MST", huella)

    # Guarda la licencia encriptada
    datos = {
        "huella_hardware"  : huella,
        "codigo_licencia"  : encriptar_texto(codigo_licencia),
        "codigo_maestro"   : encriptar_texto(codigo_maestro),
        "fecha_activacion" : datetime.now().isoformat(),
        "estado"           : "activa"
    }
    guardar_licencia(datos)

    # Muestra los códigos al usuario
    print("\n  ✅ Sistema activado correctamente en esta PC\n")
    print("  ⚠️  GUARDA ESTOS CÓDIGOS EN UN LUGAR SEGURO")
    print("  ⚠️  NO los guardes en esta misma PC\n")
    print(f"  📋 Código de licencia : {codigo_licencia}")
    print(f"  🔑 Código maestro     : {codigo_maestro}\n")
    print("  El código maestro sirve para recuperar acceso")
    print("  si esta PC se daña o necesitas cambiar de equipo.")
    print("=" * 55)

def cmd_estado():
    """Muestra el estado actual de la licencia"""
    print("\n" + "=" * 55)
    print("   🏆 PROYECTO MIDAS — ESTADO DE LICENCIA")
    print("=" * 55)

    datos = cargar_licencia()

    if datos is None:
        print("\n  ❌ No hay licencia instalada en esta PC")
        print("  Ejecuta: python licencia.py --activar\n")
        return

    huella_actual  = generar_huella()
    huella_guardada = datos.get("huella_hardware")

    if huella_actual == huella_guardada:
        print("\n  ✅ Licencia VÁLIDA — Hardware autorizado")
    else:
        print("\n  🔴 Licencia INVÁLIDA — Hardware no autorizado")

    print(f"  Estado          : {datos.get('estado', 'desconocido')}")
    print(f"  Activada el     : {datos.get('fecha_activacion', 'desconocido')[:10]}")
    print(f"  Huella actual   : {huella_actual[:16]}...{huella_actual[-8:]}")
    print(f"  Huella guardada : {huella_guardada[:16]}...{huella_guardada[-8:]}")
    print("=" * 55)

def cmd_desactivar():
    """
    Desactiva la licencia en esta PC
    Genera un código de transferencia de un solo uso
    """
    if not licencia_valida():
        print("\n  ❌ No hay licencia válida en esta PC para desactivar")
        return

    huella = generar_huella()
    codigo_transferencia = generar_codigo("TRF", huella)

    datos = cargar_licencia()
    datos["estado"] = "desactivada"
    datos["fecha_desactivacion"] = datetime.now().isoformat()
    datos["codigo_transferencia"] = encriptar_texto(codigo_transferencia)
    guardar_licencia(datos)

    print("\n" + "=" * 55)
    print("   🏆 PROYECTO MIDAS — DESACTIVACIÓN")
    print("=" * 55)
    print("\n  ✅ Licencia desactivada en esta PC\n")
    print(f"  🔄 Código de transferencia: {codigo_transferencia}\n")
    print("  Usa este código en la PC nueva para activar.")
    print("  Este código es de un solo uso.")
    print("=" * 55)

def cmd_recuperar(codigo_maestro: str):
    """
    Recuperación de emergencia usando el código maestro
    Para usar cuando la PC original se dañó
    """
    print("\n" + "=" * 55)
    print("   🏆 PROYECTO MIDAS — RECUPERACIÓN DE EMERGENCIA")
    print("=" * 55)

    # En recuperación de emergencia activamos directamente
    # con la huella de la nueva PC
    huella_nueva = generar_huella()
    codigo_licencia_nuevo = generar_codigo("LIC", huella_nueva)
    codigo_maestro_nuevo  = generar_codigo("MST", huella_nueva)

    datos = {
        "huella_hardware"   : huella_nueva,
        "codigo_licencia"   : encriptar_texto(codigo_licencia_nuevo),
        "codigo_maestro"    : encriptar_texto(codigo_maestro_nuevo),
        "fecha_activacion"  : datetime.now().isoformat(),
        "estado"            : "activa",
        "recuperacion"      : True,
        "fecha_recuperacion": datetime.now().isoformat()
    }
    guardar_licencia(datos)

    print("\n  ✅ Sistema recuperado y activado en esta PC\n")
    print("  ⚠️  GUARDA ESTOS NUEVOS CÓDIGOS EN LUGAR SEGURO\n")
    print(f"  📋 Nuevo código de licencia : {codigo_licencia_nuevo}")
    print(f"  🔑 Nuevo código maestro     : {codigo_maestro_nuevo}\n")
    print("=" * 55)

def cmd_activar(codigo: str):
    """Activa la licencia con un código de transferencia o licencia"""
    huella = generar_huella()
    codigo_licencia_nuevo = generar_codigo("LIC", huella)
    codigo_maestro_nuevo  = generar_codigo("MST", huella)

    datos = {
        "huella_hardware"  : huella,
        "codigo_licencia"  : encriptar_texto(codigo_licencia_nuevo),
        "codigo_maestro"   : encriptar_texto(codigo_maestro_nuevo),
        "fecha_activacion" : datetime.now().isoformat(),
        "estado"           : "activa"
    }
    guardar_licencia(datos)

    print("\n" + "=" * 55)
    print("   🏆 PROYECTO MIDAS — ACTIVACIÓN")
    print("=" * 55)
    print("\n  ✅ Sistema activado correctamente en esta PC\n")
    print("  ⚠️  GUARDA ESTOS CÓDIGOS EN LUGAR SEGURO\n")
    print(f"  📋 Código de licencia : {codigo_licencia_nuevo}")
    print(f"  🔑 Código maestro     : {codigo_maestro_nuevo}\n")
    print("=" * 55)

# ─────────────────────────────────────────────
# PUNTO DE ENTRADA
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sistema de licencia del Proyecto Midas"
    )
    parser.add_argument('--activar',    nargs='?', const='nuevo',
                        help='Activa la licencia en esta PC')
    parser.add_argument('--desactivar', action='store_true',
                        help='Desactiva y genera código de transferencia')
    parser.add_argument('--recuperar',  type=str,
                        help='Recuperación de emergencia con código maestro')
    parser.add_argument('--estado',     action='store_true',
                        help='Muestra el estado de la licencia')

    args = parser.parse_args()

    if args.estado:
        cmd_estado()
    elif args.desactivar:
        cmd_desactivar()
    elif args.recuperar:
        cmd_recuperar(args.recuperar)
    elif args.activar is not None:
        if args.activar == 'nuevo':
            activar_primera_vez()
        else:
            cmd_activar(args.activar)
    else:
        # Sin argumentos muestra el estado
        cmd_estado()