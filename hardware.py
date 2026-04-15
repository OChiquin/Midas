# hardware.py
# Lee la información del hardware de la PC para generar una huella única

import uuid
import hashlib
import platform
import psutil
import subprocess

def obtener_id_cpu():
    """Obtiene el identificador único del procesador"""
    try:
        # Comando para obtener el ID del CPU en Windows
        resultado = subprocess.check_output(
            'wmic cpu get ProcessorId',
            shell=True
        ).decode()
        # Limpia el resultado y devuelve solo el ID
        lineas = [l.strip() for l in resultado.strip().split('\n') if l.strip()]
        return lineas[-1] if len(lineas) > 1 else "CPU_DESCONOCIDO"
    except:
        return "CPU_DESCONOCIDO"

def obtener_mac():
    """Obtiene la dirección MAC de la tarjeta de red principal"""
    try:
        mac = uuid.getnode()
        # Convierte el número a formato XX:XX:XX:XX:XX:XX
        return ':'.join(('%012X' % mac)[i:i+2] for i in range(0, 12, 2))
    except:
        return "MAC_DESCONOCIDA"

def obtener_serial_disco():
    """Obtiene el número de serie del disco duro principal"""
    try:
        resultado = subprocess.check_output(
            'wmic diskdrive get SerialNumber',
            shell=True
        ).decode()
        lineas = [l.strip() for l in resultado.strip().split('\n') if l.strip()]
        return lineas[-1] if len(lineas) > 1 else "DISCO_DESCONOCIDO"
    except:
        return "DISCO_DESCONOCIDO"

def generar_huella():
    """
    Combina CPU + MAC + Disco y genera un hash único de 64 caracteres
    Este hash es la identidad de esta PC
    """
    cpu = obtener_id_cpu()
    mac = obtener_mac()
    disco = obtener_serial_disco()

    # Une los tres datos con un separador
    datos_combinados = f"{cpu}|{mac}|{disco}"

    # Genera el hash SHA-256
    huella = hashlib.sha256(datos_combinados.encode()).hexdigest()

    return huella

def mostrar_info_hardware():
    """Muestra la información del hardware en pantalla (para diagnóstico)"""
    print("=" * 50)
    print("   INFORMACIÓN DE HARDWARE DETECTADA")
    print("=" * 50)
    print(f"  CPU ID    : {obtener_id_cpu()}")
    print(f"  MAC       : {obtener_mac()}")
    print(f"  Disco     : {obtener_serial_disco()}")
    print(f"  Sistema   : {platform.system()} {platform.release()}")
    print("=" * 50)
    huella = generar_huella()
    print(f"  Huella    : {huella[:16]}...{huella[-16:]}")
    print("=" * 50)
    return huella

# Si ejecutas este archivo directamente, muestra la info
if __name__ == "__main__":
    mostrar_info_hardware()