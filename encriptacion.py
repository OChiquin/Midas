# encriptacion.py
# Encripta y desencripta archivos usando la huella del hardware como llave

import os
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from hardware import generar_huella

# Sal fija del proyecto Midas (no es secreta, solo hace más robusto el hash)
SALT = b'MidasTradingBot2024'

def generar_llave_desde_huella(huella: str) -> bytes:
    """
    Convierte la huella de hardware en una llave de encriptación válida
    La misma huella siempre genera la misma llave
    """
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=SALT,
        iterations=100000,
    )
    llave = base64.urlsafe_b64encode(
        kdf.derive(huella.encode())
    )
    return llave

def obtener_fernet() -> Fernet:
    """Crea el objeto Fernet con la llave de esta PC"""
    huella = generar_huella()
    llave = generar_llave_desde_huella(huella)
    return Fernet(llave)

def encriptar_archivo(ruta_archivo: str):
    """
    Encripta un archivo en disco
    El archivo original es reemplazado por la versión encriptada
    """
    if not os.path.exists(ruta_archivo):
        print(f"  ❌ Archivo no encontrado: {ruta_archivo}")
        return False
    
    try:
        fernet = obtener_fernet()
        
        # Lee el contenido original
        with open(ruta_archivo, 'rb') as f:
            contenido = f.read()
        
        # Encripta el contenido
        contenido_encriptado = fernet.encrypt(contenido)
        
        # Sobreescribe el archivo con la versión encriptada
        with open(ruta_archivo, 'wb') as f:
            f.write(contenido_encriptado)
        
        print(f"  ✅ Encriptado: {ruta_archivo}")
        return True
    except Exception as e:
        print(f"  ❌ Error al encriptar {ruta_archivo}: {e}")
        return False

def desencriptar_archivo(ruta_archivo: str) -> bytes:
    """
    Desencripta un archivo y devuelve su contenido en memoria
    NO sobreescribe el archivo en disco
    """
    if not os.path.exists(ruta_archivo):
        print(f"  ❌ Archivo no encontrado: {ruta_archivo}")
        return None
    
    try:
        fernet = obtener_fernet()
        
        with open(ruta_archivo, 'rb') as f:
            contenido_encriptado = f.read()
        
        contenido = fernet.decrypt(contenido_encriptado)
        return contenido
    except Exception as e:
        print(f"  ❌ Error al desencriptar: hardware no autorizado o archivo corrupto")
        return None

def encriptar_texto(texto: str) -> str:
    """Encripta un texto y devuelve el resultado como string"""
    fernet = obtener_fernet()
    return fernet.encrypt(texto.encode()).decode()

def desencriptar_texto(texto_encriptado: str) -> str:
    """Desencripta un texto encriptado y devuelve el original"""
    try:
        fernet = obtener_fernet()
        return fernet.decrypt(texto_encriptado.encode()).decode()
    except:
        return None

# Prueba si ejecutas directamente
if __name__ == "__main__":
    print("Probando sistema de encriptación...")
    
    # Crea un archivo de prueba
    with open("prueba_temp.txt", "w") as f:
        f.write("Este es un mensaje secreto del Proyecto Midas")
    
    print("Antes de encriptar:")
    with open("prueba_temp.txt", "r") as f:
        print(f"  {f.read()}")
    
    # Encripta
    encriptar_archivo("prueba_temp.txt")
    
    print("Después de encriptar (contenido ilegible):")
    with open("prueba_temp.txt", "rb") as f:
        print(f"  {f.read()[:50]}...")
    
    # Desencripta en memoria
    contenido = desencriptar_archivo("prueba_temp.txt")
    print("Desencriptado en memoria:")
    print(f"  {contenido.decode()}")
    
    # Limpia el archivo de prueba
    os.remove("prueba_temp.txt")
    print("\n✅ Sistema de encriptación funcionando correctamente")