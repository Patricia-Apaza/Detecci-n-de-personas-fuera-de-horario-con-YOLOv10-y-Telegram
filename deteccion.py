import cv2
import time
import requests
from datetime import datetime
from ultralytics import YOLO

# 1. CONFIGURACION - EDITAR ESTOS VALORES

# Datos de tu bot de Telegram
TELEGRAM_TOKEN = "8191050704:AAGq946Jqw6Ntk7ZhQw-BJgY1ZCiMR3bVeU"
TELEGRAM_CHAT_ID = "6167866010"

# Horario de toque de queda (formato 24 horas)
HORA_INICIO_TOQUE_QUEDA = 22   # 10 pm
HORA_FIN_TOQUE_QUEDA = 4       # 4 am del dia siguiente

# Cada cuantos segundos como minimo se puede enviar una alerta (para no inundar el chat de Telegram con fotos repetidas)
COOLDOWN_SEGUNDOS = 30

# Confianza minima para considerar una deteccion valida (0 a 1)
CONFIANZA_MINIMA = 0.5

# Indice de la camara (0 = camara principal de la laptop)
INDICE_CAMARA = 0

# Carpeta donde se guardan las capturas de las alertas
CARPETA_CAPTURAS = "capturas"

# 2. CARGAR EL MODELO YOLOv10

print("Cargando modelo YOLOv10...")
modelo = YOLO("yolov10n.pt")
print("Modelo cargado correctamente.")

# La clase 0 en el dataset COCO (con el que viene entrenado YOLO) es "person"
CLASE_PERSONA = 0


# 3. FUNCIONES AUXILIARES

def es_horario_toque_queda(hora_actual):
    """
    Devuelve True si la hora actual (objeto datetime.time o int de hora)
    esta dentro del horario de toque de queda.
    Maneja el caso especial en que el horario cruza la medianoche
    (ej: de 22 a 4, el rango "envuelve" el dia).
    """
    hora = hora_actual.hour

    if HORA_INICIO_TOQUE_QUEDA > HORA_FIN_TOQUE_QUEDA:
        # El rango cruza la medianoche, ej: 22 -> 4
        return hora >= HORA_INICIO_TOQUE_QUEDA or hora < HORA_FIN_TOQUE_QUEDA
    else:
        # Rango normal, no cruza medianoche
        return HORA_INICIO_TOQUE_QUEDA <= hora < HORA_FIN_TOQUE_QUEDA


def enviar_alerta_telegram(ruta_imagen, mensaje):
    """
    Envia una foto + texto al chat de Telegram configurado,
    usando la API HTTP de Telegram (sendPhoto).
    """
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"

    try:
        with open(ruta_imagen, "rb") as foto:
            respuesta = requests.post(
                url,
                data={"chat_id": TELEGRAM_CHAT_ID, "caption": mensaje},
                files={"photo": foto},
                timeout=10,
            )

        if respuesta.status_code == 200:
            print("Alerta enviada a Telegram correctamente.")
        else:
            print(f"Error al enviar a Telegram: {respuesta.status_code} - {respuesta.text}")

    except Exception as error:
        print(f"Excepcion al enviar a Telegram: {error}")


# 4. PROGRAMA PRINCIPAL

def main():
    import os
    os.makedirs(CARPETA_CAPTURAS, exist_ok=True)

    captura = cv2.VideoCapture(INDICE_CAMARA)

    if not captura.isOpened():
        print("ERROR: No se pudo abrir la camara. Revisa el indice de camara o si otra app la esta usando.")
        return

    print("Camara iniciada. Presiona 'q' en la ventana de video para salir.")

    ultimo_envio = 0  # marca de tiempo (timestamp) del ultimo envio a Telegram

    while True:
        ret, frame = captura.read()
        if not ret:
            print("No se pudo leer el frame de la camara.")
            break

        ahora = datetime.now()
        en_toque_de_queda = es_horario_toque_queda(ahora)

        # Corremos la deteccion de YOLO sobre el frame actual
        resultados = modelo(frame, verbose=False)[0]

        hay_persona = False

        # Recorremos cada deteccion del frame
        for caja in resultados.boxes:
            clase_detectada = int(caja.cls[0])
            confianza = float(caja.conf[0])

            if clase_detectada == CLASE_PERSONA and confianza >= CONFIANZA_MINIMA:
                hay_persona = True

                # Dibujamos el rectangulo y la etiqueta sobre el frame
                x1, y1, x2, y2 = map(int, caja.xyxy[0])
                color = (0, 0, 255) if en_toque_de_queda else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                texto = f"Persona {confianza:.2f}"
                cv2.putText(frame, texto, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Texto de estado en la esquina de la ventana
        estado_texto = "TOQUE DE QUEDA" if en_toque_de_queda else "Horario libre"
        color_estado = (0, 0, 255) if en_toque_de_queda else (0, 255, 0)
        cv2.putText(frame, f"{ahora.strftime('%H:%M:%S')} - {estado_texto}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_estado, 2)

        # Si hay persona Y es horario de toque de queda Y ya paso el cooldown
        tiempo_actual = time.time()
        if hay_persona and en_toque_de_queda and (tiempo_actual - ultimo_envio) > COOLDOWN_SEGUNDOS:
            nombre_archivo = f"{CARPETA_CAPTURAS}/alerta_{ahora.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(nombre_archivo, frame)

            mensaje = (
                f"ALERTA - Residente fuera de horario\n"
                f"Fecha y hora: {ahora.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Se detecto una persona durante el toque de queda."
            )

            enviar_alerta_telegram(nombre_archivo, mensaje)
            ultimo_envio = tiempo_actual

        # Mostramos la ventana con el video y las detecciones
        cv2.imshow("Alarma Residencial - YOLOv10", frame)

        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    captura.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
