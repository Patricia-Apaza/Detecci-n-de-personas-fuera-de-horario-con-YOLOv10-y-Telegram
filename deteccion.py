import cv2
import requests
import warnings
import logging
import os
from datetime import datetime
from ultralytics import YOLO
from ultralytics.utils import LOGGER as ultralytics_logger

# Suprimir warnings del modulo warnings de Python
warnings.filterwarnings("ignore", message=".*not enough matching points.*")

# Suprimir el WARNING del logger interno de ultralytics
ultralytics_logger.setLevel(logging.ERROR)

# 1. CONFIGURACION
TELEGRAM_TOKEN  = "8191050704:AAGq946Jqw6Ntk7ZhQw-BJgY1ZCiMR3bVeU"
TELEGRAM_CHAT_ID = "-5332802869"   # grupo "Alarma Residencial"

HORA_INICIO_TOQUE_QUEDA = 22      # 10 pm
HORA_FIN_TOQUE_QUEDA    = 4       # 4 am del dia siguiente

CONFIANZA_MINIMA = 0.5
INDICE_CAMARA    = 0
CARPETA_CAPTURAS = "capturas"

# Cuantos frames consecutivos debe aparecer un ID nuevo
# antes de considerarlo una deteccion real (evita IDs fantasma de 1 frame)
FRAMES_CONFIRMACION = 5

# 2. CARGAR MODELO
print("Cargando modelo YOLOv10...")
modelo = YOLO("yolov10n.pt")
print("Modelo cargado correctamente.")
print(f"Toque de queda: {HORA_INICIO_TOQUE_QUEDA}:00 - {HORA_FIN_TOQUE_QUEDA}:00")

CLASE_PERSONA = 0

# 3. FUNCIONES AUXILIARES
def es_horario_toque_queda(hora_actual):
    hora = hora_actual.hour
    if HORA_INICIO_TOQUE_QUEDA > HORA_FIN_TOQUE_QUEDA:
        return hora >= HORA_INICIO_TOQUE_QUEDA or hora < HORA_FIN_TOQUE_QUEDA
    return HORA_INICIO_TOQUE_QUEDA <= hora < HORA_FIN_TOQUE_QUEDA


def enviar_alerta_telegram(ruta_imagen, mensaje):
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
            print("Alerta enviada al grupo de Telegram correctamente.")
        else:
            print(f"Error Telegram: {respuesta.status_code} - {respuesta.text}")
    except Exception as e:
        print(f"Excepcion al enviar a Telegram: {e}")

# 4. PROGRAMA PRINCIPAL
def main():
    os.makedirs(CARPETA_CAPTURAS, exist_ok=True)

    captura = cv2.VideoCapture(INDICE_CAMARA)
    if not captura.isOpened():
        print("ERROR: No se pudo abrir la camara.")
        return

    print("Camara iniciada. Presiona 'q' para salir.")

    # IDs que ya generaron una alerta
    ids_alertados = set()

    # Contador de frames en que cada ID ha sido visto (para confirmar que es real y no un ID fantasma)
    conteo_frames = {}   # {track_id: cantidad_de_frames_visto}

    # Conjunto de IDs vistos en el frame anterior
    ids_frame_anterior = set()

    while True:
        ret, frame = captura.read()
        if not ret:
            print("No se pudo leer el frame.")
            break

        ahora = datetime.now()
        en_toque_de_queda = es_horario_toque_queda(ahora)

        #Deteccion + tracking
        resultados = modelo.track(frame, persist=True, verbose=False)[0]

        # IDs de personas validas detectadas en ESTE frame
        ids_este_frame = set()

        for caja in resultados.boxes:
            if int(caja.cls[0]) != CLASE_PERSONA:
                continue
            if float(caja.conf[0]) < CONFIANZA_MINIMA:
                continue
            if caja.id is None:
                continue

            track_id = int(caja.id[0])
            ids_este_frame.add(track_id)

            # Acumular conteo de frames para este ID
            conteo_frames[track_id] = conteo_frames.get(track_id, 0) + 1

            # Dibujar caja
            x1, y1, x2, y2 = map(int, caja.xyxy[0])
            confianza = float(caja.conf[0])
            color = (0, 0, 255) if en_toque_de_queda else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID:{track_id} {confianza:.2f}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Limpiar conteo de IDs que ya no estan en camara
        ids_desaparecidos = set(conteo_frames.keys()) - ids_este_frame
        for tid in ids_desaparecidos:
            conteo_frames.pop(tid, None)

        # IDs confirmados (vistos al menos FRAMES_CONFIRMACION frames seguidos)
        ids_confirmados = {
            tid for tid in ids_este_frame
            if conteo_frames.get(tid, 0) >= FRAMES_CONFIRMACION
        }

        # IDs nuevos confirmados que no habian sido alertados antes
        ids_nuevos = ids_confirmados - ids_alertados

        # Logica de alerta
        # Solo alertar si hay personas nuevas Y estamos en toque de queda
        if ids_nuevos and en_toque_de_queda:
            total_personas = len(ids_confirmados)

            nombre_archivo = (
                f"{CARPETA_CAPTURAS}/alerta_"
                f"{ahora.strftime('%Y%m%d_%H%M%S')}"
                f"_{total_personas}personas.jpg"
            )
            cv2.imwrite(nombre_archivo, frame)

            ids_nuevos_str = ", ".join(str(i) for i in sorted(ids_nuevos))
            mensaje = (
                f"ALERTA - Residentes fuera de horario\n"
                f"Fecha y hora: {ahora.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Personas en camara: {total_personas}\n"
                f"ID(s) nuevo(s) detectado(s): {ids_nuevos_str}"
            )

            enviar_alerta_telegram(nombre_archivo, mensaje)

            # Marcar todos los IDs nuevos como ya alertados
            ids_alertados.update(ids_nuevos)

        ids_frame_anterior = ids_este_frame.copy()

        # Texto de estado en pantalla
        estado = "TOQUE DE QUEDA" if en_toque_de_queda else "Horario libre"
        color_estado = (0, 0, 255) if en_toque_de_queda else (0, 255, 0)
        cv2.putText(frame, f"{ahora.strftime('%H:%M:%S')} - {estado}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_estado, 2)

        personas_en_camara = len(ids_confirmados)
        if personas_en_camara > 0:
            cv2.putText(frame, f"Personas: {personas_en_camara}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_estado, 2)

        cv2.imshow("Alarma Residencial - YOLOv10", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    captura.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()