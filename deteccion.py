import cv2
import requests
import warnings
import logging
import os
import threading
import time
import shutil
import json
from datetime import datetime, timedelta
from ultralytics import YOLO
from ultralytics.utils import LOGGER as ultralytics_logger

warnings.filterwarnings("ignore", message=".*not enough matching points.*")
ultralytics_logger.setLevel(logging.ERROR)

# 2. CONFIGURACIÓN DEL SISTEMA
TELEGRAM_TOKEN   = "8191050704:AAGq946Jqw6Ntk7ZhQw-BJgY1ZCiMR3bVeU"
TELEGRAM_CHAT_ID = "-1004353380145"

CONFIANZA_MINIMA    = 0.8
INDICE_CAMARA       = 0
CARPETA_CAPTURAS    = "capturas"
FRAMES_CONFIRMACION = 5
CLASE_PERSONA       = 0

# 3. ESTADO COMPARTIDO (PROCESAMIENTO MULTIHILO SEGURO)
estado = {
    "hora_inicio": 22,
    "hora_fin": 4,
    "total_alertas_sesion": 0
}
conversaciones = {}
lock_estado = threading.Lock()

print("Cargando arquitectura de Redes Convolucionales (YOLOv10)...")
modelo = YOLO("yolov10n.pt")
print("Modelo de IA para personas listo.")

# Cargar el detector de rostros nativo de OpenCV (Haar Cascade)
print("Cargando detector facial secundario...")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 4. CAPA DE ABSTRACCIÓN DE TIEMPO Y GESTIÓN DE TURNOS
def es_horario_toque_queda(hora_actual):
    with lock_estado:
        inicio = estado["hora_inicio"]
        fin    = estado["hora_fin"]
    hora = hora_actual.hour
    if inicio > fin:
        return hora >= inicio or hora < fin
    return inicio <= hora < fin

def obtener_nombre_turno(ahora):
    with lock_estado:
        inicio = estado["hora_inicio"]
        fin    = estado["hora_fin"]
        
    if inicio > fin:  # El turno cruza la medianoche
        if ahora.hour >= inicio:
            fecha_ini = ahora
            fecha_fin = ahora + timedelta(days=1)
        else:
            fecha_ini = ahora - timedelta(days=1)
            fecha_fin = ahora
            
        if fecha_ini.month == fecha_fin.month:
            return f"{fecha_ini.strftime('%Y-%m-%d')}_al_{fecha_fin.strftime('%d')}"
        else:
            return f"{fecha_ini.strftime('%Y-%m-%d')}_al_{fecha_fin.strftime('%m-%d')}"
    else:  # Turno lineal en el mismo día
        return ahora.strftime('%Y-%m-%d')

# 5. CLIENTE API DE TELEGRAM (MÉTODOS POST HTTP)
def enviar_mensaje_telegram(texto, chat_id=None, reply_markup=None):
    destino = str(chat_id) if chat_id else TELEGRAM_CHAT_ID
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": destino, "text": texto, "parse_mode": "Markdown"}
    if reply_markup:
        payload["reply_markup"] = json.dumps(reply_markup)
    try:
        requests.post(url, data=payload, timeout=10)
    except Exception as e:
        print(f"[API Error] Mensaje no enviado: {e}")

def enviar_alerta_foto(ruta_imagen, mensaje):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    try:
        with open(ruta_imagen, "rb") as foto:
            r = requests.post(
                url,
                data={"chat_id": TELEGRAM_CHAT_ID, "caption": mensaje, "parse_mode": "Markdown"},
                files={"photo": foto},
                timeout=15,
            )
        if r.status_code == 200:
            with lock_estado:
                estado["total_alertas_sesion"] += 1
    except Exception as e:
        print(f"[API Error] Captura no enviada: {e}")

def enviar_documento_zip(ruta_zip, titulo, chat_id):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendDocument"
    try:
        with open(ruta_zip, "rb") as doc:
            requests.post(
                url,
                data={"chat_id": chat_id, "caption": titulo, "parse_mode": "Markdown"},
                files={"document": doc},
                timeout=45
            )
    except Exception as e:
        print(f"[API Error] Compresión ZIP no enviada: {e}")

# CONFIGURACIÓN DEL MENÚ NATIVO DE TELEGRAM
def configurar_menu_comandos():
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/setMyCommands"
    comandos = [
        {"command": "panel", "description": "Ver estado y alertas del sistema"},
        {"command": "descargar", "description": "Extraer evidencias multimedia"},
        {"command": "horario", "description": "Modificar reglas de toque de queda"},
        {"command": "limpiar", "description": "Mantenimiento y depuración de disco"}
    ]
    try:
        respuesta = requests.post(url, json={"commands": comandos}, timeout=10)
        if respuesta.status_code == 200:
            print("Menú de comandos nativo de Telegram inyectado exitosamente.")
    except Exception as e:
        print(f"[API Error] Falla al conectar con setMyCommands: {e}")

# 6. LOGICA DE COMPRESIÓN ESTRUCTURADA
def generar_zip_rango_fechas(f_inicio_str, f_fin_str):
    try:
        fecha_ini = datetime.strptime(f_inicio_str, "%Y-%m-%d").date()
        fecha_fin = datetime.strptime(f_fin_str, "%Y-%m-%d").date()
    except ValueError:
        return None, "El formato de fecha es incorrecto."

    if fecha_ini > fecha_fin:
        return None, "La fecha de inicio no puede ser posterior a la de fin."

    temp_compilacion = f"temp_reporte_{int(time.time())}"
    os.makedirs(temp_compilacion, exist_ok=True)
    
    carpetas_copiadas = 0
    if os.path.exists(CARPETA_CAPTURAS):
        for carpeta in os.listdir(CARPETA_CAPTURAS):
            ruta_carpeta = os.path.join(CARPETA_CAPTURAS, carpeta)
            if os.path.isdir(ruta_carpeta):
                try:
                    fecha_carpeta = datetime.strptime(carpeta[:10], "%Y-%m-%d").date()
                    if fecha_ini <= fecha_carpeta <= fecha_fin:
                        shutil.copytree(ruta_carpeta, os.path.join(temp_compilacion, carpeta))
                        carpetas_copiadas += 1
                except Exception:
                    continue
                    
    if carpetas_copiadas == 0:
        shutil.rmtree(temp_compilacion)
        return None, f"No se encontraron evidencias en turnos iniciados entre el `{f_inicio_str}` y el `{f_fin_str}`."

    nombre_zip = f"Reporte_Rango_{f_inicio_str}_al_{f_fin_str}"
    ruta_zip = shutil.make_archive(nombre_zip, 'zip', temp_compilacion)
    shutil.rmtree(temp_compilacion)
    return ruta_zip, None

# 7. DEMONIO INTERACTIVO (ESCUCHA DE MENSAJES Y BOTONES INTEGRADOS)
def escuchar_telegram():
    ultimo_update_id = None
    print("Pasarela de comunicación activa. Escuchando peticiones...")
    
    while True:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
            params = {"timeout": 10, "allowed_updates": json.dumps(["message", "callback_query"])}
            if ultimo_update_id is not None:
                params["offset"] = ultimo_update_id + 1
            
            r = requests.get(url, params=params, timeout=15)
            data = r.json()
            if not data.get("ok"):
                time.sleep(2)
                continue
            
            for update in data.get("result", []):
                ultimo_update_id = update["update_id"]
                
                # EVENTOS: BOTONES INTERACTIVOS (INLINE)
                if "callback_query" in update:
                    cb = update["callback_query"]
                    cb_id = cb["id"]
                    cb_data = cb.get("data", "")
                    chat_id_msg = str(cb["message"]["chat"]["id"])
                    user_id = cb["from"]["id"]
                    nombre_user = cb["from"].get("first_name", "Admin")
                    
                    if chat_id_msg != TELEGRAM_CHAT_ID: continue
                    requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/answerCallbackQuery", data={"callback_query_id": cb_id})
                    
                    if cb_data == "dl_today":
                        nombre_turno = obtener_nombre_turno(datetime.now())
                        ruta_target = os.path.join(CARPETA_CAPTURAS, nombre_turno)
                        
                        if os.path.exists(ruta_target) and os.listdir(ruta_target):
                            enviar_mensaje_telegram(f"Compilando registros estructurados para el turno actual (`{nombre_turno}`)...", chat_id_msg)
                            temp_compilacion = f"temp_hoy_{int(time.time())}"
                            os.makedirs(temp_compilacion, exist_ok=True)
                            shutil.copytree(ruta_target, os.path.join(temp_compilacion, nombre_turno))
                            
                            nombre_zip = f"Reporte_Turno_{nombre_turno}"
                            ruta_zip = shutil.make_archive(nombre_zip, 'zip', temp_compilacion)
                            
                            enviar_documento_zip(ruta_zip, f"*Reporte del Turno:* `{nombre_turno}`\nOperador: {nombre_user}", chat_id_msg)
                            
                            shutil.rmtree(temp_compilacion)
                            if os.path.exists(ruta_zip): os.remove(ruta_zip)
                        else:
                            enviar_mensaje_telegram(f"*Sin registros:* No hay evidencias guardadas para el turno activo (`{nombre_turno}`).", chat_id_msg)
                    
                    elif cb_data == "dl_range":
                        conversaciones[user_id] = {"paso": "dl_esperando_inicio"}
                        enviar_mensaje_telegram(
                            "*FILTRADO POR RANGO DE FECHAS*\n\n"
                            "Escriba la **FECHA DE INICIO** (`AAAA-MM-DD`). El sistema buscará los turnos que comenzaron a partir de ese día.\n"
                            "Ejemplo: `2026-06-29`", 
                            chat_id_msg
                        )
                    
                    elif cb_data == "clear_today":
                        nombre_turno = obtener_nombre_turno(datetime.now())
                        ruta_hoy = os.path.join(CARPETA_CAPTURAS, nombre_turno)
                        if os.path.exists(ruta_hoy): shutil.rmtree(ruta_hoy)
                        enviar_mensaje_telegram(f"*Operación Exitosa:* Se borró la carpeta del turno activo (`{nombre_turno}`).", chat_id_msg)
                        
                    elif cb_data == "clear_all":
                        if os.path.exists(CARPETA_CAPTURAS): shutil.rmtree(CARPETA_CAPTURAS)
                        os.makedirs(CARPETA_CAPTURAS, exist_ok=True)
                        with lock_estado: estado["total_alertas_sesion"] = 0
                        enviar_mensaje_telegram("*Purga Completa:* Se destruyó todo el historial de turnos local.", chat_id_msg)
                    
                    continue

                # EVENTOS: COMANDOS Y ENTRADA DE TEXTO
                if "message" in update:
                    message = update["message"]
                    texto_msg = message.get("text", "").strip()
                    user_id = message["from"]["id"]
                    nombre = message["from"].get("first_name", "Admin")
                    chat_id_msg = str(message["chat"]["id"])
                    
                    if chat_id_msg != TELEGRAM_CHAT_ID: continue
                    if texto_msg.startswith("/"):
                        comando = texto_msg.lower().split()[0].split('@')[0]
                    else:
                        comando = ""
                                        
                    if comando == "/panel":
                        with lock_estado:
                            ini, fin = estado["hora_inicio"], estado["hora_fin"]
                            total = estado["total_alertas_sesion"]
                        en_tq = es_horario_toque_queda(datetime.now())
                        estado_txt = "*RESTRICCIÓN ACTIVA*" if en_tq else "*ACCESO LIBRE*"
                        turno_actual_str = obtener_nombre_turno(datetime.now())
                        
                        panel_info = (
                            "*SISTEMA DE CONTROL GENERAL*\n"
                            f"*Fase Operativa:* {estado_txt}\n"
                            f"*Toque de Queda:* {ini:02d}:00 a {fin:02d}:00 hrs\n"
                            f"*Turno de Registro:* `{turno_actual_str}`\n"
                            f"*Alertas de la Sesión:* {total}\n"
                            "Use los comandos `/descargar` o `/limpiar` según requiera."
                        )
                        enviar_mensaje_telegram(panel_info, chat_id_msg)
                        continue

                    if comando == "/descargar":
                        markup_criterios = {
                            "inline_keyboard": [
                                [{"text": "Descargar Turno Actual (Hoy)", "callback_data": "dl_today"}],
                                [{"text": "Filtrar rango de fechas", "callback_data": "dl_range"}]
                            ]
                        }
                        enviar_mensaje_telegram("*EXTRACCIÓN DE MULTIMEDIA POR TURNOS*\nSeleccione el criterio de empaquetado para la descarga:", chat_id_msg, reply_markup=markup_criterios)
                        continue

                    if comando == "/limpiar":
                        markup_limpieza = {
                            "inline_keyboard": [
                                [{"text": "Eliminar Turno de Hoy", "callback_data": "clear_today"}],
                                [{"text": "Purgar Todo el Historial", "callback_data": "clear_all"}]
                            ]
                        }
                        enviar_mensaje_telegram("*MANTENIMIENTO DE DISCO*\nSeleccione la acción de eliminación sobre el almacenamiento:", chat_id_msg, reply_markup=markup_limpieza)
                        continue

                    if comando == "/horario":
                        conversaciones[user_id] = {"paso": "esperando_inicio"}
                        with lock_estado: ini, fin = estado["hora_inicio"], estado["hora_fin"]
                        enviar_mensaje_telegram(f"*REGLAS DE HORARIOS*\nRango actual: {ini:02d}:00 a {fin:02d}:00.\nEscriba la nueva *HORA DE INICIO* (0 a 23):", chat_id_msg)
                        continue
                    
                    if user_id in conversaciones:
                        paso = conversaciones[user_id]["paso"]
                        
                        if paso == "esperando_inicio":
                            try:
                                hora_inicio = int(texto_msg)
                                if not (0 <= hora_inicio <= 23): raise ValueError()
                                conversaciones[user_id]["paso"]   = "esperando_fin"
                                conversaciones[user_id]["inicio"] = hora_inicio
                                enviar_mensaje_telegram(f"Hora de inicio establecida: *{hora_inicio:02d}:00*.\nEscriba la *HORA DE FIN* (0 a 23):", chat_id_msg)
                            except (ValueError, TypeError):
                                enviar_mensaje_telegram("*Error:* Ingrese un entero de 0 a 23.", chat_id_msg)
                        
                        elif paso == "esperando_fin":
                            try:
                                hora_fin = int(texto_msg)
                                if not (0 <= hora_fin <= 23): raise ValueError()
                                hora_inicio = conversaciones[user_id]["inicio"]
                                with lock_estado:
                                    estado["hora_inicio"] = hora_inicio
                                    estado["hora_fin"]    = hora_fin
                                del conversaciones[user_id]
                                enviar_mensaje_telegram(f"*HORARIOS ACTUALIZADOS*\n Restricción establecida: *{hora_inicio:02d}:00 a {hora_fin:02d}:00 hrs*", chat_id_msg)
                            except (ValueError, TypeError):
                                enviar_mensaje_telegram("*Error:* Ingrese un entero de 0 a 23.", chat_id_msg)
                        
                        elif paso == "dl_esperando_inicio":
                            try:
                                datetime.strptime(texto_msg, "%Y-%m-%d")
                                conversaciones[user_id]["paso"] = "dl_esperando_fin"
                                conversaciones[user_id]["fecha_inicio"] = texto_msg
                                enviar_mensaje_telegram(f"Fecha de inicio guardada: `{texto_msg}`.\n\nAhora escriba la **FECHA DE FIN** (`AAAA-MM-DD`):\nEjemplo: `2026-06-30`", chat_id_msg)
                            except ValueError:
                                enviar_mensaje_telegram("*Formato Erróneo:* Use la estructura exacta `AAAA-MM-DD`.", chat_id_msg)
                                
                        elif paso == "dl_esperando_fin":
                            try:
                                datetime.strptime(texto_msg, "%Y-%m-%d")
                                f_inicio = conversaciones[user_id]["fecha_inicio"]
                                f_fin = texto_msg
                                del conversaciones[user_id]
                                
                                enviar_mensaje_telegram(f"Analizando y empaquetando turnos desde `{f_inicio}` hasta `{f_fin}`...", chat_id_msg)
                                
                                path_zip, error = generar_zip_rango_fechas(f_inicio, f_fin)
                                if error:
                                    enviar_mensaje_telegram(error, chat_id_msg)
                                else:
                                    enviar_documento_zip(path_zip, f" *Evidencias por Rango Operativo*\n Período de inicio: `{f_inicio}` a `{f_fin}`\n👤 Operador: {nombre}", chat_id_msg)
                                    if os.path.exists(path_zip): os.remove(path_zip)
                            except ValueError:
                                enviar_mensaje_telegram("*Formato Erróneo:* Use la estructura exacta `AAAA-MM-DD`.", chat_id_msg)
        except Exception as e:
            time.sleep(3)
            
# 8. LÓGICA PRINCIPAL (NÚCLEO DE VISIÓN COMPUTACIONAL)
def main():
    os.makedirs(CARPETA_CAPTURAS, exist_ok=True)
    configurar_menu_comandos()
    
    hilo_telegram = threading.Thread(target=escuchar_telegram, daemon=True)
    hilo_telegram.start()
    
    captura = cv2.VideoCapture(INDICE_CAMARA)
    if not captura.isOpened():
        print("Error de Hardware: Interfaz de cámara inaccesible.")
        return
    
    print("Sistema desplegado.")
    
    ids_alertados = set()
    conteo_frames = {}
    
    while True:
        ret, frame = captura.read()
        if not ret: break
        
        ahora = datetime.now()
        en_toque_de_queda = es_horario_toque_queda(ahora)
        resultados = modelo.track(frame, persist=True, verbose=False)[0]
        ids_este_frame = set()
        
        for caja in resultados.boxes:
            if int(caja.cls[0]) != CLASE_PERSONA: continue
            if float(caja.conf[0]) < CONFIANZA_MINIMA: continue
            if caja.id is None: continue
            
            track_id = int(caja.id[0])
            ids_este_frame.add(track_id)
            conteo_frames[track_id] = conteo_frames.get(track_id, 0) + 1
            
            x1, y1, x2, y2 = map(int, caja.xyxy[0])
            confianza = float(caja.conf[0])
            color = (0, 0, 255) if en_toque_de_queda else (0, 255, 0)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID:{track_id} ({confianza:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        for tid in list(conteo_frames.keys()):
            if tid not in ids_este_frame: conteo_frames.pop(tid)
            
        ids_confirmados = {tid for tid in ids_este_frame if conteo_frames.get(tid, 0) >= FRAMES_CONFIRMACION}
        ids_nuevos = ids_confirmados - ids_alertados
        
        # --- NUEVA LÓGICA DE DETECCIÓN Y CORTE DE ROSTROS/CUERPO ---
        if ids_nuevos and en_toque_de_queda:
            nombre_turno_actual = obtener_nombre_turno(ahora)
            carpeta_turno = os.path.join(CARPETA_CAPTURAS, nombre_turno_actual)
            os.makedirs(carpeta_turno, exist_ok=True)
            
            # Recorremos las detecciones del frame para ubicar solo a los intrusos nuevos
            for caja in resultados.boxes:
                if int(caja.cls[0]) != CLASE_PERSONA or caja.id is None: continue
                
                track_id = int(caja.id[0])
                if track_id in ids_nuevos:
                    x1, y1, x2, y2 = map(int, caja.xyxy[0])
                    
                    # Asegurar que las coordenadas no salgan de los límites del frame
                    h_frame, w_frame = frame.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w_frame, x2), min(h_frame, y2)
                    
                    # Aislar (recortar) la imagen del cuerpo del intruso
                    cuerpo_img = frame[y1:y2, x1:x2].copy()
                    
                    # Convertir a escala de grises para el detector de rostros
                    gray_cuerpo = cv2.cvtColor(cuerpo_img, cv2.COLOR_BGR2GRAY)
                    rostros = face_cascade.detectMultiScale(
                        gray_cuerpo, 
                        scaleFactor=1.1, 
                        minNeighbors=4, 
                        minSize=(20, 20)
                    )
                    
                    if len(rostros) > 0:
                        # Si hay rostro, lo encerramos en un cuadro (dibujado sobre el recorte del cuerpo)
                        for (rx, ry, rw, rh) in rostros:
                            cv2.rectangle(cuerpo_img, (rx, ry), (rx+rw, ry+rh), (255, 0, 0), 2)
                        estado_mensaje = "✅ *Se detectó el rostro de la persona.*"
                    else:
                        estado_mensaje = "❌ *No se detectó rostro de la persona.*"
                    
                    # Guardamos la imagen (el cuerpo recortado, con o sin el cuadro en el rostro)
                    nombre_archivo = f"{carpeta_turno}/alerta_{ahora.strftime('%H%M%S')}_id_{track_id}.jpg"
                    cv2.imwrite(nombre_archivo, cuerpo_img)
                    
                    mensaje = (
                        "*ALERTA DE INTRUSIÓN*\n"
                        f"*Turno Operativo:* `{nombre_turno_actual}`\n"
                        f"*Hora Exacta:* {ahora.strftime('%H:%M:%S')}\n"
                        f"*Sujeto:* ID {track_id}\n\n"
                        f"{estado_mensaje}"
                    )
                    
                    # Enviamos por Telegram individualmente
                    enviar_alerta_foto(nombre_archivo, mensaje)
            
            # Registramos todos los nuevos como alertados
            ids_alertados.update(ids_nuevos)
            
        if not en_toque_de_queda: ids_alertados.clear()
            
        # Interfaz Gráfica Local (HUD de Monitoreo)
        turno_hud = obtener_nombre_turno(ahora)
        estado_txt = f"TOQUE DE QUEDA ACTIVO | TURNO: {turno_hud}" if en_toque_de_queda else "ACCESO PERMITIDO"
        color_hud = (0, 0, 255) if en_toque_de_queda else (0, 255, 0)
        
        cv2.putText(frame, f"{estado_txt}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_hud, 2)
        cv2.imshow("System Dashboard", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
            
    captura.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()