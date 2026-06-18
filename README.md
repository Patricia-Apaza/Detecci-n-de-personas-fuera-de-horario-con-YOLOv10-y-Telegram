# Detección de personas fuera de horario con YOLOv10 y Telegram

Sistema de videovigilancia para residencial de varones que detecta personas circulando durante el horario de toque de queda (10:00 p.m. a 4:00 a.m.) usando el modelo YOLOv10, y envía automáticamente una alerta con foto a un chat de Telegram.

## Tecnologías usadas

- Python 3.10
- YOLOv10 (librería `ultralytics`)
- OpenCV (captura de cámara y dibujo de detecciones)
- PyTorch (con soporte GPU/CUDA, o CPU como alternativa)
- API de Bots de Telegram (gratuita)

---

## 1. Clonar el repositorio

Abre una terminal (o el Anaconda Prompt) en la carpeta donde quieras descargar el proyecto, y ejecuta:

```bash
git clone https://github.com/Patricia-Apaza/Detecci-n-de-personas-fuera-de-horario-con-YOLOv10-y-Telegram.git
```

Entra a la carpeta del proyecto:

```bash
cd Detecci-n-de-personas-fuera-de-horario-con-YOLOv10-y-Telegram
```

---

## 2. Instalar Miniconda (si la PC no lo tiene)

Si la computadora que vas a usar no tiene Anaconda/Miniconda instalado:

1. Descargar desde: https://www.anaconda.com/download/success (elegir Miniconda para Windows).
2. Instalar dejando las opciones por defecto.
3. Abrir **"Anaconda Prompt (Miniconda3)"** desde el menú de inicio. Todos los comandos siguientes van en esa ventana.

---

## 3. Crear el entorno de Python

```bash
conda create -n yolo python=3.10 -y
conda activate yolo
```

> Si conda pide aceptar términos de servicio (mensaje "Terms of Service have not been accepted"), ejecuta los tres comandos `conda tos accept ...` que te indique en pantalla, y vuelve a correr `conda create`.

---

## 4. IMPORTANTE: Revisar si la PC tiene GPU NVIDIA antes de instalar PyTorch

Antes de instalar PyTorch, hay que saber si la laptop/PC tiene tarjeta gráfica NVIDIA, porque el comando de instalación cambia.

### Cómo saber si tiene GPU NVIDIA

En el Anaconda Prompt (o cmd), ejecutar:

```bash
nvidia-smi
```

- **Si muestra una tabla con el nombre de una GPU NVIDIA** (ej. RTX, GTX, Quadro) → la PC sí tiene GPU compatible. Ir a la **Opción A**.
- **Si da un error como** `'nvidia-smi' no se reconoce como un comando...` → la PC no tiene GPU NVIDIA, o no tiene los drivers instalados. Ir a la **Opción B**.

### Opción A: La PC SÍ tiene GPU NVIDIA

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verificar que PyTorch detecta la GPU:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Debe imprimir `True`.

### Opción B: La PC NO tiene GPU NVIDIA

Instalar la versión de PyTorch para CPU (más simple, no requiere drivers especiales):

```bash
pip install torch torchvision torchaudio
```

El programa funciona igual, solo que el procesamiento de cada frame de video será más lento (puede notarse que el video va con menos fluidez). Esto es normal y no afecta que la detección y el envío de alertas funcionen correctamente.

> **Nota:** el código de este proyecto detecta automáticamente si hay GPU disponible o no (usa `cuda` si existe, y si no, usa `cpu` sin que el programa se caiga). No es necesario tocar el código para cambiar entre GPU y CPU, solo instalar la versión correcta de PyTorch según el caso.

---

## 5. Instalar las demás librerías

Con el entorno `yolo` activado:

```bash
pip install ultralytics opencv-python requests
```

---

## 6. Configurar el bot de Telegram

El archivo `deteccion.py` ya incluye un token y chat_id de ejemplo. Si vas a usar tu propio bot, edita estas dos líneas al inicio del archivo:

```python
TELEGRAM_TOKEN = "TU_TOKEN_AQUI"
TELEGRAM_CHAT_ID = "TU_CHAT_ID_AQUI"
```

**Importante:** para que el bot pueda escribirte, primero tienes que abrir un chat con tu bot en Telegram y darle "Start" o mandarle cualquier mensaje. Si no, el envío de alertas falla con el error `"chat not found"`.

---

## 7. Ejecutar el programa

Con el entorno activado y dentro de la carpeta del proyecto:

```bash
python deteccion.py
```

- Se abre una ventana mostrando el video de la cámara.
- Recuadro **verde** alrededor de una persona = detectada en horario libre (no envía alerta).
- Recuadro **rojo** alrededor de una persona = detectada en horario de toque de queda (envía foto + mensaje a Telegram).
- Para salir, presionar la tecla `q` con la ventana de video activa.

---

## 8. Resumen rápido (PC nueva, sin nada instalado)

```bash
git clone https://github.com/Patricia-Apaza/Detecci-n-de-personas-fuera-de-horario-con-YOLOv10-y-Telegram.git
cd Detecci-n-de-personas-fuera-de-horario-con-YOLOv10-y-Telegram
conda create -n yolo python=3.10 -y
conda activate yolo

:: Si la PC tiene GPU NVIDIA:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

:: Si la PC NO tiene GPU NVIDIA:
pip install torch torchvision torchaudio

pip install ultralytics opencv-python requests
python deteccion.py
```

---

## Notas

- El modelo `yolov10n.pt` se descarga automáticamente la primera vez que se ejecuta el programa (requiere conexión a internet en ese primer uso).
- El horario de toque de queda (22:00 a 04:00) y los demás parámetros se pueden modificar directamente al inicio del archivo `deteccion.py`.