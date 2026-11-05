import sys
sys.stdout.reconfigure(encoding='utf-8')
"""
generate_video.py
=================
Generador automatizado de video de sustentación para el proyecto
"Clasificación Inteligente de Residuos Sólidos (CRS)"
Maestría en IA, Educación e Investigación — UFHEC

Requisitos:
    pip install Pillow
    ffmpeg en PATH o ajustar FFMPEG_PATH más abajo

Uso:
    C:\\Users\\Feibert\\AppData\\Local\\Programs\\Python\\Python311\\python.exe video\\generate_video.py
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIGURACIÓN GLOBAL
# ---------------------------------------------------------------------------

# Directorio base del proyecto
PROJECT_ROOT = Path(__file__).parent.parent
VIDEO_DIR    = Path(__file__).parent          # …/Project_ML/video/
FRAMES_DIR   = VIDEO_DIR / "frames"
OUTPUT_MP4   = VIDEO_DIR / "crs_sustentacion.mp4"

# Música
MUSIC_SRC = Path(
    r"C:\Users\Feibert\.gemini\config\skills\video_doc\assets\music"
    r"\happy-beats-business-moves-vol-9-by-ende-dot-app.mp3"
)
MUSIC_DST = VIDEO_DIR / "happy-beats-business-moves-vol-9-by-ende-dot-app.mp3"

# FFmpeg (ajusta si no está en PATH)
FFMPEG_PATH = "ffmpeg"

# Dimensiones y duración
WIDTH, HEIGHT = 1920, 1080
FPS           = 30
SCENE_DURATIONS = [3, 3, 5, 5, 4]   # segundos por escena (total=20s)

# Paleta del proyecto
BG_COLOR     = (11, 17, 32)          # #0b1120
ACCENT_COLOR = (16, 185, 129)        # #10b981
TEXT_COLOR   = (241, 245, 249)       # #f1f5f9
DIM_COLOR    = (100, 116, 139)       # slate-500
WHITE        = (255, 255, 255)
BLACK        = (0, 0, 0)
DARK_CARD    = (22, 33, 55)          # card background

# ---------------------------------------------------------------------------
# UTILIDADES DE PILLOW
# ---------------------------------------------------------------------------

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("ERROR: Pillow no instalado. Ejecuta: pip install Pillow")
    sys.exit(1)

def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Carga fuente del sistema con fallback a Arial o default."""
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibrib.ttf" if bold else r"C:\Windows\Fonts\calibri.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    # Último recurso: fuente bitmap por defecto
    return ImageFont.load_default()


def new_frame() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    """Crea un lienzo 1920×1080 con el fondo del proyecto."""
    img  = Image.new("RGB", (WIDTH, HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)
    return img, draw


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    y: int,
    font: ImageFont.FreeTypeFont,
    color: tuple = TEXT_COLOR,
    max_width: int = WIDTH - 160,
) -> None:
    """Dibuja texto centrado horizontalmente en la posición y dada."""
    # Manejar texto largo: truncar si es necesario
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    x = (WIDTH - text_w) // 2
    draw.text((x, y), text, font=font, fill=color)


def draw_rounded_rect(
    draw: ImageDraw.ImageDraw,
    x0: int, y0: int, x1: int, y1: int,
    radius: int = 20,
    fill: tuple = DARK_CARD,
    outline: tuple = None,
    outline_width: int = 2,
) -> None:
    """Dibuja un rectángulo con esquinas redondeadas."""
    draw.rounded_rectangle([x0, y0, x1, y1], radius=radius, fill=fill,
                           outline=outline, width=outline_width)


def draw_progress_bar(
    draw: ImageDraw.ImageDraw,
    x0: int, y0: int, x1: int, y1: int,
    pct: float,                 # 0.0 – 1.0
    bg_color: tuple = (30, 45, 65),
    fill_color: tuple = ACCENT_COLOR,
    radius: int = 10,
) -> None:
    """Barra de progreso horizontal con relleno proporcional."""
    draw.rounded_rectangle([x0, y0, x1, y1], radius=radius, fill=bg_color)
    filled_x1 = int(x0 + (x1 - x0) * min(max(pct, 0), 1))
    if filled_x1 > x0 + 2 * radius:
        draw.rounded_rectangle([x0, y0, filled_x1, y1], radius=radius, fill=fill_color)


def add_accent_line(draw: ImageDraw.ImageDraw, y: int) -> None:
    """Línea de acento esmeralda."""
    draw.rectangle([80, y, WIDTH - 80, y + 3], fill=ACCENT_COLOR)


def add_watermark(draw: ImageDraw.ImageDraw) -> None:
    """Marca de agua UFHEC en la esquina inferior derecha."""
    font = load_font(22)
    draw.text((WIDTH - 300, HEIGHT - 48), "UFHEC — Maestría IA 2026",
              font=font, fill=DIM_COLOR)


# ---------------------------------------------------------------------------
# ESCENA 1 — HOOK ESTADÍSTICO
# ---------------------------------------------------------------------------

def scene1_hook() -> Image.Image:
    """
    Fondo oscuro. Tres líneas de texto aparecen de forma secuencial.
    Dado que generamos un frame estático, mostramos el estado final.
    """
    img, draw = new_frame()

    # Degradado sutil: líneas horizontales con ligera variación de brillo
    for row in range(HEIGHT):
        ratio = row / HEIGHT
        r = int(BG_COLOR[0] + (20 - BG_COLOR[0]) * ratio * 0.3)
        g = int(BG_COLOR[1] + (30 - BG_COLOR[1]) * ratio * 0.3)
        b = int(BG_COLOR[2] + (50 - BG_COLOR[2]) * ratio * 0.3)
        draw.line([(0, row), (WIDTH, row)], fill=(r, g, b))

    # Línea de acento superior
    add_accent_line(draw, 180)

    font_xl  = load_font(110, bold=True)
    font_lg  = load_font(80,  bold=True)
    font_md  = load_font(60)
    font_sm  = load_font(36)

    # Línea 1 — número principal
    draw_centered_text(draw, "2,527", y=240, font=font_xl, color=ACCENT_COLOR)
    draw_centered_text(draw, "imágenes de residuos sólidos", y=380, font=font_md, color=TEXT_COLOR)

    # Separador
    add_accent_line(draw, 480)

    # Línea 2 — features
    draw_centered_text(draw, "→  4 variables", y=510, font=font_lg, color=WHITE)
    draw_centered_text(draw, "mean_r · mean_g · mean_b · edge_density", y=630, font=font_sm, color=DIM_COLOR)

    # Línea 3 — clases
    draw_centered_text(draw, "→  6 categorías de residuos", y=720, font=font_lg, color=WHITE)

    # Clases badge row
    classes = ["📦 cardboard", "🫙 glass", "⚙️ metal", "📄 paper", "🧴 plastic", "🗑️ trash"]
    badge_w = 240
    total_w = badge_w * len(classes) + 20 * (len(classes) - 1)
    start_x = (WIDTH - total_w) // 2
    badge_font = load_font(26)
    for i, cls in enumerate(classes):
        bx = start_x + i * (badge_w + 20)
        draw_rounded_rect(draw, bx, 840, bx + badge_w, 900, radius=14,
                          fill=(22, 40, 60), outline=ACCENT_COLOR)
        bbox = draw.textbbox((0, 0), cls, font=badge_font)
        tx = bx + (badge_w - (bbox[2] - bbox[0])) // 2
        draw.text((tx, 852), cls, font=badge_font, fill=TEXT_COLOR)

    add_watermark(draw)
    return img


# ---------------------------------------------------------------------------
# ESCENA 2 — LOGO Y TÍTULO DEL PROYECTO
# ---------------------------------------------------------------------------

def scene2_title() -> Image.Image:
    img, draw = new_frame()

    font_icon  = load_font(160, bold=True)
    font_title = load_font(72,  bold=True)
    font_sub   = load_font(42)
    font_badge = load_font(32)

    # Icono ♻️  (texto grande centrado)
    draw_centered_text(draw, "♻", y=100, font=font_icon, color=ACCENT_COLOR)

    # Título principal
    draw_centered_text(draw,
                       "Clasificación Inteligente",
                       y=350, font=font_title, color=WHITE)
    draw_centered_text(draw,
                       "de Residuos Sólidos",
                       y=445, font=font_title, color=WHITE)

    # Sigla CRS
    draw_centered_text(draw, "— CRS —", y=550, font=font_sub, color=ACCENT_COLOR)

    add_accent_line(draw, 630)

    # Badge UFHEC
    badge_text = "  Maestría en IA, Educación e Investigación  |  UFHEC  "
    bbox = draw.textbbox((0, 0), badge_text, font=font_badge)
    bw = bbox[2] - bbox[0] + 60
    bh = 60
    bx = (WIDTH - bw) // 2
    by = 660
    draw_rounded_rect(draw, bx, by, bx + bw, by + bh, radius=16,
                      fill=(16, 185, 129, 40), outline=ACCENT_COLOR)
    draw_centered_text(draw, badge_text, y=by + 10, font=font_badge, color=ACCENT_COLOR)

    # Autor
    font_author = load_font(30)
    draw_centered_text(draw, "Investigador: Feibert Alirio Giraldo", y=760, font=font_author, color=DIM_COLOR)
    draw_centered_text(draw, "Machine Learning  •  Noviembre 2026",   y=800, font=font_author, color=DIM_COLOR)

    # Línea inferior
    add_accent_line(draw, HEIGHT - 80)
    add_watermark(draw)
    return img


# ---------------------------------------------------------------------------
# ESCENA 3 — PIPELINE ETL EN ACCIÓN
# ---------------------------------------------------------------------------

def scene3_pipeline() -> Image.Image:
    img, draw = new_frame()

    font_title = load_font(54, bold=True)
    font_step  = load_font(32, bold=True)
    font_code  = load_font(26)
    font_arrow = load_font(60, bold=True)

    # Título
    draw_centered_text(draw, "Pipeline ETL — Ingeniería de Características",
                       y=60, font=font_title, color=WHITE)
    add_accent_line(draw, 140)

    # 5 pasos del pipeline
    steps = [
        ("📷", "Imagen\noriginal", "JPEG / PNG\ncualquier res."),
        ("🔄", "Resize\n100 × 100", "cv2.resize()\ninter=AREA"),
        ("🎨", "Extracción\nRGB + Canny", "mean_r, mean_g\nmean_b, edge_density"),
        ("📊", "DataFrame\nCSV", "2,527 filas\n4 columnas"),
        ("🌲", "Random\nForest", "100 árboles\nbalanced • Gini"),
    ]

    box_w, box_h = 290, 260
    total_boxes_w = len(steps) * box_w + (len(steps) - 1) * 60
    start_x = (WIDTH - total_boxes_w) // 2
    box_y = 200

    for i, (icon, name, detail) in enumerate(steps):
        bx = start_x + i * (box_w + 60)

        # Caja del paso
        outline_col = ACCENT_COLOR if i in (2, 4) else (60, 90, 130)
        draw_rounded_rect(draw, bx, box_y, bx + box_w, box_y + box_h,
                          radius=18, fill=DARK_CARD, outline=outline_col, outline_width=3)

        # Número de paso
        step_num = load_font(20)
        draw.text((bx + 14, box_y + 12), f"PASO {i+1}", font=step_num, fill=ACCENT_COLOR)

        # Icono
        font_ico = load_font(50)
        bbox = draw.textbbox((0, 0), icon, font=font_ico)
        ix = bx + (box_w - (bbox[2] - bbox[0])) // 2
        draw.text((ix, box_y + 42), icon, font=font_ico, fill=TEXT_COLOR)

        # Nombre del paso
        for j, line in enumerate(name.split("\n")):
            bbox = draw.textbbox((0, 0), line, font=font_step)
            lx = bx + (box_w - (bbox[2] - bbox[0])) // 2
            draw.text((lx, box_y + 112 + j * 42), line, font=font_step, fill=WHITE)

        # Detalle (código)
        for j, line in enumerate(detail.split("\n")):
            bbox = draw.textbbox((0, 0), line, font=font_code)
            lx = bx + (box_w - (bbox[2] - bbox[0])) // 2
            draw.text((lx, box_y + 198 + j * 32), line, font=font_code, fill=DIM_COLOR)

        # Flecha entre pasos
        if i < len(steps) - 1:
            ax = bx + box_w + 10
            ay = box_y + box_h // 2 - 20
            draw.text((ax, ay), "→", font=font_arrow, fill=ACCENT_COLOR)

    # Bloque resultado
    res_y = box_y + box_h + 60
    draw_rounded_rect(draw, 200, res_y, WIDTH - 200, res_y + 180, radius=20,
                      fill=(10, 30, 20), outline=ACCENT_COLOR, outline_width=2)

    font_res   = load_font(42, bold=True)
    font_res_s = load_font(30)
    draw_centered_text(draw, "🎯  Clase predicha con 62.8% de exactitud (Random Forest)",
                       y=res_y + 30, font=font_res, color=ACCENT_COLOR)
    draw_centered_text(draw,
                       "Inferencia < 5 ms  •  Ejecutable en Raspberry Pi  •  Modelo interpretable",
                       y=res_y + 100, font=font_res_s, color=DIM_COLOR)

    add_watermark(draw)
    return img


# ---------------------------------------------------------------------------
# ESCENA 4 — COMPARATIVA DE MODELOS
# ---------------------------------------------------------------------------

def scene4_comparison() -> Image.Image:
    img, draw = new_frame()

    font_title = load_font(54, bold=True)
    font_model = load_font(44, bold=True)
    font_pct   = load_font(100, bold=True)
    font_label = load_font(28)
    font_badge = load_font(26)

    draw_centered_text(draw, "Comparativa de Modelos", y=50, font=font_title, color=WHITE)
    add_accent_line(draw, 130)

    # --- Panel izquierdo: Random Forest ---
    panel_w = 820
    left_x  = 60
    panel_y = 160
    panel_h = 760

    draw_rounded_rect(draw, left_x, panel_y, left_x + panel_w, panel_y + panel_h,
                      radius=24, fill=DARK_CARD, outline=(60, 90, 130), outline_width=2)

    draw.text((left_x + 40, panel_y + 30), "🌲 Random Forest Classifier",
              font=font_model, fill=TEXT_COLOR)

    # Barra RF
    bar_x0, bar_y0 = left_x + 40, panel_y + 150
    bar_x1, bar_y1 = left_x + panel_w - 40, panel_y + 230
    draw_progress_bar(draw, bar_x0, bar_y0, bar_x1, bar_y1, pct=0.628,
                      bg_color=(30, 45, 65), fill_color=(99, 102, 241))  # indigo

    # Porcentaje RF
    bbox = draw.textbbox((0, 0), "62.8%", font=font_pct)
    pct_x = left_x + (panel_w - (bbox[2] - bbox[0])) // 2
    draw.text((pct_x, panel_y + 255), "62.8%", font=font_pct, fill=(99, 102, 241))

    draw_centered_text(draw, "Accuracy Global", y=panel_y + 390, font=font_label, color=DIM_COLOR)

    # Detalles RF
    rf_details = [
        ("Tipo de entrada", "4 features tabulares"),
        ("Clases objetivo", "6 categorías de residuos"),
        ("Tiempo de inferencia", "< 5 ms"),
        ("Hardware mínimo", "Raspberry Pi Zero"),
        ("Interpretabilidad", "🟢 ALTA"),
    ]
    for k, (key, val) in enumerate(rf_details):
        y_det = panel_y + 450 + k * 55
        draw.text((left_x + 40, y_det),         f"{key}:",  font=font_badge, fill=DIM_COLOR)
        draw.text((left_x + 300, y_det),  val,  font=font_badge, fill=TEXT_COLOR)

    # --- Panel derecho: MobileNetV2 ---
    right_x = WIDTH - left_x - panel_w

    draw_rounded_rect(draw, right_x, panel_y, right_x + panel_w, panel_y + panel_h,
                      radius=24, fill=(10, 25, 15), outline=ACCENT_COLOR, outline_width=3)

    draw.text((right_x + 40, panel_y + 30), "🧠 MobileNetV2 (Transfer Learning)",
              font=font_model, fill=TEXT_COLOR)

    # Barra TL
    bar_rx0, bar_ry0 = right_x + 40, panel_y + 150
    bar_rx1, bar_ry1 = right_x + panel_w - 40, panel_y + 230
    draw_progress_bar(draw, bar_rx0, bar_ry0, bar_rx1, bar_ry1, pct=0.914,
                      bg_color=(10, 40, 25), fill_color=ACCENT_COLOR)

    # Porcentaje TL
    bbox = draw.textbbox((0, 0), "91.4%", font=font_pct)
    pct_x = right_x + (panel_w - (bbox[2] - bbox[0])) // 2
    draw.text((pct_x, panel_y + 255), "91.4%", font=font_pct, fill=ACCENT_COLOR)

    draw_centered_text(draw, "Accuracy Global", y=panel_y + 390, font=font_label, color=DIM_COLOR)

    # Detalles TL
    tl_details = [
        ("Tipo de entrada", "Tensor 224×224×3"),
        ("Clases objetivo", "3 clases botellas PET"),
        ("Tiempo de inferencia", "25-50 ms"),
        ("Hardware mínimo", "CPU moderna"),
        ("Interpretabilidad", "🟡 MEDIA"),
    ]
    for k, (key, val) in enumerate(tl_details):
        y_det = panel_y + 450 + k * 55
        draw.text((right_x + 40, y_det),          f"{key}:", font=font_badge, fill=DIM_COLOR)
        draw.text((right_x + 300, y_det), val,    font=font_badge, fill=TEXT_COLOR)

    # Flecha de mejora en el centro
    center_x = left_x + panel_w + (right_x - left_x - panel_w) // 2
    font_delta = load_font(36, bold=True)
    font_delta_lbl = load_font(24)
    delta_text = "+28.6%"
    bbox = draw.textbbox((0, 0), delta_text, font=font_delta)
    dx = center_x - (bbox[2] - bbox[0]) // 2
    draw.text((dx, panel_y + 320), delta_text, font=font_delta, fill=ACCENT_COLOR)
    draw.text((center_x - 20, panel_y + 370), "↑", font=font_delta, fill=ACCENT_COLOR)
    draw_centered_text(draw, "mejora", y=panel_y + 420, font=font_delta_lbl, color=DIM_COLOR)

    add_watermark(draw)
    return img


# ---------------------------------------------------------------------------
# ESCENA 5 — CTA Y CIERRE
# ---------------------------------------------------------------------------

def scene5_cta() -> Image.Image:
    img, draw = new_frame()

    font_headline = load_font(68, bold=True)
    font_sub      = load_font(38)
    font_url      = load_font(30)
    font_badge    = load_font(28)

    # Título de cierre
    draw_centered_text(draw, "Código Abierto. Desplegado en la Nube.", y=140, font=font_headline, color=WHITE)
    draw_centered_text(draw, "Listo para Economía Circular.", y=230, font=font_headline, color=ACCENT_COLOR)

    add_accent_line(draw, 330)

    # Badges de plataformas
    badge_items = [
        ("🐙 GitHub", "feibertciin/Project_ML_CRS",    (40, 60, 100)),
        ("🚀 Streamlit", "Project_ML_CRS.streamlit.app", (120, 30, 50)),
        ("🌐 GitHub Pages", "feibertciin.github.io/Project_ML_CRS", (20, 80, 60)),
    ]
    badge_bw = 540
    total_bw = badge_bw * 3 + 40 * 2
    start_bx = (WIDTH - total_bw) // 2

    for i, (plat, url, fill) in enumerate(badge_items):
        bx = start_bx + i * (badge_bw + 40)
        draw_rounded_rect(draw, bx, 370, bx + badge_bw, 530, radius=20,
                          fill=fill, outline=ACCENT_COLOR, outline_width=2)
        bbox = draw.textbbox((0, 0), plat, font=font_badge)
        draw.text((bx + (badge_bw - (bbox[2] - bbox[0])) // 2, 390),
                  plat, font=font_badge, fill=TEXT_COLOR)
        bbox = draw.textbbox((0, 0), url, font=font_url)
        draw.text((bx + (badge_bw - (bbox[2] - bbox[0])) // 2, 460),
                  url, font=font_url, fill=DIM_COLOR)

    # Métricas resumen
    add_accent_line(draw, 570)
    metrics = [
        ("2,527", "Imágenes procesadas"),
        ("4",     "Variables extraídas"),
        ("62.8%", "Exactitud RF"),
        ("91.4%", "Exactitud MobileNetV2"),
        ("5 ms",  "Inferencia RF"),
    ]
    metric_w = WIDTH // len(metrics)
    font_mval = load_font(50, bold=True)
    font_mlbl = load_font(24)
    for i, (val, lbl) in enumerate(metrics):
        mx = i * metric_w
        bbox = draw.textbbox((0, 0), val, font=font_mval)
        vx = mx + (metric_w - (bbox[2] - bbox[0])) // 2
        draw.text((vx, 610), val, font=font_mval, fill=ACCENT_COLOR)
        bbox = draw.textbbox((0, 0), lbl, font=font_mlbl)
        lx = mx + (metric_w - (bbox[2] - bbox[0])) // 2
        draw.text((lx, 680), lbl, font=font_mlbl, fill=DIM_COLOR)

    # Badge UFHEC grande
    add_accent_line(draw, 760)
    ufhec_text = "🎓  UFHEC  —  Maestría en Inteligencia Artificial, Educación e Investigación  —  2026"
    font_ufhec = load_font(30)
    draw_centered_text(draw, ufhec_text, y=800, font=font_ufhec, color=TEXT_COLOR)

    font_author = load_font(26)
    draw_centered_text(draw,
                       "Feibert Alirio Giraldo  |  Machine Learning  |  Noviembre 2026",
                       y=855, font=font_author, color=DIM_COLOR)

    # Línea final
    add_accent_line(draw, HEIGHT - 80)
    return img


# ---------------------------------------------------------------------------
# MAIN: GENERAR FRAMES Y VIDEO
# ---------------------------------------------------------------------------

def generate_frames() -> list[Path]:
    """Genera los 5 frames PNG y devuelve sus rutas."""
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    scene_funcs = [scene1_hook, scene2_title, scene3_pipeline, scene4_comparison, scene5_cta]
    frame_paths = []

    for i, func in enumerate(scene_funcs, start=1):
        path = FRAMES_DIR / f"scene_{i:02d}.png"
        print(f"  Generando frame {i}/5: {path.name} …")
        img = func()
        img.save(str(path), format="PNG", optimize=False)
        frame_paths.append(path)
        print(f"    ✓ Guardado ({img.size[0]}×{img.size[1]})")

    return frame_paths


def copy_music() -> Path:
    """Copia el track de música al directorio de video."""
    if MUSIC_SRC.exists():
        shutil.copy2(str(MUSIC_SRC), str(MUSIC_DST))
        print(f"  ✓ Música copiada: {MUSIC_DST.name}")
    else:
        print(f"  ⚠ Música no encontrada en: {MUSIC_SRC}")
        print("    El video se generará sin audio.")
    return MUSIC_DST


def build_ffmpeg_command(frame_paths: list[Path], music_path: Path) -> list[str]:
    """
    Construye el comando FFmpeg para:
    1. Combinar frames PNG en video (duración variable por escena)
    2. Mezclar con música a 0.25 de volumen con fade-in/out
    3. Exportar H.264 1920×1080 30fps
    """
    # Crear archivo de lista de frames con duración por escena
    concat_file = FRAMES_DIR / "concat.txt"
    with open(str(concat_file), "w", encoding="utf-8") as f:
        for frame_path, duration in zip(frame_paths, SCENE_DURATIONS):
            f.write(f"file '{frame_path.resolve()}'\n")
            f.write(f"duration {duration}\n")
        # FFmpeg concat requiere repetir el último frame
        f.write(f"file '{frame_paths[-1].resolve()}'\n")

    total_duration = sum(SCENE_DURATIONS)  # 20 seconds

    has_music = music_path.exists()

    if has_music:
        cmd = [
            FFMPEG_PATH,
            "-y",                               # Sobrescribir output si existe
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),             # Input 1: frames
            "-i", str(music_path),              # Input 2: música
            "-filter_complex",
            (
                # Video: escala a 1920x1080 y aplica cross-dissolve simple
                "[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,"
                "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=#0b1120,"
                "fps=30[vout];"
                # Audio: recortar al total, fade-in 0.5s, fade-out 1s, bajar volumen
                f"[1:a]atrim=0:{total_duration},"
                f"afade=t=in:st=0:d=0.5,"
                f"afade=t=out:st={total_duration-1}:d=1,"
                "volume=0.25[aout]"
            ),
            "-map", "[vout]",
            "-map", "[aout]",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-c:a", "aac",
            "-b:a", "192k",
            "-movflags", "+faststart",
            "-t", str(total_duration),
            str(OUTPUT_MP4),
        ]
    else:
        # Sin audio
        cmd = [
            FFMPEG_PATH,
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            "-filter_complex",
            (
                "[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,"
                "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=#0b1120,"
                "fps=30[vout]"
            ),
            "-map", "[vout]",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-movflags", "+faststart",
            "-t", str(sum(SCENE_DURATIONS)),
            str(OUTPUT_MP4),
        ]

    return cmd


def render_video(frame_paths: list[Path], music_path: Path) -> bool:
    """Llama a FFmpeg para renderizar el video final."""
    cmd = build_ffmpeg_command(frame_paths, music_path)
    print(f"\n  Ejecutando FFmpeg …")
    print("  Comando:", " ".join(cmd[:6]), "…")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )
        if result.returncode == 0:
            size_mb = OUTPUT_MP4.stat().st_size / (1024 * 1024)
            print(f"  ✓ Video generado: {OUTPUT_MP4}")
            print(f"    Tamaño: {size_mb:.1f} MB")
            return True
        else:
            print("  ✗ Error en FFmpeg:")
            print(result.stderr[-3000:])  # Últimas 3000 chars del log
            return False
    except FileNotFoundError:
        print("  ✗ FFmpeg no encontrado. Instala FFmpeg y agrega al PATH.")
        print("    Descarga: https://www.gyan.dev/ffmpeg/builds/")
        return False
    except subprocess.TimeoutExpired:
        print("  ✗ Timeout al ejecutar FFmpeg (>5 min)")
        return False


def update_enlace_video(success: bool) -> None:
    """Actualiza enlace_video.txt con la información del video generado."""
    enlace_path = VIDEO_DIR / "enlace_video.txt"
    content = f"""# Video de Sustentación — Clasificación Inteligente de Residuos Sólidos (CRS)
# Proyecto Final Machine Learning | UFHEC — Maestría en IA | Noviembre 2026
# Investigador: Feibert Alirio Giraldo

## Estado del Video
{"✅ GENERADO EXITOSAMENTE" if success else "⚠ Generación de video pendiente (revisar FFmpeg)"}

## Archivo Local
{OUTPUT_MP4}

## Descripción del Video
Duración: 20 segundos | Resolución: 1920×1080 (Full HD) | FPS: 30 | Codec: H.264 + AAC

## Escenas
1. [0s-3s]   Hook Estadístico — "2,527 imágenes → 4 variables → 6 categorías"
2. [3s-6s]   Logo y Título — CRS + Badge UFHEC
3. [6s-11s]  Pipeline ETL — 5 pasos del flujo de datos y extracción de características
4. [11s-16s] Comparativa de Modelos — Random Forest 62.8% vs MobileNetV2 91.4%
5. [16s-20s] CTA y Cierre — GitHub, Streamlit, GitHub Pages + Badge UFHEC

## URLs del Proyecto
- GitHub:        https://github.com/feibertciin/Project_ML_CRS
- Streamlit App: https://Project_ML_CRS.streamlit.app
- Landing Page:  https://feibertciin.github.io/Project_ML_CRS
- Notebook:      Project_ML_CRS.ipynb (en raíz del repositorio)

## Música
Track: happy-beats-business-moves-vol-9-by-ende-dot-app.mp3 (114.84 BPM)
Volumen: 0.25 (ducked) | Fade-in: 0.5s | Fade-out: 1.0s

## Generado con
Pillow (frames PNG 1920×1080) + FFmpeg (renderizado H.264) + Python 3.11
Script: video/generate_video.py
Fecha: Noviembre 2026
"""
    with open(str(enlace_path), "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  ✓ enlace_video.txt actualizado")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("  CRS — Generador de Video de Sustentación")
    print("  Clasificación Inteligente de Residuos Sólidos | UFHEC 2026")
    print("=" * 70)

    print("\n[1/4] Preparando directorios …")
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Output: {OUTPUT_MP4}")

    print("\n[2/4] Generando frames PNG (Pillow) …")
    frame_paths = generate_frames()
    print(f"  ✓ {len(frame_paths)} frames generados en {FRAMES_DIR}")

    print("\n[3/4] Preparando audio …")
    music_path = copy_music()

    print("\n[4/4] Renderizando video con FFmpeg …")
    success = render_video(frame_paths, music_path)

    print("\n[+] Actualizando enlace_video.txt …")
    update_enlace_video(success)

    print("\n" + "=" * 70)
    if success:
        print("  ✅ VIDEO GENERADO EXITOSAMENTE")
        print(f"  📹 {OUTPUT_MP4}")
    else:
        print("  ⚠  Los frames PNG fueron generados correctamente.")
        print("  ⚠  Instala FFmpeg para completar el video:")
        print("      https://www.gyan.dev/ffmpeg/builds/")
        print(f"  📁 Frames disponibles en: {FRAMES_DIR}")
    print("=" * 70)

