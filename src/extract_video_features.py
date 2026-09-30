import cv2
import json
import numpy as np
import pandas as pd
from pathlib import Path
import mediapipe as mp


BASE_DIR = Path(__file__).resolve().parent


VIDEOS_DIR = BASE_DIR / "videos"


RESULTS_DIR = BASE_DIR / "resultados_revision_individual"
RESULTS_DIR.mkdir(exist_ok=True)


VIDEO_NAME = "Longi_MASC_022" \
".avi"
VIDEO_PATH = VIDEOS_DIR / VIDEO_NAME
STEM = Path(VIDEO_NAME).stem

FPS_REAL = 240
DISTANCIA_REFERENCIA_M = 3.0


USAR_CALIBRACION_COMUN = True

CALIBRACION_COMUN = {
    "x_tabla": 1205,
    "x_foso": 1443,
    "distancia_pixeles": 238,
    "escala_m_por_pixel": 3.0 / 238,
    "frame_calibracion": "comun"
}


MARGEN_X = 140
MARGEN_Y = 80
MIN_AREA = 300
CROP_W = 550
CROP_H_FIXED = None


VIS_THRESHOLD = 0.55
MAX_JUMP_DEGREES = 35
SMOOTH_WINDOW = 5


MIN_SEPARACION_FRAMES = 18
N_APOYOS = 6
DIST_MAX_ANTES_TABLA_M = 13.0
MARGEN_DESPUES_TABLA_M = -0.40


MIN_GAP_APOYOS = 24
MIN_GAP_ULTIMO_BATIDA = 38
MAX_GAP_APOYOS = 90


MIN_DISTANCIA_ENTRE_APOYOS_M = 0.75
MIN_DISTANCIA_ULTIMO_BATIDA_M = 1.00


ACTIVAR_FILTRO_APOYOS_EN_AIRE = True


TOLERANCIA_Y_APOYO_PX = 18


MAX_VELOCIDAD_PLAUSIBLE_M_S = 13.0


ACTIVAR_RESCATE_APOYO_PREBATIDA = True
GAP_SOSPECHA_FALTA_PREBATIDA = 65
FRAME_ESPERADO_PREBATIDA = 45
MARGEN_BUSQUEDA_PREBATIDA = 18


ACTIVAR_REPARACION_SECUENCIA = True


BATIDA_DIST_MIN_M = -0.50
BATIDA_DIST_MAX_M = 1.30


MIN_LONGITUD_RECONSTRUIR_M = 0.40
MAX_LONGITUD_RECONSTRUIR_M = 4.20
GAP_IDEAL_FRAMES = 45
LONGITUD_IDEAL_PASO_M = 1.80


USAR_VENTANA_ANALISIS = False
FRAME_INICIO_ANALISIS = 450
FRAME_FIN_ANALISIS = 950


VENTANAS_SALIDA_FRAMES = [10, 15, 20]
MIN_FRAMES_VALIDOS_SALIDA = 5
VENTANA_SUAVIZADO_CADERA = 5


ACTIVAR_FILTRO_POSE_ATLETA_POR_CROP = True
MAX_DIST_POSE_A_CROP_PX = 50
MAX_SALTO_CADERA_TRACKING_PX = 180
MAX_FRAMES_SIN_POSE_ANTES_RESET = 20


OUTPUT_CROP = RESULTS_DIR / f"crop_atleta_{STEM}.mp4"
OUTPUT_DEBUG_CROP = RESULTS_DIR / f"debug_movimiento_{STEM}.mp4"
OUTPUT_CROP_INFO = RESULTS_DIR / f"crop_info_{STEM}.csv"
OUTPUT_POSE_VIDEO = RESULTS_DIR / f"video_pose_{STEM}.mp4"
OUTPUT_KEYPOINTS = RESULTS_DIR / f"keypoints_angulos_{STEM}.csv"
OUTPUT_FINAL = RESULTS_DIR / f"eventos_automaticos_{STEM}.csv"

CALIBRATION_JSON = RESULTS_DIR / f"calibracion_{STEM}.json"

DEBUG_DIR = RESULTS_DIR / f"debug_eventos_auto_{STEM}"
DEBUG_DIR.mkdir(exist_ok=True)


print("=== PIPELINE COMPLETO AUTOMÁTICO ===")
print("Vídeo:", VIDEO_PATH)
print("Existe:", VIDEO_PATH.exists())

if not VIDEO_PATH.exists():
    print("\nNo encuentro el vídeo. Archivos en la carpeta:")
    for f in BASE_DIR.iterdir():
        print("-", f.name)
    raise FileNotFoundError("No existe el vídeo indicado.")


def calcular_angulo(a, b, c):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)

    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba == 0 or norm_bc == 0:
        return np.nan

    coseno = np.dot(ba, bc) / (norm_ba * norm_bc)
    coseno = np.clip(coseno, -1.0, 1.0)

    return np.degrees(np.arccos(coseno))


def inclinacion_segmento(p1, p2):
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)

    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    if dx == 0 and dy == 0:
        return np.nan

    return np.degrees(np.arctan2(dx, dy))


calibration_points = []
calibration_frame = None


def click_calibracion(event, x, y, flags, param):
    global calibration_points, calibration_frame

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(calibration_points) < 2:
            calibration_points.append((x, y))

            label = "tabla" if len(calibration_points) == 1 else "inicio_foso"

            print(f"{label}: x={x}, y={y}")

            cv2.circle(calibration_frame, (x, y), 7, (0, 0, 255), -1)
            cv2.putText(
                calibration_frame,
                label,
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
            cv2.imshow("Calibracion", calibration_frame)


def cargar_o_crear_calibracion():
    global calibration_points, calibration_frame

    if USAR_CALIBRACION_COMUN:
        print("\n=== USANDO CALIBRACIÓN COMÚN ===")
        print(CALIBRACION_COMUN)
        return CALIBRACION_COMUN

    if CALIBRATION_JSON.exists():
        usar = input("Ya existe calibración para este vídeo. ¿Usarla? s/n: ").strip().lower()
        if usar == "s":
            with open(CALIBRATION_JSON, "r", encoding="utf-8") as f:
                calibracion = json.load(f)
            print("Calibración cargada:", calibracion)
            return calibracion

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise ValueError("No se pudo abrir el vídeo para calibrar.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("\n=== CALIBRACIÓN ===")
    print("Marca 2 puntos en el vídeo original:")
    print("1. Tabla de batida")
    print("2. Inicio del foso")
    print("Distancia real entre ambos = 3 m")

    frame_number = input("Frame donde se vea bien tabla/foso. Prueba 650 o 680: ").strip()

    try:
        frame_number = int(frame_number)
    except ValueError:
        frame_number = total_frames // 2

    frame_number = max(0, min(frame_number, total_frames - 1))

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError("No se pudo leer frame de calibración.")

    calibration_points = []
    calibration_frame = frame.copy()

    cv2.namedWindow("Calibracion", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Calibracion", 1200, 500)
    cv2.imshow("Calibracion", calibration_frame)
    cv2.setMouseCallback("Calibracion", click_calibracion)

    print("\nHaz clic en tabla y después en inicio del foso. Luego pulsa cualquier tecla.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(calibration_points) != 2:
        raise ValueError("No marcaste los dos puntos de calibración.")

    x_tabla = calibration_points[0][0]
    x_foso = calibration_points[1][0]

    distancia_pixeles = abs(x_foso - x_tabla)

    if distancia_pixeles == 0:
        raise ValueError("Tabla e inicio del foso tienen la misma x. No se puede calibrar.")

    escala_m_por_pixel = DISTANCIA_REFERENCIA_M / distancia_pixeles

    calibracion = {
        "x_tabla": x_tabla,
        "x_foso": x_foso,
        "distancia_pixeles": distancia_pixeles,
        "escala_m_por_pixel": escala_m_por_pixel,
        "frame_calibracion": frame_number
    }

    with open(CALIBRATION_JSON, "w", encoding="utf-8") as f:
        json.dump(calibracion, f, indent=4)

    print("Calibración guardada:", calibracion)

    return calibracion


def generar_crop_automatico():
    print("\n=== PASO 1: GENERANDO CROP AUTOMÁTICO ===")

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise ValueError("No se pudo abrir el vídeo original.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    crop_h = height if CROP_H_FIXED is None else CROP_H_FIXED

    print("FPS:", fps)
    print("Frames:", total_frames)
    print("Resolución:", width, "x", height)
    print("Crop:", CROP_W, "x", crop_h)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out_debug = cv2.VideoWriter(str(OUTPUT_DEBUG_CROP), fourcc, fps, (width, height))
    out_crop = cv2.VideoWriter(str(OUTPUT_CROP), fourcc, fps, (CROP_W, crop_h))

    back_sub = cv2.createBackgroundSubtractorMOG2(
        history=300,
        varThreshold=25,
        detectShadows=False
    )

    last_box = None
    crop_rows = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        fg_mask = back_sub.apply(frame)

        kernel = np.ones((5, 5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            fg_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        boxes = []

        for cnt in contours:
            area = cv2.contourArea(cnt)

            if area < MIN_AREA:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            if h < 40 or w < 20:
                continue

            boxes.append((x, y, w, h, area))

        detected = False

        if boxes:
            x, y, w, h, area = max(boxes, key=lambda b: b[4])

            x1 = max(0, x - MARGEN_X)
            y1 = max(0, y - MARGEN_Y)
            x2 = min(width, x + w + MARGEN_X)
            y2 = min(height, y + h + MARGEN_Y)

            last_box = (x1, y1, x2, y2)
            detected = True

        if last_box is not None:
            x1, y1, x2, y2 = last_box
        else:
            x1, y1, x2, y2 = 0, 0, width, height

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            crop = frame.copy()
            x1, y1, x2, y2 = 0, 0, width, height

        crop_original_w = x2 - x1
        crop_original_h = y2 - y1

        crop_resized = cv2.resize(crop, (CROP_W, crop_h))

        scale_x = crop_original_w / CROP_W
        scale_y = crop_original_h / crop_h

        crop_rows.append({
            "video": VIDEO_NAME,
            "frame": frame_idx,
            "time_s": frame_idx / fps if fps > 0 else np.nan,
            "detected_motion_this_frame": detected,
            "x1_crop_original": x1,
            "y1_crop_original": y1,
            "x2_crop_original": x2,
            "y2_crop_original": y2,
            "crop_original_w": crop_original_w,
            "crop_original_h": crop_original_h,
            "crop_saved_w": CROP_W,
            "crop_saved_h": crop_h,
            "scale_x_crop_to_original": scale_x,
            "scale_y_crop_to_original": scale_y,
            "centro_crop_x_original": (x1 + x2) / 2,
            "centro_crop_y_original": (y1 + y2) / 2
        })

        debug_frame = frame.copy()
        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            debug_frame,
            f"Frame {frame_idx}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        out_debug.write(debug_frame)
        out_crop.write(crop_resized)

        frame_idx += 1

    cap.release()
    out_debug.release()
    out_crop.release()

    crop_df = pd.DataFrame(crop_rows)
    crop_df.to_csv(OUTPUT_CROP_INFO, index=False, encoding="utf-8-sig")

    print("Crop guardado:", OUTPUT_CROP)
    print("Debug crop guardado:", OUTPUT_DEBUG_CROP)
    print("Info crop guardada:", OUTPUT_CROP_INFO)


def detectar_pose_y_angulos():
    print("\n=== PASO 2: DETECTANDO POSE Y ÁNGULOS ===")

    crop_info = pd.read_csv(OUTPUT_CROP_INFO)

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    )

    idx = {
        "hombro_izq": mp_pose.PoseLandmark.LEFT_SHOULDER.value,
        "hombro_der": mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
        "cadera_izq": mp_pose.PoseLandmark.LEFT_HIP.value,
        "cadera_der": mp_pose.PoseLandmark.RIGHT_HIP.value,
        "rodilla_izq": mp_pose.PoseLandmark.LEFT_KNEE.value,
        "rodilla_der": mp_pose.PoseLandmark.RIGHT_KNEE.value,
        "tobillo_izq": mp_pose.PoseLandmark.LEFT_ANKLE.value,
        "tobillo_der": mp_pose.PoseLandmark.RIGHT_ANKLE.value,
    }

    cap = cv2.VideoCapture(str(OUTPUT_CROP))

    if not cap.isOpened():
        raise ValueError("No se pudo abrir el vídeo crop.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_pose = cv2.VideoWriter(
        str(OUTPUT_POSE_VIDEO),
        fourcc,
        fps,
        (width, height)
    )

    rows = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_idx >= len(crop_info):
            break

        info = crop_info.iloc[frame_idx]

        x1 = info["x1_crop_original"]
        y1 = info["y1_crop_original"]
        scale_x = info["scale_x_crop_to_original"]
        scale_y = info["scale_y_crop_to_original"]

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        annotated = frame.copy()

        centro_crop_x = (info["x1_crop_original"] + info["x2_crop_original"]) / 2
        centro_crop_y = (info["y1_crop_original"] + info["y2_crop_original"]) / 2

        row = {
            "video": VIDEO_NAME,
            "frame": frame_idx,
            "time_s": frame_idx / fps if fps > 0 else np.nan,
            "pose_detected": False,
            "centro_crop_x_original": centro_crop_x,
            "centro_crop_y_original": centro_crop_y,
            "dist_pose_a_crop_px": np.nan
        }

        if results.pose_landmarks:
            row["pose_detected"] = True
            landmarks = results.pose_landmarks.landmark

            mp_drawing.draw_landmarks(
                annotated,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            for name, landmark_idx in idx.items():
                lm = landmarks[landmark_idx]

                x_crop = lm.x * width
                y_crop = lm.y * height

                x_original = x1 + x_crop * scale_x
                y_original = y1 + y_crop * scale_y

                row[f"{name}_x_crop"] = x_crop
                row[f"{name}_y_crop"] = y_crop
                row[f"{name}_x_original"] = x_original
                row[f"{name}_y_original"] = y_original
                row[f"{name}_vis"] = lm.visibility

            hi = (row["hombro_izq_x_crop"], row["hombro_izq_y_crop"])
            hd = (row["hombro_der_x_crop"], row["hombro_der_y_crop"])
            ci = (row["cadera_izq_x_crop"], row["cadera_izq_y_crop"])
            cd = (row["cadera_der_x_crop"], row["cadera_der_y_crop"])
            ri = (row["rodilla_izq_x_crop"], row["rodilla_izq_y_crop"])
            rd = (row["rodilla_der_x_crop"], row["rodilla_der_y_crop"])
            ti = (row["tobillo_izq_x_crop"], row["tobillo_izq_y_crop"])
            td = (row["tobillo_der_x_crop"], row["tobillo_der_y_crop"])

            row["angulo_rodilla_izq"] = calcular_angulo(ci, ri, ti)
            row["angulo_rodilla_der"] = calcular_angulo(cd, rd, td)
            row["angulo_cadera_izq"] = calcular_angulo(hi, ci, ri)
            row["angulo_cadera_der"] = calcular_angulo(hd, cd, rd)

            hombro_medio = ((hi[0] + hd[0]) / 2, (hi[1] + hd[1]) / 2)
            cadera_media = ((ci[0] + cd[0]) / 2, (ci[1] + cd[1]) / 2)

            row["inclinacion_tronco"] = inclinacion_segmento(hombro_medio, cadera_media)


            row["cadera_media_x_original"] = np.nanmean([
                row["cadera_izq_x_original"],
                row["cadera_der_x_original"]
            ])
            row["cadera_media_y_original"] = np.nanmean([
                row["cadera_izq_y_original"],
                row["cadera_der_y_original"]
            ])

            row["dist_pose_a_crop_px"] = np.sqrt(
                (row["cadera_media_x_original"] - centro_crop_x) ** 2
                + (row["cadera_media_y_original"] - centro_crop_y) ** 2
            )

            if row["dist_pose_a_crop_px"] <= MAX_DIST_POSE_A_CROP_PX:
                texto_track = f"POSE CERCA CROP: {row['dist_pose_a_crop_px']:.0f}px"
                color_track = (0, 255, 0)
            else:
                texto_track = f"POSE SOSPECHOSA: {row['dist_pose_a_crop_px']:.0f}px"
                color_track = (0, 0, 255)

            cv2.putText(
                annotated,
                texto_track,
                (20, 220),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color_track,
                2
            )

            cv2.putText(
                annotated,
                f"Rod I: {row['angulo_rodilla_izq']:.1f}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

            cv2.putText(
                annotated,
                f"Rod D: {row['angulo_rodilla_der']:.1f}",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

            cv2.putText(
                annotated,
                f"Cad I: {row['angulo_cadera_izq']:.1f}",
                (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

            cv2.putText(
                annotated,
                f"Cad D: {row['angulo_cadera_der']:.1f}",
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

            cv2.putText(
                annotated,
                f"Tronco: {row['inclinacion_tronco']:.1f}",
                (20, 190),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

        else:
            cv2.putText(
                annotated,
                "POSE NO DETECTADA",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        cv2.putText(
            annotated,
            f"Frame: {frame_idx}/{total_frames}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        rows.append(row)
        out_pose.write(annotated)

        frame_idx += 1

    cap.release()
    out_pose.release()
    pose.close()

    df = pd.DataFrame(rows)

    print("Vídeo con esqueleto guardado en:")
    print(OUTPUT_POSE_VIDEO)

    return df


def aplicar_filtro_pose_atleta_por_crop(df):
    """
    Marca qué frames de pose pertenecen probablemente al atleta.

    El problema que corrige es que MediaPipe a veces pierde al saltador
    cerca de la batida o en el vuelo y se engancha al juez o a otra persona.
    Para evitar contaminar ángulos y velocidades, la pose solo se considera
    válida si está cerca del crop de movimiento y mantiene continuidad temporal.
    """

    print("\n=== PASO 2B: FILTRO POSE ATLETA POR CROP/TRACKING ===")

    df = df.copy()

    if not ACTIVAR_FILTRO_POSE_ATLETA_POR_CROP:
        df["pose_valida_atleta"] = df["pose_detected"] == True
        df["motivo_pose_no_valida"] = ""
        return df

    for col in ["cadera_media_x_original", "cadera_media_y_original", "dist_pose_a_crop_px"]:
        if col not in df.columns:
            df[col] = np.nan

    df["pose_valida_atleta"] = False
    df["motivo_pose_no_valida"] = ""

    ultima_x = None
    ultima_y = None
    ultimo_frame_valido = None

    for idx, row in df.iterrows():
        if row.get("pose_detected", False) != True:
            df.loc[idx, "motivo_pose_no_valida"] = "pose_no_detectada"
            continue

        x = row.get("cadera_media_x_original", np.nan)
        y = row.get("cadera_media_y_original", np.nan)
        d_crop = row.get("dist_pose_a_crop_px", np.nan)

        if pd.isna(x) or pd.isna(y):
            df.loc[idx, "motivo_pose_no_valida"] = "cadera_nan"
            continue

        if pd.isna(d_crop) or d_crop > MAX_DIST_POSE_A_CROP_PX:
            df.loc[idx, "motivo_pose_no_valida"] = "pose_lejos_del_crop_atleta"
            continue

        frame = int(row["frame"])

        if ultima_x is None:
            df.loc[idx, "pose_valida_atleta"] = True
            df.loc[idx, "motivo_pose_no_valida"] = ""
            ultima_x = x
            ultima_y = y
            ultimo_frame_valido = frame
            continue

        frames_gap = frame - ultimo_frame_valido
        salto_cadera = np.sqrt((x - ultima_x) ** 2 + (y - ultima_y) ** 2)

        if frames_gap <= MAX_FRAMES_SIN_POSE_ANTES_RESET and salto_cadera > MAX_SALTO_CADERA_TRACKING_PX:
            df.loc[idx, "motivo_pose_no_valida"] = "salto_identidad_pose"
            continue

        df.loc[idx, "pose_valida_atleta"] = True
        df.loc[idx, "motivo_pose_no_valida"] = ""
        ultima_x = x
        ultima_y = y
        ultimo_frame_valido = frame


    df.loc[~df["pose_valida_atleta"], "cadera_media_x_original"] = np.nan
    df.loc[~df["pose_valida_atleta"], "cadera_media_y_original"] = np.nan

    print("Frames con pose detectada:", int((df["pose_detected"] == True).sum()))
    print("Frames pose válida del atleta:", int((df["pose_valida_atleta"] == True).sum()))
    print("Rechazados lejos del crop:", int((df["motivo_pose_no_valida"] == "pose_lejos_del_crop_atleta").sum()))
    print("Rechazados por salto de identidad:", int((df["motivo_pose_no_valida"] == "salto_identidad_pose").sum()))

    return df


def limpiar_angulos(df):
    print("\n=== PASO 3: LIMPIANDO ÁNGULOS ===")

    df = df.copy()

    if "pose_valida_atleta" not in df.columns:
        df["pose_valida_atleta"] = df["pose_detected"] == True

    for col in [
        "cadera_izq_vis", "rodilla_izq_vis", "tobillo_izq_vis",
        "cadera_der_vis", "rodilla_der_vis", "tobillo_der_vis",
        "hombro_izq_vis", "hombro_der_vis"
    ]:
        if col not in df.columns:
            df[col] = np.nan

    df["rodilla_izq_valida"] = (
        (df["pose_valida_atleta"] == True)
        & (df["cadera_izq_vis"] >= VIS_THRESHOLD)
        & (df["rodilla_izq_vis"] >= VIS_THRESHOLD)
        & (df["tobillo_izq_vis"] >= VIS_THRESHOLD)
    )

    df["rodilla_der_valida"] = (
        (df["pose_valida_atleta"] == True)
        & (df["cadera_der_vis"] >= VIS_THRESHOLD)
        & (df["rodilla_der_vis"] >= VIS_THRESHOLD)
        & (df["tobillo_der_vis"] >= VIS_THRESHOLD)
    )

    df["cadera_izq_valida"] = (
        (df["pose_valida_atleta"] == True)
        & (df["hombro_izq_vis"] >= VIS_THRESHOLD)
        & (df["cadera_izq_vis"] >= VIS_THRESHOLD)
        & (df["rodilla_izq_vis"] >= VIS_THRESHOLD)
    )

    df["cadera_der_valida"] = (
        (df["pose_valida_atleta"] == True)
        & (df["hombro_der_vis"] >= VIS_THRESHOLD)
        & (df["cadera_der_vis"] >= VIS_THRESHOLD)
        & (df["rodilla_der_vis"] >= VIS_THRESHOLD)
    )

    df["tronco_valido"] = (
        (df["pose_valida_atleta"] == True)
        & (df["hombro_izq_vis"] >= VIS_THRESHOLD)
        & (df["hombro_der_vis"] >= VIS_THRESHOLD)
        & (df["cadera_izq_vis"] >= VIS_THRESHOLD)
        & (df["cadera_der_vis"] >= VIS_THRESHOLD)
    )

    angle_cols = [
        ("angulo_rodilla_izq", "rodilla_izq_valida"),
        ("angulo_rodilla_der", "rodilla_der_valida"),
        ("angulo_cadera_izq", "cadera_izq_valida"),
        ("angulo_cadera_der", "cadera_der_valida"),
        ("inclinacion_tronco", "tronco_valido"),
    ]

    for angle_col, valid_col in angle_cols:
        if angle_col not in df.columns:
            df[angle_col] = np.nan

        filtered = angle_col + "_filtrado"
        smooth = angle_col + "_suavizado"

        df[filtered] = df[angle_col].where(df[valid_col], np.nan)

        diff = df[filtered].diff().abs()
        df.loc[diff > MAX_JUMP_DEGREES, filtered] = np.nan

        df[smooth] = (
            df[filtered]
            .interpolate(limit=3, limit_direction="both")
            .rolling(window=SMOOTH_WINDOW, center=True, min_periods=1)
            .mean()
        )

    df["score_angulos"] = (
        df["rodilla_izq_valida"].astype(int)
        + df["rodilla_der_valida"].astype(int)
        + df["cadera_izq_valida"].astype(int)
        + df["cadera_der_valida"].astype(int)
        + df["tronco_valido"].astype(int)
    )


    if "cadera_media_x_original" in df.columns:
        df["cadera_media_x_suavizada"] = (
            df["cadera_media_x_original"]
            .interpolate(limit=3, limit_direction="both")
            .rolling(window=VENTANA_SUAVIZADO_CADERA, center=True, min_periods=1)
            .mean()
        )

    if "cadera_media_y_original" in df.columns:
        df["cadera_media_y_suavizada"] = (
            df["cadera_media_y_original"]
            .interpolate(limit=3, limit_direction="both")
            .rolling(window=VENTANA_SUAVIZADO_CADERA, center=True, min_periods=1)
            .mean()
        )

    df.to_csv(OUTPUT_KEYPOINTS, index=False, encoding="utf-8-sig")

    print("Keypoints + ángulos guardados:", OUTPUT_KEYPOINTS)
    print("Frames con tronco válido:", df["tronco_valido"].sum(), "/", len(df))
    print("Frames con score 5:", (df["score_angulos"] == 5).sum(), "/", len(df))

    return df


def seleccionar_ultimos_apoyos_con_cadencia(apoyos_df, n_apoyos=6):
    """
    Selecciona los últimos apoyos evitando:
    - duplicados demasiado cercanos en frames
    - duplicados demasiado cercanos en distancia a tabla
    - doble detección del mismo contacto de batida

    La clave es que dos apoyos consecutivos no pueden avanzar muy poco
    hacia la tabla. Si el avance es demasiado pequeño, normalmente es el
    mismo apoyo detectado al entrar y al salir del contacto.
    """

    if apoyos_df.empty:
        return apoyos_df

    apoyos_df = apoyos_df.sort_values("frame").reset_index(drop=True)

    seleccionados = []


    ultimo = apoyos_df.iloc[-1].to_dict()
    seleccionados.append(ultimo)

    frame_actual = int(ultimo["frame"])
    dist_actual = float(ultimo["distancia_a_tabla_m"])

    for i in range(len(apoyos_df) - 2, -1, -1):
        candidato = apoyos_df.iloc[i].to_dict()

        frame_candidato = int(candidato["frame"])
        dist_candidato = float(candidato["distancia_a_tabla_m"])

        gap_frames = frame_actual - frame_candidato
        gap_distancia = dist_candidato - dist_actual


        if len(seleccionados) == 1:
            min_gap_frames = MIN_GAP_ULTIMO_BATIDA
            min_gap_distancia = MIN_DISTANCIA_ULTIMO_BATIDA_M
        else:
            min_gap_frames = MIN_GAP_APOYOS
            min_gap_distancia = MIN_DISTANCIA_ENTRE_APOYOS_M


        if gap_frames < min_gap_frames:
            continue


        if gap_distancia < min_gap_distancia:
            continue


        if min_gap_frames <= gap_frames <= MAX_GAP_APOYOS:
            seleccionados.append(candidato)
            frame_actual = frame_candidato
            dist_actual = dist_candidato


        elif gap_frames > MAX_GAP_APOYOS:
            seleccionados.append(candidato)
            frame_actual = frame_candidato
            dist_actual = dist_candidato

        if len(seleccionados) >= n_apoyos:
            break

    seleccionados = list(reversed(seleccionados))

    return pd.DataFrame(seleccionados)

def rescatar_apoyo_prebatida(apoyos_df, cand_original):
    """
    Intenta corregir un fallo frecuente:
    que falte el apoyo inmediatamente anterior a la batida.

    Si entre el último apoyo detectado y la batida hay demasiados frames,
    busca un candidato intermedio cerca del frame esperado.
    """

    if not ACTIVAR_RESCATE_APOYO_PREBATIDA:
        return apoyos_df

    if apoyos_df.empty or len(apoyos_df) < 2:
        return apoyos_df

    apoyos_df = apoyos_df.sort_values("frame").reset_index(drop=True)
    cand_original = cand_original.sort_values("frame").reset_index(drop=True)

    batida = apoyos_df.iloc[-1]
    anterior = apoyos_df.iloc[-2]

    frame_batida = int(batida["frame"])
    frame_anterior = int(anterior["frame"])

    gap = frame_batida - frame_anterior

    if gap <= GAP_SOSPECHA_FALTA_PREBATIDA:
        return apoyos_df

    print("\n=== POSIBLE APOYO PREBATIDA PERDIDO ===")
    print("Frame anterior detectado:", frame_anterior)
    print("Frame batida:", frame_batida)
    print("Gap:", gap)

    frame_esperado = frame_batida - FRAME_ESPERADO_PREBATIDA

    candidatos = cand_original[
        (cand_original["frame"] >= frame_esperado - MARGEN_BUSQUEDA_PREBATIDA)
        & (cand_original["frame"] <= frame_esperado + MARGEN_BUSQUEDA_PREBATIDA)
    ].copy()

    if candidatos.empty:
        print("No se encontró candidato intermedio para rescatar.")
        return apoyos_df


    frames_usados = set(apoyos_df["frame"].astype(int).tolist())

    candidatos = candidatos[
        ~candidatos["frame"].astype(int).isin(frames_usados)
    ].copy()

    if candidatos.empty:
        print("Los candidatos intermedios ya estaban usados.")
        return apoyos_df


    candidatos["dist_frame_esperado"] = (
        candidatos["frame"].astype(int) - frame_esperado
    ).abs()

    candidatos["score_rescate"] = (
        candidatos["y"] * 1.0
        + candidatos["vis"] * 80.0
        - candidatos["dist_frame_esperado"] * 3.0
    )

    elegido = candidatos.sort_values("score_rescate", ascending=False).iloc[0].to_dict()

    print("Apoyo rescatado:")
    print(
        "frame =", int(elegido["frame"]),
        "| lado =", elegido["lado"],
        "| distancia =", round(elegido["distancia_a_tabla_m"], 2)
    )

    apoyos_df = pd.concat(
        [apoyos_df, pd.DataFrame([elegido])],
        ignore_index=True
    )

    apoyos_df = apoyos_df.sort_values("frame").reset_index(drop=True)


    apoyos_df = apoyos_df.tail(N_APOYOS).reset_index(drop=True)

    return apoyos_df


def secuencia_sospechosa(apoyos_df):
    """
    Detecta si la secuencia final de apoyos parece contaminada.

    Casos típicos:
    - la batida no está cerca de la tabla
    - la distancia a tabla vuelve a aumentar muchísimo
    - se mezclan dos atletas o dos carreras distintas
    """

    if apoyos_df.empty or len(apoyos_df) < 2:
        return True

    apoyos_df = apoyos_df.sort_values("frame").reset_index(drop=True)


    batida = apoyos_df.iloc[-1]
    dist_batida = batida["distancia_a_tabla_m"]

    if pd.isna(dist_batida):
        return True

    if not (BATIDA_DIST_MIN_M <= dist_batida <= BATIDA_DIST_MAX_M):
        print("\nSecuencia sospechosa: la batida no está cerca de la tabla.")
        print("Distancia batida:", dist_batida)
        return True


    distancias = apoyos_df["distancia_a_tabla_m"].values

    for i in range(1, len(distancias)):

        if distancias[i] > distancias[i - 1] + 1.00:
            print("\nSecuencia sospechosa: la distancia a tabla vuelve a aumentar.")
            print("Distancias:", distancias)
            return True

    return False


def reconstruir_apoyos_desde_batida(cand_original):
    """
    Reconstruye la secuencia anclándola a una batida cercana a la tabla.
    Sirve para evitar coger otra atleta/persona después del salto real.
    """

    print("\n=== RECONSTRUYENDO SECUENCIA DESDE BATIDA CERCANA A TABLA ===")

    if cand_original.empty:
        return pd.DataFrame()

    cand = cand_original.copy()
    cand = cand.sort_values("frame").reset_index(drop=True)


    cand_batida = cand[
        (cand["distancia_a_tabla_m"] >= BATIDA_DIST_MIN_M)
        & (cand["distancia_a_tabla_m"] <= BATIDA_DIST_MAX_M)
    ].copy()

    if cand_batida.empty:
        print("No hay candidatos de batida cerca de la tabla.")
        return pd.DataFrame()

    mejores_secuencias = []

    for _, batida in cand_batida.iterrows():
        secuencia = [batida.to_dict()]
        actual = batida.to_dict()

        while len(secuencia) < N_APOYOS:
            frame_actual = int(actual["frame"])
            dist_actual = float(actual["distancia_a_tabla_m"])
            lado_actual = actual["lado"]

            previos = cand[
                (cand["frame"] < frame_actual - MIN_GAP_APOYOS)
                & (cand["frame"] >= frame_actual - MAX_GAP_APOYOS)
                & (cand["distancia_a_tabla_m"] > dist_actual + MIN_LONGITUD_RECONSTRUIR_M)
                & (cand["distancia_a_tabla_m"] < dist_actual + MAX_LONGITUD_RECONSTRUIR_M)
            ].copy()

            if previos.empty:
                break

            previos["gap_frames"] = frame_actual - previos["frame"].astype(int)
            previos["longitud_estim_m"] = previos["distancia_a_tabla_m"] - dist_actual

            previos["bonus_alternancia"] = np.where(
                previos["lado"] != lado_actual,
                20.0,
                -15.0
            )

            previos["penalizacion_gap"] = (
                previos["gap_frames"] - GAP_IDEAL_FRAMES
            ).abs() * 0.50

            previos["penalizacion_longitud"] = (
                previos["longitud_estim_m"] - LONGITUD_IDEAL_PASO_M
            ).abs() * 12.0

            previos["score_reconstruccion"] = (
                previos["y"] * 0.40
                + previos["vis"] * 80.0
                + previos["bonus_alternancia"]
                - previos["penalizacion_gap"]
                - previos["penalizacion_longitud"]
            )

            elegido = previos.sort_values(
                "score_reconstruccion",
                ascending=False
            ).iloc[0].to_dict()

            secuencia.append(elegido)
            actual = elegido

        secuencia_ordenada = list(reversed(secuencia))
        sec_df = pd.DataFrame(secuencia_ordenada)


        num_apoyos = len(sec_df)

        if num_apoyos >= 2:
            gaps = sec_df["frame"].diff().dropna()
            dist = sec_df["distancia_a_tabla_m"].values

            penalizacion_gaps_raros = 0
            for g in gaps:
                if g < MIN_GAP_APOYOS or g > MAX_GAP_APOYOS:
                    penalizacion_gaps_raros += 30

            penalizacion_distancias = 0
            for i in range(1, len(dist)):
                if dist[i] > dist[i - 1] + 1.00:
                    penalizacion_distancias += 50
        else:
            penalizacion_gaps_raros = 100
            penalizacion_distancias = 100

        dist_batida = abs(float(sec_df.iloc[-1]["distancia_a_tabla_m"]))

        score_total = (
            num_apoyos * 200
            - dist_batida * 100
            - penalizacion_gaps_raros
            - penalizacion_distancias
        )

        mejores_secuencias.append({
            "score_total": score_total,
            "num_apoyos": num_apoyos,
            "secuencia": sec_df
        })

    if not mejores_secuencias:
        print("No se pudo reconstruir ninguna secuencia.")
        return pd.DataFrame()

    mejores_secuencias = sorted(
        mejores_secuencias,
        key=lambda x: (x["num_apoyos"], x["score_total"]),
        reverse=True
    )

    mejor = mejores_secuencias[0]["secuencia"].copy()
    mejor = mejor.sort_values("frame").reset_index(drop=True)


    mejor = mejor.tail(N_APOYOS).reset_index(drop=True)

    print("Secuencia reconstruida:")
    print(mejor[["frame", "lado", "distancia_a_tabla_m", "y", "vis"]])

    return mejor


def filtrar_apoyos_en_aire(apoyos_df, cand_original=None):
    """
    Filtro SUAVE de apoyos en el aire.

    Importante:
    - NO elimina un apoyo sospechoso si no puede sustituirlo por otro candidato mejor.
    - Evita que la secuencia se quede con 3, 4 o 5 apoyos y se renombre mal.
    - Si encuentra un candidato alternativo entre el apoyo anterior y el siguiente,
      reemplaza el apoyo sospechoso.
    - Si no encuentra alternativa clara, mantiene el apoyo y lo deja para revisión.
    """

    if not ACTIVAR_FILTRO_APOYOS_EN_AIRE:
        return apoyos_df

    if apoyos_df.empty or len(apoyos_df) < 3:
        return apoyos_df

    apoyos_df = apoyos_df.sort_values("frame").reset_index(drop=True).copy()

    if cand_original is None or cand_original.empty:
        print("\nFiltro aire: no hay cand_original. No se eliminan apoyos, solo se conserva la secuencia.")
        return apoyos_df

    cand_original = cand_original.sort_values("frame").reset_index(drop=True).copy()


    y_mediana = apoyos_df["y"].median()
    y_minima_aceptable = y_mediana - TOLERANCIA_Y_APOYO_PX

    sospechosos = set()

    for i in range(len(apoyos_df)):

        if i == len(apoyos_df) - 1:
            continue

        y = apoyos_df.loc[i, "y"]

        if y < y_minima_aceptable:
            print("\n=== APOYO SOSPECHOSO POR ALTURA ===")
            print("Índice:", i)
            print("Frame:", int(apoyos_df.loc[i, "frame"]))
            print("y apoyo:", round(float(y), 2))
            print("y mediana:", round(float(y_mediana), 2))
            sospechosos.add(i)

    for i in range(len(apoyos_df) - 1):
        d1 = apoyos_df.loc[i, "distancia_a_tabla_m"]
        d2 = apoyos_df.loc[i + 1, "distancia_a_tabla_m"]
        t1 = apoyos_df.loc[i, "time_s"]
        t2 = apoyos_df.loc[i + 1, "time_s"]

        tiempo = abs(t2 - t1)

        if tiempo <= 0:
            continue

        longitud = abs(d1 - d2)
        velocidad = longitud / tiempo

        if velocidad > MAX_VELOCIDAD_PLAUSIBLE_M_S:
            y_i = apoyos_df.loc[i, "y"]
            y_next = apoyos_df.loc[i + 1, "y"]


            if i + 1 == len(apoyos_df) - 1:
                candidato_sospechoso = i
            else:
                candidato_sospechoso = i if y_i < y_next else i + 1

            print("\n=== APOYO SOSPECHOSO POR VELOCIDAD ===")
            print("Frames:", int(apoyos_df.loc[i, "frame"]), "->", int(apoyos_df.loc[i + 1, "frame"]))
            print("Velocidad:", round(float(velocidad), 2), "m/s")
            print("Candidato sospechoso:", int(apoyos_df.loc[candidato_sospechoso, "frame"]))
            sospechosos.add(candidato_sospechoso)

    if not sospechosos:
        return apoyos_df


    apoyos_corregidos = apoyos_df.copy()
    frames_usados = set(apoyos_corregidos["frame"].astype(int).tolist())

    for idx_sospechoso in sorted(sospechosos):

        if idx_sospechoso <= 0 or idx_sospechoso >= len(apoyos_corregidos) - 1:
            print("No se reemplaza índice", idx_sospechoso, "porque no tiene dos vecinos claros.")
            continue

        anterior = apoyos_corregidos.loc[idx_sospechoso - 1]
        actual = apoyos_corregidos.loc[idx_sospechoso]
        siguiente = apoyos_corregidos.loc[idx_sospechoso + 1]

        frame_min = int(anterior["frame"]) + MIN_GAP_APOYOS
        frame_max = int(siguiente["frame"]) - MIN_GAP_APOYOS


        if frame_min > frame_max:
            frame_min = int(anterior["frame"]) + 10
            frame_max = int(siguiente["frame"]) - 10

        dist_anterior = float(anterior["distancia_a_tabla_m"])
        dist_siguiente = float(siguiente["distancia_a_tabla_m"])


        dist_max = dist_anterior - 0.35
        dist_min = dist_siguiente + 0.35

        candidatos = cand_original[
            (cand_original["frame"].astype(int) >= frame_min)
            & (cand_original["frame"].astype(int) <= frame_max)
            & (~cand_original["frame"].astype(int).isin(frames_usados))
            & (cand_original["distancia_a_tabla_m"] <= dist_max)
            & (cand_original["distancia_a_tabla_m"] >= dist_min)
        ].copy()


        if not candidatos.empty:
            candidatos = candidatos[candidatos["y"] >= y_minima_aceptable].copy()

        if candidatos.empty:
            print("\nNo se encontró reemplazo claro para frame", int(actual["frame"]))
            print("Se mantiene el apoyo para no romper la secuencia.")
            continue


        frame_esperado = (int(anterior["frame"]) + int(siguiente["frame"])) / 2
        dist_esperada = (dist_anterior + dist_siguiente) / 2

        candidatos["penalizacion_frame"] = (candidatos["frame"].astype(int) - frame_esperado).abs()
        candidatos["penalizacion_distancia"] = (candidatos["distancia_a_tabla_m"] - dist_esperada).abs()

        candidatos["score_reemplazo"] = (
            candidatos["y"] * 1.0
            + candidatos["vis"] * 80.0
            - candidatos["penalizacion_frame"] * 2.0
            - candidatos["penalizacion_distancia"] * 20.0
        )

        elegido = candidatos.sort_values("score_reemplazo", ascending=False).iloc[0]

        print("\n=== REEMPLAZO DE APOYO SOSPECHOSO ===")
        print("Sospechoso frame:", int(actual["frame"]), "-> nuevo frame:", int(elegido["frame"]))
        print("Distancia:", round(float(actual["distancia_a_tabla_m"]), 2), "->", round(float(elegido["distancia_a_tabla_m"]), 2))
        print("y:", round(float(actual["y"]), 2), "->", round(float(elegido["y"]), 2))

        frames_usados.discard(int(actual["frame"]))
        frames_usados.add(int(elegido["frame"]))

        for col in apoyos_corregidos.columns:
            if col in elegido.index:
                apoyos_corregidos.loc[idx_sospechoso, col] = elegido[col]

    apoyos_corregidos = apoyos_corregidos.sort_values("frame").reset_index(drop=True)

    print("\nSecuencia tras filtro suave de apoyos en el aire:")
    print(apoyos_corregidos[["frame", "lado", "distancia_a_tabla_m", "y", "vis"]])

    return apoyos_corregidos

def detectar_apoyos_automaticos(df, calibracion):
    print("\n=== PASO 4: DETECTANDO APOYOS AUTOMÁTICOS ESTABLES + CADENCIA ===")

    x_tabla = calibracion["x_tabla"]
    escala = calibracion["escala_m_por_pixel"]

    VIS_APOYO = 0.45

    candidatos = []

    for _, row in df.iterrows():
        if "pose_valida_atleta" in df.columns:
            if row.get("pose_valida_atleta", False) != True:
                continue
        else:
            if not row.get("pose_detected", False):
                continue

        tobillos = []

        for lado in ["izq", "der"]:
            x_col = f"tobillo_{lado}_x_original"
            y_col = f"tobillo_{lado}_y_original"
            v_col = f"tobillo_{lado}_vis"

            if x_col not in df.columns or y_col not in df.columns or v_col not in df.columns:
                continue

            x = row[x_col]
            y = row[y_col]
            vis = row[v_col]

            if pd.isna(x) or pd.isna(y) or pd.isna(vis):
                continue

            if vis < VIS_APOYO:
                continue

            distancia_a_tabla_m = (x_tabla - x) * escala

            if not (MARGEN_DESPUES_TABLA_M <= distancia_a_tabla_m <= DIST_MAX_ANTES_TABLA_M):
                continue

            tobillos.append({
                "frame": int(row["frame"]),
                "time_s": row["time_s"],
                "lado": lado,
                "x": x,
                "y": y,
                "vis": vis,
                "distancia_a_tabla_m": distancia_a_tabla_m
            })

        if not tobillos:
            continue

        tobillo_bajo = max(tobillos, key=lambda p: p["y"])
        candidatos.append(tobillo_bajo)

    if not candidatos:
        print("No hay candidatos de apoyo.")
        return pd.DataFrame()

    cand_original = pd.DataFrame(candidatos)
    cand_original = cand_original.sort_values("frame").reset_index(drop=True)

    debug_candidatos_path = RESULTS_DIR / f"candidatos_apoyo_{STEM}.csv"
    cand_original.to_csv(debug_candidatos_path, index=False, encoding="utf-8-sig")
    print("Candidatos de apoyo guardados en:")
    print(debug_candidatos_path)

    def detectar_con_quantile(cand_base, q):
        cand = cand_base.copy()

        umbral_y = cand["y"].quantile(q)
        cand = cand[cand["y"] >= umbral_y].copy()
        cand = cand.sort_values("frame").reset_index(drop=True)

        apoyos = []

        for i in range(1, len(cand) - 1):
            y_prev = cand.loc[i - 1, "y"]
            y_curr = cand.loc[i, "y"]
            y_next = cand.loc[i + 1, "y"]

            if y_curr >= y_prev and y_curr >= y_next:
                frame_curr = int(cand.loc[i, "frame"])

                if apoyos:
                    diferencia_frames = frame_curr - int(apoyos[-1]["frame"])

                    if diferencia_frames < MIN_GAP_APOYOS:
                        if y_curr > apoyos[-1]["y"]:
                            apoyos[-1] = cand.loc[i].to_dict()
                        continue

                apoyos.append(cand.loc[i].to_dict())

        return pd.DataFrame(apoyos)

    apoyos_df = pd.DataFrame()

    for q in [0.70, 0.60, 0.50, 0.40, 0.30]:
        temp = detectar_con_quantile(cand_original, q)

        print(f"Intento quantile {q}: {len(temp)} apoyos")

        if len(temp) >= N_APOYOS:
            apoyos_df = temp
            break

        if len(temp) > len(apoyos_df):
            apoyos_df = temp

    if len(apoyos_df) < N_APOYOS:
        print(f"Solo se detectaron {len(apoyos_df)} apoyos por máximos locales.")
        print("Forzando selección de candidatos bajos y separados...")

        cand = cand_original.copy()

        umbral_y = cand["y"].quantile(0.35)
        cand = cand[cand["y"] >= umbral_y].copy()

        cand["score_apoyo"] = (
            cand["y"] * 1.0
            + cand["vis"] * 80.0
            - cand["distancia_a_tabla_m"].abs() * 2.0
        )

        cand = cand.sort_values("score_apoyo", ascending=False).reset_index(drop=True)

        seleccionados = []

        for _, row in cand.iterrows():
            frame = int(row["frame"])

            demasiado_cerca = False

            for sel in seleccionados:
                if abs(frame - int(sel["frame"])) < MIN_GAP_APOYOS:
                    demasiado_cerca = True
                    break

            if demasiado_cerca:
                continue

            seleccionados.append(row.to_dict())

            if len(seleccionados) >= N_APOYOS:
                break

        if seleccionados:
            apoyos_df = pd.DataFrame(seleccionados)

    if apoyos_df.empty:
        print("No se detectaron apoyos.")
        return apoyos_df

    apoyos_df = apoyos_df.sort_values("frame").reset_index(drop=True)

    apoyos_df = seleccionar_ultimos_apoyos_con_cadencia(
        apoyos_df,
        n_apoyos=N_APOYOS
    )

    apoyos_df = apoyos_df.sort_values("frame").reset_index(drop=True)


    apoyos_df = rescatar_apoyo_prebatida(apoyos_df, cand_original)

    apoyos_df = apoyos_df.sort_values("frame").reset_index(drop=True)


    apoyos_df = seleccionar_ultimos_apoyos_con_cadencia(
        apoyos_df,
        n_apoyos=N_APOYOS
    )

    apoyos_df = apoyos_df.sort_values("frame").reset_index(drop=True)


    if ACTIVAR_REPARACION_SECUENCIA and secuencia_sospechosa(apoyos_df):
        apoyos_reconstruidos = reconstruir_apoyos_desde_batida(cand_original)

        if not apoyos_reconstruidos.empty:
            apoyos_df = apoyos_reconstruidos.copy()
            apoyos_df = apoyos_df.sort_values("frame").reset_index(drop=True)


            apoyos_df = rescatar_apoyo_prebatida(apoyos_df, cand_original)
            apoyos_df = apoyos_df.sort_values("frame").reset_index(drop=True)

            apoyos_df = seleccionar_ultimos_apoyos_con_cadencia(
                apoyos_df,
                n_apoyos=N_APOYOS
            )

            apoyos_df = apoyos_df.sort_values("frame").reset_index(drop=True)


    apoyos_df = filtrar_apoyos_en_aire(apoyos_df, cand_original)
    apoyos_df = apoyos_df.sort_values("frame").reset_index(drop=True)

    nombres = ["apoyo_-5", "apoyo_-4", "apoyo_-3", "apoyo_-2", "apoyo_-1", "batida"]
    apoyos_df["evento"] = nombres[-len(apoyos_df):]

    print("Apoyos detectados:")
    print(apoyos_df[["evento", "frame", "lado", "distancia_a_tabla_m", "y", "vis"]])

    return apoyos_df


def elegir_fila_angulo(df, frame_objetivo, ventana=12):
    sub = df[
        (df["frame"] >= frame_objetivo - ventana)
        & (df["frame"] <= frame_objetivo + ventana)
    ].copy()

    if sub.empty:
        return None

    sub["distancia_evento"] = (sub["frame"] - frame_objetivo).abs()

    sub = sub.sort_values(
        by=["score_angulos", "distancia_evento"],
        ascending=[False, True]
    )

    return sub.iloc[0]


def obtener_angulo_lado(row, lado):
    if lado == "izq":
        return (
            row.get("angulo_rodilla_izq_suavizado", np.nan),
            row.get("angulo_cadera_izq_suavizado", np.nan)
        )

    if lado == "der":
        return (
            row.get("angulo_rodilla_der_suavizado", np.nan),
            row.get("angulo_cadera_der_suavizado", np.nan)
        )

    return np.nan, np.nan


def guardar_debug_evento(evento_row, angle_row):
    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        return

    frame_idx = int(evento_row["frame"])

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return

    x = int(evento_row["x"])
    y = int(evento_row["y"])
    evento = evento_row["evento"]

    cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)

    texto = (
        f"{evento} | frame={frame_idx} | lado={evento_row['lado']} | "
        f"dist={evento_row['distancia_a_tabla_m']:.2f}m"
    )

    cv2.putText(
        frame,
        texto,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    if angle_row is not None:
        cv2.putText(
            frame,
            f"score_angulos={angle_row['score_angulos']}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    out_path = DEBUG_DIR / f"{evento}_frame_{frame_idx}.jpg"
    cv2.imwrite(str(out_path), frame)


def validar_secuencia_apoyos(res):
    res = res.copy()

    alternancia = []
    longitud_ok = []
    velocidad_ok = []
    frames_ok = []
    distancia_entre_apoyos_ok = []

    for i in range(len(res)):

        if i == 0:
            alternancia.append(True)
            frames_ok.append(True)
            distancia_entre_apoyos_ok.append(True)
        else:
            lado_actual = res.loc[i, "lado_apoyo"]
            lado_anterior = res.loc[i - 1, "lado_apoyo"]
            alternancia.append(lado_actual != lado_anterior)

            gap_frames = int(res.loc[i, "frame_evento"]) - int(res.loc[i - 1, "frame_evento"])

            if i == len(res) - 1:
                frames_ok.append(MIN_GAP_ULTIMO_BATIDA <= gap_frames <= MAX_GAP_APOYOS)
            else:
                frames_ok.append(MIN_GAP_APOYOS <= gap_frames <= MAX_GAP_APOYOS)

            dist_anterior = float(res.loc[i - 1, "distancia_a_tabla_m"])
            dist_actual = float(res.loc[i, "distancia_a_tabla_m"])
            gap_distancia = dist_anterior - dist_actual

            if i == len(res) - 1:
                distancia_entre_apoyos_ok.append(gap_distancia >= MIN_DISTANCIA_ULTIMO_BATIDA_M)
            else:
                distancia_entre_apoyos_ok.append(gap_distancia >= MIN_DISTANCIA_ENTRE_APOYOS_M)

        long_paso = res.loc[i, "longitud_hasta_siguiente_m"] if "longitud_hasta_siguiente_m" in res.columns else np.nan

        if pd.isna(long_paso):
            longitud_ok.append(True)
        else:
            longitud_ok.append(0.80 <= long_paso <= 3.00)

        vel = res.loc[i, "velocidad_paso_m_s"] if "velocidad_paso_m_s" in res.columns else np.nan

        if pd.isna(vel):
            velocidad_ok.append(True)
        else:
            velocidad_ok.append(5.0 <= vel <= 12.5)

    res["alternancia_correcta"] = alternancia
    res["separacion_frames_ok"] = frames_ok
    res["distancia_entre_apoyos_ok"] = distancia_entre_apoyos_ok
    res["longitud_paso_ok"] = longitud_ok
    res["velocidad_paso_ok"] = velocidad_ok

    calidad = []

    for i in range(len(res)):
        if (
            res.loc[i, "alternancia_correcta"]
            and res.loc[i, "separacion_frames_ok"]
            and res.loc[i, "distancia_entre_apoyos_ok"]
            and res.loc[i, "longitud_paso_ok"]
            and res.loc[i, "velocidad_paso_ok"]
        ):
            calidad.append("OK")
        else:
            calidad.append("DUDOSO")

    res["calidad_apoyo"] = calidad

    if (res["calidad_apoyo"] == "DUDOSO").any():
        res["calidad_salto"] = "REVISAR"
    else:
        res["calidad_salto"] = "OK"

    return res


def calcular_velocidad_entrada_batida(res):
    """
    Calcula la velocidad de entrada a la batida a partir de las velocidades
    entre los últimos apoyos ya detectados por el pipeline bueno.
    """
    out = {}

    eventos = res.sort_values("frame_evento").reset_index(drop=True).copy()

    velocidades = pd.to_numeric(
        eventos.loc[eventos["evento"] != "batida", "velocidad_paso_m_s"],
        errors="coerce"
    ).dropna().values

    out["velocidad_entrada_ultimo_tramo_m_s"] = velocidades[-1] if len(velocidades) >= 1 else np.nan
    out["velocidad_entrada_media_2tramos_m_s"] = np.mean(velocidades[-2:]) if len(velocidades) >= 2 else np.nan
    out["velocidad_entrada_media_3tramos_m_s"] = np.mean(velocidades[-3:]) if len(velocidades) >= 3 else np.nan


    if pd.notna(out["velocidad_entrada_media_3tramos_m_s"]):
        out["velocidad_entrada_principal_m_s"] = out["velocidad_entrada_media_3tramos_m_s"]
        out["criterio_velocidad_entrada"] = "media_3_tramos"
    elif pd.notna(out["velocidad_entrada_media_2tramos_m_s"]):
        out["velocidad_entrada_principal_m_s"] = out["velocidad_entrada_media_2tramos_m_s"]
        out["criterio_velocidad_entrada"] = "media_2_tramos"
    else:
        out["velocidad_entrada_principal_m_s"] = out["velocidad_entrada_ultimo_tramo_m_s"]
        out["criterio_velocidad_entrada"] = "ultimo_tramo"


    try:
        apoyo_m1 = eventos[eventos["evento"] == "apoyo_-1"].iloc[0]
        batida = eventos[eventos["evento"] == "batida"].iloc[0]

        d1 = float(apoyo_m1["distancia_a_tabla_m"])
        d2 = float(batida["distancia_a_tabla_m"])
        t1 = float(apoyo_m1["time_s"])
        t2 = float(batida["time_s"])

        out["velocidad_entrada_directa_apoyo1_batida_m_s"] = abs(d1 - d2) / abs(t2 - t1) if abs(t2 - t1) > 0 else np.nan
    except Exception:
        out["velocidad_entrada_directa_apoyo1_batida_m_s"] = np.nan

    return out


def estimar_velocidad_salida_batida(df, frame_batida, calibracion, ventana_frames):
    """
    Estima la velocidad de salida tras la batida usando solo frames donde la
    pose sigue perteneciendo al atleta. Si MediaPipe se engancha al juez,
    esos frames no entran en el cálculo.
    """
    escala = calibracion["escala_m_por_pixel"]

    sub = df[
        (df["frame"] >= frame_batida)
        & (df["frame"] <= frame_batida + ventana_frames)
    ].copy()

    if "pose_valida_atleta" in sub.columns:
        sub = sub[sub["pose_valida_atleta"] == True].copy()

    necesarias = ["time_s", "cadera_media_x_suavizada", "cadera_media_y_suavizada"]

    for col in necesarias:
        if col not in sub.columns:
            return np.nan, np.nan, np.nan, 0, "sin_columnas_cadera"

    sub = sub.dropna(subset=necesarias).copy()

    if len(sub) < MIN_FRAMES_VALIDOS_SALIDA:
        return np.nan, np.nan, np.nan, len(sub), "pose_atleta_insuficiente_postbatida"

    t = sub["time_s"].values.astype(float)
    t0 = t - t[0]

    if np.max(t0) == 0:
        return np.nan, np.nan, np.nan, len(sub), "tiempo_insuficiente"

    x_m = sub["cadera_media_x_suavizada"].values.astype(float) * escala
    y_m = sub["cadera_media_y_suavizada"].values.astype(float) * escala

    vx = np.polyfit(t0, x_m, 1)[0]
    vy = np.polyfit(t0, y_m, 1)[0]

    v_horizontal = abs(vx)
    v_vertical = abs(vy)
    v_resultante = np.sqrt(v_horizontal ** 2 + v_vertical ** 2)

    return v_horizontal, v_vertical, v_resultante, len(sub), "OK"

def calcular_variables_velocidad_batida(res, df, calibracion):
    """
    Calcula todas las variables nuevas:
    - velocidad de entrada a batida
    - velocidad horizontal de salida
    - velocidad resultante de salida
    - pérdida de velocidad horizontal
    - ratio salida/entrada
    """
    out = {}
    out.update(calcular_velocidad_entrada_batida(res))

    try:
        frame_batida = int(res.loc[res["evento"] == "batida", "frame_evento"].iloc[0])
        time_batida = float(res.loc[res["evento"] == "batida", "time_s"].iloc[0])
    except Exception:
        frame_batida = int(res.sort_values("frame_evento").iloc[-1]["frame_evento"])
        time_batida = float(res.sort_values("frame_evento").iloc[-1]["time_s"])

    out["frame_batida_velocidad"] = frame_batida
    out["time_batida_velocidad_s"] = time_batida

    v_entrada = out.get("velocidad_entrada_principal_m_s", np.nan)

    for ventana in VENTANAS_SALIDA_FRAMES:
        vh, vv, vr, n, estado = estimar_velocidad_salida_batida(
            df=df,
            frame_batida=frame_batida,
            calibracion=calibracion,
            ventana_frames=ventana
        )

        out[f"velocidad_salida_horizontal_{ventana}f_m_s"] = vh
        out[f"velocidad_salida_vertical_{ventana}f_m_s"] = vv
        out[f"velocidad_salida_resultante_{ventana}f_m_s"] = vr
        out[f"num_frames_salida_{ventana}f"] = n
        out[f"estado_velocidad_salida_{ventana}f"] = estado

        if pd.notna(v_entrada) and pd.notna(vh) and v_entrada != 0:
            out[f"perdida_velocidad_horizontal_batida_{ventana}f_m_s"] = v_entrada - vh
            out[f"ratio_salida_entrada_{ventana}f"] = vh / v_entrada
        else:
            out[f"perdida_velocidad_horizontal_batida_{ventana}f_m_s"] = np.nan
            out[f"ratio_salida_entrada_{ventana}f"] = np.nan

    return out


def crear_tabla_final(df, apoyos_df, calibracion):
    filas = []

    for _, apoyo in apoyos_df.iterrows():
        angle_row = elegir_fila_angulo(df, int(apoyo["frame"]), ventana=12)

        if angle_row is None:
            ang_rodilla = np.nan
            ang_cadera = np.nan
            inc_tronco = np.nan
            score = np.nan
            frame_angulo = np.nan
        else:
            ang_rodilla, ang_cadera = obtener_angulo_lado(angle_row, apoyo["lado"])
            inc_tronco = angle_row.get("inclinacion_tronco_suavizado", np.nan)
            score = angle_row.get("score_angulos", np.nan)
            frame_angulo = int(angle_row["frame"])

        filas.append({
            "video": VIDEO_NAME,
            "evento": apoyo["evento"],
            "frame_evento": int(apoyo["frame"]),
            "time_s": apoyo["time_s"],
            "lado_apoyo": apoyo["lado"],
            "x_original": apoyo["x"],
            "y_original": apoyo["y"],
            "vis_tobillo": apoyo["vis"],
            "distancia_a_tabla_m": apoyo["distancia_a_tabla_m"],
            "frame_angulo_usado": frame_angulo,
            "score_angulos": score,
            "angulo_rodilla": ang_rodilla,
            "angulo_cadera": ang_cadera,
            "inclinacion_tronco": inc_tronco,
        })

        guardar_debug_evento(apoyo, angle_row)

    res = pd.DataFrame(filas)

    longitudes = []
    tiempos = []
    velocidades = []

    for i in range(len(res)):
        if i < len(res) - 1:
            d1 = res.loc[i, "distancia_a_tabla_m"]
            d2 = res.loc[i + 1, "distancia_a_tabla_m"]
            t1 = res.loc[i, "time_s"]
            t2 = res.loc[i + 1, "time_s"]

            longitud = abs(d1 - d2)
            tiempo = abs(t2 - t1)
            velocidad = longitud / tiempo if tiempo > 0 else np.nan
        else:
            longitud = np.nan
            tiempo = np.nan
            velocidad = np.nan

        longitudes.append(longitud)
        tiempos.append(tiempo)
        velocidades.append(velocidad)

    res["longitud_hasta_siguiente_m"] = longitudes
    res["tiempo_hasta_siguiente_s"] = tiempos
    res["velocidad_paso_m_s"] = velocidades

    res = validar_secuencia_apoyos(res)


    variables_velocidad = calcular_variables_velocidad_batida(res, df, calibracion)

    for nombre_variable, valor in variables_velocidad.items():
        res[nombre_variable] = valor

    output_velocidades = RESULTS_DIR / f"variables_velocidad_batida_{STEM}.csv"
    pd.DataFrame([{ "video": VIDEO_NAME, **variables_velocidad }]).to_csv(
        output_velocidades,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nVariables de velocidad de entrada/salida guardadas en:")
    print(output_velocidades)

    res.to_csv(OUTPUT_FINAL, index=False, encoding="utf-8-sig")

    print("\n=== RESULTADO FINAL ===")
    print(res)

    print("\nCSV final guardado en:")
    print(OUTPUT_FINAL)

    print("\nImágenes de revisión guardadas en:")
    print(DEBUG_DIR)


def main():
    calibracion = cargar_o_crear_calibracion()

    generar_crop_automatico()

    df = detectar_pose_y_angulos()
    df = aplicar_filtro_pose_atleta_por_crop(df)
    df = limpiar_angulos(df)

    df_apoyos = df.copy()

    if USAR_VENTANA_ANALISIS:
        print("\n=== USANDO VENTANA TEMPORAL PARA APOYOS ===")
        print(f"Frames analizados: {FRAME_INICIO_ANALISIS} - {FRAME_FIN_ANALISIS}")

        df_apoyos = df_apoyos[
            (df_apoyos["frame"] >= FRAME_INICIO_ANALISIS)
            & (df_apoyos["frame"] <= FRAME_FIN_ANALISIS)
        ].copy()

        print("Frames disponibles para apoyos:", len(df_apoyos))

    apoyos_df = detectar_apoyos_automaticos(df_apoyos, calibracion)

    if apoyos_df.empty:
        print("\nNo se pudo generar resultado final porque no se detectaron apoyos.")
        return

    crear_tabla_final(df, apoyos_df, calibracion)

    print("\n=== PIPELINE COMPLETO TERMINADO ===")


if __name__ == "__main__":
    main()
