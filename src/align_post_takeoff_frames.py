import pandas as pd
import numpy as np
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


RESULTS_DIR = BASE_DIR / "resultados_revision_individual"


if not RESULTS_DIR.exists():
    RESULTS_DIR = BASE_DIR

FPS_REAL = 240
MAX_DIST_POSE_A_CROP_PX = 50


MAX_FRAMES_BUSQUEDA_POSTBATIDA = 300


MIN_FRAMES_DIFERENCIA_POSTBATIDA = 40


USAR_POSE_VALIDA_ATLETA = True


RADIO_MAX_BUSQUEDA_ALREDEDOR_OBJETIVO = 300


OUTPUT_RESUMEN_MAXIMO = RESULTS_DIR / "resumen_mayor_diferencia_frames_postbatida_min40.csv"
OUTPUT_FRAME_COMUN = RESULTS_DIR / "pose_postbatida_misma_distancia_frames_min40.csv"
OUTPUT_DETALLE_RECUPERACION = RESULTS_DIR / "detalle_recuperacion_pose_por_video_min40.csv"


def normalizar_bool(serie):
    """Convierte columnas True/False que a veces llegan como texto."""
    if serie.dtype == bool:
        return serie
    return serie.astype(str).str.lower().isin(["true", "1", "sí", "si", "yes"])


def extraer_stem_desde_eventos(path_eventos):
    return path_eventos.stem.replace("eventos_automaticos_", "")


def preparar_pose(df_pose):
    df_pose = df_pose.copy()

    if "frame" not in df_pose.columns:
        raise ValueError("El archivo de keypoints no tiene columna frame")

    df_pose["frame"] = pd.to_numeric(df_pose["frame"], errors="coerce")
    df_pose = df_pose.dropna(subset=["frame"]).copy()
    df_pose["frame"] = df_pose["frame"].astype(int)

    if "pose_detected" not in df_pose.columns:
        df_pose["pose_detected"] = False
    else:
        df_pose["pose_detected"] = normalizar_bool(df_pose["pose_detected"])

    if USAR_POSE_VALIDA_ATLETA and "pose_valida_atleta" in df_pose.columns:
        df_pose["pose_valida_para_analisis"] = normalizar_bool(df_pose["pose_valida_atleta"])
        df_pose["criterio_pose_usado"] = "pose_valida_atleta"
        return df_pose

    if "dist_pose_a_crop_px" in df_pose.columns:
        df_pose["dist_pose_a_crop_px"] = pd.to_numeric(df_pose["dist_pose_a_crop_px"], errors="coerce")
        df_pose["pose_valida_para_analisis"] = (
            (df_pose["pose_detected"] == True)
            & (df_pose["dist_pose_a_crop_px"] <= MAX_DIST_POSE_A_CROP_PX)
        )
        df_pose["criterio_pose_usado"] = f"pose_detected_y_dist_crop_le_{MAX_DIST_POSE_A_CROP_PX}"
    else:
        df_pose["pose_valida_para_analisis"] = df_pose["pose_detected"] == True
        df_pose["criterio_pose_usado"] = "solo_pose_detected"

    return df_pose


def obtener_frame_batida(path_eventos):
    eventos = pd.read_csv(path_eventos)

    if eventos.empty:
        raise ValueError("CSV de eventos vacío")

    if "evento" not in eventos.columns or "frame_evento" not in eventos.columns:
        raise ValueError("El CSV de eventos no tiene evento/frame_evento")

    batidas = eventos[eventos["evento"] == "batida"].copy()

    if batidas.empty:

        fila = eventos.sort_values("frame_evento").iloc[-1]
    else:
        fila = batidas.iloc[0]

    frame_batida = int(fila["frame_evento"])
    time_batida_s = float(fila.get("time_s", np.nan))

    return frame_batida, time_batida_s


def primera_pose_valida_postbatida(pose, frame_batida):


    frame_minimo = frame_batida + MIN_FRAMES_DIFERENCIA_POSTBATIDA

    ventana = pose[
        (pose["frame"] >= frame_minimo)
        & (pose["frame"] <= frame_batida + MAX_FRAMES_BUSQUEDA_POSTBATIDA)
    ].copy()

    validos = ventana[ventana["pose_valida_para_analisis"] == True].copy()

    if validos.empty:
        return np.nan, np.nan, np.nan, f"no_hay_pose_valida_postbatida_desde_{MIN_FRAMES_DIFERENCIA_POSTBATIDA}f"

    primer_frame = int(validos["frame"].iloc[0])
    frames_diferencia = primer_frame - frame_batida
    tiempo_s = frames_diferencia / FPS_REAL

    return primer_frame, frames_diferencia, tiempo_s, "OK"


def elegir_frame_cercano_al_objetivo(pose, frame_batida, offset_objetivo):
    """
    Busca una pose válida en batida + offset_objetivo.
    Si no existe exactamente, prueba objetivo-1, objetivo+1, objetivo-2, objetivo+2...
    """
    frame_objetivo = int(frame_batida + offset_objetivo)

    validos = pose[
        (pose["pose_valida_para_analisis"] == True)
        & (pose["frame"] >= frame_batida)
        & (pose["frame"] <= frame_batida + MAX_FRAMES_BUSQUEDA_POSTBATIDA)
    ].copy()

    if validos.empty:
        return None, "sin_pose_valida_en_ventana"


    validos_por_frame = {
        int(row["frame"]): row
        for _, row in validos.sort_values("frame").iterrows()
    }


    candidatos_frames = [frame_objetivo]

    for delta in range(1, RADIO_MAX_BUSQUEDA_ALREDEDOR_OBJETIVO + 1):
        candidatos_frames.append(frame_objetivo - delta)
        candidatos_frames.append(frame_objetivo + delta)

    for fr in candidatos_frames:
        if fr < frame_batida:
            continue
        if fr > frame_batida + MAX_FRAMES_BUSQUEDA_POSTBATIDA:
            continue
        if fr in validos_por_frame:
            row = validos_por_frame[fr]
            estado = "exacto" if fr == frame_objetivo else "ajustado_frame_arriba_abajo"
            return row, estado


    validos["distancia_a_objetivo"] = (validos["frame"] - frame_objetivo).abs()
    row = validos.sort_values(["distancia_a_objetivo", "frame"]).iloc[0]
    return row, "fallback_mas_cercano"


def extraer_valor(row, col):
    if row is None:
        return np.nan
    try:
        return row.get(col, np.nan)
    except Exception:
        return np.nan


def main():
    eventos_files = sorted(RESULTS_DIR.glob("eventos_automaticos_*.csv"))

    if not eventos_files:
        print("No se encontraron eventos_automaticos_*.csv en:")
        print(RESULTS_DIR)
        return

    datos = []
    errores = []

    print("\n=== BUSCANDO RECUPERACIÓN MÁS TARDÍA DE POSE POSTBATIDA ===")
    print("Carpeta:", RESULTS_DIR)
    print("Mínimo de frames después de batida:", MIN_FRAMES_DIFERENCIA_POSTBATIDA)

    for path_eventos in eventos_files:
        stem = extraer_stem_desde_eventos(path_eventos)
        path_pose = RESULTS_DIR / f"keypoints_angulos_{stem}.csv"

        if not path_pose.exists():
            errores.append({"video": stem, "error": f"No existe {path_pose.name}"})
            continue

        try:
            frame_batida, time_batida_s = obtener_frame_batida(path_eventos)
            pose = preparar_pose(pd.read_csv(path_pose))

            primer_frame, frames_diff, tiempo_diff_s, estado = primera_pose_valida_postbatida(
                pose=pose,
                frame_batida=frame_batida
            )

            criterio = pose["criterio_pose_usado"].iloc[0] if len(pose) else "desconocido"

            datos.append({
                "video": stem,
                "path_eventos": path_eventos,
                "path_pose": path_pose,
                "frame_batida": frame_batida,
                "time_batida_s": time_batida_s,
                "primer_frame_pose_valida_postbatida": primer_frame,
                "frames_hasta_recuperar_pose_postbatida": frames_diff,
                "tiempo_hasta_recuperar_pose_postbatida_s": tiempo_diff_s,
                "estado_recuperacion": estado,
                "criterio_pose_usado": criterio,
                "min_frames_diferencia_postbatida": MIN_FRAMES_DIFERENCIA_POSTBATIDA,
            })

        except Exception as e:
            errores.append({"video": stem, "error": str(e)})

    df_recuperacion = pd.DataFrame(datos)

    if df_recuperacion.empty:
        print("No se pudo analizar ningún vídeo.")
        if errores:
            print(pd.DataFrame(errores))
        return

    df_recuperacion.to_csv(OUTPUT_DETALLE_RECUPERACION, index=False, encoding="utf-8-sig")

    validos_max = df_recuperacion.dropna(subset=["frames_hasta_recuperar_pose_postbatida"]).copy()

    if validos_max.empty:
        print("Ningún vídeo recupera pose válida en la ventana configurada.")
        print("Guardado detalle en:", OUTPUT_DETALLE_RECUPERACION)
        return

    idx_max = validos_max["frames_hasta_recuperar_pose_postbatida"].idxmax()
    fila_max = df_recuperacion.loc[idx_max]

    offset_maximo = int(fila_max["frames_hasta_recuperar_pose_postbatida"])
    video_maximo = fila_max["video"]
    frame_batida_maximo = int(fila_max["frame_batida"])
    primer_frame_maximo = int(fila_max["primer_frame_pose_valida_postbatida"])
    tiempo_maximo_s = float(fila_max["tiempo_hasta_recuperar_pose_postbatida_s"])

    resumen_maximo = pd.DataFrame([{
        "video_con_recuperacion_mas_tardia": video_maximo,
        "frame_batida": frame_batida_maximo,
        "primer_frame_pose_valida_postbatida": primer_frame_maximo,
        "mayor_diferencia_frames": offset_maximo,
        "mayor_diferencia_tiempo_s": tiempo_maximo_s,
        "min_frames_diferencia_postbatida": MIN_FRAMES_DIFERENCIA_POSTBATIDA,
        "fps_usado": FPS_REAL,
        "explicacion": f"Este offset se usa como distancia común postbatida para todos los vídeos. La búsqueda ignora recuperaciones anteriores a {MIN_FRAMES_DIFERENCIA_POSTBATIDA} frames."
    }])

    resumen_maximo.to_csv(OUTPUT_RESUMEN_MAXIMO, index=False, encoding="utf-8-sig")

    print("\nMayor diferencia encontrada:")
    print("Vídeo:", video_maximo)
    print("Frame batida:", frame_batida_maximo)
    print("Primer frame válido postbatida:", primer_frame_maximo)
    print("Diferencia:", offset_maximo, "frames")
    print("Tiempo:", round(tiempo_maximo_s, 4), "s")

    print("\n=== BUSCANDO EN TODOS LOS VÍDEOS EL FRAME BATIDA +", offset_maximo, "===")

    filas_frame_comun = []

    for _, info_video in df_recuperacion.iterrows():
        stem = info_video["video"]
        frame_batida = int(info_video["frame_batida"])
        frame_objetivo = frame_batida + offset_maximo

        try:
            pose = preparar_pose(pd.read_csv(info_video["path_pose"]))
            row_elegida, estado_busqueda = elegir_frame_cercano_al_objetivo(
                pose=pose,
                frame_batida=frame_batida,
                offset_objetivo=offset_maximo
            )

            if row_elegida is None:
                frame_elegido = np.nan
                offset_real = np.nan
                desviacion = np.nan
                time_elegido_s = np.nan
            else:
                frame_elegido = int(row_elegida["frame"])
                offset_real = frame_elegido - frame_batida
                desviacion = offset_real - offset_maximo
                time_elegido_s = extraer_valor(row_elegida, "time_s")

            filas_frame_comun.append({
                "video": stem,
                "frame_batida": frame_batida,
                "offset_objetivo_comun_frames": offset_maximo,
                "frame_objetivo_comun": frame_objetivo,
                "frame_pose_elegido": frame_elegido,
                "offset_real_elegido_frames": offset_real,
                "desviacion_respecto_objetivo_frames": desviacion,
                "time_pose_elegido_s": time_elegido_s,
                "estado_busqueda_frame_comun": estado_busqueda,
                "primer_frame_pose_valida_postbatida": info_video["primer_frame_pose_valida_postbatida"],
                "frames_hasta_recuperar_pose_postbatida": info_video["frames_hasta_recuperar_pose_postbatida"],
                "dist_pose_a_crop_px_frame_elegido": extraer_valor(row_elegida, "dist_pose_a_crop_px"),
                "pose_rescatada_subcrop_frame_elegido": extraer_valor(row_elegida, "pose_rescatada_subcrop"),
                "modo_pose_detectada_frame_elegido": extraer_valor(row_elegida, "modo_pose_detectada"),
                "cadera_media_x_original_frame_elegido": extraer_valor(row_elegida, "cadera_media_x_original"),
                "cadera_media_y_original_frame_elegido": extraer_valor(row_elegida, "cadera_media_y_original"),
                "cadera_media_x_suavizada_frame_elegido": extraer_valor(row_elegida, "cadera_media_x_suavizada"),
                "cadera_media_y_suavizada_frame_elegido": extraer_valor(row_elegida, "cadera_media_y_suavizada"),
                "score_angulos_frame_elegido": extraer_valor(row_elegida, "score_angulos"),
                "criterio_pose_usado": info_video["criterio_pose_usado"],
            })

        except Exception as e:
            filas_frame_comun.append({
                "video": stem,
                "frame_batida": frame_batida,
                "offset_objetivo_comun_frames": offset_maximo,
                "frame_objetivo_comun": frame_objetivo,
                "frame_pose_elegido": np.nan,
                "offset_real_elegido_frames": np.nan,
                "desviacion_respecto_objetivo_frames": np.nan,
                "estado_busqueda_frame_comun": f"error: {e}",
            })

    df_frame_comun = pd.DataFrame(filas_frame_comun)
    df_frame_comun = df_frame_comun.sort_values("video").reset_index(drop=True)
    df_frame_comun.to_csv(OUTPUT_FRAME_COMUN, index=False, encoding="utf-8-sig")

    print("\n=== RESULTADO FRAME COMÚN POSTBATIDA ===")
    cols_mostrar = [
        "video",
        "frame_batida",
        "offset_objetivo_comun_frames",
        "frame_objetivo_comun",
        "frame_pose_elegido",
        "offset_real_elegido_frames",
        "desviacion_respecto_objetivo_frames",
        "estado_busqueda_frame_comun"
    ]
    print(df_frame_comun[cols_mostrar])

    print("\nGuardado resumen máximo en:")
    print(OUTPUT_RESUMEN_MAXIMO)

    print("\nGuardado detalle de recuperación por vídeo en:")
    print(OUTPUT_DETALLE_RECUPERACION)

    print("\nGuardado frame común elegido para todos los vídeos en:")
    print(OUTPUT_FRAME_COMUN)

    if errores:
        print("\nAvisos/errores:")
        print(pd.DataFrame(errores))

    n_sin_pose = int(df_frame_comun["frame_pose_elegido"].isna().sum())
    if n_sin_pose > 0:
        print("\nATENCIÓN:", n_sin_pose, "vídeos no tienen pose válida ni buscando alrededor del objetivo.")


if __name__ == "__main__":
    main()
