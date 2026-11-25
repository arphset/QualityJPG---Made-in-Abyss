"""
Сильное JPEG-сжатие (фото пережимали при пересылке в мессенджерах, перезаливке, «экономии места»).

Качество сжатия оценивается по таблице квантования яркости, записанной в самом JPEG-файле:
чем сильнее сжатие, тем крупнее коэффициенты таблицы. Оценка обратна формуле масштабирования
стандартной таблицы IJG (её используют libjpeg, Photoshop «Сохранить как», большинство сервисов).
Пиксельные метрики «квадратиков» проверялись и отброшены: обычные веб-JPEG (качество 75–90)
по ним неотличимы от сильно пережатых.

Ограничение: если пережатое фото потом пересохранили с высоким качеством, артефакты остались,
а в файле записано новое (высокое) качество — такой случай не обнаруживается.
"""
import numpy as np
from PIL import Image

# Стандартная таблица квантования яркости JPEG (ITU T.81, приложение K)
_STD_LUMA = np.array([
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], dtype=float)

QUALITY_OK = 55     # с этого качества и выше артефакты обычно не видны
QUALITY_BAD = 25    # при таком качестве и ниже «квадратики» заметны всегда


def estimate_jpeg_quality(image_path):
    """Качество JPEG 1..100 или None (не JPEG / нет таблиц)"""
    try:
        with Image.open(image_path) as im:
            if im.format != "JPEG":
                return None
            tables = im.quantization
    except Exception:
        return None
    if not tables or 0 not in tables:
        return None
    scale = np.mean(np.asarray(tables[0], dtype=float)) / np.mean(_STD_LUMA) * 100
    quality = (200 - scale) / 2 if scale <= 100 else 5000 / scale
    return float(np.clip(quality, 1, 100))


def check_artifacts(image_path):
    quality = estimate_jpeg_quality(image_path)
    severity = 0.0
    if quality is not None:
        severity = float(np.clip((QUALITY_OK - quality) / (QUALITY_OK - QUALITY_BAD), 0.0, 1.0))
        print(f"[4] Качество JPEG: ~{quality:.0f}")
    return {
        "jpeg_quality": quality,
        "compression_severity": severity,
        "is_compressed": severity >= 0.3,
    }
