"""
Оценка резкости, не зависящая от сюжета, контраста и разрешения.

Основная метрика — ширина краёв (в пикселях кадра, приведённого к стороне 1024):
у резкого края перепад яркости C происходит за 1–2 пикселя, у размытого — растягивается.
Для каждого блока с заметным перепадом считаем  ширина ≈ C / max|градиент|.
Однотонные области (небо, белый фон, глянцевый кузов, чёрная ткань) в оценку не входят —
раньше именно они «топили» медиану лапласиана и резкие фото признавались размытыми.

Берём 25-й перцентиль ширины по блокам объекта: фото считается резким, если резка заметная
часть объекта (боке и размытый фон в портретной съёмке не штрафуются).
Смаз движения размывает края только одного направления, поэтому ширина считается ещё и
отдельно по 4 направлениям; худшее направление входит в оценку с весом DIRECTIONAL_WEIGHT.
На шумных фото перед расчётом градиентов применяется медианный фильтр.
"""
import cv2
import numpy as np

BLOCK = 32
MIN_BLOCK_CONTRAST = 20.0   # перепад яркости (p98 - p2) блока, при котором блок считается «с деталями»
MIN_EDGE_BLOCKS = 6         # меньше — оценивать резкость не по чему (почти пустой кадр)
EDGE_PERCENTILE = 25
WIDTH_SHARP = 2.6           # ширина края резкого фото (с учётом JPEG и демозаики)
WIDTH_BLURRY = 4.2          # ширина края, начиная с которой фото полностью размыто
BLURRY_THRESHOLD = 0.45     # индекс чёткости ниже — фото размыто
DIRECTIONAL_WEIGHT = 0.6    # вклад худшего направления (смаз движения)
MOTION_RATIO = 1.6          # во сколько раз худшее направление шире лучшего, чтобы назвать это смазом
NOISE_PREFILTER = 1.5       # у 95% чистых фото оценка шума ≤ ~1.1, при шуме σ≈10 — ≥ 2


def _legacy_metrics(gray):
    """Старые метрики (ACF, лапласиан, Собель) — только для отчёта и БД, в оценку не входят"""
    g = gray.astype(np.float64)
    norm = (g - g.mean()) / (g.std() + 1e-8)
    f = np.fft.fft2(norm)
    acf = np.fft.fftshift(np.fft.ifft2(f * np.conj(f)).real)
    h, w = g.shape
    cy, cx = h // 2, w // 2
    acf /= acf[cy, cx] + 1e-8
    acf_lag5 = (abs(acf[cy - 5, cx]) + abs(acf[cy, cx - 5])) / 2

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    sobel = np.hypot(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))
    local = [lap[y:y + 64, x:x + 64].var() for y in range(0, h - 63, 64) for x in range(0, w - 63, 64)]
    return {
        "acf_score": float(acf_lag5),
        "laplacian_var": float(lap.var()),
        "median_laplacian": float(np.median(local)) if local else float(lap.var()),
        "sobel_var": float(sobel.var()),
    }


def _noise_sigma(gray, block=8, percentile=2):
    """
    Оценка цифрового шума (σ, уровни яркости). Отклик ядра Иммеркера (почти нулевой на гладких
    градиентах) усредняется по блокам 8×8 и берётся по 2% самых гладких блоков: шум есть и там,
    а текстура (трава, гравий, ткань, текст) — нет. Блоки с выбитыми в белое/чёрное пикселями
    пропускаются: на них шум не виден.
    """
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    g = gray.astype(np.float32)
    response = np.abs(cv2.filter2D(g, -1, kernel))[1:-1, 1:-1] * np.sqrt(np.pi / 2) / 6
    g = g[1:-1, 1:-1]
    h, w = response.shape
    h, w = h // block * block, w // block * block
    if h == 0 or w == 0:
        return 0.0
    blocks_r = response[:h, :w].reshape(h // block, block, w // block, block).mean(axis=(1, 3)).ravel()
    blocks_g = g[:h, :w].reshape(h // block, block, w // block, block)
    usable = (blocks_g.max(axis=(1, 3)).ravel() < 250) & (blocks_g.min(axis=(1, 3)).ravel() > 5)
    if usable.sum() < 20:
        return 0.0
    return float(np.percentile(blocks_r[usable], percentile))


def check_blur(image, mask=None):
    """
    image — BGR-кадр, уже приведённый к рабочему размеру (длинная сторона ≤ 1024).
    mask — маска главного объекта (1 — объект) того же размера или None.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    noise_sigma = _noise_sigma(gray)
    # Шум (тёмная комната, дешёвая камера) завышает градиенты, и размытое фото выглядит резким.
    # Медианный фильтр 3×3 убирает шум, почти не трогая края; на чистых фото его не применяем.
    edges_src = cv2.medianBlur(gray, 3) if noise_sigma > NOISE_PREFILTER else gray
    g = edges_src.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    mag = np.sqrt(gx * gx + gy * gy)

    h, w = g.shape
    widths, angles, in_subject = [], [], []
    for y in range(0, h - BLOCK + 1, BLOCK):
        for x in range(0, w - BLOCK + 1, BLOCK):
            block = g[y:y + BLOCK, x:x + BLOCK]
            lo, hi = np.percentile(block, (2, 98))
            contrast = hi - lo
            if contrast < MIN_BLOCK_CONTRAST:
                continue
            peak = np.percentile(mag[y:y + BLOCK, x:x + BLOCK], 99)
            widths.append(contrast / (peak + 1e-6))
            # Преобладающее направление градиента в блоке (тензор структуры)
            bx, by = gx[y:y + BLOCK, x:x + BLOCK], gy[y:y + BLOCK, x:x + BLOCK]
            angles.append(0.5 * np.arctan2(2 * float((bx * by).sum()), float((bx * bx).sum() - (by * by).sum())))
            in_subject.append(mask is None or mask[y:y + BLOCK, x:x + BLOCK].mean() > 0.5)

    widths = np.array(widths)
    angles = np.array(angles)
    in_subject = np.array(in_subject, dtype=bool)
    edge_blocks = int(len(widths))
    total_blocks = max(1, (h // BLOCK) * (w // BLOCK))

    directional_width, direction_ratio = 0.0, 1.0
    if edge_blocks < MIN_EDGE_BLOCKS:
        # Почти нет деталей: однотонный кадр, «квадрат Малевича». Резкость оценить нельзя.
        edge_width, sharpness, scope = 0.0, 0.0, "нет деталей"
    else:
        use = in_subject if in_subject.sum() >= 12 else np.ones_like(in_subject)
        scope = "по объекту" if not use.all() else "по всему кадру"
        sel_w, sel_a = widths[use], angles[use]
        overall = float(np.percentile(sel_w, EDGE_PERCENTILE))

        # Смаз движения размывает только края, перпендикулярные движению, а параллельные остаются резкими
        # и «прячут» смаз в общей статистике. Поэтому ширину считаем отдельно по 4 направлениям краёв.
        bins = (((sel_a + np.pi / 8) % np.pi) // (np.pi / 4)).astype(int)
        per_dir = [float(np.percentile(sel_w[bins == k], EDGE_PERCENTILE)) for k in range(4)
                   if (bins == k).sum() >= max(8, 0.08 * len(sel_w))]
        if len(per_dir) >= 2:
            directional_width = max(per_dir)
            direction_ratio = max(per_dir) / (min(per_dir) + 1e-6)

        edge_width = max(overall, DIRECTIONAL_WEIGHT * directional_width)
        sharpness = float(np.clip((WIDTH_BLURRY - edge_width) / (WIDTH_BLURRY - WIDTH_SHARP), 0.0, 1.0))

    is_blurry = edge_blocks >= MIN_EDGE_BLOCKS and sharpness < BLURRY_THRESHOLD
    is_motion_blur = bool(is_blurry and direction_ratio > MOTION_RATIO)

    result = _legacy_metrics(gray)
    result.update({
        "sharpness_score": round(sharpness, 3),
        "edge_width": edge_width,
        "edge_blocks": edge_blocks,
        "detail_share": edge_blocks / total_blocks,
        "directional_width": directional_width,
        "noise_sigma": noise_sigma,
        "direction_ratio": direction_ratio,
        "is_blurry": bool(is_blurry),
        "is_motion_blur": is_motion_blur,
        "method": f"ширина краёв, {scope}" + (", смаз движения" if is_motion_blur else ""),
    })

    print(f"[1] Чёткость ({scope}): ширина края {edge_width:.2f} px, блоков с деталями {edge_blocks}, "
          f"индекс {sharpness:.2f} -> {'РАЗМЫТО' if is_blurry else 'ЧЁТКО'}")
    return result
