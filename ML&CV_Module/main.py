import cv2
import numpy as np
from filters.blure_filter import check_blur
from filters.brightness_filter import check_brightness
from filters.YOLO import detect_objects_yolo
from filters.subject import find_subject
from filters.artifacts import check_artifacts

# Все метрики считаются на копии с длинной стороной 1024 px: так оценка не зависит от разрешения камеры
# (4000 px с телефона и 1000 px скриншот оцениваются одинаково), а анализ идёт быстрее.
WORK_SIDE = 1024
MIN_GOOD_SIDE = 800   # маркетплейсы требуют от ~700–900 px по длинной стороне
NOISE_START, NOISE_END = 1.6, 3.2   # оценка шума на копии 1024 px: у 95% чистых фото ≤ ~1.1


def _read_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        # cv2.imread не понимает не-ASCII пути на Windows
        try:
            image = cv2.imdecode(np.fromfile(image_path, np.uint8), cv2.IMREAD_UNCHANGED)
        except (OSError, ValueError):
            image = None
    if image is None:
        return None
    if image.dtype != np.uint8:   # 16-битные PNG
        image = (image / 257).astype(np.uint8) if image.dtype == np.uint16 else image.astype(np.uint8)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        # PNG с прозрачностью: без этого прозрачный фон читается как чёрный.
        # Маркетплейсы показывают такие картинки на белом — так и оцениваем.
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        image = (image[:, :, :3].astype(np.float32) * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    return image


def _work_copy(image):
    h, w = image.shape[:2]
    scale = WORK_SIDE / max(h, w)
    if scale >= 1:
        return image
    return cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def analyze_image(image_path):
    image = _read_image(image_path)
    if image is None:
        return {"error": "Картинка не найдена"}

    h, w = image.shape[:2]
    print(f"--- Анализ: {image_path} ({w}x{h}) ---")
    work = _work_copy(image)

    yolo_data = detect_objects_yolo(work)
    subject = find_subject(work, yolo_data)
    mask = subject["mask"]

    # Резкость по маске объекта — только если объект выделен надёжно; приблизительная маска
    # (центральная область) может попасть на размытый фон, тогда честнее оценивать весь кадр
    blur_mask = mask if subject["subject_confidence"] >= 1.0 else None
    blur_data = check_blur(work, mask=blur_mask)
    brightness_data = check_brightness(work, mask=mask, background_uniform=subject["background_uniform"])
    artifacts = check_artifacts(image_path)   # по таблицам квантования в самом JPEG-файле
    noise_sigma = blur_data["noise_sigma"]
    noise_severity = float(np.clip((noise_sigma - NOISE_START) / (NOISE_END - NOISE_START), 0.0, 1.0))

    issues = []
    empty_frame = blur_data["edge_blocks"] < 6
    if empty_frame:
        issues.append("В кадре почти нет деталей — товар не виден")
    elif blur_data["is_motion_blur"]:
        issues.append("Смаз движения (дрожание камеры или движение объекта)")
    elif blur_data["is_blurry"]:
        issues.append("Фото размыто (не в фокусе)")
    elif blur_data["sharpness_score"] < 0.75:
        issues.append("Чёткость ниже желаемой")

    if brightness_data["is_dark"]:
        issues.append("Фото слишком тёмное (недодержка)")
    if brightness_data["is_bright"]:
        issues.append("Кадр засвечен, нет теней")
    if brightness_data["is_overexposed"]:
        issues.append(f"Пересвет: выбитые в белое участки ({brightness_data['overexposed_pct']:.0f}%)")
    if brightness_data["is_underexposed"]:
        issues.append(f"Провалы в тенях на объекте ({brightness_data['underexposed_pct']:.0f}%)")
    if brightness_data["is_low_contrast"]:
        issues.append("Низкий контраст (пелена, туман)")
    if brightness_data["is_backlit"]:
        issues.append("Объект в тени на фоне яркого света (снято против окна или солнца)")
    if artifacts["is_compressed"]:
        issues.append(f"Фото сильно сжато (качество JPEG ~{artifacts['jpeg_quality']:.0f}), возможны «квадратики»")
    if noise_severity >= 0.3:
        issues.append("Заметный цифровой шум (мало света при съёмке)")

    fill_rate = subject["fill_rate"]
    if subject["object_found"]:
        if fill_rate < 0.25:
            issues.append(f"Объект занимает мало места в кадре ({fill_rate*100:.0f}%), его стоит снять крупнее")
        if subject["is_off_center"] and fill_rate < 0.5:
            issues.append("Объект смещён от центра кадра")
    elif not empty_frame:
        issues.append("Главный объект не удалось выделить (сложный фон или сцена)")

    long_side = max(h, w)
    if long_side < MIN_GOOD_SIDE:
        issues.append(f"Низкое разрешение ({w}x{h}), рекомендуется от {MIN_GOOD_SIDE} px по длинной стороне")

    if not issues:
        print("\n✅ ФОТО ОТЛИЧНОЕ! Можно публиковать.")
    else:
        print(f"\n❌ ПРОБЛЕМЫ: {', '.join(issues)}")

    return {
        # Ключи прежней версии (UI, БД, LLM, тесты)
        "acf_score": blur_data["acf_score"],
        "laplacian_var": blur_data["laplacian_var"],
        "median_laplacian": blur_data["median_laplacian"],
        "sobel_var": blur_data["sobel_var"],
        "sharpness_score": blur_data["sharpness_score"],
        "blur_method": blur_data["method"],
        "is_blurry": blur_data["is_blurry"],
        "is_motion_blur": blur_data["is_motion_blur"],
        "brightness": brightness_data["brightness"],
        "overexposed_pct": brightness_data["overexposed_pct"],
        "underexposed_pct": brightness_data["underexposed_pct"],
        "is_dark": brightness_data["is_dark"],
        "is_bright": brightness_data["is_bright"],
        "is_overexposed": brightness_data["is_overexposed"],
        "is_underexposed": brightness_data["is_underexposed"],
        "fill_rate": fill_rate,
        "is_off_center": subject["is_off_center"],
        "composition_bad": subject["object_found"] and fill_rate < 0.15,
        "object_found": subject["object_found"],
        "issues": issues,
        # Новые метрики
        "edge_width": blur_data["edge_width"],
        "edge_blocks": blur_data["edge_blocks"],
        "detail_share": blur_data["detail_share"],
        "empty_frame": empty_frame,
        "subject_method": subject["subject_method"],
        "subject_confidence": subject["subject_confidence"],
        "yolo_objects": yolo_data.get("object_count", 0),
        "subject_brightness": brightness_data["subject_brightness"],
        "contrast": brightness_data["contrast"],
        "dark_severity": brightness_data["dark_severity"],
        "highlight_severity": brightness_data["highlight_severity"],
        "washed_severity": brightness_data["washed_severity"],
        "shadow_severity": brightness_data["shadow_severity"],
        "contrast_severity": brightness_data["contrast_severity"],
        "is_low_contrast": brightness_data["is_low_contrast"],
        "backlit_severity": brightness_data["backlit_severity"],
        "highlight_pile_up": brightness_data["highlight_pile_up"],
        "overexposed_effective_pct": brightness_data["overexposed_effective_pct"],
        "is_backlit": brightness_data["is_backlit"],
        "jpeg_quality": artifacts["jpeg_quality"] if artifacts["jpeg_quality"] is not None else 0.0,
        "compression_severity": artifacts["compression_severity"],
        "is_compressed": artifacts["is_compressed"],
        "noise_sigma": noise_sigma,
        "noise_severity": noise_severity,
        "is_noisy": noise_severity >= 0.3,
        "image_width": int(w),
        "image_height": int(h),
    }
