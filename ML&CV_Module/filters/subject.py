"""
Выделение главного объекта кадра без привязки к классам YOLO.

1. Если фон однородный (студийная съёмка, белый/серый/цветной фон) — объект = всё, что заметно
   отличается по цвету от фона по краям кадра. Работает для любых товаров, а не только для 80 классов COCO.
2. Иначе — главный объект YOLO (люди, машины, мебель, техника и т.п.): крупный и ближе к центру.
   Случайные объекты у края кадра (прохожий, обрезанный рамкой) главным не считаются.
3. Иначе — GrabCut от центра кадра: то, что отделяется от окружения в центральной области
   (YOLOv8n, например, не узнаёт машину в ракурсе с открытой дверью).
4. Если не сработало ничего — объект не выделен (пейзаж, интерьер): заполнение почти не штрафуется.
"""
import cv2
import numpy as np

ANALYSIS_SIDE = 512          # сторона уменьшенной копии для выделения объекта
BORDER_SHARE = 0.03          # ширина полосы по краям кадра, по которой оценивается фон
BG_TOLERANCE = 10.0          # ΔE: пиксель края похож на фон
BG_UNIFORM_SHARE = 0.70      # доля «похожих» пикселей края, при которой фон считается однородным
FG_THRESHOLD = 16.0          # ΔE: пиксель отличается от фона → объект
MIN_COMPONENT_SHARE = 0.002  # мелкие пятна (пыль, шум JPEG) отбрасываем
YOLO_MIN_CONF = 0.30
YOLO_MIN_AREA = 0.03         # объекты меньше 3% кадра главным объектом не считаются
CENTRAL_ZONE = 0.35          # центр рамки главного объекта — не дальше 35% размера кадра от центра
GRABCUT_SIDE = 256


def _lab(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0   # L в OpenCV хранится в 0..255, приводим к 0..100 как в ΔE76
    return lab


def _background_mask(small):
    """Маска объекта на однородном фоне или None, если фон неоднородный"""
    lab = _lab(small)
    h, w = lab.shape[:2]
    b = max(2, int(round(min(h, w) * BORDER_SHARE)))
    border = np.concatenate([
        lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
        lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3),
    ])
    bg = np.median(border, axis=0)
    uniform_share = float(np.mean(np.linalg.norm(border - bg, axis=1) < BG_TOLERANCE))
    if uniform_share < BG_UNIFORM_SHARE:
        return None, uniform_share

    dist = np.linalg.norm(lab - bg, axis=2)
    fg = (dist > FG_THRESHOLD).astype(np.uint8)
    # Сначала «склеиваем» тонкие детали (контуры, проволока, ткань в сетку), потом убираем шум
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    min_area = MIN_COMPONENT_SHARE * h * w
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    if not keep:
        return None, uniform_share
    mask = np.isin(labels, keep).astype(np.uint8)
    return mask, uniform_share


def _yolo_main_mask(yolo_data, w, h):
    """Маска главного объекта по YOLO или None"""
    cands = []
    for (x1, y1, x2, y2), conf in zip(yolo_data.get("boxes", []), yolo_data.get("confidences", [])):
        area = (x2 - x1) * (y2 - y1) / float(w * h)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        central = abs(cx - w / 2) < CENTRAL_ZONE * w and abs(cy - h / 2) < CENTRAL_ZONE * h
        if conf >= YOLO_MIN_CONF and area >= YOLO_MIN_AREA and central:
            cands.append(((x1, y1, x2, y2), area, conf))
    if not cands:
        return None
    main_area = max(a * c for _, a, c in cands)
    mask = np.zeros((h, w), np.uint8)
    for (x1, y1, x2, y2), area, conf in cands:
        if area * conf >= 0.3 * main_area:   # группа сопоставимых объектов (пара кроссовок, комплект)
            mask[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = 1
    return mask


def _grabcut_mask(image):
    """Выделение объекта в центральной области кадра без знания класса (или None)"""
    h, w = image.shape[:2]
    scale = min(1.0, GRABCUT_SIDE / max(h, w))
    small = cv2.resize(image, (max(8, round(w * scale)), max(8, round(h * scale))), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    rect = (int(sw * 0.08), int(sh * 0.08), int(sw * 0.84), int(sh * 0.84))
    gc = np.zeros((sh, sw), np.uint8)
    try:
        cv2.grabCut(small, gc, rect, np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64),
                    3, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None
    fg = np.isin(gc, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n <= 1:
        return None
    biggest = stats[1:, cv2.CC_STAT_AREA].max()
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 0.1 * biggest]
    fg = np.isin(labels, keep).astype(np.uint8)
    share = fg.mean()
    if share < 0.02 or share > 0.9:
        return None
    return cv2.resize(fg, (w, h), interpolation=cv2.INTER_NEAREST)


def find_subject(image, yolo_data):
    """
    Возвращает маску объекта (в размере image), рамку и метрики композиции.
    fill_rate — доля кадра, занятая рамкой объекта (0..1).
    """
    h, w = image.shape[:2]
    scale = min(1.0, ANALYSIS_SIDE / max(h, w))
    small = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))),
                       interpolation=cv2.INTER_AREA) if scale < 1 else image

    mask_small, uniform_share = _background_mask(small)
    method = "однородный фон"
    mask = None
    if mask_small is not None:
        mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)

    if mask is None:
        mask = _yolo_main_mask(yolo_data, w, h)
        method = "YOLO"

    if mask is None:
        mask = _grabcut_mask(image)
        method = "центральная область"

    if mask is None or not mask.any():
        return {
            "mask": None, "bbox": None, "object_found": False, "fill_rate": 0.0,
            "area_rate": 0.0, "is_off_center": False, "subject_method": "не выделен", "subject_confidence": 0.0,
            "background_uniform": uniform_share >= BG_UNIFORM_SHARE,
        }

    ys, xs = np.nonzero(mask)
    x1, x2, y1, y2 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    fill_rate = (x2 - x1) * (y2 - y1) / float(h * w)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    is_off_center = abs(cx - w / 2) > 0.25 * w or abs(cy - h / 2) > 0.25 * h

    return {
        "mask": mask,
        "bbox": (x1, y1, x2, y2),
        "object_found": True,
        "fill_rate": float(fill_rate),
        "area_rate": float(mask.mean()),
        "is_off_center": bool(is_off_center),
        "subject_method": method,
        # насколько можно доверять рамке объекта при оценке заполнения кадра
        "subject_confidence": {"однородный фон": 1.0, "YOLO": 1.0}.get(method, 0.5),
        "background_uniform": method == "однородный фон",
    }
