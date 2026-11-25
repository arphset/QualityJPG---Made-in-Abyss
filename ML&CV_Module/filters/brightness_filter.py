"""
Проверка экспозиции, не зависящая от цвета товара и фона.

Принципы:
- Белый (или чёрный) однородный фон — норма для маркетплейсов, а не пересвет/недосвет:
  потеря деталей ищется только внутри объекта.
- Тёмный товар (чёрная футболка) на светлом фоне — не недосвет: о недодержке говорит то, что
  во всём кадре нет светлых участков (верхний перцентиль яркости низкий).
- Пересвет — когда заметная часть объекта «выбита» в белое, либо в кадре вообще нет теней (засвечен).
Каждая проблема получает степень 0..1, итоговый штраф плавно зависит от неё.
"""
import cv2
import numpy as np

CLIP_HIGH = 250
CLIP_LOW = 5
MAX_HOLE_SHARE = 0.05   # дыра в маске больше 5% площади объекта — это фон, а не выбитый блик


def _ramp(value, start, end):
    """0 при value=start, 1 при value=end (линейно, с ограничением)"""
    return float(np.clip((value - start) / (end - start), 0.0, 1.0))


def _subject_region(mask):
    """
    Маска объекта с заполненными небольшими «дырами»: выбитый в белое участок, со всех сторон окружённый
    товаром (блик на этикетке, засвеченная грань), тоже относится к объекту. Крупные дыры — это фон,
    видный сквозь товар (внутри кольца, браслета, ручки кружки), и они остаются фоном. Просветы,
    открытые наружу (фон между листьями растения), фоном являются всегда.
    """
    m = (mask > 0).astype(np.uint8)
    padded = cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    ff_mask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
    cv2.floodFill(padded, ff_mask, (0, 0), 1)
    holes = (padded[1:-1, 1:-1] == 0).astype(np.uint8)
    region = m > 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(holes, connectivity=4)
    limit = MAX_HOLE_SHARE * max(1, int(region.sum()))
    small = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] <= limit]
    if small:
        region |= np.isin(labels, small)
    return region


def check_brightness(image, mask=None, background_uniform=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    brightness = float(np.mean(gray))
    p2, p50, p98, p99 = (float(v) for v in np.percentile(gray, (2, 50, 98, 99)))

    # Для недодержки — максимум по каналам (HSV V): насыщенный красный товар не должен считаться тёмным
    value = image.max(axis=2)
    v99 = float(np.percentile(value, 99))

    region = None
    if mask is not None and mask.any():
        region = _subject_region(mask)
    subject = gray[region] if region is not None and region.any() else gray.ravel()
    s5, s50 = (float(v) for v in np.percentile(subject, (5, 50)))

    luma_clip = float(np.mean(subject >= CLIP_HIGH) * 100)
    # Выбитый в один-два канала цвет (жёлтый, светло-зелёный) теряет детали так же, как белый,
    # но яркость (luma) у него ниже 250. Считаем такие пиксели с половинным весом.
    value_subject = value[region] if region is not None and region.any() else value.ravel()
    channel_clip = float(np.mean(value_subject >= 254) * 100)
    overexposed_pct = max(luma_clip, 0.5 * channel_clip)
    underexposed_pct = float(np.mean(subject <= CLIP_LOW) * 100)

    # Недодержка всего кадра: даже самые светлые участки тёмные
    dark_severity = _ramp(-v99, -170, -80)
    # Выбитые света на объекте (блики на глянце — норма, поэтому порог в несколько процентов)
    highlight_severity = _ramp(overexposed_pct, 6, 30)

    # Передержка по форме гистограммы: у нормального фото светлые тона плавно сходят на нет до 255,
    # у передержанного — «упираются в потолок»: выбитых пикселей больше, чем просто светлых.
    # Объект оценивается полностью. Для сцен дополнительно смотрим весь кадр, но с половинным весом:
    # выбитое небо или окно — дефект, но меньший, чем выбитый сам товар. Студийный белый фон не в счёт.
    def _clipping(values):
        blown = float(np.count_nonzero(values >= 253))
        light = float(np.count_nonzero((values >= 200) & (values < 253)))
        pile = blown / max(1.0, light)
        pct = blown / max(1, values.size) * 100
        return min(_ramp(pile, 1.0, 2.5), _ramp(pct, 4, 15)), pile, pct

    # blown_pct — реальная доля выбитого (её видит пользователь и пишем в БД),
    # effective_pct — с учётом половинного веса фона (идёт только в расчёт штрафа)
    clipping_severity, pile_up, blown_pct = _clipping(value_subject)
    effective_pct = blown_pct
    if not background_uniform and region is None:
        # сцена без выделенного объекта (пейзаж, улица): выбитое здесь — в основном небо
        clipping_severity, effective_pct = 0.5 * clipping_severity, 0.5 * blown_pct
    elif not background_uniform:
        frame_sev, frame_pile, frame_pct = _clipping(value.ravel())
        if 0.5 * frame_sev > clipping_severity:
            clipping_severity, pile_up = 0.5 * frame_sev, frame_pile
            blown_pct, effective_pct = frame_pct, 0.5 * frame_pct
    highlight_severity = max(highlight_severity, clipping_severity)
    overexposed_effective_pct = max(overexposed_pct, effective_pct if clipping_severity > 0 else 0.0)
    overexposed_pct = max(overexposed_pct, blown_pct if clipping_severity > 0 else 0.0)
    # Засвеченный кадр: нет теней ни в кадре, ни на объекте
    washed_severity = min(_ramp(p2, 110, 190), _ramp(s5, 150, 215))
    # Провалы в чёрное на объекте
    # (чёрный товар сам по себе почти чёрный, поэтому порог высокий)
    shadow_severity = _ramp(underexposed_pct, 35, 75)
    # Низкий контраст (туман, грязный объектив, серая «пелена»)
    contrast = p98 - p2
    contrast_severity = _ramp(-contrast, -80, -30)

    # Контровой свет: объект тёмный, а позади него яркий, местами выбитый свет (окно, небо).
    # На однородном студийном фоне не проверяем: чёрный товар на белом — это норма, а не контровой свет.
    backlit_severity = 0.0
    subject_value = bg_value = bg_clip = 0.0
    if region is not None and region.any() and (~region).any() and not background_uniform:
        subject_value = float(np.median(value[region]))
        background = value[~region]
        bg_value = float(np.median(background))
        bg_clip = float(np.mean(background >= 250) * 100)
        backlit_severity = min(_ramp(-subject_value, -100, -55),          # объект тёмный
                               _ramp(bg_value - subject_value, 70, 130),  # фон намного светлее
                               _ramp(bg_clip, 8, 25))                    # и местами выбит
    is_backlit = backlit_severity >= 0.3

    is_dark = dark_severity >= 0.3
    is_overexposed = highlight_severity >= 0.3
    is_bright = washed_severity >= 0.3
    is_underexposed = shadow_severity >= 0.3
    is_low_contrast = contrast_severity >= 0.3

    print(f"[2] Экспозиция: средняя {brightness:.0f}, p2={p2:.0f}, p99={p99:.0f}, объект: медиана {s50:.0f}, "
          f"выбито {overexposed_pct:.1f}%, провалы {underexposed_pct:.1f}%")

    return {
        "brightness": brightness,
        "overexposed_pct": overexposed_pct,
        "underexposed_pct": underexposed_pct,
        "subject_brightness": s50,
        "contrast": contrast,
        "highlights_p99": v99,
        "shadows_p2": p2,
        "dark_severity": dark_severity,
        "highlight_severity": highlight_severity,
        "washed_severity": washed_severity,
        "shadow_severity": shadow_severity,
        "contrast_severity": contrast_severity,
        "backlit_severity": backlit_severity,
        "highlight_pile_up": pile_up,
        "overexposed_effective_pct": overexposed_effective_pct,
        "is_backlit": is_backlit,
        "is_overexposed": is_overexposed,
        "is_underexposed": is_underexposed,
        "is_dark": is_dark,
        "is_bright": is_bright,
        "is_low_contrast": is_low_contrast,
    }
