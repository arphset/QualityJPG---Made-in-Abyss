"""
Итоговая оценка качества фото по 10-балльной шкале.

Каждая группа проблем даёт плавный штраф (без скачков на порогах), а у группы есть потолок,
чтобы несколько родственных признаков (например, «темно» и «провалы в тенях») не складывались
в непропорционально большой штраф.

  Чёткость       — до 5.5 баллов (главный критерий: размытое фото товара непригодно)
  Экспозиция     — до 4 баллов (темно, засвет, пересвет, тени, контраст, контровой свет)
  Композиция     — до 2.5 баллов
  Качество файла — до 4.5 баллов (разрешение, артефакты сжатия JPEG, цифровой шум)
  Пустой кадр    — 4 балла (оценивать нечего)
"""

SHARP_OK = 0.85         # индекс чёткости, начиная с которого штрафа нет
SHARP_PENALTY_MAX = 5.5
EXPOSURE_CAP = 4.0
COMPOSITION_CAP = 2.5
FILL_OK = 0.25          # доля кадра под объектом, начиная с которой штрафа нет
FILL_MIN = 0.05
MIN_GOOD_SIDE = 800
MIN_SIDE = 300
RESOLUTION_PENALTY_MAX = 4.0   # при 300 px и меньше; 600 px — около 1 балла (штраф растёт нелинейно)
FILE_QUALITY_CAP = 4.5


def _clamp01(x):
    return max(0.0, min(1.0, float(x)))


def _severity(result, key, flag):
    """Степень проблемы 0..1; для старых результатов без степеней — по булеву флагу"""
    if key in result and result[key] is not None:
        return _clamp01(result[key])
    return 1.0 if result.get(flag, False) else 0.0


def _fmt(value):
    return f"{value:.1f}"


def calculate_quality_score(analysis_result: dict) -> dict:
    r = analysis_result
    score = 10.0
    penalties = []

    def apply(label, value):
        nonlocal score
        value = round(value, 1)
        if value >= 0.1:
            score -= value
            penalties.append(f"{label}: -{_fmt(value)}")

    def apply_group(items, cap):
        """Штрафы группы; если сумма выше потолка — все ужимаются пропорционально (каждая проблема видна)"""
        total = sum(value for _, value in items)
        factor = cap / total if total > cap else 1.0
        for label, value in sorted(items, key=lambda item: -item[1]):
            apply(label, value * factor)

    # 1. ЧЁТКОСТЬ
    if r.get("empty_frame", False):
        apply("Кадр почти пустой, товар не виден", 4.0)
    else:
        sharpness = float(r.get("sharpness_score", 1.0) or 0.0)
        penalty = SHARP_PENALTY_MAX * _clamp01((SHARP_OK - sharpness) / (SHARP_OK - 0.05))
        if r.get("is_motion_blur", False):
            label = "Смаз движения"
        elif r.get("is_blurry", False) or sharpness < 0.45:
            label = "Размытие"
        else:
            label = "Пограничная чёткость"
        apply(label, penalty)

    # 2. ЭКСПОЗИЦИЯ
    exposure = [
        ("Слишком темно", 2.5 * _severity(r, "dark_severity", "is_dark")),
        ("Кадр засвечен", 2.0 * _severity(r, "washed_severity", "is_bright")),
        ("Пересвет, выбитые света" + (f" ({r['overexposed_pct']:.0f}%)" if "overexposed_pct" in r else ""),
         2.0 * _severity(r, "highlight_severity", "is_overexposed")
         # выбита большая часть кадра/объекта — штраф растёт с площадью потерянных деталей
         + 2.0 * _clamp01((float(r.get("overexposed_effective_pct", r.get("overexposed_pct", 0)) or 0) - 30) / 40)
         * _severity(r, "highlight_severity", "is_overexposed")),
        ("Провалы в тенях", 0.75 * _severity(r, "shadow_severity", "is_underexposed")),
        ("Низкий контраст", 1.0 * _severity(r, "contrast_severity", "is_low_contrast")),
        ("Объект в тени на фоне яркого света", 1.0 * _severity(r, "backlit_severity", "is_backlit")),
    ]
    apply_group(exposure, EXPOSURE_CAP)

    # 3. КОМПОЗИЦИЯ
    try:
        fill_rate = float(r.get("fill_rate", 0) or 0)
    except (TypeError, ValueError):
        fill_rate = 0.0
    composition = []
    if r.get("empty_frame", False):
        pass
    elif not r.get("object_found", True):
        # Сцена без явного главного объекта (интерьер, пейзаж, сложный фон) — мягкий штраф
        composition.append(("Главный объект не выделен", 0.5))
    else:
        # Если объект выделен приблизительно (по центральной области), штраф за заполнение вдвое мягче
        confidence = _clamp01(r.get("subject_confidence", 1.0))
        composition.append((f"Заполнение кадра {fill_rate*100:.0f}%",
                            2.0 * confidence * _clamp01((FILL_OK - fill_rate) / (FILL_OK - FILL_MIN))))
        if r.get("is_off_center", False) and fill_rate < 0.5:
            composition.append(("Смещение от центра", 0.5))
    apply_group(composition, COMPOSITION_CAP)

    # 4. КАЧЕСТВО ФАЙЛА
    file_quality = []
    if r.get("image_width") and r.get("image_height"):
        long_side = max(r["image_width"], r["image_height"])
        lack = _clamp01((MIN_GOOD_SIDE - long_side) / (MIN_GOOD_SIDE - MIN_SIDE))
        file_quality.append((f"Низкое разрешение ({r['image_width']}x{r['image_height']})",
                             RESOLUTION_PENALTY_MAX * lack ** 1.5))
    file_quality.append(("Артефакты сжатия JPEG", 2.0 * _severity(r, "compression_severity", "is_compressed")))
    file_quality.append(("Цифровой шум", 1.5 * _severity(r, "noise_severity", "is_noisy")))
    apply_group(file_quality, FILE_QUALITY_CAP)

    score = max(1.0, min(10.0, score))

    if score >= 9.0:
        verdict, emoji = "Отлично", "🏆"
    elif score >= 7.5:
        verdict, emoji = "Хорошо", "✅"
    elif score >= 6.0:
        verdict, emoji = "Удовлетворительно", "⚠️"
    elif score >= 4.0:
        verdict, emoji = "Плохо", "❌"
    else:
        verdict, emoji = "Критично", "🚫"

    return {
        "score": round(score, 1),
        "verdict": verdict,
        "emoji": emoji,
        "penalties": penalties,
        "max_score": 10.0,
    }
