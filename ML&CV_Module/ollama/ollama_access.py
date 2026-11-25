import os
import httpx
from openai import OpenAI
from filters.score_calculator import FILL_OK, SHARP_OK


# Отключаем прокси
for proxy_var in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
    os.environ.pop(proxy_var, None)

# Адрес Ollama из переменной окружения
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "60"))
# Модель держится в памяти Ollama между запросами (иначе выгружается через 5 минут простоя)
OLLAMA_KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "60m")
# Ограничение длины ответа: оценка + 3–5 рекомендаций укладываются с запасом
MAX_TOKENS = int(os.environ.get("OLLAMA_MAX_TOKENS", "350"))

LLM_ERROR_PREFIX = "⚠️ Ошибка связи с Ollama"

# КЛИЕНТ БЕЗ ПРОКСИ — trust_env=False игнорирует ВСЕ системные прокси.
# Создаётся один раз: переиспользует HTTP-соединение между запросами.
_client = OpenAI(
    base_url=f"{OLLAMA_HOST}/v1",
    api_key="ollama",
    http_client=httpx.Client(timeout=OLLAMA_TIMEOUT, trust_env=False),
)


def _safe_fmt(value, format_spec):
    try:
        if value is None:
            return "N/A"
        return f"{float(value):{format_spec}}"
    except (ValueError, TypeError):
        return "N/A"


def _build_prompt(data: dict):
    """
    Возвращает (блок метрик, промпт).
    Блок метрик формируется кодом и подставляется в начало отчёта — раньше LLM переписывала
    его сама (~150 лишних токенов на каждый запрос, основная часть времени ответа на CPU).
    """
    fill_rate_val = data.get('fill_rate')
    fill_rate_display = _safe_fmt(fill_rate_val * 100 if fill_rate_val is not None else None, '.1f')

    metrics_text = (
        f"ACF (автокорреляция, справочно, в оценку не входит): {_safe_fmt(data.get('acf_score'), '.3f')}\n"
        f"Лапласиан глобальный (текстура): {_safe_fmt(data.get('laplacian_var'), '.0f')}\n"
        f"Лапласиан локальный (медиана по блокам): {_safe_fmt(data.get('median_laplacian'), '.0f')}\n"
        f"Вариация градиента Собеля (чёткость краёв): {_safe_fmt(data.get('sobel_var'), '.0f')}\n"
        f"Индекс чёткости (0.0-1.0): {_safe_fmt(data.get('sharpness_score'), '.2f')}\n"
        f"Ширина краёв (px, меньше — резче): {_safe_fmt(data.get('edge_width'), '.2f')}\n"
        f"Шум (оценка σ, норма ≤ 1.6): {_safe_fmt(data.get('noise_sigma'), '.1f')}\n"
        f"Качество JPEG (0 — не JPEG; норма ≥ 55): {_safe_fmt(data.get('jpeg_quality'), '.0f')}\n"
        f"Яркость: {_safe_fmt(data.get('brightness'), '.1f')}\n"
        f"Пересвет: {_safe_fmt(data.get('overexposed_pct'), '.1f')}%\n"
        f"Недосвет: {_safe_fmt(data.get('underexposed_pct'), '.1f')}%\n"
        f"Заполнение кадра: {fill_rate_display}%\n"
        f"Смещение от центра: {'Да' if data.get('is_off_center') else 'Нет'}\n"
        f"Товар найден: {'Да' if data.get('object_found', True) else 'Нет'}"
    )

    # Шум и качество JPEG показываются пользователю в блоке метрик, но в промпт не передаются:
    # маленькая модель (3B) от лишних чисел чаще сбивается с формата ответа, а сами проблемы
    # и так перечислены в списке ПРОБЛЕМЫ.
    prompt_metrics = "\n".join(line for line in metrics_text.split("\n")
                               if not line.startswith(("Шум", "Качество JPEG")))

    issues = data.get('issues', [])
    issues_text = ", ".join(issues) if issues else "Проблем не выявлено"

    sharpness_val = _safe_fmt(data.get('sharpness_score'), '.2f')

    # Вердикты по чёткости и заполнению считаются в Python по тем же порогам, что и итоговая оценка
    # (filters/score_calculator.py), чтобы текст LLM не противоречил баллу
    try:
        sharpness = float(data.get('sharpness_score', 0) or 0)
    except (TypeError, ValueError):
        sharpness = 0.0
    if data.get('empty_frame'):
        sharpness_verdict = "В кадре почти нет деталей, товар не виден"
    elif sharpness >= SHARP_OK:
        sharpness_verdict = f"Фото чёткое (индекс чёткости {sharpness_val})"
    elif sharpness >= 0.45 and not data.get('is_blurry'):
        sharpness_verdict = f"Чёткость приемлемая, но не идеальная (индекс чёткости {sharpness_val})"
    elif data.get('is_motion_blur'):
        sharpness_verdict = f"Фото смазано движением (индекс чёткости {sharpness_val})"
    else:
        sharpness_verdict = f"Фото размытое (индекс чёткости {sharpness_val})"

    try:
        fill_rate_val = float(data.get('fill_rate', 0) or 0)
    except (TypeError, ValueError):
        fill_rate_val = 0

    if not data.get('object_found', True):
        fill_comment = ("Главный объект автоматически не выделен (сцена, интерьер или сложный фон). "
                        "НЕ делай выводов о заполнении кадра, можно посоветовать выделить товар крупнее.")
    elif fill_rate_val >= FILL_OK:
        fill_comment = f"Заполнение кадра в норме (>= {FILL_OK*100:.0f}%). НЕ упоминай заполнение как проблему."
    elif fill_rate_val >= 0.15:
        fill_comment = f"Заполнение кадра {fill_rate_val*100:.1f}% — товар стоит снять крупнее."
    else:
        fill_comment = f"Заполнение кадра {fill_rate_val*100:.1f}% — критически мало, товар плохо виден."

    prompt = f"""Ты эксперт по фотографии для маркетплейсов. Дай краткий отчёт строго по формату ниже. НЕ повторяй инструкции и метрики. НЕ пиши заголовки разделов типа "ШАБЛОН" или "ПРАВИЛА".

МЕТРИКИ:
{prompt_metrics}

ПРОБЛЕМЫ: {issues_text}

ФРАЗА О ЧЁТКОСТИ (используй её как есть в разделе "Оценка качества"):
{sharpness_verdict}

КОММЕНТАРИЙ ПО ЗАПОЛНЕНИЮ (учти в рекомендациях):
{fill_comment}

ФОРМАТ ОТВЕТА (пример):

Оценка качества:
Фото чёткое (индекс чёткости 0.85). Товар хорошо виден, экспозиция в норме.

Рекомендации:
1. Фото готово к публикации.
2. Можно немного увеличить заполнение кадра.
3. Попробуйте добавить больше деталей в фон.

ТВОЙ ОТВЕТ (начни со строки "Оценка качества:", точно в таком же формате, только с реальными данными из МЕТРИКИ и ПРОБЛЕМЫ выше):"""

    return f"Метрики фото:\n{metrics_text}\n\n", prompt


# Запасные советы на случай, если модель не дописала раздел «Рекомендации» (маленькая модель
# при temperature 0.7 изредка останавливается после оценки). Ключ — фрагмент текста проблемы.
_FALLBACK_TIPS = [
    (("размыт", "смаз", "чёткость"), "Переснимите товар: держите камеру неподвижно или используйте опору, "
                                     "наведите фокус на товар."),
    (("тёмное", "темно"), "Добавьте света или увеличьте экспозицию."),
    (("тени на фоне",), "Не снимайте против окна или солнца: свет должен падать на товар."),
    (("засвечен", "пересвет"), "Уменьшите экспозицию и уберите прямой свет или вспышку."),
    (("провалы в тенях",), "Подсветите тёмные участки товара (отражатель, второй источник света)."),
    (("контраст",), "Протрите объектив и снимайте при более ярком, направленном свете."),
    (("мало места", "смещён"), "Снимите товар крупнее и разместите его по центру кадра."),
    (("не удалось выделить",), "Снимите товар на однородном фоне, чтобы он был главным в кадре."),
    (("разрешение",), "Загрузите фото в исходном размере — не меньше 800 px по длинной стороне."),
    (("сжато",), "Загрузите оригинал фото без пересжатия (не из мессенджера)."),
    (("шум",), "Снимайте при хорошем освещении: шум появляется при нехватке света."),
]


def _fallback_recommendations(data: dict):
    tips = []
    for issue in data.get('issues', []):
        low = issue.lower()
        for keys, tip in _FALLBACK_TIPS:
            if any(k in low for k in keys) and tip not in tips:
                tips.append(tip)
    if not tips:
        tips = ["Фото готово к публикации."]
    return "\n\nРекомендации:\n" + "\n".join(f"{i}. {t}" for i, t in enumerate(tips[:4], 1))


def stream_recommendations(data: dict):
    """
    Генератор текста рекомендаций по частям (для потокового вывода в интерфейсе).
    Первым фрагментом отдаётся блок метрик, затем — ответ модели по мере генерации.
    Ошибки не пробрасываются: возвращается понятное сообщение (R.01.03).
    """
    if "error" in data:
        yield f"❌ Анализ не выполнен: {data['error']}. Проверьте имя файла и путь."
        return

    metrics_block, prompt = _build_prompt(data)
    yield metrics_block

    try:
        stream = _client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=MAX_TOKENS,
            timeout=OLLAMA_TIMEOUT,
            stream=True,
        )
        answer = []
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                answer.append(chunk.choices[0].delta.content)
                yield chunk.choices[0].delta.content
        if "Рекомендации" not in "".join(answer):
            yield _fallback_recommendations(data)
    except Exception as e:
        yield f"\n{LLM_ERROR_PREFIX}: {str(e)}. Проверьте, запущен ли сервер."


def get_recommendations(data: dict):
    """Генерирует рекомендации через локальную Ollama (целиком, без потока)"""
    return "".join(stream_recommendations(data))


def warmup():
    """Загрузить модель в память Ollama заранее, чтобы первый запрос не ждал холодного старта"""
    try:
        httpx.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": "", "keep_alive": OLLAMA_KEEP_ALIVE},
            timeout=OLLAMA_TIMEOUT,
            trust_env=False,
        )
        return True
    except Exception as e:
        print(f"Ollama недоступна при прогреве: {e}")
        return False
