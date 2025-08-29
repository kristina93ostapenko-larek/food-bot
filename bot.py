#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FoodBot — телеграм-бот «что приготовить из того, что есть»
• aiogram 3.7+ (FSM, inline-кнопки)
• OpenAI Async SDK (основная модель + авто-фолбэк)
• Стрим с основной моделью; при ошибке/пустом ответе — переключение на gpt-4o-mini
• Правильные параметры: max_completion_tokens (без temperature для моделей, где не поддерживается)
• Устойчивый long-poll, сброс webhook, диагностика /ping, /id
"""

import os
import re
import time
import random
import asyncio
import logging
from typing import List, Dict, Optional, Any, Tuple
from logging.handlers import RotatingFileHandler
from functools import lru_cache
from contextlib import suppress

from dotenv import load_dotenv

from aiogram import Bot, Dispatcher, types, F
from aiogram.enums import ParseMode
from aiogram.filters import Command, BaseFilter
from aiogram.client.default import DefaultBotProperties
from aiogram.utils.chat_action import ChatActionSender

# FSM
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import StatesGroup, State
from aiogram.fsm.storage.memory import MemoryStorage

# ===================== КОНФИГ =====================

class Config:
    MAX_PRODUCTS = 15
    RATE_LIMIT_PER_MIN = 5
    REQUEST_TIMEOUT = 60              # общий сторож на генерацию
    MAX_RETRIES = 2                   # ретраи стрима основной модели
    CACHE_SIZE = 128

    # Модели
    OPENAI_MODEL = "gpt-4"            # основная модель (если недоступна — фолбэк)
    OPENAI_FALLBACK_MODEL = "gpt-3.5-turbo"
    OPENAI_MAX_TOKENS = 1200          # используем max_completion_tokens

    POLLING_TIMEOUT = 25
    RECONNECT_DELAY = 5
    MAX_RECONNECT_ATTEMPTS = 0        # 0 = бесконечно

    EDIT_INTERVAL_SEC = 0.6           # частота редактирования стрима

config = Config()

# ================= ЛОГГИРОВАНИЕ ==================

logger = logging.getLogger("foodbot")
logger.setLevel(logging.INFO)
fmt = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')
file_handler = RotatingFileHandler('foodbot.log', maxBytes=2 * 1024 * 1024, backupCount=3, encoding='utf-8')
file_handler.setFormatter(fmt)
console_handler = logging.StreamHandler()
console_handler.setFormatter(fmt)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# ================== ENV / OpenAI ==================

load_dotenv()
TOKEN = os.getenv('TELEGRAM_TOKEN')
OPENAI_KEY = os.getenv('OPENAI_API_KEY')
if not TOKEN or not OPENAI_KEY:
    logger.critical("Отсутствуют TELEGRAM_TOKEN или OPENAI_API_KEY в .env")
    raise SystemExit(1)

from openai import AsyncOpenAI
from aiohttp import ServerDisconnectedError, ClientOSError
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.exceptions import TelegramNetworkError
oa_client = AsyncOpenAI(api_key=OPENAI_KEY)

# Глобальный bot создадим после запуска event loop
bot: Optional[Bot] = None

# ================== FSM ==================

class RecipeFlow(StatesGroup):
    choosing_meal = State()
    entering_ingredients = State()

# ============== Рейт-лимит ==============

feedback_stats: Dict[str, int] = {"👍": 0, "👎": 0}
_user_window: Dict[int, List[float]] = {}

class RateLimitFilter(BaseFilter):
    async def __call__(self, message: types.Message) -> bool:
        uid = message.from_user.id
        now = time.time()
        bucket = _user_window.setdefault(uid, [])
        bucket = [t for t in bucket if now - t < 60]
        _user_window[uid] = bucket
        if len(bucket) >= config.RATE_LIMIT_PER_MIN:
            await message.answer("⚠️ Слишком много сообщений. Подождите 1 минуту и попробуйте снова.")
            return False
        bucket.append(now)
        return True

rate_limit = RateLimitFilter()

# ================== UI / Тексты ==================

def meal_keyboard() -> types.InlineKeyboardMarkup:
    return types.InlineKeyboardMarkup(inline_keyboard=[
        [
            types.InlineKeyboardButton(text="🥐 Завтрак", callback_data="meal:Завтрак"),
            types.InlineKeyboardButton(text="🍲 Обед", callback_data="meal:Обед"),
        ],
        [
            types.InlineKeyboardButton(text="🍽 Ужин", callback_data="meal:Ужин"),
            types.InlineKeyboardButton(text="🎲 Удиви меня", callback_data="meal:Удиви меня"),
        ]
    ])

def feedback_keyboard() -> types.InlineKeyboardMarkup:
    return types.InlineKeyboardMarkup(inline_keyboard=[
        [types.InlineKeyboardButton(text="👍", callback_data="fb:up"),
         types.InlineKeyboardButton(text="👎", callback_data="fb:down")],
        [types.InlineKeyboardButton(text="🔁 Новый запрос", callback_data="restart")]
    ])

WELCOME = (
    "👋 *Привет! Я NamutiFoodBot.*\n\n"
    "Не хочешь докупать продукты? Помогу приготовить еду из того, что осталось в холодильнике.\n\n"
    "Сначала выбери тип приёма пищи:"
)

HELP = (
    "ℹ️ *Как пользоваться:*\n"
    "1) Выбери: завтрак / обед / ужин / «удиви меня».\n"
    "2) Перечисли ингредиенты через запятую — я предложу несколько простых рецептов.\n"
    f"3) Максимум {config.MAX_PRODUCTS} ингредиентов.\n"
    "4) Я печатаю рецепты по мере генерации.\n"
)

def render_header(meal: str) -> str:
    emoji = {"Завтрак": "🥐", "Обед": "🍲", "Ужин": "🍽"}.get(meal, "🎲")
    return f"{emoji} *Подбор рецептов* · _{meal}_\n"

def render_footer() -> str:
    return "\n—\nНу как? 👇"

# ================== Утилиты ==================

def normalize_products(text: str) -> List[str]:
    if not text:
        return []
    lowered = text.lower().strip()
    unified = re.sub(r'\s*(и|,|;|\n|\r)\s*', ',', lowered)
    parts = [p.strip() for p in unified.split(',') if p.strip()]
    seen, uniq = set(), []
    for p in parts:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq

@lru_cache(maxsize=config.CACHE_SIZE)
def system_prompt() -> str:
    return (
        "Ты — креативный и практичный шеф-повар. Твоя задача — составить рецепты ИСКЛЮЧИТЕЛЬНО из предложенных ингредиентов.\n\n"
        "**АБСОЛЮТНЫЕ ЗАПРЕТЫ (нарушать нельзя!):**\n"
        "1) 🚫 ЗАПРЕЩЕНО использовать любые ингредиенты, которых нет в списке пользователя. Даже если блюдо классически готовится с ними.\n"
        "2) 🚫 ЗАПРЕЩЕНО предлагать добавлять ингредиенты, которых нет в списке (ни в советах, ни в шагах).\n"
        "3) 🚫 ЗАПРЕЩЕНО заменять ингредиенты на другие (например, курицу на краба).\n\n"
        "**Разрешены ТОЛЬКО эти базовые продукты (и то только если они логично дополняют рецепт):**\n"
        "соль, перец, растительное/сливочное масло, вода, мука, сахар, специи.\n\n"
        "**ВАЖНОЕ ПРАВИЛО ДЛЯ КОЛИЧЕСТВ ИНГРЕДИЕНТОВ:**\n"
        "1) 🟢 ОБЯЗАТЕЛЬНО указывай примерные количества для ВСЕХ ингредиентов\n"
        "2) 🟢 Используй стандартные меры: граммы (г), миллилитры (мл), столовые/чайные ложки (ст.л./ч.л.), штуки (шт.), щепотки\n"
        "3) 🟢 Для базовых разрешенных продуктов указывай реалистичные количества (например: 1 ст.л. масла, 100 г муки, щепотка соли)\n"
        "4) 🟢 Для основных ингредиентов из списка пользователя указывай примерные пропорции относительно других ингредиентов\n\n"
        "**Правила генерации рецептов:**\n"
        "1) **Сбалансированность:** Старайся предложить меню из 2-3 сочетающихся блюд (суп + горячее + салат/гарнир).\n"
        "2) **Креативность:** Избегай примитивных рецептов ('жареный X'). Предлагай интересные блюда: запеканки, рагу, фаршированные овощи, котлеты, супы-пюре.\n"
        "3) **Обязательно предлагай супы:** Если есть овощи и жидкость (вода/бульон/молоко/сливки) — предложи суп.\n"
        "4) **Полное использование:** Старайся задействовать максимальное количество из предложенных ингредиентов.\n\n"
        "**Качество советов:**\n"
        "- Совет должен быть НЕТРИВИАЛЬНЫМ. Если нет хорошей идеи — не добавляй блок 'Совет:'.\n"
        "- 🚫 ЗАПРЕЩЕНО: 'подавать горячим', 'посолить по вкусу' — это очевидные вещи.\n"
        "- ✅ Разрешено: лайфхаки по приготовлению, неочевидные сочетания, советы по подаче.\n\n"
        "**ФОРМАТ ОТВЕТА (Markdown):**\n"
        "*Название блюда*\n"
        "Ингредиенты: (только те, что используются в этом блюде из списка пользователя + разрешенные базовые)\n"
        "  - [ингредиент 1] - [количество, например: 200 г]\n"
        "  - [ингредиент 2] - [количество, например: 2 ст.л.]\n"
        "  - [ингредиент 3] - [количество, например: 1 шт.]\n"
        "Шаги:\n"
        "1) ... (чёткие шаги, 5-7 пунктов, с указанием количеств где это уместно)\n"
        "_Совет:_ (ТОЛЬКО если есть действительно полезный и неочевидный совет)\n\n"
        "**ПРИМЕР ПРАВИЛЬНОГО ФОРМАТА:**\n"
        "*Омлет с сыром*\n"
        "Ингредиенты:\n"
        "  - яйца - 3 шт.\n"
        "  - сыр - 50 г\n"
        "  - растительное масло - 1 ст.л.\n"
        "  - соль - щепотка\n"
        "  - перец - по вкусу\n"
        "Шаги:\n"
        "1) Взбейте яйца с солью и перцем...\n\n"
        "Проверь каждый рецепт на соответствие ингредиентам и указание количеств перед отправкой!"
    )

def build_messages(products: List[str], meal_type: str) -> List[Dict[str, Any]]:
    user_msg = (
        f"Тип приёма пищи: {meal_type or 'любой'}.\n"
        f"Ингредиенты: {', '.join(products)}.\n"
        f"ВАЖНО: используй только эти ингредиенты, не предлагай добавлять новые.\n"
        f"ОБЯЗАТЕЛЬНО указывай примерные количества для всех ингредиентов в понятных единицах измерения (г, мл, ст.л., ч.л., шт., щепотки).\n"
        f"Названия блюд оформляй как *жирный текст* используя звездочки: *Название блюда*"
    )
    return [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": user_msg}
    ]

def format_dish_names(text: str) -> str:
    """Форматирует названия блюд в жирный текст"""
    # Простой подход: ищем строки, которые выглядят как названия блюд
    # и оборачиваем их в **
    lines = text.split('\n')
    formatted_lines = []
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Если строка выглядит как название блюда (не пустая, не начинается с дефиса, не номер шага)
        if (stripped and 
            not stripped.startswith('-') and 
            not stripped.startswith('Шаги:') and 
            not stripped.startswith('Ингредиенты:') and
            not re.match(r'^\d+[\)\.]', stripped) and  # не начинается с цифры и точки/скобки
            not stripped.startswith('_Совет:') and
            (i == 0 or not lines[i-1].strip()) and  # предыдущая строка пустая или это начало
            not stripped.startswith('*') and  # уже не форматировано
            len(stripped) < 100):  # не слишком длинное
            formatted_lines.append(f"*{stripped}*")
        else:
            formatted_lines.append(line)
    
    return '\n'.join(formatted_lines)

# ===== OpenAI вызовы =====

async def openai_stream(model: str, messages: List[Dict[str, Any]]) -> Tuple[bool, str, Optional[str]]:
    """
    Возвращает (ok, text, error_code).
    ok=True и text непустой — успех.
    ok=False и error_code — текстовый код ошибки/диагностика.
    """
    content_buf = ""
    try:
        kwargs = dict(model=model, max_completion_tokens=config.OPENAI_MAX_TOKENS, messages=messages, stream=True)
        stream = await oa_client.chat.completions.create(**kwargs)
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                content_buf += chunk.choices[0].delta.content
        if content_buf.strip():
            return True, content_buf, None
        return False, "", "empty_stream"
    except Exception as e:
        return False, "", str(e)

async def openai_non_stream(model: str, messages: List[Dict[str, Any]]) -> Tuple[bool, str, Optional[str]]:
    try:
        kwargs = dict(model=model, max_completion_tokens=config.OPENAI_MAX_TOKENS, messages=messages, stream=False)
        resp = await oa_client.chat.completions.create(**kwargs)
        text = (resp.choices[0].message.content or "").strip()
        if text:
            return True, text, None
        return False, "", "empty_response"
    except Exception as e:
        return False, "", str(e)

# ================== Стрим с авто-фолбэком ==================

async def stream_with_fallback(
    primary_model: str,
    fallback_model: str,
    messages: List[Dict[str, Any]],
) -> Tuple[bool, str, str]:
    """
    Пытаемся: 
      1) стрим с primary_model (с ретраями),
      2) если пусто/ошибка — нестриминговый запрос primary_model,
      3) если пусто/ошибка — нестриминговый запрос fallback_model.
    Возвращает (ok, text, used_model)
    """
    # 1) primary stream с ретраями
    last_err = None
    for attempt in range(1, config.MAX_RETRIES + 1):
        ok, text, err = await openai_stream(primary_model, messages)
        if ok and text.strip():
            return True, text, primary_model
        last_err = err
        await asyncio.sleep(min(2 ** (attempt - 1), 4))  # backoff

    # 2) primary non-stream
    ok, text, err = await openai_non_stream(primary_model, messages)
    if ok and text.strip():
        return True, text, primary_model
    last_err = err

    # 3) fallback non-stream
    ok, text, err = await openai_non_stream(fallback_model, messages)
    if ok and text.strip():
        return True, text, fallback_model

    # Логируем причину
    logger.error(f"All attempts failed. Primary err: {last_err}, Fallback err: {err}")
    return False, "", fallback_model

# ================== UI-стрим для чата ==================

async def stream_gpt_to_message(
    chat_id: int,
    header: str,
    footer: str,
    messages: List[Dict[str, Any]],
    reply_markup: Optional[types.InlineKeyboardMarkup] = None,
) -> None:
    assert bot is not None
    msg = await bot.send_message(chat_id, "⏳ _Готовлю рецепты…_")

    # Получаем ответ от GPT
    ok, text, used_model = await stream_with_fallback(
        config.OPENAI_MODEL,
        config.OPENAI_FALLBACK_MODEL,
        messages
    )
    
    if not ok or not text.strip():
        await msg.edit_text("❌ Не удалось сгенерировать рецепты. Попробуйте позже.")
        return

    # Форматируем ответ - делаем названия блюд жирными
    formatted_text = format_dish_names(text)
    full_text = f"{header}{formatted_text}{footer}"
    
    # Отправляем результат
    try:
        await msg.edit_text(full_text, reply_markup=reply_markup)
    except Exception as e:
        logger.error(f"Ошибка редактирования сообщения: {e}")
        # Если сообщение слишком длинное, разбиваем на части
        if "message is too long" in str(e):
            parts = []
            current_part = header
            for line in formatted_text.split('\n'):
                if len(current_part + line + '\n') > 4096:
                    parts.append(current_part)
                    current_part = line + '\n'
                else:
                    current_part += line + '\n'
            parts.append(current_part + footer)
            
            await msg.delete()
            for i, part in enumerate(parts):
                if i == len(parts) - 1:
                    await bot.send_message(chat_id, part, reply_markup=reply_markup)
                else:
                    await bot.send_message(chat_id, part)
        else:
            await msg.edit_text("❌ Ошибка форматирования сообщения. Попробуйте снова.")

# ================== Инициализация бота ==================

# Создаем диспетчер с хранилищем
storage = MemoryStorage()
dp = Dispatcher(storage=storage)

# ================== Обработчики ==================

@dp.message(Command("ping"))
async def cmd_ping(message: types.Message):
    await message.answer("pong 🟢")

@dp.message(Command("id"))
async def cmd_id(message: types.Message):
    await message.answer(f"Ваш chat_id: `{message.chat.id}`")

@dp.message(Command("start"))
async def cmd_start(message: types.Message, state: FSMContext):
    await state.clear()
    await state.set_state(RecipeFlow.choosing_meal)
    await message.answer(WELCOME, reply_markup=meal_keyboard())

@dp.message(Command("help"))
async def cmd_help(message: types.Message):
    await message.answer(HELP)

@dp.callback_query(F.data.startswith("meal:"))
async def on_meal_chosen(cq: types.CallbackQuery, state: FSMContext):
    await cq.answer()
    meal = cq.data.split(":", 1)[1]
    await state.update_data(meal_type=meal)
    await state.set_state(RecipeFlow.entering_ingredients)
    await cq.message.edit_reply_markup(reply_markup=None)
    await cq.message.answer(
        f"{render_header(meal)}Напишите список ингредиентов через запятую.\n"
        "_Пример_: `яйца, сыр, томаты` или `курица и рис, брокколи`."
    )

@dp.message(RecipeFlow.entering_ingredients, rate_limit)
async def on_ingredients(message: types.Message, state: FSMContext):
    data = await state.get_data()
    meal = data.get("meal_type", "Удиви меня")

    products = normalize_products(message.text or "")
    if not products:
        return await message.answer("Не вижу ингредиентов. Пример: _курица, рис, брокколи_")
    if len(products) > config.MAX_PRODUCTS:
        return await message.answer(f"⚠️ Слишком много позиций. Максимум {config.MAX_PRODUCTS}.")

    header = render_header(meal)
    messages = build_messages(products, meal)

    async with ChatActionSender.typing(bot=bot, chat_id=message.chat.id):
        await stream_gpt_to_message(
            chat_id=message.chat.id,
            header=header,
            footer=render_footer(),
            messages=messages,
            reply_markup=feedback_keyboard(),
        )

    await state.clear()

@dp.callback_query(F.data.in_({"fb:up", "fb:down"}))
async def on_feedback(cq: types.CallbackQuery):
    await cq.answer("Спасибо!")
    if cq.data == "fb:up":
        feedback_stats["👍"] += 1
    else:
        feedback_stats["👎"] += 1
    await cq.message.edit_reply_markup(reply_markup=None)
    await cq.message.answer("🙏 Спасибо за оценку! Нажмите «/start», чтобы начать заново.")

@dp.callback_query(F.data == "restart")
async def on_restart(cq: types.CallbackQuery, state: FSMContext):
    await cq.answer()
    await state.clear()
    await cq.message.edit_reply_markup(reply_markup=None)
    await cq.message.answer(WELCOME, reply_markup=meal_keyboard())

@dp.message(rate_limit)
async def fallback(message: types.Message, state: FSMContext):
    await message.answer("Я вас не понял 🤖\nНажмите /start или /help")

# ================== Жизненный цикл ==================

def make_bot(token: str) -> Bot:
    """Создаёт Bot с устойчивой aiohttp-сессией и корректными таймаутами."""
    session = AiohttpSession()
    return Bot(token=token, session=session, default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN))

async def on_startup(bot: Bot):
    try:
        await bot.delete_webhook(drop_pending_updates=True)
        me = await bot.get_me()
        logger.info(f"🟢 Webhook удалён. Бот: @{me.username} (id={me.id})")
    except Exception as e:
        logger.warning(f"Не удалось удалить webhook или получить getMe: {e}")
    logger.info("🟢 Бот успешно запущен (polling)")

async def on_shutdown(bot: Bot):
    logger.info("🔴 Бот останавливается")
    try:
        await bot.session.close()
    except Exception:
        pass

# =============== Запуск бота (polling) ===============

async def run_polling() -> None:
    """Устойчивый запуск polling с экспоненциальным бэкоффом."""
    if not TOKEN:
        raise RuntimeError("TELEGRAM_TOKEN не найден")

    global bot
    attempt = 0
    while True:
        try:
            bot = make_bot(TOKEN)
            await on_startup(bot)
            
            await dp.start_polling(
                bot, 
                polling_timeout=config.POLLING_TIMEOUT,
                handle_signals=False,
                allowed_updates=dp.resolve_used_update_types()
            )
            attempt = 0
            break  # штатное завершение polling
            
        except (ServerDisconnectedError, ClientOSError, asyncio.TimeoutError, TelegramNetworkError) as e:
            attempt += 1
            wait = min(60, 2 ** attempt)
            logger.warning(f"Сетевая ошибка: {type(e).__name__}: {e}. Повтор через {wait} сек (попытка {attempt})")
            await asyncio.sleep(wait)
            
        except Exception as e:
            logger.exception(f"Неожиданная ошибка: {e}")
            await asyncio.sleep(5)
            attempt += 1
            
        finally:
            if bot:
                with suppress(Exception):
                    await on_shutdown(bot)
                with suppress(Exception):
                    await bot.session.close()

if __name__ == "__main__":
    try:
        asyncio.run(run_polling())
    except KeyboardInterrupt:
        logger.info("🛑 Бот остановлен вручную")
