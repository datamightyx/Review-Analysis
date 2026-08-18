"""Пароль на вхід у Streamlit-застосунок.

Кожна сторінка Streamlit — окремий скрипт, який виконується самостійно, тому
`require_auth()` треба викликати НА КОЖНІЙ сторінці. Сторінка без цього виклику
відкривається за прямим URL повз логін.

Налаштування паролю:

    python auth.py
    # ввести пароль, скопіювати рядок APP_PASSWORD_HASH=... у .env

Джерела паролю, у порядку пріоритету:
  1. APP_PASSWORD_HASH  — pbkdf2_sha256$<iterations>$<salt_hex>$<hash_hex>
  2. st.secrets["auth"]["password_hash"] — той самий формат
  3. APP_PASSWORD — відкритим текстом (працює, але хеш кращий)

Якщо жодне не задане, вхід закритий: застосунок скаже, як згенерувати хеш.
Це навмисно — «немає паролю = пускаємо всіх» тихо зняло б захист.

Межі цього захисту (важливо розуміти перед тим, як віддавати назовні):
  * стан входу живе в st.session_state, тобто в пам'яті сервера на час
    websocket-сесії; перезавантаження вкладки = новий логін;
  * лічильник невдалих спроб теж посесійний, тож від наполегливого перебору з
    новими сесіями він не рятує — від інтернету застосунок має закривати
    реверс-проксі з нормальною автентифікацією (nginx basic auth, Cloudflare
    Access тощо) або вбудований OIDC-логін Streamlit (st.login, 1.42+);
  * пароль летить у відкритому вигляді, якщо застосунок віддається по HTTP —
    для зовнішнього доступу потрібен HTTPS.
"""

import hashlib
import hmac
import os
import time

import streamlit as st
from dotenv import load_dotenv

# .env лежить поруч із цим файлом; шукаємо саме там, а не від поточної теки
ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(ENV_PATH)

ALGO = "pbkdf2_sha256"
ITERATIONS = 260_000
SALT_BYTES = 16

MAX_ATTEMPTS = 5
LOCKOUT_SECONDS = 60


def hash_password(password, iterations=ITERATIONS):
    salt = os.urandom(SALT_BYTES)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"{ALGO}${iterations}${salt.hex()}${digest.hex()}"


def verify_password(password, stored):
    """Порівняння за сталий час. Приймає і хеш, і відкритий пароль."""
    if not stored:
        return False

    if stored.startswith(f"{ALGO}$"):
        try:
            _, iterations, salt_hex, expected_hex = stored.split("$", 3)
            digest = hashlib.pbkdf2_hmac(
                "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iterations)
            )
        except (ValueError, TypeError):
            return False
        return hmac.compare_digest(digest.hex(), expected_hex)

    return hmac.compare_digest(password, stored)


# Розкладки, які приймаємо в secrets.toml: (секція|None, ключ, це_хеш).
# Одна жорстко зашита розкладка — часта причина «не задано»: у файлі пароль є,
# але під іншим ключем, і застосунок про це мовчить.
SECRET_SHAPES = (
    ("auth", "password_hash",     True),
    ("auth", "APP_PASSWORD_HASH", True),
    (None,   "APP_PASSWORD_HASH", True),
    ("auth", "password",          False),
    ("auth", "APP_PASSWORD",      False),
    (None,   "APP_PASSWORD",      False),
)


def _read_secrets():
    """(об'єкт secrets|None, імена ключів верхнього рівня, текст помилки|None).

    Парсинг відбувається при першому доступі, тому помилковий TOML вилазить
    саме тут. Розрізняємо «файлу немає» і «файл є, але зламаний» — інакше обидва
    випадки виглядають як «пароль не налаштовано».
    """
    try:
        secrets = st.secrets
        return secrets, sorted(secrets.keys()), None
    except Exception as e:
        name = type(e).__name__
        if "SecretNotFound" in name or isinstance(e, FileNotFoundError):
            return None, [], None
        return None, [], f"{name}: {e}"


def secrets_value(*paths):
    """Перше непорожнє значення з st.secrets за списком (секція|None, ключ).

    Спільна точка доступу до секретів: зламаний або відсутній secrets.toml тут
    не кидає виняток, а дає порожній рядок — сторінки не мають падати через це.
    """
    secrets, _keys, _error = _read_secrets()
    if secrets is None:
        return ""

    for section, key in paths:
        try:
            container = secrets[section] if section else secrets
            value = container[key]
        except Exception:
            continue
        if value:
            return str(value).strip()
    return ""


def _stored_secret():
    """(значення, чи_це_хеш, діагностика) з env або st.secrets."""
    diag = {"secrets_keys": [], "secrets_error": None, "found_at": None}

    # Перечитуємо .env щоразу: Streamlit не переімпортовує вже завантажені
    # модулі між rerun-ами, тому імпортного load_dotenv замало — дописаний у
    # запущеному застосунку рядок інакше не підхопиться до рестарту
    load_dotenv(ENV_PATH, override=False)

    for var, is_hash in (("APP_PASSWORD_HASH", True), ("APP_PASSWORD", False)):
        value = (os.environ.get(var) or "").strip()
        if value:
            diag["found_at"] = f"env: {var}"
            return value, is_hash, diag

    secrets, keys, error = _read_secrets()
    diag["secrets_keys"] = keys
    diag["secrets_error"] = error
    if secrets is None:
        return None, False, diag

    for section, key, is_hash in SECRET_SHAPES:
        try:
            container = secrets[section] if section else secrets
            value = container[key]
        except Exception:
            continue
        value = str(value).strip()
        if value:
            diag["found_at"] = f"secrets: {section + '.' if section else ''}{key}"
            return value, is_hash, diag

    return None, False, diag


def _lockout_left():
    until = st.session_state.get("auth_locked_until", 0)
    return max(0, int(until - time.time()))


def _login_form(stored):
    st.title("🔒 Вхід")

    left, _ = st.columns([1, 1])
    with left:
        locked = _lockout_left()
        if locked:
            st.error(f"Забагато невдалих спроб. Спробуй через {locked} с.")
            st.stop()

        with st.form("login"):
            password = st.text_input("Пароль", type="password")
            submitted = st.form_submit_button("Увійти", type="primary")

        if submitted:
            if verify_password(password, stored):
                st.session_state["auth_ok"] = True
                st.session_state["auth_attempts"] = 0
                st.rerun()

            attempts = st.session_state.get("auth_attempts", 0) + 1
            st.session_state["auth_attempts"] = attempts
            if attempts >= MAX_ATTEMPTS:
                st.session_state["auth_locked_until"] = time.time() + LOCKOUT_SECONDS
                st.error(f"Забагато невдалих спроб. Спробуй через {LOCKOUT_SECONDS} с.")
            else:
                st.error(f"Невірний пароль ({attempts} з {MAX_ATTEMPTS}).")


def _setup_hint(diag):
    st.title("🔒 Пароль не налаштовано")
    st.error("Вхід закритий: пароль не знайдено ні в env, ні в `st.secrets`.")

    if diag["secrets_error"]:
        st.warning(
            "`st.secrets` не читається — найчастіше це синтаксис TOML "
            "(значення без лапок, зайва кома, збита секція):\n\n"
            f"```\n{diag['secrets_error']}\n```"
        )
    elif diag["secrets_keys"]:
        st.info(
            "`st.secrets` прочитано, ключі верхнього рівня: "
            + ", ".join(f"`{k}`" for k in diag["secrets_keys"])
            + " — але жоден із очікуваних не підійшов."
        )
    else:
        st.info("`st.secrets` порожній або файлу секретів немає.")

    st.markdown(
        "Приймається будь-яка з цих розкладок у `secrets.toml` "
        "(або в Streamlit Cloud → Settings → Secrets):\n\n"
        "```toml\n"
        "[auth]\n"
        'password_hash = "pbkdf2_sha256$260000$<salt>$<hash>"\n'
        "```\n"
        "або плоским ключем:\n"
        "```toml\n"
        'APP_PASSWORD_HASH = "pbkdf2_sha256$260000$<salt>$<hash>"\n'
        "```\n\n"
        "Значення обов'язково **в подвійних лапках** — без них TOML не "
        "розбереться, і секрети не завантажаться цілком.\n\n"
        "Локально те саме працює через `.env` поруч з `app.py`:\n"
        "```\n"
        "APP_PASSWORD_HASH=pbkdf2_sha256$260000$<salt>$<hash>\n"
        "```\n"
        "(у `.env` лапки не потрібні)\n\n"
        "Згенерувати новий хеш: `python auth.py`."
    )


def render_logout():
    """Кнопка виходу в сайдбарі. Це st-команда, тож лише після set_page_config."""
    with st.sidebar:
        if st.button("Вийти", use_container_width=True):
            for key in ("auth_ok", "auth_attempts", "auth_locked_until"):
                st.session_state.pop(key, None)
            st.rerun()


def require_auth(logout_button=True):
    """Пускає далі або малює логін і зупиняє скрипт сторінки.

    logout_button=False потрібне там, де сторінка ще не викликала
    set_page_config (напр. обгортка над чужим app.py): будь-яка st-команда до
    нього ламає Streamlit. Кнопку виходу тоді малюють окремо, пізніше.
    """
    if st.session_state.get("auth_ok"):
        if logout_button:
            render_logout()
        return

    stored, is_hash, diag = _stored_secret()
    if not stored:
        _setup_hint(diag)
        st.stop()

    if is_hash and not stored.startswith(f"{ALGO}$"):
        st.warning(
            "Значення схоже не на хеш `pbkdf2_sha256$...`. Якщо це відкритий "
            "пароль — він однаково спрацює, але краще згенеруй хеш: `python auth.py`."
        )
    elif is_hash and len(stored.split("$")) != 4:
        st.error(
            "Хеш обрізаний або склеєний: очікується "
            "`pbkdf2_sha256$<iterations>$<salt>$<hash>` — чотири частини через `$`. "
            "Перевір, чи значення скопійоване цілком."
        )

    _login_form(stored)
    if not is_hash:
        st.caption(
            "Пароль зберігається відкритим текстом у `APP_PASSWORD`. "
            "Надійніше — `APP_PASSWORD_HASH` (`python auth.py`)."
        )
    st.stop()


if __name__ == "__main__":
    import getpass

    pwd = getpass.getpass("Новий пароль: ")
    if not pwd:
        raise SystemExit("Порожній пароль.")
    if pwd != getpass.getpass("Ще раз: "):
        raise SystemExit("Паролі не збігаються.")
    print("\nДодай цей рядок у .env:\n")
    print(f"APP_PASSWORD_HASH={hash_password(pwd)}")
