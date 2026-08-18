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
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

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


def _stored_secret():
    """(значення, чи_це_хеш) з env або st.secrets."""
    env_hash = (os.environ.get("APP_PASSWORD_HASH") or "").strip()
    if env_hash:
        return env_hash, True

    try:
        secret = st.secrets["auth"]["password_hash"]
    except Exception:
        secret = None
    if secret:
        return str(secret).strip(), True

    env_plain = (os.environ.get("APP_PASSWORD") or "").strip()
    if env_plain:
        return env_plain, False

    return None, False


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


def _setup_hint():
    st.title("🔒 Пароль не налаштовано")
    st.error("Вхід закритий: не задано ні `APP_PASSWORD_HASH`, ні `APP_PASSWORD`.")
    st.markdown(
        "Згенеруй хеш і поклади його в `.env` поруч з `app.py`:\n\n"
        "```bash\n"
        "python auth.py\n"
        "```\n\n"
        "Скрипт запитає пароль і надрукує готовий рядок:\n\n"
        "```\n"
        "APP_PASSWORD_HASH=pbkdf2_sha256$260000$<salt>$<hash>\n"
        "```\n\n"
        "Після цього перезапусти застосунок."
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

    stored, is_hash = _stored_secret()
    if not stored:
        _setup_hint()
        st.stop()

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
