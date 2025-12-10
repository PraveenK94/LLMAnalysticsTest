
# app.py
import json
import time
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    StaleElementReferenceException,
    WebDriverException,
)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Optional auto-refresh helper
try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False

# ==========================
# CONFIG
# ==========================

COOKIES_PATH = Path("cookies.json")
AVIATOR_URL = "https://topx.one/games/provider/spribe-aviatorz"

CSV_FILE = "aviator_multipliers_streamlit.csv"
JSON_FILE = "aviator_multipliers_streamlit.json"

SESSIONS_DIR = Path("sessions")
SESSIONS_DIR.mkdir(exist_ok=True)
SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

OLLAMA_MODEL = "gemma3:1b"
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

# ==========================
# SELENIUM HELPERS
# ==========================


def create_driver() -> webdriver.Chrome:
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    # options.add_argument("--headless=new")
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options,
    )
    return driver


def load_cookies_from_disk(path: Path) -> list:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Please create cookies.json.")
    with path.open("r", encoding="utf-8") as f:
        cookies = json.load(f)
    if not isinstance(cookies, list):
        raise ValueError("cookies.json must contain a JSON list of cookie dicts.")
    return cookies


def inject_cookies(driver: webdriver.Chrome, cookies: list):
    for c in cookies:
        if "topx.one" not in c.get("domain", ""):
            continue

        cookie_dict = {
            "name": c["name"],
            "value": c["value"],
            "domain": c.get("domain", "topx.one"),
            "path": c.get("path", "/"),
            "secure": c.get("secure", False),
            "httpOnly": c.get("httpOnly", False),
        }

        if "expirationDate" in c:
            try:
                cookie_dict["expiry"] = int(c["expirationDate"])
            except Exception:
                pass

        try:
            driver.add_cookie(cookie_dict)
        except Exception as e:
            print(f"Skipping cookie {cookie_dict['name']}: {e}")


def get_payout_values(driver: webdriver.Chrome):
    payout_elems = WebDriverWait(driver, 20).until(
        EC.presence_of_all_elements_located(
            (
                By.XPATH,
                "//div[contains(@class, 'payouts-block')]"
                "//div[contains(@class, 'payout')]",
            )
        )
    )

    values = []
    for el in payout_elems:
        txt = el.text.strip()
        if not txt:
            continue
        txt = txt.lower().replace("x", "").strip()
        try:
            values.append(float(txt))
        except ValueError:
            continue

    return values


# --- new: bets extractor ---
def get_bets_count(driver: webdriver.Chrome):
    """
    Return tuple (bets_current:int|None, bets_total:int|None, raw_text:str|None)
    Tries several selectors; returns (None, None, None) if not found.
    """
    try:
        selectors = [
            "app-total-win-widget .bets-count",
            "app-total-win-widget .bets .bets-count",
            ".total-win-container .bets-count",
            ".bets-count"
        ]
        text = None
        for sel in selectors:
            try:
                elem = WebDriverWait(driver, 2).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, sel))
                )
                text = elem.text.strip()
                if text:
                    break
            except Exception:
                continue

        if not text:
            return None, None, None

        raw = text.replace(" ", "")
        if "/" in raw:
            parts = raw.split("/", 1)
            try:
                cur = int(parts[0])
            except Exception:
                cur = None
            try:
                tot = int(parts[1])
            except Exception:
                tot = None
        else:
            try:
                cur = int(raw)
            except Exception:
                cur = None
            tot = None

        return cur, tot, text

    except Exception as e:
        print(f"[SCRAPER] get_bets_count error: {e}")
        return None, None, None


# ==========================
# SHARED STORE
# ==========================


@st.cache_resource
def get_store():
    return {
        "records": [],
        "logs": [],
        "running": False,
        "thread": None,
    }


def save_records(records):
    if not records:
        return
    df_out = pd.DataFrame(records)

    # overwrite latest snapshot
    df_out.to_csv(CSV_FILE, index=False)
    with open(JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    # per-session archive
    session_csv = SESSIONS_DIR / f"aviator_session_{SESSION_ID}.csv"
    session_json = SESSIONS_DIR / f"aviator_session_{SESSION_ID}.json"
    df_out.to_csv(session_csv, index=False)
    with open(session_json, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def scraper_loop(store, interval_sec: float, cookies: list):
    """
    Background thread that scrapes payout history and bets count,
    appends records to shared store.
    """
    store["running"] = True
    driver = None

    try:
        driver = create_driver()
        driver.get("https://topx.one")
        time.sleep(2)
        inject_cookies(driver, cookies)

        driver.get(AVIATOR_URL)
        iframe = WebDriverWait(driver, 15).until(
            EC.presence_of_element_located(
                (By.XPATH, "//iframe[@class='game-view_iframeBlock__5_zt2']")
            )
        )
        driver.switch_to.frame(iframe)

        start = time.time()
        last_history = None

        while store["running"]:
            # --- read history safely ---
            try:
                history_vals = get_payout_values(driver)

            except StaleElementReferenceException as e:
                print("[SCRAPER] Stale element while reading history, retrying…", e)
                time.sleep(0.5)
                continue

            except WebDriverException as e:
                msg = str(e)
                print(f"[SCRAPER] WebDriver error while getting history: {msg}")

                lower_msg = msg.lower()
                fatal = (
                    "invalid session id" in lower_msg
                    or "disconnected" in lower_msg
                    or "chrome not reachable" in lower_msg
                    or "session deleted" in lower_msg
                )

                if fatal:
                    print("[SCRAPER] Fatal webdriver error, stopping scraper loop.")
                    break

                time.sleep(float(interval_sec))
                continue

            except Exception as e:
                print(f"[SCRAPER] Unexpected error getting history: {repr(e)}")
                time.sleep(float(interval_sec))
                continue

            # --- normal processing when we *did* get history ---
            if history_vals:
                if history_vals != last_history:
                    new_val = history_vals[0]  # left-most is latest
                    now = datetime.now()
                    t = time.time() - start

                    # safe bets extraction
                    bets_current, bets_total, bets_raw = None, None, None
                    try:
                        bets_current, bets_total, bets_raw = get_bets_count(driver)
                    except Exception as e:
                        print(f"[SCRAPER] Error reading bets count: {e}")

                    rec = {
                        "t_seconds_from_start": t,
                        "iso_time": now.isoformat(timespec="seconds"),
                        "new_round_value": new_val,
                        "bets_current": bets_current,
                        "bets_total": bets_total,
                        "bets_raw_text": bets_raw,
                        "history_snapshot_left_to_right": history_vals,
                        "history_snapshot_right_to_left": list(reversed(history_vals)),
                    }
                    store["records"].append(rec)

                    preview = " ".join(f"{v}x" for v in history_vals[:12])
                    log_line = (
                        f"{now.strftime('%H:%M:%S')} | "
                        f"value={new_val}x | bets={bets_raw or 'N/A'} | history_LR={preview}"
                    )
                    store["logs"].append(log_line)

                    last_history = history_vals

            time.sleep(float(interval_sec))

    finally:
        save_records(store["records"])
        store["running"] = False
        try:
            if driver is not None:
                driver.quit()
        except Exception:
            pass
        print("[SCRAPER] Exited, driver closed.")


# ==========================
# STATS & LLM
# ==========================


def describe_distribution(df: pd.DataFrame, small_max: float, large_min: float) -> str:
    if df.empty:
        return "No data yet."

    vals = df["new_round_value"].values
    n = len(vals)

    small_mask = vals <= small_max
    large_mask = vals >= large_min
    mid_mask = (~small_mask) & (~large_mask)

    n_small = int(small_mask.sum())
    n_mid = int(mid_mask.sum())
    n_large = int(large_mask.sum())

    pct_small = 100 * n_small / n if n > 0 else 0
    pct_mid = 100 * n_mid / n if n > 0 else 0
    pct_large = 100 * n_large / n if n > 0 else 0

    mean_val = float(df["new_round_value"].mean())
    std_val = float(df["new_round_value"].std()) if n > 1 else 0.0
    min_val = float(df["new_round_value"].min())
    max_val = float(df["new_round_value"].max())

    q10, q25, q50, q75, q90 = df["new_round_value"].quantile(
        [0.1, 0.25, 0.5, 0.75, 0.9]
    ).round(2).tolist()

    p_small = n_small / n if n > 0 else 0.0
    p_mid = n_mid / n if n > 0 else 0.0
    p_large = n_large / n if n > 0 else 0.0

    text = f"""
Live stats
Rounds captured: {n}
Latest value: {df['new_round_value'].iloc[-1]:.2f}x
Mean multiplier: {mean_val:.2f}x
Std dev (volatility): {std_val:.2f}x
Min: {min_val:.2f}x
Max: {max_val:.2f}x

Quantiles (approx):
10%: {q10}x
25% (Q1): {q25}x
50% (median): {q50}x
75% (Q3): {q75}x
90%: {q90}x

Bins:
Small (≤ {small_max}x): {n_small} ({pct_small:.1f}%)
Mid ({small_max}x – {large_min}x): {n_mid} ({pct_mid:.1f}%)
Massive (≥ {large_min}x): {n_large} ({pct_large:.1f}%)

Empirical (frequency-based) 'next-round' chances:
Next round small: ~{p_small:.2f}
Next round mid: ~{p_mid:.2f}
Next round massive: ~{p_large:.2f}
"""
    return text.strip()


def call_gemma_via_http(prompt: str, num_predict: int = 256) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": num_predict},
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", "").strip()


def ask_local_llm(df: pd.DataFrame, user_question: str, small_max: float, large_min: float) -> str:
    if df.empty:
        return "No data yet to analyze. Let the live tracker run for a while."

    summary = describe_distribution(df, small_max, large_min)
    uq = user_question.strip()
    uq_lower = uq.lower()
    detailed = "deep search" in uq_lower
    uq_clean = uq.replace("deep search", "").replace("Deep search", "")

    if detailed:
        style_instruction = (
            "Give a detailed explanation (3–5 short paragraphs). "
            "Go deeper into heavy tails, randomness and volatility."
        )
        num_predict = 512
    else:
        style_instruction = (
            "Answer briefly (max 3 sentences). "
            "Give intuition only, not long essays."
        )
        num_predict = 256

    system_prompt = """
You are a friendly data science tutor.
You explain randomness, volatility, heavy tails, and distribution patterns.
You MUST NOT give any advice about gambling, betting, strategies, or how to profit.
Only educational explanation and intuition.
"""

    user_prompt = f"""
{system_prompt}

Here is a summary of a random multiplier sequence:

{summary}

The user asks:
{uq_clean}

{style_instruction}
"""

    try:
        return call_gemma_via_http(user_prompt, num_predict=num_predict)
    except Exception as e:
        return f"Error calling local Gemma (model={OLLAMA_MODEL}): {e}"


# ==========================
# PLOT HELPERS
# ==========================


def make_live_figure(df: pd.DataFrame, window: int, theme: str) -> go.Figure:
    df = df.copy()
    df["round_index"] = np.arange(1, len(df) + 1)
    template = "plotly_dark" if theme == "Dark" else "plotly"

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["iso_time"],
            y=df["new_round_value"],
            mode="lines+markers",
            name="Multiplier",
            customdata=df[["round_index"]],
            hovertemplate=(
                "Round #%{customdata[0]}<br>"
                "Time=%{x|%H:%M:%S}<br>"
                "Value=%{y:.2f}x"
                "<extra></extra>"
            ),
        )
    )

    if len(df) >= window:
        roll = df["new_round_value"].rolling(window).mean()
        fig.add_trace(
            go.Scatter(
                x=df["iso_time"],
                y=roll,
                mode="lines",
                name=f"Rolling mean ({window})",
                line=dict(dash="dash"),
            )
        )

    last_row = df.iloc[-1]
    fig.add_trace(
        go.Scatter(
            x=[last_row["iso_time"]],
            y=[last_row["new_round_value"]],
            mode="markers+text",
            name="Latest",
            marker=dict(color="green", size=10),
            text=[f"#{int(last_row['round_index'])} • {last_row['new_round_value']:.2f}x"],
            textposition="top center",
        )
    )

    fig.update_layout(
        title="Live Aviator Multiplier Stream",
        xaxis_title="Time",
        yaxis_title="Multiplier (x)",
        xaxis=dict(showgrid=True, tickformat="%H:%M:%S", rangeslider=dict(visible=True)),
        yaxis=dict(showgrid=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=20, t=40, b=40),
        height=360,
        template=template,
        uirevision="live",
    )

    return fig


def make_bets_over_time_figure(df: pd.DataFrame, theme: str):
    df = df.copy()
    if "bets_current" not in df.columns:
        return None
    df["iso_time"] = pd.to_datetime(df["iso_time"])
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["iso_time"],
            y=df["bets_current"].fillna(0),
            name="Bets current",
            opacity=0.6,
        )
    )
    # overlay rolling mean of multipliers on secondary y-axis
    df["mult_roll"] = df["new_round_value"].rolling(10).mean()
    fig.add_trace(
        go.Scatter(
            x=df["iso_time"],
            y=df["mult_roll"],
            mode="lines",
            name="Multiplier rolling mean",
            yaxis="y2",
        )
    )
    template = "plotly_dark" if theme == "Dark" else "plotly"
    fig.update_layout(
        title="Bets current per timestamp (bar) + rolling mean multiplier (line)",
        xaxis_title="Time",
        yaxis=dict(title="Bets count"),
        yaxis2=dict(title="Multiplier (x)", overlaying="y", side="right"),
        template=template,
        height=300,
        margin=dict(l=40, r=40, t=40, b=40),
        uirevision="bets-plot",
    )
    return fig


# ==========================
# ML helpers: Random Forest
# ==========================


def make_supervised_from_df(df: pd.DataFrame, window: int = 10):
    vals = df["new_round_value"].values.astype(float)
    if len(vals) <= window:
        return np.empty((0, window)), np.empty((0,))
    X, y = [], []
    for i in range(len(vals) - window):
        X.append(vals[i : i + window])
        y.append(vals[i + window])
    return np.array(X), np.array(y)


def train_random_forest(df: pd.DataFrame, window: int = 10, test_ratio: float = 0.2):
    X, y = make_supervised_from_df(df, window)
    if len(X) < 20:
        return None

    split_idx = int(len(X) * (1 - test_ratio))
    if split_idx <= window:
        split_idx = max(window + 1, len(X) - 5)

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    if len(X_test) == 0:
        return None

    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)

    mae_per_round = []
    for i in range(1, len(y_test) + 1):
        mae_i = mean_absolute_error(y_test[:i], y_pred_test[:i])
        mae_per_round.append(mae_i)

    vals = df["new_round_value"].values.astype(float)
    last_window = vals[-window:].copy()
    future_preds = []
    for _ in range(5):
        x_input = last_window.reshape(1, -1)
        next_val = model.predict(x_input)[0]
        future_preds.append(next_val)
        last_window = np.concatenate([last_window[1:], [next_val]])

    return {
        "y_test": y_test,
        "y_pred_test": y_pred_test,
        "mae_per_round": mae_per_round,
        "future_preds": future_preds,
        "window": window,
        "test_ratio": test_ratio,
    }


def make_history_surface(df: pd.DataFrame, window: int = 10, max_windows: int = 80):
    vals = df["new_round_value"].values.astype(float)
    if len(vals) <= window:
        return None, None, None

    X_list = []
    for i in range(len(vals) - window):
        X_list.append(vals[i : i + window])
    X = np.array(X_list)

    n_windows = min(len(X), max_windows)
    X = X[-n_windows:]  # latest windows

    window_idx = np.arange(X.shape[0])
    lag_idx = np.arange(window)
    W, L = np.meshgrid(window_idx, lag_idx, indexing="ij")
    Z = X
    return W, L, Z


# ==========================
# LOAD SNAPSHOT
# ==========================


def load_records_from_file():
    p = Path(JSON_FILE)
    if not p.exists():
        return pd.DataFrame(columns=["iso_time", "new_round_value"])
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        if df.empty:
            return df
        df["iso_time"] = pd.to_datetime(df["iso_time"])
        return df
    except Exception:
        return pd.DataFrame(columns=["iso_time", "new_round_value"])


# ==========================
# STREAMLIT UI
# ==========================

st.set_page_config(page_title="Aviator Live Tracker + Tutor", layout="wide")
store = get_store()

if "rf_results" not in st.session_state:
    st.session_state["rf_results"] = None
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    auto_refresh_ms = st.number_input(
        "UI auto-refresh (ms)",
        min_value=500,
        max_value=5000,
        value=1500,
        step=500,
        key="auto_refresh_ms",
    )
    if HAS_AUTOREFRESH:
        st_autorefresh(interval=int(auto_refresh_ms), key="data_refresh")
        st.caption("Auto-refresh is ON.")
    else:
        st.caption("Install `streamlit-autorefresh` for auto refresh.")

    theme = st.selectbox("Dashboard theme", options=["Light", "Dark"], index=1, key="theme_select")

# global dashboard CSS (not just charts)
if theme == "Dark":
    st.markdown(
        """
        <style>
        .stApp { background-color: #0e1117; color: #e5e7eb; }
        [data-testid="stSidebar"] { background-color: #020617; }
        .block-container { padding-top: 1.5rem; padding-bottom: 3rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <style>
        .stApp { background-color: #f3f4f6; color: #111827; }
        [data-testid="stSidebar"] { background-color: #e5e7eb; }
        .block-container { padding-top: 1.5rem; padding-bottom: 3rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

st.title("🎰 Aviator Live Tracker + Gemma Tutor")

# Controls
st.subheader("🛰 Live tracking controls")

c1, c2, c3, c4 = st.columns(4)
with c1:
    interval_sec = st.number_input(
        "Sampling interval (sec)", min_value=0.5, max_value=10.0, value=1.0, step=0.5, key="interval_live"
    )
with c2:
    rolling_window = st.number_input(
        "Rolling mean window", min_value=2, max_value=100, value=10, step=1, key="rolling_live"
    )
with c3:
    small_max = st.number_input("Small ≤ x", min_value=1.0, max_value=10.0, value=2.0, step=0.1, key="small_max_live")
with c4:
    large_min = st.number_input("Massive ≥ x", min_value=3.0, max_value=100.0, value=10.0, step=0.5, key="large_min_live")

b1, b2, b3 = st.columns(3)
with b1:
    start_btn = st.button("🚀 Start scraping", type="primary", key="start_scrape")
with b2:
    stop_btn = st.button("⏹ Stop scraping", key="stop_scrape")
with b3:
    st.button("🔁 Manual refresh", key="refresh_view")

cookies_for_scraper = None
cookies_error = None
if COOKIES_PATH.exists():
    try:
        cookies_for_scraper = load_cookies_from_disk(COOKIES_PATH)
    except Exception as e:
        cookies_error = str(e)
else:
    cookies_error = f"{COOKIES_PATH} not found. Create cookies.json first."

if start_btn:
    if cookies_error:
        st.error(f"Cannot start scraper: {cookies_error}")
    else:
        if not store["running"]:
            t = threading.Thread(target=scraper_loop, args=(store, float(interval_sec), cookies_for_scraper), daemon=True)
            store["thread"] = t
            t.start()
            st.success("Scraper started in background (Chrome will open).")
        else:
            st.info("Scraper already running.")

if stop_btn:
    if store["running"]:
        store["running"] = False
        st.warning("Stopping scraper (wait for current cycle).")
    else:
        st.info("Scraper not running.")

st.markdown("> Educational **randomness lab** only. No betting, no strategy, no financial advice.")
st.markdown("---")

left, right = st.columns([2, 1])

# ==========================
# LEFT: live + tabs
# ==========================
with left:
    st.subheader("📉 Live stream")

    records = store["records"]
    logs = store["logs"]

    if not records:
        if cookies_error:
            st.error(f"Cookies error: {cookies_error}")
        st.info("No records yet. Start the scraper.")
    else:
        df_live = pd.DataFrame(records)
        df_live["iso_time"] = pd.to_datetime(df_live["iso_time"])

        fig_live = make_live_figure(df_live[["iso_time", "new_round_value"]], window=int(rolling_window), theme=theme)
        st.plotly_chart(fig_live, use_container_width=True)

        tab_stats, tab_rf, tab_hist = st.tabs(["📊 Stats & patterns", "🤖 RF analysis", "📜 History"])

        vals = df_live["new_round_value"]
        n = len(vals)
        latest_val = float(vals.iloc[-1])
        mean_val = float(vals.mean())
        std_val = float(vals.std()) if n > 1 else 0.0
        min_val = float(vals.min())
        max_val = float(vals.max())

        q10, q25, q50, q75, q90 = vals.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(2).tolist()

        small_mask = vals <= small_max
        large_mask = vals >= large_min
        mid_mask = (~small_mask) & (~large_mask)

        n_small = int(small_mask.sum())
        n_mid = int(mid_mask.sum())
        n_large = int(large_mask.sum())

        pct_small = 100 * n_small / n if n > 0 else 0
        pct_mid = 100 * n_mid / n if n > 0 else 0
        pct_large = 100 * n_large / n if n > 0 else 0

        # bets metric (latest)
        bets_cur = None
        bets_tot = None
        if "bets_current" in df_live.columns and not df_live["bets_current"].isna().all():
            bets_cur = df_live["bets_current"].iloc[-1]
        if "bets_total" in df_live.columns and not df_live["bets_total"].isna().all():
            bets_tot = df_live["bets_total"].iloc[-1]
        bets_label = (f"{bets_cur}" + (f"/{bets_tot}" if bets_tot is not None else "")) if bets_cur is not None else "N/A"

        template = "plotly_dark" if theme == "Dark" else "plotly"

        # ----- STATS TAB -----
        with tab_stats:
            st.markdown("### Live stats overview")
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Rounds", n)
            with m2:
                st.metric("Latest", f"{latest_val:.2f}x")
            with m3:
                st.metric("Mean", f"{mean_val:.2f}x")
            with m4:
                st.metric("Std dev", f"{std_val:.2f}x")

            # show bets as separate metric row
            b1, b2 = st.columns([3, 1])
            with b1:
                st.write("")  # spacer
            with b2:
                st.metric("Bets (cur/total)", bets_label)

            # Range / quantiles / bins
            row1_c1, row1_c2 = st.columns(2)
            with row1_c1:
                st.markdown("#### Range & quantiles")
                st.write(f"Min: **{min_val:.2f}x**  •  Max: **{max_val:.2f}x**")
                q_df = pd.DataFrame({
                    "Quantile": ["10%", "25% (Q1)", "50% (Median)", "75% (Q3)", "90%"],
                    "Value (x)": [q10, q25, q50, q75, q90],
                })
                st.dataframe(q_df, use_container_width=True, height=180)
            with row1_c2:
                st.markdown("#### Bins by threshold")
                bins_df = pd.DataFrame({
                    "Bin": [f"Small (≤ {small_max}x)", f"Mid ({small_max}x – {large_min}x)", f"Massive (≥ {large_min}x)"],
                    "Count": [n_small, n_mid, n_large],
                    "Percent": [f"{pct_small:.1f}%", f"{pct_mid:.1f}%", f"{pct_large:.1f}%"],
                })
                st.dataframe(bins_df, use_container_width=True, height=180)

            row2_c1, row2_c2 = st.columns(2)
            with row2_c1:
                st.markdown("#### Histogram (value distribution)")
                fig_hist = go.Figure(data=[go.Histogram(x=vals, nbinsx=30, hovertemplate="x=%{x:.2f}x<br>count=%{y}<extra></extra>")])
                fig_hist.update_layout(xaxis_title="Multiplier (x)", yaxis_title="Count", template=template, height=260)
                st.plotly_chart(fig_hist, use_container_width=True)
            with row2_c2:
                st.markdown("#### Boxplot (spread & outliers)")
                fig_box = go.Figure(data=[go.Box(y=vals, boxpoints="outliers", name="Multipliers", hovertemplate="value=%{y:.2f}x<extra></extra>")])
                fig_box.update_layout(yaxis_title="Multiplier (x)", template=template, height=260)
                st.plotly_chart(fig_box, use_container_width=True)

            # bets over time
            with st.expander("Bets over time chart (bar)"):
                fig_bets = make_bets_over_time_figure(df_live, theme=theme)
                if fig_bets is not None:
                    st.plotly_chart(fig_bets, use_container_width=True)
                else:
                    st.info("No bets data yet (will appear after scraper captures bets).")

            st.markdown("#### Text summary")
            stats_text = describe_distribution(df_live, small_max, large_min)
            st.text_area("Summary (copy / notes)", value=stats_text, height=220)

        # ----- RF TAB -----
        with tab_rf:
            st.markdown("### Random Forest: pattern demo & rendering")
            ml1, ml2, ml3 = st.columns(3)
            with ml1:
                ml_window = st.number_input("History window (rounds)", min_value=5, max_value=60, value=10, step=1, key="ml_window")
            with ml2:
                ml_test_ratio = st.slider("Test ratio", min_value=0.05, max_value=0.5, value=0.2, step=0.05, key="ml_test_ratio")
            with ml3:
                train_btn = st.button("Train RF model", key="train_rf_btn")

            if train_btn:
                if len(df_live) <= ml_window + 10:
                    st.warning(f"Need more data to train (≥ {ml_window + 10} rounds).")
                else:
                    with st.spinner("Training Random Forest on history..."):
                        rf_info = train_random_forest(df_live[["iso_time", "new_round_value"]], window=int(ml_window), test_ratio=float(ml_test_ratio))
                    if rf_info is None:
                        st.warning("Not enough data for stable train/test split.")
                    else:
                        st.session_state["rf_results"] = rf_info

            rf_results = st.session_state.get("rf_results")
            if rf_results is None:
                st.info("Train the RF model to see predictions and error analysis.")
            else:
                y_test = rf_results["y_test"]
                y_pred_test = rf_results["y_pred_test"]
                mae_per_round = rf_results["mae_per_round"]
                future_preds = rf_results["future_preds"]

                st.markdown("#### Next 5 predicted multipliers (model forecast)")
                metrics_cols = st.columns(5)
                for i, (col, val) in enumerate(zip(metrics_cols, future_preds), start=1):
                    with col:
                        st.metric(f"t+{i}", f"{val:.2f}x")

                rf_row1_c1, rf_row1_c2 = st.columns(2)
                with rf_row1_c1:
                    st.markdown("#### Tail history + forecast")
                    tail_n = min(40, len(vals))
                    actual_vals = vals.iloc[-tail_n:].values
                    idx_actual = np.arange(1, tail_n + 1)
                    idx_pred = np.arange(tail_n + 1, tail_n + 1 + 5)
                    fig_tail = go.Figure()
                    fig_tail.add_trace(go.Scatter(x=idx_actual, y=actual_vals, mode="lines+markers", name="Actual (tail)"))
                    fig_tail.add_trace(go.Scatter(x=idx_pred, y=future_preds, mode="lines+markers", name="Predicted next 5", line=dict(dash="dash")))
                    fig_tail.update_layout(xaxis_title="Tail index", yaxis_title="Multiplier (x)", template=template, height=260, uirevision="rf-forecast")
                    st.plotly_chart(fig_tail, use_container_width=True)
                with rf_row1_c2:
                    st.markdown("#### MAE progression")
                    idx_mae = np.arange(1, len(mae_per_round) + 1)
                    fig_mae = go.Figure(data=[go.Scatter(x=idx_mae, y=mae_per_round, mode="lines+markers", hovertemplate="idx=%{x}<br>MAE=%{y:.4f}<extra></extra>")])
                    fig_mae.update_layout(xaxis_title="Test index", yaxis_title="MAE", template=template, height=260)
                    st.plotly_chart(fig_mae, use_container_width=True)

                # 3D history surface (actual values)
                st.markdown("#### 3D history surface (how model sees sliding windows of history)")
                W, L, Z = make_history_surface(df_live, window=int(ml_window), max_windows=80)
                if W is not None:
                    fig_hist3d = go.Figure(data=[go.Surface(x=L, y=W, z=Z, colorbar=dict(title="x (multiplier)"))])
                    fig_hist3d.update_layout(scene=dict(xaxis_title="Lag in window", yaxis_title="Window index (time)", zaxis_title="Multiplier (x)"), template=template, height=320, uirevision="rf-hist-3d")
                    st.plotly_chart(fig_hist3d, use_container_width=True)
                else:
                    st.info("Need more data for 3D history surface (window too large or insufficient history).")

                # 3D test vs predicted
                st.markdown("#### 3D scatter: test index vs actual vs predicted")
                idx_test = np.arange(1, len(y_test) + 1)
                fig_rf_3d = go.Figure(data=[go.Scatter3d(x=idx_test, y=y_test, z=y_pred_test, mode="markers", name="Test samples", hovertemplate="idx=%{x}<br>actual=%{y:.2f}x<br>pred=%{z:.2f}x<extra></extra>")])
                fig_rf_3d.update_layout(scene=dict(xaxis_title="Test index", yaxis_title="Actual (x)", zaxis_title="Predicted (x)"), template=template, height=320, uirevision="rf-3d")
                st.plotly_chart(fig_rf_3d, use_container_width=True)

                st.markdown(
                    """
**Theory & facts (RF demo)**  
- Random Forest models treat each sliding window as a single sample in a high-dimensional space.  
- The 3D history surface visualizes those sliding windows; the model uses that geometry to split/regress.  
- MAE shows how errors evolve across the test set.  
**Important:** this is an educational exercise. The underlying process is heavy-tailed and largely random — the model can learn patterns but not guarantee future outcomes.
"""
                )

        # ----- HISTORY TAB -----
        with tab_hist:
            st.markdown("### Recent rounds & raw logs (scrollable grid)")

            h1, h2 = st.columns(2)
            with h1:
                st.markdown("#### Recent rounds (latest 300)")
                # include bets columns; will scroll horizontally if needed
                tail_df = df_live[["iso_time", "new_round_value", "bets_current", "bets_total", "bets_raw_text"]].tail(300)
                st.dataframe(tail_df, use_container_width=True, height=360)
            with h2:
                st.markdown("#### Live log (last 400 lines)")
                st.text_area("Log", value="\n".join(logs[-400:]), height=360)

            with st.expander("Download data (CSV snapshot)"):
                st.download_button("Download CSV", data=pd.DataFrame(records).to_csv(index=False), file_name=f"aviator_snapshot_{SESSION_ID}.csv", mime="text/csv")

# ==========================
# RIGHT: Gemma tutor
# ==========================
with right:
    st.subheader("😄 Gemma3:1B Data Tutor")

    if store["records"]:
        df_for_chat = pd.DataFrame(store["records"])
        df_for_chat["iso_time"] = pd.to_datetime(df_for_chat["iso_time"])
    else:
        df_for_chat = load_records_from_file()

    t1, t2 = st.columns(2)
    with t1:
        small_max_chat = st.number_input("Small threshold (≤ x)", min_value=1.0, max_value=10.0, value=2.0, step=0.1, key="small_max_chat_key")
    with t2:
        large_min_chat = st.number_input("Massive threshold (≥ x)", min_value=3.0, max_value=100.0, value=10.0, step=0.5, key="large_min_chat_key")

    chat_history = st.session_state["chat_history"]

    def latest_answer_text():
        for msg in reversed(chat_history):
            if msg["role"] == "tutor":
                return msg["content"]
        return "Ask a question to see Gemma's answer here."

    st.markdown("### 🧠 Latest tutor response")
    st.text_area("Latest response", value=latest_answer_text(), height=160)

    st.markdown("### Ask a question")
    st.caption("Use **Cmd+Enter / Ctrl+Enter** to send. Add `deep search` for a longer, more theoretical answer.")
    with st.form("tutor_form", clear_on_submit=True):
        question = st.text_area("Your question:", height=80)
        submitted = st.form_submit_button("Send")

    if submitted and question.strip():
        if df_for_chat.empty:
            answer = "No live data yet. Let the tracker run a bit, then ask again."
        else:
            answer = ask_local_llm(df_for_chat, question, small_max_chat, large_min_chat)
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "tutor", "content": answer})
        st.session_state["chat_history"] = chat_history

    st.markdown("### 💬 Chat history (scrollable)")
    lines = []
    for msg in chat_history[-120:]:
        who = "You" if msg["role"] == "user" else "Tutor"
        lines.append(f"{who}: {msg['content']}")
    st.text_area("History", value="\n\n".join(lines) if lines else "No questions yet.", height=300)

    if st.button("Clear chat history"):
        st.session_state["chat_history"] = []