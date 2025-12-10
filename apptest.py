# ===============================================================
# FINAL STREAMLIT CLOUD APP — PASSWORD PROTECTED / COOKIE BASED
# ===============================================================

import json
import time
import threading
from datetime import datetime
from pathlib import Path
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from bs4 import BeautifulSoup

# Optional ML
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error
    SKLEARN_AVAILABLE = True
except:
    SKLEARN_AVAILABLE = False

# ===============================================================
# AUTH — PASSWORD FROM SECRETS
# ===============================================================
if "auth_ok" not in st.session_state:
    st.session_state.auth_ok = False

if not st.session_state.auth_ok:
    st.set_page_config(page_title="Sign in", layout="centered")
    st.title("🔐 Private Dashboard")
    pw = st.text_input("Enter password", type="password")

    if st.button("Sign in"):
        correct = st.secrets.get("APP_PASSWORD", None)
        if pw and correct and pw == correct:
            st.session_state.auth_ok = True
            st.rerun()
        else:
            st.error("Invalid password.")
    st.stop()

# ===============================================================
# AFTER LOGIN — LOAD APP
# ===============================================================
st.set_page_config(page_title="Aviator Tracker", layout="wide")
st.title("📊 Aviator Tracker — Cookies-Only Cloud Version")

AVIATOR_URL = "https://topx.one/games/provider/spribe-aviatorz"
JSON_FILE = "snapshot.json"
CSV_FILE = "snapshot.csv"
SESSIONS_DIR = Path("sessions")
SESSIONS_DIR.mkdir(exist_ok=True)
SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

# ===============================================================
# READ COOKIES FROM SECRETS
# ===============================================================
def load_cookies_from_secrets():
    """
    Reads TOML entry:

    [cookies]
    data = \"\"\" [ {...}, {...} ] \"\"\"

    Returns list of cookie dicts.
    """
    if "cookies" not in st.secrets:
        return None

    raw = st.secrets["cookies"]
    data = raw.get("data") if isinstance(raw, dict) else raw
    try:
        return json.loads(data)
    except Exception as e:
        st.error(f"Failed to parse cookies from secrets: {e}")
        return None

cookies = load_cookies_from_secrets()
if cookies is None:
    st.error("❌ No cookies found in secrets. Add `[cookies] data = \"\"\"[...]\"\"\"` first.")
    st.stop()

# ===============================================================
# BUILD REQUESTS SESSION WITH COOKIES
# ===============================================================
def session_from_cookies(cookies):
    s = requests.Session()
    s.headers.update({
        "User-Agent": 
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
    })
    for c in cookies:
        cookie = requests.cookies.create_cookie(
            name=c.get("name"),
            value=c.get("value"),
            domain=c.get("domain"),
            path=c.get("path", "/")
        )
        s.cookies.set_cookie(cookie)
    return s

session = session_from_cookies(cookies)

# ===============================================================
# PARSE MULTIPLIERS (REQUESTS ONLY)
# ===============================================================
def extract_values(html):
    soup = BeautifulSoup(html, "html.parser")
    import re

    text_blocks = [
        soup.get_text(" ", strip=True)
    ]

    pattern = re.compile(r"(\d+(?:\.\d+)?)\s*x", re.I)
    out = []

    for block in text_blocks:
        for m in pattern.findall(block):
            try:
                val = float(m)
                out.append(val)
            except:
                pass

    unique = []
    seen = set()
    for v in out:
        rv = round(v, 4)
        if rv not in seen:
            unique.append(rv)
            seen.add(rv)
    return unique

def try_fetch():
    try:
        r = session.get(AVIATOR_URL, timeout=20)
        if r.status_code != 200:
            return None
        vals = extract_values(r.text)
        if len(vals) == 0:
            return None
        return vals
    except:
        return None

# ===============================================================
# STORE (CACHE)
# ===============================================================
@st.cache_resource
def get_store():
    return {"records": [], "running": False, "logs": []}

store = get_store()

# ===============================================================
# BACKGROUND THREAD
# ===============================================================
def poll_loop(store, interval):
    store["running"] = True
    last = None
    start = time.time()

    while store["running"]:
        vals = try_fetch()
        if vals and vals != last:
            now = datetime.now()
            record = {
                "iso_time": now.isoformat(timespec="seconds"),
                "t": time.time() - start,
                "new_round_value": vals[0],
                "history": vals
            }
            store["records"].append(record)
            store["logs"].append(f"{now.strftime('%H:%M:%S')} → {vals[0]}x ({len(vals)} entries)")
            save_snapshot(store["records"])
            last = vals
        time.sleep(interval)

def save_snapshot(records):
    if not records:
        return
    df = pd.DataFrame(records)
    df.to_csv(CSV_FILE, index=False)
    with open(JSON_FILE, "w") as f:
        json.dump(records, f, indent=2)

# ===============================================================
# SIDEBAR CONTROLS
# ===============================================================

st.sidebar.header("Controls")
interval = st.sidebar.number_input("Polling interval (sec)", 2.0, 60.0, 5.0)
theme = st.sidebar.selectbox("Theme", ["Dark", "Light"])

start_btn = st.sidebar.button("▶ Start")
stop_btn = st.sidebar.button("⏹ Stop")

if start_btn:
    if not store["running"]:
        test = try_fetch()
        if not test:
            st.error("❌ Failed to fetch multipliers via HTTP. Site likely requires JS.")
        else:
            st.success(f"Success! Found {len(test)} values. Starting tracker.")
            t = threading.Thread(target=poll_loop, args=(store, interval), daemon=True)
            t.start()
    else:
        st.info("Already running.")

if stop_btn:
    store["running"] = False
    st.warning("Stopped.")

# ===============================================================
# UI (MAIN)
# ===============================================================
records = store["records"]

if not records:
    st.info("No data yet. Click ▶ Start.")
    st.stop()

df = pd.DataFrame(records)
df["iso_time"] = pd.to_datetime(df["iso_time"])

# PLOT
def make_plot(df):
    df["idx"] = range(1, len(df) + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["iso_time"], y=df["new_round_value"],
        mode="lines+markers", name="Multiplier"
    ))
    fig.update_layout(
        title="Live Multiplier Stream",
        xaxis_title="Time",
        yaxis_title="Value (x)",
        template="plotly_dark" if theme=="Dark" else "plotly"
    )
    return fig

st.plotly_chart(make_plot(df), use_container_width=True)

# LOGS
with st.expander("📜 Logs"):
    st.text("\n".join(store["logs"][-300:]))

# ===============================================================
# RF MODEL (OPTIONAL)
# ===============================================================
if SKLEARN_AVAILABLE and len(df) > 40:

    st.subheader("🤖 Random Forest Demo")
    window = 10

    def make_xy(df, window):
        vals = df["new_round_value"].values
        X, y = [], []
        for i in range(len(vals) - window):
            X.append(vals[i:i+window])
            y.append(vals[i+window])
        return np.array(X), np.array(y)

    X, y = make_xy(df, window)
    if len(X) > 20:
        model = RandomForestRegressor(n_estimators=200)
        split = int(len(X)*0.8)
        model.fit(X[:split], y[:split])
        preds = model.predict(X[split:])

        st.write("Next 5 predicted values (demo):")
        last = df["new_round_value"].values[-window:]
        f = []
        x = last.copy()
        for _ in range(5):
            pred = model.predict([x])[0]
            f.append(pred)
            x = np.append(x[1:], pred)
        st.write([round(v, 2) for v in f])

# ===============================================================
# UPLOAD SNAPSHOT
# ===============================================================
with st.expander("Upload snapshot from local Selenium run"):
    file = st.file_uploader("Upload JSON", type="json")
    if file:
        data = json.load(file)
        if isinstance(data, list):
            with open(JSON_FILE, "w") as f:
                json.dump(data, f, indent=2)
            st.success("Snapshot uploaded.")
        else:
            st.error("Invalid format: must be list.")