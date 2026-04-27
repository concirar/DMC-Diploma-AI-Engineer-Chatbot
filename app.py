"""
app.py — Streamlit Credit Scoring Assistant
Chat conversacional para análisis de riesgo crediticio por ID de cliente.

Uso:
    streamlit run app.py

Requiere:
    - clients.csv (generado por generate_clients_db.py)
    - mlflow_credit.db (generado por el notebook)
    - Ollama corriendo: ollama serve
"""
import re
import warnings
from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import ollama
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

MLFLOW_TRACKING_URI = "sqlite:///mlflow_credit.db"
MODEL_URI = "models:/credit-scoring-polish@production_80"
CLIENTS_CSV = "clients.csv"
OLLAMA_MODEL = "gemma4:e2b"

FEATURE_DESC = {
    "year": "horizonte de pronóstico (años antes de quiebra)",
    "A1": "utilidad neta / activos totales",
    "A2": "pasivos totales / activos totales",
    "A3": "capital de trabajo / activos totales",
    "A4": "activos corrientes / pasivos corrientes (liquidez)",
    "A5": "ciclo de caja en días",
    "A6": "utilidades retenidas / activos totales",
    "A7": "EBIT / activos totales",
    "A8": "patrimonio contable / pasivos totales",
    "A9": "ventas / activos totales (rotación)",
    "A10": "patrimonio / activos totales",
    "A11": "utilidad bruta + gastos financieros / activos totales",
    "A12": "utilidad bruta / pasivos corrientes",
    "A13": "margen bruto + depreciación / ventas",
    "A14": "utilidad bruta + intereses / activos totales",
    "A15": "días para pagar pasivos totales",
    "A16": "utilidad bruta + depreciación / pasivos totales",
    "A17": "activos totales / pasivos totales",
    "A18": "utilidad bruta / activos totales",
    "A19": "margen bruto (utilidad bruta / ventas)",
    "A20": "rotación de inventarios en días",
    "A21": "crecimiento de ventas",
    "A22": "utilidad operativa / activos totales",
    "A23": "utilidad neta / ventas (margen neto)",
    "A24": "utilidad bruta 3 años / activos totales",
    "A25": "(patrimonio - capital social) / activos totales",
    "A26": "(utilidad neta + depreciación) / pasivos totales",
    "A27": "utilidad operativa / gastos financieros",
    "A28": "capital de trabajo / activos fijos",
    "A29": "logaritmo de activos totales (tamaño empresa)",
    "A30": "(pasivos totales - caja) / ventas",
    "A31": "(utilidad bruta + intereses) / ventas",
    "A32": "días de pasivos corrientes sobre costo de ventas",
    "A33": "gastos operativos / pasivos corrientes",
    "A34": "gastos operativos / pasivos totales",
    "A35": "utilidad en ventas / activos totales",
    "A36": "ventas totales / activos totales",
    "A37": "(activos corrientes - inventarios) / pasivos a largo plazo",
    "A38": "capital constante / activos totales",
    "A39": "margen de utilidad en ventas",
    "A40": "prueba ácida (sin inventarios)",
    "A41": "días para pagar pasivos con EBITDA",
    "A42": "margen operativo",
    "A43": "rotación de cobros + días de inventario",
    "A44": "días de cobro (cuentas por cobrar / ventas × 365)",
    "A45": "utilidad neta / inventarios",
    "A46": "activos corrientes sin inventarios / pasivos corrientes",
    "A47": "días de inventario (costo de ventas)",
    "A48": "EBITDA / activos totales",
    "A49": "EBITDA / ventas",
    "A50": "activos corrientes / pasivos totales",
    "A51": "pasivos corrientes / activos totales",
    "A52": "días de pasivos corrientes sobre costo de ventas",
    "A53": "patrimonio / activos fijos",
    "A54": "capital constante / activos fijos",
    "A55": "capital de trabajo (valor absoluto)",
    "A56": "margen bruto (ventas - costos) / ventas",
    "A57": "índice de liquidez ajustada",
    "A58": "costos totales / ventas totales",
    "A59": "deuda a largo plazo / patrimonio",
    "A60": "rotación de inventarios (ventas / inventario)",
    "A61": "rotación de cuentas por cobrar",
    "A62": "días de pasivos corrientes sobre ventas",
    "A63": "ventas / pasivos corrientes",
    "A64": "ventas / activos fijos",
}

RISK_CONFIG = {
    "BAJO":     {"emoji": "🟢", "color": "#2ecc71"},
    "MEDIO":    {"emoji": "🟡", "color": "#f39c12"},
    "ALTO":     {"emoji": "🟠", "color": "#e67e22"},
    "MUY ALTO": {"emoji": "🔴", "color": "#e74c3c"},
}

HELP_TEXT = """**Comandos disponibles:**
- `analiza CLI-00418` — análisis completo de un cliente
- `CLI-00418` — forma corta
- `ayuda` — mostrar este mensaje

Los IDs van de `CLI-00001` a `CLI-08681`."""


# ---------------------------------------------------------------------------
# Carga de recursos (cacheados)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Cargando modelo XGBoost y SHAP explainer...")
def load_model_and_explainer():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model = mlflow.xgboost.load_model(MODEL_URI)
    explainer = shap.TreeExplainer(model)
    return model, explainer


@st.cache_data(show_spinner="Cargando base de clientes...")
def load_clients() -> pd.DataFrame:
    return pd.read_csv(CLIENTS_CSV, index_col="id")


# ---------------------------------------------------------------------------
# Funciones de análisis
# ---------------------------------------------------------------------------

def classify_risk(proba: float) -> tuple[str, str]:
    if proba < 0.20:
        return "BAJO", "🟢"
    if proba < 0.50:
        return "MEDIO", "🟡"
    if proba < 0.75:
        return "ALTO", "🟠"
    return "MUY ALTO", "🔴"


def get_top_drivers(shap_vals: np.ndarray, features_df: pd.DataFrame, n: int = 6) -> list[dict]:
    impact = pd.Series(shap_vals, index=features_df.columns)
    top_idx = impact.abs().sort_values(ascending=False).head(n).index
    return [
        {
            "feature": feat,
            "descripcion": FEATURE_DESC.get(feat, feat),
            "valor": float(features_df[feat].values[0]),
            "shap": float(impact[feat]),
            "sube_riesgo": impact[feat] > 0,
        }
        for feat in top_idx
    ]


def shap_chart(drivers: list[dict]) -> go.Figure:
    labels = [f"{d['feature']}<br><sup>{d['descripcion'][:40]}</sup>" for d in drivers]
    values = [d["shap"] for d in drivers]
    colors = [RISK_CONFIG["MUY ALTO"]["color"] if v > 0 else RISK_CONFIG["BAJO"]["color"] for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.4f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title="Factores de impacto en el score (SHAP)",
        xaxis_title="Impacto sobre la probabilidad de quiebra",
        yaxis={"autorange": "reversed"},
        height=320,
        margin=dict(l=10, r=60, t=40, b=10),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="#fafafa",
    )
    fig.add_vline(x=0, line_width=1, line_color="#555")
    return fig


def build_ollama_prompt(score_pct: float, nivel: str, drivers: list[dict], client_id: str) -> str:
    lineas = []
    for d in drivers:
        efecto = "AUMENTA el riesgo" if d["sube_riesgo"] else "REDUCE el riesgo"
        lineas.append(f"  - {d['feature']} ({d['descripcion']}): valor={d['valor']:.4f} → {efecto}")
    return (
        f"Eres un analista senior de riesgo crediticio corporativo.\n"
        f"Empresa analizada: \"{client_id}\"\n\n"
        f"RESULTADO DEL MODELO:\n"
        f"- Probabilidad de quiebra: {score_pct:.1f}%\n"
        f"- Nivel de riesgo: {nivel}\n\n"
        f"FACTORES DE MAYOR IMPACTO EN EL SCORE:\n"
        + "\n".join(lineas)
        + "\n\nRedacta un dictamen crediticio profesional en español de máximo 200 palabras que:\n"
        "1. Resuma el nivel de riesgo en una oración.\n"
        "2. Explique los 3 factores de mayor impacto en lenguaje claro.\n"
        "3. No uses términos de machine learning (no menciones SHAP ni XGBoost).\n"
        "4. Tono formal de comité de crédito.\n"
    )


def stream_dictamen(prompt: str):
    try:
        stream = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        for chunk in stream:
            yield chunk["message"]["content"]
    except Exception as e:
        yield f"\n\n*[Ollama no disponible: {e}. Ejecuta `ollama serve` en una terminal.]*"


# ---------------------------------------------------------------------------
# Parsing de intención
# ---------------------------------------------------------------------------

def parse_intent(text: str) -> tuple[str, str | None]:
    text = text.strip()
    if text.lower() in ("ayuda", "help", "?"):
        return "help", None

    # Extrae cualquier ID con patrón CLI-XXXXX (o variantes con guión, sin guión)
    m = re.search(r"CLI[- ]?(\d+)", text, re.IGNORECASE)
    if m:
        num = int(m.group(1))
        return "analyze", f"CLI-{num:05d}"

    # Solo número → interpretar como ID
    m = re.match(r"^\d+$", text)
    if m:
        return "analyze", f"CLI-{int(text):05d}"

    return "unknown", None


# ---------------------------------------------------------------------------
# Flujo principal de análisis
# ---------------------------------------------------------------------------

def analyze_client(client_id: str) -> None:
    clients = load_clients()
    model, explainer = load_model_and_explainer()

    # 1. Lookup
    if client_id not in clients.index:
        st.error(
            f"No encontré el cliente `{client_id}` en la base de datos. "
            f"Los IDs van de `CLI-00001` a `CLI-{len(clients):05d}`."
        )
        return

    row = clients.loc[client_id]
    clase_real = int(row["class"])
    feature_cols = [c for c in clients.columns if c != "class"]
    features_df = pd.DataFrame([row[feature_cols].values], columns=feature_cols)

    # 2. Score
    proba = float(model.predict_proba(features_df)[0][1])
    score_pct = proba * 100
    nivel, emoji = classify_risk(proba)
    color = RISK_CONFIG[nivel]["color"]
    clase_real_label = "Quiebra real" if clase_real == 1 else "No quiebra real"
    clase_real_icon = "🔴" if clase_real == 1 else "🟢"

    # 3. SHAP
    shap_vals = explainer.shap_values(features_df)[0]
    drivers = get_top_drivers(shap_vals, features_df)

    # 4. Renderizar respuesta del asistente
    st.markdown(f"### {emoji} Análisis — `{client_id}`")

    col1, col2, col3 = st.columns(3)
    col1.metric("Score de quiebra", f"{score_pct:.1f}%")
    col2.metric("Nivel de riesgo", f"{emoji} {nivel}")
    col3.metric("Clase real", f"{clase_real_icon} {clase_real_label}")

    st.plotly_chart(shap_chart(drivers), use_container_width=True)

    with st.expander("Ver detalle de los 6 ratios más influyentes"):
        tabla = pd.DataFrame([
            {
                "Ratio": d["feature"],
                "Descripción": d["descripcion"],
                "Valor": f"{d['valor']:.4f}",
                "Impacto SHAP": f"{d['shap']:+.4f}",
                "Efecto": "↑ Aumenta riesgo" if d["sube_riesgo"] else "↓ Reduce riesgo",
            }
            for d in drivers
        ])
        st.dataframe(tabla, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("**Dictamen del Asistente IA (Gemma via Ollama):**")
    prompt = build_ollama_prompt(score_pct, nivel, drivers, client_id)
    st.write_stream(stream_dictamen(prompt))

    # 5. Actualizar historial en session state
    st.session_state.history.append({
        "id": client_id,
        "score": score_pct,
        "nivel": nivel,
        "emoji": emoji,
        "clase_real": clase_real,
    })


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 🏦 Credit Scoring")
        st.markdown("Asistente de riesgo crediticio — empresas polacas")
        st.divider()

        st.markdown("### Métricas del modelo")
        col1, col2 = st.columns(2)
        col1.metric("ROC-AUC", "0.9491")
        col2.metric("Avg Precision", "0.6824")
        col1.metric("F1 quiebra", "0.62")
        col2.metric("Precisión", "0.71")
        st.caption("XGBoost · SMOTE · 65 ratios financieros")
        st.divider()

        if st.session_state.history:
            st.markdown("### Clientes analizados")
            for entry in reversed(st.session_state.history):
                real_icon = "🔴" if entry["clase_real"] == 1 else "🟢"
                st.markdown(
                    f"`{entry['id']}` {entry['emoji']} **{entry['nivel']}** "
                    f"({entry['score']:.1f}%) {real_icon}"
                )
            st.divider()

        clients_ok = Path(CLIENTS_CSV).exists()
        st.markdown("### Estado del sistema")
        st.markdown(f"{'✅' if clients_ok else '❌'} Base de clientes (`clients.csv`)")

        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            client = mlflow.MlflowClient()
            client.get_registered_model("credit-scoring-polish")
            st.markdown("✅ MLflow Model Registry")
        except Exception:
            st.markdown("⚠️ MLflow Model Registry")

        try:
            ollama.list()
            st.markdown("✅ Ollama / Gemma")
        except Exception:
            st.markdown("❌ Ollama (ejecuta `ollama serve`)")

        if st.session_state.history:
            st.divider()
            if st.button("🗑️ Limpiar sesión", use_container_width=True):
                st.session_state.messages = []
                st.session_state.history = []
                st.rerun()


# ---------------------------------------------------------------------------
# App principal
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Credit Scoring Assistant",
        page_icon="🏦",
        layout="wide",
    )

    # Inicializar session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "history" not in st.session_state:
        st.session_state.history = []

    render_sidebar()

    st.title("🤖 Asistente de Riesgo Crediticio")
    st.caption("Escribe un ID de cliente para obtener el análisis completo.")

    # Verificar que clients.csv existe antes de cargar el modelo
    if not Path(CLIENTS_CSV).exists():
        st.error(
            "No se encontró `clients.csv`. Ejecuta primero:\n\n"
            "```bash\npython generate_clients_db.py\n```"
        )
        st.stop()

    # Pre-cargar recursos en background al iniciar
    load_clients()
    load_model_and_explainer()

    # Renderizar historial de mensajes del chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                # El contenido de los mensajes del asistente ya se renderizó
                # durante el análisis; aquí solo mostramos el resumen
                st.markdown(msg["content"])

    # Input del chat
    user_input = st.chat_input("Escribe un ID de cliente o escribe 'ayuda'...")
    if not user_input:
        return

    # Mostrar mensaje del usuario
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Procesar intención
    intent, client_id = parse_intent(user_input)

    with st.chat_message("assistant"):
        if intent == "help":
            st.markdown(HELP_TEXT)
            st.session_state.messages.append({"role": "assistant", "content": HELP_TEXT})

        elif intent == "analyze":
            analyze_client(client_id)
            summary = f"Análisis completado para `{client_id}`."
            st.session_state.messages.append({"role": "assistant", "content": summary})

        else:
            msg = (
                f"No entendí `{user_input}`. "
                "Escribe un ID de cliente como `CLI-00418` o escribe `ayuda`."
            )
            st.markdown(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})


if __name__ == "__main__":
    main()
