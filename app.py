
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from typing import Optional, List, Dict

# ML & TS
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.set_page_config(page_title="Soya ML — UPSA | SolverTic", layout="wide")

# ---------------- Branding ----------------
DEFAULT_BRAND = {
    "org": "SolverTic SRL · UPSA",
    "primary_hex": "#4F46E5",   # indigo-600
    "accent_hex": "#06B6D4",    # cyan-500
    "dark": False,
    "footer": "© 2025 — SolverTic SRL · UPSA — Pronósticos de Precios de Soya (ML)"
}
if "brand" not in st.session_state:
    st.session_state["brand"] = DEFAULT_BRAND.copy()

def inject_css(brand: Dict):
    bg = "#0f172a" if brand.get("dark") else "#ffffff"
    fg = "#e2e8f0" if brand.get("dark") else "#111827"
    primary = brand.get("primary_hex", "#4F46E5")
    accent = brand.get("accent_hex", "#06B6D4")
    st.markdown(f"""
    <style>
      .stApp {{
        background: {bg};
        color: {fg};
      }}
      h1, h2, h3, h4, h5, h6 {{ color: {fg}; }}
      .css-1cypcdb a, a {{ color: {accent}; }}
      .stDownloadButton button, .stButton button {{
        background: {primary}; color: white; border-radius: 8px;
      }}
    </style>
    """, unsafe_allow_html=True)

inject_css(st.session_state["brand"])

# --------------- Sidebar ---------------
with st.sidebar:
    st.title("🎛️ Configuración")
    st.subheader("Branding")
    org = st.text_input("Organización", value=st.session_state["brand"]["org"])
    primary = st.color_picker("Primario", value=st.session_state["brand"]["primary_hex"])
    accent = st.color_picker("Acento", value=st.session_state["brand"]["accent_hex"])
    dark = st.checkbox("Tema oscuro", value=st.session_state["brand"]["dark"])
    logo = st.file_uploader("Logo (opcional, PNG/JPG)", type=["png","jpg","jpeg"], key="logo_brand")
    if st.button("Aplicar estilo"):
        st.session_state["brand"].update({"org": org, "primary_hex": primary, "accent_hex": accent, "dark": dark})
        st.experimental_rerun()

    st.markdown("---")
    PAGE = st.radio("Navegación", [
        "0) Proyecto",
        "1) Cargar datos",
        "2) EDA",
        "3) Ingeniería",
        "4) Modelos (TS-CV)",
        "5) Walk-Forward",
        "6) Pronósticos / Escenarios",
        "7) Benchmark (Bloomberg)",
        "8) Cobertura (Collar)",
        "9) Reporte / Exportación",
        "ℹ️ Glosario / Metodología"
    ])

if logo is not None:
    st.sidebar.image(logo, caption=st.session_state["brand"]["org"], use_column_width=True)
else:
    st.sidebar.write(st.session_state["brand"]["org"])

# --------------- Helpers ---------------
@st.cache_data
def load_table(file) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(file)
    return pd.DataFrame()

def parse_date(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        try:
            pd.to_datetime(df[c])
            return c
        except: pass
    return None

def add_calendar(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    d["year"] = d[date_col].dt.year
    d["month"] = d[date_col].dt.month
    d["quarter"] = d[date_col].dt.quarter
    d["week"] = d[date_col].dt.isocalendar().week.astype(int)
    d["is_month_start"] = d[date_col].dt.is_month_start.astype(int)
    d["is_month_end"] = d[date_col].dt.is_month_end.astype(int)
    return d

def add_lags_rolls(df: pd.DataFrame, target: str, lags: List[int], rolls: List[int]) -> pd.DataFrame:
    d = df.copy()
    for L in lags:
        d[f"{target}_lag{L}"] = d[target].shift(L)
    for R in rolls:
        d[f"{target}_rollmean{R}"] = d[target].rolling(R).mean()
        d[f"{target}_rollstd{R}"] = d[target].rolling(R).std()
    return d.dropna()

def metric_set(y_true, y_pred) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    mape = (np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-9))).mean() * 100
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}

def plotly_template():
    return st.selectbox("Plantilla de gráfico", ["plotly_white","plotly","ggplot2","seaborn","simple_white","plotly_dark","presentation"], index=0)

def models_dict(seed=42) -> Dict[str, object]:
    return {
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=seed),
        "Lasso": Lasso(alpha=0.001, random_state=seed, max_iter=10000),
        "RandomForest": RandomForestRegressor(n_estimators=400, random_state=seed),
        "GradientBoosting": GradientBoostingRegressor(random_state=seed),
        "XGBoost": XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=6, subsample=0.9, colsample_bytree=0.8, random_state=seed, objective="reg:squarederror"),
        "LightGBM": LGBMRegressor(n_estimators=800, learning_rate=0.03, num_leaves=31, subsample=0.9, colsample_bytree=0.8, random_state=seed),
        "CatBoost": CatBoostRegressor(iterations=600, learning_rate=0.03, depth=6, loss_function="RMSE", random_seed=seed, verbose=False),
    }

def tscv(X, y, n=5):
    splitter = TimeSeriesSplit(n_splits=n)
    for tr, te in splitter.split(X):
        yield tr, te

def walk_forward(df: pd.DataFrame, date_col: str, target: str, feats: List[str], model, initial: int, step: int):
    df = df.sort_values(date_col).reset_index(drop=True)
    out = []
    for start in range(initial, len(df)-step+1, step):
        tr = df.iloc[:start]; te = df.iloc[start:start+step]
        model.fit(tr[feats], tr[target])
        pred = model.predict(te[feats])
        out.append(pd.DataFrame({date_col: te[date_col].values, "y_true": te[target].values, "y_pred": pred}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=[date_col,"y_true","y_pred"])

# Session init
for k, v in {"data": pd.DataFrame(), "date_col":"", "target":"", "features":[], "engineered":pd.DataFrame(), "freq":"MS", "best_name":"", "best_model":None}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------- Pages ----------------
if PAGE.startswith("0"):
    st.title("📈 Pronósticos de Precios de Soya (ML)")
    st.markdown("""
**Objetivo:** entregar una herramienta **simple y completa** para cargar datos, explorarlos, crear características,
entrenar modelos de última generación (XGBoost, LightGBM, CatBoost), validar con **TimeSeriesSplit y Walk-Forward**,
generar **pronósticos y escenarios**, comparar con **benchmarks trimestrales** y simular **coberturas**.
""")
    st.success("Sugerencia: usa XGBoost/CatBoost; mide MAPE; valida con Walk-Forward.")

elif PAGE.startswith("1"):
    st.header("📥 Cargar datos")
    up = st.file_uploader("CSV/XLSX con columna de fecha y objetivo (precio soya)", type=["csv","xlsx"])
    if up:
        df = load_table(up)
        st.session_state["data"] = df.copy()
        st.write("Vista previa:", df.head())
        guess = parse_date(df) or ""
        date_col = st.selectbox("Columna de fecha", [""] + list(df.columns), index=(list(df.columns).index(guess)+1) if guess in df.columns else 0)
        target = st.selectbox("Objetivo (precio)", [""] + [c for c in df.columns if c != date_col])
        feats = st.multiselect("Features (opcionales)", [c for c in df.columns if c not in [date_col, target]])
        freq = st.selectbox("Frecuencia", ["D","W","MS","M","Q","YS"], index=2)
        if st.button("Guardar"):
            st.session_state.update({"date_col":date_col, "target":target, "features":feats, "freq":freq})
            st.success("Configuración guardada.")

elif PAGE.startswith("2"):
    st.header("🔎 EDA")
    df = st.session_state["data"]
    if df.empty:
        st.info("Carga datos primero.")
    else:
        template = plotly_template()
        st.write("Dimensiones:", df.shape)
        st.write("Tipos:", df.dtypes)
        with st.expander("Estadísticos"):
            st.write(df.describe(include="all"))
        x = st.selectbox("Eje X", list(df.columns))
        ys = st.multiselect("Ejes Y", [c for c in df.columns if c != x], default=[c for c in df.columns if c != x][:1])
        kind = st.selectbox("Tipo", ["Línea","Área","Barras","Dispersión","Boxplot","Heatmap (correlación)"])
        if ys:
            try:
                if kind=="Línea": fig = px.line(df, x=x, y=ys, template=template)
                elif kind=="Área": fig = px.area(df, x=x, y=ys, template=template)
                elif kind=="Barras": fig = px.bar(df, x=x, y=ys, template=template, barmode="group")
                elif kind=="Dispersión": fig = px.scatter(df, x=x, y=ys[0], template=template)
                elif kind=="Boxplot": fig = px.box(df, x=x, y=ys[0], template=template)
                else: fig = px.imshow(df.select_dtypes(include=[np.number]).corr(), template=template, color_continuous_scale="RdBu_r", origin="lower")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"No se pudo graficar: {e}")
        st.markdown("#### Tendencia/Ciclo (HP) y Descomposición (STL)")
        date_col, target = st.session_state["date_col"], st.session_state["target"]
        if date_col and target and date_col in df and target in df:
            s = df.copy(); s[date_col] = pd.to_datetime(s[date_col]); s = s.sort_values(date_col).set_index(date_col)
            lam = st.number_input("λ HP (mensual≈129,600)", value=129600, step=1000)
            try:
                cycle, trend = hpfilter(s[target], lamb=lam)
                st.plotly_chart(px.line(pd.DataFrame({"trend":trend,"cycle":cycle}), template=template, title="HP: tendencia & ciclo"), use_container_width=True)
            except Exception as e:
                st.info(f"HP filter no disponible: {e}")
            try:
                stl = STL(s[target], period=12).fit()
                st.plotly_chart(px.line(pd.DataFrame({"observed":s[target], "trend":stl.trend, "seasonal":stl.seasonal, "resid":stl.resid}), template=template, title="STL"), use_container_width=True)
            except Exception as e:
                st.info(f"STL no disponible: {e}")

elif PAGE.startswith("3"):
    st.header("🧪 Ingeniería")
    df = st.session_state["data"]; date_col = st.session_state["date_col"]; target = st.session_state["target"]
    if df.empty or not date_col or not target:
        st.info("Define fecha/objetivo en 'Cargar datos'.")
    else:
        lags = st.multiselect("Lags", [1,2,3,6,9,12,18,24], default=[1,2,3,6,12])
        rolls = st.multiselect("Ventanas móviles", [3,6,12,24], default=[3,6,12])
        d = df.copy(); d[date_col] = pd.to_datetime(d[date_col]); d = d.sort_values(date_col)
        d = add_calendar(d, date_col)
        d = add_lags_rolls(d, target, lags, rolls)
        d = d.dropna()
        st.session_state["engineered"] = d.copy()
        st.write("Vista previa:", d.head())
        st.download_button("Descargar CSV ingenierizado", d.to_csv(index=False).encode("utf-8"), "datos_ingenierizados.csv")

elif PAGE.startswith("4"):
    st.header("🤖 Modelos (TimeSeriesSplit)")
    df = st.session_state["engineered"]; date_col = st.session_state["date_col"]; target = st.session_state["target"]
    if df.empty:
        st.info("Realiza ingeniería primero.")
    else:
        feats_all = [c for c in df.columns if c not in [date_col, target]]
        feats = st.multiselect("Predictores", feats_all, default=[c for c in feats_all if ("lag" in c) or ("roll" in c) or (c in ["year","month","quarter","week"])])
        n = st.slider("Folds", 3, 8, 5, 1)
        template = plotly_template()
        X, y = df[feats], df[target]
        models = models_dict()
        rows = []
        for name, m in models.items():
            maes, rmses, mapes = [], [], []
            for tr, te in tscv(X, y, n):
                m.fit(X.iloc[tr], y.iloc[tr])
                yp = m.predict(X.iloc[te])
                me = metric_set(y.iloc[te].values, yp)
                maes.append(me["MAE"]); rmses.append(me["RMSE"]); mapes.append(me["MAPE"])
            rows.append({"Modelo": name, "MAE": np.mean(maes), "RMSE": np.mean(rmses), "MAPE": np.mean(mapes)})
        res = pd.DataFrame(rows).sort_values("MAPE")
        st.subheader("Ranking (MAPE promedio)")
        st.dataframe(res, use_container_width=True)
        st.plotly_chart(px.bar(res, x="Modelo", y="MAPE", template=template, title="MAPE (TS-CV)"), use_container_width=True)
        best = res.iloc[0]["Modelo"]
        st.session_state["best_name"] = best
        st.session_state["best_model"] = models[best].fit(X, y)
        st.success(f"Mejor modelo: {best} (entrenado full data).")

elif PAGE.startswith("5"):
    st.header("🚶 Walk-Forward")
    df = st.session_state["engineered"]; date_col = st.session_state["date_col"]; target = st.session_state["target"]
    best = st.session_state["best_model"]
    if df.empty or best is None:
        st.info("Entrena modelos en la sección 4.")
    else:
        feats = [c for c in df.columns if c not in [date_col, target]]
        initial = st.number_input("Tamaño inicial (meses)", value=max(36, int(len(df)*0.6)), min_value=12, step=1)
        step = st.number_input("Paso (meses)", value=1, min_value=1, step=1)
        template = plotly_template()
        wf = walk_forward(df[[date_col, target] + feats], date_col, target, feats, best, int(initial), int(step))
        st.write(wf.tail())
        ms = metric_set(wf["y_true"].values, wf["y_pred"].values)
        st.info(f"Walk-Forward → MAPE={ms['MAPE']:.2f}% · RMSE={ms['RMSE']:.2f} · MAE={ms['MAE']:.2f} · R²={ms['R2']:.3f}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=wf[date_col], y=wf["y_true"], name="Real"))
        fig.add_trace(go.Scatter(x=wf[date_col], y=wf["y_pred"], name="Pred"))
        fig.update_layout(template=template, title="Real vs Pred (Walk-Forward)")
        st.plotly_chart(fig, use_container_width=True)

elif PAGE.startswith("6"):
    st.header("🔮 Pronósticos / Escenarios")
    df = st.session_state["engineered"]; date_col = st.session_state["date_col"]; target = st.session_state["target"]
    best = st.session_state["best_model"]; freq = st.session_state["freq"]
    if df.empty or best is None:
        st.info("Entrena el mejor modelo (sección 4).")
    else:
        horizon = st.slider("Horizonte", 1, 36, 12, 1)
        feats = [c for c in df.columns if c not in [date_col, target]]
        base = df[feats].iloc[[-1]].copy()
        futureX = pd.concat([base]*horizon, ignore_index=True)
        st.subheader("Ajustes (Δ %) sobre rolling")
        editable = [c for c in feats if ("rollmean" in c) or ("rollstd" in c)]
        for c in editable:
            dv = st.number_input(c, value=0.0, step=0.5, format="%.2f")
            futureX[c] = futureX[c] * (1 + dv/100.0)
        last_date = pd.to_datetime(df[date_col]).max()
        idx = pd.date_range(last_date, periods=horizon+1, freq=freq)[1:]
        yhat = best.predict(futureX)
        pred = pd.DataFrame({date_col: idx, "forecast": yhat})
        template = plotly_template()
        st.plotly_chart(px.line(pred, x=date_col, y="forecast", template=template, title="Pronóstico (ML)"), use_container_width=True)
        st.dataframe(pred, use_container_width=True)
        st.download_button("Descargar pronóstico (CSV)", pred.to_csv(index=False).encode("utf-8"), "pronostico.csv")

        st.markdown("---")
        st.info("SARIMAX (opcional)")
        p = st.number_input("p", 0, 5, 1); d = st.number_input("d", 0, 2, 1); q = st.number_input("q", 0, 5, 1)
        P = st.number_input("P", 0, 5, 1); D = st.number_input("D", 0, 2, 1); Q = st.number_input("Q", 0, 5, 1); s = st.number_input("s (mensual=12)", 0, 24, 12)
        if st.button("Entrenar SARIMAX"):
            try:
                srs = df.sort_values(date_col).set_index(pd.to_datetime(df[date_col]))[target].asfreq(freq).ffill()
                model = SARIMAX(srs, order=(p,d,q), seasonal_order=(P,D,Q,s), enforce_stationarity=False, enforce_invertibility=False)
                res = model.fit(disp=False)
                fc = res.get_forecast(steps=horizon)
                ci = fc.conf_int(); mean = fc.predicted_mean.reset_index(); mean.columns = [date_col, "sarimax"]
                plot = mean.copy(); ci = ci.reset_index(drop=True); plot["lower"] = ci.iloc[:,0]; plot["upper"] = ci.iloc[:,1]
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=plot[date_col], y=plot["sarimax"], name="SARIMAX"))
                fig2.add_trace(go.Scatter(x=plot[date_col], y=plot["upper"], name="Upper", mode="lines"))
                fig2.add_trace(go.Scatter(x=plot[date_col], y=plot["lower"], name="Lower", mode="lines", fill="tonexty"))
                fig2.update_layout(template=template, title="SARIMAX con bandas")
                st.plotly_chart(fig2, use_container_width=True)
            except Exception as e:
                st.error(f"SARIMAX falló: {e}")

elif PAGE.startswith("7"):
    st.header("📊 Benchmark (Bloomberg)")
    st.write("Archivo con columnas: trimestre (ej. 2025Q3) y valor (USD/TM).")
    bench_file = st.file_uploader("Benchmark (.csv/.xlsx)", type=["csv","xlsx"])
    df = st.session_state["engineered"]; date_col = st.session_state["date_col"]
    if bench_file and not df.empty and date_col in df:
        bench = load_table(bench_file)
        fc_file = st.file_uploader("Pronóstico (archivo CSV exportado en sección 6)", type=["csv"])
        if fc_file:
            fc = pd.read_csv(fc_file)
            fc[date_col] = pd.to_datetime(fc[date_col]); fc["quarter"] = fc[date_col].dt.to_period("Q")
            agg = fc.groupby("quarter")["forecast"].mean().reset_index()
            agg.columns = ["trimestre","forecast_modelo"]
            merged = pd.merge(agg, bench, left_on="trimestre", right_on=bench.columns[0], how="inner")
            merged["diff"] = merged["forecast_modelo"] - merged.iloc[:,2]
            st.dataframe(merged, use_container_width=True)
            template = plotly_template()
            fig = go.Figure()
            fig.add_trace(go.Bar(x=merged["trimestre"].astype(str), y=merged["forecast_modelo"], name="Modelo"))
            fig.add_trace(go.Bar(x=merged["trimestre"].astype(str), y=merged.iloc[:,2], name="Bloomberg"))
            fig.update_layout(barmode="group", template=template, title="Modelo vs Bloomberg")
            st.plotly_chart(fig, use_container_width=True)
            st.info(f"Diferencia promedio absoluta: {merged['diff'].abs().mean():.2f} USD/TM")
    else:
        st.info("Genera y sube tu pronóstico para comparar.")

elif PAGE.startswith("8"):
    st.header("🛡️ Cobertura (collar modificado)")
    spot = st.number_input("Spot esperado (USD/TM)", value=380.0, step=1.0)
    k_put = st.number_input("Strike PUT", value=360.0, step=1.0)
    k_call = st.number_input("Strike CALL", value=375.0, step=1.0)
    premium_put = st.number_input("Prima PUT", value=4.41, step=0.1)
    premium_call = st.number_input("Prima CALL", value=19.84, step=0.1)
    template = plotly_template()
    prices = np.linspace(spot*0.8, spot*1.2, 120)
    base = prices
    put_payoff = np.maximum(k_put - prices, 0) - premium_put
    call_payoff = -np.maximum(prices - k_call, 0) + premium_call
    net = base + put_payoff + call_payoff
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices, y=base, name="Sin cobertura"))
    fig.add_trace(go.Scatter(x=prices, y=net, name="Collar modificado"))
    fig.update_layout(template=template, title="Payoff — 1 TM (ilustrativo)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Protección total bajo PUT; upside limitado sobre CALL; prima neta = CALL - PUT.")

elif PAGE.startswith("9"):
    st.header("📝 Reporte / Exportación")
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    summary = {
        "fecha": now,
        "org": st.session_state["brand"]["org"],
        "mejor_modelo": st.session_state.get("best_name",""),
        "freq": st.session_state.get("freq","MS"),
        "date_col": st.session_state.get("date_col",""),
        "target": st.session_state.get("target",""),
        "features": st.session_state.get("features", []),
    }
    st.json(summary)
    comments = st.text_area("Comentarios adicionales (opcional)")
    md = f"""# Reporte — {summary['org']}
**Fecha:** {now}

## Configuración
- Mejor modelo: {summary['mejor_modelo']}
- Frecuencia: {summary['freq']}
- Fecha: {summary['date_col']}
- Objetivo: {summary['target']}
- Features base: {', '.join(summary['features'])}

## Comentarios
{comments}

> Generado con Soya ML — Streamlit.
"""
    st.download_button("Descargar reporte (.md)", md, file_name="reporte_soya_brand.md")
    if "engineered" in st.session_state and not st.session_state["engineered"].empty:
        st.download_button("Descargar datos ingenierizados (CSV)", st.session_state["engineered"].to_csv(index=False).encode("utf-8"), "datos_ingenierizados.csv")

elif PAGE.startswith("ℹ️"):
    st.header("ℹ️ Glosario / Metodología (resumen)")
    st.markdown("""
- **MAPE**: error porcentual absoluto medio; compara precisión proporcional al nivel de la serie.
- **RMSE / MAE**: errores en unidades (USD/TM). RMSE penaliza grandes errores.
- **TimeSeriesSplit**: validación cruzada respetando el orden temporal.
- **Walk-Forward**: retrotesteo incremental que emula el uso en producción.
- **Boosting (XGBoost/LightGBM/CatBoost)**: ensambles de árboles en secuencia que corrigen errores previos.
- **Lags / Rolling**: retardos y ventanas móviles del precio (capturan inercia y suavizan ruido).
- **HP Filter / STL**: herramientas para separar tendencia, ciclo y estacionalidad.
- **SARIMAX**: modelo clásico de series temporales con sazonalidad.
- **Collar**: estrategia de cobertura: PUT largo + CALL corto para acotar rango de precios.
""")

st.caption(st.session_state["brand"]["footer"])
