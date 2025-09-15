
import os, json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from typing import List, Dict, Optional

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

# ---------- Branding ----------
DEFAULT_BRAND = {"org": "SolverTic SRL · UPSA", "primary_hex": "#4F46E5", "accent_hex": "#06B6D4", "dark": False}
if "brand" not in st.session_state:
    st.session_state["brand"] = DEFAULT_BRAND.copy()

def inject_css():
    b = st.session_state["brand"]
    bg = "#0f172a" if b["dark"] else "#ffffff"
    fg = "#e2e8f0" if b["dark"] else "#111827"
    st.markdown(f"""
    <style>
      .stApp {{ background:{bg}; color:{fg}; }}
      h1,h2,h3,h4,h5,h6 {{ color:{fg}; }}
      .stButton button, .stDownloadButton button {{ background:{b['primary_hex']}; color:white; border-radius:8px; }}
      a {{ color:{b['accent_hex']}; }}
      .badge {{ display:inline-block; padding:2px 8px; border-radius:12px; background:{b['accent_hex']}; color:white; font-size:11px; }}
    </style>
    """, unsafe_allow_html=True)

inject_css()

# ---------- Sidebar ----------
with st.sidebar:
    st.title("🎛️ Configuración")
    st.subheader("Branding")
    b = st.session_state["brand"]
    org = st.text_input("Organización", b["org"])
    primary = st.color_picker("Primario", b["primary_hex"])
    accent = st.color_picker("Acento", b["accent_hex"])
    dark = st.checkbox("Tema oscuro", b["dark"])
    logo = st.file_uploader("Logo (PNG/JPG) opcional", type=["png","jpg","jpeg"])
    if st.button("Aplicar estilo"):
        st.session_state["brand"].update({"org": org, "primary_hex": primary, "accent_hex": accent, "dark": dark})
        st.rerun()

    st.markdown("---")
    PAGE = st.radio("Navegación", [
        "0) Proyecto","1) Cargar datos","2) EDA","3) Ingeniería",
        "4) Modelos (TS-CV)","5) Walk-Forward","6) Pronósticos / Escenarios",
        "7) Benchmark","8) Cobertura","9) Reporte","ℹ️ Glosario"
    ])

    st.markdown("---")
    st.subheader("🧭 Perfil de unidades")
    if "units_profile" not in st.session_state:
        st.session_state["units_profile"] = {"percent_cols": [], "percent_0_100": True, "currency_cols": [], "index_cols": []}
    prof = st.session_state["units_profile"]

    data_cols = list(st.session_state.get("data", pd.DataFrame()).columns) if "data" in st.session_state else []
    if data_cols:
        # Correct arg order
        prof["percent_cols"] = st.multiselect("Columnas en % (ratios)", options=data_cols, default=prof.get("percent_cols", []), help="Ej: stock/consumo, participación, tasas.")
        prof["percent_0_100"] = st.checkbox("Esquema % en 0–100 (no 0–1)", value=prof.get("percent_0_100", True))
        prof["currency_cols"] = st.multiselect("Columnas en moneda (USD)", options=data_cols, default=prof.get("currency_cols", []), help="Ej: precio_soya, costos en USD.")
        prof["index_cols"] = st.multiselect("Columnas índice (base=100)", options=data_cols, default=prof.get("index_cols", []))

        c1, c2 = st.columns(2)
        with c1:
            if st.button("💾 Guardar perfil (JSON)"):
                st.session_state["units_profile"] = prof
                st.session_state["units_profile_json"] = json.dumps(prof, ensure_ascii=False, indent=2)
        with c2:
            up_prof = st.file_uploader("📤 Cargar perfil (JSON)", type=["json"], key="profile_upload")
            if up_prof is not None:
                try:
                    loaded = json.loads(up_prof.read().decode("utf-8"))
                    st.session_state["units_profile"] = loaded
                    st.success("Perfil cargado.")
                except Exception as e:
                    st.error(f"No se pudo cargar el perfil: {e}")
    else:
        st.info("Carga datos para definir el perfil de unidades.")

if logo is not None:
    st.sidebar.image(logo, caption=st.session_state["brand"]["org"], use_column_width=True)
else:
    st.sidebar.write(st.session_state["brand"]["org"])

# ---------- Helpers ----------
@st.cache_data
def load_table(file)->pd.DataFrame:
    n=file.name.lower()
    if n.endswith(".csv"): return pd.read_csv(file)
    if n.endswith(".xlsx") or n.endswith(".xls"): return pd.read_excel(file)
    return pd.DataFrame()

def parse_date(df)->Optional[str]:
    for c in df.columns:
        try:
            pd.to_datetime(df[c]); return c
        except: pass
    return None

def try_parse_dates_series(s: pd.Series, dayfirst=False):
    return pd.to_datetime(s.astype(str).str.strip(), errors='coerce', infer_datetime_format=True, dayfirst=dayfirst, utc=False)

def robust_to_datetime(df, col):
    s = df[col].astype(str).str.strip()
    dt1 = try_parse_dates_series(s, dayfirst=False)
    if dt1.isna().mean() > 0.3:
        dt2 = try_parse_dates_series(s, dayfirst=True)
        dt = dt2
        if dt.isna().mean() > 0.3:
            s2 = s.str.replace('/', '-', regex=False)
            dt3 = try_parse_dates_series(s2, dayfirst=False)
            if dt3.isna().mean() > 0.3:
                dt4 = try_parse_dates_series(s2, dayfirst=True)
                dt = dt4
            else:
                dt = dt3
    else:
        dt = dt1
    out = df.copy()
    out[col] = dt
    return out

def invalid_date_rows(df, col):
    orig = df.copy()
    parsed = robust_to_datetime(df, col)
    bad_mask = parsed[col].isna()
    bad = orig.loc[bad_mask, [col]].copy()
    bad.columns = [f"{col}_original"]
    bad = bad.reset_index().rename(columns={"index": "row_idx"})
    return bad

def filter_dataset_by_valid_dates(df, col):
    parsed = robust_to_datetime(df, col)
    good = parsed.loc[parsed[col].notna()].copy()
    bad = df.loc[parsed[col].isna()].copy()
    return good, bad

def add_calendar(df, date_col):
    d=df.copy(); d[date_col]=pd.to_datetime(d[date_col])
    d["year"]=d[date_col].dt.year; d["month"]=d[date_col].dt.month; d["quarter"]=d[date_col].dt.quarter; d["week"]=d[date_col].dt.isocalendar().week.astype(int)
    d["is_month_start"]=d[date_col].dt.is_month_start.astype(int); d["is_month_end"]=d[date_col].dt.is_month_end.astype(int)
    return d

def add_lags_rolls(df, target, lags, rolls):
    if pd.api.types.is_datetime64_any_dtype(df[target]):
        raise TypeError(f"El objetivo '{target}' es de tipo fecha. Selecciona una columna numérica.")
    d = df.copy()
    ser = d[target].astype(str).str.replace(r'[^0-9,.\-]', '', regex=True).str.replace(',', '', regex=False)
    d[target] = pd.to_numeric(ser, errors='coerce')
    for L in lags:
        d[f"{target}_lag{L}"] = d[target].shift(L)
    for R in rolls:
        if R < len(d):
            d[f"{target}_rollmean{R}"] = d[target].rolling(R, min_periods=max(1, R//2)).mean()
            d[f"{target}_rollstd{R}"] = d[target].rolling(R, min_periods=max(2, R//2)).std()
    return d.dropna(subset=[target])

def numeric_features(df, cols): return [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]

def clean_xy(X, y):
    Xc=X.replace([np.inf,-np.inf], np.nan).ffill().bfill().fillna(0)
    yc=y.replace([np.inf,-np.inf], np.nan).ffill().bfill()
    return Xc, yc

def models(seed=42)->Dict[str,object]:
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

def metric_set(y_true, y_pred):
    mae=mean_absolute_error(y_true,y_pred); rmse=mean_squared_error(y_true,y_pred,squared=False); r2=r2_score(y_true,y_pred)
    mape=(np.abs((y_true - y_pred)/np.maximum(np.abs(y_true),1e-9))).mean()*100
    return {"MAE":mae,"RMSE":rmse,"R2":r2,"MAPE":mape}

def tscv(X,y,n=5):
    sp=TimeSeriesSplit(n_splits=n)
    for tr,te in sp.split(X): yield tr,te

def walk_forward(df, date_col, target, feats, model, initial, step):
    df=df.sort_values(date_col).reset_index(drop=True)
    feats_num=numeric_features(df, feats)
    out=[]
    for start in range(initial, len(df)-step+1, step):
        tr=df.iloc[:start]; te=df.iloc[start:start+step]
        Xtr,ytr=clean_xy(tr[feats_num], tr[target])
        Xte,yte=clean_xy(te[feats_num], te[target])
        model.fit(Xtr,ytr)
        pred=model.predict(Xte)
        out.append(pd.DataFrame({date_col: te[date_col].values, "y_true": yte.values, "y_pred": pred}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=[date_col,"y_true","y_pred"])

def to_percent_series(s, assume_0_to_1=True):
    s = pd.to_numeric(s, errors="coerce")
    return s if assume_0_to_1 else s/100.0

def normalize_base_100(df, cols, base_idx=0):
    out = df.copy()
    for c in cols:
        try:
            base = float(out[c].iloc[base_idx])
            if base != 0:
                out[c] = out[c] / base * 100.0
        except Exception:
            pass
    return out

# ---------- Session ----------
for k,v in {"data":pd.DataFrame(),"date_col":"","target":"","features":[],"engineered":pd.DataFrame(),"freq":"MS","best_name":"","best_model":None}.items():
    if k not in st.session_state: st.session_state[k]=v

# ---------- Pages ----------
if PAGE.startswith("0"):
    st.title("📈 Pronósticos de Precios de Soya (ML) — UPSA | SolverTic")
    st.success("Cargue datos → EDA → Ingeniería → Modelos (TS-CV) → Walk-Forward → Pronóstico/Escenarios → Benchmark → Cobertura → Reporte")

elif PAGE.startswith("1"):
    st.header("📥 Cargar datos")
    up=st.file_uploader("CSV/XLSX con columna de fecha y precio objetivo", type=["csv","xlsx"])
    if up:
        df=load_table(up); st.session_state["data"]=df.copy()
        st.write(df.head())
        guess=parse_date(df) or ""
        date_col=st.selectbox("Columna de fecha", [""]+list(df.columns), index=(list(df.columns).index(guess)+1) if guess in df.columns else 0)
        target_options = [c for c in df.columns if not (date_col and c==date_col)]
        target=st.selectbox("Objetivo (precio numérico)", [""]+target_options)
        feats=st.multiselect("Features (opcionales)", [c for c in df.columns if c not in [date_col,target]])
        freq=st.selectbox("Frecuencia", ["D","W","MS","M","Q","YS"], index=2)

        if date_col:
            inv = invalid_date_rows(df, date_col)
            n_bad = len(inv)
            if n_bad > 0:
                st.warning(f"Se detectaron {n_bad} filas con fechas inválidas en '{date_col}'.")
                with st.expander("Ver filas con fecha inválida"):
                    st.dataframe(inv.head(300), use_container_width=True)
                    st.download_button("Descargar CSV con filas inválidas", inv.to_csv(index=False).encode("utf-8"), "fechas_invalidas.csv")
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("🧹 Filtrar dataset (excluir inválidas)"):
                        good, bad = filter_dataset_by_valid_dates(df, date_col)
                        st.session_state["data"] = good
                        st.session_state["last_clean_bad"] = bad
                        st.success(f"Se filtró el dataset. Filas válidas: {len(good)} · Removidas: {len(bad)}")
                        st.rerun()
                with c2:
                    if "last_clean_bad" in st.session_state and isinstance(st.session_state["last_clean_bad"], pd.DataFrame) and not st.session_state["last_clean_bad"].empty:
                        st.download_button("Descargar último descarte (CSV)", st.session_state["last_clean_bad"].to_csv(index=False).encode("utf-8"), "filas_descartadas.csv")
            if not st.session_state["data"].empty:
                st.download_button("⬇️ Descargar dataset actual (limpio)", st.session_state["data"].to_csv(index=False).encode("utf-8"), "dataset_limpio.csv")

        if st.button("Guardar"):
            if not date_col:
                st.error("Elige una columna de fecha válida.")
            elif not target:
                st.error("Elige una columna objetivo (numérica).")
            elif pd.api.types.is_datetime64_any_dtype(df[target]):
                st.error(f"La columna objetivo '{target}' es de tipo fecha. Selecciona una columna numérica.")
            else:
                st.session_state.update({"date_col":date_col,"target":target,"features":feats,"freq":freq})
                st.success("Configuración guardada.")

elif PAGE.startswith("2"):
    st.header("🔎 EDA")
    df=st.session_state["data"]
    if df.empty: st.info("Carga datos primero.")
    else:
        template=st.selectbox("Plantilla", ["plotly_white","plotly","ggplot2","seaborn","simple_white","plotly_dark","presentation"], index=0)
        st.write("Dimensiones:", df.shape); st.write("Tipos:", df.dtypes)
        with st.expander("Estadísticos"): st.write(df.describe(include="all"))
        x=st.selectbox("Eje X", list(df.columns))
        ys=st.multiselect("Series a graficar (Y)", [c for c in df.columns if c != x], default=[c for c in df.columns if c != x][:1])

        prof = st.session_state.get("units_profile", {"percent_cols": [], "percent_0_100": True, "currency_cols": [], "index_cols": []})
        is_percent_cols = st.multiselect("Tratar como % (perfil)", options=ys, default=[c for c in ys if c in prof.get("percent_cols", [])])
        perc_0_to_100 = st.checkbox("Valores 0–100 (no 0–1)", value=prof.get("percent_0_100", True))
        idx_defaults = [c for c in ys if c in prof.get("index_cols", [])]
        do_norm = st.checkbox("Normalizar a base=100 (para columnas marcadas como índice)", value=len(idx_defaults)>0)
        base_idx = st.number_input("Índice base", value=0, min_value=0, step=1)
        y_log = st.checkbox("Escala log Y", value=False)
        facet = st.checkbox("Pequeños múltiples (facet por variable)", value=False)
        auto_dual = st.checkbox("Eje secundario automático", value=True)

        if x:
            inv2 = invalid_date_rows(df, x)
            if len(inv2) > 0:
                st.caption(f"<span class='badge'>Diagnóstico</span> {len(inv2)} filas con fecha inválida en '{x}'.", unsafe_allow_html=True)
                with st.expander("Ver/descargar inválidas (EDA)"):
                    st.dataframe(inv2.head(300), use_container_width=True)
                    st.download_button("Descargar CSV inválidas (EDA)", inv2.to_csv(index=False).encode("utf-8"), "fechas_invalidas_EDA.csv")
                if st.button("🧹 Filtrar dataset ahora (EDA)"):
                    good, bad = filter_dataset_by_valid_dates(df, x)
                    st.session_state["data"] = good
                    st.session_state["last_clean_bad"] = bad
                    st.success(f"Dataset filtrado desde EDA. Filas válidas: {len(good)} · Removidas: {len(bad)}")
                    st.rerun()

        dplot = df.copy()
        for col in ys:
            try:
                dplot[col] = pd.to_numeric(dplot[col], errors="coerce")
            except Exception:
                pass

        idx_cols = [c for c in ys if c in prof.get("index_cols", [])]
        if do_norm and len(idx_cols)>0 and base_idx < len(dplot):
            dplot = normalize_base_100(dplot, idx_cols, base_idx=base_idx)

        on_secondary = set()
        if auto_dual:
            if any(c in prof.get("currency_cols", []) for c in ys) and any(c in prof.get("percent_cols", []) for c in ys):
                on_secondary.update([c for c in ys if c in prof.get("percent_cols", [])])

        try:
            if facet and len(ys) > 1:
                long = dplot[[x]+ys].melt(id_vars=[x], var_name="variable", value_name="valor")
                fig = px.line(long, x=x, y="valor", facet_col="variable", facet_col_wrap=2, template=template)
                if y_log:
                    fig.update_yaxes(type="log")
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = go.Figure()
                for i, ycol in enumerate(ys):
                    yser = dplot[ycol]
                    if ycol in is_percent_cols:
                        yser = to_percent_series(yser, assume_0_to_1=not perc_0_to_100)
                    sec = (ycol in on_secondary) and auto_dual
                    fig.add_trace(go.Scatter(x=dplot[x], y=yser, name=ycol, yaxis="y2" if sec else "y1"))
                fig.update_layout(template=template)
                if y_log:
                    fig.update_yaxes(type="log")
                if len(is_percent_cols)>0:
                    fig.update_yaxes(tickformat=".0%")
                if auto_dual and len(on_secondary)>0:
                    fig.update_layout(
                        yaxis=dict(title="Nivel"),
                        yaxis2=dict(title="%", overlaying="y", side="right", showgrid=False)
                    )
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"No se pudo graficar: {e}")

        dc, tg = st.session_state["date_col"], st.session_state["target"]
        if dc and tg and dc in df and tg in df and not pd.api.types.is_datetime64_any_dtype(df[tg]):
            s = robust_to_datetime(df.copy(), dc); s = s.sort_values(dc).set_index(dc)
            lam=st.number_input("λ HP (mensual≈129,600)", value=129600, step=1000)
            try:
                cycle,trend=hpfilter(s[tg], lamb=lam); st.plotly_chart(px.line(pd.DataFrame({"trend":trend,"cycle":cycle}), template=template, title="HP: tendencia & ciclo"), use_container_width=True)
            except Exception as e: st.info(f"HP no disponible: {e}")
            try:
                stl=STL(s[tg], period=12).fit()
                st.plotly_chart(px.line(pd.DataFrame({"observed":s[tg],"trend":stl.trend,"seasonal":stl.seasonal,"resid":stl.resid}), template=template, title="STL"), use_container_width=True)
            except Exception as e: st.info(f"STL no disponible: {e}")
        elif dc and tg and tg in df and pd.api.types.is_datetime64_any_dtype(df[tg]):
            st.warning(f"El objetivo '{tg}' es fecha: los descomponedores (HP/STL) requieren serie numérica.")

elif PAGE.startswith("3"):
    st.header("🧪 Ingeniería")
    df=st.session_state["data"]; dc=st.session_state["date_col"]; tg=st.session_state["target"]
    if df.empty or not dc or not tg: st.info("Define fecha/objetivo en 'Cargar datos'.")
    else:
        inv3 = invalid_date_rows(df, dc)
        if len(inv3) > 0:
            st.caption(f"<span class='badge'>Diagnóstico</span> {len(inv3)} filas con fecha inválida en '{dc}'.", unsafe_allow_html=True)
            with st.expander("Ver/descargar inválidas (Ingeniería)"):
                st.dataframe(inv3.head(300), use_container_width=True)
                st.download_button("Descargar CSV inválidas (Ingeniería)", inv3.to_csv(index=False).encode("utf-8"), "fechas_invalidas_ingenieria.csv")
            if st.button("🧹 Filtrar dataset ahora (Ingeniería)"):
                good, bad = filter_dataset_by_valid_dates(df, dc)
                st.session_state["data"] = good
                st.session_state["last_clean_bad"] = bad
                st.success(f"Dataset filtrado desde Ingeniería. Filas válidas: {len(good)} · Removidas: {len(bad)}")
                st.rerun()

        lags=st.multiselect("Lags", [1,2,3,6,9,12,18,24], default=[1,2,3,6,12])
        rolls=st.multiselect("Ventanas móviles", [3,6,12,24], default=[3,6,12])
        try:
            d=robust_to_datetime(df.copy(), dc); d=d.sort_values(dc); d=add_calendar(d,dc); d=add_lags_rolls(d,tg,lags,rolls).dropna()
            st.session_state["engineered"]=d.copy(); st.write(d.head())
            st.download_button("Descargar CSV ingenierizado", d.to_csv(index=False).encode("utf-8"), "datos_ingenierizados.csv")
        except TypeError as te:
            st.error(str(te))
        except Exception as e:
            st.error(f"No se pudo construir lags/rolling: {e}")

elif PAGE.startswith("4"):
    st.header("🤖 Modelos (TimeSeriesSplit)")
    df=st.session_state["engineered"]; dc=st.session_state["date_col"]; tg=st.session_state["target"]
    if df.empty: st.info("Realiza ingeniería primero.")
    else:
        if pd.api.types.is_datetime64_any_dtype(df[tg]):
            st.error(f"El objetivo '{tg}' es fecha. Selecciona un objetivo numérico en 'Cargar datos'.")
        else:
            feats_all=[c for c in df.columns if c not in [dc,tg] and pd.api.types.is_numeric_dtype(df[c])]
            feats=st.multiselect("Predictores", feats_all, default=[c for c in feats_all if ("lag" in c) or ("roll" in c) or (c in ["year","month","quarter","week"])])
            n=st.slider("Folds",3,8,5,1); template=st.selectbox("Plantilla", ["plotly_white","plotly","ggplot2","seaborn","simple_white","plotly_dark","presentation"], index=0)
            X,y=df[feats], df[tg]; ms=models()
            rows=[]
            for name,m in ms.items():
                maes,rmses,mapes=[],[],[]
                for tr,te in tscv(X,y,n):
                    Xtr,Ytr = clean_xy(X.iloc[tr], y.iloc[tr])
                    Xte,_ = clean_xy(X.iloc[te], y.iloc[te])
                    m.fit(Xtr,Ytr)
                    yp=m.predict(Xte)
                    met=metric_set(y.iloc[te].values, yp)
                    maes.append(met["MAE"]); rmses.append(met["RMSE"]); mapes.append(met["MAPE"])
                rows.append({"Modelo":name,"MAE":np.mean(maes),"RMSE":np.mean(rmses),"MAPE":np.mean(mapes)})
            res=pd.DataFrame(rows).sort_values("MAPE")
            st.dataframe(res, use_container_width=True); st.plotly_chart(px.bar(res,x="Modelo",y="MAPE",template=template,title="MAPE (TS-CV)"), use_container_width=True)
            best=res.iloc[0]["Modelo"]; st.session_state["best_name"]=best; st.session_state["best_model"]=ms[best].fit(*clean_xy(X, y))
            st.success(f"Mejor modelo: {best} (entrenado full data).")

elif PAGE.startswith("5"):
    st.header("🚶 Walk-Forward")
    df=st.session_state["engineered"]; dc=st.session_state["date_col"]; tg=st.session_state["target"]; best=st.session_state["best_model"]
    if df.empty or best is None: st.info("Entrena modelos en la sección 4.")
    else:
        feats=[c for c in df.columns if c not in [dc,tg] and pd.api.types.is_numeric_dtype(df[c])]
        initial=st.number_input("Tamaño inicial (meses)", value=max(36, int(len(df)*0.6)), min_value=12, step=1)
        step=st.number_input("Paso (meses)", value=1, min_value=1, step=1)
        template=st.selectbox("Plantilla", ["plotly_white","plotly","ggplot2","seaborn","simple_white","plotly_dark","presentation"], index=0)
        wf=walk_forward(df[[dc,tg]+feats], dc, tg, feats, best, int(initial), int(step))
        st.write(wf.tail()); met=metric_set(wf["y_true"].values, wf["y_pred"].values)
        st.info(f"Walk-Forward → MAPE={met['MAPE']:.2f}% · RMSE={met['RMSE']:.2f} · MAE={met['MAE']:.2f} · R²={met['R2']:.3f}")
        fig=go.Figure(); fig.add_trace(go.Scatter(x=wf[dc], y=wf["y_true"], name="Real")); fig.add_trace(go.Scatter(x=wf[dc], y=wf["y_pred"], name="Pred"))
        fig.update_layout(template=template, title="Real vs Pred (Walk-Forward)"); st.plotly_chart(fig, use_container_width=True)

elif PAGE.startswith("6"):
    st.header("🔮 Pronósticos / Escenarios")
    df=st.session_state["engineered"]; dc=st.session_state["date_col"]; tg=st.session_state["target"]; best=st.session_state["best_model"]; freq=st.session_state["freq"]
    if df.empty or best is None: st.info("Entrena el mejor modelo (sección 4).")
    else:
        h=st.slider("Horizonte",1,36,12,1); feats=[c for c in df.columns if c not in [dc,tg] and pd.api.types.is_numeric_dtype(df[c])]
        base=df[feats].iloc[[-1]].copy(); futureX=pd.concat([base]*h, ignore_index=True)
        st.subheader("Ajustes (Δ %) sobre rolling"); editable=[c for c in feats if ("rollmean" in c) or ("rollstd" in c)]
        for c in editable:
            dv=st.number_input(c, value=0.0, step=0.5, format="%.2f"); futureX[c]=futureX[c]*(1+dv/100.0)
        last=pd.to_datetime(df[dc]).max(); idx=pd.date_range(last, periods=h+1, freq=freq)[1:]
        yhat=best.predict(futureX); pred=pd.DataFrame({dc:idx,"forecast":yhat})
        template=st.selectbox("Plantilla", ["plotly_white","plotly","ggplot2","seaborn","simple_white","plotly_dark","presentation"], index=0)
        st.plotly_chart(px.line(pred, x=dc, y="forecast", template=template, title="Pronóstico (ML)"), use_container_width=True); st.dataframe(pred); st.download_button("Descargar pronóstico (CSV)", pred.to_csv(index=False).encode("utf-8"), "pronostico.csv")
        st.markdown("---"); st.info("SARIMAX (opcional)")
        p=st.number_input("p",0,5,1); d=st.number_input("d",0,2,1); q=st.number_input("q",0,5,1); P=st.number_input("P",0,5,1); D=st.number_input("D",0,2,1); Q=st.number_input("Q",0,5,1); s=st.number_input("s (mensual=12)",0,24,12)
        if st.button("Entrenar SARIMAX"):
            try:
                tmp=robust_to_datetime(df.copy(), dc); srs=tmp.sort_values(dc).set_index(tmp[dc])[tg].asfreq(freq).ffill()
                model=SARIMAX(srs, order=(p,d,q), seasonal_order=(P,D,Q,s), enforce_stationarity=False, enforce_invertibility=False)
                res=model.fit(disp=False); fc=res.get_forecast(steps=h); ci=fc.conf_int(); mean=fc.predicted_mean.reset_index(); mean.columns=[dc,"sarimax"]
                plot=mean.copy(); ci=ci.reset_index(drop=True); plot["lower"]=ci.iloc[:,0]; plot["upper"]=ci.iloc[:,1]
                fig2=go.Figure(); fig2.add_trace(go.Scatter(x=plot[dc], y=plot["sarimax"], name="SARIMAX")); fig2.add_trace(go.Scatter(x=plot[dc], y=plot["upper"], name="Upper", mode="lines")); fig2.add_trace(go.Scatter(x=plot[dc], y=plot["lower"], name="Lower", mode="lines", fill="tonexty"))
                fig2.update_layout(template=template, title="SARIMAX con bandas"); st.plotly_chart(fig2, use_container_width=True)
            except Exception as e: st.error(f"SARIMAX falló: {e}")

elif PAGE.startswith("7"):
    st.header("📊 Benchmark (trimestral)")
    bench_file=st.file_uploader("Benchmark (.csv/.xlsx) con columnas: trimestre (e.g., 2025Q3) y valor (USD/TM)", type=["csv","xlsx"])
    df=st.session_state["engineered"]; dc=st.session_state["date_col"]
    if bench_file and not df.empty and dc in df:
        bench=load_table(bench_file); fc_file=st.file_uploader("Pronóstico (CSV exportado)", type=["csv"])
        if fc_file:
            fc=pd.read_csv(fc_file); fc[dc]=pd.to_datetime(fc[dc]); fc["quarter"]=fc[dc].dt.to_period("Q")
            agg=fc.groupby("quarter")["forecast"].mean().reset_index(); agg.columns=["trimestre","forecast_modelo"]
            merged=pd.merge(agg, bench, left_on="trimestre", right_on=bench.columns[0], how="inner"); merged["diff"]=merged["forecast_modelo"]-merged.iloc[:,2]
            st.dataframe(merged, use_container_width=True); fig=go.Figure(); fig.add_trace(go.Bar(x=merged["trimestre"].astype(str), y=merged["forecast_modelo"], name="Modelo")); fig.add_trace(go.Bar(x=merged["trimestre"].astype(str), y=merged.iloc[:,2], name="Benchmark")); fig.update_layout(barmode="group")
            st.plotly_chart(fig, use_container_width=True); st.info(f"Diferencia promedio absoluta: {merged['diff'].abs().mean():.2f} USD/TM")
    else:
        st.info("Genera el pronóstico y súbelo para comparar.")

elif PAGE.startswith("8"):
    st.header("🛡️ Cobertura (collar modificado)")
    spot=st.number_input("Spot esperado (USD/TM)", value=380.0, step=1.0); k_put=st.number_input("Strike PUT", value=360.0, step=1.0); k_call=st.number_input("Strike CALL", value=375.0, step=1.0); premium_put=st.number_input("Prima PUT", value=4.41, step=0.1); premium_call=st.number_input("Prima CALL", value=19.84, step=0.1)
    template=st.selectbox("Plantilla", ["plotly_white","plotly","ggplot2","seaborn","simple_white","plotly_dark","presentation"], index=0)
    prices=np.linspace(spot*0.8, spot*1.2, 120); base=prices; put=np.maximum(k_put-prices,0)-premium_put; call=-np.maximum(prices-k_call,0)+premium_call; net=base+put+call
    fig=go.Figure(); fig.add_trace(go.Scatter(x=prices,y=base,name="Sin cobertura")); fig.add_trace(go.Scatter(x=prices,y=net,name="Collar modificado")); fig.update_layout(template=template, title="Payoff — 1 TM (ilustrativo)"); st.plotly_chart(fig, use_container_width=True)

elif PAGE.startswith("9"):
    st.header("📝 Reporte")
    now=datetime.now().strftime("%Y-%m-%d %H:%M")
    summary={"fecha":now,"org":st.session_state["brand"]["org"]}
    for key in ["best_name","freq","date_col","target","features"]:
        summary[key]=st.session_state.get(key,"")
    st.json(summary)
    comments=st.text_area("Comentarios (opcional)")
    md=f"""# Reporte — {summary['org']}
**Fecha:** {now}

## Configuración
- Mejor modelo: {summary.get('best_name','')}
- Frecuencia: {summary.get('freq','MS')}
- Columna de fecha: {summary.get('date_col','')}
- Objetivo: {summary.get('target','')}
- Features base: {', '.join(summary.get('features', []))}

## Comentarios
{comments}

> Generado con Soya ML — Streamlit.
"""
    st.download_button("Descargar reporte (.md)", md, file_name="reporte_soya_ml.md")
    if "engineered" in st.session_state and not st.session_state["engineered"].empty:
        st.download_button("Descargar datos ingenierizados (CSV)", st.session_state["engineered"].to_csv(index=False).encode("utf-8"), "datos_ingenierizados.csv")

elif PAGE.startswith("ℹ️"):
    st.header("ℹ️ Glosario")
    st.markdown("""
- **MAPE**: error porcentual absoluto medio.
- **RMSE / MAE**: errores en USD/TM; RMSE penaliza más grandes errores.
- **TS-CV (TimeSeriesSplit)** y **Walk-Forward**: validaciones que respetan el tiempo.
- **Boosting** (XGBoost/LightGBM/CatBoost) vs **RandomForest/GBM**.
- **Lags / Rolling**; **HP** y **STL**; **SARIMAX**.
- **Collar**: PUT larga + CALL corta.
""")

st.caption("© 2025 — SolverTic SRL · UPSA")
