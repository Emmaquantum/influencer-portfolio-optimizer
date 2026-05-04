"""
Influencer Portfolio Optimizer
Motor: Euler-Maruyama SDE + Markowitz Portfolio Theory
Arquitectura de parámetros:
  - γ, μ, σ  → derivados del influencer (histórico)
  - δ, κ     → campaña (sliders del manager)
  - plataforma, formato, timing → contexto (multiplicadores)
"""

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.optimize import minimize
import warnings

warnings.filterwarnings("ignore")

np.random.seed(42)


def plotly_html(fig: go.Figure, height: int = 500) -> None:
    html = fig.to_html(
        full_html=False, include_plotlyjs="cdn",
        config={"responsive": True, "displayModeBar": False},
    )
    components.html(html, height=height + 40, scrolling=False)


# ─────────────────────────────────────────────
# PÁGINA
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Influencer Portfolio Optimizer",
    page_icon="📊", layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>
[data-testid="stAppViewContainer"]{background:#0e1117}
[data-testid="stSidebar"]{background:#161b22;border-right:1px solid #30363d}
.metric-card{background:linear-gradient(135deg,#1c2128,#21262d);
    border:1px solid #30363d;border-radius:12px;padding:20px;text-align:center}
.metric-value{font-size:2.2rem;font-weight:700;color:#58a6ff;margin:0}
.metric-label{font-size:.85rem;color:#8b949e;margin-top:4px}
.metric-delta{font-size:.9rem;color:#3fb950}
.champion-card{background:linear-gradient(135deg,#1c2128,#21262d);
    border:1px solid #30363d;border-radius:10px;padding:14px 16px;margin-bottom:10px}
.champion-cat{font-size:.75rem;color:#8b949e;text-transform:uppercase;letter-spacing:1px}
.champion-name{font-size:1.1rem;font-weight:700;color:#c9d1d9;margin:4px 0 2px}
.champion-score{font-size:1.6rem;font-weight:700;color:#3fb950}
.champion-meta{font-size:.8rem;color:#8b949e;margin-top:4px}
.param-box{background:#161b22;border:1px solid #30363d;border-radius:8px;
    padding:12px 16px;margin-bottom:8px;font-size:.85rem;color:#c9d1d9}
.param-tag{display:inline-block;padding:2px 8px;border-radius:12px;font-size:.75rem;
    font-weight:600;margin-right:4px}
.tag-inf{background:#1f3a5c;color:#58a6ff}
.tag-camp{background:#1f3a1f;color:#3fb950}
.tag-ctx{background:#3a2a1f;color:#d29922}
</style>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PERFILES: 45 INFLUENCERS
# handle: (grupo, tier, followers, eng_rate, costo_COP, categoria)
# ─────────────────────────────────────────────
INFLUENCER_PROFILES = {
    "@alebananac":           (1,"Micro",  85_000, 0.048, 1_800_000,"Gastronomía"),
    "@amadorat":             (1,"Micro",  62_000, 0.052, 1_400_000,"Educación & Negocios"),
    "@amirodriguezz":        (1,"Macro", 420_000, 0.031, 5_500_000,"Entretenimiento"),
    "@camilaesguerra":       (1,"Micro",  95_000, 0.045, 2_000_000,"Lifestyle & Viajes"),
    "@camilomartinezbotero": (1,"Macro", 310_000, 0.028, 4_200_000,"Educación & Negocios"),
    "@danielagaravitoo":     (1,"Micro",  78_000, 0.051, 1_700_000,"Cultura & Arte"),
    "@dr.negrete":           (1,"Macro", 280_000, 0.034, 3_800_000,"Salud & Bienestar"),
    "@duvanjimenez":         (1,"Micro", 110_000, 0.042, 2_100_000,"Entretenimiento"),
    "@erikaenfrancia":       (2,"Micro",  55_000, 0.061, 1_200_000,"Lifestyle & Viajes"),
    "@evapartis":            (2,"Micro",  88_000, 0.047, 1_900_000,"Cultura & Arte"),
    "@gabrielaruedarueda":   (2,"Nano",   38_000, 0.072,   800_000,"Educación & Negocios"),
    "@gfermg":               (2,"Micro",  72_000, 0.053, 1_550_000,"Tech & Gaming"),
    "@itsjacobots":          (2,"Nano",   29_000, 0.081,   650_000,"Tech & Gaming"),
    "@kimdesutter":          (2,"Micro",  93_000, 0.044, 1_980_000,"Belleza & Moda"),
    "@lagranjadelborrego":   (2,"Micro",  67_000, 0.058, 1_450_000,"Gastronomía"),
    "@lamamadelasrecetas":   (2,"Macro", 350_000, 0.027, 4_800_000,"Gastronomía"),
    "@leicysantos10":        (3,"Macro", 890_000, 0.022, 9_500_000,"Salud & Bienestar"),
    "@losdenam":             (3,"Micro", 105_000, 0.041, 2_200_000,"Entretenimiento"),
    "@makisdeangulo":        (3,"Micro",  76_000, 0.055, 1_650_000,"Gastronomía"),
    "@___mange":             (3,"Nano",   33_000, 0.076,   720_000,"Entretenimiento"),
    "@marcelarestrepo10":    (3,"Micro",  82_000, 0.049, 1_750_000,"Belleza & Moda"),
    "@mariaelviraram":       (3,"Macro", 260_000, 0.033, 3_600_000,"Cultura & Arte"),
    "@medicennmar":          (3,"Micro",  58_000, 0.059, 1_250_000,"Salud & Bienestar"),
    "@megumihasebep":        (3,"Nano",   41_000, 0.068,   880_000,"Cultura & Arte"),
    "@nelagonzalez":         (4,"Macro", 540_000, 0.026, 6_800_000,"Entretenimiento"),
    "@olgaluciavives":       (4,"Macro", 390_000, 0.029, 5_200_000,"Cultura & Arte"),
    "@pautips":              (4,"Mega",4_200_000, 0.018,35_000_000,"Belleza & Moda"),
    "@raquelcarpe":          (4,"Micro",  99_000, 0.043, 2_100_000,"Lifestyle & Viajes"),
    "@raqueldebedout":       (4,"Macro", 470_000, 0.025, 6_100_000,"Lifestyle & Viajes"),
    "@sofiacastro":          (4,"Mega",2_800_000, 0.019,22_000_000,"Entretenimiento"),
    "@soybans":              (4,"Micro",  86_000, 0.046, 1_850_000,"Salud & Bienestar"),
    "@stephmoreno":          (4,"Micro",  74_000, 0.054, 1_600_000,"Belleza & Moda"),
    "@valelizcano":          (5,"Macro", 620_000, 0.024, 7_900_000,"Lifestyle & Viajes"),
    "@victoriadallier":      (5,"Micro",  91_000, 0.044, 1_950_000,"Belleza & Moda"),
    "@zamirvillamil":        (5,"Micro",  68_000, 0.057, 1_480_000,"Entretenimiento"),
    "@laurabarjum":          (5,"Macro", 330_000, 0.030, 4_500_000,"Lifestyle & Viajes"),
    "@yurivargas1":          (5,"Micro",  80_000, 0.050, 1_720_000,"Salud & Bienestar"),
    "@majovargas1":          (5,"Nano",   45_000, 0.065,   960_000,"Lifestyle & Viajes"),
    "@emilyrococo":          (5,"Micro",  63_000, 0.060, 1_350_000,"Cultura & Arte"),
    "@lolavargas":           (5,"Nano",   36_000, 0.073,   780_000,"Lifestyle & Viajes"),
    "@alexlarra":            (6,"Micro",  97_000, 0.043, 2_050_000,"Entretenimiento"),
    "@monalimentos":         (6,"Micro", 115_000, 0.039, 2_400_000,"Gastronomía"),
    "@tatacalde7":           (6,"Nano",   42_000, 0.067,   900_000,"Educación & Negocios"),
    "@pongamoslo_a_prueba":  (6,"Nano",   31_000, 0.079,   680_000,"Tech & Gaming"),
    "@dipaksalazar":         (6,"Micro",  71_000, 0.055, 1_530_000,"Tech & Gaming"),
}

TIER_COLORS = {"Nano":"#f85149","Micro":"#d29922","Macro":"#58a6ff","Mega":"#bc8cff"}
CATEGORY_ICONS = {
    "Gastronomía":"🍽️","Salud & Bienestar":"💪","Belleza & Moda":"✨",
    "Lifestyle & Viajes":"🌍","Entretenimiento":"🎭","Tech & Gaming":"🎮",
    "Educación & Negocios":"📚","Cultura & Arte":"🎨",
}
CATEGORY_COLORS = {
    "Gastronomía":"#e67e22","Salud & Bienestar":"#27ae60","Belleza & Moda":"#e91e8c",
    "Lifestyle & Viajes":"#3498db","Entretenimiento":"#9b59b6","Tech & Gaming":"#1abc9c",
    "Educación & Negocios":"#f39c12","Cultura & Arte":"#e74c3c",
}

# ─────────────────────────────────────────────
# MULTIPLICADORES DE CONTEXTO
# ─────────────────────────────────────────────

# Efecto en: mu (alcance drift), sigma (volatilidad), delta (conversión base)
PLATFORM_MULT = {
    "Instagram":{"mu":1.00,"sigma":1.00,"delta":1.00},
    "TikTok":   {"mu":1.45,"sigma":1.60,"delta":0.75},
    "YouTube":  {"mu":0.65,"sigma":0.65,"delta":1.70},
}
FORMAT_MULT = {
    "Reel / Short":{"mu":1.15,"sigma":1.35,"delta":0.75},
    "Story":       {"mu":0.85,"sigma":0.80,"delta":1.25},
    "Post / Foto": {"mu":1.00,"sigma":1.00,"delta":1.00},
    "Video Largo": {"mu":0.70,"sigma":0.60,"delta":1.85},
}
TIMING_MULT = {
    "Normal":          {"mu":1.00,"sigma":1.00,"delta":1.00},
    "Alta temporada":  {"mu":1.25,"sigma":1.15,"delta":1.45},
    "Tendencia viral": {"mu":1.65,"sigma":2.10,"delta":0.85},
    "Baja temporada":  {"mu":0.80,"sigma":0.90,"delta":0.65},
}

# κ fit marca-audiencia: campaign_categoria → influencer_categoria → multiplicador
KAPPA_FIT = {
    "Gastronomía":         {"Gastronomía":1.00,"Salud & Bienestar":0.75,"Lifestyle & Viajes":0.65,"Cultura & Arte":0.50,"Belleza & Moda":0.45,"Entretenimiento":0.40,"Tech & Gaming":0.25,"Educación & Negocios":0.35},
    "Salud & Bienestar":   {"Salud & Bienestar":1.00,"Gastronomía":0.75,"Lifestyle & Viajes":0.70,"Educación & Negocios":0.60,"Belleza & Moda":0.55,"Cultura & Arte":0.40,"Entretenimiento":0.35,"Tech & Gaming":0.25},
    "Belleza & Moda":      {"Belleza & Moda":1.00,"Lifestyle & Viajes":0.80,"Cultura & Arte":0.65,"Entretenimiento":0.60,"Salud & Bienestar":0.50,"Gastronomía":0.45,"Educación & Negocios":0.35,"Tech & Gaming":0.25},
    "Lifestyle & Viajes":  {"Lifestyle & Viajes":1.00,"Belleza & Moda":0.80,"Entretenimiento":0.75,"Cultura & Arte":0.65,"Gastronomía":0.60,"Salud & Bienestar":0.55,"Educación & Negocios":0.45,"Tech & Gaming":0.35},
    "Entretenimiento":     {"Entretenimiento":1.00,"Lifestyle & Viajes":0.75,"Cultura & Arte":0.75,"Belleza & Moda":0.60,"Gastronomía":0.50,"Tech & Gaming":0.55,"Educación & Negocios":0.45,"Salud & Bienestar":0.40},
    "Tech & Gaming":       {"Tech & Gaming":1.00,"Educación & Negocios":0.75,"Entretenimiento":0.60,"Lifestyle & Viajes":0.45,"Cultura & Arte":0.40,"Salud & Bienestar":0.35,"Belleza & Moda":0.30,"Gastronomía":0.25},
    "Educación & Negocios":{"Educación & Negocios":1.00,"Tech & Gaming":0.75,"Salud & Bienestar":0.65,"Cultura & Arte":0.55,"Entretenimiento":0.50,"Lifestyle & Viajes":0.50,"Gastronomía":0.40,"Belleza & Moda":0.35},
    "Cultura & Arte":      {"Cultura & Arte":1.00,"Entretenimiento":0.80,"Lifestyle & Viajes":0.70,"Educación & Negocios":0.60,"Belleza & Moda":0.55,"Gastronomía":0.50,"Salud & Bienestar":0.40,"Tech & Gaming":0.30},
}

# μ y σ base según tier (derivados de histórico, no sliders)
TIER_MU_BASE    = {"Nano":0.012,"Micro":0.009,"Macro":0.006,"Mega":0.003}
TIER_SIGMA_BASE = {"Nano":0.22, "Micro":0.16, "Macro":0.10, "Mega":0.06}

# ─────────────────────────────────────────────
# DATOS
# ─────────────────────────────────────────────

def build_demo_dataframe() -> pd.DataFrame:
    """
    μ y σ derivados del tier + ruido histórico (±12%).
    γ = eng_rate directamente (dato observado del influencer).
    """
    rng = np.random.default_rng(42)
    rows = []
    for handle, (grupo, tier, followers, er, costo, categoria) in INFLUENCER_PROFILES.items():
        noise = rng.uniform(0.88, 1.12)
        rows.append({
            "handle":    handle,
            "grupo":     grupo,
            "tier":      tier,
            "categoria": categoria,
            "followers": int(followers * rng.uniform(0.90, 1.10)),
            "eng_rate":  float(np.clip(er * rng.uniform(0.88, 1.12), 0.01, 0.25)),
            "cost_cop":  int(costo * rng.uniform(0.92, 1.08)),
            # μ y σ base desde tier — el contexto los reescala en la simulación
            "mu_base":   round(TIER_MU_BASE[tier] * noise, 5),
            "sigma_base":round(TIER_SIGMA_BASE[tier] * rng.uniform(0.85, 1.15), 4),
        })
    return pd.DataFrame(rows).set_index("handle")


def parse_uploaded_csv(uploaded_file) -> pd.DataFrame:
    df_raw = pd.read_csv(uploaded_file)
    df_raw.columns = df_raw.columns.str.strip().str.lower()
    required = {"handle","followers","eng_rate","cost_cop"}
    missing  = required - set(df_raw.columns)
    if missing:
        st.error(f"Columnas faltantes: {missing}")
        st.stop()
    df_raw = df_raw.set_index("handle")
    if "grupo" not in df_raw.columns:
        df_raw["grupo"] = df_raw.index.map(lambda h: INFLUENCER_PROFILES.get(h,(1,))[0])
    if "tier" not in df_raw.columns:
        df_raw["tier"]  = df_raw.index.map(lambda h: INFLUENCER_PROFILES.get(h,(1,"Micro"))[1])
    if "categoria" not in df_raw.columns:
        df_raw["categoria"] = df_raw.index.map(
            lambda h: INFLUENCER_PROFILES.get(h,(1,"Micro",0,0,0,"Entretenimiento"))[5])
    df_raw["mu_base"]    = df_raw["tier"].map(TIER_MU_BASE).fillna(0.009)
    df_raw["sigma_base"] = df_raw["tier"].map(TIER_SIGMA_BASE).fillna(0.16)
    return df_raw


# ─────────────────────────────────────────────
# SCORING COMPUESTO
# ─────────────────────────────────────────────

def compute_scores(df: pd.DataFrame, sim_results: dict) -> pd.DataFrame:
    s = df.copy()
    s["roi_mean"]   = [sim_results[h]["roi_mean"]   for h in df.index]
    s["roi_std"]    = [sim_results[h]["roi_std"]     for h in df.index]
    s["reach_mean"] = [sim_results[h]["reach_mean"]  for h in df.index]
    s["kappa_eff"]  = [sim_results[h]["kappa_eff"]   for h in df.index]
    s["cost_eff"]   = s["roi_mean"] / (s["cost_cop"] / 1_000_000)

    def mm(x: pd.Series) -> pd.Series:
        r = x.max() - x.min()
        return (x - x.min()) / r if r > 1e-12 else pd.Series(0.5, index=x.index)

    s["score"] = (
        0.40 * mm(s["roi_mean"]) +
        0.35 * mm(s["eng_rate"]) +
        0.25 * mm(s["cost_eff"])
    ) * 100

    s["rank_cat"] = (
        s.groupby("categoria")["score"]
        .rank(ascending=False, method="min").astype(int)
    )
    return s


# ─────────────────────────────────────────────
# MOTOR ESTOCÁSTICO: EULER-MARUYAMA
# Parámetros efectivos por influencer:
#   mu_eff    = mu_base  * plat_mu  * fmt_mu  * timing_mu
#   sigma_eff = sigma_base * plat_sg * fmt_sg * timing_sg
#   gamma     = eng_rate  (dato del influencer, sin slider)
#   delta_eff = delta_camp * plat_d  * fmt_d  * timing_d
#   kappa_eff = kappa_camp * fit[campaign_cat][influencer_cat]
# ─────────────────────────────────────────────

def euler_maruyama_simulation(
    df: pd.DataFrame,
    ticket: float,
    delta_camp: float,
    kappa_camp: float,
    campaign_cat: str,
    ctx_mu: float,
    ctx_sigma: float,
    ctx_delta: float,
    n_sim: int = 1_000,
    T: int = 30,
    eta: float = 0.5,
) -> dict:
    dt = 1 / T
    results = {}
    for handle, row in df.iterrows():
        mu_eff    = float(row["mu_base"])    * ctx_mu
        sigma_eff = float(row["sigma_base"]) * ctx_sigma
        gamma_eff = float(row["eng_rate"])   # directo del influencer
        delta_eff = delta_camp               * ctx_delta
        kappa_eff = kappa_camp * KAPPA_FIT[campaign_cat].get(row["categoria"], 0.40)
        cost      = float(row["cost_cop"])

        X = np.zeros((n_sim, T + 1))
        C = np.zeros((n_sim, T + 1))
        X[:, 0] = float(row["followers"])

        for t in range(T):
            dW        = np.random.randn(n_sim) * np.sqrt(dt)
            X[:,t+1]  = np.maximum(X[:,t] * (1 + mu_eff * dt + sigma_eff * dW), 0)
            dW_c      = np.random.randn(n_sim) * np.sqrt(dt)
            E_t       = gamma_eff * X[:,t]
            C[:,t+1]  = np.maximum(C[:,t] + delta_eff * E_t * kappa_eff * dt + eta * dW_c, 0)

        roi = (C[:,-1] * ticket - cost) / cost
        results[handle] = {
            "X":          X,
            "C":          C,
            "roi_dist":   roi,
            "roi_mean":   float(roi.mean()),
            "roi_std":    float(roi.std()),
            "reach_mean": float(X[:,-1].mean()),
            "mu_eff":     round(mu_eff, 5),
            "sigma_eff":  round(sigma_eff, 4),
            "gamma_eff":  round(gamma_eff, 4),
            "delta_eff":  round(delta_eff, 4),
            "kappa_eff":  round(kappa_eff, 3),
        }
    return results


def build_covariance_matrix(df, sim_results):
    handles = list(df.index)
    n       = len(handles)
    grupos  = df["grupo"].values
    stds    = np.array([sim_results[h]["roi_std"] for h in handles])
    corr    = np.full((n, n), 0.15)
    for i in range(n):
        for j in range(n):
            if i == j:        corr[i,j] = 1.0
            elif grupos[i] == grupos[j]: corr[i,j] = 0.70
    return corr * np.outer(stds, stds)


# ─────────────────────────────────────────────
# OPTIMIZACIÓN DE PORTAFOLIO (MARKOWITZ)
# ─────────────────────────────────────────────

def optimize_portfolio(df, sim_results, presupuesto, lambda_risk):
    handles = list(df.index)
    n       = len(handles)
    costs   = df["cost_cop"].values.astype(float)
    mu_vec  = np.array([sim_results[h]["roi_mean"] for h in handles])
    cov     = build_covariance_matrix(df, sim_results)

    def markowitz_utility(w):
        rp  = w @ mu_vec
        vp  = w @ cov @ w
        # Minimizamos el negativo de la utilidad para maximizar la utilidad real
        # U = Retorno - (Lambda * Varianza)
        return -(rp - (lambda_risk * vp))

    res   = minimize(markowitz_utility, np.ones(n)/n*0.05, method="SLSQP",
                     bounds=[(0.0, 0.40)]*n, # Aumentado a 40% para dejar subir la frontera
                     constraints=[{"type":"ineq","fun":lambda w: presupuesto - np.dot(w,costs)}],
                     options={"maxiter":2000,"ftol":1e-10})
    w     = np.maximum(res.x, 0)
    rp    = w @ mu_vec
    vp    = w @ cov @ w
    std   = np.sqrt(max(vp, 0))

    alloc_real = {}
    presupuesto_restante = presupuesto

    indices_prioridad = np.argsort(-w)

    for i in indices_prioridad:
        h = handles[i]
        costo_unitario = float(costs[i])
        
        # Filtro: Si Markowitz le dio un peso mayor al 1% y tenemos saldo
        if w[i] > 0.01 and presupuesto_restante >= costo_unitario:
            alloc_real[h] = costo_unitario
            presupuesto_restante -= costo_unitario

    w_real = np.zeros(n)
    for i, h in enumerate(handles):
        if h in alloc_real:
            w_real[i] = costs[i] / presupuesto # Peso sobre el capital total asignado

    rp_real  = w_real @ mu_vec
    vp_real  = w_real @ cov @ w_real
    std_real = np.sqrt(max(vp_real, 0))

    roi_p_real = sum(w_real[i] * sim_results[h]["roi_dist"] for i, h in enumerate(handles) if w_real[i] > 0)

    return {
        "weights":        dict(zip(handles, w_real)), # Ahora devuelve los pesos ejecutables
        "allocation_cop": alloc_real,                 # Solo contratos al 100%
        "roi_portfolio":  float(rp_real),
        "var_portfolio":  float(vp_real),
        "std_portfolio":  float(std_real),
        "sharpe":         float(rp_real/std_real) if std_real > 1e-12 else 0.0,
        "budget_used":    float(presupuesto - presupuesto_restante),
        "prob_exito":     float((roi_p_real > 0).mean()) if isinstance(roi_p_real, np.ndarray) else 0.0,
        "roi_dist":       roi_p_real,
        "success":        res.success,
    }


def efficient_frontier_points(df, sim_results, presupuesto, n_points=50):

    handles = list(df.index)
    n = len(handles)

    costs = df["cost_cop"].values.astype(float)
    mu = np.array([sim_results[h]["roi_mean"] for h in handles])
    cov = build_covariance_matrix(df, sim_results)

    # Rango de retornos objetivo
    target_returns = np.linspace(mu.min(), mu.max(), n_points)

    frontier = []

    for target in target_returns:

        def portfolio_variance(w):
            return w @ cov @ w

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w: w @ mu - target},
            {"type": "ineq", "fun": lambda w: presupuesto - np.dot(w, costs)},
        ]

        bounds = [(0, 0.2)] * n

        w0 = np.ones(n) / n

        res = minimize(
            portfolio_variance,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000}
        )

        if res.success:
            w = res.x
            rp = w @ mu
            std = np.sqrt(w @ cov @ w)

            frontier.append({
                "roi": rp,
                "std": std
            })

    return pd.DataFrame(frontier)


# ─────────────────────────────────────────────
# VISUALIZACIONES
# ─────────────────────────────────────────────

PLOTLY_THEME = dict(
    paper_bgcolor="#0e1117", plot_bgcolor="#161b22",
    font=dict(color="#c9d1d9",family="Inter,sans-serif"),
    xaxis=dict(gridcolor="#21262d",linecolor="#30363d"),
    yaxis=dict(gridcolor="#21262d",linecolor="#30363d"),
    legend=dict(bgcolor="#161b22",bordercolor="#30363d",borderwidth=1),
)


def plot_catalog_bubble(scores_df):
    fig = go.Figure()
    for cat, color in CATEGORY_COLORS.items():
        sub = scores_df[scores_df["categoria"] == cat]
        if sub.empty: continue
        fig.add_trace(go.Scatter(
            x=sub["eng_rate"]*100, y=sub["score"],
            mode="markers+text",
            marker=dict(size=np.clip(sub["followers"]/15_000,8,50),
                        color=color, opacity=0.82,
                        line=dict(width=1.2,color="#0e1117")),
            text=[h.replace("@","") for h in sub.index],
            textposition="top center",
            textfont=dict(size=9,color="#c9d1d9"),
            name=f"{CATEGORY_ICONS.get(cat,'')} {cat}",
            customdata=np.column_stack([
                sub["followers"].map("{:,}".format),
                sub["cost_cop"].map("${:,.0f}".format),
                sub["score"].round(1), sub["tier"],
                (sub["kappa_eff"]*100).round(0),
            ]),
            hovertemplate=(
                "<b>%{text}</b><br>Score: %{customdata[2]}<br>"
                "Eng: %{x:.2f}%<br>Followers: %{customdata[0]}<br>"
                "Costo: %{customdata[1]}<br>Tier: %{customdata[3]}<br>"
                "κ fit campaña: %{customdata[4]}%<extra></extra>"
            ),
        ))
    fig.update_layout(
        title="Mapa de Influencers — Score vs Engagement (tamaño = followers, color = categoría)",
        xaxis_title="Engagement Rate (%)", yaxis_title="Score Compuesto (0–100)",
        height=580, **PLOTLY_THEME,
    )
    return fig


def plot_params_heatmap(scores_df):
    """Heatmap de parámetros SDE efectivos por influencer."""
    sub = scores_df.sort_values("score", ascending=False).head(20)
    params_cols = ["mu_eff","sigma_eff","gamma_eff","delta_eff","kappa_eff"]
    # Normalizar 0-1 por columna para heatmap visual
    mat = sub[params_cols].copy()
    for c in params_cols:
        rng = mat[c].max() - mat[c].min()
        mat[c] = (mat[c] - mat[c].min()) / rng if rng > 0 else 0.5

    labels = ["μ (alcance drift)","σ (volatilidad)","γ (engagement)","δ (conversión)","κ (fit marca)"]
    fig = go.Figure(go.Heatmap(
        z=mat.values.T,
        x=[h.replace("@","") for h in sub.index],
        y=labels,
        colorscale="Blues",
        showscale=True,
        hovertemplate="<b>%{x}</b><br>%{y}: %{z:.3f}<extra></extra>",
        colorbar=dict(title="Valor norm.", tickfont=dict(color="#c9d1d9")),
    ))
    fig.update_layout(
        title="Parámetros SDE Efectivos — Top 20 por Score (normalizado)",
        height=320, **PLOTLY_THEME,
    )
    fig.update_xaxes(tickangle=-45, gridcolor="#21262d")
    return fig


def plot_category_ranking_bars(scores_df, categoria):
    sub = scores_df[scores_df["categoria"]==categoria].sort_values("score",ascending=True)
    colors = [
        "#f1c40f" if i==len(sub)-1 else
        "#bdc3c7" if i==len(sub)-2 else
        "#cd7f32" if i==len(sub)-3 else
        TIER_COLORS.get(t,"#58a6ff")
        for i,t in enumerate(sub["tier"])
    ]
    fig = go.Figure(go.Bar(
        x=sub["score"],
        y=[h.replace("@","") for h in sub.index],
        orientation="h",
        marker=dict(color=colors),
        text=[f"{s:.1f}" for s in sub["score"]],
        textposition="outside",
        customdata=np.column_stack([
            sub["followers"].map("{:,}".format),
            (sub["eng_rate"]*100).round(2),
            sub["cost_cop"].map("${:,.0f}".format),
            sub["tier"],
            (sub["kappa_eff"]*100).round(0),
        ]),
        hovertemplate=(
            "<b>@%{y}</b><br>Score: %{x:.1f}<br>"
            "Followers: %{customdata[0]}<br>Eng: %{customdata[1]}%<br>"
            "Costo: %{customdata[2]}<br>Tier: %{customdata[3]}<br>"
            "κ fit: %{customdata[4]}%<extra></extra>"
        ),
    ))
    fig.update_layout(
        title=f"{CATEGORY_ICONS.get(categoria,'')} Ranking — {categoria}",
        xaxis_title="Score", xaxis_range=[0,115],
        height=max(280, 40*len(sub)), **PLOTLY_THEME,
    )
    return fig


def plot_efficient_frontier(df, sim_results, frontier_df, opt_result, scores_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frontier_df["std"], y=frontier_df["roi"],
        mode="lines", line=dict(color="#58a6ff",width=2.5),
        name="Frontera Eficiente",
    ))
    for cat, color in CATEGORY_COLORS.items():
        sub = scores_df[scores_df["categoria"]==cat]
        fig.add_trace(go.Scatter(
            x=[sim_results[h]["roi_std"]  for h in sub.index],
            y=[sim_results[h]["roi_mean"] for h in sub.index],
            mode="markers",
            marker=dict(size=10,color=color,opacity=0.80,
                        line=dict(width=1,color="#0e1117")),
            name=f"{CATEGORY_ICONS.get(cat,'')} {cat}",
            text=list(sub.index),
            customdata=[(sim_results[h]["kappa_eff"]*100) for h in sub.index],
            hovertemplate=(
                "<b>%{text}</b><br>Riesgo σ: %{x:.3f}<br>"
                "ROI: %{y:.3f}<br>κ fit: %{customdata:.0f}%<extra></extra>"
            ),
        ))
    fig.add_trace(go.Scatter(
        x=[opt_result["std_portfolio"]], y=[opt_result["roi_portfolio"]],
        mode="markers",
        marker=dict(size=20,color="#3fb950",symbol="star",
                    line=dict(width=2,color="#ffffff")),
        name="Portafolio Óptimo",
    ))
    fig.update_layout(
        title="Frontera Eficiente — Riesgo vs Retorno (hover = κ fit campaña)",
        xaxis_title="Riesgo (σ ROI)", yaxis_title="ROI Esperado",
        height=500, **PLOTLY_THEME,
    )
    return fig


def plot_allocation(opt_result, scores_df):
    alloc = opt_result["allocation_cop"]
    if not alloc: return go.Figure()
    adf = (
        pd.DataFrame.from_dict(alloc,orient="index",columns=["COP"])
        .join(scores_df[["tier","categoria","score","kappa_eff"]])
        .sort_values("COP",ascending=True)
    )
    fig = go.Figure(go.Bar(
        x=adf["COP"]/1_000_000,
        y=[h.replace("@","") for h in adf.index],
        orientation="h",
        marker=dict(color=adf["categoria"].map(CATEGORY_COLORS).fillna("#8b949e")),
        text=[f"${v/1e6:.1f}M" for v in adf["COP"]],
        textposition="outside",
        customdata=np.column_stack([adf["categoria"],(adf["kappa_eff"]*100).round(0)]),
        hovertemplate=(
            "<b>@%{y}</b><br>$%{x:.1f}M COP<br>"
            "%{customdata[0]}<br>κ fit: %{customdata[1]}%<extra></extra>"
        ),
    ))
    fig.update_layout(
        title="Asset Allocation — Presupuesto (color = categoría · hover = κ fit)",
        xaxis_title="Asignación (M COP)",
        height=max(350,36*len(adf)), **PLOTLY_THEME,
    )
    return fig


def plot_stochastic_projection(sim_results, opt_result, T=30, n_paths=25):
    alloc = opt_result["allocation_cop"]
    if not alloc: return go.Figure()
    top5  = sorted(alloc,key=lambda h:alloc[h],reverse=True)[:5]
    fig   = make_subplots(rows=1,cols=2,
                          subplot_titles=["Alcance (miles)","Conversiones"])
    t_ax  = np.linspace(0,30,T+1)
    for i,h in enumerate(top5):
        c   = px.colors.qualitative.Plotly[i%10]
        X,C = sim_results[h]["X"], sim_results[h]["C"]
        idx = np.random.choice(X.shape[0],min(n_paths,X.shape[0]),replace=False)
        for j,pi in enumerate(idx):
            fig.add_trace(go.Scatter(x=t_ax,y=X[pi]/1000,mode="lines",
                line=dict(color=c,width=0.7),opacity=0.25,
                name=h,legendgroup=h,showlegend=(j==0)),row=1,col=1)
            fig.add_trace(go.Scatter(x=t_ax,y=C[pi],mode="lines",
                line=dict(color=c,width=0.7),opacity=0.25,
                name=h,legendgroup=h,showlegend=False),row=1,col=2)
        fig.add_trace(go.Scatter(x=t_ax,y=X.mean(axis=0)/1000,mode="lines",
            line=dict(color=c,width=2.5),name=h,legendgroup=h,showlegend=False),row=1,col=1)
        fig.add_trace(go.Scatter(x=t_ax,y=C.mean(axis=0),mode="lines",
            line=dict(color=c,width=2.5),name=h,legendgroup=h,showlegend=False),row=1,col=2)
    fig.update_layout(title="Proyección Estocástica — Top 5 del Portafolio",
                      height=420,**PLOTLY_THEME)
    fig.update_xaxes(title_text="Días",gridcolor="#21262d",linecolor="#30363d")
    fig.update_yaxes(gridcolor="#21262d",linecolor="#30363d")
    return fig


def plot_roi_dist(opt_result):
    rd  = opt_result["roi_dist"]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=rd,nbinsx=60,
        marker=dict(color="#58a6ff",line=dict(color="#0e1117",width=0.3)),
        opacity=0.85,name="ROI"))
    m = float(rd.mean())
    fig.add_vline(x=0,line=dict(color="#f85149",width=2,dash="dash"),
                  annotation_text="Break-even",annotation_font_color="#f85149")
    fig.add_vline(x=m,line=dict(color="#3fb950",width=2),
                  annotation_text=f"μ={m:.2f}",annotation_font_color="#3fb950")
    fig.update_layout(title="Distribución ROI Portafolio (Monte Carlo)",
                      xaxis_title="ROI",yaxis_title="Frecuencia",
                      height=340,**PLOTLY_THEME)
    return fig


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

def render_sidebar():
    cats = sorted(CATEGORY_COLORS.keys())

    # ── CAMPAÑA ──
    st.sidebar.markdown("## 📣 Campaña")
    presupuesto  = st.sidebar.slider("Presupuesto (M COP)",5,500,80,5) * 1_000_000
    ticket       = st.sidebar.slider("Ticket Promedio (COP)",50_000,2_000_000,280_000,10_000,format="$%d")
    campaign_cat = st.sidebar.selectbox(
        "Categoría de la campaña",
        options=cats,
        format_func=lambda c: f"{CATEGORY_ICONS.get(c,'')} {c}",
        help="Determina el fit κ de cada influencer automáticamente.",
    )

    # ── PARÁMETROS DE CAMPAÑA ──
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🎯 Parámetros de Campaña")
    st.sidebar.markdown(
        "<span class='param-tag tag-camp'>Campaña</span> "
        "Controla el manager — cambian por brief/producto",
        unsafe_allow_html=True,
    )
    delta_camp = st.sidebar.slider(
        "δ — Tasa de conversión base (%)", 0.1, 8.0, 2.0, 0.1,
        help="% de engagements que se convierten en acción (compra/registro). "
             "Depende del CTA, landing page y producto.",
    ) / 100

    kappa_camp = st.sidebar.slider(
        "κ base — Calidad del brief (0–1)", 0.1, 1.0, 0.75, 0.05,
        help="Calidad del brief, creatividad y fit general de la marca. "
             "Se multiplica por el fit de categoría de cada influencer.",
    )

    lambda_risk = st.sidebar.slider(
        "λ — Aversión al Riesgo", 0.0, 10.0, 1.0, 0.1,
        help="λ=0: máximo retorno. λ alto: mínima varianza.",
    )

    # ── CONTEXTO ──
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 📱 Contexto de Campaña")
    st.sidebar.markdown(
        "<span class='param-tag tag-ctx'>Contexto</span> "
        "Reescalan μ, σ, δ de todos los influencers",
        unsafe_allow_html=True,
    )

    plataforma = st.sidebar.selectbox(
        "Plataforma",
        list(PLATFORM_MULT.keys()),
        help="Instagram: baseline · TikTok: mayor alcance/viralidad, menor conversión · "
             "YouTube: menor alcance, mayor conversión (intención alta).",
    )
    formato = st.sidebar.selectbox(
        "Formato de contenido",
        list(FORMAT_MULT.keys()),
        help="Reel: máximo alcance, baja conversión directa · "
             "Story: menor alcance, mejor CTA · "
             "Video largo: máxima conversión, alcance limitado.",
    )
    timing = st.sidebar.selectbox(
        "Temporada / Timing",
        list(TIMING_MULT.keys()),
        help="Alta temporada: navidad, día de la madre, etc. · "
             "Tendencia viral: mayor alcance y volatilidad, menor conversión directa.",
    )

    # ── SIMULACIÓN ──
    st.sidebar.markdown("---")
    st.sidebar.markdown("## ⚙️ Simulación")
    n_sim = st.sidebar.select_slider(
        "Iteraciones Monte Carlo", options=[200,500,1000,2000], value=1000)

    # Combinar multiplicadores de contexto
    pm = PLATFORM_MULT[plataforma]
    fm = FORMAT_MULT[formato]
    tm = TIMING_MULT[timing]
    ctx_mu    = pm["mu"]    * fm["mu"]    * tm["mu"]
    ctx_sigma = pm["sigma"] * fm["sigma"] * tm["sigma"]
    ctx_delta = pm["delta"] * fm["delta"] * tm["delta"]

    # Panel de multiplicadores activos
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Multiplicadores activos:**")
    st.sidebar.markdown(
        f"μ ×{ctx_mu:.2f} &nbsp; σ ×{ctx_sigma:.2f} &nbsp; δ ×{ctx_delta:.2f}",
        unsafe_allow_html=True,
    )

    return dict(
        presupuesto=presupuesto, ticket=ticket,
        campaign_cat=campaign_cat,
        delta_camp=delta_camp, kappa_camp=kappa_camp,
        lambda_risk=lambda_risk,
        plataforma=plataforma, formato=formato, timing=timing,
        ctx_mu=ctx_mu, ctx_sigma=ctx_sigma, ctx_delta=ctx_delta,
        n_sim=n_sim,
    )


# ─────────────────────────────────────────────
# APP PRINCIPAL
# ─────────────────────────────────────────────

def main():
    st.markdown("""
    <h1 style='color:#c9d1d9;font-size:2rem;font-weight:700;margin-bottom:0'>
        Influencer Portfolio Optimizer
    </h1>
    <p style='color:#8b949e;font-size:1rem;margin-top:4px'>
        Euler-Maruyama SDE &nbsp;·&nbsp; Markowitz Portfolio Theory &nbsp;·&nbsp; Monte Carlo
    </p>""", unsafe_allow_html=True)
    st.markdown("---")

    params = render_sidebar()

    # ── Leyenda de parámetros ──
    with st.expander("Arquitectura de Parámetros del Modelo", expanded=False):
        st.markdown("""
| Parámetro | Origen | Quién lo controla | Descripción |
|---|---|---|---|
| **γ (gamma)** | `eng_rate` del influencer | Dato histórico | Tasa de engagement observada. No es un slider — viene de los datos. |
| **μ (mu)** | Tier del influencer × contexto | Dato + plataforma/formato/timing | Drift del alcance. Nano crece % rápido, Mega es estable. |
| **σ (sigma)** | Tier del influencer × contexto | Dato + plataforma/formato/timing | Volatilidad/viralidad. Nano muy volátil, Mega predecible. |
| **δ (delta)** | Slider de campaña × contexto | Manager | Tasa de conversión base del producto/CTA. |
| **κ (kappa)** | Brief del manager × fit categoría | Manager + algoritmo | Fit marca-audiencia. Se auto-ajusta por categoría de campaña. |
| **Plataforma** | Contexto | Manager | Reescala μ, σ, δ globalmente. |
| **Formato** | Contexto | Manager | Reel sube μ/σ, Video largo sube δ. |
| **Timing** | Contexto | Manager | Alta temporada sube δ, tendencia viral explota σ. |
        """)

    # ── Datos ──
    c1, c2 = st.columns([1,2])
    with c1:
        st.markdown("### Fuente de Datos")
        modo = st.radio("Modo",["Demo (auto-generado)","Producción (CSV)"])
    df = None
    if modo == "Producción (CSV)":
        with c2:
            st.markdown("### Cargar Datos Reales")
            uploaded = st.file_uploader(
                "CSV del Manager (handle, followers, eng_rate, cost_cop)", type=["csv"])
            df = parse_uploaded_csv(uploaded) if uploaded else build_demo_dataframe()
            if uploaded: st.success(f"Dataset cargado: {len(df)} influencers")
            else:        st.info("Esperando CSV... usando Demo.")
    else:
        with c2:
            st.info("Modo Demo — μ y σ derivados del tier de cada influencer. γ = eng_rate real.")
        df = build_demo_dataframe()

    template = "handle,followers,eng_rate,cost_cop\n" + "".join(
        f"{h},{v[2]},{v[3]},{v[4]}\n" for h,v in INFLUENCER_PROFILES.items())
    st.sidebar.download_button("Descargar plantilla CSV",data=template,
                               file_name="influencers_template.csv",mime="text/csv")

    st.markdown("---")

    # ── Simulación ──
    with st.spinner("Simulando SDE por influencer (Euler-Maruyama)..."):
        sim_results = euler_maruyama_simulation(
            df,
            ticket=params["ticket"],
            delta_camp=params["delta_camp"],
            kappa_camp=params["kappa_camp"],
            campaign_cat=params["campaign_cat"],
            ctx_mu=params["ctx_mu"],
            ctx_sigma=params["ctx_sigma"],
            ctx_delta=params["ctx_delta"],
            n_sim=params["n_sim"],
        )
        scores_df = compute_scores(df, sim_results)

    # Agregar parámetros efectivos a scores_df
    for p in ["mu_eff","sigma_eff","gamma_eff","delta_eff","kappa_eff"]:
        scores_df[p] = [sim_results[h][p] for h in scores_df.index]

    # ── Optimización ──
    with st.spinner("Optimizando portafolio (Markowitz + SLSQP)..."):
        opt    = optimize_portfolio(df, sim_results, params["presupuesto"], params["lambda_risk"])
        front  = efficient_frontier_points(df, sim_results, params["presupuesto"])

    # ═══════════════════════════════════════════
    # SECCIÓN 1 — CATÁLOGO & RANKINGS
    # ═══════════════════════════════════════════
    st.markdown(f"## Catálogo — {CATEGORY_ICONS.get(params['campaign_cat'],'')} Campaña: {params['campaign_cat']}")
    st.caption(
        f"Plataforma: **{params['plataforma']}** · Formato: **{params['formato']}** · "
        f"Timing: **{params['timing']}** · δ campaña: **{params['delta_camp']*100:.1f}%** · "
        f"κ base: **{params['kappa_camp']:.2f}**"
    )
    st.caption("Score = 40% ROI · 35% Engagement · 25% Costo-eficiencia")

    # ── Campeones por categoría ──
    st.markdown("### Mejor Influencer por Categoría")
    cats_list = sorted(CATEGORY_COLORS.keys())
    for row_start in range(0, len(cats_list), 4):
        row_cats = cats_list[row_start:row_start+4]
        cols     = st.columns(len(row_cats))
        for col, cat in zip(cols, row_cats):
            sub = scores_df[scores_df["categoria"]==cat]
            if sub.empty: continue
            champ  = sub.loc[sub["score"].idxmax()]
            color  = CATEGORY_COLORS.get(cat,"#58a6ff")
            icon   = CATEGORY_ICONS.get(cat,"")
            n_cat  = len(sub)
            kfit   = champ["kappa_eff"] / params["kappa_camp"] if params["kappa_camp"] > 0 else 0
            col.markdown(f"""
            <div class="champion-card" style="border-left:4px solid {color}">
                <div class="champion-cat">{icon} {cat}</div>
                <div class="champion-name">{champ.name}</div>
                <div class="champion-score">{champ['score']:.1f}<span style="font-size:1rem;color:#8b949e"> /100</span></div>
                <div class="champion-meta">
                    {champ['tier']} · {int(champ['followers']):,} seg.<br>
                    γ={champ['eng_rate']:.2%} · ROI {champ['roi_mean']:+.2f}<br>
                    κ fit campaña: <b>{champ['kappa_eff']:.2f}</b> ({kfit*100:.0f}%)<br>
                    Costo: ${champ['cost_cop']:,.0f}<br>
                    <span style="color:#484f58">#1 de {n_cat}</span>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Mapa de todos los influencers ──
    st.markdown("### Mapa Completo (45 influencers)")
    plotly_html(plot_catalog_bubble(scores_df), height=580)

    # ── Heatmap de parámetros SDE ──
    st.markdown("### Parámetros SDE Efectivos — Top 20")
    st.caption("μ y σ vienen del tier+contexto · γ del eng_rate real · δ y κ de la campaña modulada por fit")
    plotly_html(plot_params_heatmap(scores_df), height=320)

    # ── Ranking por categoría (tabs) ──
    st.markdown("### Ranking Detallado por Categoría")
    tabs = st.tabs([f"{CATEGORY_ICONS.get(c,'')} {c}" for c in cats_list])
    for tab, cat in zip(tabs, cats_list):
        with tab:
            sub = scores_df[scores_df["categoria"]==cat].sort_values("score",ascending=False)
            plotly_html(plot_category_ranking_bars(scores_df,cat), height=max(280,40*len(sub)))
            tbl = sub.reset_index()[[
                "handle","tier","followers","eng_rate","cost_cop",
                "mu_eff","sigma_eff","gamma_eff","delta_eff","kappa_eff",
                "roi_mean","roi_std","score","rank_cat"
            ]].copy()
            tbl.columns = ["Handle","Tier","Followers","γ(eng)","Costo",
                           "μ efect.","σ efect.","γ efect.","δ efect.","κ fit",
                           "ROI","σ ROI","Score","Rank"]
            tbl["Followers"] = tbl["Followers"].map("{:,}".format)
            tbl["γ(eng)"]    = tbl["γ(eng)"].map("{:.2%}".format)
            tbl["Costo"]     = tbl["Costo"].map("${:,.0f}".format)
            for c in ["μ efect.","σ efect.","γ efect.","δ efect.","κ fit","ROI","σ ROI"]:
                tbl[c] = tbl[c].map("{:.4f}".format)
            tbl["Score"]     = tbl["Score"].map("{:.1f}".format)
            st.dataframe(tbl, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ═══════════════════════════════════════════
    # SECCIÓN 2 — PORTAFOLIO ÓPTIMO
    # ═══════════════════════════════════════════
    st.markdown("## Portafolio Óptimo (Markowitz)")

    m1,m2,m3,m4,m5 = st.columns(5)
    for col,label,val,sub_ in [
        (m1,"ROI Esperado",    f"{opt['roi_portfolio']*100:+.1f}%",  "Portafolio ponderado"),
        (m2,"Sharpe Ratio",    f"{opt['sharpe']:.2f}",               "Retorno / Riesgo"),
        (m3,"P(ROI > 0)",      f"{opt['prob_exito']*100:.1f}%",      "Prob. de éxito MC"),
        (m4,"Varianza",        f"{opt['var_portfolio']:.4f}",          "Riesgo total"),
        (m5,"Presupuesto Usado",f"${opt['budget_used']/1e6:.1f}M",   f"de ${params['presupuesto']/1e6:.0f}M"),
    ]:
        col.markdown(f"""
        <div class="metric-card">
            <p class="metric-label">{label}</p>
            <p class="metric-value">{val}</p>
            <p class="metric-delta">{sub_}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    cl,cr = st.columns([1.2,1])
    with cl: plotly_html(plot_efficient_frontier(df,sim_results,front,opt,scores_df),height=500)
    with cr: plotly_html(plot_roi_dist(opt),height=340)

    plotly_html(plot_allocation(opt,scores_df),
                height=max(350,36*len(opt["allocation_cop"])))
    plotly_html(plot_stochastic_projection(sim_results,opt),height=420)

    # Tabla del portafolio
    st.markdown("### Composición del Portafolio")
    alloc = opt["allocation_cop"]
    if alloc:
        rows = []
        for h, cop in sorted(alloc.items(),key=lambda x:-x[1]):
            r  = scores_df.loc[h]
            sr = sim_results[h]
            rows.append({
                "Influencer":  h,
                "Categoría":   r["categoria"],
                "Tier":        r["tier"],
                "Rank en Cat":f"#{int(r['rank_cat'])}",
                "Score":       f"{r['score']:.1f}",
                "γ (eng)":     f"{r['eng_rate']:.2%}",
                "μ efect.":    f"{sr['mu_eff']:.5f}",
                "σ efect.":    f"{sr['sigma_eff']:.4f}",
                "δ efect.":    f"{sr['delta_eff']:.4f}",
                "κ fit":       f"{sr['kappa_eff']:.3f}",
                "ROI Esp.":    f"{sr['roi_mean']:+.3f}",
                "Riesgo σ":    f"{sr['roi_std']:.3f}",
                "Asignado":    f"${cop:,.0f}",
                "% Budget":    f"{cop/params['presupuesto']*100:.1f}%",
            })
        port_df = pd.DataFrame(rows)
        st.dataframe(port_df, use_container_width=True, hide_index=True)
        st.download_button("Exportar Portafolio (CSV)",
                           data=port_df.to_csv(index=False),
                           file_name="portafolio_optimo.csv",mime="text/csv")

    st.markdown("---")
    st.markdown(
        "<p style='color:#484f58;text-align:center;font-size:.8rem'>"
        "Euler-Maruyama SDE · Markowitz Portfolio Theory · Monte Carlo · Sierra Analytics</p>",
        unsafe_allow_html=True)


if __name__ == "__main__":
    main()
