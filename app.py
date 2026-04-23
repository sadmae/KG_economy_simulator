import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Форсайт-симулятор КР",
    page_icon="🇰🇬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main-header {
    background: linear-gradient(135deg, #01696f 0%, #0c4e54 100%);
    padding: 1.4rem 2rem; border-radius: 12px; margin-bottom: 1.5rem; color: white;
}
.main-header h1 { margin: 0; font-size: 1.5rem; font-weight: 700; }
.main-header p  { margin: 0.3rem 0 0; opacity: 0.85; font-size: 0.85rem; }
.section-title {
    font-size: 0.95rem; font-weight: 600; color: #28251d;
    border-bottom: 2px solid #01696f; padding-bottom: 5px; margin-bottom: 0.9rem;
}
.kpi-card {
    background: #f9f8f5; border: 1px solid #dcd9d5; border-radius: 10px;
    padding: 12px 14px; text-align: center;
}
.kpi-card.neg { border-left: 4px solid #a12c7b; }
.kpi-card.neu { border-left: 4px solid #01696f; }
.kpi-card.pos { border-left: 4px solid #437a22; }
.kpi-label { font-size: 0.68rem; color: #7a7974; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 500; }
.kpi-value { font-size: 1.3rem; font-weight: 700; color: #28251d; font-variant-numeric: tabular-nums; margin: 3px 0; }
.kpi-delta { font-size: 0.75rem; font-weight: 500; }
.kpi-delta.up   { color: #437a22; }
.kpi-delta.down { color: #a12c7b; }
.kpi-delta.flat { color: #7a7974; }
.badge-neg { background:#e0ced7; color:#561740; padding:2px 10px; border-radius:99px; font-size:0.78rem; font-weight:600; }
.badge-neu { background:#cedcd8; color:#0f3638; padding:2px 10px; border-radius:99px; font-size:0.78rem; font-weight:600; }
.badge-pos { background:#d4dfcc; color:#1e3f0a; padding:2px 10px; border-radius:99px; font-size:0.78rem; font-weight:600; }
[data-testid="stSidebar"] { background: #f3f0ec; }
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTS ───────────────────────────────────────────────
BASE_YEAR = 2023

BASELINE = {
    "gdp_nominal_bsom": 1100.0,
    "gdp_growth":  0.045,
    "inflation":   0.115,
    "exchange_rate": 90.27,
    "fiscal_balance": -0.018,
    "gross_debt":  0.458,
    "gold_price":  1960.0,
    "output_gap":  0.010,
    "ext_reserves_usd": 3.20,
    "current_account": -0.070,
}

COEFS = {
    # Inflation — stagflationary model
    "inf_base":     0.075,   # KG long-run avg inflation ~7.5%
    "inf_depr":     0.35,    # depreciation pass-through (high, KG ~60% imports)
    "inf_us":       0.40,    # imported global inflation
    "inf_gap":      0.30,    # positive output gap → inflation pressure DOWN (stable demand)
    "inf_gdpshock": 2.50,    # negative gdp_shock → cost-push inflation UP (stagflation)
    # Fiscal
    "fis_const": -0.044, "fis_gdp": 0.107,  "fis_gold": 0.00003,
    # Exchange rate
    "fx_const":  31.85,  "fx_gold": 0.0227, "fx_fis":  -5.0,
    # Debt
    "debt_fis":  -6.026, "debt_gdp": -0.127,"debt_gold": -0.0003,
    # GDP growth
    "gr_const":  0.043,  "gr_gap":  0.924,  "gr_fis":   0.15,
}

SCENARIO_DEFAULTS = {
    "Негативный": {"color":"#a12c7b","badge_class":"badge-neg","gdp_shock":-0.025,"gold_shock":-300.0,"fis_shock":-0.015,"us_inf":0.035,"ext_shock":-0.015,"reform_coef":0.0},
    "Нейтральный":{"color":"#01696f","badge_class":"badge-neu","gdp_shock": 0.000,"gold_shock":   0.0,"fis_shock": 0.000,"us_inf":0.025,"ext_shock": 0.000,"reform_coef":0.0},
    "Позитивный": {"color":"#437a22","badge_class":"badge-pos","gdp_shock": 0.025,"gold_shock": 300.0,"fis_shock": 0.015,"us_inf":0.018,"ext_shock": 0.015,"reform_coef":0.5},
}
COLOR_MAP = {sc: v["color"] for sc, v in SCENARIO_DEFAULTS.items()}

# ─── SIMULATION ENGINE ────────────────────────────────────────
def simulate(params: dict, horizon: int) -> pd.DataFrame:
    rows, prev = [], BASELINE.copy()
    for yr in range(BASE_YEAR + 1, horizon + 1):
        t = yr - BASE_YEAR
        gold = max(800.0, prev["gold_price"] + params["gold_shock"] * (1 if t == 1 else 0.3))
        fx   = max(prev["exchange_rate"] * 0.85,
                   COEFS["fx_const"] + COEFS["fx_gold"] * gold + COEFS["fx_fis"] * prev["fiscal_balance"])
        depr = (fx - prev["exchange_rate"]) / prev["exchange_rate"]

        # --- Inflation: stagflationary logic ---
        # (+) depreciation pass-through (KG highly import-dependent, ~60% imports)
        # (+) US inflation (global price import)
        # (+) negative gdp_shock → supply disruption → cost-push inflation UP
        # (-) positive output gap → demand managed, inflation lower (Phillips inverse)
        gap_now = prev["output_gap"] * 0.6 + params["gdp_shock"] * 0.5 + params.get("reform_coef", 0) * 0.005
        inflation = max(0.02, min(0.40,
            COEFS["inf_base"]
            + COEFS["inf_depr"]     * max(depr, 0)
            + COEFS["inf_us"]       * params["us_inf"]
            - COEFS["inf_gap"]      * gap_now
            - COEFS["inf_gdpshock"] * params["gdp_shock"]))

        gap = gap_now
        gdp_growth = max(-0.15, min(0.20,
            COEFS["gr_const"] + COEFS["gr_gap"] * gap
            + COEFS["gr_fis"] * (prev["fiscal_balance"] + params["fis_shock"])
            + params["gdp_shock"]))
        fiscal = max(-0.12, min(0.04,
            COEFS["fis_const"] + COEFS["fis_gdp"] * gdp_growth
            + COEFS["fis_gold"] * (gold - BASELINE["gold_price"]) + params["fis_shock"]))
        debt = max(0.15,
            prev["gross_debt"] + (-fiscal) * 0.8
            + COEFS["debt_gdp"] * (gdp_growth - BASELINE["gdp_growth"])
            + COEFS["debt_gold"] * (gold - BASELINE["gold_price"]))
        gdp_nom  = prev["gdp_nominal_bsom"] * (1 + gdp_growth) * (1 + inflation)
        curr_acc = max(-0.20, min(0.10, prev["current_account"] + params["ext_shock"] * 0.5
                                  + 0.002 * (gold - BASELINE["gold_price"]) / BASELINE["gold_price"]))
        reserves = max(0.5, prev["ext_reserves_usd"] * (1 + curr_acc * 0.3 + gdp_growth * 0.1))
        row = dict(year=yr, gdp_nominal_bsom=round(gdp_nom,1), gdp_growth=round(gdp_growth,4),
                   inflation=round(inflation,4), exchange_rate=round(fx,2), depreciation=round(depr,4),
                   fiscal_balance=round(fiscal,4), gross_debt=round(debt,4), gold_price=round(gold,1),
                   output_gap=round(gap,4), current_account=round(curr_acc,4), ext_reserves_usd=round(reserves,2))
        rows.append(row)
        prev = {**prev, **row}
    return pd.DataFrame(rows)

def run_all(user_params: dict, horizon: int) -> dict:
    return {sc: simulate({**SCENARIO_DEFAULTS[sc], **user_params.get(sc, {})}, horizon)
            for sc in SCENARIO_DEFAULTS}

# ─── CHART HELPERS ────────────────────────────────────────────
L = dict(font_family="Inter, sans-serif", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f9f8f5",
         legend=dict(orientation="h",yanchor="bottom",y=1.06,xanchor="center",x=0.5,font_size=11),
         xaxis=dict(showgrid=False,tickfont_size=11),
         yaxis=dict(gridcolor="#e8e6e2",gridwidth=1,tickfont_size=11))

def fan(results, col, title, fmt=".1%", ytitle=""):
    fig = go.Figure()
    bv = BASELINE.get(col, 0)
    fig.add_trace(go.Scatter(x=[BASE_YEAR], y=[bv], mode="markers",
                             marker=dict(size=9,color="#28251d"), name="База 2023"))
    for sc, df in results.items():
        xs = [BASE_YEAR] + list(df["year"])
        ys = [bv] + list(df[col])
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=sc,
                                 line=dict(color=COLOR_MAP.get(sc,"#888"),width=2.5), marker=dict(size=5)))
    neg = [bv] + list(results["Негативный"][col])
    pos = [bv] + list(results["Позитивный"][col])
    xs2 = [BASE_YEAR] + list(results["Нейтральный"]["year"])
    fig.add_trace(go.Scatter(x=xs2+xs2[::-1], y=pos+neg[::-1], fill="toself",
                             fillcolor="rgba(1,105,111,0.06)", line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.update_layout(**L, margin=dict(l=0,r=0,t=38,b=0), height=310, title=dict(text=title,font_size=12,x=0,xanchor="left"), yaxis_title=ytitle)
    fig.update_yaxes(tickformat=fmt)
    return fig

def bar_ch(results, col, title, fmt=".1%"):
    cmp_yrs = [2025, 2027, 2030]
    fig = go.Figure()
    for sc, df in results.items():
        vals = [df.loc[df.year==y, col].values[0] if y in df.year.values else None for y in cmp_yrs]
        fig.add_trace(go.Bar(name=sc, x=[str(y) for y in cmp_yrs], y=vals,
                             marker_color=COLOR_MAP.get(sc,"#888"),
                             text=[f"{v:{fmt[1:]}}" if v is not None else "" for v in vals],
                             textposition="outside", textfont_size=11))
    fig.update_layout(**L, margin=dict(l=0,r=0,t=38,b=0), height=270, title=dict(text=title,font_size=12,x=0,xanchor="left"),
                      barmode="group", yaxis_tickformat=fmt)
    return fig

def heatmap_ch(results, horizon):
    cols_info = [("ВВП рост","gdp_growth"),("Инфляция","inflation"),
                 ("Долг/ВВП","gross_debt"),("Фис. баланс","fiscal_balance"),("Курс сом","exchange_rate")]
    tyr = min(horizon, max(results["Нейтральный"]["year"]))
    matrix = []
    for sc, df in results.items():
        r = df[df.year == tyr].iloc[0]
        row = []
        for _, col in cols_info:
            bref = BASELINE.get(col,1)
            row.append(round((r[col]-bref)/abs(bref)*100 if bref else r[col], 1))
        matrix.append(row)
    fig = go.Figure(data=go.Heatmap(
        z=matrix, x=[c[0] for c in cols_info], y=list(results.keys()),
        colorscale=[[0,"#a12c7b"],[0.5,"#f9f8f5"],[1,"#437a22"]], zmid=0,
        text=[[f"{v:+.1f}%" for v in row] for row in matrix],
        texttemplate="%{text}", textfont_size=12, showscale=True,
        colorbar=dict(title="Откл.%",ticksuffix="%",len=0.8)))
    fig.update_layout(**L, height=240, margin=dict(l=0,r=60,t=38,b=0),
                      title=dict(text=f"Отклонение от базы 2023 к {tyr} г.", font_size=12, x=0))
    return fig

def overview_subplots(results):
    fig = make_subplots(rows=2, cols=2,
        subplot_titles=("Реальный рост ВВП","Инфляция (ИПЦ)","Гос. долг (% ВВП)","Курс сом/USD"),
        vertical_spacing=0.18, horizontal_spacing=0.08)
    panels = [("gdp_growth",1,1,".1%"),("inflation",1,2,".1%"),
              ("gross_debt",2,1,".0%"),("exchange_rate",2,2,".1f")]
    for col,row,cn,fmt in panels:
        bv = BASELINE.get(col,0)
        for sc, df in results.items():
            xs = [BASE_YEAR] + list(df["year"])
            ys = [bv] + list(df[col])
            fig.add_trace(go.Scatter(x=xs,y=ys,mode="lines+markers",name=sc,
                                     line=dict(color=COLOR_MAP.get(sc,"#888"),width=2),
                                     marker=dict(size=4),legendgroup=sc,
                                     showlegend=(row==1 and cn==1)),row=row,col=cn)
        fig.update_yaxes(tickformat=fmt,row=row,col=cn,gridcolor="#e8e6e2")
        fig.update_xaxes(showgrid=False,row=row,col=cn)
    fig.update_layout(font_family="Inter, sans-serif",paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="#f9f8f5",height=500,margin=dict(l=0,r=0,t=50,b=0),
                      legend=dict(orientation="h",yanchor="bottom",y=1.04,xanchor="center",x=0.5,font_size=11))
    return fig

# ─── SIDEBAR ─────────────────────────────────────────────────
st.sidebar.markdown("### ⚙️ Параметры сценариев")
st.sidebar.markdown("---")
user_params = {}
for sc_name in ["Негативный","Нейтральный","Позитивный"]:
    d = SCENARIO_DEFAULTS[sc_name]
    st.sidebar.markdown(f"<span class='{d['badge_class']}'>{sc_name}</span>", unsafe_allow_html=True)
    with st.sidebar.expander(f"Настроить {sc_name.lower()}", expanded=(sc_name=="Нейтральный")):
        gold  = st.slider("Шок цены золота ($/унц.)",-600,600,int(d["gold_shock"]),50,key=f"g_{sc_name}")
        gdp_s = st.slider("Шок роста ВВП (п.п.)",-6,6,int(d["gdp_shock"]*100),1,key=f"gr_{sc_name}")/100
        fis_s = st.slider("Фискальный шок (% ВВП)",-4,4,int(d["fis_shock"]*100),1,key=f"fi_{sc_name}")/100
        us_inf= st.slider("Инфляция США (%)",10,80,int(d["us_inf"]*100),5,key=f"us_{sc_name}")/100
        reform= st.slider("Реформы (0–1)",0.0,1.0,float(d["reform_coef"]),0.1,key=f"re_{sc_name}")
    user_params[sc_name] = {"gold_shock":gold,"gdp_shock":gdp_s,"fis_shock":fis_s,
                             "us_inf":us_inf,"reform_coef":reform,"ext_shock":d["ext_shock"]}
st.sidebar.markdown("---")
horizon = st.sidebar.select_slider("📅 Горизонт прогноза",options=list(range(2025,2036)),value=2030)

# ─── RUN ─────────────────────────────────────────────────────
results = run_all(user_params, horizon)

# ─── HEADER ──────────────────────────────────────────────────
st.markdown(f"""
<div class="main-header">
  <h1>🇰🇬 Форсайт-симулятор Кыргызской Республики</h1>
  <p>Макроэкономическое сценарное моделирование · База: модель Правительства КР / СЕКО (Брауманн) · 2023–{horizon}</p>
</div>""", unsafe_allow_html=True)

tab1,tab2,tab3,tab4,tab5 = st.tabs(["📊 Обзор","🔴 Реальный сектор","💰 Фискальный сектор","🏦 Монетарный & Внешний","⚡ Стресс-тест"])

# ── ТАБ 1 ────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-title">KPI — ключевые показатели</div>',unsafe_allow_html=True)
    cls_map = {"Негативный":"neg","Нейтральный":"neu","Позитивный":"pos"}
    for sc_name, df in results.items():
        tyr = min(horizon, df["year"].max())
        v = df[df.year==tyr].iloc[0]
        st.markdown(f'<span class="{SCENARIO_DEFAULTS[sc_name]["badge_class"]}">{sc_name}</span>',unsafe_allow_html=True)
        kpi_items = [
            ("ВВП рост",    v["gdp_growth"],      BASELINE["gdp_growth"],      ".1%", True),
            ("Инфляция",    v["inflation"],        BASELINE["inflation"],        ".1%", False),
            ("Долг/ВВП",    v["gross_debt"],       BASELINE["gross_debt"],       ".0%", False),
            ("Фис. баланс", v["fiscal_balance"],   BASELINE["fiscal_balance"],  ".1%", True),
            ("Курс сом/$",  v["exchange_rate"],    BASELINE["exchange_rate"],   ".1f", False),
            ("ВВП (млрд₴)", v["gdp_nominal_bsom"], BASELINE["gdp_nominal_bsom"],".0f", True),
        ]
        for c,(label,val,bval,fmt,up_good) in zip(st.columns(6),kpi_items):
            delta=val-bval; fs=fmt[1:]
            dcls=("flat" if abs(delta)<0.001 and fmt.endswith("%")
                  else ("up" if (delta>0)==up_good else "down"))
            c.markdown(f"""<div class="kpi-card {cls_map[sc_name]}">
  <div class="kpi-label">{label}</div>
  <div class="kpi-value">{val:{fs}}</div>
  <div class="kpi-delta {dcls}">{delta:+{fs}} vs 2023</div>
</div>""",unsafe_allow_html=True)
        st.markdown("<br>",unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-title">Сводный дашборд</div>',unsafe_allow_html=True)
    st.plotly_chart(overview_subplots(results),width='stretch')
    st.markdown('<div class="section-title">Тепловая карта отклонений</div>',unsafe_allow_html=True)
    st.plotly_chart(heatmap_ch(results,horizon),width='stretch')

    st.markdown('<div class="section-title">Матрица корреляций (2000–2023)</div>',unsafe_allow_html=True)
    corr_labels = ["ВВП рост","Инфляция","Курс сом","Цена золота","Фис. баланс","Долг/ВВП","Разрыв выпуска"]
    corr_matrix = [
        [ 1.00, 0.34,-0.14,-0.15, 0.17,-0.13, 0.57],
        [ 0.34, 1.00,-0.08, 0.08, 0.12,-0.09, 0.06],
        [-0.14,-0.08, 1.00, 0.71, 0.07,-0.29, 0.23],
        [-0.15, 0.08, 0.71, 1.00, 0.25,-0.76, 0.14],
        [ 0.17, 0.12, 0.07, 0.25, 1.00,-0.65, 0.46],
        [-0.13,-0.09,-0.29,-0.76,-0.65, 1.00,-0.37],
        [ 0.57, 0.06, 0.23, 0.14, 0.46,-0.37, 1.00],
    ]
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_matrix, x=corr_labels, y=corr_labels,
        colorscale=[[0,"#a12c7b"],[0.5,"#f9f8f5"],[1,"#437a22"]],
        zmin=-1, zmax=1, zmid=0,
        text=[[f"{v:.2f}" for v in row] for row in corr_matrix],
        texttemplate="%{text}", textfont_size=11,
        showscale=True, colorbar=dict(title="r", len=0.9),
    ))
    fig_corr.update_layout(
        font_family="Inter, sans-serif", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f9f8f5",
        height=380, margin=dict(l=0,r=60,t=20,b=0),
    )
    st.plotly_chart(fig_corr, width='stretch')

# ── ТАБ 2 ────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-title">Реальный сектор</div>',unsafe_allow_html=True)
    c1,c2=st.columns(2)
    with c1: st.plotly_chart(fan(results,"gdp_growth","Реальный рост ВВП",".1%","% г/г"),width='stretch')
    with c2: st.plotly_chart(fan(results,"gdp_nominal_bsom","Номинальный ВВП",".0f","млрд сом"),width='stretch')
    c3,c4=st.columns(2)
    with c3: st.plotly_chart(fan(results,"output_gap","Разрыв выпуска",".1%","% потенц. ВВП"),width='stretch')
    with c4: st.plotly_chart(bar_ch(results,"gdp_growth","Рост ВВП: сравнение сценариев",".1%"),width='stretch')
    st.markdown('<div class="section-title">Данные</div>',unsafe_allow_html=True)
    rows_tab=[]
    for sc,df in results.items():
        for _,row in df.iterrows():
            rows_tab.append({"Сценарий":sc,"Год":int(row.year),
                "ВВП рост":f"{row.gdp_growth:.1%}","ВВП (млрд сом)":f"{row.gdp_nominal_bsom:.0f}",
                "Разрыв выпуска":f"{row.output_gap:.1%}"})
    st.dataframe(pd.DataFrame(rows_tab),width='stretch',hide_index=True)

# ── ТАБ 3 ────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-title">Фискальный сектор</div>',unsafe_allow_html=True)
    c1,c2=st.columns(2)
    with c1: st.plotly_chart(fan(results,"fiscal_balance","Фискальный баланс (% ВВП)",".1%","% ВВП"),width='stretch')
    with c2: st.plotly_chart(fan(results,"gross_debt","Государственный долг (% ВВП)",".0%","% ВВП"),width='stretch')
    c3,c4=st.columns(2)
    with c3: st.plotly_chart(bar_ch(results,"fiscal_balance","Фис. баланс: сравнение",".1%"),width='stretch')
    with c4: st.plotly_chart(bar_ch(results,"gross_debt","Долг/ВВП: сравнение",".0%"),width='stretch')

    st.markdown('<div class="section-title">⚠️ Анализ долговой устойчивости (DSA)</div>',unsafe_allow_html=True)
    fig_dsa=go.Figure()
    fig_dsa.add_hrect(y0=0.0,y1=0.50,fillcolor="rgba(67,122,34,0.07)",line_width=0,
                      annotation_text="Устойчивая зона (<50%)",annotation_position="top left",annotation_font_size=10)
    fig_dsa.add_hrect(y0=0.50,y1=0.70,fillcolor="rgba(209,153,0,0.07)",line_width=0,
                      annotation_text="Умеренный риск (50–70%)",annotation_position="top left",annotation_font_size=10)
    fig_dsa.add_hrect(y0=0.70,y1=1.20,fillcolor="rgba(161,44,123,0.07)",line_width=0,
                      annotation_text="Высокий риск (>70%)",annotation_position="top left",annotation_font_size=10)
    for sc,df in results.items():
        xs=[BASE_YEAR]+list(df["year"]); ys=[BASELINE["gross_debt"]]+list(df["gross_debt"])
        fig_dsa.add_trace(go.Scatter(x=xs,y=ys,mode="lines+markers",name=sc,
                                     line=dict(color=COLOR_MAP.get(sc,"#888"),width=2.5),marker=dict(size=5)))
    fig_dsa.add_hline(y=0.70,line_dash="dot",line_color="#964219",annotation_text="Порог 70%",annotation_font_size=10)
    fig_dsa.add_hline(y=0.50,line_dash="dot",line_color="#437a22",annotation_text="Порог 50%",annotation_font_size=10)
    fig_dsa.update_layout(**L, margin=dict(l=0,r=0,t=38,b=0), height=320, yaxis_tickformat=".0%",
                          title=dict(text="DSA: динамика долга по сценариям",font_size=12))
    st.plotly_chart(fig_dsa,width='stretch')

# ── ТАБ 4 ────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-title">Монетарный сектор</div>',unsafe_allow_html=True)
    c1,c2=st.columns(2)
    with c1: st.plotly_chart(fan(results,"inflation","Инфляция (ИПЦ)",".1%","% г/г"),width='stretch')
    with c2: st.plotly_chart(fan(results,"exchange_rate","Курс сом/USD",".1f","сом за $"),width='stretch')
    st.markdown('<div class="section-title">Внешний сектор</div>',unsafe_allow_html=True)
    c3,c4=st.columns(2)
    with c3: st.plotly_chart(fan(results,"current_account","Текущий счёт (% ВВП)",".1%","% ВВП"),width='stretch')
    with c4: st.plotly_chart(fan(results,"ext_reserves_usd","Валютные резервы",".2f","млрд USD"),width='stretch')

    st.markdown('<div class="section-title">Чувствительность: цена золота → курс сома</div>',unsafe_allow_html=True)
    gold_rng=np.arange(800,3200,50)
    fx_rng=COEFS["fx_const"]+COEFS["fx_gold"]*gold_rng
    fig_s=go.Figure()
    fig_s.add_trace(go.Scatter(x=gold_rng,y=fx_rng,mode="lines",name="Расчётный курс",
                               line=dict(color="#01696f",width=2.5)))
    fig_s.add_vline(x=BASELINE["gold_price"],line_dash="dot",line_color="#28251d",
                    annotation_text=f"База: ${BASELINE['gold_price']:.0f}",annotation_font_size=10)
    for sc_n,sc_clr in [("Негативный","#a12c7b"),("Позитивный","#437a22")]:
        gval=BASELINE["gold_price"]+user_params[sc_n]["gold_shock"]
        fig_s.add_vline(x=gval,line_dash="dash",line_color=sc_clr,
                        annotation_text=sc_n[:3]+".",annotation_font_size=10)
    fig_s.update_layout(**L, margin=dict(l=0,r=0,t=38,b=0), height=290,xaxis_title="Цена золота ($/унц.)",yaxis_title="Курс сом/USD",
                        title=dict(text="Золото → Курс сома (r = 0.71)",font_size=12))
    st.plotly_chart(fig_s,width='stretch')

# ── ТАБ 5 ────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-title">⚡ Стресс-тестирование</div>',unsafe_allow_html=True)
    st.info("Оцените устойчивость экономики к резким шокам.")
    shock_opts={
        "🔴 Обвал цен на золото (−40%)":         {"gold_shock":-784,"gdp_shock":-0.010,"fis_shock":-0.010,"us_inf":0.025,"ext_shock":-0.020,"reform_coef":0},
        "🟡 Курсовой кризис (+30% девальвация)":  {"gold_shock":-300,"gdp_shock":-0.020,"fis_shock":-0.010,"us_inf":0.040,"ext_shock":-0.030,"reform_coef":0},
        "🟠 Фискальный шок (+3% ВВП расходов)":  {"gold_shock":   0,"gdp_shock": 0.010,"fis_shock":-0.030,"us_inf":0.025,"ext_shock": 0.000,"reform_coef":0},
        "🔵 Внешний шок (потеря резервов)":       {"gold_shock":-200,"gdp_shock":-0.015,"fis_shock":-0.005,"us_inf":0.030,"ext_shock":-0.040,"reform_coef":0},
        "⚫ Комбинированный шок":                  {"gold_shock":-500,"gdp_shock":-0.040,"fis_shock":-0.030,"us_inf":0.050,"ext_shock":-0.050,"reform_coef":0},
    }
    shock_name=st.selectbox("Тип шока",list(shock_opts.keys()))
    sp=shock_opts[shock_name]
    stress_res={
        "Нейтральный (база)": simulate({**SCENARIO_DEFAULTS["Нейтральный"],"us_inf":0.025,"ext_shock":0},horizon),
        "Стресс-сценарий":    simulate(sp, horizon),
    }
    CM_ST={"Нейтральный (база)":"#01696f","Стресс-сценарий":"#a12c7b"}
    c1,c2=st.columns(2)
    for (col,title,fmt,key2),cx in zip([
        ("gdp_growth","Рост ВВП при стрессе",".1%","s1"),
        ("gross_debt","Долг/ВВП при стрессе",".0%","s2"),
    ],[c1,c2]):
        fig_st=go.Figure()
        for sc,df in stress_res.items():
            bv=BASELINE.get(col,0)
            xs=[BASE_YEAR]+list(df["year"]); ys=[bv]+list(df[col])
            fig_st.add_trace(go.Scatter(x=xs,y=ys,mode="lines+markers",name=sc,
                                        line=dict(color=CM_ST[sc],width=2.5),marker=dict(size=5)))
        if col=="gross_debt":
            fig_st.add_hline(y=0.70,line_dash="dot",line_color="#964219",
                             annotation_text="Порог 70%",annotation_font_size=10)
        fig_st.update_layout(**L, margin=dict(l=0,r=0,t=38,b=0), height=280,yaxis_tickformat=fmt,
                             title=dict(text=title,font_size=12))
        cx.plotly_chart(fig_st,width='stretch',key=key2)

    tyr_st=min(horizon,stress_res["Нейтральный (база)"]["year"].max())
    b_row=stress_res["Нейтральный (база)"][stress_res["Нейтральный (база)"].year==tyr_st].iloc[0]
    s_row=stress_res["Стресс-сценарий"][stress_res["Стресс-сценарий"].year==tyr_st].iloc[0]
    st.markdown(f'<div class="section-title">Оценка влияния к {tyr_st} году</div>',unsafe_allow_html=True)
    imp=[]
    for name,col,fmt in [("Рост ВВП","gdp_growth",".1%"),("Инфляция","inflation",".1%"),
                          ("Долг/ВВП","gross_debt",".0%"),("Фис. баланс","fiscal_balance",".1%"),
                          ("Курс сом/USD","exchange_rate",".1f"),("Резервы (млрд$)","ext_reserves_usd",".2f")]:
        bv2,sv2=b_row[col],s_row[col]; diff=sv2-bv2; fs=fmt[1:]
        pdev=abs(diff/bv2) if bv2 else 0
        risk="🔴 Высокий" if pdev>0.15 else ("🟡 Умеренный" if pdev>0.05 else "🟢 Низкий")
        imp.append({"Показатель":name,"База":f"{bv2:{fs}}","Стресс":f"{sv2:{fs}}","Изменение":f"{diff:+{fs}}","Риск":risk})
    st.dataframe(pd.DataFrame(imp),width='stretch',hide_index=True)

st.markdown("---")
st.markdown(
    f"<small style='color:#7a7974'>Форсайт-симулятор КР v1.0 · "
    f"База: макромодель Правительства КР и Б. Брауманна (СЕКО) · НИСИ КР · Апрель 2026</small>",
    unsafe_allow_html=True)
