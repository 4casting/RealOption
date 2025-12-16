import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import io
import math
import datetime

# ==========================================
# 0. STREAMLIT CONFIG & STATE
# ==========================================
st.set_page_config(page_title="Master Thesis Valuation", layout="wide")

if 'history' not in st.session_state: st.session_state.history = []
if 'simulation_results' not in st.session_state: st.session_state.simulation_results = None
if 'pdf_buffer' not in st.session_state: st.session_state.pdf_buffer = None

# ==========================================
# 1. CORE LOGIC
# ==========================================
def run_simulation(M, p, q, C, ARPU, kappa, Delta_CM, Fixed_Cost, start, T, 
                   mode='static', trigger_val=0.05, fallback_params=None):
    N = [0.0] * T
    W = [0.0] * T
    N[0] = start
    W[0] = N[0] * ARPU - Fixed_Cost

    check_period_start = 3
    option_exercised = False
    
    # Wir brauchen lokale Kopien der Parameter, da sie sich beim Switch ändern können
    curr_p, curr_q, curr_C = p, q, C
    curr_ARPU, curr_kappa, curr_Delta_CM = ARPU, kappa, Delta_CM
    curr_FC = Fixed_Cost
    curr_M = M

    for t in range(1, T):
        N_prev = N[t-1]
        
        # --- REAL OPTION LOGIC ---
        if mode != 'static' and not option_exercised and t >= check_period_start:
            # Trigger-Prüfung basierend auf aktuellen Parametern
            potential_acquisition = (curr_p + curr_q * (N_prev / curr_M)) * (curr_M - N_prev)
            trigger_threshold = trigger_val * curr_M
            
            if potential_acquisition < trigger_threshold:
                option_exercised = True
                
                if mode == 'switch' and fallback_params:
                    # KOMPLETTER STRATEGIEWECHSEL ZU A
                    curr_p = fallback_params['p']
                    curr_q = fallback_params['q']
                    curr_C = fallback_params['C']
                    curr_ARPU = fallback_params['ARPU']
                    curr_kappa = fallback_params['kappa']
                    curr_Delta_CM = fallback_params['Delta_CM']
                    curr_FC = fallback_params['Fixed_Cost']
                    # M ändern wir in der Regel nicht, außer es ist explizit gewollt
                    
                elif mode == 'abandon':
                    curr_C = 1.0  # Alle Kunden gehen
                    curr_FC = 0   # Kostenstopp
        # -------------------------

        retention = N_prev * (1 - curr_C)
        acquisition = (curr_p + curr_q * (N_prev / curr_M)) * (curr_M - N_prev)
        
        if acquisition < 0: acquisition = 0
        if mode == 'abandon' and option_exercised: acquisition = 0
        
        N[t] = retention + acquisition
        if N[t] > curr_M: N[t] = curr_M
        
        revenue = N[t] * curr_ARPU
        cannibalization_loss = acquisition * curr_kappa * curr_Delta_CM
        
        W[t] = revenue - cannibalization_loss - curr_FC
        
    return N, W, sum(W)

# ==========================================
# 2. STATISTIK HELPER
# ==========================================
def calculate_cochran_n(params_dict, T, mode='static', fallback=None, trigger=0.05):
    pilot_n = 250 # Etwas höherer Pilot
    results = []
    
    # Helper für Random Values
    def get_val(v): return np.random.triangular(v[0], (v[0]+v[1])/2, v[1]) if isinstance(v, tuple) else v
    
    for _ in range(pilot_n):
        curr = {k: get_val(v) for k, v in params_dict.items()}
        curr_fb = {k: get_val(v) for k, v in fallback.items()} if fallback else None
        _, _, val = run_simulation(**curr, start=1, T=T, mode=mode, trigger_val=trigger, fallback_params=curr_fb)
        results.append(val)
    
    std_dev = np.std(results)
    mean_val = np.mean(results)
    
    # Cochran mit 1% Fehler-Marge (Präzise!)
    if mean_val == 0: return 2000
    E = abs(mean_val * 0.01) 
    if E == 0: return 2000
    
    n_opt = (1.96 * std_dev / E) ** 2
    return max(int(math.ceil(n_opt)), 5000) # Mindestens 5000 für gute Optik

def get_tornado_data(base_params, ranges, T, mode, trigger, fallback_ranges):
    # Base Inputs (Mittelwerte)
    def mid(v): return (v[0]+v[1])/2 if isinstance(v, tuple) else v
    base_inputs = {k: mid(v) for k, v in ranges.items()}
    fb_inputs = {k: mid(v) for k, v in fallback_ranges.items()} if fallback_ranges else None
    
    _, _, base_val = run_simulation(**base_inputs, start=1, T=T, mode=mode, trigger_val=trigger, fallback_params=fb_inputs)
    
    data = []
    for param, val_range in ranges.items():
        if not isinstance(val_range, tuple): continue
        
        # Low Case
        low_inputs = base_inputs.copy(); low_inputs[param] = val_range[0]
        _, _, val_low = run_simulation(**low_inputs, start=1, T=T, mode=mode, trigger_val=trigger, fallback_params=fb_inputs)
        
        # High Case
        high_inputs = base_inputs.copy(); high_inputs[param] = val_range[1]
        _, _, val_high = run_simulation(**high_inputs, start=1, T=T, mode=mode, trigger_val=trigger, fallback_params=fb_inputs)
        
        data.append({
            "Parameter": param, 
            "Low": val_low - base_val, 
            "High": val_high - base_val, 
            "Range": abs(val_high - val_low),
            "Abs_Low": val_low, # Für genauere Anzeige
            "Abs_High": val_high
        })
    return pd.DataFrame(data).sort_values(by="Range", ascending=True), base_val

def get_regression_sensitivity(df_inputs, y_values):
    scaler = StandardScaler(); X_scaled = scaler.fit_transform(df_inputs)
    model = LinearRegression(); model.fit(X_scaled, y_values)
    return pd.DataFrame({"Parameter": df_inputs.columns, "Beta": model.coef_}).sort_values(by="Beta", key=abs, ascending=True), model.score(X_scaled, y_values)

# ==========================================
# 3. GUI INPUTS
# ==========================================

# --- HISTORY ---
with st.sidebar:
    st.header("History")
    def restore():
        idx = st.session_state.hist_sel
        if idx is not None:
            entry = st.session_state.history[idx]
            for k, v in entry['params'].items(): st.session_state[k] = v
            st.toast(f"Stand {entry['timestamp']} geladen!")

    if st.session_state.history:
        opts = {i: f"{e['timestamp']} (M={e['params'].get('M_val', '?')})" for i, e in enumerate(st.session_state.history)}
        st.selectbox("Wiederherstellen:", list(opts.keys()), format_func=lambda x: opts[x], key="hist_sel", index=None, on_change=restore)

st.markdown("<h1 style='text-align: center;'>Valuing Market Entry Strategies</h1>", unsafe_allow_html=True)

with st.container():
    st.markdown("### Global Settings")
    c1, c2, c3, c4 = st.columns(4)
    with c1: T_in = st.slider("Jahre (T)", 5, 20, 15, key="T_val")
    with c2: M_in = st.number_input("Marktpotenzial (M)", 300, 1000000, 500, step=50, key="M_val")
    with c3: trig_in = st.slider("Trigger (< % Growth)", 0.01, 0.3, 0.05, key="trig_val")
    with c4: 
        st.write("")
        start_btn = st.button("Start Simulation", type="primary", use_container_width=True)

st.markdown("---")

col_left, col_right = st.columns(2)
def range_in(lbl, min_v, max_v, sfx, stp=0.01, fmt="%.2f"):
    c1, c2 = st.columns(2)
    k_min, k_max = f"{lbl}_min_{sfx}", f"{lbl}_max_{sfx}"
    if k_min not in st.session_state: st.session_state[k_min] = min_v
    if k_max not in st.session_state: st.session_state[k_max] = max_v
    return (c1.number_input(f"{lbl} Min", value=st.session_state[k_min], step=stp, format=fmt, key=k_min),
            c2.number_input(f"{lbl} Max", value=st.session_state[k_max], step=stp, format=fmt, key=k_max))

with col_left:
    st.markdown("### Option A: Standard (Fallback)")
    p_a = range_in("p", 0.005, 0.010, "a", 0.001, "%.3f")
    q_a = range_in("q", 0.15, 0.25, "a")
    c_a = range_in("C", 0.03, 0.05, "a")
    arpu_a = range_in("ARPU", 3800.0, 4200.0, "a", 100.0)
    fc_a = range_in("Fixkosten", 140000.0, 160000.0, "a", 1000.0)
    kap_a = range_in("Kappa", 0.05, 0.10, "a")
    dcm_a = range_in("Delta Margin", 50.0, 100.0, "a", 10.0)

with col_right:
    st.markdown("### Option B: Fighter (Start)")
    p_b = range_in("p", 0.030, 0.050, "b", 0.001, "%.3f")
    q_b = range_in("q", 0.20, 0.30, "b")
    c_b = range_in("C", 0.08, 0.12, "b")
    arpu_b = range_in("ARPU", 3000.0, 3500.0, "b", 100.0)
    fc_b = range_in("Fixkosten", 180000.0, 200000.0, "b", 1000.0)
    kap_b = range_in("Kappa", 0.10, 0.20, "b")
    dcm_b = range_in("Delta Margin", 50.0, 100.0, "b", 10.0)

# ==========================================
# 4. EXECUTION
# ==========================================
if start_btn:
    # History Save
    snap = {k: v for k, v in st.session_state.items() if "_min_" in k or "_max_" in k or k in ["T_val", "M_val", "trig_val"]}
    st.session_state.history.append({'timestamp': datetime.datetime.now().strftime("%H:%M:%S"), 'params': snap})

    # Params
    params_A = {'M': (M_in*0.9, M_in*1.1), 'p': p_a, 'q': q_a, 'C': c_a, 'ARPU': arpu_a, 'kappa': kap_a, 'Delta_CM': dcm_a, 'Fixed_Cost': fc_a}
    params_B = {'M': (M_in*0.9, M_in*1.1), 'p': p_b, 'q': q_b, 'C': c_b, 'ARPU': arpu_b, 'kappa': kap_b, 'Delta_CM': dcm_b, 'Fixed_Cost': fc_b}

    # Cochran
    with st.spinner("Berechne optimale Iterationen (Cochran 1%)..."):
        n_A = calculate_cochran_n(params_A, T_in, 'static')
        n_B = calculate_cochran_n(params_B, T_in, 'static')
        n_Switch = calculate_cochran_n(params_B, T_in, 'switch', fallback=params_A, trigger=trig_in)
        n_Abandon = calculate_cochran_n(params_B, T_in, 'abandon', trigger=trig_in)
    
    st.toast(f"Runs: A={n_A}, B={n_B}, Sw={n_Switch}, Ab={n_Abandon}")

    # Simulation
    res_store = {}
    bar = st.progress(0)
    scenarios = [
        ("1. Standard (A)", n_A, params_A, 'static', None),
        ("2. Fighter (B)", n_B, params_B, 'static', None),
        ("3. Switch Option", n_Switch, params_B, 'switch', params_A),
        ("4. Abandon Option", n_Abandon, params_B, 'abandon', None)
    ]
    
    def rnd(v): return np.random.triangular(v[0], (v[0]+v[1])/2, v[1]) if isinstance(v, tuple) else v

    for idx, (name, n, p_rng, mode, fb_rng) in enumerate(scenarios):
        sim_sums, sim_inputs, all_N, all_W = [], [], [], []
        
        for _ in range(n):
            curr = {k: rnd(v) for k, v in p_rng.items()}
            fb = {k: rnd(v) for k, v in fb_rng.items()} if fb_rng else None
            N_t, W_t, tot = run_simulation(**curr, start=1, T=T_in, mode=mode, trigger_val=trig_in, fallback_params=fb)
            sim_sums.append(tot); all_N.append(N_t); all_W.append(W_t); sim_inputs.append(curr)
            
        bar.progress((idx+1)/4)
        
        df_in = pd.DataFrame(sim_inputs); df_in_reg = df_in.loc[:, df_in.std() > 0]
        torn, base_v = get_tornado_data(None, p_rng, T_in, mode, trig_in, fb_rng)
        reg, r2 = (None, 0)
        if not df_in_reg.empty: reg, r2 = get_regression_sensitivity(df_in_reg, sim_sums)

        res_store[name] = {
            "n": n, "sums": sim_sums, "avg_N": np.mean(all_N, axis=0), "avg_W": np.mean(all_W, axis=0),
            "tornado": (torn, base_v), "regression": (reg, r2),
            "mean": np.mean(sim_sums), "std": np.std(sim_sums), 
            "min": np.min(sim_sums), "max": np.max(sim_sums), "var5": np.percentile(sim_sums, 5)
        }
    
    st.session_state.simulation_results = res_store
    
    # PDF GEN
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        # Page 1: Overview
        fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8.27, 11.69))
        for n, d in res_store.items(): ax1.plot(d["avg_N"], label=n)
        ax1.set_title("Customer Adoption N(t)"); ax1.legend(); ax1.grid(True, alpha=0.3)
        
        for n, d in res_store.items(): ax2.plot(d["avg_W"], label=n)
        ax2.set_title("Net Value Contribution W(t) per Period"); ax2.grid(True, alpha=0.3)
        ax2.axhline(0, color='k', linewidth=0.5)
        
        nms = list(res_store.keys()); mus = [res_store[n]["mean"] for n in nms]; sigs = [res_store[n]["std"] for n in nms]
        ax3.bar(nms, mus, yerr=sigs, capsize=5, alpha=0.7, color=['tab:blue', 'tab:orange', 'tab:green', 'tab:red'])
        ax3.set_title("Total Value Comparison (Mean +/- StdDev)"); ax3.set_ylabel("EUR")
        plt.tight_layout(); pdf.savefig(fig1); plt.close(fig1)

        # Details
        for k, d in res_store.items():
            fig, (ax_t, ax_h, ax_r) = plt.subplots(3, 1, figsize=(8.27, 11.69))
            fig.suptitle(f"Detail: {k}", fontsize=16)
            
            df_t, b_v = d["tornado"]; y = np.arange(len(df_t))
            ax_t.barh(y, df_t["Low"], color='tab:red', alpha=0.6); ax_t.barh(y, df_t["High"], color='tab:green', alpha=0.6)
            ax_t.set_yticks(y); ax_t.set_yticklabels(df_t["Parameter"]); ax_t.invert_yaxis(); ax_t.axvline(0, c='k', ls='--')
            ax_t.set_title(f"Sensitivity (Tornado) - Base: {b_v/1e6:.2f}M")

            ax_h.hist(d["sums"], bins=40, color='skyblue', edgecolor='white')
            ax_h.axvline(d["mean"], c='k', ls='--', label=f"Mean: {d['mean']/1e6:.1f}M")
            ax_h.axvline(d["var5"], c='r', ls='--', label=f"VaR 5%: {d['var5']/1e6:.1f}M")
            ax_h.set_title(f"Risk Profile (n={d['n']})"); ax_h.legend()

            df_r, r2 = d["regression"]
            if df_r is not None:
                cols = ['tab:green' if c > 0 else 'tab:red' for c in df_r["Beta"]]
                ax_r.barh(df_r["Parameter"], df_r["Beta"], color=cols)
                ax_r.set_title(f"Global Sensitivity (Beta) - R2={r2:.2f}")
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95]); pdf.savefig(fig); plt.close(fig)
    st.session_state.pdf_buffer = buf

# ==========================================
# 5. DISPLAY RESULTS
# ==========================================
if st.session_state.simulation_results:
    res = st.session_state.simulation_results
    
    st.markdown("### Output Analysis")
    
    # 1. PRÄZISE TABELLE (DataFrame)
    st.markdown("#### Summary of Results")
    summary_data = []
    for k, d in res.items():
        summary_data.append({
            "Szenario": k,
            "Simulations (n)": d['n'],
            "Mean (€)": f"{d['mean']:,.0f}",
            "StdDev (€)": f"{d['std']:,.0f}",
            "Min (€)": f"{d['min']:,.0f}",
            "Max (€)": f"{d['max']:,.0f}",
            "VaR 5% (€)": f"{d['var5']:,.0f}" # Value at Risk
        })
    st.dataframe(pd.DataFrame(summary_data).set_index("Szenario"), use_container_width=True)

    # 2. ZEITREIHEN PLOTS (Groß)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Customer Evolution $N(t)$")
        fig_n, ax_n = plt.subplots(figsize=(6, 4))
        for n, d in res.items(): ax_n.plot(d["avg_N"], label=n)
        ax_n.legend(); ax_n.grid(True, alpha=0.3); ax_n.set_ylabel("Anzahl Kunden")
        st.pyplot(fig_n)
    
    with c2:
        st.markdown("#### Net Value Contribution $W(t)$")
        fig_w, ax_w = plt.subplots(figsize=(6, 4))
        for n, d in res.items(): ax_w.plot(d["avg_W"], label=n)
        ax_w.legend(); ax_w.grid(True, alpha=0.3); ax_w.set_ylabel("Wertbeitrag (€)")
        ax_w.axhline(0, color='black', linewidth=0.8) # Null-Linie
        st.pyplot(fig_w)

    # 3. DETAILS TABS
    tabs = st.tabs(list(res.keys()))
    for i, tab in enumerate(tabs):
        with tab:
            k = list(res.keys())[i]; d = res[k]
            c_d1, c_d2 = st.columns([1, 2])
            with c_d1:
                st.metric("Mean Value", f"€ {d['mean']/1e6:.2f} M")
                st.metric("Value at Risk (5%)", f"€ {d['var5']/1e6:.2f} M", delta_color="inverse")
            with c_d2:
                fig_h, ax_h = plt.subplots(figsize=(8, 3))
                ax_h.hist(d["sums"], bins=50, color='steelblue', edgecolor='white', alpha=0.8)
                ax_h.axvline(d["mean"], color='k', linestyle='--', linewidth=1.5, label='Mean')
                ax_h.axvline(d["var5"], color='r', linestyle='--', linewidth=1.5, label='VaR 5%')
                ax_h.set_title("Risiko-Profil (Histogramm der Gesamtergebnisse)"); ax_h.legend()
                st.pyplot(fig_h)

    # 4. DOWNLOAD
    if st.session_state.pdf_buffer:
        st.download_button("📄 PDF Report Download", st.session_state.pdf_buffer.getvalue(), 
                           f"Report_{datetime.datetime.now().strftime('%H%M')}.pdf", "application/pdf", use_container_width=True)



