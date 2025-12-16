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
# 0. STREAMLIT CONFIG & STATE INIT
# ==========================================
st.set_page_config(page_title="Master Thesis Valuation", layout="wide")

# Initialisierung des Session States (Gedächtnis der App)
if 'history' not in st.session_state:
    st.session_state.history = []
if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = None
if 'pdf_buffer' not in st.session_state:
    st.session_state.pdf_buffer = None

# ==========================================
# 1. CORE LOGIC & SIMULATION
# ==========================================
def run_simulation(M, p, q, C, ARPU, kappa, Delta_CM, Fixed_Cost, start, T, 
                   mode='static', trigger_val=0.05, fallback_params=None):
    N = [0.0] * T
    W = [0.0] * T
    N[0] = start
    W[0] = N[0] * ARPU - Fixed_Cost

    check_period_start = 3
    option_exercised = False

    for t in range(1, T):
        N_prev = N[t-1]
        
        # --- REAL OPTION LOGIC ---
        if mode != 'static' and not option_exercised and t >= check_period_start:
            potential_acquisition = (p + q * (N_prev / M)) * (M - N_prev)
            trigger_threshold = trigger_val * M
            
            if potential_acquisition < trigger_threshold:
                option_exercised = True
                if mode == 'switch' and fallback_params:
                    # Switch Parameters
                    p = fallback_params['p']; q = fallback_params['q']
                    C = fallback_params['C']; ARPU = fallback_params['ARPU']
                    kappa = fallback_params['kappa']; Delta_CM = fallback_params['Delta_CM']
                    Fixed_Cost = fallback_params['Fixed_Cost']
                elif mode == 'abandon':
                    C = 1.0; Fixed_Cost = 0
        # -------------------------

        retention = N_prev * (1 - C)
        acquisition = (p + q * (N_prev / M)) * (M - N_prev)
        if acquisition < 0: acquisition = 0
        if mode == 'abandon' and option_exercised: acquisition = 0
        
        N[t] = retention + acquisition
        if N[t] > M: N[t] = M
        
        revenue = N[t] * ARPU
        cannibalization_loss = acquisition * kappa * Delta_CM
        W[t] = revenue - cannibalization_loss - Fixed_Cost
        
    return N, W, sum(W)

# ==========================================
# 2. STATISTIK HELPER
# ==========================================
def calculate_cochran_n(params_dict, T, mode='static', fallback=None, trigger=0.05):
    # Pilot Run
    pilot_n = 200
    results = []
    for _ in range(pilot_n):
        curr = {k: np.random.triangular(v[0], (v[0]+v[1])/2, v[1]) if isinstance(v, tuple) else v for k, v in params_dict.items()}
        curr_fb = {k: np.random.triangular(v[0], (v[0]+v[1])/2, v[1]) if isinstance(v, tuple) else v for k, v in fallback.items()} if fallback else None
        _, _, val = run_simulation(**curr, start=1, T=T, mode=mode, trigger_val=trigger, fallback_params=curr_fb)
        results.append(val)
    
    std_dev = np.std(results)
    mean_val = np.mean(results)
    
    # FIX: Strengerer Fehler (1% statt 5%) -> Mehr Iterationen
    if mean_val == 0: return 2000
    E = abs(mean_val * 0.01) 
    
    if E == 0: return 2000
    n_opt = (1.96 * std_dev / E) ** 2
    
    # Sicherstellen, dass mindestens 1000 Läufe gemacht werden für schöne Histogramme
    return max(int(math.ceil(n_opt)), 1000)

def get_tornado_data(base_params, ranges, T, mode, trigger, fallback_ranges):
    # ... (Identisch zur vorherigen Version, verkürzt für Übersichtlichkeit) ...
    # Base Inputs
    base_inputs = {k: (v[0]+v[1])/2 if isinstance(v, tuple) else v for k, v in ranges.items()}
    fb_inputs = {k: (v[0]+v[1])/2 if isinstance(v, tuple) else v for k, v in fallback_ranges.items()} if fallback_ranges else None
    _, _, base_val = run_simulation(**base_inputs, start=1, T=T, mode=mode, trigger_val=trigger, fallback_params=fb_inputs)
    
    data = []
    for param, val_range in ranges.items():
        if not isinstance(val_range, tuple): continue
        # Low
        low_inputs = base_inputs.copy(); low_inputs[param] = val_range[0]
        _, _, val_low = run_simulation(**low_inputs, start=1, T=T, mode=mode, trigger_val=trigger, fallback_params=fb_inputs)
        # High
        high_inputs = base_inputs.copy(); high_inputs[param] = val_range[1]
        _, _, val_high = run_simulation(**high_inputs, start=1, T=T, mode=mode, trigger_val=trigger, fallback_params=fb_inputs)
        
        data.append({"Parameter": param, "Low": val_low - base_val, "High": val_high - base_val, "Range": abs(val_high - val_low)})
    return pd.DataFrame(data).sort_values(by="Range", ascending=True), base_val

def get_regression_sensitivity(df_inputs, y_values):
    scaler = StandardScaler(); X_scaled = scaler.fit_transform(df_inputs)
    model = LinearRegression(); model.fit(X_scaled, y_values)
    return pd.DataFrame({"Parameter": df_inputs.columns, "Beta": model.coef_}).sort_values(by="Beta", key=abs, ascending=True), model.score(X_scaled, y_values)

# ==========================================
# 3. GUI & HISTORY MANAGEMENT
# ==========================================

# --- HISTORY SIDEBAR ---
with st.sidebar:
    st.header("📜 Verlauf / History")
    
    # Funktion zum Wiederherstellen
    def load_history_entry():
        selected_idx = st.session_state.history_selector
        if selected_idx is not None:
            entry = st.session_state.history[selected_idx]
            # Alle Werte im Session State überschreiben
            for k, v in entry['params'].items():
                st.session_state[k] = v
            st.toast(f"Einstellungen von {entry['timestamp']} geladen!", icon="✅")

    if st.session_state.history:
        # Erstelle Labels für die Selectbox
        options = {i: f"Run {i+1}: {entry['timestamp']}" for i, entry in enumerate(st.session_state.history)}
        st.selectbox(
            "Frühere Simulation laden:", 
            options=list(options.keys()), 
            format_func=lambda x: options[x],
            key="history_selector",
            index=None,
            placeholder="Wähle einen Lauf...",
            on_change=load_history_entry
        )
    else:
        st.info("Noch keine Simulationen durchgeführt.")

st.markdown("<h1 style='text-align: center;'>Valuing Digital Market Entry Strategies</h1>", unsafe_allow_html=True)

# --- GLOBALE EINSTELLUNGEN ---
with st.container():
    st.markdown("### 🌐 Globale & Methodische Einstellungen")
    c1, c2, c3, c4 = st.columns(4)
    with c1: T_in = st.slider("Zeitraum (T)", 5, 20, 15, key="T_val")
    with c2: M_in = st.number_input("Marktpotenzial (M)", 300, 5000, 500, step=50, key="M_val")
    with c3: trig_in = st.slider("Trigger für Option C/D (< % Growth)", 0.01, 0.15, 0.05, key="trig_val")
    with c4: 
        st.write("")
        # Der Button bekommt einen Callback, um die Simulation zu starten
        start_btn = st.button("🚀 Analyse starten", type="primary", use_container_width=True)

st.markdown("---")

col_left, col_right = st.columns(2)

# Helper für Inputs mit Session State Keys
def range_input(label, def_min, def_max, k_suffix, step=0.01, fmt="%.2f"):
    c1, c2 = st.columns(2)
    # Keys sind wichtig für History Restore!
    k_min = f"{label}_min_{k_suffix}"
    k_max = f"{label}_max_{k_suffix}"
    
    # Defaults setzen, falls noch nicht im State
    if k_min not in st.session_state: st.session_state[k_min] = def_min
    if k_max not in st.session_state: st.session_state[k_max] = def_max
    
    v_min = c1.number_input(f"{label} Min", value=st.session_state[k_min], step=step, format=fmt, key=k_min)
    v_max = c2.number_input(f"{label} Max", value=st.session_state[k_max], step=step, format=fmt, key=k_max)
    return (v_min, v_max)

with col_left:
    st.markdown("### 🔵 Option A: Standard (Basis & Fallback)")
    p_a = range_input("Innovation (p)", 0.005, 0.010, "a", 0.001, "%.3f")
    q_a = range_input("Imitation (q)", 0.15, 0.25, "a")
    c_a = range_input("Churn (C)", 0.03, 0.05, "a")
    arpu_a = range_input("ARPU (€)", 3800.0, 4200.0, "a", 100.0)
    fc_a = range_input("Fixkosten (€)", 140000.0, 160000.0, "a", 1000.0)
    kap_a = range_input("Kappa", 0.05, 0.10, "a")
    dcm_a = range_input("Delta Margin", 50.0, 100.0, "a", 10.0)

with col_right:
    st.markdown("### 🔴 Option B: Fighter (Startpunkt)")
    p_b = range_input("Innovation (p)", 0.030, 0.050, "b", 0.001, "%.3f")
    q_b = range_input("Imitation (q)", 0.20, 0.30, "b")
    c_b = range_input("Churn (C)", 0.08, 0.12, "b")
    arpu_b = range_input("ARPU (€)", 3000.0, 3500.0, "b", 100.0)
    fc_b = range_input("Fixkosten (€)", 180000.0, 200000.0, "b", 1000.0)
    kap_b = range_input("Kappa", 0.10, 0.20, "b")
    dcm_b = range_input("Delta Margin", 50.0, 100.0, "b", 10.0)

# ==========================================
# 4. EXECUTION LOGIC
# ==========================================

# Wir sammeln alle aktuellen Parameter-Werte in einem Dictionary
current_params_snapshot = {k: v for k, v in st.session_state.items() if "_min_" in k or "_max_" in k or k in ["T_val", "M_val", "trig_val"]}

if start_btn:
    # 1. History speichern
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    st.session_state.history.append({
        'timestamp': timestamp,
        'params': current_params_snapshot
    })

    # 2. Parameter Dictionaries bauen
    params_A = {'M': (M_in*0.9, M_in*1.1), 'p': p_a, 'q': q_a, 'C': c_a, 'ARPU': arpu_a, 'kappa': kap_a, 'Delta_CM': dcm_a, 'Fixed_Cost': fc_a}
    params_B = {'M': (M_in*0.9, M_in*1.1), 'p': p_b, 'q': q_b, 'C': c_b, 'ARPU': arpu_b, 'kappa': kap_b, 'Delta_CM': dcm_b, 'Fixed_Cost': fc_b}

    # 3. Cochran Calculation (Auto-Iterationen)
    with st.spinner("Berechne optimale Iterationen (1% Precision)..."):
        n_A = calculate_cochran_n(params_A, T_in, 'static')
        n_B = calculate_cochran_n(params_B, T_in, 'static')
        n_Switch = calculate_cochran_n(params_B, T_in, 'switch', fallback=params_A, trigger=trig_in)
        n_Abandon = calculate_cochran_n(params_B, T_in, 'abandon', trigger=trig_in)
    
    st.toast(f"Optimiert: A={n_A}, B={n_B}, Switch={n_Switch}, Abandon={n_Abandon} Iterationen")

    # 4. Simulation Runs
    results_storage = {}
    progress_bar = st.progress(0)
    
    scenarios = [
        ("Scenario 1: Standard", n_A, params_A, 'static', None),
        ("Scenario 2: Fighter (Static)", n_B, params_B, 'static', None),
        ("Scenario 3: Switch Option", n_Switch, params_B, 'switch', params_A),
        ("Scenario 4: Abandon Option", n_Abandon, params_B, 'abandon', None)
    ]
    
    def get_rnd(v): return np.random.triangular(v[0], (v[0]+v[1])/2, v[1]) if isinstance(v, tuple) else v

    for idx, (name, n, p_ranges, mode, fallback_rng) in enumerate(scenarios):
        sim_sums = []; sim_inputs = []; all_N = []; all_W = []
        
        for _ in range(n):
            curr_p = {k: get_rnd(v) for k, v in p_ranges.items()}
            curr_fb = {k: get_rnd(v) for k, v in fallback_rng.items()} if fallback_rng else None
            N_trace, W_trace, total_val = run_simulation(**curr_p, start=1, T=T_in, mode=mode, trigger_val=trig_in, fallback_params=curr_fb)
            sim_sums.append(total_val); all_N.append(N_trace); all_W.append(W_trace); sim_inputs.append(curr_p)
            
        progress_bar.progress((idx + 1) / 4)
        
        # Analytics
        df_inputs = pd.DataFrame(sim_inputs)
        df_inputs_reg = df_inputs.loc[:, df_inputs.std() > 0]
        tornado_df, base_v = get_tornado_data(None, p_ranges, T_in, mode, trig_in, fallback_rng)
        reg_coefs, r2 = None, 0
        if not df_inputs_reg.empty: reg_coefs, r2 = get_regression_sensitivity(df_inputs_reg, sim_sums)

        results_storage[name] = {
            "n": n, "sums": sim_sums, "avg_N": np.mean(all_N, axis=0), "avg_W": np.mean(all_W, axis=0),
            "tornado": (tornado_df, base_v), "regression": (reg_coefs, r2),
            "mean": np.mean(sim_sums), "std": np.std(sim_sums), "var5": np.percentile(sim_sums, 5)
        }

    # 5. Speichern im Session State (Persistenz!)
    st.session_state.simulation_results = results_storage
    
    # 6. PDF Generierung im Hintergrund (damit es bereit ist)
    pdf_buffer = io.BytesIO()
    with PdfPages(pdf_buffer) as pdf:
        # Overview Plot
        fig_ov, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8.27, 11.69))
        for name, data in results_storage.items(): ax1.plot(data["avg_N"], label=name)
        ax1.set_title("Customer Adoption N(t)"); ax1.legend(); ax1.grid(True, alpha=0.3)
        for name, data in results_storage.items(): ax2.plot(data["avg_W"], label=name)
        ax2.set_title("Financial Value W(t)"); ax2.grid(True, alpha=0.3)
        names = list(results_storage.keys()); means = [results_storage[n]["mean"] for n in names]; stds = [results_storage[n]["std"] for n in names]
        ax3.bar(names, means, yerr=stds, capsize=5, alpha=0.7, color=['tab:blue', 'tab:orange', 'tab:green', 'tab:red'])
        ax3.set_title("Final Valuation Comparison"); ax3.set_ylabel("Total Value (€)")
        plt.tight_layout(); pdf.savefig(fig_ov); plt.close(fig_ov)

        # Details
        for key in results_storage.keys():
            res = results_storage[key]
            fig_det, (ax_t, ax_h, ax_r) = plt.subplots(3, 1, figsize=(8.27, 11.69))
            fig_det.suptitle(f"Detail Analysis: {key}", fontsize=16)
            
            df_t, base_v = res["tornado"]; y_p = np.arange(len(df_t))
            ax_t.barh(y_p, df_t["Low"], color='tab:red', alpha=0.6); ax_t.barh(y_p, df_t["High"], color='tab:green', alpha=0.6)
            ax_t.set_yticks(y_p); ax_t.set_yticklabels(df_t["Parameter"]); ax_t.invert_yaxis(); ax_t.axvline(0, color='k', ls='--')
            ax_t.set_title(f"Sensitivity (Tornado) - Base: {base_v/1e6:.2f}M")

            ax_h.hist(res["sums"], bins=40, color='skyblue', edgecolor='white')
            ax_h.axvline(res["mean"], color='k', ls='--', label=f"Mean: {res['mean']/1e6:.1f}M")
            ax_h.axvline(res["var5"], color='r', ls='--', label=f"VaR 5%: {res['var5']/1e6:.1f}M")
            ax_h.set_title(f"Risk Profile (n={res['n']})"); ax_h.legend()

            df_reg, r2 = res["regression"]
            if df_reg is not None:
                cols = ['tab:green' if c > 0 else 'tab:red' for c in df_reg["Beta"]]
                ax_r.barh(df_reg["Parameter"], df_reg["Beta"], color=cols)
                ax_r.set_title(f"Global Sensitivity (Beta) - R2={r2:.2f}")
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95]); pdf.savefig(fig_det); plt.close(fig_det)
    
    st.session_state.pdf_buffer = pdf_buffer

# ==========================================
# 5. RESULT DISPLAY (FROM STATE)
# ==========================================
if st.session_state.simulation_results is not None:
    res = st.session_state.simulation_results
    
    st.markdown("---")
    st.markdown("### 📊 Ergebnisse")

    # Tabs
    tabs = st.tabs(["Übersicht", "Detail 1", "Detail 2", "Detail 3", "Detail 4"])
    
    with tabs[0]:
        st.subheader("Strategy Comparison")
        fig_ov, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
        for name, data in res.items(): ax1.plot(data["avg_N"], label=name)
        ax1.set_title("Customer Adoption"); ax1.legend(); ax1.grid(True, alpha=0.3)
        
        names = list(res.keys()); means = [res[n]["mean"] for n in names]; stds = [res[n]["std"] for n in names]
        ax2.bar(names, means, yerr=stds, capsize=5, color=['tab:blue', 'tab:orange', 'tab:green', 'tab:red'], alpha=0.7)
        ax2.set_title("Mean Value & Volatility"); ax2.set_ylabel("EUR")
        st.pyplot(fig_ov)

    # Details
    keys = list(res.keys())
    for i, tab in enumerate(tabs[1:]):
        with tab:
            k = keys[i]; d = res[k]
            st.markdown(f"#### {k}")
            col_d1, col_d2 = st.columns(2)
            col_d1.metric("Mean Value", f"€ {d['mean']/1e6:.2f} M")
            col_d2.metric("VaR (5%)", f"€ {d['var5']/1e6:.2f} M")
            
            fig_d, ax_h = plt.subplots(figsize=(6, 3))
            ax_h.hist(d["sums"], bins=40, color='skyblue', edgecolor='white')
            ax_h.axvline(d["mean"], c='k', ls='--'); ax_h.axvline(d["var5"], c='r', ls='--')
            ax_h.set_title("Risk Profile")
            st.pyplot(fig_d)

    # PDF Download Button (Kritisch: Benutzt den Buffer aus dem State!)
    if st.session_state.pdf_buffer is not None:
        st.download_button(
            label="📄 PDF Report herunterladen",
            data=st.session_state.pdf_buffer.getvalue(),
            file_name=f"Sim_Report_{datetime.datetime.now().strftime('%H%M')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
