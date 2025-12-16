import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import io
import math

# ==========================================
# 1. CORE LOGIC & SIMULATION
# ==========================================
def run_simulation(M, p, q, C, ARPU, kappa, Delta_CM, Fixed_Cost, start, T, 
                   mode='static', trigger_val=0.05, fallback_params=None):
    """
    Führt EINE Simulation durch.
    mode: 'static', 'switch' (wechselt zu A), 'abandon' (bricht ab)
    """
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
            # Trigger Prüfung (Wachstum < X% von M)
            potential_acquisition = (p + q * (N_prev / M)) * (M - N_prev)
            trigger_threshold = trigger_val * M
            
            if potential_acquisition < trigger_threshold:
                option_exercised = True
                
                if mode == 'switch' and fallback_params:
                    # STRATEGIEWECHSEL: ALLE Parameter von Option A übernehmen
                    p = fallback_params['p']
                    q = fallback_params['q']
                    C = fallback_params['C']
                    ARPU = fallback_params['ARPU']
                    kappa = fallback_params['kappa']
                    Delta_CM = fallback_params['Delta_CM']
                    Fixed_Cost = fallback_params['Fixed_Cost']
                    M = fallback_params['M'] # Auch Marktpotenzial könnte sich theoretisch ändern
                    
                elif mode == 'abandon':
                    C = 1.0  # Totaler Churn
                    Fixed_Cost = 0 # Kostenstopp
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
# 2. STATISTIK HELPER (COCHRAN & SENSITIVITY)
# ==========================================
def calculate_cochran_n(params_dict, T, mode='static', fallback=None, trigger=0.05):
    """
    Führt Pilot-Simulation durch und berechnet n nach Cochran.
    n = (Z * s / E)^2
    """
    pilot_n = 200
    results = []
    
    # Pilot Runs
    for _ in range(pilot_n):
        # Zufallswerte ziehen
        curr = {}
        for k, v in params_dict.items():
            curr[k] = np.random.triangular(v[0], (v[0]+v[1])/2, v[1]) if isinstance(v, tuple) else v
            
        # Fallback Parameter auflösen falls Switch
        curr_fallback = None
        if fallback:
            curr_fallback = {}
            for k, v in fallback.items():
                curr_fallback[k] = np.random.triangular(v[0], (v[0]+v[1])/2, v[1]) if isinstance(v, tuple) else v

        _, _, val = run_simulation(**curr, start=1, T=T, mode=mode, trigger_val=trigger, fallback_params=curr_fallback)
        results.append(val)
    
    # Statistik
    std_dev = np.std(results)
    mean_val = np.mean(results)
    
    Z = 1.96 # 95% Konfidenz
    E = abs(mean_val * 0.05) # 5% Fehlertoleranz (Margin of Error)
    
    if E == 0: return 500 # Fallback falls Mean 0
    
    n_opt = (Z * std_dev / E) ** 2
    return int(math.ceil(n_opt))

def get_tornado_data(base_params, ranges, T, mode, trigger, fallback_ranges):
    """Berechnet Deterministic Sensitivity (Min/Max für jeden Parameter)"""
    data = []
    # Base Case berechnen (Mittelwerte)
    base_inputs = {k: (v[0]+v[1])/2 if isinstance(v, tuple) else v for k, v in ranges.items()}
    fallback_inputs = {k: (v[0]+v[1])/2 if isinstance(v, tuple) else v for k, v in fallback_ranges.items()} if fallback_ranges else None
    
    _, _, base_val = run_simulation(**base_inputs, start=1, T=T, mode=mode, trigger_val=trigger, fallback_params=fallback_inputs)
    
    for param, val_range in ranges.items():
        if not isinstance(val_range, tuple): continue # Fixe Werte überspringen
        
        # Low Case
        low_inputs = base_inputs.copy()
        low_inputs[param] = val_range[0]
        _, _, val_low = run_simulation(**low_inputs, start=1, T=T, mode=mode, trigger_val=trigger, fallback_params=fallback_inputs)
        
        # High Case
        high_inputs = base_inputs.copy()
        high_inputs[param] = val_range[1]
        _, _, val_high = run_simulation(**high_inputs, start=1, T=T, mode=mode, trigger_val=trigger, fallback_params=fallback_inputs)
        
        data.append({
            "Parameter": param,
            "Low": val_low - base_val,
            "High": val_high - base_val,
            "Range": abs(val_high - val_low)
        })
    
    return pd.DataFrame(data).sort_values(by="Range", ascending=True), base_val

def get_regression_sensitivity(df_inputs, y_values):
    """Berechnet Beta-Koeffizienten für globale Sensitivität"""
    X = df_inputs
    y = y_values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_scaled, y)
    
    coefs = pd.DataFrame({
        "Parameter": X.columns,
        "Beta": model.coef_
    }).sort_values(by="Beta", key=abs, ascending=True)
    
    r2 = model.score(X_scaled, y)
    return coefs, r2

# ==========================================
# 3. STREAMLIT GUI
# ==========================================
st.set_page_config(page_title="Master Thesis Valuation", layout="wide")
st.markdown("<h1 style='text-align: center;'>Valuing Digital Market Entry Strategies</h1>", unsafe_allow_html=True)

# --- GLOBALE EINSTELLUNGEN ---
with st.container():
    st.markdown("### 🌐 Globale & Methodische Einstellungen")
    c1, c2, c3, c4 = st.columns(4)
    with c1: T = st.slider("Zeitraum (T)", 5, 20, 15)
    with c2: M_global = st.number_input("Marktpotenzial (M)", 300, 5000, 500, step=50)
    with c3: trigger_input = st.slider("Trigger für Option C/D (< % Growth)", 0.01, 0.15, 0.05)
    with c4: 
        st.write("")
        start_btn = st.button("🚀 Analyse starten (Auto-Cochran)", type="primary", use_container_width=True)

st.markdown("---")

col_left, col_right = st.columns(2)

# --- INPUT PARAMETER (RANGES) ---
# Helper für Input Layout
def range_input(label, default_min, default_max, key_suffix, step=0.01, fmt="%.2f"):
    c1, c2 = st.columns(2)
    val_min = c1.number_input(f"{label} Min", value=default_min, step=step, format=fmt, key=f"{label}_min_{key_suffix}")
    val_max = c2.number_input(f"{label} Max", value=default_max, step=step, format=fmt, key=f"{label}_max_{key_suffix}")
    return (val_min, val_max)

with col_left:
    st.markdown("### 🔵 Option A: Standard (Basis & Fallback)")
    st.info("Parameter für konservative Strategie UND Switch-Ziel.")
    p_a = range_input("Innovation (p)", 0.005, 0.010, "a", 0.001, "%.3f")
    q_a = range_input("Imitation (q)", 0.15, 0.25, "a")
    c_a = range_input("Churn (C)", 0.03, 0.05, "a")
    arpu_a = range_input("ARPU (€)", 3800.0, 4200.0, "a", 100.0)
    fc_a = range_input("Fixkosten (€)", 140000.0, 160000.0, "a", 1000.0)
    kap_a = range_input("Kappa", 0.05, 0.10, "a")
    dcm_a = range_input("Delta Margin", 50.0, 100.0, "a", 10.0)

with col_right:
    st.markdown("### 🔴 Option B: Fighter (Startpunkt)")
    st.warning("Start-Parameter für aggressive Strategien (B, C, D).")
    p_b = range_input("Innovation (p)", 0.030, 0.050, "b", 0.001, "%.3f")
    q_b = range_input("Imitation (q)", 0.20, 0.30, "b")
    c_b = range_input("Churn (C)", 0.08, 0.12, "b")
    arpu_b = range_input("ARPU (€)", 3000.0, 3500.0, "b", 100.0)
    fc_b = range_input("Fixkosten (€)", 180000.0, 200000.0, "b", 1000.0)
    kap_b = range_input("Kappa", 0.10, 0.20, "b")
    dcm_b = range_input("Delta Margin", 50.0, 100.0, "b", 10.0)


# ==========================================
# 4. EXECUTION & REPORTING
# ==========================================
if start_btn:
    st.markdown("---")
    st.markdown("### 📊 Detaillierte Analyse & Report")
    
    # 1. Parameter Dictionaries packen (Ranges)
    params_A_ranges = {
        'M': (M_global*0.9, M_global*1.1), 'p': p_a, 'q': q_a, 'C': c_a, 
        'ARPU': arpu_a, 'kappa': kap_a, 'Delta_CM': dcm_a, 'Fixed_Cost': fc_a
    }
    params_B_ranges = {
        'M': (M_global*0.9, M_global*1.1), 'p': p_b, 'q': q_b, 'C': c_b, 
        'ARPU': arpu_b, 'kappa': kap_b, 'Delta_CM': dcm_b, 'Fixed_Cost': fc_b
    }

    # 2. Cochran Berechnungen (Automatische n)
    with st.spinner("Berechne optimale Sample Sizes nach Cochran..."):
        n_A = calculate_cochran_n(params_A_ranges, T, 'static')
        n_B = calculate_cochran_n(params_B_ranges, T, 'static')
        n_Switch = calculate_cochran_n(params_B_ranges, T, 'switch', fallback=params_A_ranges, trigger=trigger_input)
        n_Abandon = calculate_cochran_n(params_B_ranges, T, 'abandon', trigger=trigger_input)
    
    st.success(f"Optimale Iterationen berechnet (95% Conf, 5% Error): A={n_A}, B={n_B}, Switch={n_Switch}, Abandon={n_Abandon}")

    # 3. Main Monte Carlo Loop
    results_storage = {} # Zum Speichern aller Daten für PDF
    
    progress_bar = st.progress(0)
    total_steps = 4
    
    scenarios = [
        ("Scenario 1: Standard", n_A, params_A_ranges, 'static', None),
        ("Scenario 2: Fighter (Static)", n_B, params_B_ranges, 'static', None),
        ("Scenario 3: Switch Option", n_Switch, params_B_ranges, 'switch', params_A_ranges),
        ("Scenario 4: Abandon Option", n_Abandon, params_B_ranges, 'abandon', None)
    ]
    
    # Helper für Random Val aus Range
    def get_rnd(v): return np.random.triangular(v[0], (v[0]+v[1])/2, v[1]) if isinstance(v, tuple) else v

    for idx, (name, n, p_ranges, mode, fallback_rng) in enumerate(scenarios):
        
        sim_sums = []
        sim_inputs = [] # Für Regression
        
        # Arrays für Durchschnittskurven
        all_N = []
        all_W = []

        for _ in range(n):
            # Inputs ziehen
            curr_p = {k: get_rnd(v) for k, v in p_ranges.items()}
            curr_fb = {k: get_rnd(v) for k, v in fallback_rng.items()} if fallback_rng else None
            
            # Run Simulation
            N_trace, W_trace, total_val = run_simulation(**curr_p, start=1, T=T, mode=mode, trigger_val=trigger_input, fallback_params=curr_fb)
            
            sim_sums.append(total_val)
            all_N.append(N_trace)
            all_W.append(W_trace)
            sim_inputs.append(curr_p) # Eingaben speichern für Regression
            
        progress_bar.progress((idx + 1) / total_steps)
        
        # Daten verarbeiten
        df_inputs = pd.DataFrame(sim_inputs)
        # Für Regression müssen wir fixe Werte entfernen (StdDev=0)
        df_inputs_reg = df_inputs.loc[:, df_inputs.std() > 0] 
        
        # Sensitivitäten berechnen
        tornado_df, base_val_tornado = get_tornado_data(None, p_ranges, T, mode, trigger_input, fallback_rng)
        
        reg_coefs, r2 = None, 0
        if not df_inputs_reg.empty:
            reg_coefs, r2 = get_regression_sensitivity(df_inputs_reg, sim_sums)

        results_storage[name] = {
            "n": n,
            "sums": sim_sums,
            "avg_N": np.mean(all_N, axis=0),
            "avg_W": np.mean(all_W, axis=0),
            "tornado": (tornado_df, base_val_tornado),
            "regression": (reg_coefs, r2),
            "mean": np.mean(sim_sums),
            "std": np.std(sim_sums),
            "var5": np.percentile(sim_sums, 5)
        }

    # ==========================================
    # VISUALISIERUNG (WIE PDF)
    # ==========================================
    
    # TABS für Details
    tabs = st.tabs(["Übersicht", "Detail 1: Standard", "Detail 2: Fighter", "Detail 3: Switch", "Detail 4: Abandon"])
    
    # TAB 1: ÜBERSICHT
    with tabs[0]:
        st.subheader("Global Strategy Overview")
        
        # Comparison Plots
        fig_ov, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12))
        
        # N(t)
        for name, data in results_storage.items():
            ax1.plot(data["avg_N"], label=name)
        ax1.set_title("Customer Adoption N(t)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # W(t)
        for name, data in results_storage.items():
            ax2.plot(data["avg_W"], label=name)
        ax2.set_title("Financial Value W(t)")
        ax2.grid(True, alpha=0.3)
        
        # Bar Chart Mean + Error
        names = list(results_storage.keys())
        means = [results_storage[n]["mean"] for n in names]
        stds = [results_storage[n]["std"] for n in names]
        
        ax3.bar(names, means, yerr=stds, capsize=5, alpha=0.7, color=['tab:blue', 'tab:orange', 'tab:green', 'tab:red'])
        ax3.set_title("Final Valuation Comparison (Mean +/- StdDev)")
        ax3.set_ylabel("Total Value (€)")
        for i, v in enumerate(means):
            ax3.text(i, v, f"{v/1e6:.1f}M", ha='center', va='bottom', fontweight='bold')

        st.pyplot(fig_ov)

    # DETAIL TABS (Generisch)
    scenario_keys = list(results_storage.keys())
    
    for i, tab in enumerate(tabs[1:]):
        key = scenario_keys[i]
        res = results_storage[key]
        
        with tab:
            st.markdown(f"### Detail Analysis: {key}")
            
            # 3 Plots untereinander wie im PDF
            fig_det, (ax_torn, ax_hist, ax_reg) = plt.subplots(3, 1, figsize=(8, 14))
            
            # 1. Tornado Plot
            df_torn, base_v = res["tornado"]
            y_pos = np.arange(len(df_torn))
            ax_torn.barh(y_pos, df_torn["Low"], color='tab:red', alpha=0.6, label='Low Input')
            ax_torn.barh(y_pos, df_torn["High"], color='tab:green', alpha=0.6, label='High Input')
            ax_torn.set_yticks(y_pos)
            ax_torn.set_yticklabels(df_torn["Parameter"])
            ax_torn.invert_yaxis()
            ax_torn.axvline(0, color='black', linestyle='--')
            ax_torn.set_title(f"Sensitivity (Tornado) - Base: € {base_v/1e6:.2f}M")
            ax_torn.legend()

            # 2. Risk Profile (Histogram)
            ax_hist.hist(res["sums"], bins=30, alpha=0.6, color='tab:blue', edgecolor='white')
            ax_hist.axvline(res["mean"], color='black', linestyle='--', label=f"Mean: {res['mean']/1e6:.1f}M")
            ax_hist.axvline(res["var5"], color='red', linestyle='--', label=f"VaR 5%: {res['var5']/1e6:.1f}M")
            ax_hist.set_title(f"Risk Profile (n={res['n']})")
            ax_hist.legend()

            # 3. Regression Beta
            df_reg, r2 = res["regression"]
            if df_reg is not None:
                colors = ['tab:green' if c > 0 else 'tab:red' for c in df_reg["Beta"]]
                ax_reg.barh(df_reg["Parameter"], df_reg["Beta"], color=colors)
                ax_reg.set_title(f"Global Sensitivity (Beta) - R²={r2:.2f}")
                ax_reg.axvline(0, color='black', linestyle='--')
                for idx, v in enumerate(df_reg["Beta"]):
                    ax_reg.text(v, idx, f" {v:.2f}", va='center', fontsize=9)
            else:
                ax_reg.text(0.5, 0.5, "Insufficient variance for regression", ha='center')

            st.pyplot(fig_det)

    # ==========================================
    # PDF EXPORT GENERIERUNG
    # ==========================================
    pdf_buffer = io.BytesIO()
    with PdfPages(pdf_buffer) as pdf:
        # Page 1: Overview
        pdf.savefig(fig_ov)
        
        # Pages 2-5: Details
        # Wir müssen die Figuren leider neu erstellen für das PDF, da Matplotlib stateful ist
        # oder wir nutzen die bereits erstellten Figuren (da wir sie oben erstellt haben).
        # Im Loop oben haben wir "fig_det" aber überschrieben.
        # Lösung: Wir iterieren nochmal kurz für den PDF Export (schnell, da Daten berechnet sind)
        
        for key in scenario_keys:
            res = results_storage[key]
            fig_pdf, (ax_t, ax_h, ax_r) = plt.subplots(3, 1, figsize=(8.27, 11.69)) # A4 Größe
            fig_pdf.suptitle(f"Detail Analysis: {key}", fontsize=16)
            
            # Tornado Re-Draw
            df_t, base_v = res["tornado"]
            y_p = np.arange(len(df_t))
            ax_t.barh(y_p, df_t["Low"], color='tab:red', alpha=0.6); ax_t.barh(y_p, df_t["High"], color='tab:green', alpha=0.6)
            ax_t.set_yticks(y_p); ax_t.set_yticklabels(df_t["Parameter"]); ax_t.invert_yaxis(); ax_t.axvline(0, color='k', ls='--')
            ax_t.set_title(f"Sensitivity (Tornado) - Base: {base_v/1e6:.2f}M")

            # Hist Re-Draw
            ax_h.hist(res["sums"], bins=40, color='skyblue', edgecolor='white')
            ax_h.axvline(res["mean"], color='k', ls='--', label=f"Mean: {res['mean']/1e6:.1f}M")
            ax_h.axvline(res["var5"], color='r', ls='--', label=f"VaR 5%: {res['var5']/1e6:.1f}M")
            ax_h.set_title("Risk Profile (Monte Carlo)"); ax_h.legend()

            # Reg Re-Draw
            df_reg, r2 = res["regression"]
            if df_reg is not None:
                cols = ['tab:green' if c > 0 else 'tab:red' for c in df_reg["Beta"]]
                ax_r.barh(df_reg["Parameter"], df_reg["Beta"], color=cols)
                ax_r.set_title(f"Global Sensitivity (Beta) - R2={r2:.2f}")
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            pdf.savefig(fig_pdf)
            plt.close(fig_pdf)
        
        # Info Page
        fig_info, ax_i = plt.subplots(figsize=(8.27, 11.69))
        ax_i.axis('off')
        txt = "SIMULATION REPORT\n\nMethodology:\n"
        txt += "- Sample Size: Calculated via Cochran's Formula (95% Conf, 5% Error)\n"
        txt += "- Switch Option: Triggers full parameter replacement to 'Standard' Strategy\n"
        txt += "- Abandon Option: Stops operational costs, churn set to 100%\n\n"
        txt += "Scenario Iterations (n):\n"
        for k, v in results_storage.items():
            txt += f"- {k}: {v['n']} runs\n"
        ax_i.text(0.1, 0.8, txt, family='monospace', fontsize=12, va='top')
        pdf.savefig(fig_info)

    st.download_button("📄 Vollständigen PDF Report herunterladen", pdf_buffer.getvalue(), "Simulation_Report_Full.pdf", "application/pdf", use_container_width=True)
