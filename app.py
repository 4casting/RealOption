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
# 0. STREAMLIT KONFIGURATION
# ==========================================
st.set_page_config(page_title="Master Thesis Valuation", layout="wide")

if 'history' not in st.session_state: st.session_state.history = []
if 'simulation_results' not in st.session_state: st.session_state.simulation_results = None
if 'pdf_buffer' not in st.session_state: st.session_state.pdf_buffer = None

# ==========================================
# 1. KERN-LOGIK (SIMULATION MIT KOHORTEN)
# ==========================================
def run_simulation(M, p, q, C, ARPU, kappa, Delta_CM, Fixed_Cost, start, T, 
                   mode='static', trigger_val=0.05, fallback_params=None,
                   check_mode='continuous', check_year=3, growth_metric='share_of_m',
                   switch_config=None):
    
    # Initialsierung der Cohort Matrix
    # Zeile = Startjahr, Spalte = Laufzeitjahr
    cohorts = np.zeros((T, T))
    cohorts[0, 0] = start 

    N = [0.0] * T
    W = [0.0] * T
    N[0] = start
    W[0] = N[0] * ARPU - Fixed_Cost

    option_exercised = False
    project_is_dead = False
    
    curr_p, curr_q, curr_C = p, q, C
    curr_ARPU, curr_kappa, curr_Delta_CM = ARPU, kappa, Delta_CM
    curr_FC, curr_M = Fixed_Cost, M

    growth_history = []

    for t in range(1, T):
        if project_is_dead:
            N[t] = 0.0; W[t] = 0.0
            continue 

        # Bestand aus Vorperiode berechnen
        N_prev = np.sum(cohorts[:, t-1])
        
        # --- PROGNOSE ---
        potential_acquisition = (curr_p + curr_q * (N_prev / curr_M)) * (curr_M - N_prev)
        if potential_acquisition < 0: potential_acquisition = 0
        
        if growth_metric == 'share_of_m':
            current_rate = potential_acquisition / curr_M
        else:
            current_rate = (potential_acquisition / N_prev) if N_prev > 0 else 0.0

        # --- TRIGGER PRÜFUNG & OPTIONEN ---
        shock_applied = 0.0 

        if mode != 'static' and not option_exercised:
            is_check_time = False
            if check_mode == 'specific' and t == check_year: is_check_time = True
            elif check_mode == 'continuous' and t >= check_year: is_check_time = True
            
            if is_check_time:
                all_rates = growth_history + [current_rate]
                avg_growth = sum(all_rates) / len(all_rates) if all_rates else 0
                
                if avg_growth < trigger_val:
                    option_exercised = True
                    
                    if mode == 'switch' and fallback_params and switch_config:
                        # Switch Logic
                        target_ARPU = fallback_params['ARPU']
                        delta_p = (target_ARPU - curr_ARPU) / curr_ARPU if curr_ARPU > 0 else 0.0
                        
                        if delta_p <= switch_config['thresh_low']: zone_prefix = 'zone1'
                        elif delta_p <= switch_config['thresh_high']: zone_prefix = 'zone2'
                        else: zone_prefix = 'zone3'
                        
                        use_gf = switch_config['grandfathering']
                        suffix = "_gf" if use_gf else "_nogf"
                        
                        shock_factor = switch_config[f'shock_{zone_prefix}{suffix}']
                        q_multiplier = switch_config[f'q_mult_{zone_prefix}{suffix}']
                        
                        curr_p = fallback_params['p']
                        curr_C = fallback_params['C']
                        curr_ARPU = fallback_params['ARPU']
                        curr_kappa = fallback_params['kappa']
                        curr_Delta_CM = fallback_params['Delta_CM']
                        curr_FC = fallback_params['Fixed_Cost']
                        
                        base_target_q = fallback_params['q']
                        curr_q = base_target_q * q_multiplier 
                        
                        shock_applied = shock_factor
                        
                        # Simuliere Schock für Akquise-Berechnung
                        N_prev_sim = N_prev * (1.0 - shock_applied)
                        potential_acquisition = (curr_p + curr_q * (N_prev_sim / curr_M)) * (curr_M - N_prev_sim)
                        if potential_acquisition < 0: potential_acquisition = 0
                        
                    elif mode == 'abandon':
                        project_is_dead = True
                        N[t] = 0.0; W[t] = 0.0
                        continue

        growth_history.append(current_rate)

        # --- UPDATE KOHORTEN ---
        # 1. Retention auf alte Kohorten
        eff_retention = (1.0 - shock_applied) * (1.0 - curr_C)
        for start_year in range(t):
            cohorts[start_year, t] = cohorts[start_year, t-1] * eff_retention
            
        # 2. Neue Kohorte (New Adopters)
        cohorts[t, t] = potential_acquisition
        
        # 3. Summe
        N[t] = np.sum(cohorts[:, t])
        if N[t] > curr_M: N[t] = curr_M
        
        revenue = N[t] * curr_ARPU
        cannib = potential_acquisition * curr_kappa * curr_Delta_CM
        W[t] = revenue - cannib - curr_FC
    
    return N, W, sum(W), option_exercised, cohorts

# ==========================================
# 2. STATISTIK HELPER
# ==========================================
def calculate_cochran_n(params_dict, T, mode='static', fallback=None, trigger=0.05, 
                        c_mode='continuous', c_year=3, g_metric='share_of_m', sw_conf=None):
    pilot_n = 200
    results = []
    def get_val(v): return np.random.triangular(v[0], (v[0]+v[1])/2, v[1]) if isinstance(v, tuple) else v
    for _ in range(pilot_n):
        curr = {k: get_val(v) for k, v in params_dict.items()}
        curr_fb = {k: get_val(v) for k, v in fallback.items()} if fallback else None
        # Unpacking 5 values
        _, _, val, _, _ = run_simulation(**curr, start=1, T=T, mode=mode, trigger_val=trigger, 
                                        fallback_params=curr_fb, check_mode=c_mode, check_year=c_year, 
                                        growth_metric=g_metric, switch_config=sw_conf)
        results.append(val)
    std = np.std(results); mean = np.mean(results)
    if mean == 0: return 1000
    E = abs(mean * 0.01) 
    if E == 0: return 1000
    n = (1.96 * std / E) ** 2
    return max(int(math.ceil(n)), 1000)

def get_tornado_data(base_params, ranges, T, mode, trigger, fallback_ranges, c_mode, c_year, g_metric, sw_conf):
    def mid(v): return (v[0]+v[1])/2 if isinstance(v, tuple) else v
    base_inputs = {k: mid(v) for k, v in ranges.items()}
    fb_inputs = {k: mid(v) for k, v in fallback_ranges.items()} if fallback_ranges else None
    
    # Helper um nur den NPV zu holen
    def run_sim_val(in_p):
        return run_simulation(**in_p, start=1, T=T, mode=mode, trigger_val=trigger, 
                              fallback_params=fb_inputs, check_mode=c_mode, check_year=c_year, 
                              growth_metric=g_metric, switch_config=sw_conf)[2]
    
    base_val = run_sim_val(base_inputs)
    data = []
    for param, val_range in ranges.items():
        if not isinstance(val_range, tuple): continue
        low_inputs = base_inputs.copy(); low_inputs[param] = val_range[0]
        high_inputs = base_inputs.copy(); high_inputs[param] = val_range[1]
        
        v_low = run_sim_val(low_inputs)
        v_high = run_sim_val(high_inputs)
        
        data.append({"Parameter": param, "Low": v_low - base_val, "High": v_high - base_val, "Range": abs(v_high - v_low)})
    return pd.DataFrame(data).sort_values(by="Range", ascending=True), base_val

def get_regression_sensitivity(df_inputs, y_values):
    scaler = StandardScaler(); X_scaled = scaler.fit_transform(df_inputs)
    model = LinearRegression(); model.fit(X_scaled, y_values)
    return pd.DataFrame({"Parameter": df_inputs.columns, "Beta": model.coef_}).sort_values(by="Beta", key=abs, ascending=True), model.score(X_scaled, y_values)

# ==========================================
# 3. NAVIGATION & SEITEN
# ==========================================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Menü:", ["Simulation & Analyse", "Modell-Beschreibung"])

# --- SEITE: MODELL-BESCHREIBUNG ---
if page == "Modell-Beschreibung":
    st.title("Modellbeschreibung")
    st.markdown("""
    Diese Simulation basiert auf einem integrierten Bewertungsrahmen ("Integrated Valuation Framework"), der das **Synthesized Bass Diffusion Model** mit der **Real Options Analysis (ROA)** kombiniert.
    """)
    st.header("1. Mathematischer Kern")
    st.subheader("A. Kundenwachstum")
    st.markdown(r"""
    $$N(t) = N(t-1) \cdot (1-C) + \left( p + q \cdot \frac{N(t-1)}{M} \right) \cdot (M - N(t-1))$$
    """)
    st.subheader("B. Finanzielle Bewertung")
    st.markdown(r"""
    $$W(t) = (N(t) \cdot ARPU) - (\Delta N(t) \cdot \kappa \cdot \Delta CM) - \text{Fixed Costs}$$
    """)

# --- SEITE: SIMULATION & ANALYSE ---
elif page == "Simulation & Analyse":
    
    with st.sidebar:
        st.markdown("---")
        st.header("Verlauf")
        def restore():
            idx = st.session_state.hist_sel
            if idx is not None:
                entry = st.session_state.history[idx]
                for k, v in entry['params'].items(): st.session_state[k] = v
                st.toast(f"Wiederhergestellt: {entry['timestamp']}")
        if st.session_state.history:
            opts = {i: f"{e['timestamp']} (M={e['params'].get('M_val', '?')})" for i, e in enumerate(st.session_state.history)}
            st.selectbox("Frühere Eingaben laden:", list(opts.keys()), format_func=lambda x: opts[x], key="hist_sel", index=None, on_change=restore)

    st.markdown("<h1 style='text-align: center;'>Valuing Market Entry Pricing-Strategies and Real Options</h1>", unsafe_allow_html=True)

    with st.container():
        st.markdown("### Globale Settings")
        c1, c2, c3, c4 = st.columns([1, 1, 2, 1])
        with c1: T_in = st.slider("Laufzeit (T)", 5, 30, 30, key="T_val")
        with c2: M_in = st.number_input("Marktpotenzial (M)", 300, 10000, 500, step=50, key="M_val")
        with c3:
            st.markdown("**Option Trigger**")
            check_mode_in = st.selectbox("Wann prüfen?", ["specific", "continuous"], 
                                         format_func=lambda x: "Einmalig" if x == "specific" else "Fortlaufend", key="check_mode_sel")
            metric_in = st.selectbox("Metrik", ["share_of_m", "relative"], 
                                     format_func=lambda x: "Marktdurchdringung" if x == "share_of_m" else "Relatives Wachstum", key="metric_sel")
            c3_1, c3_2 = st.columns(2)
            with c3_1: check_year_in = st.number_input("Start-Periode", 1, T_in, 3, key="check_year_val")
            with c3_2: 
                mx = 0.5 if metric_in == "share_of_m" else 2.0
                def_v = 0.03 if metric_in == "share_of_m" else 0.15
                trig_val_in = st.slider("Grenzwert (<)", 0.0, mx, def_v, step=0.01, key="trig_val")
        with c4: 
            st.write(""); st.write("")
            start_btn = st.button("Simulation starten", type="primary", use_container_width=True)

    with st.expander("⚙️ Konfiguration: Switch Matrix", expanded=False):
        gf_active = st.checkbox("Grandfathering anwenden?", value=False, key="gf_active")
        col_th1, col_th2 = st.columns(2)
        with col_th1: thresh_low = st.number_input("Grenze Sicherheitszone", 0.0, 1.0, 0.10, step=0.05, key="th_low")
        with col_th2: thresh_high = st.number_input("Grenze Gefahrenzone", 0.0, 1.0, 0.20, step=0.05, key="th_high")
        
        col_z1, col_z2, col_z3 = st.columns(3)
        def zone_inputs(col, title, prefix, def_shock, def_q):
            with col:
                st.markdown(f"**{title}**")
                s_no = st.number_input(f"Churn {prefix}", 0.0, 1.0, def_shock, key=f"s_no_{prefix}")
                q_no = st.number_input(f"q-Faktor {prefix}", 0.0, 1.5, def_q, key=f"q_no_{prefix}")
                s_gf = st.number_input(f"Churn {prefix} (GF)", 0.0, 1.0, 0.0, key=f"s_gf_{prefix}", disabled=True)
                q_gf = st.number_input(f"q-Faktor {prefix} (GF)", 0.0, 1.5, def_q, key=f"q_gf_{prefix}")
                return s_no, q_no, s_gf, q_gf

        s1_no, q1_no, s1_gf, q1_gf = zone_inputs(col_z1, "Sicherheitszone", "1", 0.02, 1.0)
        s2_no, q2_no, s2_gf, q2_gf = zone_inputs(col_z2, "Warnzone", "2", 0.10, 0.8)
        s3_no, q3_no, s3_gf, q3_gf = zone_inputs(col_z3, "Gefahrenzone", "3", 0.30, 0.5)

        switch_config_dict = {
            'grandfathering': gf_active,
            'thresh_low': thresh_low, 'thresh_high': thresh_high,
            'shock_zone1_nogf': s1_no, 'q_mult_zone1_nogf': q1_no, 'shock_zone1_gf': s1_gf, 'q_mult_zone1_gf': q1_gf,
            'shock_zone2_nogf': s2_no, 'q_mult_zone2_nogf': q2_no, 'shock_zone2_gf': s2_gf, 'q_mult_zone2_gf': q2_gf,
            'shock_zone3_nogf': s3_no, 'q_mult_zone3_nogf': q3_no, 'shock_zone3_gf': s3_gf, 'q_mult_zone3_gf': q3_gf,
        }

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

    # --- SIMULATION ---
    if start_btn:
        snap = {k: v for k, v in st.session_state.items() if "_min_" in k or "_max_" in k or k in ["T_val", "M_val", "trig_val", "gf_active"]}
        st.session_state.history.append({'timestamp': datetime.datetime.now().strftime("%H:%M:%S"), 'params': snap})
        
        params_A = {'M': (M_in*0.9, M_in*1.1), 'p': p_a, 'q': q_a, 'C': c_a, 'ARPU': arpu_a, 'kappa': kap_a, 'Delta_CM': dcm_a, 'Fixed_Cost': fc_a}
        params_B = {'M': (M_in*0.9, M_in*1.1), 'p': p_b, 'q': q_b, 'C': c_b, 'ARPU': arpu_b, 'kappa': kap_b, 'Delta_CM': dcm_b, 'Fixed_Cost': fc_b}
        
        with st.spinner("Berechne..."):
            n_A = calculate_cochran_n(params_A, T_in, 'static')
            n_B = calculate_cochran_n(params_B, T_in, 'static')
            n_Sw = calculate_cochran_n(params_B, T_in, 'switch', fallback=params_A, trigger=trig_val_in, c_mode=check_mode_in, c_year=check_year_in, g_metric=metric_in, sw_conf=switch_config_dict)
            n_Ab = calculate_cochran_n(params_B, T_in, 'abandon', trigger=trig_val_in, c_mode=check_mode_in, c_year=check_year_in, g_metric=metric_in, sw_conf=switch_config_dict)
        
        res_store = {}; bar = st.progress(0)
        scenarios = [("1. Standard (A)", n_A, params_A, 'static', None, 'tab:blue'),
                     ("2. Fighter (B)", n_B, params_B, 'static', None, 'tab:red'),
                     ("3. Switch Option", n_Sw, params_B, 'switch', params_A, 'tab:green'),
                     ("4. Abandon Option", n_Ab, params_B, 'abandon', None, 'black')]
        
        def rnd(v): return np.random.triangular(v[0], (v[0]+v[1])/2, v[1]) if isinstance(v, tuple) else v

        for idx, (name, n, p_rng, mode, fb_rng, col) in enumerate(scenarios):
            sim_sums, sim_inputs, all_N, all_W = [], [], [], []
            avg_cohorts = np.zeros((T_in, T_in))
            
            exercised_count = 0
            for _ in range(n):
                curr = {k: rnd(v) for k, v in p_rng.items()}
                fb = {k: rnd(v) for k, v in fb_rng.items()} if fb_rng else None
                
                N_t, W_t, tot, exc, coh_mat = run_simulation(**curr, start=1, T=T_in, mode=mode, trigger_val=trig_val_in, 
                                                    fallback_params=fb, check_mode=check_mode_in, check_year=check_year_in, 
                                                    growth_metric=metric_in, switch_config=switch_config_dict)
                sim_sums.append(tot); all_N.append(N_t); all_W.append(W_t); sim_inputs.append(curr)
                
                avg_cohorts += coh_mat
                if exc: exercised_count += 1
            
            avg_cohorts /= n
            bar.progress((idx+1)/4)
            
            arr_N = np.array(all_N); arr_W = np.array(all_W)
            p5_N = np.percentile(arr_N, 5, axis=0); p95_N = np.percentile(arr_N, 95, axis=0)
            p5_W = np.percentile(arr_W, 5, axis=0); p95_W = np.percentile(arr_W, 95, axis=0)
            
            df_in = pd.DataFrame(sim_inputs); df_in_reg = df_in.loc[:, df_in.std() > 0]
            torn, base_v = get_tornado_data(None, p_rng, T_in, mode, trig_val_in, fb_rng, check_mode_in, check_year_in, metric_in, switch_config_dict)
            reg, r2 = (None, 0)
            if not df_in_reg.empty: reg, r2 = get_regression_sensitivity(df_in_reg, sim_sums)

            res_store[name] = {
                "n": n, "sums": sim_sums, "avg_N": np.mean(all_N, axis=0), "avg_W": np.mean(all_W, axis=0),
                "p5_N": p5_N, "p95_N": p95_N, "p5_W": p5_W, "p95_W": p95_W,
                "cohorts": avg_cohorts, 
                "tornado": (torn, base_v), "regression": (reg, r2), "color": col,
                "mean": np.mean(sim_sums), "std": np.std(sim_sums), 
                "min": np.min(sim_sums), "max": np.max(sim_sums), "var5": np.percentile(sim_sums, 5),
                "exercise_rate": (exercised_count / n) * 100
            }
        st.session_state.simulation_results = res_store

        # --- PDF GENERATOR ---
        buf = io.BytesIO()
        with PdfPages(buf) as pdf:
            # SEITE 1: GRAFIKEN
            fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8.27, 11.69))
            for n, d in res_store.items():
                ax1.plot(d["avg_N"], label=n, color=d["color"])
                ax1.fill_between(range(len(d["avg_N"])), d["p5_N"], d["p95_N"], color=d["color"], alpha=0.1)
            ax1.set_title(f"Customer Adoption (Metric: {metric_in})"); ax1.legend(); ax1.grid(True, alpha=0.1)
            
            for n, d in res_store.items():
                ax2.plot(d["avg_W"], label=n, color=d["color"])
                ax2.fill_between(range(len(d["avg_W"])), d["p5_W"], d["p95_W"], color=d["color"], alpha=0.1)
            ax2.set_title("Net Value Contribution"); ax2.grid(True, alpha=0.1)
            
            nms = list(res_store.keys()); mus = [res_store[n]["mean"] for n in nms]; sigs = [res_store[n]["std"] for n in nms]
            cols = [res_store[n]["color"] for n in nms]
            bars = ax3.bar(nms, mus, yerr=sigs, capsize=5, alpha=0.7, color=cols)
            ax3.bar_label(bars, fmt='€ %d', padding=3); ax3.set_title("Total Value Comparison"); ax3.set_ylabel("EUR")
            plt.tight_layout(); pdf.savefig(fig1); plt.close(fig1)

            # SEITE 2: TABELLE
            fig_tab, ax_tab = plt.subplots(figsize=(8.27, 11.69)); ax_tab.axis('off')
            table_data = [["Scenario", "Runs", "Mean (€)", "StdDev (€)", "VaR 5% (€)", "Exercise %"]]
            for n, d in res_store.items():
                table_data.append([n, str(d['n']), f"{d['mean']:,.0f}", f"{d['std']:,.0f}", f"{d['var5']:,.0f}", f"{d['exercise_rate']:.1f}%"])
            table = ax_tab.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.3, 0.1, 0.15, 0.15, 0.15, 0.15])
            table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 2)
            ax_tab.set_title("Simulation Results Summary", fontweight='bold'); pdf.savefig(fig_tab); plt.close(fig_tab)

            # SEITE 3: INPUTS
            fig_inp, ax_inp = plt.subplots(figsize=(8.27, 11.69)); ax_inp.axis('off')
            info_text = f"T: {T_in}, M: {M_in}, Trig: {metric_in} < {trig_val_in}"
            ax_inp.text(0.05, 0.95, info_text, transform=ax_inp.transAxes)
            pdf.savefig(fig_inp); plt.close(fig_inp)

            # SEITEN 4+: DETAILS
            for k, d in res_store.items():
                fig, (ax_t, ax_h, ax_r) = plt.subplots(3, 1, figsize=(8.27, 11.69))
                fig.suptitle(f"Detail: {k}", fontsize=16)
                
                df_t, b_v = d["tornado"]
                if not df_t.empty and df_t['Range'].sum() > 0:
                    y = np.arange(len(df_t))
                    ax_t.barh(y, df_t["Low"], color='tab:red', alpha=0.6); ax_t.barh(y, df_t["High"], color='tab:green', alpha=0.6)
                    ax_t.set_yticks(y); ax_t.set_yticklabels(df_t["Parameter"]); ax_t.invert_yaxis(); ax_t.axvline(0, c='k', ls='--')
                
                ax_h.hist(d["sums"], bins=40, color='skyblue', edgecolor='white'); ax_h.set_title("NPV Risk Profile")
                
                df_r, r2 = d["regression"]
                if df_r is not None:
                    clrs = ['tab:green' if c > 0 else 'tab:red' for c in df_r["Beta"]]
                    ax_r.barh(df_r["Parameter"], df_r["Beta"], color=clrs)
                plt.tight_layout(rect=[0, 0.03, 1, 0.95]); pdf.savefig(fig); plt.close(fig)

            # NEU: ANHANG - COHORT ANALYSES
            for k, d in res_store.items():
                if 'cohorts' in d: 
                    fig_c, ax_c = plt.subplots(figsize=(8.27, 6))
                    cohort_data = d['cohorts']
                    years = np.arange(cohort_data.shape[1])
                    lbls = [f"Y{i}" if i % 2 == 0 else "" for i in range(len(years))]
                    pal = plt.cm.viridis(np.linspace(0, 1, len(years)))
                    ax_c.stackplot(years, cohort_data, labels=lbls, colors=pal, alpha=0.8)
                    ax_c.set_title(f"Appendix: Cohort Analysis - {k}")
                    ax_c.set_ylabel("Customers"); ax_c.grid(True, alpha=0.3)
                    pdf.savefig(fig_c); plt.close(fig_c)

            # NEU: ANHANG - ADOPTION CURVES (Bell Curves)
            for k, d in res_store.items():
                if 'cohorts' in d:
                    fig_g, ax_g = plt.subplots(figsize=(8.27, 6))
                    # Diagonal der Kohorten-Matrix = Neu gewonnene Kunden pro Jahr
                    adoption_curve = d['cohorts'].diagonal()
                    years = np.arange(len(adoption_curve))
                    
                    ax_g.plot(years, adoption_curve, marker='o', color=d['color'], linewidth=2)
                    ax_g.fill_between(years, 0, adoption_curve, color=d['color'], alpha=0.2)
                    
                    ax_g.set_title(f"Appendix: Adoption Curve (New Customers) - {k}")
                    ax_g.set_xlabel("Period (Year)"); ax_g.set_ylabel("New Acquired Customers")
                    ax_g.grid(True, alpha=0.3)
                    pdf.savefig(fig_g); plt.close(fig_g)

        st.session_state.pdf_buffer = buf

    # --- ANZEIGE ---
    if st.session_state.simulation_results:
        res = st.session_state.simulation_results
        st.markdown("### Ergebnisse")
        
        summary_data = []
        for k, d in res.items():
            summary_data.append({
                "Szenario": k, "Runs": d['n'], "Mean (€)": f"{d['mean']:,.0f}", 
                "StdDev (€)": f"{d['std']:,.0f}", "VaR 5% (€)": f"{d['var5']:,.0f}", 
                "Ausübung %": f"{d['exercise_rate']:.1f}%"
            })
        st.dataframe(pd.DataFrame(summary_data).set_index("Szenario"), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Customer Evolution")
            fig_n, ax_n = plt.subplots(figsize=(6, 4))
            for n, d in res.items():
                ax_n.plot(d["avg_N"], label=n, color=d["color"])
                ax_n.fill_between(range(len(d["avg_N"])), d["p5_N"], d["p95_N"], color=d["color"], alpha=0.1)
            ax_n.legend(); ax_n.grid(True, alpha=0.3); st.pyplot(fig_n)
        
        with c2:
            st.markdown("#### Net Value Contribution")
            fig_w, ax_w = plt.subplots(figsize=(6, 4))
            for n, d in res.items():
                ax_w.plot(d["avg_W"], label=n, color=d["color"])
                ax_w.fill_between(range(len(d["avg_W"])), d["p5_W"], d["p95_W"], color=d["color"], alpha=0.1)
            ax_w.legend(); ax_w.grid(True, alpha=0.3); st.pyplot(fig_w)

        if st.session_state.pdf_buffer:
            st.download_button("📄 PDF Report Download (inkl. Kohorten & Adoptionskurven)", st.session_state.pdf_buffer.getvalue(), 
                               f"Report_{datetime.datetime.now().strftime('%H%M')}.pdf", "application/pdf", use_container_width=True)

        # ==========================================
        # DEEP DIVE SECTION
        # ==========================================
        st.markdown("---")
        st.header("📊 Deep Dive Analyse")
        
        avail_scenarios = list(res.keys())
        def_idx = 2 if len(avail_scenarios) > 2 else 0
        selected_scen = st.selectbox("Wähle Strategie für Detail-Analyse:", avail_scenarios, index=def_idx)
        
        # Robust check
        if 'cohorts' not in res[selected_scen]:
            st.warning("⚠️ Bitte Simulation neu starten, um die neuen Grafiken zu laden.")
        else:
            col_deep1, col_deep2 = st.columns(2)
            
            with col_deep1:
                st.subheader("1. Kohorten-Analyse (Stacked)")
                cohort_data = res[selected_scen]['cohorts']
                fig_coh, ax_coh = plt.subplots(figsize=(6, 4))
                years = np.arange(cohort_data.shape[1])
                lbls = [f"Y{i}" if i % 2 == 0 else "" for i in range(len(years))]
                pal = plt.cm.viridis(np.linspace(0, 1, len(years)))
                ax_coh.stackplot(years, cohort_data, labels=lbls, colors=pal, alpha=0.8)
                ax_coh.set_title("Active Customers by Cohort")
                ax_coh.set_xlabel("Year"); ax_coh.set_ylabel("Count")
                ax_coh.grid(True, alpha=0.3)
                st.pyplot(fig_coh)

            with col_deep2:
                st.subheader("2. Adoptions-Kurve (Glockenkurve)")
                # Diagonal = Neue Kunden pro Jahr
                adoption_curve = res[selected_scen]['cohorts'].diagonal()
                years = np.arange(len(adoption_curve))
                
                fig_bell, ax_bell = plt.subplots(figsize=(6, 4))
                ax_bell.plot(years, adoption_curve, marker='o', linewidth=2, color='tab:orange', label="New Customers")
                ax_bell.fill_between(years, 0, adoption_curve, color='tab:orange', alpha=0.2)
                
                ax_bell.set_title("New Customers acquired per Year")
                ax_bell.set_xlabel("Period (Year)")
                ax_bell.set_ylabel("Count")
                ax_bell.legend(); ax_bell.grid(True, alpha=0.3)
                st.pyplot(fig_bell)
                
            if "Switch" in selected_scen:
                st.info("💡 Hinweis: Der Verlauf der Kurven zeigt direkt den Einfluss der Strategie-Änderung.")
