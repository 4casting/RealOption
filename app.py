import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import io

# ==========================================
# 1. MATHEMATISCHES MODELL (CORE LOGIC)
# ==========================================
def run_simulation(M, p, q, C, ARPU, kappa, Delta_CM, Fixed_Cost, start, T, 
                   real_option_active=False, trigger_val=0.05, fallback_params=None):
    """
    Führt EINE Simulation durch.
    fallback_params: Dict mit Werten von Option A (für den Switch).
    """
    N = [0.0] * T
    W = [0.0] * T
    N[0] = start
    W[0] = N[0] * ARPU - Fixed_Cost

    check_period_start = 3
    # Hinweis: C_extreme brauchen wir nicht mehr, wir wechseln ja zu A!

    # Status-Marker, um zu sehen, ob gewechselt wurde (für Debugging/Plots)
    switched = False

    for t in range(1, T):
        N_prev = N[t-1]
        
        # --- REAL OPTION LOGIC: SWITCH TO A ---
        # Wir prüfen VOR der Berechnung der aktuellen Periode
        if real_option_active and not switched and t >= check_period_start:
            # Wir berechnen das potenzielle Wachstum mit den AKTUELLEN (aggressiven) Werten
            potential_acquisition = (p + q * (N_prev / M)) * (M - N_prev)
            trigger_threshold = trigger_val * M
            
            # Wenn Fighter-Strategie versagt:
            if potential_acquisition < trigger_threshold:
                switched = True
                # ÜBERSCHREIBEN DER PARAMETER MIT OPTION A (FALLBACK)
                if fallback_params:
                    p = fallback_params['p']
                    q = fallback_params['q']
                    C = fallback_params['C']
                    ARPU = fallback_params['ARPU']
                    kappa = fallback_params['kappa']
                    Delta_CM = fallback_params['Delta_CM']
                    Fixed_Cost = fallback_params['Fixed_Cost']
        
        # ------------------------------------

        # Berechnung (jetzt ggf. mit den neuen Parametern von A)
        retention = N_prev * (1 - C)
        acquisition = (p + q * (N_prev / M)) * (M - N_prev)
        
        if acquisition < 0: acquisition = 0
        
        N[t] = retention + acquisition
        if N[t] > M: N[t] = M
        
        revenue = N[t] * ARPU
        cannibalization_loss = acquisition * kappa * Delta_CM
        
        W[t] = revenue - cannibalization_loss - Fixed_Cost
        
    return N, W

# ==========================================
# 2. STREAMLIT GUI
# ==========================================
st.set_page_config(page_title="Master Thesis Simulation", layout="wide")

st.markdown("<h1 style='text-align: center;'>Valuing Digital Market Entry Strategies</h1>", unsafe_allow_html=True)

# --- A. OBERE LEISTE ---
with st.container():
    st.markdown("### 🌐 Globale Einstellungen")
    col_g1, col_g2, col_g3, col_g4 = st.columns(4)
    with col_g1: T = st.slider("Zeitraum (T)", 5, 20, 15)
    with col_g2: n_sim = st.number_input("Simulationen (n)", 100, 5000, 1000, step=100)
    with col_g3: M_global = st.number_input("Marktpotenzial (M)", 300, 5000, 500, step=50)
    with col_g4: 
        st.write(""); st.write("") 
        start_btn = st.button("🚀 Simulation starten", type="primary", use_container_width=True)

st.markdown("---")

# --- B. LAYOUT ---
col_left, col_center, col_right = st.columns([1, 2, 1])

# --- LINKE SPALTE: OPTION A ---
with col_left:
    st.markdown("### 🔵 Option A (Standard)")
    st.info("Konservative Strategie")
    st.markdown("**1. Diffusion**")
    p_a = st.slider("Innovation (p)", 0.001, 0.05, (0.005, 0.010), format="%.3f", key="pa")
    q_a = st.slider("Imitation (q)", 0.1, 0.5, (0.15, 0.25), key="qa")
    st.markdown("**2. Financials**")
    c_a = st.slider("Churn Rate", 0.0, 0.3, (0.03, 0.05), key="ca")
    arpu_a = st.number_input("ARPU (€)", value=4000, step=100, key="arpua")
    fc_a = st.number_input("Fixkosten (€)", value=150000, step=5000, key="fca")
    st.markdown("**3. Cannibalization**")
    kap_a = st.slider("Kappa", 0.0, 1.0, (0.05, 0.10), key="kapa")
    dcm_a = st.number_input("Margin Erosion (€)", value=75, step=10, key="dcma")

# --- RECHTE SPALTE: OPTION B/C ---
with col_right:
    st.markdown("### 🔴 Option B/C (Fighter)")
    st.warning("Aggressive Strategie")
    st.markdown("**1. Diffusion**")
    p_b = st.slider("Innovation (p)", 0.001, 0.1, (0.03, 0.05), format="%.3f", key="pb")
    q_b = st.slider("Imitation (q)", 0.1, 0.8, (0.30, 0.50), key="qb")
    st.markdown("**2. Financials**")
    c_b = st.slider("Churn Rate", 0.0, 0.5, (0.10, 0.20), key="cb")
    arpu_b = st.number_input("ARPU (€)", value=2800, step=100, key="arpub")
    fc_b = st.number_input("Fixkosten (€)", value=180000, step=5000, key="fcb")
    st.markdown("**3. Cannibalization**")
    kap_b = st.slider("Kappa", 0.0, 1.0, (0.50, 0.70), key="kapb")
    dcm_b = st.number_input("Margin Erosion (€)", value=1500, step=100, key="dcmb")
    st.markdown("---")
    st.markdown("**🟢 Real Option (Switch to A)**")
    trigger_input = st.slider("Switch Trigger (< % Growth)", 0.01, 0.15, 0.05, format="%.2f")

# --- MITTLERE SPALTE: ERGEBNISSE ---
with col_center:
    if start_btn:
        st.markdown("### 📊 Analyse Ergebnisse")
        
        def get_val(r): return np.random.triangular(r[0], (r[0]+r[1])/2, r[1]) if isinstance(r, tuple) else r
        
        hist_N_A, hist_W_A, sum_W_A = [], [], []
        hist_N_B, hist_W_B, sum_W_B = [], [], []
        hist_N_C, hist_W_C, sum_W_C = [], [], []

        prog_bar = st.progress(0)

        for i in range(n_sim):
            m_curr = np.random.uniform(M_global*0.9, M_global*1.1)
            
            # 1. Parameter für A ziehen
            curr_p_a = get_val(p_a); curr_q_a = get_val(q_a); curr_c_a = get_val(c_a)
            curr_arpu_a = arpu_a; curr_kap_a = get_val(kap_a); curr_dcm_a = dcm_a
            curr_fc_a = fc_a

            # 2. Parameter für B ziehen
            curr_p_b = get_val(p_b); curr_q_b = get_val(q_b); curr_c_b = get_val(c_b)
            curr_arpu_b = arpu_b; curr_kap_b = get_val(kap_b); curr_dcm_b = dcm_b
            curr_fc_b = fc_b

            # --- PAKET FÜR DEN SWITCH (Fallback Params) ---
            # Das übergeben wir an Option C, damit sie weiß, wohin sie wechseln soll
            fallback_pack = {
                'p': curr_p_a, 'q': curr_q_a, 'C': curr_c_a,
                'ARPU': curr_arpu_a, 'kappa': curr_kap_a, 
                'Delta_CM': curr_dcm_a, 'Fixed_Cost': curr_fc_a
            }

            # --- RUN A ---
            N_a, W_a = run_simulation(
                M=m_curr, p=curr_p_a, q=curr_q_a, C=curr_c_a,
                ARPU=curr_arpu_a, kappa=curr_kap_a, Delta_CM=curr_dcm_a,
                Fixed_Cost=curr_fc_a, start=1, T=T, real_option_active=False
            )
            hist_N_A.append(N_a); hist_W_A.append(W_a); sum_W_A.append(sum(W_a))
            
            # --- RUN B (Fighter Only) ---
            N_b, W_b = run_simulation(
                M=m_curr, p=curr_p_b, q=curr_q_b, C=curr_c_b,
                ARPU=curr_arpu_b, kappa=curr_kap_b, Delta_CM=curr_dcm_b,
                Fixed_Cost=curr_fc_b, start=1, T=T, real_option_active=False
            )
            hist_N_B.append(N_b); hist_W_B.append(W_b); sum_W_B.append(sum(W_b))
            
            # --- RUN C (Fighter with Switch Option) ---
            # Wir starten mit Parametern von B, übergeben aber A als Fallback
            N_c, W_c = run_simulation(
                M=m_curr, p=curr_p_b, q=curr_q_b, C=curr_c_b,
                ARPU=curr_arpu_b, kappa=curr_kap_b, Delta_CM=curr_dcm_b,
                Fixed_Cost=curr_fc_b, start=1, T=T, 
                real_option_active=True, 
                trigger_val=trigger_input,
                fallback_params=fallback_pack # <--- HIER PASSIERT DER SWITCH
            )
            hist_N_C.append(N_c); hist_W_C.append(W_c); sum_W_C.append(sum(W_c))
            
            if i % 50 == 0: prog_bar.progress(i/n_sim)
        
        prog_bar.progress(100)

        # KPIs
        mean_a, mean_b, mean_c = np.mean(sum_W_A), np.mean(sum_W_B), np.mean(sum_W_C)
        option_value = mean_c - mean_b

        k1, k2, k3 = st.columns(3)
        k1.metric("Option A (Std)", f"€ {mean_a/1e6:.2f} M", border=True)
        k2.metric("Option B (Fighter)", f"€ {mean_b/1e6:.2f} M", delta=f"{(mean_b-mean_a)/1e6:.2f} M", border=True)
        k3.metric("Option C (Switch)", f"€ {mean_c/1e6:.2f} M", delta=f"{option_value/1e6:.2f} M Value", border=True)

        # PLOTS
        avg_N_A = np.mean(hist_N_A, axis=0); avg_W_A = np.mean(hist_W_A, axis=0)
        avg_N_B = np.mean(hist_N_B, axis=0); avg_W_B = np.mean(hist_W_B, axis=0)
        avg_N_C = np.mean(hist_N_C, axis=0); avg_W_C = np.mean(hist_W_C, axis=0)

        fig_n, ax_n = plt.subplots(figsize=(6, 3))
        ax_n.plot(avg_N_A, label='A: Standard', color='tab:blue', alpha=0.6)
        ax_n.plot(avg_N_B, label='B: Fighter', color='tab:red', alpha=0.6)
        ax_n.plot(avg_N_C, label='C: Switched Strategy', color='tab:green', linestyle='--', linewidth=2)
        ax_n.set_title("Ø Customer Base Evolution")
        ax_n.legend()
        ax_n.grid(True, alpha=0.3)
        st.pyplot(fig_n)

        fig_w, ax_w = plt.subplots(figsize=(6, 3))
        ax_w.plot(avg_W_A, label='A', color='tab:blue', alpha=0.6)
        ax_w.plot(avg_W_B, label='B', color='tab:red', alpha=0.6)
        ax_w.plot(avg_W_C, label='C', color='tab:green', linestyle='--', linewidth=2)
        ax_w.axhline(0, color='black', linewidth=0.8)
        ax_w.set_title("Ø Net Value Contribution per Period")
        ax_w.legend()
        ax_w.grid(True, alpha=0.3)
        st.pyplot(fig_w)

        # PDF DOWNLOAD (wie gehabt, nur den Code kopieren oder lassen wenn er schon da ist)
        pdf_buffer = io.BytesIO()
        with PdfPages(pdf_buffer) as pdf:
            pdf.savefig(fig_n)
            pdf.savefig(fig_w)
            # ... Rest des PDF Codes
        st.download_button("📄 PDF Report", pdf_buffer.getvalue(), "thesis_report.pdf", "application/pdf", use_container_width=True)

    else:
        st.info("👈 Bitte Parameter anpassen und starten.")
