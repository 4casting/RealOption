import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import io

# ==========================================
# 1. MATHEMATISCHES MODELL (CORE LOGIC)
# ==========================================
def run_simulation(M, p, q, C, ARPU, kappa, Delta_CM, Fixed_Cost, start, T, 
                   real_option_active=False, trigger_val=0.05):
    """
    Führt EINE Simulation durch und gibt die Zeitreihen N und W zurück.
    """
    N = [0.0] * T
    W = [0.0] * T
    N[0] = start
    W[0] = N[0] * ARPU - Fixed_Cost

    check_period_start = 3
    C_extreme = 0.95 

    for t in range(1, T):
        N_prev = N[t-1]
        retention = N_prev * (1 - C)
        acquisition = (p + q * (N_prev / M)) * (M - N_prev)
        
        if acquisition < 0: acquisition = 0
        
        # --- REAL OPTION LOGIC ---
        if real_option_active and t >= check_period_start:
            trigger_threshold = trigger_val * M
            if acquisition < trigger_threshold:
                C = C_extreme 

        N[t] = retention + acquisition
        if N[t] > M: N[t] = M
        
        revenue = N[t] * ARPU
        cannibalization_loss = acquisition * kappa * Delta_CM
        
        W[t] = revenue - cannibalization_loss - Fixed_Cost
        
    # WICHTIG: Wir geben jetzt die Listen (Zeitreihen) zurück, nicht nur die Summe
    return N, W

# ==========================================
# 2. STREAMLIT GUI
# ==========================================
st.set_page_config(page_title="Master Thesis Simulation", layout="wide")

st.markdown("<h1 style='text-align: center;'>Valuing Digital Market Entry Strategies</h1>", unsafe_allow_html=True)

# --- A. OBERE LEISTE (GLOBALE EINSTELLUNGEN) ---
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

# --- B. LAYOUT (LINKS / MITTE / RECHTS) ---
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
    st.markdown("**🟢 Real Option (Only C)**")
    trigger_input = st.slider("Abandon Trigger (< % Growth)", 0.01, 0.15, 0.05, format="%.2f")

# --- MITTLERE SPALTE: ERGEBNISSE ---
with col_center:
    if start_btn:
        st.markdown("### 📊 Analyse Ergebnisse")
        
        def get_val(r): return np.random.triangular(r[0], (r[0]+r[1])/2, r[1]) if isinstance(r, tuple) else r
        
        # Speicher für Historien (Arrays) und Summen (Floats)
        hist_N_A, hist_W_A, sum_W_A = [], [], []
        hist_N_B, hist_W_B, sum_W_B = [], [], []
        hist_N_C, hist_W_C, sum_W_C = [], [], []

        prog_bar = st.progress(0)

        # SIMULATION LOOP
        for i in range(n_sim):
            m_curr = np.random.uniform(M_global*0.9, M_global*1.1)
            
            # --- SCENARIO A ---
            N_a, W_a = run_simulation(
                M=m_curr, p=get_val(p_a), q=get_val(q_a), C=get_val(c_a),
                ARPU=arpu_a, kappa=get_val(kap_a), Delta_CM=dcm_a,
                Fixed_Cost=fc_a, start=1, T=T, real_option_active=False
            )
            hist_N_A.append(N_a); hist_W_A.append(W_a); sum_W_A.append(sum(W_a))
            
            # --- SCENARIO B & C ---
            # Gemeinsame Parameter ziehen
            curr_p_b = get_val(p_b); curr_q_b = get_val(q_b); curr_c_b = get_val(c_b)
            curr_kap_b = get_val(kap_b)
            
            # B (No Exit)
            N_b, W_b = run_simulation(
                M=m_curr, p=curr_p_b, q=curr_q_b, C=curr_c_b,
                ARPU=arpu_b, kappa=curr_kap_b, Delta_CM=dcm_b,
                Fixed_Cost=fc_b, start=1, T=T, real_option_active=False
            )
            hist_N_B.append(N_b); hist_W_B.append(W_b); sum_W_B.append(sum(W_b))
            
            # C (With Exit)
            N_c, W_c = run_simulation(
                M=m_curr, p=curr_p_b, q=curr_q_b, C=curr_c_b,
                ARPU=arpu_b, kappa=curr_kap_b, Delta_CM=dcm_b,
                Fixed_Cost=fc_b, start=1, T=T, real_option_active=True, trigger_val=trigger_input
            )
            hist_N_C.append(N_c); hist_W_C.append(W_c); sum_W_C.append(sum(W_c))
            
            if i % 50 == 0: prog_bar.progress(i/n_sim)
        
        prog_bar.progress(100)

        # MITTELWERTE BERECHNEN (KPIs)
        mean_a, mean_b, mean_c = np.mean(sum_W_A), np.mean(sum_W_B), np.mean(sum_W_C)
        option_value = mean_c - mean_b

        # KPI Cards
        k1, k2, k3 = st.columns(3)
        k1.metric("Option A", f"€ {mean_a/1e6:.2f} M", border=True)
        k2.metric("Option B", f"€ {mean_b/1e6:.2f} M", delta=f"{(mean_b-mean_a)/1e6:.2f} M", border=True)
        k3.metric("Option C", f"€ {mean_c/1e6:.2f} M", delta=f"{option_value/1e6:.2f} M Value", border=True)

        # ---------------------------------------------
        # PLOTS ERSTELLEN
        # ---------------------------------------------
        
        # 1. Durchschnittskurven berechnen
        avg_N_A = np.mean(hist_N_A, axis=0)
        avg_N_B = np.mean(hist_N_B, axis=0)
        avg_N_C = np.mean(hist_N_C, axis=0)
        
        avg_W_A = np.mean(hist_W_A, axis=0)
        avg_W_B = np.mean(hist_W_B, axis=0)
        avg_W_C = np.mean(hist_W_C, axis=0)

        # CHART 1: WACHSTUM (N)
        fig_n, ax_n = plt.subplots(figsize=(6, 3))
        ax_n.plot(avg_N_A, label='A: Standard', color='tab:blue')
        ax_n.plot(avg_N_B, label='B: Fighter', color='tab:red')
        ax_n.plot(avg_N_C, label='C: Dynamic', color='tab:green', linestyle='--')
        ax_n.set_title("Ø Customer Base Evolution N(t)")
        ax_n.set_ylabel("Active Customers")
        ax_n.legend()
        ax_n.grid(True, alpha=0.3)
        st.pyplot(fig_n)

        # CHART 2: FINANZEN (W pro Periode)
        fig_w, ax_w = plt.subplots(figsize=(6, 3))
        ax_w.plot(avg_W_A, label='A: Standard', color='tab:blue')
        ax_w.plot(avg_W_B, label='B: Fighter', color='tab:red')
        ax_w.plot(avg_W_C, label='C: Dynamic', color='tab:green', linestyle='--')
        ax_w.axhline(0, color='black', linewidth=0.8) # Nulllinie für Break-Even
        ax_w.set_title("Ø Net Value Contribution W(t) per Period")
        ax_w.set_ylabel("Net Value (€)")
        ax_w.legend()
        ax_w.grid(True, alpha=0.3)
        st.pyplot(fig_w)

        # CHART 3: HISTOGRAMM (Gesamtwert)
        fig_hist, ax_hist = plt.subplots(figsize=(6, 3))
        ax_hist.hist(sum_W_A, bins=30, alpha=0.5, label='A', color='tab:blue')
        ax_hist.hist(sum_W_B, bins=30, alpha=0.5, label='B', color='tab:red')
        ax_hist.hist(sum_W_C, bins=30, alpha=0.5, label='C', color='tab:green', histtype='step', linewidth=2)
        ax_hist.set_title("Distribution of Total Value (Monte Carlo)")
        ax_hist.set_xlabel("Cumulative Value (€)")
        ax_hist.legend()
        ax_hist.grid(True, alpha=0.2)
        st.pyplot(fig_hist)

        # PDF DOWNLOAD
        pdf_buffer = io.BytesIO()
        with PdfPages(pdf_buffer) as pdf:
            pdf.savefig(fig_n)    # Seite 1: Kunden
            pdf.savefig(fig_w)    # Seite 2: Finanzen
            pdf.savefig(fig_hist) # Seite 3: Histogramm
            
            # Seite 4: Info & Parameter
            fig_txt, ax_txt = plt.subplots(figsize=(8, 6))
            ax_txt.axis('off')
            info_text = (f"SIMULATION REPORT\nRuns: {n_sim} | T: {T}\n\n"
                         f"Results (Mean Total Value):\nOption A: {mean_a:,.0f} €\n"
                         f"Option B: {mean_b:,.0f} €\nOption C: {mean_c:,.0f} €\n"
                         f"Option Value: {option_value:,.0f} €")
            ax_txt.text(0.1, 0.8, info_text, family='monospace', fontsize=12)
            pdf.savefig(fig_txt)

        st.download_button("📄 PDF Report (inkl. Kurven)", pdf_buffer.getvalue(), "thesis_charts.pdf", "application/pdf", use_container_width=True)

    else:
        st.info("👈 Parameter anpassen und starten.")
