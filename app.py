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
    Führt EINE Simulation über T Perioden durch.
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
        
    return sum(W)

# ==========================================
# 2. STREAMLIT GUI - NEUES LAYOUT
# ==========================================
st.set_page_config(page_title="Master Thesis Simulation", layout="wide")

st.markdown("<h1 style='text-align: center;'>Valuing Digital Market Entry Strategies</h1>", unsafe_allow_html=True)

# --- A. OBERE LEISTE (GLOBALE EINSTELLUNGEN) ---
with st.container():
    st.markdown("### 🌐 Globale Einstellungen")
    # 4 Spalten für die globalen Werte nebeneinander
    col_g1, col_g2, col_g3, col_g4 = st.columns(4)
    
    with col_g1:
        T = st.slider("Zeitraum (T)", 5, 20, 15)
    with col_g2:
        n_sim = st.number_input("Simulationen (n)", 100, 10000, 1000, step=100)
    with col_g3:
        M_global = st.number_input("Marktpotenzial (M)", 300, 2000, 500, step=50)
    with col_g4:
        # Der Start-Button sitzt prominent oben rechts in der Leiste
        st.write("") # Platzhalter für Alignment
        st.write("") 
        start_btn = st.button("🚀 Simulation starten", type="primary", use_container_width=True)

st.markdown("---")

# --- B. DREI-SPALTEN-LAYOUT ---
# Wir nutzen ein Verhältnis von 1:2:1 (Links schmal, Mitte breit, Rechts schmal)
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
        # Platzhalter für Layout
        st.markdown("### 📊 Analyse Ergebnisse")
        
        # Helper Function
        def get_val(r): 
            if isinstance(r, tuple): return np.random.triangular(r[0], (r[0]+r[1])/2, r[1])
            return r
        
        # Speicher
        res_A, res_B, res_C = [], [], []

        # Progress Bar
        prog_bar = st.progress(0)

        # Simulation Loop
        for i in range(n_sim):
            m_curr = np.random.uniform(M_global*0.9, M_global*1.1)
            
            # Scenario A
            # Hinweis: Wir bauen Ranges aus den Number Inputs (z.B. +/- 10%) oder nehmen Slider Ranges
            # Für Number Inputs nehmen wir hier den fixen Wert, für Slider die Range
            val_a = run_simulation(
                M=m_curr, p=get_val(p_a), q=get_val(q_a), C=get_val(c_a),
                ARPU=arpu_a, kappa=get_val(kap_a), Delta_CM=dcm_a,
                Fixed_Cost=fc_a, start=1, T=T, real_option_active=False
            )
            res_A.append(val_a)
            
            # Scenario B & C
            curr_p_b = get_val(p_b); curr_q_b = get_val(q_b); curr_c_b = get_val(c_b)
            curr_kap_b = get_val(kap_b)
            
            val_b = run_simulation(
                M=m_curr, p=curr_p_b, q=curr_q_b, C=curr_c_b,
                ARPU=arpu_b, kappa=curr_kap_b, Delta_CM=dcm_b,
                Fixed_Cost=fc_b, start=1, T=T, real_option_active=False
            )
            res_B.append(val_b)
            
            val_c = run_simulation(
                M=m_curr, p=curr_p_b, q=curr_q_b, C=curr_c_b,
                ARPU=arpu_b, kappa=curr_kap_b, Delta_CM=dcm_b,
                Fixed_Cost=fc_b, start=1, T=T, real_option_active=True, trigger_val=trigger_input
            )
            res_C.append(val_c)
            
            if i % 50 == 0: prog_bar.progress(i/n_sim)
        
        prog_bar.progress(100)

        # KPIs berechnen
        mean_a, mean_b, mean_c = np.mean(res_A), np.mean(res_B), np.mean(res_C)
        option_value = mean_c - mean_b

        # KPI Cards (Container)
        kpi_container = st.container()
        with kpi_container:
            k1, k2, k3 = st.columns(3)
            k1.metric("Option A (Std)", f"€ {mean_a/1e6:.2f} M", border=True)
            k2.metric("Option B (Fighter)", f"€ {mean_b/1e6:.2f} M", delta=f"{(mean_b-mean_a)/1e6:.2f} M", border=True)
            k3.metric("Option C (Real Opt)", f"€ {mean_c/1e6:.2f} M", delta=f"{option_value/1e6:.2f} M Value", border=True)

        # Plot
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(res_A, bins=40, alpha=0.5, label='A: Standard', color='tab:blue')
        ax.hist(res_B, bins=40, alpha=0.5, label='B: Fighter', color='tab:red')
        ax.hist(res_C, bins=40, alpha=0.5, label='C: Dynamic', color='tab:green', histtype='step', linewidth=2)
        ax.set_xlabel("Net Value Contribution (€)")
        ax.legend()
        ax.grid(True, alpha=0.2)
        st.pyplot(fig)

        # PDF Download
        pdf_buffer = io.BytesIO()
        with PdfPages(pdf_buffer) as pdf:
            pdf.savefig(fig)
            # Info Page Logic (gekürzt für Übersichtlichkeit)
            fig_txt, ax_txt = plt.subplots(figsize=(8, 6))
            ax_txt.text(0.1, 0.5, f"Simulation Report\nRuns: {n_sim}\nMean A: {mean_a:,.0f}", fontsize=12)
            ax_txt.axis('off')
            pdf.savefig(fig_txt)

        st.download_button("📄 PDF Report Download", pdf_buffer.getvalue(), "thesis_report.pdf", "application/pdf", use_container_width=True)
    
    else:
        st.info("👈 Bitte Parameter links/rechts anpassen und oben auf 'Starten' klicken.")
