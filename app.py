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
    # Initialer Wert (Periode 0)
    W[0] = N[0] * ARPU - Fixed_Cost

    # Real Option Parameter
    check_period_start = 3
    C_extreme = 0.95 # Projekt-Abbruch durch extremen Churn

    for t in range(1, T):
        N_prev = N[t-1]
        
        # 1. Bass Diffusion & Churn
        # Churn wirkt auf Bestand
        retention = N_prev * (1 - C)
        # Bass wirkt auf Potenzial (angepasst für B2B: Active Base treibt Word-of-Mouth)
        acquisition = (p + q * (N_prev / M)) * (M - N_prev)
        
        if acquisition < 0: acquisition = 0
        
        # --- REAL OPTION LOGIC (OPTION C) ---
        if real_option_active and t >= check_period_start:
            # Trigger: Wenn Wachstum < X% des Marktpotenzials -> Abbruch
            trigger_threshold = trigger_val * M
            if acquisition < trigger_threshold:
                C = C_extreme 
        # ------------------------------------

        N[t] = retention + acquisition
        if N[t] > M: N[t] = M
        
        # 2. Financial Valuation
        revenue = N[t] * ARPU
        # Kannibalisierung trifft nur Neukunden
        cannibalization_loss = acquisition * kappa * Delta_CM
        
        # Net Value Contribution nach Fixkosten
        W[t] = revenue - cannibalization_loss - Fixed_Cost
        
    return sum(W) # Wir geben den kumulierten Wert (Total Value) zurück

# ==========================================
# 2. STREAMLIT GUI
# ==========================================
st.set_page_config(page_title="Master Thesis Simulation", layout="wide")

st.title("Valuing Digital Market Entry Strategies")
st.markdown("### A Real Options Analysis based on Bass Diffusion")

# --- SIDEBAR: KONFIGURATION ---
st.sidebar.header("1. Globale Einstellungen")
T = st.sidebar.slider("Zeitraum (Perioden)", 5, 20, 15)
n_sim = st.sidebar.slider("Simulationen (n)", 100, 5000, 1000)
M_global = st.sidebar.slider("Marktpotenzial (M) ca.", 300, 1000, 500)

st.sidebar.markdown("---")
st.sidebar.header("2. Szenario-Definition")

# Wir nutzen Tabs für bessere Übersicht
tab_a, tab_b = st.sidebar.tabs(["Option A: Standard", "Option B/C: Fighter"])

# --- TAB A: STANDARD STRATEGIE ---
with tab_a:
    st.markdown("**Diffusion Parameters**")
    p_a = st.slider("Innov. (p) [A]", 0.001, 0.05, (0.005, 0.010), format="%.3f")
    q_a = st.slider("Imit. (q) [A]", 0.1, 0.5, (0.15, 0.25))
    
    st.markdown("**Retention & Financials**")
    c_a = st.slider("Churn (C) [A]", 0.0, 0.3, (0.03, 0.05))
    arpu_a = st.slider("ARPU (€) [A]", 2000, 5000, (3800, 4200))
    fc_a = st.number_input("Fixkosten (€) [A]", value=150000, step=10000)
    
    st.markdown("**Cannibalization Risk**")
    kap_a = st.slider("Kappa (Rate) [A]", 0.0, 1.0, (0.05, 0.10))
    dcm_a = st.slider("Delta Margin (€) [A]", 0, 2000, (50, 100))

# --- TAB B: FIGHTER / DYNAMIC ---
with tab_b:
    st.info("Parameter für die aggressive Strategie (höheres Risiko)")
    st.markdown("**Diffusion Parameters**")
    p_b = st.slider("Innov. (p) [B]", 0.001, 0.1, (0.03, 0.05), format="%.3f")
    q_b = st.slider("Imit. (q) [B]", 0.1, 0.8, (0.30, 0.50))
    
    st.markdown("**Retention & Financials**")
    c_b = st.slider("Churn (C) [B]", 0.0, 0.5, (0.10, 0.20), help="Höherer Churn erwartet bei aggressiver Strategie")
    arpu_b = st.slider("ARPU (€) [B]", 1000, 4000, (2500, 3000), help="Niedrigerer Preis")
    fc_b = st.number_input("Fixkosten (€) [B]", value=180000, step=10000)
    
    st.markdown("**Cannibalization Risk**")
    kap_b = st.slider("Kappa (Rate) [B]", 0.0, 1.0, (0.50, 0.70), help="Hohes Risiko der Kannibalisierung")
    dcm_b = st.slider("Delta Margin (€) [B]", 0, 3000, (1200, 1800), help="Verlust an Deckungsbeitrag pro Kunde")

    st.markdown("---")
    st.markdown("**Real Option Logic (Only Option C)**")
    trigger_input = st.slider("Abandon Trigger (Growth < % of M)", 0.01, 0.10, 0.05)

# --- START BUTTON ---
if st.button("🚀 Simulation starten"):
    
    # Platzhalter für Layout
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    chart_place = st.empty()
    
    # 1. Datenvorbereitung (Helper Function für Random Ranges)
    def get_val(r): 
        if isinstance(r, tuple): return np.random.triangular(r[0], (r[0]+r[1])/2, r[1])
        return r
    
    # Speicher für Ergebnisse
    res_A = []
    res_B = []
    res_C = []

    # Progress Bar
    prog_bar = st.progress(0)

    # 2. MONTE CARLO LOOP
    for i in range(n_sim):
        # M leicht variieren (+/- 10%)
        m_curr = np.random.uniform(M_global*0.9, M_global*1.1)
        
        # --- SCENARIO A ---
        val_a = run_simulation(
            M=m_curr, p=get_val(p_a), q=get_val(q_a), C=get_val(c_a),
            ARPU=get_val(arpu_a), kappa=get_val(kap_a), Delta_CM=get_val(dcm_a),
            Fixed_Cost=fc_a, start=1, T=T, real_option_active=False
        )
        res_A.append(val_a)
        
        # --- SCENARIO B & C (Parameter sind gleich, nur Option unterscheidet sich) ---
        # Wir ziehen EINMAL Parameter für B und C gemeinsam, damit der Vergleich fair ist
        curr_p_b = get_val(p_b); curr_q_b = get_val(q_b); curr_c_b = get_val(c_b)
        curr_arpu_b = get_val(arpu_b); curr_kap_b = get_val(kap_b); curr_dcm_b = get_val(dcm_b)
        
        # Lauf B (Ohne Bremse)
        val_b = run_simulation(
            M=m_curr, p=curr_p_b, q=curr_q_b, C=curr_c_b,
            ARPU=curr_arpu_b, kappa=curr_kap_b, Delta_CM=curr_dcm_b,
            Fixed_Cost=fc_b, start=1, T=T, real_option_active=False
        )
        res_B.append(val_b)
        
        # Lauf C (Mit Real Option Trigger)
        val_c = run_simulation(
            M=m_curr, p=curr_p_b, q=curr_q_b, C=curr_c_b,
            ARPU=curr_arpu_b, kappa=curr_kap_b, Delta_CM=curr_dcm_b,
            Fixed_Cost=fc_b, start=1, T=T, real_option_active=True, trigger_val=trigger_input
        )
        res_C.append(val_c)
        
        if i % 50 == 0: prog_bar.progress(i/n_sim)
    
    prog_bar.progress(100)

    # 3. ERGEBNISSE ANZEIGEN
    mean_a, mean_b, mean_c = np.mean(res_A), np.mean(res_B), np.mean(res_C)
    
    # KPIs
    col_kpi1.metric("Option A (Standard)", f"€ {mean_a/1e6:.2f} M", help="Low Risk, Low Reward")
    col_kpi2.metric("Option B (Fighter)", f"€ {mean_b/1e6:.2f} M", delta=f"{(mean_b-mean_a)/1e6:.2f} M", delta_color="normal", help="High Risk Strategy")
    
    # Value of Option Calculation
    option_value = mean_c - mean_b
    col_kpi3.metric("Option C (Dynamic)", f"€ {mean_c/1e6:.2f} M", delta=f"{option_value/1e6:.2f} M (Option Value)", help="Fighter Strategy with Abandonment Option")

    # PLOT
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(res_A, bins=40, alpha=0.5, label='A: Standard', color='blue')
    ax.hist(res_B, bins=40, alpha=0.5, label='B: Fighter (No Exit)', color='red')
    ax.hist(res_C, bins=40, alpha=0.5, label='C: Dynamic (With Exit)', color='green', histtype='step', linewidth=2)
    
    ax.set_title("Distribution of Net Value Contribution (Monte Carlo)")
    ax.set_xlabel("Cumulative Net Value (€)")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # 4. PDF DOWNLOAD
    st.subheader("📥 Export")
    pdf_buffer = io.BytesIO()
    with PdfPages(pdf_buffer) as pdf:
        pdf.savefig(fig)
        
        # Info Page
        fig_txt, ax_txt = plt.subplots(figsize=(8, 11))
        ax_txt.axis('off')
        info = (
            f"SIMULATION REPORT\n"
            f"=================\n\n"
            f"Runs: {n_sim} | Periods: {T}\n\n"
            f"RESULTS (Mean):\n"
            f"Option A: {mean_a:,.0f} EUR\n"
            f"Option B: {mean_b:,.0f} EUR\n"
            f"Option C: {mean_c:,.0f} EUR\n"
            f"-> Value of Real Option: {option_value:,.0f} EUR\n\n"
            f"INPUT PARAMETERS (Ranges):\n"
            f"---------------------------\n"
            f"Scenario A (Standard):\n"
            f"  Churn: {c_a}\n"
            f"  ARPU: {arpu_a}\n"
            f"  Kappa: {kap_a}\n"
            f"  Delta CM: {dcm_a}\n\n"
            f"Scenario B (Fighter):\n"
            f"  Churn: {c_b}\n"
            f"  ARPU: {arpu_b}\n"
            f"  Kappa: {kap_b}\n"
            f"  Delta CM: {dcm_b}\n"
            f"  Abandon Trigger: < {trigger_input*100}% Growth"
        )
        ax_txt.text(0.1, 0.9, info, family='monospace', fontsize=10, va='top')
        pdf.savefig(fig_txt)

    st.download_button("Download PDF Report", pdf_buffer.getvalue(), "simulation_thesis.pdf", "application/pdf")
