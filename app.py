import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import io # Wichtig für den Download im RAM

# ---------------------------------------------------------
# 1. SIMULATIONS-LOGIK (Identisch zu deinem Code)
# ---------------------------------------------------------
def run_simulation(M, p, q, C, ARPU, kappa, Delta_CM, Fixed_Cost, start, T, real_option_active=False):
    N = [0.0] * T
    W = [0.0] * T
    N[0] = start
    W[0] = N[0] * ARPU - Fixed_Cost # Initial Fixkostenabzug

    trigger_threshold = 0.05 * M 
    check_period_start = 3
    C_extreme = 0.95 

    for t in range(1, T):
        N_prev = N[t-1]
        retention = N_prev * (1 - C)
        acquisition = (p + q * (N_prev / M)) * (M - N_prev)
        if acquisition < 0: acquisition = 0
        
        # Real Option Logic
        if real_option_active and t >= check_period_start:
            if acquisition < trigger_threshold:
                C = C_extreme 

        N[t] = retention + acquisition
        if N[t] > M: N[t] = M
        
        revenue = N[t] * ARPU
        cannibalization_loss = acquisition * kappa * Delta_CM
        
        # Wichtig: Fixkosten hier abziehen
        W[t] = revenue - cannibalization_loss - Fixed_Cost
        
    return N, W

# ---------------------------------------------------------
# 2. STREAMLIT APP AUFBAU
# ---------------------------------------------------------
st.set_page_config(page_title="Market Entry Simulation", layout="wide")

st.title("Valuing Digital Market Entry Strategies")
st.markdown("Interaktive Simulation basierend auf Bass Diffusion & Real Options.")

# --- SIDEBAR: INPUT PARAMETER ---
st.sidebar.header("1. Globale Einstellungen")
T = st.sidebar.slider("Laufzeit (Perioden)", 5, 20, 15)
n_simulations = st.sidebar.slider("Anzahl Monte-Carlo Simulationen", 100, 5000, 1000)

st.sidebar.header("2. Strategie-Parameter (Ranges)")
st.sidebar.markdown("**Marktpotenzial (M)**")
M_min = st.sidebar.number_input("M Min", value=400)
M_max = st.sidebar.number_input("M Max", value=600)

st.sidebar.markdown("**Innovationskoeffizient (p)**")
p_min = st.sidebar.number_input("p Min", value=0.02, format="%.3f")
p_max = st.sidebar.number_input("p Max", value=0.04, format="%.3f")

st.sidebar.markdown("**Imitationskoeffizient (q)**")
q_min = st.sidebar.number_input("q Min", value=0.30, format="%.2f")
q_max = st.sidebar.number_input("q Max", value=0.50, format="%.2f")

st.sidebar.markdown("**Fixkosten pro Periode**")
fc_min = st.sidebar.number_input("Fixkosten Min", value=150000)
fc_max = st.sidebar.number_input("Fixkosten Max", value=200000)

# Button zum Starten
if st.button("Simulation starten"):
    
    # Ladebalken
    progress_bar = st.progress(0)
    
    # --- SIMULATION DURCHFÜHREN ---
    # Wir simulieren hier exemplarisch Option B (Fighter Brand) und C (Real Option)
    # basierend auf den User-Inputs
    
    results = []
    
    # Parameter für Simulation packen
    params = {
        "M": (M_min, M_max), "p": (p_min, p_max), "q": (q_min, q_max),
        "C": (0.08, 0.15), "ARPU": (2500, 3000), "kappa": (0.4, 0.6),
        "Delta_CM": (1000, 1500), "Fixed_Cost": (fc_min, fc_max),
        "start": 1
    }

    # Monte Carlo Loop
    data_opt_b = [] # Ohne Real Option
    data_opt_c = [] # Mit Real Option

    for i in range(n_simulations):
        # Zufallswerte ziehen
        curr = {}
        for k, v in params.items():
            if isinstance(v, tuple):
                curr[k] = np.random.triangular(v[0], (v[0]+v[1])/2, v[1])
            else:
                curr[k] = v
        
        # Lauf 1: Ohne Option (B)
        _, W_b = run_simulation(**curr, T=T, real_option_active=False)
        data_opt_b.append(sum(W_b))
        
        # Lauf 2: Mit Option (C)
        _, W_c = run_simulation(**curr, T=T, real_option_active=True)
        data_opt_c.append(sum(W_c))
        
        if i % 100 == 0:
            progress_bar.progress(i / n_simulations)
            
    progress_bar.progress(100)

    # --- ERGEBNISSE ANZEIGEN ---
    st.subheader("Simulations-Ergebnisse")
    col1, col2 = st.columns(2)
    
    mean_b = np.mean(data_opt_b)
    mean_c = np.mean(data_opt_c)
    
    col1.metric("Ø Wert Option B (Standard)", f"€ {mean_b/1e6:.2f} M")
    col2.metric("Ø Wert Option C (Real Option)", f"€ {mean_c/1e6:.2f} M", delta=f"{(mean_c-mean_b)/1e6:.2f} M")

    # Plotting für die Webseite
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(data_opt_b, bins=50, alpha=0.5, label='Option B (No Exit)', color='red')
    ax.hist(data_opt_c, bins=50, alpha=0.5, label='Option C (With Exit)', color='green')
    ax.legend()
    ax.set_title("Verteilung des Net Value Contribution (Monte Carlo)")
    ax.set_xlabel("Total Value (EUR)")
    st.pyplot(fig)

    # ---------------------------------------------------------
    # 3. PDF GENERIERUNG (IN MEMORY)
    # ---------------------------------------------------------
    st.subheader("Download Report")
    
    # Virtueller Speicher für das PDF
    pdf_buffer = io.BytesIO()
    
    with PdfPages(pdf_buffer) as pdf:
        # Seite 1: Der Plot von oben
        pdf.savefig(fig)
        
        # Seite 2: Parameter-Tabelle (als neuer Plot)
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.axis('off')
        text_info = f"""
        Simulation Parameters:
        ----------------------
        Iterations: {n_simulations}
        Period T: {T}
        
        User Inputs:
        M: {M_min} - {M_max}
        p: {p_min} - {p_max}
        q: {q_min} - {q_max}
        Fixed Costs: {fc_min} - {fc_max}
        """
        ax2.text(0.1, 0.9, text_info, family='monospace', fontsize=12, va='top')
        pdf.savefig(fig2)
        
    # PDF fertigstellen und Pointer an den Anfang setzen
    # Wichtig: Nichts schließen, nur Buffer nutzen
    
    # Download Button
    st.download_button(
        label="📄 PDF Report herunterladen",
        data=pdf_buffer.getvalue(),
        file_name="simulation_report.pdf",
        mime="application/pdf"
    )