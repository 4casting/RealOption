import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import io

# ==========================================
# 1. MATHEMATISCHES MODELL (CORE LOGIC)
# ==========================================
def run_simulation(M, p, q, C, ARPU, kappa, Delta_CM, Fixed_Cost, start, T, 
                   mode='static', trigger_val=0.05, fallback_params=None):
    """
    Führt EINE Simulation durch.
    mode: 'static' (keine Option), 'switch' (wechselt zu A), 'abandon' (bricht ab)
    """
    N = [0.0] * T
    W = [0.0] * T
    N[0] = start
    W[0] = N[0] * ARPU - Fixed_Cost

    # Start-Periode für die Überprüfung (z.B. ab Jahr 3)
    check_period_start = 3
    
    # Status-Marker
    option_exercised = False

    for t in range(1, T):
        N_prev = N[t-1]
        
        # --- REAL OPTION LOGIC ---
        # Wir prüfen VOR der Berechnung der aktuellen Periode
        if mode != 'static' and not option_exercised and t >= check_period_start:
            
            # Wir berechnen das potenzielle Wachstum mit den AKTUELLEN Werten
            potential_acquisition = (p + q * (N_prev / M)) * (M - N_prev)
            trigger_threshold = trigger_val * M
            
            # TRIGGER BEDINGUNG: Wachstum zu schwach?
            if potential_acquisition < trigger_threshold:
                option_exercised = True
                
                if mode == 'switch' and fallback_params:
                    # STRATEGIEWECHSEL: Wir übernehmen die Parameter von Option A
                    p = fallback_params['p']
                    q = fallback_params['q']
                    C = fallback_params['C']
                    ARPU = fallback_params['ARPU']
                    kappa = fallback_params['kappa']
                    Delta_CM = fallback_params['Delta_CM']
                    Fixed_Cost = fallback_params['Fixed_Cost']
                    
                elif mode == 'abandon':
                    # ABBRUCH: Maximale Abwanderung, Kostenstopp
                    C = 1.0  # Alle Kunden gehen (Projekt wird eingestellt)
                    Fixed_Cost = 0 # Keine weiteren operativen Kosten
                    # (Optional: Hier könnte man noch Liquidationskosten abziehen)
        
        # ------------------------------------

        # Berechnung der Periode (mit ggf. neuen Parametern)
        retention = N_prev * (1 - C)
        acquisition = (p + q * (N_prev / M)) * (M - N_prev)
        
        if acquisition < 0: acquisition = 0
        if mode == 'abandon' and option_exercised: acquisition = 0 # Keine Neukunden bei Abbruch
        
        N[t] = retention + acquisition
        if N[t] > M: N[t] = M
        
        revenue = N[t] * ARPU
        cannibalization_loss = acquisition * kappa * Delta_CM
        
        W[t] = revenue - cannibalization_loss - Fixed_Cost
        
    return N, W

# ==========================================
# 2. STREAMLIT GUI
# ==========================================
st.set_page_config(page_title="Real Options Valuation", layout="wide")

st.markdown("<h1 style='text-align: center;'>Valuing Digital Market Entry Strategies</h1>", unsafe_allow_html=True)

# --- GLOBALE EINSTELLUNGEN ---
with st.container():
    st.markdown("### 🌐 Globale Einstellungen")
    c1, c2, c3, c4 = st.columns(4)
    with c1: T = st.slider("Zeitraum (Jahre)", 5, 20, 15)
    with c2: n_sim = st.number_input("Monte Carlo Iterationen", 100, 5000, 1000, step=100)
    with c3: M_global = st.number_input("Marktpotenzial (M)", 300, 5000, 500, step=50)
    with c4: 
        st.write(""); st.write("")
        start_btn = st.button("🚀 Simulation starten", type="primary", use_container_width=True)

st.markdown("---")

col_left, col_center, col_right = st.columns([1, 2, 1])

# --- LINKS: OPTION A (STANDARD) ---
with col_left:
    st.markdown("### 🔵 Option A (Standard)")
    st.info("Basis-Szenario & Fallback für Switch")
    
    st.markdown("**Diffusion & Retention**")
    p_a = st.slider("Innovation (p)", 0.001, 0.05, (0.005, 0.010), format="%.3f", key="pa")
    q_a = st.slider("Imitation (q)", 0.1, 0.5, (0.15, 0.25), key="qa")
    c_a = st.slider("Churn Rate", 0.0, 0.3, (0.03, 0.05), key="ca")
    
    st.markdown("**Financials**")
    arpu_a = st.number_input("ARPU (€)", value=4000, step=100, key="arpua")
    fc_a = st.number_input("Fixkosten (€)", value=150000, step=5000, key="fca")
    
    st.markdown("**Cannibalization**")
    kap_a = st.slider("Kappa", 0.0, 1.0, (0.05, 0.10), key="kapa")
    dcm_a = st.number_input("Margin Erosion (€)", value=75, step=10, key="dcma")

# --- RECHTS: OPTION B (FIGHTER BASIS) ---
with col_right:
    st.markdown("### 🔴 Option B (Fighter)")
    st.warning("Aggressive Basis für B, Switch & Abandon")
    
    st.markdown("**Diffusion & Retention**")
    p_b = st.slider("Innovation (p)", 0.001, 0.1, (0.03, 0.05), format="%.3f", key="pb")
    q_b = st.slider("Imitation (q)", 0.1, 0.8, (0.30, 0.50), key="qb")
    c_b = st.slider("Churn Rate", 0.0, 0.5, (0.10, 0.20), key="cb")
    
    st.markdown("**Financials**")
    arpu_b = st.number_input("ARPU (€)", value=2800, step=100, key="arpub")
    fc_b = st.number_input("Fixkosten (€)", value=180000, step=5000, key="fcb")
    
    st.markdown("**Cannibalization**")
    kap_b = st.slider("Kappa", 0.0, 1.0, (0.50, 0.70), key="kapb")
    dcm_b = st.number_input("Margin Erosion (€)", value=1500, step=100, key="dcmb")
    
    st.markdown("---")
    st.markdown("**⚡ Trigger Logic**")
    trigger_input = st.slider("Ausübung wenn Wachstum < % von M", 0.01, 0.15, 0.05)

# --- MITTE: ERGEBNISSE ---
with col_center:
    if start_btn:
        st.markdown("### 📊 Analyse Ergebnisse")
        
        # Helper für Random Ranges
        def get_val(r): return np.random.triangular(r[0], (r[0]+r[1])/2, r[1]) if isinstance(r, tuple) else r
        
        # Speicher
        data = {
            "A": {"N": [], "W": [], "Sum": []},
            "B": {"N": [], "W": [], "Sum": []},
            "Switch": {"N": [], "W": [], "Sum": []},
            "Abandon": {"N": [], "W": [], "Sum": []}
        }

        prog_bar = st.progress(0)

        # MONTE CARLO LOOP
        for i in range(n_sim):
            m_curr = np.random.uniform(M_global*0.9, M_global*1.1)
            
            # 1. Parameter ziehen
            # A
            pa, qa, ca = get_val(p_a), get_val(q_a), get_val(c_a)
            kapa = get_val(kap_a)
            # B
            pb, qb, cb = get_val(p_b), get_val(q_b), get_val(c_b)
            kapb = get_val(kap_b)

            # Fallback Pack für Switch
            fallback = {
                'p': pa, 'q': qa, 'C': ca, 'ARPU': arpu_a, 
                'kappa': kapa, 'Delta_CM': dcm_a, 'Fixed_Cost': fc_a
            }

            # 2. Simulationen laufen lassen
            
            # A: Static Standard
            Na, Wa = run_simulation(m_curr, pa, qa, ca, arpu_a, kapa, dcm_a, fc_a, 1, T, mode='static')
            data["A"]["N"].append(Na); data["A"]["W"].append(Wa); data["A"]["Sum"].append(sum(Wa))

            # B: Static Fighter (No Exit)
            Nb, Wb = run_simulation(m_curr, pb, qb, cb, arpu_b, kapb, dcm_b, fc_b, 1, T, mode='static')
            data["B"]["N"].append(Nb); data["B"]["W"].append(Wb); data["B"]["Sum"].append(sum(Wb))

            # C: Switch Option
            Nc, Wc = run_simulation(m_curr, pb, qb, cb, arpu_b, kapb, dcm_b, fc_b, 1, T, 
                                    mode='switch', trigger_val=trigger_input, fallback_params=fallback)
            data["Switch"]["N"].append(Nc); data["Switch"]["W"].append(Wc); data["Switch"]["Sum"].append(sum(Wc))

            # D: Abandon Option
            Nd, Wd = run_simulation(m_curr, pb, qb, cb, arpu_b, kapb, dcm_b, fc_b, 1, T, 
                                    mode='abandon', trigger_val=trigger_input)
            data["Abandon"]["N"].append(Nd); data["Abandon"]["W"].append(Wd); data["Abandon"]["Sum"].append(sum(Wd))
            
            if i % 50 == 0: prog_bar.progress(i/n_sim)
        
        prog_bar.progress(100)

        # 3. KPIs BERECHNEN
        means = {k: np.mean(v["Sum"]) for k, v in data.items()}
        
        # KPI Anzeige
        k1, k2 = st.columns(2)
        k1.metric("Option A (Standard)", f"€ {means['A']/1e6:.2f} M", border=True)
        k2.metric("Option B (Fighter Static)", f"€ {means['B']/1e6:.2f} M", delta=f"{(means['B']-means['A'])/1e6:.2f} M", border=True)
        
        k3, k4 = st.columns(2)
        val_switch = means['Switch'] - means['B']
        k3.metric("Option to Switch", f"€ {means['Switch']/1e6:.2f} M", delta=f"{val_switch/1e6:.2f} M Value", border=True)
        
        val_abandon = means['Abandon'] - means['B']
        k4.metric("Option to Abandon", f"€ {means['Abandon']/1e6:.2f} M", delta=f"{val_abandon/1e6:.2f} M Value", border=True)

        # 4. PLOTS
        # Durchschnittskurven
        avg_N = {k: np.mean(v["N"], axis=0) for k, v in data.items()}
        avg_W = {k: np.mean(v["W"], axis=0) for k, v in data.items()}

        # Plot 1: Kunden N(t)
        fig_n, ax_n = plt.subplots(figsize=(7, 3.5))
        ax_n.plot(avg_N['A'], label='A: Standard', color='tab:blue', alpha=0.6)
        ax_n.plot(avg_N['B'], label='B: Fighter', color='tab:red', alpha=0.6)
        ax_n.plot(avg_N['Switch'], label='Switch', color='tab:green', linestyle='--', linewidth=2)
        ax_n.plot(avg_N['Abandon'], label='Abandon', color='black', linestyle=':', linewidth=2)
        ax_n.set_title("Ø Customer Base Evolution N(t)")
        ax_n.set_ylabel("Active Customers")
        ax_n.legend()
        ax_n.grid(True, alpha=0.3)
        st.pyplot(fig_n)

        # Plot 2: Finanzen W(t)
        fig_w, ax_w = plt.subplots(figsize=(7, 3.5))
        ax_w.plot(avg_W['A'], label='A: Standard', color='tab:blue', alpha=0.6)
        ax_w.plot(avg_W['B'], label='B: Fighter', color='tab:red', alpha=0.6)
        ax_w.plot(avg_W['Switch'], label='Switch', color='tab:green', linestyle='--', linewidth=2)
        ax_w.plot(avg_W['Abandon'], label='Abandon', color='black', linestyle=':', linewidth=2)
        ax_w.axhline(0, color='grey', linewidth=0.8)
        ax_w.set_title("Ø Net Value Contribution W(t) per Period")
        ax_w.set_ylabel("Net Value (€)")
        ax_w.legend()
        ax_w.grid(True, alpha=0.3)
        st.pyplot(fig_w)

        # Plot 3: Histogramm (Risiko-Profil)
        fig_hist, ax_hist = plt.subplots(figsize=(7, 3.5))
        # Wir zeigen B, Switch und Abandon (A lassen wir für Übersichtlichkeit raus oder machen es transparent)
        ax_hist.hist(data['B']['Sum'], bins=40, alpha=0.4, label='B (Static)', color='tab:red')
        ax_hist.hist(data['Switch']['Sum'], bins=40, alpha=0.4, label='Switch', color='tab:green')
        ax_hist.hist(data['Abandon']['Sum'], bins=40, alpha=0.4, label='Abandon', color='black')
        ax_hist.set_title("Risk Profile: Distribution of Total Value")
        ax_hist.set_xlabel("Cumulative Value (€)")
        ax_hist.legend()
        ax_hist.grid(True, alpha=0.2)
        st.pyplot(fig_hist)

        # PDF Export
        pdf_buffer = io.BytesIO()
        with PdfPages(pdf_buffer) as pdf:
            pdf.savefig(fig_n)
            pdf.savefig(fig_w)
            pdf.savefig(fig_hist)
            
            # Info Page
            fig_txt, ax_txt = plt.subplots(figsize=(8, 6))
            ax_txt.axis('off')
            txt = (f"SIMULATION REPORT\nRuns: {n_sim} | T: {T}\n\n"
                   f"MEAN TOTAL VALUES:\n"
                   f"Option A:       {means['A']:,.0f} €\n"
                   f"Option B:       {means['B']:,.0f} €\n"
                   f"Option Switch:  {means['Switch']:,.0f} €\n"
                   f"Option Abandon: {means['Abandon']:,.0f} €\n\n"
                   f"OPTION VALUES (vs B):\n"
                   f"Val(Switch):  {val_switch:,.0f} €\n"
                   f"Val(Abandon): {val_abandon:,.0f} €")
            ax_txt.text(0.1, 0.8, txt, family='monospace', fontsize=11)
            pdf.savefig(fig_txt)

        st.download_button("📄 PDF Report Download", pdf_buffer.getvalue(), "thesis_options_report.pdf", "application/pdf", use_container_width=True)

    else:
        st.info("👈 Bitte Parameter anpassen und 'Starten' klicken.")
