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
# 0. STREAMLIT KONFIGURATION & STATUS
# ==========================================
# Setzt den Titel der Browser-Tabs und das Layout auf "Wide" (breit)
st.set_page_config(page_title="Master Thesis Valuation", layout="wide")

# Initialisierung des Session States (Zwischenspeicher für Browser-Sitzung)
# Damit bleiben Daten erhalten, auch wenn man Buttons klickt.
if 'history' not in st.session_state: st.session_state.history = []
if 'simulation_results' not in st.session_state: st.session_state.simulation_results = None
if 'pdf_buffer' not in st.session_state: st.session_state.pdf_buffer = None

# ==========================================
# 1. KERN-LOGIK DER SIMULATION
# ==========================================
def run_simulation(M, p, q, C, ARPU, kappa, Delta_CM, Fixed_Cost, start, T, 
                   mode='static', trigger_val=0.05, fallback_params=None,
                   trigger_logic='share_of_m', trigger_year=3):
    """
    Führt eine einzelne Simulation über T Perioden durch.
    
    Parameter:
    - trigger_logic: 'share_of_m' (Neukunden < % von M) oder 'avg_growth' (Ø Wachstum < %)
    - trigger_year: Das Jahr, bis zu dem geprüft wird bzw. ab dem die Option greift.
    """
    
    # Listen für Kundenbestand (N) und Wertbeitrag (W) initialisieren
    N = [0.0] * T
    W = [0.0] * T
    
    # Startwerte setzen (Periode 0)
    N[0] = start
    W[0] = N[0] * ARPU - Fixed_Cost # Umsatz minus Fixkosten

    # Variable, um zu speichern, ob die Real Option (Abbruch/Wechsel) schon genutzt wurde
    option_exercised = False
    
    # Lokale Kopien der Parameter erstellen.
    # Grund: Bei einem Strategiewechsel (Switch) müssen wir diese Werte überschreiben können,
    # ohne die Original-Funktionsargumente zu verändern.
    curr_p, curr_q, curr_C = p, q, C
    curr_ARPU, curr_kappa, curr_Delta_CM = ARPU, kappa, Delta_CM
    curr_FC = Fixed_Cost
    curr_M = M

    # Liste, um die Wachstumsraten der vergangenen Jahre zu speichern (für die neue Logik)
    past_growth_rates = []

    # Schleife über alle Perioden von 1 bis T-1
    for t in range(1, T):
        N_prev = N[t-1] # Bestand der Vorperiode
        
        # --- 1. VORBERECHNUNG (Was würde mit aktueller Strategie passieren?) ---
        # Potenzielle Neukunden (Bass-Formel) mit aktuellen Parametern berechnen
        potential_acquisition = (curr_p + curr_q * (N_prev / curr_M)) * (curr_M - N_prev)
        if potential_acquisition < 0: potential_acquisition = 0
        
        # Potenzielle Abwanderung
        potential_retention = N_prev * (1 - curr_C)
        
        # Potenzieller neuer Bestand
        potential_N = potential_retention + potential_acquisition
        if potential_N > curr_M: potential_N = curr_M
        
        # Potenzielles Wachstum in dieser Periode (%)
        if N_prev > 0:
            current_growth_rate = (potential_N - N_prev) / N_prev
        else:
            current_growth_rate = 0.0

        # --- 2. REAL OPTION LOGIC (Entscheidung: Weitermachen oder Ändern?) ---
        # Nur prüfen, wenn wir nicht im "static" Modus sind und noch nicht gewechselt haben
        if mode != 'static' and not option_exercised:
            
            trigger_condition_met = False
            
            # FALL A: Alte Logik -> Neukundenanteil am Markt
            # Prüft ab dem trigger_year jedes Jahr
            if trigger_logic == 'share_of_m' and t >= trigger_year:
                # Trigger-Schwelle: z.B. 5% des Marktpotenzials (M)
                trigger_threshold = trigger_val * curr_M
                if potential_acquisition < trigger_threshold:
                    trigger_condition_met = True
            
            # FALL B: Neue Logik -> Durchschnittliches Wachstum
            # Prüft GENAU im trigger_year (z.B. nach Jahr 3 Bilanz ziehen)
            elif trigger_logic == 'avg_growth' and t == trigger_year:
                # Berechne Durchschnitt aller bisherigen Wachstumsraten + aktuelles Jahr
                all_rates = past_growth_rates + [current_growth_rate]
                avg_growth = sum(all_rates) / len(all_rates) if all_rates else 0
                
                # Wenn Ø Wachstum unter dem Schwellenwert (z.B. 15%) liegt -> Alarm
                if avg_growth < trigger_val:
                    trigger_condition_met = True

            # --- 3. AUSFÜHRUNG DER OPTION (Wenn Trigger ausgelöst) ---
            if trigger_condition_met:
                option_exercised = True
                
                if mode == 'switch' and fallback_params:
                    # STRATEGIEWECHSEL: Wir überschreiben ALLE Parameter mit denen von Option A
                    curr_p = fallback_params['p']
                    curr_q = fallback_params['q']
                    curr_C = fallback_params['C']
                    curr_ARPU = fallback_params['ARPU']
                    curr_kappa = fallback_params['kappa']
                    curr_Delta_CM = fallback_params['Delta_CM']
                    curr_FC = fallback_params['Fixed_Cost']
                    
                    # WICHTIG: Die Akquise für DIESES Jahr muss nun neu berechnet werden mit neuen Werten!
                    # Wir simulieren, dass das Management sofort reagiert.
                    
                elif mode == 'abandon':
                    # ABBRUCH: Maximale Abwanderung, Kostenstopp
                    curr_C = 1.0  # Churn = 100% (Alle Kunden weg)
                    curr_FC = 0   # Fixkosten = 0
                    
        # --- 4. ENDGÜLTIGE BERECHNUNG DER PERIODE (Mit ggf. neuen Parametern) ---
        
        # Erneute Berechnung (falls Parameter oben geändert wurden, gelten jetzt die neuen)
        retention = N_prev * (1 - curr_C)
        acquisition = (curr_p + curr_q * (N_prev / curr_M)) * (curr_M - N_prev)
        
        if acquisition < 0: acquisition = 0
        if mode == 'abandon' and option_exercised: acquisition = 0 # Keine Neukunden bei Abbruch
        
        # Bestand aktualisieren
        N[t] = retention + acquisition
        if N[t] > curr_M: N[t] = curr_M
        
        # Tatsächliches Wachstum speichern (für Historie im nächsten Durchlauf)
        actual_growth = (N[t] - N_prev) / N_prev if N_prev > 0 else 0
        past_growth_rates.append(actual_growth)
        
        # Finanzen berechnen
        revenue = N[t] * curr_ARPU
        # Kannibalisierung (Verlust durch Wechsel von Trad. zu Digital)
        cannibalization_loss = acquisition * curr_kappa * curr_Delta_CM
        
        # Net Value Contribution = Umsatz - Kannibalisierung - Fixkosten
        W[t] = revenue - cannibalization_loss - curr_FC
        
    return N, W, sum(W)

# ==========================================
# 2. STATISTIK HELPER (Hilfsfunktionen)
# ==========================================
def calculate_cochran_n(params_dict, T, mode='static', fallback=None, trigger=0.05, trig_logic='share_of_m', trig_year=3):
    """
    Berechnet die statistisch notwendige Anzahl an Simulationen (n), 
    um ein signifikantes Ergebnis zu erhalten (Cochran-Formel).
    """
    pilot_n = 250 # Kleine Test-Runde
    results = []
    
    # Hilfsfunktion, um Zufallswerte aus Ranges zu ziehen (Triangular Distribution)
    def get_val(v): return np.random.triangular(v[0], (v[0]+v[1])/2, v[1]) if isinstance(v, tuple) else v
    
    # Pilot-Simulation durchführen
    for _ in range(pilot_n):
        curr = {k: get_val(v) for k, v in params_dict.items()}
        curr_fb = {k: get_val(v) for k, v in fallback.items()} if fallback else None
        _, _, val = run_simulation(**curr, start=1, T=T, mode=mode, trigger_val=trigger, fallback_params=curr_fb, 
                                   trigger_logic=trig_logic, trigger_year=trig_year)
        results.append(val)
    
    # Standardabweichung und Mittelwert des Pilots berechnen
    std_dev = np.std(results)
    mean_val = np.mean(results)
    
    # Cochran Formel: n = (Z * s / E)^2
    # Wir setzen E (Fehlertoleranz) auf 1% des Mittelwerts für hohe Genauigkeit
    if mean_val == 0: return 1500
    E = abs(mean_val * 0.01) 
    if E == 0: return 1500
    
    n_opt = (1.96 * std_dev / E) ** 2
    
    # Wir geben mindestens 1500 zurück, damit die Grafiken schön glatt sind
    return max(int(math.ceil(n_opt)), 1500)

def get_tornado_data(base_params, ranges, T, mode, trigger, fallback_ranges, trig_logic, trig_year):
    """
    Berechnet die Daten für das Tornado-Diagramm (Sensitivitätsanalyse).
    Es wird jeder Parameter einzeln auf Min und Max gesetzt, während alle anderen konstant bleiben.
    """
    # Mittelwerte (Base Case) definieren
    def mid(v): return (v[0]+v[1])/2 if isinstance(v, tuple) else v
    base_inputs = {k: mid(v) for k, v in ranges.items()}
    fb_inputs = {k: mid(v) for k, v in fallback_ranges.items()} if fallback_ranges else None
    
    # Basis-Ergebnis berechnen
    _, _, base_val = run_simulation(**base_inputs, start=1, T=T, mode=mode, trigger_val=trigger, fallback_params=fb_inputs,
                                    trigger_logic=trig_logic, trigger_year=trig_year)
    
    data = []
    # Loop durch alle Parameter
    for param, val_range in ranges.items():
        if not isinstance(val_range, tuple): continue # Fixwerte überspringen
        
        # Teste unteren Rand (Low Case)
        low_inputs = base_inputs.copy(); low_inputs[param] = val_range[0]
        _, _, val_low = run_simulation(**low_inputs, start=1, T=T, mode=mode, trigger_val=trigger, fallback_params=fb_inputs,
                                       trigger_logic=trig_logic, trigger_year=trig_year)
        
        # Teste oberen Rand (High Case)
        high_inputs = base_inputs.copy(); high_inputs[param] = val_range[1]
        _, _, val_high = run_simulation(**high_inputs, start=1, T=T, mode=mode, trigger_val=trigger, fallback_params=fb_inputs,
                                        trigger_logic=trig_logic, trigger_year=trig_year)
        
        # Speichere die Abweichung vom Basiswert
        data.append({
            "Parameter": param, 
            "Low": val_low - base_val, 
            "High": val_high - base_val, 
            "Range": abs(val_high - val_low)
        })
    # Sortieren nach Einflussstärke (Range)
    return pd.DataFrame(data).sort_values(by="Range", ascending=True), base_val

def get_regression_sensitivity(df_inputs, y_values):
    """
    Führt eine lineare Regression durch, um die globale Sensitivität (Beta-Koeffizienten) zu ermitteln.
    """
    scaler = StandardScaler(); X_scaled = scaler.fit_transform(df_inputs)
    model = LinearRegression(); model.fit(X_scaled, y_values)
    return pd.DataFrame({"Parameter": df_inputs.columns, "Beta": model.coef_}).sort_values(by="Beta", key=abs, ascending=True), model.score(X_scaled, y_values)

# ==========================================
# 3. GUI INPUTS (Benutzeroberfläche)
# ==========================================

# --- SIDEBAR (Verlauf) ---
with st.sidebar:
    st.header("📜 History")
    
    # Funktion zum Wiederherstellen alter Einstellungen
    def restore():
        idx = st.session_state.hist_sel
        if idx is not None:
            entry = st.session_state.history[idx]
            # Lädt gespeicherte Parameter zurück in den Session State
            for k, v in entry['params'].items(): st.session_state[k] = v
            st.toast(f"Geladen: {entry['timestamp']}")
            
    if st.session_state.history:
        opts = {i: f"{e['timestamp']} (M={e['params'].get('M_val', '?')})" for i, e in enumerate(st.session_state.history)}
        st.selectbox("Laden:", list(opts.keys()), format_func=lambda x: opts[x], key="hist_sel", index=None, on_change=restore)

st.markdown("<h1 style='text-align: center;'>Valuing Digital Market Entry Strategies</h1>", unsafe_allow_html=True)

# --- GLOBALE SETTINGS (Oben) ---
with st.container():
    st.markdown("### 🌐 Globale Settings")
    
    # Spaltenaufteilung 1:1:2:1 (Mitte breiter für Trigger)
    c1, c2, c3, c4 = st.columns([1, 1, 2, 1])
    
    with c1: 
        # T: Laufzeit der Simulation
        T_in = st.slider("Jahre (T)", 5, 30, 15, key="T_val")
    with c2: 
        # M: Marktgröße
        M_in = st.number_input("Marktpotenzial (M)", 300, 10000, 500, step=50, key="M_val")
        
    with c3:
        # NEU: Trigger Logik Auswahl
        st.markdown("**Real Option Trigger Logik**")
        col_trig_1, col_trig_2 = st.columns(2)
        with col_trig_1:
            # Auswahl: Welche Metrik löst den Abbruch aus?
            trig_mode_in = st.selectbox("Trigger-Metrik", 
                                        ["share_of_m", "avg_growth"], 
                                        format_func=lambda x: "Neukunden (% von M)" if x == "share_of_m" else "Ø Wachstum (%)",
                                        key="trig_mode_sel")
        with col_trig_2:
            # Auswahl: Wann wird geprüft?
            trig_year_in = st.number_input("Prüf-Jahr (Check Year)", 1, T_in, 3, key="trig_year_val")

        # Dynamische Beschriftung des Sliders je nach Modus
        slider_label = "Trigger-Wert (< % von M)" if trig_mode_in == "share_of_m" else "Trigger-Wert (< % Ø Wachstum)"
        # Default-Werte anpassen: Wachstum ist meist höher (z.B. 15%), Share of M niedriger (z.B. 5%)
        default_trig = 0.05 if trig_mode_in == "share_of_m" else 0.15
        max_trig = 0.20 if trig_mode_in == "share_of_m" else 0.50
        
        trig_val_in = st.slider(slider_label, 0.01, max_trig, default_trig, key="trig_val")

    with c4: 
        st.write(""); st.write("") # Abstandhalter
        start_btn = st.button("🚀 Simulation starten", type="primary", use_container_width=True)

st.markdown("---")

# --- PARAMETER EINGABE (Links / Rechts) ---
col_left, col_right = st.columns(2)

# Helper-Funktion für Min/Max Input Felder
def range_in(lbl, min_v, max_v, sfx, stp=0.01, fmt="%.2f"):
    c1, c2 = st.columns(2)
    k_min, k_max = f"{lbl}_min_{sfx}", f"{lbl}_max_{sfx}"
    # Initialisierung im Session State falls noch nicht vorhanden
    if k_min not in st.session_state: st.session_state[k_min] = min_v
    if k_max not in st.session_state: st.session_state[k_max] = max_v
    return (c1.number_input(f"{lbl} Min", value=st.session_state[k_min], step=stp, format=fmt, key=k_min),
            c2.number_input(f"{lbl} Max", value=st.session_state[k_max], step=stp, format=fmt, key=k_max))

with col_left:
    st.markdown("### 🔵 Option A: Standard (Fallback)")
    # Definition der Parameter-Ranges für die konservative Strategie
    p_a = range_in("p (Innov.)", 0.005, 0.010, "a", 0.001, "%.3f")
    q_a = range_in("q (Imit.)", 0.15, 0.25, "a")
    c_a = range_in("C (Churn)", 0.03, 0.05, "a")
    arpu_a = range_in("ARPU", 3800.0, 4200.0, "a", 100.0)
    fc_a = range_in("Fixkosten", 140000.0, 160000.0, "a", 1000.0)
    kap_a = range_in("Kappa", 0.05, 0.10, "a")
    dcm_a = range_in("Delta Margin", 50.0, 100.0, "a", 10.0)

with col_right:
    st.markdown("### 🔴 Option B: Fighter (Start)")
    # Definition der Parameter-Ranges für die aggressive Strategie
    p_b = range_in("p (Innov.)", 0.030, 0.050, "b", 0.001, "%.3f")
    q_b = range_in("q (Imit.)", 0.20, 0.30, "b")
    c_b = range_in("C (Churn)", 0.08, 0.12, "b")
    arpu_b = range_in("ARPU", 3000.0, 3500.0, "b", 100.0)
    fc_b = range_in("Fixkosten", 180000.0, 200000.0, "b", 1000.0)
    kap_b = range_in("Kappa", 0.10, 0.20, "b")
    dcm_b = range_in("Delta Margin", 50.0, 100.0, "b", 10.0)

# ==========================================
# 4. AUSFÜHRUNG (Wenn Button geklickt)
# ==========================================
if start_btn:
    # Snapshot der Eingaben für die Historie speichern
    snap = {k: v for k, v in st.session_state.items() if "_min_" in k or "_max_" in k or k in ["T_val", "M_val", "trig_val", "trig_mode_sel", "trig_year_val"]}
    st.session_state.history.append({'timestamp': datetime.datetime.now().strftime("%H:%M:%S"), 'params': snap})
    
    # Parameter-Pakete schnüren
    params_A = {'M': (M_in*0.9, M_in*1.1), 'p': p_a, 'q': q_a, 'C': c_a, 'ARPU': arpu_a, 'kappa': kap_a, 'Delta_CM': dcm_a, 'Fixed_Cost': fc_a}
    params_B = {'M': (M_in*0.9, M_in*1.1), 'p': p_b, 'q': q_b, 'C': c_b, 'ARPU': arpu_b, 'kappa': kap_b, 'Delta_CM': dcm_b, 'Fixed_Cost': fc_b}
    
    # Cochran Sample Size Berechnung
    with st.spinner("Berechne optimale Iterationen (Cochran 1%)..."):
        n_A = calculate_cochran_n(params_A, T_in, 'static')
        n_B = calculate_cochran_n(params_B, T_in, 'static')
        # Für die Optionen übergeben wir jetzt auch die neue Trigger-Logik
        n_Switch = calculate_cochran_n(params_B, T_in, 'switch', fallback=params_A, trigger=trig_val_in, trig_logic=trig_mode_in, trig_year=trig_year_in)
        n_Abandon = calculate_cochran_n(params_B, T_in, 'abandon', trigger=trig_val_in, trig_logic=trig_mode_in, trig_year=trig_year_in)
    
    st.toast(f"Runs: A={n_A}, B={n_B}, Sw={n_Switch}, Ab={n_Abandon}")

    # Speicher für Ergebnisse
    res_store = {}
    bar = st.progress(0)
    
    # Definition der 4 Szenarien
    scenarios = [
        ("1. Standard (A)", n_A, params_A, 'static', None, 'tab:blue'),
        ("2. Fighter (B)", n_B, params_B, 'static', None, 'tab:red'),
        ("3. Switch Option", n_Switch, params_B, 'switch', params_A, 'tab:green'),
        ("4. Abandon Option", n_Abandon, params_B, 'abandon', None, 'black')
    ]
    
    # Zufalls-Funktion
    def rnd(v): return np.random.triangular(v[0], (v[0]+v[1])/2, v[1]) if isinstance(v, tuple) else v

    # Hauptschleife über die Szenarien
    for idx, (name, n, p_rng, mode, fb_rng, col) in enumerate(scenarios):
        sim_sums, sim_inputs, all_N, all_W = [], [], [], []
        
        # Monte Carlo Schleife (n Iterationen)
        for _ in range(n):
            curr = {k: rnd(v) for k, v in p_rng.items()}
            fb = {k: rnd(v) for k, v in fb_rng.items()} if fb_rng else None
            
            # Simulation starten
            N_t, W_t, tot = run_simulation(**curr, start=1, T=T_in, mode=mode, 
                                           trigger_val=trig_val_in, fallback_params=fb,
                                           trigger_logic=trig_mode_in, trigger_year=trig_year_in) # Neue Parameter übergeben
            
            sim_sums.append(tot); all_N.append(N_t); all_W.append(W_t); sim_inputs.append(curr)
            
        bar.progress((idx+1)/4)
        
        # Berechnung der Konfidenzintervalle (Ranges) für Plots
        arr_N = np.array(all_N); arr_W = np.array(all_W)
        p5_N = np.percentile(arr_N, 5, axis=0); p95_N = np.percentile(arr_N, 95, axis=0)
        p5_W = np.percentile(arr_W, 5, axis=0); p95_W = np.percentile(arr_W, 95, axis=0)
        
        # Analysen (Tornado & Regression)
        df_in = pd.DataFrame(sim_inputs); df_in_reg = df_in.loc[:, df_in.std() > 0]
        torn, base_v = get_tornado_data(None, p_rng, T_in, mode, trig_val_in, fb_rng, trig_mode_in, trig_year_in)
        reg, r2 = (None, 0)
        if not df_in_reg.empty: reg, r2 = get_regression_sensitivity(df_in_reg, sim_sums)

        # Ergebnis speichern
        res_store[name] = {
            "n": n, "sums": sim_sums, "avg_N": np.mean(all_N, axis=0), "avg_W": np.mean(all_W, axis=0),
            "p5_N": p5_N, "p95_N": p95_N, "p5_W": p5_W, "p95_W": p95_W, # Ranges speichern
            "tornado": (torn, base_v), "regression": (reg, r2), "color": col,
            "mean": np.mean(sim_sums), "std": np.std(sim_sums), 
            "min": np.min(sim_sums), "max": np.max(sim_sums), "var5": np.percentile(sim_sums, 5)
        }
    
    st.session_state.simulation_results = res_store

    # PDF GEN (Im Hintergrund)
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        # Seite 1: Übersicht
        fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8.27, 11.69))
        
        # Plot N(t) mit Transparenz
        for n, d in res_store.items():
            ax1.plot(d["avg_N"], label=n, color=d["color"])
            ax1.fill_between(range(len(d["avg_N"])), d["p5_N"], d["p95_N"], color=d["color"], alpha=0.3)
        ax1.set_title("Customer Adoption N(t) (mit 90% Konfidenz-Intervall)"); ax1.legend(); ax1.grid(True, alpha=0.3)
        
        # Plot W(t) mit Transparenz
        for n, d in res_store.items():
            ax2.plot(d["avg_W"], label=n, color=d["color"])
            ax2.fill_between(range(len(d["avg_W"])), d["p5_W"], d["p95_W"], color=d["color"], alpha=0.3)
        ax2.set_title("Net Value Contribution W(t) per Period"); ax2.grid(True, alpha=0.3); ax2.axhline(0, color='k', linewidth=0.5)
        
        nms = list(res_store.keys()); mus = [res_store[n]["mean"] for n in nms]; sigs = [res_store[n]["std"] for n in nms]
        cols = [res_store[n]["color"] for n in nms]
        ax3.bar(nms, mus, yerr=sigs, capsize=5, alpha=0.7, color=cols)
        ax3.set_title("Total Value Comparison (Mean +/- StdDev)"); ax3.set_ylabel("EUR")
        plt.tight_layout(); pdf.savefig(fig1); plt.close(fig1)

        # Detail-Seiten für jedes Szenario
        for k, d in res_store.items():
            fig, (ax_t, ax_h, ax_r) = plt.subplots(3, 1, figsize=(8.27, 11.69))
            fig.suptitle(f"Detail: {k}", fontsize=16)
            
            # Tornado
            df_t, b_v = d["tornado"]; y = np.arange(len(df_t))
            ax_t.barh(y, df_t["Low"], color='tab:red', alpha=0.6); ax_t.barh(y, df_t["High"], color='tab:green', alpha=0.6)
            ax_t.set_yticks(y); ax_t.set_yticklabels(df_t["Parameter"]); ax_t.invert_yaxis(); ax_t.axvline(0, c='k', ls='--')
            ax_t.set_title(f"Sensitivity (Tornado) - Base: {b_v/1e6:.2f}M")

            # Histogramm
            ax_h.hist(d["sums"], bins=40, color='skyblue', edgecolor='white')
            ax_h.axvline(d["mean"], c='k', ls='--', label=f"Mean: {d['mean']/1e6:.1f}M")
            ax_h.axvline(d["var5"], c='r', ls='--', label=f"VaR 5%: {d['var5']/1e6:.1f}M")
            ax_h.set_title(f"Risk Profile (n={d['n']})"); ax_h.legend()

            # Regression
            df_r, r2 = d["regression"]
            if df_r is not None:
                clrs = ['tab:green' if c > 0 else 'tab:red' for c in df_r["Beta"]]
                ax_r.barh(df_r["Parameter"], df_r["Beta"], color=clrs)
                ax_r.set_title(f"Global Sensitivity (Beta) - R2={r2:.2f}")
            plt.tight_layout(rect=[0, 0.03, 1, 0.95]); pdf.savefig(fig); plt.close(fig)
    st.session_state.pdf_buffer = buf

# ==========================================
# 5. ERGEBNIS-ANZEIGE
# ==========================================
if st.session_state.simulation_results:
    res = st.session_state.simulation_results
    st.markdown("### 📈 Analyse Ergebnisse")
    
    # 1. Zusammenfassung (Tabelle)
    st.markdown("#### Zusammenfassung (Exakte Werte)")
    summary_data = []
    for k, d in res.items():
        summary_data.append({
            "Szenario": k, "Runs": d['n'], "Mean (€)": f"{d['mean']:,.0f}",
            "StdDev (€)": f"{d['std']:,.0f}", "VaR 5% (€)": f"{d['var5']:,.0f}"
        })
    st.dataframe(pd.DataFrame(summary_data).set_index("Szenario"), use_container_width=True)

    # 2. Charts (mit Ranges!)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Customer Evolution $N(t)$")
        fig_n, ax_n = plt.subplots(figsize=(6, 4))
        for n, d in res.items():
            ax_n.plot(d["avg_N"], label=n, color=d["color"])
            ax_n.fill_between(range(len(d["avg_N"])), d["p5_N"], d["p95_N"], color=d["color"], alpha=0.3)
        ax_n.legend(); ax_n.grid(True, alpha=0.3); ax_n.set_ylabel("Kunden")
        st.pyplot(fig_n)
    
    with c2:
        st.markdown("#### Net Value Contribution $W(t)$")
        fig_w, ax_w = plt.subplots(figsize=(6, 4))
        for n, d in res.items():
            ax_w.plot(d["avg_W"], label=n, color=d["color"])
            ax_w.fill_between(range(len(d["avg_W"])), d["p5_W"], d["p95_W"], color=d["color"], alpha=0.3)
        ax_w.legend(); ax_w.grid(True, alpha=0.3); ax_w.set_ylabel("Wertbeitrag (€)"); ax_w.axhline(0, color='k', linewidth=0.8)
        st.pyplot(fig_w)

    # 3. Download Button
    if st.session_state.pdf_buffer:
        st.download_button("📄 PDF Report Download", st.session_state.pdf_buffer.getvalue(), 
                           f"Report_{datetime.datetime.now().strftime('%H%M')}.pdf", "application/pdf", use_container_width=True)
