import streamlit as st  # Framework für die Web-App
import matplotlib.pyplot as plt  # Bibliothek für Diagramme
from matplotlib.backends.backend_pdf import PdfPages  # Für den PDF-Export
import numpy as np  # Für numerische Berechnungen und Arrays
import pandas as pd  # Für Datenverarbeitung (DataFrames)
from sklearn.linear_model import LinearRegression  # Für die Sensitivitätsanalyse (Regression)
from sklearn.preprocessing import StandardScaler  # Zum Skalieren der Daten für die Regression
import io  # Für Datenströme (Input/Output im Speicher)
import math  # Mathematische Standardfunktionen
import datetime  # Für Zeitstempel

# ==========================================
# 0. STREAMLIT KONFIGURATION & STATUS
# ==========================================
# Setzt den Seitentitel und das Layout auf die volle Breite
st.set_page_config(page_title="Master Thesis Valuation", layout="wide")

# Initialisierung des Session States (Speicher, der erhalten bleibt, wenn man Buttons klickt)
if 'history' not in st.session_state: st.session_state.history = []  # Liste für den Verlauf
if 'simulation_results' not in st.session_state: st.session_state.simulation_results = None  # Speicher für Ergebnisse
if 'pdf_buffer' not in st.session_state: st.session_state.pdf_buffer = None  # Speicher für das PDF

# ==========================================
# 1. KERN-LOGIK DER SIMULATION
# ==========================================
def run_simulation(M, p, q, C, ARPU, kappa, Delta_CM, Fixed_Cost, start, T, 
                   mode='static', trigger_val=0.05, fallback_params=None,
                   check_mode='continuous', check_year=3):
    """
    Führt eine einzelne Simulation über T Perioden durch.
    Rückgabe: N (Kunden), W (Wert), Summe W, exercised (Wurde Option ausgeübt?)
    """
    
    # Listen für Kundenbestand (N) und Wertbeitrag (W) für alle Jahre initialisieren
    N = [0.0] * T
    W = [0.0] * T
    
    # Startwerte für das Jahr 0 setzen
    N[0] = start
    W[0] = N[0] * ARPU - Fixed_Cost # Umsatz abzüglich Fixkosten im Jahr 0

    # Marker: Wurde die Option (Abbruch/Wechsel) in diesem Lauf schon genutzt?
    option_exercised = False
    
    # Wir erstellen lokale Kopien der Parameter.
    # Grund: Wenn wir die Strategie wechseln ("Switch"), müssen wir p, q, C etc. ändern,
    # ohne die ursprünglichen Variablen der Funktion zu überschreiben.
    curr_p, curr_q, curr_C = p, q, C
    curr_ARPU, curr_kappa, curr_Delta_CM = ARPU, kappa, Delta_CM
    curr_FC = Fixed_Cost
    curr_M = M

    # Liste, um die Wachstumsraten der vergangenen Jahre zu speichern (für den Durchschnitt)
    past_growth_rates = []

    # Hauptschleife: Wir gehen Jahr für Jahr durch (von Jahr 1 bis T-1)
    for t in range(1, T):
        N_prev = N[t-1] # Bestand der Vorperiode
        
        # --- 1. VOR-BERECHNUNG (Szenario mit aktueller Strategie) ---
        # Berechnung der potenziellen Neukunden mit der Bass-Formel
        potential_acquisition = (curr_p + curr_q * (N_prev / curr_M)) * (curr_M - N_prev)
        if potential_acquisition < 0: potential_acquisition = 0 # Keine negativen Neukunden
        
        # Berechnung der potenziellen Kunden, die bleiben (Retention)
        potential_retention = N_prev * (1 - curr_C)
        
        # Potenzieller neuer Gesamtbestand
        potential_N = potential_retention + potential_acquisition
        if potential_N > curr_M: potential_N = curr_M # Darf Marktpotenzial nicht übersteigen
        
        # Berechnung des Wachstums in diesem Jahr (in Prozent)
        if N_prev > 0:
            current_growth_rate = (potential_N - N_prev) / N_prev
        else:
            current_growth_rate = 0.0

        # --- 2. REAL OPTION LOGIC (Soll abgebrochen/gewechselt werden?) ---
        # Wir prüfen nur, wenn es kein statisches Szenario ist und wir noch nicht gewechselt haben
        if mode != 'static' and not option_exercised:
            
            should_check = False # Soll in diesem Jahr geprüft werden?
            
            # Entscheidung basierend auf dem gewählten Modus
            if check_mode == 'specific':
                # Modus "Einmalig": Wir prüfen NUR im exakten Zieljahr
                if t == check_year: should_check = True
            elif check_mode == 'continuous':
                # Modus "Fortlaufend": Wir prüfen JEDES Jahr ab dem Startjahr
                if t >= check_year: should_check = True
            
            # Wenn geprüft werden soll:
            if should_check:
                # Wir berechnen das durchschnittliche Wachstum aller Jahre bis jetzt
                all_rates = past_growth_rates + [current_growth_rate]
                avg_growth = sum(all_rates) / len(all_rates) if all_rates else 0
                
                # TRIGGER: Ist das durchschnittliche Wachstum schlechter als der Grenzwert?
                if avg_growth < trigger_val:
                    option_exercised = True # Option wird ausgeübt!
                    
                    # FALL A: Switch Option (Wechsel zur Standard-Strategie)
                    if mode == 'switch' and fallback_params:
                        # Wir überschreiben ALLE Parameter mit denen von Option A (Fallback)
                        curr_p = fallback_params['p']
                        curr_q = fallback_params['q']
                        curr_C = fallback_params['C']
                        curr_ARPU = fallback_params['ARPU']
                        curr_kappa = fallback_params['kappa']
                        curr_Delta_CM = fallback_params['Delta_CM']
                        curr_FC = fallback_params['Fixed_Cost']
                        
                        # Wichtig: Die Akquise für DIESES Jahr muss nun neu berechnet werden,
                        # da wir so tun, als hätte das Management sofort reagiert.
                        
                    # FALL B: Abandon Option (Abbruch)
                    elif mode == 'abandon':
                        curr_C = 1.0  # 100% Churn -> Alle Kunden gehen
                        curr_FC = 0   # Keine Fixkosten mehr
                        # Akquise wird unten auf 0 gesetzt
                    
        # --- 3. ENDGÜLTIGE BERECHNUNG (Mit den (evtl. neuen) Parametern) ---
        
        # Retention und Akquise neu berechnen (falls Parameter oben geändert wurden)
        retention = N_prev * (1 - curr_C)
        acquisition = (curr_p + curr_q * (N_prev / curr_M)) * (curr_M - N_prev)
        
        # Sicherheitsregeln
        if acquisition < 0: acquisition = 0
        if mode == 'abandon' and option_exercised: acquisition = 0 # Bei Abbruch keine Neukunden mehr
        
        # Bestand aktualisieren
        N[t] = retention + acquisition
        if N[t] > curr_M: N[t] = curr_M
        
        # Tatsächliches Wachstum speichern (für die Historie im nächsten Jahr)
        actual_growth = (N[t] - N_prev) / N_prev if N_prev > 0 else 0
        past_growth_rates.append(actual_growth)
        
        # Finanzen berechnen
        revenue = N[t] * curr_ARPU # Umsatz
        cannibalization_loss = acquisition * curr_kappa * curr_Delta_CM # Kannibalisierung
        
        # Net Value Contribution = Umsatz - Kannibalisierung - Fixkosten
        W[t] = revenue - cannibalization_loss - curr_FC
        
    # Rückgabe der Zeitreihen, der Summe und ob die Option gezogen wurde
    return N, W, sum(W), option_exercised

# ==========================================
# 2. STATISTIK HELPER (Hilfsfunktionen)
# ==========================================
def calculate_cochran_n(params_dict, T, mode='static', fallback=None, trigger=0.05, c_mode='continuous', c_year=3):
    """
    Berechnet die statistisch notwendige Anzahl an Simulationen (n).
    """
    pilot_n = 250 # Kleine Test-Runde
    results = []
    
    # Hilfsfunktion für Zufallswerte (Dreiecksverteilung)
    def get_val(v): return np.random.triangular(v[0], (v[0]+v[1])/2, v[1]) if isinstance(v, tuple) else v
    
    # Pilot-Simulationen durchführen
    for _ in range(pilot_n):
        curr = {k: get_val(v) for k, v in params_dict.items()}
        curr_fb = {k: get_val(v) for k, v in fallback.items()} if fallback else None
        # Wir ignorieren hier den 4. Rückgabewert (exercised), da wir nur den Wert (val) brauchen
        _, _, val, _ = run_simulation(**curr, start=1, T=T, mode=mode, trigger_val=trigger, fallback_params=curr_fb, 
                                      check_mode=c_mode, check_year=c_year)
        results.append(val)
    
    # Statistik berechnen
    std_dev = np.std(results)
    mean_val = np.mean(results)
    
    # Cochran Formel anwenden (1% Fehlertoleranz)
    if mean_val == 0: return 1500
    E = abs(mean_val * 0.01) 
    if E == 0: return 1500
    
    n_opt = (1.96 * std_dev / E) ** 2
    
    # Mindestens 1500 Simulationen für glatte Kurven
    return max(int(math.ceil(n_opt)), 1500)

def get_tornado_data(base_params, ranges, T, mode, trigger, fallback_ranges, c_mode, c_year):
    """
    Berechnet Daten für das Tornado-Diagramm (Sensitivitätsanalyse).
    """
    # Mittelwerte für alle Parameter berechnen (Base Case)
    def mid(v): return (v[0]+v[1])/2 if isinstance(v, tuple) else v
    base_inputs = {k: mid(v) for k, v in ranges.items()}
    fb_inputs = {k: mid(v) for k, v in fallback_ranges.items()} if fallback_ranges else None
    
    # Basis-Ergebnis
    _, _, base_val, _ = run_simulation(**base_inputs, start=1, T=T, mode=mode, trigger_val=trigger, fallback_params=fb_inputs,
                                       check_mode=c_mode, check_year=c_year)
    
    data = []
    # Jeden Parameter einzeln variieren
    for param, val_range in ranges.items():
        if not isinstance(val_range, tuple): continue # Fixwerte überspringen
        
        # Teste unteren Rand (Low)
        low_inputs = base_inputs.copy(); low_inputs[param] = val_range[0]
        _, _, val_low, _ = run_simulation(**low_inputs, start=1, T=T, mode=mode, trigger_val=trigger, fallback_params=fb_inputs,
                                          check_mode=c_mode, check_year=c_year)
        
        # Teste oberen Rand (High)
        high_inputs = base_inputs.copy(); high_inputs[param] = val_range[1]
        _, _, val_high, _ = run_simulation(**high_inputs, start=1, T=T, mode=mode, trigger_val=trigger, fallback_params=fb_inputs,
                                           check_mode=c_mode, check_year=c_year)
        
        data.append({
            "Parameter": param, 
            "Low": val_low - base_val, 
            "High": val_high - base_val, 
            "Range": abs(val_high - val_low)
        })
    return pd.DataFrame(data).sort_values(by="Range", ascending=True), base_val

def get_regression_sensitivity(df_inputs, y_values):
    """
    Berechnet Beta-Koeffizienten mittels linearer Regression.
    """
    scaler = StandardScaler(); X_scaled = scaler.fit_transform(df_inputs)
    model = LinearRegression(); model.fit(X_scaled, y_values)
    return pd.DataFrame({"Parameter": df_inputs.columns, "Beta": model.coef_}).sort_values(by="Beta", key=abs, ascending=True), model.score(X_scaled, y_values)

# ==========================================
# 3. GUI EINGABEN (Benutzeroberfläche)
# ==========================================

# --- SIDEBAR (Verlauf laden) ---
with st.sidebar:
    st.header("📜 History")
    
    def restore():
        idx = st.session_state.hist_sel
        if idx is not None:
            entry = st.session_state.history[idx]
            # Alte Parameter in den Session State laden
            for k, v in entry['params'].items(): st.session_state[k] = v
            st.toast(f"Geladen: {entry['timestamp']}")
            
    if st.session_state.history:
        opts = {i: f"{e['timestamp']} (M={e['params'].get('M_val', '?')})" for i, e in enumerate(st.session_state.history)}
        st.selectbox("Laden:", list(opts.keys()), format_func=lambda x: opts[x], key="hist_sel", index=None, on_change=restore)

st.markdown("<h1 style='text-align: center;'>Valuing Digital Market Entry Strategies</h1>", unsafe_allow_html=True)

# --- GLOBALE EINSTELLUNGEN ---
with st.container():
    st.markdown("### 🌐 Globale Settings & Trigger Logik")
    
    # Spaltenaufteilung: Zeitraum, Markt, Trigger-Einstellungen, Start-Button
    c1, c2, c3, c4 = st.columns([1, 1, 2, 1])
    
    with c1: 
        T_in = st.slider("Jahre (T)", 5, 30, 15, key="T_val") # Laufzeit
    with c2: 
        M_in = st.number_input("Marktpotenzial (M)", 300, 10000, 500, step=50, key="M_val") # Marktgröße
        
    with c3:
        # NEU: Einstellungen für die Trigger-Logik
        st.markdown("**Option Trigger (Abbruchbedingung)**")
        col_trig_1, col_trig_2 = st.columns(2)
        
        with col_trig_1:
            # Auswahl: Einmalig oder Fortlaufend?
            check_mode_in = st.radio("Prüf-Modus", 
                                     ["specific", "continuous"], 
                                     format_func=lambda x: "Einmalig (bestimmtes Jahr)" if x == "specific" else "Fortlaufend (Jedes Jahr)",
                                     key="check_mode_sel")
        with col_trig_2:
            # Label ändert sich je nach Modus
            year_label = "Prüf-Jahr" if check_mode_in == "specific" else "Start-Jahr der Prüfung"
            check_year_in = st.number_input(year_label, 1, T_in, 3, key="check_year_val")

        # Slider für den Grenzwert (Durchschnittliches Wachstum)
        trig_val_in = st.slider("Trigger: Ø Wachstum unter X %", 0.01, 0.50, 0.15, key="trig_val")

    with c4: 
        st.write(""); st.write("") # Platzhalter für Layout
        start_btn = st.button("🚀 Simulation starten", type="primary", use_container_width=True)

st.markdown("---")

# --- PARAMETER EINGABE (Links / Rechts) ---
col_left, col_right = st.columns(2)

# Helper-Funktion für Min/Max Input Felder
def range_in(lbl, min_v, max_v, sfx, stp=0.01, fmt="%.2f"):
    c1, c2 = st.columns(2)
    k_min, k_max = f"{lbl}_min_{sfx}", f"{lbl}_max_{sfx}"
    if k_min not in st.session_state: st.session_state[k_min] = min_v
    if k_max not in st.session_state: st.session_state[k_max] = max_v
    return (c1.number_input(f"{lbl} Min", value=st.session_state[k_min], step=stp, format=fmt, key=k_min),
            c2.number_input(f"{lbl} Max", value=st.session_state[k_max], step=stp, format=fmt, key=k_max))

with col_left:
    st.markdown("### 🔵 Option A: Standard (Fallback)")
    # Eingabefelder für Szenario A
    p_a = range_in("p (Innov.)", 0.005, 0.010, "a", 0.001, "%.3f")
    q_a = range_in("q (Imit.)", 0.15, 0.25, "a")
    c_a = range_in("C (Churn)", 0.03, 0.05, "a")
    arpu_a = range_in("ARPU", 3800.0, 4200.0, "a", 100.0)
    fc_a = range_in("Fixkosten", 140000.0, 160000.0, "a", 1000.0)
    kap_a = range_in("Kappa", 0.05, 0.10, "a")
    dcm_a = range_in("Delta Margin", 50.0, 100.0, "a", 10.0)

with col_right:
    st.markdown("### 🔴 Option B: Fighter (Start)")
    # Eingabefelder für Szenario B
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
    # Snapshot der Eingaben für die History speichern
    snap = {k: v for k, v in st.session_state.items() if "_min_" in k or "_max_" in k or k in ["T_val", "M_val", "trig_val", "check_mode_sel", "check_year_val"]}
    st.session_state.history.append({'timestamp': datetime.datetime.now().strftime("%H:%M:%S"), 'params': snap})
    
    # Parameter-Pakete erstellen
    params_A = {'M': (M_in*0.9, M_in*1.1), 'p': p_a, 'q': q_a, 'C': c_a, 'ARPU': arpu_a, 'kappa': kap_a, 'Delta_CM': dcm_a, 'Fixed_Cost': fc_a}
    params_B = {'M': (M_in*0.9, M_in*1.1), 'p': p_b, 'q': q_b, 'C': c_b, 'ARPU': arpu_b, 'kappa': kap_b, 'Delta_CM': dcm_b, 'Fixed_Cost': fc_b}
    
    # Cochran Berechnung für die Iterationen
    with st.spinner("Berechne optimale Iterationen (Cochran 1%)..."):
        n_A = calculate_cochran_n(params_A, T_in, 'static')
        n_B = calculate_cochran_n(params_B, T_in, 'static')
        # Neue Trigger-Parameter übergeben
        n_Switch = calculate_cochran_n(params_B, T_in, 'switch', fallback=params_A, trigger=trig_val_in, c_mode=check_mode_in, c_year=check_year_in)
        n_Abandon = calculate_cochran_n(params_B, T_in, 'abandon', trigger=trig_val_in, c_mode=check_mode_in, c_year=check_year_in)
    
    st.toast(f"Runs: A={n_A}, B={n_B}, Sw={n_Switch}, Ab={n_Abandon}")

    # Ergebnis-Speicher
    res_store = {}
    bar = st.progress(0)
    
    # Szenarien definieren
    scenarios = [
        ("1. Standard (A)", n_A, params_A, 'static', None, 'tab:blue'),
        ("2. Fighter (B)", n_B, params_B, 'static', None, 'tab:red'),
        ("3. Switch Option", n_Switch, params_B, 'switch', params_A, 'tab:green'),
        ("4. Abandon Option", n_Abandon, params_B, 'abandon', None, 'black')
    ]
    
    def rnd(v): return np.random.triangular(v[0], (v[0]+v[1])/2, v[1]) if isinstance(v, tuple) else v

    # Hauptschleife über die 4 Szenarien
    for idx, (name, n, p_rng, mode, fb_rng, col) in enumerate(scenarios):
        sim_sums, sim_inputs, all_N, all_W = [], [], [], []
        exercised_count = 0 # Zähler für Ausübungen
        
        # Monte Carlo Runs
        for _ in range(n):
            curr = {k: rnd(v) for k, v in p_rng.items()}
            fb = {k: rnd(v) for k, v in fb_rng.items()} if fb_rng else None
            
            # Simulation starten
            N_t, W_t, tot, exc = run_simulation(**curr, start=1, T=T_in, mode=mode, 
                                           trigger_val=trig_val_in, fallback_params=fb,
                                           check_mode=check_mode_in, check_year=check_year_in) # Neue Parameter
            
            sim_sums.append(tot); all_N.append(N_t); all_W.append(W_t); sim_inputs.append(curr)
            if exc: exercised_count += 1 # Zählen wenn ausgeübt
            
        bar.progress((idx+1)/4)
        
        # Statistik für Plots
        arr_N = np.array(all_N); arr_W = np.array(all_W)
        p5_N = np.percentile(arr_N, 5, axis=0); p95_N = np.percentile(arr_N, 95, axis=0)
        p5_W = np.percentile(arr_W, 5, axis=0); p95_W = np.percentile(arr_W, 95, axis=0)
        
        # Tornado & Regression
        df_in = pd.DataFrame(sim_inputs); df_in_reg = df_in.loc[:, df_in.std() > 0]
        torn, base_v = get_tornado_data(None, p_rng, T_in, mode, trig_val_in, fb_rng, check_mode_in, check_year_in)
        reg, r2 = (None, 0)
        if not df_in_reg.empty: reg, r2 = get_regression_sensitivity(df_in_reg, sim_sums)

        # Ergebnis speichern (inklusive Ausübungsquote)
        res_store[name] = {
            "n": n, "sums": sim_sums, "avg_N": np.mean(all_N, axis=0), "avg_W": np.mean(all_W, axis=0),
            "p5_N": p5_N, "p95_N": p95_N, "p5_W": p5_W, "p95_W": p95_W,
            "tornado": (torn, base_v), "regression": (reg, r2), "color": col,
            "mean": np.mean(sim_sums), "std": np.std(sim_sums), 
            "min": np.min(sim_sums), "max": np.max(sim_sums), "var5": np.percentile(sim_sums, 5),
            "exercised_rate": (exercised_count / n) * 100 # Quote berechnen
        }
    
    st.session_state.simulation_results = res_store

    # PDF Generierung (Im Hintergrund)
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        # Übersicht Plots
        fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8.27, 11.69))
        
        for n, d in res_store.items():
            ax1.plot(d["avg_N"], label=n, color=d["color"])
            ax1.fill_between(range(len(d["avg_N"])), d["p5_N"], d["p95_N"], color=d["color"], alpha=0.3)
        ax1.set_title("Customer Adoption N(t) (mit 90% Konfidenz-Intervall)"); ax1.legend(); ax1.grid(True, alpha=0.3)
        
        for n, d in res_store.items():
            ax2.plot(d["avg_W"], label=n, color=d["color"])
            ax2.fill_between(range(len(d["avg_W"])), d["p5_W"], d["p95_W"], color=d["color"], alpha=0.3)
        ax2.set_title("Net Value Contribution W(t) per Period"); ax2.grid(True, alpha=0.3); ax2.axhline(0, color='k', linewidth=0.5)
        
        nms = list(res_store.keys()); mus = [res_store[n]["mean"] for n in nms]; sigs = [res_store[n]["std"] for n in nms]
        cols = [res_store[n]["color"] for n in nms]
        ax3.bar(nms, mus, yerr=sigs, capsize=5, alpha=0.7, color=cols)
        ax3.set_title("Total Value Comparison (Mean +/- StdDev)"); ax3.set_ylabel("EUR")
        plt.tight_layout(); pdf.savefig(fig1); plt.close(fig1)

        # Detail-Seiten
        for k, d in res_store.items():
            fig, (ax_t, ax_h, ax_r) = plt.subplots(3, 1, figsize=(8.27, 11.69))
            fig.suptitle(f"Detail: {k}", fontsize=16)
            
            df_t, b_v = d["tornado"]; y = np.arange(len(df_t))
            ax_t.barh(y, df_t["Low"], color='tab:red', alpha=0.6); ax_t.barh(y, df_t["High"], color='tab:green', alpha=0.6)
            ax_t.set_yticks(y); ax_t.set_yticklabels(df_t["Parameter"]); ax_t.invert_yaxis(); ax_t.axvline(0, c='k', ls='--')
            ax_t.set_title(f"Sensitivity (Tornado) - Base: {b_v/1e6:.2f}M")

            ax_h.hist(d["sums"], bins=40, color='skyblue', edgecolor='white')
            ax_h.axvline(d["mean"], c='k', ls='--', label=f"Mean: {d['mean']/1e6:.1f}M")
            ax_h.axvline(d["var5"], c='r', ls='--', label=f"VaR 5%: {d['var5']/1e6:.1f}M")
            ax_h.set_title(f"Risk Profile (n={d['n']})"); ax_h.legend()

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
    st.markdown("#### Zusammenfassung")
    summary_data = []
    for k, d in res.items():
        summary_data.append({
            "Szenario": k, 
            "Runs": d['n'], 
            "Mean (€)": f"{d['mean']:,.0f}",
            "StdDev (€)": f"{d['std']:,.0f}", 
            "VaR 5% (€)": f"{d['var5']:,.0f}",
            "Ausübung %": f"{d['exercised_rate']:.1f}%" # Neue Spalte!
        })
    st.dataframe(pd.DataFrame(summary_data).set_index("Szenario"), use_container_width=True)

    # 2. Charts
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

    # 3. Download
    if st.session_state.pdf_buffer:
        st.download_button("📄 PDF Report Download", st.session_state.pdf_buffer.getvalue(), 
                           f"Report_{datetime.datetime.now().strftime('%H%M')}.pdf", "application/pdf", use_container_width=True)
