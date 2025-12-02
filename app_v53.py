import streamlit as st
import pandas as pd
import numpy as np

# --- CONFIGURAZIONE ---
st.set_page_config(page_title="Value Bet Ultimate v53", page_icon="🏆", layout="wide")
st.title("🏆 Calcolatore Strategico (Rankings + Clusters v53)")
st.markdown("---")

# --- FUNZIONI MATEMATICHE ---
def get_implicit_probs(elo_home, elo_away, hfa):
    try:
        exponent = (elo_away - (elo_home + hfa)) / 400
        p_elo_h = 1 / (1 + 10**exponent)
        p_elo_a = 1 - p_elo_h
        return p_elo_h, p_elo_a
    except:
        return 0, 0

def remove_margin(odd_1, odd_x, odd_2):
    try:
        if odd_1 <= 0 or odd_x <= 0 or odd_2 <= 0: return 0, 0, 0
        inv_sum = (1/odd_1) + (1/odd_x) + (1/odd_2)
        return (1/odd_1)/inv_sum, (1/odd_x)/inv_sum, (1/odd_2)/inv_sum
    except:
        return 0, 0, 0

def calculate_row(row, base_hfa, use_dynamic):
    res = {'EV_1': -1, 'EV_X': -1, 'EV_2': -1, 'Fair_1': 0, 'Fair_X': 0, 'Fair_2': 0, 'ELO_Diff': 0, 'HFA_Used': base_hfa}
    try:
        elo_h = float(row.get('elohomeo', 1500))
        elo_a = float(row.get('eloawayo', 1500))
        o1 = float(row.get('cotaa', 0))
        ox = float(row.get('cotae', 0))
        o2 = float(row.get('cotad', 0))
        
        # --- LOGICA RANKING DINAMICO ---
        current_hfa = base_hfa
        if use_dynamic:
            # Cerchiamo le classifiche (Place 1a = rank_h_home, Place 2d = rank_a_away)
            r1 = row.get('rank_h_home')
            r2 = row.get('rank_a_away')
            
            if pd.notna(r1) and pd.notna(r2):
                # Se Casa è 1° e Ospite 20° -> Diff = 19 -> Bonus HFA
                # Se Casa è 20° e Ospite 1° -> Diff = -19 -> Malus HFA
                diff = r2 - r1
                adj = diff * 3 # 3 punti ELO per ogni posizione di differenza
                current_hfa = base_hfa + adj
                current_hfa = max(0, min(current_hfa, 250)) # Limiti di sicurezza
        
        res['HFA_Used'] = current_hfa
        
        # Calcolo con HFA Dinamico
        if o1 > 0 and ox > 0 and o2 > 0:
            pf_1, pf_x, pf_2 = remove_margin(o1, ox, o2)
            p_elo_h, p_elo_a = get_implicit_probs(elo_h, elo_a, current_hfa)
            
            rem = 1 - pf_x
            p_fin_1 = rem * p_elo_h
            p_fin_2 = rem * p_elo_a
            
            res['Fair_1'] = 1/p_fin_1 if p_fin_1>0 else 0
            res['Fair_X'] = 1/pf_x if pf_x>0 else 0
            res['Fair_2'] = 1/p_fin_2 if p_fin_2>0 else 0
            res['EV_1'] = (o1 * p_fin_1) - 1
            res['EV_X'] = (ox * pf_x) - 1
            res['EV_2'] = (o2 * p_fin_2) - 1
            
        res['ELO_Diff'] = (elo_h + current_hfa) - elo_a
        
    except:
        pass
    return pd.Series(res)

# --- CARICAMENTO DATI ---
@st.cache_data(ttl=0)
def load_data(file, base_hfa, use_dynamic):
    try:
        try:
            df = pd.read_csv(file, sep=';', encoding='latin1')
            if len(df.columns) < 5: raise ValueError
        except:
            file.seek(0)
            df = pd.read_csv(file, sep=',', encoding='latin1')

        df.columns = df.columns.str.strip().str.lower()
        df = df.loc[:, ~df.columns.duplicated()] # Fix duplicati
        
        # Mappa Completa (Quote + ELO + Ranking)
        rename_map = {
            '1': 'cotaa', 'x': 'cotae', '2': 'cotad',
            'eloc': 'elohomeo', 'eloo': 'eloawayo',
            'gfinc': 'scor1', 'gfino': 'scor2',
            'o2,5': 'cotao', 'u2,5': 'cotau',
            'data': 'datamecic', 'datameci': 'datamecic',
            'casa': 'txtechipa1', 'ospite': 'txtechipa2',
            'gg': 'cotagg', 'ng': 'cotang',
            # RANKING
            'place 1a': 'rank_h_home', 'place 2d': 'rank_a_away',
            'place 1t': 'rank_h_tot', 'place 2t': 'rank_a_tot'
        }
        
        # Mappa fuzzy
        final_map = {}
        for col in df.columns:
            for k, v in rename_map.items():
                if k in col and col not in final_map.values():
                    final_map[col] = v
                    break
        df = df.rename(columns=final_map)
        
        # Pulizia Numeri
        cols_num = ['cotaa', 'cotae', 'cotad', 'cotao', 'cotau', 'elohomeo', 'eloawayo', 'scor1', 'scor2', 'rank_h_home', 'rank_a_away']
        for c in cols_num:
            if c in df.columns:
                df[c] = df[c].astype(str).str.replace(',', '.', regex=False)
                df[c] = pd.to_numeric(df[c], errors='coerce')

        df = df.dropna(subset=['cotaa', 'cotae', 'cotad', 'elohomeo', 'eloawayo'])
        
        # Calcolo EV (passando i parametri dinamici)
        calc = df.apply(lambda r: calculate_row(r, base_hfa, use_dynamic), axis=1)
        df = pd.concat([df, calc], axis=1)
        
        # Risultati
        df['res_1x2'] = '-' 
        if 'scor1' in df.columns and 'scor2' in df.columns:
            mask = df['scor1'].notna() & df['scor2'].notna()
            df.loc[mask & (df['scor1'] > df['scor2']), 'res_1x2'] = '1'
            df.loc[mask & (df['scor1'] == df['scor2']), 'res_1x2'] = 'X'
            df.loc[mask & (df['scor1'] < df['scor2']), 'res_1x2'] = '2'
            
            df['goals_tot'] = df['scor1'] + df['scor2']
            if 'cotao' in df.columns:
                df['res_o25'] = np.nan
                df.loc[mask, 'res_o25'] = (df.loc[mask, 'goals_tot'] > 2.5).astype(int)
            
        return df, None
    except Exception as e:
        return None, f"Errore: {str(e)}"

# --- MOTORE CLUSTER (Usa i nuovi EV dinamici) ---
def analyze_input(df_hist, odd, ev, market_type):
    if df_hist is None or df_hist.empty: return 0, 0, 0
    if pd.isna(odd) or pd.isna(ev): return 0, 0, 0
    
    f_odd_min, f_odd_max = odd * 0.9, odd * 1.1
    f_ev_min = ev - 0.05 
    
    mask = pd.Series(True, index=df_hist.index)
    target, col_odd = None, None
    if market_type == "1": target, col_odd = '1', 'cotaa'
    elif market_type == "2": target, col_odd = '2', 'cotad'
    elif market_type == "X": target, col_odd = 'X', 'cotae'
    
    if col_odd not in df_hist.columns: return 0, 0, 0

    mask &= (df_hist[col_odd] >= f_odd_min) & (df_hist[col_odd] <= f_odd_max)
    col_ev = f"EV_{market_type}"
    if col_ev in df_hist.columns: mask &= (df_hist[col_ev] >= f_ev_min)
    
    cluster = df_hist[mask]
    
    if len(cluster) >= 5:
        wins = len(cluster[cluster['res_1x2'] == target])
        profit = (cluster[cluster['res_1x2'] == target][col_odd] - 1).sum() - (len(cluster) - wins)
        roi = (profit / len(cluster)) * 100
        return len(cluster), roi, profit
    return 0, 0, 0

# --- INTERFACCIA ---
st.sidebar.header("⚙️ Impostazioni Ranking")
BASE_HFA = st.sidebar.number_input("HFA Base", 100, step=10)
USE_DYN = st.sidebar.checkbox("Usa Classifica per HFA Dinamico", value=True, help="Se attivo, modifica il vantaggio campo in base a Place 1a e Place 2d.")

st.sidebar.header("📂 Caricamento")
file_hist = st.sidebar.file_uploader("1. Carica STORICO (Cervello)", type=["csv"], key="u_hist")
file_fut = st.sidebar.file_uploader("2. Carica FUTURE (Target)", type=["csv"], key="u_fut")

df_history, df_future = None, None

if file_hist:
    df_history, err_h = load_data(file_hist, BASE_HFA, USE_DYN)
    if err_h: st.sidebar.error(err_h)
    else: 
        st.sidebar.success(f"🧠 Storico OK: {len(df_history)} match")
        if USE_DYN and 'rank_h_home' not in df_history.columns:
            st.sidebar.warning("⚠️ Nel file storico mancano le colonne 'Place 1a/2d'. Ranking disattivato.")

if file_fut:
    df_future, err_f = load_data(file_fut, BASE_HFA, USE_DYN)
    if err_f: st.sidebar.error(err_f)
    else: st.sidebar.success(f"🎯 Future OK: {len(df_future)} match")

tab1, tab2, tab3 = st.tabs(["🔮 Manuale", "🚀 Future", "📊 Storico"])

with tab1:
    st.header("Analisi Singola")
    col_input, col_res = st.columns([1, 1])
    with col_input:
        team_h = st.text_input("Casa", "Home")
        team_a = st.text_input("Ospite", "Away")
        c1, c2 = st.columns(2)
        elo_h = c1.number_input("ELO Casa", 1500, min_value=0)
        elo_a = c2.number_input("ELO Ospite", 1500, min_value=0)
        
        # NUOVI INPUT PER IL RANKING MANUALE
        if USE_DYN:
            st.markdown("---")
            c_r1, c_r2 = st.columns(2)
            rank_h = c_r1.number_input("Pos. Casa (in Casa)", 1, 20, 10, help="Place 1a")
            rank_a = c_r2.number_input("Pos. Ospite (Fuori)", 1, 20, 10, help="Place 2d")
        else:
            rank_h, rank_a = None, None

        st.markdown("---")
        c3, c4, c5 = st.columns(3)
        o1 = c3.number_input("Quota 1", 2.00)
        ox = c4.number_input("Quota X", 3.00)
        o2 = c5.number_input("Quota 2", 3.50)
        btn_calc = st.button("Analizza", type="primary")

    with col_res:
        if btn_calc:
            row = {'elohomeo': elo_h, 'eloawayo': elo_a, 'cotaa': o1, 'cotae': ox, 'cotad': o2, 'rank_h_home': rank_h, 'rank_a_away': rank_a}
            res = calculate_row(row, BASE_HFA, USE_DYN)
            
            st.subheader(f"{team_h} vs {team_a}")
            st.info(f"HFA Usato: **{int(res['HFA_Used'])}** (Base: {BASE_HFA})")
            
            def show_smart_card(label, odd, ev, market_type):
                bg = "#28a745" if ev > 0 else "#dc3545" 
                text = "white"
                extra = ""
                if df_history is not None:
                    n, roi, prof = analyze_input(df_history, odd, ev, market_type)
                    if n > 0 and roi > 5: extra = f"✅ CLUSTER: ROI +{roi:.1f}% ({n})"
                    elif n > 0 and roi < -5: extra = f"❌ CLUSTER: ROI {roi:.1f}% ({n})"
                    elif n > 0: extra = f"⚖️ Neutro ({n})"
                
                st.markdown(f"""
                <div style="background-color:{bg}; color:{text}; padding:10px; border-radius:10px; text-align:center; margin-bottom:10px;">
                    <h3>{label}</h3>
                    <h1>{odd:.2f}</h1>
                    <p>EV: {ev:+.1%}</p>
                    <p style="background-color:rgba(0,0,0,0.3); padding:5px; border-radius:5px;">{extra}</p>
                </div>
                """, unsafe_allow_html=True)

            k1, k2, k3 = st.columns(3)
            with k1: show_smart_card("1", o1, res['EV_1'], "1")
            with k2: show_smart_card("X", ox, res['EV_X'],
