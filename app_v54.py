import streamlit as st
import pandas as pd
import numpy as np

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="Value Bet V55", page_icon="🛡️", layout="wide")
st.title("🛡️ Calcolatore Strategico (Final V55)")
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
    res = {'EV_1': -1, 'EV_X': -1, 'EV_2': -1, 'HFA_Used': base_hfa}
    
    try:
        # Funzione helper per convertire in float in sicurezza
        def get_float(val):
            try:
                return float(str(val).replace(',', '.'))
            except:
                return 0.0

        elo_h = get_float(row.get('elohomeo', 1500))
        elo_a = get_float(row.get('eloawayo', 1500))
        o1 = get_float(row.get('cotaa', 0))
        ox = get_float(row.get('cotae', 0))
        o2 = get_float(row.get('cotad', 0))
        
        # --- LOGICA HFA DINAMICO ---
        current_hfa = base_hfa
        if use_dynamic:
            r_h_home = row.get('rank_h_home') 
            r_a_away = row.get('rank_a_away') 
            
            if pd.notna(r_h_home) and pd.notna(r_a_away):
                try:
                    rank_diff = float(r_a_away) - float(r_h_home)
                    adj = rank_diff * 3
                    current_hfa = base_hfa + adj
                    current_hfa = max(0, min(current_hfa, 200))
                except:
                    pass
        
        res['HFA_Used'] = current_hfa
        
        # Calcolo
        pf_1, pf_x, pf_2 = remove_margin(o1, ox, o2)
        p_elo_h, p_elo_a = get_implicit_probs(elo_h, elo_a, current_hfa)
        
        rem = 1 - pf_x
        p_fin_1 = rem * p_elo_h
        p_fin_2 = rem * p_elo_a
        
        res['EV_1'] = (o1 * p_fin_1) - 1
        res['EV_X'] = (ox * pf_x) - 1
        res['EV_2'] = (o2 * p_fin_2) - 1
        
    except:
        pass
        
    return pd.Series(res)

# --- CARICAMENTO DATI ---
@st.cache_data(ttl=0)
def load_data(file, base_hfa, use_dynamic):
    try:
        # Tenta lettura con ; poi con ,
        try:
            df = pd.read_csv(file, sep=';', encoding='latin1', on_bad_lines='skip', engine='python')
            if len(df.columns) < 3: raise ValueError
        except:
            file.seek(0)
            df = pd.read_csv(file, sep=',', encoding='latin1', on_bad_lines='skip', engine='python')

        df.columns = df.columns.str.strip().str.lower()
        df = df.loc[:, ~df.columns.duplicated()] 
        
        # MAPPING ESTESO
        rename_map = {
            '1': 'cotaa', 'x': 'cotae', '2': 'cotad',
            'eloc': 'elohomeo', 'eloo': 'eloawayo',
            'gfinc': 'scor1', 'gfino': 'scor2',
            'data': 'datamecic', 'datameci': 'datamecic',
            'casa': 'txtechipa1', 'ospite': 'txtechipa2',
            # Rank varianti
            'place1a': 'rank_h_home', 'place 1a': 'rank_h_home',
            'place2d': 'rank_a_away', 'place 2d': 'rank_a_away',
            'place1t': 'rank_h_tot',  'place 1t': 'rank_h_tot',
            'place2t': 'rank_a_tot',  'place 2t': 'rank_a_tot'
        }
        
        cols_found = list(df.columns)
        final_rename = {}
        for col in cols_found:
            for key, val in rename_map.items():
                if key in col:
                    final_rename[col] = val
                    break
        
        df = df.rename(columns=final_rename)
        df = df.loc[:, ~df.columns.duplicated()]

        # Pulizia numeri
        cols_num = ['cotaa', 'cotae', 'cotad', 'elohomeo', 'eloawayo', 'scor1', 'scor2', 'rank_h_home', 'rank_a_away']
        for c in cols_num:
            if c in df.columns:
                df[c] = df[c].astype(str).str.replace(',', '.', regex=False)
                df[c] = pd.to_numeric(df[c], errors='coerce')

        df = df.dropna(subset=['cotaa', 'cotae', 'cotad', 'elohomeo', 'eloawayo'])
        
        # Calcolo
        if not df.empty:
            calc = df.apply(lambda r: calculate_row(r, base_hfa, use_dynamic), axis=1)
            df = pd.concat([df, calc], axis=1)
        
            # Determina Risultato 1X2
            df['res_1x2'] = '-' 
            if 'scor1' in df.columns and 'scor2' in df.columns:
                mask = df['scor1'].notna() & df['scor2'].notna()
                df.loc[mask & (df['scor1'] > df['scor2']), 'res_1x2'] = '1'
                df.loc[mask & (df['scor1'] == df['scor2']), 'res_1x2'] = 'X'
                df.loc[mask & (df['scor1'] < df['scor2']), 'res_1x2'] = '2'
            
        return df, None
    except Exception as e:
        return None, str(e)

# --- INTERFACCIA ---
st.sidebar.header("⚙️ Impostazioni")
BASE_HFA = st.sidebar.number_input("HFA Base", value=100, step=10)
USE_DYN = st.sidebar.checkbox("Usa HFA Dinamico", value=True)

uploaded_file = st.sidebar.file_uploader("📂 Carica CSV", type=["csv"])

if uploaded_file:
    df, error_msg = load_data(uploaded_file, BASE_HFA, USE_DYN)
    
    if error_msg:
        st.error(error_msg)
    elif df is not None and not df.empty:
        st.success(f"Dati caricati: {len(df)} partite")
        
        has_rank = 'rank_h_home' in df.columns and 'rank_a_away' in df.columns
        if USE_DYN and not has_rank:
            st.warning("⚠️ Ranking non trovato (Place1a/Place2d). Uso HFA standard.")
        elif USE_DYN:
            st.info("✅ Ranking attivo.")

        tab1, tab2 = st.tabs(["📊 Profitto", "🔍 Dati"])
        
        with tab1:
            df_played = df[df['res_1x2'] != '-'].copy()
            if not df_played.empty:
                pnl_1 = np.where(df_played['EV_1']>0, np.where(df_played['res_1x2']=='1', df_played['cotaa']-1, -1), 0).sum()
                pnl_2 = np.where(df_played['EV_2']>0, np.where(df_played['res_1x2']=='2', df_played['cotad']-1, -1), 0).sum()
                
                c1, c2 = st.columns(2)
                c1.metric("Profitto Casa", f"{pnl_1:.2f} u")
                c2.metric("Profitto Ospite", f"{pnl_2:.2f} u")
                
                # Lista divisa su più righe per evitare errori di copia
                cols_show = [
                    'datamecic', 'txtechipa1', 'txtechipa2', 
                    'HFA_Used', 'cotaa', 'cotad', 
                    'EV
