import streamlit as st
import pandas as pd
import numpy as np

# --- CONFIGURAZIONE ---
st.set_page_config(page_title="Value Bet Dynamic v49.1", page_icon="📈", layout="wide")
st.title("📈 Calcolatore Strategico (Dynamic HFA Fix)")
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
        
        # --- LOGICA HFA DINAMICO ---
        current_hfa = base_hfa
        if use_dynamic:
            r_h_home = row.get('rank_h_home') # Place 1a
            r_a_away = row.get('rank_a_away') # Place 2d
            
            if pd.notna(r_h_home) and pd.notna(r_a_away):
                rank_diff = r_a_away - r_h_home
                adj = rank_diff * 3
                current_hfa = base_hfa + adj
                current_hfa = max(0, min(current_hfa, 200)) # Limiti
        
        res['HFA_Used'] = current_hfa
        
        # Calcolo
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
        # Rimuove duplicati per evitare errori .str
        df = df.loc[:, ~df.columns.duplicated()]
        
        # Mappa Estesa (copre datameci, data, place 1a, ecc)
        rename_map = {
            '1': 'cotaa', 'x': 'cotae', '2': 'cotad',
            'eloc': 'elohomeo', 'eloo': 'eloawayo',
            'gfinc': 'scor1', 'gfino': 'scor2',
            'data': 'datamecic', 'datameci': 'datamecic',
            'casa': 'txtechipa1', 'ospite': 'txtechipa2',
            'place 1a': 'rank_h_home', 'place 2d': 'rank_a_away',
            'place 1t': 'rank_h_tot', 'place 2t': 'rank_a_tot'
        }
        
        # Mappa fuzzy: se la colonna contiene "place 1a", la rinomina
        final_map = {}
        for col in df.columns:
            for k, v in rename_map.items():
                if k in col and col not in final_map:
                    final_map[col] = v
                    break
        df = df.rename(columns=final_map)
        
        # Pulizia Numeri
        cols_num = ['cotaa', 'cotae', 'cotad', 'elohomeo', 'eloawayo', 'scor1', 'scor2', 'rank_h_home', 'rank_a_away']
        for c in cols_num:
            if c in df.columns:
                df[c] = df[c].astype(str).str.replace(',', '.', regex=False)
                df[c] = pd.to_numeric(df[c], errors='coerce')

        df = df.dropna(subset=['cotaa', 'cotae', 'cotad', 'elohomeo', 'eloawayo'])
        
        # Calcolo
        calc = df.apply(lambda r: calculate_row(r, base_hfa, use_dynamic), axis=1)
        df = pd.concat([df, calc], axis=1)
        
        # Risultati
        df['res_1x2'] = '-' 
        if 'scor1' in df.columns and 'scor2' in df.columns:
            mask = df['scor1'].notna() & df['scor2'].notna()
            df.loc[mask & (df['scor1'] > df['scor2']), 'res_1x2'] = '1'
            df.loc[mask & (df['scor1'] == df['scor2']), 'res_1x2'] = 'X'
            df.loc[mask & (df['scor1'] < df['scor2']), 'res_1x2'] = '2'
            
        return df, None
    except Exception as e:
        return None, f"Errore: {str(e)}"

# --- INTERFACCIA ---
st.sidebar.header("⚙️ Impostazioni Modello")
BASE_HFA = st.sidebar.number_input("HFA Base", value=100, step=10)
USE_DYN = st.sidebar.checkbox("Usa HFA Dinamico (Classifica)", value=True, help="Modifica l'HFA in base alla classifica Casa/Trasferta.")

uploaded_file = st.sidebar.file_uploader("📂 Carica CSV (con Ranking)", type=["csv"])

if uploaded_file:
    df, error_msg = load_data(uploaded_file, BASE_HFA, USE_DYN)
    
    if error_msg:
        st.error(error_msg)
    else:
        st.success(f"Dati caricati! ({len(df)} righe)")
        
        has_rank = 'rank_h_home' in df.columns and 'rank_a_away' in df.columns
        if USE_DYN and not has_rank:
            st.warning("⚠️ Colonne Ranking non trovate. Uso HFA Base.")

        tab1, tab2 = st.tabs(["📊 Analisi Profitto", "🔍 Dettaglio HFA"])
        
        with tab1:
            st.header("Confronto Redditività")
            df_played = df[df['res_1x2'] != '-'].copy()
            
            if not df_played.empty:
                pnl_1 = np.where(df_played['EV_1']>0, np.where(df_played['res_1x2']=='1', df_played['cotaa']-1, -1), 0).sum()
                pnl_2 = np.where(df_played['EV_2']>0, np.where(df_played['res_1x2']=='2', df_played['cotad']-1, -1), 0).sum()
                
                k1, k2 = st.columns(2)
                k1.metric("Profitto Casa (1)", f"{pnl_1:.2f} u", delta="Dinamico" if USE_DYN else "Standard")
                k2.metric("Profitto Ospite (2)", f"{pnl_2:.2f} u", delta="Dinamico" if USE_DYN else "Standard")
                
                # --- VISUALIZZAZIONE SICURA (NO CRASH) ---
                # Cerco le colonne che esistono davvero nel file
                possible_cols = ['datamecic', 'txtechipa1', 'txtechipa2', 'HFA_Used', 'cotaa', 'cotad', 'EV_1', 'EV_2', 'res_1x2']
                final_cols = [c for c in possible_cols if c in df_played.columns]
                
                st.dataframe(df_played[final_cols])
            else:
                st.info("Nessun risultato storico nel file.")
                # Mostra colonne sicure anche per future
                possible_cols = ['datamecic', 'txtechipa1', 'txtechipa2', 'HFA_Used', 'EV_1', 'EV_2']
                final_cols = [c for c in possible_cols if c in df.columns]
                st.dataframe(df[final_cols])

        with tab2:
            st.header("Impatto Classifica su HFA")
            if has_rank:
                st.write("Esempio di come il Ranking modifica il fattore campo:")
                cols_rank = ['txtechipa1', 'rank_h_home', 'txtechipa2', 'rank_a_away', 'HFA_Used']
                safe_cols = [c for c in cols_rank if c in df.columns]
                st.dataframe(df[safe_cols].head(20))
            else:
                st.write("Dati ranking non disponibili nel file.")
