import streamlit as st
import pandas as pd
import numpy as np

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="Value Bet V53", page_icon="🛡️", layout="wide")
st.title("🛡️ Calcolatore Strategico (Full Fix V53)")
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
    # Inizializza risultati default
    res = {'EV_1': -1, 'EV_X': -1, 'EV_2': -1, 'Fair_1': 0, 'Fair_X': 0, 'Fair_2': 0, 'ELO_Diff': 0, 'HFA_Used': base_hfa}
    
    try:
        # Recupero dati con fallback sicuri
        elo_h = float(row.get('elohomeo', 1500))
        elo_a = float(row.get('eloawayo', 1500))
        o1 = float(row.get('cotaa', 0))
        ox = float(row.get('cotae', 0))
        o2 = float(row.get('cotad', 0))
        
        # --- LOGICA HFA DINAMICO ---
        current_hfa = base_hfa
        if use_dynamic:
            r_h_home = row.get('rank_h_home') # Mapped from Place1a
            r_a_away = row.get('rank_a_away') # Mapped from Place2d
            
            if pd.notna(r_h_home) and pd.notna(r_a_away):
                rank_diff = r_a_away - r_h_home
                adj = rank_diff * 3
                current_hfa = base_hfa + adj
                current_hfa = max(0, min(current_hfa, 200)) # Limiti HFA (0-200)
        
        res['HFA_Used'] = current_hfa
        
        # Calcolo Probabilità
        pf_1, pf_x, pf_2 = remove_margin(o1, ox, o2)
        p_elo_h, p_elo_a = get_implicit_probs(elo_h, elo_a, current_hfa)
        
        # Distribuzione probabilità su 1X2
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
        pass # Se fallisce il calcolo, ritorna i valori default
        
    return pd.Series(res)

# --- CARICAMENTO DATI ---
@st.cache_data(ttl=0)
def load_data(file, base_hfa, use_dynamic):
    try:
        # Lettura CSV flessibile
        try:
            df = pd.read_csv(file, sep=';', encoding='latin1')
            if len(df.columns) < 5: raise ValueError
        except:
            file.seek(0)
            df = pd.read_csv(file, sep=',', encoding='latin1')

        # 1. Normalizza nomi (tutto minuscolo, niente spazi ai bordi)
        df.columns = df.columns.str.strip().str.lower()
        
        # 2. RIMUOVI COLONNE DUPLICATE (Fondamentale)
        df = df.loc[:, ~df.columns.duplicated()]
        
        # 3. Mappa nomi (AGGIORNATA per Place1a/Place2d)
        rename_map = {
            '1': 'cotaa', 'x': 'cotae', '2': 'cotad',
            'eloc': 'elohomeo', 'eloo': 'eloawayo',
            'gfinc': 'scor1', 'gfino': 'scor2',
            'data': 'datamecic', 'datameci': 'datamecic',
            'casa': 'txtechipa1', 'ospite': 'txtechipa2',
            
            # MAPPING CORRETTO PER IL TUO EXCEL
            'place1a': 'rank_h_home', 'place 1a': 'rank_h_home',
            'place2d': 'rank_a_away', 'place 2d': 'rank_a_away',
            'place1t': 'rank_h_tot',  'place 1t': 'rank_h_tot',
            'place2t': 'rank_a_tot',  'place 2t': 'rank_a_tot'
        }
        
        # Applica mappa nomi
        cols_found = list(df.columns)
        final_rename = {}
        for col in cols_found:
            for key, val in rename_map.items():
                if key in col: # Cerca la chiave nel nome colonna
                    final_rename[col] = val
                    break
        
        df = df.rename(columns=final_rename)
        
        # Rimuovi duplicati di nuovo (sicurezza extra)
        df = df.loc[:, ~df.columns.duplicated()]

        # Pulizia Numeri (gestione virgola/punto)
        cols_num = ['cotaa', 'cotae', 'cotad', 'elohomeo', 'eloawayo', 'scor1', 'scor2', 'rank_h_home', 'rank_a_away']
        for c in cols_num:
            if c in df.columns:
                df[c] = df[c].astype(str).str.replace(',', '.', regex=False)
                df[c] = pd.to_numeric(df[c], errors='coerce')

        # Filtra righe senza quote o ELO
        df = df.dropna(subset=['cotaa', 'cotae', 'cotad', 'elohomeo', 'eloawayo'])
        
        # Esegui Calcoli
        calc = df.apply(lambda r: calculate_row(r, base_hfa, use_dynamic), axis=1)
        df = pd.concat([df, calc], axis=1)
        
        # Determina Risultato 1X2 (se ci sono i gol)
        df['res_1x2'] = '-' 
        if 'scor1' in df.columns and 'scor2' in df.columns:
            mask = df['scor1'].notna() & df['scor2'].notna()
            df.loc[mask & (df['scor1'] > df['scor2']), 'res_1x2'] = '1'
            df.loc[mask & (df['scor1'] == df['scor2']), 'res_1x2'] = 'X'
            df.loc[mask & (df['scor1'] < df['scor2']), 'res_1x2'] = '2'
            
        return df, None
    except Exception as e:
        return None, f"Errore nel caricamento: {str(e)}"

# --- INTERFACCIA UTENTE ---
st.sidebar.header("⚙️ Impostazioni")
BASE_HFA = st.sidebar.number_input("HFA Base", value=100, step=10)
USE_DYN = st.sidebar.checkbox("Usa HFA Dinamico (Classifica)", value=True)

uploaded_file = st.sidebar.file_uploader("📂 Carica CSV", type=["csv"])

if uploaded_file:
    df, error_msg = load_data(uploaded_file, BASE_HFA, USE_DYN)
    
    if error_msg:
        st.error(error_msg)
    else:
        st.success(f"Dati caricati correttamente! ({len(df)} partite)")
        
        # Verifica presenza colonne Ranking
        has_rank = 'rank_h_home' in df.columns and 'rank_a_away' in df.columns
        if USE_DYN and not has_rank:
            st.warning("⚠️ Colonne 'Place1a' e 'Place2d' non trovate. Il calcolo userà solo HFA Base.")
        elif USE_DYN:
            st.info("✅ Ranking squadre rilevato correttamente. HFA Dinamico attivo.")

        # CREAZIONE TABS
        tab1, tab2 = st.tabs(["📊 Analisi Profitto", "🔍 Dati & Ranking"])
        
        # --- TAB 1: ANALISI ---
        with tab1:
            st.subheader("Confronto Redditività")
            df_played = df[df['res_1x2'] != '-'].copy()
            
            if not df_played.empty:
                # Calcolo PnL
                pnl_1 = np.where(df_played['EV_1']>0, np.where(df_played['res_1x2']=='1', df_played['cotaa']-1, -1), 0).sum()
                pnl_2 = np.where(df_played['EV_2']>0, np.where(df_played['res_1x2']=='2', df_played['cotad']-1, -1), 0).sum()
                
                k1, k2 = st.columns(2)
                k1.metric("Profitto Casa (1)", f"{pnl_1:.2f} u")
                k2.metric("Profitto Ospite (2)", f"{pnl_2:.2f} u")
                
                # Tabella Risultati
                desired_cols = ['datamecic', 'txtechipa1', 'txtechipa2', 'HFA_Used', 'cotaa', 'cotad', 'EV_1', 'EV_2', 'res_1x2']
                # Filtra solo colonne esistenti
                final_cols = [c for c in desired_cols if c in df_played.columns]
                
                # Rimuovi duplicati dalla lista visualizzazione
                final_cols = list(dict.fromkeys(final_cols))
                st.dataframe(df_played[final_cols])
            else:
                st.info("Nessuna partita con risultato finale trovato nel file.")

        # --- TAB 2: DATI COMPLETI ---
        with tab2:
            st.subheader("Dataset Completo")
            
            # Mostra dataframe pulito da duplicati
            st.dataframe(df.loc[:, ~df.columns.duplicated()])
            
            st.markdown("---")
            st.subheader("Verifica Colonne Ranking")
            
            # Debug: mostra se sta leggendo i rank
            cols_rank = ['txtechipa1', 'rank_h_home', 'txtechipa2', 'rank_a_away', 'HFA_Used']
            safe_cols = [c for c in cols_rank if c in df.columns]
            
            if len(safe_cols) > 0:
                st.dataframe(df[safe_cols].head(20))
            else:
                st.warning("Le colonne del ranking non sono visibili.")

else:
    st.info("👈 Carica il file CSV per iniziare.")
