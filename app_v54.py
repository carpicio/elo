import streamlit as st
import pandas as pd
import numpy as np

# --- CONFIGURAZIONE ---
st.set_page_config(
    page_title="Value Bet V60", 
    page_icon="🛡️", 
    layout="wide"
)
st.title("🛡️ Calcolatore Strategico (V60)")
st.markdown("---")

# --- FUNZIONI ---
def get_probs(elo_h, elo_a, hfa):
    try:
        diff = elo_a - (elo_h + hfa)
        exp = diff / 400
        p_h = 1 / (1 + 10**exp)
        return p_h, 1 - p_h
    except:
        return 0, 0

def no_margin(o1, ox, o2):
    try:
        if o1<=0 or ox<=0 or o2<=0: 
            return 0,0,0
        i1 = 1/o1
        ix = 1/ox
        i2 = 1/o2
        s = i1 + ix + i2
        return i1/s, ix/s, i2/s
    except:
        return 0,0,0

def calc_row(row, base_hfa, dyn):
    # Setup output
    res = {
        'EV_1': -1, 
        'EV_X': -1, 
        'EV_2': -1, 
        'HFA': base_hfa
    }
    
    try:
        # Helper per numeri
        def to_f(v):
            s = str(v).replace(',', '.')
            try: 
                return float(s)
            except: 
                return 0.0

        # Lettura dati
        elo_h = to_f(row.get('elohomeo', 1500))
        elo_a = to_f(row.get('eloawayo', 1500))
        o1 = to_f(row.get('cotaa', 0))
        ox = to_f(row.get('cotae', 0))
        o2 = to_f(row.get('cotad', 0))
        
        # HFA Dinamico
        curr_hfa = base_hfa
        if dyn:
            # Chiavi esatte dal tuo file
            r1 = row.get('Place 1a') 
            r2 = row.get('Place 2d')
            
            # Se non trova 'Place 1a', prova minuscolo
            if pd.isna(r1): r1 = row.get('place 1a')
            if pd.isna(r2): r2 = row.get('place 2d')

            if pd.notna(r1) and pd.notna(r2):
                try:
                    d = float(r2) - float(r1)
                    adj = d * 3
                    curr_hfa += adj
                    # Limiti 0-200
                    if curr_hfa < 0: curr_hfa = 0
                    if curr_hfa > 200: curr_hfa = 200
                except:
                    pass
        
        res['HFA'] = curr_hfa
        
        # Matematica
        f1, fx, f2 = no_margin(o1, ox, o2)
        ph, pa = get_probs(elo_h, elo_a, curr_hfa)
        
        rem = 1 - fx
        fin1 = rem * ph
        fin2 = rem * pa
        
        res['EV_1'] = (o1 * fin1) - 1
        res['EV_X'] = (ox * fx) - 1
        res['EV_2'] = (o2 * fin2) - 1
        
    except:
        pass
        
    return pd.Series(res)

# --- CARICAMENTO ---
@st.cache_data(ttl=0)
def load_file(file, hfa, dyn):
    try:
        # Lettura diretta con punto e virgola
        df = pd.read_csv(
            file, 
            sep=';', 
            encoding='latin1',
            on_bad_lines='skip',
            engine='python'
        )
        
        # Pulizia nomi colonne (toglie spazi extra ai lati)
        df.columns = df.columns.str.strip()
        
        # Verifica Colonne Chiave
        req = ['cotaa', 'cotad', 'elohomeo']
        # Controlliamo se ci sono (case insensitive)
        cols_lower = [c.lower() for c in df.columns]
        missing = [c for c in req if c not in cols_lower]
        
        if len(missing) > 0:
            return None, f"Mancano: {missing}"

        # Standardizza nomi essenziali
        # Mappa manuale per sicurezza
        ren = {
            '1': 'cotaa', 
            '2': 'cotad',
            'x': 'cotae',
            'X': 'cotae',
            'eloc': 'elohomeo',
            'eloo': 'eloawayo'
        }
        # Applica rinomina se serve
        new_cols = {}
        for c in df.columns:
            cl = c.lower()
            if cl in ren:
                new_cols[c] = ren[cl]
        df = df.rename(columns=new_cols)

        # Filtro righe vuote
        df = df.dropna(subset=['cotaa'])
        
        # Calcolo Applicato
        if not df.empty:
            calc = df.apply(
                lambda r: calc_row(r, hfa, dyn), 
                axis=1
            )
            df = pd.concat([df, calc], axis=1)
            
            # Risultato 1X2
            df['res'] = '-'
            # Gestione sicura colonne score
            if 'scor1' in df.columns:
                s1 = pd.to_numeric(df['scor1'], errors='coerce')
                s2 = pd.to_numeric(df['scor2'], errors='coerce')
                
                mask = s1.notna() & s2.notna()
                df.loc[mask & (s1 > s2), 'res'] = '1'
                df.loc[mask & (s1 == s2), 'res'] = 'X'
                df.loc[mask & (s1 < s2), 'res'] = '2'

        return df, None
        
    except Exception as e:
        return None, str(e)

# --- SIDEBAR ---
st.sidebar.header("⚙️ Opzioni")
base_hfa = st.sidebar.number_input(
    "HFA Base", 
    value=100, 
    step=10
)
use_dyn = st.sidebar.checkbox(
    "HFA Dinamico", 
    value=True
)

up_file = st.sidebar.file_uploader(
    "Carica CSV", 
    type=["csv"]
)

# --- MAIN ---
if up_file:
    df, err = load_file(up_file, base_hfa, use_dyn)
    
    if err:
        st.error(f"Errore: {err}")
    elif df is not None:
        n = len(df)
        st.success(f"Caricate {n} partite.")
        
        # Tabs
        t1, t2 = st.tabs(["Profitto", "Dati"])
        
        with t1:
            played = df[df['res'] != '-'].copy()
            if not played.empty:
                # Helper PNL
                def get_pnl(row, pick):
                    # Quote numeriche
                    o1 = float(str(row['cotaa']).replace(',','.'))
                    o2 = float(str(row['cotad']).replace(',','.'))
                    
                    if pick == '1':
                        if row['res'] == '1': return o1 - 1
                        return -1
                    if pick == '2':
                        if row['res'] == '2': return o2 - 1
                        return -1
                    return 0

                # Calcolo PNL vettorizzato
                # EV > 0
                idx1 = played['EV_1'] > 0
                idx2 = played['EV_2'] > 0
                
                p1 = played[idx1].apply(
                    lambda x: get_pnl(x, '1'), axis=1
                ).sum()
                
                p2 = played[idx2].apply(
                    lambda x: get_pnl(x, '2'), axis=1
                ).sum()
                
                c1, c2 = st.columns(2)
                c1.metric("Profitto Casa", f"{p1:.2f}u")
                c2.metric("Profitto Ospite", f"{p2:.2f}u")
                
                # Colonne da mostrare
                cols = [
                    'datamecic',
                    'txtechipa1',
                    'txtechipa2',
                    'HFA',
                    'cotaa',
                    'cotad',
                    'EV_1',
                    'EV_2',
                    'res'
                ]
                # Filtro esistenti
                final = [c for c in cols if c in played.columns]
                st.dataframe(played[final])
            else:
                st.info("Nessun risultato finale.")
                
        with t2:
            st.dataframe(df)
else:
    st.info("Carica il file CSV.")
