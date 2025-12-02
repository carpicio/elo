import streamlit as st
import pandas as pd
import numpy as np

# --- CONFIGURAZIONE ---
st.set_page_config(
    page_title="Value Bet V62 - Cluster Analyzer", 
    page_icon="🔬", 
    layout="wide"
)
st.title("🛡️ Calcolatore Strategico (V62 - League Cluster)")
st.markdown("---")

# --- CORE LOGIC ---
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
        if o1<=0 or ox<=0 or o2<=0: return 0,0,0
        i1 = 1/o1; ix = 1/ox; i2 = 1/o2
        s = i1 + ix + i2
        return i1/s, ix/s, i2/s
    except:
        return 0,0,0

def calc_row(row, base_hfa, dyn):
    res = {'EV_1': -1, 'EV_X': -1, 'EV_2': -1, 'Prob_1': 0, 'Prob_2': 0, 'HFA': base_hfa}
    try:
        def to_f(v):
            try: return float(str(v).replace(',', '.'))
            except: return 0.0

        elo_h = to_f(row.get('elohomeo', 1500))
        elo_a = to_f(row.get('eloawayo', 1500))
        o1 = to_f(row.get('cotaa', 0))
        ox = to_f(row.get('cotae', 0))
        o2 = to_f(row.get('cotad', 0))
        
        curr_hfa = base_hfa
        if dyn:
            r1 = row.get('Place 1a'); r2 = row.get('Place 2d')
            if pd.isna(r1): r1 = row.get('place 1a')
            if pd.isna(r2): r2 = row.get('place 2d')

            if pd.notna(r1) and pd.notna(r2):
                try:
                    d = float(r2) - float(r1)
                    adj = d * 3
                    curr_hfa += adj
                    curr_hfa = max(0, min(curr_hfa, 200))
                except: pass
        
        res['HFA'] = curr_hfa
        f1, fx, f2 = no_margin(o1, ox, o2)
        ph, pa = get_probs(elo_h, elo_a, curr_hfa)
        rem = 1 - fx
        fin1 = rem * ph
        fin2 = rem * pa
        
        res['Prob_1'] = fin1
        res['Prob_2'] = fin2
        res['EV_1'] = (o1 * fin1) - 1
        res['EV_X'] = (ox * fx) - 1
        res['EV_2'] = (o2 * fin2) - 1
    except: pass
    return pd.Series(res)

@st.cache_data(ttl=0)
def load_file(file, hfa, dyn):
    try:
        df = pd.read_csv(file, sep=';', encoding='latin1', on_bad_lines='skip', engine='python')
        df.columns = df.columns.str.strip()
        ren = {'1': 'cotaa', '2': 'cotad', 'x': 'cotae', 'X': 'cotae', 'eloc': 'elohomeo', 'eloo': 'eloawayo'}
        new_cols = {}
        for c in df.columns:
            if c.lower() in ren: new_cols[c] = ren[c.lower()]
        df = df.rename(columns=new_cols)
        df = df.dropna(subset=['cotaa'])

        if not df.empty:
            calc = df.apply(lambda r: calc_row(r, hfa, dyn), axis=1)
            df = pd.concat([df, calc], axis=1)
            
            df['res'] = '-'
            if 'scor1' in df.columns:
                s1 = pd.to_numeric(df['scor1'], errors='coerce')
                s2 = pd.to_numeric(df['scor2'], errors='coerce')
                mask = s1.notna() & s2.notna()
                df.loc[mask & (s1 > s2), 'res'] = '1'
                df.loc[mask & (s1 == s2), 'res'] = 'X'
                df.loc[mask & (s1 < s2), 'res'] = '2'
                
            if 'datamecic' in df.columns:
                df['Date'] = pd.to_datetime(df['datamecic'], dayfirst=True, errors='coerce')
        return df, None
    except Exception as e:
        return None, str(e)

# --- SIMULATOR & ANALYTICS ---
def run_cluster_analysis(df, min_ev, fixed_stake):
    # Filtra solo le bet valide
    bets = []
    
    for idx, row in df.iterrows():
        if row['res'] == '-': continue
        
        try:
            o1 = float(str(row['cotaa']).replace(',','.'))
            o2 = float(str(row['cotad']).replace(',','.'))
        except: continue

        pick = None; odds = 0; pnl = 0
        
        if row['EV_1'] * 100 >= min_ev:
            pick = '1'; odds = o1
            pnl = (odds - 1) * fixed_stake if row['res'] == '1' else -fixed_stake
        elif row['EV_2'] * 100 >= min_ev:
            pick = '2'; odds = o2
            pnl = (odds - 1) * fixed_stake if row['res'] == '2' else -fixed_stake
            
        if pick:
            bets.append({
                'Country': row.get('country', 'Unknown'),
                'League': row.get('league', 'Unknown'),
                'Pick': pick,
                'Odds': odds,
                'Result': 'Win' if pnl > 0 else 'Loss',
                'Profit': pnl,
                'Stake': fixed_stake
            })
            
    return pd.DataFrame(bets)

# --- UI ---
st.sidebar.header("⚙️ Parametri Modello")
base_hfa = st.sidebar.number_input("HFA Base", value=100, step=10, help="Abbassa a 60-80 se pensi che l'HFA sia sopravvalutato")
use_dyn = st.sidebar.checkbox("HFA Dinamico", value=True)

st.sidebar.header("🎯 Filtri Scommessa")
MIN_EV = st.sidebar.slider("Minimo EV (%)", 0.0, 15.0, 2.0)
MIN_ODDS, MAX_ODDS = st.sidebar.slider("Range Quote Accettate", 1.20, 10.0, (1.50, 4.00))

up_file = st.sidebar.file_uploader("Carica CSV", type=["csv"])

if up_file:
    df, err = load_file(up_file, base_hfa, use_dyn)
    
    if df is not None:
        # Analisi Cluster
        st.subheader("📊 Analisi Redditività per Campionato")
        
        bets_df = run_cluster_analysis(df, MIN_EV, 10) # 10€ flat stake
        
        if not bets_df.empty:
            # Filtro Quote
            bets_df = bets_df[(bets_df['Odds'] >= MIN_ODDS) & (bets_df['Odds'] <= MAX_ODDS)]
            
            # Raggruppamento
            summary = bets_df.groupby(['Country', 'League']).agg(
                Bets=('Profit', 'count'),
                Profit=('Profit', 'sum'),
                ROI=('Profit', lambda x: (x.sum() / (len(x)*10)) * 100)
            ).reset_index()
            
            summary = summary.sort_values('Profit', ascending=False)
            
            # Formattazione
            st.dataframe(
                summary.style.format({'Profit': '{:.2f}€', 'ROI': '{:.2f}%'}),
                use_container_width=True
            )
            
            st.metric("Profitto Totale Filtrato", f"{bets_df['Profit'].sum():.2f}€")
            
            # Grafico
            st.bar_chart(data=summary, x='Country', y='Profit')
            
            st.write("### 📝 Dettaglio Scommesse Filtrate")
            st.dataframe(bets_df)
            
        else:
            st.warning("Nessuna scommessa trovata con questi filtri.")
