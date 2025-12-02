import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# --- CONFIGURAZIONE ---
st.set_page_config(
    page_title="Value Bet V61 - Simulator", 
    page_icon="🧪", 
    layout="wide"
)
st.title("🛡️ Calcolatore Strategico & Simulatore (V61)")
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
    res = {'EV_1': -1, 'EV_X': -1, 'EV_2': -1, 'HFA': base_hfa, 'Prob_1': 0, 'Prob_2': 0}
    try:
        def to_f(v):
            s = str(v).replace(',', '.')
            try: return float(s)
            except: return 0.0

        elo_h = to_f(row.get('elohomeo', 1500))
        elo_a = to_f(row.get('eloawayo', 1500))
        o1 = to_f(row.get('cotaa', 0))
        ox = to_f(row.get('cotae', 0))
        o2 = to_f(row.get('cotad', 0))
        
        curr_hfa = base_hfa
        if dyn:
            r1 = row.get('Place 1a') 
            r2 = row.get('Place 2d')
            if pd.isna(r1): r1 = row.get('place 1a')
            if pd.isna(r2): r2 = row.get('place 2d')

            if pd.notna(r1) and pd.notna(r2):
                try:
                    d = float(r2) - float(r1)
                    adj = d * 3
                    curr_hfa += adj
                    if curr_hfa < 0: curr_hfa = 0
                    if curr_hfa > 200: curr_hfa = 200
                except:
                    pass
        
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
        
    except:
        pass
    return pd.Series(res)

@st.cache_data(ttl=0)
def load_file(file, hfa, dyn):
    try:
        df = pd.read_csv(file, sep=';', encoding='latin1', on_bad_lines='skip', engine='python')
        df.columns = df.columns.str.strip()
        
        # Rename map
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
                
            # Date Conversion for Chart
            if 'datamecic' in df.columns:
                df['Date'] = pd.to_datetime(df['datamecic'], dayfirst=True, errors='coerce')

        return df, None
    except Exception as e:
        return None, str(e)

# --- SIMULATION ENGINE ---
def run_simulation(df, bankroll, strategy, fixed_stake, kelly_fraction, min_ev):
    history = []
    current_bankroll = bankroll
    bets_placed = 0
    wins = 0
    
    # Ordina per data se possibile
    if 'Date' in df.columns and df['Date'].notna().any():
        df = df.sort_values('Date')
    
    for idx, row in df.iterrows():
        if row['res'] == '-': continue
        
        # Quote numeriche
        try:
            o1 = float(str(row['cotaa']).replace(',','.'))
            o2 = float(str(row['cotad']).replace(',','.'))
        except: continue

        bet_pick = None
        bet_odds = 0
        bet_prob = 0
        bet_ev = 0
        
        # Cerca valore
        if row['EV_1'] * 100 >= min_ev:
            bet_pick = '1'
            bet_odds = o1
            bet_prob = row['Prob_1']
            bet_ev = row['EV_1']
        elif row['EV_2'] * 100 >= min_ev:
            bet_pick = '2'
            bet_odds = o2
            bet_prob = row['Prob_2']
            bet_ev = row['EV_2']
            
        if bet_pick:
            stake = 0
            if strategy == "Fissa":
                stake = fixed_stake
            elif strategy == "Kelly":
                # Formula Kelly: (bp - q) / b
                # b = odds - 1
                # p = probability
                # q = 1 - p
                b = bet_odds - 1
                p = bet_prob
                q = 1 - p
                kelly_perc = ((b * p) - q) / b
                stake = current_bankroll * (kelly_perc * kelly_fraction)
            
            stake = max(0, min(stake, current_bankroll)) # Mai scommettere più di quello che hai
            
            if stake > 0:
                bets_placed += 1
                outcome = 0
                if bet_pick == row['res']:
                    outcome = stake * (bet_odds - 1)
                    wins += 1
                else:
                    outcome = -stake
                
                current_bankroll += outcome
                
                history.append({
                    'Match': f"{row['txtechipa1']} - {row['txtechipa2']}",
                    'Pick': bet_pick,
                    'Odds': bet_odds,
                    'Stake': round(stake, 2),
                    'Result': 'Win' if outcome > 0 else 'Loss',
                    'Profit': outcome,
                    'Bankroll': current_bankroll
                })
                
    return pd.DataFrame(history), bets_placed, wins

# --- INTERFACCIA ---
st.sidebar.header("⚙️ Impostazioni Modello")
base_hfa = st.sidebar.number_input("HFA Base", value=100, step=10)
use_dyn = st.sidebar.checkbox("HFA Dinamico", value=True)

st.sidebar.header("💰 Impostazioni Simulazione")
START_BANK = st.sidebar.number_input("Budget Iniziale", value=1000)
STRATEGY = st.sidebar.selectbox("Metodo", ["Fissa", "Kelly"])
FIXED_AMT = 10
KELLY_FRAC = 0.1
if STRATEGY == "Fissa":
    FIXED_AMT = st.sidebar.number_input("Puntata Fissa (€)", value=10)
else:
    KELLY_FRAC = st.sidebar.slider("Frazione di Kelly (0.1 = Prudente)", 0.01, 1.0, 0.1)

MIN_EV = st.sidebar.slider("Minimo Valore (EV%)", 0.0, 20.0, 2.0, help="Scommetti solo se il valore è superiore a X%")

up_file = st.sidebar.file_uploader("Carica CSV", type=["csv"])

if up_file:
    df, err = load_file(up_file, base_hfa, use_dyn)
    
    if df is not None:
        st.success(f"Caricate {len(df)} partite.")
        
        tab1, tab2, tab3 = st.tabs(["📊 Analisi", "🧪 Simulazione", "📝 Dati Grezzi"])
        
        with tab1:
            st.info("Vai nel tab 'Simulazione' per vedere l'andamento del portafoglio.")
            cols = ['datamecic','txtechipa1', 'txtechipa2','HFA', 'cotaa', 'cotad','EV_1', 'EV_2', 'res']
            final = [c for c in cols if c in df.columns]
            st.dataframe(df[final])

        with tab2:
            st.subheader("Backtest Strategia")
            
            sim_df, n_bets, n_wins = run_simulation(df, START_BANK, STRATEGY, FIXED_AMT, KELLY_FRAC, MIN_EV)
            
            if not sim_df.empty:
                final_bal = sim_df.iloc[-1]['Bankroll']
                profit = final_bal - START_BANK
                roi = (profit / sim_df['Stake'].sum()) * 100
                win_rate = (n_wins / n_bets) * 100
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Bilancio Finale", f"{final_bal:.2f} €", delta=f"{profit:.2f} €")
                c2.metric("ROI (Yield)", f"{roi:.2f}%")
                c3.metric("Win Rate", f"{win_rate:.1f}%")
                c4.metric("Bet Totali", n_bets)
                
                st.line_chart(sim_df['Bankroll'])
                st.dataframe(sim_df)
            else:
                st.warning("Nessuna scommessa trovata con i criteri attuali (EV troppo basso?). Prova ad abbassare la soglia EV.")
        
        with tab3:
            st.dataframe(df)
