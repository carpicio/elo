import streamlit as st
import pandas as pd
import numpy as np

# --- CONFIGURAZIONE ---
st.set_page_config(page_title="Value Bet HFA v48", page_icon="🎛️", layout="wide")
st.title("🎛️ Calcolatore Strategico (HFA Dinamico v48)")

# --- SIDEBAR: CONTROLLO FATTORE CAMPO ---
st.sidebar.header("⚙️ Parametri Modello")
st.sidebar.info("Modifica il Fattore Campo per vedere come cambiano le previsioni e i profitti.")
HFA_VAL = st.sidebar.number_input("🏠 Fattore Campo (HFA)", value=100, step=10, min_value=0, max_value=300, help="Punti ELO aggiunti alla squadra di casa. Standard = 100.")
st.sidebar.markdown("---")

# --- FUNZIONI MATEMATICHE ---
def get_implicit_probs(elo_home, elo_away, hfa):
    try:
        # Qui usiamo il valore HFA passato come argomento
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

def calculate_row(row, hfa):
    res = {'EV_1': -1, 'EV_X': -1, 'EV_2': -1, 'Fair_1': 0, 'Fair_X': 0, 'Fair_2': 0, 'ELO_Diff': 0}
    try:
        elo_h = float(row.get('elohomeo', 1500))
        elo_a = float(row.get('eloawayo', 1500))
        o1 = float(row.get('cotaa', 0))
        ox = float(row.get('cotae', 0))
        o2 = float(row.get('cotad', 0))
    except:
        return pd.Series(res)
    
    if o1 > 0 and ox > 0 and o2 > 0:
        pf_1, pf_x, pf_2 = remove_margin(o1, ox, o2)
        # Passiamo l'HFA dinamico alla funzione ELO
        p_elo_h, p_elo_a = get_implicit_probs(elo_h, elo_a, hfa)
        rem = 1 - pf_x
        p_fin_1 = rem * p_elo_h
        p_fin_2 = rem * p_elo_a
        
        res['Fair_1'] = 1/p_fin_1 if p_fin_1>0 else 0
        res['Fair_X'] = 1/pf_x if pf_x>0 else 0
        res['Fair_2'] = 1/p_fin_2 if p_fin_2>0 else 0
        res['EV_1'] = (o1 * p_fin_1) - 1
        res['EV_X'] = (ox * pf_x) - 1
        res['EV_2'] = (o2 * p_fin_2) - 1
        
    res['ELO_Diff'] = (elo_h + hfa) - elo_a
    return pd.Series(res)

# --- CARICAMENTO DATI (SOLO RAW) ---
@st.cache_data(ttl=0)
def load_raw_data(file):
    try:
        try:
            df = pd.read_csv(file, sep=';', encoding='latin1')
            if len(df.columns) < 5: raise ValueError
        except:
            file.seek(0)
            df = pd.read_csv(file, sep=',', encoding='latin1')

        df.columns = df.columns.str.strip().str.lower().str.replace(r'[^\w\s]', '', regex=True)
        
        rename_map = {
            '1': 'cotaa', 'x': 'cotae', '2': 'cotad',
            'eloc': 'elohomeo', 'eloo': 'eloawayo',
            'gfinc': 'scor1', 'gfino': 'scor2',
            'o25': 'cotao', 'u25': 'cotau',
            'data': 'datamecic', 'casa': 'txtechipa1', 'ospite': 'txtechipa2',
            'gg': 'cotagg', 'ng': 'cotang'
        }
        df = df.rename(columns=rename_map)
        
        cols_num = ['cotaa', 'cotae', 'cotad', 'cotao', 'cotau', 'elohomeo', 'eloawayo', 'scor1', 'scor2', 'cotagg', 'cotang']
        for c in cols_num:
            if c in df.columns:
                df[c] = df[c].astype(str).str.replace(',', '.', regex=False)
                df[c] = pd.to_numeric(df[c], errors='coerce')

        df = df.dropna(subset=['cotaa', 'cotae', 'cotad', 'elohomeo', 'eloawayo'])
        return df, None
    except Exception as e:
        return None, str(e)

# --- ELABORAZIONE DINAMICA ---
def process_data_with_hfa(df, hfa_value):
    # Ricalcola tutto usando l'HFA scelto dall'utente
    calc = df.apply(lambda r: calculate_row(r, hfa_value), axis=1)
    df_out = pd.concat([df, calc], axis=1)
    
    df_out['res_1x2'] = '-' 
    if 'scor1' in df_out.columns and 'scor2' in df_out.columns:
        mask = df_out['scor1'].notna() & df_out['scor2'].notna()
        df_out.loc[mask & (df_out['scor1'] > df_out['scor2']), 'res_1x2'] = '1'
        df_out.loc[mask & (df_out['scor1'] == df_out['scor2']), 'res_1x2'] = 'X'
        df_out.loc[mask & (df_out['scor1'] < df_out['scor2']), 'res_1x2'] = '2'
        
        df_out['goals_tot'] = df_out['scor1'] + df_out['scor2']
        if 'cotao' in df_out.columns:
            df_out['res_o25'] = np.nan
            df_out.loc[mask, 'res_o25'] = (df_out.loc[mask, 'goals_tot'] > 2.5).astype(int)
            
    return df_out

# --- MOTORE CLUSTER ---
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
    if col_ev in df_hist.columns:
        mask &= (df_hist[col_ev] >= f_ev_min)
    
    cluster = df_hist[mask]
    
    if len(cluster) >= 5:
        wins = len(cluster[cluster['res_1x2'] == target])
        profit = (cluster[cluster['res_1x2'] == target][col_odd] - 1).sum() - (len(cluster) - wins)
        roi = (profit / len(cluster)) * 100
        return len(cluster), roi, profit
    return 0, 0, 0

# --- INTERFACCIA ---
file_hist = st.sidebar.file_uploader("📂 1. Carica STORICO", type=["csv"], key="u_hist")
file_fut = st.sidebar.file_uploader("📂 2. Carica FUTURE", type=["csv"], key="u_fut")

df_history, df_future = None, None

# Caricamento e Calcolo Dinamico
if file_hist:
    raw_h, err_h = load_raw_data(file_hist)
    if err_h: st.sidebar.error(err_h)
    else: 
        # Applichiamo HFA qui!
        df_history = process_data_with_hfa(raw_h, HFA_VAL)
        st.sidebar.success(f"🧠 Storico OK ({len(df_history)} match)")

if file_fut:
    raw_f, err_f = load_raw_data(file_fut)
    if err_f: st.sidebar.error(err_f)
    else: 
        # Applichiamo HFA anche qui!
        df_future = process_data_with_hfa(raw_f, HFA_VAL)
        st.sidebar.success(f"🎯 Future OK ({len(df_future)} match)")

tab1, tab2, tab3 = st.tabs(["🔮 Manuale", "🚀 Future", "📊 Storico"])

with tab1:
    st.header(f"Analisi Singola (HFA: {HFA_VAL})")
    
    col_input, col_res = st.columns([1, 1])
    with col_input:
        team_h = st.text_input("Casa", "Home")
        team_a = st.text_input("Ospite", "Away")
        c1, c2 = st.columns(2)
        elo_h = c1.number_input("ELO Casa", value=1500, min_value=0, step=10)
        elo_a = c2.number_input("ELO Ospite", value=1500, min_value=0, step=10)
        c3, c4, c5 = st.columns(3)
        o1 = c3.number_input("Quota 1", value=2.00, min_value=1.01)
        ox = c4.number_input("Quota X", value=3.00, min_value=1.01)
        o2 = c5.number_input("Quota 2", value=3.50, min_value=1.01)
        btn_calc = st.button("Analizza Partita", type="primary")

    with col_res:
        if btn_calc:
            # Calcolo manuale usando HFA_VAL dalla sidebar
            row = {'elohomeo': elo_h, 'eloawayo': elo_a, 'cotaa': o1, 'cotae': ox, 'cotad': o2}
            res = calculate_row(row, HFA_VAL)
            
            st.subheader(f"{team_h} vs {team_a}")
            st.info(f"Differenza ELO (con HFA {HFA_VAL}): **{int(res['ELO_Diff'])}**")
            
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
                    <p style="background-color:rgba(0,0,0,0.2); padding:2px;">Imp: {res[f'Fair_{label}']:.2f}</p>
                    <p>EV: {ev:+.1%}</p>
                    <p style="font-size:12px;">{extra}</p>
                </div>
                """, unsafe_allow_html=True)

            k1, k2, k3 = st.columns(3)
            with k1: show_smart_card("1", o1, res['EV_1'], "1")
            with k2: show_smart_card("X", ox, res['EV_X'], "X")
            with k3: show_smart_card("2", o2, res['EV_2'], "2")

with tab2:
    st.header(f"🚀 Occasioni Future (HFA: {HFA_VAL})")
    if df_future is None:
        st.info("Carica file Future.")
    elif df_history is None:
        st.warning("Carica Storico per analisi cluster.")
    else:
        min_roi_target = st.slider("Min ROI Storico %", 0, 50, 10)
        results = []
        progress = st.progress(0)
        
        for i, row in df_future.iterrows():
            c_a = row.get('cotaa', 0)
            c_d = row.get('cotad', 0)
            
            n1, roi1, p1 = analyze_input(df_history, c_a, row.get('EV_1', 0), "1")
            if n1 >= 5 and roi1 >= min_roi_target and row.get('EV_1', 0) > 0:
                results.append({'Data': row.get('datamecic', '-'), 'Casa': row.get('txtechipa1', '-'), 'Ospite': row.get('txtechipa2', '-'), 
                                'Bet': '1', 'Quota': round(c_a, 2), 'EV %': round(row.get('EV_1', 0)*100, 1), 'ROI Storico %': round(roi1, 1), 'Casi': n1})
            
            n2, roi2, p2 = analyze_input(df_history, c_d, row.get('EV_2', 0), "2")
            if n2 >= 5 and roi2 >= min_roi_target and row.get('EV_2', 0) > 0:
                results.append({'Data': row.get('datamecic', '-'), 'Casa': row.get('txtechipa1', '-'), 'Ospite': row.get('txtechipa2', '-'), 
                                'Bet': '2', 'Quota': round(c_d, 2), 'EV %': round(row.get('EV_2', 0)*100, 1), 'ROI Storico %': round(roi2, 1), 'Casi': n2})
            
            progress.progress((i + 1) / len(df_future))
        progress.empty()
        
        if results:
            df_res = pd.DataFrame(results).sort_values('ROI Storico %', ascending=False)
            st.success(f"Trovate **{len(df_res)}** Occasioni!")
            st.dataframe(df_res)
        else:
            st.warning("Nessuna occasione trovata.")

with tab3:
    st.header(f"📊 Report Storico (HFA: {HFA_VAL})")
    if df_history is not None:
        df_played = df_history[df_history['res_1x2'] != '-'].copy()
        if not df_played.empty:
            pnl_1 = np.where(df_played['EV_1']>0, np.where(df_played['res_1x2']=='1', df_played['cotaa']-1, -1), 0).sum()
            pnl_2 = np.where(df_played['EV_2']>0, np.where(df_played['res_1x2']=='2', df_played['cotad']-1, -1), 0).sum()
            
            k1, k2 = st.columns(2)
            k1.metric("Profitto 1", f"{pnl_1:.2f} u")
            k2.metric("Profitto 2", f"{pnl_2:.2f} u")
            
            st.subheader("Win Rate per Fascia EV")
            df_b1 = df_played[['EV_1', 'res_1x2']].rename(columns={'EV_1':'EV'}); df_b1['Won'] = (df_b1['res_1x2']=='1').astype(int)
            df_b2 = df_played[['EV_2', 'res_1x2']].rename(columns={'EV_2':'EV'}); df_b2['Won'] = (df_b2['res_1x2']=='2').astype(int)
            stats = pd.concat([df_b1, df_b2])
            stats['Fascia'] = pd.cut(stats['EV'], bins=[-0.5, 0, 0.05, 0.10, 0.20, 10], labels=['No', '0-5%', '5-10%', '10-20%', '>20%'])
            res_tab = stats.groupby('Fascia').agg(Bets=('Won','count'), WinRate=('Won','mean'))
            res_tab['WinRate'] = (res_tab['WinRate']*100).round(1).astype(str)+'%'
            st.table(res_tab)
        else:
            st.info("Nessun risultato storico.")
    else:
        st.info("Carica Storico.")
