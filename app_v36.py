import pandas as pd
import numpy as np

# Load the file
try:
    df = pd.read_csv('set_ott_nov.csv', sep=';', encoding='latin1')
except:
    # Fallback
    df = pd.read_csv('set_ott_nov.csv', sep=',', encoding='latin1')

# Clean column names
df.columns = df.columns.str.strip().str.lower()

# Remove duplicate columns (user mentioned ELO was doubled, this fixes it if still present)
df = df.loc[:, ~df.columns.duplicated()]

# Rename map to standardize
rename_map = {
    '1': 'cotaa', 'x': 'cotae', '2': 'cotad',
    'o2,5': 'cotao', 'u2,5': 'cotau',
    'gg': 'cotagg', 'ng': 'cotang',
    'eloc': 'elohomeo', 'eloo': 'eloawayo',
    'gfinc': 'scor1', 'gfino': 'scor2',
    'data': 'datamecic'
}
# Apply rename if columns match keys
df = df.rename(columns=rename_map)

# List of columns to convert to numeric
cols_num = ['cotaa', 'cotae', 'cotad', 'cotao', 'cotau', 'elohomeo', 'eloawayo', 'scor1', 'scor2']
# Add GG/NG if they exist (based on previous file structures, sometimes they are missing or named differently)
if 'cotagg' in df.columns: cols_num.append('cotagg')
if 'cotang' in df.columns: cols_num.append('cotang')

for c in cols_num:
    if c in df.columns:
        df[c] = df[c].astype(str).str.replace(',', '.', regex=False)
        df[c] = pd.to_numeric(df[c], errors='coerce')

# Drop rows with missing essential data for analysis
df_clean = df.dropna(subset=['cotaa', 'cotae', 'cotad', 'elohomeo', 'eloawayo', 'scor1', 'scor2']).copy()

# --- CALCULATE RESULTS ---
# 1X2
conditions = [
    (df_clean['scor1'] > df_clean['scor2']),
    (df_clean['scor1'] == df_clean['scor2']),
    (df_clean['scor1'] < df_clean['scor2'])
]
df_clean['res_1x2'] = np.select(conditions, ['1', 'X', '2'], default='-')

# Over/Under 2.5
df_clean['goals_tot'] = df_clean['scor1'] + df_clean['scor2']
if 'cotao' in df_clean.columns:
    df_clean['res_o25'] = (df_clean['goals_tot'] > 2.5).astype(int)
    df_clean['res_u25'] = (df_clean['goals_tot'] < 2.5).astype(int)

# GG/NG
if 'cotagg' in df_clean.columns:
    df_clean['res_gg'] = ((df_clean['scor1'] > 0) & (df_clean['scor2'] > 0)).astype(int)
    df_clean['res_ng'] = ((df_clean['scor1'] == 0) | (df_clean['scor2'] == 0)).astype(int)

# --- CALCULATE EV (Value) ---
HFA = 100 

def calc_ev_row(row):
    try:
        o1, ox, o2 = row['cotaa'], row['cotae'], row['cotad']
        if o1 <= 0 or ox <= 0 or o2 <= 0: return 0, 0, 0
        
        # Market Probabilities
        inv_sum = (1/o1) + (1/ox) + (1/o2)
        pf_x = (1/ox) / inv_sum
        
        # ELO Probabilities
        elo_h, elo_a = row['elohomeo'], row['eloawayo']
        exp = (elo_a - (elo_h + HFA)) / 400
        p_elo_h = 1 / (1 + 10**exp)
        p_elo_a = 1 - p_elo_h
        
        # Combined
        rem = 1 - pf_x
        p_fin_1 = rem * p_elo_h
        p_fin_2 = rem * p_elo_a
        
        ev_1 = (o1 * p_fin_1) - 1
        ev_x = (ox * pf_x) - 1
        ev_2 = (o2 * p_fin_2) - 1
        
        return ev_1, ev_x, ev_2
    except:
        return 0, 0, 0

ev_data = df_clean.apply(calc_ev_row, axis=1, result_type='expand')
df_clean[['EV_1', 'EV_X', 'EV_2']] = ev_data

# --- CLUSTER ANALYSIS FUNCTION ---
def find_best_clusters(df, market_type, odd_col, metric_col, metric_type='EV'):
    results = []
    
    # Define Bins
    # Odds Bins
    odds_bins = [1.0, 1.5, 1.8, 2.0, 2.2, 2.5, 3.0, 4.0, 5.0, 10.0, 100.0]
    df['bin_odd'] = pd.cut(df[odd_col], bins=odds_bins)
    
    # Metric Bins
    if metric_type == 'EV':
        # EV Bins: <0, 0-5%, 5-10%, 10-20%, >20%
        met_bins = [-1.0, 0.0, 0.05, 0.10, 0.20, 10.0]
        met_labels = ['No Value', '0-5%', '5-10%', '10-20%', '>20%']
        df['bin_met'] = pd.cut(df[metric_col], bins=met_bins, labels=met_labels)
    else:
        # ELO Diff Abs Bins (0-50, 50-100, etc.)
        met_bins = [0, 50, 100, 150, 200, 1000]
        met_labels = ['0-50', '50-100', '100-150', '150-200', '>200']
        df['bin_met'] = pd.cut(df[metric_col], bins=met_bins, labels=met_labels)
        
    grouped = df.groupby(['bin_odd', 'bin_met'])
    
    for name, group in grouped:
        if len(group) < 10: continue # Min sample size
        
        bets = len(group)
        
        # Calculate Wins
        if market_type == '1': wins = len(group[group['res_1x2'] == '1'])
        elif market_type == 'X': wins = len(group[group['res_1x2'] == 'X'])
        elif market_type == '2': wins = len(group[group['res_1x2'] == '2'])
        elif market_type == 'Over 2.5': wins = group['res_o25'].sum()
        elif market_type == 'Under 2.5': wins = group['res_u25'].sum()
        elif market_type == 'GG': wins = group['res_gg'].sum()
        elif market_type == 'NG': wins = group['res_ng'].sum()
        
        profit = (group[odd_col] - 1).sum() - (bets - wins) if wins > 0 else -bets
        roi = (profit / bets) * 100
        
        if roi > 0:
            results.append({
                'Market': market_type,
                'Odds Range': str(name[0]),
                'Filter Range': str(name[1]),
                'Bets': bets,
                'ROI %': round(roi, 2),
                'Profit': round(profit, 2)
            })
            
    return pd.DataFrame(results).sort_values('ROI %', ascending=False)

# --- RUN ANALYSIS ---
# 1X2 Analysis
res_1 = find_best_clusters(df_clean, '1', 'cotaa', 'EV_1', 'EV')
res_x = find_best_clusters(df_clean, 'X', 'cotae', 'EV_X', 'EV')
res_2 = find_best_clusters(df_clean, '2', 'cotad', 'EV_2', 'EV')

# Goals Analysis (Needs ELO Diff)
df_clean['elo_diff_abs'] = abs((df_clean['elohomeo'] + 100) - df_clean['eloawayo'])

res_o25 = pd.DataFrame()
res_u25 = pd.DataFrame()
if 'cotao' in df_clean.columns:
    res_o25 = find_best_clusters(df_clean, 'Over 2.5', 'cotao', 'elo_diff_abs', 'ELO')
    res_u25 = find_best_clusters(df_clean, 'Under 2.5', 'cotau', 'elo_diff_abs', 'ELO')

# GG/NG Analysis
res_gg = pd.DataFrame()
res_ng = pd.DataFrame()
if 'cotagg' in df_clean.columns: # Assuming column might be named 'cotagg' or similar, strictly checking names
    # Actually user file header shows 'gg' and 'ng' columns which might contain odds? No, usually GG/NG are just boolean or text results in some files, but in others odds.
    # Let's check df_clean columns to be sure what we have.
    pass

# Check for GG/NG odds specifically
# In previous snippets GG/NG were results or odds? In "26_26_..." snippet: GG;NG were headers.
# Let's try to assume they are odds if they are floats.
# We normalized column names to lower case. 'gg' -> 'cotagg'? No, rename map didn't map 'gg' to 'cotagg' unless I add it.
# Let's check if 'gg' column exists and has odds-like values (e.g. > 1.0)
gg_col = 'gg' if 'gg' in df_clean.columns else None
ng_col = 'ng' if 'ng' in df_clean.columns else None

if gg_col:
    # Rename for consistency if needed or just use
    res_gg = find_best_clusters(df_clean, 'GG', gg_col, 'elo_diff_abs', 'ELO')
if ng_col:
    res_ng = find_best_clusters(df_clean, 'NG', ng_col, 'elo_diff_abs', 'ELO')

print("--- CLUSTER VINCENTI 1X2 ---")
print(pd.concat([res_1.head(3), res_x.head(3), res_2.head(3)]))

print("\n--- CLUSTER VINCENTI O/U 2.5 ---")
if not res_o25.empty: print(pd.concat([res_o25.head(3), res_u25.head(3)]))
else: print("Dati Over/Under non disponibili o insufficienti.")

print("\n--- CLUSTER VINCENTI GG/NG ---")
if not res_gg.empty: print(pd.concat([res_gg.head(3), res_ng.head(3)]))
else: print("Dati GG/NG non disponibili (o non riconosciuti come quote).")
