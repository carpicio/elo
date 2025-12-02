import streamlit as st
import pandas as pd
import numpy as np

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="Value Bet V59", page_icon="🛡️", layout="wide")
st.title("🛡️ Calcolatore Strategico (V59 - Precision Fix)")
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
        def get_float(val):
            try:
                return float(str(val).replace(',', '.'))
            except:
                return 0.0

        elo_h = get_float(row.get('elohomeo', 1500))
        elo_a = get_float(row.get('eloawayo', 1500))
        o1 = get_float(row.get('cotaa', 0))
        ox = get_float(row.get('cota
