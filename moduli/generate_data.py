import pandas as pd
import numpy as np

# Seed per la riproducibilità
np.random.seed(42)
n_samples = 1000

# Feature: [EmissioneBase, ProbTraffico, MeteoIdx, AC_On]
# 0.0: Tesla, 92.0: Hybrid, 110.0: Diesel
bases = np.random.choice([0.0, 92.0, 110.0], n_samples)
traffic_probs = np.random.uniform(0, 1, n_samples)
meteo_indices = np.random.randint(0, 4, n_samples)
ac_status = np.random.randint(0, 2, n_samples)

def simulate_realistic_emission(base, traffic, meteo, ac):
    if base == 0.0:
        return 0.0
    
    res = base
    
    # 1. Impatto Traffico (Stop & Go)
    # Il Diesel aumenta fino al +180% in traffico estremo
    # L'Ibrido aumenta fino al +120% (più efficiente in città)
    if base > 100: # Diesel
        res *= (1 + traffic * 1.8)
    else: # Hybrid
        res *= (1 + traffic * 1.2)
        
    # 2. Impatto Aria Condizionata
    # Incidenza del 15% + un fattore variabile basato sul traffico (motore al minimo)
    if ac == 1:
        ac_impact = 1.15 + (traffic * 0.1) 
        res *= ac_impact
        
    # 3. Impatto Meteo (Resistenza al rotolamento e riscaldamento)
    # 0: Sole (1.0), 1: Pioggia (1.08), 2: Nebbia (1.12), 3: Neve (1.25)
    meteo_weights = {0: 1.0, 1: 1.08, 2: 1.12, 3: 1.25}
    res *= meteo_weights[meteo]
    
    # 4. Varianza Reale (Rumore Gaussiano del 4%)
    noise = np.random.normal(0, res * 0.04)
    return max(base * 0.8, res + noise)

emissions = [simulate_realistic_emission(b, t, m, a) for b, t, m, a in zip(bases, traffic_probs, meteo_indices, ac_status)]

df_final = pd.DataFrame({
    'base_emission': bases,
    'traffic_prob': traffic_probs,
    'meteo_idx': meteo_indices,
    'ac_on': ac_status,
    'real_co2_per_km': emissions
})

# Arrotondamento per un look "dati reali"
df_final = df_final.round({'traffic_prob': 4, 'real_co2_per_km': 2})

# Salvataggio
df_final.to_csv('past_trips.csv', index=False)
print("File 'past_trips.csv' con 1000 righe generato con successo.")