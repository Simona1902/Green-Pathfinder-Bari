import matplotlib.pyplot as plt
import pandas as pd
from main_green import GreenPathAgent

def run_evaluation():
    agent = GreenPathAgent()
    
    # 6 Scenari precisi per la relazione (Cap. 3.5 e 4.5)
    scenarios = [
        ("Toyota_Yaris_Hybrid", 0, 0, False, "Yaris - S1 (Ottimale)"),
        ("Toyota_Yaris_Hybrid", 0, 2, False, "Yaris - S2 (Pendolare)"),
        ("Toyota_Yaris_Hybrid", 1, 5, True,  "Yaris - S3 (Stress Test)"),
        ("Tesla_Model3", 0, 0, False, "Tesla - Zero Emission"),
        ("Fiat_500_Diesel", 0, 0, False, "Diesel - Baseline"),
        ("Fiat_500_Diesel", 1, 5, True,  "Diesel - Critico")
    ]
    
    results = []
    start, goal = 'Poggiofranco', 'Torre_a_Mare'

    for v_id, meteo, ora, ac, label in scenarios:
        veicolo = agent.onto.search_one(iri=f"*{v_id}")
        res_bayes = agent.bayes_engine.query(variables=['Traffico'], evidence={'Meteo': meteo, 'Ora': ora})
        prob_alto = res_bayes.values[1]
        
        path, co2, dist = agent.a_star_search(start, goal, veicolo, prob_alto, meteo, ac)
        tempo = agent.stima_tempo(dist, prob_alto, meteo) if path else 0
        
        results.append({
            "Scenario": label,
            "P_Traffico": f"{round(prob_alto * 100)}%",
            "CO2_g": round(co2, 2) if co2 != float('inf') else 0,
            "Tempo_min": round(tempo, 1)
        })

    df = pd.DataFrame(results)
    print("\n--- DATI DEFINITIVI PER LA RELAZIONE ---")
    print(df.to_string(index=False))
    
    # Generazione Grafico
    plt.figure(figsize=(12, 6))
    colors = ['#2ecc71', '#27ae60', '#16a085', '#3498db', '#e67e22', '#c0392b']
    bars = plt.bar(df['Scenario'], df['CO2_g'], color=colors)
    plt.ylabel('CO2 Totale (grammi)')
    plt.title('Valutazione Finale: Analisi Comparativa delle Emissioni')
    plt.xticks(rotation=15, ha='right')
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, f"{bar.get_height():.1f}g", ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig('../docs/grafico_valutazione_finale.png')
    df.to_csv("../docs/risultati_finali.csv", index=False)

if __name__ == "__main__":
    run_evaluation()