import heapq
import os
import sys
import pandas as pd
from owlready2 import get_ontology
from bayesian_traffic import get_traffic_model 
from sklearn.tree import DecisionTreeRegressor 

class GreenPathAgent:
    def __init__(self, onto_path="../ontologie/green_path.owl", data_path="../data/past_trips.csv"):
        if not os.path.exists(onto_path):
            print(f"Errore: Ontologia non trovata in {onto_path}. Lancia prima setup_kb_green.py")
            sys.exit()
        
        self.onto = get_ontology(onto_path).load()
        self.bayes_engine = get_traffic_model()
        
        print("\n🧠 Inizializzazione Modulo di Apprendimento...")
        self.ml_model = self._train_learning_module(data_path)
        
        self.mappa = {
            'Poggiofranco': [('Campus', 3, False), ('Policlinico', 2, False)],
            'Policlinico': [('Poggiofranco', 2, False), ('Stazione', 2, False)],
            'Campus': [('Poggiofranco', 3, False), ('Stazione', 3, False), ('Pane_e_Pomodoro', 4, False)],
            'Stazione': [('Policlinico', 2, False), ('Campus', 3, False), ('Via_Sparano', 1, True), ('Bari_Vecchia', 2, True)],
            'Via_Sparano': [('Stazione', 1, True), ('Bari_Vecchia', 1, True), ('Pane_e_Pomodoro', 3, False)],
            'Bari_Vecchia': [('Stazione', 2, True), ('Via_Sparano', 1, True), ('Porto', 1, True)],
            'Porto': [('Bari_Vecchia', 1, True), ('Fiera_del_Levante', 4, False)],
            'Pane_e_Pomodoro': [('Campus', 4, False), ('Via_Sparano', 3, False), ('Punta_Perotti', 1, False)],
            'Punta_Perotti': [('Pane_e_Pomodoro', 1, False), ('Torre_a_Mare', 8, False)],
            'Fiera_del_Levante': [('Porto', 4, False)],
            'Torre_a_Mare': [('Punta_Perotti', 8, False)]
        }

    def _train_learning_module(self, data_path):
        if not os.path.exists(data_path):
            print(f"Avviso: Dataset {data_path} non trovato.")
            return None
        
        df = pd.read_csv(data_path)
        x_train = df[['base_emission', 'traffic_prob', 'meteo_idx', 'ac_on']]
        y_train = df['real_co2_per_km']
        
        model = DecisionTreeRegressor(random_state=42)
        model.fit(x_train, y_train)
        print("✅ L'agente ha imparato dai dati storici.")
        return model

    def get_yes_no(self, prompt):
        while True:
            scelta = input(prompt).lower().strip()
            if scelta in ['s', 'n']:
                return scelta == 's'
            print("❌ Errore: Inserisci solo 's' (sì) o 'n' (no).")

    def get_valid_input(self, prompt, min_val, max_val):
        while True:
            try:
                val = int(input(prompt))
                if min_val <= val <= max_val: 
                    return val
                print(f"❌ Inserisci un numero tra {min_val} e {max_val}.")
            except ValueError:
                print("❌ Input non valido. Inserisci un numero intero.")

    def prevedi_emissione_ml(self, veicolo, prob_traffico, meteo_idx, ac_on):
        x_input = pd.DataFrame(
            [[veicolo.emissioneBase, prob_traffico, meteo_idx, int(ac_on)]],
            columns=['base_emission', 'traffic_prob', 'meteo_idx', 'ac_on']
        )
        return self.ml_model.predict(x_input)[0]

    def a_star_search(self, start, goal, veicolo, prob_traffico, meteo_idx, ac_on):
        is_eco = isinstance(veicolo.haCarburante, (self.onto.Elettrico, self.onto.Ibrido))
        frontier = [(0, start, [], 0, 0)] 
        visited = {}

        while frontier:
            (f_score, attuale, path, dist_acc, co2_acc) = heapq.heappop(frontier)
            if attuale == goal: 
                return path + [attuale], co2_acc, dist_acc

            for (vicino, dist, is_ztl) in self.mappa.get(attuale, []):
                if is_ztl and not is_eco: 
                    continue
                co2_per_km = self.prevedi_emissione_ml(veicolo, prob_traffico, meteo_idx, ac_on)
                co2_tratto = co2_per_km * dist
                g_score = f_score + co2_tratto + (dist * 0.01)
                nuova_co2 = co2_acc + co2_tratto
                if vicino not in visited or g_score < visited[vicino]:
                    visited[vicino] = g_score
                    heapq.heappush(frontier, (g_score, vicino, path + [attuale], dist_acc + dist, nuova_co2))
        return None, float('inf'), 0

    def stima_tempo(self, distanza, prob_traffico, meteo_idx):
        v_base = 40.0 
        meteo_penalty = {0: 1.0, 1: 0.7, 2: 0.6, 3: 0.4}
        velocita_reale = v_base * (1 - (prob_traffico * 0.5)) * meteo_penalty.get(meteo_idx, 1.0)
        return (distanza / velocita_reale) * 60

    def run(self):
        print("\n" + "="*50)
        print("     GREEN-PATHFINDER BARI: MODELLO IBRIDO")
        print("="*50)

        while True:
            print("\n🚗 GARAGE:")
            veicoli_nomi = ["Tesla_Model3", "Toyota_Yaris_Hybrid", "Fiat_500_Diesel"]
            for i, v in enumerate(veicoli_nomi, 1): 
                print(f"{i}. {v.replace('_', ' ')}")
            
            scelta_v = self.get_valid_input("Auto (1-3): ", 1, 3)
            veicolo = self.onto.search_one(iri=f"*{veicoli_nomi[scelta_v-1]}")

            ac = self.get_yes_no("\nAccedenderai l'aria condizionata? (s/n): ")
            veicolo.ariaCondizionataAccesa = ac

            print("\n🌦️ METEO: 0:Sole, 1:Pioggia, 2:Nebbia, 3:Neve")
            print("Scegli la condizione meteo attuale inserendo il numero corrispondente:")
            m_idx = self.get_valid_input("Meteo: ", 0, 3)

            print("\n⏰ FASCIA ORARIA:")
            print("Scegli la fascia oraria inserendo il numero corrispondente:")
            orari = ["00:00-05:00", "05:00-07:00", "07:00-09:00", "09:00-14:00", 
                      "14:00-17:00", "17:00-22:00", "22:00-24:00"]
            for i, h in enumerate(orari): 
                print(f"{i} --> {h}")
            o_idx = self.get_valid_input("Ora: ", 0, 6)

            res = self.bayes_engine.query(variables=['Traffico'], evidence={'Meteo': m_idx, 'Ora': o_idx})
            prob_alto = res.values[1]
            
            print("\n🗺️ PUNTI DI BARI DISPONIBILI:")
            print("Scegli il punto di partenza e di arrivo inserendo il numero corrispondente:")
            nodi = list(self.mappa.keys())
            for i, n in enumerate(nodi, 1):
                print(f"{i}. {n}")
            s_idx = self.get_valid_input("Partenza: ", 1, len(nodi))
            g_idx = self.get_valid_input("Arrivo: ", 1, len(nodi))
            
            # Controllo per spostamenti nulli
            if s_idx == g_idx:
                print("\n A questo punto potresti anche camminare... Sei già a destinazione! 🚶‍♂️🌿")
            else:
                #   Esegue la ricerca solo se i punti sono diversi
                path, co2, dist_tot = self.a_star_search(nodi[s_idx-1], nodi[g_idx-1], veicolo, prob_alto, m_idx, ac)
                
                if path:
                    print("\n✅ RISULTATO:")
                    print(f"📍 Cammino: {' ➔ '.join(path)}\n📊 CO2: {co2:.2f} g\n⏱️ Tempo: {self.stima_tempo(dist_tot, prob_alto, m_idx):.0f} min")
                else:
                    print("\n❌ Nessun percorso percorribile a causa della ZTL. Prova ad usare un veicolo ibrido o elettrico.")
            if not self.get_yes_no("\nVuoi fare un'altra previsione? (s/n): "): 
                print("\nGrazie per aver usato Green-Pathfinder. Guida con prudenza! 🚗💨🌿")
                break

if __name__ == "__main__":
    agent = GreenPathAgent()
    agent.run()