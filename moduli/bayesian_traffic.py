from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

def get_traffic_model():
    # Struttura: Meteo e Ora influenzano il Traffico
    model = DiscreteBayesianNetwork([('Meteo', 'Traffico'), ('Ora', 'Traffico')])

    # Meteo: 0:Sole, 1:Pioggia, 2:Nebbia, 3:Neve
    cpd_meteo = TabularCPD(variable='Meteo', variable_card=4, 
                           values=[[0.7], [0.15], [0.1], [0.05]])

    # Ora aggiornata a 7 stati (variable_card=7)
    cpd_ora = TabularCPD(variable='Ora', variable_card=7, 
                         values=[[0.2], [0.1], [0.15], [0.15], [0.1], [0.2], [0.1]])

    # Nuova Tabella Probabilità Traffico (P(Traffico=Alto | Meteo, Ora))
    # Matrice 2x28 (4 Meteo * 7 Ore = 28 combinazioni)
    # Mapping Ora: 0:Notte(Assente), 1:Alba(Medio), 2:Mattina(Medio), 3:Giorno(Intenso), 
    # 4:Pomeriggio(Medio), 5:Sera(Intenso), 6:Pre-notte(Medio)
    cpd_traffico = TabularCPD(
        variable='Traffico', variable_card=2,
        values=[
            # Traffico BASSO (per ogni combinazione Meteo/Ora)
            [0.95, 0.6, 0.55, 0.2, 0.6, 0.2, 0.6,  # Sole
             0.8,  0.4, 0.35, 0.1, 0.4, 0.1, 0.4,  # Pioggia
             0.7,  0.3, 0.25, 0.05, 0.3, 0.05, 0.3, # Nebbia
             0.6,  0.2, 0.15, 0.01, 0.2, 0.01, 0.2],# Neve
            # Traffico ALTO
            [0.05, 0.4, 0.45, 0.8, 0.4, 0.8, 0.4,  # Sole
             0.2,  0.6, 0.65, 0.9, 0.6, 0.9, 0.6,  # Pioggia
             0.3,  0.7, 0.75, 0.95, 0.7, 0.95, 0.7, # Nebbia
             0.4,  0.8, 0.85, 0.99, 0.8, 0.99, 0.8] # Neve
        ],
        evidence=['Meteo', 'Ora'],
        evidence_card=[4, 7]
    )

    model.add_cpds(cpd_meteo, cpd_ora, cpd_traffico)
    return VariableElimination(model)