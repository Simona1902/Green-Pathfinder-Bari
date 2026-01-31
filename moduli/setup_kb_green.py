from owlready2 import (
    get_ontology, Thing, ObjectProperty, 
    DataProperty, FunctionalProperty, AllDisjoint
)

def setup_urban_ontology():
    onto = get_ontology("http://test.org/green_path.owl")

    with onto:
        class Veicolo(Thing):
            pass

        class Carburante(Thing):
            pass

        class Strada(Thing):
            pass

        class Diesel(Carburante):
            pass

        class Elettrico(Carburante):
            pass

        class Ibrido(Carburante):
            pass

        AllDisjoint([Diesel, Elettrico, Ibrido])

        class haCarburante(ObjectProperty, FunctionalProperty):
            domain = [Veicolo]
            range = [Carburante]

        class ariaCondizionataAccesa(DataProperty, FunctionalProperty):
            domain = [Veicolo]
            range = [bool]

        class emissioneBase(DataProperty, FunctionalProperty):
            domain = [Veicolo]
            range = [float]

        # Istanze con dati CO2 reali
        tesla = Veicolo("Tesla_Model3", emissioneBase=0.0, ariaCondizionataAccesa=False)
        tesla.haCarburante = Elettrico()
        
        yaris = Veicolo("Toyota_Yaris_Hybrid", emissioneBase=82.0, ariaCondizionataAccesa=False)
        yaris.haCarburante = Ibrido()
        
        fiat = Veicolo("Fiat_500_Diesel", emissioneBase=110.0, ariaCondizionataAccesa=False)
        fiat.haCarburante = Diesel()

        class AreaProtetta(Strada):
            pass

        # Via Sparano e Bari Vecchia sono protette
        AreaProtetta("Via_Sparano")
        AreaProtetta("Bari_Vecchia")

    onto.save(file="../ontologie/green_path.owl", format="rdfxml")
    print("Ontologia Green-Pathfinder creata e salvata!")

if __name__ == "__main__":
    setup_urban_ontology()