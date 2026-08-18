import os
import sys

# Ajouter le root path pour importer les modules proprement
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from complianceguard.graph_agent import compliance_agent

def test_a2a_graph():
    print("Testing A2A Graph (Drafter <-> Reviewer)...")
    
    initial_state = {
        "question": "Quelles sont les conditions pour avoir le label Startup Act en Tunisie ?", # Connu par GraphRAG
        "has_pdf": False,  
        "revision_count": 0
    }
    
    print("==================================================")
    print("TEST 1 : Question connue (Label Startup Act)")
    print("==================================================")
    print("Submitting question:", initial_state["question"])
    
    # Exécuter le graphe en streams 
    for s in compliance_agent.stream(initial_state):
        if "node_evaluator" in s:
            print(f"\n[EVALUATEUR A2A] Décision : Aller sur le Web ? -> {s['node_evaluator']['needs_web_search']}")
        if "node_web" in s:
            print("\n[RECHERCHE WEB] Outil activé !")
            
    # ------ DUXIEME QUESTION INCONNUE ------
    print("\n==================================================")
    print("TEST 2 : Question très spécifique inconnue du graphe local")
    print("==================================================")
    initial_state_2 = {
        "question": "Quelle est la date exacte de nomination du dernier ministre des technologies en 2026 selon la presse tunisienne ?",
        "has_pdf": False,
        "revision_count": 0
    }
    print("Submitting question:", initial_state_2["question"])
    
    for s in compliance_agent.stream(initial_state_2):
        if "node_evaluator" in s:
            print(f"\n[EVALUATEUR A2A] Décision : Aller sur le Web ? -> {s['node_evaluator']['needs_web_search']}")
        if "node_web" in s:
            print("\n[RECHERCHE WEB] Outil activé !")


if __name__ == "__main__":
    test_a2a_graph()