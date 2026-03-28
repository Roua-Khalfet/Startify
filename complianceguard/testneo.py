"""
ComplianceGuard — test_neo4j.py
================================
Lance ce script EN PREMIER sur ta machine pour valider la connexion.

    python test_neo4j.py

Si OK → lance ingest.py
"""

import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

URI      = os.getenv("NEO4J_URI")
USER     = os.getenv("NEO4J_USERNAME", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD")
DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

print("=" * 50)
print("Test connexion Neo4j Aura — ComplianceGuard")
print("=" * 50)
print(f"URI      : {URI}")
print(f"Username : {USER}")
print(f"Database : {DATABASE}")
print()

try:
    # neo4j+s:// gère déjà SSL, pas besoin de paramètres supplémentaires
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    driver.verify_connectivity()
    print("[OK] Connexion Neo4j Aura reussie !\n")

    with driver.session(database=DATABASE) as session:
        # Test 1 : ping
        result = session.run("RETURN 'ComplianceGuard connecte !' AS msg")
        print("[OK]", result.single()["msg"])

        # Test 2 : vérifier que la base est vide (première fois)
        result = session.run("MATCH (n) RETURN count(n) AS total")
        total = result.single()["total"]
        print(f"[OK] Noeuds existants dans la base : {total}")

        if total == 0:
            print("\n  Base vide - pret pour l'ingestion !")
            print("  Lance : python ingest.py")
        else:
            print(f"\n  Base non vide - {total} noeuds trouves.")
            print("  Lance : python ingest.py  (pour reingerer)")
            print("  ou    : python main.py    (pour tester l'agent)")

        # Test 3 : créer un noeud de test
        session.run("""
            MERGE (t:Test {name: 'ComplianceGuard'})
            SET t.status = 'connected', t.timestamp = datetime()
        """)
        result = session.run("MATCH (t:Test {name:'ComplianceGuard'}) RETURN t.status AS s")
        print(f"[OK] Noeud de test cree : status = {result.single()['s']}")

        # Nettoyer le noeud de test
        session.run("MATCH (t:Test {name:'ComplianceGuard'}) DELETE t")
        print("[OK] Noeud de test supprime")

    driver.close()
    print("\n" + "=" * 50)
    print("TOUT EST OK — Lance maintenant :")
    print("  python ingest.py")
    print("=" * 50)

except Exception as e:
    print(f"\n[ERREUR] Erreur de connexion : {e}")
    print("\nVerifie :")
    print("  1. Ton .env contient les bonnes valeurs")
    print("  2. L'instance Aura est bien demarree (console.neo4j.io)")
    print("  3. Tu as attendu 60 secondes apres la creation")
    print("  4. NEO4J_USERNAME=neo4j (pas d660a5d1)")