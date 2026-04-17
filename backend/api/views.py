"""
API Views for ComplianceGuard - Django REST Framework.

Endpoints:
- POST /api/chat/ - Chat avec l'agent GraphRAG
- POST /api/chat/knowledge/ - Chat knowledge-only via backend/knowledge/ask_graph_rag.py
- POST /api/conformite/ - Analyse de conformité avec scoring
- POST /api/documents/ - Génération de documents
- GET /api/graph/ - Données du graphe Neo4j
- GET /api/veille/ - État de la veille web
- POST /api/suggestions/ - Questions suggérées
"""

import os
import sys
import importlib.util
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from .serializers import (
    ChatRequestSerializer,
    ConformiteRequestSerializer,
    DocumentRequestSerializer,
    SuggestionsRequestSerializer,
)

# Add complianceguard to path and load .env
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Force load .env from project root
load_dotenv(PROJECT_ROOT / ".env", override=True)

_LOCAL_GRAPHRAG_FUNCTION = None

# ============================================================================
# Import agents (lazy loading to avoid startup errors)
# ============================================================================

def get_answer_function():
    """Lazy import of crag_answer and answer_question."""
    try:
        from complianceguard.ask_question import answer_question, _is_greeting_or_non_question
        from complianceguard.crag import crag_answer
        return answer_question, crag_answer, _is_greeting_or_non_question
    except Exception as e:
        import traceback
        print(f"ERROR importing ask_question: {e}")
        traceback.print_exc()
        return None, None, None

def get_redacteur():
    """Lazy import of AgentRedacteur."""
    try:
        from complianceguard.agent_redacteur import AgentRedacteur, ProjectInfo
        return AgentRedacteur, ProjectInfo
    except Exception as e:
        print(f"Warning: Could not import agent_redacteur: {e}")
        return None, None


def get_local_graph_rag_function():
    """Lazy import of backend/knowledge/ask_graph_rag.py runner."""
    global _LOCAL_GRAPHRAG_FUNCTION
    if _LOCAL_GRAPHRAG_FUNCTION is not None:
        return _LOCAL_GRAPHRAG_FUNCTION

    module_path = PROJECT_ROOT / "backend" / "knowledge" / "ask_graph_rag.py"
    try:
        spec = importlib.util.spec_from_file_location("local_graph_rag", module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load spec for {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        runner = getattr(module, "run_local_graph_rag", None)
        if runner is None:
            raise AttributeError("run_local_graph_rag not found in ask_graph_rag.py")
        _LOCAL_GRAPHRAG_FUNCTION = runner
        return _LOCAL_GRAPHRAG_FUNCTION
    except Exception as e:
        print(f"ERROR importing local GraphRAG runner: {e}")
        return None


def is_simple_greeting_or_non_question(text: str) -> bool:
    """Lightweight greeting detector to avoid loading full chat stack."""
    text_lower = (text or "").lower().strip()
    greetings = {
        "bonjour", "bonsoir", "salut", "hello", "hi", "hey",
        "coucou", "salam", "bsr", "bjr", "cc", "merci", "thanks",
        "ok", "oui", "non", "d'accord", "super", "bien", "cool",
    }

    words = text_lower.split()
    if len(words) < 3:
        if any(greeting in text_lower for greeting in greetings):
            return True
        if "?" not in text and len(text_lower) < 20:
            return True
    return False

# ============================================================================
# Advanced Conformité Scoring System
# ============================================================================

# Detailed legal requirements database
LEGAL_REQUIREMENTS = {
    "startup_act": {
        "loi": "Loi n° 2018-20 du 17 avril 2018",
        "titre": "Startup Act",
        "criteres": {
            "age_entreprise": {
                "article": "Art. 3, al. 1",
                "texte": "Être constituée depuis moins de 8 ans",
                "poids": 15,
            },
            "caractere_innovant": {
                "article": "Art. 3, al. 2",
                "texte": "Avoir un modèle économique innovant et un fort potentiel de croissance",
                "poids": 25,
                "keywords": ["innovant", "innovation", "technologie", "digital", "ia", "intelligence artificielle", 
                            "blockchain", "machine learning", "saas", "plateforme", "disruption", "scalable"],
            },
            "independance": {
                "article": "Art. 3, al. 3",
                "texte": "Ne pas être une filiale ou résulter d'une restructuration",
                "poids": 10,
            },
            "siege_tunisie": {
                "article": "Art. 3, al. 4",
                "texte": "Avoir son siège social en Tunisie",
                "poids": 10,
            },
            "capital_detenu": {
                "article": "Art. 3, al. 5",
                "texte": "Capital détenu à 2/3 par des personnes physiques ou fonds d'investissement",
                "poids": 10,
            },
        }
    },
    "societes_commerciales": {
        "loi": "Code des Sociétés Commerciales",
        "criteres": {
            "capital_suarl": {
                "article": "Art. 148",
                "texte": "Capital minimum SUARL: 1 000 TND",
                "seuil": 1000,
                "type": "SUARL",
                "poids": 10,
            },
            "capital_sarl": {
                "article": "Art. 92",
                "texte": "Capital minimum SARL: 1 000 TND",
                "seuil": 1000,
                "type": "SARL",
                "poids": 10,
            },
            "capital_sa": {
                "article": "Art. 160",
                "texte": "Capital minimum SA: 5 000 TND (sans APE)",
                "seuil": 5000,
                "type": "SA",
                "poids": 10,
            },
            "statuts_rediges": {
                "article": "Art. 96",
                "texte": "Statuts rédigés par acte authentique ou sous seing privé",
                "poids": 5,
            },
        }
    },
    "protection_donnees": {
        "loi": "Loi organique n° 2004-63",
        "criteres": {
            "declaration_inpdp": {
                "article": "Art. 7",
                "texte": "Déclaration préalable auprès de l'INPDP pour tout traitement",
                "poids": 15,
                "secteurs_critiques": ["Fintech", "HealthTech", "EdTech", "E-commerce"],
            },
            "consentement": {
                "article": "Art. 27",
                "texte": "Consentement explicite pour la collecte de données",
                "poids": 10,
            },
            "finalite": {
                "article": "Art. 9",
                "texte": "Définir la finalité du traitement des données",
                "poids": 5,
            },
            "securite": {
                "article": "Art. 19",
                "texte": "Mesures de sécurité appropriées",
                "poids": 10,
            },
            "droits_personnes": {
                "article": "Art. 32-38",
                "texte": "Garantir les droits d'accès, rectification, suppression",
                "poids": 5,
            },
        }
    },
    "bct_paiement": {
        "loi": "Loi n° 2016-48 + Circulaire BCT 2020-01",
        "criteres": {
            "agrement_etablissement": {
                "article": "Art. 34",
                "texte": "Agrément BCT obligatoire pour activité de paiement",
                "poids": 25,
                "activites": ["paiement", "transfert", "monnaie électronique", "payment", "wallet", "mobile money"],
            },
            "capital_paiement": {
                "article": "Circ. 2020-01, Art. 5",
                "texte": "Capital minimum établissement de paiement: 1 000 000 TND",
                "seuil": 1000000,
                "poids": 15,
            },
            "capital_monnaie_elec": {
                "article": "Circ. 2020-01, Art. 6",
                "texte": "Capital minimum monnaie électronique: 2 000 000 TND",
                "seuil": 2000000,
                "poids": 15,
            },
            "anti_blanchiment": {
                "article": "Loi 2003-75, Art. 74",
                "texte": "Dispositif de lutte contre le blanchiment (KYC/AML)",
                "poids": 20,
            },
        }
    },
    "commerce_electronique": {
        "loi": "Loi n° 2000-83",
        "criteres": {
            "mentions_legales": {
                "article": "Art. 9",
                "texte": "Mentions légales obligatoires sur le site",
                "poids": 10,
                "elements": ["raison sociale", "siège", "RCS", "contact", "TVA"],
            },
            "conditions_generales": {
                "article": "Art. 25",
                "texte": "CGV/CGU accessibles et acceptées avant achat",
                "poids": 10,
            },
            "droit_retractation": {
                "article": "Art. 30",
                "texte": "Droit de rétractation de 10 jours (si applicable)",
                "poids": 10,
            },
            "confirmation_commande": {
                "article": "Art. 28",
                "texte": "Confirmation écrite de la commande",
                "poids": 5,
            },
        }
    }
}

# Risk profiles by sector
SECTOR_RISK_PROFILES = {
    "Fintech": {
        "risque_global": "élevé",
        "lois_applicables": ["startup_act", "societes_commerciales", "protection_donnees", "bct_paiement", "commerce_electronique"],
        "autorisations_requises": ["Agrément BCT", "Déclaration INPDP"],
        "capital_recommande": 1000000,
        "delai_conformite": "6-12 mois",
    },
    "HealthTech": {
        "risque_global": "élevé",
        "lois_applicables": ["startup_act", "societes_commerciales", "protection_donnees"],
        "autorisations_requises": ["Autorisation Ministère Santé", "Déclaration INPDP"],
        "capital_recommande": 50000,
        "delai_conformite": "3-6 mois",
    },
    "EdTech": {
        "risque_global": "moyen",
        "lois_applicables": ["startup_act", "societes_commerciales", "protection_donnees", "commerce_electronique"],
        "autorisations_requises": ["Déclaration INPDP"],
        "capital_recommande": 10000,
        "delai_conformite": "1-3 mois",
    },
    "E-commerce": {
        "risque_global": "moyen",
        "lois_applicables": ["startup_act", "societes_commerciales", "protection_donnees", "commerce_electronique"],
        "autorisations_requises": ["Déclaration INPDP"],
        "capital_recommande": 5000,
        "delai_conformite": "1-2 mois",
    },
    "SaaS": {
        "risque_global": "faible",
        "lois_applicables": ["startup_act", "societes_commerciales", "protection_donnees", "commerce_electronique"],
        "autorisations_requises": [],
        "capital_recommande": 1000,
        "delai_conformite": "< 1 mois",
    },
}

def analyze_text_compliance(text: str, keywords: list) -> tuple[int, list]:
    """Analyse un texte pour détecter la présence de mots-clés."""
    text_lower = text.lower()
    found = [kw for kw in keywords if kw.lower() in text_lower]
    score = min(100, len(found) * 20) if keywords else 50
    return score, found

def calculate_capital_score(capital: int | None, seuil: int) -> tuple[int, str]:
    """Calcule le score basé sur le capital."""
    if capital is None:
        return 30, "Capital non spécifié - à définir"
    if capital >= seuil:
        return 100, f"✓ Capital de {capital:,} TND ≥ {seuil:,} TND requis"
    ratio = capital / seuil
    score = int(ratio * 100)
    manque = seuil - capital
    return score, f"Capital insuffisant: {capital:,} TND < {seuil:,} TND (manque {manque:,} TND)"

def analyze_conformite(data):
    """Analyse avancée de la conformité avec détails par critère."""
    project_description = data.get("project_description", "") or ""
    sector = data.get("sector", "SaaS") or "SaaS"
    capital = data.get("capital")
    type_societe = data.get("type_societe", "SUARL") or "SUARL"
    
    # Get sector risk profile
    risk_profile = SECTOR_RISK_PROFILES.get(sector, SECTOR_RISK_PROFILES["SaaS"])
    applicable_laws = risk_profile["lois_applicables"]
    
    criteres_results = []
    total_weighted_score = 0
    total_weight = 0
    recommendations = []
    
    # 1. STARTUP ACT ELIGIBILITY
    if "startup_act" in applicable_laws:
        startup_reqs = LEGAL_REQUIREMENTS["startup_act"]["criteres"]
        
        # Innovation check
        innov_crit = startup_reqs["caractere_innovant"]
        innov_score, found_keywords = analyze_text_compliance(
            project_description, 
            innov_crit["keywords"]
        )
        
        if innov_score >= 60:
            status = "check"
            details = f"Caractère innovant détecté: {', '.join(found_keywords[:3])}"
        elif innov_score >= 40:
            status = "warning"
            details = f"Innovation partielle. Renforcer: IA, blockchain, ou technologie disruptive"
            recommendations.append("Détailler le caractère innovant de votre solution")
        else:
            status = "x"
            details = "Caractère innovant non démontré dans la description"
            recommendations.append("Reformuler le projet pour mettre en avant l'innovation technologique")
        
        criteres_results.append({
            "label": "Éligibilité Label Startup",
            "score": innov_score,
            "status": status,
            "article": innov_crit["article"],
            "article_source": f"{LEGAL_REQUIREMENTS['startup_act']['loi']} - {LEGAL_REQUIREMENTS['startup_act']['titre']}",
            "details": details,
            "category": "Startup Act",
            "recommendation": "Soumettre dossier au Startup Act pour avantages fiscaux" if innov_score >= 60 else None,
        })
        total_weighted_score += innov_score * innov_crit["poids"]
        total_weight += innov_crit["poids"]
    
    # 2. CAPITAL REQUIREMENTS
    societe_reqs = LEGAL_REQUIREMENTS["societes_commerciales"]["criteres"]
    capital_key = f"capital_{type_societe.lower()}"
    if capital_key in societe_reqs:
        cap_crit = societe_reqs[capital_key]
        cap_score, cap_details = calculate_capital_score(capital, cap_crit["seuil"])
        
        status = "check" if cap_score == 100 else "warning" if cap_score >= 50 else "x"
        
        criteres_results.append({
            "label": f"Capital social ({type_societe})",
            "score": cap_score,
            "status": status,
            "article": cap_crit["article"],
            "article_source": LEGAL_REQUIREMENTS["societes_commerciales"]["loi"],
            "details": cap_details,
            "category": "Forme juridique",
            "recommendation": None if cap_score == 100 else f"Augmenter le capital à {cap_crit['seuil']:,} TND minimum",
        })
        total_weighted_score += cap_score * cap_crit["poids"]
        total_weight += cap_crit["poids"]
    
    # 3. BCT AUTHORIZATION (Fintech specific)
    if sector == "Fintech" or "bct_paiement" in applicable_laws:
        bct_reqs = LEGAL_REQUIREMENTS["bct_paiement"]["criteres"]
        agr_crit = bct_reqs["agrement_etablissement"]
        
        # Check if payment activity detected
        payment_score, found_activities = analyze_text_compliance(
            project_description,
            agr_crit["activites"]
        )
        
        if found_activities:
            # Payment activity detected - BCT required
            cap_paiement = bct_reqs["capital_paiement"]
            cap_score, cap_details = calculate_capital_score(capital, cap_paiement["seuil"])
            
            criteres_results.append({
                "label": "Agrément BCT (Paiement)",
                "score": 20,  # Low score because authorization needed
                "status": "x",
                "article": agr_crit["article"],
                "article_source": LEGAL_REQUIREMENTS["bct_paiement"]["loi"],
                "details": f"Activité de paiement détectée ({', '.join(found_activities)}). Agrément BCT OBLIGATOIRE.",
                "category": "Réglementation BCT",
                "recommendation": "Déposer demande d'agrément auprès de la BCT (délai: 3-6 mois)",
            })
            total_weighted_score += 20 * agr_crit["poids"]
            total_weight += agr_crit["poids"]
            
            criteres_results.append({
                "label": "Capital établissement paiement",
                "score": cap_score,
                "status": "check" if cap_score == 100 else "x",
                "article": cap_paiement["article"],
                "article_source": LEGAL_REQUIREMENTS["bct_paiement"]["loi"],
                "details": cap_details,
                "category": "Réglementation BCT",
                "recommendation": None if cap_score == 100 else "Capital de 1 000 000 TND requis pour établissement de paiement",
            })
            total_weighted_score += cap_score * cap_paiement["poids"]
            total_weight += cap_paiement["poids"]
            
            # AML/KYC requirement
            aml_crit = bct_reqs["anti_blanchiment"]
            criteres_results.append({
                "label": "Dispositif KYC/AML",
                "score": 40,
                "status": "warning",
                "article": aml_crit["article"],
                "article_source": "Loi 2003-75 (Anti-blanchiment)",
                "details": "Dispositif de vérification d'identité et lutte anti-blanchiment requis",
                "category": "Réglementation BCT",
                "recommendation": "Implémenter KYC (vérification identité) + monitoring des transactions",
            })
            total_weighted_score += 40 * aml_crit["poids"]
            total_weight += aml_crit["poids"]
        else:
            criteres_results.append({
                "label": "Agrément BCT",
                "score": 100,
                "status": "check",
                "article": agr_crit["article"],
                "article_source": LEGAL_REQUIREMENTS["bct_paiement"]["loi"],
                "details": "Pas d'activité de paiement détectée - agrément BCT non requis",
                "category": "Réglementation BCT",
                "recommendation": None,
            })
            total_weighted_score += 100 * 10
            total_weight += 10
    
    # 4. DATA PROTECTION
    if "protection_donnees" in applicable_laws:
        data_reqs = LEGAL_REQUIREMENTS["protection_donnees"]["criteres"]
        inpdp_crit = data_reqs["declaration_inpdp"]
        
        is_critical = sector in inpdp_crit["secteurs_critiques"]
        data_keywords = ["données", "data", "utilisateur", "client", "personnel", "information"]
        data_score, found_data = analyze_text_compliance(project_description, data_keywords)
        
        if is_critical or found_data:
            score = 35 if is_critical else 60
            status = "warning" if is_critical else "warning"
            details = f"Secteur {sector}: déclaration INPDP obligatoire avant mise en production" if is_critical else "Traitement de données détecté - déclaration INPDP recommandée"
            
            criteres_results.append({
                "label": "Déclaration INPDP",
                "score": score,
                "status": status,
                "article": inpdp_crit["article"],
                "article_source": LEGAL_REQUIREMENTS["protection_donnees"]["loi"],
                "details": details,
                "category": "Protection des données",
                "recommendation": "Effectuer déclaration sur https://www.inpdp.tn (gratuit, ~2 semaines)",
            })
            total_weighted_score += score * inpdp_crit["poids"]
            total_weight += inpdp_crit["poids"]
            
            # Consent requirement
            consent_crit = data_reqs["consentement"]
            criteres_results.append({
                "label": "Consentement utilisateurs",
                "score": 50,
                "status": "warning",
                "article": consent_crit["article"],
                "article_source": LEGAL_REQUIREMENTS["protection_donnees"]["loi"],
                "details": "Mécanisme de consentement explicite requis pour collecte de données",
                "category": "Protection des données",
                "recommendation": "Ajouter case à cocher + politique de confidentialité",
            })
            total_weighted_score += 50 * consent_crit["poids"]
            total_weight += consent_crit["poids"]
        else:
            criteres_results.append({
                "label": "Protection des données",
                "score": 80,
                "status": "check",
                "article": inpdp_crit["article"],
                "article_source": LEGAL_REQUIREMENTS["protection_donnees"]["loi"],
                "details": "Obligations standard - déclaration INPDP recommandée par précaution",
                "category": "Protection des données",
                "recommendation": None,
            })
            total_weighted_score += 80 * inpdp_crit["poids"]
            total_weight += inpdp_crit["poids"]
    
    # 5. E-COMMERCE REQUIREMENTS
    if "commerce_electronique" in applicable_laws:
        ecom_reqs = LEGAL_REQUIREMENTS["commerce_electronique"]["criteres"]
        
        mentions_crit = ecom_reqs["mentions_legales"]
        cgv_crit = ecom_reqs["conditions_generales"]
        
        criteres_results.append({
            "label": "Mentions légales site web",
            "score": 60,
            "status": "warning",
            "article": mentions_crit["article"],
            "article_source": LEGAL_REQUIREMENTS["commerce_electronique"]["loi"],
            "details": f"Requis: {', '.join(mentions_crit['elements'])}",
            "category": "Commerce électronique",
            "recommendation": "Créer page 'Mentions légales' avec toutes les informations requises",
        })
        total_weighted_score += 60 * mentions_crit["poids"]
        total_weight += mentions_crit["poids"]
        
        criteres_results.append({
            "label": "CGU/CGV",
            "score": 55,
            "status": "warning",
            "article": cgv_crit["article"],
            "article_source": LEGAL_REQUIREMENTS["commerce_electronique"]["loi"],
            "details": "Conditions générales obligatoires et acceptées avant transaction",
            "category": "Commerce électronique",
            "recommendation": "Rédiger CGU/CGV conformes (utiliser AgentRédacteur)",
        })
        total_weighted_score += 55 * cgv_crit["poids"]
        total_weight += cgv_crit["poids"]
    
    # Calculate final score
    score_global = int(total_weighted_score / total_weight) if total_weight > 0 else 50
    
    # Determine status
    if score_global >= 75:
        status_global = "conforme"
    elif score_global >= 50:
        status_global = "conforme_reserves"
    else:
        status_global = "non_conforme"
    
    # Build recommendations summary
    all_recommendations = [c["recommendation"] for c in criteres_results if c.get("recommendation")]
    
    return {
        "score_global": score_global,
        "status": status_global,
        "criteres": criteres_results,
        "risk_profile": {
            "niveau": risk_profile["risque_global"],
            "autorisations_requises": risk_profile["autorisations_requises"],
            "capital_recommande": risk_profile["capital_recommande"],
            "delai_conformite": risk_profile["delai_conformite"],
        },
        "recommendations": all_recommendations[:5],  # Top 5 recommendations
        "lois_applicables": [LEGAL_REQUIREMENTS[l]["loi"] for l in applicable_laws if l in LEGAL_REQUIREMENTS],
    }

# ============================================================================
# Suggestions Logic
# ============================================================================

SUGGESTION_TEMPLATES = {
    "Fintech": [
        "Ai-je besoin d'une autorisation BCT pour mon activité ?",
        "Quel capital minimum pour un établissement de paiement ?",
        "Quelles sont les obligations de lutte anti-blanchiment ?",
        "Comment obtenir l'agrément d'établissement de paiement ?",
    ],
    "EdTech": [
        "Quelles autorisations pour une plateforme éducative en ligne ?",
        "Protection des données des mineurs : quelles obligations ?",
        "Agrément du Ministère de l'Éducation nécessaire ?",
        "Certification des contenus pédagogiques requise ?",
    ],
    "HealthTech": [
        "Réglementation des dispositifs médicaux connectés ?",
        "Hébergement des données de santé : quelles contraintes ?",
        "Autorisation du Ministère de la Santé nécessaire ?",
        "Responsabilité médicale et applications de santé ?",
    ],
    "E-commerce": [
        "Mentions légales obligatoires pour un site e-commerce ?",
        "Droit de rétractation : quelles obligations ?",
        "TVA et facturation électronique en Tunisie ?",
        "Protection du consommateur : clauses interdites ?",
    ],
    "SaaS": [
        "Clauses essentielles d'un contrat SaaS ?",
        "Responsabilité en cas d'interruption de service ?",
        "Propriété intellectuelle du code développé ?",
        "Transfert de données hors Tunisie : conditions ?",
    ],
    "default": [
        "Quels avantages du Startup Act puis-je obtenir ?",
        "Comment obtenir le label Startup ?",
        "Quel type de société pour ma startup ?",
        "Quelles obligations fiscales pour une startup ?",
    ]
}

def generate_suggestions(project_description, sector):
    """Génère des questions suggérées basées sur le contexte."""
    sector_suggestions = SUGGESTION_TEMPLATES.get(sector, SUGGESTION_TEMPLATES["default"])
    
    desc_lower = project_description.lower()
    contextual = []
    
    if "paiement" in desc_lower or "payment" in desc_lower:
        contextual.append("Réglementation des services de paiement en Tunisie ?")
    if "données" in desc_lower or "data" in desc_lower:
        contextual.append("Obligations INPDP pour le traitement de données ?")
    if "international" in desc_lower or "export" in desc_lower:
        contextual.append("Compte en devises pour startup : conditions ?")
    if "investissement" in desc_lower or "levée" in desc_lower:
        contextual.append("Avantages fiscaux pour les investisseurs (Startup Act) ?")
    
    all_suggestions = contextual + sector_suggestions
    return all_suggestions[:6]

# ============================================================================
# API Views
# ============================================================================

@api_view(["GET"])
def api_root(request):
    """API root endpoint."""
    return Response({
        "message": "ComplianceGuard API",
        "version": "1.0.0",
        "endpoints": {
            "chat": "/api/chat/",
            "chat_knowledge": "/api/chat/knowledge/",
            "upload": "/api/upload/",
            "conformite": "/api/conformite/",
            "documents": "/api/documents/",
            "graph": "/api/graph/",
            "veille": "/api/veille/",
            "suggestions": "/api/suggestions/",
        }
    })

@api_view(["POST"])
def chat(request):
    """Chat avec l'agent GraphRAG ou le système CRAG."""
    serializer = ChatRequestSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    data = serializer.validated_data
    message = data["message"]
    project_context = data.get("project_context", "")
    mode = data.get("mode", "kb") # "kb" pour GraphRAG, "notebook" pour CRAG uploads
    
    answer_question_func, crag_func, is_greeting = get_answer_function()
    
    if answer_question_func is None or crag_func is None:
        # Fallback response if agents not available
        return Response({
            "response": "Les agents ne sont pas disponibles. Vérifiez que les dépendances sont installées.",
            "sources": [],
            "source_type": "error"
        })
    
    # Check for greetings
    if is_greeting and is_greeting(message):
        return Response({
            "response": (
                "Bonjour ! Je suis l'assistant juridique ComplianceGuard, "
                "spécialisé dans le Startup Act tunisien.\n\n"
                "Posez-moi une question juridique, par exemple :\n"
                "- Quels documents pour obtenir le label startup ?\n"
                "- Quels sont les avantages fiscaux du Startup Act ?\n"
                "- Comment obtenir le congé startup ?"
            ),
            "sources": [],
            "source_type": "system"
        })
    
    # Enrich question with context
    enriched_question = message
    # In notebook mode we keep the raw user question to avoid degrading
    # CRAG retrieval with UI-level context strings.
    if project_context and mode != "notebook":
        enriched_question = f"Contexte projet: {project_context}\n\nQuestion: {message}"
    
    try:
        if mode == "notebook":
            answer, sources, metadata = crag_func(enriched_question, enable_web_fallback=False, mode="notebook")
            source_type = metadata.get("action", "CRAG")
        else:
            answer, sources = answer_question_func(enriched_question, enable_web_fallback=True, mode="kb")
            metadata = {}
            source_type = "GraphRAG"
        
        return Response({
            "response": answer,
            "sources": sources,
            "source_type": source_type,
            "metadata": metadata
        })
    except Exception as e:
        # Fallback: return a helpful message when Qdrant/Neo4j not available
        error_msg = str(e)
        if "compliance_vectors" in error_msg or "Qdrant" in error_msg:
            return Response({
                "response": (
                    "⚠️ **Base de données non initialisée**\n\n"
                    "La collection Qdrant n'existe pas encore. "
                    "Exécutez d'abord l'ingestion des documents :\n\n"
                    "```bash\ncd complianceguard && python ingest.py\n```\n\n"
                    "Cela va indexer les PDFs du dossier `Data/` dans Neo4j et Qdrant."
                ),
                "sources": [],
                "source_type": "error"
            })
        return Response({
            "response": f"Erreur: {error_msg}",
            "sources": [],
            "source_type": "error"
        })


@api_view(["POST"])
def chat_knowledge(request):
    """Chat knowledge-only via local GraphRAG pipeline from backend/knowledge/ask_graph_rag.py."""
    serializer = ChatRequestSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    data = serializer.validated_data
    message = data["message"]
    project_context = data.get("project_context", "")

    if is_simple_greeting_or_non_question(message):
        return Response({
            "response": (
                "Bonjour ! Je suis l'assistant juridique ComplianceGuard, "
                "spécialisé dans le Startup Act tunisien.\n\n"
                "Posez-moi une question juridique, par exemple :\n"
                "- Quels documents pour obtenir le label startup ?\n"
                "- Quels sont les avantages fiscaux du Startup Act ?\n"
                "- Comment obtenir le congé startup ?"
            ),
            "sources": [],
            "source_type": "system",
            "metadata": {"mode": "knowledge-only"},
        })

    graphrag_func = get_local_graph_rag_function()
    if graphrag_func is None:
        return Response({
            "response": "Le moteur GraphRAG local n'est pas disponible actuellement.",
            "sources": [],
            "source_type": "error",
            "metadata": {"mode": "knowledge-only"},
        }, status=status.HTTP_503_SERVICE_UNAVAILABLE)

    enriched_question = message
    if project_context:
        enriched_question = f"Contexte projet: {project_context}\n\nQuestion: {message}"

    try:
        answer, sources, metadata = graphrag_func(
            enriched_question,
            llm_provider="azure",
            timeout=45,
            top_k=4,
            max_sources=6,
        )
        return Response({
            "response": answer,
            "sources": sources,
            "source_type": metadata.get("source_type", "GraphRAG Local") if isinstance(metadata, dict) else "GraphRAG Local",
            "metadata": metadata if isinstance(metadata, dict) else {"mode": "knowledge-only"},
        })
    except Exception as e:
        return Response({
            "response": f"Erreur GraphRAG local: {e}",
            "sources": [],
            "source_type": "error",
            "metadata": {"mode": "knowledge-only"},
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

import tempfile

@api_view(["POST"])
def upload_document(request):
    """Uploads a file for fast ingestion."""
    if 'file' not in request.FILES:
        return Response({"error": "No file uploaded"}, status=status.HTTP_400_BAD_REQUEST)
        
    try:
        from complianceguard.ingest import fast_ingest_file
    except ImportError:
        return Response({"error": "Document parsing not available (unstructured not installed)"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    uploaded_file = request.FILES['file']
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tf:
        for chunk in uploaded_file.chunks():
            tf.write(chunk)
        temp_path = tf.name
        
    # We rename it to the original locally so Docling uses the right extension
    try:
        target_path = Path(temp_path).parent / uploaded_file.name
        os.rename(temp_path, target_path)
        result = fast_ingest_file(target_path)
        os.unlink(target_path)
        if result.get("status") != "success" or int(result.get("chunks_indexed") or 0) <= 0:
            return Response(
                {"error": result.get("message", "Ingestion failed")},
                status=status.HTTP_422_UNPROCESSABLE_ENTITY,
            )
        return Response(result)
    except Exception as e:
        if os.path.exists(target_path):
            os.unlink(target_path)
        elif os.path.exists(temp_path):
            os.unlink(temp_path)
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(["POST"])
def conformite(request):
    """Analyse de conformité avec scoring."""
    serializer = ConformiteRequestSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    result = analyze_conformite(serializer.validated_data)
    return Response(result)

@api_view(["POST"])
def generate_documents(request):
    """Génère des documents juridiques."""
    serializer = DocumentRequestSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    data = serializer.validated_data
    AgentRedacteur, ProjectInfo = get_redacteur()
    
    if AgentRedacteur is None:
        return Response({
            "error": "Agent Rédacteur non disponible"
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    try:
        agent = AgentRedacteur()
        project = ProjectInfo(
            nom_startup=data["nom_startup"],
            activite=data["activite"],
            fondateurs=data.get("fondateurs", []),
            capital_social=data.get("capital_social", 1000),
            siege_social=data.get("siege_social", "Tunis"),
            type_societe=data.get("type_societe", "SUARL"),
        )
        
        results = []
        
        if data["doc_type"] == "all":
            docs = agent.generer_pack_complet(project)
            for doc_type, content in docs.items():
                results.append({
                    "doc_type": doc_type,
                    "content": content,
                    "filename": f"{data['nom_startup']}_{doc_type}.md"
                })
        else:
            content = agent.generer_document(data["doc_type"], project)
            results.append({
                "doc_type": data["doc_type"],
                "content": content,
                "filename": f"{data['nom_startup']}_{data['doc_type']}.md"
            })
        
        return Response(results)
    except Exception as e:
        return Response({
            "error": str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(["GET"])
def get_graph(request):
    """Récupère les données du graphe Neo4j."""
    # Default graph data (real implementation would query Neo4j)
    default_graph = {
        "nodes": [
            {"id": "loi_2018_20", "label": "Loi 2018-20 (Startup Act)", "type": "loi", "properties": {"titre": "Startup Act"}},
            {"id": "art_3", "label": "Article 3 - Conditions", "type": "article", "properties": {"sujet": "Conditions label"}},
            {"id": "art_13", "label": "Article 13 - Avantages fiscaux", "type": "article", "properties": {"sujet": "Avantages fiscaux"}},
            {"id": "art_20", "label": "Article 20 - Congé startup", "type": "article", "properties": {"sujet": "Congé création"}},
            {"id": "decret_840", "label": "Décret 2018-840", "type": "loi", "properties": {"titre": "Application"}},
            {"id": "bct_2019_01", "label": "Circulaire BCT 2019-01", "type": "loi", "properties": {"titre": "Devises"}},
            {"id": "startup", "label": "Startup", "type": "entite", "properties": {}},
            {"id": "investisseur", "label": "Investisseur", "type": "entite", "properties": {}},
            {"id": "label", "label": "Label Startup", "type": "concept", "properties": {}},
        ],
        "edges": [
            {"source": "loi_2018_20", "target": "art_3", "relation": "CONTIENT"},
            {"source": "loi_2018_20", "target": "art_13", "relation": "CONTIENT"},
            {"source": "loi_2018_20", "target": "art_20", "relation": "CONTIENT"},
            {"source": "decret_840", "target": "loi_2018_20", "relation": "APPLIQUE"},
            {"source": "bct_2019_01", "target": "loi_2018_20", "relation": "REFERENCE"},
            {"source": "art_3", "target": "startup", "relation": "DEFINIT"},
            {"source": "art_3", "target": "label", "relation": "ETABLIT"},
            {"source": "art_13", "target": "investisseur", "relation": "CONCERNE"},
            {"source": "art_20", "target": "startup", "relation": "BENEFICIE"},
        ]
    }
    
    try:
        from complianceguard.tools.retriever import get_graph as get_neo4j_graph
        graph = get_neo4j_graph()
        
        query = """
        MATCH (n)-[r]->(m)
        RETURN n, r, m
        LIMIT 50
        """
        result = graph.query(query)
        
        nodes = []
        edges = []
        seen_nodes = set()
        
        for record in result:
            n = record.get("n", {})
            m = record.get("m", {})
            r = record.get("r", {})
            
            for node in [n, m]:
                if node and hasattr(node, "element_id"):
                    node_id = str(node.element_id)
                    if node_id not in seen_nodes:
                        seen_nodes.add(node_id)
                        labels = list(node.labels) if hasattr(node, "labels") else ["Unknown"]
                        nodes.append({
                            "id": node_id,
                            "label": dict(node).get("name", dict(node).get("titre", node_id[:20])),
                            "type": labels[0].lower() if labels else "unknown",
                            "properties": dict(node)
                        })
            
            if r and hasattr(r, "type"):
                edges.append({
                    "source": str(r.start_node.element_id),
                    "target": str(r.end_node.element_id),
                    "relation": r.type
                })
        
        if nodes:
            return Response({"nodes": nodes, "edges": edges})
    except Exception as e:
        print(f"Error fetching graph: {e}")
    
    return Response(default_graph)

@api_view(["GET"])
def get_veille(request):
    """Récupère l'état de la veille web."""
    now = datetime.now().isoformat()
    
    # Try to get real veille status
    try:
        from complianceguard.agent_veille import load_cache, SITES_TO_MONITOR
        cache = load_cache()
        
        items = []
        for site in SITES_TO_MONITOR:
            page_info = cache.pages.get(site["url"])
            if page_info:
                items.append({
                    "url": site["url"],
                    "nom": site["name"],
                    "last_check": page_info.timestamp,
                    "has_changed": False,
                    "status": "ok"
                })
            else:
                items.append({
                    "url": site["url"],
                    "nom": site["name"],
                    "last_check": now,
                    "has_changed": False,
                    "status": "pending"
                })
        
        return Response({
            "items": items,
            "last_update": cache.last_update or now
        })
    except Exception:
        pass
    
    # Default response
    return Response({
        "items": [
            {"url": "https://startup.gov.tn", "nom": "Portail Startup", "last_check": now, "has_changed": False, "status": "ok"},
            {"url": "https://www.bct.gov.tn", "nom": "BCT - Circulaires", "last_check": now, "has_changed": False, "status": "ok"},
            {"url": "https://www.apii.tn", "nom": "APII", "last_check": now, "has_changed": False, "status": "ok"},
        ],
        "last_update": now
    })

@api_view(["POST"])
def get_suggestions(request):
    """Génère des questions suggérées."""
    serializer = SuggestionsRequestSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    data = serializer.validated_data
    questions = generate_suggestions(
        data["project_description"],
        data["sector"]
    )
    
    return Response({"questions": questions})
