# src/tests/test_nlp.py
# ─────────────────────────────────────────────────────────────────────
# Tests NLP pipeline — après FIX-BUG1 et FIX-BUG2
# ─────────────────────────────────────────────────────────────────────

import pytest
from src.nlp.multilingual_nlp_pipeline import MultilingualNLPPipeline

pipe = MultilingualNLPPipeline()


# ── Langue ────────────────────────────────────────────────────────────

def test_detect_french():
    r = pipe.analyze("mon réseau coupe depuis 3 jours")
    assert r["language"] == "fr"

def test_detect_arabic():
    """FIX-BUG1 : arabic_ch doit compter les CARACTÈRES, pas les mots."""
    r = pipe.analyze("شبكتي مقطوعة في تونس")
    assert r["language"] == "ar", (
        f"Attendu 'ar', reçu '{r['language']}' — "
        "BUG-1 non corrigé : detect_language compte les mots arabes "
        "au lieu des caractères arabes."
    )

def test_detect_english():
    r = pipe.analyze("My network keeps dropping in Tunis")
    assert r["language"] == "en"


# ── Classification is_complaint ───────────────────────────────────────

def test_french_complaint():
    r = pipe.analyze("mon réseau coupe depuis 3 jours")
    assert r["language"]    == "fr"
    assert r["is_complaint"] is True

def test_arabic_complaint():
    r = pipe.analyze("شبكتي مقطوعة في تونس منذ 3 أيام")
    assert r["language"]    == "ar"
    assert r["is_complaint"] is True

def test_english_complaint():
    r = pipe.analyze("My network keeps dropping in Tunis since yesterday")
    assert r["language"]    == "en"
    assert r["is_complaint"] is True

def test_french_feedback():
    """
    FIX-BUG2 : 'lent' est sous-chaîne de 'excellent'.
    Avant le fix, 'merci pour votre excellent service' déclenchait
    la catégorie 'Débit / Internet' → is_complaint=True (faux positif).
    """
    r = pipe.analyze("merci pour votre excellent service")
    assert r["language"]    == "fr"
    assert r["is_complaint"] is False, (
        f"Attendu False, reçu True — "
        "BUG-2 non corrigé : 'lent' matche comme sous-chaîne de 'excellent' "
        "dans _category(). Utiliser \\b word boundaries."
    )

def test_arabic_feedback():
    r = pipe.analyze("شكرا جزيلا، الخدمة ممتازة")
    assert r["language"]    == "ar"
    assert r["is_complaint"] is False

def test_english_feedback():
    r = pipe.analyze("Thank you, service has improved!")
    assert r["language"]    == "en"
    assert r["is_complaint"] is False

def test_question_not_complaint():
    r = pipe.analyze("Comment activer le roaming international ?")
    assert r["language"]    == "fr"
    assert r["is_complaint"] is False


# ── Urgence ───────────────────────────────────────────────────────────

def test_urgent_complaint():
    r = pipe.analyze("réseau coupé depuis 5 jours, urgent !!")
    assert r["urgency_level"] in ["urgent", "très urgent"]

def test_normal_urgency():
    r = pipe.analyze("merci pour votre service")
    assert r["urgency_level"] == "normal"


# ── Catégorie ─────────────────────────────────────────────────────────

def test_category_reseau():
    r = pipe.analyze("mon réseau coupe tout le temps")
    assert r["category"] == "Réseau / Couverture"

def test_category_not_triggered_by_substring():
    """
    FIX-BUG2 : 'lent' ne doit PAS matcher dans 'excellent'.
    """
    r = pipe.analyze("service excellent et rapide")
    assert r["category"] not in ["Débit / Internet"], (
        "'lent' (sous-chaîne de 'excellent') a déclenché la catégorie "
        "Débit / Internet — BUG-2 non corrigé."
    )