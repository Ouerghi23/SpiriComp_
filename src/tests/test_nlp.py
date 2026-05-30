from src.nlp.multilingual_nlp_pipeline import MultilingualNLPPipeline

pipe = MultilingualNLPPipeline()

def test_french_complaint():
    r = pipe.analyze("mon réseau coupe depuis 3 jours")
    assert r["is_complaint"] == True
    assert r["language"] == "fr"

def test_arabic_complaint():
    r = pipe.analyze("شبكتي مقطوعة في تونس")
    assert r["is_complaint"] == True
    assert r["language"] == "ar"

def test_positive_feedback():
    r = pipe.analyze("merci pour votre excellent service")
    assert r["is_complaint"] == False