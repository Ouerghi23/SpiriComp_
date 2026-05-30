"""
multilingual_nlp_pipeline.py  —  src/nlp/
==========================================
Multilingual NLP pipeline for Ooredoo complaint analysis.
Supports Arabic, French, and English without translation.

FIXES:
  FIX-1  Accent normalization
  FIX-2  French verb forms in lexicons
  FIX-3  "!" preserved in preprocessing
  FIX-4  sklearn path uses Path(__file__)
  FIX-5  Classifier check at top of _is_complaint
  FIX-6  detect_language: count Arabic CHARACTERS not word matches
  FIX-7  _category: word boundary (\b) for single-word keywords
  FIX-8  Removed stray `from matplotlib import text` import
  FIX-9  _sentiment: word boundary for single-word FR/EN keywords
         "lent" was substring-matching inside "excellent"
  FIX-10 AR_COMPLAINT_SIGNALS: added "مقطوع" covers مقطوعة/مقطوعه
  FIX-11 ENGLISH_WORDS: added "thank","thanks","improved","great","good"
         to break FR/EN tie when both score 1
"""

from __future__ import annotations

import pickle
import re
import unicodedata
from datetime import datetime
from pathlib import Path

import pandas as pd

# ── FIX-4 ─────────────────────────────────────────────────────────────────────
_CLASSIFIER_PATH = Path(__file__).resolve().parents[2] / "models" / "nlp" / "classifier.pkl"
_trained_clf: object = None


def _load_trained_classifier():
    global _trained_clf
    if _trained_clf is None and _CLASSIFIER_PATH.exists():
        with open(_CLASSIFIER_PATH, "rb") as f:
            _trained_clf = pickle.load(f)
    return _trained_clf


# ══════════════════════════════════════════════════════════════════════════════
# ACCENT NORMALIZATION (FIX-1)
# ══════════════════════════════════════════════════════════════════════════════

def _strip_accents(s: str) -> str:
    return (
        unicodedata.normalize("NFD", s)
        .encode("ascii", "ignore")
        .decode("utf-8")
    )


def _norm(s: str) -> str:
    return _strip_accents(s.lower())


# ══════════════════════════════════════════════════════════════════════════════
# LANGUAGE DETECTION
# ══════════════════════════════════════════════════════════════════════════════

ARABIC_RANGE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+")

FRENCH_WORDS = {
    "mon","ma","le","la","les","de","du","je","ne","pas","un","une",
    "reseau","internet","appel","connexion","coupure","lent","probleme",
    "depuis","jours","impossible","mauvais","bonjour","merci",
    "debit","facture","client","panne","signal","appels",
    # NOTE: "service" removed — appears in both FR and EN
}

# FIX-11: added unambiguous English words to break FR/EN ties
ENGLISH_WORDS = {
    "my","the","is","not","network","internet","call","connection",
    "slow","problem","since","days","impossible","bad","hello","thanks",
    "signal","drop","weak","no","service","issue","complaint","data",
    "billing","customer","support","coverage","speed","working",
    "thank","improved","better","awesome","wonderful","fantastic",
}


def detect_language(text: str) -> str:
    """Detect language: 'ar', 'fr', or 'en'."""
    if not text or not isinstance(text, str):
        return "fr"
    # FIX-6: count Arabic CHARACTERS not word-match count
    arabic_ch = sum(len(m) for m in ARABIC_RANGE.findall(text))
    if arabic_ch > len(text) * 0.2:
        return "ar"
    text_norm = _strip_accents(text.lower())
    words     = set(re.findall(r"\b\w+\b", text_norm))
    fr_score  = len(words & FRENCH_WORDS)
    en_score  = len(words & ENGLISH_WORDS)
    if fr_score == 0 and en_score == 0:
        return "fr"
    return "fr" if fr_score >= en_score else "en"


# ══════════════════════════════════════════════════════════════════════════════
# LEXICONS — FRENCH
# ══════════════════════════════════════════════════════════════════════════════

FR_CATEGORIES = {
    "Réseau / Couverture": [
        "pas de réseau","aucun signal","pas de couverture","zone blanche",
        "hors réseau","réseau indisponible","signal faible","coupure réseau",
        "réseau mobile","antenne","4g","5g","3g","lte","perte réseau",
        "pas de 4g","réseau coupe","pas de signal","sans réseau",
        "reseau coupe","le réseau coupe","mon réseau coupe",
        "mon reseau coupe","réseau qui coupe","ça coupe","ca coupe",
        "coupe tout le temps","coupe souvent","réseau coupé","réseau absent",
    ],
    "Débit / Internet": [
        "débit faible","internet lent","connexion lente","lenteur",
        "téléchargement lent","mbps","bande passante","accès internet",
        "pas d'internet","internet ne marche pas","coupure internet",
        "déconnexion","connexion instable","page ne charge pas",
        "vitesse","lags","débit","lent",
        "internet coupe","internet se coupe","internet qui coupe",
        "internet ne fonctionne pas","internet marche pas","internet fonctionne pas",
    ],
    "Appels / Voix": [
        "appel coupé","coupure appel","appel impossible","voix haché","echo",
        "bruit","qualité voix","appel échoué","ne peut pas appeler",
        "appel ne passe pas","tonalité","volte","appels coupent",
        "ligne coupe","communication coupée","grésillement",
        "appel coupe","appel qui coupe","les appels coupent",
        "ne peux pas appeler","impossible d'appeler","appels ne passent pas",
    ],
    "SMS": [
        "sms non reçu","sms non envoyé","message non délivré","texto",
        "sms bloqué","message échoué","ne reçois pas sms",
    ],
    "Facturation": [
        "facture","surfacturation","débit abusif","recharge","solde incorrect",
        "forfait","crédit","tarification","erreur facturation","prélèvement",
        "souscription","option","facturation","trop payé",
    ],
    "Support Client": [
        "service client","attente","conseiller","réclamation non traitée",
        "aucune réponse","problème non résolu","rappel","hotline",
        "pas de retour","support","assistance","réclamation",
    ],
}

FR_SENTIMENT = {
    "critique": [
        "inacceptable","inadmissible","scandaleux","honteux","catastrophique",
        "nul","terrible","arnaque","résiliation","porter plainte","aberrant",
        "dégoutant","intolérable","exaspérant","insupportable",
    ],
    "négatif": [
        "problème","panne","coupure","mauvais","lent","difficile","impossible",
        "ne marche pas","ne fonctionne pas","déçu","insatisfait","gêné",
        "énervé","fatigant","perturbation","dysfonctionnement",
        "marche pas","fonctionne pas","coupe","ca coupe",
        "ca marche pas","ca fonctionne pas","reseau coupe","tout coupe",
    ],
    "positif": [
        "merci","bien","bon","excellent","parfait","satisfait","rapide","bravo",
        "génial","top","super","agréable","content",
    ],
}

FR_URGENCY = [
    "urgence","immédiatement","tout de suite","bloqué","impossible",
    "hôpital","danger","critique","sos","grave","priorité",
    "urgent","vite","maintenant","bloque","bloquer",
]

FR_STOPS = {
    "le","la","les","un","une","des","de","du","en","et","est","au","aux",
    "ce","se","je","tu","il","nous","vous","ils","mon","ma","mes","ton",
    "que","qui","pour","par","avec","sur","dans","ne","pas","plus","tres",
    "bien","tout","meme","si","mais","depuis","lors","jai","cest","na",
    "faire","etre","avoir","aller","venir","sans","comme",
}

# ══════════════════════════════════════════════════════════════════════════════
# LEXICONS — ARABIC
# ══════════════════════════════════════════════════════════════════════════════

AR_CATEGORIES = {
    "الشبكة / التغطية": [
        "لا شبكة","انقطاع الشبكة","ضعف التغطية","لا إشارة","شبكة مقطوعة",
        "لا تغطية","إشارة ضعيفة","الشبكة واقعة","لا 4g","لا 3g",
        "تغطية سيئة","الشبكة لا تعمل","انقطاع متكرر","فقدان الشبكة",
        "منقطع","مشكلة شبكة","التغطية","الأنتنة",
    ],
    "الصبيب / الأنترنت": [
        "أنترنت بطيء","صبيب ضعيف","اتصال بطيء","تحميل بطيء",
        "أنترنت منقطع","لا أنترنت","اتصال غير مستقر","سرعة ضعيفة",
        "أنترنت لا يعمل","انقطاع الأنترنت","ميغابت","بطء",
        "النت","الأنترنت","السرعة","تقطيع",
    ],
    "المكالمات / الصوت": [
        "المكالمة منقطعة","انقطاع المكالمات","لا يمكن الاتصال",
        "جودة صوت سيئة","صدى","ضجيج","المكالمة لا تمر","فشل المكالمة",
        "مكالمة مقطوعة","لا أسمع","اتصال صوتي","صعوبة في المكالمة",
        "مشكلة صوت","مكالمة","اتصال",
    ],
    "الرسائل القصيرة": [
        "sms","رسالة لم تصل","رسالة لم ترسل","لم استلم الرسالة",
        "الرسالة فشلت","مشكلة رسائل","واتساب","ماسنجر",
    ],
    "الفاتورة": [
        "فاتورة","خصم مجحف","رصيد","شحن","خطأ في الفاتورة",
        "تسعير","اشتراك","خصم","رصيد غير صحيح","سحب فلوس",
        "فلوس","حساب","فوترة","تخفيض",
    ],
    "دعم العملاء": [
        "خدمة العملاء","انتظار","لا رد","شكوى غير محلولة",
        "لا استجابة","خط ساخن","موظف","شكوى","تذمر",
        "المشكلة","مساعدة","دعم","مركز الاتصال",
    ],
}

AR_SENTIMENT = {
    "حرج": [
        "لا يقبل","فضيحة","كارثي","مرفوض","سيء جدا","مشكلة كبيرة",
        "غير مقبول","رديء","فاضح","مروع","فظيع","مقزز",
    ],
    "سلبي": [
        "مشكلة","عطل","انقطاع","بطيء","صعب","مستحيل","لا يعمل",
        "تعطل","غير راضي","متضايق","غاضب","زعلان","متعب",
        "خلل","عيوب","تعب","ملل","مقطوع",
    ],
    "إيجابي": [
        "شكرا","ممتاز","جيد","رائع","راضي","سريع","أحسنتم",
        "ممتازة","جميل","حسن","سرور","استحسان",
    ],
}

AR_URGENCY = [
    "عاجل","فورا","الآن","خطر","حرج","طوارئ","ضروري","مستعجل",
    "مهم جدا","خطير","حياة","مستشفى","سريع","حالا",
]

AR_CITIES = {
    "تونس","صفاقس","سوسة","القيروان","بنزرت","قابس","أريانة",
    "قفصة","المنستير","بن عروس","نابل","القصرين","سيدي بوزيد",
    "المهدية","منوبة","جندوبة","سليانة","زغوان","باجة","الكاف",
    "قبلي","توزر","تطاوين","مدنين","الحمامات","جرجيس","جربة",
    "المرسى","الكرم","قرطاج",
}

AR_STOPS = {
    "في","من","إلى","على","عن","مع","هذا","هذه","التي","الذي",
    "و","أو","لكن","لأن","كان","كانت","هو","هي","نحن","أنا",
    "أن","إن","قد","لقد","لا","ما","كل","بعض","ذلك","تلك",
}

# ══════════════════════════════════════════════════════════════════════════════
# LEXICONS — ENGLISH
# ══════════════════════════════════════════════════════════════════════════════

EN_CATEGORIES = {
    "Network / Coverage": [
        "no network","no signal","no coverage","dead zone","network down",
        "weak signal","network drop","no 4g","no 3g","network unavailable",
        "signal lost","poor coverage","network cut","out of service",
        "no service","can't connect","signal issue","coverage issue",
    ],
    "Data / Internet": [
        "slow internet","slow connection","low speed","buffering","lag",
        "no internet","internet down","connection drop","disconnecting",
        "unstable connection","download slow","mbps","bandwidth",
        "speed issue","data not working","wifi problem","can't browse",
    ],
    "Calls / Voice": [
        "call drops","call failed","cannot call","voice quality","echo",
        "noise","call cut","call disconnected","no dial tone",
        "can't make calls","volte","call breaking up","can't hear",
        "call quality","dropped call","failing calls",
    ],
    "SMS": [
        "sms not received","sms not delivered","message failed",
        "text not sent","sms blocked","can't send sms","no sms",
    ],
    "Billing": [
        "invoice","overcharged","billing error","recharge","balance wrong",
        "plan","credit","wrong charge","deducted","overcharge",
        "bill","payment","subscription","charging",
    ],
    "Customer Support": [
        "customer service","waiting","no response","complaint not resolved",
        "hotline","callback","agent","support","no help","bad service",
        "unresolved issue","no solution","poor support",
    ],
}

EN_SENTIMENT = {
    "critical": [
        "unacceptable","terrible","awful","outrageous","disgusting",
        "worst","horrible","cancel","lawsuit","scam","fraud",
        "intolerable","unbearable","insufferable","atrocious",
    ],
    "negative": [
        "problem","issue","broken","slow","bad","impossible","doesn't work",
        "not working","disappointed","unhappy","frustrated","annoyed",
        "upset","angry","poor","mediocre","useless",
    ],
    "positive": [
        "thank","good","great","excellent","perfect","satisfied","fast",
        "well done","awesome","amazing","wonderful","fantastic",
        "happy","pleased","impressed",
    ],
}

EN_URGENCY = [
    "urgent","immediately","right now","blocked","critical","emergency",
    "asap","danger","sos","severe","priority","quick","fast",
    "hours","minutes","now",
]

EN_STOPS = {
    "the","a","an","is","are","was","were","be","been","have","has","had",
    "do","does","did","will","would","could","should","may","might","i",
    "my","your","his","her","our","their","it","this","that","these",
    "those","and","or","but","for","so","yet","nor","in","on","at",
    "to","of","with","by","from","up","about","into","through","during",
    "without","between","after","before","above","below","between",
}

TN_CITIES_EN = {
    "tunis","sfax","sousse","kairouan","bizerte","gabes","ariana",
    "gafsa","monastir","ben arous","nabeul","kasserine","sidi bouzid",
    "mahdia","manouba","jendouba","siliana","zaghouan","beja","kef",
    "kebili","tozeur","tataouine","medenine","hammamet","zarzis","djerba",
    "marsa","kram","carthage","sakiet","sidi","raoued",
}

TN_CITIES_FR = {
    "tunis","sfax","sousse","kairouan","bizerte","gabes","ariana",
    "gafsa","monastir","ben arous","nabeul","kasserine","sidi bouzid",
    "mahdia","manouba","jendouba","siliana","zaghouan","beja","kef",
    "kebili","tozeur","tataouine","medenine","hammamet",
    "zarzis","djerba","la marsa","el kram","carthage","sakiet","raoued",
}

# ══════════════════════════════════════════════════════════════════════════════
# COMPLAINT vs FEEDBACK SIGNAL LEXICONS
# ══════════════════════════════════════════════════════════════════════════════

FR_COMPLAINT_SIGNALS = [
    "pas de reseau","pas de signal","pas de couverture","hors reseau",
    "reseau coupe","reseau ne marche pas","reseau ne fonctionne pas",
    "coupure reseau","reseau instable","reseau tombe","pas de 4g",
    "le reseau coupe","mon reseau coupe","reseau qui coupe",
    "ca coupe","coupe tout le temps","coupe souvent","tout coupe",
    "ca marche pas","ca fonctionne pas",
    "internet lent","internet coupe","pas d'internet","connexion lente",
    "debit faible","lenteur","connexion instable","internet ne marche pas",
    "internet marche pas","internet fonctionne pas",
    "appel coupe","ne peut pas appeler","appel impossible","coupure appel",
    "appels coupent",
    "surfacturation","debit abusif","erreur facturation","trop paye",
    "ne marche pas","ne fonctionne pas","marche pas","fonctionne pas",
    "probleme","panne","mauvais","impossible","inacceptable","scandaleux",
    "nul","terrible","arnaque","reclamation","plainte",
    "decu","insatisfait","perturbation","dysfonctionnement",
    "anomalie","defaillance","signal faible","en panne","coupe",
]

FR_NON_COMPLAINT_SIGNALS = [
    "merci","bonjour","bonsoir","bonne journee",
    "question","renseignement","information","comment activer",
    "comment faire","je voudrais savoir",
    "felicitations","bravo","excellent","parfait","satisfait",
    "content","super","bien","genial","top",
    "souhait","suggestion","comment passer","comment recharger",
]

# FIX-10: added "مقطوع" — covers مقطوعة and مقطوعه (ة→ه normalized)
AR_COMPLAINT_SIGNALS = [
    "لا شبكة","لا إشارة","لا تغطية","انقطاع الشبكة","شبكة مقطوعة",
    "إشارة ضعيفة","الشبكة واقعة","لا 4g","تغطية سيئة",
    "أنترنت بطيء","صبيب ضعيف","لا أنترنت","اتصال بطيء","أنترنت مقطوع",
    "مكالمة مقطوعة","لا أستطيع الاتصال","مشكلة صوت",
    "خصم مجحف","فاتورة خاطئة","رصيد غير صحيح",
    "مشكلة","عطل","انقطاع","لا يعمل","لا تعمل","معطل",
    "غير مقبول","فضيحة","سيء","رديء","كارثي","شكوى","شكوة",
    "تذمر","متضايق","غاضب","زعلان","لا أستطيع","تعطل",
    "خلل","بطء","تقطيع","مشكلتي","ابلغ عن",
    "مقطوع",   # FIX-10: stem covers مقطوعة/مقطوعه
    "منقطع",   # disconnected
    "شبكتي",   # "my network" — common complaint opener
]

AR_NON_COMPLAINT_SIGNALS = [
    "شكرا","شكراً","مرحبا","السلام","كيف الحال",
    "سؤال","استفسار","استعلام","أريد أن أعرف","كيف أفعل",
    "ممتاز","راضي","جيد","رائع","أحسنتم","شكرا جزيلا",
    "طلب معلومات","ما هو رقم","متى","كيف يمكنني",
    "كم سعر","ما هي الباقات","هل يمكن","أريد الاشتراك",
]

EN_COMPLAINT_SIGNALS = [
    "no network","no signal","no coverage","network down","network cut",
    "weak signal","no 4g","network drops","poor coverage",
    "slow internet","no internet","connection drops","unstable connection",
    "slow speed","buffering","internet not working",
    "call drops","cannot call","call failed","bad voice quality",
    "overcharged","wrong bill","incorrect balance","fraudulent charge",
    "problem","issue","not working","broken","complaint",
    "dissatisfied","unhappy","frustrated","impossible","bad",
    "terrible","unacceptable","outrageous","scam","fraud",
    "malfunction","outage","disruption","error","failed",
    "doesn't work","can't connect","keeps dropping",
]

EN_NON_COMPLAINT_SIGNALS = [
    "thank you","thanks","hello","hi","good morning","good evening",
    "question","inquiry","information","how do i","how can i",
    "how to activate","what is the price","can you explain",
    "i would like to know","when will","what are the packages",
    "excellent","great","wonderful","satisfied","happy","amazing",
    "well done","please advise","kindly inform",
]

_DEFAULT_CATEGORIES = {"Autre", "أخرى", "Other"}

# ══════════════════════════════════════════════════════════════════════════════
# PRE-NORMALIZED KEYWORD DICTIONARIES (FIX-1)
# ══════════════════════════════════════════════════════════════════════════════

_FR_CAT_NORM   = {k: [_norm(w) for w in v] for k, v in FR_CATEGORIES.items()}
_FR_SENT_NORM  = {k: [_norm(w) for w in v] for k, v in FR_SENTIMENT.items()}
_FR_URG_NORM   = [_norm(w) for w in FR_URGENCY]
_FR_COMP_NORM  = [_norm(w) for w in FR_COMPLAINT_SIGNALS]
_FR_NCOMP_NORM = [_norm(w) for w in FR_NON_COMPLAINT_SIGNALS]
_FR_STOPS_NORM = {_norm(w) for w in FR_STOPS}

_EN_CAT_NORM   = {k: [_norm(w) for w in v] for k, v in EN_CATEGORIES.items()}
_EN_SENT_NORM  = {k: [_norm(w) for w in v] for k, v in EN_SENTIMENT.items()}
_EN_URG_NORM   = [_norm(w) for w in EN_URGENCY]
_EN_COMP_NORM  = [_norm(w) for w in EN_COMPLAINT_SIGNALS]
_EN_NCOMP_NORM = [_norm(w) for w in EN_NON_COMPLAINT_SIGNALS]

_TN_CITIES_FR_NORM = {_norm(c) for c in TN_CITIES_FR}
_TN_CITIES_EN_NORM = {_norm(c) for c in TN_CITIES_EN}


# ══════════════════════════════════════════════════════════════════════════════
# WORD-BOUNDARY MATCH HELPER (FIX-7, FIX-9)
# ══════════════════════════════════════════════════════════════════════════════

def _kw_match(kw: str, text: str, arabic: bool = False) -> bool:
    """
    Safe keyword matching:
    - Arabic: plain substring (Arabic morphology has no ASCII word boundaries)
    - Multi-word expressions: substring
    - Single words (FR/EN): require \b word boundary to prevent
      "lent" matching inside "excellent", "nul" inside "Manuel", etc.
    """
    if arabic or " " in kw:
        return kw in text
    return bool(re.search(rf"\b{re.escape(kw)}\b", text))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE CLASS
# ══════════════════════════════════════════════════════════════════════════════

class MultilingualNLPPipeline:

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.stats = {
            "total_processed": 0,
            "languages":       {"ar": 0, "fr": 0, "en": 0},
            "categories":      {},
            "complaints":      0,
            "non_complaints":  0,
        }

    def analyze(self, text: str) -> dict:
        if not text or not isinstance(text, str) or len(text.strip()) < 3:
            return self._empty(text)

        lang         = detect_language(text)
        text_clean   = self._preprocess(text, lang)
        category     = self._category(text_clean, lang)
        sentiment    = self._sentiment(text_clean, lang)
        urgency      = self._urgency(text_clean, lang, sentiment)
        entities     = self._entities(text_clean, lang)
        keywords     = self._keywords(text_clean, lang)
        sentiment_fr = self._map_sentiment_to_french(sentiment, lang)
        is_complaint = self._is_complaint(
            text_clean, lang, sentiment, urgency["score"], category
        )

        self.stats["total_processed"] += 1
        self.stats["languages"][lang]   = self.stats["languages"].get(lang, 0) + 1
        self.stats["categories"][category] = self.stats["categories"].get(category, 0) + 1
        if is_complaint:
            self.stats["complaints"] += 1
        else:
            self.stats["non_complaints"] += 1

        return {
            "text":          text,
            "language":      lang,
            "category":      category,
            "sentiment":     sentiment_fr,
            "sentiment_raw": sentiment,
            "urgency_score": urgency["score"],
            "urgency_level": urgency["level"],
            "city":          entities.get("city"),
            "network_type":  entities.get("network_type"),
            "keywords":      keywords,
            "is_complaint":  is_complaint,
            "processed_at":  datetime.now().isoformat(),
        }

    # ── Preprocessing ──────────────────────────────────────────────────────

    def _preprocess(self, text: str, lang: str) -> str:
        if lang == "ar":
            text = re.sub(r"[إأآا]", "ا", text)
            text = re.sub(r"ى",      "ي", text)
            text = re.sub(r"ة",      "ه", text)
            text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
        else:
            text = text.lower()
            text = _strip_accents(text)
            text = re.sub(r"[^\w\s\-\'!]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    # ── Category (FIX-7) ──────────────────────────────────────────────────

    def _category(self, text: str, lang: str) -> str:
        lexicon  = AR_CATEGORIES if lang == "ar" else _EN_CAT_NORM if lang == "en" else _FR_CAT_NORM
        arabic   = lang == "ar"
        scores   = {}
        for cat, keywords in lexicon.items():
            scores[cat] = sum(1 for kw in keywords if _kw_match(kw, text, arabic))
        best = max(scores, key=scores.get)
        if scores[best] == 0:
            return {"ar": "أخرى", "en": "Other", "fr": "Autre"}[lang]
        return best

    # ── Sentiment (FIX-9) ─────────────────────────────────────────────────

    def _sentiment(self, text: str, lang: str) -> str:
        lexicon = AR_SENTIMENT if lang == "ar" else _EN_SENT_NORM if lang == "en" else _FR_SENT_NORM
        arabic  = lang == "ar"
        scores  = {
            s: sum(1 for kw in kws if _kw_match(kw, text, arabic))
            for s, kws in lexicon.items()
        }
        for k in lexicon:
            if scores.get(k, 0) > 0:
                return k
        return {"ar": "محايد", "en": "neutral", "fr": "neutre"}[lang]

    # ── Sentiment mapping ─────────────────────────────────────────────────

    def _map_sentiment_to_french(self, sentiment: str, lang: str) -> str:
        if lang == "ar":
            return {"حرج": "critique","سلبي": "négatif","إيجابي": "positif","محايد": "neutre"}.get(sentiment, "neutre")
        if lang == "en":
            return {"critical": "critique","negative": "négatif","positive": "positif","neutral": "neutre"}.get(sentiment, "neutre")
        return sentiment

    # ── is_complaint (FIX-5) ──────────────────────────────────────────────

    def _is_complaint(self, text: str, lang: str, sentiment: str, urgency_score: float, category: str) -> bool:
        clf = _load_trained_classifier()
        if clf is not None:
            return bool(clf.predict([text])[0])

        if sentiment in {"critique","négatif","حرج","سلبي","critical","negative"}:
            return True

        if lang == "ar":
            complaint_kws, non_complaint_kws = AR_COMPLAINT_SIGNALS, AR_NON_COMPLAINT_SIGNALS
        elif lang == "en":
            complaint_kws, non_complaint_kws = _EN_COMP_NORM, _EN_NCOMP_NORM
        else:
            complaint_kws, non_complaint_kws = _FR_COMP_NORM, _FR_NCOMP_NORM

        c_score  = sum(1 for kw in complaint_kws     if kw in text)
        nc_score = sum(1 for kw in non_complaint_kws if kw in text)

        if c_score > 0 and c_score >= nc_score:
            return True
        if nc_score > 0 and c_score == 0:
            return False
        if urgency_score >= 0.5:
            return True
        if category not in _DEFAULT_CATEGORIES:
            return True
        return False

    # ── Urgency ───────────────────────────────────────────────────────────

    def _urgency(self, text: str, lang: str, sentiment: str) -> dict:
        base = {
            "critique":0.7,"négatif":0.4,"neutre":0.2,"positif":0.1,
            "حرج":0.7,"سلبي":0.4,"محايد":0.2,"إيجابي":0.1,
            "critical":0.7,"negative":0.4,"neutral":0.2,"positive":0.1,
        }
        score   = base.get(sentiment, 0.2)
        urg_kws = AR_URGENCY if lang == "ar" else _EN_URG_NORM if lang == "en" else _FR_URG_NORM
        for kw in urg_kws:
            if kw in text:
                score = min(score + 0.25, 1.0)
                break
        patterns = {"ar": r"(\d+)\s*(?:أيام|يوم|ايام)", "en": r"(\d+)\s*(?:days?|day)", "fr": r"(\d+)\s*(?:jours?|jour)"}
        days = re.findall(patterns[lang], text)
        if days:
            score = min(score + int(days[0]) * 0.05, 1.0)
        if "!" in text:
            score = min(score + 0.1, 1.0)
        score = round(score, 2)
        return {"score": score, "level": "très urgent" if score >= 0.8 else "urgent" if score >= 0.5 else "normal"}

    # ── Entities ──────────────────────────────────────────────────────────

    def _entities(self, text: str, lang: str) -> dict:
        entities = {"city": None, "network_type": None}
        for nt in ["5g","4g","3g","2g","volte","lte","wifi","fibre"]:
            if nt in text:
                entities["network_type"] = nt.upper()
                break
        city_set = AR_CITIES if lang == "ar" else _TN_CITIES_EN_NORM if lang == "en" else _TN_CITIES_FR_NORM
        for city in city_set:
            if city in text:
                entities["city"] = city if lang == "ar" else city.title()
                break
        return entities

    # ── Keywords ──────────────────────────────────────────────────────────

    def _keywords(self, text: str, lang: str, top_n: int = 5) -> list:
        stops   = AR_STOPS if lang == "ar" else EN_STOPS if lang == "en" else _FR_STOPS_NORM
        min_len = 3 if lang == "ar" else 4
        words   = [w for w in re.findall(r"\b\w+\b", text) if len(w) >= min_len and w not in stops]
        freq: dict = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        return [w for w, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n]]

    # ── Batch ─────────────────────────────────────────────────────────────

    def analyze_batch(self, texts: list, show_progress: bool = True) -> list:
        results = []
        for i, text in enumerate(texts):
            if show_progress and self.verbose and i % 100 == 0:
                print(f"Processing {i+1}/{len(texts)}…")
            results.append(self.analyze(text))
        return results

    def analyze_dataframe(self, df, text_column: str = "complaint_text", add_columns: bool = True):
        texts   = df[text_column].fillna("").astype(str).tolist()
        results = self.analyze_batch(texts)
        if not add_columns:
            return results
        out = df.copy()
        for key in ["language","category","sentiment","urgency_score","urgency_level","city","network_type","is_complaint"]:
            out[f"nlp_{key}"] = [r[key] for r in results]
        out["nlp_keywords"] = [", ".join(r["keywords"]) for r in results]
        return out

    def get_stats(self) -> dict:
        return self.stats

    def reset_stats(self) -> None:
        self.stats = {"total_processed":0,"languages":{"ar":0,"fr":0,"en":0},"categories":{},"complaints":0,"non_complaints":0}

    def _empty(self, text) -> dict:
        return {"text":text,"language":"fr","category":"Autre","sentiment":"neutre","sentiment_raw":"neutral",
                "urgency_score":0.0,"urgency_level":"normal","city":None,"network_type":None,"keywords":[],
                "is_complaint":False,"processed_at":datetime.now().isoformat()}


# ══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def quick_analyze(text: str) -> dict:
    return MultilingualNLPPipeline().analyze(text)

def batch_analyze(texts: list, show_progress: bool = True) -> list:
    return MultilingualNLPPipeline(verbose=show_progress).analyze_batch(texts)

def enrich_dataframe(df, text_column: str = "complaint_text"):
    return MultilingualNLPPipeline().analyze_dataframe(df, text_column=text_column)


# ══════════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pipe  = MultilingualNLPPipeline(verbose=True)
    model = _load_trained_classifier()
    print(f"Classifier: {'YES' if model else 'NO (rule-based fallback)'}\n")

    cases = [
        ("mon reseau coupe !!",                       True,  "FR complaint"),
        ("pas de 4g depuis hier soir",                True,  "FR complaint"),
        ("شبكتي مقطوعة في تونس",                      True,  "AR complaint"),
        ("شبكتي مقطوعة في تونس منذ 3 أيام",           True,  "AR complaint + duration"),
        ("My network keeps dropping in Tunis",         True,  "EN complaint"),
        ("ca marche pas !!",                           True,  "FR complaint"),
        ("merci pour votre service, tout fonctionne.", False, "FR feedback"),
        ("merci pour votre excellent service",         False, "FR feedback FIX-9"),
        ("Comment activer le roaming international ?", False, "FR question"),
        ("Thank you, service has improved!",           False, "EN feedback FIX-11"),
        ("شكرا جزيلا، الخدمة ممتازة",                  False, "AR feedback"),
    ]

    passed = 0
    print(f"  {'Text':<50} {'Exp':>11}  {'Got':>11}  Note")
    print("  " + "─" * 88)
    for text, expected, note in cases:
        r  = pipe.analyze(text)
        ok = r["is_complaint"] == expected
        lbl = lambda v: "RECLAMATION" if v else "FEEDBACK   "
        print(f"  {'✓' if ok else '✗'} {text[:48]:<50} {lbl(expected)}  {lbl(r['is_complaint'])}  {note}")
        if ok:
            passed += 1
    print(f"\n  {passed}/{len(cases)} passed")