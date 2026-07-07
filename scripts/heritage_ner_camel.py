# -*- coding: utf-8 -*-
"""
heritage_ner_camel.py

نظام تصنيف كيانات مسمّى (NER) هجين للتراث الفلسطيني:
  1) CAMeL Tools (ner-arabert) -> يكتشف الكيانات العامة (أشخاص PERS، أماكن LOC، منظمات ORG)
  2) قاموس تراثي مبني يدويًا      -> يكتشف الكيانات التراثية المتخصصة:
       PAL_FOOD     -> أكلات شعبية
       PAL_CLOTHES  -> ملابس وأزياء تراثية
       PAL_HERITAGE -> عادات / فنون / حرف / مناسبات

طريقة التشغيل:
    uv run python heritage_ner_camel.py
"""

from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.ner import NERecognizer


# =========================================================
# 1) القاموس التراثي (نفس فكرة الملف السابق، لكن كـ dict بسيط)
# =========================================================
# المفتاح: الشكل بدون "ال" -- القيمة: التصنيف
# بنولّد تلقائيًا نسخة بـ "ال" من كل كلمة، فمش محتاجين نكررها يدويًا.

HERITAGE_DICTIONARY = {
    # ---------- PAL_FOOD ----------
    "مسخن": "PAL_FOOD",
    "مقلوبة": "PAL_FOOD",
    "كنافة": "PAL_FOOD",
    "فتوش": "PAL_FOOD",
    "معمول": "PAL_FOOD",
    "قطايف": "PAL_FOOD",
    "مفتول": "PAL_FOOD",
    "ملوخية": "PAL_FOOD",
    "مسقعة": "PAL_FOOD",
    "فريكة": "PAL_FOOD",
    "شنكليش": "PAL_FOOD",
    "زعتر": "PAL_FOOD",
    "طابون": "PAL_FOOD",
    "لبنة": "PAL_FOOD",
    "منسف": "PAL_FOOD",
    "سماقية": "PAL_FOOD",
    "علاليش": "PAL_FOOD",

    # ---------- PAL_CLOTHES ----------
    "كوفية": "PAL_CLOTHES",
    "حطة": "PAL_CLOTHES",
    "شماغ": "PAL_CLOTHES",
    "عباية": "PAL_CLOTHES",
    "طاقية": "PAL_CLOTHES",
    "شرش": "PAL_CLOTHES",
    "زنار": "PAL_CLOTHES",
    "قنباز": "PAL_CLOTHES",
    "برنس": "PAL_CLOTHES",
    "شملة": "PAL_CLOTHES",
    "عقال": "PAL_CLOTHES",

    # ---------- PAL_HERITAGE ----------
    "دبكة": "PAL_HERITAGE",
    "تطريز": "PAL_HERITAGE",
    "حكواتي": "PAL_HERITAGE",
    "سامر": "PAL_HERITAGE",
    "زغرودة": "PAL_HERITAGE",
    "زغاريد": "PAL_HERITAGE",
    "زفة": "PAL_HERITAGE",
    "حنة": "PAL_HERITAGE",
    "تعليلة": "PAL_HERITAGE",
    "مهر": "PAL_HERITAGE",
    "أرجيلة": "PAL_HERITAGE",
    "نول": "PAL_HERITAGE",
    "فخار": "PAL_HERITAGE",
}


def build_full_dictionary(base_dict: dict) -> dict:
    """
    بيولّد نسخة كل كلمة بـ "ال" التعريف تلقائيًا،
    عشان "مسخن" و"المسخن" الاتنين يتلاقوا.
    """
    full_dict = {}
    for word, label in base_dict.items():
        full_dict[word] = label
        full_dict["ال" + word] = label
    return full_dict


FULL_DICTIONARY = build_full_dictionary(HERITAGE_DICTIONARY)


# =========================================================
# 2 - ب) تنضيف الكلمة من حروف العطف/الجر الملزوقة بأولها
# =========================================================
# لازم نشيل البادئات الأطول الأول (زي "بال" قبل "ب") عشان الشيل يبقى صح

PREFIXES_TO_STRIP = [
    "وبال", "فبال", "كبال",   # و/ف/ك + ب + ال
    "وال", "فال", "بال", "كال",  # حرف عطف/جر + ال
    "و", "ف", "ب", "ل", "ك",   # حرف عطف/جر لوحده
]


def normalize_token(token: str) -> str:
    """
    بتشيل بادئات شائعة (و، ب، ف، ل، ك، بال، وال...) من أول الكلمة
    عشان "والمقلوبة" تترجع لـ "المقلوبة" أو "مقلوبة" وتتطابق مع القاموس.
    بترجع أول نسخة تطابق كلمة موجودة فعلاً في القاموس، أو الكلمة الأصلية لو مفيش تطابق.
    """
    if token in FULL_DICTIONARY:
        return token

    for prefix in PREFIXES_TO_STRIP:
        if token.startswith(prefix) and len(token) > len(prefix) + 1:
            stripped = token[len(prefix):]
            if stripped in FULL_DICTIONARY:
                return stripped

    return token


# =========================================================
# 2) تحميل موديل CAMeL NER (مرة واحدة بس عند بدء البرنامج)
# =========================================================

def load_camel_ner():
    ner = NERecognizer.pretrained()
    return ner


# =========================================================
# 3) دالة استخراج الكيانات الهجينة (Hybrid)
# =========================================================

def extract_hybrid_entities(text: str, ner_model) -> list:
    """
    بتاخد نص، وترجع قائمة بكل كيان بالشكل:
        (الكلمة, التصنيف, المصدر)
    المصدر يكون "heritage_dict" أو "camel_ner"
    """
    tokens = simple_word_tokenize(text)
    results = []

    # --- الخطوة أ: نمرّ على كل كلمة، نشوف هل هي (أو نسختها المنضّفة) في القاموس ---
    heritage_hit_indices = set()
    for i, token in enumerate(tokens):
        normalized = normalize_token(token)
        if normalized in FULL_DICTIONARY:
            label = FULL_DICTIONARY[normalized]
            # بنطبع الكلمة الأصلية زي ما جت في النص، مش النسخة المنضّفة
            results.append((token, label, "heritage_dict"))
            heritage_hit_indices.add(i)

    # --- الخطوة ب: نشغّل CAMeL NER على الجملة كاملة ---
    ner_labels = ner_model.predict_sentence(tokens)

    # --- الخطوة ج: نجمع الكيانات المتتالية (B-XXX يتبعها I-XXX) ---
    current_entity_tokens = []
    current_label = None

    for i, (token, tag) in enumerate(zip(tokens, ner_labels)):
        # لو الكلمة دي أصلاً اتصنّفت من القاموس التراثي، نتجاهل تصنيف CAMeL ليها
        # عشان القاموس التراثي بياخد الأولوية (زي فكرة EntityRuler قبل ner في spaCy)
        if i in heritage_hit_indices:
            if current_entity_tokens:
                results.append((
                    " ".join(current_entity_tokens),
                    current_label,
                    "camel_ner",
                ))
                current_entity_tokens = []
                current_label = None
            continue

        if tag.startswith("B-"):
            if current_entity_tokens:
                results.append((
                    " ".join(current_entity_tokens),
                    current_label,
                    "camel_ner",
                ))
            current_entity_tokens = [token]
            current_label = tag[2:]  # PERS, LOC, ORG...

        elif tag.startswith("I-") and current_entity_tokens:
            current_entity_tokens.append(token)

        else:  # tag == "O"
            if current_entity_tokens:
                results.append((
                    " ".join(current_entity_tokens),
                    current_label,
                    "camel_ner",
                ))
                current_entity_tokens = []
                current_label = None

    # لو الجملة خلصت والكيان لسه مفتوح
    if current_entity_tokens:
        results.append((
            " ".join(current_entity_tokens),
            current_label,
            "camel_ner",
        ))

    return results


# =========================================================
# 4) طباعة النتائج بشكل منظم
# =========================================================

def print_results(text: str, entities: list):
    print(f"\nالنص: {text}")
    if not entities:
        print("  (لا توجد كيانات مكتشفة)")
        return
    for ent_text, label, source in entities:
        print(f"  {ent_text:<20} | {label:<15} | المصدر: {source}")


# =========================================================
# 5) تجربة على جمل عينة
# =========================================================

if __name__ == "__main__":
    print("جاري تحميل موديل CAMeL NER...")
    ner_model = load_camel_ner()
    print("تم تحميل الموديل بنجاح.\n")

    sample_sentences = [
        "لبست جدتي الثوب الفلسطيني المطرز في يوم العرس في نابلس.",
        "أكل محمد المسخن والمقلوبة في بيت أبو علي بمدينة الخليل.",
        "رقص الشباب الدبكة على أنغام الأغاني الشعبية في السامر بقرية سلواد.",
        "اشترت سارة كوفية وشماغ من سوق البلدة القديمة في القدس.",
        "تشتهر مدينة نابلس بالكنافة والصابون النابلسي.",
        "غنّت النساء الزغاريد فرحًا بموسم قطف الزيتون في رام الله.",
    ]

    print("=== نتائج النظام الهجين (CAMeL NER + قاموس التراث) ===")
    for sent in sample_sentences:
        entities = extract_hybrid_entities(sent, ner_model)
        print_results(sent, entities)