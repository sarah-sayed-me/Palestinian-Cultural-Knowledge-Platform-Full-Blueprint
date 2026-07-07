# -*- coding: utf-8 -*-
"""
run_ner_on_corpus.py

بيقرأ ملف data/train_corpus.json (قائمة مستندات)، ولكل مستند:
  1) بيقسم حقل "text" إلى جمل
  2) بيشغل عليه النظام الهجين (CAMeL NER + قاموس التراث الفلسطيني)
  3) بيحط النتيجة في حقل "entities" بتاع نفس المستند

وبيحفظ النتيجة في ملف جديد: data/train_corpus_with_entities.json
(عشان الملف الأصلي يفضل سليم لو حصل أي خطأ)

طريقة التشغيل:
    uv run python run_ner_on_corpus.py
"""

import json
import re

from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.ner import NERecognizer


# =========================================================
# 1) القاموس التراثي (نفس القاموس من heritage_ner_camel.py)
# =========================================================

HERITAGE_DICTIONARY = {
    # ---------- PAL_FOOD ----------
    "مسخن": "PAL_FOOD", "مقلوبة": "PAL_FOOD", "كنافة": "PAL_FOOD",
    "فتوش": "PAL_FOOD", "معمول": "PAL_FOOD", "قطايف": "PAL_FOOD",
    "مفتول": "PAL_FOOD", "ملوخية": "PAL_FOOD", "مسقعة": "PAL_FOOD",
    "فريكة": "PAL_FOOD", "شنكليش": "PAL_FOOD", "زعتر": "PAL_FOOD",
    "طابون": "PAL_FOOD", "لبنة": "PAL_FOOD", "منسف": "PAL_FOOD",
    "سماقية": "PAL_FOOD", "علاليش": "PAL_FOOD",

    # ---------- PAL_CLOTHES ----------
    "كوفية": "PAL_CLOTHES", "حطة": "PAL_CLOTHES", "شماغ": "PAL_CLOTHES",
    "عباية": "PAL_CLOTHES", "طاقية": "PAL_CLOTHES", "شرش": "PAL_CLOTHES",
    "زنار": "PAL_CLOTHES", "قنباز": "PAL_CLOTHES", "برنس": "PAL_CLOTHES",
    "شملة": "PAL_CLOTHES", "عقال": "PAL_CLOTHES",

    # ---------- PAL_HERITAGE ----------
    "دبكة": "PAL_HERITAGE", "تطريز": "PAL_HERITAGE", "حكواتي": "PAL_HERITAGE",
    "سامر": "PAL_HERITAGE", "زغرودة": "PAL_HERITAGE", "زغاريد": "PAL_HERITAGE",
    "زفة": "PAL_HERITAGE", "حنة": "PAL_HERITAGE", "تعليلة": "PAL_HERITAGE",
    "مهر": "PAL_HERITAGE", "أرجيلة": "PAL_HERITAGE", "نول": "PAL_HERITAGE",
    "فخار": "PAL_HERITAGE",
}


def build_full_dictionary(base_dict: dict) -> dict:
    full_dict = {}
    for word, label in base_dict.items():
        full_dict[word] = label
        full_dict["ال" + word] = label
    return full_dict


FULL_DICTIONARY = build_full_dictionary(HERITAGE_DICTIONARY)

PREFIXES_TO_STRIP = [
    "وبال", "فبال", "كبال",
    "وال", "فال", "بال", "كال",
    "و", "ف", "ب", "ل", "ك",
]


def normalize_token(token: str) -> str:
    if token in FULL_DICTIONARY:
        return token
    for prefix in PREFIXES_TO_STRIP:
        if token.startswith(prefix) and len(token) > len(prefix) + 1:
            stripped = token[len(prefix):]
            if stripped in FULL_DICTIONARY:
                return stripped
    return token


def strip_leading_conjunction(token: str) -> str:
    """
    تنضيف عام (مش مرتبط بالقاموس التراثي): بيشيل حرف "الواو" بس
    من أول أي كلمة، زي "والاردن" -> "الاردن"، "واسرائيل" -> "اسرائيل".
    بنقتصر على "الواو" لوحدها بس (مش ف/ب/ل/ك) عشان دول ممكن يكونوا
    جزء أصلي من الكلمة نفسها (زي "فلسطين"، "لبنان")، فشيلهم غلط.
    """
    if token.startswith("و") and len(token) > 2:
        return token[1:]
    return token


# =========================================================
# 2) تقسيم النص الطويل إلى جمل
# =========================================================

def split_into_sentences(text: str) -> list:
    """
    تقسيم بسيط للنص العربي إلى جمل بناءً على علامات الترقيم والأسطر الجديدة.
    """
    # نستبدل فواصل الأسطر بنقطة عشان نضمن فصل الفقرات كجمل منفصلة
    text = text.replace("\n", ". ")
    # نقسم على النقطة والفاصلة المنقوطة وعلامة الاستفهام والتعجب
    sentences = re.split(r"[.!؟؛]+", text)
    # نشيل الجمل الفاضية أو القصيرة جدًا (أقل من 3 كلمات)
    sentences = [s.strip() for s in sentences if len(s.strip().split()) >= 3]
    return sentences


# =========================================================
# 3) دالة الاستخراج الهجين (نفس منطق heritage_ner_camel.py)
# =========================================================

def extract_hybrid_entities(text: str, ner_model) -> list:
    tokens = simple_word_tokenize(text)
    if not tokens:
        return []

    results = []
    heritage_hit_indices = set()

    for i, token in enumerate(tokens):
        normalized = normalize_token(token)
        if normalized in FULL_DICTIONARY:
            label = FULL_DICTIONARY[normalized]
            results.append({"text": token, "label": label, "source": "heritage_dict"})
            heritage_hit_indices.add(i)

    ner_labels = ner_model.predict_sentence(tokens)

    current_entity_tokens = []
    current_label = None

    for i, (token, tag) in enumerate(zip(tokens, ner_labels)):
        if i in heritage_hit_indices:
            if current_entity_tokens:
                results.append({
                    "text": " ".join(current_entity_tokens),
                    "label": current_label,
                    "source": "camel_ner",
                })
                current_entity_tokens = []
                current_label = None
            continue

        if tag.startswith("B-"):
            if current_entity_tokens:
                cleaned = strip_leading_conjunction(current_entity_tokens[0])
                current_entity_tokens[0] = cleaned
                results.append({
                    "text": " ".join(current_entity_tokens),
                    "label": current_label,
                    "source": "camel_ner",
                })
            current_entity_tokens = [token]
            current_label = tag[2:]

        elif tag.startswith("I-") and current_entity_tokens:
            current_entity_tokens.append(token)

        else:
            if current_entity_tokens:
                cleaned = strip_leading_conjunction(current_entity_tokens[0])
                current_entity_tokens[0] = cleaned
                results.append({
                    "text": " ".join(current_entity_tokens),
                    "label": current_label,
                    "source": "camel_ner",
                })
                current_entity_tokens = []
                current_label = None

    if current_entity_tokens:
        cleaned = strip_leading_conjunction(current_entity_tokens[0])
        current_entity_tokens[0] = cleaned
        results.append({
            "text": " ".join(current_entity_tokens),
            "label": current_label,
            "source": "camel_ner",
        })

    return results


# =========================================================
# 4) معالجة مستند واحد كامل (كل جمله)
# =========================================================

def process_document(doc: dict, ner_model) -> list:
    text = doc.get("text", "") or ""
    if not text.strip():
        return []

    sentences = split_into_sentences(text)
    all_entities = []

    for sentence in sentences:
        entities = extract_hybrid_entities(sentence, ner_model)
        all_entities.extend(entities)

    return all_entities


# =========================================================
# 5) البرنامج الرئيسي
# =========================================================

if __name__ == "__main__":
    INPUT_PATH = "data/train_corpus.json"
    OUTPUT_PATH = "data/train_corpus_with_entities.json"

    # عدد المستندات المطلوب معالجتها في هذه التجربة (None = الكل)
    # نبدأ بعدد محدود للتجربة الأولى عشان نتأكد إن كل حاجة شغالة صح
    LIMIT = 3

    print("جاري تحميل موديل CAMeL NER...")
    ner_model = NERecognizer.pretrained()
    print("تم تحميل الموديل بنجاح.\n")

    print(f"جاري تحميل الملف: {INPUT_PATH}")
    with open(INPUT_PATH, encoding="utf-8") as f:
        corpus = json.load(f)

    print(f"عدد المستندات في الملف: {len(corpus)}")

    documents_to_process = corpus if LIMIT is None else corpus[:LIMIT]

    for idx, doc in enumerate(documents_to_process):
        title = doc.get("source_url", f"مستند رقم {idx}")
        print(f"\n[{idx + 1}/{len(documents_to_process)}] جاري معالجة: {title}")

        entities = process_document(doc, ner_model)
        doc["entities"] = entities

        print(f"  تم استخراج {len(entities)} كيان.")

        # نطبع أول 15 كيان بس كعينة للمراجعة السريعة
        for ent in entities[:15]:
            print(f"    {ent['text']:<20} | {ent['label']:<12} | {ent['source']}")
        if len(entities) > 15:
            print(f"    ... و {len(entities) - 15} كيان إضافي")

    print(f"\nجاري حفظ النتيجة في: {OUTPUT_PATH}")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    print("تم الحفظ بنجاح.")