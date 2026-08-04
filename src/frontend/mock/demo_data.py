"""Mock / Demo data for the Palestinian Cultural Knowledge Platform.

Not used by default — services/backend.py wires the real dashboard to
data_loaders.py / graph_store.py / the Track F reports / the RAG pipeline
instead. Kept here for offline demoing or as a schema reference for what
each services.backend.get_*_data() function must return.
"""

import random
import json

# ============================================================
# OVERVIEW DATA
# ============================================================

def get_overview_data():
    """Return overview KPI metrics and distribution data."""
    return {
        "kpis": {
            "total_documents": 882,
            "sources": 5,
            "kg_entities": 13063,
            "kg_relations": 585,
        },
        "documents_by_source": [
            {"source": "ويكيبيديا العربية", "count": 581},
            {"source": "وكالة وفا", "count": 106},
            {"source": "ويكيبيديا الإنجليزية", "count": 97},
            {"source": "GDELT", "count": 73},
            {"source": "Semantic Scholar", "count": 25},
        ],
        "documents_by_language": [
            {"language": "العربية", "count": 687},
            {"language": "الإنجليزية", "count": 195},
        ],
        "topic_distribution": [
            {"topic": "التراث والثقافة", "count": 245},
            {"topic": "المدن والتاريخ", "count": 189},
            {"topic": "الأدب والشعر", "count": 132},
            {"topic": "المأكولات", "count": 98},
            {"topic": "السياسة والنزاع", "count": 87},
            {"topic": "الفنون والحرف", "count": 76},
            {"topic": "الدين والمقدسات", "count": 55},
        ],
        "kg_entity_types": [
            {"type": "PERSON", "label": "أشخاص", "count": 2840},
            {"type": "LOCATION", "label": "أماكن", "count": 3120},
            {"type": "ORGANIZATION", "label": "منظمات", "count": 1650},
            {"type": "HERITAGE_FOOD", "label": "تراث غذائي", "count": 420},
            {"type": "HERITAGE_CRAFT", "label": "حرف تراثية", "count": 310},
            {"type": "HERITAGE_PLACE", "label": "أماكن تراثية", "count": 280},
            {"type": "MISC", "label": "أخرى", "count": 4443},
        ],
    }


# ============================================================
# TOPIC MAP DATA
# ============================================================

def get_topic_data():
    """Return topic modeling data for scatter/cluster visualization."""
    random.seed(42)
    topics = [
        {"id": 0, "name": "التراث والثقافة الشعبية", "color": "#6B8E23"},
        {"id": 1, "name": "المدن الفلسطينية التاريخية", "color": "#009736"},
        {"id": 2, "name": "الأدب والشعر الفلسطيني", "color": "#C5A55A"},
        {"id": 3, "name": "المأكولات والمطبخ", "color": "#CE1126"},
        {"id": 4, "name": "السياسة والنزاع", "color": "#1a1a1a"},
        {"id": 5, "name": "الفنون والحرف اليدوية", "color": "#556B2F"},
        {"id": 6, "name": "الدين والمقدسات الإسلامية", "color": "#8B8580"},
    ]

    # Generate scatter points for each topic
    documents = []
    for topic in topics:
        n_docs = random.randint(25, 90)
        cx = random.uniform(-3, 3)
        cy = random.uniform(-3, 3)
        for _ in range(n_docs):
            x = cx + random.gauss(0, 0.8)
            y = cy + random.gauss(0, 0.8)
            lang = random.choice(["العربية", "العربية", "العربية", "الإنجليزية"])
            source = random.choice(
                ["ويكيبيديا العربية", "وكالة وفا", "ويكيبيديا الإنجليزية", "GDELT", "Semantic Scholar"]
            )
            decade = random.choice(["1800", "1850", "1900", "1920", "1940", "1960", "1980", "2000", "2020"])
            documents.append({
                "x": x, "y": y,
                "topic_id": topic["id"],
                "topic_name": topic["name"],
                "color": topic["color"],
                "title": f"وثيقة نموذجية - {topic['name']}",
                "language": lang,
                "source": source,
                "decade": decade,
            })

    # Topic details
    topic_details = {
        0: {
            "name": "التراث والثقافة الشعبية",
            "doc_count": 245,
            "top_words": ["التطريز", "التراث", "الفولكلور", "الأزياء", "العادات", "المهرجانات", "الأغاني"],
            "example_docs": [
                "التطريز الفلسطيني هو أحد أبرز عناصر التراث الثقافي الفلسطيني...",
                "تتميز كل مدينة فلسطينية بزخارف تطريزية خاصة بها تدل على هويتها...",
                "يعد الفولكلور الفلسطيني جزءًا لا يتجزأ من الهوية الوطنية...",
            ],
        },
        1: {
            "name": "المدن الفلسطينية التاريخية",
            "doc_count": 189,
            "top_words": ["القدس", "يافا", "حيفا", "نابلس", "الخليل", "غزة", "بيت لحم", "رام الله"],
            "example_docs": [
                "تعتبر القدس من أقدم المدن المأهولة في العالم...",
                "كانت يافا تُعرف بـ 'عروس البحر' ل موقعها على ساحل البحر الأبيض المتوسط...",
                "نابلس مدينة كنعانية عريقة تُعرف بجبالها الخضراء...",
            ],
        },
        2: {
            "name": "الأدب والشعر الفلسطيني",
            "doc_count": 132,
            "top_words": ["محمود درويش", "الشعر", "القصة", "الرواية", "المنفى", "الحنين", "الهوية"],
            "example_docs": [
                "يُعد محمود درويش أهم شاعر فلسطيني معاصر...",
                "تناول الأدب الفلسطيني موضوعات الهوية والمنفى والعودة...",
                "ظهر ما يُعرف بأدب المقاومة بعد النكبة عام 1948...",
            ],
        },
        3: {
            "name": "المأكولات والمطبخ",
            "doc_count": 98,
            "top_words": ["الكنافة", "المقلوبة", "الزيتون", "الزعتر", "الفلافل", "المفتول", "المنسف"],
            "example_docs": [
                "الكنافة النابلسية من أشهر الحلويات الفلسطينية...",
                "يعد الزيتون الفلسطيني رمزًا وطنيًا وثقافيًا...",
                "المقلوبة طبق تقليدي فلسطيني يُقدم في المناسبات...",
            ],
        },
        4: {
            "name": "السياسة والنزاع",
            "doc_count": 87,
            "top_words": ["فلسطين", "الاحتلال", "النكبة", "العودة", "السلام", "المفاوضات", "الضفة"],
            "example_docs": [
                "تُشير أحداث النكبة عام 1948 إلى التهجير الجماعي...",
                "تمثل قضية اللاجئين الفلسطينيين أحد أطول أزمات اللاجئين...",
                "شهدت فلسطين تقلبات سياسية كبرى عبر القرن العشرين...",
            ],
        },
        5: {
            "name": "الفنون والحرف اليدوية",
            "doc_count": 76,
            "top_words": ["الخزف", "النسيج", "الحفر", "الزخرفة", "الفسيفساء", "الخط العربي"],
            "example_docs": [
                "يتميز الخزف الفلسطيني بزخارفه الزرقاء والبيضاء...",
                "تعد صناعة الفسيفساء من الحرف العريقة في فلسطين...",
                "يشتهر الخط العربي الفلسطيني بجماله ودقته...",
            ],
        },
        6: {
            "name": "الدين والمقدسات الإسلامية",
            "doc_count": 55,
            "top_words": ["المسجد الأقصى", "القدس", "الحرم", "الإسلام", "المسيحية", "كنيسة المهد"],
            "example_docs": [
                "يُعد المسجد الأقصى ثالث أقدس المساجد في الإسلام...",
                "تضم فلسطين أماكن مقدسة لدى الديانات السماوية الثلاث...",
                "كنيسة المهد في بيت لحم من أهم المواقع المسيحية...",
            ],
        },
    }

    return {
        "topics": topics,
        "documents": documents,
        "topic_details": topic_details,
        "filters": {
            "sources": ["الكل", "ويكيبيديا العربية", "وكالة وفا", "ويكيبيديا الإنجليزية", "GDELT", "Semantic Scholar"],
            "languages": ["الكل", "العربية", "الإنجليزية"],
            "decades": ["الكل", "1800", "1850", "1900", "1920", "1940", "1960", "1980", "2000", "2020"],
        },
    }


# ============================================================
# TIMELINE DATA
# ============================================================

def get_timeline_data():
    """Return temporal analysis data."""
    return {
        "docs_by_decade": [
            {"decade": "1800", "count": 5},
            {"decade": "1810", "count": 3},
            {"decade": "1820", "count": 2},
            {"decade": "1830", "count": 4},
            {"decade": "1840", "count": 6},
            {"decade": "1850", "count": 12},
            {"decade": "1860", "count": 8},
            {"decade": "1870", "count": 15},
            {"decade": "1880", "count": 18},
            {"decade": "1890", "count": 22},
            {"decade": "1900", "count": 35},
            {"decade": "1910", "count": 42},
            {"decade": "1920", "count": 58},
            {"decade": "1930", "count": 65},
            {"decade": "1940", "count": 120},
            {"decade": "1950", "count": 85},
            {"decade": "1960", "count": 72},
            {"decade": "1970", "count": 68},
            {"decade": "1980", "count": 75},
            {"decade": "1990", "count": 82},
            {"decade": "2000", "count": 110},
            {"decade": "2010", "count": 145},
            {"decade": "2020", "count": 132},
        ],
        "available_terms": [
            "فلسطين", "الثقافة", "القدس", "التراث", "النكبة",
            "الزيتون", "التطريز", "الشعر", "المنفى", "المقاومة"
        ],
        "term_frequencies": {
            "فلسطين": [
                {"year": 1800, "freq": 2}, {"year": 1850, "freq": 8}, {"year": 1900, "freq": 25},
                {"year": 1920, "freq": 55}, {"year": 1940, "freq": 120}, {"year": 1950, "freq": 85},
                {"year": 1970, "freq": 60}, {"year": 1990, "freq": 78}, {"year": 2000, "freq": 110},
                {"year": 2020, "freq": 130},
            ],
            "الثقافة": [
                {"year": 1800, "freq": 1}, {"year": 1850, "freq": 3}, {"year": 1900, "freq": 10},
                {"year": 1920, "freq": 18}, {"year": 1940, "freq": 35}, {"year": 1950, "freq": 40},
                {"year": 1970, "freq": 45}, {"year": 1990, "freq": 55}, {"year": 2000, "freq": 70},
                {"year": 2020, "freq": 85},
            ],
            "القدس": [
                {"year": 1800, "freq": 5}, {"year": 1850, "freq": 12}, {"year": 1900, "freq": 30},
                {"year": 1920, "freq": 60}, {"year": 1940, "freq": 95}, {"year": 1950, "freq": 70},
                {"year": 1970, "freq": 55}, {"year": 1990, "freq": 65}, {"year": 2000, "freq": 90},
                {"year": 2020, "freq": 105},
            ],
            "التراث": [
                {"year": 1800, "freq": 0}, {"year": 1850, "freq": 2}, {"year": 1900, "freq": 8},
                {"year": 1920, "freq": 15}, {"year": 1940, "freq": 25}, {"year": 1950, "freq": 30},
                {"year": 1970, "freq": 35}, {"year": 1990, "freq": 50}, {"year": 2000, "freq": 65},
                {"year": 2020, "freq": 80},
            ],
            "النكبة": [
                {"year": 1800, "freq": 0}, {"year": 1850, "freq": 0}, {"year": 1900, "freq": 0},
                {"year": 1920, "freq": 2}, {"year": 1940, "freq": 15}, {"year": 1950, "freq": 45},
                {"year": 1970, "freq": 30}, {"year": 1990, "freq": 35}, {"year": 2000, "freq": 40},
                {"year": 2020, "freq": 50},
            ],
            "الزيتون": [
                {"year": 1800, "freq": 1}, {"year": 1850, "freq": 3}, {"year": 1900, "freq": 8},
                {"year": 1920, "freq": 12}, {"year": 1940, "freq": 20}, {"year": 1950, "freq": 18},
                {"year": 1970, "freq": 22}, {"year": 1990, "freq": 28}, {"year": 2000, "freq": 35},
                {"year": 2020, "freq": 42},
            ],
            "التطريز": [
                {"year": 1800, "freq": 0}, {"year": 1850, "freq": 1}, {"year": 1900, "freq": 5},
                {"year": 1920, "freq": 10}, {"year": 1940, "freq": 18}, {"year": 1950, "freq": 22},
                {"year": 1970, "freq": 28}, {"year": 1990, "freq": 35}, {"year": 2000, "freq": 45},
                {"year": 2020, "freq": 55},
            ],
            "الشعر": [
                {"year": 1800, "freq": 2}, {"year": 1850, "freq": 5}, {"year": 1900, "freq": 12},
                {"year": 1920, "freq": 20}, {"year": 1940, "freq": 30}, {"year": 1950, "freq": 35},
                {"year": 1970, "freq": 38}, {"year": 1990, "freq": 42}, {"year": 2000, "freq": 48},
                {"year": 2020, "freq": 52},
            ],
            "المنفى": [
                {"year": 1800, "freq": 0}, {"year": 1850, "freq": 0}, {"year": 1900, "freq": 1},
                {"year": 1920, "freq": 3}, {"year": 1940, "freq": 25}, {"year": 1950, "freq": 40},
                {"year": 1970, "freq": 32}, {"year": 1990, "freq": 28}, {"year": 2000, "freq": 25},
                {"year": 2020, "freq": 22},
            ],
            "المقاومة": [
                {"year": 1800, "freq": 0}, {"year": 1850, "freq": 0}, {"year": 1900, "freq": 2},
                {"year": 1920, "freq": 8}, {"year": 1940, "freq": 20}, {"year": 1950, "freq": 30},
                {"year": 1970, "freq": 45}, {"year": 1990, "freq": 50}, {"year": 2000, "freq": 55},
                {"year": 2020, "freq": 48},
            ],
        },
        "era_info": {
            "1800": "فترة الحكم العثماني المبكر - وثائق محدودة عن الحياة الثقافية في فلسطين",
            "1900": "نهاية الحكم العثماني وبدء الانتداب البريطاني - تنامي الوعي الوطني الفلسطيني",
            "1940": "فترة النكبة والتأسيس - تحول جذري في البنية الاجتماعية والثقافية الفلسطينية",
            "1960": "ما بعد النكسة - صعود حركات المقاومة وتطور الأدب الفلسطيني",
            "1980": "الانتفاضة الأولى - تنامي الهوية الثقافية والفنية كأداة مقاومة",
            "2000": "الانتفاضة الثانية والعصر الرقمي - توسع الإنتاج الثقافي والأدبي الفلسطيني",
        },
    }


# ============================================================
# BIAS DATA
# ============================================================

def get_bias_data():
    """Return bias measurement data."""
    return {
        "dimension_scores": [
            {
                "dimension": "الثقافة",
                "dimension_en": "Cultural",
                "value": 72,
                "sources": {
                    "ويكيبيديا العربية": 78,
                    "وكالة وفا": 45,
                    "ويكيبيديا الإنجليزية": 82,
                    "GDELT": 30,
                },
                "description": "نسبة التمثيل الثقافي (الفنون، التراث، المطبخ، العادات) في المحتوى",
            },
            {
                "dimension": "الصراع",
                "dimension_en": "Conflict",
                "value": 58,
                "sources": {
                    "ويكيبيديا العربية": 50,
                    "وكالة وفا": 82,
                    "ويكيبيديا الإنجليزية": 55,
                    "GDELT": 85,
                },
                "description": "نسبة التمثيل المتعلق بالصراع والسياسة والأحداث العسكرية",
            },
            {
                "dimension": "التاريخ",
                "dimension_en": "Historical",
                "value": 65,
                "sources": {
                    "ويكيبيديا العربية": 70,
                    "وكالة وفا": 40,
                    "ويكيبيديا الإنجليزية": 75,
                    "GDELT": 35,
                },
                "description": "نسبة المحتوى التاريخي والمعماري والأثري",
            },
            {
                "dimension": "الدين والمقدسات",
                "dimension_en": "Religious",
                "value": 45,
                "sources": {
                    "ويكيبيديا العربية": 55,
                    "وكالة وفا": 25,
                    "ويكيبيديا الإنجليزية": 50,
                    "GDELT": 15,
                },
                "description": "نسبة المحتوى المتعلق بالأماكن المقدسة والممارسات الدينية",
            },
            {
                "dimension": "الاقتصاد والمعيشة",
                "dimension_en": "Economic",
                "value": 30,
                "sources": {
                    "ويكيبيديا العربية": 35,
                    "وكالة وفا": 20,
                    "ويكيبيديا الإنجليزية": 30,
                    "GDELT": 40,
                },
                "description": "نسبة المحتوى المتعلق بالاقتصاد والزراعة والتجارة",
            },
        ],
        "weat_score": -1.612,
        "weat_description": "مقياس WEAT يُظهر ارتباطًا بين المصطلحات الثقافية الفلسطينية وكلمات العنف في بعض المصادر - يشير هذا إلى أن بعض المصادر تميل إلى ربط المحتوى الفلسطيني بإطار الصراع",
        "source_comparison": [
            {
                "source": "ويكيبيديا العربية",
                "total_docs": 581,
                "cultural_pct": 42,
                "conflict_pct": 18,
                "historical_pct": 25,
                "religious_pct": 15,
            },
            {
                "source": "وكالة وفا",
                "total_docs": 106,
                "cultural_pct": 15,
                "conflict_pct": 52,
                "historical_pct": 12,
                "religious_pct": 8,
            },
            {
                "source": "ويكيبيديا الإنجليزية",
                "total_docs": 97,
                "cultural_pct": 38,
                "conflict_pct": 22,
                "historical_pct": 28,
                "religious_pct": 12,
            },
            {
                "source": "GDELT",
                "total_docs": 73,
                "cultural_pct": 12,
                "conflict_pct": 58,
                "historical_pct": 10,
                "religious_pct": 5,
            },
        ],
        "filters": {
            "sources": ["الكل", "ويكيبيديا العربية", "وكالة وفا", "ويكيبيديا الإنجليزية", "GDELT", "Semantic Scholar"],
            "languages": ["الكل", "العربية", "الإنجليزية"],
            "decades": ["الكل", "1800", "1900", "1940", "1960", "1980", "2000", "2020"],
        },
    }


# ============================================================
# KNOWLEDGE GRAPH DATA
# ============================================================

def get_kg_data():
    """Return knowledge graph data for visualization."""
    return {
        "stats": {
            "total_entities": 13063,
            "total_relations": 585,
            "linked_to_wikidata": 950,
        },
        "sample_entities": [
            "القدس", "محمود درويش", "التطريز", "يافا", "حيفا",
            "نابلس", "غزة", "بيت لحم", "الكنافة", "الزيتون",
            "المسجد الأقصى", "النكبة", "فلسطين", "الخليل", "رام الله",
        ],
    }


def search_knowledge_graph(query: str):
    """Search the knowledge graph for an entity and return its neighborhood.

    Replace this with: graph_store.search(query) or API call.
    """
    # Simulated graph neighborhoods
    graphs = {
        "القدس": {
            "center": {"name": "القدس", "type": "LOCATION", "wikidata": "Q1218"},
            "neighbors": [
                {"name": "المسجد الأقصى", "type": "HERITAGE_PLACE", "relation": "يضم", "direction": "out"},
                {"name": "كنيسة القيامة", "type": "HERITAGE_PLACE", "relation": "يضم", "direction": "out"},
                {"name": "فلسطين", "type": "LOCATION", "relation": "عاصمة", "direction": "out"},
                {"name": "الخليل", "type": "LOCATION", "relation": "قريب من", "direction": "out"},
                {"name": "البحر الميت", "type": "LOCATION", "relation": "قريب من", "direction": "out"},
                {"name": "القدس الشرقية", "type": "LOCATION", "relation": "جزء من", "direction": "in"},
                {"name": "عبدالله الثاني", "type": "PERSON", "relation": "زائر", "direction": "in"},
                {"name": "الإسلام", "type": "MISC", "relation": "مقدس لدى", "direction": "out"},
                {"name": "المسيحية", "type": "MISC", "relation": "مقدس لدى", "direction": "out"},
                {"name": "بلدة القدس القديمة", "type": "HERITAGE_PLACE", "relation": "يحتوي", "direction": "out"},
            ],
        },
        "محمود درويش": {
            "center": {"name": "محمود درويش", "type": "PERSON", "wikidata": "Q263741"},
            "neighbors": [
                {"name": "فلسطين", "type": "LOCATION", "relation": "من", "direction": "out"},
                {"name": "البروة", "type": "LOCATION", "relation": "ولد في", "direction": "out"},
                {"name": "الشعر", "type": "MISC", "relation": "يعمل في", "direction": "out"},
                {"name": "المنفى", "type": "MISC", "relation": "عاش في", "direction": "out"},
                {"name": "بيروت", "type": "LOCATION", "relation": "عاش في", "direction": "out"},
                {"name": "النكبة", "type": "MISC", "relation": "كتب عن", "direction": "out"},
                {"name": "حيفا", "type": "LOCATION", "relation": "عاش في", "direction": "out"},
                {"name": "غسان كنفاني", "type": "PERSON", "relation": "معاصر", "direction": "out"},
            ],
        },
        "التطريز": {
            "center": {"name": "التطريز", "type": "HERITAGE_CRAFT", "wikidata": None},
            "neighbors": [
                {"name": "فلسطين", "type": "LOCATION", "relation": "تراث", "direction": "out"},
                {"name": "الثلث", "type": "HERITAGE_CRAFT", "relation": "نوع من", "direction": "out"},
                {"name": "رام الله", "type": "LOCATION", "relation": "مشهور في", "direction": "out"},
                {"name": "بيت لحم", "type": "LOCATION", "relation": "مشهور في", "direction": "out"},
                {"name": "القميص الفلسطيني", "type": "HERITAGE_CLOTHING", "relation": "يُزين", "direction": "out"},
                {"name": "الأحمر", "type": "MISC", "relation": "لون رئيسي", "direction": "out"},
                {"name": "النساء الفلسطينيات", "type": "PERSON", "relation": "يمارسه", "direction": "in"},
            ],
        },
        "يافا": {
            "center": {"name": "يافا", "type": "LOCATION", "wikidata": "Q33628"},
            "neighbors": [
                {"name": "البحر الأبيض المتوسط", "type": "LOCATION", "relation": "تقع على", "direction": "out"},
                {"name": "تل أبيب", "type": "LOCATION", "relation": "أصبحت جزءًا من", "direction": "out"},
                {"name": "البرتقال", "type": "HERITAGE_PLANT", "relation": "مشهورة بـ", "direction": "out"},
                {"name": "حيفا", "type": "LOCATION", "relation": "قريب من", "direction": "out"},
                {"name": "النكبة", "type": "MISC", "relation": "تأثرت بـ", "direction": "out"},
                {"name": "غسان كنفاني", "type": "PERSON", "relation": "كتب عن", "direction": "in"},
            ],
        },
    }

    # Find best match
    for key, graph in graphs.items():
        if query.strip() in key or key in query.strip():
            return graph

    # Default: generate a generic result
    return {
        "center": {"name": query, "type": "ENTITY", "wikidata": None},
        "neighbors": [
            {"name": "فلسطين", "type": "LOCATION", "relation": "مرتبط بـ", "direction": "out"},
            {"name": "التراث", "type": "MISC", "relation": "جزء من", "direction": "out"},
            {"name": "الثقافة", "type": "MISC", "relation": "متعلق بـ", "direction": "out"},
        ],
    }


# ============================================================
# RAG / ASK DATA
# ============================================================

def ask_question(question: str):
    """Send a question to the RAG system and return the answer.

    Replace this with: rag_pipeline.ask(question) or API call.
    """
    # Simulated responses
    responses = {
        "التراث": {
            "answer": "يشتمل التراث الثقافي الفلسطيني على عدة عناصر رئيسية تشمل: التطريز الفلسطيني بأشكاله المتنوعة التي تختلف من مدينة لأخرى، حيث تتميز كل منطقة بزخارفها الخاصة مثل ثوب رام الله المطرز بالصلبان الحمراء، وثوب بيت لحم بأشكاله الهندسية الدقيقة. كما يشمل التراث الفلسطيني المأكولات الشعبية كالمقلوبة والمنسف والكنافة النابلسية، بالإضافة إلى الأغاني والأدب الشعبي والحكايات والموروثات الشفهية التي تناقلتها الأجيال. يُعد الزيتون أيضًا رمزًا ثقافيًا فلسطينيًا بارزًا يمثل الصمود والعمق التاريخي للشعب الفلسطيني وارتباطه بالأرض.",
            "citations": [
                {"id": 1, "title": "الثقافة الفلسطينية", "source": "ويكيبيديا العربية", "url": "https://ar.wikipedia.org/wiki/ثقافة_فلسطينية"},
                {"id": 2, "title": "التطريز الفلسطيني", "source": "ويكيبيديا العربية", "url": "https://ar.wikipedia.org/wiki/تطريز_فلسطيني"},
                {"id": 3, "title": "المطبخ الفلسطيني", "source": "وكالة وفا", "url": "https://wafa.ps/ar"},
            ],
            "metadata": {
                "sources_used": 3,
                "chunks_retrieved": 5,
                "response_time": "1.2s",
            },
        },
        "التطريز": {
            "answer": "التطريز الفلسطيني هو فن زخرفي تقليدي يُعتبر من أهم عناصر الهوية الثقافية الفلسطينية. يتميز كل إقليم فلسطيني بأسلوب وزخارف مميزة: فتطريز رام الله يتميز بالصلبان المطرزة باللون الأحمر، بينما يتميز تطريز بيت لحم بالأشكال الهندسية الدقيقة المستوحاة من الزخارف الإسلامية. يستخدم التطريز الفلسطيني خيوطًا حريرية ملونة بألوان زاهية مثل الأحمر والأسود والأزرق، وغالبًا ما تُزين الأثواب بالتطريز على الأكمام والأطراف والصدر. يُعد التطريز جزءًا من التراث اللامادي الفلسطيني المهدد بالانقراض بسبب الظروف السياسية والاقتصادية.",
            "citations": [
                {"id": 1, "title": "التطريز الفلسطيني", "source": "ويكيبيديا العربية", "url": "https://ar.wikipedia.org/wiki/تطريز_فلسطيني"},
                {"id": 2, "title": "أزياء فلسطينية", "source": "ويكيبيديا العربية", "url": "https://ar.wikipedia.org/wiki/أزياء_فلسطينية"},
            ],
            "metadata": {
                "sources_used": 2,
                "chunks_retrieved": 5,
                "response_time": "0.9s",
            },
        },
    }

    # Find matching response
    for key, resp in responses.items():
        if key in question:
            return resp

    # Generic response
    return {
        "answer": f"بناءً على البحث في قاعدة المعرفة الفلسطينية، تتعلق سؤالك بموضوع مهم ضمن السياق الثقافي الفلسطيني. تشير المصادر المتاحة إلى أن هذا الموضوع يحظى باهتمام كبير في الأدب والبحث الفلسطيني. نوصي بالتشريح الأعمق من خلال استكشاف الوثائق المتعلقة في الخرائط والمواضيع المتاحة على المنصة للحصول على معلومات أكثر تفصيلًا ومصادر محددة.",
        "citations": [
            {"id": 1, "title": "وثيقة مرتبطة", "source": "ويكيبيديا العربية", "url": None},
            {"id": 2, "title": "مصدر ثانوي", "source": "وكالة وفا", "url": None},
        ],
        "metadata": {
            "sources_used": 2,
            "chunks_retrieved": 5,
            "response_time": "1.5s",
        },
    }
