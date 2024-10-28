SPECIAL = [
    "1girl",
    "2girls",
    "3girls",
    "4girls",
    "5girls",
    "6+girls",
    "multiple girls",
    "1boy",
    "2boys",
    "3boys",
    "4boys",
    "5boys",
    "6+boys",
    "multiple boys",
    "male focus",
    "1other",
    "2others",
    "3others",
    "4others",
    "5others",
    "6+others",
    "multiple others",
]
POSSIBLE_QUALITY_TAGS = [
    "masterpiece",
    "best quality",
    "great quality",
    "good quality",
    "normal quality",
    "low quality",
    "worse quality",
    "very aesthetic",
    "aesthetic",
    "displeasing",
    "very displeasing",
    "newest",
    "recent",
    "mid",
    "early",
    "old",
    "score_9",
    "score_8_up",
    "score_7_up",
    "score_6_up",
    "score_5_up",
    "score_4_up",
    "source_anime",
    "source_cartoon",
    "source_furry",
    "source_pony",
]
RATING_TAGS = ["safe", "general", "sfw", "sensitive", "nsfw", "explicit"]

# minimum length of generated contents
# unit: tags
TARGET = {
    "very_short": 8,
    "short": 16,
    "long": 32,
    "very_long": 48,
}
TARGET_TIPO = {
    "very_short": 6,
    "short": 18,
    "long": 36,
    "very_long": 54,
}
TARGET_TIPO_MAX = {
    "very_short": 18,
    "short": 36,
    "long": 54,
    "very_long": 72,
}
# unit: sentences
TARGET_TIPO_NL = {
    "very_short": 1,
    "short": 2,
    "long": 4,
    "very_long": 8,
}
TARGET_TIPO_NL_MAX = {
    "very_short": 2,
    "short": 4,
    "long": 8,
    "very_long": 10000,
}

TIPO_DEFAULT_FORMAT = {
    "tag only (DTG mode)": """
<|special|>, <|characters|>, <|copyrights|>, 
<|artist|>, 

<|general|>,

<|quality|>, <|meta|>, <|rating|>
""".strip(),
    "NL only (Tag to NL)": """<|extended|>.""".strip(),
    "Both, tag first (recommend)": """
<|special|>, <|characters|>, <|copyrights|>, 
<|artist|>, 

<|general|>,

<|extended|>.

<|quality|>, <|meta|>, <|rating|>
""".strip(),
    "Both, NL first (recommend)": """
<|special|>, <|characters|>, <|copyrights|>, 
<|artist|>, 

<|extended|>.

<|general|>,

<|quality|>, <|meta|>, <|rating|>
""".strip(),
    "Both + generated NL": """
<|special|>, <|characters|>, <|copyrights|>, 
<|artist|>, 

<|generated|>.

<|general|>,

<|extended|>.

<|quality|>, <|meta|>, <|rating|>
""".strip(),
}
