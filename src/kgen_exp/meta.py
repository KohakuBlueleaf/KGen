DEFAULT_NEGATIVE_PROMPT = """
low quality, worst quality, normal quality, text, signature, jpeg artifacts, 
bad anatomy, old, early, mini skirt, nsfw, chibi, multiple girls, multiple boys, 
multiple tails, multiple views, copyright name, watermark, artist name, signature
"""

DEFAULT_FORMAT = {
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
