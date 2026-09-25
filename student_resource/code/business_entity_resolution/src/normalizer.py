"""
Business Entity Resolution — Universal Normalization Engine (Optimized)
"""

import re
import unicodedata
from typing import Tuple, List, Optional

# ==============================================================================
# 1. INDIC SCRIPT PHONETIC TRANSLITERATION TABLES (REFINED FOR STANDARD ROMANIZATION)
# ==============================================================================

# Standard Indian Romanization for Devanagari (Hindi, Marathi)
# Maps to standard English spelling (e.g., 'आ' -> 'a', 'ी' -> 'i', 'ू' -> 'u')
DEVANAGARI_MAP = {
    '\u0905': 'a', '\u0906': 'a', '\u0907': 'i', '\u0908': 'i', '\u0909': 'u', '\u090a': 'u',
    '\u090b': 'ri', '\u090e': 'e', '\u090f': 'e', '\u0910': 'ai', '\u0912': 'o', '\u0913': 'o', '\u0914': 'au',
    '\u0915': 'k', '\u0916': 'kh', '\u0917': 'g', '\u0918': 'gh', '\u0919': 'ng',
    '\u091a': 'ch', '\u091b': 'chh', '\u091c': 'j', '\u091d': 'jh', '\u091e': 'ny',
    '\u091f': 't', '\u0920': 'th', '\u0921': 'd', '\u0922': 'dh', '\u0923': 'n',
    '\u0924': 't', '\u0925': 'th', '\u0926': 'd', '\u0927': 'dh', '\u0928': 'n',
    '\u092a': 'p', '\u092b': 'ph', '\u092c': 'b', '\u092d': 'bh', '\u092e': 'm',
    '\u092f': 'ya', '\u0930': 'r', '\u0932': 'l', '\u0933': 'l', '\u0935': 'v',
    '\u0936': 'sh', '\u0937': 'sh', '\u0938': 's', '\u0939': 'h',
    # Matras (vowel signs)
    '\u093e': 'a', '\u093f': 'i', '\u0940': 'i', '\u0941': 'u', '\u0942': 'u',
    '\u0943': 'ri', '\u0947': 'e', '\u0948': 'ai', '\u094b': 'o', '\u094c': 'au',
    '\u0902': 'n', '\u0901': 'n', '\u0903': 'h', '\u094d': '',  # Halant/Virama
    '\u0958': 'q', '\u0959': 'kh', '\u095a': 'g', '\u095b': 'z', '\u095c': 'r', '\u095d': 'rh', '\u095e': 'f'
}

# Mapping for Tamil \u0B80 - \u0BFF
TAMIL_MAP = {
    '\u0b85': 'a', '\u0b86': 'a', '\u0b87': 'i', '\u0b88': 'i', '\u0b89': 'u', '\u0b8a': 'u',
    '\u0b8e': 'e', '\u0b8f': 'e', '\u0b90': 'ai', '\u0b92': 'o', '\u0b93': 'o', '\u0b94': 'au',
    '\u0b95': 'k', '\u0b99': 'ng', '\u0b9a': 'ch', '\u0b9c': 'j', '\u0b9e': 'ny',
    '\u0b9f': 't', '\u0ba3': 'n', '\u0ba4': 'th', '\u0ba8': 'n', '\u0ba9': 'n',
    '\u0baa': 'p', '\u0bae': 'm', '\u0baf': 'ya', '\u0bb0': 'r', '\u0bb1': 'r',
    '\u0bb2': 'l', '\u0bb3': 'l', '\u0bb4': 'zh', '\u0bb5': 'v', '\u0bb6': 'sh',
    '\u0bb7': 'sh', '\u0bb8': 's', '\u0bb9': 'h',
    # Matras
    '\u0bbe': 'a', '\u0bbf': 'i', '\u0bc0': 'i', '\u0bc1': 'u', '\u0bc2': 'u',
    '\u0bc6': 'e', '\u0bc7': 'e', '\u0bc8': 'ai', '\u0bca': 'o', '\u0bcb': 'o', '\u0bcc': 'au',
    '\u0bcd': '',  # Virama / Pulli
    '\u0802': 'm'
}

# Mapping for Kannada \u0C80 - \u0CFF
KANNADA_MAP = {
    '\u0c85': 'a', '\u0c86': 'a', '\u0c87': 'i', '\u0c88': 'i', '\u0c89': 'u', '\u0c8a': 'u',
    '\u0c8e': 'e', '\u0c8f': 'e', '\u0c90': 'ai', '\u0c92': 'o', '\u0c93': 'o', '\u0c94': 'au',
    '\u0c95': 'k', '\u0c96': 'kh', '\u0c97': 'g', '\u0c98': 'gh', '\u0c99': 'ng',
    '\u0c9a': 'ch', '\u0c9b': 'chh', '\u0c9c': 'j', '\u0c9d': 'jh', '\u0c9e': 'ny',
    '\u0c9f': 't', '\u0ca0': 'th', '\u0ca1': 'd', '\u0ca2': 'dh', '\u0ca3': 'n',
    '\u0ca4': 't', '\u0ca5': 'th', '\u0ca6': 'd', '\u0ca7': 'dh', '\u0ca8': 'n',
    '\u0caa': 'p', '\u0cab': 'ph', '\u0cac': 'b', '\u0cad': 'bh', '\u0cae': 'm',
    '\u0caf': 'ya', '\u0cb0': 'r', '\u0cb1': 'r', '\u0cb2': 'l', '\u0cb3': 'l', '\u0cb5': 'v',
    '\u0cb6': 'sh', '\u0cb7': 'sh', '\u0cb8': 's', '\u0cb9': 'h',
    # Matras
    '\u0cbe': 'a', '\u0cbf': 'i', '\u0cc0': 'i', '\u0cc1': 'u', '\u0cc2': 'u',
    '\u0cc6': 'e', '\u0cc7': 'e', '\u0cc8': 'ai', '\u0cca': 'o', '\u0ccb': 'o', '\u0ccc': 'au',
    '\u0ccd': '',
    '\u0c82': 'n'
}

# Mapping for Telugu \u0C00 - \u0C7F
TELUGU_MAP = {
    '\u0c05': 'a', '\u0c06': 'a', '\u0c07': 'i', '\u0c08': 'i', '\u0c09': 'u', '\u0c0a': 'u',
    '\u0c0e': 'e', '\u0c0f': 'e', '\u0c10': 'ai', '\u0c12': 'o', '\u0c13': 'o', '\u0c14': 'au',
    '\u0c15': 'k', '\u0c16': 'kh', '\u0c17': 'g', '\u0c18': 'gh', '\u0c19': 'ng',
    '\u0c1a': 'ch', '\u0c1b': 'chh', '\u0c1c': 'j', '\u0c1d': 'jh', '\u0c1e': 'ny',
    '\u0c1f': 't', '\u0c20': 'th', '\u0c21': 'd', '\u0c22': 'dh', '\u0c23': 'n',
    '\u0c24': 't', '\u0c25': 'th', '\u0c26': 'd', '\u0c27': 'dh', '\u0c28': 'n',
    '\u0c2a': 'p', '\u0c2b': 'ph', '\u0c2c': 'b', '\u0c2d': 'bh', '\u0c2e': 'm',
    '\u0c2f': 'ya', '\u0c30': 'r', '\u0c31': 'r', '\u0c32': 'l', '\u0c33': 'l', '\u0c35': 'v',
    '\u0c36': 'sh', '\u0c37': 'sh', '\u0c38': 's', '\u0c39': 'h',
    # Matras
    '\u0c3e': 'a', '\u0c3f': 'i', '\u0c40': 'i', '\u0c41': 'u', '\u0c42': 'u',
    '\u0c46': 'e', '\u0c47': 'e', '\u0c48': 'ai', '\u0c4a': 'o', '\u0c4b': 'o', '\u0c4c': 'au',
    '\u0c4d': '',
    '\u0c02': 'n'
}

# Mapping for Bengali & Gujarati
BENGALI_MAP = {
    '\u0985': 'a', '\u0986': 'a', '\u0987': 'i', '\u0988': 'i', '\u0989': 'u', '\u098a': 'u',
    '\u098f': 'e', '\u0990': 'ai', '\u0993': 'o', '\u0994': 'au',
    '\u0995': 'k', '\u0996': 'kh', '\u0997': 'g', '\u0998': 'gh', '\u0999': 'ng',
    '\u099a': 'ch', '\u099b': 'chh', '\u099c': 'j', '\u099d': 'jh', '\u099e': 'ny',
    '\u099f': 't', '\u09a0': 'th', '\u09a1': 'd', '\u09a2': 'dh', '\u09a3': 'n',
    '\u09a4': 't', '\u09a5': 'th', '\u09a6': 'd', '\u09a7': 'dh', '\u09a8': 'n',
    '\u09aa': 'p', '\u09ab': 'ph', '\u09ac': 'b', '\u09ad': 'bh', '\u09ae': 'm',
    '\u09af': 'ya', '\u09b0': 'r', '\u09b2': 'l', '\u09ac': 'v', '\u09b6': 'sh',
    '\u09b7': 'sh', '\u09b8': 's', '\u09b9': 'h',
    '\u09be': 'a', '\u09bf': 'i', '\u09c0': 'i', '\u09c1': 'u', '\u09c2': 'u',
    '\u09c7': 'e', '\u09c8': 'ai', '\u09cb': 'o', '\u09cc': 'au', '\u09cd': '', '\u0982': 'n'
}

GUJARATI_MAP = {
    '\u0a85': 'a', '\u0a86': 'a', '\u0a87': 'i', '\u0a88': 'i', '\u0a89': 'u', '\u0a8a': 'u',
    '\u0a8f': 'e', '\u0a90': 'ai', '\u0a93': 'o', '\u0a94': 'au',
    '\u0a95': 'k', '\u0a96': 'kh', '\u0a97': 'g', '\u0a98': 'gh', '\u0a99': 'ng',
    '\u0a9a': 'ch', '\u0a9b': 'chh', '\u0a9c': 'j', '\u0a9d': 'jh', '\u0a9e': 'ny',
    '\u0a9f': 't', '\u0aa0': 'th', '\u0aa1': 'd', '\u0aa2': 'dh', '\u0aa3': 'n',
    '\u0aa4': 't', '\u0aa5': 'th', '\u0aa6': 'd', '\u0aa7': 'dh', '\u0aa8': 'n',
    '\u0aaa': 'p', '\u0aab': 'ph', '\u0aac': 'b', '\u0aad': 'bh', '\u0aae': 'm',
    '\u0aaf': 'ya', '\u0ab0': 'r', '\u0ab2': 'l', '\u0ab5': 'v', '\u0ab6': 'sh',
    '\u0ab7': 'sh', '\u0ab8': 's', '\u0ab9': 'h',
    '\u0abe': 'a', '\u0abf': 'i', '\u0ac0': 'i', '\u0ac1': 'u', '\u0ac2': 'u',
    '\u0ac7': 'e', '\u0ac8': 'ai', '\u0acb': 'o', '\u0acc': 'au', '\u0acd': '', '\u0a82': 'n'
}

MASTER_UNICODE_MAP = {}
for m in [DEVANAGARI_MAP, TAMIL_MAP, KANNADA_MAP, TELUGU_MAP, BENGALI_MAP, GUJARATI_MAP]:
    MASTER_UNICODE_MAP.update(m)

# High-frequency Indic business terms dictionary
INDIC_BUSINESS_TERMS = {
    # Hindi / Marathi
    'प्राइवेट लिमिटेड': 'pvt ltd',
    'प्रा. लि.': 'pvt ltd',
    'प्रा० लि०': 'pvt ltd',
    'प्रा लि': 'pvt ltd',
    'लिमिटेड': 'ltd',
    'लि.': 'ltd',
    'एलएलपी': 'llp',
    'मार्केटिंग': 'marketing',
    'कंस्ट्रक्शंस': 'constructions',
    'कंस्ट्रक्शन': 'construction',
    'प्रॉपर्टीज': 'properties',
    'प्रॉपर्टी': 'property',
    'एंटरप्राइजेज': 'enterprises',
    'एंटरप्राइज': 'enterprise',
    'ट्रेडर्स': 'traders',
    'ट्रेडिंग': 'trading',
    'सर्विसेज': 'services',
    'सर्विस': 'service',
    'सॉल्यूशंस': 'solutions',
    'टेक्नोलॉजीज': 'technologies',
    'टेक्नोलॉजी': 'technology',
    'कंसल्टेंट्स': 'consultants',
    'कंसल्टिंग': 'consulting',
    'फूड्स': 'foods',
    'फूड': 'food',
    'हेल्थकेयर': 'healthcare',
    'फार्मा': 'pharma',
    'फाउंडेशन': 'foundation',
    'इंफ्राकॉन': 'infracon',
    'इंफ्रास्ट्रक्चर': 'infrastructure',
    'एजेंसी': 'agency',
    'हॉस्पिटल': 'hospital',
    'विद्यालय': 'vidyalaya',
    'इंटरनेशनल': 'international',
    'ग्लोबल': 'global',
    'उद्योग': 'udyog',
    'व्यापार': 'vyapar',
    
    # State & City Hindi names in addresses
    'दिल्ली': 'delhi',
    'नई दिल्ली': 'new delhi',
    'हरियाणा': 'haryana',
    'महाराष्ट्र': 'maharashtra',
    'पंजाब': 'punjab',
    'उत्तर प्रदेश': 'uttar pradesh',
    'मध्य प्रदेश': 'madhya pradesh',
    'गुजरात': 'gujarat',
    'राजस्थान': 'rajasthan',
    'कर्नाटक': 'karnataka',
    'कोलकाता': 'kolkata',
    'मुंबई': 'mumbai',
    'पुणे': 'pune',
    'अहमदाबाद': 'ahmedabad',
    'हैदराबाद': 'hyderabad',
    'चेन्नई': 'chennai',
    'बेंगलुरु': 'bengaluru',
    
    # Tamil terms
    'பிரைவேட் லிமிடெட்': 'pvt ltd',
    'லிமிடெட்': 'ltd',
    'குளோபல்': 'global',
    'பிசினஸ்': 'business',
    'தமிழ்நாடு': 'tamil nadu',
    'சென்னை': 'chennai',
    
    # Kannada terms
    'ಕರ್ನಾಟಕ': 'karnataka',
    'ಬೆಂಗಳೂರು': 'bangalore'
}

indic_terms_pattern = re.compile('|'.join(re.escape(k) for k in sorted(INDIC_BUSINESS_TERMS.keys(), key=len, reverse=True)))

def transliterate_indic(text: str) -> str:
    """Replaces known Indic business terms and phonetically transliterates remaining Indic script."""
    if not text:
        return ""
    text = indic_terms_pattern.sub(lambda m: INDIC_BUSINESS_TERMS[m.group(0)], text)
    
    chars = []
    for ch in text:
        if ch in MASTER_UNICODE_MAP:
            chars.append(MASTER_UNICODE_MAP[ch])
        else:
            chars.append(ch)
    return "".join(chars)


# ==============================================================================
# 2. EUROPEAN / FRENCH ACCENTS & LIGATURES
# ==============================================================================

LIGATURE_MAP = {
    'œ': 'oe', 'Œ': 'oe', 'æ': 'ae', 'Æ': 'ae', 'ß': 'ss',
    '«': ' ', '»': ' ', '“': ' ', '”': ' ', '’': "'", '`': "'"
}

def clean_unicode_and_accents(text: str) -> str:
    """Applies NFKD decomposition, ligature expansion, and accent stripping."""
    if not text:
        return ""
    for k, v in LIGATURE_MAP.items():
        text = text.replace(k, v)
        
    text = transliterate_indic(text)
    
    nfkd = unicodedata.normalize('NFKD', text)
    ascii_folded = "".join(c for c in nfkd if not unicodedata.combining(c))
    
    return ascii_folded.lower()


# ==============================================================================
# 3. LEGAL ENTITY SUFFIXES & NOISE DECORATORS
# ==============================================================================

# Regexes for domain / URL cleaning (stripping protocol and domain extension without wiping domain name)
RE_HTTP_WWW = re.compile(r'\b(?:https?://|www\.)', re.IGNORECASE)
RE_DOMAIN_SUFFIX = re.compile(r'\.(?:com|org|net|co\.in|in|fr|io|biz|info|edu|gov)\b', re.IGNORECASE)

RE_PREFIX_QUALIFIERS = re.compile(r'^(?:mirapyra\s+)?(?:formerly|dba|d/b/a|ex|trading\s+as|t/a)\s+', re.IGNORECASE)
RE_NOISE_SYMBOLS = re.compile(r'^[#\-<>\*\/\@\+\|]+|[#\-<>\*\/\@\+\|]+$')
RE_REGISTRATION_ID = re.compile(r'-\s*\d{6,}\b|\#\d{4,}\b')
RE_AMPERSAND = re.compile(r'&')

LEGAL_SUFFIX_MAP = {
    r'\bprivate\s+limited\b|\bpvt\s+ltd\b|\bp\s+ltd\b|\bp\.\s*ltd\b|\bp\s+limited\b': 'pvt ltd',
    r'\bpublic\s+limited\b|\bpub\s+ltd\b': 'pub ltd',
    r'\blimited\b|\bltd\b|\bltd\.': 'ltd',
    r'\blimited\s+liability\s+partnership\b|\bllp\b|\bl\.l\.p\b': 'llp',
    r'\blimited\s+liability\s+company\b|\bllc\b|\bl\.l\.c\b|\bl\.l\.c\.\b|\(llc\)': 'llc',
    r'\bincorporated\b|\binc\b|\binc\.\b|\(inc\)': 'inc',
    r'\bcorporation\b|\bcorp\b|\bcorp\.\b': 'corp',
    r'\bcompany\b|\bco\b|\bco\.\b': 'co',
    r'\bprofessional\s+corporation\b|\bpc\b|\bp\.c\b': 'pc',
    r'\blimited\s+partnership\b|\blp\b|\bl\.p\b': 'lp',
    r'\bsociete\s+a\s+responsabilite\s+limitee\b|\bsarl\b|\bs\.a\.r\.l\b': 'sarl',
    r'\bsociete\s+par\s+actions\s+simplifiee\b|\bsasu\b|\bsas\b|\bs\.a\.s\b': 'sas',
    r'\bentreprise\s+unipersonnelle\s+a\s+responsabilite\s+limitee\b|\beurl\b': 'eurl',
    r'\bet\s+fils\b|\band\s+fils\b|\b&\s+fils\b': 'et fils'
}

COMPILED_LEGAL_SUFFIXES = [(re.compile(pattern, re.IGNORECASE), canonical) for pattern, canonical in LEGAL_SUFFIX_MAP.items()]
ALL_LEGAL_TOKENS_REGEX = re.compile(r'\b(?:pvt ltd|pub ltd|ltd|llp|llc|inc|corp|co|pc|lp|sarl|sas|sasu|eurl)\b', re.IGNORECASE)

def normalize_business_name(raw_name: str) -> Tuple[str, str]:
    """
    Transforms raw business name into (clean_name_core, clean_name_full).
    """
    if not raw_name or not isinstance(raw_name, str):
        return "", ""
    
    # 1. Unicode, Accents & Indic transliteration
    text = clean_unicode_and_accents(raw_name)
    
    # 2. Conjunctions
    text = RE_AMPERSAND.sub(' and ', text)
    
    # 3. Strip URL protocol and domain extension (leaving domain brand name intact!)
    text = RE_HTTP_WWW.sub('', text)
    text = RE_DOMAIN_SUFFIX.sub(' ', text)
    text = RE_REGISTRATION_ID.sub(' ', text)
    
    # 4. Strip noise prefixes and decorators
    text = RE_NOISE_SYMBOLS.sub(' ', text)
    text = RE_PREFIX_QUALIFIERS.sub('', text)
    
    # 5. Clean punctuation (preserve alphanumeric and spaces)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 6. Canonicalize legal suffixes for clean_name_full
    full_name = text
    for pattern, canonical in COMPILED_LEGAL_SUFFIXES:
        full_name = pattern.sub(canonical, full_name)
    full_name = re.sub(r'\s+', ' ', full_name).strip()
    
    # 7. Strip legal suffixes for clean_name_core
    core_name = ALL_LEGAL_TOKENS_REGEX.sub(' ', full_name)
    core_name = re.sub(r'\s+', ' ', core_name).strip()
    
    if not core_name:
        core_name = full_name
        
    return core_name, full_name


# ==============================================================================
# 4. ADDRESS STANDARDIZATION & NUMBER EXTRACTION
# ==============================================================================

US_STATES = {
    'alabama': 'al', 'alaska': 'ak', 'arizona': 'az', 'arkansas': 'ar', 'california': 'ca',
    'colorado': 'co', 'connecticut': 'ct', 'delaware': 'de', 'florida': 'fl', 'georgia': 'ga',
    'hawaii': 'hi', 'idaho': 'id', 'illinois': 'il', 'indiana': 'in', 'iowa': 'ia',
    'kansas': 'ks', 'kentucky': 'ky', 'louisiana': 'la', 'maine': 'me', 'maryland': 'md',
    'massachusetts': 'ma', 'michigan': 'mi', 'minnesota': 'mn', 'mississippi': 'ms',
    'missouri': 'mo', 'montana': 'mt', 'nebraska': 'ne', 'nevada': 'nv', 'new hampshire': 'nh',
    'new jersey': 'nj', 'new mexico': 'nm', 'new york': 'ny', 'north carolina': 'nc',
    'north dakota': 'nd', 'ohio': 'oh', 'oklahoma': 'ok', 'oregon': 'or', 'pennsylvania': 'pa',
    'rhode island': 'ri', 'south carolina': 'sc', 'south dakota': 'sd', 'tennessee': 'tn',
    'texas': 'tx', 'utah': 'ut', 'vermont': 'vt', 'virginia': 'va', 'washington': 'wa',
    'west virginia': 'wv', 'wisconsin': 'wi', 'wyoming': 'wy'
}

ADDRESS_ABBR_MAP = {
    r'\bstreet\b|\bst\b|\bst\.': 'st',
    r'\broad\b|\brd\b|\brd\.': 'rd',
    r'\bavenue\b|\bave\b|\bave\.': 'ave',
    r'\bboulevard\b|\bblvd\b|\bbld\b|\bbd\b|\bbd\.': 'blvd',
    r'\bdrive\b|\bdr\b|\bdr\.': 'dr',
    r'\blane\b|\bln\b|\bln\.': 'ln',
    r'\bcourt\b|\bct\b|\bct\.': 'ct',
    r'\bhighway\b|\bhwy\b': 'hwy',
    r'\bcircle\b|\bcir\b': 'cir',
    r'\bparkway\b|\bpkwy\b': 'pkwy',
    r'\btrail\b|\btrl\b': 'trail',
    r'\bapartment\b|\bapt\b|\bflat\s+no\b|\bflat\b': 'apt',
    r'\bsuite\b|\bste\b|\bunit\b|\broom\b|\brm\b': 'unit',
    r'\bfloor\b|\bfl\b|\b1st\s+floor\b|\b2nd\s+floor\b|\b3rd\s+floor\b': 'fl',
    r'\bbuilding\b|\bbldg\b': 'bldg',
    r'\bopposite\b|\bopp\b|\bopp\.': 'opp',
    r'\bbehind\b|\bb\/h\b|\bb\-h\b': 'behind',
    r'\bnear\b|\bnr\b|\bnr\.': 'near',
    r'\bhouse\s+no\b|\bh\.no\b|\bhn\b|\bplot\s+no\b|\bpt\s+no\b': 'no'
}

COMPILED_ADDR_ABBRS = [(re.compile(p, re.IGNORECASE), repl) for p, repl in ADDRESS_ABBR_MAP.items()]
RE_ZERO_PADDED_NUM = re.compile(r'\b0+(\d+)\b')
RE_DIGITS = re.compile(r'\b\d+\b')

def normalize_business_address(raw_address: str) -> Tuple[str, Tuple[str, ...], int]:
    """
    Transforms raw business address into:
    - clean_address: Standardized address string
    - address_numbers: Tuple of extracted integer string tokens
    - has_address: 1 if address present, 0 if missing
    """
    if not raw_address or not isinstance(raw_address, str) or not raw_address.strip():
        return "", (), 0
    
    text = clean_unicode_and_accents(raw_address)
    text = RE_AMPERSAND.sub(' and ', text)
    
    for pattern, repl in COMPILED_ADDR_ABBRS:
        text = pattern.sub(repl, text)
        
    for state_name, abbr in US_STATES.items():
        text = re.sub(rf'\b{state_name}\b', abbr, text)
        
    text = RE_ZERO_PADDED_NUM.sub(r'\1', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    clean_addr = re.sub(r'\s+', ' ', text).strip()
    numbers = tuple(RE_DIGITS.findall(clean_addr))
    
    return clean_addr, numbers, 1


# ==============================================================================
# 5. UNIFIED RECORD NORMALIZER
# ==============================================================================

def normalize_record(name: str, address: str, country: str) -> dict:
    """Complete normalization pipeline for a single record."""
    clean_name_core, clean_name_full = normalize_business_name(name)
    clean_address, address_numbers, has_address = normalize_business_address(address)
    country_clean = (country or "").strip().upper()
    
    return {
        "clean_name": clean_name_core,
        "clean_name_full": clean_name_full,
        "clean_address": clean_address,
        "address_numbers": address_numbers,
        "has_address": has_address,
        "country": country_clean
    }
