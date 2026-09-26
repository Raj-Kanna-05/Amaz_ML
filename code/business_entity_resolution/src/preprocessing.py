"""
Data Preprocessing and Normalization Module for Business Entity Resolution.
Implements Unicode NFKC normalization, diacritic normalization, legal suffix & street
abbreviation expansion, postal/number token extraction, and country standardization.
"""

import re
import unicodedata
from typing import Dict, List, Set, Tuple
import pandas as pd


# Business Legal / Name Abbreviations
NAME_ABBREVIATIONS: Dict[str, str] = {
    "corp": "corporation",
    "inc": "incorporated",
    "ltd": "limited",
    "pvt": "private",
    "llc": "limited liability company",
    "llp": "limited liability partnership",
    "co": "company",
    "ent": "enterprises",
    "intl": "international",
    "mfg": "manufacturing",
    "tech": "technologies",
    "grp": "group",
    "assn": "association",
    "assoc": "associates",
    "dept": "department",
    "svc": "service",
    "svcs": "services",
    "univ": "university",
    "hosp": "hospital",
    "distr": "distribution",
    "sys": "systems",
    "ind": "industries",
    "soln": "solutions",
    "solns": "solutions",
    "mktg": "marketing",
}

# Legal Suffixes to identify and strip for "core_name"
LEGAL_SUFFIXES: Set[str] = {
    "corporation", "incorporated", "limited", "private", "llc", "llp",
    "company", "enterprises", "corp", "inc", "ltd", "pvt", "co", "gmbh",
    "sarl", "sa", "sas", "plc", "bv", "nv", "spa", "srl"
}

# Address and Street Abbreviations
ADDRESS_ABBREVIATIONS: Dict[str, str] = {
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "blvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "ct": "court",
    "cir": "circle",
    "hwy": "highway",
    "ste": "suite",
    "apt": "apartment",
    "fl": "floor",
    "bldg": "building",
    "opp": "opposite",
    "nr": "near",
    "plz": "plaza",
    "pl": "place",
    "sq": "square",
    "pk": "park",
    "pkwy": "parkway",
    "expy": "expressway",
    "ctr": "center",
    "nd": "second",
    "th": "",
    "po box": "pobox",
    "p o box": "pobox",
}

# Country Aliases (Preserves unseen countries dynamically for open-set test sets)
COUNTRY_MAP: Dict[str, str] = {
    "us": "us",
    "usa": "us",
    "united states": "us",
    "united states of america": "us",
    "in": "in",
    "ind": "in",
    "india": "in",
    "fr": "fr",
    "fra": "fr",
    "france": "fr",
}


def remove_accents(text: str) -> str:
    """Removes diacritics / accents for robust matching (e.g. French cafés -> cafes)."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def normalize_text(text: str) -> str:
    """General text normalization: NFKC, unidecode, lowercase, punctuation cleanup."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = remove_accents(text)
    text = text.lower()
    text = text.replace("&", " and ")
    text = text.replace("@", " at ")
    # Replace non-alphanumeric chars with space
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def expand_tokens(tokens: List[str], mapping: Dict[str, str]) -> List[str]:
    """Expands abbreviated tokens using a dictionary mapping."""
    return [mapping.get(tok, tok) for tok in tokens]


def normalize_name(name: str) -> Tuple[str, str, List[str]]:
    """
    Normalizes business name.
    Returns:
    - norm_name: full normalized name with abbreviations expanded
    - core_name: name with common legal suffixes stripped
    - tokens: list of clean token strings
    """
    cleaned = normalize_text(name)
    raw_tokens = cleaned.split()
    expanded = expand_tokens(raw_tokens, NAME_ABBREVIATIONS)
    norm_name = " ".join(expanded)

    # Core name: remove trailing legal suffixes
    core_tokens = [t for t in expanded if t not in LEGAL_SUFFIXES]
    core_name = " ".join(core_tokens) if core_tokens else norm_name

    return norm_name, core_name, expanded


def normalize_address(address: str) -> Tuple[str, List[str], List[str]]:
    """
    Normalizes business address.
    Returns:
    - norm_address: normalized address string with expanded abbreviations
    - tokens: list of word tokens
    - numbers: list of numeric strings (postal codes, street numbers)
    """
    cleaned = normalize_text(address)
    raw_tokens = cleaned.split()
    expanded = expand_tokens(raw_tokens, ADDRESS_ABBREVIATIONS)
    norm_address = " ".join(expanded)

    # Extract numeric tokens (postal codes, building numbers)
    numbers = [tok for tok in expanded if tok.isdigit() and len(tok) >= 2]

    return norm_address, expanded, numbers


def normalize_country(country: str) -> str:
    """
    Normalizes country string while preserving open-set country labels.
    """
    cleaned = normalize_text(country)
    return COUNTRY_MAP.get(cleaned, cleaned)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies full preprocessing pipeline to a raw source dataframe.
    Input must contain: entity_id, business_name, business_address, country.
    Adds parsed columns in-place or returns augmented dataframe.
    """
    df = df.copy()

    norm_names = []
    core_names = []
    name_tokens = []

    norm_addrs = []
    addr_tokens = []
    addr_numbers = []

    norm_countries = []

    names = df["business_name"].astype(str).tolist() if "business_name" in df.columns else [""] * len(df)
    addrs = df["business_address"].astype(str).tolist() if "business_address" in df.columns else [""] * len(df)
    countries = df["country"].astype(str).tolist() if "country" in df.columns else [""] * len(df)

    for b_name, b_addr, b_country in zip(names, addrs, countries):
        n_name, c_name, n_toks = normalize_name(b_name)
        norm_names.append(n_name)
        core_names.append(c_name)
        name_tokens.append(n_toks)

        n_addr, a_toks, a_nums = normalize_address(b_addr)
        norm_addrs.append(n_addr)
        addr_tokens.append(a_toks)
        addr_numbers.append(a_nums)

        norm_countries.append(normalize_country(b_country))

    df["norm_name"] = norm_names
    df["core_name"] = core_names
    df["name_tokens"] = name_tokens
    df["norm_address"] = norm_addrs
    df["address_tokens"] = addr_tokens
    df["address_numbers"] = addr_numbers
    df["norm_country"] = norm_countries

    return df
