#!/usr/bin/env python3
"""
extend_tokenizer.py — Extend MTLTokenizer with Telugu (and optionally Kannada/Bengali) graphemes.

Phase 1 of the Indic TTS plan: adds new script characters and language tags
to the existing 2454-token vocabulary.

Usage:
  python scripts/extend_tokenizer.py --languages te
  python scripts/extend_tokenizer.py --languages te kn bn
  python scripts/extend_tokenizer.py --languages te --output extended_tokenizer.json

What this does:
  1. Downloads the original tokenizer from HuggingFace
  2. Adds language tag(s): [te], [kn], [bn]
  3. Adds script-specific graphemes (vowels, consonants, vowel signs, marks)
  4. De-duplicates against existing vocab
  5. Creates Brahmic cross-script initialization map (Telugu→Devanagari, etc.)
  6. Saves extended tokenizer + init map
"""

import argparse
import json
import os
from pathlib import Path


# ═══════════════════════════════════════════════════════════
# Telugu graphemes (U+0C00 – U+0C7F)
# ═══════════════════════════════════════════════════════════
TELUGU_VOWELS = list("అఆఇఈఉఊఋఌఎఏఐఒఓఔ")
TELUGU_CONSONANTS = list("కఖగఘఙచఛజఝఞటఠడఢణతథదధనపఫబభమయరఱలళవశషసహ")
TELUGU_VOWEL_SIGNS = list("ాిీుూృెేైొోౌ")
TELUGU_MARKS = list("ంఃా్")  # anusvara, visarga, virama
TELUGU_DIGITS = list("౦౧౨౩౪౫౬౭౮౯")

# De-dup within the definition (some chars repeat across categories)
_te_all = set(TELUGU_VOWELS + TELUGU_CONSONANTS + TELUGU_VOWEL_SIGNS + TELUGU_MARKS + TELUGU_DIGITS)
TELUGU_CHARS = sorted(_te_all, key=lambda c: ord(c))


# ═══════════════════════════════════════════════════════════
# Kannada graphemes (U+0C80 – U+0CFF)
# ═══════════════════════════════════════════════════════════
KANNADA_VOWELS = list("ಅಆಇಈಉಊಋಎಏಐಒಓಔ")
KANNADA_CONSONANTS = list("ಕಖಗಘಙಚಛಜಝಞಟಠಡಢಣತಥದಧನಪಫಬಭಮಯರಲಳವಶಷಸಹ")
KANNADA_VOWEL_SIGNS = list("ಾಿೀುೂೃೆೇೈೊೋೌ")
KANNADA_MARKS = list("ಂಃ್")
KANNADA_DIGITS = list("೦೧೨೩೪೫೬೭೮೯")

_kn_all = set(KANNADA_VOWELS + KANNADA_CONSONANTS + KANNADA_VOWEL_SIGNS + KANNADA_MARKS + KANNADA_DIGITS)
KANNADA_CHARS = sorted(_kn_all, key=lambda c: ord(c))


# ═══════════════════════════════════════════════════════════
# Bengali graphemes (U+0980 – U+09FF)
# ═══════════════════════════════════════════════════════════
BENGALI_VOWELS = list("অআইঈউঊঋএঐওঔ")
BENGALI_CONSONANTS = list("কখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহড়ঢ়য়")
BENGALI_VOWEL_SIGNS = list("ািীুূৃেৈোৌ")
BENGALI_MARKS = list("ংঃ্")
BENGALI_DIGITS = list("০১২৩৪৫৬৭৮৯")

_bn_all = set(BENGALI_VOWELS + BENGALI_CONSONANTS + BENGALI_VOWEL_SIGNS + BENGALI_MARKS + BENGALI_DIGITS)
BENGALI_CHARS = sorted(_bn_all, key=lambda c: ord(c))


# ═══════════════════════════════════════════════════════════
# Brahmic cross-script initialization map
# Maps each new character to its Devanagari phonetic equivalent
# for warm-starting T3 embeddings
# ═══════════════════════════════════════════════════════════

# Telugu → Devanagari (phonetic equivalents via Unicode offset)
# Telugu block starts at U+0C00, Devanagari at U+0900, offset = 0x0300
TELUGU_TO_DEVANAGARI = {
    # Vowels
    "అ": "अ", "ఆ": "आ", "ఇ": "इ", "ఈ": "ई", "ఉ": "उ", "ఊ": "ऊ",
    "ఋ": "ऋ", "ఌ": "ऌ", "ఎ": "ए", "ఏ": "ऐ", "ఐ": "ऐ",
    "ఒ": "ओ", "ఓ": "औ", "ఔ": "औ",
    # Consonants
    "క": "क", "ఖ": "ख", "గ": "ग", "ఘ": "घ", "ఙ": "ङ",
    "చ": "च", "ఛ": "छ", "జ": "ज", "ఝ": "झ", "ఞ": "ञ",
    "ట": "ट", "ఠ": "ठ", "డ": "ड", "ఢ": "ढ", "ణ": "ण",
    "త": "त", "థ": "थ", "ద": "द", "ధ": "ध", "న": "न",
    "ప": "प", "ఫ": "फ", "బ": "ब", "భ": "भ", "మ": "म",
    "య": "य", "ర": "र", "ఱ": "र", "ల": "ल", "ళ": "ळ",
    "వ": "व", "శ": "श", "ష": "ष", "స": "स", "హ": "ह",
    # Vowel signs
    "ా": "ा", "ి": "ि", "ీ": "ी", "ు": "ु", "ూ": "ू", "ృ": "ृ",
    "ె": "े", "ే": "ै", "ై": "ै", "ొ": "ो", "ో": "ौ", "ౌ": "ौ",
    # Marks
    "ం": "ं", "ః": "ः", "్": "्",
}

KANNADA_TO_DEVANAGARI = {
    # Vowels
    "ಅ": "अ", "ಆ": "आ", "ಇ": "इ", "ಈ": "ई", "ಉ": "उ", "ಊ": "ऊ",
    "ಋ": "ऋ", "ಎ": "ए", "ಏ": "ऐ", "ಐ": "ऐ",
    "ಒ": "ओ", "ಓ": "औ", "ಔ": "औ",
    # Consonants
    "ಕ": "क", "ಖ": "ख", "ಗ": "ग", "ಘ": "घ", "ಙ": "ङ",
    "ಚ": "च", "ಛ": "छ", "ಜ": "ज", "ಝ": "झ", "ಞ": "ञ",
    "ಟ": "ट", "ಠ": "ठ", "ಡ": "ड", "ಢ": "ढ", "ಣ": "ण",
    "ತ": "त", "ಥ": "थ", "ದ": "द", "ಧ": "ध", "ನ": "न",
    "ಪ": "प", "ಫ": "फ", "ಬ": "ब", "ಭ": "भ", "ಮ": "म",
    "ಯ": "य", "ರ": "र", "ಲ": "ल", "ಳ": "ळ",
    "ವ": "व", "ಶ": "श", "ಷ": "ष", "ಸ": "स", "ಹ": "ह",
    # Vowel signs
    "ಾ": "ा", "ಿ": "ि", "ೀ": "ी", "ು": "ु", "ೂ": "ू", "ೃ": "ृ",
    "ೆ": "े", "ೇ": "ै", "ೈ": "ै", "ೊ": "ो", "ೋ": "ौ", "ೌ": "ौ",
    # Marks
    "ಂ": "ं", "ಃ": "ः", "್": "्",
}

BENGALI_TO_DEVANAGARI = {
    # Vowels
    "অ": "अ", "আ": "आ", "ই": "इ", "ঈ": "ई", "উ": "उ", "ঊ": "ऊ",
    "ঋ": "ऋ", "এ": "ए", "ঐ": "ऐ", "ও": "ओ", "ঔ": "औ",
    # Consonants
    "ক": "क", "খ": "ख", "গ": "ग", "ঘ": "घ", "ঙ": "ङ",
    "চ": "च", "ছ": "छ", "জ": "ज", "ঝ": "झ", "ঞ": "ञ",
    "ট": "ट", "ঠ": "ठ", "ড": "ড", "ঢ": "ढ", "ণ": "ण",
    "ত": "त", "থ": "थ", "দ": "द", "ধ": "ध", "ন": "न",
    "প": "प", "ফ": "फ", "ব": "ब", "ভ": "भ", "ম": "म",
    "য": "य", "র": "र", "ল": "ल", "শ": "श", "ষ": "ष", "স": "स", "হ": "ह",
    "ড়": "ड़", "ঢ়": "ढ़", "য়": "य",
    # Vowel signs
    "া": "ा", "ি": "ि", "ী": "ी", "ু": "ु", "ূ": "ू", "ৃ": "ृ",
    "ে": "े", "ৈ": "ै", "ো": "ो", "ৌ": "ौ",
    # Marks
    "ং": "ं", "ঃ": "ः", "্": "्",
}


# ═══════════════════════════════════════════════════════════
# Tamil graphemes (U+0B80 – U+0BFF)
# ═══════════════════════════════════════════════════════════
TAMIL_VOWELS = list("அஆஇஈஉஊஎஏஐஒஓஔ")
TAMIL_CONSONANTS = list("கஙசஞடணதநபமயரலவழளறன")
TAMIL_GRANTHA = list("ஜஷஸஹ")  # Grantha consonants (loanwords)
TAMIL_VOWEL_SIGNS = list("ாிீுூெேைொோௌ")
TAMIL_MARKS = list("ஂஃ்")  # anusvara, visarga, virama
TAMIL_DIGITS = list("௦௧௨௩௪௫௬௭௮௯")

_ta_all = set(TAMIL_VOWELS + TAMIL_CONSONANTS + TAMIL_GRANTHA + TAMIL_VOWEL_SIGNS + TAMIL_MARKS + TAMIL_DIGITS)
TAMIL_CHARS = sorted(_ta_all, key=lambda c: ord(c))


# ═══════════════════════════════════════════════════════════
# Malayalam graphemes (U+0D00 – U+0D7F)
# ═══════════════════════════════════════════════════════════
MALAYALAM_VOWELS = list("അആഇഈഉഊഋഎഏഐഒഓഔ")
MALAYALAM_CONSONANTS = list("കഖഗഘങചഛജഝഞടഠഡഢണതഥദധനപഫബഭമയരലളവശഷസഹ")
MALAYALAM_VOWEL_SIGNS = list("ാിീുൂൃെേൈൊോൌ")
MALAYALAM_MARKS = list("ംഃ്")
MALAYALAM_DIGITS = list("൦൧൨൩൪൫൬൭൮൯")

_ml_all = set(MALAYALAM_VOWELS + MALAYALAM_CONSONANTS + MALAYALAM_VOWEL_SIGNS + MALAYALAM_MARKS + MALAYALAM_DIGITS)
MALAYALAM_CHARS = sorted(_ml_all, key=lambda c: ord(c))


# ═══════════════════════════════════════════════════════════
# Gujarati graphemes (U+0A80 – U+0AFF)
# ═══════════════════════════════════════════════════════════
GUJARATI_VOWELS = list("અઆઇઈઉઊઋએઐઓઔ")
GUJARATI_CONSONANTS = list("કખગઘઙચછજઝઞટઠડઢણતથદધનપફબભમયરલળવશષસહ")
GUJARATI_VOWEL_SIGNS = list("ાિીુૂૃેૈોૌ")
GUJARATI_MARKS = list("ંઃ્")
GUJARATI_DIGITS = list("૦૧૨૩૪૫૬૭૮૯")

_gu_all = set(GUJARATI_VOWELS + GUJARATI_CONSONANTS + GUJARATI_VOWEL_SIGNS + GUJARATI_MARKS + GUJARATI_DIGITS)
GUJARATI_CHARS = sorted(_gu_all, key=lambda c: ord(c))


# ═══════════════════════════════════════════════════════════
# Marathi — uses Devanagari script (same as Hindi)
# No new graphemes needed, just the language tag [mr]
# ═══════════════════════════════════════════════════════════
MARATHI_CHARS = []  # All chars already in Devanagari vocab


# ═══════════════════════════════════════════════════════════
# Brahmic cross-script initialization maps (continued)
# ═══════════════════════════════════════════════════════════

TAMIL_TO_DEVANAGARI = {
    # Vowels
    "அ": "अ", "ஆ": "आ", "இ": "इ", "ஈ": "ई", "உ": "उ", "ஊ": "ऊ",
    "எ": "ए", "ஏ": "ऐ", "ஐ": "ऐ", "ஒ": "ओ", "ஓ": "औ", "ஔ": "औ",
    # Consonants
    "க": "क", "ங": "ङ", "ச": "च", "ஞ": "ञ", "ட": "ट", "ண": "ण",
    "த": "त", "ந": "न", "ப": "प", "ம": "म",
    "ய": "य", "ர": "र", "ல": "ल", "வ": "व",
    "ழ": "ळ", "ள": "ळ", "ற": "र", "ன": "न",
    # Grantha
    "ஜ": "ज", "ஷ": "ष", "ஸ": "स", "ஹ": "ह",
    # Vowel signs
    "ா": "ा", "ி": "ि", "ீ": "ी", "ு": "ु", "ூ": "ू",
    "ெ": "े", "ே": "ै", "ை": "ै", "ொ": "ो", "ோ": "ौ", "ௌ": "ौ",
    # Marks
    "ஂ": "ं", "ஃ": "ः", "்": "्",
}

MALAYALAM_TO_DEVANAGARI = {
    # Vowels
    "അ": "अ", "ആ": "आ", "ഇ": "इ", "ഈ": "ई", "ഉ": "उ", "ഊ": "ऊ",
    "ഋ": "ऋ", "എ": "ए", "ഏ": "ऐ", "ഐ": "ऐ",
    "ഒ": "ओ", "ഓ": "औ", "ഔ": "औ",
    # Consonants
    "ക": "क", "ഖ": "ख", "ഗ": "ग", "ഘ": "घ", "ങ": "ङ",
    "ച": "च", "ഛ": "छ", "ജ": "ज", "ഝ": "झ", "ഞ": "ञ",
    "ട": "ट", "ഠ": "ठ", "ഡ": "ड", "ഢ": "ढ", "ണ": "ण",
    "ത": "त", "ഥ": "थ", "ദ": "द", "ധ": "ध", "ന": "न",
    "പ": "प", "ഫ": "फ", "ബ": "ब", "ഭ": "भ", "മ": "म",
    "യ": "य", "ര": "र", "ല": "ल", "ള": "ळ",
    "വ": "व", "ശ": "श", "ഷ": "ष", "സ": "स", "ഹ": "ह",
    # Vowel signs
    "ാ": "ा", "ി": "ि", "ീ": "ी", "ു": "ु", "ൂ": "ू", "ൃ": "ृ",
    "െ": "े", "േ": "ै", "ൈ": "ै", "ൊ": "ो", "ോ": "ौ", "ൌ": "ौ",
    # Marks
    "ം": "ं", "ഃ": "ः", "്": "्",
}

GUJARATI_TO_DEVANAGARI = {
    # Vowels
    "અ": "अ", "આ": "आ", "ઇ": "इ", "ઈ": "ई", "ઉ": "उ", "ઊ": "ऊ",
    "ઋ": "ऋ", "એ": "ए", "ઐ": "ऐ", "ઓ": "ओ", "ઔ": "औ",
    # Consonants
    "ક": "क", "ખ": "ख", "ગ": "ग", "ઘ": "घ", "ઙ": "ङ",
    "ચ": "च", "છ": "छ", "જ": "ज", "ઝ": "झ", "ઞ": "ञ",
    "ટ": "ट", "ઠ": "ठ", "ડ": "ड", "ઢ": "ढ", "ણ": "ण",
    "ત": "त", "થ": "थ", "દ": "द", "ધ": "ध", "ન": "न",
    "પ": "प", "ફ": "फ", "બ": "ब", "ભ": "भ", "મ": "म",
    "ય": "य", "ર": "र", "લ": "ल", "ળ": "ळ",
    "વ": "व", "શ": "श", "ષ": "ष", "સ": "स", "હ": "ह",
    # Vowel signs
    "ા": "ा", "િ": "ि", "ી": "ी", "ુ": "ु", "ૂ": "ू", "ૃ": "ृ",
    "ે": "े", "ૈ": "ै", "ો": "ो", "ૌ": "ौ",
    # Marks
    "ં": "ं", "ઃ": "ः", "્": "्",
}

MARATHI_TO_DEVANAGARI = {}  # Marathi uses Devanagari — no mapping needed


LANG_CHARS = {
    "te": TELUGU_CHARS,
    "kn": KANNADA_CHARS,
    "bn": BENGALI_CHARS,
    "ta": TAMIL_CHARS,
    "ml": MALAYALAM_CHARS,
    "mr": MARATHI_CHARS,
    "gu": GUJARATI_CHARS,
}

LANG_INIT_MAP = {
    "te": TELUGU_TO_DEVANAGARI,
    "kn": KANNADA_TO_DEVANAGARI,
    "bn": BENGALI_TO_DEVANAGARI,
    "ta": TAMIL_TO_DEVANAGARI,
    "ml": MALAYALAM_TO_DEVANAGARI,
    "mr": MARATHI_TO_DEVANAGARI,
    "gu": GUJARATI_TO_DEVANAGARI,
}

LANG_NAMES = {
    "te": "Telugu",
    "kn": "Kannada",
    "bn": "Bengali",
    "ta": "Tamil",
    "ml": "Malayalam",
    "mr": "Marathi",
    "gu": "Gujarati",
}


def main():
    parser = argparse.ArgumentParser(description="Extend MTLTokenizer with Indic graphemes")
    parser.add_argument("--languages", nargs="+", default=["te"],
                        choices=["te", "kn", "bn", "ta", "ml", "mr", "gu"],
                        help="Languages to add (default: te)")
    parser.add_argument("--base_tokenizer", type=str, default=None,
                        help="Path to existing extended tokenizer to build on (for incremental extension). "
                             "If not provided, downloads original from HuggingFace.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output tokenizer JSON path (default: data/tokenizer/extended_tokenizer.json)")
    parser.add_argument("--init_map_output", type=str, default=None,
                        help="Output init map JSON (default: data/tokenizer/brahmic_init_map.json)")
    args = parser.parse_args()

    output_path = args.output or "data/tokenizer/extended_tokenizer.json"
    init_map_path = args.init_map_output or "data/tokenizer/brahmic_init_map.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # ─── Load base tokenizer ───
    from tokenizers import Tokenizer, AddedToken
    from huggingface_hub import snapshot_download

    if args.base_tokenizer:
        print(f"Loading base tokenizer from {args.base_tokenizer}")
        tokenizer = Tokenizer.from_file(args.base_tokenizer)
    else:
        print("Downloading original tokenizer from HuggingFace...")
        ckpt_dir = Path(
            snapshot_download(
                repo_id="ResembleAI/chatterbox",
                repo_type="model",
                revision="main",
                allow_patterns=["grapheme_mtl_merged_expanded_v1.json"],
                token=os.getenv("HF_TOKEN"),
            )
        )
        original_path = ckpt_dir / "grapheme_mtl_merged_expanded_v1.json"
        tokenizer = Tokenizer.from_file(str(original_path))

    existing_vocab = tokenizer.get_vocab()
    original_size = len(existing_vocab)

    print(f"Original vocab size: {original_size}")

    # ─── Check existing Devanagari coverage ───
    devanagari_chars = {
        char: token_id for char, token_id in existing_vocab.items()
        if len(char) == 1 and '\u0900' <= char <= '\u097F'
    }
    print(f"Existing Devanagari characters: {len(devanagari_chars)}")

    # ─── Collect new tokens to add ───
    all_new_tokens = []
    all_new_lang_tags = []
    per_lang_new = {}

    for lang in args.languages:
        lang_tag = f"[{lang}]"
        chars = LANG_CHARS[lang]

        # De-duplicate against existing vocab
        new_chars = [c for c in chars if c not in existing_vocab]
        new_tag = lang_tag if lang_tag not in existing_vocab else None

        per_lang_new[lang] = {
            "tag": lang_tag,
            "tag_is_new": new_tag is not None,
            "total_chars": len(chars),
            "new_chars": len(new_chars),
            "already_exists": len(chars) - len(new_chars),
        }

        if new_tag:
            all_new_lang_tags.append(new_tag)
        all_new_tokens.extend(new_chars)

        print(f"\n{LANG_NAMES[lang]} ({lang}):")
        print(f"  Total graphemes defined: {len(chars)}")
        print(f"  New (not in vocab):      {len(new_chars)}")
        print(f"  Already in vocab:        {len(chars) - len(new_chars)}")
        print(f"  Language tag [{lang}]:    {'NEW' if new_tag else 'already exists'}")

    # Final de-dup across languages (some combining marks might overlap)
    seen = set()
    unique_new_tokens = []
    for t in all_new_tokens:
        if t not in seen:
            seen.add(t)
            unique_new_tokens.append(t)

    print(f"\n{'='*50}")
    print(f"Total new tokens to add: {len(all_new_lang_tags)} tags + {len(unique_new_tokens)} chars = {len(all_new_lang_tags) + len(unique_new_tokens)}")
    print(f"New vocab size: {original_size} → {original_size + len(all_new_lang_tags) + len(unique_new_tokens)}")

    # ─── Add tokens to tokenizer ───
    # Add language tags as special tokens (won't be split by pre-tokenizer)
    if all_new_lang_tags:
        special_tokens = [AddedToken(t, special=True) for t in all_new_lang_tags]
        tokenizer.add_special_tokens(special_tokens)
        print(f"\nAdded special tokens: {all_new_lang_tags}")

    # Add grapheme characters as regular tokens
    if unique_new_tokens:
        regular_tokens = [AddedToken(t, special=False) for t in unique_new_tokens]
        tokenizer.add_tokens(regular_tokens)
        print(f"Added {len(unique_new_tokens)} grapheme tokens")

    # Verify new size
    new_vocab = tokenizer.get_vocab()
    new_size = len(new_vocab)
    print(f"\nFinal vocab size: {new_size} (was {original_size}, +{new_size - original_size})")

    # ─── Save extended tokenizer ───
    tokenizer.save(output_path)
    print(f"\nSaved extended tokenizer → {output_path}")

    # ─── Build Brahmic initialization map ───
    # Maps: new_token_id → devanagari_token_id (for embedding warm-start)
    init_map = {}
    init_map_readable = {}
    unmapped = []

    for lang in args.languages:
        lang_init = LANG_INIT_MAP[lang]
        for new_char, deva_char in lang_init.items():
            if new_char in new_vocab and deva_char in existing_vocab:
                new_id = new_vocab[new_char]
                deva_id = existing_vocab[deva_char]
                init_map[str(new_id)] = deva_id
                init_map_readable[f"{new_char} ({lang})"] = f"{deva_char} (hi) [id {deva_id}→{new_id}]"
            elif new_char in new_vocab:
                unmapped.append(f"{new_char} ({lang}) → {deva_char} (NOT in Devanagari vocab)")

    print(f"\nBrahmic init map: {len(init_map)} mappings")
    if unmapped:
        print(f"Unmapped (will use random init): {len(unmapped)}")
        for u in unmapped[:10]:
            print(f"  {u}")

    # Save init map
    init_data = {
        "description": "Brahmic cross-script embedding initialization map",
        "format": "new_token_id (str) → devanagari_token_id (int)",
        "original_vocab_size": original_size,
        "extended_vocab_size": new_size,
        "languages_added": args.languages,
        "n_mappings": len(init_map),
        "map": init_map,
        "readable": init_map_readable,
    }
    with open(init_map_path, "w", encoding="utf-8") as f:
        json.dump(init_data, f, ensure_ascii=False, indent=2)
    print(f"Saved init map → {init_map_path}")

    # ─── Verify encoding works ───
    print(f"\n{'='*50}")
    print("VERIFICATION")
    print(f"{'='*50}")

    test_cases = []
    if "te" in args.languages:
        test_cases.append(("te", "నమస్కారం"))
        test_cases.append(("te", "నేను office కి వెళ్తున్నాను"))  # code-mix
    if "kn" in args.languages:
        test_cases.append(("kn", "ನಮಸ್ಕಾರ"))
    if "bn" in args.languages:
        test_cases.append(("bn", "নমস্কার"))
    if "ta" in args.languages:
        test_cases.append(("ta", "வணக்கம்"))
    if "ml" in args.languages:
        test_cases.append(("ml", "നമസ്കാരം"))
    if "mr" in args.languages:
        test_cases.append(("mr", "नमस्कार"))  # Devanagari — should all resolve
    if "gu" in args.languages:
        test_cases.append(("gu", "નમસ્તે"))

    for lang, text in test_cases:
        # Simulate MTLTokenizer.encode() logic
        from unicodedata import normalize
        txt = text.lower()
        txt = normalize("NFKD", txt)
        txt = f"[{lang}]{txt}"
        txt = txt.replace(" ", "[SPACE]")
        encoded = tokenizer.encode(txt)
        token_ids = encoded.ids
        tokens = encoded.tokens

        has_unk = 2 in token_ids  # UNK token is id 2
        status = "✗ HAS UNK" if has_unk else "✓ OK"

        print(f"\n  [{lang}] \"{text}\"")
        print(f"  {status}  IDs: {token_ids[:20]}{'...' if len(token_ids) > 20 else ''}")
        print(f"  Tokens: {tokens[:20]}{'...' if len(tokens) > 20 else ''}")

    # ─── Summary ───
    print(f"\n{'='*50}")
    print("PHASE 1 COMPLETE")
    print(f"{'='*50}")
    print(f"  Extended tokenizer: {output_path}")
    print(f"  Brahmic init map:   {init_map_path}")
    print(f"  Vocab: {original_size} → {new_size} (+{new_size - original_size})")
    print(f"\nNext steps:")
    print(f"  1. python scripts/extend_t3_embeddings.py --tokenizer {output_path} --init_map {init_map_path}")
    print(f"  2. Update SUPPORTED_LANGUAGES in src/chatterbox/mtl_tts.py")
    print(f"  3. Download + preprocess Telugu data")
    print(f"  4. Round 2 LoRA training with Telugu + Hindi + English")


if __name__ == "__main__":
    main()
