"""Character identification for the Caitanya-caritāmṛta corpus.

spaCy's `en_core_web_sm` was trained on contemporary English newswire, and its
entity *labels* are not reliable on devotional Sanskrit/Bengali prose. Measured
on ~500k characters of Ādi-līlā:

    Kṛṣṇa                PERSON 275  +  ORG 188
    Caitanya             GPE    236
    Caitanya Mahāprabhu  PERSON  87
    Mahāprabhu           PERSON 112
    Nityānanda           ORG     42
    Śrīmad-Bhāgavatam    PERSON  61   <- a scripture, not a person
    Vedānta              PERSON 132   <- a philosophy, not a person

So filtering on `label_ == "PERSON"` both loses the protagonist and admits
scriptures. This module instead pools candidates across all entity labels and
resolves them by name, using:

  1. honorific stripping   -- "Śrīla Rūpa Gosvāmī" -> "Rūpa"
  2. an explicit alias map -- "Mahāprabhu" -> "Caitanya"
  3. a non-character stoplist for scriptures, places and abstractions

Limitation worth stating in any write-up: this is lexical resolution, not
coreference. Pronouns ("He", "the Lord") are not attributed, so mention counts
are lower bounds, and epithets such as "the Supreme Personality of Godhead" are
excluded rather than resolved to Kṛṣṇa.
"""

import re
import unicodedata

# Entity labels worth harvesting. The label itself is discarded -- only the
# surface form matters -- but restricting to these avoids DATE/CARDINAL noise.
USEFUL_LABELS = {"PERSON", "ORG", "GPE", "NORP", "LOC", "FAC", "WORK_OF_ART"}

# Titles and honorifics that prefix or suffix a name without changing referent.
_HONORIFIC_PREFIX = re.compile(
    r"^(the\s+)?(Śrī|Sri|Śrīla|Srila|Śrīpāda|Sripada|Śrīmatī|Srimati|"
    r"Lord|Sri̇|Mahā|Mahārāja|Maharaja|King|Queen|Śrīmad|Śrī\s+Śrī)\s+",
    re.IGNORECASE,
)

# Possessive forms. NER emits "Rādhārāṇī's" as its own entity, which then
# competed with "Rādhārāṇī" as a separate node.
_POSSESSIVE = re.compile(r"[’'’]s$", re.IGNORECASE)
_HONORIFIC_SUFFIX = re.compile(
    r"\s+(Prabhu|Gosvāmī|Gosvami|Goswami|Gosāñi|Ṭhākura|Thakura|Ācārya|Acarya|"
    r"Mahāśaya|Paṇḍita|Pandita|Dāsa|Dasa|Devī|Devi|Mahārāja|Maharaja|Purī|Bhaṭṭa|"
    r"Himself|Herself|Themselves|Itself)$",
    re.IGNORECASE,
)

# Scriptures, philosophies, places and abstractions that NER labels as people.
NON_CHARACTERS = {
    # scriptures and texts
    "bhagavad", "bhagavad-gītā", "gītā", "śrīmad-bhāgavatam", "bhāgavatam",
    "vedānta", "vedas", "veda", "upaniṣads", "purāṇas", "purāṇa", "mahābhārata",
    "rāmāyaṇa", "vedānta-sūtra", "śārīraka", "bhakti-rasāmṛta-sindhu",
    "caitanya-caritāmṛta", "caitanya-bhāgavata", "pañcarātras", "brahma-saṁhitā",
    "vivarta-vāda", "bhakti-ratnākara", "hari-vaṁśa", "śikṣāṣṭaka",
    # abstractions and epithets
    "godhead", "the supreme personality of godhead", "supreme personality of godhead",
    "the personality of godhead", "personality of godhead", "the absolute truth",
    "absolute truth", "brahman", "paramātmā", "supersoul", "the supersoul",
    "bhagavān", "supreme lord",
    "the supreme lord", "supreme brahman", "lotus", "dear sir", "hare kṛṣṇa",
    "mahā-mantra", "holy name", "the holy name",
    # groups, demonyms and categories, not individuals. Singular forms matter:
    # NER tags "Muslim", "Buddhist", "Hindu", "Bengali" as entities, and they
    # then surface in the character ranking as if they were people.
    "vaiṣṇava", "vaiṣṇavas", "māyāvādī", "māyāvādīs", "gosvāmīs", "hindu",
    "hindus", "european", "europeans", "american", "americans", "brāhmaṇa",
    "brāhmaṇas", "muslim", "muslims", "yavana", "yavanas", "kazis",
    "sannyāsī", "sannyāsīs", "gopī", "gopīs", "demigod", "demigods",
    "buddhist", "buddhists", "bengali", "bengalis", "christian", "christians",
    "jain", "jains", "śūdra", "śūdras", "kṣatriya", "kṣatriyas", "vaiśya",
    "nawab", "bhaṭṭathāris", "bhaṭṭathāri",
    # calendrical and ritual terms NER mistakes for names
    "ekādaśī", "dvādaśī", "janmāṣṭamī", "ratha-yātrā", "dola-yātrā",
    # līlā names: they appear in ~5,500 inline citations ("Cc. Madhya 22.83").
    # extract.py strips those, but keep these here as a second line of defence.
    "ādi", "madhya", "antya", "ādi-līlā", "madhya-līlā", "antya-līlā",
    # Bare honorifics. These are stripped when they SUFFIX a name ("Haridāsa
    # Ṭhākura" -> "Haridāsa"), but NER also emits them standalone, where they
    # name no one. Listing them here does not affect the suffix-stripped forms,
    # which no longer contain the honorific by the time this set is consulted.
    "ṭhākura", "thakura", "śrīla", "srila", "śrīpāda", "sripada", "prabhu",
    "gosvāmī", "gosvami", "goswami", "gosāñi", "ācārya", "acarya", "paṇḍita",
    "pandita", "mahāśaya", "mahārāja", "maharaja", "bhaṭṭācārya", "bhattacharya",
    "miśra", "misra", "bhaṭṭa", "dāsa", "dasa", "devī", "devi", "purī", "puri",
    "śrīmatī", "gosvāmīs",
    # Titles and roles, not individuals.
    "king", "queen", "emperor", "prince", "governor", "minister", "the king",
    "the queen", "lord", "the lord", "sir", "mother", "father",
    # Objects and structures treated as entities by NER.
    "deity", "the deity", "deities", "ratha", "temple", "the temple", "car",
    # Further places.
    "govardhana hill", "govardhana", "south india", "india", "bengal",
    "jagannātha temple",
    # places and realms
    "vaikuṇṭha", "śvetadvīpa", "kṛṣṇaloka", "brahmaloka", "viṣṇuloka",
    "vṛndāvana", "navadvīpa", "jagannātha purī", "purī", "māyāpur", "bengal",
    "india", "europe", "the causal ocean", "causal ocean", "the kāraṇa ocean",
    "kāraṇa ocean", "ganges", "yamunā", "rādhā-kuṇḍa", "vārāṇasī", "benares",
    "katwa", "kulia", "remuṇā", "gayā", "prayāga", "mathurā", "dvārakā",
    "goloka", "vaikuṇṭhaloka", "pañca", "vedic", "orissa", "oriya", "jagannātha purī",
    "nīlācala", "śāntipura", "kāśī", "kānyakubja", "saptagrāma", "pānihāṭi",
    "vraja", "vraja lord", "vṛndāvana-dhāma",
    # Pronouns and fragments NER emits as entities. These only surfaced once the
    # cast filter was widened to the top 50%.
    "them", "they", "him", "her", "himself", "herself",
    "amṛta",     # fragment of "Caitanya-caritāmṛta", and the common noun (nectar)
    "siṁha",     # name fragment; no consistent referent in this corpus
    "kulīna",    # from "kulīna-grāma", a category of residents, not a person
    # A 19th-century commentator quoted in the apparatus, not a figure in the
    # 16th-century narrative.
    "bhaktivinoda", "bhaktivinoda ṭhākura",
}

# Variants that unambiguously denote the same figure. Kept deliberately small:
# only merges that are certain in this text. Ambiguous names (Govinda, Gopāla,
# Rāmānanda) are left unmerged because they denote both the Lord and distinct
# human associates depending on context.
ALIASES = {
    "mahāprabhu": "Caitanya",
    "caitanya mahāprabhu": "Caitanya",
    "gaurāṅga": "Caitanya",
    "gaurasundara": "Caitanya",
    "gaurahari": "Caitanya",
    "nimāi": "Caitanya",
    "viśvambhara": "Caitanya",
    "gaura": "Caitanya",
    "kṛṣṇa caitanya": "Caitanya",
    "nityānanda prabhu": "Nityānanda",
    "nitāi": "Nityānanda",
    "balarāma": "Nityānanda",
    "advaita ācārya": "Advaita",
    "advaita prabhu": "Advaita",
    "sanātana": "Sanātana",
    "rūpa": "Rūpa",
    "haridāsa": "Haridāsa",
    "svarūpa dāmodara": "Svarūpa Dāmodara",
    "rāya rāmānanda": "Rāmānanda Rāya",
    "rāmānanda rāya": "Rāmānanda Rāya",
    # "Hari" is a name of Kṛṣṇa, not a separate figure. The corpus uses it that
    # way throughout: "nondifferent from Hari, the Supreme Lord"; "Some call Him
    # Hari"; "Lord Gaurāṅga, who is Śrī Hari Himself". Viṣṇu and Nārāyaṇa are
    # deliberately NOT merged -- in Gauḍīya theology those are distinct
    # expansions, whereas Hari is a direct synonym.
    "hari": "Kṛṣṇa",
    "śrī hari": "Kṛṣṇa",
    "hari-nāma": "Kṛṣṇa",
    # Verified splits of one person across two nodes. Counts in the corpus:
    #   Sārvabhauma 299, of which 258 are "Sārvabhauma Bhaṭṭācārya"
    #   Nārada       54, of which  34 are "Nārada Muni"
    #   Nanda        89, of which  47 are "Nanda Mahārāja"
    #   Śacī        144, no competing follower -- Śacī/Śacīmātā/Śacīdevī are
    #               all Caitanya's mother
    "sārvabhauma": "Sārvabhauma Bhaṭṭācārya",
    "nārada": "Nārada Muni",
    "nanda mahārāja": "Nanda",
    "śacī": "Śacīdevī",
    "śacīmātā": "Śacīdevī",
    "śacī-mātā": "Śacīdevī",
    # Murāri: 38 of 61 are "Murāri Gupta"; "Murāri Māhiti" (2) is a different
    # man and is protected below, so the bare form resolves to Gupta.
    "murāri": "Murāri Gupta",
}


# Full names whose trailing word looks like an honorific but actually
# distinguishes one person from another. Stripping it merges distinct figures.
#
# Verified against the corpus:
#   Dāmodara   409 = Svarūpa Dāmodara 341 + Dāmodara Paṇḍita 41 + others 27
#              ("Dāmodara Paṇḍita ... once unhesitatingly chastised the Lord";
#               his younger brother is Śaṅkara Paṇḍita — a different man from
#               Caitanya's secretary Svarūpa Dāmodara)
#   Rāmacandra 140 = Rāmacandra Purī 56 + Rāmacandra Khān 26 + Lord Rāmacandra 50
#              (a critical sannyāsī, the antagonist who plots against Haridāsa,
#               and the avatar — three figures, one node before this fix)
#   Gopīnātha  190 = Gopīnātha Ācārya 79 + Gopīnātha Paṭṭanāyaka 45 + Deity 65
PROTECTED_NAMES = {
    "dāmodara paṇḍita": "Dāmodara Paṇḍita",
    "śaṅkara paṇḍita": "Śaṅkara Paṇḍita",
    "gadādhara paṇḍita": "Gadādhara Paṇḍita",
    "gadādhara dāsa": "Gadādhara dāsa",
    # NOT protected: "Jagadānanda Paṇḍita" (103) and "Rāghava Paṇḍita" (67) have
    # no competing referent in this corpus -- nothing else follows either name --
    # so protecting them split one person into two nodes. They fall through to
    # normal suffix stripping and merge with their bare forms, as they should.
    "dāmodara dāsa": "Dāmodara dāsa",
    "rāmacandra purī": "Rāmacandra Purī",
    "rāmacandra khān": "Rāmacandra Khān",
    "gopīnātha ācārya": "Gopīnātha Ācārya",
    "gopīnātha paṭṭanāyaka": "Gopīnātha Paṭṭanāyaka",
    "mādhavendra purī": "Mādhavendra Purī",
    "paramānanda purī": "Paramānanda Purī",
    "īśvara purī": "Īśvara Purī",
    "brahmānanda bhāratī": "Brahmānanda Bhāratī",
    "keśava bhāratī": "Keśava Bhāratī",
    "haridāsa ṭhākura": "Haridāsa",
    "svarūpa dāmodara": "Svarūpa Dāmodara",
    "murāri māhiti": "Murāri Māhiti",
    "murāri paṇḍita": "Murāri Paṇḍita",
}


def canonicalise(surface):
    """Reduce an entity surface form to a canonical character name, or None."""
    name = unicodedata.normalize("NFC", surface).strip(" \"-.,;:!?()[]")
    name = _POSSESSIVE.sub("", name).strip(" '’")
    name = re.sub(r"\s+", " ", name)
    if not name or len(name) < 3:
        return None

    # Strip leading honorifics first ("Śrīla", "Lord"), since a protected name
    # may still carry one: "Śrī Rāmacandra Purī".
    previous = None
    while previous != name:
        previous = name
        name = _HONORIFIC_PREFIX.sub("", name).strip()

    # A protected name keeps its final word -- that word is what identifies the
    # person, not a title to be discarded.
    protected = PROTECTED_NAMES.get(name.lower())
    if protected:
        return protected

    # Otherwise strip trailing honorifics ("Rūpa Gosvāmī" -> "Rūpa").
    previous = None
    while previous != name:
        previous = name
        name = _HONORIFIC_PREFIX.sub("", name).strip()
        name = _HONORIFIC_SUFFIX.sub("", name).strip()
    if not name or len(name) < 3:
        return None

    key = name.lower()
    if key in ALIASES:
        return ALIASES[key]
    if key in NON_CHARACTERS:
        return None
    # Reject anything that still looks like a title or a common-noun phrase.
    if name[0].islower() or key.startswith("the "):
        return None
    if len(name.split()) > 4:
        return None
    return name
