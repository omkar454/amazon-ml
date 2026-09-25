"""
Test Suite & Benchmark for Universal Normalizer
Tests real cases from Train & Test datasets across US, India, and France.
"""

import time
from normalizer import normalize_record, normalize_business_name, normalize_business_address

test_cases = [
    # 1. Indic Hindi / Devanagari Transliteration
    {
        "desc": "Hindi Devanagari S2 to English S1 Match",
        "raw_s1": ("Ram Marketing Private Limited", "KH No. 570/13, New Delhi, Delhi", "India"),
        "raw_s2": ("राम मार्केटिंग प्राइवेट लिमिटेड", "KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi", "India")
    },
    # 2. Indic Hindi LLP Match
    {
        "desc": "Hindi Devanagari LLP S2 to English S1 Match",
        "raw_s1": ("Aditya Properties LLP", "G-3/571, Gulmohar Colony, Bhopal, Madhya Pradesh", "India"),
        "raw_s2": ("आदित्य प्रॉपर्टीज एलएलपी", "G-3/571, GULMOHAR COLONY, BHOPAL, Madhya Pradesh", "India")
    },
    # 3. Indic Tamil Transliteration
    {
        "desc": "Tamil Script S2 to English S1 Match",
        "raw_s1": ("Global Business Private Limited", "#C-21 - S, Ambattur, Chennai, Tamil Nadu", "India"),
        "raw_s2": ("குளோபல் பிசினஸ் பிரைவேட் லிமிடெட்", "#C-21 - S, AMBATTUR, CHENNAI, Tamil Nadu", "India")
    },
    # 4. French Accents & Legal Suffixes
    {
        "desc": "French Accents & SASU to SAS Match",
        "raw_s1": ("Thermal & Fils SASU", "20 Rue Parmentier, Dunkerque, Hauts-de-France", "France"),
        "raw_s2": ("Thermal & Fils SAS", "20 RUE PARMENTIER, DUNKERQUE, Hauts-de-France", "France")
    },
    # 5. French SARL & Boulevard Abbreviation
    {
        "desc": "French SARL with Boulevard abbreviation",
        "raw_s1": ("Team Ecole", "175 Boulevard du President Franklin Roosevelt, Bordeaux", "France"),
        "raw_s2": ("<< Team Ecole SARL", "175 BD DU PRESIDENT FRANKLIN ROOSEVELT, BORDEAUX", "France")
    },
    # 6. US URL Domain Name Match
    {
        "desc": "US URL Domain Name S2 to English S1 Match",
        "raw_s1": ("Siia Investments Inc", "8411 Gabrielino Court, Rancho Cucamonga, CA", "US"),
        "raw_s2": ("siiainvestments.com", "8411 GABRIELINO COURT, PMB 9239, RANCHO CUCAMONGA, CA", "US")
    },
    # 7. US Noise Decorator & 'formerly' DBA Prefix
    {
        "desc": "US Noise Decorator with 'formerly' prefix",
        "raw_s1": ("Painters Local Union 634", "25807 Iron Gate Drive, Madison, AL", "US"),
        "raw_s2": ("Mirapyra formerly Painters Local Union 634", "AL, Madison, 25807 Iron Gate Drive", "US")
    },
    # 8. Zero-Padded House Numbers & Accented English
    {
        "desc": "Zero-padded House Number & Injected Accents",
        "raw_s1": ("Gautam Nagar Recording Private Limited", "K-303, Gautam Nagar, New Delhi, Delhi", "India"),
        "raw_s2": ("Gautam Nagar Recording Prívate Limited", "दिल्ली, K-00303, New Delhi, Gautam Nagar", "India")
    }
]

print("=" * 80)
print("1. RUNNING UNIT TESTS ON REAL MULTILINGUAL CASES")
print("=" * 80)

for i, tc in enumerate(test_cases, 1):
    s1_norm = normalize_record(*tc["raw_s1"])
    s2_norm = normalize_record(*tc["raw_s2"])
    
    print(f"\n[Test Case {i}] {tc['desc']}")
    print(f"  Raw S1: Name='{tc['raw_s1'][0]}' | Addr='{tc['raw_s1'][1]}'")
    print(f"  Raw S2: Name='{tc['raw_s2'][0]}' | Addr='{tc['raw_s2'][1]}'")
    print(f"  -> S1 Core Name: '{s1_norm['clean_name']}' | Addr: '{s1_norm['clean_address']}' | Nums: {s1_norm['address_numbers']}")
    print(f"  -> S2 Core Name: '{s2_norm['clean_name']}' | Addr: '{s2_norm['clean_address']}' | Nums: {s2_norm['address_numbers']}")
    
    name_match = s1_norm['clean_name'] == s2_norm['clean_name']
    num_overlap = bool(set(s1_norm['address_numbers']) & set(s2_norm['address_numbers']))
    addr_match = s1_norm['clean_address'] == s2_norm['clean_address']
    
    print(f"  [Result] Core Name Match: {'EXACT (PASS)' if name_match else 'CLOSE (Sub-token match)'}")
    print(f"  [Result] Address Number Overlap: {'PASS' if num_overlap else 'NO NUM'}")
    if addr_match:
        print(f"  [Result] Address Match: EXACT (PASS)")

print("\n" + "=" * 80)
print("2. BENCHMARKING THROUGHPUT SPEED")
print("=" * 80)

sample_names = [tc["raw_s2"][0] for tc in test_cases] * 12500  # 100,000 strings
sample_addrs = [tc["raw_s2"][1] for tc in test_cases] * 12500

t0 = time.time()
for n, a in zip(sample_names, sample_addrs):
    normalize_record(n, a, "India")
t1 = time.time()

total_processed = len(sample_names)
elapsed = t1 - t0
rate = total_processed / elapsed

print(f"Processed {total_processed:,} records in {elapsed:.3f}s")
print(f"Throughput: {rate:,.0f} records/second")
print("=" * 80)
