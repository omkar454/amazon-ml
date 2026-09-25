# Step 1: Normalization Specification & Implementation Plan

**Challenge:** Business Entity Resolution Challenge (Amazon ML Challenge 2026)  
**Component:** Data Preprocessing & Canonical Text Normalization  
**Scope:** Train and Test Splits (Source 1, Source 2, Source 3)  
**Supported Geographies:** United States (`US`), India (`India`), France (`France`) *(Open set)*

---

## 1. Objectives & Overview

In multi-source Entity Resolution, dirty records from independent sources exhibit severe noise, including:
1. Accented characters and OCR corruptions (`CÓNSULTANTS`, `Béque`, `Électricité`).
2. Synthesized decorators, URLs, domain names, hashtags, and registration IDs (`siiainvestments.com`, `#centraleducation`, `- 2067865001`).
3. Legal entity suffix inconsistencies (`Pvt Ltd` vs `Private Limited`, `LLC` vs `L.L.C.`, `SARL`, `SASU`).
4. Address component abbreviations, reordering, landmark references, and leading zero-padding (`St` vs `Street`, `K-00303` vs `K-303`).
5. Transliteration / native scripts in Indic entities (Hindi / Devanagari, Tamil, Kannada).

**Goal:** Transform raw `business_name` and `business_address` fields from all sources into canonical, comparable representations that maximize blocking recall and pairwise matching accuracy.

---

## 2. Architectural Pipeline

```
Raw Record (S1 / S2 / S3)
  ├── entity_id
  ├── business_name
  ├── business_address
  └── country
       │
       ▼
[Universal Normalizer Pipeline]
  ├── Phase 1: Unicode NFKD Decomposition & Accent Stripping
  ├── Phase 2: Name Decorator, URL & ID Cleaner
  ├── Phase 3: Legal Entity Suffix Normalizer & Stripper
  ├── Phase 4: Address Standardization & State Mapping
  └── Phase 5: Numeric Identifier Extractor
       │
       ▼
[Standardized Output Schema]
  ├── clean_name_core      (Core business name without legal suffixes)
  ├── clean_name_full      (Business name with standardized legal suffixes)
  ├── clean_address        (Standardized address string)
  ├── address_numbers      (Tuple/List of discrete numeric tokens)
  └── has_address          (Boolean flag: 1 if address present, 0 if empty)
```

---

## 3. Detailed Normalization Rules

### 3.1. Phase 1: Universal Unicode & Accent Stripping
- **NFKD Normalization**: Decomposes characters with diacritics into base characters + combining marks.
- Strips combining diacritical marks:
  - `é`, `è`, `ê`, `ë` $\rightarrow$ `e`
  - `á`, `à`, `â`, `ä` $\rightarrow$ `a`
  - `ó`, `ò`, `ô`, `ö` $\rightarrow$ `o`
  - `í`, `ì`, `î`, `ï` $\rightarrow$ `i`
  - `ú`, `ù`, `û`, `ü` $\rightarrow$ `u`
  - `ç` $\rightarrow$ `c`
  - `ñ` $\rightarrow$ `n`
- Replaces special conjunctions: `&` $\rightarrow$ `and`, `@` $\rightarrow$ `at`.

---

### 3.2. Phase 2: Business Name Cleaning
1. **Noise Prefix/Decorator Stripping**:
   - Strip leading/trailing symbols: `--`, `<<`, `>>`, `##`, `**`, `//`, `|`.
   - Strip prefix markers: `formerly ...`, `dba ...`, `d/b/a ...`, `ex ...`, `t/a ...`.
2. **Domain & URL Cleaning**:
   - Matches and strips web domain extensions: `.com`, `.org`, `.net`, `.in`, `.co.in`, `.fr`, `.io`, `.biz`, `www.`.
   - Example: `siiainvestments.com` $\rightarrow$ `siia investments`.
3. **Appended ID / Hashtag Cleaning**:
   - Matches and removes appended registration strings: `- 2067865001`, `#98825`.
   - Strips hashtags: `#centraleducation` $\rightarrow$ `central education`.

---

### 3.3. Phase 3: Legal Entity Suffix Standardization & Stripping

We produce two representations:
- **`clean_name_full`**: Standardizes all legal variations into a single canonical token.
- **`clean_name_core`**: Completely removes the legal suffix to isolate the pure brand name.

| Region | Raw Variations | Standardized (`clean_name_full`) | Core Name (`clean_name_core`) |
| :--- | :--- | :--- | :--- |
| **India** | `Private Limited`, `Pvt Ltd`, `P. Ltd`, `P Ltd`, `Pvt. Ltd.` | `pvt ltd` | *Stripped* |
| **India / UK** | `Public Limited`, `Pub Ltd`, `Limited`, `Ltd.` | `ltd` | *Stripped* |
| **India / Global** | `Limited Liability Partnership`, `LLP`, `L.L.P.` | `llp` | *Stripped* |
| **US** | `Limited Liability Company`, `LLC`, `L.L.C.`, `(LLC)` | `llc` | *Stripped* |
| **US** | `Incorporated`, `Inc.`, `Inc`, `(Inc)` | `inc` | *Stripped* |
| **US** | `Corporation`, `Corp.`, `Corp` | `corp` | *Stripped* |
| **US** | `Company`, `Co.`, `Co` | `co` | *Stripped* |
| **France** | `Société à Responsabilité Limitée`, `SARL`, `S.A.R.L.` | `sarl` | *Stripped* |
| **France** | `Société par Actions Simplifiée`, `SAS`, `SASU`, `S.A.S.` | `sas` | *Stripped* |
| **France** | `Entreprise Unipersonnelle à Responsabilité Limitée`, `EURL` | `eurl` | *Stripped* |
| **France** | `& Fils`, `et Fils`, `and Fils` | `et fils` | *Preserved / Canonical* |

---

### 3.4. Phase 4: Address Standardization

1. **Missing Value Imputation**: Missing or `NaN` address becomes `""` with `has_address = 0`.
2. **Street & Road Types**:
   - `street`, `st.` $\rightarrow$ `st`
   - `road`, `rd.` $\rightarrow$ `rd`
   - `avenue`, `ave.` $\rightarrow$ `ave`
   - `boulevard`, `blvd.` $\rightarrow$ `blvd`
   - `drive`, `dr.` $\rightarrow$ `dr`
   - `lane`, `ln.` $\rightarrow$ `ln`
   - `court`, `ct.` $\rightarrow$ `ct`
   - `highway`, `hwy.` $\rightarrow$ `hwy`
   - `circle`, `cir.` $\rightarrow$ `cir`
   - `parkway`, `pkwy.` $\rightarrow$ `pkwy`
3. **Unit & Floor Types**:
   - `apartment`, `apt.`, `flat no`, `flat` $\rightarrow$ `apt`
   - `suite`, `ste.`, `unit`, `room`, `rm.` $\rightarrow$ `unit`
   - `floor`, `fl.`, `1st floor`, `2nd floor` $\rightarrow$ `fl`
   - `building`, `bldg.` $\rightarrow$ `bldg`
4. **Landmark Tags**:
   - `opposite`, `opp.`, `opp` $\rightarrow$ `opp`
   - `behind`, `b/h`, `b-h` $\rightarrow$ `behind`
   - `near`, `nr.` $\rightarrow$ `near`
   - `beside`, `next to` $\rightarrow$ `near`
5. **US State Mapping**:
   - Converts 2-letter state codes to standardized state names (or vice versa) to resolve discrepancies like `TX` vs `Texas`, `AL` vs `Alabama`, `NC` vs `North Carolina`.
6. **Number & Zero-Padding Standardization**:
   - Strips leading zeros in house, plot, and unit numbers: `K-00303` $\rightarrow$ `k-303`, `#05131` $\rightarrow$ `5131`.

---

### 3.5. Phase 5: Numeric Token Extraction (`address_numbers`)
- Extracts all integer digit sequences found in the address string.
- Example: `"6207 Ocean Front Avenue, Unit 11, VA"` $\rightarrow$ `('6207', '11')`.
- *Why this is critical:* In commercial entity resolution, two businesses on the same street (e.g. `100 Main St` vs `500 Main St`) share almost all text tokens. Comparing exact extracted numbers prevents false positive merges with near 100% precision.

---

## 4. Performance & Scalability Targets

| Metric | Target |
| :--- | :--- |
| **Throughput** | $\ge 60,000$ records/sec (single core) |
| **Memory Footprint** | Streaming chunk processing $\le 1.5$ GB RAM |
| **Country Portability** | 100% language-open (Handles `US`, `India`, `France`, and arbitrary unseen countries) |
| **Code Structure** | Self-contained in `src/normalizer.py` (Standard library + `re` / `unicodedata`) |

---

## 5. Output Verification Matrix

| Raw Input (S1, S2, or S3) | Output `clean_name_core` | Output `clean_name_full` | Output `clean_address` | Output `address_numbers` |
| :--- | :--- | :--- | :--- | :--- |
| `Urology Partners Inc` | `urology partners` | `urology partners inc` | `6207 ocean front ave virginia beach va` | `('6207',)` |
| `siiainvestments.com` | `siia investments` | `siia investments` | `8411 gabrielino ct pmb 9239 rancho cucamonga ca` | `('8411', '9239')` |
| `Mirapyra formerly Painters Local Union 634` | `painters local union 634` | `painters local union 634` | `25807 iron gate dr madison al` | `('25807',)` |
| `GOLDEN ONE CÓNSULTANTS PRIVATE LTD` | `golden one consultants` | `golden one consultants pvt ltd` | `220a sector 11 shri ram vatika park shivaji nagar haryana` | `('220', '11')` |
| `ZNB Club SARL` | `znb club` | `znb club sarl` | `5 bis rue pierre dignac la teste-de-buch nouvelle-aquitaine` | `('5',)` |
