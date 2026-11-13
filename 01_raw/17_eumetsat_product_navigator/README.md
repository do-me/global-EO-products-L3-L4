# `17_eumetsat_product_navigator` — EUMETSAT Product Navigator Elasticsearch API (EUMETSAT & All 8 SAFs)

## Overview
- **Source ID**: `17_eumetsat_product_navigator`
- **Authoritative Endpoint**: `GET https://user.eumetsat.int/catalogue/es/csw/_search?size=1000`
- **Output Directory**: `./data/`
- **Provenance & Checksum Log**: `./last_access.txt`

## How to Run
From inside this folder (or from the repository root):
```bash
uv run mine.py
```

## Particularities & Technical Notes
Retrieves all 896 collections from EUMETSAT and its 8 Satellite Application Facilities (CM SAF, OSI SAF, LSA SAF, AC SAF, H SAF, NWC SAF, NWPSAF, ROM SAF) in a single call. Crucial discovery: 208 records have an empty `processingLevels` array in upstream metadata, including flagship CM SAF, OSI SAF, LSA SAF, and H SAF Climate Data Records (L3/L4). `02_preparation` excludes explicit Level 1, Level 2, and Visualisation products while classifying unlabelled SAF CDR grids.
