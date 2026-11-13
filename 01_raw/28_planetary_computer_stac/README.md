# `28_planetary_computer_stac` — Microsoft Planetary Computer STAC v1 API

## Overview
- **Source ID**: `28_planetary_computer_stac`
- **Authoritative Endpoint**: `GET https://planetarycomputer.microsoft.com/api/stac/v1/collections`
- **Output Directory**: `./data/`
- **Provenance & Checksum Log**: `./last_access.txt`

## How to Run
From inside this folder (or from the repository root):
```bash
uv run mine.py
```

## Particularities & Technical Notes
Retrieves all 139 STAC Collections from the Microsoft Planetary Computer STAC v1 API (`planetarycomputer.microsoft.com/api/stac/v1/collections`). In `03_harmonization`, each collection is attributed to its upstream **producing institution** while recording `Microsoft Planetary Computer` as an access/hosting platform.
