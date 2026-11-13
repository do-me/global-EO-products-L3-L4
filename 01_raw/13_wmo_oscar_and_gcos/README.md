# `13_wmo_oscar_and_gcos` — WMO OSCAR (Observing Systems Capability Analysis and Review Tool) Space & GCOS Variables

## Overview
- **Source ID**: `13_wmo_oscar_and_gcos`
- **Authoritative Endpoint**: `GET https://space.oscar.wmo.int/variables + GET https://space.oscar.wmo.int/satellites`
- **Output Directory**: `./data/`
- **Provenance & Checksum Log**: `./last_access.txt`

## How to Run
From inside this folder (or from the repository root):
```bash
uv run mine.py
```

## Particularities & Technical Notes
Extracts the World Meteorological Organization (WMO) OSCAR Space geophysical variable taxonomy and satellite mission registry (`space.oscar.wmo.int`) used for cross-referencing GCOS/WMO observational requirements.
