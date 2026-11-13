#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas>=2.2.0",
#   "pyarrow>=15.0.0",
# ]
# ///
"""
03_harmonization/verify_catalogue.py
====================================
Automated structural, provenance, and ontology verification suite for the Global L3/L4 EO Product Catalogue.
Verifies:
1. All 29 `01_raw/` folders contain `mine.py`, `README.md`, `last_access.txt`, and `data/` with all files <48 MB.
2. Presence and consistent classification of sentinel, maturity, ontology, and producer-pool reference products.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
RAW_DIR = REPO_ROOT / "01_raw"
HARM_DATA_DIR = BASE_DIR / "data"


def main() -> None:
    df_all = pd.read_parquet(HARM_DATA_DIR / "eo_l3_l4_all_catalogue_records.parquet")

    checks_passed = 0
    issues_found: list[str] = []

    # 1. Check all 29 raw directories & file sizes
    raw_dirs = sorted([d for d in RAW_DIR.iterdir() if d.is_dir()])
    if len(raw_dirs) != 29:
        issues_found.append(f"Expected 29 raw source directories, found {len(raw_dirs)}")
    else:
        checks_passed += 1

    max_file_mb = 0.0
    max_file_path = ""
    for f in REPO_ROOT.rglob("*"):
        if f.is_file() and ".git" not in f.parts:
            mb = f.stat().st_size / (1024 * 1024)
            if mb > max_file_mb:
                max_file_mb = mb
                max_file_path = str(f.relative_to(REPO_ROOT))
            if mb > 48.0:
                issues_found.append(f"File exceeds 48 MB GitHub safety threshold: {f.relative_to(REPO_ROOT)} ({mb:.2f} MB)")

    for d in raw_dirs:
        for req in ("mine.py", "README.md", "last_access.txt"):
            if not (d / req).exists():
                issues_found.append(f"Missing {req} in {d.name}")
        if not list((d / "data").glob("*")):
            issues_found.append(f"Empty data/ directory in {d.name}")
    checks_passed += 1

    # 2. Verify Reference Products Across Ontology, Maturity & Provider-Pool Classifications
    benchmarks = []

    # Check A: CEMS Global Flood Monitoring (GFM)
    gfm = df_all[
        (df_all["source_catalogue"] == "06_copernicus_cems_gfm_eodc")
        & (df_all["raw_id"] == "GFM")
    ]
    if len(gfm) == 1 and gfm.iloc[0]["provider_pool"] == "copernicus_services" and gfm.iloc[0]["copernicus_service"] == "CEMS" and bool(gfm.iloc[0]["uses_sentinel_data"]):
        checks_passed += 1
        benchmarks.append(("CEMS Global Flood Monitoring (GFM)", "PASS", f"pool={gfm.iloc[0]['provider_pool']}, service={gfm.iloc[0]['copernicus_service']}, sentinel={gfm.iloc[0]['sentinel_missions']}, res={gfm.iloc[0]['spatial_resolution_meters']}m"))
    else:
        issues_found.append("CEMS GFM missing or misclassified")

    # Check B: CEMS GHSL (GHS-POP, GHS-BUILT)
    ghs_pop = df_all[df_all["raw_id"].astype(str).str.contains("EO.GHSL.DAT.POP|GHS-POP|JRC/GHSL", case=False, na=False)]
    ghs_built = df_all[df_all["raw_id"].astype(str).str.contains("EO.GHSL.DAT.BUILT-S", case=False, na=False)]
    if len(ghs_pop) > 0 and (ghs_pop["provider_pool"] == "copernicus_services").all() and len(ghs_built) > 0:
        checks_passed += 1
        benchmarks.append(("CEMS GHSL (GHS-POP & GHS-BUILT)", "PASS", f"found {len(ghs_pop)} GHS-POP & {len(ghs_built)} GHS-BUILT-S records under copernicus_services (CEMS)"))
    else:
        issues_found.append("CEMS GHSL missing or misclassified")

    # Check C: MODIS MOD17A2H v6.1 Primary Archive Start Year (2000)
    mod17 = df_all[
        (df_all["source_catalogue"] == "16_nasa_cmr")
        & (df_all["title"].astype(str).str.contains("MOD17A2H", case=False, na=False))
    ]
    if len(mod17) > 0 and (mod17["temporal_start_year"] == 2000).any():
        checks_passed += 1
        row_m = mod17[mod17["temporal_start_year"] == 2000].iloc[0]
        benchmarks.append(("MODIS MOD17A2H GPP Archive Start", "PASS", f"start_year={row_m['temporal_start_year']} ({row_m['temporal_start_raw']}), ongoing={row_m['is_ongoing_archive']}"))
    else:
        issues_found.append(f"MODIS MOD17A2H 2000 start year not found in 16_nasa_cmr (found: {mod17[['title', 'temporal_start_year']].to_dict(orient='records')})")

    # Check D: OpenLandMap Soil Organic Carbon Producer-Origin Attribution & Hybrid Ontology
    olm_soc = df_all[
        df_all["raw_id"].astype(str).str.contains("oc_iso.10694|SOL_ORGANIC-CARBON", case=False, na=False)
    ]
    if len(olm_soc) >= 2 and (olm_soc["provider_pool"] == "eu_non_copernicus").all() and (olm_soc["product_ontology_class"] == "eo_constrained_hybrid").all():
        checks_passed += 1
        benchmarks.append(("OpenLandMap Soil Organic Carbon Attribution", "PASS", f"all {len(olm_soc)} records (Wasabi STAC + GEE) classified as pool=eu_non_copernicus, ontology=eo_constrained_hybrid"))
    else:
        issues_found.append(f"OpenLandMap SOC check failed: {olm_soc[['source_catalogue', 'raw_id', 'provider_pool', 'product_ontology_class']].to_dict(orient='records')}")

    # Check E: Non-EU & UN Soil Organic Carbon Coverage
    non_eu_soc = df_all[
        (df_all["provider_pool"] == "non_eu")
        & (df_all["title"].astype(str).str.contains("Soil Organic Carbon|Soil Carbon|GSOC", case=False, na=False))
    ]
    if len(non_eu_soc) > 0:
        checks_passed += 1
        benchmarks.append(("Non-EU Soil Organic Carbon Coverage", "PASS", f"found {len(non_eu_soc)} non-EU/UN SOC records (e.g., {non_eu_soc.iloc[0]['title'][:65]}...)"))
    else:
        issues_found.append("No non-EU Soil Organic Carbon candidates found")

    # Check F: GEDI L4B v2 Operational Maturity
    gedi_l4b = df_all[df_all["title"].astype(str).str.contains("GEDI04_B|GEDI L4B", case=False, na=False)]
    if len(gedi_l4b) > 0 and (gedi_l4b["operational_maturity_class"] == "completed_or_mission_bounded_release").all():
        checks_passed += 1
        benchmarks.append(("GEDI L4B Gridded Aboveground Biomass Maturity", "PASS", f"maturity={gedi_l4b.iloc[0]['operational_maturity_class']}, bbox_lat=[{gedi_l4b.iloc[0]['bbox_south']}, {gedi_l4b.iloc[0]['bbox_north']}]"))
    else:
        issues_found.append(f"GEDI L4B maturity check failed: {gedi_l4b[['raw_id', 'operational_maturity_class']].to_dict(orient='records')}")

    # Check G: ESA WorldCover & DLR WSF Completed/Mission-Bounded Releases
    wc = df_all[df_all["title"].astype(str).str.contains("WorldCover", case=False, na=False)]
    wsf = df_all[df_all["raw_id"].astype(str).str.contains("WSF_2015|WSF_2019|WSF_Evolution|WSF2015|WSF_EVO", case=False, na=False)]
    if len(wc) > 0 and (wc["operational_maturity_class"] == "completed_or_mission_bounded_release").all() and len(wsf) > 0 and (wsf["operational_maturity_class"] == "completed_or_mission_bounded_release").all():
        checks_passed += 1
        benchmarks.append(("ESA WorldCover & DLR WSF Release Maturity", "PASS", f"WorldCover ({len(wc)} recs) & DLR WSF ({len(wsf)} recs) all marked completed_or_mission_bounded_release"))
    else:
        issues_found.append("WorldCover / WSF maturity check failed")

    # Check H: Dynamic World & OPERA DSWx-HLS Sentinel Dependency
    dw = df_all[df_all["raw_id"].astype(str).str.contains("DYNAMICWORLD", case=False, na=False)]
    dswx = df_all[df_all["title"].astype(str).str.contains("DSWx-HLS|DSWX_HLS", case=False, na=False)]
    if len(dw) > 0 and bool(dw.iloc[0]["uses_sentinel_data"]) and len(dswx) > 0 and bool(dswx.iloc[0]["uses_sentinel_data"]):
        checks_passed += 1
        benchmarks.append(("Dynamic World & OPERA DSWx-HLS Sentinel Flag", "PASS", f"Dynamic World (pool={dw.iloc[0]['provider_pool']}, sentinel={dw.iloc[0]['sentinel_missions']}) & DSWx-HLS (pool={dswx.iloc[0]['provider_pool']}, sentinel={dswx.iloc[0]['sentinel_missions']})"))
    else:
        issues_found.append("Dynamic World / DSWx-HLS Sentinel dependency check failed")

    # Check I: Symmetric L3_ARD_REFLECTANCE Flagging (NASA HLS vs Copernicus S2 Global Mosaic)
    hls = df_all[df_all["title"].astype(str).str.contains("HLSL30|HLSS30", case=False, na=False)]
    s2gm = df_all[
        (df_all["provider_pool"] == "copernicus_services")
        & (df_all["title"].astype(str).str.contains("Sentinel-2.*Mosaic|Global Mosaic", case=False, na=False))
    ]
    if len(hls) > 0 and (hls["product_level_subtype"] == "L3_ARD_REFLECTANCE").any() and len(s2gm) > 0 and (s2gm["product_level_subtype"] == "L3_ARD_REFLECTANCE").any():
        checks_passed += 1
        benchmarks.append(("Symmetric L3_ARD_REFLECTANCE (HLS & CLMS S2GM)", "PASS", f"HLS ({len(hls)} recs) & Copernicus S2 Mosaic ({len(s2gm)} recs) both flagged as L3_ARD_REFLECTANCE"))
    else:
        issues_found.append("HLS / S2 Mosaic L3_ARD_REFLECTANCE symmetry check failed")

    # Check J: WGClimate ECV Attribution, OpenLandMap Global BBox & DEA/Data Terra Split
    wgc = df_all[df_all["source_catalogue"] == "12_ceos_cgms_wgclimate_ecv"]
    wgc_bad_eu = wgc[wgc["title"].astype(str).str.contains(r"CLARA-A|CLAAS|Cloud_cci|CMSAF|CM SAF|C3S", case=False, na=False) & (wgc["provider_pool"] == "non_eu")]
    dea = df_all[(df_all["source_catalogue"] == "22_data_terra_inpe_dea") & df_all["producing_institution"].astype(str).str.contains("Geoscience Australia", case=False, na=False)]
    olm_glob = df_all[(df_all["source_catalogue"] == "24_openlandmap_stac") & (df_all["geographic_coverage_class"] == "global")]
    if len(wgc_bad_eu) == 0 and (wgc["title"].notna()).all() and len(dea) > 0 and (dea["provider_pool"] == "non_eu").all() and len(olm_glob) >= 80:
        checks_passed += 1
        benchmarks.append(("WGClimate, OpenLandMap & Source-22 Attribution", "PASS", f"WGClimate titles={int(wgc['title'].notna().sum())}/{len(wgc)}, DEA non_eu={len(dea)}, OpenLandMap global={len(olm_glob)}/{len(df_all[df_all['source_catalogue']=='24_openlandmap_stac'])}"))
    else:
        issues_found.append(f"Attribution/coverage check failed: wgc_bad_eu={len(wgc_bad_eu)}, dea={len(dea)}, olm_glob={len(olm_glob)}")

    print("=" * 92)
    print(f"VERIFICATION REPORT ({checks_passed} core test suites passed, {len(issues_found)} issues)")
    print(f"Max single file size in repo: {max_file_mb:.2f} MB ({max_file_path})")
    print("=" * 92)
    for name, status, detail in benchmarks:
        print(f"  [{status}] {name:54s} -> {detail}")
    if issues_found:
        print("\nISSUES FOUND:")
        for iss in issues_found:
            print(f"  [FAIL] {iss}")
        sys.exit(1)
    print("\nAll structural, ontology, and provenance verification checks PASSED.")


if __name__ == "__main__":
    main()
