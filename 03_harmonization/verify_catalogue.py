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
1. All 35 `01_raw/` folders contain `mine.py`, `README.md`, `last_access.txt`, and `data/` with all files <48 MB.
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
    df_canon = pd.read_parquet(HARM_DATA_DIR / "eo_l3_l4_canonical_products.parquet")

    checks_passed = 0
    issues_found: list[str] = []

    # 1. Check all 35 raw directories & file sizes
    raw_dirs = sorted([d for d in RAW_DIR.iterdir() if d.is_dir()])
    if len(raw_dirs) != 35:
        issues_found.append(f"Expected 35 raw source directories, found {len(raw_dirs)}")
    else:
        checks_passed += 1

    max_file_mb = 0.0
    max_file_path = ""
    for f in REPO_ROOT.rglob("*"):
        if f.is_file() and ".git" not in f.parts:
            # Release-hosted search index artifacts are gitignored and distributed via GitHub Releases
            if "06_semantic_hybrid_search" in f.parts and f.suffix in {".duckdb", ".npz"}:
                continue
            mb = f.stat().st_size / (1024 * 1024)
            if mb > max_file_mb:
                max_file_mb = mb
                max_file_path = str(f.relative_to(REPO_ROOT))
            if mb > 48.0:
                issues_found.append(f"File exceeds 48 MB GitHub safety threshold: {f.relative_to(REPO_ROOT)} ({mb:.2f} MB)")

    raw_check_failed = False
    for d in raw_dirs:
        for req in ("mine.py", "README.md", "last_access.txt"):
            if not (d / req).exists():
                issues_found.append(f"Missing {req} in {d.name}")
                raw_check_failed = True
        if not list((d / "data").glob("*")):
            issues_found.append(f"Empty data/ directory in {d.name}")
            raw_check_failed = True
    if not raw_check_failed:
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

    # Check K: Resolution Regex (.25 deg) & Future Projection Maturity (>= 2028)
    dyocn = df_all[df_all["title"].astype(str).str.contains("AE_DyOcn|AU_DyOcn", case=False, na=False)]
    fut_ongoing = df_all[(df_all["temporal_end_year"].fillna(0) >= 2028) & (df_all["is_ongoing_archive"] == True)]  # noqa: E712
    if len(dyocn) >= 2 and (dyocn["title_token_resolution_meters"] == 27830.0).all() and len(fut_ongoing) == 0:
        checks_passed += 1
        benchmarks.append(("Resolution Regex (.25 deg) & Future Maturity Guard", "PASS", f"AE/AU_DyOcn={dyocn.iloc[0]['title_token_resolution_meters']}m, ongoing>=2028={len(fut_ongoing)}"))
    else:
        issues_found.append(f"Resolution/maturity check failed: dyocn={dyocn['title_token_resolution_meters'].tolist()}, fut_ongoing={len(fut_ongoing)}")

    # Check L: Two-Layer Provider Pool vs Jurisdiction Tier & Schema Completeness Coverage
    comp_csv = pd.read_csv(BASE_DIR / "schema_completeness_report.csv")
    tier_valid = (
        df_all[df_all["provider_pool"] == "copernicus_services"]["provider_jurisdiction_tier"].eq("copernicus_service").all()
        and df_all[df_all["provider_pool"] == "eu_non_copernicus"]["provider_jurisdiction_tier"].isin(["eu27_national_or_eu_agency", "european_space_met_igo", "copernicus_associated_state"]).all()
        and df_all[df_all["provider_pool"] == "non_eu"]["provider_jurisdiction_tier"].isin(["third_country_non_european", "un_or_international_body", "european_esa_non_copernicus_state"]).all()
    )
    era_c3s = df_all[df_all["raw_id"].astype(str).str.contains(r"era5-pds|era5_repack|terrascope-sis-agrometeorological|ce-ag-era5", case=False, na=False)]
    if len(comp_csv) == len(df_all.columns) and tier_valid and len(era_c3s) >= 4 and (era_c3s["provider_pool"] == "copernicus_services").all():
        checks_passed += 1
        benchmarks.append(("Jurisdiction Tier, Completeness & ERA5 C3S Mapping", "PASS", f"schema_cols={len(comp_csv)}, era5_c3s={len(era_c3s)}, tier_invariants=valid"))
    else:
        issues_found.append(f"Jurisdiction/completeness check failed: comp_cols={len(comp_csv)}/{len(df_all.columns)}, tier_valid={tier_valid}, era_bad={era_c3s[era_c3s['provider_pool']!='copernicus_services']['raw_id'].tolist()}")

    # Check M: JRC Prose Exemption (ARDECO/LUCAS), FedEO SAMeT/NASA/ORNL & INPE BDC
    jrc_noncop = df_all[(df_all["source_catalogue"] == "07_ec_jrc_data_catalogue") & df_all["title"].astype(str).str.contains("ARDECO|LUCAS Copernicus", case=False, na=False)]
    fedeo_noneu = df_all[(df_all["source_catalogue"] == "14_ceos_fedeo_agencies") & df_all["raw_id"].astype(str).str.contains(r"samet_|opera-s1-rtc|mod11a2|ORNL_CLOUD", case=False, na=False)]
    inpe = df_all[(df_all["source_catalogue"] == "22_data_terra_inpe_dea") & df_all["hosting_platform"].astype(str).str.contains("INPE", case=False, na=False)]
    if (
        len(jrc_noncop) >= 3
        and (jrc_noncop["provider_pool"] == "eu_non_copernicus").all()
        and (~jrc_noncop["strict_eo_c1_eligible"]).all()
        and len(fedeo_noneu) >= 5
        and (fedeo_noneu["provider_pool"] == "non_eu").all()
        and len(inpe) >= 50
        and (inpe["provider_pool"] == "non_eu").all()
    ):
        checks_passed += 1
        benchmarks.append(("JRC ARDECO/LUCAS, FedEO Non-EU & INPE BDC Split", "PASS", f"jrc_ex={len(jrc_noncop)}, fedeo_noneu={len(fedeo_noneu)}, inpe_bdc={len(inpe)}"))
    else:
        issues_found.append(f"JRC/FedEO/INPE split failed: jrc={len(jrc_noncop)}, fedeo_noneu={len(fedeo_noneu)}, inpe={len(inpe)}")

    # Check N: High-Recall Rescued L3/L4 Composites, CDRs & Multi-Partner Coverage
    wapor_gee = df_all[(df_all["source_catalogue"] == "27_gee_official_stac") & df_all["raw_id"].astype(str).str.startswith("FAO/WAPOR/")]
    s3_syn = df_all[df_all["raw_id"].astype(str).str.contains("syn-2-v10|syn-2-vg1|s3-syn-vgt|synergy-v10", case=False, na=False)]
    transcom = df_all[df_all["title"].astype(str).str.contains("TransCom 3", case=False, na=False)]
    sedac = df_all[(df_all["source_catalogue"] == "16_nasa_cmr") & df_all["title"].astype(str).str.contains("PEST-CHEMGRIDS|Gridded Population of the World", case=False, na=False)]
    if len(wapor_gee) >= 12 and len(s3_syn) >= 6 and len(transcom) >= 2 and len(sedac) >= 10:
        checks_passed += 1
        benchmarks.append(("High-Recall Rescued Products (WaPOR/SYN/TransCom/SEDAC)", "PASS", f"wapor_gee={len(wapor_gee)}, s3_syn={len(s3_syn)}, transcom={len(transcom)}, sedac={len(sedac)}"))
    else:
        issues_found.append(f"High-recall rescue check failed: wapor={len(wapor_gee)}, syn={len(s3_syn)}, transcom={len(transcom)}, sedac={len(sedac)}")

    # Check O: Flagship Resolution, Suomi-NPP L3, CLMS/CEMS & Metadata Recovery
    max_res = df_all["spatial_resolution_meters"].max()
    id_title = df_all["raw_id"].astype(str) + " " + df_all["title"].astype(str)
    burnt = df_all[
        (df_all["source_catalogue"] == "02_copernicus_clms")
        & id_title.str.contains("Burnt Area.*300 m", case=False, na=False)
    ]
    burnt_ok = len(burnt) > 0 and bool((burnt["spatial_resolution_meters"] < 500.0).all())
    suomi_l4 = df_all[id_title.str.contains("VIIRSN_L3|VNP43|OMPS_NPP.*L3", case=False, na=False) & (df_all["processing_level"] == "L4")]
    pml_bad = df_all[id_title.str.contains("CAS/IGSNRR/PML", case=False, na=False) & (df_all["provider_pool"] != "non_eu")]
    geo_cats = [
        "01_copernicus_cmems",
        "02_copernicus_clms",
        "11_eea_and_emodnet",
        "12_ceos_cgms_wgclimate_ecv",
        "17_eumetsat_product_navigator",
    ]
    geo_rates = {c: float(df_all.loc[df_all["source_catalogue"] == c, "geographic_coverage_class"].notna().mean()) for c in geo_cats}
    geo_ok = all(rate >= 0.50 for rate in geo_rates.values())
    clms_ongoing = int(((df_all["source_catalogue"] == "02_copernicus_clms") & (df_all["is_ongoing_archive"] == True)).sum())
    if bool(max_res <= 600_000.0) and burnt_ok and len(suomi_l4) == 0 and len(pml_bad) == 0 and geo_ok and clms_ongoing >= 30:
        checks_passed += 1
        geo_summary = ", ".join(f"{k[:2]}={v:.1%}" for k, v in geo_rates.items())
        benchmarks.append(("Flagship Resolution, Suomi-NPP L3, CLMS/CEMS & Metadata Recovery", "PASS", f"max_res={max_res:.1f}m, burnt={len(burnt)} (<500m), ongoing_clms={clms_ongoing}, geo[{geo_summary}]"))
    else:
        issues_found.append(
            f"Check O failed: max_res={max_res}, burnt_ok={burnt_ok} (n={len(burnt)}), suomi_l4={len(suomi_l4)}, pml_bad={len(pml_bad)}, geo_rates={geo_rates}, clms_ongoing={clms_ongoing}"
        )

    # Check P: Deep UMM/OneStop Harvest, Cadence Support, Operational Status & Lineage Invariants
    noaa_c = df_canon[df_canon["source_catalogue"] == "20_noaa_onestop"]
    cmr_c = df_canon[df_canon["source_catalogue"] == "16_nasa_cmr"]
    noaa_desc_rate = float(noaa_c["description"].notna().mean()) if len(noaa_c) else 0.0
    noaa_bbox_rate = float(noaa_c["bbox_west"].notna().mean()) if len(noaa_c) else 0.0
    cmr_doi_rate = float(cmr_c["doi"].notna().mean()) if len(cmr_c) else 0.0
    cmr_spat_cnt = int(cmr_c["spatial_resolution_meters"].notna().sum())
    img_url_cnt = int(
        df_canon["access_url"]
        .fillna("")
        .str.contains(r"\.(?:png|jpe?g|gif|bmp|webp|svg|kmz)(?:[?#]|$)|/browse[/?#]", case=False, regex=True)
        .sum()
    )
    multi_day_ttl = df_canon["title"].fillna("").str.contains(
        r"\b(?:8-day|10-day|10-daily|16-day|5-day|dekad|dekadal|weekly|biweekly)\b", case=False, regex=True
    ) & df_canon["temporal_aggregation_window_raw"].isna()
    cadence_contra_cnt = int((multi_day_ttl & (df_canon["temporal_cadence_class"] == "sub_daily_or_daily")).sum())
    op_null_cnt = int(
        ((df_canon["operational_maturity_class"] == "operational_ongoing_series") & df_canon["is_ongoing_archive"].isna()).sum()
    )
    pre2014_sent_cnt = int(
        (
            (df_canon["uses_sentinel_data"] == True)  # noqa: E712
            & (df_canon["is_ongoing_archive"] != True)  # noqa: E712
            & (df_canon["temporal_end_year"].fillna(9999) < 2014)
        ).sum()
    )
    has_eum_0088 = bool(((df_all["source_catalogue"] == "17_eumetsat_product_navigator") & (df_all["raw_id"] == "EO:EUM:DAT:0088")).any())
    has_fluxcom = bool(df_all["doi"].fillna("").str.lower().str.contains("10.18160/5nzg-jmje").any())
    has_lcspp = bool(df_all["title"].fillna("").str.contains("LCSPP", case=False).any())
    has_troposif = bool(df_all["doi"].fillna("").str.lower().str.contains("10.5281/zenodo.10212472").any())
    has_smos_ic = bool(df_all["doi"].fillna("").str.lower().str.contains("10.15454/tihhyp").any())
    cds_oc = df_all[(df_all["source_catalogue"] == "03_copernicus_c3s_cds") & (df_all["raw_id"] == "satellite-ocean-colour")]
    cds_oc_ok = bool(
        len(cds_oc) == 1
        and cds_oc.iloc[0]["processing_level"] == "L3"
        and cds_oc.iloc[0]["processing_level_provenance"] == "provider_documented_specification"
        and cds_oc.iloc[0]["adjudication_source_citation"] == "https://climate.copernicus.eu/ocean"
        and "daily" in str(cds_oc.iloc[0]["temporal_resolution_raw"]).lower()
        and "monthly" in str(cds_oc.iloc[0]["temporal_resolution_raw"]).lower()
        and pd.isna(cds_oc.iloc[0]["temporal_cadence_class"])
    )
    gee_cadence_literal = int((df_all["temporal_resolution_raw"] == "cadence").sum())
    merra_2 = df_all[(df_all["source_catalogue"] == "16_nasa_cmr") & (df_all["raw_id"] == "C2531963170-GES_DISC")]
    merra_2_ok = bool(
        len(merra_2) == 1
        and "10.5067/o8ax56do60mi" in str(merra_2.iloc[0]["access_url"]).lower()
        and "0.625 x 0.5" in str(merra_2.iloc[0]["spatial_resolution_raw"])
    )
    op_closed_cnt = int(
        (
            (df_canon["operational_maturity_class"] == "operational_ongoing_series")
            & (df_canon["temporal_end_year"].fillna(9999) <= 2024)
        ).sum()
    )
    one_day_null_cnt = int(
        (
            df_canon["temporal_resolution_raw"].isin(["1 Day", "1 day", "1 year", "1 Year", "1-month", "0.5 days"])
            & df_canon["temporal_cadence_class"].isna()
        ).sum()
    )
    eu_global_cnt = int(
        (
            df_canon["record_id"].str.contains("_europe_|^05_copernicus_cems_ewds::efas-", case=False, regex=True)
            & (df_canon["geographic_coverage_class"] == "global")
        ).sum()
    )
    s26_dup_cnt = int(
        df_all[df_all["source_catalogue"] == "26_fao_unep_and_thematic_benchmarks"]["doi"]
        .dropna()
        .duplicated()
        .sum()
    )

    if (
        noaa_desc_rate >= 0.98
        and noaa_bbox_rate >= 0.98
        and cmr_doi_rate >= 0.90
        and cmr_spat_cnt >= 2300
        and img_url_cnt == 0
        and cadence_contra_cnt == 0
        and op_null_cnt == 0
        and pre2014_sent_cnt == 0
        and has_eum_0088
        and has_fluxcom
        and has_lcspp
        and has_troposif
        and has_smos_ic
        and cds_oc_ok
        and gee_cadence_literal == 0
        and merra_2_ok
        and op_closed_cnt == 0
        and one_day_null_cnt == 0
        and eu_global_cnt == 0
        and s26_dup_cnt == 0
    ):
        checks_passed += 1
        benchmarks.append(
            (
                "UMM/OneStop Deep Metadata, Cadence, Maturity & Lineage",
                "PASS",
                f"noaa_desc={noaa_desc_rate:.1%}, cmr_doi={cmr_doi_rate:.1%}, cmr_spat={cmr_spat_cnt}, contra={cadence_contra_cnt}, op_closed={op_closed_cnt}, eu_glob={eu_global_cnt}, s26_dup={s26_dup_cnt}",
            )
        )
    else:
        issues_found.append(
            f"Check P failed: noaa_desc={noaa_desc_rate:.3f}, noaa_bbox={noaa_bbox_rate:.3f}, cmr_doi={cmr_doi_rate:.3f}, "
            f"cmr_spat={cmr_spat_cnt}, img_urls={img_url_cnt}, cadence_contra={cadence_contra_cnt}, op_null={op_null_cnt}, "
            f"pre2014_sent={pre2014_sent_cnt}, eum_0088={has_eum_0088}, fluxcom={has_fluxcom}, lcspp={has_lcspp}, cds_oc={cds_oc_ok}, "
            f"op_closed={op_closed_cnt}, one_day_null={one_day_null_cnt}, eu_global={eu_global_cnt}, s26_dup={s26_dup_cnt}"
        )

    # Check Q: Provenance, Domains, ARD, Supersession & Sentinel Lineage (TCSIF v1, HTML decode, Zenodo URLs, GEO disks, ARD vs output, L3+L4 provenance, supersession, HLS/FedEO Sentinel)
    has_tcsif_v1 = bool(df_all["doi"].fillna("").str.lower().str.contains("10.5281/zenodo.8242928").any())
    zen_rows = df_all[(df_all["source_catalogue"] == "26_fao_unep_and_thematic_benchmarks") & df_all["raw_id"].astype(str).str.startswith("ZENODO:")]
    zen_null_urls = int(zen_rows["access_url"].isna().sum())
    html_remnants = int(df_all["description"].fillna("").str.contains(r"&(?:deg|times|ndash|nbsp|rarr);|</?(?:p|div|strong|ul|li)\b", regex=True).sum())
    ltdr_n18 = df_all[df_all["raw_id"].astype(str).str.contains("N18_AVH13C1", case=False) | df_all["title"].fillna("").str.contains("N18_AVH13C1", case=False)]
    ltdr_n18_ok = len(ltdr_n18) > 0 and bool(ltdr_n18["spatial_resolution_meters"].isna().all() and ltdr_n18["spatial_resolution_conflict"].all())
    cbers4 = df_all[df_all["raw_id"] == "CBERS4-MUX-2M-1"]
    cbers4_ok = len(cbers4) >= 1 and bool((cbers4["spatial_resolution_meters"] == 20.0).all() and not cbers4["spatial_resolution_conflict"].any())
    abi_g16 = df_all[df_all["raw_id"].astype(str).str.contains("AERDB_M3_ABI_G16", case=False) | df_all["title"].fillna("").str.contains("AERDB_M3_ABI_G16", case=False)]
    abi_g16_ok = len(abi_g16) >= 1 and bool((abi_g16["observation_domain_class"] == "geostationary_disk_or_sector").all() and (abi_g16["geographic_coverage_class"] != "global").all())
    lcspp_modis = df_all[df_all["raw_id"] == "ZENODO:20683570"]
    lcspp_ok = len(lcspp_modis) == 1 and lcspp_modis.iloc[0]["temporal_cadence_class"] == "multi_day_or_dekadal" and not bool(lcspp_modis.iloc[0]["is_ard_reflectance_only"])
    gosif_z = df_all[df_all["raw_id"] == "ZENODO:15105312"]
    gosif_ok = len(gosif_z) == 1 and gosif_z.iloc[0]["temporal_cadence_class"] == "multi_day_or_dekadal"
    troposif_z = df_all[df_all["raw_id"] == "ZENODO:10212472"]
    troposif_ok = len(troposif_z) == 1 and troposif_z.iloc[0]["temporal_cadence_class"] == "sub_daily_or_daily"
    mcd43_v061 = df_all[df_all["raw_id"].isin(["C2218719731-LPCLOUD", "MODIS/061/MCD43A4"])]
    mcd43_ok = len(mcd43_v061) == 2 and bool(mcd43_v061["is_ard_reflectance_only"].all() and (mcd43_v061["temporal_cadence_class"] == "sub_daily_or_daily").all() and mcd43_v061["temporal_aggregation_window_raw"].notna().all())
    coco2 = df_all[df_all["title"].fillna("").str.contains("CoCO2-MOSAIC", case=False)]
    coco2_ok = len(coco2) == 1 and coco2.iloc[0]["product_ontology_class"] == "non_eo_inventory_or_census" and not bool(coco2.iloc[0]["is_ard_reflectance_only"])
    c3s_sst = df_all[(df_all["source_catalogue"] == "03_copernicus_c3s_cds") & (df_all["raw_id"] == "satellite-sea-surface-temperature")]
    c3s_sst_ok = len(c3s_sst) == 1 and c3s_sst.iloc[0]["processing_level"] == "L3; L4" and c3s_sst.iloc[0]["product_level_subtype"] == "L3_L4_MULTI_LEVEL"
    c3s_alb = df_all[(df_all["source_catalogue"] == "03_copernicus_c3s_cds") & (df_all["raw_id"] == "satellite-albedo")]
    c3s_alb_ok = len(c3s_alb) == 1 and bool(c3s_alb.iloc[0]["is_ongoing_archive"] == False)  # noqa: E712
    hlsl30_all = df_all[df_all["raw_id"].isin(["C2021957657-LPCLOUD", "NASA/HLS/HLSL30/v002", "hls2-l30"])]
    hlsl30_ok = len(hlsl30_all) == 3 and not bool(hlsl30_all["uses_sentinel_data"].any())
    boreal_agb = df_all[df_all["raw_id"] == "C2756302505-ORNL_CLOUD"]
    boreal_ok = len(boreal_agb) >= 2 and bool(boreal_agb["uses_sentinel_data"].nunique() == 1 and not boreal_agb["uses_sentinel_data"].any())
    socat25 = df_all[df_all["title"].fillna("").str.contains("SOCATv2025", case=False)]
    socat25_ok = len(socat25) >= 1 and bool((socat25["operational_maturity_class"] == "deprecated_or_superseded").all())

    if (
        has_tcsif_v1
        and zen_null_urls == 0
        and html_remnants == 0
        and ltdr_n18_ok
        and cbers4_ok
        and abi_g16_ok
        and lcspp_ok
        and gosif_ok
        and troposif_ok
        and mcd43_ok
        and coco2_ok
        and c3s_sst_ok
        and c3s_alb_ok
        and hlsl30_ok
        and boreal_ok
        and socat25_ok
    ):
        checks_passed += 1
        benchmarks.append(
            (
                "Provenance, Domains, ARD, Supersession & Sentinel Lineage",
                "PASS",
                f"tcsif_v1=True, zen_urls=100%, html_clean=True, geo_disks=non-global, hlsl30_s2=False, c3s_sst=L3;L4",
            )
        )
    else:
        issues_found.append(
            f"Check Q failed: tcsif_v1={has_tcsif_v1}, zen_null_urls={zen_null_urls}, html={html_remnants}, "
            f"n18={ltdr_n18_ok}, cbers4={cbers4_ok}, abi_g16={abi_g16_ok}, lcspp={lcspp_ok}, gosif={gosif_ok}, "
            f"troposif={troposif_ok}, mcd43={mcd43_ok}, coco2={coco2_ok}, sst={c3s_sst_ok}, alb={c3s_alb_ok}, "
            f"hlsl30={hlsl30_ok}, boreal={boreal_ok}, socat25={socat25_ok}"
        )

    # Check R: Variant Table, Cited Adjudications, Semimonthly/Prose Cadence, GEO Global Restoration & Sentinel Split
    var_pq = BASE_DIR / "data" / "eo_l3_l4_product_variants.parquet"
    var_jl = BASE_DIR / "data" / "eo_l3_l4_product_variants.jsonl"
    var_ok = False
    n_vars = 0
    if var_pq.exists() and var_jl.exists():
        df_var = pd.read_parquet(var_pq)
        n_vars = len(df_var)
        v_tcsif = df_var[df_var["raw_id"] == "ZENODO:8242928"]
        v_sst = df_var[df_var["raw_id"] == "satellite-sea-surface-temperature"]
        var_ok = (
            n_vars >= 50
            and set(v_tcsif["variant_level"]) >= {"L2", "L3"}
            and bool((v_tcsif[v_tcsif["variant_level"] == "L3"]["variant_spatial_resolution_meters"] == 55660.0).all())
            and set(v_sst["variant_level"]) >= {"L3", "L4"}
        )

    geo_restored = df_all[df_all["raw_id"].isin(["C1233767575-GES_DISC", "C1266136072-GES_DISC"])]
    sndrd_glob = df_all[df_all["raw_id"].astype(str).str.contains(r"^C3944483", regex=True)]
    merra_geo = int((df_all["title"].fillna("").str.contains(r"\bmerra[-_ ]?2\b", case=False, regex=True) & (df_all["observation_domain_class"] == "geostationary_disk_or_sector")).sum())
    geo_restore_ok = len(geo_restored) == 2 and bool((geo_restored["geographic_coverage_class"] == "global").all()) and len(sndrd_glob) >= 16 and bool((sndrd_glob["geographic_coverage_class"] == "global").all()) and merra_geo == 0

    sm_ok = (
        bool((df_all[df_all["raw_id"] == "C2759076389-ORNL_CLOUD"]["temporal_output_cadences"] == "semimonthly").all())
        and bool((df_all[df_all["raw_id"] == "ZENODO:8253971"]["temporal_output_cadences"] == "semimonthly").all())
        and int((df_all["raw_id"].astype(str).str.contains("bimonthly", case=False) & (df_all["temporal_output_cadences"] == "semimonthly")).sum()) == 0
        and bool((df_all[df_all["raw_id"] == "ZENODO:7479028"]["adjudicated_output_grid_meters"] == 5566.0).all())
        and bool(df_all[df_all["raw_id"] == "ZENODO:7479028"]["spatial_resolution_meters"].isna().all())
        and bool((df_all[df_all["raw_id"] == "ZENODO:15382965"]["temporal_cadence_class"] == "monthly_or_seasonal").all())
    )

    conf_r = df_all[df_all["spatial_resolution_conflict"] == True]  # noqa: E712
    cited_conf = conf_r[conf_r["adjudication_source_citation"].notna()]
    unres_conf = conf_r[conf_r["adjudication_source_citation"].isna()]
    cite_ok = (
        len(cited_conf) >= 33
        and bool(conf_r["spatial_resolution_meters"].isna().all())
        and bool(cited_conf["adjudicated_output_grid_meters"].notna().all())
        and bool(cited_conf["adjudicated_spatial_grid_units"].notna().all())
        and bool(cited_conf["adjudicated_spatial_grid_axes"].notna().all())
        and bool(cited_conf["adjudication_source_citation"].astype(str).str.startswith("https://doi.org/").all())
        and bool(unres_conf["adjudicated_output_grid_meters"].isna().all())
    )

    sent_vc = df_all["sentinel_lineage_status"].value_counts().to_dict()
    gee29_sent_vc = df_all[df_all["source_catalogue"] == "29_awesome_gee_community"]["sentinel_lineage_status"].value_counts().to_dict()
    sent_split_ok = (
        sent_vc.get("no_sentinel_lineage", 0) == 0
        and sent_vc.get("explicit_non_sentinel", 0) >= 19000
        and sent_vc.get("unassessed_no_platform_metadata", 0) >= 16000
        and gee29_sent_vc.get("unassessed_no_platform_metadata", 0) >= 4200
    )
    msat_r = df_all[df_all["raw_id"].astype(str).str.startswith("projects/edf-methanesat-ee/assets/msat/")]
    mebd_r = df_all[df_all["raw_id"].isin(["a49dc43a-d799-11f1-8257-02ffd92199af", "d4e9b21f-6991-48ba-9b32-d1b4a08d76f0"])]
    op25_ok = len(msat_r) >= 3 and bool((msat_r["operational_maturity_class"] == "unknown").all()) and len(mebd_r) == 2 and bool((mebd_r["operational_maturity_class"] == "completed_or_mission_bounded_release").all())

    if var_ok and geo_restore_ok and sm_ok and cite_ok and sent_split_ok and op25_ok:
        checks_passed += 1
        benchmarks.append(
            (
                "Variants, Cited Adjudications, Semimonthly & Sentinel Split",
                "PASS",
                f"variants={n_vars}, geo_restored=True, semimonthly=True, conflict_cites={len(cited_conf)}/{len(conf_r)}, sent_explicit={sent_vc.get('explicit_non_sentinel', 0)}",
            )
        )
    else:
        issues_found.append(
            f"Check R failed: var_ok={var_ok} (n={n_vars}), geo_restore={geo_restore_ok}, sm={sm_ok}, cite={cite_ok}, sent_split={sent_split_ok}, op25={op25_ok}"
        )

    # Check S: Prose Provenance Separation, Zenodo Additions, Variant Conflict Guards & EUMETSAT Crosswalk
    amsr2_r = df_all[df_all["raw_id"] == "C1886605828-LANCEAMSR2"]
    remo_r = df_all[df_all["raw_id"] == "cd0b21c3-dc1c-4e44-860d-eeb714da19c6"]
    gedi4a_r = df_all[df_all["record_id"] == "16_nasa_cmr::C4212593885-ORNL_CLOUD"]
    gl_lai_r = df_all[df_all["raw_id"] == "ZENODO:12698637"]
    wk_0088_r = df_all[df_all["record_id"] == "09_wekeo_dias::EO:EUM:DAT:0088"]
    new_zen_ok = all(bool((df_all["raw_id"] == z).any()) for z in ("ZENODO:10589730", "ZENODO:5546316", "ZENODO:6659543", "ZENODO:8210518"))
    snpp_end_on = int((df_all["title"].fillna("").str.contains(r"end of mission|2026-11-02", case=False, regex=True) & (df_all["operational_maturity_class"] == "operational_ongoing_series")).sum())
    s_prose_ok = (
        len(amsr2_r) == 1 and pd.isna(amsr2_r.iloc[0]["spatial_resolution_raw"]) and float(amsr2_r.iloc[0]["adjudicated_output_grid_meters"]) == 6250.0
        and len(remo_r) == 1 and pd.isna(remo_r.iloc[0]["spatial_resolution_raw"]) and float(remo_r.iloc[0]["adjudicated_output_grid_meters"]) == 5566.0
        and pd.isna(remo_r.iloc[0]["observation_domain_mask"])
        and len(gl_lai_r) == 1 and gl_lai_r.iloc[0]["temporal_output_cadences"] == "semimonthly; 8-day"
        and "1981-2000: semimonthly" in str(gl_lai_r.iloc[0]["temporal_cadence_by_epoch"])
    )
    s_var_ok = False
    if var_pq.exists():
        df_var_s = pd.read_parquet(var_pq)
        v_821 = df_var_s[df_var_s["raw_id"] == "ZENODO:8210518"]
        v_sst_s = df_var_s[df_var_s["raw_id"] == "satellite-sea-surface-temperature"]
        v_gl_s = df_var_s[df_var_s["raw_id"] == "ZENODO:12698637"]
        s_var_ok = (
            len(gedi4a_r) == 1
            and bool(gedi4a_r.iloc[0]["processing_level_conflict"])
            and float(gedi4a_r.iloc[0]["spatial_footprint_meters"]) == 25.0
            and pd.isna(gedi4a_r.iloc[0]["adjudicated_output_grid_meters"])
            and len(df_var_s[df_var_s["raw_id"] == "C4212593885-ORNL_CLOUD"]) == 0
            and set(v_821["variant_level"]) == {"L2", "L3"}
            and bool(v_821[v_821["variant_level"] == "L2"]["variant_spatial_resolution_meters"].isna().all())
            and bool((v_821[v_821["variant_level"] == "L3"]["variant_spatial_resolution_meters"] == 55660.0).all())
            and len(v_sst_s) == 2 and bool((v_sst_s["variant_spatial_resolution_meters"] == 5566.0).all())
            and len(v_gl_s) == 2
        )
    s_wk_ok = (
        len(wk_0088_r) == 1
        and wk_0088_r.iloc[0]["processing_level"] == "L3"
        and wk_0088_r.iloc[0]["processing_level_provenance"] == "crosswalked_primary_eumetsat_navigator"
        and wk_0088_r.iloc[0]["temporal_cadence_class"] == "sub_daily_or_daily"
    )
    if new_zen_ok and s_prose_ok and s_var_ok and s_wk_ok and snpp_end_on == 0:
        checks_passed += 1
        benchmarks.append(
            (
                "Prose Separation, Zenodo Coverage, Variant & EUMETSAT Crosswalk",
                "PASS",
                f"amsr2=6250m, remo=5566m, globmap=epochs, gedi4a=footprint25m, sst_var=5566m, wekeo_0088=L3/15min, snpp_ended_on={snpp_end_on}",
            )
        )
    else:
        issues_found.append(
            f"Check S failed: new_zen={new_zen_ok}, s_prose={s_prose_ok}, s_var={s_var_ok}, s_wk={s_wk_ok}, snpp_end_on={snpp_end_on}"
        )

    # Check T: Strict RFC 8259 JSONL, FedEO CEDA DOIs, Antimeridian, Variable-Cadence Variants, Sublevels, NH SWE, Thematic Releases & Harvest Lifecycle
    import json as _json_t

    t_jsonl_ok = True
    for jp in sorted(HARM_DATA_DIR.glob("*.jsonl")):
        with jp.open("r", encoding="utf-8") as jf:
            for jl in jf:
                try:
                    _json_t.loads(jl, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
                except Exception:
                    t_jsonl_ok = False
                    break
        if not t_jsonl_ok:
            break

    ceda_t = df_all[
        (df_all["source_catalogue"] == "14_ceos_fedeo_agencies")
        & (df_all["raw_id"].astype(str).str.match(r"^[0-9a-f]{32}$"))
    ]
    t_ceda_ok = (
        int((ceda_t["doi"].fillna("").str.contains(r"MOD021KM|10\.1016/|10\.5194/", regex=True, case=False)).sum()) == 0
        and int((ceda_t["doi_source"].fillna("") == "stac_link_describedby").sum()) == 0
    )
    tpca_t = df_all[df_all["raw_id"] == "ZENODO:4922168"]
    v1081_t = df_all[df_all["raw_id"] == "ZENODO:10815170"]
    hugo_t = df_all[df_all["raw_id"] == "DATACITE:10.6096/13"]
    jof_t = df_all[df_all["raw_id"] == "DATACITE:10.20783/dias.667"]
    coseb_t = df_all[df_all["raw_id"] == "DATACITE:10.57760/sciencedb.27228"]
    pml_t = df_all[df_all["raw_id"] == "DATACITE:10.11888/Terre.tpdc.303314"]
    s5p_t = df_all[df_all["raw_id"] == "SENTINEL5P_DAILY_AUT"]
    sub_null_t = int(
        (df_canon["title"].fillna("").str.contains(r"\bLevel[- ]?[34][A-Z]\b", regex=True, case=False) & df_canon["processing_level"].isna()).sum()
    )
    nh_swe_t = df_all[df_all["record_id"] == "14_ceos_fedeo_agencies::edf8abd23f4a40aabd4d52e48dec06ea"]
    pool_sum_t = pd.read_csv(BASE_DIR / "provider_pool_and_sentinel_summary.csv")
    t_ok = (
        t_jsonl_ok
        and t_ceda_ok
        and len(tpca_t) == 1 and float(tpca_t.iloc[0]["bbox_west"]) == 150.0 and bool(tpca_t.iloc[0]["bbox_crosses_antimeridian"])
        and len(v1081_t) == 1 and v1081_t.iloc[0]["temporal_output_cadences"] == "monthly; annual; static"
        and sub_null_t == 0
        and len(nh_swe_t) == 1 and nh_swe_t.iloc[0]["geographic_coverage_class"] == "hemispheric_or_pantropical"
        and nh_swe_t.iloc[0]["processing_level"] == "L3" and str(nh_swe_t.iloc[0]["doi"]).lower() == "10.5285/edf8abd23f4a40aabd4d52e48dec06ea"
        and len(hugo_t) == 1 and float(hugo_t.iloc[0]["adjudicated_output_grid_meters"]) == 100.0 and hugo_t.iloc[0]["provider_pool"] == "eu_non_copernicus"
        and len(jof_t) == 1 and jof_t.iloc[0]["observation_domain_mask"] == "ice_free_ocean_mask"
        and len(coseb_t) == 1 and float(coseb_t.iloc[0]["adjudicated_output_grid_meters"]) == 5566.0
        and len(pml_t) == 1 and float(pml_t.iloc[0]["adjudicated_output_grid_meters"]) == 11132.0
        and len(s5p_t) >= 1 and bool((s5p_t["operational_maturity_class"] != "operational_ongoing_series").all())
        and "sentinel_lineage_status" in pool_sum_t.columns
    )
    if t_ok:
        checks_passed += 1
        benchmarks.append(
            (
                "Strict JSONL, FedEO CEDA DOIs, Antimeridian, Sublevels, Thematic Releases & Lifecycle",
                "PASS",
                f"jsonl_ok={t_jsonl_ok}, ceda_ok={t_ceda_ok}, sublevel_null={sub_null_t}, snow_swe=L3/NH, tpca=antim, hugonnet=100m, s5p_aut!=ongoing",
            )
        )
    else:
        issues_found.append(f"Check T failed: jsonl={t_jsonl_ok}, ceda={t_ceda_ok}, sub_null={sub_null_t}")

    # Check U: HTML comparison preservation, URL query preservation, Sentinel-2 lineage, paired SMAP vs Snow CCI SWE + HTAPv3 geo, GPCP v3.3 & CERES 4.2.1
    jrc_u = df_all[df_all["record_id"] == "07_ec_jrc_data_catalogue::162768e1-570f-462c-b757-37e57085d773"]
    fco2_u = df_all[df_all["raw_id"] == "ZENODO:15201316"]
    smap_u = df_all[df_all["record_id"] == "16_nasa_cmr::C2938664439-NSIDC_CPRD"]
    htap_u = df_all[df_all["record_id"] == "07_ec_jrc_data_catalogue::8626f4f4-67e3-4d0a-9457-d432f35ae3da"]
    swe_c_u = df_canon[df_canon["canonical_product_key"] == "doi:10.5285/edf8abd23f4a40aabd4d52e48dec06ea"]
    gpcp_u = df_all[df_all["record_id"] == "20_noaa_onestop::7180fc6a-a5a4-4857-925a-a0fa4f7f7727"]
    gpcp_day_u = df_all[df_all["record_id"] == "20_noaa_onestop::07090ac2-b1fa-4335-b195-745b22b68398"]
    ce1_u = df_all[df_all["record_id"] == "16_nasa_cmr::C3880496704-LARC_CLOUD"]
    wekeo_utci_u = df_all[df_all["record_id"] == "09_wekeo_dias::EO:ECMWF:DAT:DERIVED_UTCI_HISTORICAL"]
    noaa_mesa_u = df_all[df_all["record_id"] == "20_noaa_onestop::6dc622ed-8446-4bc0-af4c-0b6b15bcacdd"]
    usgs_glccd_u = df_all[df_all["record_id"] == "16_nasa_cmr::C1220566586-USGS_LTA"]
    tellus_gr_u = df_all[df_all["record_id"] == "16_nasa_cmr::C3206299308-POCLOUD"]
    tellus_oc_u = df_all[df_all["record_id"] == "16_nasa_cmr::C3206302778-POCLOUD"]
    cmems_omi_u = df_all[df_all["record_id"] == "01_copernicus_cmems::OMI_CLIMATE_THSL_GLOBAL_area_averaged_anomalies_0_700"]
    url_bare_amp_u = int(df_all["access_url"].fillna("").str.contains(r"&[A-Za-z0-9_]+$", regex=True).sum())
    u_ok = (
        len(jrc_u) == 1 and bool(jrc_u.iloc[0]["uses_sentinel_data"]) and "<2010 and 2019" in str(jrc_u.iloc[0]["description"])
        and len(fco2_u) == 1 and bool(fco2_u.iloc[0]["uses_sentinel_data"]) and "CH<5m" in str(fco2_u.iloc[0]["description"])
        and len(smap_u) == 1 and smap_u.iloc[0]["geographic_coverage_class"] == "global"
        and len(htap_u) == 1 and htap_u.iloc[0]["geographic_coverage_class"] == "global"
        and len(swe_c_u) == 1 and swe_c_u.iloc[0]["geographic_coverage_class"] == "hemispheric_or_pantropical"
        and len(gpcp_u) == 1 and gpcp_u.iloc[0]["temporal_output_cadences"] == "monthly_or_seasonal" and float(gpcp_u.iloc[0]["adjudicated_output_grid_meters"]) == 55660.0
        and len(gpcp_day_u) == 1 and float(gpcp_day_u.iloc[0]["adjudicated_output_grid_meters"]) == 55660.0 and gpcp_day_u.iloc[0]["adjudicated_spatial_resolution_source"] == "prose_delivered_output_grid"
        and len(ce1_u) == 1 and ce1_u.iloc[0]["geographic_coverage_class"] == "global"
        and len(tellus_gr_u) == 1 and tellus_gr_u.iloc[0]["geographic_coverage_class"] == "regional_or_local" and pd.isna(tellus_gr_u.iloc[0]["adjudicated_output_grid_meters"]) and tellus_gr_u.iloc[0]["prose_spatial_role"] == "parent_input_grid_for_aggregate_time_series"
        and len(tellus_oc_u) == 1 and pd.isna(tellus_oc_u.iloc[0]["adjudicated_output_grid_meters"]) and tellus_oc_u.iloc[0]["prose_spatial_role"] == "parent_input_grid_for_aggregate_time_series"
        and len(cmems_omi_u) == 1 and pd.isna(cmems_omi_u.iloc[0]["adjudicated_output_grid_meters"])
        and len(wekeo_utci_u) == 1 and "&dataset=EO:ECMWF:DAT:DERIVED_UTCI_HISTORICAL" in str(wekeo_utci_u.iloc[0]["access_url"])
        and len(noaa_mesa_u) == 1 and "&view=getDataView&header=none" in str(noaa_mesa_u.iloc[0]["access_url"])
        and len(usgs_glccd_u) == 1 and "&full_datasource=" in str(usgs_glccd_u.iloc[0]["access_url"])
        and url_bare_amp_u == 0
    )
    if u_ok:
        checks_passed += 1
        benchmarks.append(
            (
                "HTML Comparison Hygiene, URL Query Preservation, S2 Lineage, Paired SMAP/SWE & Spot Panel Invariants",
                "PASS",
                f"jrc_162768e1=S2, forestco2=S2, url_bare_amp={url_bare_amp_u}, smap_spl3ftp_e=global, htapv3=global, snow_swe=NH, gpcp_v33=0.5deg(monthly+daily), ceres_ebaf=global",
            )
        )
    else:
        issues_found.append(f"Check U failed: HTML/URL/S2/SMAP/GPCP/CERES invariant check (url_bare_amp={url_bare_amp_u})")

    # Check V: Non-European Primary Portals (JAXA, DE Africa/Pacific, NRCan) & Cross-Pool Mirror Hygiene
    s30_v = df_all[df_all["source_catalogue"] == "30_jaxa_earth_stac"]
    s31_v = df_all[df_all["source_catalogue"] == "31_digital_earth_africa_and_pacific"]
    s32_v = df_all[df_all["source_catalogue"] == "32_nrcan_ccmeo_stac"]
    jaxa_c3s_v = s30_v[s30_v["raw_id"] == "Copernicus.C3S_PROBA-V_LCCS_global_yearly"]
    jaxa_aw3d_v = s30_v[s30_v["raw_id"] == "JAXA.EORC_ALOS.PRISM_AW3D30.v4.1_global"]
    dep_v = s31_v[s31_v["raw_id"].astype(str).str.startswith("DEPACIFIC:")]
    nrcan_can_v = s32_v[s32_v["raw_id"].isin(["modis-land-cover", "canadian-wetland-inventory"])]
    ghsl_v = df_canon[(df_canon["title"].fillna("") + " " + df_canon["raw_id"].fillna("")).str.contains(r"\bghs[-_ ]|\bghsl\b|global human settlement", case=False, regex=True)]
    s2gm_fedeo_v = df_all[df_all["record_id"] == "14_ceos_fedeo_agencies::sentinel-2-global-mosaics"]
    asap_v = df_canon[(df_canon["source_catalogue"] == "15_geobon_and_geokhub") & df_canon["title"].fillna("").str.contains(r"\bASAP\s+(?:Warnings|Hotspots|warnings and hotspots)\b", case=False, regex=True)]
    wsf_v = df_canon[(df_canon["title"].fillna("") + " " + df_canon["raw_id"].fillna("")).str.contains(r"world settlement footprint|\bwsf[-_](?:2015|2019|evolution|3d)\b", case=False, regex=True)]
    wpop_v = df_canon[(df_canon["title"].fillna("") + " " + df_canon["raw_id"].fillna("")).str.contains(r"\bworldpop\b", case=False, regex=True)]
    spot_l1_v = df_all[df_all["raw_id"].astype(str).str.startswith("EO:CNES:DAT:SWH")]
    s2_cprob_v = df_all[df_all["record_id"] == "27_gee_official_stac::COPERNICUS/S2_CLOUD_PROBABILITY"]

    v_ok = (
        len(s30_v) == 112
        and len(jaxa_c3s_v) == 1 and jaxa_c3s_v.iloc[0]["provider_pool"] == "copernicus_services" and jaxa_c3s_v.iloc[0]["copernicus_service"] == "C3S"
        and len(jaxa_aw3d_v) == 1 and jaxa_aw3d_v.iloc[0]["provider_pool"] == "non_eu" and str(jaxa_aw3d_v.iloc[0]["doi"]) == "10.57746/EO.01kkfx0ejw8r5ndc9bg5nt13ws"
        and len(s31_v) >= 85
        and len(dep_v) == 13 and (dep_v["geographic_coverage_class"] == "regional_or_local").all() and dep_v["temporal_start_year"].dropna().min() >= 2017
        and len(s32_v) == 39 and (s32_v["provider_pool"] == "non_eu").all() and (nrcan_can_v["geographic_coverage_class"] == "regional_or_local").all()
        and len(ghsl_v) >= 300 and (ghsl_v["provider_pool"] == "copernicus_services").all()
        and len(s2gm_fedeo_v) == 1 and s2gm_fedeo_v.iloc[0]["provider_pool"] == "copernicus_services" and s2gm_fedeo_v.iloc[0]["product_level_subtype"] == "L3_ARD_REFLECTANCE"
        and len(asap_v) == 3 and (asap_v["provider_pool"] == "eu_non_copernicus").all()
        and len(wsf_v) >= 10 and (wsf_v["provider_pool"] == "eu_non_copernicus").all()
        and len(wpop_v) >= 15 and (wpop_v["provider_pool"] == "eu_non_copernicus").all()
        and len(spot_l1_v) == 0 and len(s2_cprob_v) == 0
    )
    if v_ok:
        checks_passed += 1
        benchmarks.append(
            (
                "Primary Non-EU Portals (JAXA, DE Africa/Pacific, NRCan) & Cross-Pool Hygiene",
                "PASS",
                f"jaxa={len(s30_v)}, deap={len(s31_v)}, nrcan={len(s32_v)}, ghsl_cop={len(ghsl_v)}/{len(ghsl_v)}, s2gm=CLMS/ARD, spot_l1=0, s2_cprob=0",
            )
        )
    else:
        issues_found.append("Check V failed: new primary portals or cross-pool mirror hygiene invariant check")

    # Check W: US GHG Center (33), FAO Geospatial CKAN (34), CNES GEODES (35) & Flagship DataCite Additions
    s33_w = df_all[df_all["source_catalogue"] == "33_us_ghg_center_stac"]
    s34_w = df_all[df_all["source_catalogue"] == "34_fao_geospatial_catalog_ckan"]
    s35_w = df_all[df_all["source_catalogue"] == "35_cnes_geodes_stac"]
    ghg_oco2_w = s33_w[s33_w["raw_id"] == "oco2-mip-co2budget-yeargrid-v1"]
    ghg_gosat_w = s33_w[s33_w["raw_id"] == "gosat-based-ch4budget-yeargrid-v1"]
    ghg_epa_w = s33_w[s33_w["raw_id"] == "epa-ch4emission-yeargrid-v2express"]
    ghg_odiac_w = s33_w[s33_w["raw_id"].astype(str).str.startswith("odiac-ffco2-")]
    fao_glc_w = s34_w[s34_w["title"].fillna("").str.contains("GLC-SHARE", case=False)]
    fao_hwsd2_w = s34_w[s34_w["title"].fillna("").str.contains("Harmonized World Soils Database v\\.2", case=False, regex=True)]
    geodes_oso_w = s35_w[s35_w["raw_id"] == "THEIA_OSO_RASTER_L3B"]
    geodes_flat_w = s35_w[s35_w["raw_id"] == "FLATSIM_TIMESERIE"]
    nceo_w = df_all[df_all["raw_id"].astype(str).str.lower() == "datacite:10.25392/leicester.data.15060270.v1"]
    wfdei_w = df_all[df_all["raw_id"].astype(str).str.lower() == "datacite:10.5065/486n-8109"]
    mirad_w = df_all[df_all["raw_id"].astype(str).str.lower() == "datacite:10.5066/p9na3eo8"]
    lanid_w = df_all[df_all["raw_id"].astype(str).str.lower() == "datacite:10.5281/zenodo.5548555"]
    xu_bio_w = df_all[df_all["raw_id"].astype(str).str.lower() == "datacite:10.5281/zenodo.4161694"]
    mirca_w = df_all[df_all["raw_id"].astype(str).str.lower() == "datacite:10.5281/zenodo.7422506"]
    mircaos_w = df_all[df_all["raw_id"].astype(str).str.lower() == "datacite:10.4211/hs.60a890eb841c460192c03bb590687145"]

    s33_emit_w = s33_w[s33_w["raw_id"].isin(["emit-ch4plume-v1", "emit-ch4plume-v2"])]
    exc_log_w = pd.read_parquet(REPO_ROOT / "02_preparation/data/excluded_records_log.parquet")
    s33_emit_exc_w = exc_log_w[
        (exc_log_w["source_id"] == "33_us_ghg_center_stac")
        & (exc_log_w["raw_id"].isin(["emit-ch4plume-v1", "emit-ch4plume-v2"]))
    ]
    w_ok = (
        len(s33_w) == 35 and len(s33_emit_w) == 0 and len(s33_emit_exc_w) == 2 and (s33_w["provider_pool"] == "non_eu").all()
        and len(ghg_oco2_w) == 1 and str(ghg_oco2_w.iloc[0]["doi"]) == "10.48588/npf6-sw92"
        and len(ghg_gosat_w) == 1 and ghg_gosat_w.iloc[0]["geographic_coverage_class"] == "global"
        and len(ghg_epa_w) == 1 and ghg_epa_w.iloc[0]["geographic_coverage_class"] == "regional_or_local" and ghg_epa_w.iloc[0]["product_ontology_class"] == "non_eo_inventory_or_census"
        and len(ghg_odiac_w) == 4 and (ghg_odiac_w["product_ontology_class"] == "eo_constrained_hybrid").all()
        and len(s34_w) >= 840 and (s34_w["provider_pool"] == "non_eu").all() and (s34_w["provider_jurisdiction_tier"] == "un_or_international_body").all()
        and len(fao_glc_w) == 12 and len(fao_hwsd2_w) >= 2
        and len(s35_w) == 23 and (s35_w["provider_pool"] == "eu_non_copernicus").all()
        and len(geodes_oso_w) == 1 and geodes_oso_w.iloc[0]["geographic_coverage_class"] == "europe_regional" and bool(geodes_oso_w.iloc[0]["uses_sentinel_data"])
        and len(geodes_flat_w) == 1 and bool(geodes_flat_w.iloc[0]["uses_sentinel_data"]) and "S1" in str(geodes_flat_w.iloc[0]["sentinel_missions"])
        and len(nceo_w) == 1 and nceo_w.iloc[0]["provider_pool"] == "eu_non_copernicus" and nceo_w.iloc[0]["provider_jurisdiction_tier"] == "copernicus_associated_state"
        and len(wfdei_w) == 1 and wfdei_w.iloc[0]["provider_pool"] == "eu_non_copernicus" and wfdei_w.iloc[0]["geographic_coverage_class"] == "global"
        and len(mirad_w) == 1 and mirad_w.iloc[0]["provider_pool"] == "non_eu"
        and len(lanid_w) == 1 and lanid_w.iloc[0]["provider_pool"] == "non_eu"
        and len(xu_bio_w) == 1 and xu_bio_w.iloc[0]["provider_pool"] == "non_eu"
        and len(mirca_w) == 1 and mirca_w.iloc[0]["provider_pool"] == "eu_non_copernicus" and mirca_w.iloc[0]["product_ontology_class"] == "eo_constrained_hybrid"
        and len(mircaos_w) == 1 and mircaos_w.iloc[0]["provider_pool"] == "non_eu" and mircaos_w.iloc[0]["product_ontology_class"] == "eo_constrained_hybrid"
    )
    if w_ok:
        checks_passed += 1
        benchmarks.append(
            (
                "US GHG Center (33), FAO CKAN (34), CNES GEODES (35) & Flagship DataCite Releases",
                "PASS",
                f"ghg={len(s33_w)}, fao_ckan={len(s34_w)}(glc_share={len(fao_glc_w)}, hwsd2={len(fao_hwsd2_w)}), geodes={len(s35_w)}(flatsim=S1), datacite_8=OK",
            )
        )
    else:
        issues_found.append("Check W failed: US GHG Center, FAO Geospatial CKAN, CNES GEODES, or DataCite benchmark invariant check")

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
