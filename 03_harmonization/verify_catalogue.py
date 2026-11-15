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
    df_canon = pd.read_parquet(HARM_DATA_DIR / "eo_l3_l4_canonical_products.parquet")

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
    )
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
        and "daily" in str(cds_oc.iloc[0]["temporal_resolution_raw"]).lower()
        and "monthly" in str(cds_oc.iloc[0]["temporal_resolution_raw"]).lower()
    )
    gee_cadence_literal = int((df_all["temporal_resolution_raw"] == "cadence").sum())
    merra_2 = df_all[(df_all["source_catalogue"] == "16_nasa_cmr") & (df_all["raw_id"] == "C2531963170-GES_DISC")]
    merra_2_ok = bool(
        len(merra_2) == 1
        and "10.5067/o8ax56do60mi" in str(merra_2.iloc[0]["access_url"]).lower()
        and "0.625 x 0.5" in str(merra_2.iloc[0]["spatial_resolution_raw"])
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
    ):
        checks_passed += 1
        benchmarks.append(
            (
                "UMM/OneStop Deep Metadata, Cadence, Maturity & Lineage",
                "PASS",
                f"noaa_desc={noaa_desc_rate:.1%}, cmr_doi={cmr_doi_rate:.1%}, cmr_spat={cmr_spat_cnt}, contra={cadence_contra_cnt}, op_null={op_null_cnt}",
            )
        )
    else:
        issues_found.append(
            f"Check P failed: noaa_desc={noaa_desc_rate:.3f}, noaa_bbox={noaa_bbox_rate:.3f}, cmr_doi={cmr_doi_rate:.3f}, "
            f"cmr_spat={cmr_spat_cnt}, img_urls={img_url_cnt}, cadence_contra={cadence_contra_cnt}, op_null={op_null_cnt}, "
            f"pre2014_sent={pre2014_sent_cnt}, eum_0088={has_eum_0088}, fluxcom={has_fluxcom}, lcspp={has_lcspp}, cds_oc={cds_oc_ok}"
        )

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
