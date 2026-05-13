"""
Demurrage & Detention Analyzer - IKEA CSV version
====================================================

This keeps the original D&D shipment/event/reporting logic and uses one IKEA
CSV tariff path.

Core shipment event logic retained:
  POL Demurrage = CLL - CGI
  POD Demurrage = CGO - CDD
  POD Detention = CER - CGO

IKEA contract logic:
  - Tariff match is country-level only.
  - Ignore Port of discharge state.
  - Match shipment DESTINATION_COUNTRY to contract Port of discharge country.
  - Container mapping:
      "20 ft Standard Container" -> 22GP
      "40 ft Standard Container" -> 42GP
      "40 ft High Cube Container" -> 42GP
    Everything else is an unmapped contract gap.
  - IKEA file prices only POD demurrage and POD detention.
  - No tiers, no combined free days, no POL demurrage pricing.
  - Cost = max(0, days - free days) * flat daily tariff.

Run:
  streamlit run ikea_demurrage_detention_analyzer.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime
from io import BytesIO

# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Demurrage & Detention Analyzer",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# COLORS
# -----------------------------------------------------------------------------
POL_DEM_COLOR = "#00a6ff"
DEM_COLOR = "#f5a623"
DET_COLOR = "#7b61ff"
TOTAL_COLOR = "#00d4aa"
ALERT_COLOR = "#ff5c5c"
TIER1_COLOR = "#8fd694"
TIER2_COLOR = "#f5a623"
THEREAFTER_COLOR = "#ff5c5c"

# -----------------------------------------------------------------------------
# CUSTOM CSS - same visual intent as original
# -----------------------------------------------------------------------------
st.markdown(
    """
<style>
    .block-container { padding-top: 1.5rem; max-width: 1250px; }
    div[data-testid="stTabs"] { margin-top: 0.5rem !important; }
    div[data-testid="stTabs"] div[role="tablist"] {
        min-height: 64px !important; height: 64px !important;
        padding-top: 8px !important; padding-bottom: 14px !important;
        margin-bottom: 18px !important; border-bottom: 1px solid #2a2d3a !important;
        overflow: visible !important; gap: 18px !important;
    }
    div[data-testid="stTabs"] button[role="tab"] {
        min-height: 48px !important; height: 48px !important;
        padding: 8px 8px 12px 8px !important; margin: 0 !important;
        overflow: visible !important; border-bottom: none !important; background: transparent !important;
    }
    div[data-testid="stTabs"] button[role="tab"] p {
        color: #cbd5e1 !important; font-size: 15px !important;
        font-weight: 800 !important; line-height: 22px !important; margin: 0 !important;
        padding: 0 !important; white-space: nowrap !important; overflow: visible !important;
        text-overflow: unset !important;
    }
    div[data-testid="stTabs"] button[role="tab"]:hover p { color: #ffffff !important; }
    div[data-testid="stTabs"] button[aria-selected="true"] p { color: #ff4b4b !important; font-weight: 900 !important; }
    div[data-testid="stTabs"] button[aria-selected="true"] { border-bottom: 4px solid #ff4b4b !important; }
    div[data-testid="stTabs"] div { overflow: visible !important; }
    div[data-testid="stMetric"] {
        background: #111827; border: 1px solid #2a2d3a; border-radius: 10px; padding: 18px 20px;
    }
    div[data-testid="stMetric"] label {
        color: #ffffff !important; font-size: 13px !important; text-transform: uppercase;
        letter-spacing: 0.8px; font-weight: 900 !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #ffffff !important; font-size: 32px !important; font-weight: 900 !important;
    }
    div[data-testid="stMetricDelta"] { color: #22c55e !important; font-weight: 900 !important; }
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# STANDARD CONTRACT HELPERS - retained internally, not exposed in IKEA-only UI
# -----------------------------------------------------------------------------
NUMERIC_CONTRACT_COLS = [
    "freeDemurrageDays",
    "firstDemurrageDays",
    "firstDemurrageRate",
    "secondDemurrageDays",
    "secondDemurrageRate",
    "thereafterDemurrageRate",
    "freeDetentionDays",
    "firstDetentionDays",
    "firstDetentionRate",
    "secondDetentionDays",
    "secondDetentionRate",
    "thereafterDetentionRate",
    "combinedFreeDays",
]


def parse_contracts_csv(contract_file):
    cdf = pd.read_csv(contract_file)
    optional_cols = [
        "terminalIdentifier", "demurrageStartEventType", "demurrageTariffCalculationMethod",
        "validityStartDate", "validityEndDate", "freeDemurrageDays", "firstDemurrageDays",
        "firstDemurrageRate", "secondDemurrageDays", "secondDemurrageRate", "thereafterDemurrageRate",
        "detentionStartEventType", "detentionTariffCalculationMethod", "freeDetentionDays",
        "firstDetentionDays", "firstDetentionRate", "secondDetentionDays", "secondDetentionRate",
        "thereafterDetentionRate", "currency", "carrierScac", "ffwScac", "portOfLoadingLocode",
        "combinedFreeDays",
    ]
    for col in optional_cols:
        if col not in cdf.columns:
            cdf[col] = np.nan
    for col in NUMERIC_CONTRACT_COLS:
        cdf[col] = pd.to_numeric(cdf[col], errors="coerce")
    records = cdf.to_dict(orient="records")
    for rec in records:
        for col, val in list(rec.items()):
            if isinstance(val, float) and np.isnan(val):
                rec[col] = None
            elif isinstance(val, str) and val.strip() == "":
                rec[col] = None
    return records, cdf


def _event_scope(value):
    text = str(value or "").upper()
    if "POL" in text:
        return "POL"
    if "POD" in text:
        return "POD"
    return None


def _has_any_number(rec, cols):
    for col in cols:
        val = rec.get(col)
        if val is None:
            continue
        try:
            if not np.isnan(val):
                return True
        except (TypeError, ValueError):
            return True
    return False


def _has_demurrage_terms(rec):
    return _has_any_number(rec, [
        "freeDemurrageDays", "firstDemurrageDays", "firstDemurrageRate",
        "secondDemurrageDays", "secondDemurrageRate", "thereafterDemurrageRate",
    ])


def _has_detention_terms(rec):
    return _has_any_number(rec, [
        "freeDetentionDays", "firstDetentionDays", "firstDetentionRate",
        "secondDetentionDays", "secondDetentionRate", "thereafterDetentionRate",
    ])

# -----------------------------------------------------------------------------
# IKEA CONTRACT PARSER AND MAPPING
# -----------------------------------------------------------------------------
IKEA_REQUIRED_COLUMNS = [
    "Port of discharge country",
    "Country name",
    "Container type",
    "Demurrage Free Time",
    "Demurrage Tariff",
    "Detention Freetime",
    "Detention Tariff",
]

IKEA_CONTAINER_MAP = {
    "20 FT STANDARD CONTAINER": "22GP",
    "40 FT STANDARD CONTAINER": "42GP",
    "40 FT HIGH CUBE CONTAINER": "42GP",
}


def _clean_text(val):
    if val is None:
        return ""
    try:
        if pd.isna(val):
            return ""
    except (TypeError, ValueError):
        pass
    return str(val).replace("\u00a0", " ").replace("\r", " ").replace("\n", " ").strip()


def _norm_key(val):
    # Robust key cleanup: removes leading/trailing spaces, non-breaking spaces, line breaks,
    # collapses repeated internal spaces, and uppercases both shipment and tariff values.
    text = _clean_text(val)
    return " ".join(text.split()).upper()


def map_ikea_container_type(container_type):
    """Strict IKEA mapping for shipment container values supported by IKEA tariff."""
    text = _norm_key(container_type)
    return IKEA_CONTAINER_MAP.get(text)


def parse_ikea_tariff_csv(contract_file):
    """Read IKEA tariff CSV and create a lookup by destination country + container type."""
    cdf = pd.read_csv(contract_file)
    cdf.columns = [str(c).strip() for c in cdf.columns]
    missing = [c for c in IKEA_REQUIRED_COLUMNS if c not in cdf.columns]
    if missing:
        raise ValueError(f"Missing IKEA tariff columns: {', '.join(missing)}")

    for col in ["Demurrage Free Time", "Demurrage Tariff", "Detention Freetime", "Detention Tariff"]:
        cdf[col] = pd.to_numeric(cdf[col], errors="coerce")

    records = []
    lookup = {}
    duplicate_keys = []
    for _, row in cdf.iterrows():
        pod_country = _norm_key(row.get("Port of discharge country"))
        container_type = _norm_key(row.get("Container type"))
        if not pod_country or not container_type:
            continue
        rec = {
            "POD_COUNTRY": pod_country,
            "COUNTRY_NAME": _clean_text(row.get("Country name")),
            "CONTAINER_TYPE_CONTRACT": container_type,
            "DEM_FREE_DAYS": row.get("Demurrage Free Time"),
            "DEM_RATE": row.get("Demurrage Tariff"),
            "DET_FREE_DAYS": row.get("Detention Freetime"),
            "DET_RATE": row.get("Detention Tariff"),
            "RATE_SOURCE": "IKEA Tariff",
        }
        key = f"{pod_country}|{container_type}"
        if key in lookup:
            existing = lookup[key]
            same = (
                _safe(existing.get("DEM_FREE_DAYS"), None) == _safe(rec.get("DEM_FREE_DAYS"), None)
                and _safe(existing.get("DEM_RATE"), None) == _safe(rec.get("DEM_RATE"), None)
                and _safe(existing.get("DET_FREE_DAYS"), None) == _safe(rec.get("DET_FREE_DAYS"), None)
                and _safe(existing.get("DET_RATE"), None) == _safe(rec.get("DET_RATE"), None)
            )
            if not same:
                duplicate_keys.append(key)
            # Same key can repeat by state; keep the first country-level rate.
            continue
        lookup[key] = rec
        records.append(rec)

    if duplicate_keys:
        raise ValueError(
            "IKEA tariff has conflicting duplicate rates for: " + ", ".join(sorted(set(duplicate_keys))[:20])
        )

    return records, cdf, lookup

# -----------------------------------------------------------------------------
# CALCULATION HELPERS
# -----------------------------------------------------------------------------
def _safe(val, default=0):
    if val is None:
        return default
    try:
        if np.isnan(val):
            return default
    except (TypeError, ValueError):
        pass
    return val


def _days_between(end_ts, start_ts):
    if pd.isna(end_ts) or pd.isna(start_ts):
        return None
    return max(0, (end_ts - start_ts).total_seconds() / 86400)


def calc_tiered_cost_breakdown(chargeable_days, t1_days, t1_rate, t2_days, t2_rate, thereafter_rate):
    result = {
        "tier1_days": 0.0, "tier1_cost": 0.0,
        "tier2_days": 0.0, "tier2_cost": 0.0,
        "thereafter_days": 0.0, "thereafter_cost": 0.0,
        "total_cost": 0.0,
    }
    if chargeable_days <= 0:
        return result
    remaining = chargeable_days
    t1 = min(remaining, _safe(t1_days))
    result["tier1_days"] = t1
    result["tier1_cost"] = t1 * _safe(t1_rate)
    remaining -= t1
    if remaining > 0:
        t2 = min(remaining, _safe(t2_days))
        result["tier2_days"] = t2
        result["tier2_cost"] = t2 * _safe(t2_rate)
        remaining -= t2
    if remaining > 0:
        result["thereafter_days"] = remaining
        result["thereafter_cost"] = remaining * _safe(thereafter_rate)
    result["total_cost"] = round(result["tier1_cost"] + result["tier2_cost"] + result["thereafter_cost"], 2)
    return result


def calc_flat_cost_breakdown(chargeable_days, rate):
    """Populate tier-like fields so existing tier columns/reports do not break."""
    chargeable_days = max(0, _safe(chargeable_days, 0))
    cost = round(chargeable_days * _safe(rate, 0), 2)
    return {
        "tier1_days": chargeable_days,
        "tier1_cost": cost,
        "tier2_days": 0.0,
        "tier2_cost": 0.0,
        "thereafter_days": 0.0,
        "thereafter_cost": 0.0,
        "total_cost": cost,
    }


def _blank_contract_profile():
    return {"pol_dem": None, "pod_dem": None, "pod_det": None, "source_records": [], "is_estimate": False}


def _put_profile(lookup, key):
    if key not in lookup:
        lookup[key] = _blank_contract_profile()
    return lookup[key]


def build_contract_lookup(contracts_list):
    carrier_lookup = {}
    ffw_lookup = {}
    for c in contracts_list or []:
        terminal = str(c.get("terminalIdentifier", "") or "").strip()
        pol = str(c.get("portOfLoadingLocode", "") or "").strip()
        carrier = c.get("carrierScac")
        ffw = c.get("ffwScac")
        keys = []
        if carrier:
            keys.append((carrier_lookup, f"{terminal}|{str(carrier).strip()}|{pol}"))
        if ffw:
            keys.append((ffw_lookup, f"{terminal}|{str(ffw).strip()}|{pol}"))
        dem_scope = _event_scope(c.get("demurrageStartEventType"))
        has_dem = _has_demurrage_terms(c)
        has_det = _has_detention_terms(c)
        for lookup, key in keys:
            profile = _put_profile(lookup, key)
            profile["source_records"].append(c)
            if has_dem and dem_scope == "POL" and profile["pol_dem"] is None:
                profile["pol_dem"] = c
            elif has_dem and dem_scope == "POD" and profile["pod_dem"] is None:
                profile["pod_dem"] = c
            if has_det and profile["pod_det"] is None:
                profile["pod_det"] = c
    return carrier_lookup, ffw_lookup


def make_estimate_contract_profile(
    pol_free_dem, pol_t1_days, pol_t1_rate, pol_t2_days, pol_t2_rate, pol_thereafter_rate,
    use_combined_pod_free, combined_pod_free_days,
    pod_free_dem, pod_t1_days, pod_t1_rate, pod_t2_days, pod_t2_rate, pod_thereafter_rate,
    pod_free_det, det_t1_days, det_t1_rate, det_t2_days, det_t2_rate, det_thereafter_rate,
):
    pol_dem = {
        "terminalIdentifier": "ESTIMATE", "portOfLoadingLocode": "ESTIMATE",
        "demurrageStartEventType": "POL_ESTIMATE", "freeDemurrageDays": pol_free_dem,
        "firstDemurrageDays": pol_t1_days, "firstDemurrageRate": pol_t1_rate,
        "secondDemurrageDays": pol_t2_days, "secondDemurrageRate": pol_t2_rate,
        "thereafterDemurrageRate": pol_thereafter_rate, "combinedFreeDays": None,
    }
    pod_dem = {
        "terminalIdentifier": "ESTIMATE", "portOfLoadingLocode": "ESTIMATE",
        "demurrageStartEventType": "POD_ESTIMATE",
        "freeDemurrageDays": 0 if use_combined_pod_free else pod_free_dem,
        "firstDemurrageDays": pod_t1_days, "firstDemurrageRate": pod_t1_rate,
        "secondDemurrageDays": pod_t2_days, "secondDemurrageRate": pod_t2_rate,
        "thereafterDemurrageRate": pod_thereafter_rate,
        "combinedFreeDays": combined_pod_free_days if use_combined_pod_free else None,
    }
    pod_det = {
        "terminalIdentifier": "ESTIMATE", "portOfLoadingLocode": "ESTIMATE",
        "detentionStartEventType": "POD_ESTIMATE",
        "freeDetentionDays": 0 if use_combined_pod_free else pod_free_det,
        "firstDetentionDays": det_t1_days, "firstDetentionRate": det_t1_rate,
        "secondDetentionDays": det_t2_days, "secondDetentionRate": det_t2_rate,
        "thereafterDetentionRate": det_thereafter_rate,
        "combinedFreeDays": combined_pod_free_days if use_combined_pod_free else None,
    }
    return {"pol_dem": pol_dem, "pod_dem": pod_dem, "pod_det": pod_det, "source_records": [pol_dem, pod_dem, pod_det], "is_estimate": True}


def _first_existing_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _clean_scac(val):
    return _clean_text(val)


def _shipment_match_identity(row):
    carrier = _clean_scac(row.get("CARRIER_SCAC", ""))
    ffw = _clean_scac(row.get("FFW_SCAC", ""))
    if carrier:
        return carrier, "Carrier"
    if ffw:
        return ffw, "FFW"
    return "", "Missing"


def normalize_required_columns(df):
    ffw_aliases = [
        "FFW_SCAC", "FFW", "FFW_SCAC_CODE", "FREIGHT_FORWARDER_SCAC",
        "FREIGHT_FORWARDER", "FORWARDER_SCAC", "FORWARDER", "FREIGHT_FORWARDER_CODE",
    ]
    ffw_col = _first_existing_column(df, ffw_aliases)
    if ffw_col is not None and ffw_col != "FFW_SCAC":
        df["FFW_SCAC"] = df[ffw_col]

    container_aliases = ["CONTAINER_TYPE", "EQUIPMENT_TYPE", "CONTAINER_SIZE_TYPE", "CONTAINER_SIZE"]
    container_col = _first_existing_column(df, container_aliases)
    if container_col is not None and container_col != "CONTAINER_TYPE":
        df["CONTAINER_TYPE"] = df[container_col]

    for col in [
        "SHIPMENT_ID", "CONTAINER_NUMBER", "CONTAINER_TYPE", "CARRIER_SCAC", "CARRIER_NAME", "FFW_SCAC",
        "POL_LOCODE", "POL", "POL_COUNTRY", "POD_LOCODE", "POD", "POD_COUNTRY",
        "DESTINATION_COUNTRY", "SUBSCRIPTION_STATUS", "LIFECYCLE_STATUS", "SHIPMENT_MODIFIED_DATE",
    ]:
        if col not in df.columns:
            df[col] = pd.NaT if col == "SHIPMENT_MODIFIED_DATE" else ""
    for col in ["CDD", "CGO", "CER", "VAD", "VDL", "CGI", "CEP", "CLL"]:
        if col not in df.columns:
            df[col] = pd.NaT
    return df


def _get_combined_pod_free_days(pod_dem_contract, pod_det_contract):
    for c in [pod_dem_contract, pod_det_contract]:
        if c is None:
            continue
        val = c.get("combinedFreeDays")
        if val is not None:
            return val
    return None


def _contract_label(profile, component_key):
    c = profile.get(component_key) if profile else None
    if c is None:
        return "Not configured"
    if profile.get("is_estimate"):
        return "Estimate"
    return str(c.get("terminalIdentifier", "")) or "Contract"


def _profile_first_value(profile, field):
    if not profile:
        return ""
    for rec in profile.get("source_records", []) or []:
        val = _clean_scac(rec.get(field, ""))
        if val:
            return val
    for component in ["pol_dem", "pod_dem", "pod_det"]:
        rec = profile.get(component)
        if rec:
            val = _clean_scac(rec.get(field, ""))
            if val:
                return val
    return ""

# -----------------------------------------------------------------------------
# D&D CALCULATION ENGINE
# -----------------------------------------------------------------------------
def _base_row_values(row, analysis_run_date):
    cgi = row["CGI"]
    cll = row["CLL"]
    cdd = row["CDD"]
    cgo = row["CGO"]
    cer = row["CER"]
    sub_status = row.get("SUBSCRIPTION_STATUS", "")

    pol_dem_total_days = _days_between(cll, cgi)
    pod_dem_total_days = _days_between(cgo, cdd)

    pod_det_total_days = None
    det_accumulating = False
    det_end_source = ""
    det_end_ts = pd.NaT
    if not pd.isna(cgo):
        if not pd.isna(cer):
            pod_det_total_days = _days_between(cer, cgo)
            det_end_source = "CER"
            det_end_ts = cer
        elif sub_status == "ACTIVE":
            pod_det_total_days = _days_between(analysis_run_date, cgo)
            det_accumulating = True
            det_end_source = "TODAY"
            det_end_ts = analysis_run_date
        elif sub_status == "COMPLETED":
            modified = row.get("SHIPMENT_MODIFIED_DATE", pd.NaT)
            if not pd.isna(modified):
                pod_det_total_days = _days_between(modified, cgo)
                det_end_ts = modified
            det_end_source = "MODIFIED_DATE"

    return {
        "cgi": cgi, "cll": cll, "cdd": cdd, "cgo": cgo, "cer": cer,
        "sub_status": sub_status,
        "pol_dem_total_days": pol_dem_total_days,
        "pod_dem_total_days": pod_dem_total_days,
        "pod_det_total_days": pod_det_total_days,
        "det_accumulating": det_accumulating,
        "det_end_source": det_end_source,
        "det_end_ts": det_end_ts,
    }


def _make_base_record(row, values, key, matched_lookup_type):
    carrier_scac_value = _clean_scac(row.get("CARRIER_SCAC", ""))
    ffw_scac_value = _clean_scac(row.get("FFW_SCAC", ""))
    carrier_ffw_scac_value = _clean_scac(row.get("CARRIER_FFW_SCAC", ""))
    return {
        "SHIPMENT_ID": row["SHIPMENT_ID"],
        "CONTAINER_NUMBER": row.get("CONTAINER_NUMBER", ""),
        "CONTAINER_TYPE": row.get("CONTAINER_TYPE", ""),
        "IKEA_CONTAINER_TYPE": row.get("IKEA_CONTAINER_TYPE", ""),
        "CONTAINER_MAPPING_STATUS": row.get("CONTAINER_MAPPING_STATUS", ""),
        "CARRIER_SCAC": carrier_scac_value,
        "CARRIER_NAME": row.get("CARRIER_NAME", ""),
        "FFW_SCAC": ffw_scac_value,
        "CARRIER_FFW_SCAC": carrier_ffw_scac_value,
        "MATCHED_PARTY_TYPE": matched_lookup_type,
        "POL_LOCODE": row["POL_LOCODE"],
        "POL": row.get("POL", ""),
        "POL_COUNTRY": row.get("POL_COUNTRY", ""),
        "POD_LOCODE": row["POD_LOCODE"],
        "POD": row.get("POD", ""),
        "POD_COUNTRY": row.get("POD_COUNTRY", ""),
        "SUBSCRIPTION_STATUS": values["sub_status"],
        "LIFECYCLE_STATUS": row.get("LIFECYCLE_STATUS", ""),
        "CGI": values["cgi"] if not pd.isna(values["cgi"]) else pd.NaT,
        "CLL": values["cll"] if not pd.isna(values["cll"]) else pd.NaT,
        "CDD": values["cdd"] if not pd.isna(values["cdd"]) else pd.NaT,
        "CGO": values["cgo"] if not pd.isna(values["cgo"]) else pd.NaT,
        "CER": values["cer"] if not pd.isna(values["cer"]) else pd.NaT,
        "DET_END_TS": values["det_end_ts"],
        "DD_ANCHOR_DATE": values["cdd"] if not pd.isna(values["cdd"]) else (values["cgo"] if not pd.isna(values["cgo"]) else (values["cer"] if not pd.isna(values["cer"]) else (values["cll"] if not pd.isna(values["cll"]) else values["cgi"]))),
        "POL_DEM_TOTAL_DAYS": round(values["pol_dem_total_days"], 2) if values["pol_dem_total_days"] is not None else None,
        "POD_DEM_TOTAL_DAYS": round(values["pod_dem_total_days"], 2) if values["pod_dem_total_days"] is not None else None,
        "POD_DET_TOTAL_DAYS": round(values["pod_det_total_days"], 2) if values["pod_det_total_days"] is not None else None,
        "DET_ACCUMULATING": values["det_accumulating"],
        "DET_END_SOURCE": values["det_end_source"],
        "LANE": f"{row['POL_LOCODE']} → {row['POD_LOCODE']}",
        "MATCH_KEY": key,
    }


def process_shipments(df, contracts_list=None, estimate_profile=None, use_estimate=False, ikea_lookup=None, rate_mode="Standard Contract"):
    df = normalize_required_columns(df.copy())
    event_cols = ["CDD", "CGO", "CER", "VAD", "VDL", "CGI", "CEP", "CLL"]
    for col in event_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    if "REPORTING_DATE" in df.columns:
        df["REPORTING_DATE"] = pd.to_datetime(df["REPORTING_DATE"], errors="coerce", utc=True)
    df["SHIPMENT_MODIFIED_DATE"] = pd.to_datetime(df["SHIPMENT_MODIFIED_DATE"], errors="coerce", utc=True)

    analysis_run_date = pd.Timestamp.now(tz="UTC")
    carrier_lookup, ffw_lookup = build_contract_lookup(contracts_list or [])

    df["SUBSCRIPTION_STATUS"] = df["SUBSCRIPTION_STATUS"].fillna("").astype(str).str.upper()
    cancelled_count = (df["SUBSCRIPTION_STATUS"] == "CANCELLED").sum()
    original_count = len(df)
    df = df[df["SUBSCRIPTION_STATUS"] != "CANCELLED"].copy()

    match_identity = df.apply(_shipment_match_identity, axis=1, result_type="expand")
    df["CARRIER_FFW_SCAC"] = match_identity[0]
    df["MATCHED_PARTY_TYPE"] = match_identity[1]

    is_ikea = rate_mode == "IKEA Tariff CSV"
    if is_ikea:
        df["IKEA_COUNTRY"] = df["DESTINATION_COUNTRY"].apply(_norm_key)
        df["IKEA_CONTAINER_TYPE"] = df["CONTAINER_TYPE"].apply(map_ikea_container_type)
        df["CONTAINER_MAPPING_STATUS"] = np.where(df["IKEA_CONTAINER_TYPE"].notna(), "MAPPED", "UNMAPPED")
        # Keep POD_COUNTRY populated for existing reporting tabs, but source it from DESTINATION_COUNTRY for IKEA.
        df["POD_COUNTRY"] = df["IKEA_COUNTRY"]
        df["_match_key"] = df["IKEA_COUNTRY"] + "|" + df["IKEA_CONTAINER_TYPE"].fillna("")
        df["IKEA_MATCH_KEY"] = df["_match_key"]
    else:
        df["IKEA_CONTAINER_TYPE"] = ""
        df["CONTAINER_MAPPING_STATUS"] = ""
        df["_match_key"] = (
            df["POD_LOCODE"].fillna("").astype(str).str.strip()
            + "|" + df["CARRIER_FFW_SCAC"].fillna("").astype(str).str.strip()
            + "|" + df["POL_LOCODE"].fillna("").astype(str).str.strip()
        )

    matched_results = []
    unmatched_results = []

    for _, row in df.iterrows():
        key = row["_match_key"]
        values = _base_row_values(row, analysis_run_date)
        matched_lookup_type = row.get("MATCHED_PARTY_TYPE", "")

        if is_ikea:
            tariff = (ikea_lookup or {}).get(key)
            base_record = _make_base_record(row, values, key, "IKEA Tariff" if tariff else matched_lookup_type)
            if tariff is None:
                reasons = []
                if not _clean_text(row.get("DESTINATION_COUNTRY")):
                    reasons.append("Missing destination country")
                if not _clean_text(row.get("CONTAINER_TYPE")):
                    reasons.append("Missing container type")
                if row.get("CONTAINER_MAPPING_STATUS") == "UNMAPPED":
                    reasons.append("Unsupported container type for IKEA tariff")
                if _clean_text(row.get("DESTINATION_COUNTRY")) and row.get("CONTAINER_MAPPING_STATUS") == "MAPPED":
                    reasons.append("No IKEA tariff for destination country | container type")
                if pd.isna(values["cdd"]) or pd.isna(values["cgo"]):
                    reasons.append("Cannot evaluate POD demurrage; missing CDD or CGO")
                if pd.isna(values["cgo"]):
                    reasons.append("Cannot evaluate POD detention; missing CGO")
                if pd.isna(values["cer"]) and values["sub_status"] not in ["ACTIVE", "COMPLETED"]:
                    reasons.append("Cannot evaluate POD detention end; missing CER and status not ACTIVE/COMPLETED")
                unmatched_record = base_record.copy()
                unmatched_record.update({
                    "MISSING_CONTRACT_REASON": "No IKEA tariff for destination country | container type",
                    "DATA_LIMITATION": "; ".join(reasons) if reasons else "Dwell days available; fees cannot be calculated without IKEA tariff",
                    "RISK_FLAG": False,
                    "RISK_REASONS": "",
                })
                unmatched_results.append(unmatched_record)
                continue

            pol_dem_chargeable = 0.0
            pol_dem_breakdown = calc_flat_cost_breakdown(0, 0)
            pod_dem_chargeable = 0.0
            pod_dem_breakdown = calc_flat_cost_breakdown(0, 0)
            pod_det_chargeable = 0.0
            pod_det_breakdown = calc_flat_cost_breakdown(0, 0)

            if values["pod_dem_total_days"] is not None:
                pod_dem_chargeable = max(0, values["pod_dem_total_days"] - _safe(tariff.get("DEM_FREE_DAYS")))
                pod_dem_breakdown = calc_flat_cost_breakdown(pod_dem_chargeable, tariff.get("DEM_RATE"))
            if values["pod_det_total_days"] is not None:
                pod_det_chargeable = max(0, values["pod_det_total_days"] - _safe(tariff.get("DET_FREE_DAYS")))
                pod_det_breakdown = calc_flat_cost_breakdown(pod_det_chargeable, tariff.get("DET_RATE"))

            pol_dem_cost = pol_dem_breakdown["total_cost"]
            pod_dem_cost = pod_dem_breakdown["total_cost"]
            pod_det_cost = pod_det_breakdown["total_cost"]
            total_cost = round(pol_dem_cost + pod_dem_cost + pod_det_cost, 2)

            matched_record = base_record.copy()
            matched_record.update({
                "RATE_SOURCE": "IKEA Tariff",
                "POL_DEM_CHARGEABLE_DAYS": 0.0,
                "POL_DEM_COST": 0.0,
                "POD_DEM_CHARGEABLE_DAYS": round(pod_dem_chargeable, 2),
                "POD_DEM_COST": pod_dem_cost,
                "POD_DET_CHARGEABLE_DAYS": round(pod_det_chargeable, 2),
                "POD_DET_COST": pod_det_cost,
                "DEM_COST": round(pod_dem_cost, 2),
                "DET_COST": pod_det_cost,
                "TOTAL_DD_COST": total_cost,
                "POL_FREE_DEM_DAYS": None,
                "POD_FREE_DEM_DAYS": _safe(tariff.get("DEM_FREE_DAYS"), None),
                "FREE_DEM_DAYS": _safe(tariff.get("DEM_FREE_DAYS"), None),
                "FREE_DET_DAYS": _safe(tariff.get("DET_FREE_DAYS"), None),
                "COMBINED_FREE_DAYS": None,
                "CONTRACT_TYPE": "IKEA Flat Rate",
                "POL_DEM_CONTRACT_STATUS": "Not configured for IKEA",
                "POD_DEM_CONTRACT_STATUS": "IKEA Tariff",
                "POD_DET_CONTRACT_STATUS": "IKEA Tariff",
                "CONTRACT_IDENTIFIER": f"{tariff.get('POD_COUNTRY')} | {tariff.get('CONTAINER_TYPE_CONTRACT')}",
                "CONTRACT_POL": "",
                "IKEA_COUNTRY_NAME": tariff.get("COUNTRY_NAME", ""),
                "IKEA_DEM_RATE": tariff.get("DEM_RATE"),
                "IKEA_DET_RATE": tariff.get("DET_RATE"),
                "POL_DEM_TIER1_DAYS": 0.0,
                "POL_DEM_TIER1_COST": 0.0,
                "POL_DEM_TIER2_DAYS": 0.0,
                "POL_DEM_TIER2_COST": 0.0,
                "POL_DEM_THEREAFTER_DAYS": 0.0,
                "POL_DEM_THEREAFTER_COST": 0.0,
                "POD_DEM_TIER1_DAYS": round(pod_dem_breakdown["tier1_days"], 2),
                "POD_DEM_TIER1_COST": round(pod_dem_breakdown["tier1_cost"], 2),
                "POD_DEM_TIER2_DAYS": 0.0,
                "POD_DEM_TIER2_COST": 0.0,
                "POD_DEM_THEREAFTER_DAYS": 0.0,
                "POD_DEM_THEREAFTER_COST": 0.0,
                "POD_DET_TIER1_DAYS": round(pod_det_breakdown["tier1_days"], 2),
                "POD_DET_TIER1_COST": round(pod_det_breakdown["tier1_cost"], 2),
                "POD_DET_TIER2_DAYS": 0.0,
                "POD_DET_TIER2_COST": 0.0,
                "POD_DET_THEREAFTER_DAYS": 0.0,
                "POD_DET_THEREAFTER_COST": 0.0,
            })
            matched_results.append(matched_record)
            continue

        # Original standard/estimate path
        if use_estimate:
            contract_profile = estimate_profile
            matched_lookup_type = "Estimate"
        else:
            contract_profile = carrier_lookup.get(key)
            if contract_profile is not None:
                matched_lookup_type = "Carrier"
            else:
                contract_profile = ffw_lookup.get(key)
                if contract_profile is not None:
                    matched_lookup_type = "FFW"

        base_record = _make_base_record(row, values, key, matched_lookup_type)

        if contract_profile is None:
            reason_parts = []
            if pd.isna(values["cgi"]) or pd.isna(values["cll"]):
                reason_parts.append("Cannot evaluate POL demurrage; missing CGI or CLL")
            if pd.isna(values["cdd"]) or pd.isna(values["cgo"]):
                reason_parts.append("Cannot evaluate POD demurrage; missing CDD or CGO")
            if pd.isna(values["cgo"]):
                reason_parts.append("Cannot evaluate POD detention; missing CGO")
            if pd.isna(values["cer"]) and values["sub_status"] not in ["ACTIVE", "COMPLETED"]:
                reason_parts.append("Cannot evaluate POD detention end; missing CER and status not ACTIVE/COMPLETED")
            unmatched_record = base_record.copy()
            unmatched_record.update({
                "MISSING_CONTRACT_REASON": "No contract for POD | Carrier/FFW | POL",
                "DATA_LIMITATION": "; ".join(reason_parts) if reason_parts else "Dwell days available; fees cannot be calculated without contract",
                "RISK_FLAG": False,
                "RISK_REASONS": "",
            })
            unmatched_results.append(unmatched_record)
            continue

        pol_dem_contract = contract_profile.get("pol_dem")
        pod_dem_contract = contract_profile.get("pod_dem")
        pod_det_contract = contract_profile.get("pod_det")
        combined_free = _get_combined_pod_free_days(pod_dem_contract, pod_det_contract)
        has_combined = combined_free is not None

        pol_dem_chargeable = 0.0
        pol_dem_breakdown = calc_tiered_cost_breakdown(0, 0, 0, 0, 0, 0)
        if pol_dem_contract is not None and values["pol_dem_total_days"] is not None:
            pol_dem_chargeable = max(0, values["pol_dem_total_days"] - _safe(pol_dem_contract.get("freeDemurrageDays")))
            pol_dem_breakdown = calc_tiered_cost_breakdown(
                pol_dem_chargeable,
                pol_dem_contract.get("firstDemurrageDays"), pol_dem_contract.get("firstDemurrageRate"),
                pol_dem_contract.get("secondDemurrageDays"), pol_dem_contract.get("secondDemurrageRate"),
                pol_dem_contract.get("thereafterDemurrageRate"),
            )

        pod_dem_chargeable = 0.0
        remaining_free_for_det = 0.0
        pod_dem_breakdown = calc_tiered_cost_breakdown(0, 0, 0, 0, 0, 0)
        if pod_dem_contract is not None and values["pod_dem_total_days"] is not None:
            if has_combined:
                pod_dem_chargeable = max(0, values["pod_dem_total_days"] - combined_free)
                remaining_free_for_det = max(0, combined_free - values["pod_dem_total_days"])
            else:
                pod_dem_chargeable = max(0, values["pod_dem_total_days"] - _safe(pod_dem_contract.get("freeDemurrageDays")))
            pod_dem_breakdown = calc_tiered_cost_breakdown(
                pod_dem_chargeable,
                pod_dem_contract.get("firstDemurrageDays"), pod_dem_contract.get("firstDemurrageRate"),
                pod_dem_contract.get("secondDemurrageDays"), pod_dem_contract.get("secondDemurrageRate"),
                pod_dem_contract.get("thereafterDemurrageRate"),
            )

        pod_det_chargeable = 0.0
        pod_det_breakdown = calc_tiered_cost_breakdown(0, 0, 0, 0, 0, 0)
        if pod_det_contract is not None and values["pod_det_total_days"] is not None:
            if has_combined:
                pod_det_chargeable = max(0, values["pod_det_total_days"] - remaining_free_for_det)
            else:
                pod_det_chargeable = max(0, values["pod_det_total_days"] - _safe(pod_det_contract.get("freeDetentionDays")))
            pod_det_breakdown = calc_tiered_cost_breakdown(
                pod_det_chargeable,
                pod_det_contract.get("firstDetentionDays"), pod_det_contract.get("firstDetentionRate"),
                pod_det_contract.get("secondDetentionDays"), pod_det_contract.get("secondDetentionRate"),
                pod_det_contract.get("thereafterDetentionRate"),
            )

        pol_dem_cost = pol_dem_breakdown["total_cost"]
        pod_dem_cost = pod_dem_breakdown["total_cost"]
        pod_det_cost = pod_det_breakdown["total_cost"]
        total_cost = round(pol_dem_cost + pod_dem_cost + pod_det_cost, 2)

        matched_record = base_record.copy()
        matched_record.update({
            "RATE_SOURCE": "Estimate" if use_estimate else "Contract",
            "POL_DEM_CHARGEABLE_DAYS": round(pol_dem_chargeable, 2),
            "POL_DEM_COST": pol_dem_cost,
            "POD_DEM_CHARGEABLE_DAYS": round(pod_dem_chargeable, 2),
            "POD_DEM_COST": pod_dem_cost,
            "POD_DET_CHARGEABLE_DAYS": round(pod_det_chargeable, 2),
            "POD_DET_COST": pod_det_cost,
            "DEM_COST": round(pol_dem_cost + pod_dem_cost, 2),
            "DET_COST": pod_det_cost,
            "TOTAL_DD_COST": total_cost,
            "POL_FREE_DEM_DAYS": _safe(pol_dem_contract.get("freeDemurrageDays"), None) if pol_dem_contract else None,
            "POD_FREE_DEM_DAYS": _safe(pod_dem_contract.get("freeDemurrageDays"), None) if pod_dem_contract else None,
            "FREE_DEM_DAYS": _safe(pod_dem_contract.get("freeDemurrageDays"), None) if pod_dem_contract else None,
            "FREE_DET_DAYS": _safe(pod_det_contract.get("freeDetentionDays"), None) if pod_det_contract else None,
            "COMBINED_FREE_DAYS": combined_free,
            "CONTRACT_TYPE": "Estimate" if use_estimate else ("Combined" if has_combined else "Separate"),
            "POL_DEM_CONTRACT_STATUS": _contract_label(contract_profile, "pol_dem"),
            "POD_DEM_CONTRACT_STATUS": _contract_label(contract_profile, "pod_dem"),
            "POD_DET_CONTRACT_STATUS": _contract_label(contract_profile, "pod_det"),
            "CONTRACT_IDENTIFIER": _contract_label(contract_profile, "pod_dem"),
            "CONTRACT_POL": str((pod_dem_contract or pod_det_contract or pol_dem_contract or {}).get("portOfLoadingLocode", "")),
            "POL_DEM_TIER1_DAYS": round(pol_dem_breakdown["tier1_days"], 2),
            "POL_DEM_TIER1_COST": round(pol_dem_breakdown["tier1_cost"], 2),
            "POL_DEM_TIER2_DAYS": round(pol_dem_breakdown["tier2_days"], 2),
            "POL_DEM_TIER2_COST": round(pol_dem_breakdown["tier2_cost"], 2),
            "POL_DEM_THEREAFTER_DAYS": round(pol_dem_breakdown["thereafter_days"], 2),
            "POL_DEM_THEREAFTER_COST": round(pol_dem_breakdown["thereafter_cost"], 2),
            "POD_DEM_TIER1_DAYS": round(pod_dem_breakdown["tier1_days"], 2),
            "POD_DEM_TIER1_COST": round(pod_dem_breakdown["tier1_cost"], 2),
            "POD_DEM_TIER2_DAYS": round(pod_dem_breakdown["tier2_days"], 2),
            "POD_DEM_TIER2_COST": round(pod_dem_breakdown["tier2_cost"], 2),
            "POD_DEM_THEREAFTER_DAYS": round(pod_dem_breakdown["thereafter_days"], 2),
            "POD_DEM_THEREAFTER_COST": round(pod_dem_breakdown["thereafter_cost"], 2),
            "POD_DET_TIER1_DAYS": round(pod_det_breakdown["tier1_days"], 2),
            "POD_DET_TIER1_COST": round(pod_det_breakdown["tier1_cost"], 2),
            "POD_DET_TIER2_DAYS": round(pod_det_breakdown["tier2_days"], 2),
            "POD_DET_TIER2_COST": round(pod_det_breakdown["tier2_cost"], 2),
            "POD_DET_THEREAFTER_DAYS": round(pod_det_breakdown["thereafter_days"], 2),
            "POD_DET_THEREAFTER_COST": round(pod_det_breakdown["thereafter_cost"], 2),
        })
        matched_results.append(matched_record)

    matched_df = pd.DataFrame(matched_results)
    unmatched_df = pd.DataFrame(unmatched_results)
    unmatched_df = enrich_unmatched_risk(unmatched_df, matched_df)
    return matched_df, unmatched_df, original_count, cancelled_count


def enrich_unmatched_risk(unmatched_df, matched_df):
    if unmatched_df.empty:
        return unmatched_df

    def positive_mean(df, col):
        if df.empty or col not in df.columns:
            return np.nan
        s = pd.to_numeric(df[col], errors="coerce")
        s = s[s > 0]
        return s.mean() if len(s) else np.nan

    global_avg_pol_dem = positive_mean(matched_df, "POL_DEM_TOTAL_DAYS")
    global_avg_pod_dem = positive_mean(matched_df, "POD_DEM_TOTAL_DAYS")
    global_avg_pod_det = positive_mean(matched_df, "POD_DET_TOTAL_DAYS")
    if np.isnan(global_avg_pol_dem):
        global_avg_pol_dem = 3.0
    if np.isnan(global_avg_pod_dem):
        global_avg_pod_dem = 3.0
    if np.isnan(global_avg_pod_det):
        global_avg_pod_det = 5.0

    unmatched_df["AVG_POL_DEM_BENCHMARK"] = round(global_avg_pol_dem, 2)
    unmatched_df["AVG_POD_DEM_BENCHMARK"] = round(global_avg_pod_dem, 2)
    unmatched_df["AVG_POD_DET_BENCHMARK"] = round(global_avg_pod_det, 2)

    risk_flags, risk_reasons, risk_score = [], [], []
    for _, row in unmatched_df.iterrows():
        reasons = []
        score = 0
        pol_days = row.get("POL_DEM_TOTAL_DAYS")
        pod_dem_days = row.get("POD_DEM_TOTAL_DAYS")
        pod_det_days = row.get("POD_DET_TOTAL_DAYS")
        if pd.notna(pol_days) and pol_days > global_avg_pol_dem:
            reasons.append(f"POL demurrage {pol_days:.1f}d vs avg {global_avg_pol_dem:.1f}d")
            score += 1
        if pd.notna(pod_dem_days) and pod_dem_days > global_avg_pod_dem:
            reasons.append(f"POD demurrage {pod_dem_days:.1f}d vs avg {global_avg_pod_dem:.1f}d")
            score += 1
        if pd.notna(pod_det_days) and pod_det_days > global_avg_pod_det:
            reasons.append(f"POD detention {pod_det_days:.1f}d vs avg {global_avg_pod_det:.1f}d")
            score += 1
        risk_flags.append(score > 0)
        risk_score.append(score)
        risk_reasons.append("; ".join(reasons) if reasons else "No above-average dwell risk detected")
    unmatched_df["RISK_FLAG"] = risk_flags
    unmatched_df["RISK_SCORE"] = risk_score
    unmatched_df["RISK_REASONS"] = risk_reasons
    return unmatched_df

# -----------------------------------------------------------------------------
# DOWNLOAD HELPERS
# -----------------------------------------------------------------------------
def format_datetime_cols(dl, cols):
    for col in cols:
        if col in dl.columns:
            dl[col] = pd.to_datetime(dl[col], errors="coerce").dt.strftime("%Y-%m-%d %H:%M").fillna("")
    return dl


def build_download_df(data):
    dl = data.copy()
    dl = format_datetime_cols(dl, ["CGI", "CLL", "CDD", "CGO", "CER", "DET_END_TS"])
    rename_map = {
        "SHIPMENT_ID": "Shipment ID",
        "CONTAINER_NUMBER": "Container",
        "CONTAINER_TYPE": "Container Type",
        "IKEA_CONTAINER_TYPE": "IKEA Contract Container Type",
        "CONTAINER_MAPPING_STATUS": "Container Mapping Status",
        "CARRIER_SCAC": "Carrier SCAC",
        "CARRIER_NAME": "Carrier Name",
        "FFW_SCAC": "Freight Forwarder SCAC",
        "CARRIER_FFW_SCAC": "Carrier / FFW SCAC",
        "MATCHED_PARTY_TYPE": "Matched Party Type",
        "POL_LOCODE": "Port of Loading",
        "POL_COUNTRY": "POL Country",
        "POD_LOCODE": "Port of Discharge",
        "POD_COUNTRY": "Destination Country",
        "LANE": "Lane",
        "SUBSCRIPTION_STATUS": "Subscription Status",
        "CGI": "Gate In at POL (CGI)",
        "CLL": "Loaded on Vessel (CLL)",
        "CDD": "Discharge at POD (CDD)",
        "CGO": "Gate Out Full at POD (CGO)",
        "CER": "Empty Return (CER)",
        "POL_DEM_TOTAL_DAYS": "POL Demurrage Total Days",
        "POL_DEM_CHARGEABLE_DAYS": "POL Demurrage Chargeable Days",
        "POL_DEM_COST": "POL Demurrage Cost (USD)",
        "POD_DEM_TOTAL_DAYS": "POD Demurrage Total Days",
        "POD_DEM_CHARGEABLE_DAYS": "POD Demurrage Chargeable Days",
        "POD_DEM_COST": "POD Demurrage Cost (USD)",
        "POD_DET_TOTAL_DAYS": "POD Detention Total Days",
        "POD_DET_CHARGEABLE_DAYS": "POD Detention Chargeable Days",
        "POD_DET_COST": "POD Detention Cost (USD)",
        "DEM_COST": "Total Demurrage Cost (USD)",
        "DET_COST": "Total Detention Cost (USD)",
        "TOTAL_DD_COST": "Total D&D Cost (USD)",
        "CONTRACT_TYPE": "Free Days Type",
        "FREE_DEM_DAYS": "Free Demurrage Days",
        "FREE_DET_DAYS": "Free Detention Days",
        "COMBINED_FREE_DAYS": "Combined Free Days",
        "IKEA_DEM_RATE": "IKEA Demurrage Rate (USD/day)",
        "IKEA_DET_RATE": "IKEA Detention Rate (USD/day)",
        "MATCH_KEY": "Contract Match Key",
    }
    dl = dl.rename(columns={k: v for k, v in rename_map.items() if k in dl.columns})
    drop_cols = [c for c in ["POL", "POD", "LIFECYCLE_STATUS"] if c in dl.columns]
    dl = dl.drop(columns=drop_cols, errors="ignore")
    desired_order = [
        "Shipment ID", "Container", "Container Type", "IKEA Contract Container Type", "Container Mapping Status",
        "Carrier SCAC", "Carrier Name", "Freight Forwarder SCAC", "Carrier / FFW SCAC", "Matched Party Type",
        "Lane", "Port of Loading", "POL Country", "Port of Discharge", "Destination Country", "Subscription Status",
        "Gate In at POL (CGI)", "Loaded on Vessel (CLL)", "Discharge at POD (CDD)",
        "Gate Out Full at POD (CGO)", "Empty Return (CER)", "Free Days Type",
        "Free Demurrage Days", "Free Detention Days", "Combined Free Days",
        "IKEA Demurrage Rate (USD/day)", "IKEA Detention Rate (USD/day)",
        "POL Demurrage Total Days", "POL Demurrage Chargeable Days", "POL Demurrage Cost (USD)",
        "POD Demurrage Total Days", "POD Demurrage Chargeable Days", "POD Demurrage Cost (USD)",
        "POD Detention Total Days", "POD Detention Chargeable Days", "POD Detention Cost (USD)",
        "Total Demurrage Cost (USD)", "Total Detention Cost (USD)", "Total D&D Cost (USD)",
        "Contract Match Key",
    ]
    existing = [c for c in desired_order if c in dl.columns]
    remaining = [c for c in dl.columns if c not in existing]
    return dl[existing + remaining]


def build_unmatched_download_df(data):
    if data.empty:
        return data.copy()
    dl = data.copy()
    dl = format_datetime_cols(dl, ["CGI", "CLL", "CDD", "CGO", "CER", "DET_END_TS"])
    rename_map = {
        "SHIPMENT_ID": "Shipment ID",
        "CONTAINER_NUMBER": "Container",
        "CONTAINER_TYPE": "Container Type",
        "IKEA_CONTAINER_TYPE": "IKEA Contract Container Type",
        "CONTAINER_MAPPING_STATUS": "Container Mapping Status",
        "CARRIER_SCAC": "Carrier SCAC",
        "CARRIER_NAME": "Carrier Name",
        "FFW_SCAC": "Freight Forwarder SCAC",
        "CARRIER_FFW_SCAC": "Carrier / FFW SCAC",
        "MATCHED_PARTY_TYPE": "Matched Party Type",
        "POL_LOCODE": "Port of Loading",
        "POL_COUNTRY": "POL Country",
        "POD_LOCODE": "Port of Discharge",
        "POD_COUNTRY": "Destination Country",
        "LANE": "Lane",
        "SUBSCRIPTION_STATUS": "Subscription Status",
        "CGI": "Gate In at POL (CGI)",
        "CLL": "Loaded on Vessel (CLL)",
        "CDD": "Discharge at POD (CDD)",
        "CGO": "Gate Out Full at POD (CGO)",
        "CER": "Empty Return (CER)",
        "POL_DEM_TOTAL_DAYS": "POL Demurrage Days",
        "POD_DEM_TOTAL_DAYS": "POD Demurrage Days",
        "POD_DET_TOTAL_DAYS": "POD Detention Days",
        "MATCH_KEY": "Missing Contract Key",
        "MISSING_CONTRACT_REASON": "Missing Contract Reason",
        "RISK_FLAG": "Risk Flag",
        "RISK_SCORE": "Risk Score",
        "RISK_REASONS": "Risk Reasons",
        "DATA_LIMITATION": "Data Limitation",
    }
    dl = dl.rename(columns={k: v for k, v in rename_map.items() if k in dl.columns})
    drop_cols = [c for c in ["POL", "POD", "LIFECYCLE_STATUS"] if c in dl.columns]
    dl = dl.drop(columns=drop_cols, errors="ignore")
    return dl

# -----------------------------------------------------------------------------
# SIDEBAR INPUTS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🚢 Demurrage & Detention Analyzer")
    st.markdown("---")

    rate_source = "Upload IKEA Tariff CSV"

    uploaded_ikea_file = st.file_uploader(
        "Upload IKEA Tariff CSV",
        type=["csv"],
        help="Upload IKEA tariff CSV. Matching is shipment DESTINATION_COUNTRY + mapped CONTAINER_TYPE.",
        key="ikea_contract_uploader",
    )
    st.caption(
        'Mapping: "20 ft Standard Container" → 22GP; '
        '"40 ft Standard Container" → 42GP; '
        '"40 ft High Cube Container" → 42GP. Everything else is a contract gap. '
        'Country matching uses shipment DESTINATION_COUNTRY and tariff Port of discharge country after whitespace cleanup.'
    )

    uploaded_file = st.file_uploader(
        "Upload Shipment CSV",
        type=["csv"],
        help="Upload the ocean shipment export CSV with milestone events.",
        key="shipment_uploader",
    )
    st.markdown("---")

ikea_lookup = None
ikea_records = None
ikea_df = None

if uploaded_ikea_file is not None:
    try:
        ikea_records, ikea_df, ikea_lookup = parse_ikea_tariff_csv(uploaded_ikea_file)
        with st.sidebar:
            st.success(f"✅ Loaded {len(ikea_lookup)} IKEA country/container tariffs")
            countries = sorted(set(r["POD_COUNTRY"] for r in ikea_records))
            containers = sorted(set(r["CONTAINER_TYPE_CONTRACT"] for r in ikea_records))
            st.markdown(f"**Countries:** {len(countries):,}")
            st.markdown(f"**Contract container types:** {', '.join(containers) if containers else '—'}")
            st.markdown("**Pricing:** flat POD demurrage + flat POD detention")
    except Exception as e:
        st.sidebar.error(f"❌ Error parsing IKEA tariff CSV: {e}")
        ikea_lookup = None

# -----------------------------------------------------------------------------
# LANDING PAGE
# -----------------------------------------------------------------------------
missing_ikea_contract = uploaded_ikea_file is None
missing_shipments = uploaded_file is None
if missing_ikea_contract or missing_shipments:
    st.markdown("## 🚢 Demurrage & Detention Analyzer")
    st.markdown("---")
    if missing_ikea_contract and missing_shipments:
        st.info("Upload both the **IKEA Tariff CSV** and a **Shipment CSV** from the sidebar to get started.")
    elif missing_ikea_contract:
        st.info("Upload the **IKEA Tariff CSV** from the sidebar.")
    elif missing_shipments:
        st.info("Upload a **Shipment CSV** from the sidebar to continue.")

    st.markdown("**IKEA tariff behavior:**")
    st.code(
        "Contract Port of discharge country -> matched to shipment DESTINATION_COUNTRY\n"
        "Shipment CONTAINER_TYPE '20 ft Standard Container' -> contract 22GP\n"
        "Shipment CONTAINER_TYPE '40 ft Standard Container' -> contract 42GP\n"
        "Shipment CONTAINER_TYPE '40 ft High Cube Container' -> contract 42GP\n"
        "Demurrage Free Time / Tariff -> POD demurrage flat pricing\n"
        "Detention Freetime / Tariff -> POD detention flat pricing\n"
        "Port of discharge state -> ignored\n"
        "POL demurrage -> not priced from IKEA tariff",
        language=None,
    )
    st.markdown("**Expected shipment milestone columns:**")
    st.code(
        "CEP → CGI → CLL → VDL → VAD → CDD → CGO → CER\n\n"
        "POL Demurrage = CGI → CLL\n"
        "POD Demurrage = CDD → CGO\n"
        "POD Detention = CGO → CER",
        language=None,
    )
    st.stop()

if ikea_lookup is None:
    st.error("IKEA tariff CSV could not be parsed. Please check the format and re-upload.")
    st.stop()

# -----------------------------------------------------------------------------
# LOAD AND PROCESS
# -----------------------------------------------------------------------------
with st.spinner("Processing shipments against IKEA tariffs..."):
    raw_df = pd.read_csv(uploaded_file)
    rdf, unmatched_df, total_shipments, cancelled_count = process_shipments(
        raw_df,
        contracts_list=None,
        estimate_profile=None,
        use_estimate=False,
        ikea_lookup=ikea_lookup,
        rate_mode="IKEA Tariff CSV",
    )

if rdf.empty and unmatched_df.empty:
    st.error("No usable shipments found after excluding cancelled shipments.")
    st.stop()

# -----------------------------------------------------------------------------
# SIDEBAR FILTERS
# -----------------------------------------------------------------------------
for _df in [rdf, unmatched_df]:
    if not _df.empty:
        if "DD_ANCHOR_DATE" not in _df.columns:
            _df["DD_ANCHOR_DATE"] = pd.NaT
        _df["DD_ANCHOR_DATE"] = pd.to_datetime(_df["DD_ANCHOR_DATE"], errors="coerce", utc=True)

with st.sidebar:
    st.markdown("---")
    st.markdown("### Filters")
    combined_for_filters = pd.concat([
        rdf[[c for c in ["CARRIER_SCAC", "FFW_SCAC", "CARRIER_FFW_SCAC", "POD_LOCODE", "POD_COUNTRY", "POL_LOCODE", "CONTAINER_TYPE", "IKEA_CONTAINER_TYPE", "DD_ANCHOR_DATE"] if c in rdf.columns]] if not rdf.empty else pd.DataFrame(),
        unmatched_df[[c for c in ["CARRIER_SCAC", "FFW_SCAC", "CARRIER_FFW_SCAC", "POD_LOCODE", "POD_COUNTRY", "POL_LOCODE", "CONTAINER_TYPE", "IKEA_CONTAINER_TYPE", "DD_ANCHOR_DATE"] if c in unmatched_df.columns]] if not unmatched_df.empty else pd.DataFrame(),
    ], ignore_index=True)

    carriers = sorted(combined_for_filters.get("CARRIER_FFW_SCAC", pd.Series(dtype=str)).dropna().astype(str).unique())
    pods = sorted(combined_for_filters.get("POD_LOCODE", pd.Series(dtype=str)).dropna().astype(str).unique())
    pols = sorted(combined_for_filters.get("POL_LOCODE", pd.Series(dtype=str)).dropna().astype(str).unique())
    countries = sorted(combined_for_filters.get("POD_COUNTRY", pd.Series(dtype=str)).dropna().astype(str).unique())
    container_types = sorted(combined_for_filters.get("CONTAINER_TYPE", pd.Series(dtype=str)).dropna().astype(str).unique())

    sel_carriers = st.multiselect("Carrier / FFW", carriers, default=carriers)
    sel_pods = st.multiselect("POD Terminal", pods, default=pods)
    if rate_source == "Upload IKEA Tariff CSV":
        sel_countries = st.multiselect("Destination Country", countries, default=countries)
        sel_container_types = st.multiselect("Container Type", container_types, default=container_types)
    else:
        sel_countries = countries
        sel_container_types = container_types
    sel_pols = st.multiselect("POL", pols, default=pols)
    show_zero = st.checkbox("Include $0 charge shipments", value=True)

    st.markdown("### Time Filters")
    trend_grain = st.radio("Trend View", ["Weekly", "Monthly"], horizontal=True)
    valid_dates = combined_for_filters["DD_ANCHOR_DATE"].dropna() if "DD_ANCHOR_DATE" in combined_for_filters.columns else pd.Series(dtype="datetime64[ns, UTC]")
    if not valid_dates.empty:
        min_date = valid_dates.min().date()
        max_date = valid_dates.max().date()
        date_range = st.date_input("D&D Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    else:
        date_range = None
        st.caption("No valid D&D anchor dates found for date filtering.")


def apply_common_filters(data, require_cost_filter=False):
    if data.empty:
        return data.copy()
    out = data.copy()
    if sel_carriers and "CARRIER_FFW_SCAC" in out.columns:
        out = out[out["CARRIER_FFW_SCAC"].astype(str).isin(sel_carriers)]
    if sel_pods and "POD_LOCODE" in out.columns:
        out = out[out["POD_LOCODE"].astype(str).isin(sel_pods)]
    if sel_countries and "POD_COUNTRY" in out.columns:
        out = out[out["POD_COUNTRY"].astype(str).isin(sel_countries)]
    if sel_container_types and "CONTAINER_TYPE" in out.columns:
        out = out[out["CONTAINER_TYPE"].astype(str).isin(sel_container_types)]
    if sel_pols and "POL_LOCODE" in out.columns:
        out = out[out["POL_LOCODE"].astype(str).isin(sel_pols)]
    if date_range and len(date_range) == 2 and "DD_ANCHOR_DATE" in out.columns:
        start_date, end_date = date_range
        anchor = pd.to_datetime(out["DD_ANCHOR_DATE"], errors="coerce", utc=True)
        out = out[(anchor.dt.date >= start_date) & (anchor.dt.date <= end_date)]
    if require_cost_filter and not show_zero and "TOTAL_DD_COST" in out.columns:
        out = out[out["TOTAL_DD_COST"] > 0]
    return out


fdf = apply_common_filters(rdf, require_cost_filter=True) if not rdf.empty else rdf.copy()
ufdf = apply_common_filters(unmatched_df, require_cost_filter=False) if not unmatched_df.empty else unmatched_df.copy()


def fill_grouping_blanks(data):
    if data.empty:
        return data
    out = data.copy()
    for col in [
        "CARRIER_FFW_SCAC", "MATCHED_PARTY_TYPE", "CARRIER_SCAC", "FFW_SCAC", "CARRIER_NAME",
        "POD_LOCODE", "POD_COUNTRY", "POL_LOCODE", "CONTAINER_TYPE", "IKEA_CONTAINER_TYPE", "MATCH_KEY",
    ]:
        if col in out.columns:
            out[col] = out[col].fillna("").astype(str)
    return out


fdf = fill_grouping_blanks(fdf)
ufdf = fill_grouping_blanks(ufdf)
unmatched_df_display = fill_grouping_blanks(unmatched_df)

# -----------------------------------------------------------------------------
# TABS
# -----------------------------------------------------------------------------
tab_overview, tab_trends, tab_carrier, tab_port, tab_ships, tab_gaps, tab_tiers, tab_download = st.tabs([
    "📊 Overview", "📈 Trends", "🚛 Carrier / FFW", "🏗️ Ports & Lanes", "📦 Shipments", "⚠️ Contract Gaps", "🔥 Tier Exposure", "📥 Download",
])

# -----------------------------------------------------------------------------
# OVERVIEW
# -----------------------------------------------------------------------------
with tab_overview:
    st.markdown("### Executive Summary")
    if rate_source == "Upload IKEA Tariff CSV":
        st.caption("IKEA mode: matching is destination country + strict mapped container type. POL demurrage is shown as days but not priced from the IKEA tariff.")
    if fdf.empty:
        st.warning("No matched shipments available for the selected filters.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total D&D Cost", f"${fdf['TOTAL_DD_COST'].sum():,.0f}", f"{len(fdf)} matched of {total_shipments:,}")
        c2.metric("POL Demurrage", f"${fdf['POL_DEM_COST'].sum():,.0f}", f"{(fdf['POL_DEM_COST'] > 0).sum()} shipments")
        c3.metric("POD Demurrage", f"${fdf['POD_DEM_COST'].sum():,.0f}", f"{(fdf['POD_DEM_COST'] > 0).sum()} shipments")
        c4.metric("POD Detention", f"${fdf['POD_DET_COST'].sum():,.0f}", f"{(fdf['POD_DET_COST'] > 0).sum()} shipments")

        c1, c2, c3 = st.columns(3)
        avg_pol_dem = fdf.loc[fdf["POL_DEM_COST"] > 0, "POL_DEM_CHARGEABLE_DAYS"].mean()
        avg_pod_dem = fdf.loc[fdf["POD_DEM_COST"] > 0, "POD_DEM_CHARGEABLE_DAYS"].mean()
        avg_det = fdf.loc[fdf["POD_DET_COST"] > 0, "POD_DET_CHARGEABLE_DAYS"].mean()
        c1.metric("Avg POL Dem Days", f"{avg_pol_dem:.1f}d" if not np.isnan(avg_pol_dem) else "—")
        c2.metric("Avg POD Dem Days", f"{avg_pod_dem:.1f}d" if not np.isnan(avg_pod_dem) else "—")
        c3.metric("Avg POD Det Days", f"{avg_det:.1f}d" if not np.isnan(avg_det) else "—")
        if cancelled_count > 0:
            st.caption(f"ℹ️ {cancelled_count} cancelled shipments excluded from analysis.")

        st.markdown("---")
        carrier_agg = fdf.groupby("CARRIER_FFW_SCAC").agg(
            POL_Demurrage=("POL_DEM_COST", "sum"), POD_Demurrage=("POD_DEM_COST", "sum"), POD_Detention=("POD_DET_COST", "sum")
        ).reset_index()
        carrier_melt = carrier_agg.melt(id_vars="CARRIER_FFW_SCAC", var_name="Type", value_name="Cost")
        carrier_melt["Type"] = carrier_melt["Type"].replace({"POL_Demurrage": "POL Demurrage", "POD_Demurrage": "POD Demurrage", "POD_Detention": "POD Detention"})
        chart_carrier = alt.Chart(carrier_melt).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            y=alt.Y("CARRIER_FFW_SCAC:N", sort="-x", title="Carrier / FFW"),
            x=alt.X("Cost:Q", title="Cost (USD)"),
            color=alt.Color("Type:N", scale=alt.Scale(domain=["POL Demurrage", "POD Demurrage", "POD Detention"], range=[POL_DEM_COLOR, DEM_COLOR, DET_COLOR])),
            tooltip=["CARRIER_FFW_SCAC", "Type", alt.Tooltip("Cost:Q", format="$,.0f")],
        ).properties(title="D&D Cost by Carrier / FFW", height=280)
        st.altair_chart(chart_carrier, use_container_width=True)

        pod_group = "POD_COUNTRY" if rate_source == "Upload IKEA Tariff CSV" else "POD_LOCODE"
        pod_agg = fdf.groupby(pod_group).agg(
            POL_Demurrage=("POL_DEM_COST", "sum"), POD_Demurrage=("POD_DEM_COST", "sum"), POD_Detention=("POD_DET_COST", "sum")
        ).reset_index()
        pod_melt = pod_agg.melt(id_vars=pod_group, var_name="Type", value_name="Cost")
        pod_melt["Type"] = pod_melt["Type"].replace({"POL_Demurrage": "POL Demurrage", "POD_Demurrage": "POD Demurrage", "POD_Detention": "POD Detention"})
        chart_pod = alt.Chart(pod_melt).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            y=alt.Y(f"{pod_group}:N", sort="-x", title="Destination Country" if pod_group == "POD_COUNTRY" else "POD Terminal"),
            x=alt.X("Cost:Q", title="Cost (USD)"),
            color=alt.Color("Type:N", scale=alt.Scale(domain=["POL Demurrage", "POD Demurrage", "POD Detention"], range=[POL_DEM_COLOR, DEM_COLOR, DET_COLOR])),
            tooltip=[pod_group, "Type", alt.Tooltip("Cost:Q", format="$,.0f")],
        ).properties(title="D&D Cost by Destination Country" if pod_group == "POD_COUNTRY" else "D&D Cost by POD Terminal", height=250)
        st.altair_chart(chart_pod, use_container_width=True)

# -----------------------------------------------------------------------------
# TRENDS
# -----------------------------------------------------------------------------
with tab_trends:
    st.markdown("### D&D Trends")
    st.caption("Trend date uses CDD when available, then CGO, then CER, then CLL/CGI as fallback.")
    if fdf.empty:
        st.warning("No matched/priced shipments available for the selected filters.")
    else:
        trend_df = fdf.copy()
        trend_df["DD_ANCHOR_DATE"] = pd.to_datetime(trend_df["DD_ANCHOR_DATE"], errors="coerce", utc=True)
        trend_df = trend_df.dropna(subset=["DD_ANCHOR_DATE"])
        if trend_df.empty:
            st.warning("No valid D&D anchor dates found for the selected filters.")
        else:
            if trend_grain == "Weekly":
                trend_df["PERIOD"] = trend_df["DD_ANCHOR_DATE"].dt.to_period("W").apply(lambda r: r.start_time)
                period_title = "Week"
            else:
                trend_df["PERIOD"] = trend_df["DD_ANCHOR_DATE"].dt.to_period("M").apply(lambda r: r.start_time)
                period_title = "Month"
            trend_agg = trend_df.groupby("PERIOD").agg(
                Shipments=("SHIPMENT_ID", "count"),
                POL_Demurrage=("POL_DEM_COST", "sum"), POD_Demurrage=("POD_DEM_COST", "sum"), Detention=("POD_DET_COST", "sum"),
                Total=("TOTAL_DD_COST", "sum"),
                Avg_POL_Dem_Days=("POL_DEM_CHARGEABLE_DAYS", "mean"), Avg_POD_Dem_Days=("POD_DEM_CHARGEABLE_DAYS", "mean"), Avg_Det_Days=("POD_DET_CHARGEABLE_DAYS", "mean"),
            ).reset_index().sort_values("PERIOD")
            c1, c2, c3 = st.columns(3)
            c1.metric("Periods", f"{len(trend_agg):,}")
            c2.metric("Total Cost", f"${trend_agg['Total'].sum():,.0f}")
            c3.metric("Avg Cost / Shipment", f"${(trend_agg['Total'].sum() / max(trend_agg['Shipments'].sum(), 1)):,.0f}")
            cost_melt = trend_agg.melt(id_vars=["PERIOD"], value_vars=["POL_Demurrage", "POD_Demurrage", "Detention"], var_name="Charge Type", value_name="Cost")
            cost_melt["Charge Type"] = cost_melt["Charge Type"].replace({"POL_Demurrage": "POL Demurrage", "POD_Demurrage": "POD Demurrage"})
            cost_chart = alt.Chart(cost_melt).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                x=alt.X("PERIOD:T", title=period_title), y=alt.Y("Cost:Q", title="Cost"),
                color=alt.Color("Charge Type:N", scale=alt.Scale(domain=["POL Demurrage", "POD Demurrage", "Detention"], range=[POL_DEM_COLOR, DEM_COLOR, DET_COLOR])),
                tooltip=[alt.Tooltip("PERIOD:T", title=period_title), "Charge Type:N", alt.Tooltip("Cost:Q", format="$,.0f")],
            ).properties(height=350)
            st.altair_chart(cost_chart, use_container_width=True)
            st.markdown("#### Trend Summary")
            st.dataframe(trend_agg.style.format({"POL_Demurrage": "${:,.0f}", "POD_Demurrage": "${:,.0f}", "Detention": "${:,.0f}", "Total": "${:,.0f}", "Avg_POL_Dem_Days": "{:.1f}", "Avg_POD_Dem_Days": "{:.1f}", "Avg_Det_Days": "{:.1f}"}), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### Contract Gap Trend")
    if ufdf.empty:
        st.info("No unmatched/contract-gap shipments for the selected filters.")
    else:
        gap_trend = ufdf.copy()
        gap_trend["DD_ANCHOR_DATE"] = pd.to_datetime(gap_trend["DD_ANCHOR_DATE"], errors="coerce", utc=True)
        gap_trend = gap_trend.dropna(subset=["DD_ANCHOR_DATE"])
        if gap_trend.empty:
            st.info("Contract-gap shipments do not have valid anchor dates for trend analysis.")
        else:
            if trend_grain == "Weekly":
                gap_trend["PERIOD"] = gap_trend["DD_ANCHOR_DATE"].dt.to_period("W").apply(lambda r: r.start_time)
            else:
                gap_trend["PERIOD"] = gap_trend["DD_ANCHOR_DATE"].dt.to_period("M").apply(lambda r: r.start_time)
            gap_agg = gap_trend.groupby("PERIOD").agg(
                Unmatched_Shipments=("SHIPMENT_ID", "count"), Risk_Shipments=("RISK_FLAG", "sum"), Missing_Contract_Keys=("MATCH_KEY", "nunique"),
                Avg_POL_Dem_Days=("POL_DEM_TOTAL_DAYS", "mean"), Avg_POD_Dem_Days=("POD_DEM_TOTAL_DAYS", "mean"), Avg_POD_Det_Days=("POD_DET_TOTAL_DAYS", "mean"),
            ).reset_index().sort_values("PERIOD")
            st.dataframe(gap_agg.style.format({"Avg_POL_Dem_Days": "{:.1f}", "Avg_POD_Dem_Days": "{:.1f}", "Avg_POD_Det_Days": "{:.1f}"}), use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# CARRIERS
# -----------------------------------------------------------------------------
with tab_carrier:
    st.markdown("### Carrier / FFW Summary")
    if fdf.empty:
        st.warning("No matched shipments available for the selected filters.")
    else:
        carrier_detail = fdf.groupby(["CARRIER_FFW_SCAC", "MATCHED_PARTY_TYPE", "CARRIER_SCAC", "FFW_SCAC", "CARRIER_NAME"], dropna=False).agg(
            Ships=("SHIPMENT_ID", "count"),
            POL_Dem_Ships=("POL_DEM_COST", lambda x: (x > 0).sum()), POD_Dem_Ships=("POD_DEM_COST", lambda x: (x > 0).sum()), Det_Ships=("POD_DET_COST", lambda x: (x > 0).sum()),
            POL_Dem_Cost=("POL_DEM_COST", "sum"), POD_Dem_Cost=("POD_DEM_COST", "sum"), Det_Cost=("POD_DET_COST", "sum"),
            Avg_POL_Dem_Days=("POL_DEM_CHARGEABLE_DAYS", lambda x: x[x > 0].mean() if (x > 0).any() else 0),
            Avg_POD_Dem_Days=("POD_DEM_CHARGEABLE_DAYS", lambda x: x[x > 0].mean() if (x > 0).any() else 0),
            Avg_Det_Days=("POD_DET_CHARGEABLE_DAYS", lambda x: x[x > 0].mean() if (x > 0).any() else 0),
        ).reset_index()
        carrier_detail["Total_Cost"] = carrier_detail["POL_Dem_Cost"] + carrier_detail["POD_Dem_Cost"] + carrier_detail["Det_Cost"]
        carrier_detail = carrier_detail.sort_values("Total_Cost", ascending=False)
        st.dataframe(carrier_detail.style.format({"POL_Dem_Cost": "${:,.0f}", "POD_Dem_Cost": "${:,.0f}", "Det_Cost": "${:,.0f}", "Total_Cost": "${:,.0f}", "Avg_POL_Dem_Days": "{:.1f}", "Avg_POD_Dem_Days": "{:.1f}", "Avg_Det_Days": "{:.1f}"}), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("#### Carrier / FFW × POD Breakdown")
        pod_col = "POD_COUNTRY" if rate_source == "Upload IKEA Tariff CSV" else "POD_LOCODE"
        cp = fdf.groupby(["CARRIER_FFW_SCAC", "MATCHED_PARTY_TYPE", pod_col], dropna=False).agg(
            Ships=("SHIPMENT_ID", "count"), POL_Dem=("POL_DEM_COST", "sum"), POD_Dem=("POD_DEM_COST", "sum"), Det=("POD_DET_COST", "sum")
        ).reset_index()
        cp["Total"] = cp["POL_Dem"] + cp["POD_Dem"] + cp["Det"]
        cp = cp[cp["Total"] > 0].sort_values("Total", ascending=False)
        st.dataframe(cp.style.format({"POL_Dem": "${:,.0f}", "POD_Dem": "${:,.0f}", "Det": "${:,.0f}", "Total": "${:,.0f}"}), use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# PORTS & LANES
# -----------------------------------------------------------------------------
with tab_port:
    st.markdown("### Ports & Lanes")
    if fdf.empty:
        st.warning("No matched shipments available for the selected filters.")
    else:
        group_col = "POD_COUNTRY" if rate_source == "Upload IKEA Tariff CSV" else "POD_LOCODE"
        st.markdown("#### Destination Country Summary" if group_col == "POD_COUNTRY" else "#### POD Terminal Summary")
        pod_sum = fdf.groupby(group_col).agg(
            Ships=("SHIPMENT_ID", "count"), POL_Dem=("POL_DEM_COST", "sum"), POD_Dem=("POD_DEM_COST", "sum"), Det=("POD_DET_COST", "sum")
        ).reset_index()
        pod_sum["Total"] = pod_sum["POL_Dem"] + pod_sum["POD_Dem"] + pod_sum["Det"]
        pod_sum = pod_sum.sort_values("Total", ascending=False)
        st.dataframe(pod_sum.style.format({"POL_Dem": "${:,.0f}", "POD_Dem": "${:,.0f}", "Det": "${:,.0f}", "Total": "${:,.0f}"}), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("#### Top 20 Lanes by Total D&D Cost")
        lane_agg = fdf.groupby("LANE").agg(
            Ships=("SHIPMENT_ID", "count"),
            Carrier_FFWs=("CARRIER_FFW_SCAC", lambda x: ", ".join(sorted(x.dropna().astype(str).unique()))),
            POL_Dem=("POL_DEM_COST", "sum"), POD_Dem=("POD_DEM_COST", "sum"), Det=("POD_DET_COST", "sum"),
            Avg_POL_Dem_Days=("POL_DEM_CHARGEABLE_DAYS", lambda x: round(x[x > 0].mean(), 1) if (x > 0).any() else 0),
            Avg_POD_Dem_Days=("POD_DEM_CHARGEABLE_DAYS", lambda x: round(x[x > 0].mean(), 1) if (x > 0).any() else 0),
            Avg_Det_Days=("POD_DET_CHARGEABLE_DAYS", lambda x: round(x[x > 0].mean(), 1) if (x > 0).any() else 0),
        ).reset_index()
        lane_agg["Total"] = lane_agg["POL_Dem"] + lane_agg["POD_Dem"] + lane_agg["Det"]
        lane_agg = lane_agg.sort_values("Total", ascending=False).head(20)
        st.dataframe(lane_agg.style.format({"POL_Dem": "${:,.0f}", "POD_Dem": "${:,.0f}", "Det": "${:,.0f}", "Total": "${:,.0f}"}), use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# SHIPMENT EXPLORER
# -----------------------------------------------------------------------------
with tab_ships:
    st.markdown("### Shipment-Level D&D Detail")
    if fdf.empty:
        st.warning("No matched shipments available for the selected filters.")
    else:
        st.caption(f"Showing {len(fdf)} matched shipments. Use sidebar filters to narrow.")
        sort_options = ["TOTAL_DD_COST", "POL_DEM_COST", "POD_DEM_COST", "POD_DET_COST", "POL_DEM_CHARGEABLE_DAYS", "POD_DEM_CHARGEABLE_DAYS", "POD_DET_CHARGEABLE_DAYS"]
        sort_col = st.selectbox("Sort by", sort_options)
        top_n = st.slider("Show top N", 10, min(500, max(len(fdf), 10)), min(50, max(len(fdf), 10)))
        display_cols = [
            "CONTAINER_NUMBER", "SHIPMENT_ID", "CONTAINER_TYPE", "IKEA_CONTAINER_TYPE", "CONTAINER_MAPPING_STATUS",
            "CARRIER_SCAC", "FFW_SCAC", "CARRIER_FFW_SCAC", "MATCHED_PARTY_TYPE", "LANE", "POD_COUNTRY",
            "CGI", "CLL", "CDD", "CGO", "CER",
            "POL_DEM_TOTAL_DAYS", "POL_DEM_CHARGEABLE_DAYS", "POL_DEM_COST",
            "POD_DEM_TOTAL_DAYS", "POD_DEM_CHARGEABLE_DAYS", "POD_DEM_COST",
            "POD_DET_TOTAL_DAYS", "POD_DET_CHARGEABLE_DAYS", "POD_DET_COST", "TOTAL_DD_COST",
            "CONTRACT_TYPE",
        ]
        show_df = fdf[[c for c in display_cols if c in fdf.columns]].sort_values(sort_col, ascending=False).head(top_n).copy()
        for dc in ["CGI", "CLL", "CDD", "CGO", "CER"]:
            if dc in show_df.columns:
                show_df[dc] = pd.to_datetime(show_df[dc], errors="coerce").dt.strftime("%Y-%m-%d").fillna("—")
        st.dataframe(show_df.style.format({"POL_DEM_COST": "${:,.2f}", "POD_DEM_COST": "${:,.2f}", "POD_DET_COST": "${:,.2f}", "TOTAL_DD_COST": "${:,.2f}"}), use_container_width=True, hide_index=True, height=600)

# -----------------------------------------------------------------------------
# CONTRACT GAPS
# -----------------------------------------------------------------------------
with tab_gaps:
    st.markdown("### ⚠️ Contract Gaps")
    if rate_source == "Upload IKEA Tariff CSV":
        st.caption("For IKEA, contract gaps include missing destination country, unsupported container type, or no tariff for destination country + mapped container type.")
    else:
        st.caption("These shipments did not match a contract, so fees are not calculated.")
    if ufdf.empty:
        st.success("No unmatched shipments found for the selected filters. All visible non-cancelled shipments matched uploaded contracts.")
    else:
        gap_source = fill_grouping_blanks(ufdf)
        risk_df = gap_source[gap_source["RISK_FLAG"] == True].copy()
        c1, c2, c3 = st.columns(3)
        c1.metric("Unmatched Shipments", f"{len(gap_source):,}")
        c2.metric("Risk Containers", f"{len(risk_df):,}", "above avg dwell")
        c3.metric("Missing Contract Keys", f"{gap_source['MATCH_KEY'].nunique():,}")
        st.markdown("---")
        st.markdown("#### Missing Contract Combinations")
        group_cols = ["POD_COUNTRY", "CONTAINER_TYPE", "IKEA_CONTAINER_TYPE", "CONTAINER_MAPPING_STATUS", "MATCH_KEY"] if rate_source == "Upload IKEA Tariff CSV" else ["POD_LOCODE", "CARRIER_FFW_SCAC", "MATCHED_PARTY_TYPE", "CARRIER_SCAC", "FFW_SCAC", "POL_LOCODE", "MATCH_KEY"]
        combo = gap_source.groupby([c for c in group_cols if c in gap_source.columns], dropna=False).agg(
            Shipments=("SHIPMENT_ID", "count"), Containers=("CONTAINER_NUMBER", lambda x: x.nunique()), Risk_Containers=("RISK_FLAG", lambda x: int(x.sum())),
            Avg_POL_Dem_Days=("POL_DEM_TOTAL_DAYS", "mean"), Avg_POD_Dem_Days=("POD_DEM_TOTAL_DAYS", "mean"), Avg_POD_Det_Days=("POD_DET_TOTAL_DAYS", "mean"),
            Max_POD_Det_Days=("POD_DET_TOTAL_DAYS", "max"),
        ).reset_index().sort_values(["Risk_Containers", "Shipments"], ascending=False)
        st.dataframe(combo.style.format({"Avg_POL_Dem_Days": "{:.1f}", "Avg_POD_Dem_Days": "{:.1f}", "Avg_POD_Det_Days": "{:.1f}", "Max_POD_Det_Days": "{:.1f}"}), use_container_width=True, hide_index=True)
        st.markdown("---")
        st.markdown("#### Container-Level Contract Gap Risk")
        gap_cols = [
            "CONTAINER_NUMBER", "SHIPMENT_ID", "CONTAINER_TYPE", "IKEA_CONTAINER_TYPE", "CONTAINER_MAPPING_STATUS",
            "CARRIER_SCAC", "FFW_SCAC", "CARRIER_FFW_SCAC", "MATCHED_PARTY_TYPE", "LANE", "POD_COUNTRY",
            "CGI", "CLL", "CDD", "CGO", "CER",
            "POL_DEM_TOTAL_DAYS", "POD_DEM_TOTAL_DAYS", "POD_DET_TOTAL_DAYS",
            "RISK_SCORE", "RISK_REASONS", "MATCH_KEY", "MISSING_CONTRACT_REASON", "DATA_LIMITATION",
        ]
        gap_show = gap_source[[c for c in gap_cols if c in gap_source.columns]].copy()
        sort_cols = [c for c in ["RISK_SCORE", "POD_DET_TOTAL_DAYS", "POD_DEM_TOTAL_DAYS", "POL_DEM_TOTAL_DAYS"] if c in gap_show.columns]
        if sort_cols:
            gap_show = gap_show.sort_values(sort_cols, ascending=False)
        for dc in ["CGI", "CLL", "CDD", "CGO", "CER"]:
            if dc in gap_show.columns:
                gap_show[dc] = pd.to_datetime(gap_show[dc], errors="coerce").dt.strftime("%Y-%m-%d").fillna("—")
        st.dataframe(gap_show, use_container_width=True, hide_index=True, height=520)

# -----------------------------------------------------------------------------
# TIER EXPOSURE
# -----------------------------------------------------------------------------
with tab_tiers:
    st.markdown("### 🔥 Tier Exposure")
    if rate_source == "Upload IKEA Tariff CSV":
        st.info("IKEA tariffs are flat-rate after free days. Tier 1 below represents all flat-rate chargeable days; Tier 2 and Thereafter remain zero.")
    if fdf.empty:
        st.warning("No matched shipments available for the selected filters.")
    else:
        tier_cost_cols = [
            "POL_DEM_TIER1_COST", "POL_DEM_TIER2_COST", "POL_DEM_THEREAFTER_COST",
            "POD_DEM_TIER1_COST", "POD_DEM_TIER2_COST", "POD_DEM_THEREAFTER_COST",
            "POD_DET_TIER1_COST", "POD_DET_TIER2_COST", "POD_DET_THEREAFTER_COST",
        ]
        for col in tier_cost_cols:
            if col not in fdf.columns:
                fdf[col] = 0.0
        total_tier_cost = fdf[tier_cost_cols].sum().sum()
        thereafter_cost = fdf[["POL_DEM_THEREAFTER_COST", "POD_DEM_THEREAFTER_COST", "POD_DET_THEREAFTER_COST"]].sum().sum()
        thereafter_containers = ((fdf["POL_DEM_THEREAFTER_COST"] > 0) | (fdf["POD_DEM_THEREAFTER_COST"] > 0) | (fdf["POD_DET_THEREAFTER_COST"] > 0)).sum()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Tiered Cost", f"${total_tier_cost:,.0f}")
        c2.metric("Thereafter Cost", f"${thereafter_cost:,.0f}")
        c3.metric("% in Thereafter", f"{(thereafter_cost / total_tier_cost * 100):.1f}%" if total_tier_cost else "0.0%")
        c4.metric("Containers Hitting Thereafter", f"{thereafter_containers:,}")
        tier_summary = pd.DataFrame([
            {"Charge Type": "POL Demurrage", "Tier 1 Cost": fdf["POL_DEM_TIER1_COST"].sum(), "Tier 2 Cost": fdf["POL_DEM_TIER2_COST"].sum(), "Thereafter Cost": fdf["POL_DEM_THEREAFTER_COST"].sum()},
            {"Charge Type": "POD Demurrage", "Tier 1 Cost": fdf["POD_DEM_TIER1_COST"].sum(), "Tier 2 Cost": fdf["POD_DEM_TIER2_COST"].sum(), "Thereafter Cost": fdf["POD_DEM_THEREAFTER_COST"].sum()},
            {"Charge Type": "POD Detention", "Tier 1 Cost": fdf["POD_DET_TIER1_COST"].sum(), "Tier 2 Cost": fdf["POD_DET_TIER2_COST"].sum(), "Thereafter Cost": fdf["POD_DET_THEREAFTER_COST"].sum()},
        ])
        st.dataframe(tier_summary.style.format({"Tier 1 Cost": "${:,.0f}", "Tier 2 Cost": "${:,.0f}", "Thereafter Cost": "${:,.0f}"}), use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# DOWNLOAD
# -----------------------------------------------------------------------------
with tab_download:
    st.markdown("### 📥 Download Results")
    st.markdown("Download matched priced shipments, unmatched contract-gap shipments, and the uploaded contracts/tariffs.")
    matched_dl = build_download_df(fdf) if not fdf.empty else pd.DataFrame()
    unmatched_dl = build_unmatched_download_df(unmatched_df) if not unmatched_df.empty else pd.DataFrame()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Matched / Priced Shipments")
        st.markdown(f"**Rows:** {len(matched_dl)} | **Columns:** {len(matched_dl.columns)}")
        if not matched_dl.empty:
            st.dataframe(matched_dl.head(10), use_container_width=True, hide_index=True)
            csv_bytes = matched_dl.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Matched Results CSV", data=csv_bytes, file_name=f"Demurrage_Detention_Matched_Results_{datetime.now().strftime('%Y-%m-%d')}.csv", mime="text/csv")
    with c2:
        st.markdown("#### Contract Gaps / Unmatched Shipments")
        st.markdown(f"**Rows:** {len(unmatched_dl)} | **Columns:** {len(unmatched_dl.columns)}")
        if not unmatched_dl.empty:
            st.dataframe(unmatched_dl.head(10), use_container_width=True, hide_index=True)
            csv_bytes = unmatched_dl.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Contract Gaps CSV", data=csv_bytes, file_name=f"Demurrage_Detention_Contract_Gaps_{datetime.now().strftime('%Y-%m-%d')}.csv", mime="text/csv")

    st.markdown("---")
    st.markdown("#### Excel Workbook")
    try:
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            if not matched_dl.empty:
                matched_dl.to_excel(writer, sheet_name="Matched Results", index=False)
            if not unmatched_dl.empty:
                unmatched_dl.to_excel(writer, sheet_name="Contract Gaps", index=False)
            summary_data = {
                "Metric": [
                    "Total Shipments in Upload", "Cancelled Excluded", "Matched / Priced Shipments", "Unmatched Shipments",
                    "Unmatched Risk Containers", "Total D&D Cost", "POL Demurrage", "POD Demurrage", "POD Detention",
                    "Rate Source", "Analysis Date",
                ],
                "Value": [
                    total_shipments, cancelled_count, len(rdf), len(unmatched_df),
                    int(unmatched_df["RISK_FLAG"].sum()) if not unmatched_df.empty else 0,
                    f"${rdf['TOTAL_DD_COST'].sum():,.2f}" if not rdf.empty else "$0.00",
                    f"${rdf['POL_DEM_COST'].sum():,.2f}" if not rdf.empty else "$0.00",
                    f"${rdf['POD_DEM_COST'].sum():,.2f}" if not rdf.empty else "$0.00",
                    f"${rdf['POD_DET_COST'].sum():,.2f}" if not rdf.empty else "$0.00",
                    rate_source,
                    datetime.now().strftime("%Y-%m-%d %H:%M"),
                ],
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary", index=False)
            if ikea_df is not None:
                ikea_df.to_excel(writer, sheet_name="IKEA Tariffs", index=False)
        st.download_button("📥 Download Excel Workbook", data=buffer.getvalue(), file_name=f"Demurrage_Detention_Analyzer_{datetime.now().strftime('%Y-%m-%d')}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except ImportError:
        st.info("Excel download requires openpyxl. Add openpyxl to requirements.txt, or use CSV downloads above.")
