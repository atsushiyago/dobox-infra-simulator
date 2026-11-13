"""国土数値情報 500mメッシュ将来推計人口データの読み込み・空間結合モジュール."""

from __future__ import annotations

import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.strtree import STRtree

from data_loader import MunicipalityBounds

DEFAULT_POP_SHP = "500m_mesh_2018_34.shp"

_REF_LAT = 34.4
_M_PER_LAT = 111320.0
_M_PER_LON = 111320.0 * math.cos(math.radians(_REF_LAT))


def _detect_pop_column(columns: list[str], year_str: str) -> str:
    """PTN_2020 / PTN_2040 または年号を含む総人口カラムを自動検出する."""
    preferred = f"PTN_{year_str}"
    if preferred in columns:
        return preferred
    candidates = [c for c in columns if year_str in c and c.upper().startswith("PT")]
    if candidates:
        return candidates[0]
    fallback = [c for c in columns if year_str in c]
    if fallback:
        return fallback[0]
    raise KeyError(f"{year_str}年の総人口カラムが見つかりません。")


def load_population_mesh(
    shp_path: str | Path = DEFAULT_POP_SHP,
    bounds: MunicipalityBounds | None = None,
) -> gpd.GeoDataFrame:
    """500mメッシュ将来推計人口Shapefileを読み込み、WGS84変換・空間フィルタリングして返す."""
    path = Path(shp_path)
    if not path.exists():
        raise FileNotFoundError(f"人口メッシュデータが見つかりません: {path}")

    gdf = gpd.read_file(path)
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    cols = list(gdf.columns)
    col_2020 = _detect_pop_column(cols, "2020")
    col_2040 = _detect_pop_column(cols, "2040")
    mesh_col = "MESH_ID" if "MESH_ID" in cols else cols[0]

    out = gpd.GeoDataFrame(
        {
            "mesh_id": gdf[mesh_col].astype(str),
            "pop_2020": pd.to_numeric(gdf[col_2020], errors="coerce").fillna(0.0),
            "pop_2040": pd.to_numeric(gdf[col_2040], errors="coerce").fillna(0.0),
        },
        geometry=gdf.geometry,
        crs="EPSG:4326",
    )

    if bounds is not None:
        pad = 0.02
        out = out.cx[
            bounds.min_lon - pad : bounds.max_lon + pad,
            bounds.min_lat - pad : bounds.max_lat + pad,
        ].copy()

    # 2040年人口増減率 (%)
    out["pop_change_rate"] = np.where(
        out["pop_2020"] > 0.0,
        ((out["pop_2040"] - out["pop_2020"]) / out["pop_2020"]) * 100.0,
        0.0,
    ).round(1)

    return out.reset_index(drop=True)


def attach_population_to_facilities(
    df: pd.DataFrame,
    mesh_gdf: gpd.GeoDataFrame,
    radius_m: float = 1000.0,
) -> pd.DataFrame:
    """各施設の半径1km以内にあるメッシュの2040年人口合計と平均人口増減率を集計・付与する."""
    out = df.copy()
    if out.empty or mesh_gdf is None or mesh_gdf.empty:
        out["pop_2020_1km"] = 0.0
        out["pop_2040_1km"] = 0.0
        out["pop_change_rate_1km"] = 0.0
        return out

    # 半径1kmを度単位に換算してSTRtreeで高速検索
    deg_radius_lat = radius_m / _M_PER_LAT
    deg_radius_lon = radius_m / _M_PER_LON
    deg_radius = max(deg_radius_lat, deg_radius_lon)

    geoms = list(mesh_gdf.geometry)
    tree = STRtree(geoms)

    pop_2020_arr = mesh_gdf["pop_2020"].to_numpy(dtype=float)
    pop_2040_arr = mesh_gdf["pop_2040"].to_numpy(dtype=float)
    rate_arr = mesh_gdf["pop_change_rate"].to_numpy(dtype=float)

    p20_sums: list[float] = []
    p40_sums: list[float] = []
    rate_avgs: list[float] = []

    for lat, lon in zip(out["lat"].astype(float), out["lon"].astype(float)):
        buf = Point(lon, lat).buffer(deg_radius)
        cand_indices = tree.query(buf)
        matched = [int(i) for i in cand_indices if geoms[int(i)].intersects(buf)]

        if not matched:
            p20_sums.append(0.0)
            p40_sums.append(0.0)
            rate_avgs.append(0.0)
            continue

        p20_sub = pop_2020_arr[matched]
        p40_sub = pop_2040_arr[matched]
        rate_sub = rate_arr[matched]

        p20_sums.append(round(float(p20_sub.sum()), 1))
        p40_sums.append(round(float(p40_sub.sum()), 1))

        valid_rates = rate_sub[p20_sub > 0.0]
        if len(valid_rates) > 0:
            rate_avgs.append(round(float(valid_rates.mean()), 1))
        else:
            rate_avgs.append(0.0)

    out["pop_2020_1km"] = p20_sums
    out["pop_2040_1km"] = p40_sums
    out["pop_change_rate_1km"] = rate_avgs
    return out
