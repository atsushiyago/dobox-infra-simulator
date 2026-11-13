"""広島県インフラマネジメント基盤（DoboX）オープンデータの統合・前処理モジュール."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

BASE_YEAR = 2026
BRIDGE_CSV_NAME = "340006_road_bridge_2023.csv"
TUNNEL_CSV_NAME = "340006_road_tunnel_2023.csv"

FEATURED_MUNICIPALITIES = ["尾道市", "呉市", "安芸太田町", "庄原市"]

COMMON_COLUMNS = [
    "id",
    "name",
    "category",
    "route_name",
    "road_type",
    "built_year",
    "built_year_imputed",
    "age",
    "length",
    "width",
    "municipality",
    "lat",
    "lon",
]


def _normalize_built_year(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """建設・架設年度を数値化し、欠損・異常値を中央値で補完する."""
    years = pd.to_numeric(series, errors="coerce")
    # 西暦として妥当な範囲外を欠損扱いとする
    years = years.where((years >= 1850) & (years <= BASE_YEAR))
    imputed_flag = years.isna()
    median_year = years.median()
    if pd.isna(median_year):
        median_year = 1975.0
    filled_years = years.fillna(round(float(median_year))).astype(int)
    return filled_years, imputed_flag


def _load_single_csv(
    csv_path: Path,
    id_col: str,
    year_col: str,
    category_label: str,
) -> pd.DataFrame:
    """橋梁またはトンネルCSVを読み込み、共通スキーマに正規化する."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"入力データが見つかりません: {csv_path}\n"
            f"同じディレクトリに {BRIDGE_CSV_NAME} と {TUNNEL_CSV_NAME} を配置してください。"
        )

    raw_df = pd.read_csv(csv_path, encoding="utf-8-sig", dtype=str)
    raw_df.columns = [col.strip() for col in raw_df.columns]

    df = pd.DataFrame(
        {
            "id": raw_df[id_col].fillna("").astype(str).str.strip(),
            "name": raw_df["施設名"].fillna("名称不明").astype(str).str.strip(),
            "category": category_label,
            "route_name": raw_df["路線名"].fillna("不明").astype(str).str.strip(),
            "road_type": raw_df["道路種別"].fillna("不明").astype(str).str.strip(),
            "built_year_raw": raw_df[year_col],
            "length": pd.to_numeric(raw_df["延長(m)"], errors="coerce").fillna(0.0),
            "width": pd.to_numeric(raw_df["幅員(m)"], errors="coerce").fillna(0.0),
            "municipality": raw_df["住所(市町)"].fillna("").astype(str).str.strip(),
            "lat": pd.to_numeric(raw_df["緯度（10進数）"], errors="coerce"),
            "lon": pd.to_numeric(raw_df["経度（10進数）"], errors="coerce"),
        }
    )

    # 緯度・経度が欠損しているレコードを除外（広島県周辺の妥当範囲でフィルタ）
    df = df.dropna(subset=["lat", "lon"]).copy()
    df = df[(df["lat"].between(33.5, 35.5)) & (df["lon"].between(131.5, 134.0))].copy()

    # 施設名やIDが空の場合の補完
    df.loc[df["name"] == "", "name"] = "名称不明"
    df.loc[df["id"] == "", "id"] = [
        f"{category_label}_{idx}" for idx in range(len(df[df["id"] == ""]))
    ]

    return df


def load_infrastructure_data(data_dir: str | Path = ".") -> pd.DataFrame:
    """橋梁CSVとトンネルCSVを統合し、経過年数を算出して返す."""
    base_dir = Path(data_dir)
    bridges = _load_single_csv(
        base_dir / BRIDGE_CSV_NAME,
        id_col="橋梁番号",
        year_col="架設年度",
        category_label="道路橋",
    )
    tunnels = _load_single_csv(
        base_dir / TUNNEL_CSV_NAME,
        id_col="トンネル番号",
        year_col="建設年度",
        category_label="トンネル",
    )

    combined = pd.concat([bridges, tunnels], ignore_index=True)

    built_year, imputed_flag = _normalize_built_year(combined["built_year_raw"])
    combined["built_year"] = built_year
    combined["built_year_imputed"] = imputed_flag
    combined["age"] = np.maximum(0, BASE_YEAR - combined["built_year"]).astype(int)
    combined = combined.drop(columns=["built_year_raw"])

    # ID重複がある場合に備えてユニークキーを保証
    combined["id"] = combined["category"].str[0] + "_" + combined["id"]

    return combined[COMMON_COLUMNS].reset_index(drop=True)


def get_available_municipalities(df: pd.DataFrame) -> list[str]:
    """橋梁データに含まれる市町一覧を返す（主要市町を先頭に配置）."""
    bridge_munis = (
        df.loc[df["category"] == "道路橋", "municipality"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    unique_munis = sorted({m for m in bridge_munis if m})
    featured = [m for m in FEATURED_MUNICIPALITIES if m in unique_munis]
    others = [m for m in unique_munis if m not in featured]
    return featured + others


@dataclass
class MunicipalityBounds:
    """市町の橋梁座標から算出した空間範囲・中心座標・推奨ズームレベル."""

    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float
    center_lat: float
    center_lon: float
    zoom: int


def get_municipality_bounds(
    df: pd.DataFrame,
    municipality: str,
    margin_deg: float = 0.01,
) -> MunicipalityBounds | None:
    """橋梁CSVの住所(市町)一致行から緯度・経度の最小/最大・中心座標・ズームを算出する."""
    bridge_subset = df[
        (df["category"] == "道路橋") & (df["municipality"] == municipality)
    ].copy()
    if bridge_subset.empty:
        return None

    # 座標誤記等の極端な外れ値（中央値から0.4度以上乖離）を除外して範囲を計算
    med_lat = float(bridge_subset["lat"].median())
    med_lon = float(bridge_subset["lon"].median())
    inliers = bridge_subset[
        ((bridge_subset["lat"] - med_lat).abs() < 0.4)
        & ((bridge_subset["lon"] - med_lon).abs() < 0.4)
    ]
    if inliers.empty:
        inliers = bridge_subset

    min_lat = float(inliers["lat"].min())
    max_lat = float(inliers["lat"].max())
    min_lon = float(inliers["lon"].min())
    max_lon = float(inliers["lon"].max())
    center_lat = float(inliers["lat"].mean())
    center_lon = float(inliers["lon"].mean())

    span = max(max_lat - min_lat, max_lon - min_lon)
    if span > 0.35:
        zoom = 10
    elif span > 0.18:
        zoom = 11
    elif span > 0.08:
        zoom = 12
    else:
        zoom = 13

    return MunicipalityBounds(
        min_lat=min_lat - margin_deg,
        max_lat=max_lat + margin_deg,
        min_lon=min_lon - margin_deg,
        max_lon=max_lon + margin_deg,
        center_lat=center_lat,
        center_lon=center_lon,
        zoom=zoom,
    )


def filter_by_municipality(
    df: pd.DataFrame,
    municipality: str,
    categories: Iterable[str] | None = None,
) -> pd.DataFrame:
    """指定市町の橋梁およびその緯度・経度範囲(±0.01度)内のトンネルを抽出する."""
    bridge_mask = (df["category"] == "道路橋") & (df["municipality"] == municipality)
    bridge_subset = df[bridge_mask].copy()

    tunnels = df[df["category"] == "トンネル"].copy()
    bounds = get_municipality_bounds(df, municipality, margin_deg=0.01)

    if bounds is not None and not tunnels.empty:
        in_bbox = (
            (tunnels["lat"] >= bounds.min_lat)
            & (tunnels["lat"] <= bounds.max_lat)
            & (tunnels["lon"] >= bounds.min_lon)
            & (tunnels["lon"] <= bounds.max_lon)
        )
        tunnel_subset = tunnels[in_bbox].copy()
    else:
        tunnel_subset = tunnels[
            tunnels["municipality"].str.contains(municipality, na=False)
        ].copy()

    result = pd.concat([bridge_subset, tunnel_subset], ignore_index=True)

    if categories is not None:
        cat_set = set(categories)
        result = result[result["category"].isin(cat_set)].copy()

    return result.reset_index(drop=True)
