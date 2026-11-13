"""緊急輸送道路ネットワークおよび防災拠点（医療機関・避難施設）の読み込み・空間判定モジュール."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import osmnx as ox
import pandas as pd
import requests
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

from data_loader import MunicipalityBounds

EMERGENCY_ROUTE_FILES = [
    (1, "1次", "第1次", "05_kinkyu_route_01.json", "#070690", 4),
    (2, "2次", "第2次", "05_kinkyu_route_02.json", "#1C26F9", 3),
    (3, "3次", "第3次", "05_kinkyu_route_03.json", "#10B981", 3),
    (4, "4次", "第4次", "05_kinkyu_route_04.json", "#F7811D", 2),
]

EMERGENCY_BONUS_MAP = {
    "1次": 12.0,
    "2次": 7.0,
    "3次": 4.0,
    "4次": 2.0,
    "対象外": 0.0,
}

# 平面近似用（広島県中心付近の緯度34.4度を基準）
_REF_LAT = 34.4
_M_PER_LAT = 111320.0
_M_PER_LON = 111320.0 * math.cos(math.radians(_REF_LAT))


@dataclass
class EmergencyRouteSegment:
    """緊急輸送道路の1区間データ."""

    rank_num: int
    rank_short: str
    rank_label: str
    color: str
    weight: int
    coords_latlon: list[tuple[float, float]]
    geom_meters: LineString


def _parse_bracketless_json(path: Path) -> list:
    """最外周の [] が省略されたカンマ区切りJSONをパースする."""
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []
    # 先頭・末尾の余分なカンマを除去してから [] で囲む
    text = text.strip().rstrip(",").lstrip(",")
    if text.startswith("[") and text.endswith("]"):
        # すでに [a, b], [c, d] のような複数配列が並んでいるケースにも対応するため
        # まずそのまま json.loads を試し、Extra data の場合のみ [] で囲む
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    return json.loads(f"[{text}]")


def _load_route_styles(data_dir: Path) -> dict[str, tuple[str, int]]:
    """05_kinkyu_route.json から路線種別ごとの色・太さ設定を読み込む."""
    style_path = data_dir / "05_kinkyu_route.json"
    styles: dict[str, tuple[str, int]] = {}
    if not style_path.exists():
        return styles
    try:
        items = _parse_bracketless_json(style_path)
        for item in items:
            if isinstance(item, dict) and "name" in item:
                styles[str(item["name"])] = (
                    str(item.get("color", "#1C26F9")),
                    int(item.get("weight", 3)),
                )
    except Exception:
        pass
    return styles


def load_emergency_routes(data_dir: str | Path = ".") -> list[EmergencyRouteSegment]:
    """第1次〜第4次緊急輸送道路JSONを読み込み、空間データリストを返す."""
    base_dir = Path(data_dir)
    styles = _load_route_styles(base_dir)
    segments: list[EmergencyRouteSegment] = []

    for rank_num, rank_short, rank_label, filename, default_color, default_weight in EMERGENCY_ROUTE_FILES:
        file_path = base_dir / filename
        if not file_path.exists():
            continue

        style_key = f"route_0{rank_num}"
        color, weight = styles.get(style_key, (default_color, default_weight))
        # 第3次の #10F51C は白地図上でやや眩しいため視認性を保ちつつ定義色を使用
        raw_lines = _parse_bracketless_json(file_path)

        for line_pts in raw_lines:
            if not isinstance(line_pts, list) or len(line_pts) < 2:
                continue
            coords_latlon: list[tuple[float, float]] = []
            coords_m: list[tuple[float, float]] = []
            for pt in line_pts:
                if not isinstance(pt, dict) or "e" not in pt or "d" not in pt:
                    continue
                lon = float(pt["e"])
                lat = float(pt["d"])
                coords_latlon.append((lat, lon))
                coords_m.append((lon * _M_PER_LON, lat * _M_PER_LAT))

            if len(coords_latlon) >= 2:
                segments.append(
                    EmergencyRouteSegment(
                        rank_num=rank_num,
                        rank_short=rank_short,
                        rank_label=rank_label,
                        color=color,
                        weight=weight,
                        coords_latlon=coords_latlon,
                        geom_meters=LineString(coords_m),
                    )
                )

    return segments


def classify_emergency_routes(
    df: pd.DataFrame,
    routes: list[EmergencyRouteSegment],
    threshold_m: float = 30.0,
) -> pd.DataFrame:
    """各インフラ座標から最寄りの緊急輸送道路までの距離を算出し、30m以内を判定する."""
    out = df.copy()
    if out.empty or not routes:
        out["emergency_route_rank"] = "対象外"
        out["emergency_route_display"] = "-"
        out["emergency_bonus"] = 0.0
        return out

    geoms = [seg.geom_meters for seg in routes]
    tree = STRtree(geoms)

    ranks: list[str] = []
    displays: list[str] = []
    bonuses: list[float] = []

    for lat, lon in zip(out["lat"].astype(float), out["lon"].astype(float)):
        pt_m = Point(lon * _M_PER_LON, lat * _M_PER_LAT)
        # 30mバッファに交差する候補ラインを空間インデックスから取得
        cand_indices = tree.query(pt_m.buffer(threshold_m))

        best_seg: EmergencyRouteSegment | None = None
        for idx in cand_indices:
            seg = routes[int(idx)]
            if pt_m.distance(seg.geom_meters) <= threshold_m:
                if best_seg is None or seg.rank_num < best_seg.rank_num:
                    best_seg = seg

        if best_seg is not None:
            ranks.append(best_seg.rank_short)
            displays.append(best_seg.rank_label)
            bonuses.append(EMERGENCY_BONUS_MAP.get(best_seg.rank_short, 0.0))
        else:
            ranks.append("対象外")
            displays.append("-")
            bonuses.append(0.0)

    out["emergency_route_rank"] = ranks
    out["emergency_route_display"] = displays
    out["emergency_bonus"] = bonuses
    return out


def _features_from_bbox_compat(
    north: float, south: float, east: float, west: float, tags: dict
):
    """OSMnx v1系/v2系双方のbbox引数仕様に対応してPOIを取得する."""
    major_ver = int(str(ox.__version__).split(".")[0])
    if major_ver >= 2:
        return ox.features_from_bbox(bbox=(west, south, east, north), tags=tags)
    return ox.features_from_bbox(
        north=north, south=south, east=east, west=west, tags=tags
    )


_OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

# Overpass API タイムアウト・未取得時に計算を止めないための主要病院フォールバック座標（広島県内全22市町＋広島市）
FALLBACK_HOSPITALS: dict[str, list[tuple[str, float, float]]] = {
    "尾道市": [
        ("JA尾道総合病院", 34.4024, 133.1879),
        ("尾道市立市民病院", 34.4264, 133.1966),
        ("公立みつぎ総合病院", 34.5131, 133.1333),
        ("因島医師会病院", 34.3161, 133.1731),
    ],
    "呉市": [
        ("国立病院機構 呉医療センター", 34.2427, 132.5673),
        ("中国労災病院", 34.2349, 132.6258),
        ("国家公務員共済組合連合会 呉共済病院", 34.2496, 132.5608),
    ],
    "福山市": [
        ("福山市民病院", 34.4831, 133.3941),
        ("公立学校共済組合 中国中央病院", 34.5396, 133.3688),
        ("日本鋼管福山病院", 34.4998, 133.4282),
    ],
    "三原市": [
        ("三原赤十字病院", 34.3982, 133.0812),
        ("興生総合病院", 34.3991, 133.0764),
    ],
    "東広島市": [
        ("国立病院機構 東広島医療センター", 34.4126, 132.7289),
        ("本永病院", 34.4272, 132.7428),
    ],
    "三次市": [
        ("市立三次中央病院", 34.7966, 132.8669),
    ],
    "庄原市": [
        ("庄原赤十字病院", 34.8559, 133.0143),
        ("庄原市立西城病院", 34.9225, 133.1112),
    ],
    "竹原市": [
        ("竹原地区医師会立 竹原病院", 34.3375, 132.9056),
    ],
    "大竹市": [
        ("国立病院機構 広島西医療センター", 34.2289, 132.2219),
    ],
    "廿日市市": [
        ("JA広島総合病院", 34.3422, 132.3189),
    ],
    "安芸高田市": [
        ("安芸高田市立吉田総合病院", 34.6661, 132.7031),
    ],
    "江田島市": [
        ("江田島市立江田島病院", 34.2242, 132.4731),
    ],
    "府中市": [
        ("府中市民病院", 34.5736, 133.2411),
    ],
    "府中町": [
        ("あかね会 土谷総合病院（近隣基幹）", 34.3895, 132.4992),
    ],
    "海田町": [
        ("済生会広島病院（近隣基幹）", 34.3578, 132.5332),
    ],
    "熊野町": [
        ("熊野町立熊野病院", 34.3361, 132.5858),
    ],
    "坂町": [
        ("済生会広島病院", 34.3408, 132.5122),
    ],
    "安芸太田町": [
        ("町立安芸太田病院", 34.6143, 132.2756),
    ],
    "北広島町": [
        ("北広島町立雄鹿原診療所・千代田病院", 34.6744, 132.5397),
    ],
    "大崎上島町": [
        ("大崎上島町立診療所", 34.2419, 132.8936),
    ],
    "世羅町": [
        ("公立世羅中央病院", 34.5881, 133.0508),
    ],
    "神石高原町": [
        ("町立神石高原病院", 34.7736, 133.2486),
    ],
    "広島市": [
        ("広島大学病院", 34.3792, 132.4783),
        ("県立広島病院", 34.3694, 132.4631),
        ("広島市立広島市民病院", 34.3988, 132.4567),
    ],
}


def extract_facilities_from_elements(elements: list[dict]) -> list[dict]:
    """Overpass JSONのelements配列から医療機関・避難施設レコードを抽出する（out center / out geom 両対応）."""
    records: list[dict] = []
    for elem in elements:
        tags = elem.get("tags") or {}
        if not tags:
            continue
        amenity = str(tags.get("amenity", ""))
        healthcare = str(tags.get("healthcare", ""))
        if amenity in ("hospital", "clinic") or healthcare in ("hospital", "clinic"):
            f_type = "医療機関"
            sub_type = amenity if amenity in ("hospital", "clinic") else healthcare
        elif amenity in ("school", "community_centre"):
            f_type = "避難施設"
            sub_type = amenity
        else:
            continue

        if "lat" in elem and "lon" in elem:
            lat, lon = float(elem["lat"]), float(elem["lon"])
        elif "center" in elem and "lat" in elem["center"] and "lon" in elem["center"]:
            lat, lon = float(elem["center"]["lat"]), float(elem["center"]["lon"])
        elif "bounds" in elem and isinstance(elem["bounds"], dict):
            b = elem["bounds"]
            lat = 0.5 * (float(b["minlat"]) + float(b["maxlat"]))
            lon = 0.5 * (float(b["minlon"]) + float(b["maxlon"]))
        elif "geometry" in elem and isinstance(elem["geometry"], list) and elem["geometry"]:
            pts = [p for p in elem["geometry"] if isinstance(p, dict) and "lat" in p and "lon" in p]
            if not pts:
                continue
            lat = sum(float(p["lat"]) for p in pts) / len(pts)
            lon = sum(float(p["lon"]) for p in pts) / len(pts)
        else:
            continue

        raw_name = tags.get("name")
        if not raw_name or not str(raw_name).strip():
            default_label = {
                "hospital": "病院",
                "clinic": "診療所",
                "school": "学校（避難所相当）",
                "community_centre": "公民館（避難所相当）",
            }.get(sub_type, f_type)
            name = default_label
        else:
            name = str(raw_name).strip()

        records.append(
            {
                "name": name,
                "facility_type": f_type,
                "amenity": sub_type,
                "lat": lat,
                "lon": lon,
            }
        )
    return records


def save_facilities_from_elements(
    elements: list[dict],
    municipality: str,
    bounds: MunicipalityBounds | None,
    cache_dir: str | Path = ".cache/osm",
) -> pd.DataFrame:
    """統合Overpassレスポンスから防災拠点を抽出・フォールバック補完してCSVキャッシュに保存する."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in municipality)
    csv_cache = cache_path / f"{safe_name}_disaster_facilities.csv"

    records = extract_facilities_from_elements(elements)
    records = _append_fallback_hospitals(records, municipality, bounds)
    result = pd.DataFrame(records)
    if not result.empty:
        result = result.drop_duplicates(subset=["name", "facility_type", "lat", "lon"]).reset_index(drop=True)
        result.to_csv(csv_cache, index=False, encoding="utf-8-sig")
    return result


def _fetch_facilities_overpass_fast(
    north: float, south: float, east: float, west: float
) -> list[dict]:
    """Overpass API の out center を使って医療機関・避難施設の座標を軽量高速取得する."""
    query = (
        f"[out:json][timeout:12];"
        f"("
        f'nwr["amenity"~"^(hospital|clinic|school|community_centre)$"]({south},{west},{north},{east});'
        f'nwr["healthcare"~"^(hospital|clinic)$"]({south},{west},{north},{east});'
        f");"
        f"out center tags;"
    )
    data = None
    last_err: Exception | None = None
    for endpoint in _OVERPASS_ENDPOINTS:
        try:
            resp = requests.post(endpoint, data={"data": query}, timeout=8)
            resp.raise_for_status()
            data = resp.json()
            break
        except Exception as exc:
            last_err = exc
            continue
    if data is None:
        raise RuntimeError(f"Overpass API からの取得に失敗しました: {last_err}")

    records: list[dict] = []
    for elem in data.get("elements", []):
        tags = elem.get("tags") or {}
        amenity = str(tags.get("amenity", ""))
        healthcare = str(tags.get("healthcare", ""))
        if amenity in ("hospital", "clinic") or healthcare in ("hospital", "clinic"):
            f_type = "医療機関"
            sub_type = amenity if amenity in ("hospital", "clinic") else healthcare
        elif amenity in ("school", "community_centre"):
            f_type = "避難施設"
            sub_type = amenity
        else:
            continue

        if "lat" in elem and "lon" in elem:
            lat, lon = float(elem["lat"]), float(elem["lon"])
        elif "center" in elem and "lat" in elem["center"] and "lon" in elem["center"]:
            lat, lon = float(elem["center"]["lat"]), float(elem["center"]["lon"])
        else:
            continue

        raw_name = tags.get("name")
        if not raw_name or not str(raw_name).strip():
            default_label = {
                "hospital": "病院",
                "clinic": "診療所",
                "school": "学校（避難所相当）",
                "community_centre": "公民館（避難所相当）",
            }.get(sub_type, f_type)
            name = default_label
        else:
            name = str(raw_name).strip()

        records.append(
            {
                "name": name,
                "facility_type": f_type,
                "amenity": sub_type,
                "lat": lat,
                "lon": lon,
            }
        )
    return records


def _fetch_hospitals_osmnx_fallback(
    north: float, south: float, east: float, west: float
) -> list[dict]:
    """OSMnx features_from_bbox で医療機関（Polygonはcentroid）を取得する."""
    tags = {
        "amenity": ["hospital", "clinic"],
        "healthcare": ["hospital", "clinic"],
    }
    try:
        ox.settings.requests_timeout = 12
        gdf = _features_from_bbox_compat(north, south, east, west, tags=tags)
    except Exception:
        return []

    if gdf is None or gdf.empty:
        return []

    records: list[dict] = []
    for _, row in gdf.iterrows():
        geom = row.get("geometry")
        if geom is None or geom.is_empty:
            continue
        pt = geom if geom.geom_type == "Point" else geom.centroid
        raw_name = row.get("name")
        name = (
            str(raw_name).strip()
            if (raw_name is not None and not pd.isna(raw_name) and str(raw_name).strip())
            else "医療機関"
        )
        amenity = str(row.get("amenity") or row.get("healthcare") or "hospital")
        records.append(
            {
                "name": name,
                "facility_type": "医療機関",
                "amenity": amenity,
                "lat": float(pt.y),
                "lon": float(pt.x),
            }
        )
    return records


def _append_fallback_hospitals(
    records: list[dict],
    municipality: str,
    bounds: MunicipalityBounds | None,
) -> list[dict]:
    """医療機関が0件の場合、対象市町の主要病院フォールバック座標を補完する."""
    has_med = any(r.get("facility_type") == "医療機関" for r in records)
    if has_med:
        return records

    fb_list = FALLBACK_HOSPITALS.get(municipality, [])
    if not fb_list and bounds is not None:
        fb_list = [(f"{municipality}中心医療機関（代替）", bounds.center_lat, bounds.center_lon)]

    for name, lat, lon in fb_list:
        records.append(
            {
                "name": name,
                "facility_type": "医療機関",
                "amenity": "hospital",
                "lat": lat,
                "lon": lon,
            }
        )
    return records


def load_disaster_facilities(
    municipality: str,
    bounds: MunicipalityBounds | None,
    df_area: pd.DataFrame | None = None,
    cache_dir: str | Path = ".cache/osm",
) -> pd.DataFrame:
    """選択市町内の医療機関・避難施設を取得しローカルキャッシュする."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in municipality)
    csv_cache = cache_path / f"{safe_name}_disaster_facilities.csv"

    if csv_cache.exists():
        cached_df = pd.read_csv(csv_cache, encoding="utf-8-sig")
        if not cached_df.empty and (cached_df["facility_type"] == "医療機関").any():
            return cached_df

    if bounds is not None:
        north, south = bounds.max_lat, bounds.min_lat
        east, west = bounds.max_lon, bounds.min_lon
    elif df_area is not None and not df_area.empty:
        north = float(df_area["lat"].max()) + 0.01
        south = float(df_area["lat"].min()) - 0.01
        east = float(df_area["lon"].max()) + 0.01
        west = float(df_area["lon"].min()) - 0.01
    else:
        return pd.DataFrame(columns=["name", "facility_type", "amenity", "lat", "lon"])

    # 1. 軽量・高速な Overpass out center クエリで取得（タイムアウト8秒）
    try:
        records = _fetch_facilities_overpass_fast(north, south, east, west)
    except Exception:
        records = []

    # 2. 医療機関が0件または未取得の場合は市町の主要病院フォールバック座標で即座に補完
    records = _append_fallback_hospitals(records, municipality, bounds)

    result = pd.DataFrame(records)
    if not result.empty:
        result = result.drop_duplicates(subset=["name", "facility_type", "lat", "lon"]).reset_index(drop=True)
        result.to_csv(csv_cache, index=False, encoding="utf-8-sig")
    return result
