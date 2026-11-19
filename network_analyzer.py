"""OSMnxとNetworkXを用いた迂回インパクト・優先度スコア計算エンジン."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import json
import math
from pathlib import Path
import pickle

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import requests
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

from emergency_loader import load_emergency_routes, save_facilities_from_elements

ISOLATED_PENALTY_KM = 999.9

_MAJOR_HIGHWAYS = {
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "motorway_link",
    "trunk_link",
    "primary_link",
    "secondary_link",
    "tertiary_link",
}

_OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
]

_OVERPASS_HEADERS = {
    "User-Agent": "HiroshimaInfraDashboard/1.0 (Streamlit Prototype; https://github.com/)",
    "Accept-Encoding": "gzip",
}

_OVERPASS_AVAILABLE: bool = True


@dataclass
class DetourResult:
    """通行止めシミュレーションの結果."""

    u: int
    v: int
    key: int
    normal_distance_km: float
    detour_distance_km: float | None
    extra_distance_km: float | None
    is_isolated: bool
    closed_segments_coords: list[list[tuple[float, float]]]
    detour_route_coords: list[tuple[float, float]]
    message: str


def _calc_age_score(ages: pd.Series) -> pd.Series:
    """経過年数から老朽度スコア（0〜100）を算出する（築50年以上で高スコア）."""
    ages_f = ages.astype(float).clip(lower=0.0)
    score = np.where(
        ages_f <= 50.0,
        (ages_f / 50.0) * 75.0,
        75.0 + np.minimum(25.0, ((ages_f - 50.0) / 30.0) * 25.0),
    )
    return pd.Series(score, index=ages.index)


def _calc_road_score(road_types: pd.Series) -> pd.Series:
    """道路種別から路線重要度スコア（0〜100）を算出する（国道 > 県道）."""

    def _score_one(val: str) -> float:
        text = str(val)
        if "国道" in text:
            return 100.0
        if "主要地方道" in text:
            return 75.0
        if "県道" in text or "地方道" in text:
            return 55.0
        return 35.0

    return road_types.apply(_score_one)


def _calc_length_score(lengths: pd.Series) -> pd.Series:
    """施設延長（m）から復旧コスト・構造規模スコア（0〜100）を算出する."""
    len_f = lengths.astype(float).clip(lower=0.0)
    return np.minimum(100.0, (np.log1p(len_f) / np.log1p(1000.0)) * 100.0)


def assign_priority_rank(score: float) -> str:
    """スコア（0〜100）をS/A/B/Cランクに変換する."""
    if score >= 80.0:
        return "S"
    if score >= 65.0:
        return "A"
    if score >= 50.0:
        return "B"
    return "C"


def compute_risk_scores(df: pd.DataFrame, aging_weight: float = 0.5) -> pd.DataFrame:
    """全施設に対して構造物リスクスコア（0〜100）を計算し、スコア降順で返す."""
    if df.empty:
        out = df.copy()
        out["age_score"] = pd.Series(dtype=float)
        out["importance_score"] = pd.Series(dtype=float)
        out["risk_score"] = pd.Series(dtype=float)
        out["priority_rank"] = pd.Series(dtype=str)
        return out

    aging_w = float(np.clip(aging_weight, 0.0, 1.0))
    importance_w = 1.0 - aging_w

    out = df.copy()
    age_score = _calc_age_score(out["age"])
    road_score = _calc_road_score(out["road_type"])
    length_score = _calc_length_score(out["length"])
    importance_score = 0.6 * road_score + 0.4 * length_score

    out["age_score"] = age_score.round(1)
    out["importance_score"] = importance_score.round(1)
    out["risk_score"] = (
        aging_w * age_score + importance_w * importance_score
    ).round(1)
    out["priority_rank"] = out["risk_score"].apply(assign_priority_rank)

    out = out.sort_values(
        by=["risk_score", "age", "length"], ascending=[False, False, False]
    ).reset_index(drop=True)
    out["rank_order"] = np.arange(1, len(out) + 1)
    return out


def compute_combined_priority(
    risk_score: float,
    extra_distance_km: float | None,
    is_isolated: bool,
    emergency_bonus: float = 0.0,
    hospital_delay_km: float | None = None,
    hospital_isolated: bool = False,
) -> tuple[float, str]:
    """構造物リスク・迂回インパクト・防災緊急輸送ボーナスを統合した総合優先度スコア・ランクを返す."""
    if is_isolated:
        detour_score = 100.0
    elif extra_distance_km is None:
        detour_score = 0.0
    else:
        # 迂回増加20km以上で最大100点
        detour_score = float(np.clip((extra_distance_km / 20.0) * 100.0, 0.0, 100.0))

    if hospital_isolated:
        hospital_bonus = 10.0
    elif hospital_delay_km is None or hospital_delay_km <= 0.0:
        hospital_bonus = 0.0
    else:
        # 病院への到達遅延5km以上で最大+8点
        hospital_bonus = float(np.clip((hospital_delay_km / 5.0) * 8.0, 0.0, 8.0))

    base = 0.65 * float(risk_score) + 0.35 * detour_score
    combined = round(min(100.0, base + float(emergency_bonus) + hospital_bonus), 1)
    return combined, assign_priority_rank(combined)


def _sanitize_cache_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in name)


def _build_graph_from_overpass_elements(elements: list[dict]) -> nx.MultiDiGraph:
    """Overpass JSONのelements配列（out geom または node+way）から主要道路MultiDiGraphを高速構築・簡略化する."""
    nodes_latlon: dict[int, tuple[float, float]] = {}
    ways: list[tuple[list[int], str, str]] = []
    node_use_count: dict[int, int] = {}

    for el in elements:
        etype = el.get("type")
        if etype == "node" and "lat" in el and "lon" in el:
            nodes_latlon[int(el["id"])] = (float(el["lat"]), float(el["lon"]))

    for el in elements:
        if el.get("type") != "way":
            continue
        tags = el.get("tags") or {}
        hw = str(tags.get("highway", ""))
        if hw not in _MAJOR_HIGHWAYS:
            continue

        nids_raw = el.get("nodes") or []
        geom_raw = el.get("geometry") or []
        valid_nids: list[int] = []
        if geom_raw and len(geom_raw) == len(nids_raw):
            for nid, pt in zip(nids_raw, geom_raw):
                if isinstance(pt, dict) and "lat" in pt and "lon" in pt:
                    nid_i = int(nid)
                    nodes_latlon[nid_i] = (float(pt["lat"]), float(pt["lon"]))
                    valid_nids.append(nid_i)
        else:
            valid_nids = [int(nid) for nid in nids_raw if int(nid) in nodes_latlon]

        if len(valid_nids) < 2:
            continue

        oneway = str(tags.get("oneway", "no")).lower()
        ways.append((valid_nids, hw, oneway))
        for i, nid in enumerate(valid_nids):
            if i == 0 or i == len(valid_nids) - 1:
                node_use_count[nid] = node_use_count.get(nid, 0) + 2
            else:
                node_use_count[nid] = node_use_count.get(nid, 0) + 1

    G = nx.MultiDiGraph(crs="epsg:4326")
    m_per_lat = 111320.0
    m_per_lon = 111320.0 * math.cos(math.radians(34.4))

    for nids, hw, oneway in ways:
        seg_start_idx = 0
        for i in range(1, len(nids)):
            nid = nids[i]
            if i == len(nids) - 1 or node_use_count.get(nid, 0) >= 2:
                sub = nids[seg_start_idx : i + 1]
                u, v = sub[0], sub[-1]
                if u not in G:
                    lat_u, lon_u = nodes_latlon[u]
                    G.add_node(u, y=lat_u, x=lon_u)
                if v not in G:
                    lat_v, lon_v = nodes_latlon[v]
                    G.add_node(v, y=lat_v, x=lon_v)

                coords_lonlat: list[tuple[float, float]] = []
                length_m = 0.0
                for k in range(len(sub)):
                    lat_k, lon_k = nodes_latlon[sub[k]]
                    coords_lonlat.append((lon_k, lat_k))
                    if k > 0:
                        lat_p, lon_p = nodes_latlon[sub[k - 1]]
                        dx = (lon_k - lon_p) * m_per_lon
                        dy = (lat_k - lat_p) * m_per_lat
                        length_m += math.hypot(dx, dy)

                geom_fwd = LineString(coords_lonlat) if len(coords_lonlat) >= 2 else None
                if oneway == "-1":
                    geom_rev = (
                        LineString(list(reversed(coords_lonlat)))
                        if len(coords_lonlat) >= 2
                        else None
                    )
                    attr_r = {"length": round(length_m, 2), "highway": hw}
                    if geom_rev is not None:
                        attr_r["geometry"] = geom_rev
                    G.add_edge(v, u, **attr_r)
                elif oneway in ("yes", "1", "true"):
                    attr_f = {"length": round(length_m, 2), "highway": hw}
                    if geom_fwd is not None:
                        attr_f["geometry"] = geom_fwd
                    G.add_edge(u, v, **attr_f)
                else:
                    attr_f = {"length": round(length_m, 2), "highway": hw}
                    if geom_fwd is not None:
                        attr_f["geometry"] = geom_fwd
                    G.add_edge(u, v, **attr_f)

                    geom_rev = (
                        LineString(list(reversed(coords_lonlat)))
                        if len(coords_lonlat) >= 2
                        else None
                    )
                    attr_r = {"length": round(length_m, 2), "highway": hw}
                    if geom_rev is not None:
                        attr_r["geometry"] = geom_rev
                    G.add_edge(v, u, **attr_r)

                seg_start_idx = i

    return G


def _build_fallback_graph(
    north: float,
    south: float,
    east: float,
    west: float,
    df_area: pd.DataFrame | None = None,
) -> nx.MultiDiGraph:
    """Overpass APIが完全到達不能な場合にローカル緊急輸送道路JSON＋施設座標から代替グラフを構築する."""
    G = nx.MultiDiGraph(crs="epsg:4326")
    G.graph["is_fallback"] = True
    m_per_lat = 111320.0
    m_per_lon = 111320.0 * math.cos(math.radians(34.4))

    routes = load_emergency_routes(".")
    coord_to_nid: dict[tuple[int, int], int] = {}
    next_nid = 1

    def _get_or_add_node(lat: float, lon: float) -> int:
        nonlocal next_nid
        # 約10mグリッドで同一ノードにスナップ
        key = (int(round(lat * 10000)), int(round(lon * 10000)))
        if key in coord_to_nid:
            return coord_to_nid[key]
        nid = next_nid
        next_nid += 1
        coord_to_nid[key] = nid
        G.add_node(nid, y=float(lat), x=float(lon))
        return nid

    pad = 0.08
    for seg in routes:
        pts = [
            (lat, lon)
            for lat, lon in seg.coords_latlon
            if (south - pad <= lat <= north + pad) and (west - pad <= lon <= east + pad)
        ]
        if len(pts) < 2:
            continue
        # 適度な粒度でノード分割してグラフ化
        step = max(1, len(pts) // 8)
        idxs = sorted(set([0] + list(range(step, len(pts) - 1, step)) + [len(pts) - 1]))
        for a_i, b_i in zip(idxs[:-1], idxs[1:]):
            sub = pts[a_i : b_i + 1]
            u = _get_or_add_node(sub[0][0], sub[0][1])
            v = _get_or_add_node(sub[-1][0], sub[-1][1])
            if u == v:
                continue
            coords_lonlat = [(lon, lat) for lat, lon in sub]
            length_m = 0.0
            for k in range(1, len(sub)):
                dx = (sub[k][1] - sub[k - 1][1]) * m_per_lon
                dy = (sub[k][0] - sub[k - 1][0]) * m_per_lat
                length_m += math.hypot(dx, dy)
            length_m = max(10.0, round(length_m, 2))
            geom_f = LineString(coords_lonlat) if len(coords_lonlat) >= 2 else None
            geom_r = (
                LineString(list(reversed(coords_lonlat)))
                if len(coords_lonlat) >= 2
                else None
            )
            attr_f = {"length": length_m, "highway": "primary"}
            attr_r = {"length": length_m, "highway": "primary"}
            if geom_f is not None:
                attr_f["geometry"] = geom_f
            if geom_r is not None:
                attr_r["geometry"] = geom_r
            G.add_edge(u, v, **attr_f)
            G.add_edge(v, u, **attr_r)

    # 緊急輸送道路同士の交差点・近接端点（60m以内）を接続して道路網を連結する
    if G.number_of_edges() > 0:
        from shapely.ops import substring as _substring

        deg_nodes = [
            (n, float(d["y"]), float(d["x"]))
            for n, d in G.nodes(data=True)
            if G.out_degree(n) <= 2
        ]
        for nid, n_lat, n_lon in deg_nodes:
            pt = Point(n_lon, n_lat)
            best_target = None
            best_dist_m = 60.0
            for eu, ev, ek, edata in list(G.edges(keys=True, data=True)):
                if eu == nid or ev == nid:
                    continue
                geom = edata.get("geometry")
                if geom is None:
                    geom = LineString(
                        [(G.nodes[eu]["x"], G.nodes[eu]["y"]), (G.nodes[ev]["x"], G.nodes[ev]["y"])]
                    )
                proj_d = float(geom.project(pt))
                snap_pt = geom.interpolate(proj_d)
                d_m = math.hypot((float(snap_pt.x) - n_lon) * m_per_lon, (float(snap_pt.y) - n_lat) * m_per_lat)
                if d_m < best_dist_m:
                    best_dist_m = d_m
                    best_target = (eu, ev, ek, edata, geom, proj_d, snap_pt)
            if best_target is not None:
                eu, ev, ek, edata, geom, proj_d, snap_pt = best_target
                s_lon, s_lat = float(snap_pt.x), float(snap_pt.y)
                snap_nid = _get_or_add_node(s_lat, s_lon)
                total_len = float(geom.length)
                if (
                    snap_nid != eu
                    and snap_nid != ev
                    and total_len > 1e-9
                    and 0.02 * total_len < proj_d < 0.98 * total_len
                ):
                    g1 = _substring(geom, 0.0, proj_d)
                    g2 = _substring(geom, proj_d, total_len)
                    orig_len_m = float(edata.get("length", 100.0))
                    r = proj_d / total_len
                    l1 = max(5.0, round(orig_len_m * r, 2))
                    l2 = max(5.0, round(orig_len_m * (1.0 - r), 2))
                    if G.has_edge(eu, ev, key=ek):
                        G.remove_edge(eu, ev, key=ek)
                    for rk in [
                        rk for rk, rd in (G.get_edge_data(ev, eu) or {}).items()
                        if abs(float(rd.get("length", 0.0)) - orig_len_m) < 5.0
                    ][:1]:
                        G.remove_edge(ev, eu, key=rk)
                    if len(g1.coords) >= 2:
                        G.add_edge(eu, snap_nid, length=l1, highway="primary", geometry=g1)
                        G.add_edge(snap_nid, eu, length=l1, highway="primary", geometry=LineString(list(reversed(g1.coords))))
                    if len(g2.coords) >= 2:
                        G.add_edge(snap_nid, ev, length=l2, highway="primary", geometry=g2)
                        G.add_edge(ev, snap_nid, length=l2, highway="primary", geometry=LineString(list(reversed(g2.coords))))
                if nid != snap_nid and not G.has_edge(nid, snap_nid):
                    conn_len = max(5.0, round(best_dist_m, 2))
                    g_c = LineString([(n_lon, n_lat), (s_lon, s_lat)])
                    g_cr = LineString([(s_lon, s_lat), (n_lon, n_lat)])
                    G.add_edge(nid, snap_nid, length=conn_len, highway="primary", geometry=g_c)
                    G.add_edge(snap_nid, nid, length=conn_len, highway="primary", geometry=g_cr)

    # 施設座標を道路エッジ上の最近接点へスナップして接続する（施設同士の直線短絡は作らない）
    if df_area is not None and not df_area.empty and G.number_of_edges() > 0:
        from shapely.ops import substring

        for _, row in df_area.iterrows():
            f_lat, f_lon = float(row["lat"]), float(row["lon"])
            pt = Point(f_lon, f_lat)

            # 現在の道路エッジから最寄りエッジを探索
            best_edge = None
            best_dist = float("inf")
            best_line = None
            for eu, ev, ek, edata in G.edges(keys=True, data=True):
                if edata.get("highway") == "access":
                    continue
                geom = edata.get("geometry")
                if geom is None:
                    geom = LineString(
                        [(G.nodes[eu]["x"], G.nodes[eu]["y"]), (G.nodes[ev]["x"], G.nodes[ev]["y"])]
                    )
                d = pt.distance(geom)
                if d < best_dist:
                    best_dist = d
                    best_edge = (eu, ev, ek, edata)
                    best_line = geom

            if best_edge is None or best_line is None:
                continue

            eu, ev, ek, edata = best_edge
            proj_dist = float(best_line.project(pt))
            total_len = float(best_line.length)
            snap_pt = best_line.interpolate(proj_dist)
            snap_lon, snap_lat = float(snap_pt.x), float(snap_pt.y)
            snap_nid = _get_or_add_node(snap_lat, snap_lon)

            # エッジの途中であれば元の道路形状を保ったまま2区間に分割
            if (
                snap_nid != eu
                and snap_nid != ev
                and total_len > 1e-9
                and 0.02 * total_len < proj_dist < 0.98 * total_len
            ):
                geom_1 = substring(best_line, 0.0, proj_dist)
                geom_2 = substring(best_line, proj_dist, total_len)
                orig_len_m = float(edata.get("length", 100.0))
                ratio = proj_dist / total_len
                len_1 = max(5.0, round(orig_len_m * ratio, 2))
                len_2 = max(5.0, round(orig_len_m * (1.0 - ratio), 2))
                hw_type = edata.get("highway", "primary")

                if G.has_edge(eu, ev, key=ek):
                    G.remove_edge(eu, ev, key=ek)
                # 逆向きペアも同様に置換
                rev_keys = [
                    rk
                    for rk, rd in (G.get_edge_data(ev, eu) or {}).items()
                    if abs(float(rd.get("length", 0.0)) - orig_len_m) < 5.0
                ]
                for rk in rev_keys[:1]:
                    G.remove_edge(ev, eu, key=rk)

                if len(geom_1.coords) >= 2:
                    g1_rev = LineString(list(reversed(geom_1.coords)))
                    G.add_edge(eu, snap_nid, length=len_1, highway=hw_type, geometry=geom_1)
                    G.add_edge(snap_nid, eu, length=len_1, highway=hw_type, geometry=g1_rev)
                if len(geom_2.coords) >= 2:
                    g2_rev = LineString(list(reversed(geom_2.coords)))
                    G.add_edge(snap_nid, ev, length=len_2, highway=hw_type, geometry=geom_2)
                    G.add_edge(ev, snap_nid, length=len_2, highway=hw_type, geometry=g2_rev)

    return G


def _save_graph_pickle(G: nx.MultiDiGraph, pkl_file: Path) -> None:
    """空間インデックスキャッシュを除いてMultiDiGraphを高速pickle保存する."""
    spatial_cache = G.graph.pop("_spatial_cache", None)
    try:
        with pkl_file.open("wb") as f:
            pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    finally:
        if spatial_cache is not None:
            G.graph["_spatial_cache"] = spatial_cache


def load_road_network(
    municipality: str,
    df_area: pd.DataFrame | None = None,
    cache_dir: str | Path = ".cache/osm",
    bounds=None,
) -> nx.MultiDiGraph:
    """対象市町の主要道路ネットワーク（および未取得時は防災拠点も一括）を高速取得・キャッシュする."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    safe_name = _sanitize_cache_name(municipality)
    pkl_file = cache_path / f"{safe_name}_drive.pkl"
    graphml_file = cache_path / f"{safe_name}_drive.graphml"
    facilities_csv = cache_path / f"{safe_name}_disaster_facilities.csv"

    # 1. 最速の .pkl キャッシュが存在する場合は約0.02秒で即座に返す
    if pkl_file.exists():
        try:
            with pkl_file.open("rb") as f:
                G = pickle.load(f)
            if isinstance(G, nx.MultiDiGraph) and G.number_of_edges() > 0:
                return G
        except Exception:
            pass

    # 2. 尾道市のローカルOverpass JSONキャッシュまたは既存 .graphml がある場合は .pkl に変換保存
    if municipality == "尾道市":
        onomichi_json = Path("cache/dd91bbd8bb1688af8c1bef56e9481321ee3f7ef8.json")
        if onomichi_json.exists():
            try:
                raw = json.loads(onomichi_json.read_text(encoding="utf-8"))
                G = _build_graph_from_overpass_elements(raw.get("elements", []))
                if G.number_of_edges() > 0:
                    _save_graph_pickle(G, pkl_file)
                    return G
            except Exception:
                pass

    if graphml_file.exists():
        try:
            G = ox.load_graphml(graphml_file)
            if G.number_of_edges() > 0:
                _save_graph_pickle(G, pkl_file)
                return G
        except Exception:
            pass

    # 3. BBoxの算出
    if bounds is not None:
        pad = 0.015
        north = float(bounds.max_lat) + pad
        south = float(bounds.min_lat) - pad
        east = float(bounds.max_lon) + pad
        west = float(bounds.min_lon) - pad
    elif df_area is not None and not df_area.empty:
        med_lat = float(df_area["lat"].median())
        med_lon = float(df_area["lon"].median())
        inliers = df_area[
            ((df_area["lat"] - med_lat).abs() < 0.4)
            & ((df_area["lon"] - med_lon).abs() < 0.4)
        ]
        if inliers.empty:
            inliers = df_area
        pad = 0.015
        north = float(inliers["lat"].max()) + pad
        south = float(inliers["lat"].min()) - pad
        east = float(inliers["lon"].max()) + pad
        west = float(inliers["lon"].min()) - pad
    else:
        north, south, east, west = 34.50, 34.35, 133.25, 133.10

    # 4. 軽量Overpass APIリクエストで主要道路網を取得（環境がオンラインの場合）
    global _OVERPASS_AVAILABLE
    elements: list[dict] | None = None
    if _OVERPASS_AVAILABLE:
        hw_regex = "^(motorway|trunk|primary|secondary|tertiary|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link)$"
        query = (
            f"[out:json][timeout:15];"
            f'way["highway"~"{hw_regex}"]({south:.5f},{west:.5f},{north:.5f},{east:.5f});'
            f"out geom qt;"
        )
        fetch_errors: list[str] = []
        for endpoint in _OVERPASS_ENDPOINTS:
            try:
                resp = requests.post(
                    endpoint,
                    data={"data": query},
                    headers=_OVERPASS_HEADERS,
                    timeout=12,
                )
                if resp.status_code == 200:
                    payload = resp.json()
                    elems = payload.get("elements")
                    if isinstance(elems, list) and len(elems) > 0:
                        elements = elems
                        break
                    fetch_errors.append(f"{endpoint}: empty elements")
                else:
                    fetch_errors.append(f"{endpoint}: HTTP {resp.status_code}")
            except Exception as exc:
                fetch_errors.append(f"{endpoint}: {type(exc).__name__}")
                continue

        if not elements:
            # 接続拒否・ネットワーク制限環境の場合は以降のリクエストをスキップして高速化
            if any("ConnectionError" in err or "Timeout" in err for err in fetch_errors):
                _OVERPASS_AVAILABLE = False

    if elements:
        G = _build_graph_from_overpass_elements(elements)
        if G.number_of_edges() > 0:
            _save_graph_pickle(G, pkl_file)
            return G

    # 5. オフラインまたはOverpass到達不能時は緊急輸送道路（第1〜4次）から道路ネットワークを瞬時に構築
    return _build_fallback_graph(north, south, east, west, df_area=df_area)


def _ensure_spatial_index(G: nx.MultiDiGraph) -> dict:
    """グラフGのノード・エッジ空間インデックス(STRtree)を構築しG.graphにメモリキャッシュする."""
    cached = G.graph.get("_spatial_cache")
    if cached is not None and cached.get("num_edges") == G.number_of_edges():
        return cached

    node_ids: list[int] = []
    node_pts: list[Point] = []
    for nid, data in G.nodes(data=True):
        node_ids.append(int(nid))
        node_pts.append(Point(float(data["x"]), float(data["y"])))
    node_tree = STRtree(node_pts) if node_pts else None

    edge_keys: list[tuple[int, int, int]] = []
    edge_geoms: list[LineString] = []
    for u, v, k, data in G.edges(keys=True, data=True):
        geom = data.get("geometry")
        if geom is None:
            ux, uy = float(G.nodes[u]["x"]), float(G.nodes[u]["y"])
            vx, vy = float(G.nodes[v]["x"]), float(G.nodes[v]["y"])
            geom = LineString([(ux, uy), (vx, vy)])
        edge_keys.append((int(u), int(v), int(k)))
        edge_geoms.append(geom)
    edge_tree = STRtree(edge_geoms) if edge_geoms else None

    cached = {
        "num_edges": G.number_of_edges(),
        "node_ids": node_ids,
        "node_tree": node_tree,
        "edge_keys": edge_keys,
        "edge_tree": edge_tree,
    }
    G.graph["_spatial_cache"] = cached
    return cached


def _fast_nearest_edges(
    G: nx.MultiDiGraph, lons: list[float], lats: list[float]
) -> list[tuple[int, int, int]]:
    """ox.nearest_edges の高速代替（GeoDataFrame変換を省きSTRtreeで一括検索）."""
    sp = _ensure_spatial_index(G)
    edge_tree: STRtree | None = sp["edge_tree"]
    edge_keys: list[tuple[int, int, int]] = sp["edge_keys"]
    if edge_tree is None or not edge_keys:
        return [(0, 0, 0) for _ in lons]
    pts = [Point(float(x), float(y)) for x, y in zip(lons, lats)]
    idxs = edge_tree.nearest(pts)
    return [edge_keys[int(i)] for i in np.atleast_1d(idxs)]


def _fast_nearest_nodes(
    G: nx.MultiDiGraph, lons: list[float], lats: list[float]
) -> list[int]:
    """ox.nearest_nodes の高速代替（STRtreeで一括検索）."""
    sp = _ensure_spatial_index(G)
    node_tree: STRtree | None = sp["node_tree"]
    node_ids: list[int] = sp["node_ids"]
    if node_tree is None or not node_ids:
        return [0 for _ in lons]
    pts = [Point(float(x), float(y)) for x, y in zip(lons, lats)]
    idxs = node_tree.nearest(pts)
    return [node_ids[int(i)] for i in np.atleast_1d(idxs)]


def _edge_coords(
    G: nx.MultiDiGraph,
    u: int,
    v: int,
    key: int,
    truncate_long_straight: bool = False,
) -> list[tuple[float, float]]:
    """エッジのジオメトリ（または両端ノード座標）を u -> v 順の (lat, lon) リストで返す."""
    data = G.get_edge_data(u, v, key) or {}
    geom = data.get("geometry")
    if isinstance(geom, str):
        try:
            from shapely import wkt

            geom = wkt.loads(geom)
        except Exception:
            geom = None

    u_lat, u_lon = float(G.nodes[u]["y"]), float(G.nodes[u]["x"])
    v_lat, v_lon = float(G.nodes[v]["y"]), float(G.nodes[v]["x"])

    if geom is not None:
        if getattr(geom, "geom_type", "") == "MultiLineString":
            raw_coords = [pt for part in geom.geoms for pt in part.coords]
        else:
            raw_coords = list(getattr(geom, "coords", []))
        if len(raw_coords) >= 2:
            coords = [(float(y), float(x)) for x, y in raw_coords]
            if u != v:
                d_start_u = math.hypot(coords[0][0] - u_lat, coords[0][1] - u_lon)
                d_start_v = math.hypot(coords[0][0] - v_lat, coords[0][1] - v_lon)
                d_end_u = math.hypot(coords[-1][0] - u_lat, coords[-1][1] - u_lon)
                d_end_v = math.hypot(coords[-1][0] - v_lat, coords[-1][1] - v_lon)
                if d_start_v + d_end_u < d_start_u + d_end_v:
                    coords.reverse()
            return coords

    dist_deg = math.hypot(v_lat - u_lat, v_lon - u_lon)
    # 単一通行止め区間のフォールバック表示時のみ、極端に長い直線（>200m）を短い区間（約60m）に抑える
    if truncate_long_straight and dist_deg > 0.002:
        ratio = 0.0006 / dist_deg
        return [
            (u_lat, u_lon),
            (u_lat + (v_lat - u_lat) * ratio, u_lon + (v_lon - u_lon) * ratio),
        ]
    return [
        (u_lat, u_lon),
        (v_lat, v_lon),
    ]


def _path_coords(G: nx.MultiDiGraph, node_path: list[int]) -> list[tuple[float, float]]:
    """ノード列から各エッジのジオメトリを連結した (lat, lon) リストを生成する."""
    if len(node_path) < 2:
        return []
    coords: list[tuple[float, float]] = []
    for u, v in zip(node_path[:-1], node_path[1:]):
        edges = G.get_edge_data(u, v)
        if not edges:
            continue
        best_key = min(
            edges.keys(),
            key=lambda k: (
                float(edges[k].get("length", float("inf"))),
                0 if edges[k].get("geometry") is not None else 1,
            ),
        )
        seg = _edge_coords(G, u, v, best_key, truncate_long_straight=False)
        for pt in seg:
            if not coords or coords[-1] != pt:
                coords.append(pt)
    return coords


def _edge_distance_meters(
    G: nx.MultiDiGraph, u: int, v: int, key: int, ref_lat: float, ref_lon: float
) -> float:
    """施設座標(ref_lat, ref_lon)からエッジまでの平面近似距離(m)を算出する."""
    m_per_lat = 111320.0
    m_per_lon = 111320.0 * math.cos(math.radians(ref_lat))

    data = G.get_edge_data(u, v, key) or {}
    geom = data.get("geometry")
    if geom is not None:
        pts = [
            ((float(x) - ref_lon) * m_per_lon, (float(y) - ref_lat) * m_per_lat)
            for x, y in geom.coords
        ]
    else:
        pts = [
            (
                (float(G.nodes[u]["x"]) - ref_lon) * m_per_lon,
                (float(G.nodes[u]["y"]) - ref_lat) * m_per_lat,
            ),
            (
                (float(G.nodes[v]["x"]) - ref_lon) * m_per_lon,
                (float(G.nodes[v]["y"]) - ref_lat) * m_per_lat,
            ),
        ]
    if len(pts) < 2:
        return float("inf")
    return float(Point(0.0, 0.0).distance(LineString(pts)))


def _find_nearby_edges(
    G: nx.MultiDiGraph, lat: float, lon: float, radius_m: float = 50.0
) -> set[tuple[int, int, int]]:
    """指定座標から半径radius_m以内の全エッジ（双方向・並行重複含む）を特定する."""
    # 高速化のため約500m四方以内のノードに接続するエッジのみ走査
    deg_buf = 0.005
    cand_nodes = {
        n
        for n, d in G.nodes(data=True)
        if abs(float(d["y"]) - lat) <= deg_buf and abs(float(d["x"]) - lon) <= deg_buf
    }

    target_edges: set[tuple[int, int, int]] = set()
    checked_pairs: set[tuple[int, int, int]] = set()
    for n in cand_nodes:
        for eu, ev, ek in list(G.out_edges(n, keys=True)) + list(
            G.in_edges(n, keys=True)
        ):
            if (eu, ev, ek) in checked_pairs:
                continue
            checked_pairs.add((eu, ev, ek))
            if _edge_distance_meters(G, eu, ev, ek, lat, lon) <= radius_m:
                # 双方向・並行重複エッジすべてを対象に追加
                for k_fwd in (G.get_edge_data(eu, ev) or {}).keys():
                    target_edges.add((eu, ev, int(k_fwd)))
                for k_rev in (G.get_edge_data(ev, eu) or {}).keys():
                    target_edges.add((ev, eu, int(k_rev)))
    return target_edges


def _pick_approach_node(
    G_undir: nx.Graph,
    G_detour: nx.MultiDiGraph,
    anchor: int,
    other_anchor: int,
    min_m: float = 200.0,
    max_m: float = 500.0,
) -> int | None:
    """通行止め区間の少し手前（200m〜500m）にある非孤立ネットワークノードを探す."""
    dist_anchor = nx.single_source_dijkstra_path_length(
        G_undir, anchor, cutoff=1000.0, weight="length"
    )
    dist_other = nx.single_source_dijkstra_path_length(
        G_undir, other_anchor, cutoff=1000.0, weight="length"
    )

    # 通行止め区間を挟んでanchor側（anchorに近い側）にあり、G_detour上で非孤立のノードを抽出
    candidates: list[tuple[float, int]] = []
    for node, d in dist_anchor.items():
        if node in (anchor, other_anchor):
            continue
        if G_detour.out_degree(node) == 0 or G_detour.in_degree(node) == 0:
            continue
        d_other = dist_other.get(node, float("inf"))
        if d <= d_other and min_m <= d <= max_m:
            candidates.append((d, node))

    if candidates:
        candidates.sort()
        return candidates[0][1]

    # 200〜500mに見つからない場合は100〜1000mで最も200mに近い側を採用
    fallback: list[tuple[float, int]] = []
    for node, d in dist_anchor.items():
        if node in (anchor, other_anchor):
            continue
        if G_detour.out_degree(node) == 0 or G_detour.in_degree(node) == 0:
            continue
        d_other = dist_other.get(node, float("inf"))
        if d <= d_other and d >= 80.0:
            fallback.append((abs(d - 300.0), node))

    if fallback:
        fallback.sort()
        return fallback[0][1]
    return None


def simulate_detour(G: nx.MultiDiGraph, lat: float, lon: float) -> DetourResult:
    """指定座標の最寄りエッジ1本（および逆方向ペアエッジのみ）を通行止めとして迂回シミュレーションを行う."""
    nearest = _fast_nearest_edges(G, [float(lon)], [float(lat)])[0]
    u, v, key = int(nearest[0]), int(nearest[1]), int(nearest[2])

    # (1) 最寄りエッジ1本 (u, v, key) とその逆方向ペアエッジ (v, u, *) のみを削除対象にする
    #     並走する別道路（例: 尾道大橋に対する新尾道大橋）を巻き込まないため
    edges_to_remove: set[tuple[int, int, int]] = {(u, v, key)}
    rev_edges = G.get_edge_data(v, u) or {}
    if rev_edges:
        # 同じ距離・形状に最も近い逆方向エッジ1本（またはu-v間の逆向きペア）を対象にする
        fwd_len = float((G.get_edge_data(u, v, key) or {}).get("length", 0.0))
        best_rev_key = min(
            rev_edges.keys(),
            key=lambda rk: abs(float(rev_edges[rk].get("length", 0.0)) - fwd_len),
        )
        edges_to_remove.add((v, u, int(best_rev_key)))

    closed_segments: list[list[tuple[float, float]]] = []
    seen_undir: set[tuple[int, int, int]] = set()
    G_detour = G.copy()
    for eu, ev, ek in edges_to_remove:
        if G_detour.has_edge(eu, ev, key=ek):
            pair_id = (min(eu, ev), max(eu, ev), ek)
            if pair_id not in seen_undir:
                seen_undir.add(pair_id)
                closed_segments.append(_edge_coords(G, eu, ev, ek))
            G_detour.remove_edge(eu, ev, key=ek)

    # (2) メインエッジ両端ノードの孤立判定と (3) 孤立時の手前/先ノード(200〜500m)探索
    src: int | None = u
    dst: int | None = v
    u_isolated = G_detour.out_degree(u) == 0 or G_detour.in_degree(u) == 0
    v_isolated = G_detour.out_degree(v) == 0 or G_detour.in_degree(v) == 0

    if u == v or u_isolated or v_isolated:
        G_undir = G.to_undirected()
        if u_isolated or u == v:
            src = _pick_approach_node(G_undir, G_detour, u, v)
        if v_isolated or u == v:
            dst = _pick_approach_node(G_undir, G_detour, v, u)

    if src is None or dst is None or src == dst:
        edge_len_m = float((G.get_edge_data(u, v, key) or {}).get("length", 0.0))
        return DetourResult(
            u=u,
            v=v,
            key=key,
            normal_distance_km=round(edge_len_m / 1000.0, 2),
            detour_distance_km=None,
            extra_distance_km=ISOLATED_PENALTY_KM,
            is_isolated=True,
            closed_segments_coords=closed_segments,
            detour_route_coords=[],
            message="孤立（迂回路なし）: 通行止め区間の代替ルートが見つかりません。",
        )

    # 通常時の最短距離（src -> dst）
    try:
        normal_m = float(nx.shortest_path_length(G, source=src, target=dst, weight="length"))
    except nx.NetworkXNoPath:
        normal_m = float((G.get_edge_data(u, v, key) or {}).get("length", 0.0))
    normal_km = normal_m / 1000.0

    # 通行止め後の最短経路再計算（G_detour上で src -> dst）
    try:
        detour_nodes = nx.shortest_path(
            G_detour, source=src, target=dst, weight="length"
        )
        detour_m = 0.0
        for a, b in zip(detour_nodes[:-1], detour_nodes[1:]):
            edges = G_detour.get_edge_data(a, b)
            detour_m += min(float(d.get("length", 0.0)) for d in edges.values())
        detour_km = detour_m / 1000.0
        extra_km = max(0.0, detour_km - normal_km)
        detour_coords = _path_coords(G_detour, detour_nodes)
        return DetourResult(
            u=u,
            v=v,
            key=key,
            normal_distance_km=round(normal_km, 2),
            detour_distance_km=round(detour_km, 2),
            extra_distance_km=round(extra_km, 2),
            is_isolated=False,
            closed_segments_coords=closed_segments,
            detour_route_coords=detour_coords,
            message="迂回ルートを算出しました。",
        )
    except nx.NetworkXNoPath:
        return DetourResult(
            u=u,
            v=v,
            key=key,
            normal_distance_km=round(normal_km, 2),
            detour_distance_km=None,
            extra_distance_km=ISOLATED_PENALTY_KM,
            is_isolated=True,
            closed_segments_coords=closed_segments,
            detour_route_coords=[],
            message="孤立（迂回路なし）: 代替ルートが見つかりません。",
        )


@dataclass
class HospitalDelayResult:
    """最寄り医療機関への搬送遅延インパクト結果."""

    nearest_hospital_name: str
    nearest_hospital_lat: float | None
    nearest_hospital_lon: float | None
    normal_hospital_km: float | None
    detour_hospital_km: float | None
    hospital_delay_km: float | None
    hospital_isolated: bool
    message: str


def _nearest_target_distance_m(
    G: nx.MultiDiGraph,
    start_node: int,
    target_nodes: set[int],
    max_dist_m: float = 50000.0,
) -> tuple[float | None, int | None]:
    """早期終了付きDijkstraでstart_nodeから最寄りのtarget_nodeまでの距離(m)を返す."""
    if start_node in target_nodes:
        return 0.0, start_node
    pq: list[tuple[float, int]] = [(0.0, start_node)]
    visited: dict[int, float] = {}
    while pq:
        d, curr = heapq.heappop(pq)
        if curr in visited:
            continue
        visited[curr] = d
        if curr in target_nodes:
            return d, curr
        if d > max_dist_m:
            break
        for _, nxt, edata in G.out_edges(curr, data=True):
            if nxt in visited:
                continue
            w = float(edata.get("length", 0.0))
            nd = d + w
            if nd <= max_dist_m:
                heapq.heappush(pq, (nd, nxt))
    return None, None


def compute_hospital_delay(
    G: nx.MultiDiGraph,
    detour_result: DetourResult | None,
    facilities_df: pd.DataFrame,
) -> HospitalDelayResult:
    """対象施設が通行止めになった際の最寄り病院への搬送遅延インパクトを算出する."""
    if detour_result is None or facilities_df is None or facilities_df.empty:
        return HospitalDelayResult(
            nearest_hospital_name="-",
            nearest_hospital_lat=None,
            nearest_hospital_lon=None,
            normal_hospital_km=None,
            detour_hospital_km=None,
            hospital_delay_km=None,
            hospital_isolated=False,
            message="医療機関データがありません。",
        )

    med_df = facilities_df[facilities_df["facility_type"] == "医療機関"]
    if med_df.empty:
        return HospitalDelayResult(
            nearest_hospital_name="-",
            nearest_hospital_lat=None,
            nearest_hospital_lon=None,
            normal_hospital_km=None,
            detour_hospital_km=None,
            hospital_delay_km=None,
            hospital_isolated=False,
            message="対象エリア内に医療機関が見つかりませんでした。",
        )

    h_lons = med_df["lon"].astype(float).tolist()
    h_lats = med_df["lat"].astype(float).tolist()
    h_names = med_df["name"].astype(str).tolist()
    h_nodes = _fast_nearest_nodes(G, h_lons, h_lats)

    node_to_info: dict[int, tuple[str, float, float]] = {}
    target_nodes: set[int] = set()
    for n, name, h_lat, h_lon in zip(h_nodes, h_names, h_lats, h_lons):
        nid = int(n)
        target_nodes.add(nid)
        node_to_info.setdefault(nid, (name, float(h_lat), float(h_lon)))

    u, v, key = detour_result.u, detour_result.v, detour_result.key

    # 対象エッジ削除後グラフを作成
    G_detour = G.copy()
    edges_to_remove: set[tuple[int, int, int]] = {(u, v, key)}
    rev_edges = G_detour.get_edge_data(v, u) or {}
    if rev_edges:
        fwd_len = float((G_detour.get_edge_data(u, v, key) or {}).get("length", 0.0))
        best_rev_key = min(
            rev_edges.keys(),
            key=lambda rk: abs(float(rev_edges[rk].get("length", 0.0)) - fwd_len),
        )
        edges_to_remove.add((v, u, int(best_rev_key)))
    for eu, ev, ek in edges_to_remove:
        if G_detour.has_edge(eu, ev, key=ek):
            G_detour.remove_edge(eu, ev, key=ek)

    # u側・v側のうち通行止めによる搬送遅延インパクトが大きい側を採用
    best_delay = -1.0
    best_res: HospitalDelayResult | None = None

    for node in (u, v):
        d_norm, h_node_norm = _nearest_target_distance_m(G, node, target_nodes)
        if d_norm is None:
            continue
        h_name, h_lat, h_lon = node_to_info.get(
            h_node_norm, ("最寄り医療機関", None, None)
        )
        d_det, _ = _nearest_target_distance_m(G_detour, node, target_nodes)

        if d_det is None:
            return HospitalDelayResult(
                nearest_hospital_name=h_name,
                nearest_hospital_lat=h_lat,
                nearest_hospital_lon=h_lon,
                normal_hospital_km=round(d_norm / 1000.0, 2),
                detour_hospital_km=None,
                hospital_delay_km=None,
                hospital_isolated=True,
                message=f"{h_name} への到達経路が遮断（孤立）されます。",
            )

        delay_m = max(0.0, d_det - d_norm)
        if delay_m > best_delay:
            best_delay = delay_m
            best_res = HospitalDelayResult(
                nearest_hospital_name=h_name,
                nearest_hospital_lat=h_lat,
                nearest_hospital_lon=h_lon,
                normal_hospital_km=round(d_norm / 1000.0, 2),
                detour_hospital_km=round(d_det / 1000.0, 2),
                hospital_delay_km=round(delay_m / 1000.0, 2),
                hospital_isolated=False,
                message=f"最寄り医療機関（{h_name}）まで +{delay_m / 1000.0:.2f} km の搬送遅延が発生します。",
            )

    if best_res is not None:
        return best_res

    return HospitalDelayResult(
        nearest_hospital_name="-",
        nearest_hospital_lat=None,
        nearest_hospital_lon=None,
        normal_hospital_km=None,
        detour_hospital_km=None,
        hospital_delay_km=None,
        hospital_isolated=False,
        message="医療機関への到達経路を算出できませんでした。",
    )


def batch_simulate_detours(
    G: nx.MultiDiGraph,
    df: pd.DataFrame,
    facilities_df: pd.DataFrame | None = None,
    municipality: str | None = None,
    cache_dir: str | Path = ".cache/osm",
) -> pd.DataFrame:
    """全施設について迂回増加距離・孤立フラグ・病院搬送遅延を一括高速計算しCSVキャッシュする."""
    cols = [
        "id",
        "normal_distance_km",
        "detour_distance_km",
        "extra_distance_km",
        "is_isolated",
        "hospital_delay_km",
        "hospital_isolated",
    ]
    if df.empty:
        return pd.DataFrame(columns=cols)

    detours_csv: Path | None = None
    if municipality and not G.graph.get("is_fallback", False):
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        safe_name = _sanitize_cache_name(municipality)
        detours_csv = cache_path / f"{safe_name}_detours.csv"
        if detours_csv.exists():
            try:
                cached = pd.read_csv(detours_csv, encoding="utf-8-sig")
                if len(cached) == len(df) and set(cols).issubset(cached.columns):
                    return cached
            except Exception:
                pass

    lats = df["lat"].astype(float).tolist()
    lons = df["lon"].astype(float).tolist()
    ids = df["id"].tolist()

    # STRtreeで一括最寄りエッジ取得（約0.02秒）
    nearest_list = _fast_nearest_edges(G, lons, lats)
    G_undir = G.to_undirected()
    G_work = G.copy()

    # 医療機関ノードと通常時最短距離・次ホップノードを事前計算
    target_hospital_nodes: set[int] = set()
    norm_hosp_dist: dict[int, float] = {}
    next_hop_to_hosp: dict[int, int] = {}
    if facilities_df is not None and not facilities_df.empty:
        med_df = facilities_df[facilities_df["facility_type"] == "医療機関"]
        if not med_df.empty:
            h_nodes = _fast_nearest_nodes(
                G,
                med_df["lon"].astype(float).tolist(),
                med_df["lat"].astype(float).tolist(),
            )
            target_hospital_nodes = {int(n) for n in h_nodes if n in G}
            if target_hospital_nodes:
                G_rev = G.reverse(copy=False)
                norm_hosp_dist, hosp_paths = nx.multi_source_dijkstra(
                    G_rev, sources=target_hospital_nodes, weight="length"
                )
                for n, p in hosp_paths.items():
                    if len(p) >= 2:
                        next_hop_to_hosp[int(n)] = int(p[-2])

    records: list[dict] = []
    for fid, nearest in zip(ids, nearest_list):
        u, v, key = int(nearest[0]), int(nearest[1]), int(nearest[2])

        edges_to_remove: set[tuple[int, int, int]] = {(u, v, key)}
        rev_edges = G_work.get_edge_data(v, u) or {}
        if rev_edges:
            fwd_len = float((G_work.get_edge_data(u, v, key) or {}).get("length", 0.0))
            best_rev_key = min(
                rev_edges.keys(),
                key=lambda rk: abs(float(rev_edges[rk].get("length", 0.0)) - fwd_len),
            )
            edges_to_remove.add((v, u, int(best_rev_key)))

        saved_edges: list[tuple[int, int, int, dict]] = []
        for eu, ev, ek in edges_to_remove:
            if G_work.has_edge(eu, ev, key=ek):
                saved_edges.append((eu, ev, ek, dict(G_work[eu][ev][ek])))
                G_work.remove_edge(eu, ev, key=ek)

        try:
            # 病院搬送遅延の一括判定（削除エッジが最短経路の次ホップでない場合は即0.0km）
            h_delay_km: float | None = None
            h_isolated = False
            if target_hospital_nodes:
                max_delay_m = 0.0
                for node, other_node in ((u, v), (v, u)):
                    d_norm_h = norm_hosp_dist.get(node)
                    if d_norm_h is None:
                        continue
                    if (
                        next_hop_to_hosp.get(node) != other_node
                        and G_work.out_degree(node) > 0
                    ):
                        continue
                    d_det_h, _ = _nearest_target_distance_m(
                        G_work,
                        node,
                        target_hospital_nodes,
                        max_dist_m=min(15000.0, d_norm_h + 10000.0),
                    )
                    if d_det_h is None:
                        h_isolated = True
                        break
                    max_delay_m = max(max_delay_m, max(0.0, d_det_h - d_norm_h))
                if not h_isolated:
                    h_delay_km = round(max_delay_m / 1000.0, 2)

            src: int | None = u
            dst: int | None = v
            u_iso = G_work.out_degree(u) == 0 or G_work.in_degree(u) == 0
            v_iso = G_work.out_degree(v) == 0 or G_work.in_degree(v) == 0
            if u == v or u_iso or v_iso:
                if u_iso or u == v:
                    src = _pick_approach_node(G_undir, G_work, u, v)
                if v_iso or u == v:
                    dst = _pick_approach_node(G_undir, G_work, v, u)

            if src is None or dst is None or src == dst:
                edge_len_m = float(
                    next((d.get("length", 0.0) for _, _, _, d in saved_edges), 0.0)
                )
                records.append(
                    {
                        "id": fid,
                        "normal_distance_km": round(edge_len_m / 1000.0, 2),
                        "detour_distance_km": None,
                        "extra_distance_km": ISOLATED_PENALTY_KM,
                        "is_isolated": True,
                        "hospital_delay_km": h_delay_km,
                        "hospital_isolated": h_isolated,
                    }
                )
                continue

            if src == u and dst == v:
                normal_m = float(
                    next((d.get("length", 0.0) for _, _, _, d in saved_edges), 0.0)
                )
            else:
                try:
                    normal_m = float(
                        nx.shortest_path_length(
                            G, source=src, target=dst, weight="length"
                        )
                    )
                except nx.NetworkXNoPath:
                    normal_m = float(
                        next((d.get("length", 0.0) for _, _, _, d in saved_edges), 0.0)
                    )
            normal_km = normal_m / 1000.0

            try:
                detour_m = float(
                    nx.shortest_path_length(
                        G_work, source=src, target=dst, weight="length"
                    )
                )
                detour_km = detour_m / 1000.0
                extra_km = max(0.0, detour_km - normal_km)
                records.append(
                    {
                        "id": fid,
                        "normal_distance_km": round(normal_km, 2),
                        "detour_distance_km": round(detour_km, 2),
                        "extra_distance_km": round(extra_km, 2),
                        "is_isolated": False,
                        "hospital_delay_km": h_delay_km,
                        "hospital_isolated": h_isolated,
                    }
                )
            except nx.NetworkXNoPath:
                records.append(
                    {
                        "id": fid,
                        "normal_distance_km": round(normal_km, 2),
                        "detour_distance_km": None,
                        "extra_distance_km": ISOLATED_PENALTY_KM,
                        "is_isolated": True,
                        "hospital_delay_km": h_delay_km,
                        "hospital_isolated": h_isolated,
                    }
                )
        finally:
            for eu, ev, ek, attrs in saved_edges:
                G_work.add_edge(eu, ev, key=ek, **attrs)

    result_df = pd.DataFrame(records)
    if detours_csv is not None and not result_df.empty:
        try:
            result_df.to_csv(detours_csv, index=False, encoding="utf-8-sig")
        except Exception:
            pass
    return result_df


def estimate_repair_costs(df: pd.DataFrame) -> pd.Series:
    """面積(延長×幅員)と経過年数から概算修繕コスト(万円)を推計する."""
    lengths = df["length"].astype(float).clip(lower=0.0)
    widths = df["width"].astype(float)
    # 幅員が0または欠損の場合はデフォルト値6.0mで補完
    effective_widths = widths.where(widths > 0.0, 6.0)
    area = lengths * effective_widths

    # 施設種別ごとの基本単価係数（万円/m2）
    unit_cost = np.where(df["category"] == "トンネル", 20.0, 15.0)
    # 老朽度係数（築20年未満=1.0、以降1年ごとに+1.5%）
    ages = df["age"].astype(float).clip(lower=0.0)
    age_factor = 1.0 + np.maximum(0.0, ages - 20.0) * 0.015

    costs = np.maximum(100.0, area * unit_cost * age_factor)
    return pd.Series(np.round(costs, 0), index=df.index)


def build_integrated_ranking(
    df_area: pd.DataFrame,
    detour_df: pd.DataFrame,
    aging_weight: float = 0.5,
) -> pd.DataFrame:
    """構造物リスク・迂回インパクト・概算修繕コストを統合し総合優先度順に並べる."""
    scored = compute_risk_scores(df_area, aging_weight=aging_weight)
    if scored.empty:
        return scored

    merged = scored.merge(detour_df, on="id", how="left")
    merged["is_isolated"] = merged["is_isolated"].fillna(False).astype(bool)
    merged["extra_distance_km"] = pd.to_numeric(
        merged["extra_distance_km"], errors="coerce"
    )
    if "hospital_isolated" not in merged.columns:
        merged["hospital_isolated"] = False
    else:
        merged["hospital_isolated"] = merged["hospital_isolated"].fillna(False).astype(bool)
    if "hospital_delay_km" not in merged.columns:
        merged["hospital_delay_km"] = np.nan
    else:
        merged["hospital_delay_km"] = pd.to_numeric(
            merged["hospital_delay_km"], errors="coerce"
        )
    if "emergency_bonus" not in merged.columns:
        merged["emergency_bonus"] = 0.0
    if "emergency_route_rank" not in merged.columns:
        merged["emergency_route_rank"] = "対象外"
    if "emergency_route_display" not in merged.columns:
        merged["emergency_route_display"] = "-"

    combined_scores: list[float] = []
    combined_ranks: list[str] = []
    detour_impact_scores: list[float] = []
    for _, row in merged.iterrows():
        iso = bool(row["is_isolated"])
        extra = None if pd.isna(row["extra_distance_km"]) else float(row["extra_distance_km"])
        e_bonus = float(row.get("emergency_bonus", 0.0))
        h_iso = bool(row.get("hospital_isolated", False))
        h_delay = (
            None
            if pd.isna(row.get("hospital_delay_km"))
            else float(row["hospital_delay_km"])
        )
        c_score, c_rank = compute_combined_priority(
            float(row["risk_score"]),
            extra,
            iso,
            emergency_bonus=e_bonus,
            hospital_delay_km=h_delay,
            hospital_isolated=h_iso,
        )
        if iso:
            d_impact = 100.0
        elif extra is None:
            d_impact = 0.0
        else:
            d_impact = float(np.clip((extra / 20.0) * 100.0, 0.0, 100.0))

        combined_scores.append(c_score)
        combined_ranks.append(c_rank)
        detour_impact_scores.append(round(d_impact, 1))

    merged["detour_impact_score"] = detour_impact_scores
    merged["combined_score"] = combined_scores
    merged["combined_rank"] = combined_ranks
    merged["estimated_cost_man_yen"] = estimate_repair_costs(merged)

    merged = merged.sort_values(
        by=["combined_score", "risk_score", "age"], ascending=[False, False, False]
    ).reset_index(drop=True)
    merged["rank_order"] = np.arange(1, len(merged) + 1)
    return merged


@dataclass
class RepairPlanResult:
    """予算配分・修繕プラン最適化の結果."""

    selected_df: pd.DataFrame
    deferred_high_risk_df: pd.DataFrame
    deferred_other_df: pd.DataFrame
    total_cost_man_yen: float
    budget_man_yen: float
    resolved_detour_risk_pct: float
    high_risk_resolved_count: int
    high_risk_total_count: int
    emergency_high_risk_resolved_count: int
    emergency_high_risk_total_count: int


def optimize_repair_plan(
    df_ranked: pd.DataFrame, budget_man_yen: float
) -> RepairPlanResult:
    """総合優先度スコアが高い順に予算上限内で修繕対象を自動選定する（貪欲法）."""
    if df_ranked.empty:
        empty = df_ranked.copy()
        return RepairPlanResult(
            selected_df=empty,
            deferred_high_risk_df=empty,
            deferred_other_df=empty,
            total_cost_man_yen=0.0,
            budget_man_yen=float(budget_man_yen),
            resolved_detour_risk_pct=0.0,
            high_risk_resolved_count=0,
            high_risk_total_count=0,
            emergency_high_risk_resolved_count=0,
            emergency_high_risk_total_count=0,
        )

    ordered = df_ranked.sort_values(
        by=["combined_score", "risk_score"], ascending=[False, False]
    ).reset_index(drop=True)

    remaining = float(budget_man_yen)
    selected_indices: list[int] = []
    deferred_indices: list[int] = []

    for idx, row in ordered.iterrows():
        cost = float(row["estimated_cost_man_yen"])
        if cost <= remaining:
            selected_indices.append(idx)
            remaining -= cost
        else:
            deferred_indices.append(idx)

    selected_df = ordered.loc[selected_indices].copy().reset_index(drop=True)
    deferred_df = ordered.loc[deferred_indices].copy().reset_index(drop=True)

    high_risk_mask_all = ordered["combined_rank"].isin(["S", "A"])
    high_risk_total = int(high_risk_mask_all.sum())
    high_risk_resolved = (
        int(selected_df["combined_rank"].isin(["S", "A"]).sum())
        if not selected_df.empty
        else 0
    )

    if "emergency_route_rank" in ordered.columns:
        emg_hr_mask_all = high_risk_mask_all & (ordered["emergency_route_rank"] != "対象外")
        emg_hr_total = int(emg_hr_mask_all.sum())
        emg_hr_resolved = (
            int(
                (
                    selected_df["combined_rank"].isin(["S", "A"])
                    & (selected_df["emergency_route_rank"] != "対象外")
                ).sum()
            )
            if not selected_df.empty
            else 0
        )
    else:
        emg_hr_total = 0
        emg_hr_resolved = 0

    deferred_high_risk_df = deferred_df[
        deferred_df["combined_rank"].isin(["S", "A"])
    ].copy()
    deferred_other_df = deferred_df[
        ~deferred_df["combined_rank"].isin(["S", "A"])
    ].copy()

    # 修繕によって解消される地域の迂回・孤立リスクの割合（%）
    total_detour_risk = float(ordered["detour_impact_score"].sum())
    if total_detour_risk > 0:
        resolved_detour_risk = (
            float(selected_df["detour_impact_score"].sum())
            if not selected_df.empty
            else 0.0
        )
        resolved_pct = round((resolved_detour_risk / total_detour_risk) * 100.0, 1)
    else:
        total_combined = float(ordered["combined_score"].sum())
        resolved_combined = (
            float(selected_df["combined_score"].sum())
            if not selected_df.empty
            else 0.0
        )
        resolved_pct = (
            round((resolved_combined / total_combined) * 100.0, 1)
            if total_combined > 0
            else 0.0
        )

    total_cost = (
        float(selected_df["estimated_cost_man_yen"].sum())
        if not selected_df.empty
        else 0.0
    )

    return RepairPlanResult(
        selected_df=selected_df,
        deferred_high_risk_df=deferred_high_risk_df,
        deferred_other_df=deferred_other_df,
        total_cost_man_yen=round(total_cost, 0),
        budget_man_yen=float(budget_man_yen),
        resolved_detour_risk_pct=resolved_pct,
        high_risk_resolved_count=high_risk_resolved,
        high_risk_total_count=high_risk_total,
        emergency_high_risk_resolved_count=emg_hr_resolved,
        emergency_high_risk_total_count=emg_hr_total,
    )


QUADRANT_LABELS = [
    "最優先で維持・修繕",
    "防災・ネットワーク維持",
    "長寿命化・経過観察",
    "将来的な集約・縮退検討候補",
]

QUADRANT_COLORS = {
    "最優先で維持・修繕": "#dc2626",       # 赤
    "防災・ネットワーク維持": "#ea580c",   # 橙
    "長寿命化・経過観察": "#2563eb",       # 青
    "将来的な集約・縮退検討候補": "#64748b", # スレート灰
}


def classify_maintenance_quadrant(
    df_ranked: pd.DataFrame,
    score_threshold: float = 60.0,
    pop_threshold: float = 500.0,
) -> pd.DataFrame:
    """総合重要度と将来人口（2040年周辺人口・減少率）の2軸で4象限に仕分ける."""
    out = df_ranked.copy()
    if out.empty:
        out["maintenance_category"] = pd.Series(dtype=str)
        return out

    if "pop_2040_1km" not in out.columns:
        out["pop_2040_1km"] = 0.0
    if "pop_change_rate_1km" not in out.columns:
        out["pop_change_rate_1km"] = 0.0

    categories: list[str] = []
    for _, row in out.iterrows():
        c_score = float(row.get("combined_score", 0.0))
        is_iso = bool(row.get("is_isolated", False))
        emg_rank = str(row.get("emergency_route_rank", "対象外"))
        pop40 = float(row.get("pop_2040_1km", 0.0))
        pop_rate = float(row.get("pop_change_rate_1km", 0.0))

        high_importance = (
            c_score >= score_threshold or is_iso or emg_rank in ("1次", "2次")
        )
        pop_maintained = (pop40 >= pop_threshold) or (pop_rate >= -15.0 and pop40 > 0.0)

        if high_importance and pop_maintained:
            categories.append("最優先で維持・修繕")
        elif high_importance and not pop_maintained:
            categories.append("防災・ネットワーク維持")
        elif not high_importance and pop_maintained:
            categories.append("長寿命化・経過観察")
        else:
            categories.append("将来的な集約・縮退検討候補")

    out["maintenance_category"] = categories
    return out
