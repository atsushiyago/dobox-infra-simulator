"""広島県インフラ維持管理・修繕優先度可視化ダッシュボード（ステップ2：緊急輸送道路・防災拠点連携）."""

from __future__ import annotations

import math
from pathlib import Path

import folium
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
from streamlit_folium import st_folium

from data_loader import (
    BRIDGE_CSV_NAME,
    TUNNEL_CSV_NAME,
    MunicipalityBounds,
    filter_by_municipality,
    get_available_municipalities,
    get_municipality_bounds,
    load_infrastructure_data,
)
from emergency_loader import (
    EmergencyRouteSegment,
    classify_emergency_routes,
    load_disaster_facilities,
    load_emergency_routes,
)
from network_analyzer import (
    QUADRANT_COLORS,
    QUADRANT_LABELS,
    DetourResult,
    HospitalDelayResult,
    RepairPlanResult,
    batch_simulate_detours,
    build_integrated_ranking,
    classify_maintenance_quadrant,
    compute_hospital_delay,
    load_road_network,
    optimize_repair_plan,
    simulate_detour,
)
from population_loader import (
    attach_population_to_facilities,
    load_population_mesh,
)

st.set_page_config(
    page_title="広島県インフラ維持管理 優先度ダッシュボード",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.8rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        display: none;
    }
    .stTabs button[data-baseweb="tab"] {
        background-color: #f0f2f6;
        border: 1px solid #d0d7de;
        border-radius: 8px;
        padding: 10px 18px;
        font-weight: 500;
    }
    .stTabs button[data-baseweb="tab"]:hover {
        background-color: #e2e6ea;
    }
    .stTabs button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #1f4e78 !important;
        color: #ffffff !important;
        border-color: #1f4e78 !important;
        font-weight: 700;
    }
    /* セレクトボックスの入力カーソル・点滅バーを非表示にし、ポインターカーソル化 */
    div[data-testid="stSelectbox"] input {
        caret-color: transparent !important;
        cursor: pointer !important;
    }
    div[data-testid="stSelectbox"] div[data-baseweb="select"] {
        cursor: pointer !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="インフラデータを読み込み中...")
def cached_load_data() -> pd.DataFrame:
    return load_infrastructure_data(".")


@st.cache_data(show_spinner="緊急輸送道路ネットワークを読み込み中...")
def cached_load_emergency_routes() -> list[EmergencyRouteSegment]:
    return load_emergency_routes(".")


@st.cache_resource(show_spinner=False)
def cached_load_network(municipality: str, _df_area: pd.DataFrame, _bounds=None):
    return load_road_network(municipality=municipality, df_area=_df_area, bounds=_bounds)


@st.cache_data(show_spinner=False)
def cached_load_disaster_facilities(
    municipality: str, _bounds: MunicipalityBounds | None, _df_area: pd.DataFrame
) -> pd.DataFrame:
    return load_disaster_facilities(
        municipality=municipality, bounds=_bounds, df_area=_df_area
    )


@st.cache_data(show_spinner="将来推計人口メッシュを読み込み中...")
def cached_load_population_mesh(
    municipality: str, _bounds: MunicipalityBounds | None
):
    try:
        return load_population_mesh("500m_mesh_2018_34.shp", bounds=_bounds)
    except FileNotFoundError:
        return None


@st.cache_data(show_spinner="全施設の迂回インパクト・病院搬送遅延を一括計算中...")
def cached_precompute_detours(
    municipality: str,
    df_muni_all: pd.DataFrame,
    facilities_df: pd.DataFrame,
    _G,
) -> pd.DataFrame:
    return batch_simulate_detours(
        _G, df_muni_all, facilities_df=facilities_df, municipality=municipality
    )


def _score_to_color(score: float) -> str:
    """総合優先度スコア（0〜100）に応じて緑→黄→赤のカラーコードを返す."""
    if score >= 80:
        return "#dc2626"  # 赤
    if score >= 65:
        return "#ea580c"  # 橙赤
    if score >= 50:
        return "#eab308"  # 黄
    return "#16a34a"      # 緑


def _format_detour_text(row: pd.Series) -> str:
    if bool(row.get("is_isolated", False)):
        return "孤立（迂回路なし）"
    val = row.get("extra_distance_km")
    if pd.isna(val):
        return "算出不可"
    return f"+{float(val):.2f} km"


def _format_hospital_delay_text(row: pd.Series) -> str:
    if bool(row.get("hospital_isolated", False)):
        return "遮断（孤立）"
    val = row.get("hospital_delay_km")
    if pd.isna(val):
        return "-"
    return f"+{float(val):.2f} km"


def _add_base_layers(m: folium.Map) -> None:
    folium.TileLayer(
        tiles="https://cyberjapandata.gsi.go.jp/xyz/pale/{z}/{x}/{y}.png",
        attr="国土地理院",
        name="国土地理院（淡色地図）",
        overlay=False,
        control=True,
    ).add_to(m)


def _pop_to_fill_color(pop_2040: float) -> str:
    if pop_2040 >= 500:
        return "#dc2626"
    if pop_2040 >= 150:
        return "#f97316"
    if pop_2040 >= 30:
        return "#facc15"
    return "#fef08a"


def _add_population_mesh_layer(
    m: folium.Map, mesh_gdf, show_pop_mesh: bool
) -> None:
    """2040年将来推計人口メッシュの半透明ポリゴンレイヤーを追加する."""
    if not show_pop_mesh or mesh_gdf is None or mesh_gdf.empty:
        return
    non_zero = mesh_gdf[mesh_gdf["pop_2040"] > 0].copy()
    if non_zero.empty:
        return

    fg_pop = folium.FeatureGroup(
        name="将来人口メッシュ（2040年推計）", show=show_pop_mesh
    )
    folium.GeoJson(
        non_zero.__geo_interface__,
        style_function=lambda feat: {
            "fillColor": _pop_to_fill_color(
                float(feat["properties"].get("pop_2040", 0.0))
            ),
            "color": "#94a3b8",
            "weight": 0.4,
            "fillOpacity": 0.35,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["mesh_id", "pop_2020", "pop_2040", "pop_change_rate"],
            aliases=["メッシュID:", "2020年人口:", "2040年人口:", "増減率(%):"],
            localize=True,
        ),
    ).add_to(fg_pop)
    fg_pop.add_to(m)


def _segment_in_bounds(
    seg: EmergencyRouteSegment, bounds: MunicipalityBounds | None
) -> bool:
    if bounds is None:
        return True
    pad = 0.05
    for lat, lon in seg.coords_latlon:
        if (
            bounds.min_lat - pad <= lat <= bounds.max_lat + pad
            and bounds.min_lon - pad <= lon <= bounds.max_lon + pad
        ):
            return True
    return False


def _add_emergency_and_disaster_layers(
    m: folium.Map,
    routes: list[EmergencyRouteSegment],
    facilities_df: pd.DataFrame,
    bounds: MunicipalityBounds | None,
    show_emergency: bool,
    show_hospitals: bool,
    show_shelters: bool,
) -> None:
    """緊急輸送道路・医療機関・避難施設をレイヤーグループとして地図に追加する."""
    if routes:
        fg_routes = folium.FeatureGroup(
            name="緊急輸送道路（第1〜4次）", show=show_emergency
        )
        for seg in routes:
            if not _segment_in_bounds(seg, bounds):
                continue
            folium.PolyLine(
                locations=seg.coords_latlon,
                color=seg.color,
                weight=seg.weight,
                opacity=0.75,
                tooltip=f"{seg.rank_label}緊急輸送道路",
            ).add_to(fg_routes)
        fg_routes.add_to(m)

    if facilities_df is not None and not facilities_df.empty:
        med_df = facilities_df[facilities_df["facility_type"] == "医療機関"]
        if not med_df.empty:
            fg_hosp = folium.FeatureGroup(
                name="医療機関（病院・診療所）", show=show_hospitals
            )
            for _, frow in med_df.iterrows():
                folium.Marker(
                    location=[float(frow["lat"]), float(frow["lon"])],
                    tooltip=f"[医療機関] {frow['name']} ({frow['amenity']})",
                    icon=folium.Icon(color="red", icon="plus-sign"),
                ).add_to(fg_hosp)
            fg_hosp.add_to(m)

        shelter_df = facilities_df[facilities_df["facility_type"] == "避難施設"]
        if not shelter_df.empty:
            fg_shelter = folium.FeatureGroup(
                name="避難施設（学校・公民館）", show=show_shelters
            )
            for _, frow in shelter_df.iterrows():
                folium.Marker(
                    location=[float(frow["lat"]), float(frow["lon"])],
                    tooltip=f"[避難施設] {frow['name']} ({frow['amenity']})",
                    icon=folium.Icon(color="green", icon="home"),
                ).add_to(fg_shelter)
            fg_shelter.add_to(m)


def _build_map(
    df_ranked: pd.DataFrame,
    selected_row: pd.Series,
    detour: DetourResult | None,
    routes: list[EmergencyRouteSegment],
    facilities_df: pd.DataFrame,
    muni_bounds: MunicipalityBounds | None,
    show_emergency: bool,
    show_hospitals: bool,
    show_shelters: bool,
    mesh_gdf=None,
    show_pop_mesh: bool = False,
    hospital_delay: HospitalDelayResult | None = None,
) -> folium.Map:
    """施設プロット・通行止め区間・迂回ルート・緊急輸送道路・防災拠点・人口メッシュを描画する."""
    sel_lat = float(selected_row["lat"])
    sel_lon = float(selected_row["lon"])

    m = folium.Map(
        location=[sel_lat, sel_lon],
        zoom_start=14,
        tiles="OpenStreetMap",
        control_scale=True,
    )
    _add_base_layers(m)
    _add_population_mesh_layer(m, mesh_gdf, show_pop_mesh)

    # 緊急輸送道路・防災拠点レイヤー
    _add_emergency_and_disaster_layers(
        m,
        routes=routes,
        facilities_df=facilities_df,
        bounds=muni_bounds,
        show_emergency=show_emergency,
        show_hospitals=show_hospitals,
        show_shelters=show_shelters,
    )

    # 全施設を総合優先度スコアに応じて色分けプロット
    for _, row in df_ranked.iterrows():
        is_selected = row["id"] == selected_row["id"]
        score_val = float(row["combined_score"])
        color = _score_to_color(score_val)
        built_note = "（中央値補完）" if row["built_year_imputed"] else ""
        detour_str = _format_detour_text(row)
        emg_str = row.get("emergency_route_display", "-")
        tooltip_html = (
            f"<b>{row['name']}</b> ({row['category']})<br>"
            f"路線: {row['route_name']} ({row['road_type']}) / 緊急輸送道路: <b>{emg_str}</b><br>"
            f"築年数: {row['age']}年 [{row['built_year']}年{built_note}] / 延長: {row['length']:.1f}m<br>"
            f"迂回影響: <b>{detour_str}</b> / 病院遅延: <b>{_format_hospital_delay_text(row)}</b><br>"
            f"総合スコア: <b>{score_val:.1f}</b> ({row['combined_rank']}) / 概算コスト: {row['estimated_cost_man_yen']:,.0f}万円"
        )

        folium.CircleMarker(
            location=[float(row["lat"]), float(row["lon"])],
            radius=9 if is_selected else 6,
            color="#111827" if is_selected else color,
            weight=3 if is_selected else 1.5,
            fill=True,
            fill_color=color,
            fill_opacity=0.95 if is_selected else 0.8,
            tooltip=folium.Tooltip(tooltip_html),
        ).add_to(m)

    # 選択施設をマーカーで強調
    folium.Marker(
        location=[sel_lat, sel_lon],
        tooltip=f"選択中: {selected_row['name']}",
        icon=folium.Icon(color="black", icon="info-sign"),
    ).add_to(m)

    # 通行止め区間（白縁＋赤太線）と迂回ルート（青破線）
    if detour is not None:
        for seg_coords in detour.closed_segments_coords:
            if seg_coords:
                folium.PolyLine(
                    locations=seg_coords,
                    color="#ffffff",
                    weight=12,
                    opacity=0.75,
                ).add_to(m)
                folium.PolyLine(
                    locations=seg_coords,
                    color="#dc2626",
                    weight=8,
                    opacity=0.9,
                    tooltip=f"通行止め区間 (通常 {detour.normal_distance_km:.2f} km)",
                ).add_to(m)

        if detour.detour_route_coords:
            folium.PolyLine(
                locations=detour.detour_route_coords,
                color="#2563eb",
                weight=5,
                opacity=0.9,
                dash_array="10, 8",
                tooltip=f"迂回ルート ({detour.detour_distance_km:.2f} km)",
            ).add_to(m)

    # 表示範囲（選択施設 + 通行止め区間 + 迂回ルート + 最寄り病院）の自動調整
    bounds_coords: list[tuple[float, float]] = [(sel_lat, sel_lon)]
    if detour is not None:
        for seg in detour.closed_segments_coords:
            bounds_coords.extend(seg)
        if not detour.is_isolated and detour.detour_route_coords:
            bounds_coords.extend(detour.detour_route_coords)
    if (
        hospital_delay is not None
        and hospital_delay.nearest_hospital_lat is not None
        and hospital_delay.nearest_hospital_lon is not None
    ):
        bounds_coords.append(
            (
                float(hospital_delay.nearest_hospital_lat),
                float(hospital_delay.nearest_hospital_lon),
            )
        )

    # 広島県域（緯度33.5〜35.5, 経度131.5〜134.5）内の有効座標のみで表示範囲を算出
    valid_coords = [
        c for c in bounds_coords
        if 33.5 <= c[0] <= 35.5 and 131.5 <= c[1] <= 134.5
    ]
    if len(valid_coords) > 1:
        v_lats = [c[0] for c in valid_coords]
        v_lons = [c[1] for c in valid_coords]
        lat_span = max(v_lats) - min(v_lats)
        lon_span = max(v_lons) - min(v_lons)
        if 1e-4 <= lat_span <= 0.6 and 1e-4 <= lon_span <= 0.6:
            lat_pad = max(0.003, lat_span * 0.18)
            lon_pad = max(0.003, lon_span * 0.18)
            m.fit_bounds(
                [
                    [min(v_lats) - lat_pad, min(v_lons) - lon_pad],
                    [max(v_lats) + lat_pad, max(v_lons) + lon_pad],
                ]
            )

    folium.LayerControl().add_to(m)
    return m


def _build_plan_map(
    plan: RepairPlanResult,
    muni_bounds: MunicipalityBounds | None,
    df_ranked: pd.DataFrame,
    routes: list[EmergencyRouteSegment],
    facilities_df: pd.DataFrame,
    show_emergency: bool,
    show_hospitals: bool,
    show_shelters: bool,
    mesh_gdf=None,
    show_pop_mesh: bool = False,
) -> folium.Map:
    """予算配分シミュレーション結果（修繕対象=青 / 見送り高リスク=赤 / その他=灰）を描画する."""
    if muni_bounds is not None:
        center_lat, center_lon = muni_bounds.center_lat, muni_bounds.center_lon
        zoom_start = muni_bounds.zoom
    else:
        center_lat = float(df_ranked["lat"].mean())
        center_lon = float(df_ranked["lon"].mean())
        zoom_start = 11

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles="OpenStreetMap",
        control_scale=True,
    )
    _add_base_layers(m)
    _add_population_mesh_layer(m, mesh_gdf, show_pop_mesh)

    _add_emergency_and_disaster_layers(
        m,
        routes=routes,
        facilities_df=facilities_df,
        bounds=muni_bounds,
        show_emergency=show_emergency,
        show_hospitals=show_hospitals,
        show_shelters=show_shelters,
    )

    def _plot_group(
        sub_df: pd.DataFrame, color: str, label: str, radius: int, opacity: float
    ) -> None:
        for _, row in sub_df.iterrows():
            tooltip_html = (
                f"<b>[{label}] {row['name']}</b> ({row['category']})<br>"
                f"緊急輸送道路: {row.get('emergency_route_display', '-')} / "
                f"総合スコア: <b>{row['combined_score']:.1f}</b> ({row['combined_rank']})<br>"
                f"迂回影響: {_format_detour_text(row)} / 病院遅延: {_format_hospital_delay_text(row)}<br>"
                f"概算修繕コスト: <b>{row['estimated_cost_man_yen']:,.0f} 万円</b>"
            )
            folium.CircleMarker(
                location=[float(row["lat"]), float(row["lon"])],
                radius=radius,
                color=color,
                weight=2,
                fill=True,
                fill_color=color,
                fill_opacity=opacity,
                tooltip=folium.Tooltip(tooltip_html),
            ).add_to(m)

    _plot_group(plan.deferred_other_df, "#9ca3af", "見送り（低・中リスク）", 4, 0.5)
    _plot_group(plan.deferred_high_risk_df, "#dc2626", "予算不足で見送り（高リスク）", 7, 0.88)
    _plot_group(plan.selected_df, "#2563eb", "今回修繕対象", 7, 0.92)

    folium.LayerControl().add_to(m)
    return m


def _build_quadrant_map(
    df_quad: pd.DataFrame,
    muni_bounds: MunicipalityBounds | None,
    mesh_gdf=None,
    show_pop_mesh: bool = True,
) -> folium.Map:
    """4象限（維持・集約仕分け）分類ごとに色分けして全施設を地図表示する."""
    if muni_bounds is not None:
        center_lat, center_lon = muni_bounds.center_lat, muni_bounds.center_lon
        zoom_start = muni_bounds.zoom
    else:
        center_lat = float(df_quad["lat"].mean())
        center_lon = float(df_quad["lon"].mean())
        zoom_start = 11

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles="OpenStreetMap",
        control_scale=True,
    )
    _add_base_layers(m)
    _add_population_mesh_layer(m, mesh_gdf, show_pop_mesh)

    for _, row in df_quad.iterrows():
        cat = str(row["maintenance_category"])
        color = QUADRANT_COLORS.get(cat, "#64748b")
        tooltip_html = (
            f"<b>[{cat}] {row['name']}</b> ({row['category']})<br>"
            f"総合スコア: <b>{row['combined_score']:.1f}</b> / 迂回影響: {_format_detour_text(row)}<br>"
            f"2040年周辺人口(1km): <b>{row['pop_2040_1km']:,.0f}人</b> ({row['pop_change_rate_1km']:+.1f}%)<br>"
            f"概算維持費: <b>{row['estimated_cost_man_yen']:,.0f} 万円</b>"
        )
        folium.CircleMarker(
            location=[float(row["lat"]), float(row["lon"])],
            radius=6,
            color=color,
            weight=2,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            tooltip=folium.Tooltip(tooltip_html),
        ).add_to(m)

    folium.LayerControl().add_to(m)
    return m


def _find_clicked_facility_id(
    map_state: dict | None, df_ranked: pd.DataFrame
) -> str | None:
    """地図上でクリックされたオブジェクト座標から最も近い施設IDを特定する."""
    if not map_state:
        return None
    clicked = map_state.get("last_object_clicked")
    if not clicked or "lat" not in clicked or "lng" not in clicked:
        return None
    clat = float(clicked["lat"])
    clng = float(clicked["lng"])
    dists = (df_ranked["lat"] - clat) ** 2 + (df_ranked["lon"] - clng) ** 2
    min_idx = dists.idxmin()
    if float(dists.loc[min_idx]) < 0.003**2:
        return str(df_ranked.loc[min_idx, "id"])
    return None


def main() -> None:
    st.title("広島県インフラ維持管理・修繕優先度ダッシュボード")
    st.caption(
        "DoboXオープンデータ × OpenStreetMap × 緊急輸送道路・防災拠点（医療機関・避難施設）連携プロトタイプ"
    )

    try:
        df_all = cached_load_data()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info(
            f"プロジェクトルートに `{BRIDGE_CSV_NAME}` および `{TUNNEL_CSV_NAME}` を配置してから再読み込みしてください。"
        )
        st.stop()

    emergency_routes = cached_load_emergency_routes()
    municipalities = get_available_municipalities(df_all)
    if not municipalities:
        st.error("有効な市町データが読み込めませんでした。")
        st.stop()

    default_muni_idx = (
        municipalities.index("尾道市") if "尾道市" in municipalities else 0
    )

    # --- 左サイドバー ---
    with st.sidebar:
        # セレクトボックス内のinput要素をreadOnly化し、キーボード入力・削除（Backspace等）を完全防止
        components.html(
            """
            <script>
            (function() {
                const parentDoc = window.parent.document;
                if (!parentDoc) return;

                function lockSelectboxes() {
                    const inputs = parentDoc.querySelectorAll('div[data-testid="stSelectbox"] input');
                    inputs.forEach(input => {
                        if (!input.readOnly) {
                            input.readOnly = true;
                        }
                    });
                }

                lockSelectboxes();
                const observer = new MutationObserver(lockSelectboxes);
                if (parentDoc.body) {
                    observer.observe(parentDoc.body, { childList: true, subtree: true });
                }

                parentDoc.addEventListener('keydown', function(e) {
                    if (e.target && e.target.matches && e.target.matches('div[data-testid="stSelectbox"] input')) {
                        const allowed = ['ArrowUp', 'ArrowDown', 'Enter', 'Escape', 'Tab'];
                        if (!allowed.includes(e.key)) {
                            e.preventDefault();
                            e.stopPropagation();
                        }
                    }
                }, true);
            })();
            </script>
            """,
            height=0,
            width=0,
        )
        st.header("分析条件設定")
        selected_muni = st.selectbox(
            "対象市町",
            options=municipalities,
            index=default_muni_idx,
        )

        category_filter = st.radio(
            "施設種別フィルタ",
            options=["すべて", "道路橋", "トンネル"],
            index=0,
            horizontal=True,
        )

        only_emergency = st.checkbox(
            "緊急輸送道路上の施設のみ表示",
            value=False,
            help="第1次〜第4次緊急輸送道路から30m以内の施設のみに絞り込みます。",
        )

        st.divider()
        st.subheader("防災・緊急輸送・人口レイヤー")
        show_emergency = st.checkbox("緊急輸送道路（第1〜4次）", value=True)
        show_hospitals = st.checkbox("医療機関（病院・診療所）", value=True)
        show_shelters = st.checkbox("避難施設（学校・公民館）", value=False)
        show_pop_mesh = st.checkbox("将来人口メッシュ（2040年推計）", value=False)

        st.divider()
        st.subheader("優先度スコアの重み付け")
        aging_weight = st.slider(
            "老朽度重視 ↔ 路線重要度重視",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="1.0に近いほど経過年数（老朽度）を重視し、0.0に近いほど道路種別・延長（路線重要度）を重視します。",
        )
        st.caption(
            f"老朽度: **{int(round(aging_weight * 100))}%** ／ "
            f"路線重要度・延長: **{int(round((1 - aging_weight) * 100))}%**"
        )

    # --- データ抽出 & 緊急輸送道路判定 & 防災拠点 & 道路ネットワーク ---
    df_muni_raw = filter_by_municipality(df_all, municipality=selected_muni)
    df_muni_all = classify_emergency_routes(
        df_muni_raw, emergency_routes, threshold_m=30.0
    )

    target_cats = None if category_filter == "すべて" else [category_filter]
    df_area = df_muni_all.copy()
    if target_cats is not None:
        df_area = df_area[df_area["category"].isin(target_cats)].copy()
    if only_emergency:
        df_area = df_area[df_area["emergency_route_rank"] != "対象外"].copy()

    if df_area.empty:
        st.warning("指定された条件に該当する施設が見つかりませんでした。フィルタを調整してください。")
        st.stop()

    muni_bounds = get_municipality_bounds(df_all, selected_muni, margin_deg=0.01)

    safe_muni = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in selected_muni)
    needs_initial_fetch = (
        not Path(f".cache/osm/{safe_muni}_drive.pkl").exists()
        and not Path(f".cache/osm/{safe_muni}_drive.graphml").exists()
    )

    loading_placeholder = st.empty()
    if needs_initial_fetch:
        loading_placeholder.markdown(
            f"""
            <style>
                @keyframes spin {{
                    0% {{ transform: rotate(0deg); }}
                    100% {{ transform: rotate(360deg); }}
                }}
            </style>
            <div style="
                position: fixed;
                inset: 0;
                z-index: 999999;
                display: flex;
                align-items: center;
                justify-content: center;
                background: rgba(15, 23, 42, 0.35);
                backdrop-filter: blur(2px);
                padding: 24px;
            ">
                <div style="
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    padding: 40px 48px;
                    width: 100%;
                    max-width: 680px;
                    background: #ffffff;
                    border: 1px solid #cbd5e1;
                    border-radius: 16px;
                    box-shadow: 0 20px 50px rgba(15, 23, 42, 0.22);
                    text-align: center;
                ">
                    <div style="
                        width: 52px;
                        height: 52px;
                        border: 6px solid #e2e8f0;
                        border-top: 6px solid #2563eb;
                        border-radius: 50%;
                        animation: spin 1s linear infinite;
                        margin-bottom: 20px;
                    "></div>
                    <h3 style="
                        margin: 0 0 12px 0;
                        color: #0f172a;
                        font-size: 1.25rem;
                        white-space: nowrap;
                        word-break: keep-all;
                    ">
                        {selected_muni} の道路ネットワーク・防災拠点を取得中...
                    </h3>
                    <p style="
                        margin: 0;
                        color: #475569;
                        font-size: 0.95rem;
                        line-height: 1.8;
                        white-space: nowrap;
                        word-break: keep-all;
                    ">
                        OpenStreetMap から道路網および医療機関・避難施設の座標を取得しています。<br>
                        （2回目以降はキャッシュから高速表示されます）
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    G = None
    network_error: str | None = None
    detour_df = pd.DataFrame(
        columns=[
            "id",
            "normal_distance_km",
            "detour_distance_km",
            "extra_distance_km",
            "is_isolated",
            "hospital_delay_km",
            "hospital_isolated",
        ]
    )
    try:
        # Overpass APIの同時接続制限(429)を防ぐため順次取得する
        G = cached_load_network(selected_muni, df_muni_all, muni_bounds)
        facilities_df = cached_load_disaster_facilities(
            selected_muni, muni_bounds, df_muni_all
        )
        detour_df = cached_precompute_detours(
            selected_muni, df_muni_all, facilities_df, G
        )
    except Exception as exc:
        network_error = str(exc)
        if "facilities_df" not in locals():
            facilities_df = cached_load_disaster_facilities(
                selected_muni, muni_bounds, df_muni_all
            )
    finally:
        loading_placeholder.empty()

    df_ranked = build_integrated_ranking(df_area, detour_df, aging_weight=aging_weight)
    mesh_gdf = cached_load_population_mesh(selected_muni, muni_bounds)
    df_ranked = attach_population_to_facilities(df_ranked, mesh_gdf, radius_m=1000.0)

    valid_ids = df_ranked["id"].tolist()
    if (
        "selected_facility_id" not in st.session_state
        or st.session_state["selected_facility_id"] not in valid_ids
    ):
        st.session_state["selected_facility_id"] = valid_ids[0]

    # サイドバーのドロップダウンと選択施設IDを双方向同期
    st.session_state["sidebar_facility_select"] = st.session_state[
        "selected_facility_id"
    ]

    def _on_sidebar_facility_change() -> None:
        st.session_state["selected_facility_id"] = st.session_state[
            "sidebar_facility_select"
        ]

    with st.sidebar:
        st.divider()
        id_to_label = {
            row["id"]: (
                f"#{row['rank_order']} {row['name']} "
                f"[{row['emergency_route_display']}] (総合{row['combined_score']:.1f}点)"
            )
            for _, row in df_ranked.iterrows()
        }
        st.selectbox(
            "個別シミュレーション対象施設",
            options=valid_ids,
            key="sidebar_facility_select",
            on_change=_on_sidebar_facility_change,
            format_func=lambda fid: id_to_label.get(fid, fid),
        )

    selected_id = st.session_state["selected_facility_id"]
    selected_row = df_ranked[df_ranked["id"] == selected_id].iloc[0]

    if network_error:
        st.warning(f"道路ネットワーク解析でエラーが発生しました: {network_error}")

    # --- メインタブ切り替え ---
    tab1, tab2, tab3 = st.tabs(
        [
            "1. 個別迂回シミュレーション",
            "2. 修繕予算最適化",
            "3. 将来人口・集約仕分け",
        ]
    )

    # ==========================================================
    # タブ1: 個別施設シミュレーション
    # ==========================================================
    with tab1:
        detour_result: DetourResult | None = None
        hospital_delay: HospitalDelayResult | None = None
        if G is not None:
            detour_result = simulate_detour(
                G, lat=float(selected_row["lat"]), lon=float(selected_row["lon"])
            )
            hospital_delay = compute_hospital_delay(G, detour_result, facilities_df)

        total_count = len(df_ranked)
        bridge_count = int((df_ranked["category"] == "道路橋").sum())
        tunnel_count = int((df_ranked["category"] == "トンネル").sum())
        aged50_count = int((df_ranked["age"] >= 50).sum())
        aged50_ratio = (aged50_count / total_count * 100.0) if total_count > 0 else 0.0

        if detour_result is None:
            detour_kpi = "算出不可"
            detour_delta = "ネットワーク未取得"
        elif detour_result.is_isolated:
            detour_kpi = "迂回路なし"
            detour_delta = "孤立（最大ペナルティ）"
        else:
            detour_kpi = f"+{detour_result.extra_distance_km:.2f} km"
            detour_delta = (
                f"通常 {detour_result.normal_distance_km:.2f}km → "
                f"迂回 {detour_result.detour_distance_km:.2f}km"
            )

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric(
            label=f"{selected_muni} 対象施設数",
            value=f"{total_count:,} 施設",
            delta=f"橋梁 {bridge_count}・トンネル {tunnel_count}",
            delta_color="off",
            delta_arrow="off",
        )
        kpi2.metric(
            label="築50年以上の施設数",
            value=f"{aged50_count:,} 施設",
            delta=f"全体の {aged50_ratio:.1f}%",
            delta_color="inverse",
            delta_arrow="off",
        )
        kpi3.metric(
            label="選択施設の迂回増加距離",
            value=detour_kpi,
            delta=detour_delta,
            delta_color=(
                "inverse" if (detour_result and detour_result.is_isolated) else "off"
            ),
            delta_arrow="off",
        )
        kpi4.metric(
            label="総合優先度ランク（防災込み）",
            value=f"ランク {selected_row['combined_rank']} ({selected_row['combined_score']:.1f}点)",
            delta=f"{selected_row['rank_order']}位/{total_count}（緊急輸送: {selected_row['emergency_route_display']}）",
            delta_color="off",
            delta_arrow="off",
        )

        # 選択中施設の詳細パネル（緊急輸送道路 & 病院搬送遅延インパクト）
        st.subheader(f"{selected_muni} インフラ総合優先度・通行止め迂回シミュレーション")
        with st.container(border=True):
            st.markdown(
                f"**選択施設:** `{selected_row['name']}`（{selected_row['category']}）｜ "
                f"路線: **{selected_row['route_name']}**（{selected_row['road_type']}）｜ "
                f"緊急輸送道路: **{selected_row['emergency_route_display']}** ｜ "
                f"築年数: **{selected_row['age']}年** ｜ "
                f"概算修繕費: **{selected_row['estimated_cost_man_yen']:,.0f} 万円**"
            )
            if hospital_delay is None or (
                hospital_delay.normal_hospital_km is None
                and not hospital_delay.hospital_isolated
            ):
                st.markdown("**最寄り病院搬送影響:** 算出データなし（市町内医療機関未取得）")
            elif hospital_delay.hospital_isolated:
                st.markdown(
                    f"**最寄り病院搬送影響:** **{hospital_delay.nearest_hospital_name}** への経路遮断（孤立）"
                )
            else:
                st.markdown(
                    f"**最寄り病院搬送影響（{hospital_delay.nearest_hospital_name}）:** "
                    f"通常 {hospital_delay.normal_hospital_km:.2f} km → "
                    f"迂回 {hospital_delay.detour_hospital_km:.2f} km "
                    f"(**+{hospital_delay.hospital_delay_km:.2f} km 遅延**)"
                )

        st.markdown(
            "凡例: <span style='color:#16a34a;'>●</span>低(&lt;50) "
            "<span style='color:#eab308;'>●</span>中(50-64) "
            "<span style='color:#ea580c;'>●</span>高(65-79) "
            "<span style='color:#dc2626;'>●</span>極高(80+) ｜ "
            "<span style='color:#dc2626;font-weight:bold;'>━━ 通行止め</span> ｜ "
            "<span style='color:#2563eb;font-weight:bold;'>┅┅ 迂回ルート</span> ｜ "
            "<span style='color:#070690;font-weight:bold;'>━ 第1次</span> "
            "<span style='color:#1C26F9;font-weight:bold;'>━ 第2次</span> "
            "<span style='color:#10B981;font-weight:bold;'>━ 第3次</span> "
            "<span style='color:#F7811D;font-weight:bold;'>━ 第4次緊急輸送道路</span>",
            unsafe_allow_html=True,
        )

        fmap = _build_map(
            df_ranked=df_ranked,
            selected_row=selected_row,
            detour=detour_result,
            routes=emergency_routes,
            facilities_df=facilities_df,
            muni_bounds=muni_bounds,
            show_emergency=show_emergency,
            show_hospitals=show_hospitals,
            show_shelters=show_shelters,
            mesh_gdf=mesh_gdf,
            show_pop_mesh=show_pop_mesh,
            hospital_delay=hospital_delay,
        )
        map_state = st_folium(
            fmap,
            width=None,
            height=460,
            returned_objects=["last_object_clicked"],
            key=f"map_{selected_muni}_{category_filter}_{only_emergency}_{selected_id}_{show_emergency}_{show_hospitals}_{show_shelters}_{show_pop_mesh}",
        )

        clicked_id = _find_clicked_facility_id(map_state, df_ranked)
        if clicked_id and clicked_id != st.session_state["selected_facility_id"]:
            st.session_state["selected_facility_id"] = clicked_id
            st.rerun()

        st.subheader("総合優先度ランキング（チェックで地図・迂回シミュレーションと連動）")
        display_df = df_ranked.copy()
        display_df["選択"] = display_df["id"] == selected_id
        display_df["迂回増加"] = display_df.apply(_format_detour_text, axis=1)
        display_df["病院搬送遅延"] = display_df.apply(_format_hospital_delay_text, axis=1)
        display_table = display_df[
            [
                "選択",
                "rank_order",
                "combined_rank",
                "combined_score",
                "emergency_route_display",
                "risk_score",
                "迂回増加",
                "病院搬送遅延",
                "estimated_cost_man_yen",
                "name",
                "category",
                "route_name",
                "road_type",
                "age",
                "length",
            ]
        ].rename(
            columns={
                "rank_order": "順位",
                "combined_rank": "総合ランク",
                "combined_score": "総合スコア",
                "emergency_route_display": "緊急輸送道路",
                "risk_score": "構造リスク",
                "estimated_cost_man_yen": "概算コスト(万円)",
                "name": "施設名",
                "category": "種別",
                "route_name": "路線名",
                "road_type": "道路種別",
                "age": "経過年数",
                "length": "延長(m)",
            }
        )

        # 現在選択中の行を薄くハイライト表示
        styled_table = display_table.style.apply(
            lambda row: [
                "background-color: #dbeafe; font-weight: 600;" if bool(row["選択"]) else ""
                for _ in row
            ],
            axis=1,
        )

        disabled_cols = [col for col in display_table.columns if col != "選択"]
        edited_df = st.data_editor(
            styled_table,
            use_container_width=True,
            hide_index=True,
            disabled=disabled_cols,
            column_config={
                "選択": st.column_config.CheckboxColumn(
                    "選択",
                    help="チェックを入れると地図・迂回シミュレーションが切り替わります",
                    default=False,
                )
            },
            key=f"ranking_editor_{selected_muni}_{category_filter}_{only_emergency}_{selected_id}",
        )

        checked_positions = [
            idx for idx, val in enumerate(edited_df["選択"].tolist()) if bool(val)
        ]
        if len(checked_positions) == 0:
            # 選択中行のチェックが外された場合は強制的に選択状態を維持
            st.rerun()
        elif len(checked_positions) >= 2:
            # 新しくチェックされた行を特定し、排他制御で切り替える
            new_id: str | None = None
            for pos in checked_positions:
                cand_id = str(df_ranked.iloc[pos]["id"])
                if cand_id != selected_id:
                    new_id = cand_id
                    break
            if new_id is not None and new_id != selected_id:
                st.session_state["selected_facility_id"] = new_id
                st.rerun()
        elif len(checked_positions) == 1:
            only_id = str(df_ranked.iloc[checked_positions[0]]["id"])
            if only_id != selected_id:
                st.session_state["selected_facility_id"] = only_id
                st.rerun()

    # ==========================================================
    # タブ2: 予算配分・修繕プラン最適化
    # ==========================================================
    with tab2:
        st.subheader(f"{selected_muni} 年間修繕予算シミュレーション")
        budget_man_yen = st.slider(
            "年間修繕予算（万円）",
            min_value=5000,
            max_value=200000,
            value=30000,
            step=5000,
            format="%d 万円",
            help="5,000万円〜20億円（200,000万円）の範囲で年間修繕予算を設定します。",
        )
        st.caption(
            f"設定予算: **{budget_man_yen / 10000:.2f} 億円**（{budget_man_yen:,} 万円）"
            "｜ 防災・緊急輸送ボーナス込みの総合優先度スコアが高い順に予算上限内で自動選定します。"
        )

        plan = optimize_repair_plan(df_ranked, budget_man_yen=float(budget_man_yen))
        budget_usage_pct = (
            (plan.total_cost_man_yen / plan.budget_man_yen * 100.0)
            if plan.budget_man_yen > 0
            else 0.0
        )

        pk1, pk2, pk3, pk4, pk5 = st.columns(5)
        pk1.metric(
            label="修繕対象 選定施設数",
            value=f"{len(plan.selected_df)} / {len(df_ranked)} 施設",
            delta=f"選定率 {len(plan.selected_df) / len(df_ranked) * 100:.1f}%",
            delta_color="off",
            delta_arrow="off",
        )
        pk2.metric(
            label="合計修繕コスト",
            value=f"{plan.total_cost_man_yen:,.0f} 万円",
            delta=f"消化率 {budget_usage_pct:.1f}%（{plan.total_cost_man_yen / 10000:.2f}億円）",
            delta_color="off",
            delta_arrow="off",
        )
        pk3.metric(
            label="解消される迂回・孤立リスク",
            value=f"{plan.resolved_detour_risk_pct:.1f} %",
            delta=f"高リスク {plan.high_risk_resolved_count}/{plan.high_risk_total_count} 施設を解消",
            delta_color="normal",
            delta_arrow="off",
        )
        pk4.metric(
            label="守られる緊急輸送高リスク施設",
            value=f"{plan.emergency_high_risk_resolved_count} / {plan.emergency_high_risk_total_count} 施設",
            delta="緊急輸送道路上のS/Aランク",
            delta_color="normal",
            delta_arrow="off",
        )
        pk5.metric(
            label="見送りの高リスク施設",
            value=f"{len(plan.deferred_high_risk_df)} 施設",
            delta="総合ランク S / A の残存数",
            delta_color="inverse" if len(plan.deferred_high_risk_df) > 0 else "off",
            delta_arrow="off",
        )

        st.markdown(
            "**凡例:** "
            "<span style='color:#2563eb;font-weight:bold;'>● 今回修繕する施設（青）</span> ｜ "
            "<span style='color:#dc2626;font-weight:bold;'>● 予算不足で見送られる高リスク施設 [S/A]（赤）</span> ｜ "
            "<span style='color:#6b7280;font-weight:bold;'>● その他の見送り施設（灰）</span>",
            unsafe_allow_html=True,
        )

        plan_map = _build_plan_map(
            plan=plan,
            muni_bounds=muni_bounds,
            df_ranked=df_ranked,
            routes=emergency_routes,
            facilities_df=facilities_df,
            show_emergency=show_emergency,
            show_hospitals=show_hospitals,
            show_shelters=show_shelters,
            mesh_gdf=mesh_gdf,
            show_pop_mesh=show_pop_mesh,
        )
        st_folium(
            plan_map,
            width=None,
            height=430,
            returned_objects=[],
            key=f"plan_map_{selected_muni}_{category_filter}_{only_emergency}_{budget_man_yen}_{aging_weight}_{show_emergency}_{show_hospitals}_{show_shelters}_{show_pop_mesh}",
        )

        col_sel, col_def = st.columns(2)
        with col_sel:
            st.subheader(f"今回修繕する施設一覧（{len(plan.selected_df)}件）")
            if plan.selected_df.empty:
                st.info("現在の予算では選定可能な施設がありません。予算を引き上げてください。")
            else:
                sel_view = plan.selected_df.copy()
                sel_view["迂回増加"] = sel_view.apply(_format_detour_text, axis=1)
                sel_view = sel_view[
                    [
                        "rank_order",
                        "combined_rank",
                        "combined_score",
                        "emergency_route_display",
                        "name",
                        "category",
                        "age",
                        "迂回増加",
                        "estimated_cost_man_yen",
                    ]
                ].rename(
                    columns={
                        "rank_order": "順位",
                        "combined_rank": "ランク",
                        "combined_score": "総合スコア",
                        "emergency_route_display": "緊急輸送道路",
                        "name": "施設名",
                        "category": "種別",
                        "age": "築年数",
                        "estimated_cost_man_yen": "概算コスト(万円)",
                    }
                )
                st.dataframe(sel_view, use_container_width=True, hide_index=True)

        with col_def:
            st.subheader(
                f"予算不足で見送られる高リスク施設（{len(plan.deferred_high_risk_df)}件）"
            )
            if plan.deferred_high_risk_df.empty:
                st.success("高リスク（総合ランク S / A）の施設はすべて修繕対象に含まれました。")
            else:
                def_view = plan.deferred_high_risk_df.copy()
                def_view["迂回増加"] = def_view.apply(_format_detour_text, axis=1)
                def_view = def_view[
                    [
                        "rank_order",
                        "combined_rank",
                        "combined_score",
                        "emergency_route_display",
                        "name",
                        "category",
                        "age",
                        "迂回増加",
                        "estimated_cost_man_yen",
                    ]
                ].rename(
                    columns={
                        "rank_order": "順位",
                        "combined_rank": "ランク",
                        "combined_score": "総合スコア",
                        "emergency_route_display": "緊急輸送道路",
                        "name": "施設名",
                        "category": "種別",
                        "age": "築年数",
                        "estimated_cost_man_yen": "概算コスト(万円)",
                    }
                )
                st.dataframe(def_view, use_container_width=True, hide_index=True)

    # ==========================================================
    # タブ3: 将来人口×インフラ維持・集約仕分け
    # ==========================================================
    with tab3:
        st.subheader(f"{selected_muni} 2040年将来人口 × インフラ維持・集約仕分け")
        if mesh_gdf is None or mesh_gdf.empty:
            st.warning(
                "将来推計人口メッシュ（500m_mesh_2018_34.shp）が読み込めませんでした。同じディレクトリにShapefile一式が配置されているかご確認ください。"
            )

        tcol1, tcol2 = st.columns(2)
        with tcol1:
            score_thresh = st.slider(
                "インフラ重要度スコア閾値（縦軸）",
                min_value=30.0,
                max_value=85.0,
                value=60.0,
                step=5.0,
                help="この値以上（または孤立・第1/2次緊急輸送道路）を『重要度・リスク高』と判定します。",
            )
        with tcol2:
            pop_thresh = st.slider(
                "2040年周辺人口閾値（横軸：半径1km圏）",
                min_value=100,
                max_value=3000,
                value=500,
                step=100,
                help="2040年周辺人口がこの値以上（または減少率15%未満）を『人口維持』と判定します。",
            )

        df_quad = classify_maintenance_quadrant(
            df_ranked, score_threshold=float(score_thresh), pop_threshold=float(pop_thresh)
        )

        # 4分類ごとのKPIカード
        q_cols = st.columns(4)
        for idx, q_label in enumerate(QUADRANT_LABELS):
            sub_q = df_quad[df_quad["maintenance_category"] == q_label]
            q_cnt = len(sub_q)
            q_cost = float(sub_q["estimated_cost_man_yen"].sum()) if not sub_q.empty else 0.0
            q_cols[idx].metric(
                label=f"【{q_label}】",
                value=f"{q_cnt:,} 施設",
                delta=f"概算維持費 {q_cost / 10000:.2f} 億円",
                delta_color="off",
                delta_arrow="off",
            )

        # 4象限散布図（Plotly） & 4分類色分け地図
        chart_col, map_col = st.columns([1.1, 1.0])
        with chart_col:
            st.markdown("#### 4象限マトリクス（2040年周辺人口 × 総合重要度）")
            plot_df = df_quad.copy()
            plot_df["迂回影響"] = plot_df.apply(_format_detour_text, axis=1)
            # 0人または欠損の施設は微小値1（対数スケールの基点）としてプロット
            plot_df["pop_2040_plot"] = plot_df["pop_2040_1km"].clip(lower=1.0)
            x_max = max(20000.0, float(plot_df["pop_2040_plot"].max()) * 1.4)

            fig = px.scatter(
                plot_df,
                x="pop_2040_plot",
                y="combined_score",
                color="maintenance_category",
                color_discrete_map=QUADRANT_COLORS,
                category_orders={"maintenance_category": QUADRANT_LABELS},
                hover_name="name",
                hover_data={
                    "pop_2040_plot": False,
                    "pop_2040_1km": ":,.0f",
                    "category": True,
                    "route_name": True,
                    "age": True,
                    "迂回影響": True,
                    "pop_change_rate_1km": ":.1f",
                    "estimated_cost_man_yen": ":,.0f",
                    "maintenance_category": False,
                },
                labels={
                    "pop_2040_plot": "2040年周辺人口（半径1km圏・人・対数軸）",
                    "pop_2040_1km": "2040年周辺人口（人）",
                    "combined_score": "インフラ総合重要度スコア（点）",
                    "maintenance_category": "仕分け区分",
                    "route_name": "路線名",
                    "age": "築年数",
                    "pop_change_rate_1km": "人口増減率(%)",
                    "estimated_cost_man_yen": "概算維持費(万円)",
                },
                height=480,
            )

            # 横軸を対数スケールに設定し、500人付近が中央に来るようバランス調整
            fig.update_xaxes(
                type="log",
                range=[0, math.log10(x_max)],
                tickvals=[1, 10, 100, 500, 1000, 5000, 20000],
                ticktext=["0~1", "10", "100", "500", "1,000", "5,000", "20,000"],
                gridcolor="rgba(203, 213, 225, 0.45)",
            )
            fig.update_yaxes(
                range=[0, 105],
                gridcolor="rgba(203, 213, 225, 0.45)",
            )

            # 4象限の背景色ブロックを追加
            fig.add_shape(type="rect", x0=pop_thresh, x1=x_max, y0=score_thresh, y1=100, fillcolor="rgba(220, 38, 38, 0.08)", line_width=0, layer="below")
            fig.add_shape(type="rect", x0=1, x1=pop_thresh, y0=score_thresh, y1=100, fillcolor="rgba(234, 88, 12, 0.08)", line_width=0, layer="below")
            fig.add_shape(type="rect", x0=pop_thresh, x1=x_max, y0=0, y1=score_thresh, fillcolor="rgba(37, 99, 235, 0.08)", line_width=0, layer="below")
            fig.add_shape(type="rect", x0=1, x1=pop_thresh, y0=0, y1=score_thresh, fillcolor="rgba(100, 116, 139, 0.08)", line_width=0, layer="below")

            # 閾値境界線（破線）
            fig.add_hline(y=score_thresh, line_dash="dash", line_color="#64748b", line_width=1.5)
            fig.add_vline(x=pop_thresh, line_dash="dash", line_color="#64748b", line_width=1.5)

            # 4象限エリアラベル（各領域の中央付近に注記表示）
            x_left_center = 10 ** (math.log10(max(2.0, pop_thresh)) / 2)
            x_right_center = 10 ** ((math.log10(pop_thresh) + math.log10(x_max)) / 2)
            y_top_center = min(94.0, score_thresh + (100.0 - score_thresh) * 0.70)
            y_bottom_center = max(8.0, score_thresh * 0.25)

            annotations_info = [
                (x_right_center, y_top_center, "最優先で維持・修繕", "利用者多・高リスク", "#dc2626"),
                (x_left_center, y_top_center, "防災・ネットワーク維持", "過疎地だが迂回不能・緊急輸送路", "#ea580c"),
                (x_right_center, y_bottom_center, "長寿命化・経過観察", "利用者多・代替路あり", "#2563eb"),
                (x_left_center, y_bottom_center, "集約・縮退検討候補", "利用者減・代替路あり", "#475569"),
            ]
            for ax, ay, main_text, sub_text, text_color in annotations_info:
                fig.add_annotation(
                    x=ax,
                    y=ay,
                    text=f"<b>{main_text}</b><br><span style='font-size:9.5px;color:#475569;'>（{sub_text}）</span>",
                    showarrow=False,
                    font=dict(size=11, color=text_color, family="sans-serif"),
                    bgcolor="rgba(255, 255, 255, 0.85)",
                    bordercolor="rgba(203, 213, 225, 0.7)",
                    borderwidth=1,
                    borderpad=4,
                )

            fig.update_layout(
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="left",
                    x=0,
                    title_text="",
                ),
                margin=dict(l=10, r=10, t=35, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.info(
                "縦軸（上に行くほど老朽化や迂回影響が深刻）と横軸（右に行くほど2040年も人口が多い）の2軸で全施設を4分類しています。"
                "左下の領域にある施設ほど、将来的な集約・再編の検討余地が大きいことを示します。"
            )

        with map_col:
            st.markdown("#### 維持・集約仕分けマップ（2040年人口メッシュ重畳）")
            quad_map = _build_quadrant_map(
                df_quad,
                muni_bounds=muni_bounds,
                mesh_gdf=mesh_gdf,
                show_pop_mesh=True if show_pop_mesh else (mesh_gdf is not None),
            )
            st_folium(
                quad_map,
                width=None,
                height=480,
                returned_objects=[],
                key=f"quad_map_{selected_muni}_{category_filter}_{only_emergency}_{score_thresh}_{pop_thresh}_{aging_weight}_{show_pop_mesh}",
            )

        # 集約・縮退検討候補テーブル
        candidates_df = df_quad[
            df_quad["maintenance_category"] == "将来的な集約・縮退検討候補"
        ].copy()
        total_reducible = (
            float(candidates_df["estimated_cost_man_yen"].sum())
            if not candidates_df.empty
            else 0.0
        )
        st.subheader(
            f"将来的な集約・縮退検討候補一覧（{len(candidates_df)}件 ／ 削減可能概算維持費: {total_reducible:,.0f} 万円 ≒ {total_reducible / 10000:.2f} 億円）"
        )
        if candidates_df.empty:
            st.info("現在の閾値では『将来的な集約・縮退検討候補』に該当する施設はありません。")
        else:
            cand_view = candidates_df.copy()
            cand_view["代替ルート距離(迂回増)"] = cand_view.apply(
                _format_detour_text, axis=1
            )
            cand_view = cand_view[
                [
                    "name",
                    "category",
                    "route_name",
                    "age",
                    "代替ルート距離(迂回増)",
                    "pop_2040_1km",
                    "pop_change_rate_1km",
                    "estimated_cost_man_yen",
                    "combined_score",
                ]
            ].rename(
                columns={
                    "name": "施設名",
                    "category": "種別",
                    "route_name": "路線名",
                    "age": "築年数",
                    "pop_2040_1km": "2040年周辺人口(人)",
                    "pop_change_rate_1km": "人口増減率(%)",
                    "estimated_cost_man_yen": "削減可能概算維持費(万円)",
                    "combined_score": "総合スコア",
                }
            )
            st.dataframe(cand_view, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
