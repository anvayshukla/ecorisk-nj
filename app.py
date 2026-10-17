import streamlit as st
import geopandas as gpd
import folium
from streamlit_folium import st_folium
from shapely.geometry import Point
from streamlit_geolocation import streamlit_geolocation
import pandas as pd
import altair as alt
import os
import math
import numpy as np
import pyarrow.parquet as pq
import base64

# --- Page Configuration ---
st.set_page_config(
    page_title="EcoRisk NJ | Cancer & Community Air Toxics Explorer",
    page_icon="🛡️",
    layout="wide"
)

# --- Professional Custom SaaS CSS Injection ---
st.markdown("""
<style>
    .main {
        background-color: #F8FAFC;
    }
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 0rem !important;
    }
    div.stMetric {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        padding: 8px 10px !important;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    div.stMetric label {
        font-size: 0.75rem !important;
        white-space: nowrap !important;
    }
    div.stMetric div[data-testid="stMetricValue"] {
        font-size: 1.1rem !important;
    }
    .exact-risk-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        padding: 16px;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    h1, h2, h3 {
        color: #0F172A;
    }
    .saas-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        padding: 20px;
        border-radius: 14px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- Clean Header Section ---
st.caption("Advanced Environmental Justice & Industrial Exposure Intelligence")
st.title("🛡️ EcoRisk NJ: Cancer & Community Air Toxics Explorer")

RISK_COL = "ETHYLENE_OXIDE_RISK_R2_2020_Total Cancer Risk (per million)"
DEFAULT_LAT, DEFAULT_LON = 40.5187, -74.4121  # Middlesex / Edison Hub
DEFAULT_RADIUS_MILES = 5

# Initialize Session State safely
if 'current_lat' not in st.session_state:
    st.session_state['current_lat'] = DEFAULT_LAT
if 'current_lon' not in st.session_state:
    st.session_state['current_lon'] = DEFAULT_LON
if 'use_gps_checked' not in st.session_state:
    st.session_state['use_gps_checked'] = False

locations = {
    "Middlesex County (Edison Hub)": {"lat": 40.5187, "lon": -74.4121},
    "Pennsville (Salem County Hub)": {"lat": 39.6643, "lon": -75.4674},
    "Atlantic County": {"lat": 39.4623, "lon": -74.6493},
    "Bergen County": {"lat": 40.8904, "lon": -74.0443},
    "Burlington County": {"lat": 39.9912, "lon": -74.7937},
    "Camden County": {"lat": 39.9259, "lon": -75.1197},
    "Cape May County": {"lat": 39.0837, "lon": -74.8236},
    "Cumberland County": {"lat": 39.4290, "lon": -75.2312},
    "Essex County": {"lat": 40.7357, "lon": -74.1724},
    "Gloucester County": {"lat": 39.8407, "lon": -75.1507},
    "Hudson County": {"lat": 40.7178, "lon": -74.0431},
    "Hunterdon County": {"lat": 40.5090, "lon": -74.8623},
    "Mercer County": {"lat": 40.2206, "lon": -74.7597},
    "Monmouth County": {"lat": 40.2587, "lon": -74.2721},
    "Morris County": {"lat": 40.7968, "lon": -74.4815},
    "Ocean County": {"lat": 39.9793, "lon": -74.2023},
    "Passaic County": {"lat": 40.9168, "lon": -74.1718},
    "Salem County": {"lat": 39.5612, "lon": -75.4645},
    "Somerset County": {"lat": 40.5732, "lon": -74.6078},
    "Sussex County": {"lat": 41.0543, "lon": -74.7479},
    "Union County": {"lat": 40.6586, "lon": -74.2109},
    "Warren County": {"lat": 40.8243, "lon": -75.0831}
}

# --- 1. Lightning-Fast Data Loaders ---
@st.cache_data
def load_base_data():
    try:
        parquet_path = "data/EtO_Risk_NJ_2020.parquet"
        gpkg_path = "data/EtO_Facilities_NJ.gpkg"
        if os.path.exists(parquet_path) and os.path.exists(gpkg_path):
            cols_to_load = ['geometry', RISK_COL]
            pf = pq.ParquetFile(parquet_path)
            all_cols = pf.schema.names
            filtered_cols = [c for c in cols_to_load if c in all_cols]
            
            blocks = gpd.read_parquet(parquet_path, columns=filtered_cols if filtered_cols else None)
            packages = gpd.read_file(gpkg_path)
            
            if blocks.crs != "EPSG:4326":
                blocks = blocks.to_crs("EPSG:4326")
            if packages.crs != "EPSG:4326":
                packages = packages.to_crs("EPSG:4326")
                
            blocks['geometry'] = blocks['geometry'].simplify(tolerance=0.002, preserve_topology=True)
            return blocks, packages, True
    except Exception as e:
        print(f"GIS load note: {e}")
    return None, None, False

@st.cache_data
def load_regional_facilities(year):
    path = f"data/TRI_Facility_Data/{year}_NJ_EastCoast_Airshed_EtO.csv"
    if os.path.exists(path):
        try:
            return pd.read_csv(path, low_memory=False)
        except Exception as e:
            st.warning(f"Error reading {path}: {e}")
    return pd.DataFrame()

with st.spinner("Initializing high-performance GIS engine..."):
    blocks_gdf, facilities_gdf, has_real_data = load_base_data()

if has_real_data and blocks_gdf is not None:
    if RISK_COL not in blocks_gdf.columns:
        st.error(f"🚨 Critical Data Error: Expected risk column '{RISK_COL}' not found in the baseline risk dataset.")
        st.stop()

if has_real_data and facilities_gdf is not None:
    _fac_lats = np.array([geom.y if geom and geom.geom_type == 'Point' else np.nan for geom in facilities_gdf.geometry])
    _fac_lons = np.array([geom.x if geom and geom.geom_type == 'Point' else np.nan for geom in facilities_gdf.geometry])
else:
    _fac_lats, _fac_lons = np.array([]), np.array([])

# --- 2. QGIS Red Color Gradient Mapping ---
def get_qgis_color(score):
    try:
        val = float(score)
    except:
        return "#FBCFE8"
    
    if val > 100:
        return "#4A0404"
    elif val > 50:
        return "#7F1D1D"
    elif val > 20:
        return "#991B1B"
    elif val > 10:
        return "#DC2626"
    elif val > 1:
        return "#FB7185"
    else:
        return "#FBCFE8"

# --- 3. Clean Dashboard Layout with 6 Optimized Tabs ---
tab_map, tab_multiyear, tab_analytics, tab_about, tab_dev, tab_paper = st.tabs([
    "🗺️ Risk Model", 
    "📈 Multi-Year Explorer", 
    "📊 Analytics & Trends",
    "💡 About EcoRisk NJ",
    "👨‍💻 Meet the Developer",
    "📄 Research Paper"
])

# --- TAB 1: 2020 EPA BASELINE RISK MODEL ---
with tab_map:
    with st.expander("📌 Quick-Start Guide for Judges & Users", expanded=False):
        st.markdown("""
        Welcome to **EcoRisk NJ**! Follow these quick steps to explore local air toxics exposure:
        1. **Select a Target Hub or GPS:** Choose a New Jersey county hub from the sidebar dropdown or check **Center on GPS** (ensure browser location permission is enabled via the address bar icon).
        2. **Adjust Analysis Radius:** Use the slider in the control panel to change the buffer zone (1, 5, or 10 miles).
        3. **Inspect Risk & Facilities:** Click anywhere **within the New Jersey regional map boundaries** to relocate the analysis center and view exact census block risk scores and nearby industrial facilities below.
        4. **Explore Multi-Year Trends:** Switch to Tabs 2 and 3 to examine multi-year industrial emission pounds (2020–2024) across the regional airshed.
        """)

    current_radius = st.session_state.get('radius_miles_slider', DEFAULT_RADIUS_MILES)

    temp_radius_degrees = current_radius / 69.0
    block_count = 0
    local_facilities_list = []
    
    if has_real_data and blocks_gdf is not None:
        center_point = Point(st.session_state['current_lon'], st.session_state['current_lat'])
        buffered_poly = center_point.buffer(temp_radius_degrees)
        possible_matches_index = list(blocks_gdf.sindex.intersection(buffered_poly.bounds))
        possible_matches = blocks_gdf.iloc[possible_matches_index]
        local_blocks = possible_matches[possible_matches.intersects(buffered_poly)].copy()
        if not local_blocks.empty:
            block_count = len(local_blocks)

    if has_real_data and facilities_gdf is not None and len(_fac_lats) > 0:
        cur_lat = st.session_state['current_lat']
        cur_lon = st.session_state['current_lon']
        cos_lat = math.cos(math.radians(cur_lat))
        d_lat = (_fac_lats - cur_lat) * 69.0
        d_lon = (_fac_lons - cur_lon) * 69.0 * cos_lat
        distances = np.sqrt(d_lat**2 + d_lon**2)

        for pos, (_, fac) in enumerate(facilities_gdf.iterrows()):
            dist_miles = distances[pos]
            if np.isnan(dist_miles):
                continue
            geom = fac.geometry
            fac_lat, fac_lon = geom.y, geom.x

            fac_name = None
            for col in fac.index:
                if any(k in col.lower() for k in ['name', 'facility', 'fac_nam', 'primary_nm']):
                    val = fac[col]
                    if pd.notna(val) and isinstance(val, str) and len(val.strip()) > 1:
                        fac_name = val.strip()
                        break
            if not fac_name:
                fac_name = "Ethylene Oxide Industrial Facility"

            fac_city = None
            for col in fac.index:
                if any(k in col.lower() for k in ['city', 'municipality', 'mun', 'town']):
                    val = fac[col]
                    if pd.notna(val) and isinstance(val, str) and len(val.strip()) > 1:
                        fac_city = val.strip()
                        break
            if not fac_city:
                fac_city = "New Jersey"

            fac_street = None
            for col in fac.index:
                if any(k in col.lower() for k in ['address', 'addr', 'location_ad', 'street']):
                    val = fac[col]
                    if pd.notna(val) and isinstance(val, str) and len(val.strip()) > 1:
                        fac_street = val.strip()
                        break
            if not fac_street:
                fac_street = ""

            fac_zip = None
            for col in fac.index:
                if any(k in col.lower() for k in ['zip', 'postal', 'mail_zip']):
                    val = fac[col]
                    if pd.notna(val):
                        s_val = str(val).strip()
                        if len(s_val) >= 5:
                            fac_zip = s_val[:5]
                            break
            if not fac_zip:
                fac_zip = ""

            addr_parts = [p for p in [fac_street, fac_city, "NJ", fac_zip] if p]
            full_address = ", ".join(addr_parts) if addr_parts else "New Jersey"

            if dist_miles <= current_radius:
                local_facilities_list.append({
                    "Facility Name": fac_name,
                    "Address": full_address,
                    "Latitude": round(fac_lat, 4),
                    "Longitude": round(fac_lon, 4),
                    "Distance (mi)": round(dist_miles, 1)
                })

    col_ctrl, col_mapview = st.columns([1.0, 2.6])
    
    with col_ctrl:
        with st.container(border=True):
            st.markdown("#### 🎯 Region Controls")
            st.markdown("<p style='font-size: 0.9rem; font-weight: 600; color: #1E293B; margin-bottom: 8px;'>📅 Dataset: 2020 Baseline Model</p>", unsafe_allow_html=True)
            
            user_loc = streamlit_geolocation()
            use_gps_toggle = st.checkbox("📍 Center on GPS", key='use_gps_checked')

            if use_gps_toggle:
                if not user_loc or not user_loc.get('latitude') or not user_loc.get('longitude'):
                    st.warning("⚠️ **Location Access Required:** Please allow browser location access in your address bar to enable GPS tracking.")
                else:
                    st.session_state['current_lat'] = user_loc['latitude']
                    st.session_state['current_lon'] = user_loc['longitude']
                    st.warning("⚠️ **GPS Active:** Please uncheck 'Center on GPS' to freely select other counties or click on the map.")
                    st.markdown("<p style='font-size: 0.8rem; color: #16A34A; font-weight: 600; margin-top: 2px;'>🟢 GPS Active</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p style='font-size: 0.8rem; color: #64748B; margin-top: 2px;'>⚪ GPS Standby</p>", unsafe_allow_html=True)
            
            def on_county_change():
                selected_name = st.session_state['county_selectbox']
                coords = locations[selected_name]
                st.session_state['current_lat'] = coords["lat"]
                st.session_state['current_lon'] = coords["lon"]

            st.selectbox(
                "Target County / Hub:", 
                list(locations.keys()), 
                key='county_selectbox',
                on_change=on_county_change,
                help="💡 Tip: Click anywhere within the New Jersey map boundary to relocate the analysis center instantly!"
            )
            
            radius_miles = st.select_slider(
                "Analysis Radius (Miles):",
                options=[1, 5, 10],
                value=DEFAULT_RADIUS_MILES,
                key='radius_miles_slider'
            )

        with st.expander("⚕️ Air Toxics Cancer Risk Scale & Legend", expanded=True):
            st.markdown("""
            <p style='font-size: 0.78rem; color: #475569; margin-bottom: 8px; line-height: 1.3;'>
            <b>What this means:</b> Estimated excess lifetime cancer cases per 1M residents exposed continuously over 70 years.
            </p>
            <span style='color:#4A0404; font-weight:bold;'>■</span> <b>Extreme:</b> > 100 / million<br>
            <span style='color:#7F1D1D; font-weight:bold;'>■</span> <b>High:</b> 50 – 100 / million<br>
            <span style='color:#991B1B; font-weight:bold;'>■</span> <b>Elevated:</b> 20 – 50 / million<br>
            <span style='color:#DC2626; font-weight:bold;'>■</span> <b>Moderate:</b> 10 – 20 / million<br>
            <span style='color:#FB7185; font-weight:bold;'>■</span> <b>Low-Mod:</b> 1 – 10 / million<br>
            <span style='color:#FBCFE8; font-weight:bold;-webkit-text-stroke: 0.5px #334155;'>■</span> <b>Lower Modeled Risk:</b> < 1 / million
            """, unsafe_allow_html=True)

        with st.container(border=True):
            with st.expander("📖 About EPA Modeling Note", expanded=False):
                st.markdown(
                    "Tab 1 displays the **2020 EPA AirToxScreen census block modeling dataset** "
                    "(estimated theoretical long-term exposure risks). This represents modeled inhalation risk "
                    "rather than direct current exposure measurement."
                )

    with col_mapview:
        zoom_level = 13 if radius_miles == 1 else (12 if radius_miles == 5 else 11)
        
        m = folium.Map(
            location=[st.session_state['current_lat'], st.session_state['current_lon']], 
            zoom_start=zoom_level, 
            min_zoom=8,
            max_zoom=16,
            max_bounds=True,
            min_lat=38.8,
            max_lat=41.5,
            min_lon=-75.8,
            max_lon=-73.8,
            tiles="OpenStreetMap"
        )

        folium.Circle(
            location=[st.session_state['current_lat'], st.session_state['current_lon']],
            radius=radius_miles * 1609.34,
            color="#2563EB",
            weight=2,
            fill=True,
            fill_color="#2563EB",
            fill_opacity=0.03,
            tooltip=f"{radius_miles}-Mile Exposure Buffer"
        ).add_to(m)
        
        if has_real_data and blocks_gdf is not None:
            center_point = Point(st.session_state['current_lon'], st.session_state['current_lat'])
            buffered_poly = center_point.buffer(radius_miles / 69.0)
            possible_matches_index = list(blocks_gdf.sindex.intersection(buffered_poly.bounds))
            possible_matches = blocks_gdf.iloc[possible_matches_index]
            local_blocks = possible_matches[possible_matches.intersects(buffered_poly)].copy()
            
            if not local_blocks.empty:
                risk_field = RISK_COL
                local_blocks['Risk_Display'] = local_blocks[risk_field].apply(
                    lambda x: f"{float(x):.1f}" if pd.notna(x) else "N/A"
                )

                folium.GeoJson(
                    local_blocks,
                    style_function=lambda feature: {
                        'fillColor': get_qgis_color(feature['properties'].get(risk_field, 0)),
                        'color': '#334155',
                        'weight': 0.1,
                        'fillOpacity': 0.4
                    },
                    tooltip=folium.GeoJsonTooltip(
                        fields=['Risk_Display'], 
                        aliases=['Air Toxics Cancer Risk (per million):']
                    )
                ).add_to(m)

        if has_real_data and facilities_gdf is not None and len(_fac_lats) > 0:
            for fac_item in local_facilities_list:
                folium.Marker(
                    location=[fac_item['Latitude'], fac_item['Longitude']],
                    popup=f"<b>{fac_item['Facility Name']}</b><br>{fac_item['Address']}<br><b>Distance:</b> {fac_item['Distance (mi)']} mi",
                    tooltip=f"{fac_item['Facility Name']} - {fac_item['Distance (mi)']} mi",
                    icon=folium.Icon(color='darkred', icon='cloud', prefix='fa')
                ).add_to(m)

        folium.Marker(
            location=[st.session_state['current_lat'], st.session_state['current_lon']],
            popup="Analysis Center",
            tooltip="Analysis Center",
            icon=folium.Icon(color='blue', icon='crosshairs', prefix='fa')
        ).add_to(m)

        map_data = st_folium(m, use_container_width=True, height=450, returned_objects=['last_clicked'])
        
        if map_data and map_data.get('last_clicked'):
            if use_gps_toggle:
                st.warning("⚠️ **GPS mode is active.** Map manual clicks are disabled. Please uncheck 'Center on GPS' in the sidebar controls to explore other locations.")
            else:
                clicked_lat = map_data['last_clicked']['lat']
                clicked_lon = map_data['last_clicked']['lng']
                if clicked_lat != st.session_state['current_lat'] or clicked_lon != st.session_state['current_lon']:
                    valid_click = True
                    if has_real_data and blocks_gdf is not None:
                        pt = Point(clicked_lon, clicked_lat)
                        possible_idx = list(blocks_gdf.sindex.intersection(pt.buffer(0.001).bounds))
                        if not possible_idx or not blocks_gdf.iloc[possible_idx].intersects(pt).any():
                            valid_click = False
                    
                    if valid_click:
                        st.session_state['current_lat'] = clicked_lat
                        st.session_state['current_lon'] = clicked_lon
                        st.rerun()
                    else:
                        st.warning("⚠️ Please select a location within New Jersey boundaries. Clicks outside New Jersey are restricted.")
        
        st.markdown(
            f"""
            <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; padding: 10px 16px; border-radius: 10px; display: flex; justify-content: space-around; align-items: center; margin-top: 10px; margin-bottom: 15px; box-shadow: 0 1px 3px rgba(0,0,0,0.03);">
                <div style="font-size: 0.85rem; color: #334155;" title="Total Census Blocks in Region: {block_count:,}">📍 <b>Blocks in Region:</b> <span style="color: #0F172A; font-weight: 600;">{block_count:,}</span></div>
                <div style="font-size: 0.85rem; color: #334155;" title="Active Radius Buffer: {radius_miles} Miles">🔵 <b>Radius Buffer:</b> <span style="color: #0F172A; font-weight: 600;">{radius_miles} Miles</span></div>
                <div style="font-size: 0.85rem; color: #334155;" title="Tracked Industrial Facilities in Radius: {len(local_facilities_list)}">🏭 <b>Facilities in Radius:</b> <span style="color: #0F172A; font-weight: 600;">{len(local_facilities_list)}</span></div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("---")
        st.markdown("### 🛡️ Active Location Inspector & Health Guidance")
        
        center_pt = Point(st.session_state['current_lon'], st.session_state['current_lat'])
        matched_block = blocks_gdf[blocks_gdf.contains(center_pt)] if (has_real_data and blocks_gdf is not None) else gpd.DataFrame()

        col_loc1, col_loc2 = st.columns([1, 2])
        with col_loc1:
            st.markdown(f"**📍 Target Coordinates:** `{st.session_state['current_lat']:.5f}, {st.session_state['current_lon']:.5f}`")
            
            spot_risk = 0.0
            has_data = False
            if not matched_block.empty:
                raw_val = matched_block.iloc[0].get(RISK_COL, None)
                if pd.notna(raw_val):
                    try:
                        spot_risk = float(raw_val)
                        spot_risk_str = f"{spot_risk:,.1f} / million"
                        st.markdown(f"""
                            <div class="exact-risk-card" title="Exact Block Cancer Risk: {spot_risk_str}">
                                <div style="font-size: 0.75rem; color: #64748B; font-weight: 600; text-transform: uppercase; margin-bottom: 4px;">Exact Block Cancer Risk</div>
                                <div style="font-size: 1.9rem; font-weight: 800; color: #0F172A;" title="{spot_risk_str}">{spot_risk_str}</div>
                            </div>
                        """, unsafe_allow_html=True)
                        has_data = True
                    except (ValueError, TypeError):
                        st.metric("Exact Block Cancer Risk", "No data available")
                else:
                    st.metric("Exact Block Cancer Risk", "No data available")
            else:
                st.metric("Exact Block Cancer Risk", "No data available")

        with col_loc2:
            if not has_data:
                st.info("ℹ️ **No Census Block Data Available:** Coordinates fall outside modeled geographic boundaries.")
            elif spot_risk > 50:
                st.error(f"⚠️ **Elevated Modeled Risk Detected ({spot_risk:,.1f} per million):** EPA estimates indicate a higher excess lifetime cancer risk associated with ethylene oxide exposure in this census block.")
                st.markdown("""
                **Recommended Community Actions:**
                * **Official Air Monitoring:** Consult official NJDEP air quality advisories and EPA monitoring updates.
                * **Ventilation Timing:** Monitor local wind direction and facility operating schedules before opening windows.
                * **Compliance Review:** Check facility air permits and compliance records via the NJDEP Data Miner.
                """)
            elif spot_risk > 10:
                st.warning(f"⚠️ **Moderate Risk Level ({spot_risk:,.1f} per million):** Monitor local air quality advisories.")
                st.markdown("""
                **Recommended Precautions:**
                * Stay informed via the EPA AirToxScreen portal for regional updates.
                * Review local facility reporting on the NJDEP Data Miner.
                """)
            else:
                st.success("✅ **Lower Modeled Ethylene Oxide Risk:** This coordinate falls within census blocks showing lower estimated baseline risk values.")
                st.markdown("""
                **General Best Practices:**
                * Continue standard outdoor air quality awareness and community preparedness.
                """)

        st.markdown("---")
        st.markdown("### 🏭 Industrial Facilities Within Selected Radius (2020 Baseline)")
        if local_facilities_list:
            for fac in local_facilities_list:
                with st.container(border=True):
                    st.markdown(f"**🏭 {fac['Facility Name']}**")
                    st.markdown(f"📍 **Address:** {fac['Address']}")
                    st.markdown(f"🌍 **Coordinates:** `{fac['Latitude']}, {fac['Longitude']}`")
        else:
            st.info(f"ℹ️ No major industrial ethylene oxide facilities are located directly within this specific radius buffer. Try expanding to 5 or 10 miles.")

# --- TAB 2: MULTI-YEAR REGIONAL ETHYLENE OXIDE EMISSION EXPLORER (2020–2024) ---
with tab_multiyear:
    st.markdown("### 📈 Multi-Year Regional Ethylene Oxide Emission Explorer (2020–2024)")
    st.markdown("`📅 Reporting Years Window: 2020–2024 (TRI Dataset)`")
    st.info("💡 **Regional Airshed Mode:** Inspect actual TRI industrial facility reporting (pounds released) across New Jersey and neighboring regional states (NY, PA, DE, CT, MD).")

    col_ctrl2, col_map2 = st.columns([1, 2.5])
    with col_ctrl2:
        with st.container(border=True):
            st.markdown("#### 🎯 Temporal Controls")
            selected_year = st.selectbox("Select Emission Year:", [2020, 2021, 2022, 2023, 2024], index=4, key="my_year_sel")
            
            def on_county_change_my():
                coords = locations[st.session_state['county_selectbox_my']]
                st.session_state['current_lat'] = coords["lat"]
                st.session_state['current_lon'] = coords["lon"]

            st.selectbox("Target County / Hub:", list(locations.keys()), key='county_selectbox_my', on_change=on_county_change_my)
            radius_miles_my = st.select_slider("Analysis Radius (Miles):", options=[1, 5, 10], value=DEFAULT_RADIUS_MILES, key="my_radius")

    with col_map2:
        zoom_level_my = 13 if radius_miles_my == 1 else (12 if radius_miles_my == 5 else 11)
        
        m_my = folium.Map(
            location=[st.session_state['current_lat'], st.session_state['current_lon']], 
            zoom_start=zoom_level_my, 
            min_zoom=8,
            max_zoom=16,
            max_bounds=True,
            min_lat=38.8,
            max_lat=41.5,
            min_lon=-75.8,
            max_lon=-73.8,
            tiles="OpenStreetMap"
        )
        folium.Circle(
            location=[st.session_state['current_lat'], st.session_state['current_lon']],
            radius=radius_miles_my * 1609.34, color="#0284C7", weight=2, fill=True, fill_color="#0EA5E9", fill_opacity=0.05
        ).add_to(m_my)

        regional_facs_list = []
        fac_df = load_regional_facilities(selected_year)
        
        if not fac_df.empty:
            cur_lat = st.session_state['current_lat']
            cur_lon = st.session_state['current_lon']
            cos_lat = math.cos(math.radians(cur_lat))

            my_lats = pd.to_numeric(fac_df.get('12. LATITUDE'), errors='coerce').to_numpy()
            my_lons = pd.to_numeric(fac_df.get('13. LONGITUDE'), errors='coerce').to_numpy()

            d_lat = (my_lats - cur_lat) * 69.0
            d_lon = (my_lons - cur_lon) * 69.0 * cos_lat
            my_distances = np.sqrt(d_lat**2 + d_lon**2)

            for idx, fac in fac_df.iterrows():
                lat = my_lats[idx]
                lon = my_lons[idx]
                dist_miles = my_distances[idx]

                if not np.isnan(lat) and not np.isnan(lon) and not np.isnan(dist_miles):
                    fac_name = str(fac.get('4. FACILITY NAME', 'FACILITY')).strip().upper()
                    fac_city = str(fac.get('6. CITY', '')).strip().title()
                    fac_st = str(fac.get('8. ST', '')).strip().upper()
                    fac_street = str(fac.get('5. STREET ADDRESS', '')).strip()
                    
                    stk_raw = pd.to_numeric(fac.get('52. 5.2 - STACK AIR', None), errors='coerce')
                    stack = stk_raw if pd.notna(stk_raw) else 0.0

                    fug_raw = pd.to_numeric(fac.get('51. 5.1 - FUGITIVE AIR', None), errors='coerce')
                    fugitive = fug_raw if pd.notna(fug_raw) else 0.0

                    total_rel = stack + fugitive
                    
                    full_address = f"{fac_street}, {fac_city}, {fac_st}"
                    
                    if dist_miles <= radius_miles_my:
                        regional_facs_list.append({
                            "Facility Name": fac_name,
                            "State": fac_st,
                            "Municipality": fac_city,
                            "Address": full_address,
                            "Stack Air (lbs)": f"{stack:,.2f}",
                            "Fugitive Air (lbs)": f"{fugitive:,.2f}",
                            f"{selected_year} Total (lbs)": f"{total_rel:,.2f}",
                            "Distance (mi)": round(dist_miles, 1)
                        })

                    folium.Marker(
                        location=[lat, lon],
                        popup=f"<b>{fac_name}</b> ({fac_st})<br>{full_address}<br><b>Distance:</b> {dist_miles:.1f} mi<br><b>Total Emissions:</b> {total_rel:,.2f} lbs",
                        tooltip=f"{fac_name} [{fac_st}] - {dist_miles:.1f} mi",
                        icon=folium.Icon(color='cadetblue' if dist_miles <= radius_miles_my else 'gray', icon='cloud', prefix='fa')
                    ).add_to(m_my)

        folium.Marker(location=[st.session_state['current_lat'], st.session_state['current_lon']], icon=folium.Icon(color='blue', icon='crosshairs', prefix='fa')).add_to(m_my)
        
        map_data_my = st_folium(m_my, use_container_width=True, height=450, returned_objects=['last_clicked'])
        
        if map_data_my and map_data_my.get('last_clicked'):
            clicked_lat = map_data_my['last_clicked']['lat']
            clicked_lon = map_data_my['last_clicked']['lng']
            if clicked_lat != st.session_state['current_lat'] or clicked_lon != st.session_state['current_lon']:
                st.session_state['current_lat'] = clicked_lat
                st.session_state['current_lon'] = clicked_lon
                st.rerun()
        
        st.markdown(f"### 🏭 Tracked Industrial Facilities in {selected_year} Within {radius_miles_my}-Mile Radius")
        st.info("💡 **Regional Context Note:** Neighboring state facilities (NY, PA, DE, CT, MD) provide valuable airshed context, but reported release volumes alone do not establish direct individual exposure levels for New Jersey residents.")

        if regional_facs_list:
            for fac in regional_facs_list:
                with st.container(border=True):
                    st.markdown(f"**🏭 {fac['Facility Name']}** [{fac['State']}] — *{fac['Distance (mi)']} miles away*")
                    st.markdown(f"📍 **Address:** {fac['Address']}")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Stack Air", fac['Stack Air (lbs)'])
                    c2.metric("Fugitive Air", fac['Fugitive Air (lbs)'])
                    c3.metric("Total Emissions", fac[f"{selected_year} Total (lbs)"])
        else:
            st.info(f"ℹ️ No industrial ethylene oxide facilities detected within this {radius_miles_my}-mile radius for {selected_year}.")

# --- TAB 3: FACILITY & EMISSION TRENDS ANALYTICS ---
with tab_analytics:
    st.markdown("### 📊 Executive Emissions Intelligence & Analytics (2020–2024)")
    st.write("Enterprise-grade analytical overview displaying macro industrial emission distributions, release pathways, and year-over-year deltas.")

    summary_data = []
    nj_summary_data = []
    nj_melted_rows = []
    reg_melted_rows = []
    unique_facs_set = set()
    all_years_rows = []

    for yr in [2020, 2021, 2022, 2023, 2024]:
        df_yr = load_regional_facilities(yr)
        if not df_yr.empty:
            stk_raw_arr = pd.to_numeric(df_yr.get('52. 5.2 - STACK AIR', pd.Series([0]*len(df_yr))), errors='coerce').fillna(0)
            fug_raw_arr = pd.to_numeric(df_yr.get('51. 5.1 - FUGITIVE AIR', pd.Series([0]*len(df_yr))), errors='coerce').fillna(0)
            
            df_yr['Stack'] = stk_raw_arr
            df_yr['Fugitive'] = fug_raw_arr
            df_yr['Total'] = df_yr['Stack'] + df_yr['Fugitive']
            
            for idx, r in df_yr.iterrows():
                tri_id = str(r.get('1. TRIID', r.get('TRIFD', ''))).strip()
                fname = str(r.get('4. FACILITY NAME', '')).strip().upper()
                fcity = str(r.get('6. CITY', '')).strip().title()
                fst = str(r.get('8. ST', '')).strip().upper()
                if fname and fname != 'NAN':
                    unique_facs_set.add((fname, fcity, fst))
                    key = tri_id if tri_id and tri_id != 'NAN' else f"{fname}_{fcity}_{fst}"
                    
                    s_val = r.get('52. 5.2 - STACK AIR', None)
                    stk = float(s_val) if pd.notna(s_val) else 0.0

                    f_val = r.get('51. 5.1 - FUGITIVE AIR', None)
                    fug = float(f_val) if pd.notna(f_val) else 0.0

                    tot = float(stk + fug)
                    all_years_rows.append({
                        "Key": key,
                        "Facility Name": fname,
                        "Municipality": fcity,
                        "State": fst,
                        "Facility Label": f"{fname} ({fcity}, {fst})",
                        "Year": str(yr),
                        "Total Emissions (lbs)": round(tot, 2)
                    })
            
            stk_sum = df_yr['Stack'].sum()
            fug_sum = df_yr['Fugitive'].sum()
            total_sum = stk_sum + fug_sum
            
            summary_data.append({
                "Year": str(yr),
                "Active Facilities": len(df_yr),
                "Total Stack Air (lbs)": round(stk_sum, 2),
                "Total Fugitive Air (lbs)": round(fug_sum, 2),
                "Combined Emissions (lbs)": round(total_sum, 2)
            })

            reg_melted_rows.append({"Year": str(yr), "Category": "Stack Air", "Emissions (lbs)": round(stk_sum, 2)})
            reg_melted_rows.append({"Year": str(yr), "Category": "Fugitive Air", "Emissions (lbs)": round(fug_sum, 2)})
            reg_melted_rows.append({"Year": str(yr), "Category": "Total Emissions", "Emissions (lbs)": round(total_sum, 2)})

            nj_df = df_yr[df_yr['8. ST'].astype(str).str.upper().str.strip() == 'NJ']
            nj_stk = nj_df['Stack'].sum() if not nj_df.empty else 0.0
            nj_fug = nj_df['Fugitive'].sum() if not nj_df.empty else 0.0
            nj_tot = nj_stk + nj_fug
            
            nj_summary_data.append({
                "Year": str(yr),
                "NJ Facilities": len(nj_df),
                "NJ Stack Air (lbs)": round(nj_stk, 2),
                "NJ Fugitive Air (lbs)": round(nj_fug, 2),
                "NJ Combined Emissions (lbs)": round(nj_tot, 2)
            })

            nj_melted_rows.append({"Year": str(yr), "Category": "Stack Air", "Emissions (lbs)": round(nj_stk, 2)})
            nj_melted_rows.append({"Year": str(yr), "Category": "Fugitive Air", "Emissions (lbs)": round(nj_fug, 2)})
            nj_melted_rows.append({"Year": str(yr), "Category": "Total Emissions", "Emissions (lbs)": round(nj_tot, 2)})

    df_nj_chart = pd.DataFrame(nj_summary_data)
    df_nj_melted = pd.DataFrame(nj_melted_rows)
    df_reg_chart = pd.DataFrame(summary_data)
    df_reg_melted = pd.DataFrame(reg_melted_rows)

    st.markdown("#### 📌 Key Performance Indicators (2024 Overview)")
    latest_nj_row = df_nj_chart[df_nj_chart['Year'] == '2024']
    latest_reg_row = df_reg_chart[df_reg_chart['Year'] == '2024']
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Active NJ Facilities", f"{int(latest_nj_row['NJ Facilities'].values[0])}" if not latest_nj_row.empty else "N/A")
    col_m2.metric("NJ Total Emissions", f"{float(latest_nj_row['NJ Combined Emissions (lbs)'].values[0]):,.1f} lbs" if not latest_nj_row.empty else "N/A")
    col_m3.metric("Regional Facilities", f"{int(latest_reg_row['Active Facilities'].values[0])}" if not latest_reg_row.empty else "N/A")
    col_m4.metric("Regional Total Air Load", f"{float(latest_reg_row['Combined Emissions (lbs)'].values[0]):,.1f} lbs" if not latest_reg_row.empty else "N/A")

    st.success("🎯 **Impact Insight:** Regional industrial emissions across tracked airsheds demonstrate multi-year accountability tracking, highlighting measurable emission shifts between 2020 baselines and 2024 reports.")

    st.markdown("---")

    st.markdown("#### 🏆 Top Emitting Facilities (Latest Reporting Year 2024)")
    st.info("💡 **Impact Visual:** Ranking top industrial facilities by total air release volume to instantly highlight primary contributors.")
    
    df_latest_facs = load_regional_facilities(2024)
    if not df_latest_facs.empty:
        df_latest_facs['Facility_Name'] = df_latest_facs['4. FACILITY NAME'].astype(str).str.upper().str.strip()
        df_latest_facs['City'] = df_latest_facs['6. CITY'].astype(str).str.strip().str.title()
        df_latest_facs['State'] = df_latest_facs['8. ST'].astype(str).str.strip().str.upper()
        df_latest_facs['Facility_Label'] = df_latest_facs['Facility_Name'] + " (" + df_latest_facs['City'] + ", " + df_latest_facs['State'] + ")"
        
        stk_l = pd.to_numeric(df_latest_facs.get('52. 5.2 - STACK AIR', 0), errors='coerce').fillna(0)
        fug_l = pd.to_numeric(df_latest_facs.get('51. 5.1 - FUGITIVE AIR', 0), errors='coerce').fillna(0)
        df_latest_facs['Total_Emissions'] = stk_l + fug_l
        
        top_facs_df = df_latest_facs.groupby(['Facility_Label', 'Facility_Name', 'City', 'State'])['Total_Emissions'].sum().reset_index().sort_values(by='Total_Emissions', ascending=False).head(5)
        
        top_bar_chart = alt.Chart(top_facs_df).mark_bar(color="#2563EB", cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
            x=alt.X('Total_Emissions:Q', title='Total Emissions (lbs)'),
            y=alt.Y('Facility_Label:N', sort='-x', title=None, axis=alt.Axis(labelLimit=0)),
            tooltip=[
                alt.Tooltip('Facility_Name:N', title='Facility'),
                alt.Tooltip('City:N', title='City'),
                alt.Tooltip('State:N', title='State'),
                alt.Tooltip('Total_Emissions:Q', title='Total Emissions (lbs)', format=',.2f')
            ]
        ).properties(height=260)
        st.altair_chart(top_bar_chart, use_container_width=True)

    st.markdown("---")

    st.markdown("### 🏭 Macro & Individual Facility Emissions Analytics")
    st.info("💡 **Feature Highlight:** Explore macro aggregated trends for New Jersey and the broader regional airshed, or drill down into specific tracked industrial facilities.")

    sorted_facs = sorted(list(unique_facs_set), key=lambda x: (0 if x[2] == 'NJ' else 1, x[2], x[0]))
    facility_display_options = [
        "🗺️ All New Jersey Facilities (Aggregated)",
        "🌐 All Regional Airshed Facilities (Aggregated)"
    ]
    facility_map = {}
    for f in sorted_facs:
        label = f"🏭 {f[0]} ({f[1]}, {f[2]})"
        facility_display_options.append(label)
        facility_map[label] = f[0]

    selected_display_label = st.selectbox(
        "Select Macro View or Individual Facility:",
        facility_display_options,
        key="global_chart_facility_filter"
    )

    if selected_display_label == "🗺️ All New Jersey Facilities (Aggregated)":
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            st.markdown("##### 📊 New Jersey Emissions Breakdown (Stack, Fugitive & Total)")
            group_bar = alt.Chart(df_nj_melted).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
                x=alt.X('Year:N', title='Reporting Year', sort=['2020', '2021', '2022', '2023', '2024'], axis=alt.Axis(labelAngle=0)),
                xOffset=alt.X('Category:N', sort=['Stack Air', 'Fugitive Air', 'Total Emissions']),
                y=alt.Y('Emissions (lbs):Q', title='Emissions (lbs)'),
                color=alt.Color('Category:N',
                                scale=alt.Scale(
                                    domain=['Stack Air', 'Fugitive Air', 'Total Emissions'],
                                    range=['#93C5FD', '#3B82F6', '#1D4ED8']
                                ),
                                legend=alt.Legend(title="Emission Type")
                ),
                tooltip=['Year', 'Category', alt.Tooltip('Emissions (lbs):Q', format=',.2f')]
            )
            st.altair_chart(group_bar.properties(height=280), use_container_width=True)

        with col_chart2:
            st.markdown("##### 📈 New Jersey Total Emissions Trend")
            base_line = alt.Chart(df_nj_chart).mark_line(color="#2563EB", point=alt.OverlayMarkDef(size=70, filled=True, fill="#2563EB"), strokeWidth=3).encode(
                x=alt.X('Year:N', title='Reporting Year', sort=['2020', '2021', '2022', '2023', '2024'], axis=alt.Axis(labelAngle=0)),
                y=alt.Y('NJ Combined Emissions (lbs):Q', title='Combined Emissions (lbs)', scale=alt.Scale(zero=True)),
                tooltip=['Year', 'NJ Facilities', alt.Tooltip('NJ Combined Emissions (lbs):Q', format=',.2f')]
            )
            text_line = alt.Chart(df_nj_chart).mark_text(align='center', baseline='bottom', dy=-12, fontWeight='bold', color='#0F172A').encode(
                x=alt.X('Year:N', sort=['2020', '2021', '2022', '2023', '2024']),
                y=alt.Y('NJ Combined Emissions (lbs):Q'),
                text=alt.Text('NJ Combined Emissions (lbs):Q', format=',.1f')
            )
            st.altair_chart((base_line + text_line).properties(height=280), use_container_width=True)

    elif selected_display_label == "🌐 All Regional Airshed Facilities (Aggregated)":
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            st.markdown("##### 📊 Regional Airshed Emissions Breakdown (Stack, Fugitive & Total)")
            group_bar_reg = alt.Chart(df_reg_melted).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
                x=alt.X('Year:N', title='Reporting Year', sort=['2020', '2021', '2022', '2023', '2024'], axis=alt.Axis(labelAngle=0)),
                xOffset=alt.X('Category:N', sort=['Stack Air', 'Fugitive Air', 'Total Emissions']),
                y=alt.Y('Emissions (lbs):Q', title='Emissions (lbs)'),
                color=alt.Color('Category:N',
                                scale=alt.Scale(
                                    domain=['Stack Air', 'Fugitive Air', 'Total Emissions'],
                                    range=['#93C5FD', '#3B82F6', '#1D4ED8']
                                ),
                                legend=alt.Legend(title="Emission Type")
                ),
                tooltip=['Year', 'Category', alt.Tooltip('Emissions (lbs):Q', format=',.2f')]
            )
            st.altair_chart(group_bar_reg.properties(height=280), use_container_width=True)

        with col_chart2:
            st.markdown("##### 📈 Regional Airshed Total Emissions Trend")
            base_line_reg = alt.Chart(df_reg_chart).mark_line(color="#0EA5E9", point=alt.OverlayMarkDef(size=70, filled=True, fill="#0EA5E9"), strokeWidth=3).encode(
                x=alt.X('Year:N', title='Reporting Year', sort=['2020', '2021', '2022', '2023', '2024'], axis=alt.Axis(labelAngle=0)),
                y=alt.Y('Combined Emissions (lbs):Q', title='Combined Emissions (lbs)', scale=alt.Scale(zero=True)),
                tooltip=['Year', 'Active Facilities', alt.Tooltip('Combined Emissions (lbs):Q', format=',.2f')]
            )
            text_line_reg = alt.Chart(df_reg_chart).mark_text(align='center', baseline='bottom', dy=-12, fontWeight='bold', color='#0F172A').encode(
                x=alt.X('Year:N', sort=['2020', '2021', '2022', '2023', '2024']),
                y=alt.Y('Combined Emissions (lbs):Q'),
                text=alt.Text('Combined Emissions (lbs):Q', format=',.1f')
            )
            st.altair_chart((base_line_reg + text_line_reg).properties(height=280), use_container_width=True)

    else:
        target_facility_name = facility_map[selected_display_label]
        fac_trend_rows = []
        fac_melted_rows = []

        for yr_tr in [2020, 2021, 2022, 2023, 2024]:
            df_tr = load_regional_facilities(yr_tr)
            emissions_val = 0.0
            stk_val = 0.0
            fug_val = 0.0
            tot_val = 0.0
            fac_status = "Not Reported / Closed"

            if not df_tr.empty and '4. FACILITY NAME' in df_tr.columns:
                match = df_tr[df_tr['4. FACILITY NAME'].astype(str).str.upper().str.strip() == target_facility_name]
                if not match.empty:
                    stk_sum_val = pd.to_numeric(match.get('52. 5.2 - STACK AIR', 0), errors='coerce').sum()
                    stk_val = float(stk_sum_val) if pd.notna(stk_sum_val) else 0.0

                    fug_sum_val = pd.to_numeric(match.get('51. 5.1 - FUGITIVE AIR', 0), errors='coerce').sum()
                    fug_val = float(fug_sum_val) if pd.notna(fug_sum_val) else 0.0

                    tot_val = float(stk_val + fug_val)
                    emissions_val = tot_val
                    fac_status = "Active / Reporting"
            
            fac_trend_rows.append({"Year": str(yr_tr), "Total Emissions (lbs)": round(emissions_val, 2), "Status": fac_status})
            fac_melted_rows.append({"Year": str(yr_tr), "Category": "Stack Air", "Emissions (lbs)": round(stk_val, 2)})
            fac_melted_rows.append({"Year": str(yr_tr), "Category": "Fugitive Air", "Emissions (lbs)": round(fug_val, 2)})
            fac_melted_rows.append({"Year": str(yr_tr), "Category": "Total Emissions", "Emissions (lbs)": round(tot_val, 2)})
        
        df_single_fac = pd.DataFrame(fac_trend_rows)
        df_single_melted = pd.DataFrame(fac_melted_rows)
        clean_fac_label = selected_display_label.replace("🏭 ", "")

        col_fc1, col_fc2 = st.columns(2)
        with col_fc1:
            st.markdown(f"##### 📊 {clean_fac_label} - Emissions Breakdown")
            group_bar_fac = alt.Chart(df_single_melted).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
                x=alt.X('Year:N', title='Reporting Year', sort=['2020', '2021', '2022', '2023', '2024'], axis=alt.Axis(labelAngle=0)),
                xOffset=alt.X('Category:N', sort=['Stack Air', 'Fugitive Air', 'Total Emissions']),
                y=alt.Y('Emissions (lbs):Q', title='Emissions (lbs)'),
                color=alt.Color('Category:N',
                                scale=alt.Scale(
                                    domain=['Stack Air', 'Fugitive Air', 'Total Emissions'],
                                    range=['#93C5FD', '#3B82F6', '#1D4ED8']
                                ),
                                legend=alt.Legend(title="Emission Type")
                ),
                tooltip=['Year', 'Category', alt.Tooltip('Emissions (lbs):Q', format=',.2f')]
            )
            st.altair_chart(group_bar_fac.properties(height=280), use_container_width=True)

        with col_fc2:
            st.markdown(f"##### 📈 {clean_fac_label} - Total Emissions Trend")
            base_line_fac = alt.Chart(df_single_fac).mark_line(color="#2563EB", point=alt.OverlayMarkDef(size=70, filled=True, fill="#2563EB"), strokeWidth=3).encode(
                x=alt.X('Year:N', title='Reporting Year', sort=['2020', '2021', '2022', '2023', '2024'], axis=alt.Axis(labelAngle=0)),
                y=alt.Y('Total Emissions (lbs):Q', title='Annual Emissions (lbs)', scale=alt.Scale(zero=True)),
                tooltip=['Year', 'Total Emissions (lbs)', 'Status']
            )
            text_line_fac = alt.Chart(df_single_fac).mark_text(align='center', baseline='bottom', dy=-12, fontWeight='bold', color='#0F172A').encode(
                x=alt.X('Year:N', sort=['2020', '2021', '2022', '2023', '2024']),
                y=alt.Y('Total Emissions (lbs):Q'),
                text=alt.Text('Total Emissions (lbs):Q', format=',.1f')
            )
            st.altair_chart((base_line_fac + text_line_fac).properties(height=280), use_container_width=True)

        st.dataframe(df_single_fac.set_index("Year"), use_container_width=True)

    st.markdown("---")
    st.markdown("### 🌐 Comprehensive Multi-Year Facility Emissions Matrix (2020–2024)")
    st.info("💡 **Data Integrity Update:** All tracked regional facilities across 2020–2024 are fully compiled and displayed below with absolute shades-of-gray cell coloring matching the legend exactly.")

    df_all_years = pd.DataFrame(all_years_rows)

    if not df_all_years.empty:
        pivoted_delta = df_all_years.pivot_table(
            index=["Facility Name", "Municipality", "State", "Facility Label"], 
            columns="Year", 
            values="Total Emissions (lbs)", 
            aggfunc="sum",
            fill_value=0.0
        ).reset_index()
        
        for yr_str in ['2020', '2021', '2022', '2023', '2024']:
            if yr_str not in pivoted_delta.columns:
                pivoted_delta[yr_str] = 0.0

        pivoted_delta['Is_NJ'] = pivoted_delta['State'].apply(lambda s: 0 if str(s).strip().upper() == 'NJ' else 1)
        pivoted_delta = pivoted_delta.sort_values(by=['Is_NJ', 'State', 'Facility Name']).drop(columns=['Is_NJ'])

        pivoted_delta['Net Change'] = pivoted_delta['2024'] - pivoted_delta['2020']
        total_net_shift = pivoted_delta['Net Change'].sum()
        num_increased = len(pivoted_delta[pivoted_delta['Net Change'] > 0])
        num_decreased = len(pivoted_delta[pivoted_delta['Net Change'] < 0])

        col_s1, col_s2, col_s3 = st.columns(3)
        col_s1.metric("Net Regional Shift (lbs)", f"{total_net_shift:+,.2f} lbs")
        col_s2.metric("Facilities with Net Increase", f"{num_increased}")
        col_s3.metric("Facilities with Net Decrease", f"{num_decreased}")

        st.markdown("#### 💨 Emissions Intensity Color Legend")
        st.markdown("""
        <div style="display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin-bottom: 16px; font-size: 0.78rem;">
            <span style="background-color: #F8FAFC; border: 1px solid #CBD5E1; color: #0F172A; padding: 3px 8px; border-radius: 4px;">0 lbs</span>
            <span style="background-color: #F1F5F9; border: 1px solid #CBD5E1; color: #0F172A; padding: 3px 8px; border-radius: 4px;">1–49 lbs</span>
            <span style="background-color: #E2E8F0; border: 1px solid #94A3B8; color: #0F172A; padding: 3px 8px; border-radius: 4px;">50–199 lbs</span>
            <span style="background-color: #CBD5E1; border: 1px solid #64748B; color: #0F172A; padding: 3px 8px; border-radius: 4px;">200–499 lbs</span>
            <span style="background-color: #94A3B8; color: #FFFFFF; padding: 3px 8px; border-radius: 4px; font-weight: 500;">500–999 lbs</span>
            <span style="background-color: #475569; color: #FFFFFF; padding: 3px 8px; border-radius: 4px; font-weight: 500;">1,000–2,499 lbs</span>
            <span style="background-color: #1E293B; color: #FFFFFF; padding: 3px 8px; border-radius: 4px; font-weight: 500;">2,500–4,068 lbs</span>
        </div>
        """, unsafe_allow_html=True)

        display_df = pivoted_delta.set_index("Facility Label")[['Facility Name', 'Municipality', 'State', '2020', '2021', '2022', '2023', '2024', 'Net Change']]
        
        def color_emissions_gray(val):
            try:
                num = float(val)
            except:
                return ''
            if num <= 0:
                return 'background-color: #F8FAFC; color: #0F172A;'
            elif num < 50:
                return 'background-color: #F1F5F9; color: #0F172A;'
            elif num < 200:
                return 'background-color: #E2E8F0; color: #0F172A;'
            elif num < 500:
                return 'background-color: #CBD5E1; color: #0F172A;'
            elif num < 1000:
                return 'background-color: #94A3B8; color: #FFFFFF;'
            elif num < 2500:
                return 'background-color: #475569; color: #FFFFFF;'
            else:
                return 'background-color: #1E293B; color: #FFFFFF;'

        def color_net_change(val):
            return 'color: #0F172A; font-weight: 500;'

        styled_table = display_df.style.map(
            color_emissions_gray, subset=['2020', '2021', '2022', '2023', '2024']
        ).map(
            color_net_change, subset=['Net Change']
        ).format({
            '2020': '{:,.2f}',
            '2021': '{:,.2f}',
            '2022': '{:,.2f}',
            '2023': '{:,.2f}',
            '2024': '{:,.2f}',
            'Net Change': '{:+,.2f}'
        })

        st.table(styled_table)

# --- TAB 4: ABOUT ECORISK NJ ---
with tab_about:
    st.markdown("### 💡 About EcoRisk NJ & Scientific Methodology")
    
    st.markdown("### 🧬 What is Ethylene Oxide ($C_2H_4O$) & Why We Chose It")
    st.markdown("""
    * **The Chemical & Classification:** Ethylene Oxide ($C_2H_4O$) is a highly reactive, ring-shaped gas widely used in industrial manufacturing and medical equipment sterilization. It is officially classified as a **Group 1 (Tier 1) human carcinogen** due to robust toxicological and epidemiological evidence linking it to cancer.
    * **Molecular Mechanism:** Because of its strained chemical ring, Ethylene Oxide reacts directly with biological molecules through **DNA alkylation** (forming N7-HEG adducts) without requiring prior metabolic activation, leading to potential genetic mutations in tumor suppressor genes.
    * **Health & Public Risk:** Long-term exposure to Ethylene Oxide is strongly associated with an increased risk of lymphohematopoietic cancers (such as leukemia and lymphoma) as well as breast cancer. 
    * **Why Ethylene Oxide for EcoRisk NJ:** Traditional municipal zoning and broad county-level averages often conceal micro-spatial risk spikes near industrial corridors. Ethylene Oxide was chosen as the primary pollutant for this application because high-resolution spatial modeling is essential to identify localized exposure hotspots that standard monitoring networks might otherwise miss.
    """)

    st.markdown("---")
    st.markdown("### 💡 Why EcoRisk NJ Was Created & Programming Breakdown")
    st.markdown("""
    * **The Environmental Problem:** Industrial air toxics—specifically Ethylene Oxide—pose significant long-term health risks to communities living near sterilization facilities and chemical plants. However, raw environmental data is often fragmented across complex government databases, making it difficult for everyday citizens and community advocates to assess exposure.
    * **Our Mission:** EcoRisk NJ bridges this gap by transforming complex geospatial risk models and multi-year industrial reporting into an intuitive, high-performance web application that empowers communities with transparent environmental intelligence.
    * **Personal Programming Contributions:** Designed and engineered entirely from scratch, including high-performance spatial querying with GeoPandas and Shapely, custom QGIS-based color mapping algorithms, vectorized NumPy calculations for instantaneous radius buffering, and a polished SaaS-grade Streamlit interface with synchronized multi-year facility matrices.
    """)

    st.markdown("---")
    st.markdown("### 🔬 Scientific Methodology & Health Risk Context")
    st.info("💡 **Note on Risk Modeling:** Tab 1 displays the **2020 EPA AirToxScreen census block modeling dataset** (estimated theoretical long-term exposure risks). This is a separate dataset from annual TRI emissions reporting and represents modeled inhalation risk rather than a direct measurement of current exposure or individual cancer incidence.")

    st.markdown("---")
    st.markdown("### 📚 Dataset Provenance & Clickable Source Citations")
    st.markdown("""
    To ensure complete scientific transparency for the **Congressional App Challenge**, **EcoRisk NJ** relies on official government datasets:
    * **1. [EPA AirToxScreen (2020 Model)](https://www.epa.gov/AirToxScreen/2020-airtoxscreen):** Census-block level cancer risk modeling and inhalation exposure assessments.
    * **2. [EPA Toxics Release Inventory (TRI, 2020–2024)](https://www.epa.gov/toxics-release-inventory-tri-program):** Multi-year industrial facility air emissions reporting tracking stack and fugitive releases.
    * **3. Cartography & Spatial Engine:** GeoPandas, Shapely, and Folium.
    """)

    st.markdown("---")
    st.markdown("### 📋 Community Action & Policy Resources (Take Action)")
    st.markdown("""
    * **NJDEP Data Miner:** [Review facility air permits](https://njems.nj.gov/dataminer).
    * **EPA AirToxScreen Portal:** [Learn more about national assessments](https://www.epa.gov/AirToxScreen/airtoxscreen-overview).
    """)

# --- TAB 5: MEET THE DEVELOPER ---
with tab_dev:
    st.markdown("### 👨‍💻 Meet the Developer")
    
    with st.container(border=True):
        col_img, col_bio = st.columns([1, 2.5])
        
        with col_img:
            photo_path = "Anvay_Shukla_photo.png"
            if os.path.exists(photo_path):
                st.image(photo_path, width=200, caption="Anvay Shukla — Lead Developer")
            else:
                st.info("📷 Developer Photo Available")
            
            st.markdown("""
            **🔗 Connect with Developer:**
            * 💼 **[LinkedIn Profile](https://www.linkedin.com/in/anvay-shukla-a14b892aa/)**
            * 💻 **[GitHub Repository](https://github.com/anvayshukla)**
            """, unsafe_allow_html=True)

        with col_bio:
            st.markdown("#### Anvay Shukla")
            st.markdown("**Junior (Grade 11) | J.P. Stevens High School — Edison, NJ**")
            st.markdown("*Aspiring Physician-Scientist & Computational Biology Researcher*")
            
            st.markdown("""
            * **Clinical & Healthcare Background:** 200+ hours of clinical support volunteering across **Hackensack Meridian Health** and **Raritan Bay Diagnostic Imaging**, assisting in diagnostic workflows, EMR intake, and patient care.
            * **Research Focus:** Lead researcher investigating Ethylene Oxide-induced DNA alkylation, spatial cancer modeling, and cellular stress dynamics.
            * **Community Leadership:** Founder & Chapter Lead of Edison Leo Club (Lions Clubs International).
            """)

    st.markdown("---")
    
    st.markdown("### 🛑 The Motivation Behind EcoRisk NJ")
    st.markdown("""
    **Why build EcoRisk NJ?** The inspiration is deeply personal. Growing up with generational awareness of historical industrial disasters—specifically the **[Bhopal Gas Tragedy](https://en.wikipedia.org/wiki/Bhopal_disaster)** that impacted family history (where relatives, including his grandfather, lived through and narrowly escaped the disaster zone)—developer Anvay Shukla recognized a silent, recurring danger: communities often live right next to hazardous industrial zones without truly knowing what is in the air they breathe.
    
    While state and federal data exist, they are often buried in complex government databases. **EcoRisk NJ** was built to bridge this transparency gap—transforming raw toxic release inventories and computational cancer risk models into an intuitive public health platform that empowers residents, patients, and local advocates before environmental risks turn into health crises.
    """)

# --- TAB 6: RESEARCH PAPER ---
with tab_paper:
    st.markdown("### 📄 Official Research Paper & Documentation")
    st.markdown("Read the complete formal research study underlying **EcoRisk NJ**, prepared for the **Congressional App Challenge**.")
    
    pdf_filename = "EcoRisk_NJ_Ethylene_Oxide_Cancer_Research_2026.pdf"
    
    if os.path.exists(pdf_filename):
        with open(pdf_filename, "rb") as pdf_file:
            pdf_bytes = pdf_file.read()
            
        st.download_button(
            label="📥 Download Official Research Paper (PDF)",
            data=pdf_bytes,
            file_name=pdf_filename,
            mime="application/pdf",
            use_container_width=True
        )
    
    st.markdown("---")
    st.markdown("### 📖 Research Paper Pages (1–13)")

    # Responsive CSS: Default 900px on Desktop, 100% on Mobile
    st.markdown("""
    <style>
    .research-img-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 20px;
    }
    .research-img-container img {
        max-width: 900px !important; /* Desktop: Default 900px width */
        width: 100% !important;     /* Mobile: Auto 100% full width */
        height: auto !important;
        border-radius: 8px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.12);
        border: 1px solid #CBD5E1;
    }
    </style>
    """, unsafe_allow_html=True)
    
    img_folder = "Research Paper Images"
    if os.path.exists(img_folder):
        for i in range(1, 14):
            img_path_png = os.path.join(img_folder, f"{i}.png")
            img_path_jpg = os.path.join(img_folder, f"{i}.jpg")
            
            target_path = img_path_png if os.path.exists(img_path_png) else (img_path_jpg if os.path.exists(img_path_jpg) else None)
            
            if target_path:
                with open(target_path, "rb") as img_file:
                    encoded_img = base64.b64encode(img_file.read()).decode()
                
                st.markdown(f"""
                <div class="research-img-container">
                    <div>
                        <a href="data:image/png;base64,{encoded_img}" target="_blank" title="Click to view full image in new tab">
                            <img src="data:image/png;base64,{encoded_img}" alt="Page {i}">
                        </a>
                        <p style="text-align: center; font-size: 0.85rem; color: #64748B; margin-top: 6px;">Page {i} of 13 — <i>(Click image to open full resolution)</i></p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("---")
    else:
        st.warning(f"⚠️ Folder `{img_folder}` was not found. Please make sure the folder containing pages 1 to 13 is placed in your project root.")

st.markdown("---")
st.caption("EcoRisk NJ | Designed for community impact.")