import streamlit as st
import numpy as np
import pandas as pd
from dataclasses import dataclass
import math
import plotly.express as px
import plotly.graph_objects as go

# 既存のクラス定義をここに含める（EarthquakeParameters, RegionData, NankaiEarthquakeSimulator）
# ※ 前のコードの@dataclassとクラス定義をそのまま使用

@dataclass
class EarthquakeParameters:
    """地震パラメータ"""
    magnitude: float
    depth: float
    epicenter_lat: float
    epicenter_lon: float

@dataclass
class RegionData:
    """地域データ"""
    name: str
    lat: float
    lon: float
    population: int
    buildings: int
    wooden_ratio: float
    coastal: bool
    elevation: float

class NankaiEarthquakeSimulator:
    # 前のコードのNankaiEarthquakeSimulatorクラスをそのまま使用
    def __init__(self):
        self.regions = [
            RegionData("静岡市", 34.9756, 138.3827, 700000, 280000, 0.4, True, 20),
            RegionData("浜松市", 34.7108, 137.7261, 800000, 320000, 0.45, True, 10),
            RegionData("名古屋市", 35.1815, 136.9066, 2300000, 920000, 0.3, False, 15),
            RegionData("津市", 34.7185, 136.5056, 280000, 112000, 0.5, True, 5),
            RegionData("大阪市", 34.6937, 135.5023, 2700000, 1080000, 0.25, True, 5),
            RegionData("和歌山市", 34.2306, 135.1708, 360000, 144000, 0.45, True, 8),
            RegionData("高知市", 33.5597, 133.5311, 330000, 132000, 0.5, True, 3),
            RegionData("徳島市", 34.0658, 134.5594, 260000, 104000, 0.48, True, 2),
        ]
    
    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        R = 6371
        dLat = math.radians(lat2 - lat1)
        dLon = math.radians(lon2 - lon1)
        a = math.sin(dLat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dLon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return R * c
    
    def calculate_seismic_intensity(self, earthquake: EarthquakeParameters, region: RegionData) -> float:
        distance = self.calculate_distance(
            earthquake.epicenter_lat, earthquake.epicenter_lon,
            region.lat, region.lon
        )
        if distance < 1:
            distance = 1
        intensity = earthquake.magnitude - 3.5 * math.log10(distance) - 0.006 * distance + 1.5
        depth_factor = 1 - (earthquake.depth / 100) * 0.3
        intensity *= depth_factor
        return min(7, max(0, intensity))
    
    def estimate_building_damage(self, intensity: float, region: RegionData) -> dict:
        if intensity >= 7:
            collapse_rate = 0.3 * region.wooden_ratio + 0.05 * (1 - region.wooden_ratio)
            severe_rate = 0.4 * region.wooden_ratio + 0.15 * (1 - region.wooden_ratio)
        elif intensity >= 6.5:
            collapse_rate = 0.15 * region.wooden_ratio + 0.02 * (1 - region.wooden_ratio)
            severe_rate = 0.3 * region.wooden_ratio + 0.1 * (1 - region.wooden_ratio)
        elif intensity >= 6:
            collapse_rate = 0.05 * region.wooden_ratio + 0.005 * (1 - region.wooden_ratio)
            severe_rate = 0.15 * region.wooden_ratio + 0.05 * (1 - region.wooden_ratio)
        elif intensity >= 5.5:
            collapse_rate = 0.01 * region.wooden_ratio
            severe_rate = 0.05 * region.wooden_ratio + 0.01 * (1 - region.wooden_ratio)
        else:
            collapse_rate = 0
            severe_rate = 0.01 * region.wooden_ratio if intensity >= 5 else 0
        
        collapsed = int(region.buildings * collapse_rate)
        severe_damage = int(region.buildings * severe_rate)
        moderate_damage = int(region.buildings * severe_rate * 0.5)
        
        return {
            "全壊": collapsed,
            "半壊": severe_damage,
            "一部損壊": moderate_damage,
            "被害なし": region.buildings - collapsed - severe_damage - moderate_damage
        }
    
    def estimate_tsunami(self, earthquake: EarthquakeParameters, region: RegionData) -> dict:
        if not region.coastal:
            return {"津波高": 0, "到達時間": 0, "浸水面積率": 0}
        
        distance = self.calculate_distance(
            earthquake.epicenter_lat, earthquake.epicenter_lon,
            region.lat, region.lon
        )
        
        if earthquake.magnitude >= 8.0:
            base_height = 10
        elif earthquake.magnitude >= 7.5:
            base_height = 5
        else:
            base_height = 2
        
        tsunami_height = base_height * math.exp(-distance / 500)
        
        if region.elevation > tsunami_height:
            tsunami_height = 0
        
        arrival_time = distance / 12
        
        if tsunami_height > 5:
            inundation_rate = 0.3
        elif tsunami_height > 2:
            inundation_rate = 0.15
        elif tsunami_height > 0:
            inundation_rate = 0.05
        else:
            inundation_rate = 0
        
        return {
            "津波高": round(tsunami_height, 1),
            "到達時間": round(arrival_time, 0),
            "浸水面積率": inundation_rate
        }
    
    def estimate_casualties(self, intensity: float, building_damage: dict, 
                          tsunami: dict, region: RegionData) -> dict:
        building_deaths = int(building_damage["全壊"] * 0.01)
        building_injuries = int((building_damage["全壊"] + building_damage["半壊"]) * 0.05)
        
        if tsunami["津波高"] > 0:
            affected_population = int(region.population * tsunami["浸水面積率"])
            evacuation_rate = min(0.8, tsunami["到達時間"] / 60)
            non_evacuated = affected_population * (1 - evacuation_rate)
            
            tsunami_deaths = int(non_evacuated * 0.1 if tsunami["津波高"] > 2 else non_evacuated * 0.02)
            tsunami_injuries = int(non_evacuated * 0.2)
        else:
            tsunami_deaths = 0
            tsunami_injuries = 0
        
        return {
            "死者": building_deaths + tsunami_deaths,
            "重傷者": int((building_injuries + tsunami_injuries) * 0.3),
            "軽傷者": int((building_injuries + tsunami_injuries) * 0.7)
        }
    
    def estimate_infrastructure_damage(self, intensity: float, region: RegionData) -> dict:
        if intensity >= 6.5:
            electricity_outage = 0.8
            water_outage = 0.9
            gas_outage = 0.85
        elif intensity >= 6:
            electricity_outage = 0.5
            water_outage = 0.6
            gas_outage = 0.55
        elif intensity >= 5.5:
            electricity_outage = 0.2
            water_outage = 0.3
            gas_outage = 0.25
        else:
            electricity_outage = 0.05 if intensity >= 5 else 0
            water_outage = 0.05 if intensity >= 5 else 0
            gas_outage = 0.05 if intensity >= 5 else 0
        
        return {
            "停電率": electricity_outage,
            "断水率": water_outage,
            "ガス供給停止率": gas_outage
        }
    
    def simulate(self, earthquake: EarthquakeParameters) -> pd.DataFrame:
        results = []
        
        for region in self.regions:
            intensity = self.calculate_seismic_intensity(earthquake, region)
            building_damage = self.estimate_building_damage(intensity, region)
            tsunami = self.estimate_tsunami(earthquake, region)
            casualties = self.estimate_casualties(intensity, building_damage, tsunami, region)
            infrastructure = self.estimate_infrastructure_damage(intensity, region)
            
            economic_loss = (
                building_damage["全壊"] * 20000000 +
                building_damage["半壊"] * 10000000 +
                building_damage["一部損壊"] * 2000000
            ) / 100000000
            
            results.append({
                "地域": region.name,
                "推定震度": round(intensity, 1),
                "全壊建物": building_damage["全壊"],
                "半壊建物": building_damage["半壊"],
                "津波高(m)": tsunami["津波高"],
                "津波到達時間(分)": tsunami["到達時間"],
                "死者": casualties["死者"],
                "負傷者": casualties["重傷者"] + casualties["軽傷者"],
                "停電世帯率": f"{infrastructure['停電率']*100:.0f}%",
                "断水世帯率": f"{infrastructure['断水率']*100:.0f}%",
                "経済被害(億円)": round(economic_loss, 0)
            })
        
        return pd.DataFrame(results)

# Streamlitアプリケーション
def main():
    st.set_page_config(
        page_title="南海トラフ地震被害シミュレーター",
        page_icon="🌊",
        layout="wide"
    )
    
    st.title("🌊 南海トラフ地震被害シミュレーター")
    st.markdown("地震のパラメータを調整して、被害を予測します")
    
    # サイドバーで地震パラメータを設定
    with st.sidebar:
        st.header("地震パラメータ設定")
        
        magnitude = st.slider(
            "マグニチュード",
            min_value=7.0,
            max_value=9.0,
            value=8.7,
            step=0.1,
            help="地震の規模を表します"
        )
        
        depth = st.slider(
            "震源深さ (km)",
            min_value=5,
            max_value=50,
            value=10,
            step=5,
            help="浅いほど被害が大きくなります"
        )
        
        epicenter_lat = st.number_input(
            "震源緯度",
            min_value=30.0,
            max_value=36.0,
            value=33.0,
            step=0.5
        )
        
        epicenter_lon = st.number_input(
            "震源経度",
            min_value=130.0,
            max_value=140.0,
            value=136.0,
            step=0.5
        )
        
        simulate_button = st.button("シミュレーション実行", type="primary")
    
    # メインエリア
    if simulate_button:
        # シミュレーション実行
        simulator = NankaiEarthquakeSimulator()
        earthquake = EarthquakeParameters(
            magnitude=magnitude,
            depth=depth,
            epicenter_lat=epicenter_lat,
            epicenter_lon=epicenter_lon
        )
        
        with st.spinner('シミュレーション中...'):
            results = simulator.simulate(earthquake)
        
        # 結果表示
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "推定死者数",
                f"{results['死者'].sum():,}人",
                delta=None,
                delta_color="inverse"
            )
        
        with col2:
            st.metric(
                "全壊建物",
                f"{results['全壊建物'].sum():,}棟"
            )
        
        with col3:
            st.metric(
                "負傷者数",
                f"{results['負傷者'].sum():,}人"
            )
        
        with col4:
            st.metric(
                "経済被害",
                f"{results['経済被害(億円)'].sum():,.0f}億円"
            )
        
        # グラフ表示
        st.subheader("📊 地域別被害状況")
        
        tab1, tab2, tab3 = st.tabs(["人的被害", "建物被害", "津波情報"])
        
        with tab1:
            fig = px.bar(
                results,
                x="地域",
                y=["死者", "負傷者"],
                title="地域別人的被害",
                labels={"value": "人数", "variable": "被害種別"}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            fig = px.bar(
                results,
                x="地域",
                y=["全壊建物", "半壊建物"],
                title="地域別建物被害",
                labels={"value": "棟数", "variable": "被害種別"}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            tsunami_data = results[results["津波高(m)"] > 0]
            if not tsunami_data.empty:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=tsunami_data["地域"],
                    y=tsunami_data["津波高(m)"],
                    name="津波高(m)",
                    yaxis="y"
                ))
                fig.add_trace(go.Scatter(
                    x=tsunami_data["地域"],
                    y=tsunami_data["津波到達時間(分)"],
                    name="到達時間(分)",
                    yaxis="y2",
                    mode="lines+markers"
                ))
                fig.update_layout(
                    title="津波被害予測",
                    yaxis=dict(title="津波高 (m)"),
                    yaxis2=dict(title="到達時間 (分)", overlaying="y", side="right")
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("この震源位置では津波被害は予測されません")
        
        # 詳細データテーブル
        st.subheader("📋 詳細データ")
        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True
        )
        
        # CSVダウンロード
        csv = results.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 CSVファイルをダウンロード",
            data=csv,
            file_name="earthquake_simulation_results.csv",
            mime="text/csv"
        )
    
    else:
        # 初期画面
        st.info("👈 左のサイドバーで地震パラメータを設定し、「シミュレーション実行」ボタンをクリックしてください")
        
        # 説明
        with st.expander("このシミュレーターについて"):
            st.markdown("""
            ### 機能
            - 南海トラフ地震の被害を推定します
            - マグニチュード、震源深さ、震源位置を調整可能
            - 地域別の被害状況を可視化
            
            ### 推定項目
            - 建物被害（全壊・半壊）
            - 人的被害（死者・負傷者）
            - 津波被害（津波高・到達時間）
            - インフラ被害（停電・断水）
            - 経済被害
            
            ### 注意事項
            - この推定は簡易モデルに基づくものです
            - 実際の被害は様々な要因により変動します
            - 防災計画の参考程度にご利用ください
            """)

if __name__ == "__main__":
    main()
