import streamlit as st
import geemap.foliumap as geemap
import ee
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from datetime import datetime

# 페이지 기본 설정
st.set_page_config(
    page_title="하늘에서 농업을 보(고싶)다",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일 추가
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stSelectbox {
        border-radius: 5px;
    }
    .plot-container {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1 {
        color: #2E7D32;
        padding: 1rem 0;
        border-bottom: 2px solid #4CAF50;
    }
    h3 {
        color: #1B5E20;
        margin-top: 2rem;
    }
    .stAlert {
        border-radius: 5px;
    }
    .sidebar .sidebar-content {
        background-color: #E8F5E9;
    }
    </style>
    """, unsafe_allow_html=True)

# Google Earth Engine 초기화
try:
    ee.Initialize()
except Exception as e:
    ee.Authenticate()
    ee.Initialize()

# 사이드바 디자인
st.sidebar.image("https://github.com/user-attachments/assets/df735a7e-7656-4634-912f-1f83b53c9d53", width=150)
st.sidebar.title("메뉴")
page = st.sidebar.radio(
    "분석 유형 선택",
    ["🗺️ 지도 시각화", "📊 통계 시각화", "🌿 양분 현황 분석"],
    key="navigation"
)

# 다각형 좌표
coords = [
    (127.129714, 35.850595),
    (127.129852, 35.850432),
    (127.129041, 35.849884),
    (127.128887, 35.850039),
]
polygon = ee.Geometry.Polygon([coords])

# 구름 마스킹 함수
def mask_clouds(image):
    cloud_mask = image.select("MSK_CLASSI_OPAQUE").Not()
    return image.updateMask(cloud_mask)

# 각 지수를 계산하는 함수
def calculate_index(image, index):
    if index == "NDVI":
        result = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    elif index == "RVI":
        result = image.select("B8").divide(image.select("B4")).rename("RVI")
    elif index == "MTVI1":
        red = image.select("B4")
        green = image.select("B3")
        nir = image.select("B8")
        numerator = nir.subtract(green).multiply(1.2).subtract(red.subtract(green).multiply(2.5))
        result = numerator.multiply(1.2).rename("MTVI1")
    elif index == "GNDVI":
        result = image.normalizedDifference(["B8", "B3"]).rename("GNDVI")
    elif index == "GRVI":
        result = image.normalizedDifference(["B3", "B4"]).rename("GRVI")

    # 0~1 범위로 정규화
    min_value = result.reduceRegion(reducer=ee.Reducer.min(), geometry=polygon, scale=10, maxPixels=1e13).get(index)
    max_value = result.reduceRegion(reducer=ee.Reducer.max(), geometry=polygon, scale=10, maxPixels=1e13).get(index)

    # 최소값과 최대값이 None이 아닌지 확인
    min_value = ee.Algorithms.If(ee.Algorithms.IsEqual(min_value, None), 0, min_value)
    max_value = ee.Algorithms.If(ee.Algorithms.IsEqual(max_value, None), 1, max_value)

    min_value_image = ee.Image.constant(min_value)
    max_value_image = ee.Image.constant(max_value)

    # 정규화: (value - min) / (max - min)
    normalized_result = result.subtract(min_value_image).divide(max_value_image.subtract(min_value_image))

    return normalized_result.rename(f"{index}_normalized")


# 지도 시각화 페이지
if page == "🗺️ 지도 시각화":
    st.title("위성 영상 기반 식생 지수 분석")

    col1, col2 = st.columns([2, 1])

    with col1:
        start_date = st.date_input(
            "시작 날짜",
            value=None,
            help="분석할 기간의 시작일을 선택하세요"
        )

    with col2:
        end_date = st.date_input(
            "종료 날짜",
            value=None,
            help="분석할 기간의 종료일을 선택하세요"
        )

    # 지도 타입 선택을 위한 라디오 버튼
    map_type = st.radio(
        "지도 타입",
        ["일반", "위성"],
        horizontal=True,
        help="지도의 배경 타입을 선택하세요"
    )

    with st.expander("🎯 식생 지수 설명", expanded=False):
        st.markdown("""
        - **NDVI**: 정규화 식생 지수 - 식물의 건강도를 나타냅니다.
        - **RVI**: 비율 식생 지수 - 바이오매스와 관련이 있습니다.
        - **MTVI1**: 수정된 삼각 식생 지수 - 엽록소 함량을 추정합니다.
        - **GNDVI**: 녹색 정규화 식생 지수 - 질소 상태를 파악합니다.
        - **GRVI**: 녹색 적색 식생 지수 - 식물 스트레스를 감지합니다.
        """)

    index_option = st.selectbox(
        "분석할 식생 지수",
        ["NDVI", "RVI", "MTVI1", "GNDVI", "GRVI"],
        help="분석하고자 하는 식생 지수를 선택하세요"
    )

    # 지도 객체 생성 (초기 위치 설정)
    initial_coords = [35.849999, 127.129064]

    # 지도 타입에 따른 베이스맵 설정
    if map_type == "위성":
        m = geemap.Map(center=initial_coords, zoom=19, basemap='HYBRID')
    else:
        m = geemap.Map(center=initial_coords, zoom=19, basemap='ROADMAP')
        # OpenStreetMap 레이어 추가
        m.add_basemap('OpenStreetMap')


    # 선택한 지수의 시각화 파라미터 설정
    def get_vis_params(index):
        return {"min": -1, "max": 1, "palette": ["red", "yellow", "green"]}


    # 지수를 시각화하고 값 출력하는 함수
    def add_index_layer(start_date, end_date, index):
        if start_date and end_date:
            collection = (
                ee.ImageCollection("COPERNICUS/S2")
                .filterDate(str(start_date), str(end_date))
                .filterBounds(polygon)
                .map(mask_clouds)
                .sort("CLOUDY_PIXEL_PERCENTAGE")
            )

            image = collection.first()

            if image:
                index_image = calculate_index(image, index)
                index_clipped = index_image.clip(polygon)
                vis_params = get_vis_params(index)
                m.addLayer(index_clipped, vis_params, f"{index} (Normalized)")
                m.centerObject(polygon, zoom=19)
                st.write(f"### {index} (Normalized) 지수 시각화")
                st.write("지도를 통해 선택한 정규화된 지수를 확인하세요.")
            else:
                st.warning("해당 기간에 이미지가 없습니다.")


    # 선택된 지수를 계산하고 지도에 반영
    if index_option and start_date and end_date:
        add_index_layer(start_date, end_date, index_option)

    # Streamlit에 지도 표시
    m.to_streamlit(height=700)

# 통계 시각화 페이지
elif page == "📊 통계 시각화":
    st.title("식생 지수 통계 분석")

    col1, col2 = st.columns([1, 1])

    with col1:
        start_date = st.date_input(
            "시작 날짜",
            value=None,
            key="stat_start",
            help="통계 분석 시작일"
        )

    with col2:
        end_date = st.date_input(
            "종료 날짜",
            value=None,
            key="stat_end",
            help="통계 분석 종료일"
        )

    # 모든 식생 지수 리스트
    indices = ["NDVI", "RVI", "MTVI1", "GNDVI", "GRVI"]

    # 박스플롯 및 바플롯을 위한 데이터 생성 함수
    def get_boxplot_data(start_date, end_date, indices):
        all_data = []
        stats_data = []  # Barplot을 위한 통계 데이터
        if start_date and end_date:
            collection = (
                ee.ImageCollection("COPERNICUS/S2")
                .filterDate(str(start_date), str(end_date))
                .filterBounds(polygon)
                .map(mask_clouds)
                .sort("CLOUDY_PIXEL_PERCENTAGE")
            )

            image = collection.first()

            if image:
                for index in indices:
                    # 선택한 지수 계산
                    index_image = calculate_index(image, index)

                    # 다각형 영역에서 픽셀 값을 샘플링
                    samples = index_image.sample(region=polygon, scale=10, numPixels=500, geometries=True)
                    values = samples.aggregate_array(f"{index}_normalized").getInfo()

                    if values:
                        # 지수별 데이터 수집
                        all_data.append((index, values))

                        # min, mean, max 계산
                        min_val = np.min(values)
                        mean_val = np.mean(values)
                        max_val = np.max(values)

                        stats_data.append({"Index": index, "Min": min_val, "Mean": mean_val, "Max": max_val})
                return all_data, stats_data
            else:
                st.warning("해당 기간에 이미지가 없습니다.")
                return None, None


    # 시각화 코드
    if start_date and end_date:
        data, stats_data = get_boxplot_data(start_date, end_date, indices)

        if data:
            # 박스플롯 생성
            fig_box = go.Figure()

            for index, values in data:
                fig_box.add_trace(go.Box(y=values, name=index, boxmean='sd'))

            # 레이아웃 설정
            fig_box.update_layout(
                title="식생 지수 통계 시각화 (Boxplot)",
                yaxis_title="Normalized 값",
                xaxis_title="식생 지수",
                width=900,
                height=600
            )

            # Streamlit에 박스플롯 표시
            st.plotly_chart(fig_box)

            # Barplot 생성
            stats_df = pd.DataFrame(stats_data)
            stats_melted = stats_df.melt(id_vars=["Index"], value_vars=["Min", "Mean", "Max"],
                                         var_name="Statistic", value_name="Value")

            fig_bar = px.bar(stats_melted, x="Index", y="Value", color="Statistic", barmode="group",
                             title="식생 지수 통계 요약 (Min, Mean, Max)")

            # Streamlit에 바플롯 표시
            st.plotly_chart(fig_bar)

        else:
            st.warning("시각화할 데이터를 찾을 수 없습니다.")


    # 시계열 데이터 생성 함수
    def get_time_series_data(start_date, end_date, indices):
        series_data = []
        if start_date and end_date:
            collection = (
                ee.ImageCollection("COPERNICUS/S2")
                .filterDate(str(start_date), str(end_date))
                .filterBounds(polygon)
                .map(mask_clouds)
                .sort("system:time_start")
            )

            for i in range(collection.size().getInfo()):
                image = ee.Image(collection.toList(collection.size()).get(i))
                timestamp = image.date().millis().getInfo()
                date = datetime.utcfromtimestamp(timestamp / 1000).strftime('%Y-%m-%d')

                for index in indices:
                    index_image = calculate_index(image, index)
                    mean_value = index_image.reduceRegion(
                        reducer=ee.Reducer.mean(),
                        geometry=polygon,
                        scale=10,
                        maxPixels=1e13
                    ).get(f"{index}_normalized").getInfo()

                    if mean_value is not None:
                        series_data.append({"Date": date, "Index": index, "Value": mean_value})

        return series_data


    # 시계열 데이터 시각화
    if start_date and end_date:
        series_data = get_time_series_data(start_date, end_date, indices)

        if series_data:
            series_df = pd.DataFrame(series_data)
            fig_line = px.line(series_df, x="Date", y="Value", color="Index", markers=True,
                               title="식생 지수 시계열 데이터")

            fig_line.update_layout(
                xaxis_title="날짜",
                yaxis_title="Normalized 값",
                width=900,
                height=600
            )

            # Streamlit에 시계열 lineplot 표시
            st.plotly_chart(fig_line)
        else:
            st.warning("시계열 데이터를 찾을 수 없습니다.")

# 비료 처방 추천 페이지
elif page == "🌿 양분 현황 분석":
    st.title("식생 지수 기반 양분 현황")

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        start_date = st.date_input(
            "시작 날짜",
            value=None,
            key="fertilizer_start"
        )

    with col2:
        end_date = st.date_input(
            "종료 날짜",
            value=None,
            key="fertilizer_end"
        )

    with col3:
        selected_fertilizer = st.selectbox(
            "분석할 토양 성분",
            ["pH", "P", "OM", "K", "Ca", "Mg", "EC"],
            help="분석하고자 하는 토양 성분을 선택하세요"
        )

    # 색상 가이드 추가
    st.markdown("""
    <div style='background-color: white; padding: 10px; border-radius: 5px; margin-bottom: 20px;'>
        <h4 style='color: #1B5E20; margin-bottom: 10px;'>🎨 색상 가이드</h4>
        <div style='display: flex; gap: 20px;'>
            <div>
                <span style='display: inline-block; width: 20px; height: 20px; background-color: red; margin-right: 5px; vertical-align: middle;'></span>
                <span>부족 상태 (기준치 미만)</span>
            </div>
            <div>
                <span style='display: inline-block; width: 20px; height: 20px; background-color: green; margin-right: 5px; vertical-align: middle;'></span>
                <span>적정 상태 (기준치 범위 내)</span>
            </div>
            <div>
                <span style='display: inline-block; width: 20px; height: 20px; background-color: blue; margin-right: 5px; vertical-align: middle;'></span>
                <span>과다 상태 (기준치 초과)</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 지도 타입 선택을 위한 라디오 버튼
    map_type = st.radio(
        "지도 타입",
        ["일반", "위성"],
        horizontal=True,
        help="지도의 배경 타입을 선택하세요"
    )

    with st.expander("💡 토양 성분 가이드", expanded=False):
        st.markdown("""
        | 성분 | 적정범위 | 부족 시 증상 | 과다 시 증상 |
        |------|----------|--------------|--------------|
        | pH | 5.5-6.2 | 양분 흡수 저하 | 미량원소 결핍 |
        | P | 250-350 | 생장 저하 | 다른 영양소 흡수 방해 |
        | OM | 20-30 | 토양 구조 악화 | 영양 불균형 |
        | K | 0.5-0.6 | 병해충 약화 | Ca, Mg 흡수 저해 |
        | Ca | 4.5-5.5 | 세포벽 약화 | 다른 양분 흡수 저해 |
        | Mg | 1.5-2.0 | 엽록소 감소 | K 흡수 저해 |
        | EC | 0.0-2.0 | 양분 부족 | 삼투압 스트레스 |
        """)

    # 성분별 적정 범위와 관련된 식생지수 및 계산 공식
    fertilizer_config = {
        "pH": {"index": "RVI", "min": 5.5, "max": 6.2, "formula": lambda i: 2.11 * i + 7.572},
        "P": {"index": "GNDVI", "min": 250, "max": 350, "formula": lambda i: 554.976 * i - 264.133},
        "OM": {"index": "MTVI1", "min": 20, "max": 30, "formula": lambda i: 90.083 * i - 32.187},
        "K": {"index": "RVI", "min": 0.5, "max": 0.6, "formula": lambda i: 5.581 * i - 2.545},
        "Ca": {"index": "GRVI", "min": 4.5, "max": 5.5, "formula": lambda i: 6.045 * i + 3.977},
        "Mg": {"index": "RVI", "min": 1.5, "max": 2.0, "formula": lambda i: 4.120 * i + 0.943},
        "EC": {"index": "RVI", "min": 0.0, "max": 2.0, "formula": lambda i: 1.489 * i + 0.151},
    }

    # 지도 객체 생성 (초기 위치 설정)
    initial_coords = [35.849999, 127.129064]

    # 지도 타입에 따른 베이스맵 설정
    if map_type == "위성":
        m = geemap.Map(center=initial_coords, zoom=19, basemap='HYBRID')
    else:
        m = geemap.Map(center=initial_coords, zoom=19, basemap='ROADMAP')
        # OpenStreetMap 레이어 추가
        m.add_basemap('OpenStreetMap')

    # 성분 계산 및 지도에 레이어 추가하는 함수
    def add_fertilizer_layer(start_date, end_date, fertilizer):
        if start_date and end_date:
            config = fertilizer_config[fertilizer]
            index = config["index"]
            min_val = config["min"]
            max_val = config["max"]
            formula = config["formula"]

            collection = (
                ee.ImageCollection("COPERNICUS/S2")
                .filterDate(str(start_date), str(end_date))
                .filterBounds(polygon)
                .map(mask_clouds)
                .sort("CLOUDY_PIXEL_PERCENTAGE")
            )

            image = collection.first()

            if image:
                # 선택한 지수 계산
                index_image = calculate_index(image, index).clip(polygon)

                # 성분 값 계산
                fertilizer_image = index_image.multiply(ee.Number(formula(1))).rename(fertilizer)

                # 적정 범위 마스크 설정
                low_mask = fertilizer_image.lt(min_val)
                high_mask = fertilizer_image.gt(max_val)
                ok_mask = fertilizer_image.gte(min_val).And(fertilizer_image.lte(max_val))

                # 색상으로 마스킹된 레이어 생성
                low_layer = fertilizer_image.updateMask(low_mask).visualize(min=min_val, max=max_val, palette=["red"])
                ok_layer = fertilizer_image.updateMask(ok_mask).visualize(min=min_val, max=max_val, palette=["green"])
                high_layer = fertilizer_image.updateMask(high_mask).visualize(min=min_val, max=max_val, palette=["blue"])

                # 지도에 레이어 추가
                m.addLayer(low_layer, {}, f"{fertilizer} 부족 (빨간색)")
                m.addLayer(ok_layer, {}, f"{fertilizer} 적정 (초록색)")
                m.addLayer(high_layer, {}, f"{fertilizer} 초과 (파란색)")

                m.centerObject(polygon, zoom=19)
                st.write(f"### {fertilizer} 비료 처방 시각화")
                st.write("지도를 통해 성분의 부족, 적정, 초과 여부를 확인하세요.")
            else:
                st.warning("해당 기간에 이미지가 없습니다.")

    # 선택된 성분을 지도에 반영
    if selected_fertilizer and start_date and end_date:
        add_fertilizer_layer(start_date, end_date, selected_fertilizer)

    # Streamlit에 지도 표시
    m.to_streamlit(height=700)

# 푸터 추가
st.markdown("""
    <div style='position: fixed; bottom: 0; width: 100%; background-color: #E8F5E9; padding: 10px; text-align: center;'>
        <p style='margin: 0; color: #1B5E20;'>© Seungwon Seok | Powered by Streamlit</p>
    </div>
    """, unsafe_allow_html=True)