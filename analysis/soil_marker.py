import os
import pandas as pd
import numpy as np
import rasterio
from rasterio.warp import transform
import matplotlib.pyplot as plt


def read_tif(tif_path):
    with rasterio.open(tif_path) as src:
        # print(src.meta)
        band1 = src.read(1)
        nodata_value = src.nodata
        if (nodata_value is not None) and (not np.isnan(nodata_value)):
            band1 = np.where(band1 == nodata_value, np.nan, band1)
    return band1, src


def transform_markers(markers, src):
    """WGS 84 좌표계를 tif 파일의 좌표계로 변환."""
    dst_crs = src.crs  # tif 파일의 좌표계
    lon, lat = zip(*markers)
    transformed_coords = transform('EPSG:4326', dst_crs, lon, lat)
    return list(zip(transformed_coords[0], transformed_coords[1]))


def display_image(data, src, title, markers=None, labels=None, output_path=None):
    plt.figure(figsize=(10, 8))
    plt.imshow(data, cmap='RdYlGn', interpolation='nearest')
    plt.colorbar(label='Value')
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')

    if markers and labels:
        converted_markers = transform_markers(markers, src)
        for i, (x, y) in enumerate(converted_markers):
            row, col = src.index(x, y)
            plt.plot(col, row, marker='o', color='blue', markersize=10)
            plt.text(col + 30, row - 100, labels[i], color='black', fontsize=12, weight='bold')

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        print(f"Saved image to {output_path}")

    plt.close()


def calculate_marker_means(data, src, markers, window_size=3):
    converted_markers = transform_markers(markers, src)
    means = []

    # 데이터의 최소값과 최대값 계산
    min_value = np.nanmin(data)
    max_value = np.nanmax(data)

    for i, (x, y) in enumerate(converted_markers):
        row, col = src.index(x, y)
        half_window = window_size // 2

        # 윈도우 경계 설정 (이미지 경계 안에 있도록 제한)
        row_start = max(row - half_window, 0)
        row_end = min(row + half_window + 1, data.shape[0])
        col_start = max(col - half_window, 0)
        col_end = min(col + half_window + 1, data.shape[1])

        # 윈도우 내 픽셀 값 추출
        window_data = data[row_start:row_end, col_start:col_end]

        # NaN 값을 제외한 평균 계산
        mean_value = np.nanmean(window_data)

        # 정규화: (값 - 최소값) / (최대값 - 최소값)
        if max_value - min_value != 0:
            normalized_value = (mean_value - min_value) / (max_value - min_value)
        else:
            normalized_value = 0  # 최대값과 최소값이 같을 경우 정규화가 불가능하므로 0으로 설정

        means.append(normalized_value)
        # print(f"Marker {i + 1} ({x:.6f}, {y:.6f}) 평균 값: {mean_value:.4f} 정규화 값: {normalized_value:.4f}")

    return means


def main():
    soil = pd.read_csv('토양분석결과.csv')
    markers = [
        (127.129064, 35.849999), (127.129172, 35.850068),
        (127.129289, 35.850240), (127.129418, 35.850326), (127.129538, 35.850410), (127.129658, 35.850490),
        (127.129355, 35.850163), (127.129485, 35.850250), (127.129605, 35.850335), (127.129722, 35.850420)
    ]

    labels = ['1_1', '1_2', '2_1', '2_2', '2_3', '2_4', '3_1', '3_2', '3_3', '3_4']

    results = []

    map_path = '../preprocess/output/cropped'
    output_dir = 'output/map_images'

    for folder in os.listdir(map_path):
        folder_path = os.path.join(map_path, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith('original_output.tif'):
                    file_path = os.path.join(folder_path, file)
                    data, src = read_tif(file_path)

                    # 이미지 저장 경로 설정
                    output_path = os.path.join(output_dir, f"{folder}_{file.replace('.tif', '.png')}")
                    display_image(data, src, title=f"{folder} - {file}", markers=markers, labels=labels, output_path=output_path)

                    means = calculate_marker_means(data, src, markers)

                    for label, mean in zip(labels, means):
                        results.append({
                            'Folder': folder,
                            'File': file,
                            '구역': label,
                            'Mean Value': mean
                        })

    df = pd.DataFrame(results)

    result = pd.merge(soil, df, on='구역')
    result.to_csv('results.csv', index=False, encoding='utf-8-sig')
    print(result)


if __name__ == '__main__':
    main()

