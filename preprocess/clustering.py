import numpy as np
import rasterio
from rasterio.warp import transform
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


def read_tif(tif_path):
    """GeoTIFF 파일을 읽고, 데이터와 소스 객체를 반환."""
    src = rasterio.open(tif_path)
    print(src.meta)
    band1 = src.read(1)
    nodata_value = src.nodata
    if (nodata_value is not None) and (not np.isnan(nodata_value)):
        band1 = np.where(band1 == nodata_value, np.nan, band1)
    return band1, src


def perform_kmeans_clustering(indices, n_clusters=4):
    """K-means 클러스터링 수행."""
    valid_mask = np.all(~np.isnan(indices), axis=0)
    valid_data = indices[:, valid_mask].T

    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(valid_data)
    labels = kmeans.labels_

    clustered_array = np.full(indices.shape[1:], np.nan)
    clustered_array[valid_mask] = labels

    return clustered_array, labels, valid_data


def transform_markers(markers, src):
    """WGS 84 좌표계를 tif 파일의 좌표계로 변환."""
    dst_crs = src.crs  # tif 파일의 좌표계
    lon, lat = zip(*markers)
    transformed_coords = transform('EPSG:4326', dst_crs, lon, lat)
    return list(zip(transformed_coords[0], transformed_coords[1]))


def plot_clustering(clustered_array, n_clusters, colors, indices_names, src, markers=None, labels=None):
    """클러스터링 결과를 시각화."""
    cmap = ListedColormap(colors)

    plt.imshow(clustered_array, cmap=cmap)
    title = '필지 Kmean 클러스터링 (' + ', '.join(indices_names) + ')'
    plt.title(title)

    if markers and labels:
        converted_markers = transform_markers(markers, src)
        for i, (x, y) in enumerate(converted_markers):
            row, col = src.index(x, y)
            plt.plot(col, row, marker='o', color='blue', markersize=10)
            plt.text(col + 30, row - 100, labels[i], color='black', fontsize=12, weight='bold')

    legend_elements = [Patch(facecolor=colors[i], edgecolor='black', label=f'군집 {i + 1}') for i in range(n_clusters)]
    plt.legend(handles=legend_elements, loc='lower right')
    plt.savefig('output/clustering_map.png')
    plt.close()


def plot_scatter(valid_data, labels, n_clusters, colors, indices_names):
    """유효 데이터에 대한 산점도 플롯."""
    plt.figure(figsize=(8, 6))
    for cluster in range(n_clusters):
        cluster_data = valid_data[labels == cluster]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {cluster + 1}', alpha=0.6, color=colors[cluster])

    plt.xlabel(indices_names[0])
    plt.ylabel(indices_names[1])
    title = '필지 Kmean 클러스터링 (' + ', '.join(indices_names) + ')'
    plt.title(title)

    legend_elements = [Patch(facecolor=colors[i], edgecolor='black', label=f'군집 {i + 1}') for i in range(n_clusters)]
    plt.legend(handles=legend_elements, loc='upper left')
    plt.savefig('output/clustering.png')
    plt.close()


def main():
    ndre_path = 'output/cropped/NDRE/NDRE_original_output.tif'
    resavi_path = 'output/cropped/RESAVI/RESAVI_original_output.tif'
    mcari1_path = 'output/cropped/MCARI1/MCARI1_original_output.tif'


    colors = ['yellow', 'green', 'limegreen', 'orange']  ## 'NDRE', 'RESAVI', 'MCARI1'

    markers = [
        (127.129064, 35.849999), (127.129172, 35.850068),
        (127.129289, 35.850240), (127.129418, 35.850326), (127.129538, 35.850410), (127.129658, 35.850490),
        (127.129355, 35.850163), (127.129485, 35.850250), (127.129605, 35.850335), (127.129722, 35.850420)
    ]

    label = ['1_1', '1_2', '2_1', '2_2', '2_3', '2_4', '3_1', '3_2', '3_3', '3_4']

    ndre, src = read_tif(ndre_path)
    resavi, _ = read_tif(resavi_path)
    mcari1, _ = read_tif(mcari1_path)

    indices = np.stack([ndre, resavi, mcari1])
    indices_names = ['NDRE', 'RESAVI', 'MCARI1']

    clustered_array, labels, valid_data = perform_kmeans_clustering(indices, n_clusters=4)
    plot_clustering(clustered_array, n_clusters=4, colors=colors, indices_names=indices_names, src=src, markers=markers, labels=label)
    plot_scatter(valid_data, labels, n_clusters=4, colors=colors, indices_names=indices_names)


if __name__ == '__main__':
    main()
