import rasterio
from rasterio.mask import mask
from shapely.geometry import Polygon, mapping
from pyproj import Proj, transform
import matplotlib.pyplot as plt

def transform_coordinates(coords, src_proj, dst_proj):
    transformed_coords = []
    for x, y in coords:
        x2, y2 = transform(src_proj, dst_proj, x, y)
        transformed_coords.append((x2, y2))
    return transformed_coords

def crop_tiff_by_coordinates(image_path, coords, output_path):
    with rasterio.open(image_path) as src:
        src_crs = src.crs

    src_proj = Proj(init='epsg:4326')
    dst_proj = Proj(src_crs)

    transformed_coords = transform_coordinates(coords, src_proj, dst_proj)
    polygon = Polygon(transformed_coords)

    with rasterio.open(image_path) as src:
        out_image, out_transform = mask(src, [mapping(polygon)], crop=True)
        out_meta = src.meta.copy()

        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(out_image)

    return out_image

def main():
    coords = [
        (127.129714, 35.850595),
        (127.129852, 35.850432),
        (127.129041, 35.849884),
        (127.128887, 35.850039)
    ]

    image_path = 'input/JBNU_grass_transparent_reflectance_red edge.tif'
    output_path = 'input/cropped_red_edge.tif'

    cropped_image = crop_tiff_by_coordinates(image_path, coords, output_path)

    plt.figure(figsize=(10, 10))
    plt.imshow(cropped_image[0], cmap='gray')
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main()