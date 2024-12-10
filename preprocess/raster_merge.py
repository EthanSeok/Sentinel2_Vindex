import rasterio
from rasterio.enums import ColorInterp
import numpy as np

def main():
    input_raster_red = './input/cropped_red.tif'
    input_raster_green = './input/cropped_green.tif'
    input_raster_blue = './input/cropped_blue.tif'
    output_raster_rgb = './output/cropped_rgb.tif'

    # 레드 채널 읽기
    with rasterio.open(input_raster_red) as red_src:
        red = red_src.read(1)
        profile = red_src.profile

    # 그린 채널 읽기
    with rasterio.open(input_raster_green) as green_src:
        green = green_src.read(1)

    # 블루 채널 읽기
    with rasterio.open(input_raster_blue) as blue_src:
        blue = blue_src.read(1)

    profile.update(count=3, dtype=rasterio.float32)

    with rasterio.open(output_raster_rgb, 'w', **profile) as dst:
        dst.write(red.astype(rasterio.float32), 1)
        dst.write(green.astype(rasterio.float32), 2)
        dst.write(blue.astype(rasterio.float32), 3)

    with rasterio.open(output_raster_rgb, 'r+') as dst:
        dst.colorinterp = [ColorInterp.red, ColorInterp.green, ColorInterp.blue]

if __name__ == '__main__':
    main()