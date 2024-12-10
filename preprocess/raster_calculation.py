import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt

def read_band(band_path):
    with rasterio.open(band_path) as src:
        band = src.read(1)
        meta = src.meta
        nodata = src.nodata
    return band, meta, nodata

def write_output(output_path, data, meta, nodata):
    meta.update(dtype=rasterio.float32, count=1, nodata=nodata)
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(data.astype(rasterio.float32), 1)

def mask_background(band, nodata):
    if nodata is not None:
        mask = (band == nodata)
        band = np.where(mask, np.nan, band)
    return band

def calculate_ndvi(red, nir):
    ndvi = (nir - red) / (nir + red)
    return np.where(np.isnan(ndvi), np.nan, ndvi)

def calculate_ndre(red_edge, nir):
    ndre = (nir - red_edge) / (nir + red_edge)
    return np.where(np.isnan(ndre), np.nan, ndre)

def calculate_gndvi(green, nir):
    gndvi = (nir - green) / (nir + green)
    return np.where(np.isnan(gndvi), np.nan, gndvi)

def calculate_evi(nir, red, blue):
    evi = 2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1))
    return np.where(np.isnan(evi), np.nan, evi)

def calculate_rvi(red, nir):
    rvi = nir / red
    return np.where(np.isnan(rvi), np.nan, rvi)

def calculate_cvi(nir, red, green):
    cvi = nir * (red / green**2)
    return np.where(np.isnan(cvi), np.nan, cvi)

def calculate_tvi(nir, red, green):
    tvi = 0.5 * (120 * (nir - green) - 200 * (red - green))
    return np.where(np.isnan(tvi), np.nan, tvi)

def calculate_vari(blue, red, green):
    vari = (green - red) / (green + red - blue)
    return np.where(np.isnan(vari), np.nan, vari)

def calculate_savi(nir, red, L=0.5):
    savi = ((nir - red) * (1 + L)) / (nir + red + L)
    return np.where(np.isnan(savi), np.nan, savi)

def calcluate_rdvi(nir, red):
    rdvi = (nir - red) / (nir + red)**0.5
    return np.where(np.isnan(rdvi), np.nan, rdvi)

def calculate_mtvi1(nir, green, red):
    mtvi1 = 1.2 * (1.2 * (nir - green) - 2.5 * (red - green))
    return np.where(np.isnan(mtvi1), np.nan, mtvi1)

def calculate_vagr(red, green, blue):
    vagr = (green - red) / (green + red - blue)
    return np.where(np.isnan(vagr), np.nan, vagr)

def calculate_grvi(red, green):
    grvi = (green - red) / (green + red)
    return np.where(np.isnan(grvi), np.nan, grvi)

def calculate_mgrvi(red, green):
    mgrvi = (green**2 - red**2) / (green**2 + red**2)
    return np.where(np.isnan(mgrvi), np.nan, mgrvi)

def calculate_CIre(nir, red_edge):
    CIre = (nir / red_edge) -1
    return np.where(np.isnan(CIre), np.nan, CIre)

def calculate_datt(red, nir, red_edge):
    datt = (nir - red_edge) / (nir - red)
    return np.where(np.isnan(datt), np.nan, datt)

def calculate_resavi(nir, red_edge):
    resavi = 1.5 * (nir - red_edge) / (nir + red_edge + 0.5)
    return np.where(np.isnan(resavi), np.nan, resavi)

def calculate_MCARI1(nir, red_edge, green):
    mcart1 = ((nir - red_edge) - 0.2 * (nir - green)) * (nir / red_edge)
    return np.where(np.isnan(mcart1), np.nan, mcart1)

def calculate_GLI(red, green, blue):
    gli = (2 * green - red - blue) / (2 * green + red + blue)
    return np.where(np.isnan(gli), np.nan, gli)

def save_colored_image(index_data, output_path, cmap='RdYlGn'):
    plt.figure(figsize=(10, 10))
    masked_data = np.ma.masked_where(np.isnan(index_data), index_data)
    plt.imshow(masked_data, cmap=cmap)
    plt.axis('off')  # Hide the axes
    plt.colorbar()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()


def normalize_index(index_data):
    min_val = np.nanmin(index_data)
    max_val = np.nanmax(index_data)
    # print(f"Normalizing index with min_val: {min_val}, max_val: {max_val}")
    # Normalize to range 0 to 1
    normalized = (index_data - min_val) / (max_val - min_val)
    return np.where(np.isnan(index_data), np.nan, normalized)

def calculate_indices(red_band, nir_band, red_edge_band, green_band, blue_band, output_dir):
    red, meta, red_nodata = read_band(red_band)
    nir, _, nir_nodata = read_band(nir_band)
    red_edge, _, red_edge_nodata = read_band(red_edge_band)
    green, _, green_nodata = read_band(green_band)
    blue, _, blue_nodata = read_band(blue_band)

    if not (red.shape == nir.shape == red_edge.shape == green.shape == blue.shape):
        raise ValueError("The input rasters do not have the same shape")

    red = mask_background(red, red_nodata)
    nir = mask_background(nir, nir_nodata)
    red_edge = mask_background(red_edge, red_edge_nodata)
    green = mask_background(green, green_nodata)
    blue = mask_background(blue, blue_nodata)

    indices = {
        'NDVI': calculate_ndvi(red, nir),
        'NDRE': calculate_ndre(red_edge, nir),
        'GNDVI': calculate_gndvi(green, nir),
        'EVI': calculate_evi(nir, red, blue),
        'RVI': calculate_rvi(red, nir),
        'CVI': calculate_cvi(nir, red, green),
        'TVI': calculate_tvi(nir, red, green),
        'VARI': calculate_vari(blue, red, green),
        'SAVI': calculate_savi(nir, red),
        'RDVI': calcluate_rdvi(nir, red),
        'MTVI1': calculate_mtvi1(nir, green, red),
        'VAGR': calculate_vagr(red, green, blue),
        'GRVI': calculate_grvi(red, green),
        'MGRVI': calculate_mgrvi(red, green),
        'CIre': calculate_CIre(nir, red_edge),
        'DATT': calculate_datt(red, nir, red_edge),
        'RESAVI': calculate_resavi(nir, red_edge),
        'MCARI1': calculate_MCARI1(nir, red_edge, green),
        'GLI': calculate_GLI(red, green, blue)
    }

    for index_name, index_data in indices.items():
        index_output_dir = os.path.join(output_dir, index_name)
        if not os.path.exists(index_output_dir):
            os.makedirs(index_output_dir)

        # print(f"{index_name} index min: {np.nanmin(index_data)}, max: {np.nanmax(index_data)}")
        original_output_path = os.path.join(index_output_dir, f'{index_name}_original_output.tif')
        write_output(original_output_path, index_data, meta, nodata=0)

        normalized_index = normalize_index(index_data)
        # print(f"{index_name} normalized min: {np.nanmin(normalized_index)}, max: {np.nanmax(normalized_index)}")  # Debugging line
        normalized_output_path = os.path.join(index_output_dir, f'{index_name}_normalized_output.tif')
        write_output(normalized_output_path, normalized_index, meta, nodata=0)

        colored_output_path = os.path.join(index_output_dir, f'{index_name}_colored.png')
        save_colored_image(normalized_index, colored_output_path)

def main():
    input_raster_red = './input/cropped_red.tif'
    input_raster_nir = './input/cropped_nir.tif'
    input_raster_red_edge = './input/cropped_red_edge.tif'
    input_raster_green = './input/cropped_green.tif'
    input_raster_blue = './input/cropped_blue.tif'
    output_dir = 'output/cropped'

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    calculate_indices(input_raster_red, input_raster_nir, input_raster_red_edge, input_raster_green, input_raster_blue, output_dir)

if __name__ == '__main__':
    main()
