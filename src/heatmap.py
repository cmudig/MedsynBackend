import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

def normalize_image(image):
    image_min = image.min()
    image_max = image.max()
    return (image - image_min) / (image_max - image_min + 1e-8)

# **Function to Average Multiple Heatmaps**
def average_heatmaps_from_files(file_list, target_shape=None):
    accumulated_heatmap = None
    count = 0  

    for file in file_list:
        heatmap = np.load(file)
        heatmap = np.flip(heatmap, axis=1)  

        if target_shape and heatmap.shape != target_shape:
            zoom_factors = (
                target_shape[0] / heatmap.shape[0],
                target_shape[1] / heatmap.shape[1],
                target_shape[2] / heatmap.shape[2]
            )
            heatmap = zoom(heatmap, zoom_factors, order=1)  
        
        heatmap = normalize_image(heatmap)

        if accumulated_heatmap is None:
            accumulated_heatmap = np.zeros_like(heatmap)
        
        accumulated_heatmap += heatmap
        count += 1
    
    if count > 0:
        averaged_heatmap = accumulated_heatmap / count
    else:
        raise ValueError("No valid heatmaps were found.")

    return averaged_heatmap

def create_heatmap(heatmap_data_path):
    # heatmap_data_path = '/media/volume/gen-ai-volume/MedSyn/results/saliency_maps/leftpleur_nocard_nocons/leftpleur_nocard_nocons_sample_0_token_2_left_heatmaps.npy'

    heatmaps = np.load(heatmap_data_path)
    print(f"Loaded heatmaps with shape: {heatmaps.shape}")

    height, width, num_frames = 64, 64, 64

    if heatmaps.shape[1] != height or heatmaps.shape[2] != width:
        resized_heatmaps = []
        for i in range(num_frames):
            heatmap_slice = heatmaps[i, :, :]  # e.g. (8, 8)
            heatmap_slice_norm = normalize_image(heatmap_slice)
            zoom_factors = (height / heatmap_slice.shape[0], width / heatmap_slice.shape[1])
            heatmap_resized = zoom(heatmap_slice_norm, zoom_factors, order=1)  # Bilinear interpolation
            resized_heatmaps.append(heatmap_resized)
        heatmap_volume = np.stack(resized_heatmaps, axis=0)  # (frames, height, width)
    else:
        heatmap_volume = heatmaps

    return heatmap_volume