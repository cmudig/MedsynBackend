import pydicom
import highdicom as hd
from datetime import datetime
from typing import Optional
from highdicom.pm import ParametricMap, RealWorldValueMapping
from pydicom.uid import generate_uid
from scipy.ndimage import zoom, binary_fill_holes
import numpy as np
import os
import re

def generate_pmap(heatmap_volume, threshold=0.5, binarize=True):
    """
    Convert a heatmap into a binary probability map (PMAP).
    
    Args:
        heatmap_volume (numpy.ndarray): The original heatmap volume (3D).
        threshold (float): The threshold to binarize the heatmap.

    Returns:
        numpy.ndarray: A binary probability map (same shape as heatmap_volume).
    """
    if binarize:
        pmap = (heatmap_volume > threshold).astype(np.uint8)  # Convert to 0 and 1
    else:
        pmap = (heatmap_volume > threshold).astype(np.float32)
    return pmap

def resize_pmap(pmap, target_shape):
    """
    Resize the probability map (PMAP) to match DICOM slice dimensions.

    Args:
        pmap (numpy.ndarray): Original PMAP (slices, height, width).
        target_shape (tuple): (num_slices, height, width) from DICOM.

    Returns:
        numpy.ndarray: Resized PMAP matching the DICOM shape.
    """
    zoom_factors = (target_shape[0] / pmap.shape[0],  # Adjust slices
                    target_shape[1] / pmap.shape[1],  # Adjust height
                    target_shape[2] / pmap.shape[2])  # Adjust width

    pmap_resized = zoom(pmap, zoom_factors, order=1)  # Bilinear interpolation
    return pmap_resized

def load_dicom_series(directory):
    """
    Load all DICOM slices from a directory and sort them by Instance Number.

    Args:
        directory (str): Path to the folder containing DICOM files.

    Returns:
        list[pydicom.Dataset]: Sorted list of DICOM datasets.
    """
    dicom_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.dcm')]
    dicom_datasets = [pydicom.dcmread(f) for f in dicom_files]

    # Sort slices by Instance Number to ensure correct order
    dicom_datasets.sort(key=lambda x: int(x.InstanceNumber))
    
    return dicom_datasets

def _sanitize_label(label: str) -> str:
    slug = re.sub(r'[^a-zA-Z0-9]+', '_', label.lower()).strip('_')
    return slug or 'token'


def _pretty_token_label(token_suffix: Optional[str]) -> str:
    if not token_suffix:
        return "Token"
    try:
        parts = token_suffix.split('_', 2)
        if len(parts) >= 3:
            return parts[2].replace('_', ' ').upper()
    except Exception:
        pass
    return token_suffix.replace('_', ' ').upper()


def _compute_body_mask(dicom_series, threshold: float = -750.0) -> np.ndarray:
    """Return a boolean mask for voxels that belong to the body volume."""
    mask_slices = []
    for dataset in dicom_series:
        pixels = dataset.pixel_array.astype(np.float32)
        slope = getattr(dataset, "RescaleSlope", 1.0) or 1.0
        intercept = getattr(dataset, "RescaleIntercept", 0.0) or 0.0
        hu = pixels * slope + intercept
        slice_mask = hu > threshold
        slice_mask = binary_fill_holes(slice_mask)
        mask_slices.append(slice_mask)
    return np.stack(mask_slices, axis=0)


def attach_pmap_to_dicom_series(dicom_dir, pmap, filename, sampleNum, series_description, typemap, combined, token_suffix=None):
    """
    Attach a PMAP to a DICOM CT series and save it as a multi-frame DICOM PMAP.

    Args:
        dicom_dir (str): Path to the directory containing DICOM slices.
        pmap (numpy.ndarray): The PMAP array (same size as the DICOM volume).

    Returns:
        str: Path to the saved PMAP DICOM file.
    """
    # Load DICOM series
    dicom_series = load_dicom_series(dicom_dir)

    # Get DICOM shape
    num_slices = len(dicom_series)
    dicom_height, dicom_width = dicom_series[0].Rows, dicom_series[0].Columns

    # Resize PMAP if dimensions do not match
    if pmap.shape != (num_slices, dicom_height, dicom_width):
        print(f"Resizing PMAP from {pmap.shape} to ({num_slices}, {dicom_height}, {dicom_width})")
        pmap = resize_pmap(pmap, (num_slices, dicom_height, dicom_width))

    pmap = pmap.astype(np.float32)

    # Mask out voxels that lie outside the body to avoid background overlays
    # body_mask = _compute_body_mask(dicom_series)

    # ✅ Flip the PMAP if it appears upside down
    pmap = np.flip(pmap, axis=1)  # Flip along height (axial view)
    # body_mask = np.flip(body_mask, axis=1)

    # pmap = np.where(body_mask, pmap, 0.0)
    # pmap = np.flip(pmap, axis=0)  # Flip along depth (coronal view)
    # pmap = np.flip(pmap, axis=2)  # Flip along width (sagittal view)
    
    # Generate metadata
    series_instance_uid = generate_uid()
    instance_uid = generate_uid()
    now = datetime.now()

    # ✅ Get correct Referenced CT Series UID
    referenced_series_uid = dicom_series[0].SeriesInstanceUID

    # ✅ Fix: Ensure correct SOP Class UID for CT Image Storage
    referenced_instances = []
    for img in dicom_series:
        ref = pydicom.Dataset()
        ref.ReferencedSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"  # ✅ CT Image Storage
        ref.ReferencedSOPInstanceUID = img.SOPInstanceUID
        referenced_instances.append(ref)

    # ✅ Define a valid RealWorldValueMapping (Identity Mapping)
    real_world_value_mappings = [
        RealWorldValueMapping(
            lut_label="Identity",
            lut_explanation="Probability Map (0-1 range)",
            unit=hd.sr.coding.Code("1", "UCUM", "No Units"),
            value_range=(0.0, 1.0),
            slope=1.0,
            intercept=0.0
        )
    ]

    # ✅ Fix: Ensure correct Modality (CT) and reference correct CT Series
    token_label = _pretty_token_label(token_suffix)
    parametric_map = ParametricMap(
        source_images=dicom_series,  # Use full DICOM series
        pixel_array=pmap,  # Already resized
        series_instance_uid=series_instance_uid,
        sop_instance_uid=instance_uid,
        manufacturer="Your Organization",
        manufacturer_model_name="AI Model XYZ",
        software_versions="1.0.0",
        device_serial_number="0000",
        contains_recognizable_visual_features=False,
        real_world_value_mappings=real_world_value_mappings,  
        window_center=0.9,  # Helps display PMAP properly (0 to 1 range)
        window_width=1.0,  # Ensures PMAP contrast scaling works
        series_description=f"{token_label} {series_description}",
        series_number=3005,  # ✅ Match expected PMAP series number in OHIF
        instance_number=1
    )

    # ✅ Explicitly set Modality to "CT"
    parametric_map.Modality = "CT"

    # ✅ Fix: Create Referenced Series Sequence Correctly
    referenced_series_item = pydicom.Dataset()
    referenced_series_item.SeriesInstanceUID = referenced_series_uid
    referenced_series_item.ReferencedInstanceSequence = referenced_instances
    parametric_map.ReferencedSeriesSequence = [referenced_series_item]

    
    # ✅ Update the Referenced Instance Sequence
    parametric_map.ReferencedImageSequence = referenced_instances

    # Save the PMAP as a DICOM file
    pmap_save_path = "/media/volume/gen-ai-volume/MedSyn/results/dicom_overlays"
    label = token_suffix or ('combined' if combined else 'single')
    label = _sanitize_label(label)
    output_pmap_path = os.path.join(pmap_save_path, filename, f"{filename}_sample_{sampleNum}_{label}_output_{typemap}.dcm")
    os.makedirs(os.path.dirname(output_pmap_path), exist_ok=True)  
    parametric_map.save_as(output_pmap_path)

    print(f"Saved PMAP DICOM to {output_pmap_path}")
    return output_pmap_path


def run_pmap_function(folder, heatmap_volume, sampleNum, threshold, saliencymap=False, saliencythresh=False, combined=False, token_suffix=None):
    dicom_dir = f"/media/volume/gen-ai-volume/MedSyn/results/dicom/{folder}_sample_{sampleNum}"
    if not saliencymap and not saliencythresh:
        pmap = generate_pmap(heatmap_volume, threshold=threshold, binarize=True)
        output_pmap_dicom = attach_pmap_to_dicom_series(dicom_dir, pmap, folder, sampleNum, "Probability Map", "pmap", combined, token_suffix)
    elif not saliencymap:
        pmap = generate_pmap(heatmap_volume, threshold=threshold, binarize=False)
        output_pmap_dicom = attach_pmap_to_dicom_series(dicom_dir, pmap, folder, sampleNum, "Saliency Map Threshold", "saliencythresh", combined, token_suffix)
    else:
        output_pmap_dicom = attach_pmap_to_dicom_series(dicom_dir, heatmap_volume, folder, sampleNum, "Saliency Map", "saliency", combined, token_suffix)


    return output_pmap_dicom
