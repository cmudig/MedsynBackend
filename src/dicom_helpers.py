import nibabel as nib
import pydicom
import numpy as np
import os
from pydicom.dataset import Dataset, FileDataset
from datetime import datetime
from pydicom.uid import generate_uid



    



def nifti_to_dicom(nifti_file,
    output_folder,
    series_description,
    series_instance_uid, # should be different for each image
    study_instance_uid, # should be the same for each study (for AI/non-AI)
    patient_name,
    patient_id,
   
    reference_dicom_file = "/media/volume/gen-ai-volume/MedSyn/results/dicom/test_dicom/slice_000.dcm",
    modality='AI',
   
    study_id='1', # should be the same for AI/non-AI

    description="",

    ):
    
    print(f"Store DICOM files in Folder: {output_folder}")
    rotate=""
    
    
    if modality=='AI':
        rotate="counterclockwise"
        apply_mirror = False
    else:
        rotate="clockwise"
        apply_mirror = True



    # Load the NIfTI file
    img = nib.load(nifti_file)
    data = img.get_fdata()
    affine = img.affine

    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Initialize common dataset attributes
    ds = pydicom.dcmread(reference_dicom_file)
    ds.PatientName = patient_name
    ds.PatientID = patient_id
    ds.AccessionNumber = study_instance_uid
    ds.StudyInstanceUID = study_instance_uid
    ds.SeriesInstanceUID = series_instance_uid
    ds.SeriesDescription= series_description
    ds.Modality = modality
    ds.SeriesNumber = 1
    ds.StudyID = study_id
    ds.StudyDescription = description
    ds.Manufacturer = "PythonDicomConversion"
    ds.Rows, ds.Columns = data.shape[:2]
    ds.SliceThickness = float(affine[2, 2])

    ds.SamplesPerPixel = 1
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1  # 1 means signed integers




    # Set additional metadata
    ds.ContentDate = str(datetime.now().date()).replace('-', '')
    ds.ContentTime = datetime.now().strftime("%H%M")

    ds.StudyDate = ds.ContentDate
    ds.StudyTime = ds.ContentTime
    ds.PatientSex = "O"
    ds.PatientBirthDate = "19000101"

    

    # Scale pixel data if necessary (e.g., to avoid issues with pixel value ranges)
    if modality=='AI':
        # set window preset for ai generated imgags
        ds.WindowWidth = 1500
        ds.WindowCenter = 0
        data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 1624 -1024 # from MedSyn Paper
    else:
        # set window preset for original imges
        ds.WindowWidth = 200
        ds.WindowCenter = 0
        data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 2000 -1000 # tried out
    data = data.astype('int16')

    # reverse in 3rd axis
    #data = data[:,:,::-1]
    # Rotate each slice to the left (90 degrees counterclockwise)
    if rotate == "counterclockwise":
        data = np.rot90(data, k=1, axes=(0, 1))
    elif rotate == "clockwise":
        data = np.rot90(data, k=3, axes=(0, 1))
    
    

    
    #print(data)
    # Iterate over each slice and update the dataset
    for i in range(data.shape[2]):
        slice_data = data[:, :, -(i+1)]

        if apply_mirror:
            slice_data = np.fliplr(slice_data)

        # Update slice-specific attributes
        ds.SOPInstanceUID= generate_uid()
        ds.InstanceNumber = i + 1
        ds.ImagePositionPatient = [0,0,-i] 
        ds.SliceLocation = i * ds.SliceThickness



        # Convert pixel data to the appropriate type and flatten the array
        ds.PixelData = slice_data.tobytes()

        # Visualize the slice
        # plt.imshow(slice_data, cmap='gray')
        # plt.title(f'Slice {i}')
        # plt.show()

        # Save the DICOM file
        dicom_filename = os.path.join(output_folder, f"slice_{i:03d}.dcm")
        ds.save_as(dicom_filename)

    print(f"Conversion complete. DICOM files are saved in {output_folder}")




