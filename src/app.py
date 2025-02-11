from flask import Flask, request, jsonify, send_file, send_from_directory
import os
from extract_text import TextExtractor
from stage1 import run_diffusion_1
from stage2 import run_diffusion_2
import threading
import io
import sys
from flask_cors import CORS
import pydicom
from dicom_helpers import nifti_to_dicom
import accelerate
import torch
import signal
import subprocess
import time
import numpy as np
from flask import Flask, jsonify, send_file
import cv2
import nibabel as nib
from scipy.ndimage import zoom


app = Flask(__name__)

# Enable CORS
CORS(app, resources={r"/*": {"origins": "*"}})

# Define the folder to serve files from
FILES_FOLDER = '/media/volume/gen-ai-volume/MedSyn/results'

# models
TEXTEXTRACTOR_MODEL_FOLDER = "/media/volume/gen-ai-volume/MedSyn/models/test_run2"
STAGE1_MODEL_FOLDER="/media/volume/gen-ai-volume/MedSyn/models/stage1"
STAGE2_MODEL_FOLDER = "/media/volume/gen-ai-volume/MedSyn/models/stage2"

process_is_running= False

# check if server is running
@app.route('/')
def base_route():
    return jsonify({"server_running": True})

# Define a route to listen to POST requests
@app.route('/api', methods=['POST'])
def api_post():
    data = request.json
    return jsonify(data), 200


def overlay_heatmap_on_ct(ct_scan_folder, heatmap_npy_path, foldername, sample_number):
    """
    Overlays a heatmap (stored as .npy) onto a CT scan (DICOM series) and saves the combined images as a new DICOM series.
    """
    dicom_output_folder = os.path.join(FILES_FOLDER, "dicom_overlays", foldername, sample_number)
    os.makedirs(dicom_output_folder, exist_ok=True)

    # Load the heatmap from .npy
    heatmap_data = np.load(heatmap_npy_path)  # Shape: (num_heads, 1, num_slices, H, W)
    print(f"✅ Loaded heatmap: {heatmap_data.shape}")

    # Remove singleton dimension if necessary
    if heatmap_data.shape[1] == 1:
        heatmap_data = np.squeeze(heatmap_data, axis=1)  # Shape: (num_heads, num_slices, H, W)

    # Average over attention heads
    heatmap_data = heatmap_data.mean(axis=0)  # Shape: (num_slices, H, W)

    # List all CT scan slices
    dicom_files = sorted([f for f in os.listdir(ct_scan_folder) if f.endswith(".dcm")])
    num_ct_slices = len(dicom_files)

    # Ensure heatmap slices match CT scan slices (from 64 → 256)
    heatmap_slices = heatmap_data.shape[0]
    if heatmap_slices != num_ct_slices:
        zoom_factor = num_ct_slices / heatmap_slices  # 256 / 64 = 4.0
        heatmap_data = zoom(heatmap_data, (zoom_factor, 1, 1), order=3)  # Interpolating slices
        print(f"🔄 Resized heatmap from {heatmap_slices} → {heatmap_data.shape[0]} slices")

    series_instance_uid = pydicom.uid.generate_uid()
    
    # Process each DICOM slice
    dicom_list = []
    for i, dicom_filename in enumerate(dicom_files):
        dicom_path = os.path.join(ct_scan_folder, dicom_filename)
        ds = pydicom.dcmread(dicom_path)

        # Read the CT image
        ct_image = ds.pixel_array.astype(np.float32)
        ct_image = (ct_image - np.min(ct_image)) / (np.max(ct_image) - np.min(ct_image))  # Normalize

        # Read the corresponding heatmap slice
        heatmap_slice = heatmap_data[i, :, :]
        heatmap_slice = cv2.resize(heatmap_slice, (ct_image.shape[1], ct_image.shape[0]), interpolation=cv2.INTER_CUBIC)
        heatmap_slice = (heatmap_slice - np.min(heatmap_slice)) / (np.max(heatmap_slice) - np.min(heatmap_slice))  # Normalize

        # Overlay heatmap onto CT scan (blend using transparency)
        overlayed_image = (ct_image * 0.7 + heatmap_slice * 0.3)  # Adjust blending ratio
        overlayed_image = (overlayed_image * 255).astype(np.uint8)

        # Update DICOM metadata
        ds.PixelData = overlayed_image.tobytes()
        ds.Rows, ds.Columns = overlayed_image.shape
        ds.SeriesDescription = "CT Scan with Attention Overlay"
        ds.SeriesInstanceUID = series_instance_uid

        # Save the new DICOM slice
        output_dicom_path = os.path.join(dicom_output_folder, f"slice_{i:03d}.dcm")
        ds.save_as(output_dicom_path)
        dicom_list.append(output_dicom_path)

    print(f"✅ Overlay process complete. DICOM saved at {dicom_output_folder}")
    return dicom_list

@app.route('/dicom_files/<foldername>/<int:sample_number>/<filename>', methods=['GET'])
def get_dicom_files(foldername, sample_number, filename):
    try:
        # Construct the path
        dicom_folder = os.path.join(FILES_FOLDER, "dicom_overlays", f"{foldername}_sample_{sample_number}")
        dicom_file_path = os.path.join(dicom_folder, filename)

        print(f"Checking file: {dicom_file_path}")

        # Ensure file exists
        if not os.path.isfile(dicom_file_path):
            return jsonify({"error": "File not found"}), 404

        # Load the DICOM file
        dicom_data = pydicom.dcmread(dicom_file_path)

        print(f"🔎 Checking DICOM metadata for {filename}:")
        print(f"   - StudyInstanceUID: {dicom_data.StudyInstanceUID}")
        print(f"   - SeriesInstanceUID: {dicom_data.SeriesInstanceUID}")

        # Convert to byte stream
        dicom_bytes = io.BytesIO()
        dicom_data.save_as(dicom_bytes)
        dicom_bytes.seek(0)

        # Serve the file
        return send_file(dicom_bytes, mimetype="application/dicom", as_attachment=False)

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

# gets the attention map
# @app.route('/attention-maps/<foldername>/<int:sample_number>/<studyInstanceUID>', methods=['GET'])

@app.route('/dicom_files/<foldername>/<int:sample_number>', methods=['GET'])
def list_dicom_files(foldername, sample_number):

    try:
        # Construct full path
        dicom_folder = os.path.join(FILES_FOLDER, "dicom_overlays", f"{foldername}_sample_{sample_number}")

        # Ensure folder exists
        if not os.path.exists(dicom_folder):
            return jsonify({"error": "Folder not found"}), 404

        # List files in the folder
        files = [f for f in os.listdir(dicom_folder) if os.path.isfile(os.path.join(dicom_folder, f))]
        return jsonify({"dicom_files": files}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# lists all files in a folder
@app.route('/files/<foldername>/<int:sample_number>', methods=['GET'])
def list_files(foldername, sample_number):
    print("OUR SAMPLE NUMBER IS ", sample_number)
    try:
        foldername = f"{foldername}_sample_{sample_number}"
        # foldername = f"{foldername}_sample_0"
        folder = os.path.join(FILES_FOLDER,"dicom",foldername)
        files = os.listdir(folder)
        files = [f for f in files if os.path.isfile(os.path.join(folder, f))]
        return jsonify(files)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# returns a dicom certain file
@app.route('/files/<foldername>/<filename>/<int:sample_number>', methods=['POST'])
def get_file(foldername, filename, sample_number):
    print("OUR SAMPLE NUMBER IS ", sample_number)
    try:
        foldername = f"{foldername}_sample_{sample_number}"
        # foldername = f"{foldername}_sample_0"
        # Build the path to the subfolder
        folder = os.path.join(FILES_FOLDER, foldername)
        print(f"Checking folder: {folder}")
        print(f"Accessing folder: {folder}")
        print(f"Requested filename: {filename}")

        dicom_file_path = os.path.join(FILES_FOLDER,"dicom",foldername, filename)
        print(dicom_file_path)
        if os.path.isfile(dicom_file_path):
            dicom_data = pydicom.dcmread(dicom_file_path)

            # Convert the DICOM data to a byte stream
            dicom_bytes = io.BytesIO()
            dicom_data.save_as(dicom_bytes)
            dicom_bytes.seek(0)

            return send_file(dicom_bytes, mimetype='application/dicom', as_attachment=False)
        else:
            # Return a 404 error if the file is not found
            print(f"File {filename} not found in folder {folder}")
            return jsonify({"error": str(e)}), 500
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500


# run model
@app.route('/files/<fileID>', methods=['POST'])
def process_text(fileID):
    try:
        # Get the prompt from the POST request
        data = request.get_json()
        prompt = data.get('prompt')
        description = data.get('description')
        studyInstanceUID = data.get('studyInstanceUID')
        filename = data.get('filename')
        patient_name = data.get('patient_name')
        patient_id = data.get('patient_id')
        read_img_flag = data.get('read_img_flag')
        num_series_exists = data.get('num_series_in_study')
        print(f"promt: {prompt}")
        print(f"description: {description}")
        print(f"studyInstanceUID: {studyInstanceUID}")
        print(f"filename: {filename}")
        print(f"patient_name: {patient_name}")
        print(f"patient_id: {patient_id}")
        print(f"num_Series: {num_series_exists}")
        series_instance_uid = pydicom.uid.generate_uid()
        
        if not prompt:
            return jsonify({"error": "Prompt is empty."}), 400
        if not description:
            return jsonify({"error": "Description is empty."}), 400
        if not studyInstanceUID:
            return jsonify({"error": "studyInstanceUID is empty."}), 400
        
        output_folder = os.path.join(FILES_FOLDER,"text_embed")
        print(f"outputfolder: {output_folder}")

        print
        # Start the process in a separate thread
        threading.Thread(target=run_text_extractor_and_models, args=(studyInstanceUID, description, prompt, output_folder, filename, patient_name, patient_id, series_instance_uid, read_img_flag, num_series_exists)).start()

        return jsonify({"message": "Process started", 
                        "filename": filename,
                        "prompt":prompt,
                        "seriesInstanceUID":series_instance_uid}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/progress')
def progress():
    last_line = "Empty Log."
    with open('log.txt') as f:
        lines = f.readlines()
        if lines:  # Check if lines list is not empty
            last_line = lines[-1]
    
    return last_line

def get_gpu_process():
    """Finds running GPU processes that are using CUDA."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,name", "--format=csv,noheader"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if result.returncode == 0:
            processes = result.stdout.strip().split("\n")
            gpu_processes = [line.split(",")[0].strip() for line in processes if "python" in line]
            return gpu_processes if gpu_processes else None
        else:
            print("Failed to retrieve GPU processes:", result.stderr)
            return None
    except Exception as e:
        print(f"Error retrieving GPU processes: {e}")
        return None

def kill_gpu_process():
    """Kills all detected GPU processes (except this script)."""
    gpu_processes = get_gpu_process()
    if gpu_processes:
        for pid in gpu_processes:
            if str(os.getpid()) not in pid:  # Avoid killing this process
                print(f"Killing GPU process: {pid}")
                os.kill(int(pid), signal.SIGKILL)
    else:
        print("[INFO] No active GPU processes found.")

@app.route('/status', methods=['GET'])
def check_running():
    global process_is_running
    return jsonify({"process_is_running": process_is_running})

def run_text_extractor_and_models(studyInstanceUID, description, prompt, output_folder, filename, patient_name, patient_id, series_instance_uid, read_img_flag, num_series_exists=0):
    # filename: e.g. test.npy
    global process_is_running
    old_stdout = sys.stdout
    sys.stdout = StreamToFile()
    process_is_running = True

    # clear output folder textembedding
    for fn in os.listdir(FILES_FOLDER+"/text_embed"):
        file_path = os.path.join(FILES_FOLDER+"/text_embed", fn)
        if os.path.isfile(file_path) and "dont_delete" not in fn:
            os.remove(file_path)

    
    # clear output folder low-resolution
    if read_img_flag:
        for fn in os.listdir(FILES_FOLDER +"/img_64_standard/" + studyInstanceUID):
            file_path = os.path.join(FILES_FOLDER +"/img_64_standard", fn)
            if os.path.isfile(file_path) and "dont_delete" not in fn:
                if "saved_noise" not in fn:
                    os.remove(file_path)

    try:
        torch.cuda.empty_cache()
        # Run the text extractor
        text_extractor = TextExtractor(resume_model=TEXTEXTRACTOR_MODEL_FOLDER)
        text_extractor.run(prompt, output_folder, filename)
        print(f"Textembedding stored in: {output_folder}")
        _save_text_to_file(folder_path=FILES_FOLDER+"/prompts", file_name=filename[:-4]+".txt", text_content=prompt)
        
        torch.cuda.empty_cache()
        accelerate.state.AcceleratorState._shared_state.clear() # dirty hack to reset accelerator state

        run_diffusion_1(input_folder=FILES_FOLDER+"/text_embed", 
                        output_folder=FILES_FOLDER +"/img_64_standard/" + studyInstanceUID, 
                        noise_folder=FILES_FOLDER+"/img_64_standard/saved_noise/" + studyInstanceUID,
                        model_folder=STAGE1_MODEL_FOLDER, 
                        dont_delete_folder=FILES_FOLDER+"/img_64_standard",
                        attention_folder=FILES_FOLDER+"/saliency_maps/"+studyInstanceUID,
                        num_sample=1,
                        read_img_flag=read_img_flag)

        torch.cuda.empty_cache()
        accelerate.state.AcceleratorState._shared_state.clear() # dirty hack to reset accelerator state

        # Run high-res model
        run_diffusion_2(input_folder=FILES_FOLDER+ "/img_64_standard/"+studyInstanceUID, 
                        output_folder=FILES_FOLDER +"/img_256_standard", 
                        model_folder=STAGE2_MODEL_FOLDER,
                        filename=filename,
                        num_series_exists=num_series_exists)

        # convert nifti to dicom
        nifti_file = os.path.join(FILES_FOLDER,"img_256_standard",filename[:-4]+"_sample_" + str(num_series_exists) + ".nii.gz")
        output_folder = os.path.join(FILES_FOLDER,"dicom",filename[:-4]+"_sample_" + str(num_series_exists))
        
        print(series_instance_uid)
        print(nifti_file)
        nifti_to_dicom(nifti_file=nifti_file,
                        output_folder=output_folder,
                        series_description=description,                      
                        series_instance_uid=series_instance_uid,
                        study_instance_uid=studyInstanceUID,
                        patient_name=patient_name,
                        patient_id=patient_id)


    finally:
        print("Uploading Data to Orthanc...")
        sys.stdout = old_stdout
        process_is_running=False
         # After diffusion completes, check and kill GPU processes
        # print("Checking for any lingering GPU processes...")
        # time.sleep(5)  # Ensure processes have updated
        # kill_gpu_process()  # Kill GPU processes

class StreamToFile(io.StringIO):
    def __init__(self):
        super().__init__()
        self.file = open('log.txt', 'w')

    def write(self, message):
        self.file.write(message)
        self.file.flush()
        super().write(message)

    def close(self):
        self.file.close()
        super().close()
def _save_text_to_file(folder_path, file_name, text_content):
    """
    Save the given text content to a .txt file with the specified file name in the specified folder.

    Parameters:
    folder_path (str): The path to the folder where the file will be saved.
    file_name (str): The name of the file (should include .txt extension).
    text_content (str): The text content to be written to the file.
    """
    # Ensure the folder exists, create if it doesn't
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Construct the full file path
    file_path = os.path.join(folder_path, file_name)
    
    # Write the text content to the file
    with open(file_path, 'w') as file:
        file.write(text_content)
    
    print(f"File '{file_name}' saved in '{folder_path}' with the provided content.")
if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=5000, debug=True)


    # studyInstanceUID, description, prompt, output_folder, filename, patient_name, patient_id, series_instance_uid, read_img_flag
    description="Calcification, Atelectasis, Opacity, Consolidation"

    run_text_extractor_and_models(
        studyInstanceUID="kate3",
        description=description, 
        prompt="left pleural effusion",
        # prompt="left pleural effusion",
        output_folder="/media/volume/gen-ai-volume/MedSyn/results/text_embed",
        filename="leftpleuraleffusion2.npy",
        # filename="20250202173128largepanco.npy",
        patient_name="k",
        patient_id="leftpleur3",
        series_instance_uid="leftpleur3",
        read_img_flag=False,
        num_series_exists=0
    )


"""
# access server through public api
$ curl "http://149.165.171.65:5000/api?marco=polo"


"""