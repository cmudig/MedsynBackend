from flask import Flask, request, jsonify, send_file, send_from_directory
import os
from extract_text import TextExtractor
from stage1 import run_diffusion_1
from stage2 import run_diffusion_2
from heatmap import create_heatmap
from pmap import run_pmap_function
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
import signal
import pynvml
import re
import re
from typing import Optional

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

@app.route('/dicom_files/<foldername>/<int:sample_number>', methods=['GET'])
def list_dicom_files(foldername, sample_number):

    try:
       
       return jsonify({"data": foldername}), 200

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
        saved_noise_path = data.get('saved_noise_path')
        print(f"promt: {prompt}")
        print(f"description: {description}")
        print(f"studyInstanceUID: {studyInstanceUID}")
        print(f"filename: {filename}")
        print(f"patient_name: {patient_name}")
        print(f"patient_id: {patient_id}")
        print(f"num_Series: {num_series_exists}")
        if saved_noise_path:
            print(f"saved_noise_path override: {saved_noise_path}")
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
        threading.Thread(target=run_text_extractor_and_models, args=(studyInstanceUID, description, prompt, output_folder, filename, patient_name, patient_id, series_instance_uid, read_img_flag, num_series_exists, saved_noise_path)).start()

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

@app.route('/status', methods=['GET'])
def check_running():
    global process_is_running
    return jsonify({"process_is_running": process_is_running})

def _build_token_suffix(token_idx, token_text):
    slug = re.sub(r'[^a-zA-Z0-9]+', '_', token_text.lower()).strip('_')
    slug = slug or f"token{token_idx}"
    return f"token_{token_idx}_{slug}"


def _token_file_path(filename: str) -> str:
    safe_name = os.path.basename(filename)
    base, ext = os.path.splitext(safe_name)
    token_file = f"{base}_tokens.npy" if ext else f"{safe_name}_tokens.npy"
    return os.path.join(FILES_FOLDER, "text_embed", token_file)


def _first_sentence_token_limit(filename: str, tokenizer) -> Optional[int]:
    """Return the largest token index to keep (exclusive of the first '.' token)."""
    tokens_path = _token_file_path(filename)
    if not os.path.exists(tokens_path):
        print(f"Token file not found at {tokens_path}; generating overlays for full prompt.")
        return None

    try:
        token_ids = np.load(tokens_path)
    except Exception as exc:
        print(f"Failed to load tokens from {tokens_path}: {exc}")
        return None

    flat_token_ids = token_ids.flatten().tolist()
    limit = None

    for idx, token_id in enumerate(flat_token_ids):
        try:
            decoded = tokenizer.decode([int(token_id)]).strip()
        except Exception as exc:
            print(f"Tokenizer decode failed at index {idx}: {exc}")
            return None

        if decoded in ("[SEP]", "[PAD]"):
            break
        if not decoded:
            continue

        if '.' in decoded:
            limit = idx - 1
            break

    if limit is None:
        return None

    return limit


def _safe_run_pmap(folder, heatmap_volume, sample_num, threshold, *, token_suffix=None, **kwargs):
    try:
        return run_pmap_function(folder, heatmap_volume, sample_num, threshold, token_suffix=token_suffix, **kwargs)
    except TypeError:
        # Fall back for environments that still have the older signature.
        return run_pmap_function(folder, heatmap_volume, sample_num, threshold, **kwargs)


def clear_processes(skip_pids=None):
    """Terminate remaining GPU processes, keeping the current process alive."""
    skip = {os.getpid()}
    if skip_pids:
        skip.update(skip_pids)

    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError as exc:
        print(f"NVML init failed: {exc}")
        return

    try:
        device_count = pynvml.nvmlDeviceGetCount()

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            for proc in processes:
                if proc.pid in skip:
                    continue
                print(f"GPU {i} - PID: {proc.pid}, GPU Memory: {proc.usedGpuMemory} bytes")
                try:
                    os.kill(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
    finally:
        pynvml.nvmlShutdown()

# def run_text_extractor_and_models(studyInstanceUID, description, prompt, output_folder, filename, patient_name, patient_id, series_instance_uid, read_img_flag, num_series_exists=0):
#     # filename: e.g. test.npy
#     global process_is_running
#     old_stdout = sys.stdout
#     sys.stdout = StreamToFile()
#     process_is_running = True

#     # clear output folder textembedding
#     for fn in os.listdir(FILES_FOLDER+"/text_embed"):
#         file_path = os.path.join(FILES_FOLDER+"/text_embed", fn)
#         if os.path.isfile(file_path) and "dont_delete" not in fn:
#             os.remove(file_path)

    
#     # clear output folder low-resolution
#     if read_img_flag:
#         full_dir = os.path.join(FILES_FOLDER, "img_64_standard", studyInstanceUID)
#         #if full_dir does not exist, make it
#         if not os.path.exists(full_dir):
#             os.makedirs(full_dir)
        
#         for fn in os.listdir(full_dir):
#             file_path = os.path.join(full_dir, fn)  # include the subfolder
#             print("HERRREEE", file_path)

#             if os.path.isfile(file_path) and "dont_delete" not in fn and "saved_noise" not in fn:
#                 print("whhahattt")
#                 os.remove(file_path)
#             elif os.path.isdir(file_path) and "dont_delete" not in fn and "saved_noise" not in fn:
#                 print("directorrrryyyyy")
#                 for f in os.listdir(file_path):
#                     os.remove(os.path.join(file_path, f))
#     else:
#         for fn in os.listdir(FILES_FOLDER +"/img_64_standard/"):
#             file_path = os.path.join(FILES_FOLDER +"/img_64_standard", fn)
#             #recurisvely delete all files in any folder in the img_64_standard folder that is not dont_delete or saved_noise
#             if os.path.isfile(file_path) and "dont_delete" not in fn and "saved_noise" not in fn:
#                 os.remove(file_path)
#             elif os.path.isdir(file_path) and "dont_delete" not in fn and "saved_noise" not in fn:
#                 for f in os.listdir(file_path):
#                     os.remove(os.path.join(file_path, f))
#                 os.rmdir(file_path)

#     try:
#         torch.cuda.empty_cache()
#         # Run the text extractor
#         text_extractor = TextExtractor(resume_model=TEXTEXTRACTOR_MODEL_FOLDER)
#         text_extractor.run(prompt, output_folder, filename)
#         print(f"Textembedding stored in: {output_folder}")
#         _save_text_to_file(folder_path=FILES_FOLDER+"/prompts", file_name=filename[:-4]+"_"+str(num_series_exists)+".txt", text_content=prompt)
        
#         torch.cuda.empty_cache()
#         accelerate.state.AcceleratorState._shared_state.clear() # dirty hack to reset accelerator state

#         run_diffusion_1(input_folder=FILES_FOLDER+"/text_embed", 
#                         output_folder=FILES_FOLDER +"/img_64_standard/" + studyInstanceUID, 
#                         noise_folder=FILES_FOLDER+"/img_64_standard/saved_noise/" + studyInstanceUID,
#                         model_folder=STAGE1_MODEL_FOLDER, 
#                         dont_delete_folder=FILES_FOLDER+"/img_64_standard",
#                         attention_folder=FILES_FOLDER+"/saliency_maps/"+studyInstanceUID,
#                         num_sample=1,
#                         tokenizer=text_extractor.tokenizer,
#                         read_img_flag=read_img_flag,
#                         num_series_exists=num_series_exists)
        
#         print("Completed low res.")

#         torch.cuda.empty_cache()
#         accelerate.state.AcceleratorState._shared_state.clear() # dirty hack to reset accelerator state

#         # Run high-res model
#         run_diffusion_2(input_folder=FILES_FOLDER+ "/img_64_standard/"+studyInstanceUID, 
#                         output_folder=FILES_FOLDER +"/img_256_standard", 
#                         model_folder=STAGE2_MODEL_FOLDER,
#                         filename=filename,
#                         num_series_exists=num_series_exists)
        
#         print("Completed high res.")

#         # convert nifti to dicom
#         nifti_file = os.path.join(FILES_FOLDER,"img_256_standard",studyInstanceUID+"_sample_" + str(num_series_exists) + ".nii.gz")
#         output_folder = os.path.join(FILES_FOLDER,"dicom",studyInstanceUID+"_sample_"+str(num_series_exists))
        
#         print(series_instance_uid)
#         print(nifti_file)
#         nifti_to_dicom(nifti_file=nifti_file,
#                         output_folder=output_folder,
#                         series_description=description,                      
#                         series_instance_uid=series_instance_uid,
#                         study_instance_uid=studyInstanceUID,
#                         patient_name=patient_name,
#                         patient_id=patient_id)
        
#         print("Now making heatmap and pmap....")
#         # first we need to get the heatmap volume
#         heatmap_data_path = FILES_FOLDER+'/saliency_maps/'+studyInstanceUID+'/'+filename[:-4]+"_sample_" + str(num_series_exists)+'_token_0_[CLS]_heatmaps.npy'
#         hm_vol = create_heatmap(heatmap_data_path)

#         print('Now maknig pmap...')
#         out_path = run_pmap_function(studyInstanceUID, hm_vol, num_series_exists, 0.6)
#         print(f"We saved the pmap at {out_path}")

#     finally:
#         print("Uploading Data to Orthanc...")
#         sys.stdout = old_stdout
#         process_is_running=False
#         clear_processes()


def run_text_extractor_and_models(studyInstanceUID, description, prompt, output_folder, filename, patient_name, patient_id, series_instance_uid, read_img_flag, num_series_exists=0, saved_noise_path: Optional[str] = None):

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
        full_dir = os.path.join(FILES_FOLDER, "img_64_standard", studyInstanceUID)
        #if full_dir does not exist, make it
        if not os.path.exists(full_dir):
            os.makedirs(full_dir)
        
        for fn in os.listdir(full_dir):
            file_path = os.path.join(full_dir, fn)  # include the subfolder
            if os.path.isfile(file_path) and "dont_delete" not in fn and "saved_noise" not in fn:
                os.remove(file_path)
            elif os.path.isdir(file_path) and "dont_delete" not in fn and "saved_noise" not in fn:
                for f in os.listdir(file_path):
                    os.remove(os.path.join(file_path, f))
    else:
        for fn in os.listdir(FILES_FOLDER +"/img_64_standard/"):
            file_path = os.path.join(FILES_FOLDER +"/img_64_standard", fn)
            #recurisvely delete all files in any folder in the img_64_standard folder that is not dont_delete or saved_noise
            if os.path.isfile(file_path) and "dont_delete" not in fn and "saved_noise" not in fn:
                os.remove(file_path)
            elif os.path.isdir(file_path) and "dont_delete" not in fn and "saved_noise" not in fn:
                for f in os.listdir(file_path):
                    os.remove(os.path.join(file_path, f))
                os.rmdir(file_path)
    
    try:
        torch.cuda.empty_cache()
        # Run the text extractor
        text_extractor = TextExtractor(resume_model=TEXTEXTRACTOR_MODEL_FOLDER)
        text_extractor.run(prompt, output_folder, filename)
        print(f"Textembedding stored in: {output_folder}")
        
        torch.cuda.empty_cache()
        accelerate.state.AcceleratorState._shared_state.clear() # dirty hack to reset accelerator state

        noise_folder = saved_noise_path or FILES_FOLDER+"/img_64_standard/saved_noise/" + studyInstanceUID
        if saved_noise_path:
            print(f"Using overridden saved noise path: {noise_folder}")

        run_diffusion_1(input_folder=FILES_FOLDER+"/text_embed", 
                        output_folder=FILES_FOLDER +"/img_64_standard/" + studyInstanceUID, 
                        noise_folder=noise_folder,
                        model_folder=STAGE1_MODEL_FOLDER, 
                        dont_delete_folder=FILES_FOLDER+"/img_64_standard",
                        attention_folder=FILES_FOLDER+"/saliency_maps/"+studyInstanceUID,
                        num_sample=1,
                        tokenizer=text_extractor.tokenizer,
                        read_img_flag=read_img_flag,
                        num_series_exists=num_series_exists)
        
        print("Completed low res.")

        torch.cuda.empty_cache()
        accelerate.state.AcceleratorState._shared_state.clear() # dirty hack to reset accelerator state

        # Run high-res model
        run_diffusion_2(input_folder=FILES_FOLDER+ "/img_64_standard/"+studyInstanceUID, 
                        output_folder=FILES_FOLDER +"/img_256_standard", 
                        model_folder=STAGE2_MODEL_FOLDER,
                        filename=filename,
                        num_series_exists=num_series_exists)
        
        print("Completed high res.")

        # convert nifti to dicom
        nifti_file = os.path.join(FILES_FOLDER,"img_256_standard",filename[:-4]+"_sample_" + str(num_series_exists) + ".nii.gz")
        output_folder = os.path.join(FILES_FOLDER,"dicom",studyInstanceUID+"_sample_"+str(num_series_exists))
        
        print(series_instance_uid)
        print(nifti_file)
        nifti_to_dicom(nifti_file=nifti_file,
                        output_folder=output_folder,
                        series_description=description,                      
                        series_instance_uid=series_instance_uid,
                        study_instance_uid=studyInstanceUID,
                        patient_name=patient_name,
                        patient_id=patient_id)
        
        print("Collecting per-token heatmaps...")
        heatmap_dir = os.path.join(FILES_FOLDER, 'saliency_maps', studyInstanceUID)
        heatmap_prefix = f"{filename[:-4]}_sample_{num_series_exists}_token_"
        heatmap_suffix = "_heatmaps.npy"
        token_heatmaps = []

        if os.path.isdir(heatmap_dir):
            for file_name in sorted(os.listdir(heatmap_dir)):
                if not file_name.startswith(heatmap_prefix) or not file_name.endswith(heatmap_suffix):
                    continue
                token_meta = file_name[len(heatmap_prefix):-len(heatmap_suffix)]
                if '_' not in token_meta:
                    print(f"Skipping heatmap with unexpected name: {file_name}")
                    continue
                token_idx_str, token_text = token_meta.split('_', 1)
                try:
                    token_idx = int(token_idx_str)
                except ValueError:
                    print(f"Unable to parse token index from {file_name}")
                    continue
                if token_text == '[PAD]':
                    continue

                heatmap_path = os.path.join(heatmap_dir, file_name)
                heatmap_volume = create_heatmap(heatmap_path)
                token_heatmaps.append({
                    'index': token_idx,
                    'token': token_text,
                    'path': heatmap_path,
                    'heatmap': heatmap_volume,
                    'suffix': _build_token_suffix(token_idx, token_text)
                })
        else:
            print(f"No heatmap directory found at {heatmap_dir}")

        if token_heatmaps:
            token_heatmaps.sort(key=lambda entry: entry['index'])
            cutoff_index = _first_sentence_token_limit(filename, text_extractor.tokenizer)
            if cutoff_index is not None:
                before_count = len(token_heatmaps)
                token_heatmaps = [entry for entry in token_heatmaps if entry['index'] <= cutoff_index]
                print(f"Limiting token overlays to first sentence tokens (<= index {cutoff_index}). {before_count} -> {len(token_heatmaps)}.")

        if token_heatmaps:
            print(f"Generating DICOM overlays for {len(token_heatmaps)} tokens (excluding [PAD]).")
            for entry in token_heatmaps:
                hm_vol = entry['heatmap']
                suffix = entry['suffix']
                token = entry['token']
                idx = entry['index']
                print(f"Now making pmap for token '{token}' (index {idx})...")
                out_path = _safe_run_pmap(studyInstanceUID, hm_vol, num_series_exists, 0.7, token_suffix=suffix)
                print(f"Saved PMAP to {out_path}")
                out_path = _safe_run_pmap(studyInstanceUID, hm_vol, num_series_exists, 0.7, saliencymap=True, saliencythresh=False, token_suffix=suffix)
                print(f"Saved saliency map to {out_path}")
                out_path = _safe_run_pmap(studyInstanceUID, hm_vol, num_series_exists, 0.7, saliencymap=False, saliencythresh=True, token_suffix=suffix)
                print(f"Saved saliency threshold map to {out_path}")
        else:
            print("No eligible token heatmaps found; skipping PMAP generation.")
        
    finally:
        print("Uploading Data to Orthanc...")
        sys.stdout = old_stdout
        process_is_running=False
        clear_processes()


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
    app.run(host='0.0.0.0', port=5000, debug=True)

    # description="no pleural effusion, no consolidation, and no cardiomegaly."

    # run_text_extractor_and_models(
    #     studyInstanceUID="3271643143176",
    #     description=description, 
    #     prompt="no pleural effusion, no consolidation, and no cardiomegaly",
    #     output_folder="/media/volume/gen-ai-volume/MedSyn/results/text_embed",
    #     filename="3271643143176.npy",
    #     patient_name="3271643143176",
    #     patient_id="3271643143176",
    #     series_instance_uid="3271643143176",
    #     read_img_flag=False,
    #     num_series_exists=0
    # )


"""
# access server through public api
$ curl "http://149.165.171.65:5000/api?marco=polo"


"""
