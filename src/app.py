from flask import Flask, request, jsonify, send_from_directory, abort
import os
from extract_text import TextExtractor
from stage1 import run_diffusion_1
from stage2 import run_diffusion_2
import torch

app = Flask(__name__)

# Define the folder to serve files from
FILES_FOLDER = '/media/volume/gen-ai-volume/MedSyn/results'



# models
TEXTEXTRACTOR_MODEL_FOLDER = "/media/volume/gen-ai-volume/MedSyn/models/test_run2"
STAGE1_MODEL_FOLDER="/media/volume/gen-ai-volume/MedSyn/models/stage1"
STAGE2_MODEL_FOLDER = "/media/volume/gen-ai-volume/MedSyn/models/stage2"


# Define a route to listen to GET requests
@app.route('/api', methods=['GET'])
def api_get():
    data = request.args.to_dict()
    return jsonify(data), 200

# Define a route to listen to POST requests
@app.route('/api', methods=['POST'])
def api_post():
    data = request.json
    return jsonify(data), 200




# lists all files in a folder
@app.route('/files/<foldername>', methods=['GET'])
def list_files(foldername):
    try:
        folder = os.path.join(FILES_FOLDER,"img_256_standard",foldername)
        files = os.listdir(folder)
        files = [f for f in files if os.path.isfile(os.path.join(folder, f))]
        return jsonify(files)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# returns a certain file
@app.route('/files/<foldername>/<filename>', methods=['GET'])
def get_file(foldername, filename):
    try:
        # Build the path to the subfolder
        folder = os.path.join(FILES_FOLDER, foldername)
        print(f"Accessing folder: {folder}")
        print(f"Requested filename: {filename}")
        
        # Ensure the file exists in the specified folder
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path):
            return send_from_directory(folder, filename)
        else:
            # Return a 404 error if the file is not found
            print(f"File {filename} not found in folder {folder}")
            abort(404)
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500


# run model
@app.route('/files/<filename>', methods=['POST'])
def process_text(filename):
    try:
        
        # Get the prompt from the POST request
        data = request.get_json()
        prompt = data.get('prompt')
        print(f"promt: {prompt}")
        
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400

        output_folder = os.path.join(FILES_FOLDER,"text_embed")
        print(f"outputfolder: {output_folder}")
        filename = filename+'.npy'
        print(f"filename: {filename}")

        
        # Run the text extractor
        text_extractor = TextExtractor(resume_model=TEXTEXTRACTOR_MODEL_FOLDER)
        print("init worked")
        text_extractor.run(prompt, output_folder, filename)
        print(f"Textembedding stored in: {output_folder}")


        # Run low-res model
        run_diffusion_1(input_folder=FILES_FOLDER+"/text_embed", 
                output_folder= FILES_FOLDER +"/img_64_standard", 
                model_folder=STAGE1_MODEL_FOLDER, 
                num_sample=1)
        
        # Run high-res model
        run_diffusion_2(input_folder=FILES_FOLDER+ "/img_64_standard", 
                output_folder=FILES_FOLDER +"/img_256_standard", 
                model_folder=STAGE1_MODEL_FOLDER) 
        

        return jsonify({"message": "Process completed successfully",
                        "filename":filename,
                        "text_embed_foldername":FILES_FOLDER+"/text_embed",
                        "low_res_foldername":FILES_FOLDER +"/img_64_standard",
                        "high_res_foldername":FILES_FOLDER +"/img_256_standard",
                        "get_request":f'files/{filename[:-4]}',
                        "prompt":prompt}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)



"""
# access server through public api
$ curl "http://149.165.171.65:5000/api?marco=polo"



"""