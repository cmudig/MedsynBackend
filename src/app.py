from flask import Flask, request, jsonify, send_from_directory, abort, Response, render_template_string
import os
from extract_text import TextExtractor
from stage1 import run_diffusion_1
from stage2 import run_diffusion_2
import threading
import time
import io
import sys
from flask_cors import CORS

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

        # Start the process in a separate thread
        threading.Thread(target=run_text_extractor_and_models, args=(prompt, output_folder, filename)).start()

        return jsonify({"message": "Process started", "filename": filename}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/progress')
def progress():
    def generate():
        with open('log.txt') as f:
            f.seek(0, os.SEEK_END)  # Move the file pointer to the end of the file
            while True:
                line = f.readline()
                if line:
                    yield f"{line.strip()}\n"
                #time.sleep(1)  # Sleep briefly to avoid busy-waiting
    return Response(generate(), content_type='text/event-stream')

@app.route('/running', methods=['POST'])
def check_running():
    global process_is_running
    return jsonify({"process_is_running": process_is_running})

def run_text_extractor_and_models(prompt, output_folder, filename):
    global process_is_running
    old_stdout = sys.stdout
    sys.stdout = StreamToFile()
    process_is_running = True

    try:
        # # Run the text extractor
        # text_extractor = TextExtractor(resume_model=TEXTEXTRACTOR_MODEL_FOLDER)
        # text_extractor.run(prompt, output_folder, filename)
        # print(f"Textembedding stored in: {output_folder}")

        # # Run low-res model
        # run_diffusion_1(input_folder=FILES_FOLDER+"/text_embed", 
        #                 output_folder=FILES_FOLDER +"/img_64_standard", 
        #                 model_folder=STAGE1_MODEL_FOLDER, 
        #                 num_sample=1)

        # # Run high-res model
        # run_diffusion_2(input_folder=FILES_FOLDER+ "/img_64_standard", 
        #                 output_folder=FILES_FOLDER +"/img_256_standard", 
        #                 model_folder=STAGE1_MODEL_FOLDER)

        # fake some progress for now
        for i in range(100):
            time.sleep(5)
            print(f"progress: {i}")
    finally:
        sys.stdout = old_stdout
        process_is_running=False

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)



"""
# access server through public api
$ curl "http://149.165.171.65:5000/api?marco=polo"


"""