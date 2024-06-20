from flask import Flask, request, jsonify, send_from_directory, abort
import os

app = Flask(__name__)

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


# Define the folder to serve files from
FILES_FOLDER = '/media/volume/ai-model-store/MedSyn/results/img_256_standard'

# lists all files in a folder
@app.route('/files/<foldername>', methods=['GET'])
def list_files(foldername):
    try:
        folder = os.path.join(FILES_FOLDER,foldername)
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





if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)



"""
# access server through public api
$ curl "http://149.165.171.65:5000/api?marco=polo"



"""