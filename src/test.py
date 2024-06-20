from flask import Flask, request, jsonify

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


"""
# access server through public api
$ curl "http://149.165.171.65:5000/api?marco=polo"



"""