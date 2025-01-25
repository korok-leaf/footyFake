from flask import Flask, jsonify, request


app = Flask(__name__)

@app.route('/get_ball', methods=['POST'])
def get_ball():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    return jsonify({'message': 'File received'}), 200


if __name__ == '__main__':
    app.run(debug=True)