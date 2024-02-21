from haufman_code import encode_image_to_huffman, decode_huffman_to_image
from flask import Flask, request, send_file, jsonify
from PIL import Image
import numpy as np
import json
import io
import pickle
import base64



app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['image']
    image = Image.open(file.stream)
    image_array = np.array(image)
    image_shape = image_array.shape

    encoded_data, huffman_dicts, list_result = encode_image_to_huffman(image_array)
    result = list_result[0] - list_result[1]

    huffman_json = json.dumps(huffman_dicts)
    encoded_data_json = json.dumps(encoded_data)
    image_shape_json = json.dumps(image_shape)

    combined_data = f"{encoded_data_json}split{huffman_json}split{image_shape_json}split{result}"
    return combined_data

@app.route('/decode', methods=['POST'])
def decode():
    file = request.files['file']

    # Read the content of the file
    content = file.read().decode('utf-8')
    parts = content.split('split')
    if len(parts) != 4:
        return "Invalid file format or delimiter not found.", len(parts)  # Return an error

    encoded_data_json, huffman_json, image_shape_json, _ = parts  # Now we can safely unpack

    encoded_data = json.loads(encoded_data_json)
    huffman_dict = json.loads(huffman_json)
    image_shape = json.loads(image_shape_json)

    image_array = decode_huffman_to_image(np.array(encoded_data, dtype='object'), np.array(huffman_dict, dtype='object'), image_shape)
    image_array = np.clip(image_array, 0, 255)  # Ensure the values are within the 0-255 range

    # Convert to uint8
    image_array = image_array.astype('uint8')
    img = Image.fromarray(image_array)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    return send_file(
        filename_or_fp= io.BytesIO(img_byte_arr),
        mimetype='image/png',
        as_attachment=True,
        attachment_filename='decode.png'
    )

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)