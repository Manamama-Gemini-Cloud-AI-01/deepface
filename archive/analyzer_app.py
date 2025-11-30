import os
import subprocess
import re
import json
from flask import Flask, request, jsonify, render_template # Import render_template

# Initialize Flask app
app = Flask(__name__, template_folder='templates') # Specify template folder


# --- Backend Logic ---
@app.route('/')
def index():
    return render_template('analyzer_index.html') # Use render_template

def run_deepface_analysis(image_path):
    """Runs deepface in a subprocess and parses the output."""
    command = ["deepface", "analyze", image_path]
    print(f"Running command: {' '.join(command)}")
    
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    stdout, stderr = process.communicate()
    
    # Print both stdout and stderr for debugging
    print("\n--- DeepFace STDOUT ---")
    print(stdout)
    print("\n--- DeepFace STDERR ---")
    print(stderr)
    
    # Find the JSON-like dictionary output in stdout
    # It typically starts with '{' and ends with '}' and can span multiple lines
    # We will find all such occurrences
    # Regex to find dictionary-like strings (handles nested braces)
    dict_pattern = re.compile(r"(\{.*?\})(?=\s*\{|\s*$)", re.DOTALL)
    matches = dict_pattern.findall(stdout)
    
    if not matches:
        return {"error": "Could not parse analysis results from deepface output.", "stderr": stderr}

    # Convert found strings to actual dictionaries
    results = []
    for match in matches:
        try:
            # The output uses numpy types, so we replace them with standard types for json
            # e.g., "np.float32(0.1)" becomes "0.1"
            clean_match = re.sub(r"np\.float32\((.*?)\)", r"\1", match)
            # The CLI output uses single quotes, which is not valid JSON.
            # Python's json.loads can't handle it, but ast.literal_eval can.
            import ast
            face_data = ast.literal_eval(clean_match)
            results.append(face_data)
        except Exception as e:
            print(f"Error parsing a match: {e}\nMatch content: {match}")
            continue
            
    return results

@app.route('/full_analyze_upload', methods=['POST'])
def full_analyze_upload():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    
    # Save the file temporarily
    temp_path = f"temp_{{file.filename}}"
    file.save(temp_path)
    
    try:
        analysis_result = run_deepface_analysis(temp_path)
        return jsonify(analysis_result)
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/full_analyze_webcam', methods=['POST'])
def full_analyze_webcam():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"error": "No image data provided"}), 400

    # The data is a base64 string, e.g., "data:image/jpeg;base64,..."
    # We need to decode it and save it as a temporary file
    try:
        header, encoded = data['image'].split(",", 1)
        import base64
        image_data = base64.b64decode(encoded)
        
        temp_path = "temp_webcam_capture.jpg"
        with open(temp_path, "wb") as f:
            f.write(image_data)
    except Exception as e:
        return jsonify({"error": f"Failed to decode base64 image: {str(e)}"}), 400

    try:
        analysis_result = run_deepface_analysis(temp_path)
        return jsonify(analysis_result)
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == '__main__':
    # Running in debug mode is helpful for development
    app.run(debug=True, port=5001)