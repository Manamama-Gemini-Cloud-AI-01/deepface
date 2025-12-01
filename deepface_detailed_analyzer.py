import os
import argparse
import subprocess
import json
import time # Import time for elapsed time measurement
import numpy as np # Import numpy
from deepface import DeepFace # Import DeepFace



class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)
        
        

def run_shell_command(command, description="Running shell command", cwd=None):
    """
    Runs a shell command and prints its output.
    """
    print(f"\n--- {description} ---")
    print(f"Command: {command}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True, cwd=cwd)
        print("STDOUT:\n", result.stdout)

        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error: {description} failed.")
        print("STDOUT:\n", e.stdout)
        raise
    except FileNotFoundError:
        print(f"Error: Command not found. Make sure '{command.split()[0]}' is in your PATH.")
        raise
                

def run_exiftool_command(command_args):
    """
    Runs an exiftool command and returns its output.
    """

    result = subprocess.run(command_args, capture_output=True, text=True, check=True)
    return result.stdout.strip()

def get_exif_usercomment(image_path):
    """
    Reads the UserComment EXIF tag for a given image and attempts to parse it as JSON.
    Returns a dictionary of the parsed JSON, or an empty dictionary if not found or invalid.
    """
    output = run_exiftool_command(["exiftool", "-UserComment", "-json", image_path])
    if output:
        try:
            data = json.loads(output)
            if data and isinstance(data, list) and len(data) > 0:
                user_comment_str = data[0].get("UserComment")
                if user_comment_str:
                    return json.loads(user_comment_str)
        except (json.JSONDecodeError, IndexError):
            pass # Not found or not valid JSON
    return {}

def write_exif_data(image_path, new_tags):
    """
    Writes EXIF data to an image by updating or creating a JSON string in the UserComment tag.
    new_tags: dictionary where keys are tag names and values are their content.
    """
    existing_tags = get_exif_usercomment(image_path)
    # Merge new tags into existing ones
    existing_tags.update(new_tags)
    
    # Serialize the combined tags into a JSON string using the custom encoder
    json_string = json.dumps(existing_tags, cls=NumpyEncoder, ensure_ascii=False)
    
    command_args = ["exiftool", "-overwrite_original", f"-UserComment={json_string}", image_path]
    run_exiftool_command(command_args)

def analyze_and_tag_image(image_path):
    """
    Analyzes an image with DeepFace and updates its EXIF data.
    """
    print(f"Processing image: {image_path}")

    # Read existing DeepFace data from UserComment
    deepface_metadata = get_exif_usercomment(image_path)
    
    # Perform face extraction first
    try:
        extracted_faces = DeepFace.extract_faces(img_path=image_path, enforce_detection=False, align=False)
        face_count = len(extracted_faces) if extracted_faces else 0
        face_detected_status = "Yes" if face_count > 0 else "No"
        
        deepface_metadata.update({
            "DeepFaceFaceDetected": face_detected_status,
            "DeepFaceFaceCount": face_count
        })
        write_exif_data(image_path, deepface_metadata)

        if face_detected_status == "No":
            print(f"  No faces detected in {image_path}. Skipping detailed analysis.")
            return # Exit if no faces found
        
        print(f"  Faces detected in {image_path}. Running detailed analysis...")
        # Perform detailed DeepFace analysis only if faces were detected
        analysis_results = DeepFace.analyze(
            img_path=image_path,
            actions=['age', 'gender', 'race', 'emotion'],
            enforce_detection=True # We know there's a face from extract_faces, so enforce is fine
        )

        if analysis_results and isinstance(analysis_results, list):
            first_face_result = analysis_results[0]
            
            deepface_metadata.update({
                "DeepFaceFaceCount": len(analysis_results), # Update face count based on successful analysis
                "DeepFaceDominantEmotion": first_face_result.get("dominant_emotion"),
                "DeepFaceAge": first_face_result.get("age"),
                "DeepFaceDominantGender": first_face_result.get("dominant_gender"),
                "DeepFaceDominantRace": first_face_result.get("dominant_race"),
                "DeepFaceAnalysisSummary": analysis_results # This will be JSON-encoded by write_exif_data
            })
            # Remove None values
            deepface_metadata = {k: v for k, v in deepface_metadata.items() if v is not None}

            print(f"  Detailed analysis results for {image_path}: {deepface_metadata}")
            write_exif_data(image_path, deepface_metadata)
        else:
            print(f"  DeepFace.analyze did not return results for {image_path} despite initial detection. Marking as no face.")
            deepface_metadata.update({
                "DeepFaceFaceDetected": "No",
                "DeepFaceFaceCount": 0,
                "DeepFaceDominantEmotion": "N/A",
                "DeepFaceAge": "N/A",
                "DeepFaceDominantGender": "N/A",
                "DeepFaceDominantRace": "N/A",
                "DeepFaceAnalysisSummary": []
            })
            write_exif_data(image_path, deepface_metadata)

    except ValueError as e:
        if "Face could not be detected" in str(e):
            print(f"  Face not detected by DeepFace.extract_faces for {image_path}. Marking as no face.")
            deepface_metadata.update({
                "DeepFaceFaceDetected": "No",
                "DeepFaceFaceCount": 0,
                "DeepFaceDominantEmotion": "N/A",
                "DeepFaceAge": "N/A",
                "DeepFaceDominantGender": "N/A",
                "DeepFaceDominantRace": "N/A",
                "DeepFaceAnalysisSummary": []
            })
            write_exif_data(image_path, deepface_metadata)
        else:
            print(f"  Error during DeepFace processing for {image_path}: {e}. Marking as no face.")
            deepface_metadata.update({
                "DeepFaceFaceDetected": "No",
                "DeepFaceFaceCount": 0,
                "DeepFaceDominantEmotion": "N/A",
                "DeepFaceAge": "N/A",
                "DeepFaceDominantGender": "N/A",
                "DeepFaceDominantRace": "N/A",
                "DeepFaceAnalysisSummary": []
            })
            write_exif_data(image_path, deepface_metadata)
        



def main():
    parser = argparse.ArgumentParser(
        description="Orchestrates video processing, initial DeepFace detection, and detailed DeepFace analysis."
    )
    parser.add_argument("input_video", type=str, help="Path to the input video file.")
    parser.add_argument("--num-images", type=int, default=3,
                        help="Number of images per scene (for detect_images_in_video.sh).")
    parser.add_argument("--scene-threshold", type=int, default=27,
                        help="Scene detection threshold (for detect_images_in_video.sh).")
    
    args = parser.parse_args()

    input_video = args.input_video
    num_images = args.num_images
    scene_threshold = args.scene_threshold



    # --- Step 1: Call detect_images_in_video.sh ---
    print("\n--- Step 1: Extracting images from video using detect_images_in_video.sh ---")
    start_time = time.time()
    
    shell_script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../Ubuntu_Scripts_1/ai_ml/detect_images_in_video.sh"))
    
    video_basename_no_ext = os.path.splitext(os.path.basename(input_video))[0]
    
    # The shell script itself determines the output directory name based on its parameters
    # It creates the output directory next to the input video.
    shell_script_output_dir_name = f"{video_basename_no_ext}_scenes_n{num_images}_t{scene_threshold}"
    full_output_dir_path = os.path.join(os.path.dirname(input_video), shell_script_output_dir_name)

    # Ensure the directory where the shell script will create its output exists.
    # This is the directory of the input video.
    os.makedirs(os.path.dirname(input_video), exist_ok=True)
    
    shell_command = (
        f"{shell_script_path} '{input_video}' "
        f"--num-images {num_images} "
        f"--scene-threshold {scene_threshold}"
    )
    
    run_shell_command(shell_command, description="Running detect_images_in_video.sh")
    
    end_time = time.time()
    print(f"Step 1 completed in {end_time - start_time:.2f} seconds.")



    # --- Step 3: Perform detailed DeepFace analysis using deepface_detailed_analyzer.py ---
    print(f"\n--- Step 3: Performing detailed DeepFace analysis using deepface_detailed_analyzer.py on {full_output_dir_path} ---")
    start_time = time.time()

 

    target_directory = full_output_dir_path

 
    print(f"Starting detailed DeepFace analysis in: {target_directory}")
    for root, _, files in os.walk(target_directory):
        for file in files:
            if file.lower().endswith("-02.jpg"): # Filter for -02.jpg files
                image_path = os.path.join(root, file)
                analyze_and_tag_image(image_path)

    print(f"Detailed DeepFace analysis completed for directory: {target_directory}")
  

    end_time = time.time()
    print(f"Step 3 completed in {end_time - start_time:.2f} seconds.")
    print("\n--- Workflow completed successfully! ---")

    # --- Step 4: Generate HTML summary report ---
    print("\n--- Step 4: Generating DeepFace HTML summary report ---")
    start_time = time.time()
    generate_deepface_summary_html(full_output_dir_path, os.path.basename(input_video))
    end_time = time.time()
    print(f"Step 4 completed in {end_time - start_time:.2f} seconds.")
    print("\n--- Full workflow completed! ---")

def format_probabilities(prob_dict, top_n=3):
    """
    Formats a dictionary of probabilities into a concise string.
    Example: {'angry': 72.6, 'fear': 14.5, 'neutral': 4.2} -> "angry: 72.6%, fear: 14.5%, neutral: 4.2%"
    """
    if not isinstance(prob_dict, dict) or not prob_dict:
        return "N/A"
    
    # Sort probabilities in descending order
    sorted_probs = sorted(prob_dict.items(), key=lambda item: item[1], reverse=True)
    
    formatted_parts = []
    for key, value in sorted_probs[:top_n]:
        # Check if the value is a numpy float and convert it to a native float
        if isinstance(value, np.floating):
            value = float(value)
        formatted_parts.append(f"{key}: {value:.1f}%")
        
    return ", ".join(formatted_parts)

def generate_deepface_summary_html(output_dir, video_filename):
    """
    Generates an HTML summary page of DeepFace analysis results.
    """
    summary_html_file = os.path.join(output_dir, f"{os.path.splitext(video_filename)[0]}_deepface_summary.html")

    html_content = []
    html_content.append("<!DOCTYPE html>")
    html_content.append("<html lang=\"en\">")
    html_content.append("<head>")
    html_content.append("    <meta charset=\"UTF-8\">")
    html_content.append(f"    <title>DeepFace Analysis Summary for: {video_filename}</title>")
    html_content.append("    <style>")
    html_content.append("        body { font-family: sans-serif; background-color: #1a1a1a; color: #e0e0e0; margin: 40px; }")
    html_content.append("        h1 { text-align: center; color: #4CAF50; }")
    html_content.append("        table { width: 90%; margin: 20px auto; border-collapse: collapse; }")
    html_content.append("        th, td { border: 1px solid #333; padding: 10px; text-align: left; }")
    html_content.append("        th { background-color: #333; color: #eee; }")
    html_content.append("        tr:nth-child(even) { background-color: #282828; }")
    html_content.append("        img { max-width: 150px; height: auto; display: block; margin: 0 auto; }")
    html_content.append("        .no-face { color: #f44336; }")
    html_content.append("        .face-detected { color: #4CAF50; }")
    html_content.append("    </style>")
    html_content.append("</head>")
    html_content.append("<body>")
    html_content.append(f"<h1>DeepFace Analysis Summary for: {video_filename}</h1>")
    html_content.append("<table>")
    html_content.append("    <tr>")
    html_content.append("        <th>Image</th>")
    html_content.append("        <th>Filename</th>")
    html_content.append("        <th>Face Count</th>")
    html_content.append("        <th>Age</th>")
    html_content.append("        <th>Emotion Probabilities</th>")
    html_content.append("        <th>Gender Probabilities</th>")
    html_content.append("        <th>Race Probabilities</th>")
    html_content.append("    </tr>")

    processed_images_data = []

    for root, _, files in os.walk(output_dir):
        for file in files:
            if file.lower().endswith("-02.jpg"):
                image_path = os.path.join(root, file)
                deepface_metadata = get_exif_usercomment(image_path)
                processed_images_data.append((file, deepface_metadata))
    
    # Sort by filename to ensure consistent order
    processed_images_data.sort(key=lambda x: x[0])

    for file, deepface_metadata in processed_images_data:
        face_detected = deepface_metadata.get("DeepFaceFaceDetected", "No") 
        face_count = deepface_metadata.get("DeepFaceFaceCount", 0)

        # If faces are detected, iterate through each face's analysis
        if face_detected == "Yes" and "DeepFaceAnalysisSummary" in deepface_metadata:
            analysis_summary = deepface_metadata["DeepFaceAnalysisSummary"]
            
            for face_idx, face_result in enumerate(analysis_summary):
                age = face_result.get("age", "N/A")
                emotion_probs_str = format_probabilities(face_result.get("emotion", {}))
                gender_probs_str = format_probabilities(face_result.get("gender", {}))
                race_probs_str = format_probabilities(face_result.get("race", {}))

                # For multiple faces, adjust face count to show current face's index
                display_face_count = f"{face_idx + 1} of {face_count}" if face_count > 1 else str(face_count)

                html_content.append("    <tr>")
                html_content.append(f"        <td><img src=\"./{file}\" alt=\"{file}\"></td>")
                html_content.append(f"        <td>{file}</td>")
                html_content.append(f"        <td class=\"face-detected\">{display_face_count}</td>") # Apply class to Face Count
                html_content.append(f"        <td>{age}</td>")
                html_content.append(f"        <td>{emotion_probs_str}</td>")
                html_content.append(f"        <td>{gender_probs_str}</td>")
                html_content.append(f"        <td>{race_probs_str}</td>")
                html_content.append("    </tr>")
        else:
            # If no faces detected, create a single row with N/A for analysis
            html_content.append("    <tr>")
            html_content.append(f"        <td><img src=\"./{file}\" alt=\"{file}\"></td>")
            html_content.append(f"        <td>{file}</td>")
            html_content.append(f"        <td class=\"no-face\">{face_count}</td>")
            html_content.append(f"        <td>N/A</td>")
            html_content.append(f"        <td>N/A</td>")
            html_content.append(f"        <td>N/A</td>")
            html_content.append(f"        <td>N/A</td>")
            html_content.append("    </tr>")

    html_content.append("</table>")
    html_content.append("</body>")
    html_content.append("</html>")

    with open(summary_html_file, "w") as f:
        f.write("\n".join(html_content))
    print(f"  HTML summary generated at: {summary_html_file}")

if __name__ == "__main__":
    main()
