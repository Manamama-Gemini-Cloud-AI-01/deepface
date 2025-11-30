import os
import argparse
import subprocess
import json
import numpy as np # Import numpy
from deepface import DeepFace

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
    face_detected_status = deepface_metadata.get("DeepFaceFaceDetected", "Unknown")

    if face_detected_status == "Unknown":
        # If DeepFaceFaceDetected is "Unknown", perform initial detection first
        print(f"  EXIF tag DeepFaceFaceDetected is 'Unknown'. Performing initial DeepFace detection...")
        try:
            extracted_faces = DeepFace.extract_faces(img_path=image_path, enforce_detection=False, align=False)
            
            face_count = 0
            face_detected_status = "No"

            if extracted_faces and isinstance(extracted_faces, list) and len(extracted_faces) > 0:
                face_count = len(extracted_faces)
                face_detected_status = "Yes"
                print(f"  Initial detection found {face_count} face(s).")
            else:
                print(f"  Initial detection found 0 faces.")

            deepface_metadata.update({
                "DeepFaceFaceDetected": face_detected_status,
                "DeepFaceFaceCount": face_count
            })
            write_exif_data(image_path, deepface_metadata)

        except Exception as e:
            print(f"  Error during initial DeepFace detection for {image_path}: {e}")
            deepface_metadata.update({
                "DeepFaceFaceDetected": "No",
                "DeepFaceFaceCount": 0
            })
            write_exif_data(image_path, deepface_metadata)
            face_detected_status = "No" # Update for subsequent logic

    if face_detected_status == "Yes":
        print(f"  Faces detected in {image_path}. Running detailed analysis...")
        try:
            # Perform detailed DeepFace analysis
            analysis_results = DeepFace.analyze(
                img_path=image_path,
                actions=['age', 'gender', 'race', 'emotion'],
                enforce_detection=True  # Ensure a face is detected before analyzing attributes
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
                print(f"  DeepFace.analyze did not return results for {image_path} despite initial detection. Skipping detailed tagging.")
                # If analysis fails, update status to reflect this stricter check
                deepface_metadata.update({
                    "DeepFaceFaceDetected": "No",
                    "DeepFaceFaceCount": 0, # Explicitly set to 0 if detailed analysis fails to return results
                    "DeepFaceDominantEmotion": "",
                    "DeepFaceAge": "",
                    "DeepFaceDominantGender": "",
                    "DeepFaceDominantRace": "",
                    "DeepFaceAnalysisSummary": []
                })
                write_exif_data(image_path, deepface_metadata)


        except ValueError as e:
            if "Face could not be detected" in str(e):
                print(f"  Detailed analysis failed for {image_path}: Face could not be detected. Updating status.")
                # Update status and clear detailed tags
                deepface_metadata.update({
                    "DeepFaceFaceDetected": "No",
                    "DeepFaceFaceCount": 0, # Explicitly set to 0 if face not detected by analyze
                    "DeepFaceDominantEmotion": "",
                    "DeepFaceAge": "",
                    "DeepFaceDominantGender": "",
                    "DeepFaceDominantRace": "",
                    "DeepFaceAnalysisSummary": []
                })
                write_exif_data(image_path, deepface_metadata)
            else:
                print(f"  Error during detailed DeepFace analysis for {image_path}: {e}")
        except Exception as e:
            print(f"  General error during detailed DeepFace analysis for {image_path}: {e}")
    else:
        print(f"  No faces detected in {image_path}. Skipping detailed analysis.")

def main():
    parser = argparse.ArgumentParser(
        description="Analyzes images with DeepFace and updates EXIF metadata with detailed facial attributes."
    )
    parser.add_argument("directory", type=str, help="Path to the directory containing images.")
    args = parser.parse_args()

    target_directory = args.directory

    if not os.path.isdir(target_directory):
        print(f"Error: Directory '{target_directory}' not found.")
        return

    print(f"Starting detailed DeepFace analysis in: {target_directory}")

    image_extensions = (".jpg", ".jpeg", ".png", ".gif", ".webp")
    for root, _, files in os.walk(target_directory):
        for file in files:
            if file.lower().endswith("-02.jpg"): # Filter for -02.jpg files
                image_path = os.path.join(root, file)
                analyze_and_tag_image(image_path)

    print(f"Detailed DeepFace analysis completed for directory: {target_directory}")

if __name__ == "__main__":
    main()
