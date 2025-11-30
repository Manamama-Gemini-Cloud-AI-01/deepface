import os
import time
from deepface import DeepFace
from deepface.commons.logger import Logger
from tabulate import tabulate # For pretty table output

# Initialize deepface's logger if needed (though DeepFace's own logging is usually sufficient)
# This will use the deepface.commons.logger.Logger class
logger = Logger()

def process_folder(folder_path: str):
    """
    Processes all image files in a given folder using DeepFace.extract_faces,
    and prints a table of results.
    """
    print(f"\n--- Processing Folder: {folder_path} ---")
    results_table = []
    headers = ["Image", "Face Detected", "Face Count", "Elapsed Time (s)"]
    
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("No image files found in this folder.")
        return

    # DeepFace will load models on the first call, subsequent calls will reuse them.
    # This simulates the "model memoizing" effect.
    for image_name in sorted(image_files): # Sort for consistent output
        image_path = os.path.join(folder_path, image_name)
        start_time = time.time()
        
        face_detected = "No"
        face_count = 0
        
        try:
            # Using 'opencv' backend which is generally fast
            # enforce_detection=False so it doesn't raise an exception if no face is found
            detected_faces = DeepFace.extract_faces(
                img_path=image_path, 
                detector_backend='opencv', 
                enforce_detection=False
            )
            
            if detected_faces:
                face_detected = "Yes"
                face_count = len(detected_faces)
        except Exception as e:
            face_detected = f"Error: {e}"
            face_count = 0
            logger.error(f"Error processing {image_path}: {e}")
        
        elapsed_time = time.time() - start_time
        results_table.append([image_name, face_detected, face_count, f"{elapsed_time:.2f}"])
    
    print(tabulate(results_table, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    test_folders = [
    
        "/home/zezen/Pictures/Gembooth_Google",
        "/home/zezen/Pictures/New"
    ]

    total_start_time = time.time()

    for folder in test_folders:
        process_folder(folder)
    
    total_elapsed_time = time.time() - total_start_time
    print(f"\n--- Total script elapsed time: {total_elapsed_time:.2f} seconds ---")
