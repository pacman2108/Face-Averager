import cv2
import mediapipe as mp
import os

def extract_landmarks(folder_path, output_ext='.txt'):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)

            if img is None:
                print(f"[!] Couldn't read {filename}")
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(img_rgb)

            if not result.multi_face_landmarks:
                print(f"[!] No face detected in {filename}")
                continue

            face_landmarks = result.multi_face_landmarks[0]
            points = []

            for lm in face_landmarks.landmark[:68]:  # First 68 for compatibility
                x = int(lm.x * img.shape[1])
                y = int(lm.y * img.shape[0])
                points.append(f"{x} {y}")

            txt_filename = os.path.splitext(filename)[0] + output_ext
            txt_path = os.path.join(folder_path, txt_filename)

            with open(txt_path, 'w') as f:
                f.write('\n'.join(points))

            print(f"[✓] Saved landmarks for {filename} -> {txt_filename}")

# Example usage
folder = input("File location: ")
extract_landmarks(folder)
