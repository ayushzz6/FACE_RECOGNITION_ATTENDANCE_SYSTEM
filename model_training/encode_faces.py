import os
import cv2
import face_recognition
import pickle

def encode_faces(dataset_path, encoding_file):

    known_encodings = []
    known_names = []

    for person_name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_folder):
            continue  

        print(f"[INFO] Processing folder: {person_name}")

        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)

            try:
                # Image load करो OpenCV से
                image = cv2.imread(image_path)

                if image is None:
                    print(f"[WARNING] Unable to read image: {image_path}")
                    continue


                
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                
                encodings = face_recognition.face_encodings(rgb_image)

                if len(encodings) == 0:
                    print(f"[WARNING] No faces found in image: {image_path}")
                    continue

               
                encoding = encodings[0]

                known_encodings.append(encoding)
                known_names.append(person_name)

                print(f"[SUCCESS] Encoded {image_path}")

            except Exception as e:
                print(f"[ERROR] Failed to process image {image_path}: {str(e)}")

  
    print(f"[INFO] Serializing encodings to {encoding_file} ...")
    data = {"encodings": known_encodings, "names": known_names}
    with open(encoding_file, "wb") as f:
        pickle.dump(data, f)

    print("[INFO] Encoding process complete!")



if __name__ == "__main__":
    dataset_path = (r"C:\downloads\FACE_REC\dataset")  
    encoding_file = "encodings.pickle"
    encode_faces(dataset_path, encoding_file)

