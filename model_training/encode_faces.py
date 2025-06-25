import os   # it is used to get the list of files in a directory
import cv2
import face_recognition
import pickle

def encode_faces(dataset_path, encoding_file):    #defining a function to encode the faces in the dataset and save them to a file

    known_encodings = []
    known_names = []

    for person_name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person_name)    # it is used to get the path of the folder of the person
        if not os.path.isdir(person_folder):                       # it is used to check if the folder is a directory
            continue  

        print(f"[INFO] Processing folder: {person_name}")

        for image_name in os.listdir(person_folder):           # it is used to get the list of files in a directory
            image_path = os.path.join(person_folder, image_name)   # it is used to get the path of the file

            try:
                image = cv2.imread(image_path)    # it is used to read the image

                if image is None:                  # it is used to check if the image is valid or corrupted
                    print(f"[WARNING] Unable to read image: {image_path}")      
                    continue     # it is used to skip the rest of the loop


                image = cv2.resize(image, (256,256))           # it is used to resize the image for faster processing
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)    # it is used to convert the image to RGB as face_recognition expects RGB

                
                encodings = face_recognition.face_encodings(rgb_image)    # it is used to get the encodings of the faces in the image in 128 dimensional vector

                if len(encodings) == 0:    # it is used to check if the image contains any faces
                    print(f"[WARNING] No faces found in image: {image_path}")  
                    continue   

               
                encoding = encodings[0]       # it is used to get the first encoding of the face

                known_encodings.append(encoding)      # it is used to add the encoding to the list
                known_names.append(person_name)       # it is used to add the name of the person to the list

                print(f"[SUCCESS] Encoded {image_path}")

            except Exception as e:        # it is used to catch any exceptions that may occur
                print(f"[ERROR] Failed to process image {image_path}: {str(e)}")

  
    print(f"[INFO] Serializing encodings to {encoding_file} ...")   # it is used to print a message to the user
    data = {"encodings": known_encodings, "names": known_names}      # it is used to store the encodings and names in a dictionary
    with open(encoding_file, "wb") as f:       # it is used to open the file in write binary mode
        pickle.dump(data, f)                    # it is used to dump the data to the file as a pickle object and save it

    print("[INFO] Encoding process complete!")



if __name__ == "__main__":             # it is used to check if the file is being run as the main program to keep the code clean
    dataset_path = (r"C:\downloads\FACE_REC\dataset")  
    encoding_file = "encodings.pickle"
    encode_faces(dataset_path, encoding_file)

