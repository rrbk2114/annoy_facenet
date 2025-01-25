import os
import cv2
import numpy as np
from annoy import AnnoyIndex
from enhancedfacenet import EnhancedFacenet
from facedetector import FaceDetector

def train_and_build_index(data_path, annoy_file):
    ef = EnhancedFacenet()
    face_detector = FaceDetector()

    dataset = []
    for label in os.listdir(data_path):
        label_path = os.path.join(data_path, label)
        if os.path.isdir(label_path):
            for img_name in os.listdir(label_path):
                img_path = os.path.join(label_path, img_name)
                dataset.append((img_path, label))

    embeddings = []
    names = []

    for img_path, label in dataset:
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = face_detector.detect(img)

        if rects:
            _, embedding = ef.alignAndEncode(img, gray, rects[0])
            embeddings.append(embedding)
            names.append(label)

    f = len(embeddings[0])
    annoy_index = AnnoyIndex(f, metric="euclidean")

    for i, embedding in enumerate(embeddings):
        annoy_index.add_item(i, embedding)

    annoy_index.build(10)
    annoy_index.save(annoy_file)

    np.save('names.npy', np.array(names))

    print("Training complete and Annoy index built.")

def live_recognition(annoy_file):
    ef = EnhancedFacenet()
    face_detector = FaceDetector()

    annoy_index = AnnoyIndex(512, metric="euclidean")
    annoy_index.load(annoy_file)

    names = np.load('names.npy')

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rects = face_detector.detect(frame)

        for rect in rects:
            x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()
            face, embedding = ef.alignAndEncode(frame, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), rect)

            indices, distances = annoy_index.get_nns_by_vector(embedding, 1, include_distances=True)

            if distances[0] > 0.6:  # Threshold for unknown face
                label = input("New face detected. Enter label: ")
                save_new_user(label, embedding, names, annoy_index, annoy_file)
            else:
                label = names[indices[0]]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def save_new_user(label, embedding, names, annoy_index, annoy_file):
    names = list(names)
    names.append(label)
    np.save('names.npy', np.array(names))

    new_index = len(names) - 1
    annoy_index.add_item(new_index, embedding)
    annoy_index.build(10)
    annoy_index.save(annoy_file)

if __name__ == "__main__":
    DATA_PATH = r"C:\Users\Rajeshwari\Desktop\face_detection\datas\short"
    ANNOY_FILE = "face_index.ann"

    choice = input("Enter 'train' to train or 'run' to start live recognition: ")

    if choice.lower() == 'train':
        train_and_build_index(DATA_PATH, ANNOY_FILE)
    elif choice.lower() == 'run':
        live_recognition(ANNOY_FILE)

