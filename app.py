import os
import cv2
import numpy as np
from annoy import AnnoyIndex
from typing import List, Tuple
import dlib

class FaceRecognitionSystem:
    def __init__(self, data_path: str, model_path: str = 'models'):
        """Initialize face recognition system"""
        self.data_path = data_path
        self.model_path = model_path
        
        # Load face detection and recognition models
        self.detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(
            os.path.join(model_path, 'shape_predictor_68_face_landmarks.dat')
        )
        self.face_recognizer = dlib.face_recognition_model_v1(
            os.path.join(model_path, 'dlib_face_recognition_resnet_model_v1.dat')
        )

    def _align_face(self, img: np.ndarray, rect: dlib.rectangle) -> np.ndarray:
        """Align detected face"""
        shape = self.shape_predictor(img, rect)
        face_aligned = dlib.get_face_chip(img, shape)
        return face_aligned

    def train_system(self, index_file: str = 'face_index.ann') -> None:
        """Train face recognition system"""
        embeddings = []
        labels = []

        for label in os.listdir(self.data_path):
            label_path = os.path.join(self.data_path, label)
            if os.path.isdir(label_path):
                for img_name in os.listdir(label_path):
                    img_path = os.path.join(label_path, img_name)
                    
                    # Read image
                    img = cv2.imread(img_path)
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Detect faces
                    faces = self.detector(img_rgb)
                    
                    for rect in faces:
                        # Align and encode face
                        aligned_face = self._align_face(img_rgb, rect)
                        embedding = np.array(
                            self.face_recognizer.compute_face_descriptor(aligned_face)
                        )
                        
                        embeddings.append(embedding)
                        labels.append(label)

        # Build Annoy index
        if embeddings:
            index = AnnoyIndex(len(embeddings[0]), 'euclidean')
            
            for i, emb in enumerate(embeddings):
                index.add_item(i, emb)
            
            index.build(10)
            index.save(index_file)
            
            # Save labels
            np.save('face_labels.npy', np.array(labels))
            print(f"Trained on {len(labels)} faces")

    def recognize_faces(self, input_source: int = 0, threshold: float = 0.6) -> None:
        """Real-time face recognition"""
        # Load trained index and labels
        index = AnnoyIndex(128, 'euclidean')
        index.load('face_index.ann')
        labels = np.load('face_labels.npy')

        cap = cv2.VideoCapture(input_source)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = self.detector(frame_rgb)

            for rect in faces:
                # Align face and get embedding
                aligned_face = self._align_face(frame_rgb, rect)
                embedding = np.array(
                    self.face_recognizer.compute_face_descriptor(aligned_face)
                )

                # Find nearest neighbor
                indices, distances = index.get_nns_by_vector(embedding, 1, include_distances=True)

                # Determine label
                label = labels[indices[0]] if distances[0] <= threshold else "Unknown"

                # Draw bounding box
                x1, y1 = rect.left(), rect.top()
                x2, y2 = rect.right(), rect.bottom()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

def main():
    DATA_PATH = r"C:\Users\Rajeshwari\Desktop\face_detection\datas\short"
    face_system = FaceRecognitionSystem(DATA_PATH)

    while True:
        print("\n1. Train System\n2. Live Recognition\n3. Exit")
        choice = input("Enter choice: ")

        if choice == '1':
            face_system.train_system()
        elif choice == '2':
            face_system.recognize_faces()
        elif choice == '3':
            break
        else:
            print("Invalid choice")

if __name__ == "__main__":
    main()
