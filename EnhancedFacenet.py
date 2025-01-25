import os
import cv2
import numpy as np
import dlib
import tensorflow as tf
import torch
from annoy import AnnoyIndex
from typing import List, Tuple

class FaceDetector:
    def __init__(self, 
                 ssd_model_file="models/res10_300x300_ssd_iter_140000_fp16.caffemodel", 
                 ssd_config_file="models/deploy.prototxt"):
        self.conf_threshold = 0.63
        self.net = cv2.dnn.readNet(ssd_config_file, ssd_model_file)

    def detect(self, img: np.ndarray) -> List[dlib.rectangle]:
        """
        Detects faces in the given image using SSD model.
        :param img: Input image as a NumPy array.
        :return: List of bounding rectangles.
        """
        # Histogram equalization for better detection
        b, g, r = cv2.split(img)
        b = cv2.equalizeHist(b)
        g = cv2.equalizeHist(g)
        r = cv2.equalizeHist(r)
        img_he = cv2.merge((b, g, r))

        blob = cv2.dnn.blobFromImage(img_he, 1.0, (300, 300), [104, 117, 123], False, False)
        self.net.setInput(blob)

        frameHeight, frameWidth = img.shape[:2]
        detections = self.net.forward()

        rects = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                rects.append(dlib.rectangle(x1, y1, x2, y2))

        return rects

class FaceEncoder:
    def __init__(self, model_path='models/facenet_model.pb'):
        """
        Initialize face encoder using a pre-trained FaceNet model
        """
        self.graph = tf.Graph()
        with self.graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.sess = tf.Session(graph=self.graph)
        self.input_tensor = self.graph.get_tensor_by_name('input:0')
        self.phase_train_tensor = self.graph.get_tensor_by_name('phase_train:0')
        self.embeddings_tensor = self.graph.get_tensor_by_name('embeddings:0')

    def prewhiten(self, x):
        """
        Preprocess input image
        """
        mean = np.mean(x)
        std = np.max(np.std(x), 1/np.sqrt(x.size))
        std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
        y = np.multiply(np.subtract(x, mean), 1/std_adj)
        return y

    def encode_face(self, face: np.ndarray) -> np.ndarray:
        """
        Generate face embedding
        """
        # Preprocess the face
        face_preprocessed = cv2.resize(face, (160, 160))
        face_preprocessed = cv2.cvtColor(face_preprocessed, cv2.COLOR_BGR2RGB)
        face_preprocessed = self.prewhiten(face_preprocessed)

        # Generate embedding
        feed_dict = {
            self.input_tensor: [face_preprocessed],
            self.phase_train_tensor: False
        }
        embeddings = self.sess.run(self.embeddings_tensor, feed_dict=feed_dict)[0]
        return embeddings

class FaceRecognitionSystem:
    def __init__(self, data_path: str, index_file: str = 'face_index.ann'):
        """
        Initialize face recognition system
        """
        self.data_path = data_path
        self.index_file = index_file
        self.face_detector = FaceDetector()
        self.face_encoder = FaceEncoder()
        
        # Face alignment
        self.predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
        self.aligner = dlib.faceAligner.FaceAligner(self.predictor, desiredFaceWidth=160)

    def train_system(self):
        """
        Train the face recognition system
        """
        # Initialize Annoy index
        embeddings = []
        labels = []

        # Iterate through dataset
        for label in os.listdir(self.data_path):
            label_path = os.path.join(self.data_path, label)
            if os.path.isdir(label_path):
                for img_name in os.listdir(label_path):
                    img_path = os.path.join(label_path, img_name)
                    
                    # Read and process image
                    img = cv2.imread(img_path)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    
                    # Detect faces
                    rects = self.face_detector.detect(img)
                    
                    # Process first detected face
                    if rects:
                        # Align face
                        aligned_face = self.aligner.align(img, gray, rects[0])
                        
                        # Generate embedding
                        embedding = self.face_encoder.encode_face(aligned_face)
                        
                        embeddings.append(embedding)
                        labels.append(label)

        # Build Annoy index
        if embeddings:
            f = len(embeddings[0])
            index = AnnoyIndex(f, 'euclidean')
            
            for i, emb in enumerate(embeddings):
                index.add_item(i, emb)
            
            index.build(10)
            index.save(self.index_file)
            
            # Save labels
            np.save('face_labels.npy', np.array(labels))
            
            print(f"Trained on {len(labels)} faces")
        else:
            print("No faces detected during training")

    def recognize_faces(self, input_source=0, threshold: float = 0.6):
        """
        Recognize faces in video stream
        """
        # Load trained index
        index = AnnoyIndex(512, 'euclidean')
        index.load(self.index_file)
        
        # Load labels
        labels = np.load('face_labels.npy')

        # Open video capture
        cap = cv2.VideoCapture(input_source)

        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break

            # Detect faces
            rects = self.face_detector.detect(frame)

            for rect in rects:
                # Convert rect to bounding box coordinates
                x1, y1 = rect.left(), rect.top()
                x2, y2 = rect.right(), rect.bottom()

                # Extract and align face
                face_crop = frame[y1:y2, x1:x2]
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                aligned_face = self.aligner.align(frame, gray, rect)

                # Generate embedding
                embedding = self.face_encoder.encode_face(aligned_face)

                # Find nearest neighbor
                indices, distances = index.get_nns_by_vector(embedding, 1, include_distances=True)

                # Determine label
                if distances[0] <= threshold:
                    label = labels[indices[0]]
                else:
                    label = "Unknown"

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

            # Display frame
            cv2.imshow("Face Recognition", frame)

            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()

def main():
    DATA_PATH = r"C:\Users\Rajeshwari\Desktop\face_detection\datas\short"
    
    # Create face recognition system
    face_system = FaceRecognitionSystem(DATA_PATH)

    # Menu
    while True:
        print("\n1. Train System")
        print("2. Live Recognition")
        print("3. Exit")
        
        choice = input("Enter your choice: ")

        if choice == '1':
            face_system.train_system()
        elif choice == '2':
            face_system.recognize_faces()
        elif choice == '3':
            break
        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    main()
