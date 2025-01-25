import dlib
from imutils.face_utils.facealigner import FaceAligner
import tensorflow as tf
import facenet
import cv2

class Encoder:
    def __init__(self, checkpoint="models/20180402-114759"):
        self.sess = tf.Session()
        with self.sess.as_default():
            facenet.load_model(checkpoint)

    def generate_embedding(self, face):
        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        prewhiten_face = facenet.prewhiten(face)

        # Run forward pass to calculate embeddings
        feed_dict = {images_placeholder: [prewhiten_face], phase_train_placeholder: False}
        return self.sess.run(embeddings, feed_dict=feed_dict)[0]

class EnhancedFacenet:
    def __init__(self, desiredFaceWidth=160, predictor_path='models/shape_predictor_68_face_landmarks.dat'):
        self.predictor = dlib.shape_predictor(predictor_path)
        self.fa = FaceAligner(self.predictor, desiredFaceWidth=desiredFaceWidth, desiredLeftEye=(0.37, 0.33))
        self.encoder = Encoder()

    def alignAndEncode(self, img, gray, face_rect):
        face = self.fa.align(img, gray, face_rect)
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        return face, self.encoder.generate_embedding(face_rgb)


if __name__ == '__main__':
    pass

