from flask import Flask, request, jsonify
from keras.models import load_model
import tensorflow as tf
import cv2
import os
import numpy as np
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt

#function for preprocessing image
class FACELOADING:
    def __init__(self):
        self.target_size = (160, 160)
        self.detector = MTCNN()

    def extract_face(self, img):
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        results = self.detector.detect_faces(img_rgb)
        
        # Check if a face is detected
        if results:
            # Get the first face (assuming there is only one)
            x, y, w, h = results[0]['box']
            
            # Crop the face region
            cropped_face = img[y:y+h, x:x+w]
            
            # Convert to grayscale and resize
            cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2GRAY)
            cropped_face = cv2.resize(cropped_face, self.target_size)
            
            return cropped_face
        else:
            # Return None if no face is detected
            return None

    def load_faces(self, dir):
        faces = []
        for im_name in os.listdir(dir):
            try:
                path = os.path.join(dir, im_name)
                img = cv2.imread(path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                single_face = self.extract_face(img)
                if single_face is not None:
                    faces.append(single_face)
            except Exception as e:
                pass
        return faces

loader = FACELOADING()

#function for custom loss
def contrastive_loss(y_true, distance):
    margin = 1.0
    return tf.reduce_mean(y_true * tf.square(distance) + (1 - y_true) * tf.square(tf.maximum(margin - distance, 0)))

app = Flask(__name__)

custom_objects = {'contrastive_loss': contrastive_loss}
model = load_model('facematching.h5', custom_objects=custom_objects)

@app.route("/data/base_image/ojiie", methods=["GET"])
@app.route("/data/verif_image/ojiie", methods=["GET"])
def predict():
    try:
        # preprocess base images and verif images, the count of base image should be the same with verif image

        base_images_dir = './data/base_image/ojiie'
        base_images = loader.load_faces(base_images_dir)
        base_images = np.array(base_images)

        verif_image_dir = './data/verif_image/ilhan'
        verif_image = loader.load_faces(verif_image_dir)
        verif_image = np.array(verif_image)

        prediction = model.predict([base_images, verif_image])

        return jsonify({"prediction": float(prediction[0][0])})

    except Exception as e:
        return jsonify({"error": str(e)}), 500  # 500 is the HTTP status code for Internal Server Error

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))