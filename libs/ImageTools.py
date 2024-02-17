from libs.TaskLooper import *
import cv2, os
import numpy as np
import tensorflow as tf
import dlib
import random

class captureImage(TaskDefault):
    def __init__(self,camID,config):
        self.id = id(self) 
        self.cfg = config
        self.interval = 1 / self.cfg['FREQ_LOOP_HZ']
        self.capture = cv2.VideoCapture(camID) 
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg['FRAME_WIDTH'])
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg['FRAME_HEIGHT'])
        self.stop_thread = False
        self.current_image = None   
        self.image_lock = threading.Lock()

    def update(self,*args,**kwargs):
        with self.image_lock:
            frame = np.array(self.current_image)
        return frame
        
    def daemon(self):   
        while not self.stop_thread:
            loop_start_time = time.time()
            ret, frame = self.capture.read()
            if not ret:
                continue
            with self.image_lock:
                self.current_image = np.array(frame)   
            sleep_time = max(0, (self.interval - (time.time() - loop_start_time)) )
            time.sleep(sleep_time)
    
    def start(self):
        self.thread = threading.Thread(target=self.daemon)
        self.thread.start()
        while self.current_image is None :
            pass
    
    def stop(self):
        self.stop_thread = True
        self.thread.join()
        self.capture.release()  

class saveImage(TaskDefault):
    def __init__(self):
        self.id = id(self) 

    def update(self,frame): 
        frm = frame.copy()
        cv2.imwrite("saved_image.jpg", frm)
        return True
    
class ObjectDetector(TaskDefault):
    def __init__(self, config):
        self.id = id(self)
        self.cfg = config
        self.tflite_model_path = 'mobilenetv2.tflite'  # Chemin vers le fichier TFLite

        # Vérifier si le fichier TFLite existe
        if not os.path.exists(self.tflite_model_path):
            print("Fichier TFLite introuvable. Génération en cours...")
            self.generate_tflite_model()

        # Charger le modèle TFLite
        self.interpreter = tf.lite.Interpreter(model_path=self.tflite_model_path)
        self.interpreter.allocate_tensors()

    def generate_tflite_model(self):
        # Charger le modèle MobileNetV2
        model = tf.keras.applications.MobileNetV2(weights='imagenet', input_shape=(224, 224, 3))

        # Convertir le modèle en format TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        # Print information after conversion
        print("After converting to TFLite")
        print("Model size:", len(tflite_model))

        # Enregistrer le modèle converti en tant que fichier TFLite
        with open(self.tflite_model_path, 'wb') as f:
            f.write(tflite_model)

    def preprocess_input(self, image):
        # Prétraitement de l'image pour s'adapter aux exigences de MobileNetV2
        image = cv2.resize(image, (224, 224))
        processed_image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        return processed_image    

    def decode_predictions(self, predictions):
        # Décodage des prédictions en noms de classe
        decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions)
        return decoded_predictions

    def draw_bounding_boxes(self, image, boxes, labels):
        for box, label in zip(boxes, labels):
            xmin, ymin, xmax, ymax = box
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
            color = (0, 255, 0)  # Couleur verte
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(image, label, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def update(self, image):
        # Prétraitement de l'image
        processed_image = self.preprocess_input(image)

        # Élargir les dimensions pour s'adapter au modèle (batch_size = 1)
        input_tensor = tf.expand_dims(processed_image, 0)

        # Définir les valeurs d'entrée du modèle TFLite
        input_details = self.interpreter.get_input_details()
        self.interpreter.set_tensor(input_details[0]['index'], input_tensor)

        # Effectuer l'inférence
        self.interpreter.invoke()

        # Obtenir les résultats de l'inférence
        output_details = self.interpreter.get_output_details()
        output_data = self.interpreter.get_tensor(output_details[0]['index'])

        # Exemple : utiliser des boîtes englobantes arbitraires pour la démo
        # Dans la pratique, vous devrez utiliser un modèle de détection d'objet approprié.
        boxes = [[50, 50, 150, 150]]  # Coordonnées xmin, ymin, xmax, ymax
        labels = ["Object"]

        # Dessiner les boîtes englobantes sur l'image
        self.draw_bounding_boxes(image, boxes, labels)

        # Décodage des prédictions
        decoded_predictions = self.decode_predictions(output_data)

        # Afficher les résultats
        for i, (imagenet_id, label, score) in enumerate(decoded_predictions[0]):
            print(f"{i + 1}: {label} ({score:.2f})")

        return image


class FaceDetector(TaskDefault):
    def __init__(self, config):
        self.id = id(self)
        self.cfg = config
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def update(self, image):
        frame = image.copy()
        yeux = image.copy()
        bouche = image.copy()
        nez = image.copy()

        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(grayscale, scaleFactor=1.3, minNeighbors=5)
        print(faces)
        for (x, y, w, h) in faces:
            # Redimensionner les images des yeux, de la bouche et du nez pour les adapter aux dimensions du visage
            yeux_resized = cv2.resize(yeux, (w, h))
            bouche_resized = cv2.resize(bouche, (w, h))
            nez_resized = cv2.resize(nez, (w, h))

            # Placer les yeux, la bouche et le nez dans le visage
            frame[y:y+h, x:x+w, 0:3] = (frame[y:y+h, x:x+w, 0:3] * (1 - yeux_resized[:, :, 3:] / 255.0) +
                                         yeux_resized[:, :, 0:3] * (yeux_resized[:, :, 3:] / 255.0))

            frame[y:y+h, x:x+w, 0:3] = (frame[y:y+h, x:x+w, 0:3] * (1 - bouche_resized[:, :, 3:] / 255.0) +
                                         bouche_resized[:, :, 0:3] * (bouche_resized[:, :, 3:] / 255.0))

            frame[y:y+h, x:x+w, 0:3] = (frame[y:y+h, x:x+w, 0:3] * (1 - nez_resized[:, :, 3:] / 255.0) +
                                         nez_resized[:, :, 0:3] * (nez_resized[:, :, 3:] / 255.0))

        return frame

    