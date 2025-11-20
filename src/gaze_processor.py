import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanAbsoluteError
import os
import sys

CUSTOM_KERAS_OBJECTS = {'mae': MeanAbsoluteError} 
MODEL_PATH = 'models/mobilenet_gaze_tl_best.h5' 
IMAGE_SIZE = 224 

class GazeModelProcessor:
    def __init__(self, screen_w, screen_h):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.image_size = IMAGE_SIZE
        self.model = self._load_model()
        self.face_cascade = self._load_haar_cascade()

    def _load_model(self):
        try:
            # pega o caminho e carrega o modelo
            script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
            model_path_full = os.path.join(script_dir, MODEL_PATH)

            model = load_model(model_path_full, custom_objects=CUSTOM_KERAS_OBJECTS, compile=False)
            print("Modelo de Gaze carregado com sucesso.")
            return model
        except Exception as e:
            print(f"ERRO ao carregar o modelo de Gaze '{MODEL_PATH}': {e}")
            return None

    def _load_haar_cascade(self):
        try:
            # Carrega o detector de face do OpenCV (Haar Cascade)
            haar_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
            if not os.path.exists(haar_path):
                 print(f"ATENÇÃO: Arquivo Haar Cascade não encontrado em: {haar_path}")
                 # Tenta um caminho relativo se o padrão falhar
                 haar_path = 'haarcascade_frontalface_default.xml' 
                 if not os.path.exists(haar_path):
                     print("ERRO: Detecção facial Haar Cascade indisponível.")
                     return None
            
            return cv2.CascadeClassifier(haar_path)
        except Exception as e:
            print(f"ERRO ao carregar Haar Cascade: {e}")
            return None

    def get_gaze_coordinates(self, full_frame):
        if self.model is None or self.face_cascade is None:
            return None, None

        h, w, _ = full_frame.shape
        gray = cv2.cvtColor(full_frame, cv2.COLOR_BGR2GRAY)
        
        # Detecção de face usando Haar Cascade
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

        if len(faces) == 0:
            return None, None 

        # Escolhe a primeira face detectada
        (x, y, fw, fh) = faces[0]

        # Define a caixa para o corte
        margin = 30
        x_min, y_min = int(max(0, x - margin)), int(max(0, y - margin))
        x_max, y_max = int(min(w, x + fw + margin)), int(min(h, y + fh + margin))
        
        # Corta a imagem do rosto
        face_img = full_frame[y_min:y_max, x_min:x_max]
        
        if face_img.size == 0:
            return None, None

        # Redimensiona a imagem
        input_img = cv2.resize(face_img, (self.image_size, self.image_size))
        # Normaliza a imagem
        input_img = input_img.astype(np.float32) / 255.0
        input_tensor = np.expand_dims(input_img, axis=0)
        
        # Previsão do modelo
        predictions = self.model.predict(input_tensor, verbose=0)[0]
        
        # armazena as coordenadas do modelo
        normalized_x, normalized_y = predictions[0], predictions[1]
        
        # Converte para a resolução da webcam/visualização (640x480)
        gaze_x_pixel = int(normalized_x * self.screen_w)
        gaze_y_pixel = int(normalized_y * self.screen_h)

        return gaze_x_pixel, gaze_y_pixel

    def draw_gaze(self, frame, x, y):
        # Desenha o ponto do olhar e o texto
        if x is not None and y is not None:
            cv2.circle(frame, (x, y), 15, (0, 0, 255), -1)
            cv2.putText(frame, f"Gaze: ({x}, {y})", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame