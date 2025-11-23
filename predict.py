"""
Script de predição/inferência para reconhecimento facial
"""

import os
import sys
import cv2
import numpy as np
import tensorflow as tf

# Compatibilidade com diferentes versões do TensorFlow/Keras
try:
    # Tenta primeiro Keras 3.x (standalone)
    import keras
except ImportError:
    try:
        # Tenta Keras 2.x via TensorFlow
        from tensorflow import keras
    except ImportError as e:
        print("❌ ERRO: Não foi possível importar Keras!")
        print(f"   Erro: {e}")
        print("\n💡 SOLUÇÃO:")
        print("   Execute: pip install tensorflow")
        sys.exit(1)

import pickle
import matplotlib.pyplot as plt
from data_preprocessing import FaceDataPreprocessor


class FaceRecognizer:
    """Classe para reconhecimento facial usando modelo treinado"""
    
    def __init__(self, model_path, label_encoder_path, target_size=(128, 128)):
        """
        Inicializa o reconhecedor facial
        
        Args:
            model_path: Caminho para o modelo treinado (.h5)
            label_encoder_path: Caminho para o label encoder (.pkl)
            target_size: Tamanho das imagens usado no treinamento
        """
        self.model = keras.models.load_model(model_path)
        self.target_size = target_size
        self.preprocessor = FaceDataPreprocessor(target_size=target_size)
        
        with open(label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        print(f"Modelo carregado de: {model_path}")
        print(f"Label encoder carregado de: {label_encoder_path}")
        print(f"Número de classes: {len(self.label_encoder.classes_)}")
    
    def predict_image(self, image_path, show_result=True):
        """
        Prediz a identidade de uma pessoa em uma imagem
        
        Args:
            image_path: Caminho para a imagem
            show_result: Se True, mostra a imagem com predição
            
        Returns:
            Tupla (predicted_label, confidence, all_predictions)
        """
        # Carrega imagem
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Não foi possível carregar a imagem: {image_path}")
        
        # Detecta rosto
        face = self.preprocessor.detect_face(image)
        if face is None:
            print("Nenhum rosto detectado na imagem!")
            return None, 0.0, None
        
        # Prepara imagem
        face_normalized = face.astype('float32') / 255.0
        if len(face_normalized.shape) == 2:
            face_normalized = np.expand_dims(face_normalized, axis=-1)
        face_normalized = np.expand_dims(face_normalized, axis=0)
        
        # Predição
        predictions = self.model.predict(face_normalized, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        predicted_label = self.label_encoder.inverse_transform([predicted_class])[0]
        
        # Todas as predições (top 5)
        top_5_indices = np.argsort(predictions[0])[-5:][::-1]
        top_5_predictions = [
            (self.label_encoder.inverse_transform([idx])[0], predictions[0][idx])
            for idx in top_5_indices
        ]
        
        if show_result:
            self._visualize_prediction(image, face, predicted_label, confidence, top_5_predictions)
        
        return predicted_label, confidence, top_5_predictions
    
    def predict_from_camera(self):
        """
        Predição em tempo real usando webcam
        """
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Erro ao abrir a câmera!")
            return
        
        print("Pressione 'q' para sair")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detecta rosto
            face = self.preprocessor.detect_face(frame)
            
            if face is not None:
                # Prepara imagem
                face_normalized = face.astype('float32') / 255.0
                if len(face_normalized.shape) == 2:
                    face_normalized = np.expand_dims(face_normalized, axis=-1)
                face_normalized = np.expand_dims(face_normalized, axis=0)
                
                # Predição
                predictions = self.model.predict(face_normalized, verbose=0)
                predicted_class = np.argmax(predictions[0])
                confidence = predictions[0][predicted_class]
                predicted_label = self.label_encoder.inverse_transform([predicted_class])[0]
                
                # Desenha resultado no frame
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.preprocessor.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    label_text = f"{predicted_label} ({confidence:.2f})"
                    cv2.putText(frame, label_text, (x, y-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            cv2.imshow('Reconhecimento Facial', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def _visualize_prediction(self, original_image, face, predicted_label, confidence, top_5_predictions):
        """
        Visualiza resultado da predição
        
        Args:
            original_image: Imagem original
            face: Rosto detectado
            predicted_label: Label predito
            confidence: Confiança da predição
            top_5_predictions: Top 5 predições
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Imagem original com rosto detectado
        axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Imagem Original')
        axes[0].axis('off')
        
        # Rosto detectado
        axes[1].imshow(face, cmap='gray')
        axes[1].set_title(f'Predito: {predicted_label}\nConfiança: {confidence:.2%}')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Mostra top 5 predições
        print("\nTop 5 Predições:")
        print("-" * 40)
        for i, (label, conf) in enumerate(top_5_predictions, 1):
            print(f"{i}. {label}: {conf:.2%}")


def batch_predict(image_dir, model_path, label_encoder_path, output_file='predictions.txt'):
    """
    Faz predições em lote para todas as imagens em um diretório
    
    Args:
        image_dir: Diretório com imagens
        model_path: Caminho para o modelo
        label_encoder_path: Caminho para o label encoder
        output_file: Arquivo para salvar resultados
    """
    recognizer = FaceRecognizer(model_path, label_encoder_path)
    
    results = []
    image_files = [f for f in os.listdir(image_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        try:
            label, confidence, _ = recognizer.predict_image(img_path, show_result=False)
            results.append(f"{img_file}: {label} (confiança: {confidence:.2%})")
            print(f"{img_file}: {label} ({confidence:.2%})")
        except Exception as e:
            results.append(f"{img_file}: Erro - {str(e)}")
            print(f"Erro ao processar {img_file}: {str(e)}")
    
    # Salva resultados
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(results))
    
    print(f"\nResultados salvos em: {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Predição de reconhecimento facial')
    parser.add_argument('--model', type=str, default='models/best_model.h5',
                       help='Caminho para o modelo treinado')
    parser.add_argument('--label_encoder', type=str, default='models/label_encoder.pkl',
                       help='Caminho para o label encoder')
    parser.add_argument('--image', type=str, default=None,
                       help='Caminho para uma imagem para predição')
    parser.add_argument('--image_dir', type=str, default=None,
                       help='Diretório com imagens para predição em lote')
    parser.add_argument('--camera', action='store_true',
                       help='Usar câmera para predição em tempo real')
    
    args = parser.parse_args()
    
    if args.camera:
        recognizer = FaceRecognizer(args.model, args.label_encoder)
        recognizer.predict_from_camera()
    elif args.image:
        recognizer = FaceRecognizer(args.model, args.label_encoder)
        label, confidence, top_5 = recognizer.predict_image(args.image)
        if label:
            print(f"\nPredição: {label}")
            print(f"Confiança: {confidence:.2%}")
    elif args.image_dir:
        batch_predict(args.image_dir, args.model, args.label_encoder)
    else:
        print("Especifique --image, --image_dir ou --camera")


