"""
Script de pré-processamento de dados para reconhecimento facial
Suporta datasets LFW e ORL Faces Dataset
"""

import os
import cv2
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from pathlib import Path


class FaceDataPreprocessor:
    """Classe para pré-processar dados de reconhecimento facial"""
    
    def __init__(self, target_size=(128, 128)):
        """
        Inicializa o pré-processador
        
        Args:
            target_size: Tamanho das imagens após redimensionamento (altura, largura)
        """
        self.target_size = target_size
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    
    def detect_face(self, image):
        """
        Detecta rosto em uma imagem
        
        Args:
            image: Imagem em formato numpy array
            
        Returns:
            Imagem com rosto detectado e recortado, ou None se não detectar
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return None
        
        # Pega o maior rosto detectado
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, self.target_size)
        
        return face
    
    def load_lfw_dataset(self, data_dir):
        """
        Carrega dataset LFW (Labeled Faces in the Wild)
        Estrutura esperada: data_dir/person_name/image.jpg
        
        Args:
            data_dir: Diretório raiz do dataset LFW
            
        Returns:
            X: Array de imagens
            y: Array de labels (nomes das pessoas)
        """
        if not os.path.exists(data_dir):
            raise ValueError(f"ERRO: Diretorio nao encontrado: {data_dir}")
        
        if not os.path.isdir(data_dir):
            raise ValueError(f"ERRO: {data_dir} nao e um diretorio")
        
        X = []
        y = []
        
        person_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        
        if len(person_dirs) == 0:
            raise ValueError(f"ERRO: Nenhum diretorio de pessoa encontrado em {data_dir}")
        
        print(f"  Encontrados {len(person_dirs)} diretorios de pessoas")
        
        loaded = 0
        failed = 0
        no_face = 0
        
        for person_dir in person_dirs:
            person_path = os.path.join(data_dir, person_dir)
            
            image_files = [f for f in os.listdir(person_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for img_file in image_files:
                img_path = os.path.join(person_path, img_file)
                image = cv2.imread(img_path)
                
                if image is None:
                    failed += 1
                    continue
                
                face = self.detect_face(image)
                if face is not None:
                    X.append(face)
                    y.append(person_dir)
                    loaded += 1
                else:
                    no_face += 1
        
        print(f"  Carregadas: {loaded} imagens, Sem rosto: {no_face}, Falharam: {failed}")
        
        if len(X) == 0:
            raise ValueError(f"ERRO: Nenhuma imagem com rosto detectado em {data_dir}")
        
        return np.array(X), np.array(y)
    
    def load_orl_dataset(self, data_dir):
        """
        Carrega dataset ORL Faces
        Estrutura esperada: data_dir/s{person_id}_{image_id}.pgm
        
        Args:
            data_dir: Diretório raiz do dataset ORL
            
        Returns:
            X: Array de imagens
            y: Array de labels (IDs das pessoas)
        """
        if not os.path.exists(data_dir):
            raise ValueError(f"ERRO: Diretorio nao encontrado: {data_dir}")
        
        if not os.path.isdir(data_dir):
            raise ValueError(f"ERRO: {data_dir} nao e um diretorio")
        
        X = []
        y = []
        
        files = os.listdir(data_dir)
        if len(files) == 0:
            raise ValueError(f"ERRO: Diretorio esta vazio: {data_dir}")
        
        print(f"  Verificando {len(files)} arquivos no diretorio...")
        
        image_files = [f for f in files if f.lower().endswith(('.pgm', '.jpg', '.jpeg', '.png'))]
        print(f"  Encontrados {len(image_files)} arquivos de imagem")
        
        if len(image_files) == 0:
            raise ValueError(f"ERRO: Nenhum arquivo de imagem encontrado em {data_dir}")
        
        loaded = 0
        failed = 0
        
        for img_file in image_files:
            # ORL: s{person_id}_{image_id}.pgm
            # Exemplo: s1_1.pgm -> pessoa 1, imagem 1
            parts = img_file.split('_')
            if len(parts) >= 2:
                person_id = parts[0].replace('s', '')
            else:
                person_id = img_file.split('.')[0]
            
            img_path = os.path.join(data_dir, img_file)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if image is None:
                failed += 1
                continue
            
            # Redimensiona para o tamanho alvo
            face = cv2.resize(image, self.target_size)
            X.append(face)
            y.append(person_id)
            loaded += 1
        
        print(f"  Carregadas: {loaded} imagens, Falharam: {failed} imagens")
        
        if len(X) == 0:
            raise ValueError(f"ERRO: Nenhuma imagem foi carregada com sucesso de {data_dir}")
        
        return np.array(X), np.array(y)
    
    def normalize_images(self, X):
        """
        Normaliza imagens para o range [0, 1]
        
        Args:
            X: Array de imagens
            
        Returns:
            Array de imagens normalizadas
        """
        X = X.astype('float32') / 255.0
        return X
    
    def prepare_data(self, X, y, test_size=0.2, val_size=0.2):
        """
        Prepara dados para treinamento
        
        Args:
            X: Array de imagens
            y: Array de labels
            test_size: Proporção do conjunto de teste
            val_size: Proporção do conjunto de validação (do conjunto de treino)
            
        Returns:
            Tupla com (X_train, X_val, X_test, y_train, y_val, y_test, label_encoder)
        """
        if len(X) == 0:
            raise ValueError("ERRO: Dataset vazio! Nenhuma imagem foi carregada.")
        
        if len(y) == 0:
            raise ValueError("ERRO: Nenhum label encontrado!")
        
        if len(X) != len(y):
            raise ValueError(f"ERRO: Numero de imagens ({len(X)}) diferente do numero de labels ({len(y)})")
        
        # Normaliza imagens
        X = self.normalize_images(X)
        
        # Adiciona dimensão de canal se necessário
        if len(X.shape) == 3:
            X = np.expand_dims(X, axis=-1)
        
        # Codifica labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Verifica se há amostras suficientes para divisão
        min_samples = max(2, int(1 / (1 - test_size) / (1 - val_size)) + 1)
        if len(X) < min_samples:
            print(f"AVISO: Dataset muito pequeno ({len(X)} amostras). Ajustando divisao...")
            if len(X) < 3:
                raise ValueError(f"ERRO: Dataset muito pequeno ({len(X)} amostras). Precisa de pelo menos 3 amostras.")
            # Ajusta test_size e val_size para datasets pequenos
            test_size = min(test_size, 0.3)
            val_size = min(val_size, 0.3)
        
        # Verifica se podemos usar stratify (precisa de pelo menos 2 amostras por classe)
        class_counts = Counter(y_encoded)
        min_class_count = min(class_counts.values())
        use_stratify = min_class_count >= 2
        
        if not use_stratify:
            print(f"AVISO: Algumas classes tem apenas 1 amostra. Usando divisao sem estratificacao.")
            print(f"  Classes com 1 amostra: {sum(1 for c in class_counts.values() if c == 1)}")
            print(f"  Classes com 2+ amostras: {sum(1 for c in class_counts.values() if c >= 2)}")
        
        # Divide em treino e teste
        if use_stratify:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=test_size, random_state=42
            )
        
        # Divide treino em treino e validação
        # Verifica se podemos usar stratify na divisão de validação
        train_class_counts = Counter(y_train)
        min_train_class_count = min(train_class_counts.values())
        use_stratify_val = min_train_class_count >= 2
        
        if use_stratify_val:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=val_size, random_state=42, stratify=y_train
            )
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=val_size, random_state=42
            )
        
        return X_train, X_val, X_test, y_train, y_val, y_test, label_encoder
    
    def visualize_samples(self, X, y, label_encoder, n_samples=9):
        """
        Visualiza amostras do dataset
        
        Args:
            X: Array de imagens
            y: Array de labels codificados
            label_encoder: LabelEncoder usado para codificar labels
            n_samples: Número de amostras para visualizar
        """
        fig, axes = plt.subplots(3, 3, figsize=(10, 10))
        axes = axes.ravel()
        
        indices = np.random.choice(len(X), n_samples, replace=False)
        
        for i, idx in enumerate(indices):
            img = X[idx].squeeze()
            label = label_encoder.inverse_transform([y[idx]])[0]
            
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f'Label: {label}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('data_samples.png', dpi=150, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    # Exemplo de uso
    preprocessor = FaceDataPreprocessor(target_size=(128, 128))
    
    # Para usar com LFW dataset
    # X, y = preprocessor.load_lfw_dataset('data/lfw')
    
    # Para usar com ORL dataset
    # X, y = preprocessor.load_orl_dataset('data/orl')
    
    print("Pré-processador de dados criado com sucesso!")
    print("Use este módulo para carregar e preparar seus datasets.")


