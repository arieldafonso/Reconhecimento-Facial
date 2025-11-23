"""
Exemplo de uso do sistema de reconhecimento facial
Este script demonstra como usar o sistema de forma programática
"""

import os
import numpy as np
from data_preprocessing import FaceDataPreprocessor
from model import create_face_recognition_model, compile_model
from predict import FaceRecognizer


def example_data_preprocessing():
    """Exemplo de pré-processamento de dados"""
    print("=" * 50)
    print("Exemplo: Pré-processamento de Dados")
    print("=" * 50)
    
    # Inicializa pré-processador
    preprocessor = FaceDataPreprocessor(target_size=(128, 128))
    
    # Carrega dataset ORL (ajuste o caminho conforme necessário)
    data_dir = 'data/orl'
    if os.path.exists(data_dir):
        X, y = preprocessor.load_orl_dataset(data_dir)
        print(f"Dataset carregado: {len(X)} imagens, {len(np.unique(y))} classes")
        
        # Prepara dados
        X_train, X_val, X_test, y_train, y_val, y_test, label_encoder = \
            preprocessor.prepare_data(X, y)
        
        print(f"Treino: {len(X_train)}, Validação: {len(X_val)}, Teste: {len(X_test)}")
        
        # Visualiza amostras
        preprocessor.visualize_samples(X_train, y_train, label_encoder)
    else:
        print(f"Diretório {data_dir} não encontrado. Por favor, baixe o dataset primeiro.")


def example_model_creation():
    """Exemplo de criação de modelo"""
    print("\n" + "=" * 50)
    print("Exemplo: Criação de Modelo")
    print("=" * 50)
    
    # Define parâmetros
    input_shape = (128, 128, 1)
    num_classes = 10
    
    # Cria modelo
    model = create_face_recognition_model(input_shape, num_classes)
    model = compile_model(model)
    
    print("\nModelo criado com sucesso!")
    print(f"Parâmetros totais: {model.count_params():,}")
    model.summary()


def example_prediction():
    """Exemplo de predição"""
    print("\n" + "=" * 50)
    print("Exemplo: Predição")
    print("=" * 50)
    
    model_path = 'models/best_model.h5'
    label_encoder_path = 'models/label_encoder.pkl'
    
    if os.path.exists(model_path) and os.path.exists(label_encoder_path):
        # Inicializa reconhecedor
        recognizer = FaceRecognizer(model_path, label_encoder_path)
        
        # Exemplo de predição em uma imagem
        test_image = 'test_image.jpg'
        if os.path.exists(test_image):
            label, confidence, top_5 = recognizer.predict_image(test_image)
            print(f"\nPredição: {label}")
            print(f"Confiança: {confidence:.2%}")
            print("\nTop 5 predições:")
            for i, (lbl, conf) in enumerate(top_5, 1):
                print(f"  {i}. {lbl}: {conf:.2%}")
        else:
            print(f"Imagem de teste '{test_image}' não encontrada.")
    else:
        print("Modelo não encontrado. Por favor, treine o modelo primeiro.")


def example_complete_pipeline():
    """Exemplo de pipeline completo"""
    print("\n" + "=" * 50)
    print("Exemplo: Pipeline Completo")
    print("=" * 50)
    
    # 1. Pré-processamento
    print("\n1. Pré-processando dados...")
    preprocessor = FaceDataPreprocessor(target_size=(128, 128))
    
    data_dir = 'data/orl'
    if not os.path.exists(data_dir):
        print(f"Dataset não encontrado em {data_dir}")
        return
    
    X, y = preprocessor.load_orl_dataset(data_dir)
    X_train, X_val, X_test, y_train, y_val, y_test, label_encoder = \
        preprocessor.prepare_data(X, y)
    
    # 2. Criar modelo
    print("\n2. Criando modelo...")
    input_shape = X_train.shape[1:]
    num_classes = len(np.unique(y_train))
    model = create_face_recognition_model(input_shape, num_classes)
    model = compile_model(model)
    
    # 3. Treinar (exemplo simplificado)
    print("\n3. Treinando modelo...")
    print("(Para treinamento completo, use: python train.py)")
    
    # 4. Avaliar
    print("\n4. Avaliando modelo...")
    # test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    # print(f"Acurácia no teste: {test_acc:.2%}")
    
    print("\nPipeline completo executado!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        example_type = sys.argv[1]
        
        if example_type == 'preprocessing':
            example_data_preprocessing()
        elif example_type == 'model':
            example_model_creation()
        elif example_type == 'prediction':
            example_prediction()
        elif example_type == 'pipeline':
            example_complete_pipeline()
        else:
            print("Uso: python example_usage.py [preprocessing|model|prediction|pipeline]")
    else:
        # Executa todos os exemplos
        example_data_preprocessing()
        example_model_creation()
        example_prediction()
        example_complete_pipeline()


