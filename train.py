"""
Script de treinamento do modelo de reconhecimento facial
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Compatibilidade com diferentes versões do TensorFlow/Keras
# Keras 3.x (TensorFlow 2.20+) usa import direto
# Keras 2.x (TensorFlow 2.15-) usa from tensorflow import keras
try:
    # Tenta primeiro Keras 3.x (standalone)
    import keras
    from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
    KERAS_SOURCE = "keras3"
except ImportError:
    try:
        # Tenta Keras 2.x via TensorFlow
        from tensorflow import keras
        from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
        KERAS_SOURCE = "tensorflow"
    except ImportError as e:
        print("❌ ERRO: Não foi possível importar Keras!")
        print(f"   Erro: {e}")
        print("\n💡 SOLUÇÃO:")
        print("   Execute: pip install tensorflow")
        sys.exit(1)
from data_preprocessing import FaceDataPreprocessor
from model import create_face_recognition_model, create_lightweight_model, compile_model


def train_model(
    data_dir,
    dataset_type='orl',
    model_type='standard',
    epochs=100,
    batch_size=32,
    target_size=(128, 128),
    save_dir='models'
):
    """
    Treina modelo de reconhecimento facial
    
    Args:
        data_dir: Diretório do dataset
        dataset_type: 'orl' ou 'lfw'
        model_type: 'standard' ou 'lightweight'
        epochs: Número de épocas
        batch_size: Tamanho do batch
        target_size: Tamanho das imagens
        save_dir: Diretório para salvar modelos
    """
    print("=" * 50)
    print("Iniciando treinamento do modelo de reconhecimento facial")
    print("=" * 50)
    
    # Cria diretório para salvar modelos
    os.makedirs(save_dir, exist_ok=True)
    
    # Pré-processamento
    print("\n[1/5] Carregando e pré-processando dados...")
    preprocessor = FaceDataPreprocessor(target_size=target_size)
    
    # Verifica se o diretório existe
    if not os.path.exists(data_dir):
        print(f"\nERRO: Diretorio nao encontrado: {data_dir}")
        print("\nSOLUCAO:")
        print("1. Verifique se o caminho esta correto")
        print("2. Baixe o dataset usando: python download_datasets.py")
        print("3. Ou crie um dataset usando: python capture_faces.py")
        raise FileNotFoundError(f"Diretorio nao encontrado: {data_dir}")
    
    print(f"Carregando dataset de: {data_dir}")
    
    try:
        if dataset_type.lower() == 'orl':
            X, y = preprocessor.load_orl_dataset(data_dir)
        elif dataset_type.lower() == 'lfw':
            X, y = preprocessor.load_lfw_dataset(data_dir)
        else:
            raise ValueError("dataset_type deve ser 'orl' ou 'lfw'")
    except Exception as e:
        print(f"\nERRO ao carregar dataset: {e}")
        print("\nSOLUCAO:")
        print("1. Verifique se o diretorio contem imagens")
        print("2. Para ORL: imagens devem estar em data/orl/ diretamente")
        print("3. Para LFW: cada pessoa deve ter seu proprio diretorio")
        print(f"4. Execute: python check_dataset.py --data_dir {data_dir} --dataset_type {dataset_type}")
        raise
    
    if len(X) == 0:
        print(f"\nERRO: Nenhuma imagem foi carregada de {data_dir}")
        print("\nSOLUCAO:")
        print("1. Verifique se o diretorio contem imagens")
        print(f"2. Execute: python check_dataset.py --data_dir {data_dir} --dataset_type {dataset_type}")
        raise ValueError("Dataset vazio!")
    
    print(f"\nTotal de imagens carregadas: {len(X)}")
    num_total_classes = len(np.unique(y))
    print(f"Numero total de classes: {num_total_classes}")
    
    # Prepara dados
    X_train, X_val, X_test, y_train, y_val, y_test, label_encoder = \
        preprocessor.prepare_data(X, y, test_size=0.2, val_size=0.2)
    
    print(f"\nDivisão dos dados:")
    print(f"  Treino: {len(X_train)} imagens ({len(np.unique(y_train))} classes)")
    print(f"  Validação: {len(X_val)} imagens ({len(np.unique(y_val))} classes)")
    print(f"  Teste: {len(X_test)} imagens ({len(np.unique(y_test))} classes)")
    
    # Verifica se há classes apenas em validação/teste
    train_classes = set(np.unique(y_train))
    val_classes = set(np.unique(y_val))
    test_classes = set(np.unique(y_test))
    all_classes = train_classes | val_classes | test_classes
    
    if len(all_classes) > len(train_classes):
        missing_in_train = all_classes - train_classes
        print(f"\nAVISO: {len(missing_in_train)} classes aparecem apenas em validacao/teste")
        print(f"  Essas classes nao serao treinadas, mas o modelo suportara todas as {len(all_classes)} classes")
    
    # Visualiza amostras
    print("\n[2/5] Visualizando amostras do dataset...")
    preprocessor.visualize_samples(X_train, y_train, label_encoder)
    
    # Cria modelo
    print("\n[3/5] Criando modelo CNN...")
    input_shape = X_train.shape[1:]
    # Usa o número total de classes do label_encoder (todas as classes únicas)
    # Isso garante que o modelo suporte todos os labels possíveis
    num_classes = len(label_encoder.classes_)
    print(f"Numero de classes no modelo: {num_classes} (total de classes unicas)")
    
    if model_type == 'lightweight':
        model = create_lightweight_model(input_shape, num_classes)
    else:
        model = create_face_recognition_model(input_shape, num_classes)
    
    model = compile_model(model, learning_rate=0.001)
    
    print("\nArquitetura do modelo:")
    model.summary()
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(save_dir, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Data augmentation
    # Compatibilidade com diferentes versões
    try:
        from keras.preprocessing.image import ImageDataGenerator
        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        use_datagen = True
    except ImportError:
        try:
            from tensorflow.keras.preprocessing.image import ImageDataGenerator
            datagen = ImageDataGenerator(
                rotation_range=10,
                width_shift_range=0.1,
                height_shift_range=0.1,
                zoom_range=0.1,
                horizontal_flip=True,
                fill_mode='nearest'
            )
            use_datagen = True
        except ImportError:
            # Keras 3.x não tem preprocessing, usa treinamento sem augmentation
            print("AVISO: ImageDataGenerator nao disponivel. Treinando sem data augmentation.")
            use_datagen = False
    
    # Treinamento
    print("\n[4/5] Treinando modelo...")
    if use_datagen:
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            steps_per_epoch=len(X_train) // batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
    else:
        history = model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
    
    # Avaliação
    print("\n[5/5] Avaliando modelo...")
    test_loss, test_accuracy, test_top_k = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nResultados no conjunto de teste:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_accuracy:.4f}")
    print(f"  Top-K Accuracy: {test_top_k:.4f}")
    
    # Salva modelo final
    final_model_path = os.path.join(save_dir, 'final_model.h5')
    model.save(final_model_path)
    print(f"\nModelo final salvo em: {final_model_path}")
    
    # Salva label encoder
    import pickle
    with open(os.path.join(save_dir, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"Label encoder salvo em: {os.path.join(save_dir, 'label_encoder.pkl')}")
    
    # Plota histórico de treinamento
    plot_training_history(history, save_dir)
    
    return model, history, label_encoder


def plot_training_history(history, save_dir):
    """
    Plota gráficos do histórico de treinamento
    
    Args:
        history: Histórico retornado por model.fit()
        save_dir: Diretório para salvar gráficos
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Treino', marker='o')
    axes[0].plot(history.history['val_accuracy'], label='Validação', marker='s')
    axes[0].set_title('Accuracy do Modelo')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Treino', marker='o')
    axes[1].plot(history.history['val_loss'], label='Validação', marker='s')
    axes[1].set_title('Loss do Modelo')
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=150, bbox_inches='tight')
    print(f"Gráficos de treinamento salvos em: {os.path.join(save_dir, 'training_history.png')}")
    plt.show()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Treina modelo de reconhecimento facial')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Diretório do dataset')
    parser.add_argument('--dataset_type', type=str, default='orl',
                       choices=['orl', 'lfw'],
                       help='Tipo de dataset (orl ou lfw)')
    parser.add_argument('--model_type', type=str, default='standard',
                       choices=['standard', 'lightweight'],
                       help='Tipo de modelo (standard ou lightweight)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Número de épocas')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Tamanho do batch')
    parser.add_argument('--target_size', type=int, nargs=2, default=[128, 128],
                       help='Tamanho das imagens (altura largura)')
    
    args = parser.parse_args()
    
    train_model(
        data_dir=args.data_dir,
        dataset_type=args.dataset_type,
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        target_size=tuple(args.target_size)
    )


