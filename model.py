"""
Modelo CNN para reconhecimento facial
Arquitetura otimizada para datasets pequenos
"""

import tensorflow as tf
import sys

# Compatibilidade com diferentes versões do TensorFlow/Keras
try:
    # Tenta primeiro Keras 3.x (standalone)
    import keras
    from keras import layers, models
    from keras.regularizers import l2
except ImportError:
    try:
        # Tenta Keras 2.x via TensorFlow
        from tensorflow import keras
        from tensorflow.keras import layers, models
        from tensorflow.keras.regularizers import l2
    except ImportError as e:
        print("❌ ERRO: Não foi possível importar Keras!")
        print(f"   Erro: {e}")
        print("\n💡 SOLUÇÃO:")
        print("   Execute: pip install tensorflow")
        sys.exit(1)


def create_face_recognition_model(input_shape, num_classes, dropout_rate=0.5):
    """
    Cria modelo CNN para reconhecimento facial
    
    Args:
        input_shape: Formato da entrada (altura, largura, canais)
        num_classes: Número de classes (pessoas) para classificar
        dropout_rate: Taxa de dropout para regularização
        
    Returns:
        Modelo Keras compilado
    """
    model = models.Sequential([
        # Primeira camada convolucional
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape,
                     kernel_regularizer=l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Segunda camada convolucional
        layers.Conv2D(64, (3, 3), activation='relu',
                     kernel_regularizer=l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Terceira camada convolucional
        layers.Conv2D(128, (3, 3), activation='relu',
                     kernel_regularizer=l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Quarta camada convolucional (para datasets maiores)
        layers.Conv2D(256, (3, 3), activation='relu',
                     kernel_regularizer=l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Flatten
        layers.Flatten(),
        
        # Camadas densas
        layers.Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        
        layers.Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        
        # Camada de saída
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def create_lightweight_model(input_shape, num_classes, dropout_rate=0.5):
    """
    Cria modelo CNN mais leve para datasets muito pequenos
    
    Args:
        input_shape: Formato da entrada (altura, largura, canais)
        num_classes: Número de classes (pessoas) para classificar
        dropout_rate: Taxa de dropout para regularização
        
    Returns:
        Modelo Keras compilado
    """
    model = models.Sequential([
        # Primeira camada convolucional
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape,
                     padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Segunda camada convolucional
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Terceira camada convolucional
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Flatten
        layers.Flatten(),
        
        # Camadas densas
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        
        # Camada de saída
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def compile_model(model, learning_rate=0.001):
    """
    Compila o modelo com otimizador e métricas
    
    Args:
        model: Modelo Keras
        learning_rate: Taxa de aprendizado
        
    Returns:
        Modelo compilado
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy']
    )
    
    return model


def create_model_with_augmentation(input_shape, num_classes, model_type='standard'):
    """
    Cria modelo com data augmentation integrado
    
    Args:
        input_shape: Formato da entrada
        num_classes: Número de classes
        model_type: 'standard' ou 'lightweight'
        
    Returns:
        Modelo Keras com data augmentation
    """
    # Data augmentation
    data_augmentation = keras.Sequential([
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomFlip("horizontal"),
    ])
    
    # Cria modelo base
    if model_type == 'lightweight':
        base_model = create_lightweight_model(input_shape, num_classes)
    else:
        base_model = create_face_recognition_model(input_shape, num_classes)
    
    # Cria modelo completo com augmentation
    inputs = keras.Input(shape=input_shape)
    x = data_augmentation(inputs)
    outputs = base_model(x)
    model = keras.Model(inputs, outputs)
    
    return model


if __name__ == "__main__":
    # Exemplo de uso
    input_shape = (128, 128, 1)
    num_classes = 10
    
    model = create_face_recognition_model(input_shape, num_classes)
    model = compile_model(model)
    
    print("Modelo criado com sucesso!")
    model.summary()


