# Sistema de Reconhecimento Facial com CNN

Sistema completo de reconhecimento facial usando Convolutional Neural Networks (CNN) e Visão Computacional Avançada. Este projeto foi desenvolvido para trabalhar com datasets pequenos, utilizando técnicas de regularização e data augmentation para melhorar o desempenho.

## 🎯 Características

- **Arquitetura CNN otimizada** para datasets pequenos
- **Suporte para múltiplos datasets**: LFW e ORL Faces Dataset
- **Detecção automática de rostos** usando Haar Cascades
- **Data augmentation** para melhorar generalização
- **Regularização** (Dropout, L2, Batch Normalization)
- **Predição em tempo real** via webcam
- **Visualizações** de resultados e histórico de treinamento

## 📋 Requisitos

- Python 3.8+
- OpenCV
- TensorFlow/Keras
- NumPy
- Matplotlib
- scikit-learn

## 🚀 Instalação

1. Clone ou baixe este repositório

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

## 📁 Estrutura do Projeto

```
face/
├── data_preprocessing.py    # Pré-processamento de dados
├── model.py                 # Arquitetura CNN
├── train.py                 # Script de treinamento
├── predict.py               # Script de predição
├── requirements.txt         # Dependências
└── README.md               # Este arquivo
```

## 📊 Datasets Suportados

### ORL Faces Dataset
- **Estrutura**: `data/orl/s{person_id}_{image_id}.pgm`
- **Exemplo**: `s1_1.pgm`, `s1_2.pgm`, ..., `s40_10.pgm`
- **Download**: Disponível em vários repositórios online

### LFW (Labeled Faces in the Wild)
- **Estrutura**: `data/lfw/person_name/image.jpg`
- **Exemplo**: `data/lfw/Aaron_Eckhart/001.jpg`
- **Download**: [LFW Dataset](http://vis-www.cs.umass.edu/lfw/)

## 🎓 Como Usar

### 1. Preparar o Dataset

**⚠️ PROBLEMAS NO PASSO 2?** Consulte o **GUIA_DATASET.md** para instruções detalhadas!

**Opção rápida:**
```bash
python download_datasets.py
```

Coloque seus dados em uma das seguintes estruturas:

**Para ORL:**
```
data/
└── orl/
    ├── s1_1.pgm
    ├── s1_2.pgm
    └── ...
```

**Para LFW:**
```
data/
└── lfw/
    ├── Person1/
    │   ├── image1.jpg
    │   └── image2.jpg
    └── Person2/
        └── ...
```

### 2. Treinar o Modelo

**Treinamento básico:**
```bash
python train.py --data_dir data/orl --dataset_type orl --epochs 100
```

**Treinamento com modelo leve (para datasets muito pequenos):**
```bash
python train.py --data_dir data/orl --dataset_type orl --model_type lightweight --epochs 50
```

**Parâmetros disponíveis:**
- `--data_dir`: Diretório do dataset (obrigatório)
- `--dataset_type`: Tipo de dataset (`orl` ou `lfw`)
- `--model_type`: Tipo de modelo (`standard` ou `lightweight`)
- `--epochs`: Número de épocas (padrão: 100)
- `--batch_size`: Tamanho do batch (padrão: 32)
- `--target_size`: Tamanho das imagens (padrão: 128 128)

### 3. Fazer Predições

**Predição em uma imagem:**
```bash
python predict.py --image path/to/image.jpg --model models/best_model.h5 --label_encoder models/label_encoder.pkl
```

**Predição em lote:**
```bash
python predict.py --image_dir path/to/images/ --model models/best_model.h5 --label_encoder models/label_encoder.pkl
```

**Predição em tempo real (webcam):**
```bash
python predict.py --camera --model models/best_model.h5 --label_encoder models/label_encoder.pkl
```

## 🏗️ Arquitetura do Modelo

O modelo CNN utiliza:

- **4 camadas convolucionais** com Batch Normalization
- **Max Pooling** para redução dimensional
- **Dropout** para regularização
- **2 camadas densas** antes da saída
- **Softmax** para classificação multi-classe

### Modelo Padrão (Standard)
- Conv2D: 32 → 64 → 128 → 256 filtros
- Dense: 512 → 256 neurônios

### Modelo Leve (Lightweight)
- Conv2D: 32 → 64 → 128 filtros
- Dense: 256 → 128 neurônios

## 📈 Melhorias para Datasets Pequenos

1. **Data Augmentation**: Rotação, zoom, translação, flip horizontal
2. **Regularização L2**: Previne overfitting
3. **Batch Normalization**: Estabiliza treinamento
4. **Early Stopping**: Para quando não há melhoria
5. **Learning Rate Reduction**: Ajusta taxa de aprendizado dinamicamente

## 📝 Exemplo de Uso Programático

```python
from data_preprocessing import FaceDataPreprocessor
from model import create_face_recognition_model, compile_model
from train import train_model

# Treinar modelo
model, history, label_encoder = train_model(
    data_dir='data/orl',
    dataset_type='orl',
    epochs=50
)

# Fazer predição
from predict import FaceRecognizer

recognizer = FaceRecognizer(
    'models/best_model.h5',
    'models/label_encoder.pkl'
)

label, confidence, top_5 = recognizer.predict_image('test_image.jpg')
print(f"Pessoa identificada: {label} (confiança: {confidence:.2%})")
```

## 🔧 Troubleshooting

### Erro: "Nenhum rosto detectado"
- Verifique se a imagem contém um rosto visível
- Tente ajustar os parâmetros do detector Haar Cascade
- Certifique-se de que a iluminação é adequada

### Overfitting
- Use o modelo `lightweight` para datasets muito pequenos
- Aumente o `dropout_rate` no modelo
- Use mais data augmentation

### Baixa acurácia
- Verifique se o dataset está balanceado
- Aumente o número de épocas
- Tente diferentes tamanhos de imagem (`target_size`)

## 📚 Referências

- [ORL Face Database](https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html)
- [LFW Dataset](http://vis-www.cs.umass.edu/lfw/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [OpenCV Documentation](https://opencv.org/)

## 📄 Licença

Este projeto é fornecido como está, para fins educacionais e de pesquisa.

## 👤 Autor

Sistema de Reconhecimento Facial com CNN - Projeto de Visão Computacional Avançada

---

**Nota**: Este sistema foi otimizado para datasets pequenos. Para melhores resultados com datasets maiores, considere usar modelos pré-treinados ou transfer learning.


