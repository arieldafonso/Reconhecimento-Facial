# 🔧 Solução para Erro de Importação do Keras

## ❌ Erro Encontrado

```
ImportError: cannot import name 'keras' from 'tensorflow'
ModuleNotFoundError: No module named 'tensorflow.compat'
```

## 🔍 Causa do Problema

Você tem uma versão **incompatível do Keras standalone** instalada. O Keras standalone está tentando usar `tensorflow.compat.v2`, mas sua versão do TensorFlow não tem esse módulo.

**Solução:** O TensorFlow já inclui o Keras, então o Keras standalone não é necessário e pode causar conflitos.

## ✅ Solução Rápida (Recomendada)

Execute o script de correção automática:

```bash
python fix_keras_installation.py
```

Este script vai:
1. ✅ Detectar o problema
2. ✅ Desinstalar o Keras standalone
3. ✅ Verificar se tudo está funcionando

## 🔧 Solução Manual

Se preferir fazer manualmente:

### Passo 1: Desinstalar Keras Standalone
```bash
pip uninstall keras -y
```

### Passo 2: Verificar/Atualizar TensorFlow
```bash
pip install --upgrade tensorflow
```

### Passo 3: Verificar Instalação
```bash
python check_installation.py
```

## ✅ Verificação

Teste se está funcionando:

```python
python -c "import tensorflow as tf; from tensorflow import keras; print('OK!')"
```

Se aparecer "OK!", está tudo certo!

## 📝 Notas Importantes

- **NÃO instale `keras` separadamente** se você já tem `tensorflow` instalado
- O TensorFlow 2.x já inclui o Keras
- Se precisar de uma versão específica, use: `pip install tensorflow==2.15.0`

## 🆘 Ainda com Problemas?

1. Verifique a versão do Python (recomendado: 3.8-3.11)
2. Tente criar um ambiente virtual limpo:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   pip install tensorflow opencv-python matplotlib numpy scikit-learn Pillow
   ```

3. Execute `python check_installation.py` para diagnóstico completo

