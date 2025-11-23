# 🔧 Solução para Erro de DLL do TensorFlow no Windows

## ❌ Erro Encontrado

```
ImportError: DLL load failed while importing _pywrap_tensorflow_internal: 
Falha numa rotina de inicialização de DLL
```

## 🔍 Causa do Problema

Este erro geralmente ocorre no Windows devido a:
1. **Falta do Microsoft Visual C++ Redistributable**
2. **DLLs do sistema incompatíveis ou faltando**
3. **Versão do TensorFlow incompatível com o sistema**

## ✅ Soluções

### Solução 1: Instalar Visual C++ Redistributable (Recomendado)

1. **Baixe e instale o Visual C++ Redistributable:**
   - Acesse: https://aka.ms/vs/17/release/vc_redist.x64.exe
   - Ou baixe de: https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist
   - Instale a versão **x64** (64-bit)

2. **Reinicie o computador** após a instalação

3. **Teste novamente:**
   ```bash
   python check_installation.py
   ```

### Solução 2: Usar TensorFlow CPU (Mais Compatível)

Se a Solução 1 não funcionar, tente instalar uma versão específica do TensorFlow:

```bash
pip uninstall tensorflow -y
pip install tensorflow-cpu==2.15.0
```

### Solução 3: Verificar Requisitos do Sistema

Certifique-se de que:
- ✅ Windows 10/11 (64-bit)
- ✅ Python 3.8-3.11 (64-bit)
- ✅ Visual C++ Redistributable instalado

### Solução 4: Usar Ambiente Virtual Limpo

Crie um ambiente virtual novo:

```bash
python -m venv venv_tf
venv_tf\Scripts\activate
pip install tensorflow-cpu==2.15.0 opencv-python matplotlib "numpy<2.0" scikit-learn Pillow
```

## 🚀 Solução Rápida (Script Automático)

Execute o script de correção:

```bash
python fix_tensorflow_dll.py
```

Este script vai:
1. Verificar se o Visual C++ está instalado
2. Tentar instalar TensorFlow CPU se necessário
3. Verificar se tudo está funcionando

## 📝 Verificação Manual

Após aplicar as soluções, teste:

```python
python -c "import tensorflow as tf; print('TensorFlow OK:', tf.__version__)"
```

## ⚠️ Notas Importantes

- **TensorFlow 2.20+** pode ter problemas de DLL no Windows
- **TensorFlow 2.15.0** é mais estável no Windows
- Use **tensorflow-cpu** se não precisar de GPU
- Sempre instale o **Visual C++ Redistributable** primeiro

## 🆘 Ainda com Problemas?

1. Verifique se está usando Python 64-bit:
   ```bash
   python -c "import platform; print(platform.architecture())"
   ```

2. Tente reinstalar TensorFlow:
   ```bash
   pip uninstall tensorflow tensorflow-cpu -y
   pip install tensorflow-cpu==2.15.0
   ```

3. Verifique logs detalhados:
   ```bash
   python -c "import tensorflow as tf" 2>&1 | more
   ```

