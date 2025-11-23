"""
Script para verificar se todas as dependências estão instaladas corretamente
"""

import sys

def check_import(module_name, import_statement=None):
    """Verifica se um módulo pode ser importado"""
    try:
        if import_statement:
            exec(import_statement)
        else:
            __import__(module_name)
        print(f"OK {module_name}")
        return True
    except ImportError as e:
        print(f"ERRO {module_name} - {str(e)}")
        return False
    except Exception as e:
        print(f"AVISO {module_name} - {str(e)}")
        return False

def main():
    print("=" * 60)
    print("VERIFICAÇÃO DE INSTALAÇÃO")
    print("=" * 60)
    
    print("\nVerificando dependências básicas...")
    basic_modules = [
        ('numpy', None),
        ('cv2', None),
        ('matplotlib', None),
        ('sklearn', None),
        ('PIL', None),
    ]
    
    basic_ok = True
    for module, import_stmt in basic_modules:
        if not check_import(module, import_stmt):
            basic_ok = False
    
    print("\nVerificando TensorFlow/Keras...")
    tf_ok = check_import('tensorflow', 'import tensorflow as tf')
    
    # Tenta diferentes formas de importar Keras
    keras_ok = False
    keras_method = None
    
    print("\nTentando importar Keras...")
    
    # Método 1: tensorflow.keras
    try:
        import tensorflow as tf
        from tensorflow import keras
        print("OK Keras importado via: from tensorflow import keras")
        keras_ok = True
        keras_method = "tensorflow.keras"
    except ImportError:
        pass
    
    # Método 2: keras standalone
    if not keras_ok:
        try:
            import keras
            print("OK Keras importado via: import keras")
            keras_ok = True
            keras_method = "keras standalone"
        except ImportError:
            print("ERRO: Nao foi possivel importar Keras")
    
    # Verifica versões
    print("\n" + "=" * 60)
    print("VERSÕES INSTALADAS")
    print("=" * 60)
    
    try:
        import numpy as np
        print(f"NumPy: {np.__version__}")
    except:
        pass
    
    try:
        import cv2
        print(f"OpenCV: {cv2.__version__}")
    except:
        pass
    
    try:
        import tensorflow as tf
        print(f"TensorFlow: {tf.__version__}")
    except:
        pass
    
    if keras_ok:
        try:
            import keras
            print(f"Keras: {keras.__version__}")
        except:
            try:
                from tensorflow import keras
                print(f"Keras (via TF): {keras.__version__}")
            except:
                pass
    
    # Resumo
    print("\n" + "=" * 60)
    print("RESUMO")
    print("=" * 60)
    
    if basic_ok and tf_ok and keras_ok:
        print("OK Todas as dependencias estao instaladas corretamente!")
        print(f"OK Metodo de importacao Keras: {keras_method}")
        return 0
    else:
        print("ERRO: Algumas dependencias estao faltando ou com problemas.")
        print("\nPara instalar/atualizar, execute:")
        print("  pip install -r requirements.txt")
        print("\nOu instale manualmente:")
        print("  pip install tensorflow opencv-python matplotlib numpy scikit-learn Pillow")
        return 1

if __name__ == "__main__":
    sys.exit(main())

