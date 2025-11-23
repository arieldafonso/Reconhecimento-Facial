"""
Script para corrigir problemas de instalação do Keras/TensorFlow
"""

import subprocess
import sys

def run_command(command):
    """Executa um comando e retorna o resultado"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def main():
    print("=" * 60)
    print("CORREÇÃO DE INSTALAÇÃO KERAS/TENSORFLOW")
    print("=" * 60)
    
    print("\n🔍 Verificando instalação atual...")
    
    # Verifica TensorFlow
    success, stdout, stderr = run_command("python -c \"import tensorflow as tf; print(tf.__version__)\"")
    if success:
        tf_version = stdout.strip()
        print(f"✓ TensorFlow encontrado: {tf_version}")
    else:
        print("✗ TensorFlow não encontrado")
        print("\n📦 Instalando TensorFlow...")
        success, _, _ = run_command("pip install tensorflow")
        if success:
            print("✓ TensorFlow instalado com sucesso!")
        else:
            print("✗ Erro ao instalar TensorFlow")
            return 1
    
    # Verifica Keras standalone
    success, stdout, stderr = run_command("python -c \"import keras; print(keras.__version__)\"")
    if success:
        keras_version = stdout.strip()
        print(f"⚠️  Keras standalone encontrado: {keras_version}")
        print("\n⚠️  PROBLEMA DETECTADO:")
        print("   Você tem Keras standalone instalado, que pode causar conflitos.")
        print("   O TensorFlow já inclui Keras, então o standalone não é necessário.")
        
        resposta = input("\nDeseja desinstalar o Keras standalone? (s/n): ").strip().lower()
        if resposta in ['s', 'sim', 'y', 'yes']:
            print("\n🗑️  Desinstalando Keras standalone...")
            success, _, _ = run_command("pip uninstall keras -y")
            if success:
                print("✓ Keras standalone desinstalado!")
            else:
                print("✗ Erro ao desinstalar Keras")
    else:
        print("✓ Keras standalone não encontrado (isso é bom!)")
    
    # Verifica se tensorflow.keras funciona
    print("\n🔍 Testando importação do Keras via TensorFlow...")
    success, stdout, stderr = run_command("python -c \"import tensorflow as tf; from tensorflow import keras; print('OK')\"")
    
    if success:
        print("✓ Keras via TensorFlow funciona corretamente!")
    else:
        print("✗ Problema ao importar Keras via TensorFlow")
        print(f"   Erro: {stderr}")
        
        print("\n📦 Tentando reinstalar TensorFlow...")
        success, _, _ = run_command("pip install --upgrade --force-reinstall tensorflow")
        if success:
            print("✓ TensorFlow reinstalado!")
        else:
            print("✗ Erro ao reinstalar TensorFlow")
            return 1
    
    print("\n" + "=" * 60)
    print("VERIFICAÇÃO FINAL")
    print("=" * 60)
    
    # Teste completo
    test_code = """
import tensorflow as tf
try:
    from tensorflow import keras
    from tensorflow.keras import layers, models
    print("✓ Todas as importações funcionaram!")
    print(f"✓ TensorFlow: {tf.__version__}")
    print(f"✓ Keras: {keras.__version__}")
except Exception as e:
    print(f"✗ Erro: {e}")
    exit(1)
"""
    
    success, stdout, stderr = run_command(f"python -c \"{test_code}\"")
    if success:
        print(stdout)
        print("\n✅ INSTALAÇÃO CORRIGIDA COM SUCESSO!")
        return 0
    else:
        print("✗ Ainda há problemas:")
        print(stderr)
        print("\n💡 Tente executar manualmente:")
        print("   pip uninstall keras -y")
        print("   pip install --upgrade tensorflow")
        return 1

if __name__ == "__main__":
    sys.exit(main())

