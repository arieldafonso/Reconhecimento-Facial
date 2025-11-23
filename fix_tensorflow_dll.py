"""
Script para corrigir problemas de DLL do TensorFlow no Windows
"""

import subprocess
import sys
import os
import platform

def run_command(command, check=True):
    """Executa um comando e retorna o resultado"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if check and result.returncode != 0:
            print(f"⚠️  Comando falhou: {command}")
            print(f"   Erro: {result.stderr}")
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def check_python_architecture():
    """Verifica se Python é 64-bit"""
    arch = platform.architecture()[0]
    if arch == '64bit':
        print(f"✓ Python 64-bit detectado")
        return True
    else:
        print(f"✗ Python 32-bit detectado - TensorFlow requer 64-bit!")
        return False

def check_vc_redistributable():
    """Verifica se Visual C++ Redistributable está instalado"""
    print("\nVerificando Visual C++ Redistributable...")
    
    # Verifica no registro do Windows
    import winreg
    try:
        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64"
        )
        winreg.CloseKey(key)
        print("✓ Visual C++ Redistributable encontrado no registro")
        return True
    except:
        pass
    
    # Tenta verificar via arquivos
    vc_redist_paths = [
        r"C:\Windows\System32\vcruntime140.dll",
        r"C:\Windows\System32\msvcp140.dll",
    ]
    
    found = False
    for path in vc_redist_paths:
        if os.path.exists(path):
            found = True
            break
    
    if found:
        print("✓ DLLs do Visual C++ encontradas")
        return True
    else:
        print("⚠️  Visual C++ Redistributable não encontrado")
        print("\n📥 Baixe e instale:")
        print("   https://aka.ms/vs/17/release/vc_redist.x64.exe")
        return False

def fix_tensorflow():
    """Tenta corrigir problemas do TensorFlow"""
    print("\n" + "=" * 60)
    print("CORREÇÃO DO TENSORFLOW")
    print("=" * 60)
    
    # Verifica arquitetura do Python
    if not check_python_architecture():
        print("\n❌ Você precisa usar Python 64-bit!")
        return False
    
    # Verifica Visual C++
    vc_ok = check_vc_redistributable()
    if not vc_ok:
        print("\n⚠️  IMPORTANTE: Instale o Visual C++ Redistributable primeiro!")
        print("   URL: https://aka.ms/vs/17/release/vc_redist.x64.exe")
        resposta = input("\nJá instalou o Visual C++? (s/n): ").strip().lower()
        if resposta not in ['s', 'sim', 'y', 'yes']:
            print("\nPor favor, instale o Visual C++ Redistributable e execute este script novamente.")
            return False
    
    # Tenta importar TensorFlow
    print("\nTestando importacao do TensorFlow...")
    success, _, _ = run_command('python -c "import tensorflow as tf; print(tf.__version__)"', check=False)
    
    if success:
        print("✓ TensorFlow está funcionando!")
        return True
    
    print("\n⚠️  TensorFlow não está funcionando. Tentando corrigir...")
    
    # Opção 1: Tentar tensorflow-cpu
    print("\nTentando instalar tensorflow-cpu (versao mais compativel)...")
    
    resposta = input("Deseja instalar tensorflow-cpu==2.15.0? (s/n): ").strip().lower()
    if resposta in ['s', 'sim', 'y', 'yes']:
        print("\nDesinstalando TensorFlow atual...")
        run_command("pip uninstall tensorflow tensorflow-cpu -y", check=False)
        
        print("\nInstalando tensorflow-cpu==2.15.0...")
        success, stdout, stderr = run_command("pip install tensorflow-cpu==2.15.0", check=False)
        
        if success:
            print("✓ TensorFlow CPU instalado!")
            
            # Testa novamente
            print("\nTestando importacao...")
            success, stdout, stderr = run_command('python -c "import tensorflow as tf; print(\'OK:\', tf.__version__)"', check=False)
            
            if success:
                print("✓ TensorFlow está funcionando agora!")
                print(f"  Versão: {stdout.strip()}")
                return True
            else:
                print("✗ Ainda há problemas:")
                print(stderr)
        else:
            print("✗ Erro ao instalar TensorFlow CPU")
            print(stderr)
    
    return False

def main():
    print("=" * 60)
    print("CORREÇÃO DE DLL DO TENSORFLOW (WINDOWS)")
    print("=" * 60)
    
    if platform.system() != 'Windows':
        print("⚠️  Este script é específico para Windows")
        print(f"   Sistema detectado: {platform.system()}")
        return 1
    
    # Verifica Python
    print(f"\nPython: {sys.version}")
    
    # Tenta corrigir
    if fix_tensorflow():
        print("\n" + "=" * 60)
        print("✅ CORREÇÃO CONCLUÍDA COM SUCESSO!")
        print("=" * 60)
        print("\nTeste novamente:")
        print("  python check_installation.py")
        return 0
    else:
        print("\n" + "=" * 60)
        print("❌ CORREÇÃO FALHOU")
        print("=" * 60)
        print("\n💡 SOLUÇÕES MANUAIS:")
        print("1. Instale Visual C++ Redistributable:")
        print("   https://aka.ms/vs/17/release/vc_redist.x64.exe")
        print("\n2. Reinstale TensorFlow:")
        print("   pip uninstall tensorflow -y")
        print("   pip install tensorflow-cpu==2.15.0")
        print("\n3. Consulte: SOLUCAO_ERRO_DLL_WINDOWS.md")
        return 1

if __name__ == "__main__":
    try:
        import winreg
    except ImportError:
        print("⚠️  Módulo 'winreg' não disponível (normal em algumas versões do Python)")
        print("   Continuando sem verificação de registro...")
    
    sys.exit(main())

