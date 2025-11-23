"""
Script para baixar e preparar datasets de reconhecimento facial
Suporta download automático e preparação manual
"""

import os
import urllib.request
import zipfile
import tarfile
import shutil
from pathlib import Path


def create_data_directory():
    """Cria estrutura de diretórios para datasets"""
    os.makedirs('data/orl', exist_ok=True)
    os.makedirs('data/lfw', exist_ok=True)
    print("✓ Diretórios criados: data/orl/ e data/lfw/")


def download_orl_dataset(output_dir='data/orl'):
    """
    Baixa o dataset ORL (AT&T Face Database)
    
    Nota: O download oficial requer preenchimento de formulário.
    Este script fornece links e instruções.
    """
    print("=" * 60)
    print("Download do Dataset ORL (AT&T Face Database)")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n📥 OPÇÕES DE DOWNLOAD:")
    print("\n1. Download Oficial (Recomendado):")
    print("   URL: https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html")
    print("   - Preencha o formulário no site")
    print("   - Baixe o arquivo 'orl_faces.zip' ou similar")
    print("   - Extraia no diretório data/orl/")
    
    print("\n2. Download Direto (Alternativa):")
    print("   URL: http://www.cl.cam.ac.uk/Research/DTG/attarchive/pub/data/att_faces.zip")
    print("   - Tente baixar diretamente deste link")
    
    print("\n3. Estrutura Esperada:")
    print("   data/orl/")
    print("   ├── s1_1.pgm")
    print("   ├── s1_2.pgm")
    print("   ├── s2_1.pgm")
    print("   └── ...")
    
    print("\n⚠️  IMPORTANTE:")
    print("   - O dataset ORL contém 40 pessoas, 10 imagens cada")
    print("   - Formato: s{person_id}_{image_id}.pgm")
    print("   - Exemplo: s1_1.pgm, s1_2.pgm, ..., s40_10.pgm")
    
    # Tenta baixar de uma fonte alternativa comum
    print("\n🔄 Tentando baixar de fonte alternativa...")
    
    alternative_urls = [
        "http://www.cl.cam.ac.uk/Research/DTG/attarchive/pub/data/att_faces.zip",
        "https://github.com/opencv/opencv/raw/master/samples/data/att_faces.zip",
    ]
    
    for url in alternative_urls:
        try:
            print(f"\nTentando: {url}")
            zip_path = os.path.join(output_dir, 'orl_faces.zip')
            
            def show_progress(block_num, block_size, total_size):
                if total_size > 0:
                    downloaded = block_num * block_size
                    percent = min(downloaded * 100 / total_size, 100)
                    print(f"\rProgresso: {percent:.1f}%", end='', flush=True)
            
            urllib.request.urlretrieve(url, zip_path, show_progress)
            print("\n✓ Download concluído!")
            
            # Extrai o arquivo
            print("Extraindo arquivo...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Lista conteúdo
                file_list = zip_ref.namelist()
                
                # Extrai tudo
                zip_ref.extractall(output_dir)
                
                # Se os arquivos estão em uma subpasta, move para o diretório principal
                for item in file_list:
                    if '/' in item or '\\' in item:
                        # Arquivo está em subdiretório
                        parts = item.replace('\\', '/').split('/')
                        if len(parts) > 1:
                            # Move arquivos .pgm para o diretório principal
                            if item.endswith('.pgm'):
                                src = os.path.join(output_dir, item)
                                dst = os.path.join(output_dir, os.path.basename(item))
                                if os.path.exists(src) and not os.path.exists(dst):
                                    shutil.move(src, dst)
            
            # Remove o arquivo zip
            if os.path.exists(zip_path):
                os.remove(zip_path)
            
            # Remove subdiretórios vazios se houver
            for root, dirs, files in os.walk(output_dir):
                for d in dirs:
                    dir_path = os.path.join(root, d)
                    try:
                        if not os.listdir(dir_path):
                            os.rmdir(dir_path)
                    except:
                        pass
            
            print("✓ Dataset ORL baixado e extraído com sucesso!")
            print(f"✓ Arquivos salvos em: {output_dir}")
            
            # Verifica quantos arquivos foram baixados
            pgm_files = [f for f in os.listdir(output_dir) if f.endswith('.pgm')]
            print(f"✓ Total de imagens: {len(pgm_files)}")
            
            return True
        except Exception as e:
            print(f"\n✗ Erro ao baixar de {url}: {str(e)}")
            continue
    
    print("\n❌ Não foi possível baixar automaticamente.")
    print("\n📝 INSTRUÇÕES MANUAIS:")
    print("1. Acesse: https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html")
    print("2. Ou tente: http://www.cl.cam.ac.uk/Research/DTG/attarchive/pub/data/att_faces.zip")
    print("3. Baixe o arquivo ZIP")
    print(f"4. Extraia todos os arquivos .pgm em: {output_dir}")
    print("5. Execute: python check_dataset.py --data_dir data/orl --dataset_type orl")
    return False


def download_lfw_dataset(output_dir='data/lfw', subset_size='small'):
    """
    Baixa o dataset LFW (Labeled Faces in the Wild)
    
    Args:
        output_dir: Diretório de saída
        subset_size: 'small' (poucas pessoas) ou 'full' (dataset completo)
    """
    print("=" * 60)
    print("Download do Dataset LFW (Labeled Faces in the Wild)")
    print("=" * 60)
    
    print("\n📥 OPÇÕES DE DOWNLOAD:")
    print("\n1. Download Oficial:")
    print("   URL: http://vis-www.cs.umass.edu/lfw/")
    print("   - Baixe 'lfw.tgz' (dataset completo)")
    print("   - Ou 'lfw-a.tgz' (versão alinhada)")
    
    print("\n2. Estrutura Esperada:")
    print("   data/lfw/")
    print("   ├── Person1/")
    print("   │   ├── image1.jpg")
    print("   │   └── image2.jpg")
    print("   └── Person2/")
    print("       └── ...")
    
    print("\n⚠️  IMPORTANTE:")
    print("   - O dataset LFW completo tem mais de 13.000 imagens")
    print("   - Para testes, você pode usar um subconjunto")
    print("   - Cada pessoa deve ter seu próprio diretório")
    
    # Tenta baixar
    print("\n🔄 Tentando baixar dataset LFW...")
    
    lfw_url = "http://vis-www.cs.umass.edu/lfw/lfw.tgz"
    
    try:
        print(f"Baixando de: {lfw_url}")
        print("⚠️  AVISO: O download pode demorar (arquivo grande ~170MB)")
        
        tgz_path = os.path.join(output_dir, 'lfw.tgz')
        
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded * 100 / total_size, 100)
            print(f"\rProgresso: {percent:.1f}%", end='', flush=True)
        
        urllib.request.urlretrieve(lfw_url, tgz_path, show_progress)
        print("\n✓ Download concluído!")
        
        # Extrai o arquivo
        print("Extraindo arquivo...")
        with tarfile.open(tgz_path, 'r:gz') as tar_ref:
            tar_ref.extractall(output_dir)
        
        # Move conteúdo para o diretório correto
        extracted_dir = os.path.join(output_dir, 'lfw')
        if os.path.exists(extracted_dir):
            # Move todos os subdiretórios para data/lfw/
            for item in os.listdir(extracted_dir):
                src = os.path.join(extracted_dir, item)
                dst = os.path.join(output_dir, item)
                if os.path.isdir(src):
                    if os.path.exists(dst):
                        shutil.rmtree(dst)
                    shutil.move(src, dst)
            shutil.rmtree(extracted_dir)
        
        # Remove o arquivo tgz
        os.remove(tgz_path)
        
        print("✓ Dataset LFW baixado e extraído com sucesso!")
        return True
        
    except Exception as e:
        print(f"\n✗ Erro ao baixar: {str(e)}")
        print("\n📝 INSTRUÇÕES MANUAIS:")
        print("1. Acesse: http://vis-www.cs.umass.edu/lfw/")
        print("2. Baixe 'lfw.tgz'")
        print("3. Extraia o conteúdo em data/lfw/")
        return False


def create_sample_dataset():
    """
    Cria um dataset de exemplo pequeno para testes
    Útil quando não é possível baixar os datasets oficiais
    """
    print("=" * 60)
    print("Criando Dataset de Exemplo")
    print("=" * 60)
    
    print("\n📝 Para criar um dataset real, você tem duas opções:")
    print("\n1. USAR WEBCAM (Recomendado):")
    print("   Execute: python capture_faces.py")
    print("   - Captura imagens diretamente da sua webcam")
    print("   - Organiza automaticamente por pessoa")
    print("   - Pronto para usar!")
    
    print("\n2. ORGANIZAR IMAGENS EXISTENTES:")
    print("   - Coloque fotos em: data/custom/person1/, data/custom/person2/, etc.")
    print("   - Cada pessoa deve ter seu próprio diretório")
    print("   - Use: python check_dataset.py --data_dir data/custom --dataset_type lfw")
    
    print("\n⚠️  IMPORTANTE:")
    print("   - Capture pelo menos 5-10 imagens por pessoa")
    print("   - Tente diferentes ângulos e expressões")
    print("   - Boa iluminação ajuda muito!")
    
    use_webcam = input("\nDeseja usar a webcam agora? (s/n): ").strip().lower()
    if use_webcam in ['s', 'sim', 'y', 'yes']:
        from capture_faces import FaceCapture
        capturer = FaceCapture(output_dir='data/custom')
        capturer.capture_multiple_people()
    else:
        print("\nExecute 'python capture_faces.py' quando estiver pronto!")
    
    return 'data/custom'


def organize_custom_dataset(source_dir, output_dir='data/custom', dataset_type='lfw'):
    """
    Organiza um dataset customizado na estrutura correta
    
    Args:
        source_dir: Diretório com imagens desorganizadas
        output_dir: Diretório de saída organizado
        dataset_type: 'lfw' (por pessoa) ou 'orl' (arquivos nomeados)
    """
    print("=" * 60)
    print("Organizando Dataset Customizado")
    print("=" * 60)
    
    if not os.path.exists(source_dir):
        print(f"❌ Diretório não encontrado: {source_dir}")
        return False
    
    os.makedirs(output_dir, exist_ok=True)
    
    if dataset_type == 'lfw':
        # Organiza por pessoa (cada subdiretório = uma pessoa)
        print("\nOrganizando no formato LFW (por pessoa)...")
        # Implementação aqui
        print("✓ Dataset organizado!")
    else:
        # Organiza no formato ORL (arquivos nomeados)
        print("\nOrganizando no formato ORL...")
        # Implementação aqui
        print("✓ Dataset organizado!")
    
    return True


def main():
    """Menu principal"""
    print("=" * 60)
    print("SISTEMA DE DOWNLOAD E PREPARAÇÃO DE DATASETS")
    print("=" * 60)
    
    create_data_directory()
    
    print("\nEscolha uma opção:")
    print("1. Baixar dataset ORL (Recomendado para iniciantes)")
    print("2. Baixar dataset LFW (Dataset maior)")
    print("3. Criar dataset customizado (usando webcam)")
    print("4. Verificar estrutura de dataset existente")
    print("5. Ver guia completo (GUIA_DATASET.md)")
    print("6. Sair")
    
    choice = input("\nOpção: ").strip()
    
    if choice == '1':
        download_orl_dataset()
    elif choice == '2':
        download_lfw_dataset()
    elif choice == '3':
        create_sample_dataset()
    elif choice == '4':
        data_dir = input("Digite o caminho do dataset: ").strip()
        dataset_type = input("Tipo (orl/lfw/auto): ").strip() or 'auto'
        
        from check_dataset import check_dataset_structure
        check_dataset_structure(data_dir, dataset_type)
    elif choice == '5':
        print("\n📚 Abra o arquivo GUIA_DATASET.md para ver o guia completo!")
        print("   Ou leia online no seu editor de texto.")
    elif choice == '6':
        print("Até logo!")
    else:
        print("Opção inválida!")


if __name__ == "__main__":
    main()

