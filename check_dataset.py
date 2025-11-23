"""
Script para verificar e validar a estrutura do dataset
"""

import os
import cv2
from pathlib import Path
from collections import Counter


def check_orl_dataset(data_dir):
    """Verifica estrutura do dataset ORL"""
    print("=" * 50)
    print("Verificando Dataset ORL")
    print("=" * 50)
    
    if not os.path.exists(data_dir):
        print(f"ERRO: Diretorio nao encontrado: {data_dir}")
        return False
    
    files = [f for f in os.listdir(data_dir) 
             if f.lower().endswith(('.pgm', '.jpg', '.jpeg', '.png'))]
    
    if len(files) == 0:
        print(f"ERRO: Nenhuma imagem encontrada em {data_dir}")
        return False
    
        print(f"OK Total de imagens encontradas: {len(files)}")
    
    # Analisa estrutura
    person_ids = []
    valid_images = 0
    invalid_images = 0
    
    for img_file in files:
        # Tenta extrair person_id
        parts = img_file.split('_')
        if len(parts) >= 2:
            person_id = parts[0].replace('s', '')
            person_ids.append(person_id)
        else:
            person_id = img_file.split('.')[0]
            person_ids.append(person_id)
        
        # Verifica se a imagem é válida
        img_path = os.path.join(data_dir, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            valid_images += 1
        else:
            invalid_images += 1
            print(f"AVISO:  Imagem inválida: {img_file}")
    
    person_counts = Counter(person_ids)
    
    print(f"\nOK Imagens validas: {valid_images}")
    if invalid_images > 0:
        print(f"AVISO: Imagens invalidas: {invalid_images}")
    
    print(f"\nOK Numero de pessoas unicas: {len(person_counts)}")
    print(f"OK Imagens por pessoa:")
    for person_id, count in sorted(person_counts.items()):
        print(f"   Pessoa {person_id}: {count} imagens")
    
    # Verifica balanceamento
    counts = list(person_counts.values())
    min_count = min(counts)
    max_count = max(counts)
    avg_count = sum(counts) / len(counts)
    
    print(f"\nEstatísticas:")
    print(f"   Mínimo de imagens por pessoa: {min_count}")
    print(f"   Máximo de imagens por pessoa: {max_count}")
    print(f"   Média de imagens por pessoa: {avg_count:.1f}")
    
    if max_count - min_count > 5:
        print("AVISO: Dataset desbalanceado! Considere balancear os dados.")
    
    return True


def check_lfw_dataset(data_dir):
    """Verifica estrutura do dataset LFW"""
    print("=" * 50)
    print("Verificando Dataset LFW")
    print("=" * 50)
    
    if not os.path.exists(data_dir):
        print(f"ERRO: Diretorio nao encontrado: {data_dir}")
        return False
    
    person_dirs = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d))]
    
    if len(person_dirs) == 0:
        print(f"ERRO: Nenhum diretorio de pessoa encontrado em {data_dir}")
        return False
    
    print(f"OK Total de pessoas encontradas: {len(person_dirs)}")
    
    total_images = 0
    person_image_counts = {}
    invalid_count = 0
    
    for person_dir in person_dirs:
        person_path = os.path.join(data_dir, person_dir)
        images = [f for f in os.listdir(person_path)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        valid = 0
        for img_file in images:
            img_path = os.path.join(person_path, img_file)
            img = cv2.imread(img_path)
            if img is not None:
                valid += 1
            else:
                invalid_count += 1
        
        person_image_counts[person_dir] = valid
        total_images += valid
    
    print(f"\nOK Total de imagens validas: {total_images}")
    if invalid_count > 0:
        print(f"AVISO: Imagens invalidas: {invalid_count}")
    
    print(f"\nOK Imagens por pessoa (primeiras 10):")
    for person, count in list(person_image_counts.items())[:10]:
        print(f"   {person}: {count} imagens")
    
    if len(person_image_counts) > 10:
        print(f"   ... e mais {len(person_image_counts) - 10} pessoas")
    
    # Estatísticas
    counts = list(person_image_counts.values())
    min_count = min(counts)
    max_count = max(counts)
    avg_count = sum(counts) / len(counts)
    
    print(f"\nEstatísticas:")
    print(f"   Mínimo de imagens por pessoa: {min_count}")
    print(f"   Máximo de imagens por pessoa: {max_count}")
    print(f"   Média de imagens por pessoa: {avg_count:.1f}")
    
    if max_count - min_count > 10:
        print("AVISO: Dataset desbalanceado! Considere balancear os dados.")
    
    return True


def check_dataset_structure(data_dir, dataset_type='auto'):
    """
    Verifica estrutura do dataset automaticamente
    
    Args:
        data_dir: Diretório do dataset
        dataset_type: 'orl', 'lfw', ou 'auto' para detecção automática
    """
    if dataset_type == 'auto':
        # Tenta detectar automaticamente
        if os.path.exists(data_dir):
            files = os.listdir(data_dir)
            # Se há arquivos .pgm diretamente, provavelmente é ORL
            pgm_files = [f for f in files if f.lower().endswith('.pgm')]
            # Se há diretórios, provavelmente é LFW
            dirs = [d for d in files if os.path.isdir(os.path.join(data_dir, d))]
            
            if pgm_files and len(pgm_files) > len(dirs):
                dataset_type = 'orl'
            elif dirs:
                dataset_type = 'lfw'
            else:
                print("AVISO:  Não foi possível detectar o tipo de dataset automaticamente.")
                print("Por favor, especifique --dataset_type orl ou --dataset_type lfw")
                return False
    
    if dataset_type == 'orl':
        return check_orl_dataset(data_dir)
    elif dataset_type == 'lfw':
        return check_lfw_dataset(data_dir)
    else:
        print(f"ERRO: Tipo de dataset invalido: {dataset_type}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Verifica estrutura do dataset')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Diretório do dataset')
    parser.add_argument('--dataset_type', type=str, default='auto',
                       choices=['auto', 'orl', 'lfw'],
                       help='Tipo de dataset (auto, orl ou lfw)')
    
    args = parser.parse_args()
    
    success = check_dataset_structure(args.data_dir, args.dataset_type)
    
    if success:
        print("\n" + "=" * 50)
        print("OK Verificacao concluida com sucesso!")
        print("=" * 50)
    else:
        print("\n" + "=" * 50)
        print("ERRO: Verificacao falhou. Verifique os erros acima.")
        print("=" * 50)


