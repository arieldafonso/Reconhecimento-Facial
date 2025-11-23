"""
Script para capturar rostos usando webcam e criar dataset customizado
"""

import cv2
import os
import numpy as np
from pathlib import Path


class FaceCapture:
    """Classe para capturar rostos da webcam"""
    
    def __init__(self, output_dir='data/custom'):
        """
        Inicializa o capturador de rostos
        
        Args:
            output_dir: Diretório para salvar as imagens
        """
        self.output_dir = output_dir
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.cap = None
        
    def detect_face(self, frame):
        """
        Detecta rosto no frame
        
        Returns:
            (x, y, w, h) se detectado, None caso contrário
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100)
        )
        
        if len(faces) > 0:
            # Retorna o maior rosto
            return max(faces, key=lambda f: f[2] * f[3])
        return None
    
    def capture_person(self, person_name, num_images=10):
        """
        Captura múltiplas imagens de uma pessoa
        
        Args:
            person_name: Nome da pessoa
            num_images: Número de imagens a capturar
        """
        # Cria diretório para a pessoa
        person_dir = os.path.join(self.output_dir, person_name)
        os.makedirs(person_dir, exist_ok=True)
        
        # Abre a câmera
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            print("❌ Erro: Não foi possível abrir a câmera!")
            return False
        
        print("\n" + "=" * 60)
        print(f"Capturando imagens para: {person_name}")
        print("=" * 60)
        print(f"\n📸 Instruções:")
        print(f"  - Pressione ESPAÇO para capturar uma imagem")
        print(f"  - Pressione 'q' para finalizar")
        print(f"  - Posicione-se bem na frente da câmera")
        print(f"  - Tente diferentes ângulos e expressões")
        print(f"\nCapturas restantes: {num_images}")
        
        captured = 0
        frame_count = 0
        
        while captured < num_images:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Detecta rosto
            face_rect = self.detect_face(frame)
            
            # Desenha retângulo se rosto detectado
            if face_rect is not None:
                x, y, w, h = face_rect
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, "Rosto Detectado!", (x, y-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Nenhum rosto detectado", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Informações na tela
            cv2.putText(frame, f"Capturadas: {captured}/{num_images}", (10, 60),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "ESPACO: Capturar | Q: Sair", (10, frame.shape[0] - 20),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Captura de Rostos - Pressione ESPACO para capturar', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n⚠️  Captura cancelada pelo usuário")
                break
            elif key == ord(' '):  # Espaço
                if face_rect is not None:
                    # Salva a imagem do rosto
                    x, y, w, h = face_rect
                    face_roi = frame[y:y+h, x:x+w]
                    
                    # Redimensiona para tamanho padrão
                    face_resized = cv2.resize(face_roi, (128, 128))
                    
                    # Salva
                    img_filename = f"img_{captured+1:03d}.jpg"
                    img_path = os.path.join(person_dir, img_filename)
                    cv2.imwrite(img_path, face_resized)
                    
                    captured += 1
                    print(f"✓ Imagem {captured}/{num_images} salva: {img_path}")
                    
                    # Feedback visual
                    frame_copy = frame.copy()
                    cv2.putText(frame_copy, "CAPTURADO!", (x, y-30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    cv2.imshow('Captura de Rostos - Pressione ESPACO para capturar', frame_copy)
                    cv2.waitKey(500)  # Mostra feedback por 0.5 segundos
                else:
                    print("⚠️  Nenhum rosto detectado! Posicione-se melhor.")
        
        self.cap.release()
        cv2.destroyAllWindows()
        
        if captured > 0:
            print(f"\n✓ Captura concluída!")
            print(f"  Total de imagens: {captured}")
            print(f"  Salvas em: {person_dir}")
            return True
        else:
            print("\n❌ Nenhuma imagem foi capturada")
            return False
    
    def capture_multiple_people(self):
        """Captura imagens de múltiplas pessoas"""
        print("=" * 60)
        print("CAPTURA DE DATASET CUSTOMIZADO")
        print("=" * 60)
        
        people = []
        
        while True:
            print("\n" + "-" * 60)
            person_name = input("Digite o nome da pessoa (ou 'fim' para terminar): ").strip()
            
            if person_name.lower() in ['fim', 'f', 'quit', 'q', '']:
                break
            
            if person_name in people:
                print(f"⚠️  '{person_name}' já foi capturado. Use outro nome.")
                continue
            
            try:
                num_images = int(input(f"Quantas imagens para {person_name}? (padrão: 10): ") or "10")
            except ValueError:
                num_images = 10
            
            if self.capture_person(person_name, num_images):
                people.append(person_name)
            else:
                print(f"⚠️  Falha ao capturar imagens para {person_name}")
            
            continue_capture = input("\nCapturar outra pessoa? (s/n): ").strip().lower()
            if continue_capture not in ['s', 'sim', 'y', 'yes']:
                break
        
        if people:
            print("\n" + "=" * 60)
            print("✓ CAPTURA CONCLUÍDA!")
            print("=" * 60)
            print(f"\nPessoas capturadas: {len(people)}")
            for person in people:
                person_dir = os.path.join(self.output_dir, person)
                num_files = len([f for f in os.listdir(person_dir) 
                                if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                print(f"  - {person}: {num_files} imagens")
            
            print(f"\n📁 Dataset salvo em: {self.output_dir}")
            print("\n✅ Próximos passos:")
            print(f"   1. Verifique o dataset: python check_dataset.py --data_dir {self.output_dir} --dataset_type lfw")
            print(f"   2. Treine o modelo: python train.py --data_dir {self.output_dir} --dataset_type lfw")
        else:
            print("\n❌ Nenhuma pessoa foi capturada")


def main():
    """Função principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Captura rostos da webcam para criar dataset')
    parser.add_argument('--output_dir', type=str, default='data/custom',
                       help='Diretório para salvar as imagens')
    parser.add_argument('--person', type=str, default=None,
                       help='Nome da pessoa (se não fornecido, será interativo)')
    parser.add_argument('--num_images', type=int, default=10,
                       help='Número de imagens a capturar')
    
    args = parser.parse_args()
    
    # Cria diretório de saída
    os.makedirs(args.output_dir, exist_ok=True)
    
    capturer = FaceCapture(output_dir=args.output_dir)
    
    if args.person:
        # Modo não-interativo: captura uma pessoa
        capturer.capture_person(args.person, args.num_images)
    else:
        # Modo interativo: captura múltiplas pessoas
        capturer.capture_multiple_people()


if __name__ == "__main__":
    main()

