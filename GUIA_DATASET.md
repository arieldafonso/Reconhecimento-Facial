# 📚 Guia Completo: Preparação de Datasets

Este guia vai te ajudar a resolver problemas no **Passo 2: Preparar o Dataset**.

## 🎯 Opções Disponíveis

Você tem 3 opções principais para obter um dataset:

### Opção 1: Baixar Dataset ORL (Recomendado para Iniciantes)
### Opção 2: Baixar Dataset LFW
### Opção 3: Criar Seu Próprio Dataset

---

## 📥 Opção 1: Dataset ORL (AT&T Face Database)

### Por que escolher ORL?
- ✅ Dataset pequeno (40 pessoas, 10 imagens cada = 400 imagens)
- ✅ Perfeito para testes e aprendizado
- ✅ Estrutura simples
- ✅ Já vem pré-processado

### Método A: Download Automático

Execute o script:
```bash
python download_datasets.py
```
Escolha a opção 1.

### Método B: Download Manual

1. **Acesse o site oficial:**
   - URL: https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html
   - Ou use link direto: http://www.cl.cam.ac.uk/Research/DTG/attarchive/pub/data/att_faces.zip

2. **Baixe o arquivo:**
   - Arquivo: `att_faces.zip` ou `orl_faces.zip`
   - Tamanho: ~1-2 MB

3. **Extraia o arquivo:**
   ```bash
   # Crie o diretório se não existir
   mkdir -p data/orl
   
   # Extraia o ZIP na pasta data/orl/
   # No Windows: clique com botão direito > Extrair Tudo
   # No Linux/Mac: unzip att_faces.zip -d data/orl/
   ```

4. **Estrutura esperada:**
   ```
   data/
   └── orl/
       ├── s1_1.pgm
       ├── s1_2.pgm
       ├── s1_3.pgm
       ...
       ├── s40_8.pgm
       ├── s40_9.pgm
       └── s40_10.pgm
   ```

5. **Verifique se está correto:**
   ```bash
   python check_dataset.py --data_dir data/orl --dataset_type orl
   ```

### Método C: Download de Repositórios Alternativos

**GitHub:**
- Procure por "ORL face dataset" ou "AT&T face database"
- Muitos repositórios têm o dataset disponível

**Kaggle:**
- Procure por "ORL Face Database"
- Alguns kernels têm o dataset disponível

---

## 📥 Opção 2: Dataset LFW (Labeled Faces in the Wild)

### Por que escolher LFW?
- ✅ Dataset maior (13.000+ imagens)
- ✅ Mais desafiador
- ✅ Mais realista (fotos do mundo real)

### Método A: Download Automático

Execute o script:
```bash
python download_datasets.py
```
Escolha a opção 2.

⚠️ **Atenção:** O download pode demorar (arquivo ~170MB)

### Método B: Download Manual

1. **Acesse o site oficial:**
   - URL: http://vis-www.cs.umass.edu/lfw/
   - Clique em "Download"

2. **Baixe o arquivo:**
   - Arquivo: `lfw.tgz` (dataset completo)
   - Ou `lfw-a.tgz` (versão alinhada - recomendado)
   - Tamanho: ~170 MB

3. **Extraia o arquivo:**
   ```bash
   # Crie o diretório
   mkdir -p data/lfw
   
   # Extraia o TGZ
   # No Windows: use 7-Zip ou WinRAR
   # No Linux/Mac: tar -xzf lfw.tgz -C data/lfw/
   ```

4. **Estrutura esperada:**
   ```
   data/
   └── lfw/
       ├── Aaron_Eckhart/
       │   ├── Aaron_Eckhart_0001.jpg
       │   └── ...
       ├── Aaron_Guiel/
       │   └── ...
       └── ...
   ```

5. **Verifique se está correto:**
   ```bash
   python check_dataset.py --data_dir data/lfw --dataset_type lfw
   ```

---

## 📸 Opção 3: Criar Seu Próprio Dataset

### Usando Webcam

1. **Execute o script de captura:**
   ```bash
   python capture_faces.py
   ```

2. **Siga as instruções:**
   - Digite o nome da pessoa
   - Pressione ESPAÇO para capturar
   - Pressione 'q' para finalizar

3. **Estrutura criada:**
   ```
   data/
   └── custom/
       ├── Person1/
       │   ├── img_001.jpg
       │   └── ...
       └── Person2/
           └── ...
   ```

### Organizando Imagens Existentes

Se você já tem fotos:

1. **Estrutura LFW (recomendado):**
   ```
   data/
   └── custom/
       ├── Person1/
       │   ├── foto1.jpg
       │   ├── foto2.jpg
       │   └── ...
       ├── Person2/
       │   └── ...
       └── Person3/
           └── ...
   ```

2. **Estrutura ORL:**
   ```
   data/
   └── custom/
       ├── s1_1.pgm
       ├── s1_2.pgm
       ├── s2_1.pgm
       └── ...
   ```

3. **Verifique:**
   ```bash
   python check_dataset.py --data_dir data/custom --dataset_type lfw
   ```

---

## ✅ Verificação Final

Após preparar seu dataset, sempre verifique:

```bash
python check_dataset.py --data_dir data/orl --dataset_type orl
```

Ou para LFW:
```bash
python check_dataset.py --data_dir data/lfw --dataset_type lfw
```

O script vai mostrar:
- ✓ Número de imagens
- ✓ Número de pessoas
- ✓ Imagens por pessoa
- ⚠️ Avisos sobre problemas

---

## 🔧 Problemas Comuns e Soluções

### Problema: "Diretório não encontrado"
**Solução:**
```bash
# Crie o diretório
mkdir -p data/orl  # ou data/lfw
```

### Problema: "Nenhuma imagem encontrada"
**Solução:**
- Verifique se as imagens estão no formato correto (.pgm, .jpg, .png)
- Verifique se estão no diretório correto
- Use `python check_dataset.py` para diagnosticar

### Problema: "Estrutura incorreta"
**Solução:**
- Para ORL: Todas as imagens devem estar em `data/orl/` diretamente
- Para LFW: Cada pessoa deve ter seu próprio diretório dentro de `data/lfw/`

### Problema: "Dataset desbalanceado"
**Solução:**
- Tente ter pelo menos 5-10 imagens por pessoa
- Se possível, balanceie o número de imagens por pessoa

### Problema: "Imagens inválidas"
**Solução:**
- Verifique se os arquivos não estão corrompidos
- Tente abrir as imagens em um visualizador
- Re-baixe as imagens problemáticas

---

## 📊 Qual Dataset Escolher?

| Característica | ORL | LFW | Custom |
|---------------|-----|-----|--------|
| Tamanho | Pequeno (400 img) | Grande (13k+ img) | Variável |
| Dificuldade | Fácil | Médio | Fácil |
| Tempo de download | Rápido | Lento | N/A |
| Melhor para | Aprendizado | Produção | Testes pessoais |
| Recomendado para | Iniciantes | Projetos sérios | Experimentação |

**Recomendação:** Comece com ORL para aprender, depois experimente com LFW ou seu próprio dataset.

---

## 🚀 Próximos Passos

Após preparar o dataset:

1. ✅ Verifique com `check_dataset.py`
2. ✅ Treine o modelo: `python train.py --data_dir data/orl --dataset_type orl`
3. ✅ Faça predições: `python predict.py --image sua_imagem.jpg`

---

## 💡 Dicas

- **Para testes rápidos:** Use ORL (pequeno e rápido)
- **Para resultados reais:** Use LFW ou seu próprio dataset
- **Para aprendizado:** Crie um dataset pequeno com 3-5 pessoas usando a webcam
- **Sempre verifique** a estrutura antes de treinar

---

## 📞 Precisa de Ajuda?

Se ainda tiver problemas:

1. Execute `python check_dataset.py` e veja os erros
2. Verifique se os arquivos estão no formato correto
3. Certifique-se de que a estrutura de diretórios está correta
4. Tente com um dataset menor primeiro (ex: apenas 2-3 pessoas)

Boa sorte! 🎉

