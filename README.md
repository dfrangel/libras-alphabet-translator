# Libras Alphabet Translator

Esse projeto é um protótipo de reconhecimento em tempo real das letras estáticas do alfabeto de Libras (Língua Brasileira de Sinais) utilizando visão computacional. O sistema detecta a mão do usuário via webcam, extrai características geométricas dos pontos-chave da mão e classifica o gesto correspondente a uma letra, permitindo digitar na tela em tempo real.

## Alfabeto suportado

21 letras estáticas: **A B C D E F G I L M N O P Q R S T U V W Y**

> As letras com movimento (H, J, K, X e Z) não estão incluídas nesta versão. O reconhecimento de gestos dinâmicos será implementado em uma próxima etapa utilizando redes neurais recorrentes (LSTM).

## Demonstração

O sistema exibe um dashboard com o feed da webcam, a letra detectada com sua confiança e uma caixa de texto onde a palavra vai sendo formada conforme os gestos são confirmados.                      
<img width="500" height="305" alt="image" src="https://github.com/user-attachments/assets/6b2e151b-1881-4aed-a7a5-10a3e4a58a78"> 
<img width="500" height="305" alt="image" src="https://github.com/user-attachments/assets/d0b5289a-c70f-45dd-a537-4553931d8e88" />
<img width="1302" height="765" alt="image" src="https://github.com/user-attachments/assets/68e675d8-8ea2-4e23-a804-d32f9fe94320" />

---

## Versões do modelo

### V1 — Distâncias ao pulso (20 features)

Calcula a **distância euclidiana 3D** de cada um dos 20 pontos da mão até o pulso (landmark 0). As distâncias são normalizadas pelo valor máximo, resultando em 20 features no intervalo [0, 1].

```
features = [dist(p1, pulso), dist(p2, pulso), ..., dist(p20, pulso)]
```

Esta representação captura o "tamanho relativo" de cada dedo em relação ao pulso, sendo suficiente para distinguir a maioria das letras.

### V2 — Distâncias ao pulso + distâncias entre dedos adjacentes (24 features)

Expande a V1 adicionando as **distâncias entre as pontas de dedos adjacentes**: polegar→indicador, indicador→médio, médio→anelar e anelar→mínimo.

```
features = [20 distâncias ao pulso] + [dist(p4,p8), dist(p8,p12), dist(p12,p16), dist(p16,p20)]
```

Essas 4 distâncias extras capturam o espaçamento entre os dedos — informação importante para diferenciar letras como R, U e V, cujos pontos ao pulso são muito similares mas cujas pontas de dedos têm separações distintas.


## Dataset

**[Libras Dataset de williansoliveira — Kaggle](https://www.kaggle.com/datasets/williansoliveira/libras)**

Dataset específico para Libras contendo imagens de mãos fazendo os gestos do alfabeto estático, já dividido nos conjuntos de treino e teste.

| Split | Amostras (após extração) |
|-------|--------------------------|
| Treino | ~34.000 |
| Teste | ~11.400 |

O dataset original não contém as letras H, J, K, X e Z.

## Instalação e execução

### Pré-requisitos

- Python 3.10 ou superior
- Webcam funcional

### 1. Clone o repositório

```bash
git clone https://github.com/dfrangel/libras-alphabet-translator.git
cd libras-alphabet-translator
```

### 2. Crie e ative um ambiente virtual

```bash
python -m venv .venv
```

```bash
# Windows
.venv\Scripts\activate

# Linux / Mac
source .venv/bin/activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Execute

```bash
python main.py
```

## Próximos passos
- [ ] Reconhecimento das letras com movimento (H, J, K, X, Z) via LSTM
- [ ] Suporte a palavras completas do dicionário de Libras
---

## Autor

**Davi F. Rangel**
[GitHub](https://github.com/dfrangel)
