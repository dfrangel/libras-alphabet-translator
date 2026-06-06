from pathlib import Path
import cv2
import mediapipe as mp
import csv
import math
#===========================================================================#
# DIFERENTES VERSÕES DA FUNÇÃO DE NORMALIZAÇÃO (EXPLICADAS NO README.md)
def normalize_v1(raw):
    coords = []
    for i in range(0, len(raw), 3):
        # Separa as coordenadas em listas [x, y, z] individuais por ponto
        coords.append([raw[i], raw[i+1], raw[i+2]])

    pulso = coords[0]

    coords_centradas = []
    for ponto in coords:
        # Ponto real - pulso = centralizado
        x_novo = ponto[0] - pulso[0]
        y_novo = ponto[1] - pulso[1]
        z_novo = ponto[2] - pulso[2]
        
        coords_centradas.append([x_novo, y_novo, z_novo])
    
    distancias = []
    # Calcula a distância de cada um dos 20 pontos até o pulso (ignora o pulso [0])
    for ponto in coords_centradas[1:]:
        x, y, z = ponto[0], ponto[1], ponto[2]
        
        # Fórmula da Distância Euclidiana 3D
        distancia = math.sqrt(x**2 + y**2 + z**2)
        distancias.append(distancia)

    maior_distancia = max(distancias)
    
    # Protege divisão por 0
    if maior_distancia == 0:
        maior_distancia = 1
    
    # Normaliza todas as distâncias (Escalonamento Min-Max)
    distancias_normalizadas = []
    for d in distancias:
        distancias_normalizadas.append(d / maior_distancia)

    return distancias_normalizadas

def normalize_v2(raw):
    coords = []
    for i in range(0, len(raw), 3):
        #Separa as coordenadas em listas [x, y, z] individuais por ponto
        coords.append([raw[i], raw[i+1], raw[i+2]])

    pulso = coords[0]

    coords_centradas = []
    for ponto in coords:
        # Ponto real - pulso = centralizado
        x_novo = ponto[0] - pulso[0]
        y_novo = ponto[1] - pulso[1]
        z_novo = ponto[2] - pulso[2]
        
        coords_centradas.append([x_novo, y_novo, z_novo])
    
    distancias = []
    #Calcula a distância de cada um dos 20 pontos até o pulso
    for ponto in coords_centradas[1:]:
        x, y, z = ponto[0], ponto[1], ponto[2]
        
        #Fórmula da Distância Euclidiana 3D
        distancia = math.sqrt(x**2 + y**2 + z**2)
        distancias.append(distancia)

    pontas = [4, 8, 12, 16, 20]
    
    # Adiciona a distância entre Dedão-Indicador, Indicador-Médio, etc.
    for i in range(len(pontas) - 1):
        p1 = coords_centradas[pontas[i]]
        p2 = coords_centradas[pontas[i+1]]
        
        # Distância entre a ponta atual e a próxima ponta
        dist_dedos = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 + (p2[2] - p1[2])**2)
        distancias.append(dist_dedos)

    maior_distancia = max(distancias)
    #Protege divisão por 0
    if maior_distancia == 0:
        maior_distancia = 1
    
    #Normaliza todas as distâncias 20 do pulso + 4 dos dedos = 24 features
    distancias_normalizadas = []
    for d in distancias:
        distancias_normalizadas.append(d / maior_distancia)

    return distancias_normalizadas
#===========================================================================#

#===========================================================================#
data_dir = Path("data/train") # -> "data/train" / "data/test"
#===========================================================================#

stats = {}

header = []
#===========================================================================#
for i in range(1, 25):
    header.append(f"distancia_{i}") # -> alterar para a respectiva lógica
header.append("label")
#===========================================================================#


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

#===========================================================================#
with open("landmarks_.csv" # -> "nome final do arquivo.csv"
#===========================================================================#
          , "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)          # escreve o cabeçalho uma vez

    #Itera todos os folderes (letras)
    for folder in data_dir.iterdir():
        label = folder.name
        stats[label] = {"detectadas": 0, "falhas": 0}
        print(f"Processando letra {label}...")

        #Itera todas as instâncias (fotos)
        for image_path in folder.iterdir():
            #Carrega a imagem
            img = cv2.imread(str(image_path))
            #Converte para RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = hands.process(img_rgb)

            #Se encontra a mão extrai seus dados
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                raw = []
                for lm in hand_landmarks.landmark:
                    raw.extend([lm.x, lm.y, lm.z])
#===========================================================================#
                normalized = normalize_v2(raw) # -> escolher versão correta
#===========================================================================#
                if normalized is not None:  # proteção caso normalize retorne None
                    row = normalized + [label]
                    writer.writerow(row)
                
                stats[label]["detectadas"] += 1
            #Se não encontra não faz nada
            else:
                stats[label]["falhas"] += 1

#Imprime o panorâma geral de cada letra
for letra, contagem in stats.items():
    total = contagem["detectadas"] + contagem["falhas"]
    taxa = contagem["falhas"] / total * 100
    print(f"{letra}: {contagem['detectadas']} detectadas, {contagem['falhas']} falhas ({taxa:.1f}%)")