from pathlib import Path
import cv2
import mediapipe as mp
import csv

def normalize(raw):
    base_x = raw[0]   # landmark 0 é o pulso
    base_y = raw[1]
    base_z = raw[2]

    normalized = []
    for i in range(21):
        normalized.append(raw[i*3]   - base_x)   # x centralizado
        normalized.append(raw[i*3+1] - base_y)   # y centralizado
        normalized.append(raw[i*3+2] - base_z)   # z centralizado

    max_val = max(abs(v) for v in normalized)     # maior valor absoluto

    if max_val == 0:                              # proteção contra divisão por zero
        return None

    normalized = [v / max_val for v in normalized]
    return normalized

data_dir = Path("data/test")

stats = {}

header = []
for i in range(21):
    header.extend([f"x{i}", f"y{i}", f"z{i}"])
header.append("label")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

with open("landmarks_test.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)          # escreve o cabeçalho uma vez

    for folder in data_dir.iterdir():
        label = folder.name
        stats[label] = {"detectadas": 0, "falhas": 0}
        print(f"Processando letra {label}...")

        for image_path in folder.iterdir():
            #Carrega a imagem
            img = cv2.imread(str(image_path))
            #Converte para RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                raw = []
                for lm in hand_landmarks.landmark:
                    raw.extend([lm.x, lm.y, lm.z])
                    
                normalized = normalize(raw)
                if normalized is not None:            # proteção caso normalize retorne None
                    row = normalized + [label]
                    writer.writerow(row)
                
                stats[label]["detectadas"] += 1
            else:

                stats[label]["falhas"] += 1
        
for letra, contagem in stats.items():
    total = contagem["detectadas"] + contagem["falhas"]
    taxa = contagem["falhas"] / total * 100
    print(f"{letra}: {contagem['detectadas']} detectadas, {contagem['falhas']} falhas ({taxa:.1f}%)")