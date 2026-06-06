import cv2
import mediapipe as mp
import joblib
import numpy as np
import math

#===========================================================================#
modelo = joblib.load("model/classifier_v2.pkl") # -> definir modelo correto
#===========================================================================#


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


def main():
    CHARS_PER_LINE = 10
    # Inicializa a webcam
    webcam = cv2.VideoCapture(0)

    # 1. Configura as ferramentas de desenho e o modelo de detecção de mãos do MediaPipe
    mp_maos = mp.solutions.hands
    mp_desenho = mp.solutions.drawing_utils
    
    # Criamos o detector de mãos configurando para detectar no máximo 1 mão por enquanto
    maos = mp_maos.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)

    if not webcam.isOpened():
        print("Erro: Não foi possível acessar a webcam.")
        return

    print("MediaPipe iniciado! Mostre sua mão para a câmera.")

    word = ""
    current_letter = ""
    frame_count = 0
    CONFIRM_FRAMES = 20

    while True:
        sucesso, frame = webcam.read()
        if not sucesso:
            break

        # Inverte o frame horizontalmente para agir como um espelho (opcional, mas recomendado para Libras)
        frame = cv2.flip(frame, 1)

        # O OpenCV captura em formato BGR, mas o MediaPipe trabalha com RGB. 
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 2. O MediaPipe processa a imagem e procura por mãos
        resultado = maos.process(frame_rgb)

        prediction = ""
        confidence = 0.0

        # 3. Se ele encontrar alguma mão, desenha os pontos nela e faz a predição
        if resultado.multi_hand_landmarks:
            for pontos_da_mao in resultado.multi_hand_landmarks:
                mp_desenho.draw_landmarks(frame, pontos_da_mao, mp_maos.HAND_CONNECTIONS)

                raw = []
                for lm in pontos_da_mao.landmark:
                    raw.extend([lm.x, lm.y, lm.z])

                # Normaliza e prediz
                normalized = normalize_v2(raw) ###ALTERAR PARA DIFERENTES MODELOS###
                if normalized is not None:
                    prediction = modelo.predict([normalized])[0]
                    confidence = modelo.predict_proba([normalized]).max()

                    # Lógica de soletração
                    threshold = 0.55 if prediction in ["R", "U", "T", "L"] else 0.70
                    if confidence > threshold:
                        if prediction == current_letter:
                            frame_count += 1
                        else:
                            current_letter = prediction
                            frame_count = 0

                        if frame_count == CONFIRM_FRAMES:
                            word += current_letter
                            frame_count = 0

#===================================================================================#
# INTERFACE GRÁFICA
#===================================================================================#

        h, w, _ = frame.shape
        panel_x = w + 60
        dash_w = panel_x + 340
        dash_h = h + 100

        dashboard = np.zeros((dash_h, dash_w, 3), dtype=np.uint8)
        dashboard[:] = (34, 34, 34)

        # Título
        cv2.putText(dashboard, "LIBRAS TRADUTOR", (20, 35),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (200, 200, 200), 1)

        # Frame da webcam
        dashboard[50:50+h, 20:20+w] = frame

        # Atalhos abaixo da webcam (sem borda)
        cv2.putText(dashboard, "ESC sair  |  BKSP apagar letra  |  C limpar tudo",
                    (20, 50+h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (140, 140, 140), 1)

        # --- Card: Letra detectada ---
        card1_y = 50
        cv2.rectangle(dashboard, (panel_x, card1_y), (panel_x+310, card1_y+200), (42, 42, 42), -1)
        cv2.rectangle(dashboard, (panel_x, card1_y), (panel_x+310, card1_y+200), (58, 58, 58), 1)
        cv2.putText(dashboard, "LETRA DETECTADA", (panel_x+12, card1_y+22),
                    cv2.FONT_HERSHEY_DUPLEX, 0.45, (120, 120, 120), 1)

        if prediction and confidence > 0.45:
            cv2.putText(dashboard, prediction, (panel_x+14, card1_y+120),
                        cv2.FONT_HERSHEY_DUPLEX, 3.8, (224, 224, 224), 4)
            cv2.putText(dashboard, f"{confidence:.0%}", (panel_x+150, card1_y+120),
                        cv2.FONT_HERSHEY_DUPLEX, 1.1, (150, 150, 150), 2)

            # Barra de confiança
            bar_w = int(286 * confidence)
            cv2.rectangle(dashboard, (panel_x+12, card1_y+148), (panel_x+298, card1_y+156), (30, 30, 30), -1)
            cv2.rectangle(dashboard, (panel_x+12, card1_y+148), (panel_x+12+bar_w, card1_y+156), (180, 180, 180), -1)
            cv2.putText(dashboard, "confianca", (panel_x+12, card1_y+175),
                        cv2.FONT_HERSHEY_DUPLEX, 0.45, (100, 100, 100), 1)

        # --- Card: Palavra formada ---
        card2_y = card1_y + 216
        cv2.rectangle(dashboard, (panel_x, card2_y), (panel_x+310, card2_y+190), (42, 42, 42), -1)
        cv2.rectangle(dashboard, (panel_x, card2_y), (panel_x+310, card2_y+190), (58, 58, 58), 1)
        cv2.putText(dashboard, "CAIXA DE TEXTO", (panel_x+12, card2_y+22),
                    cv2.FONT_HERSHEY_DUPLEX, 0.45, (120, 120, 120), 1)

        # Quebra de linha automática
        line1 = word[:CHARS_PER_LINE]
        line2 = word[CHARS_PER_LINE:CHARS_PER_LINE*2]

        # Cursor piscante — aparece só na linha ativa
        import time
        cursor = "_" if int(time.time() * 2) % 2 == 0 else " "

        if len(word) < CHARS_PER_LINE:
            cv2.putText(dashboard, line1 + cursor, (panel_x+12, card2_y+80),
                        cv2.FONT_HERSHEY_DUPLEX, 1.1, (224, 224, 224), 2)
        else:
            cv2.putText(dashboard, line1, (panel_x+12, card2_y+80),
                        cv2.FONT_HERSHEY_DUPLEX, 1.1, (224, 224, 224), 2)
            cv2.putText(dashboard, line2 + cursor, (panel_x+12, card2_y+140),
                        cv2.FONT_HERSHEY_DUPLEX, 1.1, (224, 224, 224), 2)

        cv2.imshow("Tradutor de Libras", dashboard)
#===================================================================================#

#===================================================================================#
# COMANDOS TECLADO
#===================================================================================#
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27: # 27 é o código ASCII para a tecla ESC
            break
        elif key == ord('c'):
            word = ""
        elif key == 8: # 8 é o código ASCII para a tecla Backspace
            # Remove a última letra da palavra usando fatiamento de string (slicing)
            word = word[:-1]

    webcam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()