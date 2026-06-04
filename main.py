import cv2
import mediapipe as mp

def main():
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

    print("MediaPipe iniciado! Mostre sua mão para a câmera. Pressione 'q' para fechar.")

    while True:
        sucesso, frame = webcam.read()
        if not sucesso:
            break

        # O OpenCV captura em formato BGR, mas o MediaPipe trabalha com RGB. 
        # Precisamos inverter as cores antes de enviar para o modelo.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 2. O MediaPipe processa a imagem e procura por mãos
        resultado = maos.process(frame_rgb)

        # 3. Se ele encontrar alguma mão, desenha os pontos nela
        if resultado.multi_hand_landmarks:
            for pontos_da_mao in resultado.multi_hand_landmarks:
                # Desenha os pontos e as linhas de conexão sobre o frame original
                mp_desenho.draw_landmarks(frame, pontos_da_mao, mp_maos.HAND_CONNECTIONS)

        # Exibe o frame com os desenhos na tela
        cv2.imshow("Tradutor de Libras - Rastreamento de Mao", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    webcam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()