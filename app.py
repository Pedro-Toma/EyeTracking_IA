import streamlit as st
import cv2
import numpy as np
from src.gaze_processor import GazeModelProcessor 
import time

# Título da aplicação
st.set_page_config(
    page_title="Predição de Olhar (Gaze Tracking)",
    layout="wide"
)

# Dimensões de Exibição/Tela
SCREEN_W = 1920
SCREEN_H = 1080

# Inicialização do estado da sessão do Streamlit
if 'run_detection' not in st.session_state:
    st.session_state.run_detection = False

@st.cache_resource
def load_gaze_processor():
    # Carrega o processador da imagem
    processor = GazeModelProcessor(SCREEN_W, SCREEN_H)
    return processor

def start_webcam():
    st.session_state.run_detection = True

def stop_webcam():
    st.session_state.run_detection = False

def main():
    st.title("👁️ Predição de Coordenada Olhada (Gaze Tracking)")
    st.markdown("Use o botão na barra lateral para iniciar a detecção.")

    st.sidebar.header("Controles da Webcam")

    # Controle da webcam
    if not st.session_state.run_detection:
        st.sidebar.button("▶️ Iniciar Detecção", on_click=start_webcam, type="primary")
    else:
        st.sidebar.button("⏸️ Parar Detecção", on_click=stop_webcam, type="secondary")

    # Carega processador de imagens
    processor = load_gaze_processor()

    if processor.model is None or processor.face_cascade is None:
        st.error("Falha ao carregar o modelo de Gaze ou o detector de face. Verifique os caminhos e a estrutura.")
        return

    # Loop de deteccção de cola
    if st.session_state.run_detection:
        
        st.subheader("Webcam e Resultado da Predição")
        frame_placeholder = st.empty()
        gaze_coords_placeholder = st.empty()
        
        cap = cv2.VideoCapture(0)
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_H)
        
        if not cap.isOpened():
            # Erro: não foi possível acessar a webcam
            st.error(f"🔴 ERRO: Não foi possível acessar a webcam. Tente outro índice ou verifique se a câmera não está em uso.")
            stop_webcam()
            return

        with frame_placeholder.container():
            st.info("Webcam ativa. Aguardando predição de olhar...")

        try:
            # Loop de detecção de cola com a webcam
            while st.session_state.run_detection:
                ret, frame = cap.read()
                
                if not ret:
                    st.warning("Não foi possível ler o frame da webcam. A câmera pode ter sido desconectada.")
                    time.sleep(0.1)
                    continue
                
                frame = cv2.flip(frame, 1) # Espelha o frame

                # Processar o Frame
                gaze_x, gaze_y = processor.get_gaze_coordinates(frame)
                
                # Desenhar e Exibir o ponto de previsão do modelo
                output_frame = processor.draw_gaze(frame.copy(), gaze_x, gaze_y)
                output_frame_rgb = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
                
                frame_placeholder.image(output_frame_rgb, caption="Feed da Webcam com Predição de Olhar", use_column_width=True)
                
                # Se não foi detectado olhar, possível cola
                if gaze_x is not None and gaze_y is not None:
                    gaze_coords_placeholder.success(f"**Coordenada de Olhar Predita:** (X: **{gaze_x}**, Y: **{gaze_y}**)")
                else:
                    gaze_coords_placeholder.error("Alerta de Trapaça! Nenhuma face detectada.")
                
                # Interação para atualizar o streamlit
                time.sleep(0.01)

        except Exception as e:
            st.error(f"Ocorreu um erro durante o processamento de vídeo: {e}")
            frame_placeholder.empty()

        finally:
            # Liberação de Recursos
            cap.release()
            st.info("✅ Câmera liberada. Clique em 'Iniciar' para recomeçar.")

    else:
        st.info("Clique em '▶️ Iniciar Detecção e Webcam' na barra lateral para começar a predição.")

if __name__ == "__main__":
    main()