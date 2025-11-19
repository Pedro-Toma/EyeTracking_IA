import streamlit as st
import cv2
import numpy as np
from src.gaze_processor import GazeModelProcessor 
import time

# --- Configurações da Aplicação Streamlit ---
st.set_page_config(
    page_title="Predição de Olhar (Gaze Tracking)",
    layout="wide"
)

# Dimensões de Exibição/Tela (ajuste conforme necessário)
SCREEN_W = 640
SCREEN_H = 480

# Inicialização do estado da sessão do Streamlit
if 'run_detection' not in st.session_state:
    st.session_state.run_detection = False

@st.cache_resource
def load_gaze_processor():
    """Carrega o processador de olhar (modelo Keras e Haar Cascade) 
    e armazena em cache para evitar recargas."""
    processor = GazeModelProcessor(SCREEN_W, SCREEN_H)
    return processor

def start_webcam():
    """Função chamada ao clicar no botão Iniciar."""
    st.session_state.run_detection = True

def stop_webcam():
    """Função chamada ao clicar no botão Parar."""
    st.session_state.run_detection = False

def main():
    """Função principal da aplicação Streamlit."""
    
    st.title("👁️ Predição de Coordenada Olhada (Gaze Tracking)")
    st.markdown("Use o botão na barra lateral para iniciar a detecção.")

    # 1. BARRA LATERAL (CONTROLES)
    st.sidebar.header("Controles da Webcam")

    # Botões de Iniciar/Parar
    if not st.session_state.run_detection:
        st.sidebar.button("▶️ Iniciar Detecção", on_click=start_webcam, type="primary")
    else:
        st.sidebar.button("⏸️ Parar Detecção", on_click=stop_webcam, type="secondary")

    # 2. CARREGAR MODELO
    processor = load_gaze_processor()

    if processor.model is None or processor.face_cascade is None:
        st.error("Falha ao carregar o modelo de Gaze ou o detector de face. Verifique os caminhos e a estrutura.")
        return

    # 3. LOOP DE DETECÇÃO (Só executa se o estado for True)
    if st.session_state.run_detection:
        
        st.subheader("Webcam e Resultado da Predição")
        frame_placeholder = st.empty()
        gaze_coords_placeholder = st.empty()
        
        # 3.1. Configurar a Captura de Vídeo
        # Usa o índice selecionado na sidebar
        cap = cv2.VideoCapture(0)
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_H)
        
        if not cap.isOpened():
            # AVISO CRÍTICO: Não conseguiu acessar a câmera
            st.error(f"🔴 ERRO: Não foi possível acessar a webcam. Tente outro índice ou verifique se a câmera não está em uso.")
            stop_webcam() # Reseta o estado para permitir nova tentativa
            return

        with frame_placeholder.container():
            st.info("Webcam ativa. Aguardando predição de olhar...")

        try:
            # Mantém o loop ativo ENQUANTO o estado da sessão for True
            while st.session_state.run_detection:
                ret, frame = cap.read()
                
                if not ret:
                    st.warning("Não foi possível ler o frame da webcam. A câmera pode ter sido desconectada.")
                    time.sleep(0.1)
                    continue
                
                frame = cv2.flip(frame, 1) # Espelha o frame

                # 3.2. Processar o Frame
                gaze_x, gaze_y = processor.get_gaze_coordinates(frame)
                
                # 3.3. Desenhar e Exibir
                output_frame = processor.draw_gaze(frame.copy(), gaze_x, gaze_y)
                output_frame_rgb = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
                
                frame_placeholder.image(output_frame_rgb, caption="Feed da Webcam com Predição de Olhar", use_column_width=True)
                
                if gaze_x is not None and gaze_y is not None:
                    gaze_coords_placeholder.success(f"**Coordenada de Olhar Predita:** (X: **{gaze_x}**, Y: **{gaze_y}**)")
                else:
                    gaze_coords_placeholder.error("Alerta de Cola! Nenhuma face detectada.")
                
                # O Streamlit precisa de um breve sleep ou interação para atualizar
                time.sleep(0.01)

        except Exception as e:
            st.error(f"Ocorreu um erro durante o processamento de vídeo: {e}")
            frame_placeholder.empty()

        finally:
            # 3.4. Liberação de Recursos
            cap.release()
            st.info("✅ Câmera liberada. Clique em 'Iniciar' para recomeçar.")

    else:
        st.info("Clique em '▶️ Iniciar Detecção e Webcam' na barra lateral para começar a predição.")

if __name__ == "__main__":
    main()