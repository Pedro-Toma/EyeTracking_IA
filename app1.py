import streamlit as st
import cv2
import time
import numpy as np
from src.gaze_processor import GazeModelProcessor
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import os

cv2.setNumThreads(0) 
cv2.setUseOptimized(False)

# Coloca o titulo na página web
st.set_page_config(page_title="Gaze Tracking - Testes Online", layout="wide")
st.title("👁️ Sistema de Monitoramento de Olhar (WebRTC)")

# Inicializa variáveis de estado globais
if 'cheating_start_time' not in st.session_state:
    st.session_state.cheating_start_time = None
if 'cheating_history' not in st.session_state:
    st.session_state.cheating_history = []
if 'last_status' not in st.session_state:
    st.session_state["last_status"] = "Aguardando"
if 'last_message' not in st.session_state:
    st.session_state["last_message"] = "Aguardando conexão da webcam..."
if 'is_cheating_now' not in st.session_state:
    st.session_state["is_cheating_now"] = False
if 'gaze_coords' not in st.session_state:
    st.session_state["gaze_coords"] = (0, 0)
    
CHEAT_DURATION_THRESHOLD = 3 # Mínimo de 3 segundo olhando para fora da tela

# Processamento de Vídeo
class VideoProcessor(VideoProcessorBase):
    def __init__(self):

        self.analyzer = GazeModelProcessor(screen_w=640, screen_h=480) 
        
        self.gaze_x = None
        self.gaze_y = None
        self.frame_counter = 0
        self.skip_frames = 5 # Processa a IA apenas a cada 3 frames para estabilidade

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        current_status = "INSTRUÇÃO"
        status_message = "⚠️ Inicializando processamento..."

        try:
            img = cv2.flip(img, 1)
            img = cv2.resize(img, (640, 480))
            
            self.frame_counter += 1
            
            # Pular frames
            if self.frame_counter % self.skip_frames == 0 and self.analyzer.model:
                
                # chama método de pegar coordenadas do olhar
                gaze_x, gaze_y = self.analyzer.get_gaze_coordinates(img)
                
                # Armazena as coordenadas
                self.gaze_x = gaze_x
                self.gaze_y = gaze_y
                
            
            # Desenha as coordenadas na imagem
            if self.gaze_x is not None and self.gaze_y is not None:
                img = self.analyzer.draw_gaze(img, self.gaze_x, self.gaze_y)
                status_message = f"Gaze X/Y: ({self.gaze_x}, {self.gaze_y})"
                current_status = "OK"
            else:
                current_status = "INSTRUÇÃO"
                status_message = "Movimente o rosto para o centro da tela."
                
            # Atualiza variáveis de estado
            st.session_state["last_status"] = current_status
            st.session_state["last_message"] = status_message
            st.session_state["is_cheating_now"] = False
            if self.gaze_x is not None:
                st.session_state["gaze_coords"] = (self.gaze_x, self.gaze_y)

        except Exception as e:
            st.session_state["last_status"] = "ERRO"
            st.session_state["is_cheating_now"] = False

        return img 

col1, col2 = st.columns([2, 1])

# DEFINIÇÃO DO CONTEXTO WEBRTC NO ESCOPO PRINCIPAL
webrtc_ctx = webrtc_streamer(
    key="gaze_model_stream",
    video_processor_factory=VideoProcessor,
    async_transform=True,
    media_stream_constraints={"video": True, "audio": False}
)

with col1:
    st.header("Webcam em Tempo Real")

with col2:
    st.header("Status do Aluno")
    status_placeholder = st.empty()
    st.markdown("---")
    
    # Adicionando a exibição das coordenadas no painel lateral
    st.header("Coordenadas Atuais")
    coord_placeholder = st.empty()
    
    st.markdown("---")
    st.header("Histórico de Fraude")
    history_placeholder = st.empty()
    
# Loop de atualização do status (Thread Principal do Streamlit)
while True:
    
    is_playing = getattr(webrtc_ctx.state, "playing", False)
    
    if not is_playing:
        st.session_state["last_message"] = "Aguardando conexão da webcam..."
        st.session_state["last_status"] = "Aguardando"
        status_placeholder.info(st.session_state["last_message"])
        time.sleep(0.5)
        continue 
    
    current_status = st.session_state.get("last_status", "INSTRUÇÃO") 
    status_message = st.session_state.get("last_message", "Iniciando rastreamento...")
    current_time = time.time()
    
    # --- Lógica de Estado Contínuo (FRAUDE, INSTRUÇÃO, OK) ---
    # ... (Seu código de lógica de fraude permanece o mesmo) ...
    if current_status == "FRAUDE":
        if st.session_state.cheating_start_time is None:
            st.session_state.cheating_start_time = current_time
        
        elapsed = current_time - st.session_state.cheating_start_time
        
        if elapsed >= CHEAT_DURATION_THRESHOLD:
            status_placeholder.error(f"{status_message.split(':')[0]} 🚨 Tempo Contínuo: {elapsed:.1f}s")
            
            if not st.session_state.cheating_history or (current_time - st.session_state.cheating_history[-1]['time']) > 5:
                st.session_state.cheating_history.append({
                    "time": current_time,
                    "type": "Desvio de Pose Contínuo",
                    "duration": f"{elapsed:.1f}s"
                })
                
        else:
            status_placeholder.warning(f"⚠️ Atenção: Desvio. Contagem: {elapsed:.1f}s")
            
    elif current_status == "INSTRUÇÃO":
        st.session_state.cheating_start_time = None 
        status_placeholder.info(status_message)

    elif current_status == "ERRO":
        st.session_state.cheating_start_time = None 
        status_placeholder.error(status_message) 
        
    else: # Status é "OK" ou "Aguardando"
        st.session_state.cheating_start_time = None 
        status_placeholder.success(status_message)

    # --- Atualiza as Coordenadas e o Histórico ---
    current_x, current_y = st.session_state.get("gaze_coords", (0, 0))
    coord_placeholder.markdown(f"**X:** `{current_x}` | **Y:** `{current_y}`")

    history_table_data = [
        {"Horário": time.strftime("%H:%M:%S", time.localtime(h['time'])), 
         "Tipo": h['type'], 
         "Duração": h['duration']} 
        for h in st.session_state.cheating_history
    ]
    history_placeholder.dataframe(history_table_data, use_container_width=True, hide_index=True)
    
    time.sleep(0.1)