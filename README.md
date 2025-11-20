# 👁️ EyeTracking_IA: Sistema de Predição de Olhar para Detecção de Fraude (Anti-Cola)

Este projeto implementa um sistema de **Eye Tracking** utilizando Deep Learning, com foco na predição das coordenadas $(\text{x}, \text{y})$ do olhar do usuário em tempo real. A aplicação visa monitorar o comportamento visual para auxiliar na detecção de tentativas de trapaça em ambientes de exames online.

---

## 👥 Integrantes do Projeto

| Nome | RA (Registro Acadêmico) |
| :--- | :--- |
| Gabriel Fuentes de Freitas Yamashita | 10408876 |
| Guilherme Florio Vieira | 10409698 |
| Henrique Nellessen | 10388168 |
| Pedro Akira Cardoso Toma | 10390171 |

## 🧱 Estrutura do Projeto

* **Bibliotecas Necessárias**: As bibliotecas utilizadas para o desenvolvimento do projeto estão localizado no arquivo [requirements.txt](https://github.com/Pedro-Toma/EyeTracking_IA/blob/main/requirements.txt).
* **Dataset**: Foi utilizado um dataset público chamado [MPIIGAZE](https://www.kaggle.com/datasets/dhruv413/mpiigaze), mas ele não foi upado para o github devido a grande quantida de imagens.
* **Notebook Python**: Contém o treinamento do modelo utilizando transfer learning, está localizado em [/src/model_training](https://github.com/Pedro-Toma/EyeTracking_IA/blob/main/src/model_training.ipynb).
* **Modelo**: Os pesos do modelo treinado foram armazenados em um arquivo .h5, localizado em [/models/mobilenet_gaze_tl_best.h5](https://github.com/Pedro-Toma/EyeTracking_IA/blob/main/models/mobilenet_gaze_tl_best.h5).
* **Consumo do Modelo**: O carregamento do modelo e a obtenção da predição do modelo é feito pelo arquivo gaze_processor.py, localizado em [/src/gaze_processor.py](https://github.com/Pedro-Toma/EyeTracking_IA/blob/main/src/gaze_processor.py).
* **Aplicação Streamlit**: Capta as imagens do usuário e retorna a predição do modelo, localizado em [app.py](https://github.com/Pedro-Toma/EyeTracking_IA/blob/main/app.py)

## 💻 Tecnologias Utilizadas

* **Python 3.11.9**
* **TensorFlow**
* **Keras**
* **Numpy**
* **Pandas**
* **OpenCV**
* **Scikit-learn**
* **Matplotlib**

## ▶️ Como Executar o Projeto

Para executar a aplicação, siga os passos abaixo:

**1. Instale Python 3.11.9**  
   [Python 3.11.9](https://www.python.org/downloads/release/python-3119/)
   
**2. Clone o Repositório** (em uma pasta local)
   ```bash
   git init
   git clone https://github.com/Pedro-Toma/EyeTracking_IA.git
   cd EyeTracking_IA
   ```
**3. Crie o Ambiente Virtual** 
   ```bash
   python -m venv venv
   ```
**4. SE ESTIVER NO POWERSHELL** (Adquira permissão para scripts)
   ```bash
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
   ```
**5. Ative o Ambiente Virtual**
   ```bash
   .\venv\Scripts\activate
   ```
**6. Instale as Dependências**
   ```bash
   pip install -r requirements.txt
   ```
**7. Execute a Aplicação Streamlit** (a aplicação será aberta automaticamente no navegador)
   ```bash
   streamlit run app.py
   ```

## 🎥 Vídeo de Apresentação
   [Vídeo](https://youtu.be/erq2BtiPR7c)

## 📄 Artigo do Projeto
   [Artigo](https://github.com/Pedro-Toma/EyeTracking_IA/blob/main/Artigo_IA_Eye_Tracking.pdf)


