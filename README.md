# 👁️ EyeTracking_IA: Sistema de Predição de Olhar para Detecção de Fraude (Anti-Cola)

Este projeto implementa um sistema de **Gaze Tracking** (rastreamento do olhar) utilizando Deep Learning, com foco na predição das coordenadas $(\text{x}, \text{y})$ do olhar do usuário em tempo real. A aplicação visa monitorar o comportamento visual para auxiliar na detecção de tentativas de fraude (*anti-cola*) em ambientes de exames online.

---

## 👥 Integrantes do Projeto

| Nome | RA (Registro Acadêmico) |
| :--- | :--- |
| Gabriel Fuentes de Freitas Yamashita | 10408876 |
| Guilherme Florio Vieira | 10409698 |
| Henrique Nellessen | 10388168 |
| Pedro Akira Cardoso Toma | 10390171 |

## 🚀 Tecnologias e Arquitetura

| Componente | Função | Tecnologias Chave |
| :--- | :--- | :--- |
| **Modelo** | Predição das coordenadas do olhar $(\text{x}, \text{y})$. | TensorFlow / Keras, **MobileNetV2** (Transfer Learning) |
| **Visão Computacional** | Detecção de face e pré-processamento de imagem. | OpenCV (Haar Cascade) |
| **Interface** | Aplicação web interativa para demonstração em tempo real. | Streamlit |

---

## 📊 Processo de Desenvolvimento e Treinamento

O desenvolvimento passou por uma etapa de seleção e tratamento de dados crucial para a eficácia do modelo.

### 1. Escolha e Preparação do Dataset

* **Tentativa Inicial (MSU Online Exam Proctoring Dataset):** O primeiro dataset foi descartado por possuir apenas anotações de intervalo de tempo de trapaça, o que era insuficiente para o treinamento de um modelo preditivo de coordenadas de olhar.
* **Dataset Final (MPIIGAZE):** Foi escolhido o dataset **MPIIGAZE**, que fornece **imagens da região dos olhos** e as **coordenadas $(\text{x}, \text{y})$ exatas do olhar** da pessoa na tela, sendo mais propício para o aprendizado supervisionado de predição de coordenadas.

### 2. Treinamento do Modelo

O treinamento foi executado em um notebook Python, seguindo as seguintes etapas:

* **Pré-processamento:** Extração, limpeza e normalização dos dados de entrada.
* **Separação:** Divisão dos dados em conjuntos de treinamento e testes.
* **Arquitetura do Modelo:** O **MobileNetV2** (pré-treinado) foi utilizado como *backbone* (**Transfer Learning**). O modelo foi modificado para receber as imagens dos olhos como entrada e gerar **duas saídas** (para a coordenada $x$ e a coordenada $y$ do olhar).
* **Otimização:** Foi implementado um maneira para armazenar o melhor modelo e evitar armazenar um modelo com *overfitting*.
* **Métricas de Avaliação:** O treinamento foi monitorado utilizando **Função de Perda (MSE - Erro Quadrático Médio)** e **Métrica (MAE - Erro Absoluto Médio)**, com gráficos gerados ao final.

---

## 🛠️ Aplicação e Funcionamento em Tempo Real

O projeto utiliza a classe `GazeModelProcessor` para integrar a predição do olhar à aplicação Streamlit.

1.  **Captura de Vídeo:** A aplicação Streamlit utiliza a webcam para capturar o *frame* de vídeo.
2.  **Detecção de Face:** O **OpenCV (Haar Cascade)** é usado para **detectar e localizar a face** do usuário no *frame*.
3.  **Pré-processamento:** A região da face/olhos é cortada e **redimensionada para $224 \times 224$ pixels**, o formato esperado pelo MobileNetV2.
4.  **Predição:** O modelo treinado retorna as coordenadas previstas $(\text{x}, \text{y})$ do olhar na tela.
5.  **Visualização:**
    * Um ponto é **desenhado sobre o *frame* da webcam** para visualização imediata da previsão.
    * Coordenadas x e y são exibidas sobre a própria interface Streamlit, demonstrando o ponto de predição.
6.  **Alerta de Fraude:** Caso o rastreamento do olhar **falhe (e.g., face não detectada)**, um **Alerta de Trapaça** é acionado, conforme o critério de monitoramento implementado no Streamlit.

---

## 🏁 Como Executar o Projeto

Para executar a aplicação localmente, siga os passos abaixo (assumindo que você já tenha configurado o ambiente Python e as dependências).

1.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Inicie a aplicação Streamlit:**
    ```bash
    streamlit run main.py
    ```
    *(A aplicação abrirá automaticamente no seu navegador.)*


