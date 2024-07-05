import tkinter as tk
from PIL import Image, ImageTk
import cv2
import requests

# Substitua pelas suas informações da API da Azure
PREDICTION_KEY  = ''
ENDPOINT  = ''
PROJECT_ID  = ''
PUBLISHED_NAME  = 'TaskFace'

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Reconhecimento Facial")

        #configura o frame do vídeo
        self.video_frame = tk.Label(root)
        self.video_frame.pack()

        #Botão para tirar foto
        self.btn_tirar_foto = tk.Button(root, text="Tirar Foto", command=self.tirar_foto, bg='blue', fg='white')
        self.btn_tirar_foto.pack(pady=20)

        #Inicializa a captura de vídeo com a webcam
        self.cap = cv2.VideoCapture(0)

        #Carrega o classificador de face Haar Cascade
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.recognized_name = ""
        self.update_frame()

    def update_frame(self):
        #Captura o frame da webcam
        ret, frame = self.cap.read()
        if ret:
            #Converte o frame para escala de cinza
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #Detecta as faces no frame
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            #Desenha um retângulo ao redor das faces detectadas e exibe o nome reconhecido
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                if self.recognized_name:
                    cv2.putText(frame, self.recognized_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            #Converte o frame para formato compativel e exibe na interface gráfica
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.video_frame.config(image=self.photo)
        #atualiza o frame a cada 10ms
        self.root.after(10, self.update_frame)

    def tirar_foto(self):
        #Captura um frame da webcam
        ret, frame = self.cap.read()
        if ret:
            #Salva o frame como uma imagem
            cv2.imwrite("foto.jpg", frame)
            #envia a imagem para o serviço da azure para reconhecimento
            self.enviar_para_azure("foto.jpg")
        else:
            print("Não foi possível capturar a imagem.")

    def enviar_para_azure(self, image_path):
        with open(image_path, "rb") as image:
            headers = {
                'Prediction-Key': PREDICTION_KEY,
                'Content-Type': 'application/octet-stream'
            }
            
            url = f"{ENDPOINT}/customvision/v3.0/Prediction/{PROJECT_ID}/detect/iterations/{PUBLISHED_NAME}/image"
            #envia a imagem para a API da Azure
            response = requests.post(url, headers=headers, data=image)
            if response.status_code == 200:
                #Processa os resultados da API
                resultados = response.json().get('predictions', [])
                max_probability = 0.0
                max_tag = None

                #encontra a predição com maior probabilidade
                for resultado in resultados:
                    if resultado['probability'] > max_probability:
                        max_probability = resultado['probability']
                        max_tag = resultado['tagName']

                #define o nome reconhecido baseado na probaliidade
                if max_probability < 0.9:
                    self.recognized_name = "Nao Reconhecido"
                else:
                    self.recognized_name = f"{max_tag} ({max_probability:.2f})"
            else:
                self.recognized_name = "Erro ao enviar a imagem para o Azure."

    #Libera a webcam e fecha a janela
    def on_closing(self):
        self.cap.release()
        self.root.destroy()

# Configuração da interface gráfica
root = tk.Tk()
app = App(root)
root.protocol("WM_DELETE_WINDOW", app.on_closing)
root.mainloop()
