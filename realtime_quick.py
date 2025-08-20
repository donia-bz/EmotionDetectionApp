import cv2
import numpy as np
from tensorflow.keras.models import load_model
from datetime import datetime
import csv
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# -----------------------------
# Config
# -----------------------------
MODEL_PATH = "emotion_model.h5"
LOG_CSV = "emotion_logs.csv"

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
EMOJI_MAP = {
    'Angry':'😠', 'Disgust':'🤢', 'Fear':'😨',
    'Happy':'😊', 'Sad':'😢', 'Surprise':'😲', 'Neutral':'😐'
}

# Charger modèle
model = load_model(MODEL_PATH, compile=False)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Init CSV
if not os.path.exists(LOG_CSV):
    with open(LOG_CSV, 'w', newline='') as f:
        csv.writer(f).writerow(["Timestamp", "Mode", "Emotion", "Confidence"])

# -----------------------------
# Fonctions auxiliaires
# -----------------------------
def preprocess_face(face_bgr):
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(gray, (64,64))
    face = face.astype("float32")/255.0
    face = np.expand_dims(face, -1)
    face = np.expand_dims(face, 0)
    return face

def log_emotion_csv(mode, emotion, confidence):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_CSV, 'a', newline='') as f:
        csv.writer(f).writerow([ts, mode, emotion, f"{confidence:.2f}"])

# -----------------------------
# Classe Tkinter
# -----------------------------
class EmotionApp:
    def __init__(self, root):
        self.root = root
        root.title("Détection d'émotions avancée")
        root.geometry("1000x800")

        self.cap = None
        self.pause = False
        self.mode = None
        self.emotion_history = []

        # ----------- Boutons -----------
        btn_frame = tk.Frame(root)
        btn_frame.pack(fill='x', pady=6)
        tk.Button(btn_frame, text="Webcam", command=self.start_webcam, width=12).pack(side='left', padx=5)
        tk.Button(btn_frame, text="Charger vidéo", command=self.load_video, width=12).pack(side='left', padx=5)
        tk.Button(btn_frame, text="Pause/Reprendre", command=self.toggle_pause, width=12).pack(side='left', padx=5)
        tk.Button(btn_frame, text="Résumé statistique", command=self.show_summary, width=16).pack(side='left', padx=5)
        tk.Button(btn_frame, text="Ouvrir logs CSV", command=self.open_logs, width=12).pack(side='left', padx=5)

        # ----------- Labels & Canvas -----------
        self.status_label = tk.Label(root, text="Mode non sélectionné", font=("Helvetica", 14))
        self.status_label.pack(pady=6)

        self.frame_label = tk.Label(root)
        self.frame_label.pack()

        # Probabilities histogram
        self.fig, self.ax = plt.subplots(figsize=(9,2))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(pady=6)
        self.ax.set_ylim(0, 1)
        self.ax.set_title("Probabilités des émotions")
        self.bar_container = None

        self.root.after(30, self.update_frame)

    # -----------------------------
    # Fonctions
    # -----------------------------
    def start_webcam(self):
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(0)
        self.mode = "Webcam"
        self.status_label.config(text="Mode Webcam activé")
        self.emotion_history.clear()

    def load_video(self):
        path = filedialog.askopenfilename(
            title="Sélectionner vidéo",
            filetypes=[("MP4 files","*.mp4"), ("All files","*.*")]
        )
        if path:
            if self.cap:
                self.cap.release()
            self.cap = cv2.VideoCapture(path)
            self.mode = "Vidéo"
            self.status_label.config(text=f"Vidéo chargée: {os.path.basename(path)}")
            self.emotion_history.clear()

    def toggle_pause(self):
        self.pause = not self.pause
        self.status_label.config(text="Pause" if self.pause else f"{self.mode} en cours")

    def draw_probabilities(self, probs):
        self.ax.clear()
        self.ax.bar(emotion_labels, probs, color="#4b9cd3")
        self.ax.set_ylim(0, 1)
        self.ax.set_ylabel("Probabilité")
        self.ax.set_title("Probabilités des émotions")
        self.canvas.draw()

    def update_frame(self):
        if self.cap is None:
            self.root.after(500, self.update_frame)
            return

        if self.cap.isOpened() and not self.pause:
            ret, frame = self.cap.read()
            if ret:
                display = frame.copy()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
                all_preds = []

                for (x, y, w, h) in faces:
                    face_roi = frame[y:y+h, x:x+w]
                    inp = preprocess_face(face_roi)
                    preds = model.predict(inp, verbose=0)[0]
                    idx = int(np.argmax(preds))
                    confidence = float(preds[idx])
                    emotion = emotion_labels[idx]
                    log_emotion_csv(self.mode, emotion, confidence)
                    self.emotion_history.append(emotion)

                    cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    emoji = EMOJI_MAP.get(emotion,'')
                    cv2.putText(display, f"{emoji} {emotion} ({confidence:.2f})", (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    all_preds.append(preds)

                avg_preds = np.mean(all_preds, axis=0) if all_preds else np.zeros(len(emotion_labels))
                self.draw_probabilities(avg_preds)

                rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb).resize((960,540))
                imgtk = ImageTk.PhotoImage(image=img)
                self.frame_label.imgtk = imgtk
                self.frame_label.configure(image=imgtk)
            else:
                if self.mode == "Vidéo":
                    self.status_label.config(text="Vidéo terminée")
                if self.cap:
                    self.cap.release()
                    self.cap = None

        self.root.after(30, self.update_frame)

    def show_summary(self):
        if not self.emotion_history:
            messagebox.showinfo("Résumé", "Aucune émotion détectée pour le moment.")
            return
        counter = {}
        for emo in self.emotion_history:
            counter[emo] = counter.get(emo,0)+1
        summary = "Résumé des émotions détectées:\n\n"
        for emo, count in counter.items():
            summary += f"{EMOJI_MAP.get(emo,'')} {emo}: {count}\n"
        messagebox.showinfo("Résumé statistique", summary)

    def open_logs(self):
        if os.path.exists(LOG_CSV):
            os.startfile(os.path.realpath(LOG_CSV))
        else:
            messagebox.showinfo("Info", "Aucun log présent encore.")

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionApp(root)
    root.mainloop()
