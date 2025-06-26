# app.py
from flask import Flask, render_template_string, Response
import cv2
import torch
import timm
import numpy as np
import mediapipe as mp
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

app = Flask(__name__)

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
CLASS_NAMES = [chr(i) for i in range(65, 91)]  # A-Z
last_prediction = "Nill"

# --- Load Trained ViT Model ---
model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=26)
model.load_state_dict(torch.load('resources/vit_epoch_5.pth', map_location=DEVICE))
model.eval().to(DEVICE)

# --- Transform ---
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# --- MediaPipe Hands Setup ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)


def generate_frames():
    global last_prediction
    while True:
        success, frame = cap.read()
        if not success:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        current_predictions = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]
                xmin = int(min(x_coords) * w) - 20
                ymin = int(min(y_coords) * h) - 20
                xmax = int(max(x_coords) * w) + 20
                ymax = int(max(y_coords) * h) + 20
                xmin, ymin = max(0, xmin), max(0, ymin)
                xmax, ymax = min(w, xmax), min(h, ymax)

                roi = frame[ymin:ymax, xmin:xmax]
                if roi.size == 0:
                    continue

                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                pil_img = Image.fromarray(gray_roi)
                input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    out = model(input_tensor)
                    probs = F.softmax(out, dim=1)
                    conf, pred = torch.max(probs, 1)
                    letter = CLASS_NAMES[pred.item()]
                    confidence = conf.item()

                label = f"{letter} ({confidence * 100:.1f}%)"
                current_predictions.append(letter)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(frame, label, (xmin, ymin - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if current_predictions:
            last_prediction = ', '.join(current_predictions)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    with open("templetes/index.html") as f:
        return render_template_string(f.read())


@app.route('/start')
def start():
    with open("templetes/start.html") as f:
        return render_template_string(f.read())


@app.route('/get_prediction')
def get_prediction():
    return last_prediction


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)