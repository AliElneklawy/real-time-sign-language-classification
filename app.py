import os
import pickle
import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
from dotenv import load_dotenv

app = Flask(__name__)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Load the trained model
model_path = os.path.join('models', 'model.pkl')
try:
    model = pickle.load(open(model_path, 'rb'))
    print("Model loaded successfully!")
except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}")
    model = None

# Labels dictionary
labels_dict = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F'}

# Global variables for hand tracking
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.5
)

def process_frame(frame):
    """Process a single frame for hand detection and classification."""
    if frame is None:
        return None, None, None, None
    
    H, W = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    predicted_char = None
    confidence = 0.0
    bbox = None
    
    if results.multi_hand_landmarks and model is not None:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # Prepare data for prediction
            data_aux = []
            x_ = []
            y_ = []
            
            for landmark in hand_landmarks.landmark:
                x = landmark.x
                y = landmark.y
                x_.append(x)
                y_.append(y)
            
            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x - min(x_))
                data_aux.append(landmark.y - min(y_))
            
            # Calculate bounding box
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) + 10
            y2 = int(max(y_) * H) + 10
            bbox = (x1, y1, x2, y2)
            
            # Make prediction
            try:
                if len(data_aux) == 42:  # 21 landmarks * 2 (x,y)
                    prediction = model.predict_proba([data_aux])
                    predicted_idx = np.argmax(prediction[0]) + 1  # +1 because labels start at 1
                    confidence = float(np.max(prediction[0]))
                    predicted_char = labels_dict.get(predicted_idx, '?')
                    
                    # Draw prediction on frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{predicted_char} ({confidence:.2f})", 
                              (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.9, (0, 255, 0), 2)
            except Exception as e:
                print(f"Prediction error: {e}")
    
    return frame, predicted_char, confidence, bbox

def gen_frames():
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
            
        # Process the frame
        processed_frame, _, _, _ = process_frame(frame)
        
        if processed_frame is not None:
            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    camera.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/classify', methods=['POST'])
def classify():
    if 'image' not in request.files:
        return jsonify({'status': 'error', 'message': 'No image provided'}), 400
    
    try:
        # Read and process the image
        file = request.files['image']
        nparr = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'status': 'error', 'message': 'Failed to decode image'}), 400
        
        # Process the frame
        _, predicted_char, confidence, _ = process_frame(frame)
        
        if predicted_char is None:
            return jsonify({
                'status': 'success',
                'prediction': None,
                'confidence': 0.0,
                'message': 'No hands detected'
            })
        
        return jsonify({
            'status': 'success',
            'prediction': predicted_char,
            'confidence': float(confidence)
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    load_dotenv()
    app.run(debug=True, port=5000, host='0.0.0.0')
