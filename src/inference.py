import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the trained model
try:
    model = pickle.load(open('models/model.pkl', 'rb'))
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Error: Model file not found. Make sure 'models/model.pkl' exists.")
    exit()

# Initialize video capture
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

# Set camera resolution (optional, can help with stability)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Fixed MediaPipe Hands initialization
# static_image_mode should be False for video processing
hands = mp_hands.Hands(
    static_image_mode=False,  # Changed from True to False
    max_num_hands=2,          # Specify max number of hands
    min_detection_confidence=0.3,
    min_tracking_confidence=0.5
)

labels_dict = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F'}

print("Starting camera... Press 'q' to quit")

while True:
    try:
        data_aux = []
        x_ = []
        y_ = []

        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame")
            break

        H, W, _ = frame.shape

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame
        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    frame,  # image to draw
                    hand_landmarks,  # model output
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            # Extract features for prediction
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            # Calculate bounding box
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            # Make prediction
            try:
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]

                # Draw bounding box and prediction
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
            except Exception as e:
                print(f"Prediction error: {e}")

        # Display the frame
        cv2.imshow('Sign Language Recognition', frame)
        
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    except KeyboardInterrupt:
        print("Interrupted by user")
        break
    except Exception as e:
        print(f"An error occurred: {e}")
        break

# Cleanup
print("Cleaning up...")
cap.release()
cv2.destroyAllWindows()
hands.close()
print("Done!")