import os
import cv2


DATA_DIR = './images'
num_classes = 26
imgs_per_class = 10
cap = cv2.VideoCapture(0)

if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)

for i in range(num_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(i + 1))): 
        os.mkdir(os.path.join(DATA_DIR, str(i + 1)))
        
        print(f'Collecting data for class {i + 1}')
        
        while True: # wait until the user presses 's'
            ret, frame = cap.read()
            cv2.putText(frame, 'Press "s" to start.', (100, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, 
                        (0, 255, 0), 3, cv2.LINE_AA)
            cv2.imshow('Frame', frame)
            if cv2.waitKey(25) == ord('s'): 
                break
        
        counter = 1
        while counter <= imgs_per_class:
            ret, frame = cap.read()
            cv2.imshow('Frame', frame)
            cv2.waitKey(500)
            cv2.imwrite(os.path.join(DATA_DIR, str(i + 1), '{}.jpg'.format(counter)), frame)
            counter += 1

cap.release()
cv2.destroyAllWindows()
