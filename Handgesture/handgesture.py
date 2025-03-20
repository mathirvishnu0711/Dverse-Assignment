import cv2
import mediapipe as mp

# Initialize MediaPipe Hand Tracking
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, 
                        min_detection_confidence=0.6, min_tracking_confidence=0.6)

# Open Webcam
cap = cv2.VideoCapture(0)

# Define black color for fingertip markers
color_fingertips = (0, 0, 0)  # Black

# Define the landmark indices of fingertips
fingertip_indices = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky

def is_finger_raised(hand_landmarks, index):
    """ Determines if a finger is raised based on landmark positions """
    if index == 4:  # Thumb (special case)
        return hand_landmarks.landmark[index].x < hand_landmarks.landmark[index - 1].x  # Thumb extended outward
    return hand_landmarks.landmark[index].y < hand_landmarks.landmark[index - 2].y  # Fingertip is above PIP joint

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror effect for natural interaction
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            h, w, _ = frame.shape

            for index in fingertip_indices:
                if is_finger_raised(hand_landmarks, index):  # Mark only raised fingers
                    cx, cy = int(hand_landmarks.landmark[index].x * w), int(hand_landmarks.landmark[index].y * h)

                    # Draw filled black circle
                    cv2.circle(frame, (cx, cy), 10, color_fingertips, -1, lineType=cv2.LINE_AA)

                    # Text with a white shadow effect for better visibility
                    label = ["Thumb", "Index", "Middle", "Ring", "Pinky"][fingertip_indices.index(index)]
                    text_position = (cx + 15, cy - 10)
                    cv2.putText(frame, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, 
                                0.7, (255, 255, 255), 3, lineType=cv2.LINE_AA)  # White shadow
                    cv2.putText(frame, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, 
                                0.7, color_fingertips, 2, lineType=cv2.LINE_AA)  # Black text

    else:
        # Display a message if no hands are detected
        cv2.putText(frame, "No hand detected", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, (0, 0, 255), 2, lineType=cv2.LINE_AA)

    # Show frame
    cv2.imshow("Hand Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
