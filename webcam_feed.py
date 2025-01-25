import cv2
import mediapipe as mp
import numpy as np
import random

# Initialize MediaPipe Hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Game settings
grid_size = (20, 20)  # 20x20 grid
cell_size = 30  # Size of each grid cell in pixels
snake = [(10, 10)]  # Starting position of the snake
direction = (0, 1)  # Start moving right
food = (random.randint(0, grid_size[0] - 1), random.randint(0, grid_size[1] - 1))  # Random food placement
score = 0
frame_width = grid_size[0] * cell_size
frame_height = grid_size[1] * cell_size

# Initialize the webcam capture
cap = cv2.VideoCapture(0)

# Draw the snake, food, and grid on the screen
def draw_game(frame, snake, food):
    # Draw grid
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            cv2.rectangle(frame, (i * cell_size, j * cell_size),
                          ((i + 1) * cell_size, (j + 1) * cell_size), (50, 50, 50), 1)
    # Draw food
    cv2.rectangle(frame, (food[0] * cell_size, food[1] * cell_size),
                  ((food[0] + 1) * cell_size, (food[1] + 1) * cell_size), (0, 0, 255), -1)
    # Draw snake
    for segment in snake:
        cv2.rectangle(frame, (segment[0] * cell_size, segment[1] * cell_size),
                      ((segment[0] + 1) * cell_size, (segment[1] + 1) * cell_size), (0, 255, 0), -1)
    # Display score
    cv2.putText(frame, f"Score: {score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

while True:
    # Capture frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip horizontally for natural interaction
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Hand detection
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the x, y coordinates of the index finger tip
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x, y = int(index_finger_tip.x * frame.shape[1]), int(index_finger_tip.y * frame.shape[0])

            # Map hand position to grid direction
            frame_center_x = frame.shape[1] // 2
            frame_center_y = frame.shape[0] // 2
            threshold = 50  # Sensitivity threshold for direction changes
            if x < frame_center_x - threshold:
                direction = (-1, 0)  # Move left
            elif x > frame_center_x + threshold:
                direction = (1, 0)  # Move right
            elif y < frame_center_y - threshold:
                direction = (0, -1)  # Move up
            elif y > frame_center_y + threshold:
                direction = (0, 1)  # Move down

    # Update snake position
    new_head = (snake[-1][0] + direction[0], snake[-1][1] + direction[1])

    # Check for collision with food
    if new_head == food:
        score += 1
        food = (random.randint(0, grid_size[0] - 1), random.randint(0, grid_size[1] - 1))  # Respawn food
    else:
        snake.pop(0)  # Remove tail if no food eaten

    # Add new head to the snake
    snake.append(new_head)

    # Check for collisions with walls or self
    if (new_head[0] < 0 or new_head[0] >= grid_size[0] or
            new_head[1] < 0 or new_head[1] >= grid_size[1] or
            new_head in snake[:-1]):
        break  # Game over

    # Draw the game
    game_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    draw_game(game_frame, snake, food)

    # Overlay the game frame on webcam feed
    frame[:frame_height, :frame_width] = game_frame
    cv2.imshow("Snake Game", frame)

    # Quit the game if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()