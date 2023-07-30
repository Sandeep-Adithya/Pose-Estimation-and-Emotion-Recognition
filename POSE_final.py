import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

# Load the pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Define the emotion labels
emotion_labels = {
    "Happy": 0,
    "Sad": 0,
    "Angry": 0,
    "Neutral": 0
}

# Define the thresholds for emotion detection
happy_threshold = 160
sad_threshold = 120
angry_threshold = 30

# Function to detect emotion based on body posture
def detect_emotion(landmarks):
    # Get the coordinates of relevant landmarks
    left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_elbow = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
    right_elbow = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
    left_wrist = landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

    # Calculate the angle between shoulders and wrists
    left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

    # Detect emotions based on the angle
    if left_angle > happy_threshold and right_angle > happy_threshold:
        emotion_labels["Happy"] += 1
    elif left_angle < sad_threshold and right_angle < sad_threshold:
        emotion_labels["Sad"] += 1
    elif left_angle < angry_threshold or right_angle < angry_threshold:
        emotion_labels["Angry"] += 1
    else:
        emotion_labels["Neutral"] += 1

# Function to calculate angle between three landmarks
def calculate_angle(a, b, c):
    vector1 = [a.x - b.x, a.y - b.y]
    vector2 = [c.x - b.x, c.y - b.y]
    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
    magnitude1 = (vector1[0] ** 2 + vector1[1] ** 2) ** 0.5
    magnitude2 = (vector2[0] ** 2 + vector2[1] ** 2) ** 0.5
    cosine_angle = dot_product / (magnitude1 * magnitude2)
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

# Load the video capture
cap = cv2.VideoCapture(0)

# Initialize the plot
fig, ax = plt.subplots()
emotions = list(emotion_labels.keys())
percentage_bar = ax.bar(emotions, [0] * len(emotion_labels))

# Update the plot with the latest emotion statistics
def update_plot():
    total_count = sum(emotion_labels.values())
    percentages = [(count / total_count) * 100 for count in emotion_labels.values()]
    for bar, percentage in zip(percentage_bar, percentages):
        bar.set_height(percentage)
    plt.draw()

# Update the emotion statistics and plot for each frame
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Set flag to enable drawing landmarks on the image
    drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)

    # Detect pose landmarks
    results = pose.process(image)

    # Check if landmarks are detected
    if results.pose_landmarks:
        # Get emotion from body posture
        detect_emotion(results.pose_landmarks)
        update_plot()

        total_count = sum(emotion_labels.values())
        print("Emotion Recognition:")
        for emotion, count in emotion_labels.items():
            percentage = (count / total_count) * 100
            print(f"{emotion}: {percentage:.2f}%")

    # Draw pose landmarks on the image
    mp.solutions.drawing_utils.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=drawing_spec,
        connection_drawing_spec=drawing_spec)

    # Convert the RGB image back to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Show the image
    cv2.imshow('POSE', image)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Print the final emotion statistics
total_count = sum(emotion_labels.values())
print("Emotion Recognition:")
for emotion, count in emotion_labels.items():
    percentage = (count / total_count) * 100
    print(f"{emotion}: {percentage:.2f}%")

    

# Release the video capture and close the windows
cap.release()
cv2.destroyAllWindows()
