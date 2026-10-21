import cv2


video_path = 'xxx/recording.mp4'
output_path = 'xxx/step_0.png'

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Cannot open video file")
    exit()

# Read the first frame
success, frame = cap.read()

if success:
    cv2.imwrite(output_path, frame)
    print(f"The first frame has been saved as {output_path}")
else:
    print("Cannot read video frame")

cap.release()