import cv2
from flask import Flask, render_template, Response
from ultralytics import YOLO

# Initializes your web app
app = Flask(__name__)

# Load YOLO Model
model = YOLO("yolov8n-seg.pt")

# Start Webcam
cap = cv2.VideoCapture(0)

# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Video Capture Setup
# fourcc = cv2.VideoWriter_fourcc(*'XVID') # compression format
# out = cv2.VideoWriter('output.avi',fourcc,15.0,(frame_width,frame_height))

def generate_frames():
    # Loop
    while True:
        success, frame = cap.read()

        if success:

            result = model.predict(frame,conf=0.3) # Minimum confidence 0.3
            annotated_frame = result[0].plot(conf=False) # Remove confidence label

            # Total Objects Count
            count = len(result[0].boxes)
            cv2.putText(annotated_frame,f"Count: {count}",(12,37),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0, 255, 255),2)

            # Write each frame and save the video
            # out.write(annotated_frame)

            # Encode frames as JPEG
            ret, buffer = cv2.imencode('.jpg',annotated_frame)
            frame = buffer.tobytes()

            # Stream frame (This sends frames continuously to browser)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route("/")
def index():
    # Loads HTML files from templates/ folder
    return render_template('index.html')

@app.route("/video")
def video():
    # Sends data (here: video frames) to browser
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)