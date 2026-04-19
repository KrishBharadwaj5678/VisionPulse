import cv2
from flask import Flask, render_template, Response, request
from ultralytics import YOLO

# Initializes your web app
app = Flask(__name__)

# Load YOLO Model
model = YOLO("yolov8n-seg.pt")

# Start Webcam
cap = cv2.VideoCapture(0)

selected_object = "all"

def generate_frames():
    # Loop
    while True:
        success, frame = cap.read()

        if success:

            result = model.predict(frame,conf=0.3) # Minimum confidence 0.3

            annotated_frame = frame.copy()
            count = 0
            boxes = result[0].boxes

            if boxes is not None:
                filtered_indices = [] 
                for i, box in enumerate(boxes):
                    class_id = int(box.cls[0])
                    name = model.names[class_id]

                    if selected_object == "all" or name == selected_object:
                        filtered_indices.append(i)
                        count += 1

                if len(filtered_indices) > 0:
                    filtered_result = result[0][filtered_indices]
                    annotated_frame = filtered_result.plot(conf=False)
                else:
                    annotated_frame = frame.copy()

            cv2.putText(annotated_frame,f"Count: {count}",(10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2)

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

@app.route("/set_filter")
def set_filter():
    global selected_object
    # Get the 'object' value, if not found use 'all'
    selected_object = request.args.get("object","all")
    return "OK"

if __name__ == "__main__":
    app.run(debug=True)