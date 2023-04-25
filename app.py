import cv2
from flask import Flask, render_template, Response, request

import line_detection as ld
from datsetTraining.webcamTest import objectDetection as od

app = Flask(__name__)

# Define the function that will capture frames from the camera
def gen_frames():
    camera = cv2.VideoCapture(0)
    width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            processed_frame = ld.process_frame(frame)
            
            # Add identified object text to the processed frame
            identified_object_text = od(camera)
            cv2.putText(processed_frame, identified_object_text, (10, height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            ret, buffer = cv2.imencode('.jpg', processed_frame)
            processed_frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + processed_frame + b'\r\n')

# Define the route for the home page
@app.route('/')
def index():
    return render_template('index.html')
# Define the route for the video feed
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)