from flask import Flask, render_template, Response
import torch
import torchvision
import cv2
import argparse
import time
import numpy as np
from PIL import Image

app = Flask(__name__)

#camera = cv2.VideoCapture('rtsp://freja.hiof.no:1935/rtplive/_definst_/hessdalen03.stream')  # use 0 for web camera
#  for cctv camera use rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' instead of camera
# for local webcam use cv2.VideoCapture(0)

from jetcam.csi_camera import CSICamera
fps_origin = 30

camera = CSICamera(width=400, height=225, capture_width=1280, capture_height=720, capture_fps=fps_origin)
names = "/nvdli-nano/data/action_recognition_kinetics.txt"

parser = argparse.ArgumentParser()

parser.add_argument('-f', '--file', required=False)
parser.add_argument('-l', '--clip-len', dest='clip_len', default=16, type=int,
                    help='number of frames to consider for each prediction')
parser.add_argument('-c', '--classes', default = names, help='Path to classes list.')

args = vars(parser.parse_args())
#### PRINT INFO #####
print(f"Number of frames to consider for each prediction: {args['clip_len']}")

# get the lables
class_names = open(args["classes"]).read().strip().split("\n")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device is {device}")

# load the model
model = torchvision.models.video.r2plus1d_18(pretrained=True, progress=True) #This one works.

# load the model onto the computation device
model = model.eval().to(device)

frame_count = 0 # to count total frames
total_fps = 0 # to get the final frames per second
# a clips list to append and store the individual frames
clips = []
frame_and_metadatas = []

import torchvision.transforms as T
# define the transforms
# This cell may need to be ran twice, ignore the first run error.
transform = T.Compose([
    T.Resize((128, 171)),
    T.CenterCrop((112, 112)),
    T.ToTensor(),
    T.Normalize(mean = [0.43216, 0.394666, 0.37645],
                std = [0.22803, 0.22145, 0.216989])
])

def gen_frames():  # generate frame by frame from camera
    processed=0
    while(True):
        # capture each frame of the video
        frame = camera.read()
        # get the start time
        start_time = time.time()
        image = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = transform(frame).numpy()
        clips.append(frame)
        if len(clips) == args['clip_len']:
            with torch.no_grad(): # we do not want to backprop any gradients
                input_frames = np.array(clips)
                # add an extra dimension        
                input_frames = np.expand_dims(input_frames, axis=0)
                # transpose to get [1, 3, num_clips, height, width]
                input_frames = np.transpose(input_frames, (0, 2, 1, 3, 4))
                # convert the frames to tensor
                input_frames = torch.tensor(input_frames, dtype=torch.float32)
                input_frames = input_frames.to(device)
                # forward pass to get the predictions
                outputs = model(input_frames)
                # get the prediction index
                _, preds = torch.max(outputs.data, 1)

                # map predictions to the respective class names
                label = class_names[preds].strip()
            # get the end time
            end_time = time.time()
            # get the fps
            fps = 1 / (end_time - start_time)
            # add fps to total fps
            total_fps += fps
            # increment frame count
            frame_count += 1
            wait_time = max(1, int(fps/4))
            label_with_time_stamp = label + f"  frame #: {frame_count}, time: {(cap.get(cv2.CAP_PROP_POS_MSEC))/1000:.2f}s"
            cv2.putText(image, label_with_time_stamp, (15, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2, 
                        lineType=cv2.LINE_AA)
            clips.pop(0)
            if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                break
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')  # concat frame one by one and show result


        
        
@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)