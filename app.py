from flask import Flask, render_template, Response
import torch
import torchvision
import cv2
import argparse
import time
import resnet
import numpy as np
import datetime
import time
import os
from PIL import Image

app = Flask(__name__)


from jetcam.csi_camera import CSICamera
fps_origin = 1

#camera = CSICamera(width=128, height=171, , capture_fps=fps_origin)
camera = CSICamera(width=1280, height=720, capture_width=1280, capture_height=720, capture_fps=fps_origin)


parser = argparse.ArgumentParser()

parser.add_argument('-f', '--file', required=False)
parser.add_argument('-l', '--clip-len', dest='clip_len', default=16, type=int,
                    help='number of frames to consider for each prediction')

args = vars(parser.parse_args())
#### PRINT INFO #####
print(f"Number of frames to consider for each prediction: {args['clip_len']}")


# Get the labels
class_names = open("trans_8.txt").read().strip().split("\n")

#transfered_class_list = ["ApplyLipstick", "Crawling", "BrushingTeeth", "HeadMassage", "HighJump", "Punch", "PushUps", "WritingOnBoard"]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device is {device}")

# load the model
model = resnet.generate_model(model_depth=18,
                                      n_classes=8,
                                      n_input_channels=3,
                                      shortcut_type="B",
                                        conv1_t_size=7,
                                        conv1_t_stride=1,
                                        no_max_pool=False,
                                        widen_factor=1.0)
model.load_state_dict(torch.load("/nvdli-nano/data/r3d_transfered_8.pth")["state_dict"])
# load the model onto the computation device
model = model.eval().to(device)

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
    T.Normalize(mean = [0.4345, 0.4051, 0.3775],
                std = [0.2768, 0.2713, 0.2737])
])

action = ['ApplyLipstick', 'Crawling', 'BrushingTeeth', 'HeadMassage', 'HighJump', 'Punch', 'PushUps', 'WritingOnBoard']

exercise_dict = {
  "ApplyLipstick": "No",
  "Crawling": "No",
  "BrushingTeeth": "No",
  "HeadMassage": "No",
  "HighJump": "Yes",
  "Punch": "Yes",
  "PushUps": "Yes",
  "WritingOnBoard": "No"
}

def gen_frames():  # generate frame by frame from camera
    processed=0
    frame_count = 0 # to count total frames
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
                with torch.cuda.amp.autocast():
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
            # increment frame count
            frame_count += 1
            wait_time = max(1, int(fps/4))
            label_with_time_stamp = label + f"  frame #: {frame_count}, fps: {fps}"
            cv2.putText(image, label_with_time_stamp, (15, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2, 
                        lineType=cv2.LINE_AA)
            clips.pop(0)
            if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                break
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            ct = datetime.datetime.now()
            command_str = "curl -XPUT -u \'admin:Admin123$\' \'https://search-visa-visa-cmn6kne72ob4eumf4k6xw52rgq.us-east-2.es.amazonaws.com/actions/_doc/" + str(ct)[-6::] + "\' -d '{\"name\": \"" + label + "\", \"date\": \"" + str(ct)[0:10] + "\", \"time_stamp\": \"" + str(ct)[0:19] + "\", \"exercise\": \"" + exercise_dict[label] + "\"}\' -H \'Content-Type: application/json\'"
            os.system(command_str)
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