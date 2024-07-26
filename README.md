# adas_experiments

## Getting Started
### Installing Ultralytics
 * Follow the steps given in the web page below
   * https://docs.ultralytics.com/quickstart/#install-ultralytics
 * Do not miss the customized installation command for PyTorch in the same page. For my case (Raspberry Pi 4) I got the following command based on my selection:
   * pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

**Note**: create virtual env ~/.myenv and install ultraytics

<br>


### Running Ultralytics on Raspberry Pi

Refer the following link
 * https://dagshub.com/Ultralytics/ultralytics/pulls/5227/files?file_diff=true&path=

#### Start camera
`libcamera-vid -n -t 0 --width 640 --height 480 --framerate 15 --inline --listen -o tcp://127.0.0.1:8888`

#### Start Yolov8
 * `source ~/.myenv/bin/activate`
 * `yolo predict model=yolov8n.pt source=tcp://127.0.0.1:8888 show=true`

#### A better way to run
Just run the run-yolo.sh file using command `ultralytics/run-yolo.sh`
