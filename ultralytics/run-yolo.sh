#!/bin/bash

# Default values
DEFAULT_WIDTH=640
DEFAULT_HEIGHT=480

# Validate and set width and height from arguments
WIDTH=$DEFAULT_WIDTH
HEIGHT=$DEFAULT_HEIGHT

if [ "$#" -eq 2 ]; then
    # Validate width
    if [[ $1 =~ ^[0-9]+$ ]] && [ $1 -ge 640 ] && [ $1 -le 1920 ]; then
        WIDTH=$1
    else
        echo "Error: Width must be a number between 640 and 1920."
        exit 1
    fi

    # Validate height
    if [[ $2 =~ ^[0-9]+$ ]] && [ $2 -ge 480 ] && [ $2 -le 1080 ]; then
        HEIGHT=$2
    else
        echo "Error: Height must be a number between 480 and 1080."
        exit 1
    fi
elif [ "$#" -gt 0 ]; then
    echo "Usage: $0 [width height]"
    echo "  width: Number between 640 and 1920 (default: 640)"
    echo "  height: Number between 480 and 1080 (default: 480)"
    exit 1
fi

echo "Using width: $WIDTH and height: $HEIGHT"

# Ensure the path for the YOLO model exists
MODEL_DIR=~/projects/adas
MODEL_PATH="$MODEL_DIR/yolov8n.pt"

if [ ! -d "$MODEL_DIR" ]; then
    echo "Directory $MODEL_DIR does not exist. Creating it..."
    mkdir -p "$MODEL_DIR"
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "Model file $MODEL_PATH does not exist. Please ensure the file is present."
    exit 1
fi

# Function to clean up background processes
cleanup() {
    echo "Stopping libcamera-vid..."
    pkill -f "libcamera-vid"
}

# Trap script exit to clean up background processes
trap cleanup EXIT

# Start libcamera-vid in the background
libcamera-vid -n -t 0 --width $WIDTH --height $HEIGHT --framerate 15 --inline --listen -o tcp://127.0.0.1:8888 &

# Save the PID of libcamera-vid for cleanup
LIBCAMERA_PID=$!

# Wait for a few seconds to ensure libcamera-vid starts properly
sleep 1

# Activate the Python virtual environment
source ~/.myenv/bin/activate

# Run YOLOv8
yolo predict model=$MODEL_PATH source=tcp://127.0.0.1:8888 show=true

# Ensure the cleanup function is called on script exit
trap cleanup EXIT

# Wait for YOLOv8 to finish before stopping libcamera-vid
wait $LIBCAMERA_PID
