# Squat Counter with Pose Estimation

This project is about squats counting and analysis in videos using YOLO for person detection and MediaPipe for pose estimation. It overlays real-time feedback, skeleton visualization, and performance charts on the output video.

## Features
- Person detection using YOLO
- Pose estimation using MediaPipe
- Real-time squat counting and angle analysis
- Skeleton-only visualization
- Performance chart overlay

## Requirements
- Python 3.8+
- See `requirements.txt` for dependencies

## Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/AhmedEH28/Pose-Estimation-and-Analysis-for-Squat.git
   cd Pose-Estimation-and-Analysis-for-Squat
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the YOLO model weights (e.g., `yolo11n.pt`) and place them in the project directory.
4. Place your input video (e.g., `squat1.mp4`) in the project directory.

## Usage
Run the script from the command line:
```bash
python pose_main.py
```


## Example Output

![Sample Output Image](image.png)

[![Result Video](image.png)](results/pose.mp4)

[Watch the result video](results/pose.mp4) 


## Notes
- For best results, use clear videos with visible full-body squats.
- The code is structured for easy extension and integration.

## License
MIT License 

## Acknowledgments
- Thanks to [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) and [MediaPipe](https://github.com/google/mediapipe) 