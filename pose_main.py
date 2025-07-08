import argparse
import logging
import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import os
import matplotlib.pyplot as plt
import matplotlib
from typing import Tuple, List, Optional

matplotlib.use("Agg")  # Use 'Agg' backend for non-interactive plotting

# Try to import MediaPipe as fallback
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("MediaPipe not available. Install with: pip install mediapipe")

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

class SquatCounter:
    def __init__(self, model_path: str = "yolo11n.pt"):
        """
        Initialize the SquatCounter with YOLO model and MediaPipe pose estimator.
        """
        self.model = YOLO(model_path)
        if MEDIAPIPE_AVAILABLE:
            logging.info("Using MediaPipe for pose estimation...")
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=2,
                enable_segmentation=False,
                min_detection_confidence=0.3,
                min_tracking_confidence=0.3
            )
        else:
            raise RuntimeError("MediaPipe is not available. Please install it.")
        self.BODY_PARTS = {
            "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
            "LShoulder": 5, "LElbow": 6, "LWrist": 7, "MidHip": 8, "RHip": 9,
            "RKnee": 10, "RAnkle": 11, "LHip": 12, "LKnee": 13, "LAnkle": 14,
        }
        self.POSE_PAIRS = [
            ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
            ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
            ["Neck", "MidHip"], ["MidHip", "RHip"], ["MidHip", "LHip"],
            ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["LHip", "LKnee"], ["LKnee", "LAnkle"]
        ]
        self.counter = 0
        self.stage = None
        self.angle_buffer = deque(maxlen=7)
        self.smoothed_angle = 0
        self.landmark_history = {}
        self.history_size = 5
        self.down_threshold = 110
        self.up_threshold = 135
        self.min_squat_duration = 3
        self.tracked_persons = {}
        self.person_id_counter = 0
        self.stage_duration = 0
        self.last_stage = None
        self.angle_history = deque(maxlen=200)
        self.frame_count = 0

    def detect_pose_mediapipe(self, frame, bbox):
        """Detect pose using MediaPipe within bounding box"""
        x1, y1, x2, y2 = bbox
        cropped = frame[y1:y2, x1:x2]
        
        if cropped.shape[0] == 0 or cropped.shape[1] == 0:
            return None
        
        # Convert to RGB
        rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        
        if results.pose_landmarks:
            # Convert MediaPipe landmarks to OpenPose format
            keypoints = []
            landmarks = results.pose_landmarks.landmark
            crop_h, crop_w = cropped.shape[:2]
            
            # Map MediaPipe landmarks to OpenPose-like format
            mp_to_op_map = {
                0: mp.solutions.pose.PoseLandmark.NOSE,           # Nose
                1: None,                                           # Neck (calculated)
                2: mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER, # RShoulder
                3: mp.solutions.pose.PoseLandmark.RIGHT_ELBOW,    # RElbow
                4: mp.solutions.pose.PoseLandmark.RIGHT_WRIST,    # RWrist
                5: mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,  # LShoulder
                6: mp.solutions.pose.PoseLandmark.LEFT_ELBOW,     # LElbow
                7: mp.solutions.pose.PoseLandmark.LEFT_WRIST,     # LWrist
                8: None,                                           # MidHip (calculated)
                9: mp.solutions.pose.PoseLandmark.RIGHT_HIP,      # RHip
                10: mp.solutions.pose.PoseLandmark.RIGHT_KNEE,    # RKnee
                11: mp.solutions.pose.PoseLandmark.RIGHT_ANKLE,   # RAnkle
                12: mp.solutions.pose.PoseLandmark.LEFT_HIP,      # LHip
                13: mp.solutions.pose.PoseLandmark.LEFT_KNEE,     # LKnee
                14: mp.solutions.pose.PoseLandmark.LEFT_ANKLE,    # LAnkle
            }
            
            # Convert landmarks
            for i in range(15):  # Only use first 15 keypoints for compatibility
                if i in mp_to_op_map and mp_to_op_map[i] is not None:
                    lm = landmarks[mp_to_op_map[i]]
                    x = int(lm.x * crop_w) + x1
                    y = int(lm.y * crop_h) + y1
                    keypoints.append((x, y, lm.visibility))
                elif i == 1:  # Neck - calculate from shoulders
                    try:
                        l_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
                        r_shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
                        x = int(((l_shoulder.x + r_shoulder.x) / 2) * crop_w) + x1
                        y = int(((l_shoulder.y + r_shoulder.y) / 2) * crop_h) + y1
                        conf = (l_shoulder.visibility + r_shoulder.visibility) / 2
                        keypoints.append((x, y, conf))
                    except:
                        keypoints.append(None)
                elif i == 8:  # MidHip - calculate from hips
                    try:
                        l_hip = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP]
                        r_hip = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP]
                        x = int(((l_hip.x + r_hip.x) / 2) * crop_w) + x1
                        y = int(((l_hip.y + r_hip.y) / 2) * crop_h) + y1
                        conf = (l_hip.visibility + r_hip.visibility) / 2
                        keypoints.append((x, y, conf))
                    except:
                        keypoints.append(None)
                else:
                    keypoints.append(None)
            
            return keypoints
        
        return None

    def detect_pose(self, frame, bbox):
        """Detect pose using MediaPipe"""
        return self.detect_pose_mediapipe(frame, bbox)

    def calculate_angle(self, a, b, c):
        """Calculate angle between three points"""
        if None in [a, b, c]:
            return None
        
        a, b, c = np.array(a[:2]), np.array(b[:2]), np.array(c[:2])
        ba, bc = a - b, c - b
        
        # Handle zero vectors
        norm_ba, norm_bc = np.linalg.norm(ba), np.linalg.norm(bc)
        if norm_ba == 0 or norm_bc == 0:
            return None
            
        cosine = np.dot(ba, bc) / (norm_ba * norm_bc)
        cosine = np.clip(cosine, -1.0, 1.0)
        return np.degrees(np.arccos(cosine))

    def smooth_landmarks(self, keypoints, person_id):
        """Smooth landmarks using exponential moving average"""
        if person_id not in self.landmark_history:
            self.landmark_history[person_id] = {}
        
        smoothed_keypoints = []
        for i, keypoint in enumerate(keypoints):
            if keypoint is None:
                smoothed_keypoints.append(None)
                continue
                
            x, y, conf = keypoint
            if i not in self.landmark_history[person_id]:
                self.landmark_history[person_id][i] = deque(maxlen=self.history_size)
            
            self.landmark_history[person_id][i].append((x, y, conf))
            
            if len(self.landmark_history[person_id][i]) > 1:
                positions = list(self.landmark_history[person_id][i])
                weights = np.exp(np.linspace(-2, 0, len(positions)))
                weights /= weights.sum()
                
                smooth_x = sum(p[0] * w for p, w in zip(positions, weights))
                smooth_y = sum(p[1] * w for p, w in zip(positions, weights))
                smooth_conf = sum(p[2] * w for p, w in zip(positions, weights))
                
                smoothed_keypoints.append((int(smooth_x), int(smooth_y), smooth_conf))
            else:
                smoothed_keypoints.append((int(x), int(y), conf))
        
        return smoothed_keypoints

    def smooth_angle(self, angle):
        """Smooth angle using weighted moving average"""
        if angle is None or np.isnan(angle) or np.isinf(angle):
            return self.smoothed_angle
            
        self.angle_buffer.append(angle)
        
        if len(self.angle_buffer) >= 3:
            angles = list(self.angle_buffer)
            weights = np.exp(np.linspace(-1, 0, len(angles)))
            weights /= weights.sum()
            self.smoothed_angle = sum(a * w for a, w in zip(angles, weights))
        elif len(self.angle_buffer) > 0:
            self.smoothed_angle = self.angle_buffer[-1]
        
        return self.smoothed_angle

    def find_best_person_match(self, current_bbox, previous_persons):
        """Find best matching person from previous frame"""
        if not previous_persons:
            return None
            
        cx1, cy1, cx2, cy2 = current_bbox
        current_center = ((cx1 + cx2) / 2, (cy1 + cy2) / 2)
        current_area = (cx2 - cx1) * (cy2 - cy1)
        
        best_match, best_score = None, float("inf")
        
        for person_id, (prev_bbox, _) in previous_persons.items():
            px1, py1, px2, py2 = prev_bbox
            prev_center = ((px1 + px2) / 2, (py1 + py2) / 2)
            prev_area = (px2 - px1) * (py2 - py1)
            
            # Calculate distance and area similarity
            distance = np.sqrt((current_center[0] - prev_center[0]) ** 2 + 
                             (current_center[1] - prev_center[1]) ** 2)
            area_ratio = min(current_area, prev_area) / max(current_area, prev_area)
            
            score = distance * (2 - area_ratio)
            
            if score < best_score and score < 100:
                best_score = score
                best_match = person_id
        
        return best_match

    def calculate_squat_angle(self, keypoints):
        """Calculate squat angle using hip-knee-ankle points"""
        try:
            # Get required keypoints
            left_hip = keypoints[self.BODY_PARTS["LHip"]]
            left_knee = keypoints[self.BODY_PARTS["LKnee"]]
            left_ankle = keypoints[self.BODY_PARTS["LAnkle"]]
            
            right_hip = keypoints[self.BODY_PARTS["RHip"]]
            right_knee = keypoints[self.BODY_PARTS["RKnee"]]
            right_ankle = keypoints[self.BODY_PARTS["RAnkle"]]
            
            # Calculate angles for both legs
            left_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
            right_angle = self.calculate_angle(right_hip, right_knee, right_ankle)
            
            # Return average angle and knee positions
            if left_angle is not None and right_angle is not None:
                avg_angle = (left_angle + right_angle) / 2
                return avg_angle, left_knee, right_knee
            elif left_angle is not None:
                return left_angle, left_knee, None
            elif right_angle is not None:
                return right_angle, None, right_knee
            else:
                return None, None, None
                
        except Exception as e:
            print(f"Error calculating squat angle: {e}")
            return None, None, None

    def update_squat_count(self, angle):
        """Update squat count based on angle"""
        if angle is None or len(self.angle_buffer) < 3:
            return
            
        # Determine current stage
        if angle < self.down_threshold:
            current_stage = 'down'
        elif angle > self.up_threshold:
            current_stage = 'up'
        else:
            current_stage = 'transition'
        
        # Update stage duration
        if current_stage == self.last_stage:
            self.stage_duration += 1
        else:
            self.stage_duration = 1
            self.last_stage = current_stage
        
        # Count squats with minimum duration requirement
        if self.stage_duration >= self.min_squat_duration:
            if current_stage == 'down' and self.stage != 'down':
                self.stage = 'down'
            elif current_stage == 'up' and self.stage == 'down':
                self.stage = 'up'
                self.counter += 1
                self.stage = None

    def draw_feedback(self, frame, angle):
        """Draw feedback information on frame at the top-left corner in a rectangle."""
        h, w = frame.shape[:2]
        text_x = 10
        line_height = 18 # Adjusted line height
        text_y_start = 10  # Original position

        # Determine color and status based on stage
        if self.stage == 'down':
            color, status = (0, 255, 255), "DOWN"
        elif self.stage == 'up':
            color, status = (0, 255, 0), "UP"
        else:
            color, status = (200, 200, 200), "READY"

        font_scale, thickness = 0.4, 1 # Reduced font size

        # Prepare text lines (removed Range)
        lines = []
        if angle is not None:
            lines.append(f'Angle: {int(angle)}')  # Only number, no degree symbol
        lines.append(f'Squats: {self.counter}')
        lines.append(f'Status: {status}')

        # Calculate rectangle size
        max_text_width = 0
        for line in lines:
            (text_w, text_h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            max_text_width = max(max_text_width, text_w)
        rect_width = max_text_width + 20 # Add padding
        rect_height = len(lines) * line_height  # Remove extra bottom padding

        # Draw solid black background rectangle (no transparency)
        cv2.rectangle(frame, (text_x - 8, text_y_start - 8), (text_x + rect_width, text_y_start + rect_height), (0, 0, 0), -1)

        # Draw text information
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (text_x, text_y_start + i * line_height), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color if i < 3 else (200, 200, 200), thickness)

    def draw_skeleton(self, frame, keypoints, color=(0, 255, 0), thickness=2, small_keypoints=False):
        """Draw skeleton on frame"""
        # Draw connections
        for pair in self.POSE_PAIRS:
            part_a = pair[0]
            part_b = pair[1]
            
            if part_a in self.BODY_PARTS and part_b in self.BODY_PARTS:
                id_a = self.BODY_PARTS[part_a]
                id_b = self.BODY_PARTS[part_b]
                
                if keypoints[id_a] is not None and keypoints[id_b] is not None:
                    pt1 = (keypoints[id_a][0], keypoints[id_a][1])
                    pt2 = (keypoints[id_b][0], keypoints[id_b][1])
                    cv2.line(frame, pt1, pt2, color, thickness)
                    cv2.line(frame, pt1, pt2, (255, 255, 255), 1)
        
        # Draw keypoints
        for i, keypoint in enumerate(keypoints):
            if keypoint is not None:
                x, y, conf = keypoint
                # Color coding for important joints
                if i in [self.BODY_PARTS["LKnee"], self.BODY_PARTS["RKnee"]]:
                    joint_color = (0, 0, 255)  # Red for knees
                elif i in [self.BODY_PARTS["LHip"], self.BODY_PARTS["RHip"]]:
                    joint_color = (255, 0, 0)  # Blue for hips
                elif i in [self.BODY_PARTS["LAnkle"], self.BODY_PARTS["RAnkle"]]:
                    joint_color = (0, 255, 255)  # Yellow for ankles
                else:
                    joint_color = (255, 255, 255)  # White for other joints
                if small_keypoints:
                    radius = 3 if joint_color != (255, 255, 255) else 2
                else:
                    radius = 5 if joint_color != (255, 255, 255) else 4
                cv2.circle(frame, (x, y), radius, joint_color, -1)
                cv2.circle(frame, (x, y), radius + 1, (0, 0, 0), 1)

    def generate_skeleton_only_visualization(self, keypoints, original_frame_shape, scale_factor_w=0.28, scale_factor_h=0.38):
        """Generates a black background image with only the skeleton, scaled and positioned."""
        h_orig, w_orig, _ = original_frame_shape
        new_w, new_h = int(w_orig * scale_factor_w), int(h_orig * scale_factor_h)
        
        # Create a larger black background
        skeleton_frame = np.zeros((new_h, new_w, 3), dtype=np.uint8)
        
        # Scale keypoints to the new larger frame
        scaled_keypoints = []
        # Find min/max x,y for keypoints to normalize scaling
        min_x, max_x = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')
        for kp in keypoints:
            if kp is not None:
                min_x = min(min_x, kp[0])
                max_x = max(max_x, kp[0])
                min_y = min(min_y, kp[1])
                max_y = max(max_y, kp[1])

        # Calculate scaling factors for skeleton to fit within its new_w, new_h frame
        skel_width = max_x - min_x
        skel_height = max_y - min_y

        if skel_width == 0 or skel_height == 0:
            return skeleton_frame # Return empty if no valid skeleton

        scale_x = (new_w * 0.8) / skel_width # 80% of new_w to give padding
        scale_y = (new_h * 0.8) / skel_height # 80% of new_h to give padding
        scale = min(scale_x, scale_y)

        # Offset to center the skeleton within its new frame
        offset_x = (new_w - skel_width * scale) / 2 - min_x * scale
        offset_y = (new_h - skel_height * scale) / 2 - min_y * scale

        for kp in keypoints:
            if kp is not None:
                scaled_keypoints.append((int(kp[0] * scale + offset_x), int(kp[1] * scale + offset_y), kp[2]))
            else:
                scaled_keypoints.append(None)

        # Draw skeleton with smaller keypoints (radius=2 for important, 1 for others)
        self.draw_skeleton(skeleton_frame, scaled_keypoints, color=(0, 255, 0), thickness=2, small_keypoints=True)
        return skeleton_frame

    def generate_performance_chart(self, width, height):
        """Generates a real-time performance chart of squat angle."""
        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
        
        # Ensure x-axis starts from 0 and has correct range
        x_data = np.arange(len(self.angle_history))
        y_data = list(self.angle_history)

        ax.plot(x_data, y_data, color='blue', linewidth=1) # Reduced linewidth
        ax.axhline(y=self.down_threshold, color='r', linestyle='--', label='Down Threshold', linewidth=1)
        ax.axhline(y=self.up_threshold, color='g', linestyle='--', label='Up Threshold', linewidth=1)
        ax.fill_between(x_data, self.down_threshold, self.up_threshold, color='yellow', alpha=0.2, label='Safe Zone')
        
        ax.set_ylim(50, 180) # Restored to original y-axis range
        ax.set_title('Knee Angle Over Time', fontsize=8) # Reduced title font size
        ax.set_xlabel('Frame', fontsize=6) # Reduced label font size
        ax.set_ylabel('Angle (°)', fontsize=6) # Reduced label font size
        ax.tick_params(axis='both', which='major', labelsize=6) # Reduced tick label font size
        ax.legend(fontsize=6) # Reduced legend font size
        ax.grid(True)

        # Convert plot to image
        fig.canvas.draw()
        img_plot = np.array(fig.canvas.renderer._renderer)
        img_plot = cv2.cvtColor(img_plot, cv2.COLOR_RGB2BGR)
        plt.close(fig)
        return img_plot

    def process_frame(self, frame):
        """Process a single frame"""
        self.frame_count += 1
        # Detect persons using YOLO
        detections = self.model(frame, verbose=False)[0]
        current_persons = {}
        
        output_frame = frame.copy()
        skeleton_only_frame = None

        for box in detections.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box[:4])
            
            # Filter small detections
            if (x2 - x1) < 50 or (y2 - y1) < 100:
                continue
            
            # Match with previous persons
            person_id = self.find_best_person_match((x1, y1, x2, y2), self.tracked_persons)
            if person_id is None:
                person_id = self.person_id_counter
                self.person_id_counter += 1
            
            # Detect pose using MediaPipe
            keypoints = self.detect_pose(frame, (x1, y1, x2, y2))
            
            if keypoints is not None:
                # Smooth landmarks
                smoothed_keypoints = self.smooth_landmarks(keypoints, person_id)
                
                # Draw skeleton on the main output frame
                self.draw_skeleton(output_frame, smoothed_keypoints)
                
                # Generate skeleton-only visualization
                skeleton_only_frame = self.generate_skeleton_only_visualization(smoothed_keypoints, frame.shape)

                # Calculate squat angle
                angle, l_knee, r_knee = self.calculate_squat_angle(smoothed_keypoints)
                
                if angle is not None:
                    # Smooth angle and update count
                    smoothed_angle = self.smooth_angle(angle)
                    self.update_squat_count(smoothed_angle)
                    self.angle_history.append(smoothed_angle) # Add to history for charting
                
                current_persons[person_id] = ((x1, y1, x2, y2), smoothed_keypoints)
            
            # Draw bounding box
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(output_frame, "Person", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        self.tracked_persons = current_persons

        # Draw feedback text at top-left corner
        self.draw_feedback(output_frame, self.smoothed_angle)

        # Generate and overlay chart at bottom-left
        if len(self.angle_history) > 0:
            chart_frame = self.generate_performance_chart(int(output_frame.shape[1] * 0.35), int(output_frame.shape[0] * 0.25)) # Increased height to original
            chart_h, chart_w, _ = chart_frame.shape
            output_h, output_w, _ = output_frame.shape
            
            pos_x = 10 # Left bottom corner
            pos_y = output_h - chart_h - 10
            
            if pos_x + chart_w < output_w and pos_y > 0:
                output_frame[pos_y:pos_y+chart_h, pos_x:pos_x+chart_w] = chart_frame

        # Overlay skeleton-only visualization at bottom-right
        if skeleton_only_frame is not None:
            skel_h, skel_w, _ = skeleton_only_frame.shape
            output_h, output_w, _ = output_frame.shape

            pos_x = output_w - skel_w - 10
            pos_y = output_h - skel_h - 10
            
            if pos_x > 0 and pos_y > 0:
                output_frame[pos_y:pos_y+skel_h, pos_x:pos_x+skel_w] = skeleton_only_frame

        return output_frame

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Squat Counter with Pose Estimation")
    parser.add_argument('--input', type=str, default="/home/ahmed/GSCS/VisynAi/Pose/squat1.mp4", help='Path to input video file')
    parser.add_argument('--output', type=str, default="/home/ahmed/GSCS/VisynAi/Pose/squat_output_enhancedF45.avi", help='Path to output video file')
    parser.add_argument('--model', type=str, default="yolo11n.pt", help='Path to YOLO model weights')
    return parser.parse_args()

def main():
    args = parse_args()
    input_path = args.input
    output_path = args.output
    model_path = args.model

    if not os.path.exists(input_path):
        logging.error(f"Input video not found: {input_path}")
        return
    if not os.path.exists(model_path):
        logging.error(f"YOLO model not found: {model_path}")
        return

    squat_counter = SquatCounter(model_path)
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        processed = squat_counter.process_frame(frame)
        out.write(processed)
        if count % 30 == 0:
            logging.info(f"Frame {count} | Squats: {squat_counter.counter} | Angle: {squat_counter.smoothed_angle:.1f}")
    cap.release()
    out.release()
    logging.info(f"Done. Total squats: {squat_counter.counter}")
    logging.info(f"Saved to: {output_path}")

if __name__ == "__main__":
    main()




#  python pose_main.py --input squat1.mp4 --output results.avi
