import argparse
from yacs.config import CfgNode as CN
import os.path as osp
import os
import cv2
import numpy as np
from time import time, strftime
from dataset.annotate import draw, get_dart_scores
import pickle

def bboxes_to_xy(bboxes, max_darts=3):
    xy = np.zeros((4 + max_darts, 3), dtype=np.float32)
    for cls in range(5):
        if cls == 0:
            dart_xys = bboxes[bboxes[:, 4] == 0, :2][:max_darts]
            xy[4:4 + len(dart_xys), :2] = dart_xys
        else:
            cal = bboxes[bboxes[:, 4] == cls, :2]
            if len(cal):
                xy[cls - 1, :2] = cal[0]
    xy[(xy[:, 0] > 0) & (xy[:, 1] > 0), -1] = 1
    if np.sum(xy[:4, -1]) == 4:
        return xy
    else:
        xy = est_cal_pts(xy)
    return xy

def est_cal_pts(xy):
    missing_idx = np.where(xy[:4, -1] == 0)[0]
    if len(missing_idx) == 1:
        if missing_idx[0] <= 1:
            center = np.mean(xy[2:4, :2], axis=0)
            xy[:, :2] -= center
            if missing_idx[0] == 0:
                xy[0, 0] = -xy[1, 0]
                xy[0, 1] = -xy[1, 1]
                xy[0, 2] = 1
            else:
                xy[1, 0] = -xy[0, 0]
                xy[1, 1] = -xy[0, 1]
                xy[1, 2] = 1
            xy[:, :2] += center
        else:
            center = np.mean(xy[:2, :2], axis=0)
            xy[:, :2] -= center
            if missing_idx[0] == 2:
                xy[2, 0] = -xy[3, 0]
                xy[2, 1] = -xy[3, 1]
                xy[2, 2] = 1
            else:
                xy[3, 0] = -xy[2, 0]
                xy[3, 1] = -xy[2, 1]
                xy[3, 2] = 1
            xy[:, :2] += center
    else:
        print('Missed more than 1 calibration point')
    return xy

def calculate_score(preds, cfg):
    dart_scores = get_dart_scores(preds, cfg)
    valid_scores = []
    for score in dart_scores:
        try:
            if score.startswith('T'):
                valid_scores.append(3 * int(score[1:]))
            elif score.startswith('D'):
                valid_scores.append(2 * int(score[1:]))
            else:
                valid_scores.append(int(score))
        except ValueError:
            print(f"Invalid score '{score}' encountered. Skipping this score.")
    print(f'Dart scores: {valid_scores}')
    return valid_scores

def predict_video_with_teams(yolo, cfg, video_path, output_path, max_darts=3):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_score_team1 = 301
    total_score_team2 = 301
    current_team = 1  # Start with Team 1
    detected_darts_team1 = []
    detected_darts_team2 = []
    black_screen_frame_threshold = 50  # Define intensity for black screen detection
    black_screen_detected = False
    switch_cooldown = 30  # Number of frames to wait before switching teams again
    last_switch_frame = -switch_cooldown  # Initialize with a large negative number

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect black screen to switch teams
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_intensity = gray_frame.mean()

        # Check if it's a black screen and if enough frames have passed since the last switch
        if mean_intensity < black_screen_frame_threshold and (frame_count - last_switch_frame) >= switch_cooldown:
            current_team = 2 if current_team == 1 else 1  # Switch between Team 1 and 2
            print(f"Black screen detected at frame {frame_count}. Switching to Team {current_team}")
            last_switch_frame = frame_count  # Update the last switch frame
            continue

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bboxes = yolo.predict(img)
        preds = bboxes_to_xy(bboxes, max_darts)

        if len(preds) > 0:
            dart_scores = calculate_score(preds, cfg)
            if current_team == 1:
                for score in dart_scores:
                    if score not in detected_darts_team1:
                        total_score_team1 -= score
                        detected_darts_team1.append(score)
                        if total_score_team1 < 0:
                            total_score_team1 = 0
                        print(f'Team 1 score: {total_score_team1}')
            else:
                for score in dart_scores:
                    if score not in detected_darts_team2:
                        total_score_team2 -= score
                        detected_darts_team2.append(score)
                        if total_score_team2 < 0:
                            total_score_team2 = 0
                        print(f'Team 2 score: {total_score_team2}')

        # Display the current scores of both teams
        score_text = f"Team 1: {total_score_team1} | Team 2: {total_score_team2}"
        cv2.putText(frame, score_text, (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    print(f'Saved output video to {output_path}')


if __name__ == '__main__':
    from train import build_model
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', default='deepdarts_utrecht')
    parser.add_argument('--video', required=True, help='Path to input video')
    parser.add_argument('--output', required=True, help='Path to output video')
    parser.add_argument('--max_darts', type=int, default=3, help='Maximum number of darts to detect')
    args = parser.parse_args()

    cfg = CN(new_allowed=True)
    cfg.merge_from_file(osp.join('configs', args.cfg + '.yaml'))
    cfg.model.name = args.cfg

    yolo = build_model(cfg)
    yolo.load_weights(osp.join('models', args.cfg, 'weights'), cfg.model.weights_type)

    predict_video_with_teams(yolo, cfg, args.video, args.output, args.max_darts)
