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
        # TODO: if len(missing_idx) > 1
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

def predict_video(yolo, cfg, video_path, output_path, max_darts=3):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_score = 301  # Starting score
    detected_darts = []  # List to store detected darts

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bboxes = yolo.predict(img)
        preds = bboxes_to_xy(bboxes, max_darts)

        if len(preds) > 0:
            dart_scores = calculate_score(preds, cfg)
            for score in dart_scores:
                if score not in detected_darts:  # Check if the dart has already been detected
                    total_score -= score
                    detected_darts.append(score)
                    if total_score < 0:
                        total_score = 0
                    print(f'Total score after this dart: {total_score}')  # Debugging total score

        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        bboxes = yolo.predict(img)
        preds = bboxes_to_xy(bboxes, max_darts)
        xy = preds[preds[:, -1] == 1]
        img = draw(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), xy[:, :2], cfg, circles=False, score=True)

        # Display the score on the video
        cv2.putText(img, f'Score: {total_score}', (190, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        out.write(img)

    cap.release()
    out.release()
    print(f'Saved detected video to {output_path}')

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

    predict_video(yolo, cfg, args.video, args.output, args.max_darts)
