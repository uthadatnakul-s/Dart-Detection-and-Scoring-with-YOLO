import argparse
from yacs.config import CfgNode as CN
import os.path as osp
import os
import cv2
import numpy as np
from time import time, strftime
from dataset.annotate import draw, get_dart_scores
import pickle

def bboxes_to_xy(bboxes, max_darts=3):  # ปรับค่า max_darts ตามที่ต้องการ
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

def predict_video(yolo, cfg, video_path, output_path, max_darts=3):  # ปรับค่า max_darts ตามที่ต้องการ
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bboxes = yolo.predict(img)
        preds = bboxes_to_xy(bboxes, max_darts)

        xy = preds[preds[:, -1] == 1]
        img = draw(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), xy[:, :2], cfg, circles=False, score=True)
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
    # parser.add_argument('--max_darts', type=int, default=10, help='Maximum number of darts to detect')
    parser.add_argument('--max_darts', type=int, default=3, help='Maximum number of darts to detect')


    args = parser.parse_args()

    cfg = CN(new_allowed=True)
    cfg.merge_from_file(osp.join('configs', args.cfg + '.yaml'))
    cfg.model.name = args.cfg

    yolo = build_model(cfg)
    yolo.load_weights(osp.join('models', args.cfg, 'weights'), cfg.model.weights_type)

    predict_video(yolo, cfg, args.video, args.output, args.max_darts)

