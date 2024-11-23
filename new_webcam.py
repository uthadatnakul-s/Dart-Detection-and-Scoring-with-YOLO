import argparse
from yacs.config import CfgNode as CN
import os.path as osp
import cv2
import numpy as np
from dataset.annotate import draw
from predict import bboxes_to_xy

def predict_stream(yolo, cam_index):

    # เปิดกล้องตาม index ที่ผู้ใช้ระบุ
    cam = cv2.VideoCapture(cam_index)
    
    # ตรวจสอบว่ากล้องเปิดได้หรือไม่
    if not cam.isOpened():
        print(f"Error: ไม่สามารถเปิดกล้องได้ที่ index {cam_index}")
        return

    # ตั้งค่าความละเอียดของกล้อง
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cam.set(cv2.CAP_PROP_FPS, 30)

    print("Width:", cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    print("Height:", cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("FPS:", cam.get(cv2.CAP_PROP_FPS))

    while True:
        check, frame = cam.read()
        if not check:
            print("Error: ไม่สามารถอ่านภาพจากกล้องได้")
            break

        # Resize frame เพื่อให้เหมาะกับการประมวลผลของ YOLO
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = img[50:1000, 400:1000]  # Crop ภาพ
        img = cv2.resize(img, (640, 640))

        # ตรวจจับวัตถุด้วย YOLO
        bboxes = yolo.predict(img)
        preds = bboxes_to_xy(bboxes, 3)
        xy = preds[preds[:, -1] == 1]

        # วาด bounding boxes บนภาพ
        img = draw(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), xy[:, :2], cfg, circles=False, score=True)
        cv2.imshow('video', img)

        key = cv2.waitKey(33)  # ควบคุมการแสดงภาพให้เหมาะกับ FPS
        if key == ord('z'):  # กด 'z' เพื่อหยุดการแสดงผล
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    from train import build_model
    
    # รับ argument จาก command line
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', default='deepdarts_utrecht', help='ชื่อไฟล์ config')
    parser.add_argument('-cam', '--camera', type=int, default=0, help='index ของกล้องที่ต้องการใช้ (ค่าเริ่มต้นคือ 0)')
    args = parser.parse_args()

    # โหลด config
    cfg = CN(new_allowed=True)
    cfg.merge_from_file(osp.join('configs', args.cfg + '.yaml'))
    cfg.model.name = args.cfg

    # โหลดโมเดล YOLO
    yolo = build_model(cfg)
    yolo.load_weights(osp.join('models', args.cfg, 'weights'), cfg.model.weights_type)

    # เริ่มการตรวจจับด้วยกล้องที่ระบุ
    predict_stream(yolo, args.camera)
