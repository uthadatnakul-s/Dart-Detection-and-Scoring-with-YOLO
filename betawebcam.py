import argparse
from yacs.config import CfgNode as CN
import os.path as osp
import cv2
import numpy as np
from dataset.annotate import draw
from predict import bboxes_to_xy


def letterbox_image(image, target_size):
    """ทำการ resize ภาพพร้อมรักษาอัตราส่วนเดิม และเติม padding"""
    ih, iw = image.shape[:2]
    w, h = target_size

    scale = min(w/iw, h/ih)
    nw, nh = int(iw * scale), int(ih * scale)

    resized_image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)

    new_image = np.full((h, w, 3), 128, dtype=np.uint8)
    new_image[(h - nh) // 2:(h - nh) // 2 + nh, (w - nw) // 2:(w - nw) // 2 + nw] = resized_image

    return new_image


def predict_stream(yolo, cam_index):
    # เปิดกล้องตามที่เลือก
    cam = cv2.VideoCapture(cam_index)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cam.set(cv2.CAP_PROP_FPS, 30)

    if not cam.isOpened():
        print("Error: ไม่สามารถเปิดกล้องได้")
        return

    print("Width:", cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    print("Height:", cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("FPS:", cam.get(cv2.CAP_PROP_FPS))

    while True:
        check, frame = cam.read()
        if not check:
            print("Error: ไม่สามารถอ่านภาพจากกล้องได้")
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _ = img.shape

        # Crop เฉพาะส่วนกระดานปาเป้า
        x_start, x_end = int(0.2 * width), int(0.8 * width)
        y_start, y_end = int(0.1 * height), int(0.9 * height)
        img = img[y_start:y_end, x_start:x_end]

        # Resize โดยรักษาสัดส่วนเดิม
        img = letterbox_image(img, (800, 800))

        # ตรวจจับ bounding boxes
        bboxes = yolo.predict(img)
        preds = bboxes_to_xy(bboxes, 3)
        xy = preds[preds[:, -1] == 1]

        img = draw(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), xy[:, :2], cfg, circles=False, score=True)

        cv2.imshow('video', img)

        key = cv2.waitKey(33)
        if key == ord('z'):
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    from train import build_model
    parser = argparse.ArgumentParser()

    # เพิ่ม argument สำหรับการเลือกกล้อง
    parser.add_argument('-c', '--cfg', default='deepdarts_utrecht', help="ชื่อ config")
    parser.add_argument('--camera', type=int, default=0, help="เลือกกล้อง (ค่าเริ่มต้นคือกล้อง 0)")

    args = parser.parse_args()

    cfg = CN(new_allowed=True)
    cfg.merge_from_file(osp.join('configs', args.cfg + '.yaml'))
    cfg.model.name = args.cfg

    yolo = build_model(cfg)
    yolo.load_weights(osp.join('models', args.cfg, 'weights'), cfg.model.weights_type)

    # ส่งค่า camera index ที่เลือกไปยังฟังก์ชัน predict_stream
    predict_stream(yolo, args.camera)
