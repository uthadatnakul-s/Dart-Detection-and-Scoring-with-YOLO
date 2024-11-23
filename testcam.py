import cv2

def test_camera(cam_index):
    cam = cv2.VideoCapture(cam_index)
    if not cam.isOpened():
        print(f"Error: Could not open camera with index {cam_index}")
        return

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to capture image")
            break

        cv2.imshow('Camera Test', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # ลองเปลี่ยนค่าดัชนีจาก 0 เป็น 1, 2, 3 และอื่นๆ
    test_camera(0)
    test_camera(1)
    test_camera(2)
    test_camera(3)
