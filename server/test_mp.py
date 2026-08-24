import multiprocessing
import cv2

def worker():
    print("child starting", flush=True)
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    print("opened:", cap.isOpened(), flush=True)
    ret, frame = cap.read()
    print("read:", ret, flush=True)
    cap.release()

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    p = multiprocessing.Process(target=worker)
    p.start()
    p.join()