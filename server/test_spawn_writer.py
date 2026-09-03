import multiprocessing
import os
import cv2


def worker():
    print("child LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH"))
    w = cv2.VideoWriter("recordings/test_proc.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10, (640, 480))
    print("opened (mp spawn process):", w.isOpened())
    w.release()


if __name__ == "__main__":
    print("parent LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH"))
    multiprocessing.set_start_method("spawn")
    p = multiprocessing.Process(target=worker)
    p.start()
    p.join()
