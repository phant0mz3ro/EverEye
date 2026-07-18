"""
Minimal multiprocessing test — completely isolated from the camera project.
If this fails with the same SemLock/FileNotFoundError, the problem is
environmental (something about this Python build or this machine), not
anything in multi_camera.py's design.
"""

import multiprocessing


def worker(q):
    q.put("hello from child process")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=worker, args=(q,))
    p.start()
    print(q.get())
    p.join()
    print("Test passed — multiprocessing works fine on this system")
