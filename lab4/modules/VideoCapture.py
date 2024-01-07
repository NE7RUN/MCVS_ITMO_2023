import cv2
import queue
import threading
import time


class VideoCapture:
    def __init__(self, name, fps = 20):
        """! Инициализация
        @param name идентификация потока
        """

        self.cap = cv2.VideoCapture(name)
        self.name = name
        self.q = queue.Queue()
        
        #self.logger = CustomLogger(__name__)

        self.fps = fps
        self.last_frame_time = time.time()

        t = threading.Thread(target=self._reader, daemon=True)
        t.start()

    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not self.q.empty():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            if (time.time() - self.last_frame_time > 1/self.fps):
                self.q.put((ret, frame))
                self.last_frame_time = time.time()
            if not ret:
                time.sleep(5)
                self.cap.release()
                self.cap = cv2.VideoCapture(self.name)
    def read(self):
        """! Получить последний кадр"""
        try:
            return self.q.get(timeout=0.1)
        except queue.Empty:
            return False, None

    def release(self):
        """! Освободить поток"""
        self.cap.release()