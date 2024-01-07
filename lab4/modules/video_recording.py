import datetime
import time
import numpy as np
import os
import socket
import cv2
from glob import glob
from modules.video_writer import VideoWriter


class VideoRecording:
    """! Класс для записывания видеопотока

    Начинает записывать видео, если обнаружен человек,
    до момента исчезновения людей + заданный таймаут.
    """

    def __init__(self, outdir=os.getcwd()+'/output_videos/', person_timeout=60, max_file_length = 120):
        """
        Конструктор

        @param outdir Путь, где будут лежать видео-логи
        @param person_timeout Время записи лога после исчезновения человека

        """

        if not os.path.exists(outdir):
            os.makedirs(outdir)
            os.chmod(outdir, 0o777)
        self.outdir = outdir
        self.timeout = person_timeout
        self.timer = 0
        self.out_stream = None
        self.stream_time_created = 0
        self.max_file_length = max_file_length

    def _create_writer(self):
        cur_time = datetime.datetime.now().strftime("%Y-%m-%d-%Hh-%Mm-%Ss")
        out_path = os.path.join(self.outdir,
                                socket.gethostname() + "-" + cur_time + ".mov")
        self.stream_time_created = time.time()
        return VideoWriter(out_path)

    def record(self, frame:np.ndarray, person_detected:bool):
        """
        Метод класса дял запись кадра в видеопоток вывода.

        На вход:

        * frame (np.ndarray) - текущий кадр;
        * person_detected (bool) - человек находится: за конусами, либо конус убран, либо сдвинут

        """
        #frame = cv2.resize(frame,None,fx=0.6,fy=0.6)

        if person_detected:
            self.timer = time.time()

        if time.time() - self.timer < self.timeout:

            if self.out_stream is None:
                self.out_stream = self._create_writer()

            self.out_stream.write(frame)

            if time.time() - self.stream_time_created > self.max_file_length:
                self.release()

        else:
            self.release()

    def release(self):
        """
        Освобождение потока из памяти. Запись файла на диск.

        """
        if self.out_stream is not None:
            self.out_stream.release()
            self.out_stream = None
