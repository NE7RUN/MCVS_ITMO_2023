import cv2
import yaml
import time
import numpy as np

from modules.plot import Visualisation

# класс для инференса модели и обработки логики системы
from modules.Yolov_inference_trt import Yolov_inference

from modules.VideoCapture import VideoCapture

from modules.video_recording import  VideoRecording

class Save_area:
    """ Основной класс """
    def __init__(self,config:dict) -> None:
        """ Конструктор класса Save_area

        На вход получает словарь:

         * config (dict) - словарь с конфигурацией  """
        
        # Инициализация классов
        self.record = VideoRecording(person_timeout = config['personal_timeout'],
                                     max_file_length= config['video_length'])
        self.yolo_inf = Yolov_inference(config)
        self.visual = Visualisation(config['CONES_ZONE']['radius'],config['CONES_ZONE']['draw_param'])
        
        # флаги
        self.flag_show_img = config['show_imshow']

        # переменные
        self.time_ = 0

    def __call__(self, img: np.ndarray) -> None:

        img = cv2.resize(img, None, fx=config['RESIZE']['fx'], fy=config['RESIZE']['fy'])
        try:
            _, class_ids = self.yolo_inf.yolo_inference(img)
            self.yolo_inf.used_tracker()
            xyxy_man, xyxy_cones, sorted_points = self.yolo_inf.used_tracker()
            xy_man, xy_cones = self.yolo_inf.get_middel_pt()
        except:
            sorted_points = None
        try:
            state = self.yolo_inf.check_inside_safezone()
        except:
            state = 1

        if type(sorted_points) is np.ndarray and len(sorted_points) >= 3:
            self.visual(img, xy_man, xy_cones, xyxy_man, xyxy_cones, class_ids, state, sorted_points)
        else:
            pass

        if self.flag_show_img:
            cv2.imshow("img", img)

        fps = 1 / (time.time() - self.time_)

        if config["show_fps"] == True:
            print(f"FPS: {round(fps, 2)}")

        self.time_ = time.time()

        if config['video_logs']:
            self.record.record(img, person_detected=True)

        if type(sorted_points) is np.ndarray:
            self.yolo_inf.fix_zone()

        # обработка нажатия мышки
        # Порядок важен сначала метод fix_zone()
        if cv2.waitKey(25) & 0xFF == ord('a'):
            self.yolo_inf.click_button()

    def release(self):
        """ Освобождение потоков"""
        self.record.release()
        try:
            self.record.release()
        except: pass
    


def main(config:dict) -> None:
    """ Главная функция """
    if config['rtsp_flow']:
        cap = VideoCapture(config['source'])
    else:cap = cv2.VideoCapture(config['source'])

    save_area = Save_area(config)

    while True:
        ret,img = cap.read()

        if ret:
            save_area(img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    save_area.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    with open("./config/config.yaml","r") as file:
         config = yaml.safe_load(file)
    
    main(config)



