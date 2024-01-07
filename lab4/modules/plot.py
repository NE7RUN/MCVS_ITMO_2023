import numpy as np
import cv2

class Visualisation:

     """
     
     Класс для визуализации боксов, средней точки и отображения защитной зоны вокруг конуса.

     """

     def __init__(self,radius:int,*args) -> None:

          # Инициализация переменых для перезаписи
          self.img = None
          self.xy_man = None
          self.xy_cones = None
          self.xyxy_man = None
          self.xyxy_cones = None
          self.class_ids = None
          self.state = None
          self.sorted_points = None

          self.radius = radius
          self.flag_view_rectng,self.flag_view_mid_p,self.flag_polygon_view = args[0]

     def __call__(self, *args) -> np.ndarray:
          """
          Главный метод отрисовки на вход подается кортеж в формате:

          (img:np.ndarray, xy_man:np.ndarray, xy_cones:np.ndarray, xyxy_man:np.ndarray,\
                  xyxy_cones:np.ndarray, class_ids:list, state:int, sorted_points:np.ndarray)
          
          На выход

          * img (np.ndarray) - обработанное изображение;

          """
          # Обновлние параметров
          self.img = args[0]
          self.xy_man = args[1]
          self.xy_cones = args[2]
          self.xyxy_man = args[3]
          self.xyxy_cones = args[4]
          self.class_ids = args[5]
          self.state = args[6]
          self.sorted_points = args[7] 

          if self.flag_view_mid_p:
               self.__draw_middel_point()

          if self.flag_view_rectng:
               self.__draw_rectangles()

          if self.flag_polygon_view:
               self.__draw_polygon()

          return self.img

     #Функции отображения
     def __draw_middel_point(self) -> None:

          """
          Функция отрисовка средних точек.
          """
          try:
               for coordinates in self.xy_man:
                    cv2.rectangle(self.img, (int(coordinates[0]),int(coordinates[1])), (int(coordinates[0]),int(coordinates[1])),(0,255,0),4)
          except: pass

          try:
               for coordinates in self.xy_cones:
                    cv2.rectangle(self.img, (int(coordinates[0]),int(coordinates[1])), (int(coordinates[0]),int(coordinates[1])),(0,255,0),4)
          except: pass

     def __draw_rectangles(self) -> None:

          """ 
          Функция отрисовки прямоугольников

          - 0 - людей на фрейме нет;
          - 1 - есть человек, который зашел за полигон;
          - 2 - все люди локализованы в зоне;
          - 3 - моргание при заслонение конусов,на длительный период ;
          - 4 - конус вышел из защитной зоны.
     
          """
          state_colors = {
                         0: (0, 255, 0),
                         1: (0, 0, 255),
                         2: (0, 255, 0),
                         3: (47, 171, 216),
                         4: (216, 47, 188)
                         }
          color = state_colors.get(self.state)

          xyxy_man = np.append(self.xyxy_man,self.xyxy_cones,axis = 0)

          for xmin,ymin,xmax,ymax,track_id,class_ids in  xyxy_man.astype(int):

               if class_ids == 0:
                    cv2.rectangle(self.img, (xmin,ymin), (xmax,ymax),color,1)
                    cv2.putText(self.img, text=f"Id: {track_id}", org=(xmin, ymin-10), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=color, thickness=1)
               else:
                    cv2.rectangle(self.img, (xmin,ymin), (xmax,ymax),(255,255,255),1)
                    cv2.circle(self.img,(int((xmax-xmin)/2)+xmin,ymax),self.radius,(0,255,0),3)


     def __draw_polygon(self)-> None:
          """
          Отрисовка полигонов
          
          - 0 - людей на фрейме нет;
          - 1 - есть человек, который зашел за полигон;
          - 2 - все люди локализованы в зоне;
          - 3 - моргание при заслонение конусов,на длительный период ;
          - 4 - конус вышел из защитной зоны.

          """

          state_colors = {
                         0: (0, 255, 0),
                         1: (0, 0, 255),
                         2: (0, 255, 0),
                         3: (47, 171, 216),
                         4: (216, 47, 188)
                         }

          color = state_colors.get(self.state)
          try:
               self.img = cv2.polylines(self.img, [np.array(self.sorted_points, np.int32).reshape((-1, 1, 2))], True, color, 2)
          except: pass
          