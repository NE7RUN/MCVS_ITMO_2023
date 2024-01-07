
from modules.trt.trt_used import TrtYOLO
import numpy as np
from collections import deque
#специальная библиотека для проверки принадлежности точки полигону
from shapely.geometry import Point, Polygon

from modules.sort import Sort


class Yolov_inference:

    def __init__(self,config:dict,threshold = 200) -> None:
        """
        На вход:
        
        config (dict) - хранит переменные, которые приписаны в yaml файле;

        На выход: None

        * self.source - источник видео;

        * self.model - модель Yolo;

        * self.conf - коэффициент уверенности;

        * self.boxes (np.ndarray([xyxy])) - xyxy координаты ;

        * self.class_ids (np.ndarray([0,1,1])) - id каждого  элемента self.boxes ;

        * self.old_cones (np.ndarray([xyxy])) массив старых конусов, обновление по условию см. __check_cones; 

        * self.count_cones (int) - количество конусов;

        * self.refresh - бинарный массив, True - если количесво конусов уменьшилось, если нет то False


        """
        self.source = config['source']
        self.model = TrtYOLO(config['model_pth'],config['model_size'],conf=config['confidence'])
        self.conf = config['confidence']
        self.boxes = None
        self.class_ids = None
        self.xy_cones = None
        self.xy_man = None
        self.xywh_man = np.empty((0,2))
        self.xyxy_man = np.empty((0,2))
        self.xyxy_cones = None


        ## ФИКСИРОВАННОЕ ПОЛОЖЕНИЕ
        
        # количество циклов запоминания зоны

        self.cycle_count = 5
        self.fix_xyxycones = None
        self.fix_vertices_sorted = None
        self.flag2sw = False
        self.flag_fix = True # флаг для оценки
        

        # Для сравнения
        self.old_cones = None
        self.count_cones = 0

        # бинарный массив если все значения True, то горит зеленый сигнал (мнимый фильтр)

        self.green_status = deque(maxlen=3)

        self.refresh_cones = deque(maxlen=threshold)

        # Используем модель трекера
        self.tracker = Sort(config['TRACKER']['max_age'])

        # Обработка положения конусов

        # Защитный радиус окружности вокруг конуса

        self.radius = config['CONES_ZONE']['radius']

        self.refresh_fix_cones = deque(maxlen=threshold)
        
        # бинарный массив, если все значения True то конус вышел из зоны
        self.refresh_cones_move = deque(maxlen=15)

        # Флаг сингализирует, что конус перекрыт

        self.flag_warining = False 

        # Флаг сигнализирует что конус сдвинут

        self.flag_warining_move = False


    def yolo_inference(self,img:np.ndarray) -> list:
        """ Главный метод класса 
        На вход:

        * img (np.ndarray)  -  фрейм с камеры;

        На выход:

        * self.boxes - координаты xyxy

        * self.class_ids - класс 

        """

        boxes = self.model(img)

        boxes[:, :2] = boxes[:, :2] - boxes[:, 2:4] / 2
        boxes[:, 2:4] = boxes[:, :2] + boxes[:, 2:4]

        self.boxes = np.array(boxes)[:, 0:4].astype(int).tolist()
        self.class_ids = np.array(boxes)[:, 5].astype(int).tolist()
        return self.boxes,self.class_ids

    
    def get_middel_pt(self) -> list([np.array,np.array]):
        """ Получение средних точек (x,y) для конусов и для людей:
            
            На вход: 
            
            * None;

            На выход:

            * xy_man, xy_cones (np.ndarray) 
        """
        
        # инициализация пустых элементов
        self.xy_man = np.empty((0, 2))
        self.xy_cones = np.empty((0, 2))
        self.xywh_man = np.empty((0, 2))
        self.xyxy_man = np.empty((0, 4))
        self.xyxy_cones = np.empty((0, 6))

        for coordinates, id in zip(self.boxes, self.class_ids):

            if id == 0:

                self.xy_man = np.append(self.xy_man, [[int((coordinates[0] + coordinates[2]) / 2), coordinates[3]]], axis = 0)

                self.xyxy_man = np.append(self.xyxy_man,[[int(coordinates[0]),int(coordinates[1]),int(coordinates[2]),int(coordinates[3])]],axis = 0)
                
            else:
                self.xy_cones = np.append(self.xy_cones, [[int((coordinates[0] + coordinates[2]) / 2), coordinates[3]]], axis = 0)

                self.xyxy_cones = np.append(self.xyxy_cones,[[int(coordinates[0]),int(coordinates[1]),int(coordinates[2]),int(coordinates[3]),1,1]],axis = 0)
        


        if self.flag2sw: self.xyxy_cones = self.fix_xyxycones
        
            
        return self.xy_man, self.xy_cones
    

    def get_xyxy_man(self):
        """
        Получение координат боксов людей в формате xyxy
        
        """
        self.radius

        return self.xyxy_man 

    def get_xywh_man(self):

        """
        Получение координат боксов людей в формате xywh
        
        """

        return self.xywh_man
    
    def get_radius_zone(self) -> int:

        """
        Получение радиуса защитной зоны вокруг конусов

        """
        return self.radius
        

    def used_tracker(self) -> np.ndarray:
        """ 
        Метод использования трекера

        На выход:

        * xyxy_man (np.ndarray) [x,y,x1,x1,id,class], class = 0;

        * xyxy_cones (np.ndarray) [x,y,x1,x1,class,class], class = 1, дополнительный класс нужен чтобы размерность была одинаковая у xyxy массивов ;

        * vertices_sorted (np.ndarray) - отсортированные средние точки конусов ;  
        
        """

        if self.xyxy_man.shape == (0,2):
            return None, None, None
        
        tracks = self.tracker.update(self.xyxy_man)
        tracks = tracks.astype(int)
        xyxy_man = np.empty((0,6))

        if type(tracks) is np.ndarray:
            xyxy_man = np.vstack((xyxy_man, np.column_stack((tracks[:, 0:4], tracks[:, 4], np.zeros(tracks.shape[0])))))

        vertices_sorted = self.sort_cones_vertices()
    
        return xyxy_man, self.xyxy_cones, vertices_sorted

    def sort_cones_vertices(self):

        '''
        Функция сортировки по полярному углу относительно центра многоугольника,
        также идет проверка количества конусов, если конусов  меньше чем было то их позиция сохраняется

        На вход:

        * xy (list(tuple(int,int))) - не структурированный набор точек середин bbox;
                
        На выход:

        *  sorted_coordinates ((list(tuple(int,int)))) - структурированный набор точек, по часовой стрелке;
        
        '''

        poly_center = np.mean(self.xy_cones, axis=0)
        vertices = self.xy_cones
        angles = [np.arctan2(vertice[1]-poly_center[1],
                    vertice[0]-poly_center[0]) for vertice in vertices] 
        vertices_sorted = vertices[np.argsort(angles)]
        
        # Проверка что есть фиксация конусов
        if self.flag2sw == True: 
            self.check_position_cones(vertices_sorted)
            vertices_sorted = self.fix_vertices_sorted

        # Если количество уменьшилось, то проверяем что все 
        else:
            
            if self.__check_cones(vertices_sorted) == True :

                """ 
                Версия обновления с последовательностью 
                """
                # значения True тогда обновляем количество конусов, если нет оставляем старые, а в массив записываем True
                if all( val for val in self.refresh_cones):
                    self.count_cones = len(vertices_sorted)
                else:
                    vertices_sorted = self.old_cones
                    self.refresh_cones.append(True)
            # Если количество осталось тем же или стало больше то добавляем в массив False
            else:
                self.refresh_cones.append(False)


        return vertices_sorted
    
    def __check_cones(self, vertices_sorted:np.ndarray) -> bool:
        """

        Проверка, если прошлое количество конусов меньше, либо равно количеству конусов на фрейме,
        то обновляем значение точек и количество конусов, взвращаем False,если больше то возвращаем True.

        На вход: 

        * vertices_sorted (np.ndarray) отсортированные конуса

        На выход:

        * True/False (bool)

        """

        if self.count_cones > len(vertices_sorted):

            return True
        else:
            self.old_cones = vertices_sorted
            self.count_cones = len(vertices_sorted)
            return False
        
    def check_position_cones(self,vertices_sorted) -> None:
        """ 
        Метод проверки преграждения человеком конуса:

        - Если человек переграждает зону количество конусов уменьшается, в self.refresh_fix_cones  записывается True

        - Если количество равно или осталось тем же, то False

        Таким образом, если прегражден конус на долгое время, то срабатывает флаг self.flag_warining = True

         """

        # Проверка, что количество конусов не уменьшилось
        if len(self.fix_vertices_sorted) > len(vertices_sorted):
            # если значение конусов уменьшилось в меньшую сторону
            self.refresh_fix_cones.append(True)
            self.flag_warining = False

        else:
            if len(self.fix_vertices_sorted) == len(vertices_sorted):
                            # Расстояние между текущей точкой конуса 
                            distance  = [] 
                            for point_fix in self.fix_vertices_sorted:
                                # максимальное значение
                                min_value = 18000 
                                for point in vertices_sorted:
                                    l = ((point_fix[0]-point[0])**2 + (point_fix[1]-point[1])**2 )**0.5
                                    if l < min_value:
                                        min_value = l
                                distance.append(min_value)

                            """ Можно поиграться с флагами"""
                            
                            if all(x < self.radius for x in distance) == True:
                                self.refresh_cones_move.append(False)
                                self.refresh_fix_cones.append(False)
                                self.flag_warining_move = False
                                
                            else:
                                self.refresh_cones_move.append(True)
                                self.refresh_fix_cones.append(False)

            else:
                # если значение конусов увеличилось либо равно зафиксированному значению
                self.refresh_fix_cones.append(False)
                self.refresh_cones_move.append(False)
                self.flag_warining = False

        

        # если в массиве все значения TRUE, то есть уже какое то время конус остается перекрытым
        if np.all(self.refresh_fix_cones):
            self.flag_warining = True
        if np.all(self.refresh_cones_move):
            self.flag_warining_move = True
        
        
    def check_inside_safezone(self) -> bool:

        """ Метод класса попадания точки в полигон защитной зоны

            * 0 - людей на фрейме нет;
            * 1 - есть человек, который зашел за полигон;
            * 2 - все люди локализованы в зоне;
            * 3 - моргание при заслонение конусов,на длительный период ;
            * 4 - конус вышел из защитной зоны.
        """
        if self.flag_warining:
            self.green_status.append(False)
            return 3
        if self.flag_warining_move:
            self.green_status.append(False)
            return 4

        # Преобразование массивов в объекты shapely
        human_points = [Point(point) for point in self.xy_man]
        polygon = Polygon(self.old_cones)

        # Проверка принадлежности точек многоугольнику
        points_inside_polygon = [point.within(polygon) for point in human_points]

        if len(points_inside_polygon) == 0:
            self.green_status.append(True)
            return 0 if all(self.green_status) else 1
        
        if points_inside_polygon.count(False) >= 1:
            self.green_status.append(False)
            return 1

        self.green_status.append(True)
        return 2 if all(self.green_status) else 1
        

        
    def fix_zone(self) -> None:
        """ 
        Метод Фиксирование зоны, используется перед методом click_button,
        тогда будет корректная обработка события нажатия на кнопку

        Обновление параметров:

        * self.flag_fix (bool) - логическая переменная для проверки условия в методе fixzone;

        * self.flag2sw (bool) - дополнительный флаг для фиксирования xyxy_cones; 

        На выход: None
        
        """
        if self.flag_fix == True and len(self.xyxy_cones.tolist()) > 0 and len(self.xyxy_cones.tolist()) <= 2:
            print("[НЕ ДОСТАТОЧНО КОНУСОВ ДЛЯ ФИКСАЦИИ ЗОНЫ]")
            self.flag_fix = False
        
        elif self.flag_fix == True and len(self.xyxy_cones.tolist()) > 0 and len(self.xyxy_cones.tolist()) >= 3:
            self.fix_xyxycones = self.xyxy_cones
            self.fix_vertices_sorted = self.sort_cones_vertices()
            self.flag_fix = False
            self.flag2sw = True
            print("[ЗОНА ЗАФИКСИРОВАНА]")

    def click_button(self):
        """ 
        Метод обработки нажатия кнопки
         
        * Использовать после метода fix_zone

           """
        self.flag_fix = True
        self.flag2sw = False
        print('[CLICK BUTTON!]')


        