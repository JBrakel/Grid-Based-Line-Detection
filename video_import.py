import os
import numpy as np
import cv2
import json
from bs4 import BeautifulSoup

class VideoImport:
    def __init__(self, city, race):
        self.city = city
        self.race = race

        self.video_dict = {
            'tokio': {
                1: ['Tokio-2021', 'K4_H_500m_VL_ESP_SVK'],
                2: ['Tokio-2021', 'K4_H_500m_VL_GER']
            },
            'münchen': {
                1: ['München', 'K4_D_500m_VL_EC_München']
            },
            'poznan': {
                1: ['Poznan', 'C2_H_500m_FA_WC_Poznan'],
                2: ['Poznan', 'K1_D_500m_FA_WC_Poznan']
            },
            'racice': {
                1: ['Racice', 'C1_H_200m_WC_Racice'],
                2: ['Racice', 'K4_H_500m_FA_WC_Racice']
            },
            'duisburg':{
                1: ['Duisburg','C0001'],
                # 2: ['Duisburg','C0002'],
                3: ['Duisburg','C0003'],
                4: ['Duisburg','C0004'],
                12: ['Duisburg','C0012'],
                13: ['Duisburg','C0013'],
                15: ['Duisburg','C0015'],
                16: ['Duisburg','C0016'],
                19: ['Duisburg','C0019'],
                20: ['Duisburg','C0020']
            }
        }

        self.start_positions = {
            ('tokio', 1): 5200,
            ('tokio', 2): 4800,
            ('münchen', 1): 5100,
            ('poznan', 1): 5500,
            ('poznan', 2): 6200,
            ('racice', 1): 2300,
            ('racice', 2): 4600,
            ('duisburg',1): 2725,
            # ('duisburg',2): 2800,
            ('duisburg',3): 2470,
            ('duisburg',4): 2590,
            ('duisburg',12): 1335,
            ('duisburg',13): 1150,
            ('duisburg',15): 2560,
            ('duisburg',16): 2330,
            ('duisburg',19): 1110,
            ('duisburg',20): 725
        }

        self.video = self.video_dict.get(self.city, {}).get(self.race)

        if self.video is None:
            print('Video konnte nicht gefunden werden')
            return None

        self.home_directory = os.path.expanduser('~')
        self.video_path = self.home_directory + "/canoe_video_processing/line_detection_grid/raw/" + self.video[0] + "/" + self.video[1] \
                          + "/deinterlaced/" + self.video[1] + '.MP4'#".avi"

        self.video_folder, self.video_file = os.path.split(self.video_path)
        self.video_folder, self.deinterlaced_folder = os.path.split(self.video_folder)
        self.json_path = os.path.join(self.video_folder,"label", "obj_train_data", "label.json")
        self.json_path_yolo = os.path.join(self.video_folder,"yolo", "label", "obj_train_data", "label.json")
        self.txt_path = os.path.join(self.video_folder,"label", "obj_train_data")
        self.txt_path_full_video = os.path.join(self.video_folder,"yolo", "label", "obj_train_data")
        self.path_labeled_lines = os.path.join(self.video_folder,"label_lines", "annotations.xml")

    def get_video_path(self):
        return self.video_path

    def get_txt_path(self):
        return self.txt_path

    def get_txt_path_full_video(self):
        return self.txt_path_full_video

    def get_race(self):
        return self.video

    def get_best_start_position(self):
        return self.start_positions[(self.city, self.race)]

    def get_json_path_yolo(self):
        return self.json_path_yolo

    def get_json_path(self):
        return self.json_path

    def get_path_labeled_lines(self):
        return self.path_labeled_lines

    def create_json_from_new_videos(self, txt_path):
        print('start')
        file_dict = {}
        # nr_regions = 0
        types = {
            '0': 'buoy'
        }
        names = {
            '0': 'point'
        }
        if os.path.isdir(txt_path):
            for filename in os.listdir(txt_path):
                file_path = os.path.join(txt_path, filename)
                if os.path.isfile(file_path) and filename.endswith('.txt'):
                    with open(file_path, 'r') as txt_file:
                        data = txt_file.readlines()
                        regions_data = []
                        for region in data:
                            values = region.split()
                            if len(values) == 5: cls, x, y, bb_width, bb_height = values
                            else: cls, x, y, bb_width, bb_height, _ = values
                            if cls == '32':
                                if float(_) < 0.5:
                                    continue
                                img_width = 3840
                                img_height = 2160

                                cx = (float(x) * img_width)
                                cy = (float(y) * img_height)

                                regions_data.append({
                                    'shape_attributes': {
                                        'name': 'point',
                                        'cx': int(cx),
                                        'cy': int(cy)
                                    },
                                    'region_attributes': {
                                        'type': 'buoy',
                                        'colour': 'unknown',
                                        'size': 'small'
                                    }
                                })

                            filename_dict = {
                                'filename': filename.split('.')[0] + '.jpg',
                                'regions': regions_data
                            }
                            file_dict[f'{filename}'] = filename_dict
                            file_dict = dict(sorted(file_dict.items(), key=lambda item: int(item[0].split('_')[1].split('.')[0])))
        if len(file_dict) == 0: json_file = None
        else: json_file = {'_via_img_metadata': file_dict}
        print('finish')
        return json_file


    def import_labeled_lines(self, path_labeled_lines, current_frame_position):
        self.path_labeled_lines = path_labeled_lines
        with open(self.path_labeled_lines, 'r') as f:
            data = f.read()
        Bs_data = BeautifulSoup(data, "xml")
        polylines = Bs_data.find_all('polyline')
        frames = {}
        for polyline in polylines:
            frame = int(polyline.get('frame'))
            points = polyline.get('points')
            p1 = (int(float(points.split(',')[0])), int(float(points.split(',')[1].split(';')[0])))
            p2 = (int(float(points.split(';')[1].split(',')[0])), int(float(points.split(';')[1].split(',')[1])))
            if frame not in frames:
                frames[frame] = []
            frames[frame].append([p1, p2])
        return np.array(frames[current_frame_position])


    def get_frame_annotation(self, image, number, img_metadata):
        all_buoys = []
        for key, value in img_metadata.items():
            filename = value['filename']
            frame_number = int((filename.split('.')[0]).split('_')[1])
            if number == frame_number:
                regions = value['regions']
                all_colours_buoys = {'red': (0, 0, 255), 'yellow': (0, 255, 255), 'white': (255, 255, 255),
                                     'none': (0, 0, 0), 'unknown':(0, 255, 255)}
                size_buoys = {'small': 6, 'big': 10, 'none': 0} #, 'small ': 6
                if len(regions) == 0:
                    mask = np.zeros_like(image)
                else:
                    mask = np.zeros_like(image)
                    for region in regions:
                        type = region['region_attributes'].get('type', 'none')
                        colour = region['region_attributes'].get('colour', 'none')
                        size = region['region_attributes'].get('size', 'none').strip()
                        selected_colour = all_colours_buoys[colour]
                        selected_size = size_buoys[size]
                        if type == 'buoy':
                            x = region['shape_attributes']['cx']
                            y = region['shape_attributes']['cy']
                            # cv2.circle(mask, (x, y), selected_size, selected_colour, -1)
                            cv2.circle(image, (x, y), 20, all_colours_buoys['yellow'], -1)
                            all_buoys.append((x,y))
                        elif type == 'head':
                            x = region['shape_attributes']['cx']
                            y = region['shape_attributes']['cy']
                            #cv2.circle(mask, (x, y), 6, (255, 119, 0), -1)
                        elif type == 'boat_tip':
                            x = region['shape_attributes']['cx']
                            y = region['shape_attributes']['cy']
                            #cv2.circle(mask, (x, y), 6, (255, 119, 0), -1)
                        elif type == 'line':
                            x1 = region['shape_attributes']['all_points_x'][0]
                            y1 = region['shape_attributes']['all_points_y'][0]
                            x2 = region['shape_attributes']['all_points_x'][1]
                            y2 = region['shape_attributes']['all_points_y'][1]
                            #cv2.line(mask, (x1, y1), (x2, y2), (255, 0, 255), 3)
                        elif type == 'direction':
                            x1 = region['shape_attributes']['all_points_x'][0]
                            y1 = region['shape_attributes']['all_points_y'][0]
                            x2 = region['shape_attributes']['all_points_x'][1]
                            y2 = region['shape_attributes']['all_points_y'][1]
                            #cv2.line(mask, (x1, y1), (x2, y2), (2, 84, 235), 3)
        all_buoys = np.array(all_buoys)
        return all_buoys

    def get_real_coordinates_buoys(self, json_path, frame, number):
        if json_path is not None and os.path.isfile(json_path):
            with open(json_path) as json_file:
                data = json.load(json_file)
                img_metadata = data['_via_img_metadata']
                dict_real_buoys = self.get_frame_annotation(frame, number, img_metadata)[1]

            frame_annotation_numbers = []
            for key, value in img_metadata.items():
                filename = value['filename']
                frame_number = int((filename.split('.')[0]).split('_')[1])
                frame_annotation_numbers.append(frame_number)

        coords_real_buoys = []
        for x, y in dict_real_buoys:
            coords_real_buoys.append((x,y))
        coords_real_buoys = np.array(coords_real_buoys)
        return coords_real_buoys


    def save_frames(self,video_path, frame_positions, folder):
        home_directory = os.path.expanduser('~')
        new_path = home_directory + f'/canoe_video_processing/line_detection_grid/raw/Duisburg/{folder}/frames'  # saving path
        os.chdir(new_path)
        cap = cv2.VideoCapture(video_path)
        for pos in frame_positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap.read()
            cv2.imwrite('frame_%d' % int(cap.get(cv2.CAP_PROP_POS_FRAMES)) + '.jpg', frame)
            print('frame_%d' % int(cap.get(cv2.CAP_PROP_POS_FRAMES)) + '.jpg' + ' saved in folder ' + new_path.split('raw')[1])


    def extract_multiple_frames(self,capture, time_intervall_in_seconds,folder):
        home_directory = os.path.expanduser('~')
        new_path = home_directory + f'/canoe_video_processing/line_detection_grid/raw/Duisburg/{folder}/frames'  # saving path
        os.chdir(new_path)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        print('Total numbers of frames:', frame_count)
        frames_per_second = capture.get(cv2.CAP_PROP_FPS)
        print('Frames per second      :', frames_per_second)
        duration_in_seconds = frame_count / frames_per_second
        second = 0
        capture.set(cv2.CAP_PROP_POS_MSEC, second * time_intervall_in_seconds * 1000)
        success, frame = capture.read()
        capture.set(cv2.CAP_PROP_POS_FRAMES, second)
        while success:
            cv2.imwrite('frame_%d' % int(capture.get(cv2.CAP_PROP_POS_FRAMES)) + '.jpg', frame)
            print('frame_%d' % int(capture.get(cv2.CAP_PROP_POS_FRAMES)) + '.jpg' + ' saved in folder ' + new_path)
            second += 1
            capture.set(cv2.CAP_PROP_POS_MSEC, second * time_intervall_in_seconds * 1000)
            success, frame = capture.read()
