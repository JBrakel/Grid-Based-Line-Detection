import datetime
import os
import cv2
import json
import matplotlib.pyplot as plt


class Data:
    def __init__(self):
        self.home_directory = os.path.expanduser('~')
        self.export_path = self.home_directory+'/canoe_video_processing/line_detection_grid'

    def save_image(self, image, folder, current_time, callbacks_list=None, name=None, json_file=None):
        self.folder = folder
        self.image = image
        self.name = name
        self.json_file = json_file
        self.current_time = current_time
        self.callbacks_list = callbacks_list
        if self.callbacks_list is not None: self.x_values = list(range(len(self.callbacks_list)))

        if self.name == None: self.image_name = f'{self.current_time}'
        else: self.image_name = f'{self.name}'
        # else: self.image_name = f'{self.current_time}_{self.name}'

        self.path = f"{self.export_path}/"
        # self.path = f"{self.export_path}/{self.folder}/"

        self.image_path = self.path + f'{self.folder}/' + self.image_name + '.jpg'
        # print(self.path)
        # print(self.image_path)
        os.makedirs(os.path.dirname(self.image_path), exist_ok=True)
        cv2.imwrite(self.image_path, self.image)


        self.json_path = self.path + f'{self.folder}/'+ f'json/{self.image_name}.json'
        # self.json_path = self.path + self.image_name + '.json'
        # os.makedirs(os.path.dirname(self.json_path), exist_ok=True)


        if not os.path.exists(self.json_path) and self.json_file is not None:
            with open(self.json_path, 'w') as f:
                json.dump(self.json_file, f)

        self.callback_path = self.path + f'{self.folder}/'+ self.image_name + '.png'
        if self.callbacks_list is not None:
            plt.plot(self.x_values, self.callbacks_list, marker='o', linestyle='-', color='b', label='Callbacks')
            plt.xlabel('Wiederholung')
            plt.ylabel('Fehler')
            plt.title('Restfehler nach Optimierung')
            plt.legend()
            plt.savefig(self.callback_path)
            plt.close()


    def create_json(self, video, frame_nr, params_initial, params_result, time, protocol, loss):
        self.params_initial = params_initial
        self.params_result = params_result
        self.frame_nr = frame_nr
        self.video = video
        self.time = time
        self.protocol = protocol
        if self.protocol is None: self.protocol = [None, None, None, None, None, None, None, None, None]
        self.loss = loss
        if self.loss is None: self.loss = [None]
        self.json = {}

        self.json[f'{self.video}'] = {
            'frame' : self.frame_nr,
            'optimize': {
                'function': self.protocol[0],
                'params':{
                    'maxiter': self.protocol[5],
                    'seed': self.protocol[6]
                },
                'callback': {
                    'final_error': self.protocol[7],
                    'context': self.protocol[8]
                },
                'success': self.protocol[1],
                'message': self.protocol[2],
                'nfev': self.protocol[3],
                'nit': self.protocol[4],
                'time': self.time
            },
            'loss':{
                'alpha': self.loss[0]
            },
            'grid_params_initial': {
                'rows': self.params_initial[0],
                'cols': self.params_initial[1],
                'translation_x': self.params_initial[2],
                'translation_y': self.params_initial[3],
                'translation_z': self.params_initial[4],
                'rotation_x': self.params_initial[5],
                'rotation_y': self.params_initial[6],
                'rotation_z': self.params_initial[7],
                'focal_length': self.params_initial[8],
            },
            'grid_params_result':{
                'rows': self.params_result[0],
                'cols': self.params_result[1],
                'translation_x': self.params_result[2],
                'translation_y': self.params_result[3],
                'translation_z': self.params_result[4],
                'rotation_x'   : self.params_result[5],
                'rotation_y'   : self.params_result[6],
                'rotation_z'   : self.params_result[7],
                'focal_length': self.params_result[8],
            }
        }
        return self.json
