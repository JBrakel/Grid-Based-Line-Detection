import numpy as np
import cv2
import itertools
import matplotlib.pyplot as plt
import os
import random

import scipy.optimize as optimize
from scipy.optimize import basinhopping, differential_evolution, minimize, dual_annealing, direct
from scipy.spatial.distance import cdist


class Grid():

    def __init__(self, frame, target=False):
        self.frame = frame
        self.target = target
        self.frame_height, self.frame_width, _ = self.frame.shape

        self.thickness = 2
        self.radius = 6
        self.radius_test = 7

        self.translation_x = 0
        self.translation_y = 0
        self.translation_z = 0

        self.theta_x = 0
        self.theta_y = 0
        self.theta_z = 0
        self.theta_rotation = 0
        self.distance_z = 0
        self.rotation_matrix = []

        self.sensor_size_x_mm = 6.22
        self.sensor_size_y_mm = 3.5
        self.px_x_mm = self.frame_width / self.sensor_size_x_mm
        self.px_y_mm = self.frame_height / self.sensor_size_y_mm
        self.camera_center_x = self.frame_width/2
        self.camera_center_y = self.frame_height/2

        self.home_directory = os.path.expanduser('~')
        self.export_path = self.home_directory + '/canoe_video_processing/line_detection_grid'


        self.colors = {
            'green': {
                'line' : (0,255,0),
                'point': (0,150,0)
            },
            'red': {
                'line': (0, 0, 255),
                'point': (0, 0, 255)
            },
            'yellow': {
                'line': (0, 255, 255),
                'point': (0, 210, 210)
            },
            'gray': {
                'line': (100, 100, 100),
                'point': (100, 100, 100)
            }
        }

        self.sizes = {
            'green': {
                'line' : 10,#2,
                'point': 15#6
            },
            'red': {
                'line' : 2,
                'point': 6
            },
            'yellow': {
                'line' : 10,
                'point': 20
            },
            'gray': {
                'line' : 2,
                'point': 9
            }
        }

    def create_grid(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.single_row_list = []
        self.grid_coordinates = []

        for i in range(self.rows):
            y = int((i + 1) * self.frame_height / (self.rows+1))

            for j in range(self.cols):
                x = int((j + 1) * self.frame_width / (self.cols+1))
                self.single_row_list.append((x, y))

            self.grid_coordinates.append(self.single_row_list)
            self.single_row_list = []

        self.grid_coordinates = np.array(self.grid_coordinates)
        return self.grid_coordinates

    def find_outer_corners(self, coordinates):
        self.coordinates = coordinates
        self.min_x = np.min(self.coordinates[:, :, 0])
        self.max_x = np.max(self.coordinates[:, :, 0])
        self.min_y = np.min(self.coordinates[:, :, 1])
        self.max_y = np.max(self.coordinates[:, :, 1])

        self.top_left = [self.min_x, self.min_y]
        self.top_right = [self.max_x, self.min_y]
        self.bottom_left = [self.min_x, self.max_y]
        self.bottom_right = [self.max_x, self.max_y]

        return self.top_left, self.top_right, self.bottom_left, self.bottom_right

    def calculate_center(self, corners):
        self.corners = corners
        self.x_sum = 0
        self.y_sum = 0
        try:
            for corner in self.corners:
                self.x_sum += corner[0]
                self.y_sum += corner[1]

            self.center_x = float(self.x_sum / len(self.corners))
            self.center_y = float(self.y_sum / len(self.corners))

            self.center = np.array([self.center_x, self.center_y])
        except:
            self.center = None
        return self.center

    def draw_grid(self, image, coordinates, color='green'):
        self.image = image
        self.coordinates = coordinates
        self.clr = color
        self.color = self.colors[color]
        self.size = self.sizes[color]

        if len(self.coordinates.reshape(-1, 2)) > 0:
            self.number_horizontal_lines = self.coordinates.shape[0]
            self.number_vertical_lines = self.coordinates.shape[1]

            self.corners = self.find_outer_corners(self.coordinates)
            self.center = self.calculate_center(self.corners)

            # if self.target == False:
            for line in range(self.number_horizontal_lines):
                for point in range(self.number_vertical_lines):
                    neighbor = point + 1
                    if neighbor < len(self.coordinates[line]):
                        x1 = self.coordinates[line][point][0]
                        y1 = self.coordinates[line][point][1]
                        x2 = self.coordinates[line][neighbor][0]
                        y2 = self.coordinates[line][neighbor][1]

                        if not np.isnan(x1) and not np.isnan(y1) and not np.isnan(x2) and not np.isnan(y2):
                            cv2.line(self.image, (int(x1), int(y1)), (int(x2), int(y2)), self.color['line'], self.size['line'])

            for line in range(self.number_vertical_lines):
                for point in range(self.number_horizontal_lines):
                    neighbor = point + 1
                    if neighbor < len(self.coordinates):
                        x1 = self.coordinates[point][line][0]
                        y1 = self.coordinates[point][line][1]
                        x2 = self.coordinates[neighbor][line][0]
                        y2 = self.coordinates[neighbor][line][1]
                        if not np.isnan(x1) and not np.isnan(y1) and not np.isnan(x2) and not np.isnan(y2):
                            cv2.line(self.image, (int(x1), int(y1)), (int(x2), int(y2)), self.color['line'], self.size['line'])

            for i, (x,y) in enumerate(self.coordinates.reshape(-1, 2)):
                if not np.isnan(x) and not np.isnan(y):
                    cv2.circle(self.image, (int(x), int(y)), self.size['point'], self.color['point'], -1)

        return self.image

    def draw_grid_center(self, image, center):
        self.image = image
        self.center = center

        if self.target:
            pass
        else:
            if self.clr == 'gray':
                cv2.circle(self.image, tuple(self.center), self.radius_test, self.colors['gray']['point'],-1)
            else:
                cv2.circle(self.image, tuple(self.center), self.radius, self.colors['red']['point'], -1)

    def draw_image_center(self, image):
        self.image = image
        self.l = 50
        cv2.line(self.image, (int(self.frame_width/2-self.l), int(self.frame_height/2)),(int(self.frame_width/2+self.l),int(self.frame_height/2)),(0,100,250),15)
        cv2.line(self.image, (int(self.frame_width/2), int(self.frame_height/2-self.l)),(int(self.frame_width/2),int(self.frame_height/2+self.l)),(0,100,250),15)

    def create_grid_3D(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.distance_x = 12.5
        self.distance_y = 9.0
        self.grid_coordinates = np.zeros((self.rows, self.cols, 3))

        for row in range(self.rows):
            for col in range(self.cols):
                self.steps_x = np.arange(0, self.cols * self.distance_x, self.distance_x)
                self.grid_coordinates[row, :, 0] = self.steps_x
                self.grid_coordinates[row, :, 1] = row * self.distance_y
                self.grid_coordinates[row, :, 2] = self.distance_z

        if len(self.grid_coordinates) != 0:
            self.grid_center_x, self.grid_center_y = self.calculate_center(self.find_outer_corners(self.grid_coordinates))
            self.x_shift = -self.grid_center_x
            self.y_shift = -self.grid_center_y
            self.grid_coordinates[:, :, 0] += self.x_shift
            self.grid_coordinates[:, :, 1] += self.y_shift

        return self.grid_coordinates


class GridTransformation(Grid):
    def __init__(self, frame):
        super().__init__(frame)

    def apply_transformation(self, coordinates, matrix):
        self.grid_coordinates_transformation = coordinates
        self.grid_coordinates_transformation_copy = self.grid_coordinates_transformation.copy()
        self.matrix = matrix

        if self.grid_coordinates_transformation.size > 0:
            self.flattened_coordinates = self.grid_coordinates_transformation.reshape(-1, 2)
            self.flattened_coordinates = self.flattened_coordinates - [self.frame_width / 2 + self.translation_x, self.frame_height / 2 + self.translation_y]
            self.homogeneous_coordinates = np.hstack((self.flattened_coordinates, np.ones((self.flattened_coordinates.shape[0], 1))))
            self.grid_coordinates_transformation = np.dot(self.homogeneous_coordinates, self.matrix.T).astype(int)
            self.grid_coordinates_transformation = self.grid_coordinates_transformation[:, :2].reshape(self.grid_coordinates_transformation_copy.shape)
            self.grid_coordinates_transformation = self.grid_coordinates_transformation + [self.frame_width / 2 + self.translation_x ,self.frame_height / 2 + self.translation_y]

        return self.grid_coordinates_transformation

    def translation(self, coordinates, x=0, y=0):
        self.coordinates_translation = coordinates
        self.translation_x = x
        self.translation_y = y

        self.translation_matrix = np.array([[1, 0, self.translation_x],
                                            [0, 1, self.translation_y],
                                            [0, 0, 1]])

        self.coordinates_translation = self.apply_transformation(self.coordinates_translation, self.translation_matrix)

        return self.coordinates_translation

    def rotation(self, coordinates, theta=0):
        self.grid_coordinates_rotation = coordinates
        self.theta_rotation = np.deg2rad(theta)

        self.rotation_matrix = np.array([[np.cos(self.theta_rotation), -np.sin(self.theta_rotation), 0],
                                         [np.sin(self.theta_rotation),  np.cos(self.theta_rotation), 0],
                                         [0, 0, 1]])

        self.grid_coordinates_rotation = self.apply_transformation(self.grid_coordinates_rotation, self.rotation_matrix)

        return self.grid_coordinates_rotation

    def scalation(self, coordinates, x=1, y=1):
        self.grid_coordinates_scalation = coordinates
        self.scale_x = x
        self.scale_y = y

        self.scale_matrix = np.array([[self.scale_x, 0, 0],
                                      [0, self.scale_y, 0],
                                      [0, 0, 1]])

        self.theta_rotation = -np.rad2deg(self.theta_rotation)
        self.grid_coordinates_remove_rotation = self.rotation(self.grid_coordinates_scalation, theta=self.theta_rotation)
        self.grid_coordinates_scalation = self.apply_transformation(self.grid_coordinates_remove_rotation, self.scale_matrix)
        self.theta_rotation = -np.rad2deg(self.theta_rotation)
        self.grid_coordinates_scalation = self.rotation(self.grid_coordinates_scalation, theta=self.theta_rotation)

        return self.grid_coordinates_scalation

    def random_test_params_rows_cols(self, rows=(0,0), cols=(0,0)):
        if rows == (0,0):
            self.rows_test_param = 0
        else:
            self.rows_test_param = np.random.randint(rows[0], rows[1])

        if cols == (0,0):
            self.cols_y_test_param = 0
        else:
            self.cols_y_test_param = np.random.randint(cols[0], cols[1])
        return self.rows_test_param, self.cols_y_test_param

    def random_test_params_translation(self, x=(0,0), y=(0,0)):
        if x == (0,0):
            self.translation_x_test_param = 0
        else:
            self.translation_x_test_param = np.random.randint(x[0], x[1])

        if y == (0,0):
            self.translation_y_test_param = 0
        else:
            self.translation_y_test_param = np.random.randint(y[0], y[1])
        return self.translation_x_test_param, self.translation_y_test_param

    def random_test_params_rotation(self, theta=(0,0)):
        if theta == (0,0):
            self.rotation_test_param = 0
        else:
            self.rotation_test_param = np.random.randint(theta[0], theta[1])
        return self.rotation_test_param

    def random_test_params_scalation(self, x=(1,1), y=(1,1)):
        if x == (1,1):
            self.scalation_x_test_param = 1
        else:
            self.scalation_x_test_param = np.random.uniform(x[0], x[1])

        if y == (1,1):
            self.scalation_y_test_param = 1
        else:
            self.scalation_y_test_param = np.random.uniform(y[0], y[1])
        return self.scalation_x_test_param, self.scalation_y_test_param

    ## Rotation um die Achse des Grids
    def apply_transformation_3D(self, coordinates, matrix):
        self.grid_coordinates_transformation = coordinates
        self.grid_coordinates_transformation_copy = self.grid_coordinates_transformation.copy()
        self.matrix = matrix

        if self.grid_coordinates_transformation.size > 0:
            self.flattened_coordinates = self.grid_coordinates_transformation.reshape(-1, 3)
            self.flattened_coordinates = self.flattened_coordinates - [self.translation_x, self.translation_y, self.translation_z+self.distance_z]
            self.homogeneous_coordinates = np.hstack((self.flattened_coordinates, np.ones((self.flattened_coordinates.shape[0], 1))))
            self.grid_coordinates_transformation = np.dot(self.homogeneous_coordinates, self.matrix.T)
            self.grid_coordinates_transformation = self.grid_coordinates_transformation[:, :3].reshape(self.grid_coordinates_transformation_copy.shape)
            self.grid_coordinates_transformation = self.grid_coordinates_transformation + [self.translation_x, self.translation_y, self.translation_z+self.distance_z]
        return self.grid_coordinates_transformation


    def translation_3D(self, coordinates, x=0, y=0, z=0):
        self.coordinates_translation = coordinates
        self.translation_x = x
        self.translation_y = y
        self.translation_z = z

        self.translation_matrix = np.array([[1, 0, 0, self.translation_x],
                                            [0, 1, 0, self.translation_y],
                                            [0, 0, 1, self.translation_z],
                                            [0, 0, 0, 1]])

        self.coordinates_translation = self.apply_transformation_3D(self.coordinates_translation,
                                                                    self.translation_matrix)

        return self.coordinates_translation

    def rotation_3D(self, grid_coordinates_rotation, theta_x, theta_y, theta_z):
        self.grid_coordinates_rotation = grid_coordinates_rotation

        self.theta_x = theta_x
        self.theta_y = theta_y
        self.theta_z = theta_z

        self.cos_x, self.sin_x = np.cos(np.deg2rad(self.theta_x)), np.sin(np.deg2rad(self.theta_x))
        self.cos_y, self.sin_y = np.cos(np.deg2rad(self.theta_y)), np.sin(np.deg2rad(self.theta_y))
        self.cos_z, self.sin_z = np.cos(np.deg2rad(self.theta_z)), np.sin(np.deg2rad(self.theta_z))

        self.rotation_x = np.array([[1, 0, 0, 0],
                                    [0, self.cos_x, -self.sin_x, 0],
                                    [0, self.sin_x, self.cos_x, 0],
                                    [0, 0, 0, 1]])

        self.rotation_y = np.array([[self.cos_y, 0, self.sin_y, 0],
                                    [0, 1, 0, 0],
                                    [-self.sin_y, 0, self.cos_y, 0],
                                    [0, 0, 0, 1]])

        self.rotation_z = np.array([[self.cos_z, -self.sin_z, 0, 0],
                                    [self.sin_z, self.cos_z, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])

        self.rotation_matrix = np.dot(self.rotation_z, np.dot(self.rotation_y, self.rotation_x))
        self.grid_coordinates_rotation = self.apply_transformation_3D(self.grid_coordinates_rotation,self.rotation_matrix)
        return self.grid_coordinates_rotation

    def projection_3D(self, coordinates, focal_length):
        self.grid_coordinates_projection = coordinates
        self.focal_length = focal_length
        self.shape_2d = (self.grid_coordinates_projection.shape[0], self.grid_coordinates_projection.shape[1], 2)

        self.projection_matrix = np.array([[self.px_x_mm * self.focal_length, 0, self.camera_center_x, 0],
                                           [0, self.px_y_mm * self.focal_length, self.camera_center_y, 0],
                                           [0, 0, 1, 0]])

        self.flattened_coordinates = self.grid_coordinates_projection.reshape(-1, 3)
        self.coordinates_3d_homogeneous = np.hstack((self.flattened_coordinates, np.ones((self.flattened_coordinates.shape[0], 1))))
        self.coordinates_2d_homogeneous = np.dot(self.projection_matrix, self.coordinates_3d_homogeneous.T).T
        self.coordinates_2d = self.coordinates_2d_homogeneous[:, :2] / self.coordinates_2d_homogeneous[:, 2:]
        self.coordinates_2d = self.coordinates_2d.reshape(self.shape_2d)
        return self.coordinates_2d

    def delete_false_coords_z_distance(self, grid_coordinates):
        self.grid_coordinates = grid_coordinates
        self.grid_coordinates[np.where(self.grid_coordinates[:, :, 2] <= 0)] = np.nan
        self.all_nan_rows = np.all(np.isnan(self.grid_coordinates), axis=(1, 2))
        self.grid_coordinates = self.grid_coordinates[~self.all_nan_rows]
        return  self.grid_coordinates


    def random_test_params_translation_3D(self, x=(0,0), y=(0,0), z=(0,0)):
        if x == (0,0):
            self.translation_x_test_param = 0
        else:
            self.translation_x_test_param = np.random.randint(x[0], x[1])

        if y == (0,0):
            self.translation_y_test_param = 0
        else:
            self.translation_y_test_param = np.random.randint(y[0], y[1])

        if z == (0,0):
            self.translation_z_test_param = 0
        else:
            self.translation_z_test_param = np.random.randint(z[0], z[1])
        return self.translation_x_test_param, self.translation_y_test_param, self.translation_z_test_param


    def random_test_params_rotation_3D(self, theta_x=(0,0), theta_y=(0,0), theta_z=(0,0)):
        if theta_x == (0,0):
            self.theta_x_test_param = 0
        else:
            self.theta_x_test_param = np.random.randint(theta_x[0], theta_x[1])

        if theta_y == (0,0):
            self.theta_y_test_param = 0
        else:
            self.theta_y_test_param = np.random.randint(theta_y[0], theta_y[1])

        if theta_z == (0,0):
            self.theta_z_test_param = 0
        else:
            self.theta_z_test_param = np.random.randint(theta_z[0], theta_z[1])
        return self.theta_x_test_param, self.theta_y_test_param, self.theta_z_test_param


class GridOptimization(Grid):

    callbacks_list = []
    def __init__(self, frame):
        super().__init__(frame)

    def calc_params_translation(self, center):
        self.grid_target_center = center
        self.translation_x, self.translation_y = self.grid_target_center - [int(self.frame_width / 2), int(self.frame_height / 2)]
        return self.translation_x, self.translation_y

    def calc_distance_matrix(self, grid_coordinates, grid_target):
        self.grid_coordinates = grid_coordinates
        self.grid_target = grid_target
        self.distance_matrix = cdist(self.grid_coordinates.reshape(-1, 2), self.grid_target.reshape(-1, 2))
        return self.distance_matrix

    def loss_function(self, params, grid_coordinates_target):
        self.rows, self.cols, self.translation_x, self.translation_y, self.translation_z, self.theta_x, self.theta_y, self.theta_z, self.focal_length = params
        self.rows, self.cols = int(self.rows), int(self.cols)
        self.grid_coordinates_target = grid_coordinates_target

        self.grid = Grid(self.frame)
        self.grid_transformation = GridTransformation(self.frame)
        self.grid_coordinates_3D = self.grid.create_grid_3D(self.rows, self.cols)
        self.grid_coordinates_3D = self.grid_transformation.translation_3D(self.grid_coordinates_3D, x=self.translation_x, y=self.translation_y, z=self.translation_z)
        self.grid_coordinates_3D = self.grid_transformation.rotation_3D(self.grid_coordinates_3D, theta_x=self.theta_x, theta_y=self.theta_y, theta_z=self.theta_z)
        self.grid_coordinates_2D = self.grid_transformation.projection_3D(self.grid_coordinates_3D, focal_length=self.focal_length)

        self.grid_size = self.rows * self.cols
        self.grid_size_target = np.prod(self.grid_coordinates_target.shape[:-1])
        self.distance_matrix = self.calc_distance_matrix(self.grid_coordinates_2D, self.grid_coordinates_target)

        self.min_distances = np.min(self.distance_matrix, axis=1)
        self.sum_min_distances = np.sum(self.min_distances, axis=0)
        self.error_1 = 1/self.grid_size * self.sum_min_distances

        # Penalty: Unterschied der Anzahl der Bojen zwischen beiden Grids
        # self.alpha = 0.1
        # self.difference_grid_size_target_size = np.abs(self.grid_size-self.grid_size_target)
        # self.penalty = self.alpha*self.difference_grid_size_target_size
        # self.error_2 = self.penalty

        # Penalty 2: Bestrafe, wenn der Abstand nicht zu 6 unterschiedlichen Bojen berechnet wird
        self.alpha = 0.1
        self.min_distance_indices = np.argmin(self.distance_matrix, axis=1)
        self.unique_indices = np.unique(self.min_distance_indices)
        self.num_unique_indices = len(self.unique_indices)
        self.difference_grid_size_target_size = np.abs(self.grid_size-self.num_unique_indices)
        self.penalty = self.alpha * self.difference_grid_size_target_size
        self.error_2 = self.penalty

        self.total_error = self.error_1 + self.error_2
        return self.total_error

    def minimize_loss_function_dual_annealing(self, initial_params, grid_coordinates_target, max_iter, bounds=None):
        self.grid_coordinates_target = grid_coordinates_target
        self.initial_params = initial_params

        if bounds != None:
            self.bounds = bounds
        self.callbacks_list = []
        self.final_callback = [None, None]

        self.maxiter = max_iter
        self.seed = 13 #random.randint(0, 1000)
        self.x0 = self.initial_params
        self.function_params = [self.maxiter, self.seed]


        self.result = dual_annealing(
            self.loss_function,
            bounds=self.bounds,
            args=(self.grid_coordinates_target,),
            maxiter=self.maxiter,
            callback=self.callback,
            seed=self.seed,
            x0=self.x0,
        )

        self.optimal_grid_parameters = self.result.x
        self.result_error = self.result.fun
        self.rows, self.cols, self.translation_x, self.translation_y, self.translation_z, self.theta_x, self.theta_y, self.theta_z, self.focal_length = self.optimal_grid_parameters
        self.rows, self.cols = int(self.rows), int(self.cols)

        self.result_params = [self.rows,
                              self.cols,
                              self.translation_x,
                              self.translation_y,
                              self.translation_z,
                              self.theta_x,
                              self.theta_y,
                              self.theta_z,
                              self.focal_length]
        # print('')
        self.protocol = self.save_protocol('dual_annealing', self.result, self.function_params, self.final_callback)
        self.loss_function_params = [self.alpha]

        return self.result_params + [self.protocol, self.loss_function_params]


    def save_protocol(self,function, result, function_params, final_callback):
        self.function = function
        self.result = result
        self.function_params = function_params
        self.final_callback = final_callback

        # print('function :', self.function)
        # print('success :', self.result.success)
        # print('message:', self.result.message)
        # print('nfev :', self.result.nfev)
        # print('nit :', self.result.nit)
        return [self.function, self.result.success, self.result.message[0], self.result.nfev, self.result.nit] + self.function_params + self.final_callback


    def callback(self, x, f, context):
        self.legend = {
            '0': 'minimum detected in the annealing process.',
            '1': 'detection occurred in the local search process.',
            '2': 'detection done in the dual annealing process.'
        }
        # print(f"Loss: {f:.5f} | context: {self.legend[f'{context}']}")
        GridOptimization.callbacks_list.append(f)
        self.final_callback = [f, self.legend[f'{context}']]


    def return_callback(self):
        return self.final_callback

    def return_callbacks(self):
        return GridOptimization.callbacks_list

    def empty_list_callbacks(self):
        GridOptimization.callbacks_list = []

    def plot_loss(self, callbacks_list, path, current_time, name=None):
        self.callbacks_list = callbacks_list
        self.x_values = list(range(len(self.callbacks_list)))
        self.current_time = current_time
        self.path = path
        self.name = name

        if self.name == None: self.image_name = f'{self.current_time}'
        else: self.image_name = f'{self.current_time}_{self.name}'

        plt.plot(self.x_values, self.callbacks_list, marker='o', linestyle='-', color='b',label='Callbacks')
        plt.xlabel('Wiederholung')
        plt.ylabel('Fehler')
        plt.title('Restfehler nach Optimierung')
        plt.legend()
        plt.show()
