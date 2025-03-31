import numpy as np
import cv2
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

class Metric:

    def __init__(self):
        pass

    def get_slope(self, single_line):
        self.dif_y = single_line[1, 1] - single_line[0, 1]
        self.dif_x = single_line[1, 0] - single_line[0, 0]
        if self.dif_x == 0: self.dif_x = 0.000001
        self.slope = self.dif_y/self.dif_x
        return self.slope

    def get_intercept(self, single_line, slope):
        self.x, self.y = single_line[0]
        self.slope = slope
        self.intercept = self.y-self.slope*self.x
        return self.intercept

    def draw_image_borders(self, frame, th):
        self.frame_height = frame.shape[0]
        self.frame_width = frame.shape[1]

        self.image_top_line = np.array([[0, 0],
                                        [self.frame_width, 0]])

        self.image_bottom_line = np.array([[0, self.frame_height],
                                           [self.frame_width, self.frame_height]])

        self.image_left_line = np.array([[0, 0],
                                         [0, self.frame_height]])

        self.image_right_line = np.array([[self.frame_width, 0],
                                          [self.frame_width, self.frame_height]])

        self.image_borders = np.array([self.image_top_line,
                                       self.image_bottom_line,
                                       self.image_left_line,
                                       self.image_right_line])

        for point1, point2 in self.image_borders:
            x1, y1 = int(point1[0]), int(point1[1])
            x2, y2 = int(point2[0]), int(point2[1])
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 0), th)

        return self.image_borders


    def calc_intersection_points(self, frame, line1, line_length=None, squared=False):
        self.slope1 = self.get_slope(line1)
        self.intercept1 = self.get_intercept(line1, self.slope1)
        self.frame_height = frame.shape[0]
        self.frame_width = frame.shape[1]

        if squared is True: self.frame_height = self.frame_width

        if line_length is not None:
            [self.y_min, self.y_max] = line_length[0], line_length[1]
        else:
            self.y_min = 0
            self.y_max = self.frame_height


        self.x_top = (self.y_min-self.intercept1)/self.slope1
        self.x_bottom = (self.y_max-self.intercept1)/self.slope1
        self.y_left = self.intercept1
        self.y_right = self.slope1*self.frame_width+self.intercept1

        self.intersection_points = np.array([[self.x_top, self.y_min],
                                             [self.x_bottom, self.y_max],
                                             [0, self.y_left],
                                             [self.frame_width, self.y_right]])

        self.indices = np.where((self.intersection_points[:, 0] >= 0) &
                                (self.intersection_points[:, 0] <= self.frame_width) &
                                (self.intersection_points[:, 1] >= self.y_min) &
                                (self.intersection_points[:, 1] <= self.y_max))

        self.intersection_points = self.intersection_points[self.indices]
        self.intersection_points = np.unique(self.intersection_points, axis=0)

        return self.intersection_points


    def get_final_lines(self, grid_coordinates_result_final_2D, frame, line_length=None):
        self.final_lines = []
        self.grid_coordinates_result_final_2D = grid_coordinates_result_final_2D
        self.nr_lines = self.grid_coordinates_result_final_2D.shape[1]

        print(self.grid_coordinates_result_final_2D)

        for i in range(self.nr_lines):
            self.single_line_all_cords = self.grid_coordinates_result_final_2D[:, i, :]
            # self.single_line = self.single_line_all_cords[[0, -1]]
            self.single_line = self.single_line_all_cords[[0, 1]]

            if line_length is not None:
                self.intersection_points = self.calc_intersection_points(frame,self.single_line, line_length)
            else:
                self.intersection_points = self.calc_intersection_points(frame,self.single_line)
            if len(self.intersection_points) != 0:
                self.final_lines.append(self.intersection_points)
        self.final_lines = np.array(self.final_lines)
        return self.final_lines

    def get_final_labeled_lines(self, labeled_lines, frame):
        self.final_labeled_lines = []
        self.labeled_lines = labeled_lines

        for line in self.labeled_lines:
            self.intersection_points = self.calc_intersection_points(frame, line)

            if len(self.intersection_points) != 0:
                self.final_labeled_lines.append(self.intersection_points)
        # print(self.final_labeled_lines)

        self.final_labeled_lines = np.array(self.final_labeled_lines)
        return self.final_labeled_lines

    def draw_final_lines(self, frame, final_lines, labeled_lines=None, th=15, label=False):
        self.frame = frame

        if label==True and labeled_lines is not None:
            for point1, point2 in labeled_lines:
                x1,y1 = int(point1[0]), int(point1[1])
                x2,y2 = int(point2[0]), int(point2[1])
                cv2.line(frame, (x1,y1), (x2,y2), (0,0,255), int(th*1.5))

        for point1, point2 in final_lines:
            x1,y1 = int(point1[0]), int(point1[1])
            x2,y2 = int(point2[0]), int(point2[1])
            cv2.line(frame, (x1,y1), (x2,y2), (0,255,0), th)
        return self.frame

    def calc_distance_to_line(self, point, line):
        self.x, self.y = point
        [self.a, self.b, self.c] = line
        self.distance = abs(self.a * self.x + self.b * self.y + self.c) / np.sqrt(self.a ** 2 + self.b ** 2)

        return self.distance

    def get_line_length(self, grid_coordinates_real_buoys, final_lines):
        self.grid_coordinates_real_buoys = grid_coordinates_real_buoys
        self.final_lines = final_lines
        self.y_max_list = []
        self.y_min_list = []
        self.max_distance = 30

        for line in self.final_lines[:]:
            self.slope = self.get_slope(line)
            self.intercept = self.get_intercept(line, self.slope)
            self.a = -self.slope
            self.b = 1
            self.c = -self.intercept
            self.distances = []
            for point in self.grid_coordinates_real_buoys:
                self.distance = self.calc_distance_to_line(point,[self.a, self.b, self.c])
                if self.distance <= self.max_distance:
                    self.distances.append(point)
            self.distances = np.array(self.distances)
            if len(self.distances) > 1:
                self.y_min = np.min(self.distances[:,1],axis=0)
                self.y_max = np.max(self.distances[:,1],axis=0)
                self.y_min_list.append(self.y_min)
                self.y_max_list.append(self.y_max)

        self.y_min_final = np.median(self.y_min_list)
        self.y_max_final = np.median(self.y_max_list)

        self.y_min_final = np.min(self.y_min_list)
        self.y_max_final = np.max(self.y_max_list)
        return [self.y_min_final, self.y_max_final]

    def calc_angle_two_lines(self, final_lines_normalized, labeled_line_normalized):
        self.final_lines_normalized = final_lines_normalized
        self.labeled_line_normalized = labeled_line_normalized
        self.angles_rad = []

        for line in self.final_lines_normalized:
            slope_label = self.get_slope(self.labeled_line_normalized)
            slope_final = self.get_slope(line)
            if np.isinf(slope_label): slope_label = 10**10
            if np.isinf(slope_final): slope_final = 10**10
            intercept_label = self.get_intercept(self.labeled_line_normalized, slope_label)
            intercept_final = self.get_intercept(line, slope_final)
            numerator = slope_final-slope_label
            denominator =  1+slope_label*slope_final
            tan_angle = abs(numerator/denominator)
            angle_rad = np.arctan(tan_angle)
            self.angles_rad.append(angle_rad)
        self.angles_rad = np.array(self.angles_rad)
        return self.angles_rad

    def calc_midpoints(self, lines_normalized):
        lines_normalized = lines_normalized.reshape(-1,2,2)
        self.midpoints = []
        for i in range(lines_normalized.shape[0]):
            mean_x = np.mean(lines_normalized[i, :, 0])
            mean_y = np.mean(lines_normalized[i, :, 1])
            self.midpoints.append([mean_x, mean_y])
        self.midpoints = np.array(self.midpoints)
        return self.midpoints

    def calc_score_angles(self, angles_rad):
        self.angles_rad = angles_rad
        self.denominator = np.pi/2
        self.score_angles_rad = 1 - (self.angles_rad/self.denominator)
        return self.score_angles_rad

    def calc_midpoints_distances(self, midpoints, midpoint_single_labeled):
        self.midpoints_distances = []
        self.midpoint_single_labeled = midpoint_single_labeled
        for midpoint in midpoints:
            x1, y1 = self.midpoint_single_labeled
            x2, y2 = midpoint
            distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            self.midpoints_distances.append(distance)
        self.midpoints_distances = np.array(self.midpoints_distances)
        return self.midpoints_distances

    def calc_score_midpoints_distances(self, midpoints_distances):
        self.midpoints_distances = midpoints_distances
        self.score_midpoints_distances = 1 - self.midpoints_distances
        return self.score_midpoints_distances

    def calc_final_score(self, score_angles_rad, score_midpoints_distances):
        self.score_angles_rad = score_angles_rad
        self.score_midpoints_distances = score_midpoints_distances
        self.final_scores = (self.score_angles_rad*self.score_midpoints_distances)**2
        return self.final_scores

    def display_metrics_squared(self, frame, metrics_all_params, th=10, r=20):
        self.frame = frame
        self.frame_height = self.frame.shape[0]
        self.frame_width = self.frame.shape[1]

        [self.final_lines_normalized,
         self.labeled_lines_normalized,
         self.midpoints,
         self.midpoints_labeled,
         self.midpoints_distances,
         self.score_midpoints_distances,
         self.angles_rad,
         self.score_angles_rad,
         self.final_scores] = metrics_all_params

        self.final_lines_squared = self.final_lines_normalized * [self.frame_width, self.frame_width]
        self.labeled_lines_squared = self.labeled_lines_normalized * [self.frame_width, self.frame_width]
        self.midpoints_squared = self.midpoints * [self.frame_width, self.frame_width]
        self.midpoints_labeled_squared = self.midpoints_labeled * [self.frame_width, self.frame_width]

        self.image_squared = np.ones_like(self.frame) * 255
        self.image_squared = cv2.resize(self.image_squared, (self.frame_width, self.frame_width))

        for point1, point2 in self.labeled_lines_squared:
            x1, y1 = int(point1[0]), int(point1[1])
            x2, y2 = int(point2[0]), int(point2[1])
            cv2.line(self.image_squared, (x1, y1), (x2, y2), (0, 0, 255), th)

        for point1, point2 in self.final_lines_squared:
            x1, y1 = int(point1[0]), int(point1[1])
            x2, y2 = int(point2[0]), int(point2[1])
            cv2.line(self.image_squared, (x1, y1), (x2, y2), (0, 255, 0), th)

        for (x1, y1), (x2, y2) in zip(self.midpoints_squared, self.midpoints_labeled_squared):
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.line(self.image_squared, (x1, y1), (x2, y2), (0, 150, 255), th)
            cv2.circle(self.image_squared, (x1, y1), r, (0, 255, 0), -1)
            cv2.circle(self.image_squared, (x2, y2), r, (0, 0, 255), -1)

        self.draw_image_borders(self.image_squared, int(th*1.5))
        return self.image_squared


    def create_box_plots(self, all_scores, positions, different_colors=True):
        self.bp = plt.boxplot(all_scores, positions=positions, widths=0.25, patch_artist=True)  #
        if different_colors == True:
            self.colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'brown'] * 2
            for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
                plt.setp(self.bp[element], color='black', linewidth=1.5)
            for i, patch in enumerate(self.bp['boxes']):
                patch.set(facecolor=self.colors[i])
        else:
            for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
                plt.setp(self.bp[element], color='blue', linewidth=1.5)
            for patch in self.bp['boxes']:
                patch.set(facecolor='cyan')

        return self.bp

