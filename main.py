import numpy as np
import cv2
import matplotlib.pyplot as plt
import itertools
import time
import warnings
import pickle
import os
import json
import datetime
import random
from scipy.spatial.distance import cdist
from matplotlib.ticker import FormatStrFormatter

from output import Output
from grid import Grid, GridTransformation, GridOptimization
from video_import import VideoImport
from data import Data
from metric import Metric


warnings.filterwarnings("ignore", category=RuntimeWarning)

if __name__ == '__main__':

    #### Settings True False
    methods = ['initial_grid', 'spatio_temporal_grid']
    method = methods[1]
    optimize = True
    metrics = True
    save_images = True
    save_json = False
    save_histogram = False
    create_boxplots = False
    calc_percentage = False

    display_images = False
    frame_time = 1000000
    max_iter = 2500

    # Erfassen der Monitorgröße -> Zentrieren des Outputs
    output = Output()
    window_size = output.get_window_size()
    metric = Metric()
    centered_x, centered_y, centered_x2, centered_y2 = output.center_output_two_windows()
    home_directory = os.path.expanduser('~')
    export_path = home_directory + '/canoe_video_processing/line_detection_grid/'
    current_date = datetime.datetime.now().strftime('%m-%d')
    current_time = datetime.datetime.now().strftime('%H-%M-%S')
    json_file_percentage = {}

    videos = [
        # ('duisburg',1), # 0
        ('duisburg',3), # 0
        ('duisburg',4), # 1
        ('duisburg',12),# 2
        ('duisburg',13),# 3
        ('duisburg',15),# 4
        ('duisburg',16),# 5
        ('duisburg',19),# 6
        ('duisburg',20),# 7
    ]
    n=0
    m=0

    for video in videos[n:m+1]:
        times_list = []
        number_frames_left_and_right = 30
        count_forward = count_backward = number_frames_left_and_right

        city = video[0]
        race = video[1]

        save_data = Data()
        video_import = VideoImport(city, race)
        video_path = video_import.get_video_path()

        # txt_path = video_import.get_txt_path()
        json_path = video_import.get_json_path()
        txt_path_full_video = video_import.get_txt_path_full_video()
        json_path = video_import.get_json_path_yolo()
        labeled_lines_path = video_import.get_path_labeled_lines()
        # json_file = video_import.create_json_from_new_videos(txt_path_full_video)

        output_1 = output_2 = None

        if not os.path.exists(json_path) and json_file is not None:
            with open(json_path, 'w') as output_json_file:
                json.dump(json_file, output_json_file, indent=4)

        if json_path is not None and os.path.isfile(json_path):
            with open(json_path) as json_file:
                data = json.load(json_file)
                img_metadata = data['_via_img_metadata']

            frame_annotation_numbers = []
            for key, value in img_metadata.items():
                filename = value['filename']
                frame_number = int((filename.split('.')[0]).split('_')[1])
                frame_annotation_numbers.append(frame_number)
        else:
            print("No JSON")
            pass

        # Starte Video
        cap = cv2.VideoCapture(video_path)

        # Wähle Start und Stop Position
        best_start_position = video_import.get_best_start_position()
        start_position = best_start_position
        stop_position = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1

        # Setze cap auf die Startposition
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_position)

        print(city, race)

        initial_params_C0003 = [3, 2, 0.2645823146857337, 0.6368770046691009, 77.85707986868124, -85.83579463260726, -2.7940498246119345, -0.9676735270924943, 14.568856668358753]
        initial_params_C0004 = [3, 2, -0.546995590142087, 0.12166153353046935, 95.51725488620079, -86.26992288761882, -2.6713328976288184, -0.9429009017723979, 23.361092650955637]
        initial_params_C0012 = [3, 2, 1.5101791371945725, 2.5276547504106346, 77.54766090239302, -87.24261955980856, -1.9188608714280884, -1.9237223171032762, 8.972969711429414]
        initial_params_C0013 = [3, 2, -0.0861075554569673, 2.587168737595646, 69.6813933411394, -86.866292841835, -3.428138759179636, -1.9840479662281116, 7.893220585637976]
        initial_params_C0015 = [3, 2, 0.025866488834624463, 3.6633234972253526, 69.23911589537764, -87.76420732744765, 1.3136095061574402, 1.281103505489852, 10.139586207846092]
        initial_params_C0016 = [3, 2, -0.4907432756179111, 3.9008113670362916, 69.58131741960278, -87.93672141614823, 0.9236602837388362, 1.2888081776230205, 7.894543738122084]
        initial_params_C0019 = [3, 2, 1.6812454651767392, -1.7825975756664079, 79.21139298039432, -84.00736043808142, 2.2733634296695033, 1.3594695578226026, 9.828005545639085]
        initial_params_C0020 = [3, 2, 1.217636595943642, -2.305956250673066, 86.52656778194402, -84.16738599772624, 1.795305824344798, 1.343026383807374, 10.117012828551655]

        initial_params_list = [
            initial_params_C0003,
            initial_params_C0004,
            initial_params_C0012,
            initial_params_C0013,
            initial_params_C0015,
            initial_params_C0016,
            initial_params_C0019,
            initial_params_C0020,
        ]
        initial_params_best_frame_position = initial_params_list[n]
        n+=1

        if method == 'initial_grid':
            initial_params_best_frame_position = [3, 2, -0.3164185679395755, 1.67202147412031, 96.40281110810106, -87.14223648813876,-2.587900138273654, -0.9454941257806765, 18.53377037862769]
            # initial_params_best_frame_position = [3, 2, -0.5065090524784533, 1.6782814653400882, 96.40148764238289, -87.14551007623075, -2.657159236647531, -0.9520989701520921, 18.543538868910662]

        initial_params = initial_params_best_frame_position


        txl, txu = 200, 200
        tyl, tyu = 200, 200
        tzl, tzu = 40, 40
        rxl, rxu = 90 + initial_params[5], 30
        ryl, ryu = 5, 5
        rzl, rzu = 1, 1
        fll, flu = 15,15

        play_backwards = False
        count_labeled_frames = 0

        final_scores_dict = {}

        while(cap.isOpened()):
            ret, frame = cap.read()

            if ret == True:
                current_frame_position = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                all_labeled_buoys = {}

                if json_path is not None and os.path.isfile(json_path):
                    for number in frame_annotation_numbers:
                        if number == current_frame_position:

                            print(f'*** frame_{current_frame_position} ***')

                            # Erfasse Bildabmessungen
                            frame_height = frame.shape[0]
                            frame_width = frame.shape[1]

                            grid_optimization = GridOptimization(frame)
                            grid_transformation = GridTransformation(frame)
                            grid_coordinates_real_buoys = video_import.get_frame_annotation(frame,number,img_metadata)
                            frame_copy = frame.copy()
                            frame_copy2 = frame.copy()

                            txl, txu = 200, 200
                            tyl, tyu = 200, 200
                            tzl, tzu = 40, 40

                            rxl, rxu =  90+initial_params[5], 20
                            ryl, ryu =  10, 10
                            rzl, rzu =  1, 1

                            fll, flu = 5, 5

                            bounds = [(3, 3.0000001), (2, 2.0000001),
                                          (initial_params[2] - txl, initial_params[2] + txu),
                                          (initial_params[3] - tyl, initial_params[3] + tyu),
                                          (initial_params[4] - tzl, initial_params[4] + tzu),
                                          (initial_params[5] - rxl, initial_params[5] + rxu),
                                          (initial_params[6] - ryl, initial_params[6] + ryu),
                                          (initial_params[7] - rzl, initial_params[7] + rzu),
                                          (initial_params[8] - fll, initial_params[8] + flu)]

                            # Start Zeitmessung
                            start_time = time.time()

                            # Berechnung und Minimierung der Fehlerfunkion
                            protocol, loss = None, None
                            grid_coordinates_target = grid_coordinates_real_buoys
                            [rows, cols, translation_x, translation_y, translation_z, theta_x, theta_y, theta_z, focal_length] = initial_params

                            initial_params_start = initial_params
                            if optimize == True:
                                for run in range(5):
                                    print('run', run)

                                    [rows, cols, translation_x, translation_y, translation_z, theta_x, theta_y, theta_z, focal_length, protocol, loss_function_params] = grid_optimization.minimize_loss_function_dual_annealing(
                                        initial_params, grid_coordinates_target, max_iter, bounds
                                    )
                                    final_error = protocol[6]
                                    final_error = grid_optimization.return_callback()[0]
                                    if final_error == None: final_error = 11
                                    if final_error <= 5:
                                        print('error', final_error)
                                        print('')
                                        break
                                    else:
                                        random_multipliers = [random.uniform(-0.1, 0.1) for _ in range(len(initial_params_start))]
                                        initial_params = [param + random_multiplier for param, random_multiplier in
                                                      zip(initial_params_start, random_multipliers)]
                                        initial_params[0] = 3
                                        initial_params[1] = 2
                                        print('error', final_error)
                                        print('')
                            result_params = [rows, cols, translation_x, translation_y, translation_z, theta_x, theta_y, theta_z, focal_length]
                            print(result_params)

                            # Ende Zeitmessung
                            end_time = time.time()
                            elapsed_time = end_time - start_time
                            times_list.append(elapsed_time)


                            # Erstelle json
                            data_json = save_data.create_json(f'{city}_C000{race}', current_frame_position, initial_params, result_params, elapsed_time, protocol, loss)

                            ## Erzeuge Detektions Grid
                            # Zeichne 'kleines Grid (3x2) zur Detektion
                            grid_result = Grid(frame)
                            grid_coordinates_result_3D = grid_result.create_grid_3D(rows=rows, cols=cols)
                            grid_coordinates_result_3D = grid_transformation.translation_3D(grid_coordinates_result_3D, x=translation_x, y=translation_y, z=translation_z)
                            grid_coordinates_result_3D = grid_transformation.rotation_3D(grid_coordinates_result_3D, theta_x=theta_x, theta_y=theta_y, theta_z=theta_z)
                            grid_coordinates_result_2D = grid_transformation.projection_3D(grid_coordinates_result_3D, focal_length=focal_length)

                            # Zeichne 'großes' Grid als Endergebnis
                            grid_result_final = Grid(frame)
                            rows_final = 11
                            cols_final = 4
                            grid_coordinates_result_final_3D = grid_result_final.create_grid_3D(rows=rows_final, cols=cols_final)
                            grid_coordinates_result_final_3D = grid_transformation.translation_3D(grid_coordinates_result_final_3D, x=translation_x, y=translation_y, z=translation_z)
                            grid_coordinates_result_final_3D = grid_transformation.rotation_3D(grid_coordinates_result_final_3D, theta_x=theta_x, theta_y=theta_y, theta_z=theta_z)
                            grid_coordinates_result_final_3D = grid_transformation.delete_false_coords_z_distance(grid_coordinates_result_final_3D)
                            grid_coordinates_result_final_2D = grid_transformation.projection_3D(grid_coordinates_result_final_3D, focal_length)


                            final_lines = grid_coordinates_result_final_2D
                            line_length = [850, 2100]
                            final_lines_extended = metric.get_final_lines(grid_coordinates_result_final_2D, frame, line_length)
                            final_lines_extended = final_lines_extended


                            if metrics == True:
                                labeled_lines = None
                                labeled_lines = video_import.import_labeled_lines(labeled_lines_path, current_frame_position-1)
                                labeled_lines_extended = labeled_lines
                                labeled_lines_extended = metric.get_final_labeled_lines(labeled_lines, frame)
                                final_lines_normalized   = final_lines_extended /[frame_width, frame_height]
                                labeled_lines_normalized = labeled_lines_extended/[frame_width, frame_height]

                                final_lines_normalized_selected = []
                                final_labeled_lines_normalized = []
                                final_midpoints = []
                                final_midpoints_labeled = []
                                final_midpoints_distances = []
                                final_score_midpoints_distances = []
                                final_angles_rad = []
                                final_score_angles_rad = []
                                final_scores = []
                                idx_lines = []
                                max_scores = []

                                for i, line in enumerate(labeled_lines_normalized):

                                    midpoint_single_labeled = metric.calc_midpoints(line)[0]
                                    midpoints = metric.calc_midpoints(final_lines_normalized)
                                    midpoints_distances = metric.calc_midpoints_distances(midpoints, midpoint_single_labeled)
                                    score_midpoints_distances = metric.calc_score_midpoints_distances(midpoints_distances)
                                    angles_rad = metric.calc_angle_two_lines(final_lines_normalized, line)
                                    score_angles_rad = metric.calc_score_angles(angles_rad)
                                    scores = metric.calc_final_score(score_angles_rad, score_midpoints_distances)
                                    max_score = np.argmax(scores)

                                    best_line = final_lines_normalized[max_score]
                                    labeled_line = line
                                    best_midpoint = midpoints[max_score]
                                    midpoint_single_labeled = midpoint_single_labeled
                                    best_midpoint_distance = midpoints_distances[max_score]
                                    best_score_midpoint_distances = score_midpoints_distances[max_score]
                                    best_angles_rad = angles_rad[max_score]
                                    best_score_angles = score_angles_rad[max_score]
                                    best_score = scores[max_score]

                                    final_lines_normalized_selected.append(best_line)
                                    final_labeled_lines_normalized.append(labeled_line)
                                    final_midpoints.append(best_midpoint)
                                    final_midpoints_labeled.append(midpoint_single_labeled)
                                    final_midpoints_distances.append(best_midpoint_distance)
                                    final_score_midpoints_distances.append(best_score_midpoint_distances)
                                    final_angles_rad.append(best_angles_rad)
                                    final_score_angles_rad.append(best_score_angles)
                                    final_scores.append(best_score)
                                    idx_lines.append(max_score)
                                    max_scores.append(max_score)

                                final_lines_normalized_selected = np.array(final_lines_normalized_selected)
                                final_labeled_lines_normalized = np.array(final_labeled_lines_normalized)
                                final_midpoints = np.array(final_midpoints)
                                final_midpoints_labeled = np.array(final_midpoints_labeled)
                                final_midpoints_distances = np.array(final_midpoints_distances)
                                final_score_midpoints_distances = np.array(final_score_midpoints_distances)
                                final_angles_rad = np.array(final_angles_rad)
                                final_score_angles_rad = np.array(final_score_angles_rad)
                                final_scores = np.array(final_scores)
                                final_scores = final_scores[::-1]
                                final_scores_dict[current_frame_position] = list(final_scores)

                                metrics_all_params = [
                                    final_lines_normalized_selected,
                                    final_labeled_lines_normalized,
                                    final_midpoints,
                                    final_midpoints_labeled,
                                    final_midpoints_distances,
                                    final_score_midpoints_distances,
                                    final_angles_rad,
                                    final_score_angles_rad,
                                    final_scores
                                ]

                                line_length = [1050, 2100]
                                final_lines = final_lines_extended[idx_lines]
                                final_lines = final_lines_extended
                                image_squared = metric.display_metrics_squared(frame, metrics_all_params, th=20, r=30)
                                # frame_final_lines = metric.draw_final_lines(frame_copy2, final_lines, labeled_lines_extended, th=15, label=True)
                                # final_lines = metric.get_final_lines(final_lines, frame_copy2, line_length=line_length)
                                # final_lines = metric.get_final_lines(final_lines, frame_copy2, line_length=line_length)
                                # print('f',final_lines)

                                frame_final_lines = metric.draw_final_lines(frame_copy2, final_lines, th=15, label=True)

                            # Zeichne die Ergebnisse
                            frame_small = grid_result.draw_grid(frame_copy, grid_coordinates_result_2D, color='green')
                            frame_big = grid_result_final.draw_grid(frame,grid_coordinates_result_final_2D, color='green')

                            relative_nr = best_start_position - current_frame_position + 1
                            image_name_1 = image_name_2 = f'{current_frame_position}'
                            callbacks = grid_optimization.return_callbacks()
                            output_1 = frame_small
                            output_2 = frame_final_lines

                            base_folder = f'results/Duisburg/{method}/final_results/C000{race}/'
                            folder1 = base_folder + 'detection_grid'
                            folder2 = base_folder + 'final_lines'

                            if save_images == True:
                                if output_1 is not None:
                                    save_data.save_image(
                                        output_1,
                                        folder=folder1,
                                        current_time=current_time,
                                        callbacks_list=None,
                                        json_file=None,
                                        name=image_name_1
                                    )

                                if output_2 is not None:
                                    save_data.save_image(
                                        output_2,
                                        folder=folder2,
                                        current_time=current_time,
                                        callbacks_list=None,
                                        json_file=None,
                                        name=image_name_2
                                    )

                            grid_optimization.empty_list_callbacks()

                            # Passe die Sucheinstellung für die benachbarten Frames an! Die initiale Suche benötigt mehr
                            # Spielraum zur Suche
                            if method == 'spatio_temporal_grid': initial_params = result_params

                            if display_images == True:
                                # Output
                                cv2.putText(output_1, f'{current_frame_position}', (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 4,
                                                                (255, 255, 255), 10,
                                                                cv2.LINE_4)
                                output_window = 'image 1'
                                cv2.namedWindow(f"{output_window}", cv2.WINDOW_NORMAL)
                                cv2.imshow(f"{output_window}", output_1)
                                cv2.moveWindow(f"{output_window}", centered_x, centered_y)
                                cv2.resizeWindow(f"{output_window}", window_size[0], window_size[1])

                                cv2.putText(output_2, f'{current_frame_position}', (100, 150), cv2.FONT_HERSHEY_SIMPLEX,4,(255, 255, 255), 10,
                                            cv2.LINE_4)
                                output_window = 'image 2'
                                cv2.namedWindow(f"{output_window}", cv2.WINDOW_NORMAL)
                                cv2.imshow(f"{output_window}", output_2)
                                cv2.moveWindow(f"{output_window}", centered_x2, centered_y2)
                                cv2.resizeWindow(f"{output_window}", window_size[0], window_size[1])


                                key = cv2.waitKey(frame_time) & 0xFF
                                if key == ord('q'):
                                    break

                            count_forward -= 1

                            if play_backwards == True: count_backward -= 1


                    if count_forward == -1:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, best_start_position)
                        play_backwards = True
                        initial_params = initial_params_best_frame_position
                        continue

                    if count_backward == -1:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, best_start_position)
                        break

                if play_backwards == True: cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_position - 2)

            else:
                break

        print(f'time race_{race}: {sum(times_list)/len(times_list)}')

        if save_json == True:
            base_folder = f'results/Duisburg/{method}/final_results/C000{race}/'
            with open(export_path + f'results/Duisburg/{method}/final_results/' + f'all_scores_{race}.json', "w") as json_file:
                json.dump(final_scores_dict, json_file, indent=4)


        if calc_percentage == True:
            path = export_path + base_folder
            with open(export_path + f'results/Duisburg/{method}/final_results/' + f'all_scores_{race}.json', 'r') as json_file:
                final_scores_dict = json.load(json_file)

            count_lines = 0
            count_detected_lines = 0
            threshold = 0.92
            for frame, scores in final_scores_dict.items():
                nr_lines = len(scores)
                nr_detected_lines = len([score for score in scores if score >= threshold])
                count_lines += nr_lines
                count_detected_lines += nr_detected_lines
            percentage_detected_lines = count_detected_lines/count_lines

            json_file_percentage[f'{race}'] = {
                'threshold': threshold,
                'detected_lines': count_detected_lines,
                'all_lines': count_lines,
                'percentage': percentage_detected_lines
            }

            with open(path + f'percentage.json', "w") as f:
                json.dump(json_file_percentage, f, indent=4)


        if save_histogram == True:
            path = export_path + f'/results/Duisburg/{method}/final_results/'
            frame_numbers = []
            min_scores = []
            max_scores = []
            means = []

            with open(path + f'all_scores_{race}.json', 'r') as json_file:
                final_scores_dict = json.load(json_file)

            for frame_nr, scores in final_scores_dict.items():
                scores = np.array(scores)
                min_score = np.min(scores)
                max_score = np.max(scores)
                mean = np.mean(scores)
                min_scores.append(min_score)
                max_scores.append(max_score)
                means.append(mean)
                frame_nr = int(frame_nr)
                frame_nr -= best_start_position +1
                frame_numbers.append(frame_nr)
            number_frames_left_and_right = len(frame_numbers)

            plt.figure(figsize=(8, 6))
            plt.hist(frame_numbers,
                     bins=number_frames_left_and_right,
                     weights=max_scores,
                     alpha=0.5,
                     color='b',
                     edgecolor='k',
                     linewidth=0.5,
                     label='Maximaler Score'
                     )
            plt.hist(frame_numbers,
                     bins=number_frames_left_and_right,
                     weights=min_scores,
                     alpha=1,
                     color='gray',
                     edgecolor='k',
                     linewidth=0.5,
                     label='Minimaler Score'
                     )
            plt.xlabel('Startposition', fontsize=14, fontweight='bold')
            plt.ylabel('EA-Score', fontsize=14, fontweight='bold')
            plt.ylim(0.80, 1.00)
            plt.yticks(np.arange(0.80, 1.01, 0.02), fontsize=12)
            plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.legend(loc='upper right', bbox_to_anchor=(1, 1.15))
            plt.gca().invert_xaxis()
            plt.tight_layout()
            plt.savefig(path + f'histogram_{race}_min_max.png')


            plt.figure(figsize=(8, 6))
            plt.hist(frame_numbers,
                     bins=number_frames_left_and_right,
                     weights=means,
                     alpha=0.5,
                     color='b',
                     edgecolor='k',
                     linewidth=0.5,
                     label='Mittelwert EA-Score'
                     )
            plt.xlabel('Startposition', fontsize=14, fontweight='bold')
            plt.ylabel('EA-Score', fontsize=14, fontweight='bold')
            plt.ylim(0.80, 1.00)
            plt.yticks(np.arange(0.80, 1.01, 0.02), fontsize=12)
            plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.legend(loc='upper right', bbox_to_anchor=(1, 1.15))
            plt.gca().invert_xaxis()
            plt.tight_layout()
            plt.savefig(path + f'histogram_{race}_mean.png')


    if create_boxplots == True:
        all_scores_all_videos = []
        path = export_path + f'/results/Duisburg/{method}/final_results/'
        json_files = [f for f in os.listdir(path) if f.endswith('.json')]
        json_files.sort()
        video_names = [('C00'+f.split('.json')[0].split('_')[-1]) for f in json_files]
        positions = [(p+1) for p in range(len(json_files))]
        nr_files = len(positions)
        all_boxplots = []
        for i, filename in enumerate(json_files):
            all_scores_single_video = []
            with open(path + filename, 'r') as json_file:
                data = json.load(json_file)
                for value_list in data.values():
                    all_scores_single_video.extend(value_list)
                all_scores_all_videos.append(all_scores_single_video)

        plt.figure(figsize=(10, 6))
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 5), sharey=True)
        bp = metric.create_box_plots(all_scores_all_videos, positions)
        plt.xlabel('Videos', fontsize=14, fontweight='bold')
        plt.ylabel('EA-Score', fontsize=14, fontweight='bold')
        plt.ylim(0, 1)
        axs.tick_params(axis='both', labelsize=12)

        # ####### Diese Zeile auskommentieren, wenn Videobezeichnung 1-8 angezeigt werden soll!
        # video_label = True
        # if video_label == False:
        #     plt.xticks(positions, video_names)
        # #######

        medians = [np.median(d) for d in all_scores_all_videos]
        for pos, median in zip(positions, medians):
            axs.text(pos - 0.15, median, f'{median:.2f}', horizontalalignment='right', verticalalignment='center')
        boxplots = path + 'boxplot_number.png'
        plt.tight_layout()
        plt.savefig(boxplots)

        plt.figure(figsize=(10, 6))
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 5), sharey=True)
        bp = metric.create_box_plots(all_scores_all_videos, positions)
        plt.xlabel('Videos', fontsize=14, fontweight='bold')
        plt.ylabel('EA-Score', fontsize=14, fontweight='bold')
        plt.ylim(0, 1)
        axs.tick_params(axis='both', labelsize=12)
        plt.xticks(positions, video_names)

        # ####### Diese Zeile auskommentieren, wenn Videobezeichnung 1-8 angezeigt werden soll!
        # video_label = True
        # if video_label == False:
        #     plt.xticks(positions, video_names)
        # #######

        medians = [np.median(d) for d in all_scores_all_videos]
        for pos, median in zip(positions, medians):
            axs.text(pos - 0.15, median, f'{median:.2f}', horizontalalignment='right', verticalalignment='center')
        boxplots = path + 'boxplot_video.png'
        plt.tight_layout()
        plt.savefig(boxplots)

    cap.release()
    cv2.destroyAllWindows()

