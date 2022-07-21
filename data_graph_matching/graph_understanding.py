import itertools

import cv2
from matplotlib import figure
from matplotlib.pyplot import step
import skimage.io as skio
import skimage.color as skclr
import skimage.measure as skmeasure
import skimage.morphology as skmorph
from data_graph_matching.util.dataset_analysis import *
from data_graph_matching.util.dataset_graph_subsets import *
from data_graph_matching.image_operation import ImageOp
import numpy as np
from data_graph_matching.param import *
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
from data_graph_matching.image_operation import TextPosAgainstFigureContent, ImageOp
from data_graph_matching.common_util import *
import joblib
from termcolor import cprint
from scipy.spatial import ConvexHull
from sklearn.linear_model import Ridge
from scipy.optimize import minimize
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from skimage.filters import gaussian
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
import skimage
from sklearn.neighbors import NearestNeighbors

class GraphParser:

    def __init__(self):
        self.config = dict()

    def from_pipeline(self, graph_filename, graph_type, figure_content_region, text_alignment_result, legend_result):
        self.config['filename'] = graph_filename
        self.config['graph_type'] = graph_type
        self.config['figure_content_region'] = figure_content_region
        self.config['text_alignment_result'] = text_alignment_result
        self.config['legend_result'] = legend_result

        mandatory_fields = ['xticks', 'yticks', 'xlabel', 'ylabel', 'title']
        for fn in mandatory_fields:
            self.config[fn] = None

        self._init_fields()

    @staticmethod
    def _text_entity_from_text(t):
        entity = dict(
            type='text',
            text=t['t'],
            convex_hull=np.flip(np.asarray(t['v']), axis=1).tolist(),
            from_cluster=False
        )
        return entity

    @staticmethod
    def _text_entity_from_cluster(cls):
        points = np.concatenate([np.flip(np.asarray(t['text']['v']), axis=1) for t in cls['text']], axis=0)
        hull = ConvexHull(points)
        vertices = points[hull.vertices, ...]
        entity = dict(
            type='text',
            text=' '.join([x['text']['t'] for x in cls['text']]),
            convex_hull=vertices.astype(np.int).tolist(),
            from_cluster=True
        )
        return entity

    @staticmethod
    def _find_spacing(val):
        # find minimum spacing
        val_diff = np.diff(val)
        min_spacing = val_diff.min()
        min_spacing_loc = np.argmin(val_diff)
        tick_approx_location = np.round(val / min_spacing)
        # fit a linear model to deduce the spacing
        ridge = Ridge(alpha=0.2, fit_intercept=False)
        ridge.fit(tick_approx_location.reshape(-1, 1), val)
        return ridge.coef_[0], tick_approx_location, min_spacing_loc

    @staticmethod
    def _axis_transform_pixel_to_data_abc(a, b, c, v):
        return (v - a) / c + b

    @staticmethod
    def _axis_transform_pixel_to_data(axis_entity, value):
        first_loc, first_value, scale = axis_entity['transform_param']
        return GraphParser._axis_transform_pixel_to_data_abc(first_loc, first_value, scale, value)

    @staticmethod
    def _axis_transform_data_to_pixel(axis_entity, value):
        first_loc, first_value, scale = axis_entity['transform_param']
        return (value - first_value) * scale + first_loc

    @staticmethod
    def _tick_entity_from_cluster(cls):
        vertex_ind = cls['vertex_index']
        alignment_coordinate = 1 - cls['coordinate_index']
        entity = dict(
            type='ticks',
            text=[x['text']['t'] for x in cls['text']],
            aligned_position=cls['text'][0]['points'][0][cls['coordinate_index']],
            alignment_vertex=vertex_ind
        )

        # compute the average position
        position = []
        for x in cls['text']:
            all_coords = [x['points'][i][alignment_coordinate] for i in range(len(x['points']))]
            avg_pos = np.mean(all_coords)
            position.append(avg_pos)

        # compute bbox for each text
        pos_bbox = []

        for x in cls['text']:
            all_points = np.asarray(x['points'])
            i_min, j_min = np.min(all_points, axis=0)
            i_max, j_max = np.max(all_points, axis=0)
            pos_bbox.append((i_min, i_max, j_min, j_max))

        text_pos = list(zip(entity['text'], position, pos_bbox))
        text_pos.sort(key=lambda x: x[1])
        text, position, pos_bbox = zip(*text_pos)
        entity['text'] = text
        entity['position'] = position
        entity['position_bbox'] = np.asarray(pos_bbox).astype(np.int).tolist()

        is_numeric = [is_numeric_text_strict(x) for x in entity['text']]
        is_numeric_rle = run_length_encoding(is_numeric)

        numeric_runs = []
        for i in range(len(is_numeric_rle[2])):
            if is_numeric_rle[2][i] and is_numeric_rle[0][i] > 1:
                numeric_runs.append(i)

        has_numeric_ticks = False
        if len(numeric_runs) > 0:
            # find longest increasing run
            increasing_runs = []
            for numeric_run_ind in numeric_runs:
                start = is_numeric_rle[1][numeric_run_ind]
                length = is_numeric_rle[0][numeric_run_ind]
                value = [float(x) for x in entity['text'][start: start + length]]

                increasing = np.diff(value) > 0
                increasing = np.insert(increasing, 0, increasing[0])
                increasing_rle = run_length_encoding(increasing)

                run_lengths = [(increasing_rle[0][i], i) for i in range(len(increasing_rle[0]))]
                run_lengths.sort(key=lambda x: x[0], reverse=True)

                for run_length, index in run_lengths:
                    if increasing_rle[2][index]:
                        abs_start = start + increasing_rle[1][index]
                        abs_end = abs_start + run_length
                        increasing_runs.append((abs_start, abs_end))

                # do this again for the reversed order
                value.reverse()
                increasing = np.diff(value) > 0
                increasing = np.insert(increasing, 0, increasing[0])
                increasing_rle = run_length_encoding(increasing)

                run_lengths = [(increasing_rle[0][i], i) for i in range(len(increasing_rle[0]))]
                run_lengths.sort(key=lambda x: x[0], reverse=True)

                for run_length, index in run_lengths:
                    if increasing_rle[2][index]:
                        abs_start = start + increasing_rle[1][index]
                        abs_end = abs_start + run_length
                        abs_start = len(entity['text']) - abs_start - 1
                        abs_end = len(entity['text']) - abs_end
                        assert abs_start >= 0 and abs_end >= 0
                        increasing_runs.append((abs_start, abs_end))

            if len(increasing_runs) > 0:
                has_numeric_ticks = True
                entity['found_numeric_ticks'] = True
                increasing_runs.sort(key=lambda x: abs(x[1] - x[0]), reverse=True)
                # entity['best_run_start'] = increasing_runs[0][0]
                # entity['best_run_end'] = increasing_runs[0][1]
                _rs, _re = increasing_runs[0]
                if _re < _rs:
                    pos = list(reversed(entity['position']))[_re:_rs]
                    text_items = list(reversed(entity['text']))[_re:_rs]
                else:
                    pos = entity['position'][_rs:_re]
                    text_items = entity['text'][_rs:_re]

                if len(pos) > 1:
                    all_values = [float(x) for x in text_items]

                    first_value = all_values[0]
                    spacing, _, min_spacing_loc = GraphParser._find_spacing(pos)
                    # compute the diff
                    diff = all_values[min_spacing_loc + 1] - all_values[min_spacing_loc]

                    first_loc = pos[0]
                    entity['transform_param'] = (first_loc, first_value, spacing / diff)
                    entity['increasing_run_position'] = pos
                    entity['increasing_run_values'] = all_values

        if not has_numeric_ticks:
            entity['found_numeric_ticks'] = False
            pos = entity['position']
            if len(pos) > 1:
                spacing, _, _ = GraphParser._find_spacing(pos)
                entity['transform_param'] = (entity['position'][0], 0, spacing)

        if 'transform_param' in entity:
            # now that initial transform parameters are proposed, fine tune them
            if entity['found_numeric_ticks']:
                good_pos = entity['increasing_run_position']
                good_ticks = entity['increasing_run_values']
            else:
                good_pos = np.asarray(entity['position'])
                good_ticks = np.arange(len(good_pos))

            def compute_matching_loss(p):
                numeric_ticks = GraphParser._axis_transform_pixel_to_data_abc(p[0], p[1], p[2], good_pos)
                diff = numeric_ticks - good_ticks
                return np.linalg.norm(diff)

            initial_value = entity['transform_param']
            ret = minimize(compute_matching_loss, initial_value)
            entity['transform_param'] = ret.x.tolist()

            entity['numeric_ticks'] = GraphParser._axis_transform_pixel_to_data(entity,
                                                                                np.asarray(entity['position'])).tolist()

        return entity

    def _init_fields(self):

        for cls in self.config['text_alignment_result']['cluster']:
            if cls['position'] == TextPosAgainstFigureContent.BELOW.value:
                self.config['xticks'] = self._tick_entity_from_cluster(cls)
            elif cls['position'] == TextPosAgainstFigureContent.LEFT.value:
                self.config['yticks'] = self._tick_entity_from_cluster(cls)
            elif cls['position'] == TextPosAgainstFigureContent.ABOVE.value:
                self.config['title'] = self._text_entity_from_cluster(cls)

        for unc in self.config['text_alignment_result']['unclustered']:
            if len(unc['position']) == 1 and unc['position'][0] == TextPosAgainstFigureContent.LEFT.value:
                self.config['ylabel'] = self._text_entity_from_text(unc['text'])
            elif len(unc['position']) == 1 and unc['position'][0] == TextPosAgainstFigureContent.BELOW.value:
                self.config['xlabel'] = self._text_entity_from_text(unc['text'])
            elif len(unc['position']) == 1 and unc['position'][0] == TextPosAgainstFigureContent.ABOVE.value and \
                    self.config['title'] is None:
                self.config['title'] = self._text_entity_from_text(unc['text'])

    def get_legend(self):
        return self.config['legend_result']

    def get_xticks(self):
        return self.config['xticks']

    def get_yticks(self):
        return self.config['yticks']

    def get_xlabel(self):
        return self.config['xlabel']

    def get_ylabel(self):
        return self.config['ylabel']

    def get_graph_type(self):
        return self.config['graph_type']

    def get_graph(self):
        op = ImageOp(dict(filename=self.config['filename']))
        return op.get_image()

    def get_graph_figure_content(self):
        j_min, i_min, j_max, i_max = np.round(self.config['figure_content_region']).astype(np.int)
        img = self.get_graph()
        return img[i_min: i_max, j_min: j_max], i_min, j_min

    def _parse_graph_type_bar_like(self):
        figure_region, i_min, j_min = self.get_graph_figure_content()

        figure_region_lab = skclr.rgb2lab(figure_region / 255.0)

        parsed_info = []

        for legend_name, legend_color_rgb, bbox in self.config['legend_result']:
            legend_color_std = skclr.rgb2lab(np.asarray(legend_color_rgb) / 255.0)
            figure_content_clr_diff = np.linalg.norm(figure_region_lab - legend_color_std, axis=-1)

            area_mask_naive = figure_content_clr_diff < bar_plot_clr_diff_threshold
            

            # denoise with binary opening (eliminate potential lines that has close color values)
            # area_mask = skmorph.binary_opening(area_mask_naive, skmorph.square(bar_plot_binary_opening_radius))
            # area_mask_naive = skmorph.binary_erosion(area_mask_naive, skmorph.square(bar_plot_binary_opening_radius))
            # denoise with binary closing (to eliminate lines in solid blocks)
            area_mask_naive = skmorph.binary_closing(area_mask_naive, skmorph.square(bar_plot_binary_closing_radius))
            area_mask_naive = skmorph.binary_erosion(area_mask_naive, skmorph.square(bar_plot_binary_erosion_radius))

            # use the connected component that is the closest
            labels = skmeasure.label(area_mask_naive)
            label_counts = [np.count_nonzero(labels == i) for i in range(1, labels.max() + 1)]

            if labels.max() > 0:
                max_count = max(label_counts)
                if self.get_graph_type() == 'stacked':
                    for i in range(1, labels.max() + 1):
                        if label_counts[i - 1] >= bar_like_cc_size_coef * max_count:  # try to filter out small regions
                            label_loc = np.stack(np.where(labels == i), axis=1)
                            try:
                                hull = ConvexHull(label_loc)
                                vertices = label_loc[hull.vertices, ...] + [i_min, j_min]  # plus the offset
                                v_i_data = self._axis_transform_pixel_to_data(self.get_yticks(), vertices[:, 0])
                                v_j_data = self._axis_transform_pixel_to_data(self.get_xticks(), vertices[:, 1])
                                parsed_info.append(
                                    dict(
                                        type='concentrated_region',
                                        hull=vertices.tolist(),
                                        legend_name=legend_name,
                                        legend_color=legend_color_rgb,
                                        legend_bbox=bbox,
                                        data_hull=np.stack([v_j_data, v_i_data], axis=1).tolist()  # in (x, y) form
                                    )
                                )
                            except:
                                pass
                else:
                    ind_diff = []
                    # find the best match
                    for i in range(1, labels.max() + 1):
                        if label_counts[i - 1] >= bar_like_cc_size_coef * max_count:  # try to filter out small regions
                            label_mask = labels == i
                            label_region = figure_region_lab[label_mask]
                            mean_color = np.mean(label_region.reshape(-1, 3))
                            color_diff = np.linalg.norm(mean_color - legend_color_std)
                            ind_diff.append((i, color_diff))

                    if len(ind_diff) > 0:
                        best_ind, _ = min(ind_diff, key=lambda x: x[1])
                        label_loc = np.stack(np.where(labels == best_ind), axis=1)
                        try:
                            hull = ConvexHull(label_loc)
                            vertices = label_loc[hull.vertices, ...] + [i_min, j_min]  # plus the offset
                            v_i_data = self._axis_transform_pixel_to_data(self.get_yticks(), vertices[:, 0])
                            v_j_data = self._axis_transform_pixel_to_data(self.get_xticks(), vertices[:, 1])
                            parsed_info.append(
                                dict(
                                    type='concentrated_region',
                                    hull=vertices.tolist(),
                                    legend_name=legend_name,
                                    legend_color=legend_color_rgb,
                                    legend_bbox=bbox,
                                    data_hull=np.stack([v_j_data, v_i_data], axis=1).tolist()  # in (x, y) form
                                )
                            )
                        except:
                            pass

        self.config['parse_result'] = parsed_info

    def _parse_graph_type_line(self):
        figure_region, i_min, j_min = self.get_graph_figure_content()

        figure_region_lab = skclr.rgb2lab(figure_region / 255.0)

        parsed_info = []

        all_x_min = []
        all_x_max = []

        for legend_name, legend_color_rgb, bbox in self.config['legend_result']:
            legend_color_std = skclr.rgb2lab(np.asarray(legend_color_rgb) / 255.0)
            figure_content_clr_diff = np.linalg.norm(figure_region_lab - legend_color_std, axis=-1)

            area_mask_naive = figure_content_clr_diff < line_plot_clr_diff_threshold

            if np.count_nonzero(area_mask_naive) == 0:
                continue

            # we need to estimate the range of x value
            area_pixel_coords = np.where(area_mask_naive)
            area_x_min = np.min(area_pixel_coords[1])
            area_x_max = np.max(area_pixel_coords[1])
            all_x_min.append(area_x_min)
            all_x_max.append(area_x_max)

            #area_mask = skmorph.binary_opening(area_mask_naive, skmorph.square(line_plot_binary_opening_radius))
            #area_mask = area_mask.astype(np.float32)
            #area_mask = gaussian(area_mask, sigma=1.0)
            # labels = skmeasure.label(area_mask_naive)
            # label_counts = [(i, np.count_nonzero(labels == i)) for i in range(1, labels.max() + 1)]
            # assert len(labels) >= 3
            # # use three points from the first three largest cc
            # label_counts.sort(key=lambda x: x[1], reverse=True)
            #
            # all_points = []
            #
            # # take two points out from each cc
            # for cc_ind, cc_count in label_counts[:3]:
            #     cc_mask = labels == cc_ind
            #     corner_coords = corner_peaks(corner_harris(cc_mask), min_distance=5, threshold_rel=0.02)
            #
            #     # sample two points
            #     if corner_coords.shape[0] > 0:
            #         num_samples = min(corner_coords.shape[0], 2)
            #         corner_sample_ind = np.random.choice(np.arange(corner_coords.shape[0]), size=num_samples, replace=False)
            #         corner_samples = corner_coords[corner_sample_ind]
            #         all_points.append(corner_samples)

            all_points = []

            # sample data (with far apart x values)
            x_partition = np.linspace(area_x_min, area_x_max, line_plot_num_sample + 2)
            delta_x_2 = (x_partition[1] - x_partition[0]) / 2.5
            for i in range(1, line_plot_num_sample - 1):
                x_center = x_partition[i]
                coord_dist = np.abs(area_pixel_coords[1] - x_center)
                choice_coord = np.where(coord_dist < delta_x_2)[0]
                if choice_coord.size > 0:
                    good_dist = coord_dist[choice_coord]
                    pick_ind = np.argmin(good_dist)
                    coord_ind = choice_coord[pick_ind]
                    all_points.append((area_pixel_coords[0][coord_ind], area_pixel_coords[1][coord_ind]))

            # assert area_pixel_coords[0].size >= line_plot_num_sample
            # sample_ind = np.random.choice(np.arange(area_pixel_coords[0].size), size=line_plot_num_sample, replace=False)
            # all_points = np.stack([area_pixel_coords[0][sample_ind], area_pixel_coords[1][sample_ind]], axis=1) + [i_min, j_min]

            if len(all_points) > 0:
                all_points = np.asarray(all_points)
                all_points = all_points + [i_min, j_min]

                # convert to data coord
                v_i_data = self._axis_transform_pixel_to_data(self.get_yticks(), all_points[:, 0]) # y
                v_j_data = self._axis_transform_pixel_to_data(self.get_xticks(), all_points[:, 1]) # x
                data_samples = np.stack([v_j_data, v_i_data], axis=1)

                parsed_info.append(
                    dict(
                        type='data_sample',
                        legend_name=legend_name,
                        legend_color=legend_color_rgb,
                        legend_bbox=bbox,
                        sample=data_samples.astype(np.float).tolist(),
                        # x_min=self._axis_transform_pixel_to_data(self.get_xticks(), area_x_min),
                        # x_max=self._axis_transform_pixel_to_data(self.get_xticks(), area_x_max)
                    )
                )

        for item in parsed_info:
            item['x_min'] = self._axis_transform_pixel_to_data(self.get_xticks(), np.min(all_x_min) + j_min)
            item['x_max'] = self._axis_transform_pixel_to_data(self.get_xticks(), np.max(all_x_max) + j_min)

        self.config['parse_result'] = parsed_info

    def _parse_graph_type_scatter(self):
        figure_region, i_min, j_min = self.get_graph_figure_content()

        figure_region_lab = skclr.rgb2lab(figure_region / 255.0)

        parsed_info = []

        for legend_name, legend_color_rgb, bbox in self.config['legend_result']:
            legend_color_std = skclr.rgb2lab(np.asarray(legend_color_rgb) / 255.0)
            figure_content_clr_diff = np.linalg.norm(figure_region_lab - legend_color_std, axis=-1)

            area_mask_naive = figure_content_clr_diff < scatter_plot_clr_diff_threshold
            area_mask = skmorph.binary_opening(area_mask_naive, skmorph.disk(scatter_plot_binary_erosion_radius))
            area_mask = skimage.img_as_ubyte(area_mask)

            edges = canny(area_mask, sigma=3)

            hough_radii = np.arange(2, 40, 2)
            hough_res = hough_circle(edges, hough_radii)

            accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                                       total_num_peaks=scatter_plot_num_points_per_legend,
                                                       min_xdistance=3, min_ydistance=3)

            # import matplotlib.pyplot as plt
            # fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
            # ax.imshow(area_mask, cmap='gray')
            # for center_y, center_x, radius in zip(cy, cx, radii):
            #     ax.plot(center_x, center_y, 'r+')
            # plt.show()

            if len(cx) > 0:

                all_points = np.stack([cy, cx], axis=1)
                all_points = all_points + [i_min, j_min]

                # convert to data coord
                v_i_data = self._axis_transform_pixel_to_data(self.get_yticks(), all_points[:, 0]) # y
                v_j_data = self._axis_transform_pixel_to_data(self.get_xticks(), all_points[:, 1]) # x
                data_samples = np.stack([v_j_data, v_i_data], axis=1)

                parsed_info.append(
                    dict(
                        type='data_sample',
                        legend_name=legend_name,
                        legend_color=legend_color_rgb,
                        legend_bbox=bbox,
                        sample=data_samples.astype(np.float).tolist(),
                    )
                )

        self.config['parse_result'] = parsed_info

    def _parse_graph_type_heatmap(self):
        figure_region, i_min, j_min = self.get_graph_figure_content()

        #figure_region_lab = skclr.rgb2lab(figure_region / 255.0)

        #parsed_info = []

        # check how many ticks we have
        x_ticks, y_ticks = self.get_xticks(), self.get_yticks()


        # see if there is text in the figure region
        # heatmap_data = []
        # figure_i_range = i_min, i_min + figure_region.shape[0]
        # figure_j_range = j_min, j_min + figure_region.shape[1]
        # for item in self.config['text_alignment_result']['unclustered']:
        #     center_point = np.mean(item['points'], axis=0)
        #     if figure_i_range[0] <= center_point[0] <= figure_i_range[1] and \
        #           figure_j_range[0] <= center_point[1] <= figure_j_range[1]:
        #         if is_numeric_text_strict(item['text']['t']):
        #             x = self._axis_transform_pixel_to_data(self.get_xticks(), center_point[1]) # x
        #             y = self._axis_transform_pixel_to_data(self.get_yticks(), center_point[0])
        #             heatmap_data.append((float(item['text']['t']), [x, y]))



        img = self.get_graph()


        y_data_ticks = y_ticks['position']
        # for x ticks, use the top right corner
        x_data_ticks = [bbox[3] for bbox in x_ticks['position_bbox']]
        # _x_data_ticks = x_ticks['position']
        data_array = np.zeros((len(y_data_ticks), len(x_data_ticks), 3), np.float)

        # start sampling from the image
        for i, ival in enumerate(y_data_ticks):
            for j, jval in enumerate(x_data_ticks):
                pixel_coord_i = round(ival)
                pixel_coord_j = round(jval)
                cond_check = 0 <= pixel_coord_i < img.shape[0] and 0 <= pixel_coord_j <= img.shape[1]
                if cond_check:
                    sample = img[pixel_coord_i - heatmap_sample_size : pixel_coord_i + heatmap_sample_size + 1,
                                    pixel_coord_j - heatmap_sample_size : pixel_coord_j + heatmap_sample_size + 1, ...]
                    if sample.size == 0:
                        data_array[i, j] = [np.nan] * 3
                    else:
                        avg_color = np.mean(sample.reshape((-1, 3)), axis=0)
                        data_array[i,j] = skclr.rgb2lab(avg_color / 255.0) # data in LAB space
                else:
                    data_array[i, j] = [np.nan] * 3



        # extract the colorbar

        # limit the focus to the right of the figure content
        op = ImageOp(dict(filename=self.config['filename']))
        background_color = skclr.rgb2lab(np.asarray(op.get_background_color()) / 255.0)

        right_region_j = j_min + figure_region.shape[1] + 10
        right_region = skclr.rgb2lab(img[:, right_region_j:] / 255.0)
        not_bg_color = skclr.deltaE_cie76(right_region, background_color) > 6.0

        labels = skmeasure.label(not_bg_color)
        label_counts = [np.count_nonzero(labels == i) for i in range(1, labels.max() + 1)]
        largest_group = np.argmax(label_counts) + 1
        best_label_mask = labels == largest_group
        # acquire a bounding box
        best_label_loc = np.where(best_label_mask)
        _i_min, _j_min = np.min(best_label_loc, axis=1)
        _i_max, _j_max = np.max(best_label_loc, axis=1)

        _i_min += 2
        _j_min += 2
        _i_max -= 2
        _j_max -= 2
        assert _i_max > _i_min and _j_max > _j_min

        colorbar_region = img[_i_min:_i_max, right_region_j+_j_min:right_region_j+_j_max]
        colorbar_region_avg = skclr.rgb2lab(np.mean(colorbar_region, axis=1) / 255.0) # in LAB space

        # import matplotlib.pyplot as plt
        # plt.imshow(colorbar_region)
        # plt.show()

        # generate a color map
        colors = []
        pix_ticks = np.linspace(0, _i_max - _i_min, heatmap_colorbar_size)
        ticks = np.linspace(0, 1, heatmap_colorbar_size)
        for i in range(len(ticks)):
            loc = np.clip(np.round(pix_ticks[i]).astype(np.int), 0, colorbar_region_avg.shape[0] - 1)
            colors.append(colorbar_region_avg[loc])
        colors.reverse()

        # deal with nans
        for i in range(data_array.shape[0]):
            for j in range(data_array.shape[1]):
                if np.any(np.isnan(data_array[i, j])):
                    data_array[i, j] = colors[len(colors) // 2]

        # match data
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(colors)
        nn, data_ind = nn.kneighbors(data_array.reshape((-1, 3)), n_neighbors=1)
        data_ind = data_ind.reshape(data_array.shape[:-1])
        data_transformed = ticks[data_ind]

        all_text = x_ticks['text'] + y_ticks['text']
        all_text = [s.lower() for s in all_text]
        all_text = np.unique(all_text).tolist()

        self.config['parse_result'] = dict(
                type='heatmap_data',
                colors=np.asarray(colors).tolist(),
                ticks=ticks.tolist(),
                data=data_transformed.tolist(),
                text=all_text,
                colorbar_bbox=np.array((_i_min, _i_max, _j_min, _j_max), np.int).tolist(),
                y_data_ticks=np.asarray(y_data_ticks).astype(np.int).tolist(),
                x_data_ticks=np.asarray(x_data_ticks).astype(np.int).tolist(),
                x_text=x_ticks['text'],
                y_text=y_ticks['text']
            )

    def _parse_graph_type_line(self):
        figure_region, i_min, j_min = self.get_graph_figure_content()

        figure_region_lab = skclr.rgb2lab(figure_region / 255.0)

        parsed_info = []

        all_x_min = []
        all_x_max = []

        for legend_name, legend_color_rgb, bbox in self.config['legend_result']:
            legend_color_std = skclr.rgb2lab(np.asarray(legend_color_rgb) / 255.0)
            figure_content_clr_diff = np.linalg.norm(figure_region_lab - legend_color_std, axis=-1)

            area_mask_naive = figure_content_clr_diff < line_plot_clr_diff_threshold
            if np.count_nonzero(area_mask_naive) == 0:
                continue

            # we need to estimate the range of x value
            area_pixel_coords = np.where(area_mask_naive)
            area_x_min = np.min(area_pixel_coords[1])
            area_x_max = np.max(area_pixel_coords[1])
            all_x_min.append(area_x_min)
            all_x_max.append(area_x_max)

            #area_mask = skmorph.binary_opening(area_mask_naive, skmorph.square(line_plot_binary_opening_radius))
            #area_mask = area_mask.astype(np.float32)
            #area_mask = gaussian(area_mask, sigma=1.0)
            # labels = skmeasure.label(area_mask_naive)
            # label_counts = [(i, np.count_nonzero(labels == i)) for i in range(1, labels.max() + 1)]
            # assert len(labels) >= 3
            # # use three points from the first three largest cc
            # label_counts.sort(key=lambda x: x[1], reverse=True)
            #
            # all_points = []
            #
            # # take two points out from each cc
            # for cc_ind, cc_count in label_counts[:3]:
            #     cc_mask = labels == cc_ind
            #     corner_coords = corner_peaks(corner_harris(cc_mask), min_distance=5, threshold_rel=0.02)
            #
            #     # sample two points
            #     if corner_coords.shape[0] > 0:
            #         num_samples = min(corner_coords.shape[0], 2)
            #         corner_sample_ind = np.random.choice(np.arange(corner_coords.shape[0]), size=num_samples, replace=False)
            #         corner_samples = corner_coords[corner_sample_ind]
            #         all_points.append(corner_samples)

            all_points = []

            # sample data (with far apart x values)
            x_partition = np.linspace(area_x_min, area_x_max, line_plot_num_sample + 2)
            delta_x_2 = (x_partition[1] - x_partition[0]) / 2.5
            for i in range(1, line_plot_num_sample - 1):
                x_center = x_partition[i]
                coord_dist = np.abs(area_pixel_coords[1] - x_center)
                choice_coord = np.where(coord_dist < delta_x_2)[0]
                if choice_coord.size > 0:
                    good_dist = coord_dist[choice_coord]
                    pick_ind = np.argmin(good_dist)
                    coord_ind = choice_coord[pick_ind]
                    all_points.append((area_pixel_coords[0][coord_ind], area_pixel_coords[1][coord_ind]))

            # assert area_pixel_coords[0].size >= line_plot_num_sample
            # sample_ind = np.random.choice(np.arange(area_pixel_coords[0].size), size=line_plot_num_sample, replace=False)
            # all_points = np.stack([area_pixel_coords[0][sample_ind], area_pixel_coords[1][sample_ind]], axis=1) + [i_min, j_min]

            if len(all_points) > 0:
                all_points = np.asarray(all_points)
                all_points = all_points + [i_min, j_min]

                # convert to data coord
                v_i_data = self._axis_transform_pixel_to_data(self.get_yticks(), all_points[:, 0]) # y
                v_j_data = self._axis_transform_pixel_to_data(self.get_xticks(), all_points[:, 1]) # x
                data_samples = np.stack([v_j_data, v_i_data], axis=1)

                parsed_info.append(
                    dict(
                        type='data_sample',
                        legend_name=legend_name,
                        legend_color=legend_color_rgb,
                        legend_bbox=bbox,
                        sample=data_samples.astype(np.float).tolist(),
                        # x_min=self._axis_transform_pixel_to_data(self.get_xticks(), area_x_min),
                        # x_max=self._axis_transform_pixel_to_data(self.get_xticks(), area_x_max)
                    )
                )

        for item in parsed_info:
            item['x_min'] = self._axis_transform_pixel_to_data(self.get_xticks(), np.min(all_x_min) + j_min)
            item['x_max'] = self._axis_transform_pixel_to_data(self.get_xticks(), np.max(all_x_max) + j_min)

        self.config['parse_result'] = parsed_info

    def _parse_graph_type_area(self):
        figure_region, i_min, j_min = self.get_graph_figure_content()

        figure_region_lab = skclr.rgb2lab(figure_region / 255.0)

        parsed_info = []

        all_x_min = []
        all_x_max = []

        for legend_name, legend_color_rgb, bbox in self.config['legend_result']:
            legend_color_std = skclr.rgb2lab(np.asarray(legend_color_rgb) / 255.0)
            figure_content_clr_diff = np.linalg.norm(figure_region_lab - legend_color_std, axis=-1)

            area_mask_naive = figure_content_clr_diff < area_plot_clr_diff_threshold

            # we need to estimate the range of x value
            if np.count_nonzero(area_mask_naive) == 0:
                continue
            area_pixel_coords = np.where(area_mask_naive)
            area_x_min = np.min(area_pixel_coords[1])
            area_x_max = np.max(area_pixel_coords[1])
            all_x_min.append(area_x_min)
            all_x_max.append(area_x_max)

            #
            # import matplotlib.pyplot as plt
            # plt.imshow(area_mask_naive)
            # plt.show()

            area_mask = skmorph.binary_opening(area_mask_naive, skmorph.square(area_plot_binary_opening_radius))
            if np.count_nonzero(area_mask) > 0:
                loc = np.stack(np.where(area_mask), axis=1).tolist()
                loc.sort(key=lambda x: (x[1], x[0]))

                all_points = []

                for j_ind, group in itertools.groupby(loc, key=lambda x: x[1]):
                    min_i = min(group, key=lambda x: x[0])[0]
                    all_points.append((min_i, j_ind))

                # sample data (with far apart x values)

                if len(all_points) < area_plot_num_sample:
                    # discard small legends
                    continue

                sample_inds = np.random.choice(np.arange(len(all_points)), size=area_plot_num_sample, replace=False)
                all_samples = [all_points[k] for k in sample_inds]
                all_points = all_samples


                if len(all_points) > 0:
                    all_points = np.asarray(all_points)
                    all_points = all_points + [i_min, j_min]

                    # convert to data coord
                    v_i_data = self._axis_transform_pixel_to_data(self.get_yticks(), all_points[:, 0]) # y
                    v_j_data = self._axis_transform_pixel_to_data(self.get_xticks(), all_points[:, 1]) # x
                    data_samples = np.stack([v_j_data, v_i_data], axis=1)

                    parsed_info.append(
                        dict(
                            type='data_sample',
                            legend_name=legend_name,
                            legend_color=legend_color_rgb,
                            legend_bbox=bbox,
                            sample=data_samples.astype(np.float).tolist(),
                            # x_min=self._axis_transform_pixel_to_data(self.get_xticks(), area_x_min),
                            # x_max=self._axis_transform_pixel_to_data(self.get_xticks(), area_x_max)
                        )
                    )

        for item in parsed_info:
            item['x_min'] = self._axis_transform_pixel_to_data(self.get_xticks(), np.min(all_x_min) + j_min)
            item['x_max'] = self._axis_transform_pixel_to_data(self.get_xticks(), np.max(all_x_max) + j_min)

        self.config['parse_result'] = parsed_info


    def parse(self):
        if self.get_xticks() is None or self.get_yticks() is None:
            return False, 'xticks or yticks not found'  # unsuccessful

        if self.get_graph_type() in ['bar', 'box', 'violin', 'stacked', 'hbar']:
            if self.get_legend() is None:
                return False, 'no legend information found for this plot type'
            if 'transform_param' not in self.get_xticks() or 'transform_param' not in self.get_yticks():
                return False, 'no transform parameters found for this plot'
            self._parse_graph_type_bar_like()

            return True, 'OK'
        elif self.get_graph_type() == 'line':
            if self.get_legend() is None:
                return False, 'no legend information found for this plot type'
            if 'transform_param' not in self.get_xticks() or 'transform_param' not in self.get_yticks():
                return False, 'no transform parameters found for this plot'
            self._parse_graph_type_line()

            return True, 'OK'
        elif self.get_graph_type() == 'scatter':
            if self.get_legend() is None:
                return False, 'no legend information found for this plot type'
            if 'transform_param' not in self.get_xticks() or 'transform_param' not in self.get_yticks():
                return False, 'no transform parameters found for this plot'
            self._parse_graph_type_scatter()

            return True, 'OK'
        elif self.get_graph_type() == 'heatmap':
            if 'transform_param' not in self.get_xticks() or 'transform_param' not in self.get_yticks():
                return False, 'no transform parameters found for this plot'
            self._parse_graph_type_heatmap()
            return True, 'OK'
        elif self.get_graph_type() == 'area':
            if 'transform_param' not in self.get_xticks() or 'transform_param' not in self.get_yticks():
                return False, 'no transform parameters found for this plot'
            self._parse_graph_type_area()
            return True, 'OK'

        return False, 'this graph type is not implemented'
