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
import itertools
from shapely.geometry import Polygon, LineString
from scipy.interpolate import interp1d
import functools

from pandas import read_csv

class GraphDatasetMatcher:

    def __init__(self, parse_result, dataset_paths):
        self.parse_result = parse_result
        self.dataset_paths = dataset_paths

        self.dataset_cache = None
        self.heatmap_data_cache = None

    @staticmethod
    def _get_bbox_from_data_hull(hull):
        hull = np.asarray(hull)
        mins = np.min(hull, axis=0)
        maxs = np.max(hull, axis=0)
        x_min, y_min = mins
        x_max, y_max = maxs
        return x_min, x_max, y_min, y_max

    def _is_text(self, o):
        if self._safe_float(o) is None:
            if o is not None and len(str(o)) > 0:
                return True
        return False

    @staticmethod
    def _safe_float(d):
        try:
            return float(d)
        except:
            return None

    # def _dataset_iterator(self):
    #     if self.dataset_cache is None:
    #         self.dataset_cache = []
    #         for path in self.dataset_paths:
    #             df = read_csv(path)
    #
    #             # check if it has header
    #             headers = df.columns
    #             all_text = set()
    #
    #             if headers is not None and len(headers) > 0:
    #                 for val in headers:
    #                     if self._is_text(val):
    #                         all_text.add(val)
    #
    #             # see if some columns are text
    #             data_cols = []
    #             for j in range(df.shape[1]):
    #                 if self._is_text(df.iloc[0, j]):
    #                     for val in df.iloc[:, j]:
    #                         if self._is_text(val):
    #                             all_text.add(val)
    #                 else:
    #                     data_cols.append(df.iloc[:, j].to_numpy().astype(np.float64))
    #
    #             data_df = np.stack(data_cols, axis=1)
    #             entry = (path, data_df, all_text)
    #
    #             self.dataset_cache.append(entry)
    #
    #             yield entry
    #     else:
    #         for entry in self.dataset_cache:
    #             yield entry
    #
    #
    # def _get_rows_cols(self, df):
    #     ret = []
    #
    #     for i in range(df.shape[0]):
    #         col = df[i, :]
    #         ret.append(('c', i, col))
    #
    #     for j in range(df.shape[1]):
    #         row = df[:, j]
    #         ret.append(('r', j, row))
    #
    #     return ret
    #
    # @staticmethod
    # def _nn_score(data, arr):
    #     assert arr.size > 0
    #     assert arr.size >= len(data)
    #     assert len(data) > 0
    #
    #     arr_size = arr.size
    #     arr = np.sort(arr)
    #
    #     distances = []
    #
    #     for d in data:
    #         i = np.searchsorted(arr, d)
    #
    #         dl = dr = None
    #
    #         if i > 0:
    #             dl = abs(arr[i - 1] - d)
    #
    #         if i < arr.size:
    #             dr = abs(arr[i] - d)
    #
    #         if dl is None:
    #             dl = dr + 1.0
    #         if dr is None:
    #             dr = dl + 1.0
    #
    #         if dl < dr:
    #             i -= 1
    #             distances.append(dl)
    #         else:
    #             distances.append(dr)
    #
    #         arr = np.delete(arr, i)
    #
    #     # compute total negative distance
    #     total_neg_dist = -np.sum(distances)
    #     # penalize based on the number of available candidates
    #     total_neg_dist = total_neg_dist / np.sqrt(arr_size)
    #
    #     return total_neg_dist  # non-positive, greater the better
    #
    # @staticmethod
    # def _text_score(data, arr):
    #     assert len(arr) > 0
    #
    #     arr = [x.lower() for x in arr]
    #
    #     hit = 0
    #     for d in data:
    #         if d.lower() in arr:
    #             hit += 1
    #         else:
    #             hit -= 1
    #
    #     if len(arr) == 0:
    #         hit = hit / 2
    #     else:
    #         hit = hit / len(arr)
    #     return hit  # non-negative, greater the better
    #
    # def _bar_like_score(self, graph_text, graph_number):
    #     df_results = []
    #
    #     for df_path, df, all_text in self._dataset_iterator():
    #         all_number = self._get_rows_cols(df)
    #         all_text = all_text
    #
    #         text_score = self._text_score(graph_text, all_text)
    #
    #         all_number_score = []
    #         for rc, ind, data_arr in all_number:
    #             if data_arr.size >= len(graph_number): # and data_arr.size <= data_matching_max_scale * len(graph_number):
    #                 # discard if the data is too long
    #                 number_score = self._nn_score(graph_number, np.sort(data_arr))
    #                 all_number_score.append((rc, ind, number_score))
    #
    #         if len(all_number_score) > 0:
    #             all_number_score.sort(key=lambda x:x[2], reverse=True)
    #             best_number_score = all_number_score[0][2]
    #
    #             score = best_number_score * matching_data_weight + text_score * matching_text_weight
    #             df_results.append((df_path, score))
    #
    #     if len(df_results) > 0:
    #         df_results.sort(key=lambda x: x[1], reverse=True)
    #         return df_results[:5]
    #     else:
    #         return None

    def _heatmap_data_iterator(self):
        if self.heatmap_data_cache is None:
            self.heatmap_data_cache = []

            for path in self.dataset_paths:
                df = read_csv(path)
                # see if there is a text column

                data_col_inds = []
                data_cols = []
                text_cols = []
                for j in range(df.shape[1]):
                    if self._is_text(df.iloc[0, j]):
                        text_cols.append(df.iloc[:, j].to_numpy())
                    else:
                        data_cols.append(df.iloc[:, j].to_numpy().astype(np.float64))
                        data_col_inds.append(j)

                assert len(text_cols) <= 1

                header_text = []

                headers = df.columns
                if headers is not None and len(headers) > 0:
                    header_text = list(headers)

                all_text = np.concatenate([header_text, np.asarray(text_cols).flat])
                all_text = [s.lower() for s in all_text]
                all_text = np.unique(all_text).tolist()

                data_df = np.stack(data_cols, axis=1)
                if data_df.size > 0:
                    entry = (path, all_text, data_df)
                    self.heatmap_data_cache.append(entry)
                    yield entry
        else:
            for entry in self.heatmap_data_cache:
                yield entry


    def _dataset_iterator(self):
        if self.dataset_cache is None:
            self.dataset_cache = []

            for path in self.dataset_paths:
                df = read_csv(path)
                # see if there is a text column

                data_col_inds = []
                data_cols = []
                text_cols = []
                for j in range(df.shape[1]):
                    if self._is_text(df.iloc[0, j]):
                        text_cols.append(df.iloc[:, j].to_numpy())
                    else:
                        data_cols.append(df.iloc[:, j].to_numpy().astype(np.float64))
                        data_col_inds.append(j)

                assert len(text_cols) <= 1

                data_df = np.stack(data_cols, axis=1)
                if data_df.size == 0:
                    continue

                data_views = []
                if len(text_cols) == 1:
                    unique_text = np.unique(text_cols[0])
                    text_row_indices = dict()

                    for t in unique_text:
                        text_row_indices[t] = np.where(text_cols[0] == t)[0]

                    for j in range(data_df.shape[1]):
                        data_view = dict()
                        data = data_df[:, j]
                        for t in unique_text:
                            data_t = data[text_row_indices[t]]
                            data_view[t.lower()] = data_t # standardize to lower case
                        data_views.append(data_view)

                headers = df.columns
                if headers is not None and len(headers) > 0:

                    data_view = dict()

                    for col_ind, j in enumerate(data_col_inds):
                        data_col = data_df[:, col_ind]
                        data_view[headers[j]] = data_col

                    data_views.append(data_view)

                entry = (path, data_views)
                self.dataset_cache.append(entry)
                yield entry
        else:
            for entry in self.dataset_cache:
                yield entry

    @staticmethod
    def _data_view_nn_score(pairs, view, size_penalty=True):
        all_keys = view.keys()
        used_keys = set()

        text_hit = 0
        data_hit = 0
        total_distance = None

        for text, values in pairs:
            text = text.lower()

            target_data = None
            if text in all_keys:
                target_data = view[text]
                used_keys.add(text)
            else:
                for key in view.keys():
                    if text in key and key not in used_keys:
                        target_data = view[key]
                        used_keys.add(key)
                        break

            if target_data is not None:
                text_hit += 1

                if target_data.size >= len(values):
                    data_hit += 1

                    # compute distance
                    distances = []
                    arr = np.sort(target_data)
                    arr_size = arr.size

                    for d in values:
                        i = np.searchsorted(arr, d)

                        dl = dr = None

                        if i > 0:
                            dl = abs(arr[i - 1] - d)

                        if i < arr.size:
                            dr = abs(arr[i] - d)

                        if dl is None:
                            dl = dr + 1.0
                        if dr is None:
                            dr = dl + 1.0

                        if dl < dr:
                            i -= 1
                            distances.append(dl)
                        else:
                            distances.append(dr)

                        arr = np.delete(arr, i)

                    # compute total negative distance
                    total_neg_dist = -np.sum(distances)
                    # penalize based on the number of available candidates
                    if arr.size > 0 and size_penalty:
                        total_neg_dist = total_neg_dist * np.sqrt(arr_size)

                    if total_distance is None:
                        total_distance = total_neg_dist # non-positive, greater the better
                    else:
                        total_distance += total_neg_dist


        return text_hit, data_hit, total_distance

    @staticmethod
    def _solve_lin_interval(xs, ys, y):
        dy = np.diff(ys)
        dx = np.diff(xs)
        dx[np.abs(dx) < 1.0e-4] = 1.0e-4
        ks = dy / dx
        ks[np.abs(ks) < 1.0e-4] = 1.0e-4
        all_x = (y - ys[:-1]) / ks + xs[:-1]
        ret = []
        for i in range(all_x.size):
            if xs[i] <= all_x[i] <= xs[i+1]:
                ret.append(all_x[i])
        return np.asarray(ret)

    @staticmethod
    def _data_view_nn_interpolated_score(pairs, view):
        all_keys = view.keys()
        used_keys = set()

        text_hit = 0
        data_hit = 0
        total_distance = None

        for text, values, x_range in pairs:

            text = text.lower()

            target_data = None
            if text in all_keys:
                target_data = view[text]
                used_keys.add(text)
            else:
                for key in view.keys():
                    if text in key and key not in used_keys:
                        target_data = view[key]
                        used_keys.add(key)
                        break

            if target_data is not None:
                text_hit += 1

                # basic range check
                arr = np.asarray(target_data)
                arr_size = arr.size

                _, value_ys = zip(*values)
                if arr.size > 2 and np.min(value_ys) >= arr.min() and np.max(value_ys) <= arr.max():
                    data_hit += 1
                    # np.seterr(all='raise')
                    # get all possible solutions for those values
                    t = np.linspace(x_range[0], x_range[1], arr.size)

                    distances = []

                    # sort values by x
                    last_x_value = None
                    values.sort(key=lambda x: x[0])
                    for i in range(len(values)):
                        x, y = values[i]
                        # solve for y
                        x_solns = GraphDatasetMatcher._solve_lin_interval(t, arr, y)
                        if last_x_value is not None:
                            x_solns = x_solns[x_solns >= last_x_value]
                        if x_solns.size == 0:
                            distances.append(line_plot_match_penalty)
                        else:
                            x_diff = np.abs(x_solns - x)
                            best_x_ind = np.argmin(x_diff)
                            distances.append(x_diff[best_x_ind])
                            last_x_value = x_solns[best_x_ind]

                    # compute total negative distance
                    total_neg_dist = -np.sum(distances)

                    if total_distance is None:
                        total_distance = total_neg_dist # non-positive, greater the better
                    else:
                        total_distance += total_neg_dist


        return text_hit, data_hit, None if total_distance is None else float(total_distance)

    @staticmethod
    def _data_view_nn_interpolated_score_2d(pairs, view):
        all_keys = view.keys()
        used_keys = set()

        text_hit = 0
        data_hit = 0
        total_distance = None

        for text, values, x_range in pairs:

            text = text.lower()

            target_data = None
            if text in all_keys:
                target_data = view[text]
                used_keys.add(text)
            else:
                for key in view.keys():
                    if text in key and key not in used_keys:
                        target_data = view[key]
                        used_keys.add(key)
                        break
            
            unused_keys = set(all_keys).difference(used_keys)

            if target_data is not None:
                text_hit += 1

                # basic range check
                arr = np.asarray(target_data)
                arr_size = arr.size

                _, value_ys = zip(*values)
                if arr.size > 2 and np.min(value_ys) >= arr.min() and np.max(value_ys) <= arr.max():
                    data_hit += 1
                    
                    # we need to find a possible x column
                    x_columns = []
                    for key in unused_keys:
                        col = view[key]
                        if col.min() <= x_range[0] + 0.5 and col.max() >= x_range[1] - 0.5:
                            # # make sure it's monotonically increasing or decreasing
                            # if col.size > 1:
                            #     col_diff = np.diff(col)
                            #     if np.all(col_diff >= 0) or np.all(col_diff <= 0):
                            #         x_columns.append(col)
                            x_columns.append(col)

                    all_neg_dist = []

                    for x_col in x_columns:
                        assert x_col.size == arr.size

                        distances = []

                        for i in range(len(values)):
                            x, y = values[i]
                            x_solns = GraphDatasetMatcher._solve_lin_interval(x_col, arr, y)
                            if len(x_solns) > 0:
                                # find the nearest solution
                                x_diff = np.abs(x_solns - x)
                                best_x_ind = np.argmin(x_diff)
                                distances.append(x_diff[best_x_ind])
                            else:
                                distances.append(line_plot_match_penalty)

                        # compute total negative distance
                        total_neg_dist = -np.sum(distances)
                        all_neg_dist.append(total_neg_dist)

                    if len(all_neg_dist) > 0:
                        if total_distance is None:
                            total_distance = max(all_neg_dist) # non-positive, greater the better
                        else:
                            total_distance += max(all_neg_dist)


        return text_hit, data_hit, None if total_distance is None else float(total_distance)

    @staticmethod
    def _data_view_quantile_score(pairs, view):
        all_keys = view.keys()
        used_keys = set()

        text_hit = 0
        data_hit = 0
        total_distance = None

        for text, values in pairs:
            assert len(values) == 2
            text = text.lower()

            target_data = None
            if text in all_keys:
                target_data = view[text]
                used_keys.add(text)
            else:
                for key in view.keys():
                    if text in key and key not in used_keys:
                        target_data = view[key]
                        used_keys.add(key)
                        break

            if target_data is not None:
                text_hit += 1

                if target_data.size >= 4:
                    data_hit += 1

                    # compute distance
                    arr = np.sort(target_data)
                    quantiles = np.quantile(arr, [0.25, 0.75])
                    d1 = abs(quantiles[0] - values[0])
                    d2 = abs(quantiles[1] - values[1])

                    total_neg_dist = -(d1 + d2)

                    if total_distance is None:
                        total_distance = total_neg_dist # non-positive, greater the better
                    else:
                        total_distance += total_neg_dist

        return text_hit, data_hit, total_distance


    def _generic_score(self, pairs, score_func, score_sort_key):
        all_scores = []
        for df_path, df_views in self._dataset_iterator():
            for view in df_views:
                score = score_func(pairs, view)
                if score[2] is not None:
                    all_scores.append((df_path, score))

        if len(all_scores) > 0:
            all_scores.sort(key=score_sort_key, reverse=True)
            return all_scores[:min(len(all_scores), data_matching_top_k)]
        else:
            return None

    @staticmethod
    def _triplet_sort_key(x):
        return (x[1][0], x[1][1], x[1][2])

    # def _heatmap_score(self, parse_result):
    #     assert parse_result['type'] == 'heatmap_data'
    #     data = np.asarray(parse_result['data'])
    #     data_center_i, data_center_j = data.shape[0] // 2, data.shape[1] // 2
    #
    #     data_si_4, data_sj_4 = int(data.shape[0] / 3), int(data.shape[1] / 3)
    #
    #     data_h = data[data_center_i - data_si_4: data_center_i + data_si_4 + 1,
    #                     data_center_j - data_sj_4 : data_center_j + data_sj_4 + 1]
    #     data_h_flip_0 = np.flip(data_h, axis=0)
    #     data_h_flip_1 = np.flip(data_h, axis=1)
    #
    #     text_match_count = []
    #
    #     for data_path, all_text, df in self._heatmap_data_iterator():
    #         # determine text match
    #         match = 0
    #         for text in parse_result['text']:
    #             if text in all_text:
    #                 match += 1
    #         text_match_count.append((data_path, match))
    #
    #     text_match_count.sort(key=lambda x: x[1], reverse=True)
    #     most_text_match = text_match_count[0][1]
    #     df_candidate = []
    #     for df_path, count in text_match_count:
    #         if count < most_text_match:
    #             break
    #         df_candidate.append(df_path)
    #
    #     df_result = dict()
    #
    #     for data_path, all_text, df in self._heatmap_data_iterator():
    #
    #         if data_path not in df_candidate:
    #             continue
    #
    #         best_diff = None
    #
    #         # normalize df
    #         df = np.array(df, np.float64)
    #         df = df - df.min()
    #         df = df / df.max()
    #
    #         # determine text match
    #
    #         for _i in range(2):
    #             df_view = df if _i == 0 else df.transpose()
    #             if df_view.shape[0] >= data.shape[0] and df_view.shape[1] >= data.shape[1]:
    #                 best_diff = 1.0e30
    #
    #                 for i in range(df_view.shape[0] - data_h.shape[0]):
    #                     for j in range(df_view.shape[1] - data_h.shape[1]):
    #                         win = df_view[i : i + data_h.shape[0], j : j + data_h.shape[1]]
    #
    #                         for data_view in (data_h, data_h_flip_0, data_h_flip_1):
    #                             diff = np.sum(np.abs(win - data_view))
    #                             if diff < best_diff:
    #                                 best_diff = diff
    #
    #         if best_diff is not None:
    #             df_result[data_path] = best_diff
    #
    #     result = [(a, b) for a, b in df_result.items()]
    #     result.sort(key=lambda x: x[1])
    #
    #     return result[:data_matching_top_k]


    def _heatmap_score(self, parse_result):
        assert parse_result['type'] == 'heatmap_data'
        data = np.asarray(parse_result['data'])
        data_flip_1 = np.flip(data, axis=1)

        text_match_count = []

        for data_path, all_text, df in self._heatmap_data_iterator():
            # determine text match
            match = 0
            for text in parse_result['text']:
                if text in all_text:
                    match += 1
            text_match_count.append((data_path, match))

        text_match_count.sort(key=lambda x: x[1], reverse=True)
        most_text_match = text_match_count[0][1]
        df_candidate = []
        for df_path, count in text_match_count:
            if count < most_text_match:
                break
            df_candidate.append(df_path)

        df_result = dict()

        for data_path, all_text, df in self._heatmap_data_iterator():

            if data_path not in df_candidate:
                continue

            best_diff = None

            # normalize df
            df = np.array(df, np.float64)
            df = df - df.min()
            df = df / (df.max() + 1.0e-8)

            # find closest column

            for _i in range(2):
                df_view = df if _i == 0 else df.transpose()

                if df_view.shape[0] >= data.shape[0] and df_view.shape[1] >= data.shape[1]:
                    df_view_diff = [0.0] * 2

                    for _j in range(2):
                        data_view = data if _j == 0 else data_flip_1

                        # for each column...

                        for _k in range(data_view.shape[1]):
                            col = data_view[:, _k]

                            # find the closest column in this dataset
                            col_diff = np.inf

                            for i in range(df_view.shape[0] - col.size + 1):
                                win = df_view[i : i + col.size]

                                all_diff = np.abs(win - col.reshape(-1,1))
                                diff_sum = np.sum(all_diff, axis=0)
                                min_diff = np.min(diff_sum)
                                if min_diff < col_diff:
                                    col_diff = min_diff

                            df_view_diff[_j] += col_diff


                    df_view_diff = min(df_view_diff)

                    if best_diff is not None:
                        best_diff = min(best_diff, df_view_diff)
                    else:
                        best_diff = df_view_diff

            if best_diff is not None:
                df_result[data_path] = best_diff

        result = [(a, b) for a, b in df_result.items()]
        result.sort(key=lambda x: x[1])

        return result[:data_matching_top_k]

    def _nn_score(self, pairs, size_penalty=True):
        if size_penalty:
            return self._generic_score(pairs, self._data_view_nn_score, self._triplet_sort_key)
        else:
            pfunc = functools.partial(self._data_view_nn_score, size_penalty=size_penalty)
            return self._generic_score(pairs, pfunc, self._triplet_sort_key)

    def _quantile_score(self, pairs):
        return self._generic_score(pairs, self._data_view_quantile_score, self._triplet_sort_key)

    def _nn_interp_score(self, pairs):
        return self._generic_score(pairs, self._data_view_nn_interpolated_score, self._triplet_sort_key)

    def _nn_interp_score_2d(self, pairs):
        return self._generic_score(pairs, self._data_view_nn_interpolated_score_2d, self._triplet_sort_key)

    def match(self):
        if 'parse_result' not in self.parse_result:
            return None, 'no parsing results for this graph'

        if self.parse_result['graph_type'] in ['bar', 'hbar', 'stacked']:
            matching_pairs = []
            all_concentrated_regions = list(filter(lambda x: x['type'] == 'concentrated_region',
                                              self.parse_result['parse_result']))
            all_concentrated_regions.sort(key=lambda x: x['legend_name'])
            legend_groups = itertools.groupby(all_concentrated_regions, lambda x: x['legend_name'])
            for legend_name, legend_items in legend_groups:
                values = []
                for item in legend_items:
                    bbox = self._get_bbox_from_data_hull(item['data_hull'])
                    if self.parse_result['graph_type'] == 'hbar':
                        values.append(bbox[1])
                    else:
                        values.append(bbox[3])
                matching_pairs.append((legend_name, values))

                if self.parse_result['graph_type'] in ['bar', 'hbar']:
                    assert len(values) == 1

            match_result = self._nn_score(matching_pairs)
            return match_result
        elif self.parse_result['graph_type'] == 'box':
            matching_pairs = []
            for item in self.parse_result['parse_result']:
                if item['type'] == 'concentrated_region':
                    bbox = self._get_bbox_from_data_hull(item['data_hull'])
                    matching_pairs.append((item['legend_name'], [bbox[2], bbox[3]]))
            match_result = self._quantile_score(matching_pairs)
            return match_result
        elif self.parse_result['graph_type'] == 'scatter':
            matching_pairs = []
            x_var, y_var = [], []

            for item in self.parse_result['parse_result']:
                if item['type'] == 'data_sample':
                    sample = item['sample']
                    sample_x, sample_y = zip(*sample)
                    # see which one has the greater variance
                    x_var.append(np.var(sample_x))
                    y_var.append(np.var(sample_y))
            
            if len(x_var) == 0:
                return None

            if np.mean(x_var) > np.mean(y_var):
                use_x = True
            else:
                use_x = False

            for item in self.parse_result['parse_result']:
                if item['type'] == 'data_sample':
                    sample = item['sample']
                    sample_x, sample_y = zip(*sample)
                    if use_x:
                        matching_pairs.append((item['legend_name'], sample_x))
                    else:
                        matching_pairs.append((item['legend_name'], sample_y))


            match_result = self._nn_score(matching_pairs, size_penalty=False)
            return match_result
        elif self.parse_result['graph_type'] == 'violin':
            matching_pairs = []

            for item in self.parse_result['parse_result']:
                if item['type'] == 'concentrated_region':
                    hull = item['data_hull']
                    bbox = self._get_bbox_from_data_hull(hull)
                    polygon = Polygon(hull)
                    centroid = polygon.centroid
                    center_line_p1 = (centroid.x, bbox[2]-0.5)
                    center_line_p2 = (centroid.x, bbox[3]+0.5)
                    line = LineString([center_line_p1, center_line_p2])
                    itsc_points = line.intersection(polygon)
                    itsc_ymin, itsc_ymax = itsc_points.bounds[1], itsc_points.bounds[3]
                    new_points = [(centroid.x, itsc_ymin)]
                    for p in hull:
                        if p[0] <= centroid.x and itsc_ymin <= p[1] <= itsc_ymax:
                            new_points.append(p)
                    new_points.append((centroid.x, itsc_ymax))
                    new_points.sort(key=lambda x: x[1])
                    new_points = np.asarray(new_points)
                    new_points[:, 0] = centroid.x - new_points[:, 0]

                    # convert density to CDF
                    new_x = new_points[:, 1]
                    new_y = new_points[:, 0]
                    xy_interp = interp1d(new_x, new_y)
                    t = np.linspace(new_x.min(), new_x.max(), 50)
                    tv = xy_interp(t)
                    dt = t[1] - t[0]
                    trapz_values = []
                    for i in range(t.size - 1):
                        _y = tv[i: i+2]
                        _x = t[i: i+2]
                        _int = np.trapz(_y, _x, dx=dt)
                        trapz_values.append(_int)
                    trapz_cumsum = np.cumsum(trapz_values)
                    if trapz_cumsum.max() > 0.0:
                        trapz_cumsum = trapz_cumsum / trapz_cumsum.max()
                        q25_ind = np.searchsorted(trapz_cumsum, 0.25)
                        q75_ind = np.searchsorted(trapz_cumsum, 0.75)
                        q25_v = t[q25_ind]
                        q75_v = t[q75_ind]

                        matching_pairs.append((item['legend_name'], [q25_v, q75_v]))

            match_result = self._quantile_score(matching_pairs)
            return match_result
        elif self.parse_result['graph_type'] in ['line','area']:
            matching_pairs = []
            for item in self.parse_result['parse_result']:
                if item['type'] == 'data_sample':
                    sample = item['sample']
                    x_range = item['x_min'], item['x_max']
                    #matching_pairs.append((item['legend_name'], [bbox[2], bbox[3]]))
                    matching_pairs.append((item['legend_name'], sample, x_range))
            match_result = self._nn_interp_score(matching_pairs)
            return match_result
        elif self.parse_result['graph_type'] == 'heatmap':
            match_result = self._heatmap_score(self.parse_result['parse_result'])
            return match_result

        return None, 'unable to match this graph with a dataset'



