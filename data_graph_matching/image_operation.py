import skimage
import skimage.io as skio
import skimage.color as skclr
import skimage.morphology as skmorph
import skimage.draw as skdraw
import skimage.measure as skmeasure
import skimage.morphology as skmorph
import skimage.draw as skdraw
import scipy.ndimage as ndimage
from scipy.spatial import ConvexHull
from dataclasses import dataclass
from typing import Tuple
import functools
import numpy as np
from data_graph_matching.common_util import run_length_encoding
import data_graph_matching.param as param
import cv2
from enum import Enum
from dataclasses_json import dataclass_json
from sklearn.cluster import KMeans
from data_graph_matching.util.dataset_analysis import *



class TextPosAgainstFigureContent(Enum):
    LEFT = 1
    RIGHT = 2
    ABOVE = 3
    BELOW = 4
    INSIDE = 5


@dataclass_json
@dataclass
class ImageText:
    text: str
    index: int
    points: Tuple[Tuple[int, int]]


class ImageOp:

    def __init__(self, img_item):
        self.has_text_info = True
        if isinstance(img_item, np.ndarray):
            self.has_text_info = False

        self.img_item = img_item

    # @functools.lru_cache(maxsize=1)
    def get_image(self):
        if isinstance(self.img_item, np.ndarray):
            return skimage.img_as_ubyte(self.img_item)
        elif isinstance(self.img_item, dict):
            path = self.img_item['filename']
            image_original = skio.imread(path)
            if image_original.shape[2] == 4:
                image_original = skclr.rgba2rgb(image_original)
            return skimage.img_as_ubyte(image_original)
        else:
            raise RuntimeError('unsupported image information type found')

    @staticmethod
    def image_to_gray(img):
        return skimage.img_as_ubyte(skclr.rgb2gray(img))

    # @functools.lru_cache(maxsize=1)
    def get_image_gray(self):
        return self.image_to_gray(self.get_image())

    # @functools.lru_cache(maxsize=1)
    # get the text and the mask
    def get_text(self, dilation=param.text_dilation_radius):
        if not self.has_text_info:
            raise RuntimeError('no text information is provided with this image')

        assert '\n' in self.img_item['text'][0]['t']
        image_shape = self.get_image().shape

        ret = []
        mask = np.empty(image_shape[:2], dtype=np.int32)
        mask.fill(-1)

        locations = []

        for text_ind, text in enumerate(self.img_item['text'][1:]):
            ws = [v[0] for v in text['v']]
            hs = [v[1] for v in text['v']]
            rc, cc = skdraw.polygon(hs, ws, shape=image_shape)

            # apply dilation
            if dilation > 0:
                mask.fill(0)
                mask[rc, cc] = 1
                mask = skmorph.binary_dilation(mask, skmorph.disk(dilation))
                rc, cc = np.where(mask == 1)

            locations.append((rc, cc))
            ret.append(
                ImageText(
                    text=text,
                    index=text_ind,
                    points=tuple((v[1], v[0]) for v in text['v'])
                )
            )

        # apply dilation
        # if dilation > 0:
        #     for item in ret:
        #         bin_mask = mask == item.index
        #         bin_mask = skmorph.binary_dilation(bin_mask, skmorph.disk(dilation))
        #         mask[bin_mask] = item.index

        return ret, locations

    # # get the background color of a graph
    # @functools.lru_cache(maxsize=1)
    # def get_background_color(self):
    #     # attain all text region
    #     text_entries, mask = self.get_text()
    #     image = self.get_image()
    #     total_counter = dict()
    #     for entry in text_entries:
    #         bin_mask_loc = np.where(mask == entry.index)
    #         text_pixels = image[bin_mask_loc[0], bin_mask_loc[1]]
    #         unique_colors, unique_counts = np.unique(text_pixels, return_counts=True, axis=0)
    #         for color, count in zip(unique_colors, unique_counts):
    #             color = tuple(color)
    #             if color not in total_counter:
    #                 total_counter[color] = 0
    #             total_counter[color] += count
    #
    #     most_color = max(total_counter.items(), key=lambda x: x[1])[0]
    #     return most_color

    def get_bottom_right_region(self, image=None):
        if image is None:
            image = self.get_image()
        return image[-param.background_extraction_size:, -param.background_extraction_size:]

    # background color is extracted from the bottom right corner of the graph
    def get_background_color(self):
        region = self.get_bottom_right_region()
        region = region.reshape((np.prod(region.shape[:2]), region.shape[2]))
        unique_clr = np.unique(region, axis=0)
        if unique_clr.shape[0] != 1:
            raise RuntimeError(f'unable to extract background color from graph: '
                               f'found {unique_clr.shape[0]} colors, expected 1')
        return unique_clr.squeeze()

    def remove_text_from_image(self, image=None, fill=None, text_dilation=0):
        if image is None:
            image = self.get_image().copy()
        if fill is None:
            fill = self.get_background_color()
        text_entries, locations = self.get_text(dilation=text_dilation)

        for entry in text_entries:
            loc = locations[entry.index]
            image[loc[0], loc[1], ...] = fill

        return image

    # get the position of the text wrt to the figure content
    # @functools.lru_cache(maxsize=1)
    def get_text_position(self):
        tinfo, locations = self.get_text()

        text_pos_flags = []
        for entry in tinfo:
            all_flags = []
            loc = locations[entry.index]
            mean_loc = np.mean(loc, axis=1)
            mean_loc = np.round(mean_loc).astype(np.int)
            ti, tj = mean_loc
            j_min, i_min, j_max, i_max = self.img_item['figure_content_box']

            # check if text is inside figure content area
            if i_min < ti < i_max and j_min < tj < j_max:
                all_flags.append(TextPosAgainstFigureContent.INSIDE)
            else:
                if ti <= i_min:
                    all_flags.append(TextPosAgainstFigureContent.ABOVE)
                if ti >= i_max:
                    all_flags.append(TextPosAgainstFigureContent.BELOW)
                if tj <= j_min:
                    all_flags.append(TextPosAgainstFigureContent.LEFT)
                if tj >= j_max:
                    all_flags.append(TextPosAgainstFigureContent.RIGHT)

            text_pos_flags.append(all_flags)

        return list(zip(tinfo, text_pos_flags))

    # align text in the image
    def get_text_alignment_kmeans(self):
        tinfo_pos = self.get_text_position()

        align_order = [TextPosAgainstFigureContent.LEFT,
                       TextPosAgainstFigureContent.BELOW,
                       TextPosAgainstFigureContent.RIGHT,
                       TextPosAgainstFigureContent.ABOVE]

        clusters = []

        for flag in align_order:
            if len(tinfo_pos) == 0:
                break

            # extract all remaining text items that has the current flag
            valid_tinfo_pos = []

            for i_tinfo_pos in enumerate(tinfo_pos):
                if flag in i_tinfo_pos[1][1]:
                    valid_tinfo_pos.append(i_tinfo_pos)

            if len(valid_tinfo_pos) == 0:
                continue

            cluster_proposals = []
            # for the four vertices
            for i in range(4):
                # for the two coordinates
                for j in range(2):
                    # extract this particular coordinate
                    all_coord = np.asarray([x[1][0].points[i][j] for x in valid_tinfo_pos]).reshape((-1, 1))
                    kmeans = KMeans(n_clusters=max(len(np.unique(all_coord)) // 3, 1))
                    kmeans.fit(all_coord)
                    cluster_proposals.append((kmeans.inertia_, (i, j), kmeans.predict(all_coord)))

            # find the best clustering approach
            cluster_proposals.sort(key=lambda x: x[0])
            cluster_inds = cluster_proposals[0][2]
            unique_cluster_inds = np.unique(cluster_inds)

            used_indices = []
            for cind in unique_cluster_inds:
                if np.count_nonzero(cluster_inds == cind) > 1:
                    cind_loc = np.where(cluster_inds == cind)[0]
                    # create a new cluster
                    cluster_info = dict(
                        text=[valid_tinfo_pos[x][1][0] for x in cind_loc],
                        vertex_index=cluster_proposals[0][1][0],
                        coordinate_index=cluster_proposals[0][1][1],
                        position=flag
                    )
                    clusters.append(cluster_info)
                    # remove points in this cluster from the set of all points
                    used_indices.extend(valid_tinfo_pos[x][0] for x in cind_loc)

            used_indices.sort(reverse=True)
            for ind in used_indices:
                tinfo_pos.pop(ind)

        # take out remaining unclustered text
        unclustered = [x[0] for x in tinfo_pos]
        return clusters, unclustered

    # align text in the image
    def get_text_alignment_threshold_based(self, threshold=2):
        tinfo_pos = self.get_text_position()

        align_order = [
                    TextPosAgainstFigureContent.ABOVE,
                    TextPosAgainstFigureContent.BELOW,
                    TextPosAgainstFigureContent.LEFT,
                    TextPosAgainstFigureContent.RIGHT,
                    ]

        clusters = []

        for flag in align_order:
            if len(tinfo_pos) == 0:
                break

            # extract all remaining text items that has the current flag
            valid_tinfo_pos = []

            for i_tinfo_pos in enumerate(tinfo_pos):
                if flag in i_tinfo_pos[1][1]:
                    valid_tinfo_pos.append(i_tinfo_pos)

            if len(valid_tinfo_pos) == 0:
                continue

            cluster_proposals = []
            # for the four vertices
            for i in range(4):
                # for the two coordinates
                for j in range(2):
                    # extract this particular coordinate
                    all_coord = np.asarray([x[1][0].points[i][j] for x in valid_tinfo_pos]).reshape((-1, 1))

                    if all_coord.size > 0:
                        cluster_info = []
                        used_array = np.zeros_like(all_coord, dtype=np.bool)
                        for k in range(len(all_coord)):
                            if not used_array[k]:
                                diff = all_coord - all_coord[k]
                                pos = np.where(np.abs(diff) < threshold)[0]
                                assert k in pos
                                # mark as a cluster
                                for p in pos:
                                    used_array[p] = True
                                cluster_info.append(pos)

                        # assert np.concatenate(cluster_info).size == all_coord.size
                        # create an indexing result
                        result_array = np.zeros_like(all_coord, np.int32)
                        for ind, cls in enumerate(cluster_info):
                            for ele_ind in cls:
                                result_array[ele_ind] = ind

                        # kmeans = KMeans(n_clusters=max(len(np.unique(all_coord)) // 3, 1))
                        # kmeans.fit(all_coord)

                        cluster_proposals.append((len(cluster_info), (i, j), result_array))

            # find the best clustering approach
            cluster_proposals.sort(key=lambda x: x[0])
            cluster_inds = cluster_proposals[0][2]
            unique_cluster_inds = np.unique(cluster_inds)


            # used_indices = []
            # for cind in unique_cluster_inds:
            #     if np.count_nonzero(cluster_inds == cind) > 1:
            #         cind_loc = np.where(cluster_inds == cind)[0]
            #         # create a new cluster
            #         cluster_info = dict(
            #             text=[valid_tinfo_pos[x][1][0] for x in cind_loc],
            #             vertex_index=cluster_proposals[0][1][0],
            #             coordinate_index=cluster_proposals[0][1][1],
            #             position=flag
            #         )
            #         clusters.append(cluster_info)
            #         # remove points in this cluster from the set of all points
            #         used_indices.extend(valid_tinfo_pos[x][0] for x in cind_loc)

            # we only focus on the biggest cluster in the best clustering proposal
            ind_counts = [(i, np.count_nonzero(cluster_inds == i)) for i in unique_cluster_inds]
            ind_counts.sort(key=lambda x: x[1])
            assert ind_counts[-1][1] > 0
            cind = ind_counts[-1][0]
            used_indices = []

            if ind_counts[-1][1] > 1:
                cind_loc = np.where(cluster_inds == cind)[0]
                # create a new cluster
                cluster_info = dict(
                    text=[valid_tinfo_pos[x][1][0] for x in cind_loc],
                    vertex_index=cluster_proposals[0][1][0],
                    coordinate_index=cluster_proposals[0][1][1],
                    position=flag
                )
                clusters.append(cluster_info)
                # remove points in this cluster from the set of all points
                used_indices.extend(valid_tinfo_pos[x][0] for x in cind_loc)

            used_indices.sort(reverse=True)
            for ind in used_indices:
                tinfo_pos.pop(ind)

        # take out remaining unclustered text
        # add position information to unclustered text
        unclustered = []
        for i in range(len(tinfo_pos)):
            x = tinfo_pos[i]
            x_dict = x[0].to_dict()
            x_dict['position'] = [y.value for y in x[1]]
            unclustered.append(x_dict)
        #unclustered = [x[0] for x in tinfo_pos]
        return clusters, unclustered


    def get_legend_colors(self, legend_text_group, delta_e_threshold=1.0, j_offset=3):
        image = self.get_image()
        assert image.dtype == np.uint8

        legend_colors = []

        for item in legend_text_group['text']:
            v = np.flip(np.asarray(item['text']['v']), axis=1)
            # find the vertex with smallest j index
            v_max = np.max(v, axis=0)
            v_min = np.min(v, axis=0)

            i_max, j_max = v_max
            i_min, j_min = v_min
            # i_center = np.mean([i_min, i_max]).astype(np.int)
            
            img_line = image[i_min : i_max, ...]
            img_line_lab = skclr.rgb2lab(img_line)

            # sample bg color
            bg_color_1 = img_line_lab[0, j_min - j_offset * 2]
            bg_color_2 = img_line_lab[-1, j_min - j_offset * 2]
            if skclr.deltaE_cie76(bg_color_1, bg_color_2) > 2.0:
                # cannot find a consistent background color
                continue
            bg_color = bg_color_1

            delta_e = skclr.deltaE_cie76(img_line_lab, bg_color)
            img_line_mask = delta_e > delta_e_threshold
            #skio.imsave(os.path.join(binary_path, 'img_line_mask.png'), img_line_mask)

            labels = skmeasure.label(img_line_mask)

            # import matplotlib.pyplot as plt
            # plt.imshow(labels, cmap='viridis', interpolation='none')
            # plt.savefig(os.path.join(binary_path, 'img_labels.png'), dpi=300)

            # find the cluster that is closest to the text
            cluster_info = []
            for i in range(1, labels.max() + 1):
                pixel_loc = np.where(labels == i)
                cluster_center = np.mean(pixel_loc, axis=1)
                #print(i, cluster_center)
                assert cluster_center.size == 2
                if cluster_center[1] < j_min - j_offset and len(pixel_loc[0]) > 3:
                    cluster_info.append((i, j_min - cluster_center[1]))

            best_cluster_ind, _ = min(cluster_info, key=lambda x: x[1])
            best_cluster_loc = np.where(labels == best_cluster_ind)


            # sometimes legends has frames...
            # we need to get rid of that
            cluster_points = np.transpose(np.asarray(best_cluster_loc))
            try:
                chull = ConvexHull(cluster_points)
            except:
                continue
            poly_r, poly_c = skdraw.polygon(cluster_points[chull.vertices, 0], cluster_points[chull.vertices, 1])
            poly_mask = np.zeros_like(labels, np.int8)
            poly_mask[poly_r, poly_c] = 1
            for radius in range(param.legend_color_erosion_radius, 0, -1):
                #poly_mask = skmorph.binary_erosion(poly_mask, skmorph.square(radius))
                # use the erosion from ndimage with supports border value assignment
                poly_mask = ndimage.binary_erosion(poly_mask, skmorph.square(radius))
                if np.count_nonzero(poly_mask) > 0:
                    best_cluster_loc = np.where(poly_mask == 1)
                    break
            

            best_cluster_loc_min = np.min(best_cluster_loc, axis=1)
            best_cluster_loc_max = np.max(best_cluster_loc, axis=1)
            bbox = dict(i_min=int(best_cluster_loc_min[0] + i_min), j_min=int(best_cluster_loc_min[1]), 
                        i_max=int(best_cluster_loc_max[0] + i_min), j_max=int(best_cluster_loc_max[1]))
            best_cluster_pixels = img_line[best_cluster_loc[0], best_cluster_loc[1]]
            best_cluster_color = np.mean(best_cluster_pixels, axis=0)
            legend_color = best_cluster_color
            legend_color = legend_color.tolist()
            legend_colors.append((item['text']['t'], legend_color, bbox))

            # delta_e = skclr.deltaE_cie76(img_line, bg_color)
            # far_away = delta_e > delta_e_threshold
            # far_away_rle = run_length_encoding(far_away)

            # if len(far_away_rle[0]) >= 2:
            #     if far_away_rle[2][-1] == False and far_away_rle[2][-2] == True:
            #         color_range_l = far_away_rle[1][-2]
            #         color_range_r = color_range_l + far_away_rle[0][-2]
            #         legend_color = img_line[color_range_l : color_range_r]
            #         legend_color = skclr.lab2rgb(np.mean(legend_color, axis=0)) * 255.0
            #         legend_color = legend_color.tolist()
            #         legend_colors.append((item['text']['t'], legend_color))

        # the result is in RGB format
        return legend_colors
            



    # # get contents of interest from the graph
    # def get_coi(self):
    #     # run self.get_background_color to check the integrity of the graph
    #     self.get_background_color()

    #     image = self.get_image()
    #     labels = skmeasure.label(self.image_to_gray(image))

    #     bottom_right_label = self.get_bottom_right_region(labels)[0,0]
    #     # not a good idea to remove text from the original image because some suptitle may bleed into the figure region...
    #     # now, removing text region from the connected components...
    #     labels = self.remove_text_from_image(image=labels, fill=bottom_right_label)
    #     non_mask = labels != bottom_right_label
    #     non_mask = skmorph.binary_closing(non_mask, skmorph.disk(param.coi_closing_radius))

    #     # plt.imshow(non_mask)
    #     # plt.show()

    #     output_mask = np.empty(labels.shape, np.int32)
    #     output_mask.fill(-1)

    #     # detect shapes in this mask
    #     contours, h = cv2.findContours(non_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #     for cnt in contours:
    #         approx = cv2.approxPolyDP(cnt, param.coi_approx_poly_epsilon * cv2.arcLength(cnt, True), True)
    #         if len(approx) == 4:
    #             cv2.drawContours(output_mask, [cnt], 0, int(output_mask.max()+1), -1)

    #     return output_mask


if __name__ == '__main__':
    from data_graph_matching.util.dataset_analysis import *
    from data_graph_matching.util.dataset_graph_subsets import train_dataset_graph_subset
    from data_graph_matching.util.dataset_manager import DatasetManager
    import numpy as np
    import matplotlib.pyplot as plt

    plt.rcParams['figure.dpi'] = 150


    # def path_tf(p):
    #     return p.replace('/home/xiang71/data-graph-matching/',
    #                      '/data2/xiang71/data-graph-matching-cri-conf-2022/')

    path_tf = lambda x : x

    dsman = DatasetManager(path_tf=path_tf)
    graph_name = 'graph_1'
    graph_item = dsman.get_train_item_by_name('graph', graph_name)
    image_op = ImageOp(graph_item)
    ret = image_op.get_text_alignment_threshold_based()
    for item in ret[0]:
        print(item)

    # rnd_state = np.random.RandomState(42)
    # output_result_dir = os.path.join(binary_path, 'image_operation_test')
    # if not os.path.exists(output_result_dir):
    #     os.makedirs(output_result_dir)

    # for graph_type, graphs in train_dataset_graph_subset.items():
    #     graph_sample_ind = rnd_state.choice(np.arange(len(graphs)), size=4, replace=False)
    #     graph_sample = [graphs[x] for x in graph_sample_ind]

    #     for graph_name in graph_sample:
    #         fig, axs = plt.subplots(1, 2)
    #         img_op = ImageOp(dsman.get_train_item_by_name('graph', graph_name))
    #         coi = img_op.get_coi()
    #         axs[0].imshow(img_op.get_image())
    #         axs[1].imshow(coi)
    #         plt.savefig(os.path.join(output_result_dir, f'{graph_name}.png'))
    #         plt.close()

    # img_op = ImageOp(dsman.get_train_item_by_name('graph', 'graph_403'))
    # img_op.get_coi()

    # img = img_op.get_image()
    # plt.imshow(img)
    # plt.show()
    #
    # img = img_op.get_image_gray()
    # plt.imshow(img, cmap='gray')
    # plt.show()

    # text_entries, mask = img_op.get_text(dilation=1)
    #
    # img = img_op.get_image()
    #
    # for entry in text_entries:
    #     entry_mask = np.where(mask == entry.index)
    #     img[entry_mask[0], entry_mask[1]] = (1.0, 0.0, 0.0)
    #
    # plt.imshow(img)
    # plt.show()

    # print(img_op.get_background_color())

    # img_new = img_op.remove_text_from_image()
    # plt.imshow(img_new)
    # plt.show()

    # img_op.get_coi()
