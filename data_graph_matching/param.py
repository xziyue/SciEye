text_dilation_radius=1
background_extraction_size=4
num_distinct_colors_for_heatmap = 45
legend_color_erosion_radius=8
bar_plot_iou_threshold = 0.75
#bar_plot_delta_e_threshold = 10.0
#bar_plot_clr_diff_threshold = 0.1
bar_plot_clr_diff_threshold = 23.0
bar_plot_binary_erosion_radius = 6
bar_plot_binary_closing_radius = 12
bar_plot_non_rectangle_threshold = 2
bar_box_plot_stddev_threshold = 4.0
hbar_stddev_threshold = 4.0
median_cut_depth = 8
bar_like_cc_size_coef = 0.02

line_plot_clr_diff_threshold = 10.0
line_plot_binary_opening_radius = 3
line_plot_num_sample = 8
line_plot_match_penalty = 10

scatter_plot_clr_diff_threshold = 10.0
scatter_plot_binary_erosion_radius = 1
scatter_plot_num_points_per_legend = 5

area_plot_clr_diff_threshold = 10.0
area_plot_binary_opening_radius = 3
area_plot_num_sample = 8

heatmap_sample_size = 12
heatmap_colorbar_size = 100


matching_text_weight = 1.0
matching_data_weight = 1.0
data_matching_max_scale = 4
data_matching_top_k = 15

parallel_n_jobs = 10
# coi_closing_radius=2
# coi_approx_poly_epsilon=0.01

graph_segmentation_labels = [
    'background', # background is always the 0-th element
    'graph_content',
    'x_label',
    'y_label',
    'x_tick',
    'y_tick',
    'colorbar',
    'colorbar_tick',
    'legend',
    'title'
]