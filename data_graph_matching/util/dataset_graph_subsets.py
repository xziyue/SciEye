train_dataset_graph_subset = {
    'bar': [f'graph_{i}' for i in range(1, 51)], # 0  -----
    'area': [f'graph_{i}' for i in range(51, 101)], # 1
    'heatmap': [f'graph_{i}' for i in range(101, 151)], # 2
    'line': [f'graph_{i}' for i in range(151, 201)], # 3 -----
    'hbar': [f'graph_{i}' for i in range(201, 251)], # 4 -----
    'violin': [f'graph_{i}' for i in range(251, 301)], # 5 -----
    'box': [f'graph_{i}' for i in range(301, 351)], # 6 -----
    'stacked_bar': [f'graph_{i}' for i in range(351, 401)], # 7 -----
    'scatter': [f'graph_{i}' for i in range(401, 451)] # 8 --------
}

