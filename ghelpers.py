import numpy as np
import motmetrics as mm
import networkx as nx

def graph_to_accumulator(gt_graph, pred_graph, acc):
    """
    Update a motmetrics accumulator with data from ground truth and predicted graphs.
    
    Each graph is expected to have nodes with attributes:
        - 't': frame number
        - 'x': x-coordinate (centroid)
        - 'y': y-coordinate (centroid)
    
    Parameters:
      gt_graph: networkx graph representing ground truth detections.
      pred_graph: networkx graph representing predicted detections.
      acc: a motmetrics.MOTAccumulator instance.
    """
    # Determine all frame numbers from both graphs
    frames = set()
    for _, data in gt_graph.nodes(data=True):
        frames.add(data['t'])
    for _, data in pred_graph.nodes(data=True):
        frames.add(data['t'])
    frames = sorted(frames)
    
    # Process frame-by-frame
    for t in frames:
        # Get ground truth detections for frame t: (node_id, attributes)
        gt_nodes = [(n, data) for n, data in gt_graph.nodes(data=True) if data['t'] == t]
        # Get predicted detections for frame t
        pred_nodes = [(n, data) for n, data in pred_graph.nodes(data=True) if data['t'] == t]
        
        # Use node ids as detection identifiers
        gt_ids = [n for n, _ in gt_nodes]
        pred_ids = [n for n, _ in pred_nodes]
        
        # Build the cost (distance) matrix between each ground truth and prediction
        cost_matrix = []
        if gt_nodes and pred_nodes:
            for _, gt_data in gt_nodes:
                row = []
                for _, pred_data in pred_nodes:
                    dx = gt_data['x'] - pred_data['x']
                    dy = gt_data['y'] - pred_data['y']
                    dist = np.sqrt(dx**2 + dy**2)
                    row.append(dist)
                cost_matrix.append(row)
        
        # Update the accumulator for this frame
        acc.update(gt_ids, pred_ids, cost_matrix)


