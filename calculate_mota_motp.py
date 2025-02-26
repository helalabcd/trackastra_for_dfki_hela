import os
import pickle
import networkx as nx

from evaluate_aogm.helpers import digraph_from_bust
from evaluate_aogm.aogm import calculate_AOGM

from ghelpers import graph_to_accumulator
import motmetrics as mm

trackastra_dfki_metrics = []

for split in ["train", "test"]:
    for f in [x for x in os.listdir(f"trackastra_graphs_nx/{split}") if x.endswith(".p")]:
        a = f.split(".")[0]
        print(a)

        label_graph = digraph_from_bust(a, split=split)
    
        print(f)

        trackastra_graph = pickle.load( open( f"trackastra_graphs_nx/{split}/" + f, "rb" ) )

        empty_graph = nx.empty_graph(1, create_using=nx.DiGraph)
        graph_true = label_graph
        graph_predicted = trackastra_graph

        # Create accumulator
        acc = mm.MOTAccumulator(auto_id=True)

        empty = nx.Graph()

        # Convert the graphs into the accumulator
        graph_to_accumulator(graph_true, graph_predicted, acc)

        # Print MOT events for inspection
        print("MOT events:")
        print(acc.mot_events)

        # Compute metrics (e.g., MOTA, MOTP)
        mh = mm.metrics.create()
        summary = mh.compute(acc, metrics=['num_frames', 'mota', 'motp'], name='acc')
        print("\nTracking summary:")
        print(summary)

        trackastra_dfki_metrics.append({
           "split": split,
           "burst": a,
           "mota": float(summary["mota"]),
           "motp": float(summary["motp"])
        })


pickle.dump( trackastra_dfki_metrics, open( "trackastra_dfki_metrics.p", "wb" ) )

