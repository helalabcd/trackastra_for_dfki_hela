import os
from PIL import Image
from glob import glob
import torch
from trackastra.model import Trackastra
from trackastra.data import example_data_bacteria
from trackastra.tracking import graph_to_ctc, graph_to_napari_tracks
import numpy as np
import pickle
import tifffile as tiff
import networkx as nx

def convert_graph(b):
    """
    Converts a graph 'b' with node attributes:
      - 'time': time frame (0-based)
      - 'coords': a tuple (x, y)
      - plus other attributes (like 'label', 'weight') that are ignored in the conversion
      
    into a new graph with:
      - sequential integer keys (0, 1, 2, ...)
      - node attributes 't', 'x', 'y'
        where 't' is (time + 1), and (x, y) are taken from 'coords'
      
    Edges from the original graph are copied over with the new node IDs.
    """
    # Create a new directed graph (use Graph() if the original is undirected)
    newG = nx.DiGraph()
    
    # Build a mapping from original node ID to new sequential node ID
    mapping = {}
    for new_id, old_id in enumerate(b.nodes()):
        data = b.nodes[old_id]
        # Convert the time attribute: add 1 so that frame 0 becomes t=1
        t = data.get('time', 0) + 1
        # Extract coordinates
        coords = data.get('coords', (None, None))
        x, y = coords
        # Add the node with the new attributes

        # NOTE: Please note how the dimensions are flipped:
        newG.add_node(new_id, t=t, x=y, y=x)
        mapping[old_id] = new_id

    # Re-map edges using the new node IDs
    for u, v in b.edges():
        newG.add_edge(mapping[u], mapping[v])
        
    return newG

HELAPATH = os.getenv('helapath')
HELAMASKPATH = os.getenv('helamaskpath')

os.system("mkdir trackastra_graphs")
os.system("mkdir trackastra_graphs_nx trackastra_graphs_nx/train trackastra_graphs_nx/test")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load a pretrained model
model = Trackastra.from_pretrained("general_2d", device=device)

def fix_chroma_subsampling(folder_path, output_folder):

    os.makedirs(output_folder, exist_ok=True)

    # Get all .tiff files
    tiff_files = sorted(glob(f"{folder_path}/*.tiff"))

    for file in tiff_files:
        try:
            with Image.open(file) as img:
                img = img.convert("RGB")  # Convert to RGB (removes chroma subsampling)
                output_file = os.path.join(output_folder, os.path.basename(file))
                img.save(output_file, format="TIFF")  # Save in a standard TIFF format
                print(f"✅ Converted: {file} → {output_file}")
        except Exception as e:
            print(f"❌ Error processing {file}: {e}")


def load_tiff_images(directory: str) -> np.ndarray:
    """Loads all TIFF images from a directory into a NumPy array."""
    file_paths = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".tiff")])
    images = [tiff.imread(fp) for fp in file_paths]
    return np.array(images)

def example_data_tiff(images_dir, masks_dir) -> list[tuple[np.ndarray, dict, str]]:
    """Loads TIFF images and masks from the specified base directory."""
    
    imgs = load_tiff_images(images_dir)[:, :, :, 0]
    masks = load_tiff_images(masks_dir)
    
    return [
        imgs,
        masks,
    ]




for split in ("train", "test"):

    splitpath = os.path.join(HELAPATH, split)
    for burst in os.listdir(splitpath):
        tmp_imgpath = f"/tmp/{split}_{burst}"
        fix_chroma_subsampling(os.path.join(splitpath, burst, "img1"), tmp_imgpath)
        
        imgs_path = tmp_imgpath
        masks_path = os.path.join(HELAMASKPATH, split, burst)

        print(imgs_path)
        print(masks_path)

        imgs, masks = example_data_tiff(imgs_path, masks_path)
        print(imgs.shape, masks.shape)

        track_graph = model.track(imgs, masks, mode="greedy")

        print("Graph", track_graph)

        ctc_tracks, masks_tracked = graph_to_ctc(
          track_graph,
          masks,
          outdir=f"trackastra_graphs/{burst}",
        )

        pickle.dump( convert_graph(track_graph), open( f"trackastra_graphs_nx/{split}/{burst}.p", "wb" ) )
        

