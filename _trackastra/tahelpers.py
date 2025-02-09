import os
import glob
import numpy as np
import tifffile

from model import Trackastra
from tracking import graph_to_ctc, ctc_to_napari_tracks


def read_tiff_stack(folder: str) -> np.ndarray:
    """
    Read all .tif/.tiff files in `folder`, sort them by name,
    and stack them into a numpy array of shape (T, Y, X) (or (T, Z, Y, X), etc.)

    Returns
    -------
    stack : np.ndarray
        The stacked image data. The first axis is assumed to be time (T).
    """
    # Gather all tiff files in the given folder
    tiff_files = sorted(
        glob.glob(os.path.join(folder, "*.tif")) + glob.glob(os.path.join(folder, "*.tiff"))
    )
    if not tiff_files:
        raise ValueError(f"No .tif or .tiff files found in {folder}")

    # Read each file and store as a list of numpy arrays
    frames = []
    for fn in tiff_files:
        data = tifffile.imread(fn)
        frames.append(data)

    # Stack along the first dimension => shape = (T, Y, X) or (T, Z, Y, X), etc.
    stack = np.stack(frames, axis=0)
    return stack


def run_tracking(
    imgs_folder: str,
    masks_folder: str,
    model_source: str = "general_2d",
    device: str | None = None,
    mode: str = "greedy_nodiv",
    max_distance: int = 128,
    outdir: str | None = None,
):
    """
    Read a folder of image files and a folder of mask files, then run Trackastra tracking.

    Parameters
    ----------
    imgs_folder : str
        Path to the folder containing .tiff image files.
    masks_folder : str
        Path to the folder containing .tiff label mask files.
    model_source : str
        - If a known pretrained model name (e.g. "general_2d"), loads that model.
        - If a path to a folder, loads a custom trained model from there.
    device : str or None
        Torch device to use ("cpu", "cuda", etc.). If None, Trackastra tries to auto-detect.
    mode : str
        Linking mode for Trackastra: "greedy_nodiv", "greedy", or "ilp".
    max_distance : int
        The maximum distance allowed for linking objects from frame to frame.
    outdir : str or None
        If provided, results are saved in standard CTC folder format at `outdir`.

    Returns
    -------
    track_graph : networkx.DiGraph
        The raw track graph returned by Trackastra.
    masks_tracked : np.ndarray
        Same shape as input masks, with instance IDs tracked consistently across frames.
    napari_tracks : np.ndarray
        An array of shape (N, 4) or (N, 5) representing the tracking data in napari format.
    napari_tracks_graph : dict
        Graph metadata for napari’s Tracks layer (optional).
    """
    # 1) Read the image and mask stacks
    print(f"Reading images from {imgs_folder}")
    imgs = read_tiff_stack(imgs_folder)

    print(f"Reading masks from {masks_folder}")
    masks = read_tiff_stack(masks_folder)

    if imgs.shape != masks.shape:
        raise ValueError(
            f"Image shape {imgs.shape} doesn't match mask shape {masks.shape}.\n"
            "Make sure both folders contain the same number of frames, in the same order."
        )

    # 2) Load the Trackastra model
    if os.path.isdir(model_source):
        # Assume it's a folder containing a custom trained model
        model = Trackastra.from_folder(model_source, device=device)
    else:
        # Otherwise treat it as a key for a pretrained model
        model = Trackastra.from_pretrained(model_source, device=device)

    print(f"Running Trackastra tracking with mode='{mode}' and max_distance={max_distance}...")
    # 3) Run the tracking
    track_graph = model.track(
        imgs,
        masks,
        mode=mode,
        max_distance=max_distance,
        # If you want a progress bar in the console:
        # progbar_class=tqdm.tqdm,
    )

    # 4) Convert track graph to “CTC style” + tracked masks
    df, masks_tracked = graph_to_ctc(track_graph, masks, outdir=None)

    # 5) Convert to napari-friendly tracks format (optional)
    napari_tracks, napari_tracks_graph = ctc_to_napari_tracks(
        segmentation=masks_tracked,
        man_track=df
    )

    # 6) (Optional) Save results in CTC format if outdir is provided
    if outdir is not None:
        print(f"Saving CTC-format results to {outdir} ...")
        from napari_ctc_io import save_labels, save_tracks

        # Create outdir if needed
        os.makedirs(outdir, exist_ok=True)

        # Save tracked masks => outdir/masks_tracked/mask000.tif, mask001.tif, etc.
        save_labels(masks_tracked, outdir)

        # Save tracks => outdir/tracks/man_track.txt, etc.
        save_tracks(napari_tracks, outdir, graph=napari_tracks_graph)

    return track_graph, masks_tracked, napari_tracks, napari_tracks_graph

