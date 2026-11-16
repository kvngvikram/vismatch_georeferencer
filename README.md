# VisMatch Georeferencer (QGIS plugin)

Semi-automatic GCP picking with any [vismatch](https://github.com/gmberton/vismatch) matcher, without ever loading the full rasters into RAM. Only the part you have zoomed into is rendered and matched.

## Install
1. Install vismatch in **QGIS's** Python (OSGeo4W Shell on Windows, the QGIS Python on Linux/macOS):
   `python -m pip install vismatch` (torch is pulled in; use a CPU or CUDA build as you like).
2. Copy the `vismatch_georef` folder to your QGIS plugins folder (Settings → User Profiles → Open Active Profile Folder → `python/plugins`), or use *Plugins → Manage and Install Plugins → Install from ZIP*. Enable it.
3. Open it from **Raster → VisMatch Georeferencer**.

## Quick start
1. Choose the **Basemap (georeferenced)** and the **Raster to register**.
2. Zoom both views to roughly the same area.
3. Type a **Model** name (autocomplete lists `vismatch.available_models`, default `superpoint-lightglue`) and click **1. Match current windows**. Weights download on first use.
4. Check the lines joining the matched keypoints in the preview, then click **2. Accept → add GCPs**.
   Keyboard shortcuts: press **1** to match, **2** to accept and **3** to delete the last accepted region (they don't fire while you are typing in a text or number box, or when the button is disabled).
5. Pan to another region and repeat.
6. When you have enough GCPs, click **Warp with GDAL…** or **Open QGIS Georeferencer**.

## Settings
| Setting | What it does |
|---|---|
| **Match size (px)** | Size (longest side) of the image of each zoomed view given to the matcher. Larger gives more detail and better precision but uses more memory and time. It does not change what the canvases show. |
| **RANSAC thresh (px)** | Maximum error for a match to count as an inlier (`ransac_reproj_thresh`). Raise it if you get too few inliers; lower it for stricter, more accurate matches. Some models ignore it. |
| **Max GCPs / region** | Maximum number of GCPs kept from one match. |
| **GCP selection** | *Vismatch score* keeps the highest-confidence inliers (falls back to spatial spread if the model gives no scores). *Spatial spread* keeps a well-distributed subset. |
| **Min inliers** | A match with fewer inliers than this is rejected. |
| **Device** | `auto` tries CUDA, then MPS, then CPU. `cpu` works everywhere; `cuda` is an NVIDIA GPU; `mps` is the GPU of Apple Silicon Macs. Choosing a device your system lacks makes the match fail. |
| **Source coords in .points** | *Auto* detects whether the raster already has a georeference. Switch to *Pixel* or *Layer map coords* if points look shifted in the Georeferencer. |

## Working with GCPs
- **Table:** click a column header to sort (click again to reverse). Click a row to zoom both canvases to that GCP (the "Zoom to GCP" box sets how many pixels are shown across).
- **Select from the canvas:** hover over a GCP marker (the cursor becomes a pointing hand) and click to select its table row without moving the canvas.
- **Residuals:** the *res px* and *res map* columns, the RMSE label and the yellow lines on the canvases follow the selected fitting method (Polynomial 1/2/3 or Thin plate spline). Lines are at true scale; raise *Residual line ×* if they are too small to see. The spline passes through every GCP, so its residuals are zero.
- **Manual GCPs:** turn on **+ Manual GCP**, click the point on the basemap and then the same point on the other image (any order). Manual GCPs get region 0. Pan with middle or right mouse drag, zoom with the wheel; a plain left click only places a point.
- **Sync views:** keeps both canvases on the same area using the selected fitting method and shows a blue X on the other canvas at the location matching your cursor. It needs 3 / 6 / 10 / 3 GCPs (Polynomial 1 / 2 / 3 / spline). Polynomials can drift far outside the area covered by the GCPs.
- **Match preview:** mouse wheel zooms at the cursor, left-drag pans, double-click resets.
- **3. Delete last accepted** removes every GCP of the most recently accepted region from the table and canvases (manual GCPs are kept). **Delete selected** and **Clear all** also remove GCPs.

## Output
- **Save .points:** the file name defaults to `<raster>.points`; you confirm it in the dialog. It is rewritten after every accept or manual GCP once a file is chosen.
- **Open QGIS Georeferencer:** QGIS has no Python API for that window, so open your raster there and use *Load GCP Points*. The `.points` file uses the basemap CRS, with source Y negative (rows downward).
- **Warp with GDAL…:** warps the raster with the chosen fitting method (bilinear resampling, compressed tiled GeoTIFF) and adds the result to QGIS. The log shows progress, and **Cancel** stops the warp midway and deletes the partial file. Choose a new output name if you want to keep an existing file.

## Multi-band and special images
- **Matching** uses what QGIS draws: a 3-channel 8-bit image from the bands your layer style shows (usually bands 1 to 3 as RGB; a single band is drawn as grey). Change the band mapping or contrast stretch in the layer's Symbology tab. For 16-bit or float data, a fixed min/max helps matching.
- **Warping** processes every band at the original data type and does not set a nodata value or alpha band. Empty areas may be black unless the source file has a nodata value.

## Tips
- Windows at similar scale and orientation match best.
- If you run out of memory, lower **Match size**.
- Too few inliers: raise the RANSAC threshold, zoom in so both views show the same area at a similar scale, or try a different model.
- GCPs look shifted in the Georeferencer: change **Source coords in .points**.
