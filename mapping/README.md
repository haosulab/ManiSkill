# 3D Scene Scanning with ManiSkill

This project uses the ManiSkill 3 simulation framework to generate camera data, including intrinsics and poses, formatted to match the KITTI dataset. The primary goal is to create a synthetic dataset for developing and testing a 3D mapping pipeline. It features a custom environment where a camera orbits a table to capture images from multiple viewpoints, simulating a real-world data collection process.

---

## The Simulation Environment (`TableScan-v0`)

The core of the simulation is the custom `TableScan-v0` environment. Its main purpose is to create a simple, repeatable scene for scanning tasks.

### Key Features

* **Scene Setup**: The environment programmatically builds a scene with a table and places several simple cube objects with random colors on top of it.
* **Orbiting Camera**: Instead of being mounted on a robot arm, the camera is attached to a separate kinematic mount. This mount is programmed to follow a specific trajectory:
    1.  It starts at a set height and distance from the table's center.
    2.  It sweeps in a 180-degree arc around the table, capturing images along the way.
    3.  Once it reaches the end of the arc, it moves up to the next height level.
    4.  It then sweeps back in the opposite direction.
    5.  This process repeats for all predefined heights, ensuring comprehensive coverage of the scene.

---

## The Scanning Script (`scan.py`)

The `scan.py` script is the main executable that runs the simulation, captures the data, and generates the output files.

### Usage

The script is controlled via command-line flags to customize its behavior.

* **Basic Execution**
    This runs the simulation without a GUI and generates the point cloud files.
    ```bash
    python scan_table.py
    ```

* **Enable Viewer**
    Use the `--viewer` flag to open the SAPIEN GUI and watch the simulation in real-time.
    ```bash
    python scan_table.py --viewer
    ```

* **Visualize Camera Frustums**
    Use the `--frustum` flag to draw the camera's view cones (frustums) in the final HTML visualization.
    ```bash
    python scan_table.py --frustum
    ```

* **Combined Usage**
    You can combine flags to both watch the simulation and visualize the frustums.
    ```bash
    python scan_table.py --viewer --frustum
    ```

---

## Understanding the Output Files 📁

After running the script, an `images/` directory will be created in the same folder as `scan.py`. It contains the following files:

### 3D Model Files

* `point_cloud.ply`
    The final, merged 3D point cloud stored in the PLY (Polygon File Format).

* `point_cloud.html`
    An interactive 3D plot of the point cloud.

### Camera Data Files 📸

This is how the camera's intrinsic and extrinsic values are saved for each captured frame.

* `intrinsic.txt`
    This file stores the camera's **3x3 intrinsic matrix (K)**. This matrix defines the camera's internal properties like focal length and optical center. It is saved only once, as it doesn't change during the scan. The format is:

    $
    K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}
    $

    Where:
    * $f_x, f_y$ are the focal lengths in pixels.
    * $c_x, c_y$ are the coordinates of the principal point (the image center).

* `poses.txt`
    This file contains the **extrinsic pose** for each image that was captured. Each line in the file corresponds to one captured frame.
    * **Format**: Each line contains 12 space-separated numbers, which represent a flattened **3x4 camera-to-world transformation matrix**.
    * **Purpose**: This matrix defines the camera's position and orientation in the world for a specific frame. It tells you exactly where the camera was and where it was looking when it took the picture, which is essential for correctly placing the points from that frame into the final global point cloud.