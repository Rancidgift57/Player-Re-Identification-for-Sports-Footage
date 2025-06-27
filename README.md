This detailed README outlines a **Player Tracking System** designed for sports videos, specifically 15-second, 720p highlight clips. The system ingeniously combines several advanced computer vision and deep learning techniques to provide robust and consistent player tracking across various scenes within a video.

-----

## Overview

This project delivers a sophisticated **Player Tracking System** for sports videos, focusing on 15-second highlight clips in 720p resolution. It's engineered to track individual players seamlessly, even through scene changes. The system achieves this by integrating:

  * **Object Detection:** Identifying players within each frame.
  * **Person Re-identification (Re-ID):** Maintaining consistent player identities across frames and scenes.
  * **Jersey Number Recognition:** Extracting jersey numbers for enhanced identification.
  * **Homography-Based Field Position Mapping:** Translating player locations to a standardized field coordinate system.

By leveraging cutting-edge deep learning models and computer vision algorithms, this system provides highly reliable player identification and tracking capabilities, even when camera angles shift or players are temporarily obscured. The output is an enriched video stream, overlaid with essential tracking information such as bounding boxes, stable player IDs, detected jersey numbers, and precise field coordinates for each tracked player.

-----

## Key Components

The system's robust functionality is built upon the synergy of the following core components:

  * **YOLOv8 for Person Detection:** Utilizes the state-of-the-art YOLOv8 model to accurately detect human figures (players) in real-time within each video frame.
  * **ResNet50 for Feature Extraction and Player Re-identification:** Employs a pre-trained ResNet50 model to extract distinctive feature embeddings from detected players. These embeddings are crucial for recognizing the same player even after occlusions, changes in pose, or scene transitions.
  * **DeepSORT for Multi-Object Tracking:** Integrates the DeepSORT algorithm, which combines appearance-based re-identification with traditional Kalman filtering. This allows for persistent tracking of multiple players, assigning stable IDs and managing occlusions effectively.
  * **Tesseract OCR for Jersey Number Recognition:** Leverages the powerful Tesseract OCR engine to identify jersey numbers from cropped player images. Enhanced image preprocessing steps are applied to improve accuracy even with varying image quality.
  * **PySceneDetect for Scene Detection:** Automatically detects scene boundaries within the video. This is vital for handling camera angle changes and applying the correct precomputed homography matrices.
  * **Homography for Field Position Mapping:** Transforms pixel coordinates of detected players in the video frame into real-world field coordinates. This relies on precomputed homography matrices specific to different camera angles/scenes.

-----

## Features

This player tracking system offers a comprehensive set of features, making it a powerful tool for sports analytics and visualization:

  * **High-Accuracy Player Detection:** Employs **YOLOv8** to pinpoint players in every frame with impressive precision, ensuring that nearly all players are identified.
  * **Consistent Player Re-Identification:** Leverages **ResNet50** to generate robust feature embeddings, guaranteeing that each player maintains a consistent and unique ID throughout the entire video, even across different scenes or when briefly out of sight.
  * **Reliable Jersey Number Recognition:** Incorporates **Tesseract OCR** with sophisticated image preprocessing to accurately detect jersey numbers, adding another layer of identification for players.
  * **Stable Multi-Object Tracking:** Integrates **DeepSORT** to manage and maintain player tracks with stable IDs, effectively handling common challenges like occlusions, temporary disappearances, and missed detections.
  * **Automated Scene Boundary Detection:** Utilizes **PySceneDetect** to automatically identify changes in camera angles or scene transitions, allowing the system to dynamically apply the appropriate homography transformations for accurate field mapping.
  * **Precise Field Position Mapping:** Maps the real-time positions of players from the video frame to a standardized field coordinate system using **precomputed homography matrices**, providing valuable spatial data.
  * **Intuitive Real-Time Visualization:** Presents real-time tracking results directly on the video stream, displaying clear bounding boxes, persistent player IDs, detected jersey numbers (when available), and precise field coordinates for immediate insight.

-----

## Requirements

To set up and run this player tracking system, ensure you have the following prerequisites installed:

### Software Dependencies

  * **Python:** Version 3.8 or higher is required.

  * **Python Libraries:** Install all necessary libraries using pip. A `requirements.txt` file is provided for convenience.

    ```bash
    pip install opencv-python torch torchvision ultralytics numpy pytesseract scenedetect deep-sort-realtime scipy
    ```

    Alternatively, if you have a `requirements.txt` file:

    ```bash
    pip install -r requirements.txt
    ```

    The `requirements.txt` file should contain:

    ```
    opencv-python
    torch
    torchvision
    ultralytics
    numpy
    pytesseract
    scenedetect
    deep-sort-realtime
    scipy
    ```

  * **Tesseract OCR:** This is crucial for jersey number recognition.

      * **On Windows:** Download and install Tesseract from the official GitHub repository: [https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki). After installation, you'll need to set the path to the executable in your code (e.g., `./highlights/tesseract.exe`).
      * **On Linux:** Install via apt-get:
        ```bash
        sudo apt-get install tesseract-ocr
        ```
      * **On macOS:** Install via Homebrew:
        ```bash
        brew install tesseract
        ```

### Pretrained Models

The system relies on the following pretrained deep learning models, which are typically handled automatically or can be downloaded:

  * **YOLOv8:** The `yolov8n.pt` model will be automatically downloaded by the Ultralytics library upon its first use.
  * **ResNet50:** Pretrained weights are retrieved from `torchvision.models.ResNet50_Weights.IMAGENET1K_V1`, which will be downloaded by `torchvision` as needed.

### Hardware Recommendations

  * **CUDA-compatible GPU:** Highly recommended for significantly faster processing and real-time performance. The system is designed to automatically utilize a CUDA-enabled GPU if available.
  * **CPU:** If a GPU is not available, the code will seamlessly fall back to CPU processing, though performance will be slower, especially for high-resolution videos.

-----

## File Structure

The project expects a specific directory structure to ensure all necessary files are accessible. Please arrange your files as follows:

```
project_directory/
├── highlights/
│   ├── 15sec_input_720p.mp4          # The main input video file (15-second, 720p highlight clip)
│   ├── homography_angle1.npy          # Precomputed homography matrix for scene 1
│   ├── homography_angle2.npy          # Precomputed homography matrix for scene 2
│   ├── ...                           # Additional homography matrices for other scenes
│   ├── tesseract.exe                 # Tesseract OCR executable (primarily for Windows installations)
├── player_tracking.py                # The main script that runs the player tracking system
├── README.md                         # This README file
```

### Important Notes on Files:

  * **Input Video:** The `15sec_input_720p.mp4` file is your primary video input. Ensure it is placed in the `highlights/` directory and matches the specified format.
  * **Homography Files:** These `.npy` files contain precomputed homography matrices. Each file should correspond to a specific scene or camera angle within your video. The system will attempt to load these based on detected scenes. If a specific homography file is missing for a scene, the system will default to an identity matrix, which will result in inaccurate field coordinates.
  * **Tesseract Executable:** If you are on Windows, place the `tesseract.exe` file in the `highlights/` directory or ensure its path is correctly specified in the `player_tracking.py` script.

-----

## Installation

Follow these steps to get the player tracking system up and running on your local machine:

1.  **Clone or Download the Repository:**
    Start by obtaining the project files. You can clone the Git repository:

    ```bash
    git clone <repository_url>
    cd player_tracking_system
    ```

    Or, if you downloaded a zip file, extract its contents.

2.  **Install Required Python Packages:**
    Navigate to the project's root directory in your terminal and install all necessary Python libraries using the provided `requirements.txt` file:

    ```bash
    pip install -r requirements.txt
    ```

    This command will install `opencv-python`, `torch`, `torchvision`, `ultralytics`, `numpy`, `pytesseract`, `scenedetect`, `deep-sort-realtime`, and `scipy`.

3.  **Install Tesseract OCR:**
    As mentioned in the requirements, install Tesseract OCR based on your operating system. After installation, **it's crucial to update the path to the Tesseract executable** within the `player_tracking.py` script. Locate the line that sets `pytesseract.pytesseract.tesseract_cmd` and modify it to point to your Tesseract installation (e.g., `pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'` for Windows, or just ensure it's in your system's PATH for Linux/macOS).

4.  **Place Input Video and Homography Files:**
    Create a `highlights/` directory within your project's root folder. Place your input video (`15sec_input_720p.mp4`) and all precomputed homography files (`homography_angleX.npy`) inside this directory. Ensure the homography files are appropriately named (e.g., `homography_angle1.npy` for the first scene detected, `homography_angle2.npy` for the second, and so on) to correspond with the scene changes in your video.

-----

## Usage

Once all dependencies are installed and the file structure is correctly set up, you can run the player tracking system:

1.  **Verify Setup:**
    Double-check that all required Python libraries are installed, Tesseract OCR is correctly configured, and your `highlights/` directory contains the input video and relevant homography files.

2.  **Update Configuration Variables (Optional):**
    Open the `player_tracking.py` script. You'll find several configuration variables at the beginning of the script that you might want to adjust based on your specific video or performance needs:

      * `HIGHLIGHT_PATH`: The path to your input video file.
      * `HOMOGRAPHY_DIR`: The directory containing your homography matrices.
      * `YOLO_MODEL`: The YOLO model file to use (default: `yolov8n.pt`).
      * `FEATURE_THRESHOLD`: A critical parameter for Re-ID matching. Lower values mean stricter matching (default: `0.25`).
      * `JERSEY_CONFIDENCE`: The minimum confidence required for a detected jersey number to be accepted (default: `0.7`).
      * `MIN_TRACK_LENGTH`: The minimum number of frames a track must persist to be considered confirmed (default: `5`).
      * `MAX_MISSED_FRAMES`: The maximum number of frames a track can go without being updated before it is considered lost and removed (default: `20`).

3.  **Run the Script:**
    Execute the main script from your terminal:

    ```bash
    python player_tracking.py
    ```

4.  **Observe Output:**
    A new window will appear, displaying the processed video with real-time tracking results. You will see bounding boxes, stable player IDs, detected jersey numbers, and field coordinates overlaid on each frame.
    To exit the visualization, simply press the `q` key.

-----

## How It Works: A Detailed Breakdown

The player tracking system operates through a meticulously orchestrated pipeline of computer vision and deep learning processes:

1.  **Video Input Processing:**
    The system initiates by reading the specified 15-second, 720p video file from the designated path (`HIGHLIGHT_PATH`). Each frame is then passed sequentially through the subsequent processing stages.

2.  **Scene Detection:**
    Before processing begins, **PySceneDetect** is employed to analyze the video and automatically identify scene boundaries. This is crucial because camera angles and perspectives often change between scenes in sports highlights. A configurable threshold (default: `30.0`) determines the sensitivity to content changes. Detecting these boundaries allows the system to apply the correct homography matrix for each distinct scene.

3.  **Player Detection:**
    For every incoming video frame, **YOLOv8** (specifically, the `yolov8n.pt` model) is used to detect all instances of "person" (class 0). The confidence threshold for detection is intentionally lowered (default: `0.2`) to improve recall, ensuring that even partially obscured or less prominent players are initially detected for further processing.

4.  **Feature Extraction for Re-Identification:**
    Once players are detected, **ResNet50** (with ImageNet pretrained weights) extracts a 2048-dimensional feature vector (embedding) for each bounding box. These feature vectors are rich in visual information, capturing unique characteristics of each player's appearance. These embeddings are fundamental for the re-identification process, allowing the system to recognize the same player even when they move, change pose, or reappear after being occluded.

5.  **Jersey Number Detection:**
    To further refine player identification, cropped images of detected players are passed to **Tesseract OCR**. Before OCR is applied, these images undergo a series of enhanced preprocessing steps:

      * **Grayscale Conversion:** Reduces complexity and focuses on intensity.
      * **Gaussian Blur:** Smooths out noise and helps in character segmentation.
      * **Otsu Thresholding:** Dynamically binarizes the image, separating text from background.
      * **Morphological Operations (e.g., erosion, dilation):** Further refines character shapes and removes small artifacts, optimizing the image for OCR.
        This meticulous preprocessing significantly improves the accuracy of jersey number detection, even in challenging lighting or resolution conditions.

6.  **Multi-Object Tracking with DeepSORT:**
    The core of the tracking mechanism lies with **DeepSORT**. This algorithm integrates the bounding box detections and the appearance-based feature embeddings (from ResNet50) to maintain stable tracks. It intelligently associates current detections with existing tracks, minimizing ID switches. A crucial aspect is the **global player database**, which stores the feature embeddings and IDs of players seen across different scenes. This ensures that players maintain consistent IDs even when they disappear in one scene and reappear in another with a different camera angle. The **FEATURE\_THRESHOLD** (cosine distance) plays a vital role here: a lower value means a stricter match is required for re-identification. DeepSORT also utilizes parameters like `max_age` (set to `MAX_MISSED_FRAMES`) and `n_init` (set to `MIN_TRACK_LENGTH`) to manage track lifecycle, confirming tracks only after they persist for a certain duration and removing them if lost for too long.

7.  **Homography Mapping:**
    For each detected scene, the system identifies the corresponding precomputed **homography matrix** (e.g., `homography_angle1.npy`). This matrix is then used to transform the pixel coordinates of each tracked player's position (typically the bottom-center of their bounding box) from the image plane to a standardized 2D field coordinate system. This provides a consistent and measurable representation of player locations on the sports field, regardless of the camera's perspective. If a homography file is missing for a scene, an identity matrix is used, resulting in inaccurate field coordinates.

8.  **Real-time Visualization:**
    Finally, the processed frames are displayed in real-time. On each frame, the system overlays:

      * **Green Bounding Boxes:** Encircling each detected and tracked player.
      * **Stable Player IDs:** A unique, persistent numerical ID assigned to each player, maintained across frames and scenes.
      * **Jersey Numbers:** Displayed next to the player's ID if detected with sufficient confidence (`JERSEY_CONFIDENCE`).
      * **Field Coordinates:** The player's estimated (x, y) position on the field, based on the homography mapping.
        Concurrently, the console logs provide useful information, such as the number of detections per frame and the timestamps of detected scene boundaries.

-----

## Configuration Parameters

The `player_tracking.py` script contains several parameters that can be adjusted to fine-tune the system's performance for different videos or specific requirements.

  * **`DEVICE`**:

      * **Description**: Automatically determines the computational device to use.
      * **Value**: `cuda` if a CUDA-compatible GPU is detected; otherwise, `cpu`. You generally don't need to change this manually.

  * **`FEATURE_THRESHOLD`**:

      * **Description**: This is the cosine distance threshold used by DeepSORT for feature matching during re-identification. It dictates how similar feature embeddings must be for two detections to be considered the same player.
      * **Default**: `0.25`
      * **Impact**: **Lower values are stricter**, meaning a higher similarity is required for a match, which can reduce ID switches but may lead to more lost tracks. **Higher values are more lenient**, potentially improving track continuity but increasing the risk of ID switches between different players.

  * **`JERSEY_CONFIDENCE`**:

      * **Description**: The minimum confidence score required for a jersey number detected by Tesseract OCR to be accepted and displayed.
      * **Default**: `0.7` (70%)
      * **Impact**: **Higher values** lead to fewer, but more accurate, jersey number displays. **Lower values** will display more numbers, but with a higher chance of incorrect detections.

  * **`MIN_TRACK_LENGTH`**:

      * **Description**: The minimum number of consecutive frames a newly created track must be observed before it is considered a "confirmed" track and assigned a stable ID.
      * **Default**: `5` frames
      * **Impact**: **Higher values** reduce "false positive" tracks but might delay the appearance of new player IDs. **Lower values** confirm tracks faster but might lead to more spurious, short-lived tracks.

  * **`MAX_MISSED_FRAMES`**:

      * **Description**: The maximum number of frames a track can go without being updated (i.e., without a new detection being associated with it) before it is considered lost and removed from the active tracking list.
      * **Default**: `20` frames
      * **Impact**: **Higher values** allow tracks to persist through longer occlusions or missed detections. **Lower values** make the tracker more responsive to disappearances but increase the likelihood of losing tracks during brief occlusions.

  * **DeepSORT Specific Parameters (Internal Configuration):**
    These parameters are often set directly within the DeepSORT initialization based on the above high-level parameters for consistency.

      * **`max_age`**:

          * **Description**: Corresponds directly to `MAX_MISSED_FRAMES`. Maximum number of frames a track is kept alive without an associated detection.
          * **Value**: Set to `MAX_MISSED_FRAMES`.

      * **`n_init`**:

          * **Description**: Corresponds directly to `MIN_TRACK_LENGTH`. Number of consecutive detections required to initialize a new track.
          * **Value**: Set to `MIN_TRACK_LENGTH`.

      * **`nn_budget`**:

          * **Description**: Maximum number of feature vectors (appearance descriptors) stored for each track. These are used for appearance-based re-identification.
          * **Default**: `100`
          * **Impact**: A larger budget allows for more robust re-identification over longer periods but consumes more memory.

      * **`max_cosine_distance`**:

          * **Description**: Maximum cosine distance allowed between a detection's feature vector and a track's stored feature vectors for association.
          * **Value**: Set to `FEATURE_THRESHOLD`. This is the core similarity metric for Re-ID.

-----

## Output

When the `player_tracking.py` script is executed, the system produces the following outputs:

  * **Video Visualization Window:**
    A dedicated display window will open, showing the input video in real-time with several annotations overlaid on each tracked player:

      * **Bounding Box:** A green rectangular box drawn around the detected player, indicating their precise location in the frame.
      * **Stable ID:** A unique, persistent numerical ID assigned to each player. This ID remains consistent for the same player across different frames, scenes, and even after brief occlusions, ensuring reliable identification.
      * **Jersey Number:** If successfully detected with a confidence level above the `JERSEY_CONFIDENCE` threshold, the player's jersey number will be displayed alongside their stable ID.
      * **Field Coordinates:** The player's estimated position on the field, mapped to a standardized coordinate system, displayed as `(x, y)` coordinates. This provides valuable spatial information for tactical analysis.

  * **Console Logs:**
    The terminal where you ran the script will display real-time logs providing insights into the system's operation:

      * **Detections per Frame:** Information on the number of players detected in each processed video frame.
      * **Scene Boundaries:** Notifications when the system detects a scene change, indicating the start and end frame of each distinct scene. This helps in understanding when different homography matrices are being applied.

-----

## Troubleshooting

Here are common issues you might encounter and how to address them:

  * **Video Not Found:**

      * **Problem:** The system reports that the input video file cannot be found.
      * **Solution:** Double-check the `HIGHLIGHT_PATH` variable in `player_tracking.py` to ensure it correctly points to the `15sec_input_720p.mp4` file within your `highlights/` directory. Verify the file name and path for any typos.

  * **Homography File Missing:**

      * **Problem:** Field coordinates appear incorrect or are not displayed for certain scenes. The console might log a warning about a missing homography file.
      * **Solution:** Ensure that you have precomputed and placed all necessary homography matrices (`homography_angleX.npy`) in the `highlights/` directory, and that their names correctly correspond to the scene angles detected by `PySceneDetect`. If a file is missing, the system will use an identity matrix, leading to inaccurate field position mapping.

  * **Tesseract Errors (e.g., "TesseractNotFoundError", "Command 'tesseract' not found"):**

      * **Problem:** Jersey number detection fails, and you see errors related to Tesseract.
      * **Solution:** This typically means Tesseract OCR is not installed correctly or its executable path is not properly configured.
        1.  **Installation:** Re-verify that Tesseract OCR is installed on your system (refer to the "Requirements" section).
        2.  **Path Configuration:** In `player_tracking.py`, locate the line `pytesseract.pytesseract.tesseract_cmd = '...'`. Update this line to the absolute path of your Tesseract executable. For example:
              * **Windows:** `pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'`
              * **Linux/macOS:** Ensure `tesseract` is in your system's PATH. If not, provide the full path like `/usr/bin/tesseract` or `/opt/homebrew/bin/tesseract`.

  * **Performance Issues (Slow Processing):**

      * **Problem:** The video processing is very slow or not running in real-time.
      * **Solution:**
        1.  **Use a GPU:** A CUDA-compatible GPU is highly recommended. Ensure your system has one and that PyTorch is configured to use it (check `DEVICE` variable in logs).
        2.  **Reduce Video Resolution:** While the system is designed for 720p, for testing on lower-end hardware, you could temporarily use a lower resolution input video if performance is critical.
        3.  **Adjust Model:** Consider using an even smaller YOLO model (e.g., `yolov8n-tiny.pt` if available, though `yolov8n.pt` is already lightweight) if provided by Ultralytics, but this might slightly reduce detection accuracy.

  * **Tracking Issues (ID Switches, Lost Tracks):**

      * **Problem:** Players' IDs frequently switch, or tracks are lost prematurely during occlusions or brief disappearances.
      * **Solution:** Adjust the tracking configuration parameters in `player_tracking.py`:
          * **`FEATURE_THRESHOLD`**:
              * **If too many ID switches**: Try **lowering** this value (e.g., to `0.20` or `0.15`) to make Re-ID matching stricter, reducing the chance of associating different players.
              * **If too many lost tracks**: Try **increasing** this value slightly (e.g., to `0.30`) to make Re-ID matching more lenient, allowing tracks to persist through minor appearance changes.
          * **`MIN_TRACK_LENGTH`**:
              * **If many spurious, short-lived tracks**: **Increase** this value (e.g., to `8` or `10`) to require more frames before a track is confirmed.
          * **`MAX_MISSED_FRAMES`**:
              * **If tracks are lost during long occlusions**: **Increase** this value (e.g., to `30` or `40`) to allow tracks to persist for longer without updates.
              * **If tracks are "ghosting" after players leave**: **Decrease** this value (e.g., to `10` or `15`) to remove lost tracks more quickly.

-----

## Limitations

While highly capable, the current player tracking system has certain limitations:

  * **Jersey Number Detection Accuracy:**

      * The accuracy of jersey number recognition can be significantly affected by video quality. Low-resolution, blurry, or poorly lit images of players may lead to failed or incorrect detections.
      * While preprocessing steps are implemented, highly challenging visual conditions can still pose an issue. Further fine-tuning of preprocessing or OCR configuration might be needed for specific scenarios.

  * **Impact of Scene Changes:**

      * Although scene detection is integrated, sudden and drastic changes in camera angles or field layouts between scenes can still disrupt tracking.
      * The system relies on the availability and accuracy of precomputed homography files for each distinct scene. If a homography file is missing or inaccurate for a particular scene, the field coordinates will be incorrect.

  * **Occlusions:**

      * While DeepSORT is designed to mitigate the effects of occlusions (when players are temporarily hidden behind others or objects), severe or prolonged occlusions can still lead to track loss or ID switches.
      * The performance under heavy occlusion depends largely on the quality and distinctiveness of the extracted player features.

  * **Lighting Variations:**

      * Significant variations in lighting conditions across different frames or scenes can degrade the effectiveness of feature matching for re-identification. Players might appear sufficiently different under poor lighting to be considered new individuals, leading to ID switches.

-----

## Future Improvements

The current system provides a strong foundation, and several avenues exist for further enhancement:

  * **Adaptive Thresholding for Jersey Number Detection:** Implement dynamic adjustment of preprocessing parameters (like thresholding values) for Tesseract OCR based on the real-time quality and characteristics of the cropped player images. This could significantly improve robustness to varying image conditions.
  * **Dynamic Homography Estimation:** Incorporate methods for on-the-fly or semi-automatic homography estimation. This would reduce the reliance on precomputed matrices, making the system more adaptable to new environments or videos where homography files are unavailable. Techniques like vanishing point detection or feature matching to a known field template could be explored.
  * **Team Classification:** Add a module for classifying players by team, potentially based on jersey color, team logos, or other visual cues. This would add another layer of valuable information for sports analytics.
  * **Optimization for Real-time Performance:** Further optimize the feature extraction and tracking pipelines to achieve true real-time performance on lower-end hardware, making the system more accessible without requiring high-end GPUs.
  * **Exporting Tracking Results:** Implement functionality to save the comprehensive tracking results (player IDs, bounding box coordinates, jersey numbers, field coordinates) to structured file formats such as JSON, CSV, or XML. This would enable easy integration with external analysis tools and databases for deeper post-processing and statistical analysis.
  * **Player Event Detection:** Integrate modules for detecting specific player actions or events, such as shots, passes, or tackles, building upon the core tracking data.

-----

## License

This project is open-source and distributed under the **MIT License**. For full details regarding usage, modification, and distribution, please refer to the `LICENSE` file included in the project repository.

-----

## Contact

For any questions, feedback, or potential contributions to this project, please feel free to:

 
  * **Contact the project maintainer**
      LinkedIn: https://www.linkedin.com/in/nikhil-nair-809248286?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app
      Email: nnair7598@gmail.com
