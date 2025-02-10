# Cardiomyocyte Contraction Analysis

This repository contains a set of Python scripts designed to analyze cardiomyocyte beating videos. By converting each frame of the video into an image and applying Digital Image Correlation (DIC), the scripts produce vector fields representing pixel-by-pixel motion. Subsequent calculations on these vector fields provide insights into the contraction and relaxation of cardiomyocytes, which are visualized as heatmaps overlaid (or placed side by side with) the original video.

---

## Table of Contents
1. [Overview](#overview)  
2. [Repository Structure](#repository-structure)  
3. [How It Works](#how-it-works)  
4. [Contact](#contact)  

---

## Overview

1. **Frame Extraction**  
   - A script (`video_to_image_conversion.py`) that takes a cardiomyocyte beating video and converts each frame into an individual image.

2. **Digital Image Correlation (DIC)**  
   - Python scripts (named starting with `strain_field_...`) that analyze the extracted images using DIC.  
   - These scripts generate **vector fields** that represent how each pixel in the image moves between frames.

3. **Heatmap Visualization**  
   - Additional scripts compute the divergence of the vectors to visualize contraction/relaxation.  
   - Other scripts (with `vector_size` in the filename) focus on plotting the magnitude of the vectors as a heatmap.  
   - Each script can produce:  
     1. **Overlay**: Heatmap on top of the original video.  
     2. **Side-by-Side**: Heatmap video placed next to the original.  
     3. **Heatmap-Only**: Returns just the heatmap as a standalone video.

---

## Repository Structure

```plaintext
.
├── video_to_image_conversion.py
├── strain_field_*.py
├── strain_field_vector_size_*.py
└── README.md
```

> **Note:** Filenames may vary slightly. The naming convention shown above is for illustrative purposes.

- **video_to_image_conversion.py**  
  Converts the cardiomyocyte beating video into individual image frames.

- **strain_field_*.py**  
  Calculates the divergence of vector fields (indicating contraction or relaxation) and creates a heatmap.

- strain_field_vector_size_*.py**  
  Applies DIC to sequential image frames to generate vector fields representing pixel displacement and visualizes the magnitude of the vectors (overall displacement) as a heatmap.

---

## How It Works

1. **Frame Extraction**  
   - The process begins by taking the raw video of the beating cardiomyocyte and splitting it into a sequence of image frames.

2. **Vector Field Computation (DIC)**  
   - Each pair of consecutive frames is analyzed using DIC to track how each pixel moves from one frame to the next.  
   - The result is a **vector field** showing displacement vectors for every pixel.

3. **Analysis and Visualization**  
   - **Divergence Calculation**:  
     - Scripts compute the divergence (a measure of spatial extent of pixel motion, which can indicate contraction or relaxation).  
     - Results are shown as color-coded heatmaps.  
   - **Magnitude Heatmap**:  
     - Instead of divergence, some scripts create heatmaps based on the magnitude (length) of each displacement vector.  
   - **Video Output Variants**:  
     - **Overlay**: Heatmap is superimposed on the original frame.  
     - **Side-by-Side**: Heatmap and original video frames are placed next to each other.  
     - **Heatmap-Only**: A video of the heatmap alone, allowing clearer visualization of the displacement patterns.

---

## Contact

If you have any questions regarding this project or the included scripts, please reach out to:

- **Changhyeon Kim**  
  *Email:* [changhyeon.kim123@gmail.com](mailto:changhyeon.kim123@gmail.com)

- **Dr. Juyong Yoon, KIST Europe**  
  *Email:* [juyong.yoon@kist-europe.de](mailto:juyong.yoon@kist-europe.de)

We welcome any feedback, suggestions, or inquiries about the code or research behind this project.

---

**Thank you for your interest in this project!**  
Feel free to open an [issue](../../issues) if you encounter any problems or have specific questions.
