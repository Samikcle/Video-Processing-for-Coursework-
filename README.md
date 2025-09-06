# Video Processing

This project implements a complete video post-processing pipeline using OpenCV and NumPy for a Digital Image Processing course. The code demonstrates practical applications of image and video enhancement, detection, and overlay techniques.

## Function of the program

1) Detect whether a video is taken during daytime or nighttime (e.g., based on average
brightness, histogram, etc.). If a video is found taken during nighttime, increase the
brightness of the video before moving to the next step. You can decide the amount of
brightness that should increase.
2) Blur all the faces (camera facing) that appear in a video.
3) Resize and overlay the video that she talks about her life as a YouTuber (talking.mp4)
on the top left of each video. You can decide the location to overlay the video.
4) Add different watermarks (watermark1.png and watermark2.png) to the videos to
protect them from being stolen.
5) Add the end screen video (endscreen.mp4) to the end of each video.

## What I Learned

Working on this project gave me hands-on experience with applying digital image processing techniques to real-world video workflows. Some key lessons I learned include:

1) Video I/O and Metadata Handling

    • How to use OpenCV to open, read, and write video streams frame by frame.

    • The importance of handling metadata like FPS, resolution, and frame count to ensure output videos remain consistent with the input.

2) Brightness and Illumination Adjustment

    • How to measure video brightness by converting frames to grayscale and computing average intensity.

    • How to automatically classify videos as “day” or “night” based on thresholding.

    • Techniques for safely brightening dark videos in HSV color space while avoiding pixel overflow.

3) Face Detection and Privacy Protection

    • Learned to use Haar cascade classifiers for detecting faces in frames.

    • Gained practice with region-based processing (selecting ROIs) and applying Gaussian blur to anonymize sensitive areas.

4) Overlaying and Blending Visuals
  
    • How to embed a smaller video into a main video, resize it, and add a border for clarity.

    • How to overlay watermarks using masks to blend foreground and background effectively.

5) Video Composition and Storytelling

    • Techniques for adding watermarks that alternate over time, improving security and branding.

    • How to append an endscreen clip seamlessly after the main video.

6) Visualization and Debugging

    • Using Matplotlib to plot brightness values across sampled frames helped in debugging and understanding illumination conditions.

Overall, this project helped me connect theoretical image processing concepts (filtering, masking, blending, intensity adjustment, and detection) with practical applications in video editing, security, and media production.
