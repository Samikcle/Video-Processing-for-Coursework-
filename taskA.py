import cv2
import numpy as np
import matplotlib.pyplot as plt

# List of input video files to process
video_files = ["alley.mp4", "traffic.mp4", "office.mp4", "singapore.mp4"]

# Supporting assets: overlay video, watermark images, and endscreen
overlay_video = "talking.mp4"
watermark1 = cv2.imread("watermark1.png")
watermark2 = cv2.imread("watermark2.png")
endscreen_video = "endscreen.mp4"

# Helper function to open input video and create output writer
def get_video_data(input_path, output_path):
    vid = cv2.VideoCapture(input_path)
    if not vid.isOpened():
        raise IOError(f"Cannot open video: {input_path}")

    # Get metadata
    fps = vid.get(cv2.CAP_PROP_FPS)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize writer with MP4 codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    return vid, out, fps, width, height, total_frames

# Determine whether a video is recorded at night based on brightness
def is_night_video(vid, sample_frames=30, video_name="", brightness_threshold = 100):
    brightness_values = []

    # Sample the first few frames and calculate their average brightness
    for i in range(sample_frames):
        success, frame = vid.read()
        if not success:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        brightness_values.append(brightness)

    vid.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video position to start

    avg = np.mean(brightness_values)
    is_night = avg < brightness_threshold

    # Plot brightness graph for debugging
    plt.figure()
    plt.plot(brightness_values, marker='o', color='blue')
    plt.axhline(y=brightness_threshold, color='red', linestyle='--', label='Threshold')
    plt.title(f"Brightness Over First {len(brightness_values)} Frames: {video_name}")
    plt.xlabel("Frame Index")
    plt.ylabel("Average Brightness")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"{video_name}: Avg brightness = {avg:.2f} → {'NIGHT' if is_night else 'DAY'}")
    return is_night

# Increase brightness of a frame 
def brighten_frame(img, value=50): # Adjus the value number to change howw much the brightness is increased
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Truncation to prevent overflow
    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img
        
# Process brightness for a whole video if it is deemed to be recorded at night
def process_video_brightness(input_path, output_path, sample_frames=30):
    vid, out, fps, width, height, total_frames = get_video_data(input_path, output_path)

    # Check if video is taken at night
    is_night = is_night_video(vid, sample_frames, input_path)

    for _ in range(total_frames):
        success, frame = vid.read()
        if not success:
            break

        # Brighten frame only if it's a night video
        if is_night:
            frame = brighten_frame(frame)

        out.write(frame)

    vid.release()
    out.release()
    print(f"Processed brightness and saved to: {output_path}")
        
# Blur all faces detected in the video 
def blur_faces(input_path, output_path):
    face_cascade_path = 'face_detector.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    vid, out, fps, width, height, total_frames = get_video_data(input_path, output_path)

    for _ in range(total_frames):
        success, frame = vid.read()
        if not success:
            break

        # Detect faces
        faces = face_cascade.detectMultiScale(frame, 1.3, 5)

        # Apply Gaussian blur to detected faces
        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            blurred_face = cv2.GaussianBlur(face_roi, (99, 99), 30)
            frame[y:y+h, x:x+w] = blurred_face

        out.write(frame)

    vid.release()
    out.release()
    print(f"Face-blurred and saved to: {output_path}")
    
# Add a small overlay video to the top-left corner of the main video
def add_overlay(input_path, overlay_path, output_path):
    vid, out, fps, width, height, total_frames = get_video_data(input_path, output_path)
    overlay_vid = cv2.VideoCapture(overlay_path)

    if not overlay_vid.isOpened():
        raise IOError("Cannot open overlay video.")

    # Resize and position overlay
    # Adjust these values to change overlay size and position
    overlay_width = int(width * 0.25)
    overlay_height = int(height * 0.25)
    offset_x, offset_y = 50, 50
    border_thickness = 3
    border_color = (30, 50, 20)

    for _ in range(total_frames):
        success_main, main_frame = vid.read()
        if not success_main:
            break

        success_overlay, overlay_frame = overlay_vid.read()
        if not success_overlay:
            # Restart overlay if it ends before main video
            overlay_vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success_overlay, overlay_frame = overlay_vid.read()
            if not success_overlay:
                break

        # Resize and frame the overlay video
        overlay_resized = cv2.resize(overlay_frame, (overlay_width, overlay_height))
        overlay_framed = cv2.copyMakeBorder(
            overlay_resized,
            top=border_thickness,
            bottom=border_thickness,
            left=border_thickness,
            right=border_thickness,
            borderType=cv2.BORDER_CONSTANT,
            value=border_color
        )

        # Define overlay position
        y1, y2 = offset_y, offset_y + overlay_framed.shape[0]
        x1, x2 = offset_x, offset_x + overlay_framed.shape[1]

        # Paste overlay onto main frame
        if y2 <= height and x2 <= width:
            main_frame[y1:y2, x1:x2] = overlay_framed

        out.write(main_frame)

    vid.release()
    overlay_vid.release()
    out.release()
    print(f"Overlay video added and saved to: {output_path}")

# Blend watermark into a frame using masking
def overlay_watermark(frame, watermark):
    lower = np.array([0, 0, 0], dtype=np.uint8)
    upper = np.array([60, 60, 60], dtype=np.uint8)
    mask = cv2.inRange(watermark, lower, upper)
    mask_inv = cv2.bitwise_not(mask)

    # Extract foreground from watermark and background from frame
    fg = cv2.bitwise_and(watermark, watermark, mask=mask_inv)
    bg = cv2.bitwise_and(frame, frame, mask=mask)
    combined = cv2.add(bg, fg)

    return combined
    
# Add alternating watermarks to a video
def add_watermark(input_path, output_path):
    vid, out, fps, width, height, total_frames = get_video_data(input_path, output_path)

    for frame_idx in range(total_frames):
        success, frame = vid.read()
        if not success:
            break

        # Calculate which 5-second block we're in
        current_time = frame_idx / fps
        block = int(current_time // 5) # Change this value to change the frequency the watermark alternates

        # Alternate watermarks based on time
        if block % 2 == 0:
            frame = overlay_watermark(frame, watermark1)
        else:
            frame = overlay_watermark(frame, watermark2)

        out.write(frame)

    vid.release()
    out.release()
    print(f"Watermarked video and saved to: {output_path}")

# Append an endscreen video to the end of the main video
def add_endscreen(input_path, endscreen_path, output_path):
    vid, out, fps, width, height, main_total = get_video_data(input_path, output_path)
    end_vid = cv2.VideoCapture(endscreen_path)

    if not end_vid.isOpened():
        raise IOError(f"Cannot open endscreen video: {endscreen_path}")

    end_total = int(end_vid.get(cv2.CAP_PROP_FRAME_COUNT))

    # Write all frames from main video
    for _ in range(main_total):
        success, frame = vid.read()
        if not success:
            break
        out.write(frame)

    # Append endscreen
    for _ in range(end_total):
        success, frame = end_vid.read()
        if not success:
            break
        frame = cv2.resize(frame, (width, height))
        out.write(frame)

    vid.release()
    end_vid.release()
    out.release()
    print(f"Endscreen added and saved to: {output_path}")

# Main Execution Loop
for video_file in video_files:
    print(f"\nProcessing {video_file}...")
    name = video_file.split(".")[0]

    # Apply all processing steps sequentially
    process_video_brightness(video_file, f"processed_{name}.mp4")
    blur_faces(f"processed_{name}.mp4", f"processed1_{name}.mp4")
    add_overlay(f"processed1_{name}.mp4", overlay_video, f"processed2_{name}.mp4")
    add_watermark(f"processed2_{name}.mp4", f"processed3_{name}.mp4")
    add_endscreen(f"processed3_{name}.mp4", endscreen_video, f"final_{name}.mp4")

    print(f"Video finished processing and saved to: final_{name}.mp4")
