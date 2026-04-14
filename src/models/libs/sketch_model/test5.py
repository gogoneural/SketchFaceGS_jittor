import cv2
import numpy as np
import random
import os
from scipy.ndimage import gaussian_filter1d
from PIL import Image

# calculate_average_distance function remains the same as before
def calculate_average_distance(contour1, contour2):
    if len(contour1) == 0 or len(contour2) < 3: return float('inf')
    total_distance = 0
    count = 0
    points1 = contour1.squeeze()
    if points1.ndim == 1: points1 = np.array([points1])
    if points1.shape[0] == 0: return float('inf') # Check after squeeze

    for pt in points1:
        distance = cv2.pointPolygonTest(contour2, tuple(pt.astype(float)), True)
        total_distance += abs(distance)
        count += 1
    return total_distance / count if count > 0 else float('inf')


def sketchify_final(image_path, output_path,
                     num_draws=3,           # How many times to draw each line
                     max_global_offset=1,   # Max global shift per redraw pass
                     base_thickness=1,      # Base line thickness
                     thickness_variation=0, # Max random variation in thickness
                     wiggle_magnitude=1.5,  # How much the line can deviate (pixels)
                     wiggle_smoothness=3.0, # How smooth the wiggle is
                     min_contour_length=10, # Filter: Min pixels in contour perimeter
                     similarity_threshold=2.0, # De-dup: Max avg distance
                     length_ratio_threshold=1.5, # De-dup: Max length ratio
                     # **** Morphological Closing Parameters ****
                     closing_kernel_size=3,  # Size of kernel for closing gaps (e.g., 3, 5). 0 to disable.
                     # **** End Closing Params ****
                     add_blur=False,        # Add a final subtle blur?
                     blur_kernel_size=3):   # Kernel size for the final blur
    """
    Transforms line drawing to sketch: wiggly lines, length filter, de-duplicate,
    and optionally fills narrow gaps using morphological closing.
    """
    # --- 1. Load Image ---
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # if img is None: print(f"Error: Could not load image at {image_path}"); return

    # --- 2. Threshold ---
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # --- 3. Find Contours ---
    all_contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # print(f"Found {len(all_contours)} initial contours.")

    # --- 4. Filter by Length ---
    length_filtered_contours = []
    if min_contour_length > 0:
        for contour in all_contours:
            if len(contour) >= 3:
                length = cv2.arcLength(contour, closed=True)
                if length >= min_contour_length: length_filtered_contours.append(contour)
        # print(f"Kept {len(length_filtered_contours)} contours after length filtering (min={min_contour_length}px).")
    else:
        length_filtered_contours = [c for c in all_contours if len(c) >= 3]
        # print("Skipping length filtering.")

    # if not length_filtered_contours: print("Warning: No contours after length filtering."); return

    # --- 5. De-duplicate Contours (Optional but recommended) ---
    if similarity_threshold > 0 and len(length_filtered_contours) > 1:
        # print(f"Starting de-duplication (similarity threshold={similarity_threshold}px)...")
        sorted_contours = sorted(length_filtered_contours, key=lambda c: cv2.arcLength(c, True), reverse=True)
        contour_lengths = [cv2.arcLength(c, True) for c in sorted_contours]
        kept_indices = set(range(len(sorted_contours)))
        for i in range(len(sorted_contours)):
            if i not in kept_indices: continue
            c_i, len_i = sorted_contours[i], contour_lengths[i]
            for j in range(i + 1, len(sorted_contours)):
                if j not in kept_indices: continue
                c_j, len_j = sorted_contours[j], contour_lengths[j]
                if len_i > len_j * length_ratio_threshold: break
                avg_dist = calculate_average_distance(c_j, c_i)
                if avg_dist < similarity_threshold: kept_indices.remove(j)
        deduplicated_contours = [sorted_contours[k] for k in sorted(list(kept_indices))]
        # print(f"Kept {len(deduplicated_contours)} contours after de-duplication.")
    else:
        deduplicated_contours = length_filtered_contours
        # print("Skipping de-duplication or not enough contours to de-duplicate.")

    # if not deduplicated_contours: print("Warning: No contours after de-duplication."); return

    # --- 6. Create Canvas ---
    drawn_sketch_canvas = np.ones_like(img) * 255 # White background

    # --- 7. Generate Wiggly Versions & Draw ---
    # (Simplified: Generate wiggly versions directly inside the draw loop for variation per pass)
    # print(f"Drawing {len(deduplicated_contours)} base contours {num_draws} times with wiggle...")
    for i in range(num_draws):
        offset_x = random.randint(-max_global_offset, max_global_offset)
        offset_y = random.randint(-max_global_offset, max_global_offset)
        current_thickness = max(1, base_thickness + random.randint(-thickness_variation, thickness_variation))
        # print(f" Pass {i+1}: Global Offset=({offset_x},{offset_y}), Thickness={current_thickness}")

        # Generate wiggly contours *for this specific pass*
        temp_pass_canvas = np.ones_like(img) * 255
        for contour in deduplicated_contours:
             points = contour.squeeze()
             if points.ndim == 1 or len(points) < 3: continue

             # Generate wiggle specific to this pass
             random_offsets = np.random.randn(points.shape[0], 2) * wiggle_magnitude
             smoothed_offsets_x = gaussian_filter1d(random_offsets[:, 0], sigma=wiggle_smoothness, mode='wrap')
             smoothed_offsets_y = gaussian_filter1d(random_offsets[:, 1], sigma=wiggle_smoothness, mode='wrap')
             smoothed_offsets = np.stack((smoothed_offsets_x, smoothed_offsets_y), axis=-1)
             wiggly_points = points + smoothed_offsets
             wiggly_contour_formatted = wiggly_points.astype(np.int32)

             # Apply global offset for this pass
             offset_wc = wiggly_contour_formatted + np.array([offset_x, offset_y])

             if offset_wc.shape[0] > 1:
                  cv2.polylines(temp_pass_canvas, [offset_wc.astype(np.int32)], isClosed=True, color=(0, 0, 0),
                                thickness=current_thickness, lineType=cv2.LINE_AA)

        # Combine pass onto main canvas
        drawn_sketch_canvas = np.minimum(drawn_sketch_canvas, temp_pass_canvas)

    # --- 8. Morphological Closing (Fill Gaps) ---
    final_canvas = drawn_sketch_canvas # Start with the drawn result
    if closing_kernel_size > 0:
        # print(f"Applying Morphological Closing with kernel size {closing_kernel_size}x{closing_kernel_size}...")
        # Invert: Lines become white (255), background black (0)
        inverted_canvas = cv2.bitwise_not(drawn_sketch_canvas)

        # Define kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (closing_kernel_size, closing_kernel_size))
        # Could also try cv2.MORPH_ELLIPSE

        # Apply closing
        closed_inverted = cv2.morphologyEx(inverted_canvas, cv2.MORPH_CLOSE, kernel)

        # Invert back: Lines black (0), background white (255)
        final_canvas = cv2.bitwise_not(closed_inverted)
        # print("Gap filling complete.")
    else:
        # print("Skipping morphological closing.")
        pass


    # --- 9. Optional Blur ---
    if add_blur and blur_kernel_size > 0:
        # print(f"Applying Gaussian Blur...")
        if blur_kernel_size % 2 == 0: blur_kernel_size += 1
        final_canvas = cv2.GaussianBlur(final_canvas, (blur_kernel_size, blur_kernel_size), 0)

    # --- 10. Save ---
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir)
    # success = cv2.imwrite(output_path, final_canvas)
    print(Image.fromarray(final_canvas).convert("RGB").save(output_path)) # Save the final canvas as RGB
    # if success: print(f"Final sketch saved to {output_path}")
    # else: print(f"Error saving image to {output_path}")


# --- How to Use ---
if __name__ == "__main__":
    # input_image_file = 'input.png' # Replace
    # output_image_file = 'output_car_sketch.png' # Replace
    input_image_file = './test_sketch/kang_sketch.jpg'
    output_image_file = './test_sketch/kang_sketch_gai.jpg'
    if not os.path.exists(input_image_file):
         print(f"Error: Input file not found at '{input_image_file}'")
    else:
        sketchify_final(
            image_path=input_image_file,
            output_path=output_image_file,
            num_draws=1, #感觉设置为1比较好，每条线重复几次？
            max_global_offset=0,
            base_thickness=1, #粗细基本
            thickness_variation=0, #粗细变化范围
            wiggle_magnitude=1, #线条抖动幅度（模拟人的手抖）
            wiggle_smoothness=1, #线条抖动平滑度（越大越平滑）
            min_contour_length=150, #过滤掉小于这个长度的线条
            similarity_threshold=2.0, # Keep some de-dup to avoid excessive overlap
            length_ratio_threshold=1.5, # 这两个参数是用来去除重合的线条的，但感觉用处不大
            # ---- Tune Closing ----
            closing_kernel_size=5,  # 填合粗细小于这个值的狭长的间隙
            # ----------------------
            add_blur=False,
            blur_kernel_size=3 #高斯模糊
        )