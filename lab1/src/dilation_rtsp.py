import cv2
import numpy as np
import time


rtsp_url = 'адрес_rtsp_камеры'  # RTSP URL

def gstreamer_pipeline(
        capture_width=1280,
        capture_height=720,
        display_width=1280,
        display_height=720,
        framerate=30,
        flip_method=0,
):
    return (
        "rtspsrc location={} latency=0 ! "
        "rtph264depay ! h264parse ! omxh264dec ! "
        "videoconvert ! "
        "video/x-raw, width=(int){}, height=(int){}, format=(string)BGRx ! "
        "appsink"
        .format(rtsp_url, capture_width, capture_height, framerate)
    )

def dilate_with_opencv(binary_image, kernel):
    dilated_image = cv2.dilate(binary_image, kernel, iterations=1)
    return dilated_image

def dilate_without_opencv(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Return dilated image
    >>> dilate_without_opencv(np.array([[True, False, True]]), np.array([[0, 1, 0]]))
    array([[False, False, False]])
    >>> dilate_without_opencv(np.array([[False, False, True]]), np.array([[1, 0, 1]]))
    array([[False, False, False]])
    """
    output = np.zeros_like(image)
    image_padded = np.zeros(
        (image.shape[0] + kernel.shape[0] - 1, image.shape[1] + kernel.shape[1] - 1)
    )

    # Copy image to padded image
    image_padded[kernel.shape[0] - 2 : -1 :, kernel.shape[1] - 2 : -1 :] = image

    # Iterate over image & apply kernel
    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            summation = (
                kernel * image_padded[y : y + kernel.shape[0], x : x + kernel.shape[1]]
            ).sum()
            output[y, x] = int(summation > 0)
    return output

def compare_execution_time(binary_image, kernel):
    start_time_opencv = time.time()
    dilate_with_opencv(binary_image, kernel)
    end_time_opencv = time.time()

    start_time_native = time.time()
    dilate_without_opencv(binary_image, kernel)
    end_time_native = time.time()

    print("Runtime (OpenCV): {:.6f} sec".format(end_time_opencv - start_time_opencv))
    print("Runtime (Native): {:.6f} sec".format(end_time_native - start_time_native))

def main():
    print(gstreamer_pipeline())
    cap = cv2.VideoCapture(rtsp_url)
    kernel = np.array([[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]], dtype=np.uint8)
    if cap.isOpened():
        while True:
            ret_val, frame = cap.read()
            if not ret_val:
                print("Failed to capture frame")
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            binary_frame = cv2.threshold(gray_frame, 127, 255, cv2.THRESH_BINARY)[1]

            dilated_frame_opencv = dilate_with_opencv(binary_frame, kernel)
            dilated_frame_native = dilate_without_opencv(binary_frame, kernel)

            compare_execution_time(binary_frame, kernel)

            #cv2.imshow('Original', frame)
            #cv2.imshow('Binary', binary_frame)
            #cv2.imshow('Dilated (OpenCV)', dilated_frame_opencv)
            #cv2.imshow('Dilated (Native)', dilated_frame_native)

            keyCode = cv2.waitKey(1) & 0xFF
            if keyCode == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Unable to open camera")

if __name__ == "__main__":
    main()