import cv2
import numpy as np
import time

def gstreamer_pipeline(
        capture_width=1280,
        capture_height=720,
        display_width=1280,
        display_height=720,
        framerate=30,
        flip_method=0,
):
    return (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), "
            "width=(int)%d, height=(int)%d, "
            "format=(string)NV12, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink max-buffers=1 drop=true"
            % (
                capture_width,
                capture_height,
                framerate,
                flip_method,
                display_width,
                display_height,
            )
    )


# -------------------------------------------------------------------------#
# OpenCV binary
def binary_thresholding(image):
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    return binary_image


# OpenCV dilatation
def dilate_with_opencv(binary_image, kernel):
    dilated_image = cv2.dilate(binary_image, kernel, iterations=1)
    return dilated_image


# Native dilatation
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


# -------------------------------------------------------------------------#

def main():
    print(gstreamer_pipeline(flip_method=4))
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=4), cv2.CAP_GSTREAMER)
    kernel = np.array([[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]], dtype=np.uint8)
    if cap.isOpened():
        window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
        # Window
        while cv2.getWindowProperty("CSI Camera", 0) >= 0:

            ret_val, frame = cap.read()
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            binary_frame = binary_thresholding(gray_frame)

            dilated_frame_opencv = dilate_with_opencv(binary_frame, kernel)
            dilated_frame_native = dilate_without_opencv(binary_frame, kernel)

            compare_execution_time(binary_frame, kernel)

            # cv2.imshow('Original', frame)
            # cv2.imshow('Binary', binary_frame)
            # cv2.imshow('Dilated (OpenCV)', dilated_frame_opencv)
            # cv2.imshow('Dilated (Native)', dilated_frame_native)

            keyCode = cv2.waitKey(1) & 0xFF
            # Stop the program on the ESC key
            if keyCode == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Unable to open camera")


if __name__ == "__main__":
    main()