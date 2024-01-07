import cv2
import numpy as np


class VideoWriter:
    """ Class that modifies a video stream using gstreamer
    and records it using opencv with cuda support.
    The built-in codec for Jetson nano - NVENC is used for recording    
    """

    def __init__(self, filename: str) -> None:
        """
        __init__ 
            Constructor class CustomVideoWriter
        Parameters
        ----------
        filename : str
            Name of the currently being written file
        """

        self.process: None or cv2.VideoWriter = None
        self.gst_out: str = f"appsrc ! video/x-raw, format=BGR ! queue ! videoconvert ! video/x-raw,format=RGBA ! \
        nvvidconv ! nvv4l2h264enc bitrate=300000 ! h264parse ! qtmux ! filesink location={filename}"

    def write(self, frame: np.ndarray) -> None:
        """
        write 

        The function of initializing the pipe gstreamer and writing the frame to a file

        Parameters
        ----------
        frame : np.ndarray
            Frame for record
        """
        if self.process is None:
            h, w = frame.shape[:2]
            self.process = cv2.VideoWriter(
                self.gst_out, cv2.CAP_GSTREAMER, 0, float(2), (w, h))
        self.process.write(
            frame
        )

    def release(self) -> None:
        """
        release 

        The function of freeing memory from the video stream for recording
        """
        if self.process is None:
            return
        self.process.release()
