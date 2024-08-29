"""This handles the CLI for the Eye Tracking application."""

import sys
import argparse
import logging
import cv2
import mediapipe as mp
from .vision.landmarker import landmarker_and_result



def main() -> None:
    """
    A simple CLI tool that converts docx document to Alstom SwRS format (Sphinx).

    Args:
        input (str)         : The path of the docx document.
        output (str)        : The destination path

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="A simple CLI tool")
    parser.add_argument(
        "-v",
        "--verbose",
        help="Be vebose",
        action="count",
        dest="loglevel",
        default=0,
    )
    args = parser.parse_args()
    # Logging level start at Warning (level = 30)
    # Each -v sub 10 to logging level (minimum is 0)
    loglevel = max(logging.WARNING - (args.loglevel * 10), 0)
    # Force UTF-8 encoding cause in certain Windows OS,
    # it may be other encoder by default (case encounter : 'cp1252')
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore
    sys.stdin.reconfigure(encoding="utf-8")  # type: ignore
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore
    logging.basicConfig(level=loglevel, format="%(message)s")

    # access webcam
    cap = cv2.VideoCapture(0)
   
   
   # create landmarker
    hand_landmarker = landmarker_and_result()
   
    while True:
        # pull frame
        ret, frame = cap.read()
        # mirror frame
        frame = cv2.flip(frame, 1)
        # update landmarker results
        hand_landmarker.detect_async(frame)
        print(hand_landmarker.result)
        # draw landmarks on frame
        # frame = draw_landmarks_on_image(frame,hand_landmarker.result)
        if cv2.waitKey(1) == ord('q'):
            break

    # release everything
    cap.release()
    hand_landmarker.close()
    cv2.destroyAllWindows()

