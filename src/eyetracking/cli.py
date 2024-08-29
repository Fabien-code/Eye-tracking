"""This handles the CLI for the Eye Tracking application."""

import sys
import argparse
import logging
import cv2
import imutils
from .vision.getFace import EyeTracker


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
    parser.add_argument(
        "-f",
        "--face",
        help="Face Cascade file path",
    )
    parser.add_argument(
        "-e",
        "--eye",
        help="Eye Cascade file path",
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

    # construct the eye tracker
    et = EyeTracker(args.face, args.eye)
    # if a video path was not supplied, grab the reference
    camera = cv2.VideoCapture(0)
    # keep loopings
    while True:
        # grab the current frame
        (grabbed, frame) = camera.read()
        # resize the frame and convert it to grayscale
        frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # detect faces and eyes in the image
        rects = et.track(gray)
        # loop over the face bounding boxes and draw them
        for rect in rects:
            cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
        # show the tracked eyes and face
        cv2.imshow("Tracking", frame)
        # if the ‘q’ key is pressed, stop the loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    # cleanup the camera and close any open windows
    camera.release()
    cv2.destroyAllWindows()
