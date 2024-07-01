import cv2
from utils import detect_face


def main():
    """
    Main function to access, display the webcam feed and run Face Recognition
    with Haar Cascade features.

    This function initializes a connection to the default webcam and
    continuously captures frames in a loop, displaying them in a window.
    The loop can be exited by pressing the 'q' key. During the acquisition the
    Haar Cascade are used to detect a face in the webcam and display, and
    eventually display the relative bounding boxes on the screen.
    Once the loop is exited, the function releases the webcam and closes all
    OpenCV windows.

    Steps:
    1. Open a connection to the default webcam.
    2. Continuously capture frames from the webcam.
    3. Detect faces in the webcam and display them in a window.
    3. Display each captured frame in a window.
    4. Exit the loop and close the window when the 'q' key is pressed.

    Usage:
        python main.py

    Notes:
    - If the webcam cannot be accessed, an error message is printed and
      the function returns immediately.
    - If a frame cannot be read from the webcam, an error message is printed
      and the loop is exited.
    """

    # init opencv
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open the webcam.")
        return

    while True:
        # read frames sequentially
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # face detection
        frame = detect_face(frame)

        # display the resulting frame
        cv2.imshow('Face Recognition', frame)

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
