import cv2

face_classifier = cv2.CascadeClassifier('features/haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier('features/haarcascade_eye.xml')


def detect_face(img, eyes=False):
    """
    Detect faces (and optionally eyes) in an image.

    This function converts the input image to grayscale and uses a Haar
    Cascade Classifier to detect faces. Optionally, it can also detect
    eyes within the detected faces. Detected faces and eyes are highlighted
    with bounding boxes in the image.

    Notes:
    - If no faces are detected, the function returns the original image.
    - The face bounding boxes are drawn in magenta (255, 0, 255).
    - The eye bounding boxes are drawn in cyan (255, 255, 0) if detect_eyes
      is True.

    :param img: (numpy.ndarray): The input image in which faces (and eyes) will
                be detected.
    :param eyes: (bool): A flag indicating whether to detect eyes within
                 detected faces. Default is False.
    :return: numpy.ndarray: The image with bounding boxes drawn around detected
             faces (and eyes, if detect_eyes is True).
    """
    # convert the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # cascade classifier
    faces = face_classifier.detectMultiScale(gray, 1.2, 5)

    # when no faces detected, face_classifier returns the original frame
    if faces == ():
        return img

    # draw bounding boxes on the image
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        if eyes:
            # retrieve face
            roi_gray = gray[y:y + h, x:x + w]
            eyes = eye_classifier.detectMultiScale(roi_gray, 1.1, 3)
            for (ex, ey, ew, eh) in eyes:
                ex = ex + x
                ey = ey + y
                cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 2)

    return img
