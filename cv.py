# Video to frames + crop and resize to 1024x1024

import cv2

video = cv2.VideoCapture('3sec.mp4')
success, image = video.read()
count = 0

while success:
    count += 1

    # Read frame
    success, image = video.read()

    if not success:
        break

    # Rotate
    image = cv2.rotate(image, 2)
    # Crop
    image = image[400:-650, 105:-115]
    # Scale
    image = cv2.resize(image, (1024, 1024))
    # Save .jpg
    cv2.imwrite("frames/frame%03d.jpg" % count, image)
    # Display
    cv2.imshow('Frame', image)
    # Press Q on keyboard to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

print(count)
