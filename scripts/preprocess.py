from PIL import Image
import face_recognition
import os

images = open('list.txt','r')

for line in images:
    image = face_recognition.load_image_file(line.strip('\n'))

    face_locations = face_recognition.face_locations(image)

    for face_location in face_locations:
        top, right, bottom, left = face_location
        print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

        face_image = image[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)
        pil_image.save(line.strip('\n'))
