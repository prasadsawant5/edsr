import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.config.experimental.set_memory_growth(gpus[1], True)

from tensorflow.keras.models import load_model
import cv2
import numpy as np

SAVED_MODEL = './saved_models/1627308536071/model/'
VIDEO = 'test.mp4'
LOW_RES_SIZE = (240, 180)

if __name__ == '__main__':
    model = load_model(SAVED_MODEL)

    cap = cv2.VideoCapture(VIDEO)

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            image = cv2.resize(frame, LOW_RES_SIZE, interpolation=cv2.INTER_CUBIC)
            img = image / 255.
            img = tf.expand_dims(img, axis=0)

            prediction = model.predict(img, batch_size=1)
            prediction = np.array(prediction[0] * 255, dtype=np.uint8)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            cv2.imshow('Original', frame)
            cv2.moveWindow('Original', 10, 0)
            cv2.imshow('Input', image)
            cv2.moveWindow("Input", 800, 0)
            cv2.imshow('Inference', prediction)
            cv2.moveWindow('Inference', 10, 600)

        else:
            break

    cap.release()
    cv2.destroyAllWindows()


