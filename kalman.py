import cv2
import numpy as np
class Kalman2D():

    def __init__(self, x, y, v_x, v_y):

        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],
            np.float32)
        self.kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],
                [0,0,0,1]],np.float32)
        self.kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],
                [0,0,0,1]],np.float32) * 10
        #self.kalman.measurementNoiseCov = np.array([[1,0],[0,1]],np.float32) * 10
        self.kalman.statePost = np.array([[x],[y],[v_x],[v_y]],np.float32)



        self.prediction = []
        self.predict()

    def predict(self):
        self.prediction = self.kalman.predict()


    def predict_and_correct(self,x,y):
        self.prediction = self.kalman.predict()
        mp = np.array([[np.float32(x)],[np.float32(y)]])

        self.kalman.correct(mp)


    def get_prediction(self):

        return (self.prediction[0], self.prediction[1])
