import cv2
import numpy as np

class KalmanFilter:
    def __init__(self):
        ### Initialize Kalman filter ###
        # The initial state (6x1).
        self.x = np.array([[0], [0], [0], [0], [0], [0]])

        # The initial uncertainty (6x6).
        self.P = np.diag([1000, 1000, 1000, 1000, 1000, 1000])

        # The external motion (6x1).
        self.u = np.array([[0], [0], [0], [0], [0], [0]])

        # The transition matrix (6x6).
        self.F = np.array([[1, 1, 0.5, 0, 0, 0],
                    [0, 1, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 1, 0.5],
                    [0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 0, 1]])
            

        # The observation matrix (2x6).
        self.H = np.array([[1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]])

        # The measurement uncertainty.
        self.R = np.array([[1], [1]])

        self.I = np.diag([1, 1, 1, 1, 1, 1])


    def update(self, Z):
        ### Insert update function
        Y = Z - np.dot(self.H, self.x)
        S = np.linalg.multi_dot([self.H, self.P, np.transpose(self.H)]) + self.R
        K = np.linalg.multi_dot([self.P, np.transpose(self.H), np.linalg.pinv(S)])
        self.x = self.x + np.dot(K, Y)
        self.P = np.dot((self.I - np.dot(K, self.H)), self.P)


    def predict(self):
        ### insert predict function
        self.x = np.dot(self.F, self.x) + self.u
        self.P = np.linalg.multi_dot([self.F, self.P, np.transpose(self.F)])

    def get_state(self):
        return self.x, self.P
