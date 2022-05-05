import numpy as np

#Dummy class lige nu
class InferenceModel:
    def __init__(self):
        self.i = 0

    def perform_inference(self, img1, img2):
        self.i += 1
        return np.array([[self.i],[self.i]])