import numpy as np


def update(x, P, Z, H, R):
    I = np.diag([1, 1, 1, 1, 1, 1])
    ### Insert update function
    Y = Z - np.dot(H, x)
    S = np.linalg.multi_dot([H, P, np.transpose(H)]) + R
    K = np.linalg.multi_dot([P, np.transpose(H), np.linalg.pinv(S)])
    X_f = x + np.dot(K, Y)
    P_f = np.dot((I - np.dot(K, H)), P)
    return X_f, P_f


def predict(x, P, F, u):
    ### insert predict function
    X_f = np.dot(F, x) + u
    #P_f = np.linalg.multi_dot([F, P, np.transpose(F)])
    P_f = np.dot(np.dot(F, P),np.transpose(F))
    return X_f, P_f


def init():
    ### Initialize Kalman filter ###
    # The initial state (6x1).
    x = np.array([[0], [0], [0], [0], [0], [0]])
    
    # The initial uncertainty (6x6).
    P = np.diag([1000, 1000, 1000, 1000, 1000, 1000])
    
    # The external motion (6x1).
    u = np.array([[0], [0], [0], [0], [0], [0]])
    
    # The transition matrix (6x6).
    F = np.array([[1, 1, 0.5, 0, 0, 0],
                  [0, 1, 1, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 1, 1, 0.5],
                  [0, 0, 0, 0, 1, 1],
                  [0, 0, 0, 0, 0, 1]])
        
    
    # The observation matrix (2x6).
    H = np.array([[1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]])
    
    # The measurement uncertainty.
    R = np.array([[1], [1]])
    
    I = np.diag([1, 1, 1, 1, 1, 1])
    return x, P, u, F, H, R, I
