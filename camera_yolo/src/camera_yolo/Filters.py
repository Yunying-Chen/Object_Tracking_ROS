import numpy as np

class KalmanFilter:
    def __init__(self,dt=1,u_x=0,u_y=0,std_acc=1,std_meas=1):
        self.dt = dt
        self.u = np.array([[u_x],[u_y]])
        self.A = np.array([[1,0],
                          [0,1]]
                          )
        self.H = np.array([[1,0],
                          [0,1]])
        self.Q = np.eye(2)*std_acc
        self.R = np.eye(2)*std_meas
        self.P = np.eye(2)*1000      #covariance
        self.X = np.array([[-1,-1]]).reshape(2,1)
 #       self.X = np.zeros((4,1))
 	    


    def predict(self):
        self.X = np.dot(self.A,self.X)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q  # 预测协方差
        return self.X[:2]
    
    def update(self,Z):
        Z = np.array(Z).reshape(2,1)
        Y = Z - np.dot(self.H, self.X)
        S = np.dot(self.H, np.dot(self.P,self.H.T)) + self.R
        K = np.dot(np.dot(self.P,self.H.T),np.linalg.inv(S))
        self.X = self.X + np.dot(K,Y)
        I = np.eye(self.P.shape[0])
        self.P = np.dot((I - np.dot(K, self.H)),self.P)
        return self.X[:2]
        
   
    
if __name__=='__main__':
    kf = KalmanFilter(dt=1,std_acc=0.1,std_meas=1)
    measurements = [(100, 200), (102, 202), (103, 203), (107, 208)]  # 目标检测得到的中心点

    for z in measurements:
        pred = kf.predict()
        print(f"Predicted position: {pred.ravel()}")

        upd = kf.update(z)
        print(f"Updated position: {upd.ravel()}")
