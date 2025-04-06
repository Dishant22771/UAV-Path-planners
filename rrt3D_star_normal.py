import numpy as np
from numpy.matlib import repmat
from collections import defaultdict
import time
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../Sampling_based_Planning/")
from rrt_3D.env3D import env
from rrt_3D.utils3D import getDist, sampleFree, nearest, steer, isCollide, near, visualization, cost, path


class rrtstar():
    def __init__(self):
        self.env = env()
        self.Parent = {}
        self.V = []
        self.COST = {}
        self.i = 0
        self.maxiter = 4000 
        self.stepsize = 2
        self.gamma = 7
        self.eta = self.stepsize
        self.Path = []
        self.done = False
        self.x0 = tuple(self.env.start)
        self.xt = tuple(self.env.goal)
        self.V.append(self.x0)
        self.ind = 0
        
    def wireup(self,x,y):
        self.Parent[x] = y

    def removewire(self,xnear):
        xparent = self.Parent[xnear]
        a = [xnear,xparent]

    def reached(self):
        self.done = True
        goal = self.xt
        xn = near(self,self.env.goal)
        c = [cost(self,tuple(x)) for x in xn]
        xncmin = xn[np.argmin(c)]
        self.wireup(goal , tuple(xncmin))
        self.V.append(goal)
        self.Path,self.D = path(self)

    def calculate_path_length(self):
        length = 0
        if len(self.Path) > 1:
            for i in range(len(self.Path) - 1):
                start_point = np.array(self.Path[i][0])
                end_point = np.array(self.Path[i + 1][0])
                length += np.linalg.norm(end_point - start_point)
        return length




    def calculate_path_smoothness(self):
        angles = []
        if len(self.Path) < 3:
            return 0 
        for i in range(1, len(self.Path) - 1):
            a = np.array(self.Path[i-1][0])
            b = np.array(self.Path[i][0])
            c = np.array(self.Path[i+1][0])

            ba = a - b
            bc = c - b

            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.arccos(np.clip(cosine_angle, -1, 1))
            angles.append(np.degrees(angle))
        return np.mean(angles) if angles else 0

    def calculate_path_safety(self):
        min_distances = []
        for segment in self.Path:
            point = np.array(segment[0])
            for sphere in self.env.balls:
                center = np.array(sphere[:3])
                radius = sphere[3]
                distance = np.linalg.norm(point - center) - radius
                min_distances.append(distance)
            for block in self.env.blocks:
                block_center = np.array([(block[0] + block[3])/2, (block[1] + block[4])/2, (block[2] + block[5])/2])
                distance = np.linalg.norm(point - block_center)
                min_distances.append(distance)
        return min(min_distances) if min_distances else float('inf')

    def calculate_path_directness(self):
        if not self.Path:
            return float('inf')
        start = np.array(self.env.start)
        goal = np.array(self.env.goal)
        straight_line_distance = np.linalg.norm(goal - start)
        path_length = self.calculate_path_length()
        return path_length / straight_line_distance if straight_line_distance > 0 else float('inf')   

    def run(self):
        xnew = self.x0
        print('start rrt*... ')
        self.fig = plt.figure(figsize = (10,8))
        while self.ind < self.maxiter:
            xrand    = sampleFree(self)
            xnearest = nearest(self,xrand)
            xnew, dist  = steer(self,xnearest,xrand)
            collide, _ = isCollide(self,xnearest,xnew,dist=dist)
            if not collide:
                Xnear = near(self,xnew)
                self.V.append(xnew) 
                visualization(self)
                plt.title('rrt*')
                xmin, cmin = xnearest, cost(self, xnearest) + getDist(xnearest, xnew)
                Collide = []
                for xnear in Xnear:
                    xnear = tuple(xnear)
                    c1 = cost(self, xnear) + getDist(xnew, xnear)
                    collide, _ = isCollide(self, xnew, xnear)
                    Collide.append(collide)
                    if not collide and c1 < cmin:
                        xmin, cmin = xnear, c1
                self.wireup(xnew, xmin)
                for i in range(len(Xnear)):
                    collide = Collide[i]
                    xnear = tuple(Xnear[i])
                    c2 = cost(self, xnew) + getDist(xnew, xnear)
                    if not collide and c2 < cost(self, xnear):
                        self.wireup(xnear, xnew)
                self.i += 1
            self.ind += 1
        self.reached()
        self.path_smoothness = self.calculate_path_smoothness()
        self.path_safety = self.calculate_path_safety()
        self.path_directness = self.calculate_path_directness()
        print('time used = ' + str(time.time()-starttime))
        print('Total distance = '+str(self.D))
        print(f"Path Smoothness: {self.path_smoothness}")
        print(f"Path Safety: {self.path_safety}")
        print(f"Path Directness: {self.path_directness}")
        visualization(self)
        plt.show()
        

if __name__ == '__main__':
    p = rrtstar()
    starttime = time.time()
    p.run()
