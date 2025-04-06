

import numpy as np
import matplotlib.pyplot as plt
from numpy.matlib import repmat
import time
import copy


import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../Sampling_based_Planning/")
from rrt_3D.env3D import env
from rrt_3D.utils3D import getDist, sampleFree, nearest, steer, isCollide, isinside, isinbound
from rrt_3D.plot_util3D import set_axes_equal, draw_block_list, draw_Spheres, draw_obb, draw_line, make_transparent
from rrt_3D.queue import MinheapPQ
def CreateUnitSphere(r = 1):
    phi = np.linspace(0,2*np.pi, 256).reshape(256, 1) 
    theta = np.linspace(0, np.pi, 256).reshape(-1, 256) 
    radius = r
    x = radius*np.sin(theta)*np.cos(phi)
    y = radius*np.sin(theta)*np.sin(phi)
    z = radius*np.cos(theta)
    return (x, y, z)

def draw_ellipsoid(ax, C, L, xcenter):
    (xs, ys, zs) = CreateUnitSphere()
    pts = np.array([xs, ys, zs])
    pts_in_world_frame = C@L@pts + xcenter
    ax.plot_surface(pts_in_world_frame[0], pts_in_world_frame[1], pts_in_world_frame[2], alpha=0.05, color="g")

class BIT_star:
    def __init__(self, show_ellipse=False):
        self.env = env()
        self.xstart, self.xgoal = tuple(self.env.start), tuple(self.env.goal)
        self.x0, self.xt = tuple(self.env.start), tuple(self.env.goal)
        self.maxiter = 1000 
        self.eta = 7
        self.m = 400 
        self.d = 3
        self.g = {self.xstart:0, self.xgoal:np.inf}
        self.show_ellipse = show_ellipse
        self.done = False
        self.Path = []
        self.C = np.zeros([3,3])
        self.L = np.zeros([3,3])
        self.xcenter = np.zeros(3)
        self.show_ellipse = show_ellipse

        self.execution_time = 0
        self.path_length = 0




        
    def calculate_path_smoothness(self):
        path = self.Path
        angles = []
        if len(path) < 3:
            return 0

        for i in range(1, len(path) - 1):
            a = np.array(path[i - 1])
            b = np.array(path[i])
            c = np.array(path[i + 1])

            ba = a - b
            bc = c - b
            if ba.ndim > 1:
                ba = ba.flatten()
            if bc.ndim > 1:
                bc = bc.flatten()
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.arccos(np.clip(cosine_angle, -1, 1))

            angles.append(np.degrees(angle))

        return np.mean(angles) if angles else 0


    def calculate_path_safety(self):
        path = self.Path
        if not hasattr(self.env, 'blocks') and not hasattr(self.env, 'balls'):
            print("No obstacles defined in the environment.")
            return float('inf')

        min_distances = []
        for block in self.env.blocks:
            center = np.array([(block[0] + block[3]) / 2, (block[1] + block[4]) / 2, (block[2] + block[5]) / 2])
            for point in path:
                dist = np.linalg.norm(np.array(point) - center)
                min_distances.append(dist)
        for sphere in self.env.balls:
            center = np.array(sphere[:3])
            radius = sphere[3] 
            for point in path:
                dist = np.linalg.norm(np.array(point) - center) - radius
                min_distances.append(dist)

        return min(min_distances) if min_distances else float('inf')


    def calculate_path_directness(self):
        path = self.Path
        if not path:
            return float('inf')
        start = np.array(self.xstart)
        goal = np.array(self.xgoal)
        straight_line_distance = np.linalg.norm(goal - start)
        path_length = sum(np.linalg.norm(np.array(path[i]) - np.array(path[i-1])) for i in range(1, len(path)))
        return path_length / straight_line_distance if straight_line_distance > 0 else float('inf')




    def run(self):
        start_time = time.time() 
        
        self.V = {self.xstart}
        self.E = set() 
        self.Parent = {} 
        self.Xsamples = {self.xgoal} 
        self.QE = set() 
        self.QV = set() 
        self.r = np.inf 
        self.ind = 0
        num_resample = 0
        while True:
            print('round '+str(self.ind))
            self.visualization()
            if len(self.QE) == 0 and len(self.QV) == 0:
                self.Prune(self.g_T(self.xgoal))
                self.Xsamples = self.Sample(self.m, self.g_T(self.xgoal)) 
                self.Xsamples.add(self.xgoal) 
                self.Vold = {v for v in self.V}
                self.QV = {v for v in self.V} 
                if self.done:
                    self.r = 2 
                    num_resample += 1
                else:
                    self.r = self.radius(len(self.V) + len(self.Xsamples)) 
            while self.BestQueueValue(self.QV, mode = 'QV') <= self.BestQueueValue(self.QE, mode = 'QE'):
                self.ExpandVertex(self.BestInQueue(self.QV, mode = 'QV'))
            (vm, xm) = self.BestInQueue(self.QE, mode = 'QE')
            self.QE.remove((vm, xm))
            if self.g_T(vm) + self.c_hat(vm, xm) + self.h_hat(xm) < self.g_T(self.xgoal):
                cost = self.c(vm, xm)
                if self.g_hat(vm) + cost + self.h_hat(xm) < self.g_T(self.xgoal):
                    if self.g_T(vm) + cost < self.g_T(xm):
                        if xm in self.V:
                            self.E.difference_update({(v, x) for (v, x) in self.E if x == xm})
                        else:
                            self.Xsamples.remove(xm)
                            self.V.add(xm)
                            self.QV.add(xm)
                        self.g[xm] = self.g[vm] + cost
                        self.E.add((vm, xm))
                        self.Parent[xm] = vm 
                        self.QE.difference_update({(v, x) for (v, x) in self.QE if x == xm and (self.g_T(v) + self.c_hat(v, xm)) >= self.g_T(xm)})
            
            else:
                self.QE = set()
                self.QV = set()
            self.ind += 1
            if self.xgoal in self.Parent:
                print('locating path...')
                self.done = True
                self.Path = self.path()
            if self.ind > self.maxiter:
                break



        self.path_smoothness = self.calculate_path_smoothness()
        self.path_safety = self.calculate_path_safety()
        self.path_directness = self.calculate_path_directness()
        print(f"Path Smoothness: {self.path_smoothness}")
        print(f"Path Safety: {self.path_safety}")
        print(f"Path Directness: {self.path_directness}")



        self.execution_time = time.time() - start_time
        self.path_length = self.calculate_path_length()
        print(f"Execution Time: {self.execution_time} seconds")
        print(f"Path Length: {self.path_length} units")
            
        print('complete')
        print('number of times resampling ' + str(num_resample))


    def calculate_path_length(self):
        length = 0
        if hasattr(self, 'Path') and len(self.Path) > 1:
            for segment in self.Path:
                length += getDist(segment[0], segment[1])
        return length


    def getDist(pos1, pos2):
        x1, y1, z1 = pos1
        x2, y2, z2 = pos2
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
            


    def Sample(self, m, cmax, bias = 0.05, xrand = set()):
        print('new sample')
        if cmax < np.inf:
            cmin = getDist(self.xgoal, self.xstart)
            xcenter = np.array([(self.xgoal[0] + self.xstart[0]) / 2, (self.xgoal[1] + self.xstart[1]) / 2, (self.xgoal[2] + self.xstart[2]) / 2])
            C = self.RotationToWorldFrame(self.xstart, self.xgoal)
            r = np.zeros(3)
            r[0] = cmax /2
            for i in range(1,3):
                r[i] = np.sqrt(cmax**2 - cmin**2) / 2
            L = np.diag(r) 
            xball = self.SampleUnitBall(m) 
            x =  (C@L@xball).T + repmat(xcenter, len(xball.T), 1)
            self.C = C 
            self.xcenter = xcenter
            self.L = L
            x2 = set(map(tuple, x[np.array([not isinside(self, state) and isinbound(self.env.boundary, state) for state in x])])) 
            xrand.update(x2)
            if len(x2) < m:
                return self.Sample(m - len(x2), cmax, bias=bias, xrand=xrand)
        else:
            for i in range(m):
                xrand.add(tuple(sampleFree(self, bias = bias)))
        return xrand

    def SampleUnitBall(self, n):
        r = np.random.uniform(0.0, 1.0, size = n)
        theta = np.random.uniform(0, np.pi, size = n)
        phi = np.random.uniform(0, 2 * np.pi, size = n)
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return np.array([x,y,z])

    def RotationToWorldFrame(self, xstart, xgoal):
        d = getDist(xstart, xgoal)
        xstart, xgoal = np.array(xstart), np.array(xgoal)
        a1 = (xgoal - xstart) / d
        M = np.outer(a1,[1,0,0])
        U, S, V = np.linalg.svd(M)
        C = U@np.diag([1, 1, np.linalg.det(U)*np.linalg.det(V)])@V.T
        return C


    def ExpandVertex(self, v):
        self.QV.remove(v)
        Xnear = {x for x in self.Xsamples if getDist(x, v) <= self.r}
        self.QE.update({(v, x) for x in Xnear if self.g_hat(v) + self.c_hat(v, x) + self.h_hat(x) < self.g_T(self.xgoal)})
        if v not in self.Vold:
            Vnear = {w for w in self.V if getDist(w, v) <= self.r}
            self.QE.update({(v,w) for w in Vnear if \
                ((v,w) not in self.E) and \
                (self.g_hat(v) + self.c_hat(v, w) + self.h_hat(w) < self.g_T(self.xgoal)) and \
                (self.g_T(v) + self.c_hat(v, w) < self.g_T(w))})

    def Prune(self, c):
        self.Xsamples = {x for x in self.Xsamples if self.f_hat(x) >= c}
        self.V.difference_update({v for v in self.V if self.f_hat(v) >= c})
        self.E.difference_update({(v, w) for (v, w) in self.E if (self.f_hat(v) > c) or (self.f_hat(w) > c)})
        self.Xsamples.update({v for v in self.V if self.g_T(v) == np.inf})
        self.V.difference_update({v for v in self.V if self.g_T(v) == np.inf})

    def radius(self, q):
        return 2 * self.eta * (1 + 1/self.d) ** (1/self.d) * \
            (self.Lambda(self.Xf_hat(self.V)) / self.Zeta() ) ** (1/self.d) * \
            (np.log(q) / q) ** (1/self.d)

    def Lambda(self, inputset):
        return len(inputset)

    def Xf_hat(self, X):
        cbest = self.g_T(self.xgoal)
        return {x for x in X if self.f_hat(x) <= cbest}

    def Zeta(self):
        return 4/3 * np.pi

    def BestInQueue(self, inputset, mode):
        if mode == 'QV':
            V = {state: self.g_T(state) + self.h_hat(state) for state in self.QV}
        if mode == 'QE':
            V = {state: self.g_T(state[0]) + self.c_hat(state[0], state[1]) + self.h_hat(state[1]) for state in self.QE}
        if len(V) == 0:
            print(mode + 'empty')
            return None
        return min(V, key = V.get)

    def BestQueueValue(self, inputset, mode):
        if mode == 'QV':
            V = {self.g_T(state) + self.h_hat(state) for state in self.QV}
        if mode == 'QE':
            V = {self.g_T(state[0]) + self.c_hat(state[0], state[1]) + self.h_hat(state[1]) for state in self.QE}
        if len(V) == 0:
            return np.inf
        return min(V)

    def g_hat(self, v):
        return getDist(self.xstart, v)

    def h_hat(self, v):
        return getDist(self.xgoal, v)

    def f_hat(self, v):
        return self.g_hat(v) + self.h_hat(v)

    def c(self, v, w):
        collide, dist = isCollide(self, v, w)
        if collide:
            return np.inf
        else: 
            return dist

    def c_hat(self, v, w):
        return getDist(v, w)

    def g_T(self, v):
        if v not in self.g:
            self.g[v] = np.inf
        return self.g[v]

    def path(self):
        path = []
        s = self.xgoal
        i = 0
        while s != self.xstart:
            path.append((s, self.Parent[s]))
            s = self.Parent[s]
            if i > self.m:
                break
            i += 1
        return path
         
    def visualization(self):
        if self.ind % 20 == 0:
            V = np.array(list(self.V))
            Xsample = np.array(list(self.Xsamples))
            edges = list(map(list, self.E))
            Path = np.array(self.Path)
            start = self.env.start
            goal = self.env.goal
            ax = plt.subplot(111, projection='3d')
            ax.view_init(elev=90., azim=60.)
            ax.clear()

            draw_Spheres(ax, self.env.balls)
            draw_block_list(ax, self.env.blocks)
            if self.env.OBB is not None:
                draw_obb(ax, self.env.OBB)
            draw_block_list(ax, np.array([self.env.boundary]), alpha=0)
            draw_line(ax, edges, visibility=0.75, color='g')
            draw_line(ax, Path, color='r')
            if self.show_ellipse:
                draw_ellipsoid(ax, self.C, self.L, self.xcenter)
            if len(V) > 0:
                ax.scatter3D(V[:, 0], V[:, 1], V[:, 2], s=2, color='g', )
            if len(Xsample) > 0: 
                ax.scatter3D(Xsample[:, 0], Xsample[:, 1], Xsample[:, 2], s=1, color='b',)
            ax.plot(start[0:1], start[1:2], start[2:], 'go', markersize=7, markeredgecolor='k')
            ax.plot(goal[0:1], goal[1:2], goal[2:], 'ro', markersize=7, markeredgecolor='k')
            ax.dist = 11
            set_axes_equal(ax)
            make_transparent(ax)
            ax.set_axis_off()
            plt.pause(0.0001)


if __name__ == '__main__':
    Newprocess = BIT_star(show_ellipse=False)
    Newprocess.run()
