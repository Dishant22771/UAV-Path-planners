
import numpy as np
import time
import matplotlib.pyplot as plt

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../Sampling_based_Planning/")
from rrt_3D.env3D import env
from rrt_3D.utils3D import getDist, sampleFree, nearest, steer, isCollide
from rrt_3D.plot_util3D import set_axes_equal, draw_block_list, draw_Spheres, draw_obb, draw_line, make_transparent


class dynamic_rrt_3D:

    def __init__(self):
        self.env = env()
        self.x0, self.xt = tuple(self.env.start), tuple(self.env.goal)
        self.qrobot = self.x0
        self.current = tuple(self.env.start)
        self.stepsize = 0.25
        self.maxiter = 10000
        self.GoalProb = 0.05  
        self.WayPointProb = 0.02
        self.done = False
        self.invalid = False

        self.V = [] 
        self.Parent = {}  
        self.Edge = set()  
        self.Path = []
        self.flag = {}  
        self.ind = 0
        self.i = 0
        self.execution_time = 0
        self.path_length = 0


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




    
    def RegrowRRT(self):
        self.TrimRRT()
        self.GrowRRT()

    def TrimRRT(self):
        S = []
        i = 1
        print('trimming...')
        while i < len(self.V):
            qi = self.V[i]
            qp = self.Parent[qi]
            if self.flag[qp] == 'Invalid':
                self.flag[qi] = 'Invalid'
            if self.flag[qi] != 'Invalid':
                S.append(qi)
            i += 1
        self.CreateTreeFromNodes(S)

    def InvalidateNodes(self, obstacle):
        Edges = self.FindAffectedEdges(obstacle)
        for edge in Edges:
            qe = self.ChildEndpointNode(edge)
            self.flag[qe] = 'Invalid'
    def initRRT(self):
        self.V.append(self.x0)
        self.flag[self.x0] = 'Valid'

    def GrowRRT(self):
        print('growing...')
        qnew = self.x0
        distance_threshold = self.stepsize
        self.ind = 0
        while self.ind <= self.maxiter:
            qtarget = self.ChooseTarget()
            qnearest = self.Nearest(qtarget)
            qnew, collide = self.Extend(qnearest, qtarget)
            if not collide:
                self.AddNode(qnearest, qnew)
                if getDist(qnew, self.xt) < distance_threshold:
                    self.AddNode(qnearest, self.xt)
                    self.flag[self.xt] = 'Valid'
                    break
                self.i += 1
            self.ind += 1

    def ChooseTarget(self):
        p = np.random.uniform()
        if len(self.V) == 1:
            i = 0
        else:
            i = np.random.randint(0, high=len(self.V) - 1)
        if 0 < p < self.GoalProb:
            return self.xt
        elif self.GoalProb < p < self.GoalProb + self.WayPointProb:
            return self.V[i]
        elif self.GoalProb + self.WayPointProb < p < 1:
            return tuple(self.RandomState())

    def RandomState(self):
        xrand = sampleFree(self, bias=0)
        return xrand

    def AddNode(self, nearest, extended):
        self.V.append(extended)
        self.Parent[extended] = nearest
        self.Edge.add((extended, nearest))
        self.flag[extended] = 'Valid'

    def Nearest(self, target):
        return nearest(self, target, isset=True)

    def Extend(self, nearest, target):
        extended, dist = steer(self, nearest, target, DIST=True)
        collide, _ = isCollide(self, nearest, target, dist)
        return extended, collide

    def Main(self):
        start_time = time.time()
        self.x0 = tuple(self.env.goal)
        self.xt = tuple(self.env.start)
        self.initRRT()
        self.GrowRRT()
        self.Path, D = self.path()
        self.done = True
        self.visualization()
        t = 0
        while True:
            new, _ = self.env.move_block(a=[0.2, 0, -0.2], mode='translation')
            self.InvalidateNodes(new)
            self.TrimRRT()
            self.visualization()
            self.invalid = self.PathisInvalid(self.Path)
            if self.invalid:
                self.done = False
                self.RegrowRRT()
                self.Path = []
                self.Path, D = self.path()
                self.done = True
                self.visualization()
            if t == 8:
                break
            t += 1
        self.visualization()
        plt.show()
        self.execution_time = time.time() - start_time  
        self.path_length = self.calculate_path_length() 
        print(f"Execution Time: {self.execution_time} seconds")
        print(f"Path Length: {self.path_length} units")


            
        
        self.path_smoothness = self.calculate_path_smoothness()
        self.path_safety = self.calculate_path_safety()
        self.path_directness = self.calculate_path_directness()


        print(f"Path Smoothness: {self.path_smoothness}")
        print(f"Path Safety: {self.path_safety}")
        print(f"Path Directness: {self.path_directness}")




    def calculate_path_length(self):
        length = 0
        if self.Path:
            for segment in self.Path:
                start_point = segment[0] 
                end_point = segment[1]    
                length += self.getDist(start_point, end_point)
        return length

    def getDist(self, pos1, pos2):
        return np.linalg.norm(pos1 - pos2)  


    
    def FindAffectedEdges(self, obstacle):
        print('finding affected edges...')
        Affectededges = []
        for e in self.Edge:
            child, parent = e
            collide, _ = isCollide(self, child, parent)
            if collide:
                Affectededges.append(e)
        return Affectededges

    def ChildEndpointNode(self, edge):
        return edge[0]

    def CreateTreeFromNodes(self, Nodes):
        print('creating tree...')
        self.V = [node for node in Nodes]
        self.Edge = {(node, self.Parent[node]) for node in Nodes}

    def PathisInvalid(self, path):
        for edge in path:
            if self.flag[tuple(edge[0])] == 'Invalid' or self.flag[tuple(edge[1])] == 'Invalid':
                return True

    def path(self, dist=0):
        Path=[]
        x = self.xt
        i = 0
        while x != self.x0:
            x2 = self.Parent[x]
            Path.append(np.array([x, x2]))
            dist += getDist(x, x2)
            x = x2
            if i > 10000:
                print('Path is not found')
                return 
            i+= 1
        return Path, dist


    def visualization(self):
        if self.ind % 100 == 0 or self.done:
            V = np.array(self.V)
            Path = np.array(self.Path)
            start = self.env.start
            goal = self.env.goal
            edges = np.array([list(i) for i in self.Edge])
            ax = plt.subplot(111, projection='3d')
            ax.view_init(elev=90., azim=0.)
            ax.clear()
            draw_Spheres(ax, self.env.balls)
            if self.env.OBB is not None:
                draw_obb(ax, self.env.OBB)
            draw_block_list(ax, np.array([self.env.boundary]), alpha=0)
            draw_line(ax, edges, visibility=0.75, color='g')
            draw_line(ax, Path, color='r')
            ax.plot(start[0:1], start[1:2], start[2:], 'go', markersize=7, markeredgecolor='k')
            ax.plot(goal[0:1], goal[1:2], goal[2:], 'ro', markersize=7, markeredgecolor='k')
            set_axes_equal(ax)
            make_transparent(ax)
            ax.set_axis_off()
            plt.pause(0.0001)


if __name__ == '__main__':
    rrt = dynamic_rrt_3D()
    rrt.Main()
