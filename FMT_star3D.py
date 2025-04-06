
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../Sampling_based_Planning/")
from rrt_3D.env3D import env
from rrt_3D.utils3D import getDist, sampleFree, nearest, steer, isCollide
from rrt_3D.plot_util3D import set_axes_equal, draw_block_list, draw_Spheres, draw_obb, draw_line, make_transparent
from rrt_3D.queue import MinheapPQ

class FMT_star:

    def __init__(self, radius = 1, n = 1000):
        self.env = env()
        self.xinit, self.xgoal = tuple(self.env.start), tuple(self.env.goal)
        self.x0, self.xt = tuple(self.env.start), tuple(self.env.goal) 
        self.n = n 
        self.radius = radius 
        self.Vopen, self.Vopen_queue, self.Vclosed, self.V, self.Vunvisited, self.c = self.initNodeSets()
        self.neighbors = {}
        self.done = True
        self.Path = []
        self.Parent = {}

    def generateSampleSet(self, n):
        V = set()
        for i in range(n):
            V.add(tuple(sampleFree(self, bias = 0.0)))
        return V

    def initNodeSets(self):
        Vopen = {self.xinit} 
        closed = set() 
        V = self.generateSampleSet(self.n - 2) 
        Vunvisited = copy.deepcopy(V)
        Vunvisited.add(self.xgoal)
        V.add(self.xinit)
        V.add(self.xgoal)
        c = {node : np.inf for node in V}
        c[self.xinit] = 0
        Vopen_queue = MinheapPQ()
        Vopen_queue.put(self.xinit, c[self.xinit]) 
        return Vopen, Vopen_queue, closed, V, Vunvisited, c

    def Near(self, nodeset, node, rn):
        if node in self.neighbors:
            return self.neighbors[node]
        validnodes = {i for i in nodeset if getDist(i, node) < rn}
        return validnodes

    def Save(self, V_associated, node):
        self.neighbors[node] = V_associated

    def path(self, z, initT):
        path = []
        s = self.xgoal
        i = 0
        while s != self.xinit:
            path.append((s, self.Parent[s]))
            s = self.Parent[s]
            if i > self.n:
                break
            i += 1
        return path

    def Cost(self, x, y):
        return getDist(x, y)

    def FMTrun(self):
        start_time = time.time()
        
        z = self.xinit
        rn = self.radius
        Nz = self.Near(self.Vunvisited, z, rn)
        E = set()
        self.Save(Nz, z)
        ind = 0
        while z != self.xgoal:
            Vopen_new = set()
            Xnear = self.Near(self.Vunvisited, z ,rn)
            self.Save(Xnear, z)
            for x in Xnear:
                Ynear = list(self.Near(self.Vopen, x, rn))
                ymin = Ynear[np.argmin([self.c[y] + self.Cost(y,x) for y in Ynear])] 
                collide, _ = isCollide(self, ymin, x)
                if not collide:
                    E.add((ymin, x)) 
                    Vopen_new.add(x)
                    self.Parent[x] = z
                    self.Vunvisited = self.Vunvisited.difference({x})
                    self.c[x] = self.c[ymin] + self.Cost(ymin, x) 
            self.Vopen = self.Vopen.union(Vopen_new).difference({z})
            self.Vclosed.add(z)
            if len(self.Vopen) == 0:
                print('Failure')
                return 
            ind += 1
            print(str(ind) + ' node expanded')
            self.visualization(ind, E)
            Vopenlist = list(self.Vopen)
            z = Vopenlist[np.argmin([self.c[y] for y in self.Vopen])]
        T = (self.Vopen.union(self.Vclosed), E)
        self.done = True
        self.Path = self.path(z, T)
        self.visualization(ind, E)
        plt.show()

        self.execution_time = time.time() - start_time  
        self.path_length = self.calculate_path_length()  
        self.path_smoothness = self.calculate_path_smoothness()
        self.path_safety = self.calculate_path_safety()
        self.path_directness = self.calculate_path_directness()
        print(f"Execution Time: {self.execution_time} seconds")
        print(f"Path Length: {self.path_length} units")
        print(f"Path Smoothness: {self.path_smoothness}")
        print(f"Path Safety: {self.path_safety}")
        print(f"Path Directness: {self.path_directness}")

    def calculate_path_length(self):
        length = 0
        if self.Path and len(self.Path) > 1:
            for i in range(len(self.Path) - 1):
                length += np.linalg.norm(np.array(self.Path[i]) - np.array(self.Path[i + 1]))
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




        

    def visualization(self, ind, E):
        if ind % 100 == 0 or self.done:
            Path = np.array(self.Path)
            start = self.env.start
            goal = self.env.goal
            edges = np.array(list(E))
            ax = plt.subplot(111, projection='3d')
            ax.view_init(elev=65., azim=60.)
            ax.dist = 15
            ax.clear()
            draw_Spheres(ax, self.env.balls)
            draw_block_list(ax, self.env.blocks)
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
    A = FMT_star(radius = 1, n = 3000)
    A.FMTrun()



