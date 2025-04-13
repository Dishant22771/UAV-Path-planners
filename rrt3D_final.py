import numpy as np
from numpy.matlib import repmat
from collections import defaultdict
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../Sampling_based_Planning/")
from rrt_3D.environment import env
from rrt_3D.utils3D import getDist, sampleFree, nearest, steer, isCollide, near, visualization, cost, path
from animal_observation import calculate_observation_position

class rrtstar():
    def __init__(self):
        self.env = env()
        self.env.goal = calculate_observation_position(self.env)
        self.Parent = {}
        self.V = []
        self.COST = {}
        self.i = 0
        self.maxiter = 4000 
        self.stepsize = 1
        self.gamma = 7
        self.eta = self.stepsize
        self.Path = []
        self.done = False
        self.x0 = tuple(self.env.start)
        self.xt = tuple(self.env.goal)

        self.V.append(self.x0)
        self.ind = 0
        
        # Initialize the figure for visualization
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
    def wireup(self, x, y):
        self.Parent[x] = y

    def removewire(self, xnear):
        xparent = self.Parent[xnear]
        a = [xnear, xparent]

    def reached(self):
        self.done = True
        goal = self.xt
        xn = near(self, self.env.goal)
        c = [cost(self, tuple(x)) for x in xn]
        xncmin = xn[np.argmin(c)]
        self.wireup(goal, tuple(xncmin))
        self.V.append(goal)
        self.Path, self.D = path(self)

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
    
    def calculate_animal_visibility(self):
        if not self.Path:
            return 0.0
        final_position = np.array(self.env.goal)
        visible_count = 0
        total_count = len(self.env.balls)
        
        for animal in self.env.balls:
            animal_position = animal[:3]
            visible = True
            for block in self.env.blocks:
                if self.line_intersects_block(final_position, animal_position, block):
                    visible = False
                    break
            
            if visible:
                visible_count += 1
        
        return visible_count / total_count if total_count > 0 else 0.0
    
    def line_intersects_block(self, start, end, block):
        start = np.array(start)
        end = np.array(end)
        direction = end - start
        dirfrac = np.zeros(3)
        for i in range(3):
            if direction[i] != 0:
                dirfrac[i] = 1.0 / direction[i]
            else:
                dirfrac[i] = float('inf')
        
        t1 = (block[0] - start[0]) * dirfrac[0]
        t2 = (block[3] - start[0]) * dirfrac[0]
        t3 = (block[1] - start[1]) * dirfrac[1]
        t4 = (block[4] - start[1]) * dirfrac[1]
        t5 = (block[2] - start[2]) * dirfrac[2]
        t6 = (block[5] - start[2]) * dirfrac[2]
        tmin = max(min(t1, t2), min(t3, t4), min(t5, t6))
        tmax = min(max(t1, t2), max(t3, t4), max(t5, t6))
        
        if tmax < 0 or tmin > tmax or tmin > 1:
            return False
        
        return tmin <= 1

    def visualize(self):
        """Simplified visualization similar to extend_rrt3D approach"""
        self.ax.clear()
        
        # Set axis limits
        self.ax.set_xlim(self.env.boundary[0], self.env.boundary[3])
        self.ax.set_ylim(self.env.boundary[1], self.env.boundary[4])
        self.ax.set_zlim(self.env.boundary[2], self.env.boundary[5])
        
        # Labels and title
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('RRT* Path Planning for Animal Observation')
        
        # Plot start and goal points
        self.ax.scatter(self.env.start[0], self.env.start[1], self.env.start[2], 
                   c='g', s=100, marker='o', label='Start')
        self.ax.scatter(self.env.goal[0], self.env.goal[1], self.env.goal[2], 
                   c='r', s=100, marker='*', label='Goal (Observation Point)')
        
        # Plot obstacles (blocks as tree trunks)
        for block in self.env.blocks:
            x_min, y_min, z_min = block[0], block[1], block[2]
            x_max, y_max, z_max = block[3], block[4], block[5]
            
            vertices = [
                [x_min, y_min, z_min], [x_max, y_min, z_min], [x_max, y_max, z_min], [x_min, y_max, z_min],
                [x_min, y_min, z_max], [x_max, y_min, z_max], [x_max, y_max, z_max], [x_min, y_max, z_max]
            ]
            
            faces = [
                [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
                [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
                [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
                [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
                [vertices[0], vertices[3], vertices[7], vertices[4]],  # left
                [vertices[1], vertices[2], vertices[6], vertices[5]]   # right
            ]
            
            poly = Poly3DCollection(faces, alpha=0.5, linewidths=1, edgecolors='k')
            poly.set_facecolor('#8B4513')  # Brown color for trees
            self.ax.add_collection3d(poly)
        
        # Plot animals (spheres)
        for sphere in self.env.balls:
            center = sphere[:3]
            radius = sphere[3]
            
            # Simple sphere visualization using scatter with size based on radius
            self.ax.scatter(center[0], center[1], center[2], 
                       c='blue', s=radius*100, alpha=0.7)
        
        # Plot RRT* tree
        for i in range(len(self.V)):
            if i > 0:
                if self.V[i] in self.Parent:
                    xs = np.array([self.V[i][0], self.Parent[self.V[i]][0]])
                    ys = np.array([self.V[i][1], self.Parent[self.V[i]][1]])
                    zs = np.array([self.V[i][2], self.Parent[self.V[i]][2]])
                    self.ax.plot(xs, ys, zs, 'r-', alpha=0.2)
        
        # Plot final path if available
        if self.done and self.Path:
            for i in range(len(self.Path) - 1):
                x1, y1, z1 = self.Path[i][0]
                x2, y2, z2 = self.Path[i+1][0]
                self.ax.plot([x1, x2], [y1, y2], [z1, z2], 'r-', linewidth=2)
            
            # Add path statistics if available
            path_stats = ""
            if hasattr(self, 'D'):
                path_stats += f"Path length: {self.D:.2f}\n"
            if hasattr(self, 'path_smoothness'):
                path_stats += f"Smoothness: {self.path_smoothness:.2f}°\n"
            if hasattr(self, 'path_safety'):
                path_stats += f"Safety margin: {self.path_safety:.2f}\n"
            if hasattr(self, 'path_directness'):
                path_stats += f"Directness: {self.path_directness:.2f}\n"
            if hasattr(self, 'animal_visibility'):
                path_stats += f"Animal visibility: {self.animal_visibility*100:.1f}%"
            
            if path_stats:
                self.ax.text2D(0.02, 0.02, path_stats, transform=self.ax.transAxes, 
                          bbox=dict(facecolor='white', alpha=0.7))
        
        # Set a better viewpoint
        self.ax.view_init(elev=30, azim=45)
        
        # Draw and pause
        plt.draw()
        plt.pause(0.01)

    def run(self):
        xnew = self.x0
        print('start rrt*... ')
        print(f'Goal position (animal observation point): {self.env.goal}')
        print(f'Number of animals: {len(self.env.balls)}')
        
        while self.ind < self.maxiter:
            xrand = sampleFree(self)
            xnearest = nearest(self, xrand)
            xnew, dist = steer(self, xnearest, xrand)
            collide, _ = isCollide(self, xnearest, xnew, dist=dist)
            
            if not collide:
                Xnear = near(self, xnew)
                self.V.append(xnew) 
                
                # Visualization at intervals or when done
                if self.ind % 100 == 0 or self.done:
                    self.visualize()
                
                # Minimal path and minimal cost
                xmin, cmin = xnearest, cost(self, xnearest) + getDist(xnearest, xnew)
                
                # Connecting along minimal cost path
                Collide = []
                for xnear in Xnear:
                    xnear = tuple(xnear)
                    c1 = cost(self, xnear) + getDist(xnew, xnear)
                    collide, _ = isCollide(self, xnew, xnear)
                    Collide.append(collide)
                    if not collide and c1 < cmin:
                        xmin, cmin = xnear, c1
                
                self.wireup(xnew, xmin)
                
                # Rewire
                for i in range(len(Xnear)):
                    collide = Collide[i]
                    xnear = tuple(Xnear[i])
                    c2 = cost(self, xnew) + getDist(xnew, xnear)
                    if not collide and c2 < cost(self, xnear):
                        self.wireup(xnear, xnew)
                
                self.i += 1
            
            self.ind += 1
        
        # Max sample reached, calculate final path
        self.reached()
        self.path_smoothness = self.calculate_path_smoothness()
        self.path_safety = self.calculate_path_safety()
        self.path_directness = self.calculate_path_directness()
        self.animal_visibility = self.calculate_animal_visibility()
        
        # Final visualization with statistics
        self.visualize()
        
        # Print statistics
        print('Total distance = ' + str(self.D))
        print(f"Path Smoothness: {self.path_smoothness}")
        print(f"Path Safety: {self.path_safety}")
        print(f"Path Directness: {self.path_directness}")
        print(f"Animal Visibility: {self.animal_visibility * 100:.2f}%")
        
        # Display final result
        plt.show()

if __name__ == '__main__':
    p = rrtstar()
    starttime = time.time()
    p.run()
    print('time used = ' + str(time.time() - starttime))

