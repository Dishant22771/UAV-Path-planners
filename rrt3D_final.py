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
    
    def calculate_animal_visibility(self):
        """
        Calculate the percentage of animals that are visible from the final position.
        
        Returns:
            float: Percentage of visible animals (0.0 to 1.0)
        """
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
        """
        Check if a line segment intersects with an AABB block.
        
        Args:
            start: Starting point of the line segment
            end: Ending point of the line segment
            block: AABB block [xmin, ymin, zmin, xmax, ymax, zmax]
            
        Returns:
            bool: True if intersection occurs, False otherwise
        """
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
        
        # If tmax < 0, the ray is intersecting the AABB, but the whole AABB is behind the ray
        # If tmin > tmax, the ray doesn't intersect the AABB
        # If tmin > 1, the AABB is beyond the segment's end
        if tmax < 0 or tmin > tmax or tmin > 1:
            return False
        
        # Ray does intersect AABB, and intersection point is at start + tmin * direction
        return tmin <= 1



    def draw_cuboid(ax, block, color='#8B4513'):
        """Draw a cuboid representing a tree trunk"""
        x_min, y_min, z_min, x_max, y_max, z_max = block
        vertices = [
            [x_min, y_min, z_min], [x_max, y_min, z_min], [x_max, y_max, z_min], [x_min, y_max, z_min],
            [x_min, y_min, z_max], [x_max, y_min, z_max], [x_max, y_max, z_max], [x_min, y_max, z_max]
        ]
        
        # Define faces using vertices
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
            [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
            [vertices[0], vertices[3], vertices[7], vertices[4]],  # left
            [vertices[1], vertices[2], vertices[6], vertices[5]]   # right
        ]
        poly = Poly3DCollection(faces, alpha=0.7, linewidths=1, edgecolors='k')
        poly.set_facecolor(color)
        ax.add_collection3d(poly)

    def draw_tree_canopy(ax, base_center, width, height, color='#228B22'):
       
    def draw_animal(ax, sphere, animal_type, color):
        x, y, z, r = sphere
        
        if animal_type == 'elephant':
            # Body
            u = np.linspace(0, 2 * np.pi, 10)
            v = np.linspace(0, np.pi, 10)
            x_body = x + r * 1.2 * np.outer(np.cos(u), np.sin(v))
            y_body = y + r * 0.8 * np.outer(np.sin(u), np.sin(v))
            z_body = z + r * 0.9 * np.outer(np.ones(np.size(u)), np.cos(v))
            ax.plot_surface(x_body, y_body, z_body, color=color, alpha=0.7)
            ax.plot([x+r*0.8, x+r*1.5], [y, y], [z+r*0.4, z+r*0.7], color=color, linewidth=5)
            ax.plot([x+r*1.5, x+r*1.8], [y, y], [z+r*0.7, z], color=color, linewidth=3)
            
        elif animal_type == 'rabbit':
            u = np.linspace(0, 2 * np.pi, 10)
            v = np.linspace(0, np.pi, 10)
            x_body = x + r * 0.7 * np.outer(np.cos(u), np.sin(v))
            y_body = y + r * 0.7 * np.outer(np.sin(u), np.sin(v))
            z_body = z + r * 0.5 * np.outer(np.ones(np.size(u)), np.cos(v))
            ax.plot_surface(x_body, y_body, z_body, color=color, alpha=0.7)
            
            # Ears
            ax.plot([x-r*0.2, x-r*0.2], [y, y], [z+r*0.5, z+r*1.2], color=color, linewidth=3)
            ax.plot([x+r*0.2, x+r*0.2], [y, y], [z+r*0.5, z+r*1.2], color=color, linewidth=3)
            
        elif animal_type == 'giraffe':
            # Body
            u = np.linspace(0, 2 * np.pi, 10)
            v = np.linspace(0, np.pi, 10)
            x_body = x + r * 0.7 * np.outer(np.cos(u), np.sin(v))
            y_body = y + r * 1.0 * np.outer(np.sin(u), np.sin(v))
            z_body = z + r * 0.6 * np.outer(np.ones(np.size(u)), np.cos(v))
            ax.plot_surface(x_body, y_body, z_body, color=color, alpha=0.7)
            
            # Neck
            ax.plot([x, x+r*0.4], [y, y], [z+r*0.6, z+r*2.0], color=color, linewidth=5)
            
            # Head
            head_radius = r * 0.3
            u = np.linspace(0, 2 * np.pi, 8)
            v = np.linspace(0, np.pi, 8)
            x_head = x+r*0.4 + head_radius * np.outer(np.cos(u), np.sin(v))
            y_head = y + head_radius * np.outer(np.sin(u), np.sin(v))
            z_head = z+r*2.0 + head_radius * np.outer(np.ones(np.size(u)), np.cos(v))
            ax.plot_surface(x_head, y_head, z_head, color=color, alpha=0.7)
            
        else:
            u = np.linspace(0, 2 * np.pi, 15)
            v = np.linspace(0, np.pi, 15)
            x_sphere = x + r * np.outer(np.cos(u), np.sin(v))
            y_sphere = y + r * np.outer(np.sin(u), np.sin(v))
            z_sphere = z + r * np.outer(np.ones(np.size(u)), np.cos(v))
            ax.plot_surface(x_sphere, y_sphere, z_sphere, color=color, alpha=0.7)
            
            # Add "ears" for some animals
            if animal_type in ['deer', 'fox', 'mouse', 'monkey']:
                ear_size = r * 0.3
                ax.plot([x-r*0.5, x-r*0.5], [y, y], [z+r*0.7, z+r*0.7+ear_size], color=color, linewidth=3)
                ax.plot([x+r*0.5, x+r*0.5], [y, y], [z+r*0.7, z+r*0.7+ear_size], color=color, linewidth=3)
            
            # Add "horns" for deer or buffalo
            if animal_type in ['deer', 'buffalo']:
                ax.plot([x-r*0.3, x-r*0.7], [y, y], [z+r*0.8, z+r*1.3], color=color, linewidth=2)
                ax.plot([x+r*0.3, x+r*0.7], [y, y], [z+r*0.8, z+r*1.3], color=color, linewidth=2)


    def run(self):
        xnew = self.x0
        print('start rrt*... ')
        print(f'Goal position (animal observation point): {self.env.goal}')
        print(f'Number of animals: {len(self.env.balls)}')
        self.fig = plt.figure(figsize = (10,8))
        
        # Define animal types and colors
        animal_types = ['elephant', 'rabbit', 'deer', 'mouse', 'bear', 'giraffe', 'fox', 'buffalo', 'monkey', 'tiger']
        colors = ['#A0A0A0', '#D2B48C', '#8B4513', '#696969', '#654321', '#FFFF00', '#FF4500', '#000000', '#A52A2A', '#FFA500']
        
        while self.ind < self.maxiter:
            xrand    = sampleFree(self)
            xnearest = nearest(self,xrand)
            xnew, dist  = steer(self,xnearest,xrand)
            collide, _ = isCollide(self,xnearest,xnew,dist=dist)
            if not collide:
                Xnear = near(self,xnew)
                self.V.append(xnew) 
                if self.ind % 100 == 0 or self.done:
                    plt.clf()
                    ax = plt.axes(projection='3d')
                    
                    # Set axis limits
                    ax.set_xlim(self.env.boundary[0], self.env.boundary[3])
                    ax.set_ylim(self.env.boundary[1], self.env.boundary[4])
                    ax.set_zlim(self.env.boundary[2], self.env.boundary[5])
                    
                    # Labels and title
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Z')
                    ax.set_title('RRT* Path to Optimal Animal Observation Point')
                    
                    # Plot start and goal points
                    ax.scatter(self.env.start[0], self.env.start[1], self.env.start[2], c='g', s=100, marker='o', label='Start')
                    ax.scatter(self.env.goal[0], self.env.goal[1], self.env.goal[2], c='r', s=100, marker='*', label='Goal (Observation Point)')
                    
                    # Plot trees (blocks) with realistic tree shapes
                    for i, block in enumerate(self.env.blocks):
                        is_trunk = i % 2 == 0  # Even indices are trunks, odd are canopies
                        
                        if is_trunk:
                            # Draw trunk as brown cuboid
                            rrtstar.draw_cuboid(ax, block, color='#8B4513')
                        else:
                            # Draw tree canopy as green cone/pyramid on top of trunk
                            trunk_block = self.env.blocks[i-1]
                            # Get the center point of the top of the trunk
                            trunk_top_center = [(trunk_block[0] + trunk_block[3])/2, 
                                              (trunk_block[1] + trunk_block[4])/2, 
                                              trunk_block[5]]
                            
                            # Get the dimensions of the canopy
                            canopy_width = block[3] - block[0]
                            canopy_height = block[5] - block[2]
                            
                            # Draw a cone for the canopy
                            rrtstar.draw_tree_canopy(ax, trunk_top_center, canopy_width, canopy_height)
                    
                    # Plot animals (spheres) with animal shapes
                    for i, sphere in enumerate(self.env.balls):
                        animal_type = animal_types[i % len(animal_types)]
                        color = colors[i % len(colors)]
                        rrtstar.draw_animal(ax, sphere, animal_type, color)
                    
                    # Plot RRT* tree
                    for i in range(len(self.V)):
                        if i > 0:
                            if self.V[i] in self.Parent:
                                xs = np.array([self.V[i][0], self.Parent[self.V[i]][0]])
                                ys = np.array([self.V[i][1], self.Parent[self.V[i]][1]])
                                zs = np.array([self.V[i][2], self.Parent[self.V[i]][2]])
                                ax.plot(xs, ys, zs, 'b-', alpha=0.2)
                    
                    # Plot final path
                    if self.done:
                        for i in range(len(self.Path) - 1):
                            x1, y1, z1 = self.Path[i][0]
                            x2, y2, z2 = self.Path[i+1][0]
                            ax.plot([x1, x2], [y1, y2], [z1, z2], 'r-', linewidth=3)
                        
                        # Add path statistics to the plot
                        path_stats = f"Path length: {self.D:.2f}\n"
                        if hasattr(self, 'path_smoothness'):
                            path_stats += f"Smoothness: {self.path_smoothness:.2f}°\n"
                        if hasattr(self, 'path_safety'):
                            path_stats += f"Safety margin: {self.path_safety:.2f}\n"
                        if hasattr(self, 'path_directness'):
                            path_stats += f"Directness: {self.path_directness:.2f}\n"
                        if hasattr(self, 'animal_visibility'):
                            path_stats += f"Animal visibility: {self.animal_visibility*100:.1f}%"
                        
                        ax.text2D(0.02, 0.02, path_stats, transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.7))
                    
                    # Create custom legend
                    legend_elements = [
                        mpatches.Patch(color='g', label='Start Point'),
                        mpatches.Patch(color='r', label='Observation Point'),
                        mpatches.Patch(color='#8B4513', label='Tree Trunk'),
                        mpatches.Patch(color='#228B22', label='Tree Canopy'),
                        mpatches.Patch(color='blue', label='RRT* Tree'),
                        mpatches.Patch(color='red', label='Final Path')
                    ]
                    
                    # Add some animals to the legend
                    for i in range(min(3, len(animal_types))):
                        legend_elements.append(mpatches.Patch(color=colors[i], label=animal_types[i].capitalize()))
                    
                    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
                    
                    # Set a better viewpoint
                    ax.view_init(elev=30, azim=45)
                    plt.draw()
                    plt.pause(0.001)
                
                # minimal path and minimal cost
                xmin, cmin = xnearest, cost(self, xnearest) + getDist(xnearest, xnew)
                # connecting along minimal cost path
                Collide = []
                for xnear in Xnear:
                    xnear = tuple(xnear)
                    c1 = cost(self, xnear) + getDist(xnew, xnear)
                    collide, _ = isCollide(self, xnew, xnear)
                    Collide.append(collide)
                    if not collide and c1 < cmin:
                        xmin, cmin = xnear, c1
                self.wireup(xnew, xmin)
                # rewire
                for i in range(len(Xnear)):
                    collide = Collide[i]
                    xnear = tuple(Xnear[i])
                    c2 = cost(self, xnew) + getDist(xnew, xnear)
                    if not collide and c2 < cost(self, xnear):
                        # self.removewire(xnear)
                        self.wireup(xnear, xnew)
                self.i += 1
            self.ind += 1
        # max sample reached
        self.reached()
        self.path_smoothness = self.calculate_path_smoothness()
        self.path_safety = self.calculate_path_safety()
        self.path_directness = self.calculate_path_directness()
        self.animal_visibility = self.calculate_animal_visibility()
        print('time used = ' + str(time.time()-starttime))
        print('Total distance = '+str(self.D))
        print(f"Path Smoothness: {self.path_smoothness}")
        print(f"Path Safety: {self.path_safety}")
        print(f"Path Directness: {self.path_directness}")
        print(f"Animal Visibility: {self.animal_visibility * 100:.2f}%")
        
        # Final visualization
        plt.clf()
        ax = plt.axes(projection='3d')
        
        # Set axis limits
        ax.set_xlim(self.env.boundary[0], self.env.boundary[3])
        ax.set_ylim(self.env.boundary[1], self.env.boundary[4])
        ax.set_zlim(self.env.boundary[2], self.env.boundary[5])
        
        # Labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('RRT* Path to Optimal Animal Observation Point')
        
        # Plot start and goal points
        ax.scatter(self.env.start[0], self.env.start[1], self.env.start[2], c='g', s=100, marker='o', label='Start')
        ax.scatter(self.env.goal[0], self.env.goal[1], self.env.goal[2], c='r', s=100, marker='*', label='Goal (Observation Point)')
        
        # Plot trees (blocks) with realistic tree shapes
        for i, block in enumerate(self.env.blocks):
            is_trunk = i % 2 == 0  # Even indices are trunks, odd are canopies
            
            if is_trunk:
                # Draw trunk as brown cuboid
                rrtstar.draw_cuboid(ax, block, color='#8B4513')
            else:
                # Draw tree canopy as green cone/pyramid on top of trunk
                trunk_block = self.env.blocks[i-1]
                # Get the center point of the top of the trunk
                trunk_top_center = [(trunk_block[0] + trunk_block[3])/2, 
                                  (trunk_block[1] + trunk_block[4])/2, 
                                  trunk_block[5]]
                
                # Get the dimensions of the canopy
                canopy_width = block[3] - block[0]
                canopy_height = block[5] - block[2]
                
                # Draw a cone for the canopy
                rrtstar.draw_tree_canopy(ax, trunk_top_center, canopy_width, canopy_height)
        
        # Plot animals (spheres) with animal shapes
        for i, sphere in enumerate(self.env.balls):
            animal_type = animal_types[i % len(animal_types)]
            color = colors[i % len(colors)]
            rrtstar.draw_animal(ax, sphere, animal_type, color)
        
        # Plot RRT* tree
        for i in range(len(self.V)):
            if i > 0:
                if self.V[i] in self.Parent:
                    xs = np.array([self.V[i][0], self.Parent[self.V[i]][0]])
                    ys = np.array([self.V[i][1], self.Parent[self.V[i]][1]])
                    zs = np.array([self.V[i][2], self.Parent[self.V[i]][2]])
                    ax.plot(xs, ys, zs, 'b-', alpha=0.2)
        
        # Plot final path
        for i in range(len(self.Path) - 1):
            x1, y1, z1 = self.Path[i][0]
            x2, y2, z2 = self.Path[i+1][0]
            ax.plot([x1, x2], [y1, y2], [z1, z2], 'r-', linewidth=3)
        
        # Add path statistics to the plot
        path_stats = f"Path length: {self.D:.2f}\n"
        path_stats += f"Smoothness: {self.path_smoothness:.2f}°\n"
        path_stats += f"Safety margin: {self.path_safety:.2f}\n" 
        path_stats += f"Directness: {self.path_directness:.2f}\n"
        path_stats += f"Animal visibility: {self.animal_visibility*100:.1f}%"
        
        ax.text2D(0.02, 0.02, path_stats, transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.7))
        
        # Create custom legend
        legend_elements = [
            mpatches.Patch(color='g', label='Start Point'),
            mpatches.Patch(color='r', label='Observation Point'),
            mpatches.Patch(color='#8B4513', label='Tree Trunk'),
            mpatches.Patch(color='#228B22', label='Tree Canopy'),
            mpatches.Patch(color='blue', label='RRT* Tree'),
            mpatches.Patch(color='red', label='Final Path')
        ]
        
        # Add some animals to the legend
        for i in range(min(3, len(animal_types))):
            legend_elements.append(mpatches.Patch(color=colors[i], label=animal_types[i].capitalize()))
        
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
        
        # Set a better viewpoint
        ax.view_init(elev=30, azim=45)
        plt.draw()
        plt.show()
        

if __name__ == '__main__':
    p = rrtstar()
    starttime = time.time()
    p.run()
