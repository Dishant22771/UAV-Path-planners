import numpy as np

def calculate_observation_position(env):
    """
    Calculate an optimal position for observing all animals (spheres) in the environment.
    
    Args:
        env: The environment object containing animals (balls) and obstacles
        
    Returns:
        numpy.ndarray: The optimal observation position [x, y, z]
    """
    if len(env.balls) == 0:
        return np.array([10.0, 10.0, 3.0])
    animal_positions = np.array([ball[:3] for ball in env.balls])
    center = np.mean(animal_positions, axis=0)
    avg_radius = np.mean([ball[3] for ball in env.balls])
    max_dist = 0
    for i in range(len(env.balls)):
        for j in range(i+1, len(env.balls)):
            dist = np.linalg.norm(animal_positions[i] - animal_positions[j])
            max_dist = max(max_dist, dist)
    max_height = max([ball[2] + ball[3] for ball in env.balls])
    observation_height = max_height + 2.0
    observation_distance = max(max_dist * 0.5, avg_radius * 5)
    observation_height = min(observation_height, env.boundary[5] - 1.0)
    candidates = []
    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    
    for angle in angles:
        pos = np.array([
            center[0] + observation_distance * np.cos(angle),
            center[1] + observation_distance * np.sin(angle),
            observation_height
        ])
        if (pos[0] >= env.boundary[0] and pos[0] <= env.boundary[3] and
            pos[1] >= env.boundary[1] and pos[1] <= env.boundary[2] and
            pos[2] >= env.boundary[2] and pos[2] <= env.boundary[5]):
            
            visible = True
            for animal in env.balls:
                animal_pos = animal[:3]
                direction = animal_pos - pos
                distance = np.linalg.norm(direction)
                direction = direction / distance
                for block in env.blocks:
                    if line_intersects_block(pos, animal_pos, block):
                        visible = False
                        break
                
                if not visible:
                    break
            
            if visible:
                score = calculate_visibility_score(pos, env.balls)
                candidates.append((pos, score))
    if not candidates:
        return np.array([center[0], center[1], observation_height])
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]

def calculate_visibility_score(position, animals):
    """
    Calculate a score representing how well animals can be observed from a position.
    Higher score is better.
    
    Args:
        position: The observation position
        animals: List of animals (spheres) [x, y, z, radius]
        
    Returns:
        float: Visibility score
    """
    score = 0
    
    for animal in animals:
        animal_pos = animal[:3]
        dist = np.linalg.norm(animal_pos - position)
        optimal_dist = animal[3] * 5 
        dist_score = 1 / (abs(dist - optimal_dist) + 1)
        height_diff = position[2] - animal_pos[2]
        height_score = min(height_diff / (animal[3] * 2), 1.0) if height_diff > 0 else 0
        
        score += dist_score + height_score
    
    return score

def line_intersects_block(start, end, block):
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
