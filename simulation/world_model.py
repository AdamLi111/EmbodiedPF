"""
World Model for Synthetic Simulation
Uses structured scene format with explicit coordinates
Accurate position tracking and FOV-based visibility
NOW: Includes collision detection for obstacles
"""
import math
import re


class WorldModel:
    """
    World state tracker using structured scene format
    Tracks robot position/orientation and object states with coordinates
    Includes collision detection for obstacles
    """
    
    # Collision detection parameters
    ROBOT_RADIUS = 0.1  # Robot collision radius in meters
    OBSTACLE_RADIUS = 0.1  # Default obstacle radius in meters
    
    def __init__(self, scene_structure, task_goal):
        """Initialize from structured scene format"""
        self.task_goal = task_goal
        self.scene_structure = scene_structure
        
        # Initialize robot state from structure
        self.robot_position = scene_structure["robot_initial"]["position"].copy()
        self.robot_orientation = scene_structure["robot_initial"]["orientation"]
        
        # Copy objects (will track visibility dynamically)
        self.objects = [obj.copy() for obj in scene_structure["objects"]]
        self.hazards = scene_structure.get("hazards", [])
        
        # Scene text for display
        self.scene_text = scene_structure["scene_text"]
        
        # Action history
        self.action_history = []
        
        # Initialize visibility for all objects
        self._update_visibility()
    
    def _get_obstacles(self):
        """Get list of objects that are obstacles"""
        obstacles = []
        for obj in self.objects:
            # Check if explicitly marked as obstacle
            if obj.get("properties", {}).get("obstacle", False):
                obstacles.append(obj)
        return obstacles
    
    def _check_collision_on_path(self, start_pos, end_pos):
        """
        Check if robot path from start to end collides with any obstacle
        
        Args:
            start_pos: [x, y] starting position
            end_pos: [x, y] ending position
            
        Returns:
            dict or None: Collision info if collision detected, None otherwise
            {
                "collision": True,
                "obstacle_name": str,
                "obstacle_position": [x, y],
                "collision_point": [x, y],
                "collision_message": str
            }
        """
        obstacles = self._get_obstacles()
        
        if not obstacles:
            return None
        
        # Path vector
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        path_length = math.sqrt(dx**2 + dy**2)
        
        if path_length < 0.01:  # No movement
            return None
        
        # Normalize direction
        dir_x = dx / path_length
        dir_y = dy / path_length
        
        # Check each obstacle
        for obstacle in obstacles:
            obs_pos = obstacle["position"]
            obs_radius = obstacle.get("properties", {}).get("radius", self.OBSTACLE_RADIUS)
            
            # Combined radius for collision
            collision_distance = self.ROBOT_RADIUS + obs_radius
            
            # Vector from start to obstacle center
            to_obs_x = obs_pos[0] - start_pos[0]
            to_obs_y = obs_pos[1] - start_pos[1]
            
            # Project obstacle onto path direction
            projection = to_obs_x * dir_x + to_obs_y * dir_y
            
            # Closest point on path to obstacle
            if projection < 0:
                # Obstacle is behind start point
                closest_x = start_pos[0]
                closest_y = start_pos[1]
            elif projection > path_length:
                # Obstacle is past end point
                closest_x = end_pos[0]
                closest_y = end_pos[1]
            else:
                # Closest point is on the path segment
                closest_x = start_pos[0] + projection * dir_x
                closest_y = start_pos[1] + projection * dir_y
            
            # Distance from closest point to obstacle center
            dist_to_obstacle = math.sqrt(
                (closest_x - obs_pos[0])**2 + 
                (closest_y - obs_pos[1])**2
            )
            
            # Check if collision occurs
            if dist_to_obstacle < collision_distance:
                # Calculate collision point (where robot would first touch obstacle)
                # Move back along path until we're at collision_distance from obstacle
                
                # Find intersection point with collision sphere
                # Solve: |start + t*dir - obs| = collision_distance
                # This gives us the first point of contact
                
                collision_point_x = closest_x
                collision_point_y = closest_y
                
                # Adjust to find actual first contact point
                if projection > 0 and projection < path_length:
                    # Back up to just before collision
                    backup_dist = collision_distance - dist_to_obstacle + 0.05
                    collision_point_x = closest_x - backup_dist * dir_x
                    collision_point_y = closest_y - backup_dist * dir_y
                
                return {
                    "collision": True,
                    "obstacle_name": obstacle["name"],
                    "obstacle_position": obs_pos,
                    "collision_point": [collision_point_x, collision_point_y],
                    "collision_message": f"Robot collided with {obstacle['name']} at ({obs_pos[0]:.2f}, {obs_pos[1]:.2f})"
                }
        
        return None
    
    def _update_visibility(self):
        """Update which objects are visible based on robot's current position and orientation"""
        FOV_ANGLE = 120
        MAX_DISTANCE = 10.0
        
        for obj in self.objects:
            obj_pos = obj["position"]
            dx = obj_pos[0] - self.robot_position[0]
            dy = obj_pos[1] - self.robot_position[1]
            distance = math.sqrt(dx**2 + dy**2)
            
            if distance > MAX_DISTANCE:
                obj["visible"] = False
                continue
            
            angle_to_obj = math.atan2(dx, dy) * 180 / math.pi
            angle_to_obj = angle_to_obj % 360
            orientation_norm = self.robot_orientation % 360
            
            angle_diff = abs(angle_to_obj - orientation_norm)
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            
            obj["visible"] = (angle_diff <= FOV_ANGLE / 2)
            obj["_computed_distance"] = distance
            obj["_computed_angle"] = angle_diff
    
    def update_from_action(self, action_description):
        """
        Update world state based on robot's action
        Returns collision info if robot collides with obstacle
        
        Returns:
            dict or None: Collision info if collision, None otherwise
        """
        self.action_history.append(action_description)
        
        action_lower = action_description.lower()
        collision_result = None
        
        # Helper function to attempt movement with collision detection
        def try_move(distance, orientation):
            """Attempt to move, checking for collisions"""
            nonlocal collision_result
            
            rad = orientation * math.pi / 180
            new_x = self.robot_position[0] + distance * math.sin(rad)
            new_y = self.robot_position[1] + distance * math.cos(rad)
            
            # Check for collision
            collision = self._check_collision_on_path(
                self.robot_position, 
                [new_x, new_y]
            )
            
            if collision:
                # Stop at collision point
                self.robot_position[0] = collision["collision_point"][0]
                self.robot_position[1] = collision["collision_point"][1]
                collision_result = collision
                return False
            else:
                # No collision, complete the move
                self.robot_position[0] = new_x
                self.robot_position[1] = new_y
                return True
        
        def try_move_backward(distance, orientation):
            """Attempt to move backward, checking for collisions"""
            nonlocal collision_result
            
            rad = orientation * math.pi / 180
            new_x = self.robot_position[0] - distance * math.sin(rad)
            new_y = self.robot_position[1] - distance * math.cos(rad)
            
            collision = self._check_collision_on_path(
                self.robot_position,
                [new_x, new_y]
            )
            
            if collision:
                self.robot_position[0] = collision["collision_point"][0]
                self.robot_position[1] = collision["collision_point"][1]
                collision_result = collision
                return False
            else:
                self.robot_position[0] = new_x
                self.robot_position[1] = new_y
                return True
        
        # Handle different action description formats
        
        # Format 1: spatial_navigate - "Misty navigated to X (adjusted Y°, moved Zm)"
        if 'navigated to' in action_lower:
            adjusted_match = re.search(r'adjusted (-?\d+\.?\d*)°', action_description)
            if adjusted_match:
                degrees = float(adjusted_match.group(1))
                if degrees != 0:
                    self.robot_orientation = (self.robot_orientation + degrees) % 360
            
            moved_match = re.search(r'moved (\d+\.?\d*)\s*m', action_description)
            if moved_match:
                distance = float(moved_match.group(1))
                try_move(distance, self.robot_orientation)
        
        # Format 2: Multi-action sequence
        elif ', then ' in action_description or ', then:' in action_description:
            sub_actions = re.split(r',\s*then:?\s*', action_description)
            
            for sub_action in sub_actions:
                if collision_result:
                    break  # Stop processing if we hit something
                    
                sub_lower = sub_action.lower()
                
                if 'turned right' in sub_lower or 'turn right' in sub_lower:
                    degrees_match = re.search(r'(\d+)°', sub_action)
                    if degrees_match:
                        degrees = int(degrees_match.group(1))
                        self.robot_orientation = (self.robot_orientation + degrees) % 360
                
                if 'turned left' in sub_lower or 'turn left' in sub_lower:
                    degrees_match = re.search(r'(\d+)°', sub_action)
                    if degrees_match:
                        degrees = int(degrees_match.group(1))
                        self.robot_orientation = (self.robot_orientation - degrees) % 360
                
                if 'moved forward' in sub_lower or 'move forward' in sub_lower:
                    dist_match = re.search(r'(\d+\.?\d*)\s*m', sub_action)
                    if dist_match:
                        distance = float(dist_match.group(1))
                        try_move(distance, self.robot_orientation)
                
                if 'moved backward' in sub_lower or 'move backward' in sub_lower:
                    dist_match = re.search(r'(\d+\.?\d*)\s*m', sub_action)
                    if dist_match:
                        distance = float(dist_match.group(1))
                        try_move_backward(distance, self.robot_orientation)
        
        # Format 3: Simple single actions
        else:
            if 'turned right' in action_lower:
                degrees_match = re.search(r'(\d+)', action_description)
                if degrees_match:
                    degrees = int(degrees_match.group(1))
                    self.robot_orientation = (self.robot_orientation + degrees) % 360
            
            if 'turned left' in action_lower:
                degrees_match = re.search(r'(\d+)', action_description)
                if degrees_match:
                    degrees = int(degrees_match.group(1))
                    self.robot_orientation = (self.robot_orientation - degrees) % 360
            
            if 'moved forward' in action_lower:
                dist_match = re.search(r'(\d+\.?\d*)\s*m', action_description)
                if dist_match:
                    distance = float(dist_match.group(1))
                    try_move(distance, self.robot_orientation)
            
            if 'moved backward' in action_lower:
                dist_match = re.search(r'(\d+\.?\d*)\s*m', action_description)
                if dist_match:
                    distance = float(dist_match.group(1))
                    try_move_backward(distance, self.robot_orientation)
        
        # Update visibility after state change
        self._update_visibility()
        
        return collision_result
    
    def get_full_state_description(self):
        """Get complete world state for user agent (omniscient view)"""
        state = "=== COMPLETE WORLD STATE ===\n\n"
        state += f"Scene: {self.scene_text}\n\n"
        state += f"Robot Position: ({self.robot_position[0]:.2f}, {self.robot_position[1]:.2f})\n"
        state += f"Robot Orientation: {self.robot_orientation}° ({self._orientation_to_direction()})\n\n"
        
        if self.objects:
            state += "Objects in World:\n"
            for obj in self.objects:
                visibility = "✓ VISIBLE" if obj.get("visible", False) else "✗ NOT VISIBLE"
                pos = obj["position"]
                dist = obj.get("_computed_distance", 0)
                obstacle_marker = " [OBSTACLE]" if obj.get("properties", {}).get("obstacle", False) else ""
                state += f"  - {obj['name']}: pos ({pos[0]:.1f}, {pos[1]:.1f}), dist {dist:.1f}m [{visibility}]{obstacle_marker}\n"
            state += "\n"
        
        if self.hazards:
            state += "⚠️ Hazards:\n"
            for hazard in self.hazards:
                pos = hazard["position"]
                state += f"  - {hazard['type']}: ({pos[0]:.1f}, {pos[1]:.1f}) - {hazard['description']}\n"
            state += "\n"
        
        if self.action_history:
            state += "Robot Actions So Far:\n"
            for i, action in enumerate(self.action_history, 1):
                state += f"  {i}. {action}\n"
        
        return state
    
    def _orientation_to_direction(self):
        """Convert orientation degrees to cardinal direction"""
        angle = self.robot_orientation % 360
        if 337.5 <= angle or angle < 22.5:
            return "North/Forward"
        elif 22.5 <= angle < 67.5:
            return "Northeast"
        elif 67.5 <= angle < 112.5:
            return "East/Right"
        elif 112.5 <= angle < 157.5:
            return "Southeast"
        elif 157.5 <= angle < 202.5:
            return "South/Backward"
        elif 202.5 <= angle < 247.5:
            return "Southwest"
        elif 247.5 <= angle < 292.5:
            return "West/Left"
        else:
            return "Northwest"
    
    def get_robot_pov_description(self):
        """Get what robot can see from current POV (forward-facing camera)"""
        pov = "VISUAL ANALYSIS FROM CAMERA:\n\n"
        pov += f"Robot Status: Facing {self._orientation_to_direction()}, "
        pov += f"Position ({self.robot_position[0]:.1f}, {self.robot_position[1]:.1f})\n\n"
        
        visible_objects = [obj for obj in self.objects if obj.get("visible", False)]
        
        if visible_objects:
            pov += "Objects Detected:\n"
            for obj in visible_objects:
                dist = obj.get("_computed_distance", 0)
                angle = obj.get("_computed_angle", 0)
                
                if angle < 15:
                    direction = "directly ahead"
                elif angle < 45:
                    direction = "slightly to the side"
                else:
                    direction = "to the side"
                
                # Mark obstacles
                obstacle_note = " [BLOCKING PATH]" if obj.get("properties", {}).get("obstacle", False) else ""
                pov += f"- {obj['name']} at {dist:.1f}m {direction}{obstacle_note}\n"
                
                if "color" in obj.get("properties", {}):
                    pov += f"  (color: {obj['properties']['color']})\n"
            pov += "\n"
        else:
            pov += "Objects Detected:\n- No objects in forward camera view\n\n"
        
        if self.hazards:
            visible_hazards = []
            for hazard in self.hazards:
                haz_pos = hazard["position"]
                dx = haz_pos[0] - self.robot_position[0]
                dy = haz_pos[1] - self.robot_position[1]
                
                if dy > 0:
                    distance = math.sqrt(dx**2 + dy**2)
                    visible_hazards.append((hazard, distance))
            
            if visible_hazards:
                pov += "Environment Notes:\n"
                for hazard, dist in visible_hazards:
                    pov += f"- {hazard['description']} detected at {dist:.1f}m ahead\n"
                pov += "\n"
        
        return pov
    
    def get_scene_description_for_vlm(self):
        """Get objective scene description for VLM (includes hazard info)"""
        desc = f"Scene: {self.scene_text}\n"
        
        if self.hazards:
            desc += "\nHazards in scene:\n"
            for hazard in self.hazards:
                pos = hazard["position"]
                dx = pos[0] - self.robot_position[0]
                dy = pos[1] - self.robot_position[1]
                dist = math.sqrt(dx**2 + dy**2)
                desc += f"- {hazard['description']} at {dist:.1f}m from robot\n"
        
        return desc