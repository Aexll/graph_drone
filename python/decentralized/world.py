import numpy as np
import dearpygui.dearpygui as dpg
from drone import Drone
from target import Target


class WorldConfiguration:
    """Configuration for the world, such as map size and bounds."""
    MAP_SIZE = np.array([800, 800], dtype=np.float32)
    BOUNDS = np.array([[0, 0], MAP_SIZE], dtype=np.float32)
    DRONES: list[tuple] = [(300, 400),
                           (400, 400),
                           (500, 400)]
    TARGETS: list[tuple|None] = [(100, 100),
                None,
                None]
    
    def __init__(self, map_size:np.ndarray = MAP_SIZE, bounds:np.ndarray = BOUNDS, drones:list[tuple|None] = DRONES, targets:list[tuple|None] = TARGETS):
        self.map_size = map_size
        self.bounds = bounds
        self.drones = drones
        self.targets = targets






class World:

    MAP_SIZE = np.array([800, 800], dtype=np.float32)

    def __init__(self, config:WorldConfiguration|None = None, parent=None, bounds:np.ndarray = None):
        self.parent = parent
        self.bounds = bounds
        self.drones:list['Drone'] = []
        self.targets:list['Target'] = []
        if config:
            self.map_size = config.map_size
            self.bounds = config.bounds
            for drone_pos,target_pos in zip(config.drones, config.targets):
                drone = Drone(position=np.array(drone_pos), target=None, scan_for_drones_method=self.get_drones_in_range, scan_for_targets_method=self.get_targets_in_range)
                if target_pos is not None:
                    target = Target(position=np.array(target_pos), target_type=0, bounds=self.bounds)
                    drone.target = target
                    self.add_target(target)
                self.add_drone(drone)

        self.global_time:float = 0.0
        self.delta_time:float = 0.08  # Time step for updates, can be adjusted as needed


    def draw(self, draw_list):
        for drone in self.drones:
            drone.draw(draw_list)
        for target in self.targets:
            target.draw(draw_list)

    def update(self):
        for target in self.targets:
            target.update(self.delta_time)
        for drone in self.drones:
            drone.update(self.delta_time)

    def add_drone(self, drone:Drone):
        self.drones.append(drone)

    def add_target(self, target:Target):
        self.targets.append(target)

    def remove_drone(self, idx:int):
        drone = self.drones.pop(idx)
        drone.__del__()
        del drone  # Remove reference to the drone

    def remove_target(self, idx:int):
        self.targets.pop(idx)

    def add_drone_with_target(self, drone:Drone, target:Target):
        """Add a drone with a target."""
        drone.target = target
        if drone not in self.drones:
            drone.scan_for_drones_method = self.get_drones_in_range
            drone.scan_for_targets_method = self.get_targets_in_range
            drone.targets = self.targets
            self.add_drone(drone)
        if target not in self.targets:
            self.add_target(target)
    
    def generate_random_drone(self) -> Drone:
        position = np.random.rand(2) * self.MAP_SIZE[1]
        return Drone(position=position, target=None, scan_for_drones_method=self.get_drones_in_range, scan_for_targets_method=self.get_targets_in_range)

    def generate_random_target(self) -> Target:
        position = np.random.rand(2) * self.MAP_SIZE[1]
        target_type = np.random.choice([0, 1])
        bounds = np.array([[0, 0], self.MAP_SIZE])
        return Target(position=position, target_type=target_type, bounds=bounds)

    def generate_random_drone_with_target(self):
        """Generate a random drone with a target."""
        drone = self.generate_random_drone()
        target = self.generate_random_target()
        drone.target = target
        if drone not in self.drones:
            drone.scan_for_drones_method = self.get_drones_in_range
            drone.scan_for_targets_method = self.get_targets_in_range
            drone.targets = self.targets
            self.add_drone(drone)
        if target not in self.targets:
            self.add_target(target)

    # world query methods
    def get_drones_in_range(self, position:np.ndarray, range:float) -> list[Drone]:
        """Get all drones within a certain range of a position."""
        return [drone for drone in self.drones if np.linalg.norm(drone.position - position) < range]
    
    def get_targets_in_range(self, position:np.ndarray, range:float) -> list[Target]:
        """Get all targets within a certain range of a position."""
        return [target for target in self.targets if np.linalg.norm(target.position - position) < range]
    
if __name__ == "__main__":
    dpg.create_context()
    dpg.create_viewport(title='World', width=1000, height=800)
    dpg.setup_dearpygui()
    dpg.show_viewport()

    with dpg.window(label="Simulation", width=800, height=800):
        draw_list = dpg.add_drawlist(width=800, height=800)
    
    with dpg.window(label="Controls", width=200, height=800, pos=(800, 0)):
        dpg.add_text("Controls")
        dpg.add_button(label="Add Drone", callback=lambda: world.add_drone(Drone(position=np.random.rand(2) * 700, target=None, scan_for_drones_method=world.get_drones_in_range, scan_for_targets_method=world.get_targets_in_range)))
        dpg.add_button(label="Add Target", callback=lambda: world.add_target(Target(position=np.random.rand(2) * 700, target_type=1, bounds=np.array([[0, 0], world.MAP_SIZE]))))
        dpg.add_button(label="Add Drone with Target", callback=lambda: world.generate_random_drone_with_target())
        dpg.add_button(label="Remove Drone", callback=lambda: world.remove_drone(0) if world.drones else None)
        dpg.add_button(label="Remove Target", callback=lambda: world.remove_target(0) if world.targets else None)

    config = WorldConfiguration(
        map_size=np.array([800, 800]),
        bounds=np.array([[0, 0], [800, 800]]),
        drones=[(200, 200),
                (190, 390),
                (390, 190),
                (390, 390),
                ],
        targets=[None,
                 (200, 100),
                 None,
                 None,
                 (200, 500),
                 None,
                 None,
                 None,
                 None
                ]
        )
    
    world = World( config=config,  bounds=np.array([[0, 0], [800, 800]]), parent=None)

    focusing_drone = world.drones[0]

    while dpg.is_dearpygui_running():
        dpg.delete_item(draw_list, children_only=True)
        world.update()
        world.draw(draw_list)
        world.global_time += world.delta_time
        dpg.draw_text((10, 10), f"Global Time: {world.global_time:.2f}", color=(255, 255, 255), parent=draw_list)

        trigsize = 20
        offset = np.array([0, trigsize/3])
        pos1, pos2, pos3 = focusing_drone.position + np.array([-trigsize, -trigsize]), focusing_drone.position + np.array([trigsize, -trigsize]), focusing_drone.position + np.array([0, trigsize*0.866])
        dpg.draw_triangle(pos1+offset, pos2+offset, pos3+offset,
                          color=(255, 0, 0,100), parent=draw_list)

        for xied in focusing_drone.compute_xi(4):
            xied.draw_notify("simple", draw_list=draw_list)
        
        # delta = {}
        # for key, value in focusing_drone.compute_omega(10).items():
        #     if key is not None:
        #         delta[key] = value
        #         # key.draw_notify("omega", draw_list=draw_list, value=focusing_drone.compute_omega(4)[key])
        # if len(focusing_drone.connections) > 0:
        #     for key, value in focusing_drone.connections[0].compute_omega(8).items():
        #         if key is not None:
        #             key.draw_notify("omega", draw_list=draw_list, value=value - delta.get(key, 0))


        # if len(focusing_drone.connections) > 0:
        #     for key, value in focusing_drone.compute_delta(focusing_drone.connections[0],8).items():
        #         if key is not None:
        #             key.draw_notify("omega", draw_list=draw_list, value=value)
        #     for key, value in focusing_drone.connections[0].compute_delta(focusing_drone, 8).items():
        #         if key is not None:
        #             key.draw_notify("delta", draw_list=draw_list, value=value)
        
        for drone in world.drones:
            drone.draw_notify("omega", draw_list=draw_list, value=f"{drone.count}")
        # focusing_drone.compute_criticality(focusing_drone.connections[0], 8)

        dpg.render_dearpygui_frame()

    dpg.destroy_context()