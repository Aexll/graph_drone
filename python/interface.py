import dearpygui.dearpygui as dpg
import numpy as np
import graphx as gx # type: ignore
import getimg as gi
from spectral import histories_to_shapes_dict_and_transition_history, spectral_decomposition, filter_shapes_dict
from getimg import get_mini_graph_image
from getimg import SKINS
import json
import os


class GraphMaker:
    SAVE_FILE = "saved_builds.json"
    def __init__(self):
        self.nodes = np.empty((0, 2))
        self.targets = np.empty((0, 2))
        self.dist_threshold = 200

        # Spectral Decomposition
        self.spectral_decomposition_n_shapes = 10
        self.spectral_decomposition_num_layers = 5
        self.spectral_decomposition_skin = "default"
        self.spectral_decomposition_max_distance = 8
        # Genetic Optimisation
        self.opti_gen_stepsize = 0.1
        self.opti_gen_steps = 100000
        self.opti_gen_pop = 100
        self.opti_gen_keep_best_ratio = 0.1
        
        # UI
        self.window_tag = "graph_maker_window"
        self.canvas_tag = "graph_maker_canvas"
        self.canvas_width = 700
        self.canvas_height = 700
        self.node_radius = 10
        self.selected_node = None
        self.bg_color = (255, 255, 255, 255)
        self.node_color = (0, 0, 255, 255)
        self.target_color = (255, 0, 0, 255)
        self.edge_color = (100, 100, 100, 255)
        self.selected_target = None
        self.dragging_target = None

        # Visual
        self.font_size = 15
        self.show_numbers = True

        # Builds
        self.build_types = ["capture","genetic", "simple", "history", "parallel", "history_parallel"]
        self.builds = []
        self.selected_builds = set()
        self.load_saved_builds()
        """
        example : 
        builds = [
            {
                "name": "Build 1",
                "type": "capture",
                "data": Object,
                "targets": np.empty((0, 2)),
                "dist_threshold": 200,
                "opti_gen_stepsize": 0.1,
                "opti_gen_steps": 100000,
                "opti_gen_pop": 100,
                "opti_gen_keep_best_ratio": 0.1,
            }
        ]
        """



    def add_node(self, x, y):
        self.nodes = np.concatenate((self.nodes, np.array([[x, y]])))
    
    def add_target(self, x, y):
        self.targets = np.concatenate((self.targets, np.array([[x, y]])))
    
    def get_nodes(self):
        return np.array(self.nodes)
    
    def get_targets(self):
        return np.array(self.targets)
    
    def get_dist_threshold(self):
        return self.dist_threshold
    
    def add_random_targets(self):
        self.add_target(np.random.randint(self.node_radius, self.canvas_width - self.node_radius), np.random.randint(self.node_radius, self.canvas_height - self.node_radius))

    def clear_graph(self):
        self.nodes = []
        self.targets = []
        self.draw_graph()

    def get_target_at(self, x, y):
        for i, target in enumerate(self.targets):
            if np.linalg.norm(target - np.array([x, y])) <= self.node_radius:
                return i
        return None
    
    # save and load builds

    def add_build(self, build_type, data):
        self.builds.append({
            "name": f"Build {build_type} {len(self.builds) + 1}",
            "type": build_type,
            "data": data,
            "targets": self.targets,
            "dist_threshold": self.dist_threshold,
            "opti_gen_stepsize": self.opti_gen_stepsize,
            "opti_gen_steps": self.opti_gen_steps,
            "opti_gen_pop": self.opti_gen_pop,
            "opti_gen_keep_best_ratio": self.opti_gen_keep_best_ratio,
            "spectral_decomposition_n_shapes": self.spectral_decomposition_n_shapes,
        })
        self.draw_builds_table()

    def get_shape_name(self):
        return gx.get_shape_string(gx.get_shape(self.nodes, self.dist_threshold))

    def save_graph_callback(self):
        pass

    def load_graph_callback(self):
        pass

    

    ## Handlers

    def mouse_click_handler(self, sender, app_data):
        mouse_pos = dpg.get_mouse_pos(local=False)
        canvas_pos = dpg.get_item_rect_min(self.canvas_tag)
        x = mouse_pos[0] - canvas_pos[0]
        y = mouse_pos[1] - canvas_pos[1]

        clicked_target = self.get_target_at(x, y)

        if clicked_target is not None:
            self.dragging_target = clicked_target
            self.draw_graph()
        else:
            if 0 <= x <= self.canvas_width and 0 <= y <= self.canvas_height:
                self.add_target(x, y)
                self.dragging_target = len(self.targets) - 1

        self.draw_graph()

    def mouse_drag_handler(self, sender, app_data):
        if self.dragging_target is None:
            return
        
        mouse_pos = dpg.get_mouse_pos(local=False)
        canvas_pos = dpg.get_item_rect_min(self.canvas_tag)
        x = mouse_pos[0] - canvas_pos[0]
        y = mouse_pos[1] - canvas_pos[1]

        # clamp x and y to the canvas
        x = max(self.node_radius, min(x, self.canvas_width-self.node_radius))
        y = max(self.node_radius, min(y, self.canvas_height-self.node_radius))

        self.targets[self.dragging_target] = np.array([x, y])
        self.draw_graph()
    
    def mouse_release_handler(self, sender, app_data):
        self.dragging_target = None
        self.draw_graph()

    def draw_graph(self):
        dpg.delete_item(self.canvas_tag, children_only=True)
        dpg.draw_rectangle(
            parent=self.canvas_tag,
            pmin=[0, 0],
            pmax=[self.canvas_width, self.canvas_height],
            color=self.bg_color,
            fill=self.bg_color
        )
        
        # draw edges
        edges = gx.get_adjacency_matrix(self.nodes, self.dist_threshold)
        for i, row in enumerate(edges):
            for j, edge in enumerate(row):
                if edge == 1:
                    dpg.draw_line(
                        parent=self.canvas_tag,
                        p1=tuple(self.nodes[i]),
                        p2=tuple(self.nodes[j]),
                        color=self.edge_color,
                        thickness=3
                    )
                    
        for i,target in enumerate(self.targets):
            dpg.draw_circle(
                parent=self.canvas_tag,
                center=tuple(target),
                radius=self.node_radius,
                color=self.target_color,
                fill=self.target_color,
            )
            if self.show_numbers:
                font_size = self.node_radius * 1.5
                # Estimate text width as 0.6 * font_size per character
                text_width = 0.6 * font_size * len(str(i))
                text_height = font_size  # Height is roughly the font size
                dpg.draw_text(
                    parent=self.canvas_tag,
                    pos=[target[0] - text_width / 2, target[1] - text_height / 2],
                    color=(255, 255, 255, 255),
                    text=f"{i}",
                    size=font_size
                )

        for i, node in enumerate(self.nodes):
            dpg.draw_circle(
                parent=self.canvas_tag,
                center=tuple(node),
                radius=self.node_radius,
                color=self.node_color,
                fill=self.node_color
            )
            if self.show_numbers:
                font_size = self.node_radius * 1.5
                # Estimate text width as 0.6 * font_size per character
                text_width = 0.6 * font_size * len(str(i))
                text_height = font_size  # Height is roughly the font size
                dpg.draw_text(
                    parent=self.canvas_tag,
                    pos=[node[0] - text_width / 2, node[1] - text_height / 2],
                    color=(255, 255, 255, 255),
                    text=f"{i}",
                    size=font_size
                )

        # show dist threshold line 
        dpg.draw_line(
            parent=self.canvas_tag,
            p1=[10, 10],
            p2=[10 + self.dist_threshold, 10],
            color=(0, 0, 0, 255),
            thickness=3
        )
        dpg.draw_line(
            parent=self.canvas_tag,
            p1=[10 + self.dist_threshold, 5],
            p2=[10 + self.dist_threshold, 15],
            color=(0, 0, 0, 255),
            thickness=3
        )
        dpg.draw_line(
            parent=self.canvas_tag,
            p1=[10, 5],
            p2=[10, 15],
            color=(0, 0, 0, 255),
            thickness=3
        )

        shape_name = self.get_shape_name()

        # write shape name
        dpg.draw_text(
            parent=self.canvas_tag,
            pos=[10 , 20],
            color=(0, 0, 0, 255),
            text=shape_name,
            size=20
        )

        # write error
        if len(self.targets) > 0 and len(self.nodes) == len(self.targets):
            error = gx.cout_graph_p2(self.nodes, self.targets)
            dpg.draw_text(
                parent=self.canvas_tag,
            pos=[10, 40],
            color=(0, 0, 0, 255),
            text=f"Error: {error}",
            size=20
        )

    def clear_targets(self):
        self.targets = np.empty((0, 2))
        self.draw_graph()
    
    def generate_nodes_callback(self):
        if len(self.get_targets()) == 0:
            return
        self.nodes = np.array([[np.mean(self.get_targets()[:, 0]), np.mean(self.get_targets()[:, 1])]] * len(self.get_targets()))
        self.draw_graph()

    
    # Genetic Optimisation

    def set_opti_gen_stepsize(self, sender, app_data):
        self.opti_gen_stepsize = app_data

    def set_opti_gen_steps(self, sender, app_data):
        self.opti_gen_steps = app_data

    def set_opti_gen_pop(self, sender, app_data):
        self.opti_gen_pop = app_data

    def set_opti_gen_keep_best_ratio(self, sender, app_data):
        self.opti_gen_keep_best_ratio = app_data

    def optimise_genetic_nodes_callback(self):
        if len(self.get_targets()) == 0:
            return
        if len(self.nodes) != len(self.get_targets()):
            self.nodes = np.array([[np.mean(self.get_targets()[:, 0]), np.mean(self.get_targets()[:, 1])]] * len(self.get_targets()))
        self.nodes = gx.optimize_nodes_genetic(self.nodes, self.targets, self.dist_threshold, self.opti_gen_stepsize, self.opti_gen_steps, self.opti_gen_pop, self.opti_gen_keep_best_ratio)
        self.draw_graph()



    def set_dist_threshold(self, sender, app_data):
        self.dist_threshold = app_data
        self.draw_graph()

    def set_show_numbers(self, sender, app_data):
        self.show_numbers = app_data
        self.draw_graph()

    # Optimisation Simple

    def optimise_simple_nodes_callback(self):
        if len(self.get_targets()) == 0:
            return
        if len(self.nodes) != len(self.get_targets()):
            self.nodes = np.array([[np.mean(self.get_targets()[:, 0]), np.mean(self.get_targets()[:, 1])]] * len(self.get_targets()))
        self.nodes = gx.optimize_nodes(self.nodes, self.targets, self.dist_threshold, self.opti_gen_stepsize, self.opti_gen_steps, False)
        self.add_build("simple", self.nodes)
        self.draw_graph()

    # Spectral Decomposition

    def spectral_decomposition_callback(self):
        if len(self.get_targets()) == 0:
            return
        if len(self.nodes) != len(self.get_targets()):
            self.nodes = np.array([[np.mean(self.get_targets()[:, 0]), np.mean(self.get_targets()[:, 1])]] * len(self.get_targets()))
        histories = gx.optimize_nodes_history_parallel(self.nodes, self.targets, self.dist_threshold, self.opti_gen_stepsize, self.opti_gen_steps, self.opti_gen_pop, False)
        self.add_build("history_parallel", histories)
        
        # error_min = np.min([gx.cout_graph_p2(histories[i][0], self.targets) for i in range(len(histories))])
        shapes_dict, transition_history = histories_to_shapes_dict_and_transition_history(histories, self.targets, self.dist_threshold)
        
        ## filter to keep only the best n shapes (n = 10)
        shapes_sorted : list[str] = sorted(shapes_dict.keys(), key=lambda x: shapes_dict[x]['score'])
        best_shapes = shapes_sorted[:self.spectral_decomposition_n_shapes]
        shapes_dict = {k: shapes_dict[k] for k in best_shapes}
        shapes_dict, transition_history = filter_shapes_dict(shapes_dict, transition_history, lambda shape_key, info: shape_key in best_shapes)

        spectral_decomposition(
            shapes_dict, 
            transition_history, 
            get_mini_graph_image, 
            self.targets, 
            self.dist_threshold, 
            parent="spectral_window", 
            num_layers=self.spectral_decomposition_num_layers, 
            image_size=100, 
            layer_spacing=120, 
            vertical_spacing=120, 
            skin=self.spectral_decomposition_skin,
            max_distance=self.spectral_decomposition_max_distance
        )

        if not dpg.is_item_shown("spectral_window"):
            dpg.show_item("spectral_window")

        self.add_build("spectral", (shapes_dict, transition_history))


    def set_spectral_decomposition_n_shapes(self, sender, app_data):
        self.spectral_decomposition_n_shapes = app_data

    def set_spectral_decomposition_num_layers(self, sender, app_data):
        self.spectral_decomposition_num_layers = app_data

    def set_spectral_decomposition_skin(self, sender, app_data):
        self.spectral_decomposition_skin = app_data

    def set_spectral_decomposition_max_distance(self, sender, app_data):
        self.spectral_decomposition_max_distance = app_data
        # print("Max distance set to:", self.spectral_decomposition_max_distance)

    # Run

    def run(self):
        dpg.create_context()
        dpg.create_viewport(title="Interface Graphe Classique", width=1200, height=700)
        with dpg.window(label="Graphe Classique", width=700, height=700, tag=self.window_tag):
            dpg.add_separator()
            dpg.add_drawlist(width=self.canvas_width, height=self.canvas_height, tag=self.canvas_tag)
            with dpg.handler_registry():
                dpg.add_mouse_click_handler(callback=self.mouse_click_handler)
                dpg.add_mouse_drag_handler(callback=self.mouse_drag_handler)
                dpg.add_mouse_release_handler(callback=self.mouse_release_handler)
        # Nouvelle fenêtre latérale à droite
        with dpg.window(label="Actions", pos=(700, 0), width=400, height=700, tag="side_panel"):
            dpg.add_button(label="Generate Nodes", callback=self.generate_nodes_callback)
            dpg.add_button(label="Add Random Targets", callback=self.add_random_targets)
            dpg.add_button(label="Clear Targets", callback=self.clear_targets)
            dpg.add_separator()
            dpg.add_text("Options")
            dpg.add_slider_float(label="Dist Threshold", min_value=0.1, max_value=self.canvas_width, default_value=self.dist_threshold, callback=self.set_dist_threshold)
            dpg.add_slider_float(label="Stepsize", min_value=0.01, max_value=1.0, default_value=self.opti_gen_stepsize, callback=self.set_opti_gen_stepsize)
            dpg.add_input_int(label="Steps", min_value=1, max_value=10000000, default_value=self.opti_gen_steps, callback=self.set_opti_gen_steps)
            dpg.add_input_int(label="Population", min_value=1, max_value=500, default_value=self.opti_gen_pop, callback=self.set_opti_gen_pop)
            dpg.add_slider_float(label="Keep Best Ratio", min_value=0.01, max_value=1.0, default_value=self.opti_gen_keep_best_ratio, callback=self.set_opti_gen_keep_best_ratio)
            dpg.add_text("Genetic Optimisation")
            dpg.add_separator()
            dpg.add_button(label="Optimise Genetic", callback=self.optimise_genetic_nodes_callback)
            dpg.add_separator()
            dpg.add_text("Optimisation Simple")
            dpg.add_button(label="Optimise Simple", callback=self.optimise_simple_nodes_callback)
            dpg.add_separator()
            dpg.add_text("Spectral Decomposition")
            dpg.add_input_int(label="Number of Shapes", default_value=self.spectral_decomposition_n_shapes, callback=self.set_spectral_decomposition_n_shapes)
            dpg.add_input_int(label="Number of Layers", default_value=self.spectral_decomposition_num_layers, callback=self.set_spectral_decomposition_num_layers)
            dpg.add_input_int(label="Max Distance", default_value=self.spectral_decomposition_max_distance, callback=self.set_spectral_decomposition_max_distance)
            dpg.add_combo(items=SKINS, default_value=self.spectral_decomposition_skin, label="Skin", callback=self.set_spectral_decomposition_skin)
            dpg.add_button(label="Spectral Decomposition", callback=self.spectral_decomposition_callback)
            dpg.add_separator()
            dpg.add_checkbox(label="Show Numbers", default_value=self.show_numbers, callback=self.set_show_numbers)
            dpg.add_separator()
            dpg.add_button(label="Save Graph", callback=self.save_graph_callback)
            dpg.add_separator()
            dpg.add_text("Load Graph")
            dpg.add_input_text(label="Path", tag="load_graph_path")
            dpg.add_button(label="Load Graph", callback=self.load_graph_callback)
            dpg.add_separator()
            dpg.add_text("Builds : ")
            self.draw_builds_table()
            self.draw_graph()

        with dpg.window(label="Spectral", pos=(0, 0), width=1000, height=700, tag="spectral_window",show=False):
            pass

        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.start_dearpygui()
        dpg.destroy_context()

    def draw_builds_table(self):
        # Supprimer l'ancien tableau s'il existe
        if dpg.does_item_exist("builds_table"):
            dpg.delete_item("builds_table")
        with dpg.table(parent="side_panel", tag="builds_table", header_row=True, borders_innerH=True, borders_outerH=True, borders_innerV=True, borders_outerV=True):
            dpg.add_table_column(label="S", init_width_or_weight=30)
            dpg.add_table_column(label="L", init_width_or_weight=30)
            dpg.add_table_column(label="V", init_width_or_weight=30)
            dpg.add_table_column(label="Nom", init_width_or_weight=150)
            dpg.add_table_column(label="Type", init_width_or_weight=80)
            for i, build in enumerate(self.builds):
                with dpg.table_row():
                    dpg.add_checkbox(
                        label="",
                        default_value=(i in self.selected_builds),
                        callback=self.make_toggle_selection_callback(i)
                    ) 
                    dpg.add_button(label="L", callback=self.make_load_callback(i))
                    dpg.add_button(label="V", callback=self.make_view_spectre_callback(i))
                    dpg.add_text(build["name"])
                    dpg.add_text(build["type"])

    def toggle_build_selection(self, index, value):
        if value:
            self.selected_builds.add(index)
        else:
            self.selected_builds.discard(index)
        self.save_selected_builds()

    def make_toggle_selection_callback(self, index):
        def callback(sender, app_data):
            self.toggle_build_selection(index, app_data)
        return callback

    def make_load_callback(self, idx):
        def callback(sender, app_data):
            self.load_build_button_callback(idx)
        return callback

    def load_build_button_callback(self, index):
        build = self.builds[index]
        # self.nodes = build["data"]["nodes"]
        if build["type"] == "simple":
            self.nodes = build["data"]
        elif build["type"] == "genetic":
            self.nodes = build["data"]
        elif build["type"] == "history":
            self.nodes = build["data"][-1]
        elif build["type"] == "parallel":
            self.nodes = build["data"][-1]
        elif build["type"] == "history_parallel":
            self.nodes = build["data"][0][-1]
        self.targets = build["targets"]
        self.dist_threshold = build["dist_threshold"]
        self.opti_gen_stepsize = build["opti_gen_stepsize"]
        self.opti_gen_steps = build["opti_gen_steps"]
        self.opti_gen_pop = build["opti_gen_pop"]
        self.opti_gen_keep_best_ratio = build["opti_gen_keep_best_ratio"]
        # self.spectral_decomposition_n_shapes = build["spectral_decomposition_n_shapes"]
        # self.spectral_decomposition_num_layers = build["spectral_decomposition_num_layers"]
        # self.spectral_decomposition_skin = build["spectral_decomposition_skin"]
        self.draw_graph()

    def make_view_spectre_callback(self, idx):
        def callback(sender, app_data):
            self.view_spectre_callback(idx)
        return callback

    def view_spectre_callback(self, idx):
        build = self.builds[idx]
        # self.targets = build["targets"]
        self.load_build_button_callback(idx)
        if build["type"] == "spectral":
            shapes_dict, transition_history = build["data"]
            spectral_decomposition(
                shapes_dict, 
                transition_history, 
                get_mini_graph_image, 
                self.targets, 
                self.dist_threshold, 
                parent="spectral_window", 
                num_layers=self.spectral_decomposition_num_layers, 
                image_size=100, 
                layer_spacing=120, 
                vertical_spacing=120, 
                skin=self.spectral_decomposition_skin,
                max_distance=self.spectral_decomposition_max_distance
            )

        if not dpg.is_item_shown("spectral_window"):
            dpg.show_item("spectral_window")

    def to_serializable(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self.to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.to_serializable(x) for x in obj]
        elif isinstance(obj, tuple):
            return [self.to_serializable(x) for x in obj]  # convertit aussi les tuples en listes
        elif isinstance(obj, set):
            return list([self.to_serializable(x) for x in obj])
        else:
            return obj

    def save_selected_builds(self):
        # Sauvegarde les builds sélectionnés dans un fichier JSON
        selected_builds = []
        for idx in self.selected_builds:
            build = self.builds[idx]
            build_serializable = self.to_serializable(build)
            selected_builds.append(build_serializable)
        try:
            with open(self.SAVE_FILE, "w") as f:
                json.dump(selected_builds, f)
        except TypeError as e:
            print("Erreur de sérialisation JSON:", e)
            for b in selected_builds:
                try:
                    json.dumps(b)
                except Exception as ex:
                    print("Objet non sérialisable:", b)
            raise

    def load_saved_builds(self):
        # Charge les builds sauvegardés et les ajoute à la liste
        if os.path.exists(self.SAVE_FILE):
            with open(self.SAVE_FILE, "r") as f:
                try:
                    saved_builds = json.load(f)
                except Exception:
                    saved_builds = []
            for build in saved_builds:
                # Reconversion des listes en np.array si besoin
                build_copy = build.copy()
                if isinstance(build_copy["targets"], list):
                    build_copy["targets"] = np.array(build_copy["targets"])
                if isinstance(build_copy["data"], list):
                    # On tente de deviner la structure
                    if build_copy["type"] in ["simple", "genetic"]:
                        build_copy["data"] = np.array(build_copy["data"])
                    elif build_copy["type"] in ["history", "parallel"]:
                        build_copy["data"] = [np.array(x) for x in build_copy["data"]]
                    elif build_copy["type"] == "history_parallel":
                        build_copy["data"] = [ [np.array(xx) for xx in x] for x in build_copy["data"] ]
                    elif build_copy["type"] == "spectral":
                        formated_data_0 = {}
                        for key, value in build_copy["data"][0].items():
                            new_value = value.copy()
                            new_value["graph"] = np.array(new_value["graph"])
                            formated_data_0[key] = new_value.copy()
                        formated_data_1 = set()
                        for v in build_copy["data"][1]:
                            formated_data_1.add(tuple(v))
                        print(build_copy["targets"])
                        build_copy["data"] = (formated_data_0, formated_data_1)
                self.builds.append(build_copy)
                self.selected_builds.add(len(self.builds)-1)

print(gx.version())
interface = GraphMaker()
interface.run()