import dearpygui.dearpygui as dpg
import math
import random

class ProximityGraphEditor:
    def __init__(self):
        self.nodes = {}  # {id: {"pos": [x, y], "name": str}}
        self.edges = set()  # {(node1_id, node2_id)}
        self.node_counter = 0
        self.proximity_distance = 100.0
        self.node_radius = 10
        self.dragging_node = None
        self.canvas_width = 800
        self.canvas_height = 600
        self.canvas_tag = "canvas_layer"
        self.window_tag = "main_window"
    
    def distance(self, pos1, pos2):
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def update_edges(self):
        """Met à jour les arêtes basées sur la distance de proximité"""
        self.edges.clear()
        node_ids = list(self.nodes.keys())
        
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                id1, id2 = node_ids[i], node_ids[j]
                pos1, pos2 = self.nodes[id1]["pos"], self.nodes[id2]["pos"]
                
                if self.distance(pos1, pos2) <= self.proximity_distance:
                    self.edges.add((id1, id2))
    
    def add_node(self, x, y, name=None, draw=True):
        """Ajoute un nouveau nœud"""
        if name is None:
            name = f"Node {self.node_counter}"
        
        self.nodes[self.node_counter] = {
            "pos": [x, y],
            "name": name
        }
        self.node_counter += 1
        self.update_edges()
        if draw:
            self.draw_graph()
    
    def get_node_at_position(self, x, y):
        """Trouve le nœud à une position donnée"""
        for node_id, node_data in self.nodes.items():
            node_pos = node_data["pos"]
            if self.distance([x, y], node_pos) <= self.node_radius:
                return node_id
        return None
    
    def mouse_click_handler(self, sender, app_data):
        """Gestionnaire de clic de souris sur le canvas"""
        mouse_pos = dpg.get_mouse_pos(local=False)
        canvas_pos = dpg.get_item_pos(self.canvas_tag)
        
        # Convertir en coordonnées relatives au canvas
        x = mouse_pos[0] - canvas_pos[0]
        y = mouse_pos[1] - canvas_pos[1]
        
        # Vérifier si on clique sur un nœud existant
        clicked_node = self.get_node_at_position(x, y)
        
        if clicked_node is not None:
            # Commencer le drag
            self.dragging_node = clicked_node
        else:
            # Ajouter un nouveau nœud
            if 0 <= x <= self.canvas_width and 0 <= y <= self.canvas_height:
                self.add_node(x, y)
    
    def mouse_drag_handler(self, sender, app_data):
        """Gestionnaire de drag de souris"""
        if self.dragging_node is not None:
            mouse_pos = dpg.get_mouse_pos(local=False)
            canvas_pos = dpg.get_item_pos(self.canvas_tag)
            
            # Nouvelles coordonnées
            x = mouse_pos[0] - canvas_pos[0]
            y = mouse_pos[1] - canvas_pos[1]
            
            # Contraindre dans les limites du canvas
            x = max(self.node_radius, min(self.canvas_width - self.node_radius, x))
            y = max(self.node_radius, min(self.canvas_height - self.node_radius, y))
            
            # Mettre à jour la position du nœud
            self.nodes[self.dragging_node]["pos"] = [x, y]
            self.update_edges()
            self.draw_graph()
    
    def mouse_release_handler(self, sender, app_data):
        """Gestionnaire de relâchement de souris"""
        self.dragging_node = None
    
    def distance_changed_handler(self, sender, app_data, user_data=None):
        """Gestionnaire de changement de distance de proximité"""
        self.proximity_distance = app_data
        self.update_edges()
        self.draw_graph()
    
    def clear_graph(self):
        """Efface tous les nœuds et arêtes"""
        self.nodes.clear()
        self.edges.clear()
        self.node_counter = 0
        self.draw_graph()
    
    def generate_random_nodes(self):
        """Génère des nœuds aléatoires"""
        self.clear_graph()
        for i in range(8):
            x = random.randint(self.node_radius, self.canvas_width - self.node_radius)
            y = random.randint(self.node_radius, self.canvas_height - self.node_radius)
            self.add_node(x, y, draw=False)
        self.update_edges()
        self.draw_graph()
    
    def draw_graph(self):
        """Dessine le graphe sur le canvas"""
        # Effacer le canvas
        if dpg.does_item_exist(self.canvas_tag):
            dpg.delete_item(self.canvas_tag, children_only=True)
        
        # Fond du canvas
        dpg.draw_rectangle(
            parent=self.canvas_tag,
            pmin=[0, 0],
            pmax=[self.canvas_width, self.canvas_height],
            color=(30, 30, 30, 255),
            fill=(30, 30, 30, 255)
        )
        
        # Dessiner les arêtes
        for edge in self.edges:
            node1_pos = self.nodes[edge[0]]["pos"]
            node2_pos = self.nodes[edge[1]]["pos"]
            
            dpg.draw_line(
                parent=self.canvas_tag,
                p1=node1_pos,
                p2=node2_pos,
                color=(100, 100, 255, 255),
                thickness=2
            )
        
        # Dessiner les nœuds
        for node_id, node_data in self.nodes.items():
            pos = node_data["pos"]
            name = node_data["name"]
            
            # Cercle du nœud
            color = (255, 100, 100, 255) if self.dragging_node == node_id else (255, 200, 100, 255)
            dpg.draw_circle(
                parent=self.canvas_tag,
                center=pos,
                radius=self.node_radius,
                color=color,
                fill=color
            )
            
            # Contour
            dpg.draw_circle(
                parent=self.canvas_tag,
                center=pos,
                radius=self.node_radius,
                color=(50, 50, 50, 255),
                thickness=2
            )
            
            # Nom du nœud
            dpg.draw_text(
                parent=self.canvas_tag,
                pos=[pos[0] - len(name) * 3, pos[1] - self.node_radius - 15],
                text=name,
                color=(255, 255, 255, 255),
                size=12
            )
    
    def setup_ui(self):
        """Configure l'interface utilisateur"""
        dpg.destroy_context()  # Nettoyer le contexte avant de créer un nouveau
        dpg.create_context()
        dpg.create_viewport(title="Éditeur de Graphe de Proximité", width=1000, height=700)
        
        with dpg.window(label="Graphe de Proximité", width=1000, height=700, 
                       no_move=True, no_resize=True, no_collapse=True, no_close=True, tag=self.window_tag):
            
            # Panneau de contrôle
            with dpg.group(horizontal=True):
                dpg.add_text("Distance de proximité:")
                dpg.add_slider_float(
                    default_value=self.proximity_distance,
                    min_value=20.0,
                    max_value=200.0,
                    width=200,
                    callback=self.distance_changed_handler
                )
                dpg.add_button(label="Effacer", callback=lambda: self.clear_graph())
                dpg.add_button(label="Nœuds aléatoires", callback=lambda: self.generate_random_nodes())
            
            dpg.add_separator()
            
            # Instructions
            dpg.add_text("Instructions:")
            dpg.add_text("• Cliquez pour ajouter un nœud")
            dpg.add_text("• Glissez-déposez les nœuds pour les déplacer")
            dpg.add_text("• Ajustez la distance pour modifier la connectivité")
            
            dpg.add_separator()
            
            # Canvas pour le graphe
            dpg.add_draw_layer(width=self.canvas_width, height=self.canvas_height, tag=self.canvas_tag)
            
            # Gestionnaires d'événements de souris
            with dpg.handler_registry():
                dpg.add_mouse_click_handler(callback=self.mouse_click_handler)
                dpg.add_mouse_drag_handler(callback=self.mouse_drag_handler)
                dpg.add_mouse_release_handler(callback=self.mouse_release_handler)
        
        dpg.setup_dearpygui()
        dpg.show_viewport()
    
    def run(self):
        """Lance l'application"""
        self.setup_ui()
        
        # Ajouter quelques nœuds de démonstration
        self.add_node(100, 100, "A", draw=False)
        self.add_node(200, 150, "B", draw=False)
        self.add_node(300, 100, "C", draw=False)
        self.add_node(150, 250, "D", draw=False)
        self.update_edges()
        self.draw_graph()
        
        while dpg.is_dearpygui_running():
            dpg.render_dearpygui_frame()
        
        dpg.destroy_context()

# Lancement de l'application
if __name__ == "__main__":
    editor = ProximityGraphEditor()
    editor.run()