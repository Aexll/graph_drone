import dearpygui.dearpygui as dpg
import random
import math
import time
from enum import Enum
from typing import Set, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

class MessageType(Enum):
    LOCK = "LOCK"
    UNLOCK = "UNLOCK"
    BREAK_LINK = "BREAK_LINK"

class NodeState(Enum):
    FREE = "FREE"
    LOCKED = "LOCKED"
    # BREAK = "BREAK"

@dataclass
class Message:
    type: MessageType
    sender: int
    destination: int
    edge: Tuple[int, int]
    id: int
    
    def __hash__(self):
        return hash((self.type, self.sender, self.id, self.edge))

    def __str__(self):
        return f"[{self.sender} -> {self.destination}: {'LOCK' if self.type == MessageType.LOCK else 'UNLOCK'} ({self.id})]"

    def __repr__(self):
        return self.__str__()

class Node:

    def __init__(self, id: int, position: Optional[Tuple[float, float]] = None):
        self.id: int = id
        self.neighbors: Set[int] = set()
        self.position: Tuple[float, float] = position if position else (random.uniform(0, 800), random.uniform(0, 600))
        self.stored_edge: Optional[Tuple[int, int]] = None
        self.stored_id: Optional[int] = None
        self.state: NodeState = NodeState.FREE
        self.unlock_set: Set[int] = set()
        self.pending_messages: Set[Message] = set()
        self.received_messages: Set[Message] = set()
        self.position: Tuple[float, float] = (0.0, 0.0)


    def send_message(self, message: Message):
        self.pending_messages.add(message)

    def receive_message(self, message: Message):
        self.received_messages.add(message)

    def add_neighbor(self, neighbor_id: int):
        if neighbor_id not in self.neighbors:
            self.neighbors.add(neighbor_id)

    def remove_neighbor(self, neighbor_id: int, send=True):
        if neighbor_id in self.neighbors:
            self.neighbors.remove(neighbor_id)

            if send:
                self.send_message(Message(
                    type=MessageType.BREAK_LINK,
                sender=self.id,
                destination=neighbor_id,
                id=self.stored_id if self.stored_id else 0,
                edge=self.stored_edge if self.stored_edge else (0, 0)
                ))
        # print(f"Node {self.id} removed neighbor {neighbor_id}. Neighbors now: {self.neighbors}")


    def clear_messages(self):
        self.pending_messages.clear()
        self.received_messages.clear()

    def process(self):
        node = self
        """Traite les messages pour un noeud selon l'algorithme"""
        # Séparer les messages par type
        lock_messages = {msg for msg in node.received_messages if msg.type == MessageType.LOCK}
        unlock_messages = {msg for msg in node.received_messages if msg.type == MessageType.UNLOCK}
        break_messages = {msg for msg in node.received_messages if msg.type == MessageType.BREAK_LINK}

        # print(unlock_messages, lock_messages)

        # Trouver le message de blocage maximum (par id)
        max_lock_msg = None
        if lock_messages:
            max_lock_msg = max(lock_messages, key=lambda m: m.id)

        # # On retire les voisins qui veulent casser l'arête
        # for msg in break_messages:
        #     self.remove_neighbor(msg.sender, send=False)
             
        # Traitement BLOCKED
        if node.state == NodeState.LOCKED:
       
            for msg in unlock_messages:
                if (msg.edge == node.stored_edge and 
                    msg.sender in node.unlock_set):
                    node.unlock_set.remove(msg.sender)
                    
                    if not node.unlock_set:
                        node.state = NodeState.FREE
                        # Envoyer UNLOCK à tous les voisins sauf l'expéditeur original
                        for neighbor_id in node.neighbors:
                            if neighbor_id != msg.sender:
                                node.pending_messages.add(Message(
                                    type=MessageType.UNLOCK,
                                    sender=node.id,
                                    destination=neighbor_id,
                                    id=node.stored_id,
                                    edge=node.stored_edge
                                ))
                        
        # Renvoyer UNLOCK à tout ceux qui ont envoyé LOCK excepter ceux qui ont envoyé un UNLOCK
        if node.state == NodeState.LOCKED:
            for msg in lock_messages:
                self.send_message(Message(
                    type=MessageType.UNLOCK,
                    destination=msg.sender,
                    sender=self.id,
                    id=msg.id,
                    edge=msg.edge
                ))
        
            

        # Traitement selon l'état du noeud
        if node.state == NodeState.FREE:
            if lock_messages:

                # Si le noeud ayant recus le message appartient à l'edge du message on peut casser l'arête
                if max_lock_msg.edge[0] == node.id or max_lock_msg.edge[1] == node.id:
                    nei = max_lock_msg.edge[1] if max_lock_msg.edge[0] == node.id else max_lock_msg.edge[0]
                    self.send_message(Message(
                        type=MessageType.BREAK_LINK,
                        sender=node.id,
                        destination=nei,
                        id=max_lock_msg.id,
                        edge=max_lock_msg.edge
                    ))

                node.state = NodeState.LOCKED
                node.stored_edge = max_lock_msg.edge
                node.stored_id = max_lock_msg.id
                node.unlock_set = set()

                # renvoyer UNLOCK à tous les noeuds ayant envoyé LOCK mais pas l'expéditeur du message max_lock_msg
                sent_unlock_nei = set()
                for msg in lock_messages:
                    if msg.sender != max_lock_msg.sender:
                        node.send_message(Message(
                            type=MessageType.UNLOCK,
                            sender=node.id,
                            destination=msg.sender,
                            id=msg.id,
                            edge=msg.edge
                        ))
                        if msg.id == max_lock_msg.id:
                            sent_unlock_nei.add(msg.sender)
                
                # Envoyer le message à tous les voisins sauf aux expéditeur de locks
                for neighbor_id in node.neighbors:
                    if neighbor_id != max_lock_msg.sender and neighbor_id not in sent_unlock_nei:
                        node.unlock_set.add(neighbor_id)
                        node.send_message(Message(
                            type=MessageType.LOCK,
                            sender=node.id,
                            destination=neighbor_id,
                            id=max_lock_msg.id,
                            edge=max_lock_msg.edge
                        ))
           
        pass

    def try_remove_edge(self, other_node_id: int):
        """Essaie de supprimer une arête avec un autre noeud"""
        if not other_node_id in self.neighbors:
            print(f"Node {self.id} cannot remove edge with {other_node_id} because it is not a neighbor.")
            return
        
        edge = (self.id, other_node_id)
        self.stored_edge = edge
        self.stored_id = other_node_id + self.id**10
        self.state = NodeState.LOCKED
        for nei in self.neighbors:
            if nei != other_node_id:
                self.unlock_set.add(nei)
                self.pending_messages.add(Message(
                    type=MessageType.LOCK,
                    sender=self.id,
                    destination=nei,
                    id=self.stored_id,
                    edge=self.stored_edge
                )) 


class Graph:

    nodeCount = 0


    def __init__(self):
        self.nodes: Dict[int, Node] = {}
        self.edges: Set[Tuple[int, int]] = set()
    
    def add_node(self, position: Optional[Tuple[float, float]] = None):
        if Graph.nodeCount not in self.nodes:
            self.nodes[Graph.nodeCount] = Node(Graph.nodeCount, position)
            Graph.nodeCount += 1

    def add_edge(self, node1_id: int, node2_id: int):
        if node1_id in self.nodes and node2_id in self.nodes:
            edge = (node1_id, node2_id)
            self.edges.add(edge)
            self.nodes[node1_id].add_neighbor(node2_id)
            self.nodes[node2_id].add_neighbor(node1_id)
    
    def remove_edge(self, node1_id: int, node2_id: int):
        edge = (node1_id, node2_id)
        edge_reverse = (node2_id, node1_id)
        if edge in self.edges or edge_reverse in self.edges:
            self.edges.discard(edge)
            self.edges.discard(edge_reverse)  # Remove reverse edge if it exists
            self.nodes[node1_id].remove_neighbor(node2_id, send=False)
            self.nodes[node2_id].remove_neighbor(node1_id, send=False)

    def gather_messages(self):
        """Collecte les messages de tous les noeuds"""
        pending_messages = set()
        for node in self.nodes.values():
            pending_messages.update(node.pending_messages)
        return pending_messages
    
    def clear_messages(self):
        """Efface les messages de tous les noeuds"""
        for node in self.nodes.values():
            node.clear_messages()

    def distribute_messages(self):
        pending_messages = self.gather_messages()
        self.clear_messages()  # Clear messages after gathering
        for msg in pending_messages:
            if msg.destination in self.nodes:
                self.nodes[msg.destination].received_messages.add(msg)
            if msg.type == MessageType.BREAK_LINK:
                # If the message is a BREAK_LINK, we remove the neighbor
                self.remove_edge(msg.sender, msg.destination)
                # print(f"Node {msg.sender} broke link with {msg.destination}")

    def update(self):
        self.distribute_messages()
        for node in self.nodes.values():
            node.process()


def get_color_from_edge(edge: Optional[Tuple[int, int]]) -> Tuple[int, int, int]:
    """Retourne une couleur basée sur l'ID de l'arête"""
    if edge is None:
        return (255, 255, 255)  # Couleur par défaut si l'arête est None
    r = (edge[0] * 123 + edge[1] * 456) % 256
    g = (edge[0] * 789 + edge[1] * 101) % 256
    b = (edge[0] * 112 + edge[1] * 131) % 256
    return (r, g, b)
    

def draw_msg_arrow(msg: Message, drawlist_tag="graph_drawlist", offset=0):
    from_node = graph.nodes[msg.sender]
    to_node = graph.nodes[msg.destination]
    msg_from = np.array(from_node.position)
    msg_to = np.array(to_node.position)
    direction = msg_to - msg_from
    if np.linalg.norm(direction) != 0:
        direction = direction / np.linalg.norm(direction)
        direction_left = np.array([-direction[1], direction[0]])
        msg_from = msg_from + direction * (NODE_RADIUS * 2)
        msg_to = msg_to - direction * (NODE_RADIUS * 2)
        msg_from += direction_left * offset
        msg_to += direction_left * offset
    # colors = {
    #     # MessageType.LOCK: (255, 0, 0, 255),
    #     MessageType.LOCK: get_color_from_edge(msg.edge),
    #     MessageType.UNLOCK: (0, 255, 0, 255),
    #     MessageType.BREAK_LINK: (255, 165, 0, 255)
    # }
    color = get_color_from_edge(msg.edge)
    if msg.type == MessageType.LOCK:
        dpg.draw_arrow(
            msg_to,
            msg_from,
            color=color,
            thickness=3,
            parent=drawlist_tag
        )
    if msg.type == MessageType.UNLOCK:
        dpg.draw_arrow(
            msg_to,
            msg_from,
            color=color,
            thickness=1,
            parent=drawlist_tag
        )

NODE_RADIUS = 10

def draw_graph(drawlist_tag="graph_drawlist"):   


    dpg.delete_item(drawlist_tag, children_only=True)
    graph_edges_unique = set()
    for edge in graph.edges:
        if edge[0] > edge[1]:
            graph_edges_unique.add((edge[1], edge[0]))
        else:
            graph_edges_unique.add(edge)

    for edge in graph_edges_unique:
        node1 = graph.nodes[edge[0]]
        node2 = graph.nodes[edge[1]]
        msg_from = np.array(node1.position)
        msg_to = np.array(node2.position)
        direction = msg_to - msg_from
        if np.linalg.norm(direction) != 0:
            direction = direction / np.linalg.norm(direction)
            msg_from = msg_from + direction * (NODE_RADIUS * 2)
            msg_to = msg_to - direction * (NODE_RADIUS * 2)
        if hovered_edge and hovered_edge == edge:
            dpg.draw_line(msg_from, msg_to, color=(255, 0, 0, 255), thickness=4, parent=drawlist_tag)
        else:
            dpg.draw_line(msg_from, msg_to, color=(100,100,100,255), thickness=1, parent=drawlist_tag)

    for node_id, node in graph.nodes.items():
        x, y = node.position
        # print(f"Node {node_id} at position ({x}, {y})")
        colors = {
            NodeState.FREE: (0, 255, 0),
            # NodeState.LOCKED: (255, 0, 0),
            NodeState.LOCKED: get_color_from_edge(node.stored_edge),
            # NodeState.BREAK: (255, 165, 0)
        }
        selected_color = colors.get(node.state, (255, 255, 255))
        dpg.draw_circle((x, y), NODE_RADIUS, color=selected_color, fill=list(selected_color)+[100], parent=drawlist_tag)
        dpg.draw_text((x - 5, y - 5), str(node_id), color=(255, 255, 255, 255), parent=drawlist_tag)
        for nei in node.neighbors:
            neighbor_node = graph.nodes[nei]
            if neighbor_node.position != node.position:
                vector =  np.array(neighbor_node.position) - np.array(node.position) 
                direction = vector / np.linalg.norm(vector) if np.linalg.norm(vector) != 0 else (0, 0)
                drawpos = np.array(node.position) + direction * (NODE_RADIUS + 5)
                selected_color = (get_color_from_edge(node.stored_edge) if nei in node.unlock_set else (0,255,0, 255))
                dpg.draw_circle(drawpos, NODE_RADIUS/3, color=selected_color, fill= selected_color, parent=drawlist_tag)

    messages = graph.gather_messages()

    # msg_to_nodes = {node: {msg for msg in messages if msg.destination == node.id} for node in graph.nodes.values()}
    msg_on_edge = {}
    for msg in messages:
        edge = tuple(sorted([msg.sender, msg.destination]))
        if edge not in msg_on_edge:
            msg_on_edge[edge] = []
        msg_on_edge[edge].append(msg)

    # print(f"Messages on edges: {msg_on_edge}")
    for edge, msgs in msg_on_edge.items():
        # node_msg_count = len(msg_to_nodes.get(node, []))
        for i, msg in enumerate(msgs):
            # print(f"Node {node.id} has message {msg.type} from {msg.sender} to {msg.destination} with edge {msg.edge}")
            draw_msg_arrow(msg, drawlist_tag=drawlist_tag, offset=(i*5))

    deleted_edges = set()
    for msg in messages:
        deleted_edges.add(msg.edge)

    for edge in deleted_edges:
        node1 = graph.nodes[edge[0]]
        node2 = graph.nodes[edge[1]]
        msg_from = np.array(node1.position)
        msg_to = np.array(node2.position)
        # direction = msg_to - msg_from
        midpoint = (msg_from + msg_to) / 2
        # draw a cross at the midpoint of the edge
        CROSS_SIZE = 7
        CROSS_THICKNESS = 4
        color = get_color_from_edge(edge)
        dpg.draw_polyline(
            points=[(midpoint[0] - CROSS_SIZE, midpoint[1] - CROSS_SIZE), (midpoint[0] + CROSS_SIZE, midpoint[1] + CROSS_SIZE)],
            color=color,
            thickness=CROSS_THICKNESS,
            parent=drawlist_tag
        )
        dpg.draw_polyline(
            points=[(midpoint[0] - CROSS_SIZE, midpoint[1] + CROSS_SIZE), (midpoint[0] + CROSS_SIZE, midpoint[1] - CROSS_SIZE)],
            color=color,
            thickness=CROSS_THICKNESS,
            parent=drawlist_tag
        )
            # if np.linalg.norm(direction) != 0:
            #     direction = direction / np.linalg.norm(direction)
            #     msg_from = msg_from + direction * (NODE_RADIUS * 2)
            #     msg_to = msg_to - direction * (NODE_RADIUS * 2)
            # dpg.draw_line(msg_from, msg_to, color=(100, 100, 100, 255), thickness=1, parent=drawlist_tag)

    # for msg in messages:
    #     from_node = graph.nodes[msg.sender]
    #     to_node = graph.nodes[msg.destination]
    #     msg_from = np.array(from_node.position)
    #     msg_to = np.array(to_node.position)
    #     direction = msg_to - msg_from
    #     if np.linalg.norm(direction) != 0:
    #         direction = direction / np.linalg.norm(direction)
    #         msg_from = msg_from + direction * (NODE_RADIUS * 2)
    #         msg_to = msg_to - direction * (NODE_RADIUS * 2)
    #     colors = {
    #         # MessageType.LOCK: (255, 0, 0, 255),
    #         MessageType.LOCK: get_color_from_edge(msg.edge),
    #         MessageType.UNLOCK: (0, 255, 0, 255),
    #         MessageType.BREAK_LINK: (255, 165, 0, 255)
    #     }
    #     dpg.draw_arrow(
    #         msg_to,
    #         msg_from,
    #         color=colors.get(msg.type, (255, 255, 255, 255)),
    #         thickness=1,
    #         parent=drawlist_tag
    #     )

    

def _random_break(graph: Graph):
    node = random.choice(list(graph.nodes.values()))
    if node.state == NodeState.FREE:
        node.try_remove_edge(random.choice(list(node.neighbors)))
        print(f"Node {node.id} is trying to break an edge with a neighbor.")

def _reset():
    global graph
    for node in graph.nodes.values():
        node.clear_messages()
        node.state = NodeState.FREE
        node.stored_edge = None
        node.stored_id = None
        node.unlock_set.clear()
        node.pending_messages.clear()
        node.received_messages.clear()



def get_edge_at_position(position: Tuple[float, float], graph: Graph, threshold: float = 15.0) -> Optional[Tuple[int, int]]:
    """Retourne l'arête la plus proche d'une position donnée, si elle est sous le seuil de distance"""
    closest_edge = None
    closest_distance = float('inf')
    
    # Créer un ensemble d'arêtes uniques pour éviter les doublons
    graph_edges_unique = set()
    for edge in graph.edges:
        # Normaliser l'arête pour éviter les doublons (plus petit ID en premier)
        normalized_edge = (min(edge[0], edge[1]), max(edge[0], edge[1]))
        graph_edges_unique.add(normalized_edge)
    
    for edge in graph_edges_unique:
        node1 = graph.nodes[edge[0]]
        node2 = graph.nodes[edge[1]]
        edge_vec = np.array(node2.position) - np.array(node1.position)
        edge_len = np.linalg.norm(edge_vec)
        if edge_len == 0:
            continue
        edge_dir = edge_vec / edge_len
        pos_vec = np.array(position) - np.array(node1.position)
        proj = np.dot(pos_vec, edge_dir)
        proj = max(0, min(edge_len, proj))
        closest_point = np.array(node1.position) + proj * edge_dir
        dist_to_edge = np.linalg.norm(np.array(position) - closest_point)
        
        # Debug output
        # print(f"Edge {edge}: distance to position {position} is {dist_to_edge}")

        if dist_to_edge < closest_distance:
            closest_distance = dist_to_edge
            closest_edge = edge

    if closest_distance < threshold:
        return closest_edge
    else:
        return None
    
def get_closest_node(position: Tuple[float, float], graph: Graph, threshold: float = 15.0) -> Optional[Node]:
    """Retourne le noeud le plus proche d'une position donnée, si elle est sous le seuil de distance"""
    closest_node = None
    closest_distance = float('inf')
    
    for node in graph.nodes.values():
        dist = np.linalg.norm(np.array(node.position) - np.array(position))
        if dist < closest_distance:
            closest_distance = dist
            closest_node = node

    if closest_distance < threshold:
        return closest_node
    else:
        return None

# Mouse click handler
def mouse_click_handler(sender, app_data):
    mouse_pos = dpg.get_mouse_pos()

    global graph
    global hovered_edge

    node = get_closest_node(mouse_pos, graph, threshold=NODE_RADIUS * 8)
    edge = get_edge_at_position(mouse_pos, graph, threshold=NODE_RADIUS * 2)

    if edge :
        if node:
            node.try_remove_edge(edge[1] if edge[0] == node.id else edge[0])
        else:
            # If no node is clicked, we can still break the edge
            node1 = graph.nodes[edge[0]]
            node2 = graph.nodes[edge[1]]
            node1.try_remove_edge(node2.id)





    # for node in graph.nodes.values():
    #     if (mouse_pos[0] - node.position[0]) ** 2 + (mouse_pos[1] - node.position[1]) ** 2 < NODE_SPACING ** 2:
    #         print(f"Clicked on Node {node.id} at position {node.position}")
    #         print(f"State: {node.state.value}")
    #         # # Toggle the state of the node
    #         # if node.state == NodeState.FREE:
    #         #     node.state = NodeState.LOCKED
    #         #     print(f"Node {node.id} is now LOCKED.")
    #         # elif node.state == NodeState.LOCKED:
    #         #     node.state = NodeState.FREE
    #         #     print(f"Node {node.id} is now FREE.")
    #         # break
        
        # clicking of edges
        # mouse_pos = np.array(mouse_pos)
        # for nei in node.neighbors:
        #     neighbor_node = graph.nodes[nei]
        #     nei_pos = np.array(neighbor_node.position)
        #     node_pos = np.array(node.position)
        #     # Compute distance from mouse to edge (node_pos <-> nei_pos)
        #     edge_vec = nei_pos - node_pos
        #     edge_len = np.linalg.norm(edge_vec)
        #     if edge_len == 0:
        #         continue
        #     edge_dir = edge_vec / edge_len
        #     mouse_vec = mouse_pos - node_pos
        #     proj = np.dot(mouse_vec, edge_dir)
        #     # Clamp projection to edge segment
        #     proj = max(0, min(edge_len, proj))
        #     closest_point = node_pos + proj * edge_dir
        #     dist_to_edge = np.linalg.norm(mouse_pos - closest_point)
        #     EDGE_CLICK_THRESHOLD = 15  # pixels
        #     if dist_to_edge < EDGE_CLICK_THRESHOLD:
        #         print(f"Clicked on edge between Node {node.id} and Node {nei} at position {mouse_pos}")
        #         # Toggle the state of the edge
        #         if node.state == NodeState.FREE:
        #             if nei > node.id:
        #                 node.try_remove_edge(nei)
        #             # node.try_remove_edge(nei)
        #             print(f"Node {node.id} is trying to break an edge with neighbor {nei}.")
        #         # elif node.state == NodeState.LOCKED:
        #         #     node.remove_neighbor(nei)
        #         #     print(f"Node {node.id} is removing edge with neighbor {nei}.")
        #         # break



def mouse_move_handler(sender, app_data):
    mouse_pos = dpg.get_mouse_pos()
    global hovered_edge
    hovered_edge = get_edge_at_position(mouse_pos, graph, threshold=30)
    # if edge:
        # print(f"Mouse is over edge between Node {edge[0]} and Node {edge[1]} at position {mouse_pos}")
        # dpg.draw_line(
        #     (mouse_pos[0] - 10, mouse_pos[1]),
        #     (mouse_pos[0] + 10, mouse_pos[1]),
        #     color=(255, 0, 0, 255),
        #     thickness=10,
        #     parent="graph_drawlist"
        # )



hovered_edge:Optional[Tuple[int, int]] = None
hovered_node:Optional[Node] = None



DRAW_WIDTH = 800
DRAW_HEIGHT = 600
SPAWN_OFFSET = 100
CONNEXION_RADIUS = 100
NODE_SPACING = 100
NODE_COUNT = 15
NODE_PLACEMENT_CIRCLE = True
NODE_PLACEMENT_CIRCLE_RADIUS = 200  # Radius for circular placement of nodes

if __name__ == "__main__":
    graph = Graph()

    nodecount = NODE_COUNT
    for i in range(nodecount):
        graph.add_node()

        if NODE_PLACEMENT_CIRCLE:
            angle = 2 * np.pi * i / nodecount
            newnode_pos = (DRAW_WIDTH // 2 + int(np.cos(angle) * NODE_PLACEMENT_CIRCLE_RADIUS),
                           DRAW_HEIGHT // 2 + int(np.sin(angle) * NODE_PLACEMENT_CIRCLE_RADIUS))
        else:
            newnode_pos = (random.uniform(SPAWN_OFFSET, DRAW_WIDTH - SPAWN_OFFSET), random.uniform(SPAWN_OFFSET, DRAW_HEIGHT - SPAWN_OFFSET))
            while any(
                (newnode_pos[0] - n.position[0]) ** 2 + (newnode_pos[1] - n.position[1]) ** 2 < NODE_SPACING ** 2
                for n in graph.nodes.values()
            ):
                newnode_pos = (random.uniform(SPAWN_OFFSET, DRAW_WIDTH - SPAWN_OFFSET), random.uniform(SPAWN_OFFSET, DRAW_HEIGHT - SPAWN_OFFSET))


        graph.nodes[i].position = newnode_pos

    for n in graph.nodes.values():
        for m in graph.nodes.values():
            if (n.position[0] - m.position[0]) ** 2 + (n.position[1] - m.position[1]) ** 2 < CONNEXION_RADIUS ** 2:
                graph.add_edge(n.id, m.id)





    dpg.create_context()
    dpg.create_viewport(title='Graph Visualization', width=1200, height=600)
    dpg.setup_dearpygui()
    with dpg.window(label="Graph Window", width=800, height=600):
        with dpg.drawlist(width=800, height=600, tag="graph_drawlist"):
            pass
    
    with dpg.window(label="Graph Nodes", width=400, height=600, pos=(800, 0)):
        dpg.add_button(label="Break", callback=lambda: _random_break(graph))
        dpg.add_button(label="Tick", callback=lambda: graph.update())
        dpg.add_button(label="Reset", callback=lambda: _reset())
        for node_id, node in graph.nodes.items():
            dpg.add_text(f"Node {node_id}", tag=f"node_{node_id}")
            with dpg.collapsing_header(label=f"Node {node_id} Details"):
                dpg.add_text(f"Position: {node.position}")
                dpg.add_text(f"State: {node.state.value}")
                dpg.add_text(f"Neighbors: {', '.join(map(str, node.neighbors))}")
                dpg.add_text(f"Stored Edge: {node.stored_edge}")
                dpg.add_text(f"Stored ID: {node.stored_id}")
                dpg.add_text(f"Unlock Set: {', '.join(map(str, node.unlock_set))}")
                dpg.add_text(f"Pending Messages: {len(node.pending_messages)}")
                dpg.add_text(f"Received Messages: {len(node.received_messages)}")
            dpg.add_separator()

    with dpg.handler_registry():
        dpg.add_mouse_click_handler(callback=mouse_click_handler)
        dpg.add_mouse_move_handler(callback=mouse_move_handler)

    dpg.show_viewport()
    while dpg.is_dearpygui_running():
        draw_graph()
        # graph.update()
        dpg.render_dearpygui_frame()
        time.sleep(0.01)
    dpg.destroy_context()
