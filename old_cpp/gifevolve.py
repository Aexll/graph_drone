"""

génère un gif d'évolution d'un graph, sous forme de mosaïque de graphes

on génère un historique de graphes, on les affiche dans une mosaïque, et on génère un gif


"""
import graphx as gx # type: ignore
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import getimg as gi




def generate_gif(nodes, targets, dist_threshold, mutation_stepsize, steps, ngraphs,save_prefix="results",save_gif=True):

    histories = gx.optimize_nodes_history_parallel(nodes, targets, dist_threshold, mutation_stepsize, steps, ngraphs,False)


