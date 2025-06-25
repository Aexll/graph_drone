#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <random>



double raw_distance(double* a, double* b, size_t dim) {
    double dist = 0;
    for (size_t i = 0; i < dim; ++i) {
        dist += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return std::sqrt(dist);
}


double raw_cout_graph_p2(double* nodes, double* targets, size_t n, size_t m) {
    double cout = 0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            cout += raw_distance(nodes + i * 2, targets + j * 2, 2);
        }
    }
    return cout;
}

bool raw_is_connected(double* a, double* b, double dist_threshold) {
    double dx = a[0] - b[0];
    double dy = a[1] - b[1];
    return dx * dx + dy * dy < dist_threshold * dist_threshold;
}

bool raw_is_graph_connected_bfs(double* nodes, size_t n, double dist_threshold) {
    std::vector<bool> visited(n, false);
    std::vector<size_t> queue;
    queue.push_back(0);
    visited[0] = true;

    size_t visited_count = 1;
    while (!queue.empty()) {
        size_t current = queue.back();
        queue.pop_back();
        for (size_t i = 0; i < n; ++i) {
            if (!visited[i]) {
                if (raw_is_connected(nodes + current * 2, nodes + i * 2, dist_threshold)) {
                    visited[i] = true;
                    queue.push_back(i);
                    ++visited_count;
                }
            }
        }
    }
    return visited_count == n;
}


double* raw_mutate_nodes(double* nodes, size_t n, double stepsize) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0.0, stepsize);
    for (size_t i = 0; i < n * 2; ++i) {
        nodes[i] += d(gen);
    }
    return nodes;
}










// distance entre deux points
double distance(pybind11::array_t<double> a, pybind11::array_t<double> b) {
    pybind11::buffer_info a_info = a.request();
    pybind11::buffer_info b_info = b.request();

    // Check that both arrays are 1D and of size 2
    if (a_info.ndim != 1 || b_info.ndim != 1) {
        throw std::runtime_error("Input arrays must be 1-dimensional");
    }
    if (a_info.size != 2 || b_info.size != 2) {
        throw std::runtime_error("Input arrays must have size 2");
    }

    // Cast pointers to double*
    double* a_ptr = static_cast<double*>(a_info.ptr);
    double* b_ptr = static_cast<double*>(b_info.ptr);

    double dx = a_ptr[0] - b_ptr[0];
    double dy = a_ptr[1] - b_ptr[1];
    return std::sqrt(dx * dx + dy * dy);
}


bool is_connected(pybind11::array_t<double> a, pybind11::array_t<double> b, double dist_threshold) {
    // ne pas utiliser distance, car on ne veut pas calculer faire de racine carrée
    pybind11::buffer_info a_info = a.request();
    pybind11::buffer_info b_info = b.request();

    // Check that both arrays are 1D and of size 2
    if (a_info.ndim != 1 || b_info.ndim != 1) {
        throw std::runtime_error("Input arrays must be 1-dimensional");
    }
    if (a_info.size != 2 || b_info.size != 2) {
        throw std::runtime_error("Input arrays must have size 2");
    }

    double* a_ptr = static_cast<double*>(a_info.ptr);
    double* b_ptr = static_cast<double*>(b_info.ptr);

    double dx = a_ptr[0] - b_ptr[0];
    double dy = a_ptr[1] - b_ptr[1];
    return (dx * dx + dy * dy) < (dist_threshold * dist_threshold);
}

/*
L'algorithme is_graph_connected_bfs utilise une recherche en largeur (BFS) 
pour vérifier si tous les noeuds d'un graphe 
(défini par des positions de points et un seuil de distance) sont accessibles à partir du premier noeud. 
On marque les noeuds visités en partant du noeud 0, 
et on ajoute à la file tous les voisins (points à distance inférieure au seuil) non encore visités. 
Si à la fin tous les noeuds ont été visités, (visited[i] == true pour tout i)
le graphe est connexe.
*/
bool is_graph_connected_bfs(pybind11::array_t<double> nodes, double dist_threshold) {
    // nodes: shape (n, 2)
    pybind11::buffer_info nodes_info = nodes.request();
    if (nodes_info.ndim != 2 || nodes_info.shape[1] != 2) {
        throw std::runtime_error("Input nodes must be a 2D array with shape (n, 2)");
    }
    size_t n = nodes_info.shape[0];
    double* nodes_ptr = static_cast<double*>(nodes_info.ptr);

    // Visited array
    std::vector<bool> visited(n, false);
    std::vector<size_t> queue;
    queue.push_back(0);
    visited[0] = true;

    while (!queue.empty()) {
        size_t current = queue.front();
        queue.erase(queue.begin());

        double* current_ptr = nodes_ptr + current * 2;
        for (size_t i = 0; i < n; ++i) {
            if (!visited[i]) {
                double* other_ptr = nodes_ptr + i * 2;
                double dx = current_ptr[0] - other_ptr[0];
                double dy = current_ptr[1] - other_ptr[1];
                double dist2 = dx * dx + dy * dy;
                if (dist2 < dist_threshold * dist_threshold) {
                    visited[i] = true;
                    queue.push_back(i);
                }
            }
        }
    }

    // Check if all nodes are visited
    for (size_t i = 0; i < n; ++i) {
        if (!visited[i]) return false;
    }
    return true;
}





/*
fonction de cout
racines carrée de la somme des distances entre tous les noeuds et tous les targets
*/
double cout_graph_p2(pybind11::array_t<double> nodes, pybind11::array_t<double> targets) {
    // nodes: shape (n, 2)
    pybind11::buffer_info nodes_info = nodes.request();
    if (nodes_info.ndim != 2 || nodes_info.shape[1] != 2) {
        throw std::runtime_error("Input nodes must be a 2D array with shape (n, 2)");
    }

    pybind11::buffer_info targets_info = targets.request();
    if (targets_info.ndim != 2 || targets_info.shape[1] != 2) {
        throw std::runtime_error("Input targets must be a 2D array with shape (m, 2)");
    }

    size_t n = nodes_info.shape[0];
    double* nodes_ptr = static_cast<double*>(nodes_info.ptr);
    double* targets_ptr = static_cast<double*>(targets_info.ptr);

    // return raw_cout_graph_p2(nodes_ptr, targets_ptr, n, m);

    // on calcule la distance entre les noeuds et les targets
    double cout = 0;
    for (size_t i = 0; i < n; ++i) {
        double dx = nodes_ptr[i * 2] - targets_ptr[i * 2];
        double dy = nodes_ptr[i * 2 + 1] - targets_ptr[i * 2 + 1];
        cout += std::abs(dx * dx + dy * dy);
    }
    return std::sqrt(cout);
}


/* COUT TOTAL
double cout_graph_p2(pybind11::array_t<double> nodes, pybind11::array_t<double> targets) {
    // nodes: shape (n, 2)
    pybind11::buffer_info nodes_info = nodes.request();
    if (nodes_info.ndim != 2 || nodes_info.shape[1] != 2) {
        throw std::runtime_error("Input nodes must be a 2D array with shape (n, 2)");
    }

    pybind11::buffer_info targets_info = targets.request();
    if (targets_info.ndim != 2 || targets_info.shape[1] != 2) {
        throw std::runtime_error("Input targets must be a 2D array with shape (m, 2)");
    }

    size_t n = nodes_info.shape[0];
    size_t m = targets_info.shape[0];
    double* nodes_ptr = static_cast<double*>(nodes_info.ptr);
    double* targets_ptr = static_cast<double*>(targets_info.ptr);

    // return raw_cout_graph_p2(nodes_ptr, targets_ptr, n, m);

    // on calcule la distance entre les noeuds et les targets
    double cout = 0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            double dx = nodes_ptr[i * 2] - targets_ptr[j * 2];
            double dy = nodes_ptr[i * 2 + 1] - targets_ptr[j * 2 + 1];
            cout += dx * dx + dy * dy;
        }
    }
    return std::sqrt(cout);
}
*/


// Optimized mutate_nodes: use a single random generator for all elements, and avoid repeated vector construction
pybind11::array_t<double> mutate_nodes(pybind11::array_t<double> nodes, double stepsize) {
    pybind11::buffer_info nodes_info = nodes.request();
    if (nodes_info.ndim != 2 || nodes_info.shape[1] != 2) {
        throw std::runtime_error("Input nodes must be a 2D array with shape (n, 2)");
    }
    size_t n = nodes_info.shape[0];
    double* nodes_ptr = static_cast<double*>(nodes_info.ptr);

    // Preallocate result array with the same shape as input
    pybind11::array_t<double> result(nodes_info.shape);
    pybind11::buffer_info result_info = result.request();
    double* result_ptr = static_cast<double*>(result_info.ptr);

    // Efficient random number generation: fill all at once
    static thread_local std::mt19937 gen(std::random_device{}());
    std::normal_distribution<> d(0.0, stepsize);

    // Mutate all coordinates in a single loop (2*n elements)
    for (size_t i = 0; i < n * 2; ++i) {
        result_ptr[i] = nodes_ptr[i] + d(gen);
    }
    return result;
}


/*
on mutate les noeuds n fois, on vérifie si le graph est connecté, sinon on garde les noeuds précédents
on répète alors n fois l'opération
*/
pybind11::array_t<double> optimize_nodes(pybind11::array_t<double> nodes, pybind11::array_t<double> targets, double dist_threshold, double stepsize, size_t n) {
    double best_error = cout_graph_p2(nodes, targets);
    for (size_t i = 0; i < n; ++i) {
        double current_stepsize = stepsize * (static_cast<double>(n - i) / n);
        pybind11::array_t<double> new_nodes = mutate_nodes(nodes, current_stepsize);
        double new_error = cout_graph_p2(new_nodes, targets);
        if (new_error < best_error) {
            if (is_graph_connected_bfs(new_nodes, dist_threshold)) {
                nodes = new_nodes;
                best_error = new_error;
            }
        }
    }
    return nodes;
}

/*
Optimise les noeuds en gardant une historique des noeuds optimisés au fur et à mesure
*/
std::vector<pybind11::array_t<double>> optimize_nodes_history(pybind11::array_t<double> nodes, pybind11::array_t<double> targets, double dist_threshold, double stepsize, size_t n) {
    std::vector<pybind11::array_t<double>> history;
    double best_error = cout_graph_p2(nodes, targets);
    // Store the initial state
    history.push_back(nodes);
    for (size_t i = 0; i < n; ++i) {
        double current_stepsize = stepsize * (static_cast<double>(n - i) / n);
        pybind11::array_t<double> new_nodes = mutate_nodes(nodes, current_stepsize);
        double new_error = cout_graph_p2(new_nodes, targets);
        if (new_error < best_error) {
            if (is_graph_connected_bfs(new_nodes, dist_threshold)) {
                nodes = new_nodes;
                best_error = new_error;
                // Store a copy of the new nodes in the history
                history.push_back(nodes);
            }
        }
    }
    return history;
}



PYBIND11_MODULE(graphx, m) {
    m.def("distance", &distance, "Distance entre deux points");
    m.def("is_connected", &is_connected, "Vérifie si deux points sont connectés");
    m.def("is_graph_connected_bfs", &is_graph_connected_bfs, "Vérifie si un graph est connecté");
    m.def("cout_graph_p2", &cout_graph_p2, "Calcul le cout d'un graph");
    m.def("mutate_nodes", &mutate_nodes, "Muter les noeuds");
    m.def("optimize_nodes", &optimize_nodes, "Optimiser les noeuds");
    m.def("optimize_nodes_history", &optimize_nodes_history, "Optimiser les noeuds et garder l'historique");
}