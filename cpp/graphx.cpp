#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <random>
#include <thread>
#include <mutex>
#include <iostream>
#include <deque>
#include <algorithm>
#include <set>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>



// pour ne pas avoir à récupérer les informations de l'array à chaque fois
struct ArrayInfo {
    double* ptr;
    size_t n;
    size_t stride; // toujours 2
    
    ArrayInfo(pybind11::array_t<double>& arr) {
        pybind11::buffer_info info = arr.request();
        if (info.ndim != 2 || info.shape[1] != 2) {
            throw std::runtime_error("Input must be a 2D array with shape (n, 2)");
        }
        ptr = static_cast<double*>(info.ptr);
        n = info.shape[0];
        stride = info.strides[0] / sizeof(double);
    }
    
    // Accès optimisé aux coordonnées
    inline double* get_point(size_t i) { return ptr + i * stride; }
    inline const double* get_point(size_t i) const { return ptr + i * stride; }

    // for the adjacency matrix (2D int array)
    inline int& get_int_point(size_t i, size_t j) {
        return reinterpret_cast<int*>(ptr)[i * stride + j];
    }
};

// ===== FONCTIONS DE BASE OPTIMISÉES =====

// Version inline pour éviter les appels de fonction
inline double distance_squared(const double* a, const double* b) {
    double dx = a[0] - b[0];
    double dy = a[1] - b[1];
    return dx * dx + dy * dy;
}

inline double distance_fast(const double* a, const double* b) {
    return std::sqrt(distance_squared(a, b));
}

// Distance entre deux points - version optimisée
double distance(pybind11::array_t<double> a, pybind11::array_t<double> b) {
    ArrayInfo a_info(a), b_info(b);
    if (a_info.n != 1 || b_info.n != 1) {
        throw std::runtime_error("Input arrays must have exactly one point");
    }
    return distance_fast(a_info.get_point(0), b_info.get_point(0));
}


/*
si deux points sont à distance inférieure à dist_threshold, alors ils sont connectés
*/
bool is_connected(pybind11::array_t<double> a, pybind11::array_t<double> b, double dist_threshold) {
    ArrayInfo a_info(a), b_info(b);
    if (a_info.n != 1 || b_info.n != 1) {
        throw std::runtime_error("Input arrays must have exactly one point");
    }
    
    double threshold_sq = dist_threshold * dist_threshold;
    return distance_squared(a_info.get_point(0), b_info.get_point(0)) < threshold_sq;
}


/*
return adjacency matrix of a graph
*/
pybind11::array_t<int> get_adjacency_matrix(pybind11::array_t<double> nodes, double dist_threshold) {
    ArrayInfo nodes_info(nodes);
    size_t n = nodes_info.n;
    auto adj = pybind11::array_t<int>({n, n});
    auto adj_buf = static_cast<int*>(adj.request().ptr);
    size_t adj_stride = adj.strides(0) / sizeof(int);
    double threshold_sq = dist_threshold * dist_threshold;
    for (size_t i = 0; i < n; ++i) {
        double* pi = nodes_info.get_point(i);
        for (size_t j = 0; j < n; ++j) {
            double* pj = nodes_info.get_point(j);
            adj_buf[i * adj_stride + j] = (distance_squared(pi, pj) < threshold_sq) ? 1 : 0;
        }
    }
    return adj;
}

/*
BFS optimisé avec deque et early exit
deque : permet de push/pop en O(1)
early exit : si le graph est connecté, on peut stopper la recherche
si le graph est connecté, retourne true, sinon false
*/
bool is_graph_connected_bfs(pybind11::array_t<double> nodes, double dist_threshold) {
    ArrayInfo nodes_info(nodes);
    if (nodes_info.n == 0) return true;
    if (nodes_info.n == 1) return true;
    
    const double threshold_sq = dist_threshold * dist_threshold;
    std::vector<bool> visited(nodes_info.n, false);
    std::deque<size_t> queue; // deque plus efficace que vector pour push/pop
    
    queue.push_back(0);
    visited[0] = true;
    size_t visited_count = 1;

    while (!queue.empty() && visited_count < nodes_info.n) {
        size_t current = queue.front();
        queue.pop_front();
        
        const double* current_ptr = nodes_info.get_point(current);
        
        for (size_t i = 0; i < nodes_info.n; ++i) {
            if (!visited[i]) {
                const double* other_ptr = nodes_info.get_point(i);
                if (distance_squared(current_ptr, other_ptr) < threshold_sq) {
                    visited[i] = true;
                    queue.push_back(i);
                    ++visited_count;
                }
            }
        }
    }
    
    return visited_count == nodes_info.n;
}


/*
retourne le cout d'un graph, le cout est la somme des distances au carré entre les noeuds et les cibles (p2)
le tout est ensuite racine carrée pour avoir la distance euclidienne
*/
double cout_graph_p2(pybind11::array_t<double> nodes, pybind11::array_t<double> targets) {
    ArrayInfo nodes_info(nodes), targets_info(targets);
    
    if (nodes_info.n != targets_info.n) {
        throw std::runtime_error("nodes and targets must have the same number of points");
    }
    
    double cout = 0;
    for (size_t i = 0; i < nodes_info.n; ++i) {
        const double* node_ptr = nodes_info.get_point(i);
        const double* target_ptr = targets_info.get_point(i);
        cout += distance_squared(node_ptr, target_ptr);
    }
    return std::sqrt(cout);
}


/*
muter les noeuds en ajoutant une valeur aléatoire et retourne un nouveau tableau de noeuds
*/
pybind11::array_t<double> mutate_nodes(pybind11::array_t<double> nodes, double stepsize) {
    ArrayInfo nodes_info(nodes);
    
    auto result = pybind11::array_t<double>(std::vector<ptrdiff_t>{static_cast<ptrdiff_t>(nodes_info.n), 2});
    ArrayInfo result_info(result);
    
    // Générateur thread-local pour avoir des valeurs aléatoires différentes pour chaque thread
    static thread_local std::mt19937 gen;
    static thread_local bool seeded = false;
    if (!seeded) {
        gen.seed(std::random_device{}() + std::hash<std::thread::id>{}(std::this_thread::get_id()) + std::chrono::steady_clock::now().time_since_epoch().count());
        seeded = true;
    }
    std::normal_distribution<double> dist(0.0, stepsize);
    
    for (size_t i = 0; i < nodes_info.n; ++i) {
        const double* src = nodes_info.get_point(i);
        double* dst = result_info.get_point(i);
        dst[0] = src[0] + dist(gen);
        dst[1] = src[1] + dist(gen);
    }
    
    return result;
}



/*
muter les noeuds en ajoutant une valeur aléatoire sans allouer de mémoire
on ne retourne rien, on modifie directement le buffer 'result_info'
*/
void mutate_nodes_inplace(
    const ArrayInfo& nodes_info, // Les nœuds source (const)
    ArrayInfo& result_info,      // Les nœuds de destination à modifier
    double stepsize
) {
    // Générateur thread-local pour avoir des valeurs aléatoires différentes pour chaque thread
    static thread_local std::mt19937 gen;
    static thread_local bool seeded = false;
    if (!seeded) {
        gen.seed(std::random_device{}() + std::hash<std::thread::id>{}(std::this_thread::get_id()) + std::chrono::steady_clock::now().time_since_epoch().count());
        seeded = true;
    }
    std::normal_distribution<double> dist(0.0, stepsize);
    
    // on fait une mutation vectorisée pour éviter les boucles
    for (size_t i = 0; i < nodes_info.n; ++i) {
        const double* src = nodes_info.get_point(i);
        double* dst = result_info.get_point(i);
        dst[0] = src[0] + dist(gen);
        dst[1] = src[1] + dist(gen);
    }
}


/*
optimiser l'erreur d'un graph par mutation aléatoire
on retourne le meilleur graph trouvé
*/
pybind11::array_t<double> optimize_nodes(
    pybind11::array_t<double> nodes, 
    pybind11::array_t<double> targets, 
    double dist_threshold, 
    double stepsize, 
    size_t n,
    bool failure_stop_enabled = false
) {
    // Création des buffers UNE SEULE FOIS
    auto current_nodes = pybind11::array_t<double>(nodes); // Copie initiale
    auto candidate_nodes = pybind11::array_t<double>(nodes.request()); // Buffer pour les mutations

    ArrayInfo current_info(current_nodes);
    ArrayInfo candidate_info(candidate_nodes);
    
    double best_error = cout_graph_p2(current_nodes, targets);
    size_t consecutive_failures = 0;
    const size_t max_failures = n / 10;
    
    for (size_t i = 0; i < n; ++i) {
        double progress = static_cast<double>(i) / n;
        double current_stepsize = stepsize * std::exp(-2.0 * progress);
        
        // Muter dans le buffer 'candidate' sans allouer de mémoire
        mutate_nodes_inplace(current_info, candidate_info, current_stepsize);
        
        double new_error = cout_graph_p2(candidate_nodes, targets);
        
        if (new_error < best_error) {
            if (is_graph_connected_bfs(candidate_nodes, dist_threshold)) {
                // La mutation est acceptée : copier le candidat dans l'état actuel
                // C'est beaucoup plus rapide qu'une nouvelle allocation
                std::copy(
                    static_cast<double*>(candidate_info.ptr),
                    static_cast<double*>(candidate_info.ptr) + candidate_info.n * candidate_info.stride,
                    static_cast<double*>(current_info.ptr)
                );
                best_error = new_error;
                consecutive_failures = 0;
            } else {
                ++consecutive_failures;
            }
        } else {
            ++consecutive_failures;
        }
        
        if (failure_stop_enabled && consecutive_failures > max_failures) {
            break;
        }
    }
    
    return current_nodes;
}


/*
optimiser l'erreur d'un graph par mutation aléatoire et garder l'historique
on retourne une liste de graphes optimisés qui sont les meilleurs à chaque étape
*/
std::vector<pybind11::array_t<double>> optimize_nodes_history(
    pybind11::array_t<double> nodes, 
    pybind11::array_t<double> targets, 
    double dist_threshold, 
    double stepsize, 
    size_t n,
    bool failure_stop_enabled = false,
    bool push_always = true
) {
    std::vector<pybind11::array_t<double>> history;
    history.reserve(n); // Estimation intelligente de la taille

    auto current_nodes = pybind11::array_t<double>(nodes);
    auto candidate_nodes = pybind11::array_t<double>(nodes.request());
    
    ArrayInfo current_info(current_nodes);
    ArrayInfo candidate_info(candidate_nodes);
    
    double best_error = cout_graph_p2(current_nodes, targets);
    history.push_back(pybind11::array_t<double>(current_nodes.request()));
    
    size_t consecutive_failures = 0;
    const size_t max_failures = n / 10;
    
    for (size_t i = 0; i < n; ++i) {
        double progress = static_cast<double>(i) / n;
        double current_stepsize = stepsize * std::exp(-2.0 * progress);
        
        mutate_nodes_inplace(current_info, candidate_info, current_stepsize);

        double new_error = cout_graph_p2(candidate_nodes, targets);
        
        if (new_error < best_error) {
            if (is_graph_connected_bfs(candidate_nodes, dist_threshold)) {
                // current_nodes = std::move(candidate_nodes);
                std::copy(
                    static_cast<double*>(candidate_info.ptr),
                    static_cast<double*>(candidate_info.ptr) + candidate_info.n * candidate_info.stride,
                    static_cast<double*>(current_info.ptr)
                );
                best_error = new_error;
                consecutive_failures = 0;
                
                // Ne stocker que certains snapshots pour économiser la mémoire
                // if (i % (n / 100 + 1) == 0 || i == n - 1) {
                history.push_back(pybind11::array_t<double>(current_nodes.request()));
                
            } else {
                ++consecutive_failures;
            }
        } else {
            ++consecutive_failures;
        }

        
        if (push_always) {
            history.push_back(pybind11::array_t<double>(current_nodes.request()));
        }
        
        if (failure_stop_enabled && consecutive_failures > max_failures) {
            // std::cout << "Early stopping due to consecutive failures" << std::endl;
            break;
        }
    }
    
    // std::cout << "History size: " << history.size() << std::endl;
    return history;
}

/*
retournes une liste de n_threads graphes optimisés en parallèle, chaque graphe est un tableau de noeuds
*/
std::vector<pybind11::array_t<double>> optimize_nodes_parallel(
    pybind11::array_t<double> nodes,
    pybind11::array_t<double> targets,
    double dist_threshold,
    double stepsize,
    size_t n,
    size_t n_threads,
    bool failure_stop_enabled = false
) {
    std::vector<std::thread> threads;
    std::vector<pybind11::array_t<double>> all_nodes(n_threads);

    pybind11::gil_scoped_release release;

    auto worker = [&](size_t thread_id) {
        try {
            pybind11::gil_scoped_acquire acquire;
            auto nodes_copy = pybind11::array_t<double>(nodes.request()); // Copie profonde du buffer
            all_nodes[thread_id] = optimize_nodes(nodes_copy, targets, dist_threshold, stepsize, n, failure_stop_enabled);
        } catch (const std::exception& e) {
            std::cout << "Exception in thread " << thread_id << ": " << e.what() << std::endl;
        }
    };

    for (size_t i = 0; i < n_threads; ++i) {
        threads.emplace_back(worker, i);
    }
    for (auto& t : threads) {
        t.join();
    }
    return all_nodes;
}



/*
retournes une liste de n_threads historiques de graphes optimisés en parallèle, chaque historique est une liste de graphes
*/
std::vector<std::vector<pybind11::array_t<double>>> optimize_nodes_history_parallel(
    pybind11::array_t<double> nodes,
    pybind11::array_t<double> targets,
    double dist_threshold,
    double stepsize,
    size_t n,
    size_t n_threads,
    bool failure_stop_enabled = false
) {
    std::vector<std::vector<pybind11::array_t<double>>> all_histories(n_threads);
    std::vector<std::thread> threads;

    pybind11::gil_scoped_release release;

    auto worker = [&](size_t thread_id) {
        try {
            pybind11::gil_scoped_acquire acquire;
            auto nodes_copy = pybind11::array_t<double>(nodes.request()); // Copie profonde du buffer
            auto history = optimize_nodes_history(nodes_copy, targets, dist_threshold, stepsize, n, failure_stop_enabled);
            all_histories[thread_id] = std::move(history);
        } catch (const std::exception& e) {
            std::cout << "Exception in thread " << thread_id << ": " << e.what() << std::endl;
        }
    };

    for (size_t i = 0; i < n_threads; ++i) {
        threads.emplace_back(worker, i);
    }
    // std::cout << "All threads launched\n";
    for (auto& t : threads) {
        t.join();
    }
    // std::cout << "All threads joined\n";
    return all_histories;
}



/*
Retourne un tuple des connections de chaque noeud
*/
pybind11::tuple get_shape(pybind11::array_t<double> nodes, double dist_threshold) {
    ArrayInfo nodes_info(nodes);
    std::vector<std::pair<size_t, size_t>> edges;
    const double threshold_sq = dist_threshold * dist_threshold;
    for (size_t i = 0; i < nodes_info.n; ++i) {
        const double* node_i = nodes_info.get_point(i);
        for (size_t j = i + 1; j < nodes_info.n; ++j) { // j > i pour éviter les doublons
            const double* node_j = nodes_info.get_point(j);
            if (distance_squared(node_i, node_j) < threshold_sq) {
                edges.emplace_back(i, j);
            }
        }
    }
    std::sort(edges.begin(), edges.end());
    // Conversion en tuple de tuples pour Python
    pybind11::list edge_list;
    for (const auto& e : edges) {
        edge_list.append(pybind11::make_tuple(e.first, e.second));
    }
    return pybind11::tuple(edge_list);
}


/*
Retourne la distance entre deux formes (tuple d'arêtes),
c'est-à-dire le nombre d'arêtes différentes entre les deux graphes.
*/
int get_shape_distance(pybind11::tuple shape1, pybind11::tuple shape2) {
    std::set<std::pair<size_t, size_t>> set1, set2;
    for (auto item : shape1) {
        auto t = pybind11::cast<pybind11::tuple>(item);
        set1.emplace(pybind11::cast<size_t>(t[0]), pybind11::cast<size_t>(t[1]));
    }
    for (auto item : shape2) {
        auto t = pybind11::cast<pybind11::tuple>(item);
        set2.emplace(pybind11::cast<size_t>(t[0]), pybind11::cast<size_t>(t[1]));
    }
    // Distance = taille de la différence symétrique
    std::vector<std::pair<size_t, size_t>> diff;
    std::set_symmetric_difference(
        set1.begin(), set1.end(),
        set2.begin(), set2.end(),
        std::back_inserter(diff)
    );
    return static_cast<int>(diff.size());
}


std::string get_shape_string(pybind11::tuple shape) {
    // Trouver le nombre de noeuds (max index + 1)
    size_t n = 0;
    std::vector<std::vector<size_t>> neighbors;
    for (auto item : shape) {
        auto t = pybind11::cast<pybind11::tuple>(item);
        size_t i = pybind11::cast<size_t>(t[0]);
        size_t j = pybind11::cast<size_t>(t[1]);
        n = std::max(n, std::max(i, j) + 1);
    }
    neighbors.resize(n);

    // Remplir les voisins d'index strictement supérieur
    for (auto item : shape) {
        auto t = pybind11::cast<pybind11::tuple>(item);
        size_t i = pybind11::cast<size_t>(t[0]);
        size_t j = pybind11::cast<size_t>(t[1]);
        if (j > i) {
            neighbors[i].push_back(j);
        }
        // Si tu veux aussi les voisins inférieurs, tu pourrais ajouter : neighbors[j].push_back(i);
    }

    // Construire la string
    std::stringstream ss;
    for (size_t i = 0; i < n; ++i) {
        for (size_t k = 0; k < neighbors[i].size(); ++k) {
            if (k > 0) ss << '\'';
            ss << neighbors[i][k];
        }
        if (i < n - 1) ss << ',';
    }
    return ss.str();
}

/*
 * Depuis un historique de graphes (liste de graphes), renvoie une liste des transitions de string shapes (shape sous forme de string)
 * associée à chaque graphes de l'historique, par exemple si on a 
 * history = [g1, g2, g3, g4, g5] avec
 * g1 = "1,2," g2 = "1,2" g3 = "2,1," g4 = "2,1," g5 = "2'3,,"
 * alors la fonction retournera :
 * {("1,2,","2,1,"),("2,1,","2'3,")}
 */
std::set<std::pair<std::string, std::string>> get_shape_string_transition_history(const std::vector<pybind11::array_t<double>>& history, double dist_threshold) {
    std::set<std::pair<std::string, std::string>> transitions;
    std::string last_shape;
    bool first = true;
    for (const auto& graph : history) {
        auto shape = get_shape(graph, dist_threshold);
        std::string shape_str = get_shape_string(shape);
        if (first) {
            last_shape = shape_str;
            first = false;
            continue;
        }
        if (shape_str != last_shape) {
            transitions.emplace(last_shape, shape_str);
            last_shape = shape_str;
        }
    }
    return transitions;
}






/*
depuis un historique de shapes donnée
retournes un set de paire de shapes qui représente les transitions entre les shapes de l'historique
*/




/*
 * Décompose un historique de graphes en un dictionnaire associant à chaque shape string :
 *   - la shape sous forme de string (clé)
 *   - le graphe ayant le meilleur score pour cette shape (valeur : dict avec 'graph' et 'score')
 *   - le score du meilleur graphe pour cette shape (valeur : 'score')
 *
 * Retourne un dict Python : { shape_string : { 'graph': <ndarray>, 'score': <float> } }
 */
pybind11::dict decompose_history_by_shape(const std::vector<pybind11::array_t<double>>& history, pybind11::array_t<double> targets, double dist_threshold) {
    pybind11::dict result;
    std::unordered_map<std::string, std::pair<double, pybind11::array_t<double>>> bests;

    for (const auto& graph : history) {
        auto shape = get_shape(graph, dist_threshold);
        std::string shape_str = get_shape_string(shape);
        double score = cout_graph_p2(graph, targets);

        auto it = bests.find(shape_str);
        if (it == bests.end() || score < it->second.first) {
            // Nouvelle shape ou meilleur score trouvé
            bests[shape_str] = std::make_pair(score, graph);
        }
    }

    // Remplir le dict Python
    for (const auto& kv : bests) {
        pybind11::dict entry;
        entry["graph"] = kv.second.second;
        entry["score"] = kv.second.first;
        result[pybind11::str(kv.first)] = entry;
    }
    return result;
}




// ===== MODULE PYBIND11 =====

PYBIND11_MODULE(graphx, m) {
    m.def("distance", &distance, "Distance entre deux points");
    m.def("is_connected", &is_connected, "Vérifie si deux points sont connectés");
    m.def("is_graph_connected_bfs", &is_graph_connected_bfs, "Vérifie si un graph est connecté");
    m.def("cout_graph_p2", &cout_graph_p2, "Calcul le cout d'un graph");
    m.def("mutate_nodes", &mutate_nodes, "Muter les noeuds");
    m.def("optimize_nodes", &optimize_nodes, "Optimiser les noeuds");
    m.def("optimize_nodes_history", &optimize_nodes_history, "Optimiser les noeuds et garder l'historique");
    m.def("optimize_nodes_history_parallel", &optimize_nodes_history_parallel, "Optimiser les noeuds et garder l'historique en parallèle");
    m.def("get_shape", &get_shape, "Retourne un tuple de la taille du nombre de noeuds, qui représente le nombre de connections de chaque noeud");
    m.def("optimize_nodes_parallel", &optimize_nodes_parallel, "Optimiser les noeuds en parallèle");
    m.def("get_shape_distance", &get_shape_distance, "Retourne la distance entre deux formes");
    m.def("get_shape_string", &get_shape_string, "Retourne la représentation sous forme de chaîne de caractères d'un tuple de connections");
    m.def("get_shape_string_transition_history", &get_shape_string_transition_history, "Retourne un set de transitions (from_shape, to_shape) entre les shapes successives de l'historique, en sautant les répétitions consécutives.",
        pybind11::arg("history"), pybind11::arg("dist_threshold"));
    m.def("decompose_history_by_shape", &decompose_history_by_shape, "Décompose un historique de graphes en un dictionnaire associant à chaque shape string le meilleur graphe et son score.",
        pybind11::arg("history"), pybind11::arg("targets"), pybind11::arg("dist_threshold"));
    m.def("get_adjacency_matrix", &get_adjacency_matrix, "Retourne la matrice d'adjacence d'un graphe");
}