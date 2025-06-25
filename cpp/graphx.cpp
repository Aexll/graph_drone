#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <random>
#include <thread>
#include <mutex>
#include <iostream>
#include <deque>
#include <algorithm>

// ===== OPTIMISATIONS GÉNÉRALES =====

// Cache pour éviter les validations répétées
struct ArrayInfo {
    double* ptr;
    size_t n;
    size_t stride;
    
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

bool is_connected(pybind11::array_t<double> a, pybind11::array_t<double> b, double dist_threshold) {
    ArrayInfo a_info(a), b_info(b);
    if (a_info.n != 1 || b_info.n != 1) {
        throw std::runtime_error("Input arrays must have exactly one point");
    }
    
    double threshold_sq = dist_threshold * dist_threshold;
    return distance_squared(a_info.get_point(0), b_info.get_point(0)) < threshold_sq;
}

// BFS optimisé avec deque et early exit
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

// ===== FONCTION DE COÛT OPTIMISÉE =====

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

// ===== MUTATION OPTIMISÉE =====

pybind11::array_t<double> mutate_nodes(pybind11::array_t<double> nodes, double stepsize) {
    ArrayInfo nodes_info(nodes);
    
    // Créer le résultat avec la même forme - syntaxe correcte
    auto result = pybind11::array_t<double>(std::vector<ptrdiff_t>{static_cast<ptrdiff_t>(nodes_info.n), 2});
    ArrayInfo result_info(result);
    
    // Générateur thread-local optimisé
    static thread_local std::mt19937 gen(std::random_device{}());
    std::normal_distribution<double> dist(0.0, stepsize);
    
    // Mutation vectorisée
    for (size_t i = 0; i < nodes_info.n; ++i) {
        const double* src = nodes_info.get_point(i);
        double* dst = result_info.get_point(i);
        dst[0] = src[0] + dist(gen);
        dst[1] = src[1] + dist(gen);
    }
    
    return result;
}

// ===== OPTIMISATION AVEC COOLING SCHEDULE AMÉLIORÉ =====

pybind11::array_t<double> optimize_nodes(
    pybind11::array_t<double> nodes, 
    pybind11::array_t<double> targets, 
    double dist_threshold, 
    double stepsize, 
    size_t n
) {
    auto current_nodes = nodes;
    double best_error = cout_graph_p2(current_nodes, targets);
    size_t consecutive_failures = 0;
    const size_t max_failures = n / 10; // Adaptative early stopping
    
    for (size_t i = 0; i < n; ++i) {
        // Cooling schedule non-linéaire plus efficace
        double progress = static_cast<double>(i) / n;
        double current_stepsize = stepsize * std::exp(-2.0 * progress); // Décroissance exponentielle
        
        auto new_nodes = mutate_nodes(current_nodes, current_stepsize);
        double new_error = cout_graph_p2(new_nodes, targets);
        
        if (new_error < best_error) {
            if (is_graph_connected_bfs(new_nodes, dist_threshold)) {
                current_nodes = std::move(new_nodes);
                best_error = new_error;
                consecutive_failures = 0;
            } else {
                ++consecutive_failures;
            }
        } else {
            ++consecutive_failures;
        }
        
        // Early stopping si trop d'échecs consécutifs
        if (consecutive_failures > max_failures) {
            break;
        }
    }
    
    return current_nodes;
}

// ===== VERSION AVEC HISTORIQUE OPTIMISÉE =====

std::vector<pybind11::array_t<double>> optimize_nodes_history(
    pybind11::array_t<double> nodes, 
    pybind11::array_t<double> targets, 
    double dist_threshold, 
    double stepsize, 
    size_t n
) {
    std::vector<pybind11::array_t<double>> history;
    history.reserve(n / 10 + 1); // Estimation intelligente de la taille
    
    auto current_nodes = nodes;
    double best_error = cout_graph_p2(current_nodes, targets);
    history.push_back(current_nodes);
    
    size_t consecutive_failures = 0;
    const size_t max_failures = n / 10;
    
    for (size_t i = 0; i < n; ++i) {
        double progress = static_cast<double>(i) / n;
        double current_stepsize = stepsize * std::exp(-2.0 * progress);
        
        auto new_nodes = mutate_nodes(current_nodes, current_stepsize);
        double new_error = cout_graph_p2(new_nodes, targets);
        
        if (new_error < best_error) {
            if (is_graph_connected_bfs(new_nodes, dist_threshold)) {
                current_nodes = std::move(new_nodes);
                best_error = new_error;
                consecutive_failures = 0;
                
                // Ne stocker que certains snapshots pour économiser la mémoire
                if (i % (n / 100 + 1) == 0 || i == n - 1) {
                    history.push_back(current_nodes);
                }
            } else {
                ++consecutive_failures;
            }
        } else {
            ++consecutive_failures;
        }
        
        if (consecutive_failures > max_failures) {
            break;
        }
    }
    
    return history;
}

// ===== VERSION PARALLÈLE ULTRA-OPTIMISÉE =====

struct NodeDataOptimized {
    std::vector<double> data;
    size_t n;
    
    explicit NodeDataOptimized(pybind11::array_t<double> arr) {
        ArrayInfo info(arr);
        n = info.n;
        data.resize(n * 2);
        
        // Copie vectorisée
        for (size_t i = 0; i < n; ++i) {
            const double* src = info.get_point(i);
            data[i * 2] = src[0];
            data[i * 2 + 1] = src[1];
        }
    }
    
    inline double* get_point(size_t i) { return &data[i * 2]; }
    inline const double* get_point(size_t i) const { return &data[i * 2]; }
    
    pybind11::array_t<double> to_array() const {
        auto result = pybind11::array_t<double>(std::vector<ptrdiff_t>{static_cast<ptrdiff_t>(n), 2});
        ArrayInfo result_info(result);
        
        for (size_t i = 0; i < n; ++i) {
            double* dst = result_info.get_point(i);
            const double* src = get_point(i);
            dst[0] = src[0];
            dst[1] = src[1];
        }
        return result;
    }
};

// Versions natives ultra-rapides
double cout_graph_p2_native(const NodeDataOptimized& nodes, const NodeDataOptimized& targets) {
    double cout = 0;
    for (size_t i = 0; i < nodes.n; ++i) {
        const double* node_ptr = nodes.get_point(i);
        const double* target_ptr = targets.get_point(i);
        cout += distance_squared(node_ptr, target_ptr);
    }
    return std::sqrt(cout);
}

bool is_graph_connected_bfs_native(const NodeDataOptimized& nodes, double dist_threshold) {
    if (nodes.n <= 1) return true;
    
    const double threshold_sq = dist_threshold * dist_threshold;
    std::vector<bool> visited(nodes.n, false);
    std::deque<size_t> queue;
    
    queue.push_back(0);
    visited[0] = true;
    size_t visited_count = 1;

    while (!queue.empty() && visited_count < nodes.n) {
        size_t current = queue.front();
        queue.pop_front();
        
        const double* current_ptr = nodes.get_point(current);
        
        for (size_t i = 0; i < nodes.n; ++i) {
            if (!visited[i]) {
                const double* other_ptr = nodes.get_point(i);
                if (distance_squared(current_ptr, other_ptr) < threshold_sq) {
                    visited[i] = true;
                    queue.push_back(i);
                    ++visited_count;
                }
            }
        }
    }
    
    return visited_count == nodes.n;
}

NodeDataOptimized mutate_nodes_native(const NodeDataOptimized& nodes, double stepsize) {
    NodeDataOptimized result = nodes;
    
    static thread_local std::mt19937 gen(std::random_device{}());
    std::normal_distribution<double> dist(0.0, stepsize);
    
    for (size_t i = 0; i < nodes.n * 2; ++i) {
        result.data[i] += dist(gen);
    }
    
    return result;
}

std::vector<NodeDataOptimized> optimize_nodes_history_native(
    NodeDataOptimized nodes, 
    const NodeDataOptimized& targets, 
    double dist_threshold, 
    double stepsize, 
    size_t n
) {
    std::vector<NodeDataOptimized> history;
    history.reserve(n / 100 + 1);
    
    double best_error = cout_graph_p2_native(nodes, targets);
    history.push_back(nodes);
    
    size_t consecutive_failures = 0;
    const size_t max_failures = n / 10;
    
    for (size_t i = 0; i < n; ++i) {
        double progress = static_cast<double>(i) / n;
        double current_stepsize = stepsize * std::exp(-2.0 * progress);
        
        NodeDataOptimized new_nodes = mutate_nodes_native(nodes, current_stepsize);
        double new_error = cout_graph_p2_native(new_nodes, targets);
        
        if (new_error < best_error) {
            if (is_graph_connected_bfs_native(new_nodes, dist_threshold)) {
                nodes = std::move(new_nodes);
                best_error = new_error;
                consecutive_failures = 0;
                
                if (i % (n / 100 + 1) == 0 || i == n - 1) {
                    history.push_back(nodes);
                }
            } else {
                ++consecutive_failures;
            }
        } else {
            ++consecutive_failures;
        }
        
        if (consecutive_failures > max_failures) {
            break;
        }
    }
    
    return history;
}

std::vector<std::vector<pybind11::array_t<double>>> optimize_nodes_history_parallel(
    pybind11::array_t<double> nodes,
    pybind11::array_t<double> targets,
    double dist_threshold,
    double stepsize,
    size_t n,
    size_t n_threads
) {
    // Conversion avant parallélisation
    NodeDataOptimized nodes_native(nodes);
    NodeDataOptimized targets_native(targets);
    
    std::vector<std::vector<NodeDataOptimized>> all_histories_native(n_threads);
    std::vector<std::thread> threads;
    threads.reserve(n_threads);

    // Relâcher le GIL pour le parallélisme pur
    pybind11::gil_scoped_release release;

    auto worker = [&](size_t thread_id) {
        try {
            auto history = optimize_nodes_history_native(
                nodes_native, targets_native, dist_threshold, stepsize, n
            );
            all_histories_native[thread_id] = std::move(history);
        } catch (const std::exception& e) {
            std::cerr << "Exception in thread " << thread_id << ": " << e.what() << std::endl;
        }
    };

    for (size_t i = 0; i < n_threads; ++i) {
        threads.emplace_back(worker, i);
    }
    
    for (auto& t : threads) {
        t.join();
    }

    // Reconversion efficace
    std::vector<std::vector<pybind11::array_t<double>>> all_histories(n_threads);
    for (size_t i = 0; i < n_threads; ++i) {
        all_histories[i].reserve(all_histories_native[i].size());
        for (const auto& node_data : all_histories_native[i]) {
            all_histories[i].push_back(node_data.to_array());
        }
    }
    
    return all_histories;
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

    // ===== FONCTIONS NATIVES =====
    m.def("cout_graph_p2_native", &cout_graph_p2_native, "Calcul le cout d'un graph");
    m.def("is_graph_connected_bfs_native", &is_graph_connected_bfs_native, "Vérifie si un graph est connecté");
    m.def("mutate_nodes_native", &mutate_nodes_native, "Muter les noeuds");
    m.def("optimize_nodes_history_native", &optimize_nodes_history_native, "Optimiser les noeuds et garder l'historique");
}