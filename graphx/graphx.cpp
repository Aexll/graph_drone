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


std::string version(){
    return "graphx version 0.1.2";
}

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

// Distance entre deux points - version optimisée (prend deux vecteurs 2D)
double distance(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != 2 || b.size() != 2) {
        throw std::runtime_error("Input vectors must have exactly two elements representing a 2D point");
    }
    double dx = a[0] - b[0];
    double dy = a[1] - b[1];
    return std::sqrt(dx * dx + dy * dy);
}


/*
si deux points sont à distance inférieure à dist_threshold, alors ils sont connectés
*/
bool is_connected(const std::vector<double>& a, const std::vector<double>& b, double dist_threshold) {
    if (a.size() != 2 || b.size() != 2) {
        throw std::runtime_error("Input vectors must have exactly two elements representing a 2D point");
    }
    double dx = a[0] - b[0];
    double dy = a[1] - b[1];
    double threshold_sq = dist_threshold * dist_threshold;
    return (dx * dx + dy * dy) < threshold_sq;
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
            adj_buf[i * adj_stride + j] = (distance_squared(pi, pj) <= threshold_sq) ? 1 : 0;
        }
    }
    return adj;
}


/*
get_node_contact_array 

retourne un une liste d'index des noeuds, ordonée par le nombre d'arrets minimum pour atteindre le noeud passé en paramètre
puis ordonnée par leurs indices

ainsi si le noeud 0 est connecté au noeud 1 et au noeud 3, et le noeud 1 est connecté au noeud 2 et au noeud 3,
alors si on appelle get_node_contact_array(nodes, 0), on obtient [0, 1, 3, 2]

O(n) en temps et O(n) en mémoire
*/
std::vector<int> get_node_contact_array(pybind11::array_t<double> nodes, int start_idx, double dist_threshold) {
    ArrayInfo nodes_info(nodes);
    if (nodes_info.n < 1) {
        throw std::runtime_error("nodes must have at least one point");
    }

    std::vector<int> contact_array(nodes_info.n);
    std::vector<bool> visited(nodes_info.n, false);
    std::deque<int> queue;
    queue.push_back(start_idx);
    visited[start_idx] = true;
    int count = 0;
    while (!queue.empty()) {
        int current = queue.front();
        queue.pop_front();
        contact_array[count] = current;
        count++;
        for (int i = 0; i < nodes_info.n; ++i) {
            if (distance_squared(nodes_info.get_point(current), nodes_info.get_point(i)) < dist_threshold * dist_threshold && !visited[i]) {
                visited[i] = true;
                queue.push_back(i); // on ajoute le noeud à la queue
            }
        }
    }

    return contact_array;
}






/*
Cercles et probabilités
*/


/*
    etant donné une liste de noeud et un point de départ,
    choisis au hasard un noeud dans une liste de noeuds, tel que le noeud le plus proches du point de départ a plus de chance d'être choisi
    on retourne l'index du noeud choisi
*/
int chose_node_near_node_weighted(pybind11::array_t<double> nodes, pybind11::array_t<double> start_node, double dist_threshold, double sigma) {
    pybind11::buffer_info info = start_node.request();
    if (info.ndim != 1 || info.shape[0] != 2) {
        throw std::runtime_error("start_node must be shape (2,)");
    }
    ArrayInfo nodes_info(nodes);
    if (nodes_info.n < 1) {
        throw std::runtime_error("nodes must have at least one point");
    }
    const double* start_ptr = static_cast<const double*>(info.ptr);
    // on calcule les distances entre le point de départ et les autres points
    std::vector<double> distances(nodes_info.n);
    for (size_t i = 0; i < nodes_info.n; ++i) {
        distances[i] = distance_squared(nodes_info.get_point(i), start_ptr);
    }
    // on calcule les poids
    std::vector<double> weights(nodes_info.n);
    for (size_t i = 0; i < nodes_info.n; ++i) {
        weights[i] = std::exp(-distances[i] / (2 * sigma * sigma));
    }
    std::discrete_distribution<> dist(weights.begin(), weights.end());
    static thread_local std::mt19937 gen;
    static thread_local bool seeded = false;
    if (!seeded) {
        gen.seed(std::random_device{}() + std::hash<std::thread::id>{}(std::this_thread::get_id()) + std::chrono::steady_clock::now().time_since_epoch().count());
        seeded = true;
    }
    size_t chosen_idx = dist(gen);
    return static_cast<int>(chosen_idx);
}



/*
    random_points_in_disk_with_attraction_point

    etant donnée un disque de centre C de rayon r et un point attractif A,
    renvoie un point B dans le disque aléatoirement, tel que, plus la distance entre A et B est grande, plus la probabilité que B soit choisi est faible
    retournes un point 2D

    (approche par rejet)

    O(1) en temps et O(1) en mémoire (sauf si on compte le générateur de nombres aléatoires)
*/
pybind11::array_t<double> random_points_in_disk_with_attraction_point(pybind11::array_t<double> disk_center, double radius, pybind11::array_t<double> attract_point, double sigma) {
    // Ensure inputs are 1D arrays of shape (2,)
    pybind11::buffer_info center_info = disk_center.request();
    pybind11::buffer_info attract_info = attract_point.request();
    if (center_info.ndim != 1 || center_info.shape[0] != 2) {
        throw std::runtime_error("disk_center must be a 1D array of shape (2,)");
    }
    if (attract_info.ndim != 1 || attract_info.shape[0] != 2) {
        throw std::runtime_error("attract_point must be a 1D array of shape (2,)");
    }
    const double* center_ptr = static_cast<const double*>(center_info.ptr);
    const double* attract_ptr = static_cast<const double*>(attract_info.ptr);
    double x = center_ptr[0];
    double y = center_ptr[1];
    double r = radius;
    double attract_x = attract_ptr[0];
    double attract_y = attract_ptr[1];

    static thread_local std::mt19937 gen;
    static thread_local bool seeded = false;
    if (!seeded) {
        // on seed le générateur de manière unique pour chaque thread pour éviter les collisions
        gen.seed(std::random_device{}() + std::hash<std::thread::id>{}(std::this_thread::get_id()) + std::chrono::steady_clock::now().time_since_epoch().count());
        seeded = true;
    }

    double x_new, y_new;
    while (true) {
        double theta = std::uniform_real_distribution<>(0, 2 * M_PI)(gen);
        double rho = r * std::sqrt(std::uniform_real_distribution<>(0.0, 1.0)(gen));
        x_new = x + rho * std::cos(theta);
        y_new = y + rho * std::sin(theta);

        double dx = x_new - attract_x;
        double dy = y_new - attract_y;
        double dist2 = dx * dx + dy * dy;
        double weight = std::exp(-dist2 / (2 * sigma * sigma));
        double u = std::uniform_real_distribution<>(0.0, 1.0)(gen);
        if (u < weight) {
            break;
        }
    }

    // Return as a 1D numpy array of shape (2,)
    auto result = pybind11::array_t<double>(2);
    pybind11::buffer_info result_info = result.request();
    double* result_ptr = static_cast<double*>(result_info.ptr);
    result_ptr[0] = x_new;
    result_ptr[1] = y_new;
    return result;
}



/*
    random_points_in_disk_with_attraction_point_vectorized

    etant donnée un disque de centre C de rayon r et un point attractif A,
    renvoie un point B dans le disque aléatoirement, tel que, plus la distance entre A et B est grande, plus la probabilité que B soit choisi est faible
    retournes un point 2D

    (approche vectorisée)
*/






/*
    safe_mutate_nodes

    muter les noeuds de sorte que le graph soit connecté dans tout les cas

    etant donnée une liste de noeuds et un index de départ,
    on commences par trier la liste des noeuds par leurs distances au noeud de départ en terme de liens connectés
    dans une nouvelle liste, on commence par ajouter le noeud de départ (à l'index de depart)
    puis successivement, on ajoute un noeud généré aléatoirement sur l'union des disques de centre les noeuds 
    de la liste des noeuds déjà générés et de rayon r, avec un point attractif à l'endroit du noeud de la liste des noeuds
    on continue jusqu'à ce qu'il ne reste plus de noeuds à générer
    on retourne la nouvelle liste de noeuds

    O(n) en temps et O(n) en mémoire
*/
pybind11::array_t<double> safe_mutate_nodes(pybind11::array_t<double> nodes, int start_idx, double radius, double sigma) {
    ArrayInfo nodes_info(nodes);
    if (nodes_info.n < 1) {
        throw std::runtime_error("nodes must have at least one point");
    }
    if (start_idx < 0 || start_idx >= nodes_info.n) {
        throw std::runtime_error("start_idx must be between 0 and the number of nodes");
    }

    std::vector<int> contact_array = get_node_contact_array(nodes, start_idx, radius);

    // On commence avec un vector vide
    std::vector<std::array<double, 2>> new_nodes_vec;

    // Ajoute le noeud de départ
    const double* src = nodes_info.get_point(start_idx);
    new_nodes_vec.push_back({src[0], src[1]});

    for (size_t i = 1; i < nodes_info.n; ++i) {

        const double* current_node_ptr = nodes_info.get_point(contact_array[i]);

        // std::cout << "current_node_ptr: " << current_node_ptr[0] << ", " << current_node_ptr[1] << std::endl;

        pybind11::array_t<double> current_node_arr(std::vector<ptrdiff_t>{2}, current_node_ptr);

        // Crée un tableau numpy temporaire pour les noeuds déjà générés
        pybind11::array_t<double> new_nodes_arr(std::vector<ptrdiff_t>{(ptrdiff_t)new_nodes_vec.size(), 2});
        auto buf = new_nodes_arr.mutable_unchecked<2>();
        for (size_t j = 0; j < new_nodes_vec.size(); ++j) {
            buf(j, 0) = new_nodes_vec[j][0];
            buf(j, 1) = new_nodes_vec[j][1];
        }

        // for (size_t j = 0; j < new_nodes_vec.size(); ++j) {
        //     std::cout << new_nodes_vec[j][0] << ", " << new_nodes_vec[j][1] << " \n";
        // }
        // std::cout << std::endl;

        int selected_disk_idx = chose_node_near_node_weighted(new_nodes_arr, current_node_arr, radius, sigma);

        // std::cout << "selected_disk_idx: " << selected_disk_idx << std::endl;

        const double* selected_disk_node_ptr = new_nodes_vec[selected_disk_idx].data();
        pybind11::array_t<double> selected_disk_node_arr(std::vector<ptrdiff_t>{2}, selected_disk_node_ptr);

        auto new_node = random_points_in_disk_with_attraction_point(selected_disk_node_arr, radius, current_node_arr, sigma);
        pybind11::buffer_info new_node_info = new_node.request();
        const double* new_node_ptr = static_cast<const double*>(new_node_info.ptr);

        new_nodes_vec.push_back({new_node_ptr[0], new_node_ptr[1]});
    }

    // Conversion finale en pybind11::array_t<double>
    pybind11::array_t<double> result(std::vector<ptrdiff_t>{(ptrdiff_t)new_nodes_vec.size(), 2});
    auto result_buf = result.mutable_unchecked<2>();
    for (size_t i = 0; i < new_nodes_vec.size(); ++i) {
        result_buf(i, 0) = new_nodes_vec[i][0];
        result_buf(i, 1) = new_nodes_vec[i][1];
    }
    return result;
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





// Version hybride avec partage de meilleurs résultats
std::vector<pybind11::array_t<double>> optimize_nodes_parallel_hybrid(
    pybind11::array_t<double> nodes,
    pybind11::array_t<double> targets,
    double dist_threshold,
    double stepsize,
    size_t n,
    size_t n_threads
) {
    std::vector<pybind11::array_t<double>> results(n_threads);
    std::vector<std::thread> threads;
    
    // Partage périodique des meilleurs résultats
    std::mutex best_mutex;
    pybind11::array_t<double> global_best = nodes;
    double global_best_error = cout_graph_p2(nodes, targets);
    
    const size_t sync_interval = n / 10;
    
    pybind11::gil_scoped_release release;
    
    auto worker = [&](size_t thread_id) {
        try {
            pybind11::gil_scoped_acquire acquire;
            
            std::mt19937 gen(std::random_device{}() + thread_id);
            auto local_best = nodes;
            double local_best_error = cout_graph_p2(local_best, targets);
            
            for (size_t i = 0; i < n; ++i) {
                // Synchronisation périodique
                if (i % sync_interval == 0 && i > 0) {
                    std::lock_guard<std::mutex> lock(best_mutex);
                    if (global_best_error < local_best_error) {
                        local_best = global_best;
                        local_best_error = global_best_error;
                    } else if (local_best_error < global_best_error) {
                        global_best = local_best;
                        global_best_error = local_best_error;
                    }
                }
                
                double progress = static_cast<double>(i) / n;
                double current_stepsize = stepsize * std::exp(-2.0 * progress);
                
                auto new_nodes = mutate_nodes(local_best, current_stepsize);
                double new_error = cout_graph_p2(new_nodes, targets);
                
                if (new_error < local_best_error && 
                    is_graph_connected_bfs(new_nodes, dist_threshold)) {
                    local_best = std::move(new_nodes);
                    local_best_error = new_error;
                }
            }
            
            results[thread_id] = local_best;
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
    
    return results;
}



/*
optimiser l'erreur avec un algorithme génétique
retournes un seul graphe

on a une liste d'individu, générés aléatoirement de taille population_size,
on fait muter tout les individus plusieurs fois, en s'assurant qu'ils restent connectés et que
leurs erreurs sont inférieures à la meilleure erreur trouvée jusqu'à présent (keep_best_ratio)

tout les n/10 étapes on garde les meilleurs keep_best_ratio de graphes qui ont les meilleurs erreurs
et on remplace les autres par des copies des meilleurs


*/
pybind11::array_t<double> optimize_nodes_genetic(
    pybind11::array_t<double> nodes,
    pybind11::array_t<double> targets,
    double dist_threshold,
    double stepsize,
    size_t n,
    size_t population_size,
    double keep_best_ratio
) {
    // 1. Initialize population
    std::vector<pybind11::array_t<double>> population(population_size);
    std::vector<double> errors(population_size);
    for (size_t i = 0; i < population_size; ++i) {
        // Start with a mutated version of the input nodes
        auto ind = mutate_nodes(nodes, stepsize);
        // Ensure connectivity
        int tries = 0;
        while (!is_graph_connected_bfs(ind, dist_threshold) && tries < 10) {
            ind = mutate_nodes(nodes, stepsize);
            ++tries;
        }
        population[i] = ind;
        errors[i] = cout_graph_p2(ind, targets);
    }

    size_t elitism_interval = std::max(size_t(1), n / 10);
    size_t n_elite = std::max(size_t(1), static_cast<size_t>(population_size * keep_best_ratio));

    // 2. Main loop
    for (size_t gen = 0; gen < n; ++gen) {
        double progress = static_cast<double>(gen) / n;
        double current_stepsize = stepsize * std::exp(-2.0 * progress);

        // Mutation and selection
        for (size_t i = 0; i < population_size; ++i) {
            auto candidate = mutate_nodes(population[i], current_stepsize);
            if (!is_graph_connected_bfs(candidate, dist_threshold)) continue;
            double candidate_error = cout_graph_p2(candidate, targets);
            if (candidate_error < errors[i]) {
                population[i] = candidate;
                errors[i] = candidate_error;
            }
        }

        // Elitism: every elitism_interval generations
        if ((gen + 1) % elitism_interval == 0) {
            // Sort indices by error
            std::vector<size_t> idx(population_size);
            std::iota(idx.begin(), idx.end(), 0);
            std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return errors[a] < errors[b]; });
            // Copy elite
            std::vector<pybind11::array_t<double>> elite;
            std::vector<double> elite_errors;
            for (size_t i = 0; i < n_elite; ++i) {
                elite.push_back(population[idx[i]]);
                elite_errors.push_back(errors[idx[i]]);
            }
            // Replace the rest with random elite
            for (size_t i = n_elite; i < population_size; ++i) {
                size_t e = i % n_elite;
                population[idx[i]] = elite[e];
                errors[idx[i]] = elite_errors[e];
            }
        }
    }

    // 3. Return the best individual
    size_t best_idx = std::min_element(errors.begin(), errors.end()) - errors.begin();
    return population[best_idx];
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
    // Map: shape_str -> tuple(score, graph, age_min, age_max)
    std::unordered_map<std::string, std::tuple<double, pybind11::array_t<double>, size_t, size_t, size_t>> bests;

    for (size_t i = 0; i < history.size(); ++i) {
        auto graph = history[i];
        auto shape = get_shape(graph, dist_threshold);
        std::string shape_str = get_shape_string(shape);
        double score = cout_graph_p2(graph, targets);

        auto it = bests.find(shape_str);
        if (it == bests.end()) {
            // First occurrence: set everything
            bests[shape_str] = std::make_tuple(score, graph, i, i, 1);
        } else {
            double prev_score = std::get<0>(it->second);
            pybind11::array_t<double> prev_graph = std::get<1>(it->second);
            size_t age_min = std::get<2>(it->second);
            size_t age_max = i; // always update age_max
            size_t age = std::get<4>(it->second);
            if (score < prev_score) {
                // New best score for this shape
                bests[shape_str] = std::make_tuple(score, graph, age_min, age_max, age + 1);
            } else {
                // Keep previous best, just update age_max
                bests[shape_str] = std::make_tuple(prev_score, prev_graph, age_min, age_max, age + 1);
            }
        }
    }

    // Fill the Python dict
    for (const auto& kv : bests) {
        pybind11::dict entry;
        entry["graph"] = std::get<1>(kv.second); // the graph
        entry["score"] = std::get<0>(kv.second);  // the score
        entry["age_min"] = std::get<2>(kv.second); // first occurrence
        entry["age_max"] = std::get<3>(kv.second); // last occurrence
        entry["age"] = std::get<4>(kv.second); // number of times the shape has been seen
        entry["shape"] = kv.first; // the shape string
        result[pybind11::str(kv.first)] = entry;
    }
    return result;
}





// ===== MODULE PYBIND11 =====

PYBIND11_MODULE(graphx, m) {
    m.def("version", &version, "Retourne la version du module");
    m.def("distance", &distance, "Distance entre deux points");
    m.def("is_connected", &is_connected, "Vérifie si deux points sont connectés");
    m.def("is_graph_connected_bfs", &is_graph_connected_bfs, "Vérifie si un graph est connecté");
    m.def("cout_graph_p2", &cout_graph_p2, "Calcul le cout d'un graph");
    m.def("mutate_nodes", &mutate_nodes, "Muter les noeuds");
    m.def("optimize_nodes", &optimize_nodes, "Optimiser les noeuds");
    m.def("optimize_nodes_parallel_hybrid", &optimize_nodes_parallel_hybrid, "Optimiser les noeuds en parallèle avec partage de meilleurs résultats");
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

    m.def("optimize_nodes_genetic", &optimize_nodes_genetic, "Optimiser les noeuds avec un algorithme génétique",
        pybind11::arg("nodes"), pybind11::arg("targets"), pybind11::arg("dist_threshold"), pybind11::arg("stepsize"), pybind11::arg("n"), pybind11::arg("population_size"), pybind11::arg("keep_best_ratio"));

    m.def("chose_node_near_node_weighted", &chose_node_near_node_weighted, "Retourne l'index du noeud choisi pondéré par la proximité au start_node (qui doit être un point 1D)",
        pybind11::arg("nodes"), pybind11::arg("start_node"), pybind11::arg("dist_threshold"), pybind11::arg("sigma"));

    m.def("random_points_in_disk_with_attraction_point", &random_points_in_disk_with_attraction_point, "Génère des points dans un disque avec un point attractif",
        pybind11::arg("disk_center"), pybind11::arg("radius"), pybind11::arg("attract_point"), pybind11::arg("sigma"));
    
    m.def("safe_mutate_nodes", &safe_mutate_nodes, "Muter les noeuds de sorte que le graph soit connecté",
        pybind11::arg("nodes"), pybind11::arg("start_idx"), pybind11::arg("radius"), pybind11::arg("sigma"));

    m.def("get_node_contact_array", &get_node_contact_array, "Retourne un tableau d'index des noeuds ordonnés par le nombre d'arrets minimum pour atteindre le noeud de départ",
        pybind11::arg("nodes"), pybind11::arg("start_idx"), pybind11::arg("dist_threshold"));
}