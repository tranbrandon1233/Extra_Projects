#include <iostream>
#include <vector>
#include <queue>
#include <limits>

using namespace std;

// Structure to represent an edge
struct Edge {
    int destination;
    int weight;
};

// Structure to represent a node in the priority queue
struct Node {
    int vertex;
    int key;

    bool operator>(const Node& other) const {
        return key > other.key;
    }
};

// Function to implement Prim's algorithm
vector<pair<int, int>> primMST(vector<vector<Edge>>& graph) {
    int n = graph.size();

    // Priority queue to store nodes with their keys
    priority_queue<Node, vector<Node>, greater<Node>> pq;

    // Array to store the key value of each vertex
    vector<int> key(n, numeric_limits<int>::max());

    // Array to store the parent of each vertex in the MST
    vector<int> parent(n, -1);

    // Array to track whether a vertex is included in the MST
    vector<bool> inMST(n, false);

    // Start with the first vertex
    key[0] = 0;
    pq.push({ 0, 0 });

    while (!pq.empty()) {
        // Extract the vertex with minimum key value
        int u = pq.top().vertex;
        pq.pop();

        // Include the vertex in MST
        inMST[u] = true;

        // Iterate over all adjacent vertices of u
        for (const Edge& edge : graph[u]) {
            int v = edge.destination;
            int weight = edge.weight;

            // If v is not in MST and the weight of edge (u, v) is less than key[v]
            if (!inMST[v] && weight < key[v]) {
                // Update key[v] and parent[v]
                key[v] = weight;
                parent[v] = u;
                pq.push({ v, key[v] });
            }
        }
    }

    // Construct the MST from parent array
    vector<pair<int, int>> mst;
    for (int i = 1; i < n; ++i) {
        mst.push_back({ parent[i], i });
    }

    return mst;
}

int main() {
    // Example graph represented as adjacency list
    vector<vector<Edge>> graph = {
      {{1, 4}, {7, 8}},
      {{0, 4}, {2, 8}, {7, 11}},
      {{1, 8}, {3, 7}, {5, 4}, {8, 2}},
      {{2, 7}, {4, 9}, {5, 14}},
      {{3, 9}, {5, 10}},
      {{2, 4}, {3, 14}, {4, 10}, {6, 2}},
      {{5, 2}, {7, 1}, {8, 6}},
      {{0, 8}, {1, 11}, {6, 1}, {8, 7}},
      {{2, 2}, {6, 6}, {7, 7}}
    };

    // Find the Minimum Spanning Tree using Prim's algorithm
    vector<pair<int, int>> mst = primMST(graph);

    // Print the MST edges
    cout << "Minimum Spanning Tree edges:" << endl;
    for (const auto& edge : mst) {
        cout << edge.first << " - " << edge.second << endl;
    }

    return 0;
}