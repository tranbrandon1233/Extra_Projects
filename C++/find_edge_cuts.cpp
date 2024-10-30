#include <vector>
#include <iostream>
#include <algorithm>
using namespace std;

class Graph {
    int V;
    vector<vector<int>> adj;
    void bridgeUtil(int u, vector<bool>& visited, vector<int>& disc,
        vector<int>& low, vector<int>& parent, vector<pair<int, int>>& bridges);

public:
    Graph(int V);
    void addEdge(int v, int w);
    vector<pair<int, int>> findBridges();
};

Graph::Graph(int V) {
    this->V = V;
    adj.resize(V);
}

void Graph::addEdge(int v, int w) {
    adj[v].push_back(w);
    adj[w].push_back(v);
}

void Graph::bridgeUtil(int u, vector<bool>& visited, vector<int>& disc,
    vector<int>& low, vector<int>& parent, vector<pair<int, int>>& bridges) {
    static int time = 0;
    visited[u] = true;
    disc[u] = low[u] = ++time;

    for (int v : adj[u]) {
        if (!visited[v]) {
            parent[v] = u;
            bridgeUtil(v, visited, disc, low, parent, bridges);

            low[u] = min(low[u], low[v]);

            // If the lowest vertex reachable from subtree under v is below v in DFS tree,
            // then u-v is a bridge
            if (low[v] > disc[u]) {
                bridges.push_back({ min(u,v), max(u,v) });
            }
        }
        else if (v != parent[u]) {
            low[u] = min(low[u], disc[v]);
        }
    }
}

vector<pair<int, int>> Graph::findBridges() {
    vector<bool> visited(V, false);
    vector<int> disc(V);
    vector<int> low(V);
    vector<int> parent(V, -1);
    vector<pair<int, int>> bridges;

    for (int i = 0; i < V; i++) {
        if (!visited[i]) {
            bridgeUtil(i, visited, disc, low, parent, bridges);
        }
    }

    // Sort bridges for consistent output
    sort(bridges.begin(), bridges.end());
    return bridges;
}

int main() {
    int V, E;
	cout << "Enter number of vertices and edges: ";
    cin >> V >> E;

    Graph g(V);

    // Input edges (0-based indexing)
    for (int i = 0; i < E; i++) {
        int u, v;
		cout << "Enter edge " << i + 1 << " of " << E << ": ";
        cin >> u >> v;
        // Convert to 0-based indexing
        u--; v--;
        g.addEdge(u, v);
    }

    vector<pair<int, int>> bridges = g.findBridges();

    if (bridges.empty()) {
        cout << "\nNo bridges found\n";
    }
    else {
        cout << "\nBridges are:\n";
        for (auto bridge : bridges) {
            // Convert back to 1-based indexing for output
            cout << bridge.first + 1 << " " << bridge.second + 1 << "\n";
        }
    }

    return 0;
}