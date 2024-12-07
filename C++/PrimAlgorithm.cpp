#include <bits/stdc++.h>
#include <random>
using namespace std;

#define V 5

struct Location {
    double latitude;
    double longitude;
};

double calculateDistance(const Location& loc1, const Location& loc2) {
    double dx = loc1.longitude - loc2.longitude;
    double dy = loc1.latitude - loc2.latitude;
    return sqrt(dx * dx + dy * dy);
}

int calculateWeight(double distance, int traffic) {
    // Traffic multiplier ranges from 1.0 to 2.0 based on traffic level (0-100)
    double trafficMultiplier = 1.0 + (traffic / 100.0);
    return static_cast<int>(distance * trafficMultiplier * 100); // Multiply by 100 to avoid decimals
}

int minKey(vector<int>& key, vector<bool>& mstSet) {
    int min = INT_MAX, min_index;
    for (int v = 0; v < V; v++)
        if (mstSet[v] == false && key[v] < min)
            min = key[v], min_index = v;
    return min_index;
}

int printMST(vector<int>& parent, vector<vector<int>>& graph) {
    int totalWeight = 0;
    cout << "Edge \tWeight\n";
    for (int i = 1; i < V; i++) {
        cout << parent[i] << " - " << i << " \t"
             << graph[parent[i]][i] << " \n";
        totalWeight += graph[parent[i]][i];
    }
    return totalWeight;
}

void primMST(vector<vector<int>>& graph) {
    vector<int> parent(V);
    vector<int> key(V);
    vector<bool> mstSet(V);

    for (int i = 0; i < V; i++)
        key[i] = INT_MAX, mstSet[i] = false;

    key[0] = 0;
    parent[0] = -1;

    for (int count = 0; count < V - 1; count++) {
        int u = minKey(key, mstSet);
        mstSet[u] = true;

        for (int v = 0; v < V; v++)
            if (graph[u][v] && mstSet[v] == false && graph[u][v] < key[v])
                parent[v] = u, key[v] = graph[u][v];
    }

    int totalWeight = printMST(parent, graph);
    cout << "\nTotal weight of the Minimum Spanning Tree: " << totalWeight << endl;
}

int main() {
    // Random number generation setup
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis_coord(-10.0, 10.0);  // Location coordinates between -10 and 10
    uniform_int_distribution<> dis_traffic(0, 100);      // Traffic levels between 0 and 100

    // Generate random locations
    vector<Location> locations;
    cout << "Generated Locations:\n";
    for (int i = 0; i < V; i++) {
        Location loc = {dis_coord(gen), dis_coord(gen)};
        locations.push_back(loc);
        cout << "Location " << i << ": (" << loc.latitude << ", " << loc.longitude << ")\n";
    }

    // Generate random traffic levels and calculate graph weights
    vector<vector<int>> graph(V, vector<int>(V, 0));
    cout << "\nTraffic Levels Matrix:\n";
    for (int i = 0; i < V; i++) {
        for (int j = i + 1; j < V; j++) {
            int traffic = dis_traffic(gen);
            double distance = calculateDistance(locations[i], locations[j]);
            int weight = calculateWeight(distance, traffic);
            graph[i][j] = graph[j][i] = weight;
            cout << traffic << "\t";
        }
        cout << endl;
    }

    cout << "\nCalculated Weights Matrix:\n";
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
            cout << graph[i][j] << "\t";
        }
        cout << endl;
    }

    cout << "\nMinimum Spanning Tree:\n";
    primMST(graph);

    return 0;
}