#include <iostream>
#include <vector>
#include <queue>
#include <utility>

using namespace std;

const int dx[] = {-1, 1, 0, 0};
const int dy[] = {0, 0, -1, 1};

bool is_valid(int x, int y, int rows, int cols) {
    return x >= 0 && x < rows && y >= 0 && y < cols;
}

void solve_maze(vector<vector<char>>& maze) {
    int rows = maze.size();
    int cols = maze[0].size();
    pair<int, int> start, end;

    // Find start and end positions
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (maze[i][j] == 'S') start = {i, j};
            if (maze[i][j] == 'E') end = {i, j};
        }
    }

    // BFS
    queue<pair<int, int>> q;
    vector<vector<pair<int, int>>> parent(rows, vector<pair<int, int>>(cols, {-1, -1}));
    q.push(start);

    while (!q.empty()) {
        auto [x, y] = q.front();
        q.pop();

        if (x == end.first && y == end.second) break;

        for (int i = 0; i < 4; ++i) {
            int nx = x + dx[i], ny = y + dy[i];
            if (is_valid(nx, ny, rows, cols) && (maze[nx][ny] == '.' || maze[nx][ny] == 'E') && parent[nx][ny].first == -1) {
                parent[nx][ny] = {x, y};
                q.push({nx, ny});
            }
        }
    }

    // Reconstruct path
    if (parent[end.first][end.second].first != -1) {
        int x = end.first, y = end.second;
        while (x != start.first || y != start.second) {
            if (maze[x][y] != 'S' && maze[x][y] != 'E') maze[x][y] = '+';
            auto [px, py] = parent[x][y];
            x = px;
            y = py;
        }
        
        // Print solved maze
        for (const auto& row : maze) {
            for (char c : row) {
                cout << c;
            }
            cout << endl;
        }
    } else {
        cout << "The maze has no solution." << endl;
        // Print empty maze
        for (const auto& row : maze) {
            for (char c : row) {
                cout << c;
            }
            cout << endl;
        }
    }
}

int main() {
    vector<vector<char>> maze = {
        {'#', '#', '#', '#', '#', '#', '#'},
        {'#', 'S', '.', '.', '.', 'E', '#'},
        {'#', '#', '#', '#', '#', '#', '#'}
    };

    solve_maze(maze);

    return 0;
}