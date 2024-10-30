#include <iostream>
#include <vector>
#include <queue>
#include <map>
#include <tuple>

using namespace std;

int minTurns(vector<pair<int, int>>& ladders, vector<pair<int, int>>& snakes) {
    map<int, int> board; // Store ladder and snake connections
    for (auto& ladder : ladders) board[ladder.first] = ladder.second;
    for (auto& snake : snakes) board[snake.first] = snake.second;

    queue<tuple<int, int, int>> q; // Queue for BFS (position, turns, extraRolls)
    q.push({ 1, 0, 0 });

    vector<vector<int>> visited(101, vector<int>(4, 0)); // Track visited with extra rolls
    visited[1][0] = 1;

    while (!q.empty()) {
        int pos = get<0>(q.front());
        int turns = get<1>(q.front());
        int extraRolls = get<2>(q.front());
        q.pop();

        if (pos == 100) return turns;

        for (int dice = 1; dice <= 6; ++dice) {
            int nextPos = pos + dice;
            int nextExtraRolls = extraRolls;

            if (nextPos > 100) continue; // Discard moves past 100

            if (board.count(nextPos)) {
                if (board[nextPos] > nextPos) nextExtraRolls++; // Ladder
                nextPos = board[nextPos];
            }

            if (dice == 6) nextExtraRolls++; // Rolled a 6

            if (nextExtraRolls <= 3 && !visited[nextPos][nextExtraRolls]) {
                visited[nextPos][nextExtraRolls] = 1;
                if (board.count(pos) && board[pos] < pos) { // Snake
                    q.push({ nextPos, turns + 1, 0 }); // Reset extra rolls after snake
                }
                else {
                    q.push({ nextPos, turns + (nextExtraRolls - extraRolls == 0 ? 1 : 0), nextExtraRolls });
                }
            }
        }
    }

    return -1; // Should not reach here if a solution exists
}

int main() {
    // Same ladders and snakes as before
    vector<pair<int, int>> ladders = { {2, 38}, {7, 14}, {8, 31}, {28, 84}, {21, 42}, {51, 67}, {72, 91}, {80, 99} };
    vector<pair<int, int>> snakes = { {17, 7}, {54, 34}, {62, 19}, {64, 60}, {87, 36}, {93, 73}, {95, 75}, {98, 79} };

    int minTurnsRequired = minTurns(ladders, snakes);

    if (minTurnsRequired != -1) {
        cout << "Minimum turns required to win: " << minTurnsRequired << endl;
    }
    else {
        cout << "No solution found (should not happen in a valid game)." << endl;
    }

    return 0;
}