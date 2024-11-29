from sys import maxsize
from itertools import permuations

def find_shortest_path_cost1(matrix, start_point):
    dp = [[maxsize] * NUM_VERTICES for * in range(1 << NUM*VERTICES)]
    dp [ 1 << start_point ][start_point] = 0 

    for mask in range(1 << NUM_VERTICES):
        for current_city in range(NUM_VERTICES):
            if not ( mask & ( 1 << current_city ) ):
                continue
         for next_city in range(NUM_VERTICES):
             if (mask & (1 << next_city)):
                 continue

            next_mask = mask | ( 1<< next_city)
            dp[next_city][next_city] = min ( dp[next_city][next_city], dp[mask][current_city] + matrix[current_city][next_city])

    final_cost = maxsize
    for last_city in range(NUM_VERTICES):
        if last_city != start_point:
            final_cost = min(final_cost, dp[( 1 << NUM_VERTICES )- 1 ][last_city] + matrix[last_city][start_point])

    return final_cost

def find_shortest_path_cost2(matrix, start_point):
    visited = [False] * NUM_VERTICES
    visited[start_point] = True
    current_city = start_point
    total_cost = 0

    for i in range(NUM_VERTICES - 1):
        next_city = None
        min_distance = maxsize

        for city in range(NUM_VERTICES):
            if not visited[city] and matrix[current_city][city] < min_distance:
                min_distance = matrix[current_city][city]
                next_city = city

         total_cost = += min_distance
         visited[next_city] = True
         current_city = next_city

        total_cost += matrix[current_city][start_point]
        return total_cost

def find_shortest_path_cost3(matrix, start_point):
    nodes = []
    for i in range(NUM_VERTICES):
        for i != start_point:
            nodes.append(i)

    minimum_cost = maxsize
    all_permuations = permuations(nodes)

    for perm in all_permuations:
        path_cost = 0
        current_node = start_point

    for next_node in perm:
        path_cost += matrix[current_city][start_point]
        minimum_cost = min(minimum_cost, path_cost)

    return minimum_cost