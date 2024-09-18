import java.util.*;
import java.util.stream.Collectors;

class Edge {
    int source, destination, weight;

    public Edge(int source, int destination, int weight) {
        this.source = source;
        this.destination = destination;
        this.weight = weight;
    }
}

class Graph {
    List<List<Edge>> adjacencyList;

    public Graph(int vertices) {
        adjacencyList = new ArrayList<>(vertices);
        for (int i = 0; i < vertices; i++) {
            adjacencyList.add(new ArrayList<>());
        }
    }

    public void addEdge(int source, int destination, int weight) {
        adjacencyList.get(source).add(new Edge(source, destination, weight));
        adjacencyList.get(destination).add(new Edge(destination, source, weight));
    }

    public List<Integer> dijkstra(int startVertex) {
        int vertices = adjacencyList.size();
        int[] distance = new int[vertices];
        int[] parent = new int[vertices];
        boolean[] visited = new boolean[vertices];

        Arrays.fill(distance, Integer.MAX_VALUE);
        Arrays.fill(parent, -1);
        distance[startVertex] = 0;

        PriorityQueue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(a -> a[1]));
        pq.add(new int[]{startVertex, 0});

        while (!pq.isEmpty()) {
            int[] current = pq.poll();
            int u = current[0];

            if (visited[u]) continue;

            visited[u] = true;

            for (Edge edge : adjacencyList.get(u)) {
                int v = edge.destination;
                int weight = edge.weight;

                if (distance[u] + weight < distance[v]) {
                    distance[v] = distance[u] + weight;
                    parent[v] = u;
                    pq.add(new int[]{v, distance[v]});
                }
            }
        }

        return Arrays.stream(parent).boxed().collect(Collectors.toList());
    }

    public List<Integer> getShortestPath(int startVertex, int endVertex) {
        List<Integer> parent = dijkstra(startVertex);
        List<Integer> path = new ArrayList<>();

        for (int at = endVertex; at != -1; at = parent.get(at)) {
            path.add(at);
        }

        Collections.reverse(path);
        return path;
    }
}

public class GPSNavigationSystem {
    public static void main(String[] args) {
        // Example graph representing a map
        Graph graph = new Graph(6);
        graph.addEdge(0, 1, 7);
        graph.addEdge(0, 2, 9);
        graph.addEdge(0, 5, 14);
        graph.addEdge(1, 2, 10);
        graph.addEdge(1, 3, 15);
        graph.addEdge(2, 3, 11);
        graph.addEdge(2, 5, 2);
        graph.addEdge(3, 4, 6);
        graph.addEdge(4, 5, 9);

        int startVertex = 0;
        int endVertex = 4;
        List<Integer> shortestPath = graph.getShortestPath(startVertex, endVertex);

        System.out.println("Shortest path from " + startVertex + " to " + endVertex + ": " + shortestPath);
    }
}