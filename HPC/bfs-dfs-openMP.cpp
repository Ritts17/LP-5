#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <omp.h>

using namespace std;

const int MAX = 100;
vector<int> graph[MAX];
bool visited_serial[MAX], visited_parallel[MAX];
int nodes;

// Serial BFS
void serial_bfs(int start) {
    fill(visited_serial, visited_serial + MAX, false);
    queue<int> q;
    visited_serial[start] = true;
    q.push(start);

    while (!q.empty()) {
        int v = q.front();
        q.pop();
        cout << v << " ";
        for (int u : graph[v]) {
            if (!visited_serial[u]) {
                visited_serial[u] = true;
                q.push(u);
            }
        }
    }
}

// Parallel BFS
void parallel_bfs(int start) {
    fill(visited_parallel, visited_parallel + MAX, false);
    queue<int> q;
    visited_parallel[start] = true;
    q.push(start);

    while (!q.empty()) {
        int v = q.front();
        q.pop();
        cout << v << " ";

        #pragma omp parallel for
        for (int i = 0; i < graph[v].size(); i++) {
            int u = graph[v][i];
            if (!visited_parallel[u]) {
                #pragma omp critical
                {
                    if (!visited_parallel[u]) {
                        visited_parallel[u] = true;
                        q.push(u);
                    }
                }
            }
        }
    }
}

// Serial DFS
void serial_dfs(int start) {
    fill(visited_serial, visited_serial + MAX, false);
    stack<int> s;
    s.push(start);

    while (!s.empty()) {
        int v = s.top();
        s.pop();
        if (!visited_serial[v]) {
            visited_serial[v] = true;
            cout << v << " ";
            for (int i = graph[v].size() - 1; i >= 0; --i) {
                int u = graph[v][i];
                if (!visited_serial[u])
                    s.push(u);
            }
        }
    }
}

// Parallel DFS
void parallel_dfs(int start) {
    fill(visited_parallel, visited_parallel + MAX, false);
    stack<int> s;
    s.push(start);

    while (!s.empty()) {
        int v = s.top();
        s.pop();
        if (!visited_parallel[v]) {
            visited_parallel[v] = true;
            cout << v << " ";
            #pragma omp parallel for
            for (int i = 0; i < graph[v].size(); i++) {
                int u = graph[v][i];
                if (!visited_parallel[u]) {
                    #pragma omp critical
                    s.push(u);
                }
            }
        }
    }
}

int main() {
    int edges, u, v, start;
    cout << "Enter number of nodes and edges: ";
    cin >> nodes >> edges;

    cout << "Enter " << edges << " edges (u v):\n";
    for (int i = 0; i < edges; i++) {
        cin >> u >> v;
        graph[u].push_back(v);
        graph[v].push_back(u); // For undirected graph
    }

    cout << "Enter starting node for traversal: ";
    cin >> start;

    // BFS Comparison
    double start_time = omp_get_wtime();
    cout << "\nSerial BFS: ";
    serial_bfs(start);
    double serial_bfs_time = omp_get_wtime() - start_time;

    start_time = omp_get_wtime();
    cout << "\nParallel BFS: ";
    parallel_bfs(start);
    double parallel_bfs_time = omp_get_wtime() - start_time;

    // DFS Comparison
    start_time = omp_get_wtime();
    cout << "\n\nSerial DFS: ";
    serial_dfs(start);
    double serial_dfs_time = omp_get_wtime() - start_time;

    start_time = omp_get_wtime();
    cout << "\nParallel DFS: ";
    parallel_dfs(start);
    double parallel_dfs_time = omp_get_wtime() - start_time;

    // Time comparison
    cout << "\n\nTime Comparison (in seconds):";
    cout << "\nSerial BFS: " << serial_bfs_time;
    cout << "\nParallel BFS: " << parallel_bfs_time;
    cout << "\nSerial DFS: " << serial_dfs_time;
    cout << "\nParallel DFS: " << parallel_dfs_time << "\n";

    return 0;
}

/*
Enter number of nodes and edges: 6
7
Enter 7 edges (u v):
0 1 
0 2
1 2
1 3
2 4
2 5
3 4
Enter starting node for traversal: 0

Serial BFS: 0 1 2 3 4 5 
Parallel BFS: 0 2 1 4 5 3 

Serial DFS: 0 1 2 4 3 5 
Parallel DFS: 0 1 3 4 2 5 

Time Comparison (in seconds):
Serial BFS: 2.9591e-05
Parallel BFS: 0.0179986
Serial DFS: 0.00801875
Parallel DFS: 0.0171293
*/
