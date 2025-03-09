def canFinish(numCourses: int, prerequisites) -> bool:
        graph = [[] for _ in range(numCourses)]
        for course, prereq in prerequisites:
            graph[prereq].append(course)

        visited = [0]*numCourses
        def dfs(course):
            if visited[course] == 1:
                return False
            if visited[course] == 2:
                return True
            visited[course] = 1
            for next_course in graph[course]:
                if not dfs(next_course):
                    return False
            visited[course] = 2
            return True
        for course in range(numCourses):
            if visited[course] == 0 and not dfs(course):
                    return False
        return True

# Example usage:
print(canFinish(2, [[1,0]])) # True
print(canFinish(2, [[1,0],[0,1]])) # False