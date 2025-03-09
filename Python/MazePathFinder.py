def uniquePathsWithObstacles( obstacleGrid) -> int:
        if not obstacleGrid or obstacleGrid[0][0] == 1:
            return 0
        m, n = len(obstacleGrid), len(obstacleGrid[0])
        dp = [[0]*n for _ in range(m)] # m x n DP table
        dp[0][0] = 1 # starting point
        
        for i in range(1,m):
            if obstacleGrid[i][0] == 0 and dp[i-1][0] == 1:
                dp[i][0] = 1
        for j in range (1,n):
            if obstacleGrid[0][j] == 0 and dp[0][j-1]==1:
                dp[0][j] = 1
        for i in range(1,m):
            for j in range(1,n):
                if obstacleGrid[i][j] == 0:
                    dp[i][j] = dp[i-1][j] + dp[i][j-1]
                else:
                    dp[i][j] = 0
        return dp[m-1][n-1]
    
grid = [[0,0,0],[0,1,0],[0,0,0]]
print(uniquePathsWithObstacles(grid))