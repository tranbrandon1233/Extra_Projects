
import java.util.HashMap;
import java.util.Map;


public class Main {
       public static int minCostII(int[][] costs) {
       if(costs == null || costs.length == 0) return 0;
        int n = costs.length;
        int k = costs[0].length;
        int prevMin = Integer.MAX_VALUE, prevSecondMin = Integer.MAX_VALUE, prevColor = -1;
        for (int j = 0; j < k; j++){
            if(costs[0][j] < prevMin){
                prevSecondMin = prevMin;
                prevMin = costs[0][j];
                prevColor = j;
            }
            else if(costs[0][j] < prevSecondMin){
                prevSecondMin = costs[0][j];
            }
        }
        for(int i = 1; i < n; i++){
            int currMin = Integer.MAX_VALUE, currSecondMin = Integer.MAX_VALUE, currColor = -1;
            for (int j = 0; j < k; j++){
                int cost = costs[i][j] + (j==prevColor ? prevSecondMin : prevMin);
                if(cost < currMin){
                    currSecondMin = currMin;
                    currMin = cost;
                    currColor = j;
                }
                else if(cost < currSecondMin){
                    currSecondMin = cost;
                }
            } 
            prevMin = currMin;
            prevSecondMin = currSecondMin;
            prevColor = currColor;
        }
        return prevMin;
    }

    public static void main(String[] args) {
        int[][] costs1 = {{1,5,3}, {2,9,4}};
        int[][] costs2 = {{1,3}, {2,4}};
        System.out.println(minCostII(costs1));
        System.out.println(minCostII(costs2));
    }
} 
