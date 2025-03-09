function minTotalDistance(grid) {
        const rows = [];
        const cols = [];
        for (let i = 0; i < grid.length; i++){
            for(let j = 0; j < grid[0].length; j++){
                if(grid[i][j] === 1){
                    rows.push(i);
                }
            }
        }
         for (let j = 0; j < grid[0].length; j++){
            for(let i = 0; i < grid.length; i++){
                if(grid[i][j] === 1){
                    cols.push(j);
                }
            }
        }
        const midRow = rows[Math.floor(rows.length/2)];
        const midCol = cols[Math.floor(cols.length/2)];

        let distance = 0;
        for (const r of rows){
            distance += Math.abs(r-midRow);
        }
        for (const c of cols){
            distance += Math.abs(c-midCol);
        }
        return distance;
}

// Test cases
console.log(minTotalDistance([[1,0,0,0,1],[0,0,0,0,0],[0,0,1,0,0]])); // 6
console.log(minTotalDistance([[1,1]])); // 1
