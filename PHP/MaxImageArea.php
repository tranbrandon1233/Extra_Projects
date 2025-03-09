<?php
class Solution {
    public function minArea(array $image, int $x, int $y): int {
       $m = count($image);
        $n = count($image[0]);
        $low = 0;
        $high = $y;
        while($low <= $high){
            $mid = intdiv($low + $high,2);
            if(hasBlackInColumn($image, $mid, $m)){
                $high = $mid -1;
            }
            else{
                $low = $mid +1;
            }
        }
        $left = $low;
        $low = $y;
        $high = $n-1;
         while($low <= $high){
            $mid = intdiv($low + $high,2);
            if(hasBlackInColumn($image, $mid, $m)){
                $low = $mid +1;
            }
            else{
                $high= $mid -1;
            }
        }
        $right = $high;
        $low = 0;
        $high = $x;
        while($low <= $high){
            $mid = intdiv($low + $high,2);
            if(hasBlackInRow($image, $mid, $n)){
                $high = $mid -1;
            }
            else{
                $low = $mid +1;
            }
        }
        $top = $low;
        $low = $x;
        $high = $m-1;
      while($low <= $high){
            $mid = intdiv($low + $high,2);
            if(hasBlackInRow($image, $mid, $n)){
                $low= $mid +1;
            }
            else{
                $high = $mid -1;
            }
        }
        $bottom= $high;
        return ($right-$left+1)*($bottom-$top+1);
    }
    private function hasBlackInColumn($image, $col, $m){
        for($i = 0; $i < $m; $i++){
            if($image[$i][$col] === '1'){
                return true;
            }
        }
        return false;
    }

    private function hasBlackInRow($image, $row, $n){
        for($i = 0; $i < $n; $i++){
            if($image[$row][$i] === '1'){
                return true;
            }
        }
        return false;
    }
}
$image1 = [["0", "0", "1", "0"], ["0","1", "1", "0"], ["0","1","0","0"]];
$image2 = [["$1"]];
echo minArea($image1, 0,2) . "\n";

echo minArea($image2, 0,0)
?>