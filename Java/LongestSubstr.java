
class LongestSubstr{
    public static int lengthOfLongestSubstringTwoDistinct(s String) {
        if(s==null || s.length() == 0){
            return 0;
        }
        Map<Character, Integer> map = new HashMap<>();
        int left = 0, maxLength = 0;
        for (int right = 0; right < s.length(); right++){
            char rightChar = s.charAt(right);
            map.put(rightChar, map.getOrDefault(c,0)+1);
            while(map.size() > 2){
                char leftMost = s.charAt(left);
                map.put(leftChar, map.get(leftMost)-1);
                if (map.get(leftMost) == 0){
                    map.remove(leftMost);
                }
                left++;
            }
            maxLength = Math.max(maxLength, right-left+1);
        }
        return maxLength;
    }
    public static void main(String[] args) {
        System.out.println(lengthOfLongestSubstringTwoDistinct("eceba"));
        System.out.println(lengthOfLongestSubstringTwoDistinct("ccaabbb"));
        System.out.println(lengthOfLongestSubstringTwoDistinct("abc"));
        System.out.println(lengthOfLongestSubstringTwoDistinct("a"));
        System.out.println(lengthOfLongestSubstringTwoDistinct("aa"));
        System.out.println(lengthOfLongestSubstringTwoDistinct("aabbcc"));
        System.out.println(lengthOfLongestSubstringTwoDistinct("abababab"));
        System.out.println(lengthOfLongestSubstringTwoDistinct("aabb"));
        System.out.println(lengthOfLongestSubstringTwoDistinct("aabbccddeeff"));
        System.out.println(lengthOfLongestSubstringTwoDistinct("aabbccddeeffgghhiijj"));
        System.out.println(lengthOfLongestSubstringTwoDistinct(""));
    }
}