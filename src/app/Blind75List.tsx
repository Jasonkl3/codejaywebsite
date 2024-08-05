import Image from "next/image";
import CodeButton from './CodeButton';
import ListCheckbox from "./ListCheckbox";

enum Difficulty {
  Easy = 'Easy',
  Medium = 'Medium',
  Hard = 'Hard',
}

type Question = {
  leetcodeNumber: number; // aka the Id for the question
  title: string;
  topic: string; // we should make this a list of enums?
  difficulty: Difficulty;
  videoId: string; // youtubeVideo Id
  leetcodeUrl: string; // link to LC problem description
  codeSnippet: string; // code solution
}

export default function Blind75List() {

  // // sorts questions by topics
  // const getTopicMap = () => {
  //   const topicMap = new Map<string, Question[]>();
  //   questions.forEach((question: Question, index: number) => {
  //     if (topicMap.has(question.topic)) {
  //       const questions = topicMap.get(question.topic) ?? [];
  //       questions?.push(question);
  //       topicMap.set(question.topic, questions);
  //     } else {
  //       topicMap.set(question.topic, [question]);
  //     }
  //   });
  //   return topicMap;
  // }

  // const topicMap = getTopicMap();
  // console.log('topicMap', topicMap);

  const getDifficultyColor = (difficulty: string) => {
    if (difficulty === 'Easy') {
      return 'text-green-500';
    } else if (difficulty === 'Medium') {
      return 'text-orange-500';
    } else {
      return 'text-red-500'
    }
  }

  return (
    <div className="max-w-screen-lg mx-auto relative my-12 overflow-x-auto shadow-md sm:rounded-lg">
      <table className="w-full text-sm text-left rtl:text-right text-gray-500 dark:text-gray-400">
        <caption className="p-5 text-lg font-semibold text-left rtl:text-right text-gray-900 bg-white dark:text-white dark:bg-gray-800">
          LeetCode 75 Roadmap
          <details open className="my-2">
            <summary className="text-sm">What is this?</summary>
            <p className="mt-1 text-sm font-normal text-gray-500 dark:text-gray-400">
              This table contains a curated list of 75 <a href="https://www.leetcode.com" target="_blank" rel="noopener noreferrer" className="underline">LeetCode</a> questions that cover a wide range of topics and difficulty levels. Each question has been selected based on its relevance to common interview problems and its ability to help you build a strong foundation in data structures and algorithms.<br/>
            </p>
          </details>
          <details className="mb-2">
            <summary className="text-sm">Why do this?</summary>
            <p className="mt-1 text-sm font-normal text-gray-500 dark:text-gray-400">
              Knowing and practicing these questions is crucial for anyone preparing for technical coding interviews. These questions have been chosen because they frequently appear in interviews for top tech companies and provide a comprehensive overview of the problem-solving skills required to succeed.<br/>
            </p>
          </details>
          <details className="mb-2">
            <summary className="text-sm">How to use?</summary>
            <p className="mt-1 text-sm font-normal text-gray-500 dark:text-gray-400">
              Use this table as a roadmap for your interview preparation. Start by familiarizing yourself with each question, then implement your solutions in a coding environment. After solving each question, review the optimal solutions and understand the underlying concepts. This practice will help you improve your problem-solving abilities, coding efficiency, and confidence during interviews.
            </p>
          </details>
        </caption>
        <thead className="text-xs text-gray-700 uppercase bg-gray-50 dark:bg-gray-700 dark:text-gray-400">
          <tr>
            <th scope="col" className="p-4 text-2xl" title="Solved">
              {/* Solved column */}
              {/* suggestion: this should be a clear icon that opens a confirmation popup when clicked */}
              <svg
                viewBox="0 0 1024 1024"
                fill="currentColor"
                className="size-4"
              >
                <path d="M880 112H144c-17.7 0-32 14.3-32 32v736c0 17.7 14.3 32 32 32h736c17.7 0 32-14.3 32-32V144c0-17.7-14.3-32-32-32zM695.5 365.7l-210.6 292a31.8 31.8 0 01-51.7 0L308.5 484.9c-3.8-5.3 0-12.7 6.5-12.7h46.9c10.2 0 19.9 4.9 25.9 13.3l71.2 98.8 157.2-218c6-8.3 15.6-13.3 25.9-13.3H689c6.5 0 10.3 7.4 6.5 12.7z" />
              </svg>
            </th>
            <th scope="col" className="px-6 py-3">
              Question
            </th>
            <th scope="col" className="px-6 py-3">
              Topic
            </th>
            <th scope="col" className="px-6 py-3">
              Difficulty
            </th>
            <th scope="col" className="px-6 py-3">
              Video Solution
            </th>
            <th scope="col" className="px-6 py-3">
              Code
            </th>
          </tr>
        </thead>
        <tbody>
          {questions?.length > 0 && questions.map((question: Question, index: number) => (
            <tr id={`question-row-${question.leetcodeNumber}`} key={`question-row-${question.leetcodeNumber}`} className={`bg-white border-b dark:bg-gray-800 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-600`}>
              <td className="w-4 p-4">
                <ListCheckbox id={question.leetcodeNumber} />
              </td>
              <th scope="row" className="px-6 py-4 font-medium text-gray-900 whitespace-nowrap dark:text-white">
                <a 
                  href={question.leetcodeUrl ? question.leetcodeUrl : '/#'} 
                  target="_blank" 
                  rel="noopener 
                  noreferrer"
                  className={question.leetcodeUrl ? "hover:underline" : ""}
                >
                  {/* {question.title} */}
                  <div className="flex">
                    {question.title}
                    <div className="ml-2 hover:text-blue-500">
                      <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth="1.5" stroke="currentColor" className="size-4">
                        <path strokeLinecap="round" strokeLinejoin="round" d="M13.5 6H5.25A2.25 2.25 0 0 0 3 8.25v10.5A2.25 2.25 0 0 0 5.25 21h10.5A2.25 2.25 0 0 0 18 18.75V10.5m-10.5 6L21 3m0 0h-5.25M21 3v5.25" />
                      </svg>
                    </div>
                  </div>
                </a>
              </th>
              <td className="px-6 py-4">
                {question.topic}
              </td>
              <td className={`px-6 py-4 font-medium ${getDifficultyColor(question.difficulty)}`}>
                {question.difficulty}
              </td>
              <td className="px-6 py-4">
                {question.videoId && <a href={`https://www.youtube.com/shorts/${question.videoId}`} target="_blank" rel="noopener noreferrer">
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="size-6">
                    <path d="M4.5 4.5a3 3 0 0 0-3 3v9a3 3 0 0 0 3 3h8.25a3 3 0 0 0 3-3v-9a3 3 0 0 0-3-3H4.5ZM19.94 18.75l-2.69-2.69V7.94l2.69-2.69c.944-.945 2.56-.276 2.56 1.06v11.38c0 1.336-1.616 2.005-2.56 1.06Z" />
                  </svg>
                </a>}
              </td>
              <td className="px-6 py-4">
                  <CodeButton leetcodeNumber={question.leetcodeNumber} codeSnippet={question.codeSnippet} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const questions = [
  {
    leetcodeNumber: 217,
    title: 'Contains Duplicate',
    topic: 'Arrays & Hashing',
    difficulty: Difficulty.Easy,
    leetcodeUrl: 'https://leetcode.com/problems/contains-duplicate/',
    videoId: 'KgCEw82PgNw',
    codeSnippet: 
`class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        unique_nums = set()
        for num in nums:
            if num in unique_nums:
                return True
            unique_nums.add(num)
        return False
        # T: O(n), S:O(n)
`
  },
  {
    leetcodeNumber: 242,
    title: 'Valid Anagram',
    topic: 'Arrays & Hashing',
    difficulty: Difficulty.Easy,
    leetcodeUrl: 'https://leetcode.com/problems/valid-anagram/',
    videoId: 'ygZa-rqLmxQ',
    codeSnippet:
`from collections import Counter
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False
        charCount = dict(Counter(s))
        for ch in t:
            if ch not in charCount or charCount[ch] == 0:
                return False
            charCount[ch] -= 1
        return True
        # T: O(n), S:O(n)
`
  },
  {
    leetcodeNumber: 1,
    title: 'Two Sum',
    topic: 'Arrays & Hashing',
    difficulty: Difficulty.Easy,
    leetcodeUrl: 'https://leetcode.com/problems/two-sum/',
    videoId: '5qyLkBRzUXM',
    codeSnippet:
`class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        num_to_i = {}
        for i, num in enumerate(nums):
            comp = target - num
            if comp in num_to_i:
                return i, num_to_i[comp]
            num_to_i[num] = i
        # T: O(n), S:O(n)
`
  },
  {
    leetcodeNumber: 249,
    title: 'Group Anagrams',
    topic: 'Arrays & Hashing',
    difficulty: Difficulty.Medium,
    leetcodeUrl: 'https://leetcode.com/problems/group-anagrams/',
    videoId: 'uZKv7zO4ESY',
    codeSnippet:
`class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        sublists = {}
        for word in strs:
            char_frequency = [0] * 26
            for ch in word:
                char_frequency[ord(ch) - ord('a')] += 1
            key = tuple(char_frequency)
            if key in sublists:
                sublists[key].append(word)
            else:
                sublists[key] = [word]
        return sublists.values()
        # T: O(n*m), S:O(n*m)
`
  },
  {
    leetcodeNumber: 347,
    title: 'Top K Frequent Elements',
    topic: 'Arrays & Hashing',
    difficulty: Difficulty.Medium,
    leetcodeUrl: 'https://leetcode.com/problems/top-k-frequent-elements/',
    videoId: 'kqKqcJYUY08',
    codeSnippet:
`class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        count_arr = [[] for _ in range(len(nums))]
        num_count = {}
        # count numbers and store in count dictionary
        for num in nums:
            num_count[num] = 1 + num_count.get(num, 0)
        # perform bucket sort
        for num, count in num_count.items():
            count_arr[count-1].append(num)
        # reverse count array, and for each bucket append its items into top_k
        top_k = []
        for bucket in count_arr[::-1]:
            for num in bucket:
                top_k.append(num)
                if len(top_k) == k:
                    return top_k
`
  },
  {
    leetcodeNumber: 271,
    title: 'Encode and Decode Strings',
    topic: 'Arrays & Hashing',
    difficulty: Difficulty.Medium,
    leetcodeUrl: 'https://leetcode.com/problems/encode-and-decode-strings/description/',
    videoId: 'NgalRGKmErU',
    codeSnippet: `class Codec:
    def encode(self, strs: List[str]) -> str:
        """Encodes a list of strings to a single string.
        """
        s = ''
        for word in strs:
            s += str(len(word)) + '#' + word
        return s

    def decode(self, s: str) -> List[str]:
        """Decodes a single string to a list of strings.
        """
        original_list = []
        i = 0
        while i < len(s):
            j = i
            while s[j] != '#':
                j += 1
            word_len = int(s[i:j])
            i = j + 1
            j = i + word_len
            original_list.append(s[i:j])
            i = j
        return original_list

        


# Your Codec object will be instantiated and called as such:
# codec = Codec()
# codec.decode(codec.encode(strs))`
  },
  {
    leetcodeNumber: 0,
    title: 'Product of Array Except Self',
    topic: 'Arrays & Hashing',
    difficulty: Difficulty.Medium,
    leetcodeUrl: 'https://leetcode.com/problems/product-of-array-except-self/description/',
    videoId: '',
    codeSnippet: `class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        ans = [0] * len(nums)
        pre = [1] * len(nums) # prefix array
        suff = [1] * len(nums) # suffix array

        # populate pre: pre[i] = nums[i-1] * pref[i-1] // we don't include nums[i]
        prefix = pre[0]
        for i in range(1, len(nums)):
            pre[i] = nums[i-1] * prefix
            prefix = pre[i]

        # populate suff: suff[i] = nums[i-1] * suff[i-1] // don't include nums[i]
        suffix = suff[-1]
        for i in range(len(nums)-2,-1,-1):
            suff[i] = nums[i+1] * suffix
            suffix = suff[i]

        # populate ans: ans[i] = pre[i] * suf[i]
        for i in range(len(pre)):
            ans[i] = pre[i] * suff[i]
        return ans

`
  },
  {
    leetcodeNumber: 128,
    title: 'Longest Consecutive Sequence',
    topic: 'Arrays & Hashing',
    difficulty: Difficulty.Medium,
    leetcodeUrl: 'https://leetcode.com/problems/longest-consecutive-sequence/description/',
    videoId: '',
    codeSnippet: `class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        s = set(nums)
        max_length = 0
        for num in nums:
            if num-1 not in s:
                length = 0
                temp_num = num
                while temp_num in s:
                    length += 1
                    temp_num += 1
                max_length = max(max_length, length)
        return max_length`
  },
  {
    leetcodeNumber: 125,
    title: 'Valid Palindrome',
    topic: 'Two Pointers',
    difficulty: Difficulty.Easy,
    leetcodeUrl: 'https://leetcode.com/problems/valid-palindrome/',
    videoId: '',
    codeSnippet: `class Solution:
    def isPalindrome(self, s: str) -> bool:
        l, r = 0, len(s) - 1
        while l <= r:
            while l < r and not s[l].isalnum():
                l += 1
            while l < r and not s[r].isalnum():
                r -= 1
            if s[l].lower() != s[r].lower():
                return False
            l += 1
            r -= 1
        return True`
  },
  {
    leetcodeNumber: 15,
    title: '3Sum',
    topic: 'Two Pointers',
    difficulty: Difficulty.Medium,
    leetcodeUrl: 'https://leetcode.com/problems/3sum/',
    videoId: '',
    codeSnippet: `class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        ans = []
        nums.sort()
        seen = set()
        for i in range(len(nums)):
            if i > 0 and nums[i] == nums[i-1]:
                continue
            l = i+1
            r = len(nums)-1
            while l < r:
                tot = nums[i] + nums[l] + nums[r]
                if tot == 0:
                    ans.append([nums[i], nums[l], nums[r]])
                    l+=1
                    r-=1
                    while l < len(nums) and nums[l-1] == nums[l]:
                        l+=1
                    while r >= 0 and nums[r+1] == nums[r]:
                        r-=1
                elif tot > 0:
                    r-=1
                else:
                    l+=1
        return ans`
  },
  {
    leetcodeNumber: 11,
    title: 'Container With Most Water',
    topic: 'Two Pointers',
    difficulty: Difficulty.Medium,
    leetcodeUrl: 'https://leetcode.com/problems/container-with-most-water/description/',
    videoId: '',
    codeSnippet: `class Solution:
    def maxArea(self, height: List[int]) -> int:
        l = 0
        r = len(height) - 1
        maxArea = 0
        while l < r:
            area = min(height[l], height[r]) * (r - l)
            maxArea = max(maxArea, area)
            if height[l] <= height[r]:
                l += 1
            else:
                r -= 1
        return maxArea`
  },
  {
    leetcodeNumber: 121,
    title: 'Best Time to Buy and Sell Stock',
    topic: 'Sliding Window',
    difficulty: Difficulty.Easy,
    leetcodeUrl: 'https://leetcode.com/problems/best-time-to-buy-and-sell-stock/description/',
    videoId: '',
    codeSnippet: `class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        # Initialize variables to track the minimum price seen so far and the maximum profit
        min_price = float('inf')
        max_profit = 0

        # Iterate through each price in the list
        for price in prices:
            # Update the minimum price if the current price is lower than the known minimum
            if price < min_price:
                min_price = price
            # Calculate the potential profit if selling at the current price
            elif price - min_price > max_profit:
                max_profit = price - min_price

        return max_profit
`
  },
  {
    leetcodeNumber: 3,
    title: 'Longest Substring Without Repeating Characters',
    topic: 'Sliding Window',
    difficulty: Difficulty.Medium,
    leetcodeUrl: 'https://leetcode.com/problems/longest-substring-without-repeating-characters/description/',
    videoId: '',
    codeSnippet: `class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        charSet = set()
        l = 0
        maxLength = 0
        for r in range(len(s)):
            while s[r] in charSet:
                charSet.remove(s[l])
                l += 1
            charSet.add(s[r])
            maxLength = max(maxLength, len(charSet))
        return maxLength`
  },
  {
    leetcodeNumber: 424,
    title: 'Longest Repeating Character Replacement',
    topic: 'Sliding Window',
    difficulty: Difficulty.Medium,
    leetcodeUrl: 'https://leetcode.com/problems/longest-repeating-character-replacement/description/',
    videoId: '',
    codeSnippet: `class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        from collections import defaultdict
        
        # Initialize variables
        max_count = 0
        max_length = 0
        count = defaultdict(int)
        
        left = 0

        # Iterate over the string with the right pointer
        for right in range(len(s)):
            count[s[right]] += 1
            max_count = max(max_count, count[s[right]])

            # Check the window size - if we need more than k replacements, shrink the window
            if (right - left + 1) - max_count > k:
                count[s[left]] -= 1
                left += 1

            max_length = max(max_length, right - left + 1)
        
        return max_length
`
  },
  {
    leetcodeNumber: 76,
    title: 'Minimum Window Substring',
    topic: 'Sliding Window',
    difficulty: Difficulty.Hard,
    leetcodeUrl: 'https://leetcode.com/problems/minimum-window-substring/description/',
    videoId: '',
    codeSnippet: `from collections import Counter, defaultdict

class Solution:
    def minWindow(self, s: str, t: str) -> str:
        if not s or not t:
            return ""
        
        # Dictionary to count characters in t
        dict_t = Counter(t)
        # Dictionary to count characters in the current window
        window_counts = defaultdict(int)
        
        # Number of unique characters in t that need to be present in the window
        required = len(dict_t)
        # Number of unique characters in the current window which match the required count in t
        formed = 0
        
        # (window length, left, right) for the smallest window found
        ans = float("inf"), None, None
        
        left, right = 0, 0
        
        while right < len(s):
            # Add the character from the right to the window
            character = s[right]
            window_counts[character] += 1
            
            # If the character's frequency matches the required count in t
            if character in dict_t and window_counts[character] == dict_t[character]:
                formed += 1
            
            # Try to contract the window until it ceases to be 'desirable'
            while left <= right and formed == required:
                character = s[left]
                
                # Update the result if this window is smaller than the previously found ones
                if right - left + 1 < ans[0]:
                    ans = (right - left + 1, left, right)
                
                # Remove the character from the left of the window
                window_counts[character] -= 1
                if character in dict_t and window_counts[character] < dict_t[character]:
                    formed -= 1
                
                # Move the left pointer ahead
                left += 1
            
            # Keep expanding the window by moving the right pointer
            right += 1
        
        # Return the smallest window, or an empty string if no such window exists
        return "" if ans[0] == float("inf") else s[ans[1]: ans[2] + 1]
`
  },
  {
    leetcodeNumber: 20,
    title: 'Valid Parentheses',
    topic: 'Stack',
    difficulty: Difficulty.Easy,
    leetcodeUrl: 'https://leetcode.com/problems/valid-parentheses/description/',
    videoId: '',
    codeSnippet: `class Solution:
    def isValid(self, s: str) -> bool:
        # Stack to keep track of opening brackets
        stack = []
        
        # Mapping of closing brackets to their corresponding opening brackets
        bracket_map = {')': '(', '}': '{', ']': '['}
        
        # Iterate through each character in the string
        for char in s:
            if char in bracket_map:
                # If the stack is empty or the top of the stack is not the matching opening bracket
                if not stack or stack[-1] != bracket_map[char]:
                    return False
                stack.pop()
            else:
                # If the character is an opening bracket, push it onto the stack
                stack.append(char)
        
        # The string is valid if the stack is empty at the end
        return not stack
`
  },
  {
    leetcodeNumber: 153,
    title: 'Find Minimum in Rotated Sorted Array',
    topic: 'Binary Search',
    difficulty: Difficulty.Medium,
    leetcodeUrl: 'https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/description/',
    videoId: '',
    codeSnippet: `class Solution:
    def findMin(self, nums: List[int]) -> int:
        left, right = 0, len(nums) - 1
        
        while left < right:
            mid = (left + right) // 2
            
            # Check if the middle element is greater than the rightmost element
            if nums[mid] > nums[right]:
                # The minimum is in the right half
                left = mid + 1
            else:
                # The minimum is in the left half, including mid
                right = mid
        
        # At the end of the loop, left == right, pointing to the minimum element
        return nums[left]
`
  },
  {
    leetcodeNumber: 33,
    title: 'Search in Rotated Sorted Array',
    topic: 'Binary Search',
    difficulty: Difficulty.Medium,
    leetcodeUrl: 'https://leetcode.com/problems/search-in-rotated-sorted-array/',
    videoId: '',
    codeSnippet: `class Solution:
    def search(self, nums: List[int], target: int) -> int:
        pivot = self.getPivot(nums)
        
        if pivot == -1:
            return self.binarySearch(nums, target, 0, len(nums) - 1)
        if (nums[pivot] == target):
            return pivot
        if (target < nums[0]):
            return self.binarySearch(nums, target, pivot + 1, len(nums) - 1)
        return self.binarySearch(nums, target, 0, pivot)
        

    def binarySearch(self, nums, target, l, r):
        while (l <= r):
            m = (l + r) // 2
            if (nums[m] == target):
                return m
            
            if (nums[m] < target):
                l = m + 1
            else:
                r = m - 1
        
        return -1

    def getPivot(self, nums):
        l = 0
        r = len(nums) - 1
        while (l < r):
            m = (l + r) // 2
            if (nums[m] > nums[m+1]):
                return m
            if (nums[m] < nums[m-1]):
                return m - 1
            if (nums[m] > nums[0]):
                l = m + 1
            else:
                r = m - 1
        return -1
`
  },
  {
    leetcodeNumber: 206,
    title: 'Reverse Linked List',
    topic: 'Linked List',
    difficulty: Difficulty.Easy,
    leetcodeUrl: 'https://leetcode.com/problems/reverse-linked-list/description/',
    videoId: '',
    codeSnippet: `# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        p = None
        c = head
        while c is not None:
            tmp = c.next
            c.next = p
            p = c
            c = tmp
        return p
        `
  },
  {
    leetcodeNumber: 21,
    title: "Merge Two Sorted Lists",
    topic: "Linked List",
    difficulty: Difficulty.Easy,
    videoId: "",
    leetcodeUrl: "https://leetcode.com/problems/merge-two-sorted-lists/",
    codeSnippet: "class Solution:\n    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:\n        dummy = ListNode(0)\n        current = dummy\n        while l1 and l2:\n            if l1.val < l2.val:\n                current.next = l1\n                l1 = l1.next\n            else:\n                current.next = l2\n                l2 = l2.next\n            current = current.next\n        current.next = l1 or l2\n        return dummy.next"
  },
  {
    leetcodeNumber: 143,
    title: "Reorder List",
    topic: "Linked List",
    difficulty: Difficulty.Medium,
    videoId: "",
    leetcodeUrl: "https://leetcode.com/problems/reorder-list/",
    codeSnippet: "class Solution:\n    def reorderList(self, head: ListNode) -> None:\n        if not head:\n            return\n        # Find the middle of the linked list\n        slow, fast = head, head\n        while fast and fast.next:\n            slow = slow.next\n            fast = fast.next.next\n        # Reverse the second half of the list\n        prev, curr = None, slow\n        while curr:\n            next_temp = curr.next\n            curr.next = prev\n            prev = curr\n            curr = next_temp\n        # Merge the two halves\n        first, second = head, prev\n        while second.next:\n            first.next, first = second, first.next\n            second.next, second = first, second.next"
  },
  {
    leetcodeNumber: 19,
    title: "Remove Nth Node From End of List",
    topic: "Linked List",
    difficulty: Difficulty.Medium,
    videoId: "",
    leetcodeUrl: "https://leetcode.com/problems/remove-nth-node-from-end-of-list/",
    codeSnippet: "class Solution:\n    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:\n        dummy = ListNode(0)\n        dummy.next = head\n        first = dummy\n        second = dummy\n        for _ in range(n + 1):\n            first = first.next\n        while first:\n            first = first.next\n            second = second.next\n        second.next = second.next.next\n        return dummy.next"
  },
  {
    leetcodeNumber: 141,
    title: "Linked List Cycle",
    topic: "Linked List",
    difficulty: Difficulty.Easy,
    videoId: "",
    leetcodeUrl: "https://leetcode.com/problems/linked-list-cycle/",
    codeSnippet: "class Solution:\n    def hasCycle(self, head: ListNode) -> bool:\n        slow, fast = head, head\n        while fast and fast.next:\n            slow = slow.next\n            fast = fast.next.next\n            if slow == fast:\n                return True\n        return False"
  },
  {
    leetcodeNumber: 23,
    title: "Merge K Sorted Lists",
    topic: "Linked List",
    difficulty: Difficulty.Hard,
    videoId: "",
    leetcodeUrl: "https://leetcode.com/problems/merge-k-sorted-lists/",
    codeSnippet: "class Solution:\n    def mergeKLists(self, lists: List[ListNode]) -> ListNode:\n        import heapq\n        ListNode.__lt__ = lambda self, other: self.val < other.val\n        heap = [node for node in lists if node]\n        heapq.heapify(heap)\n        dummy = ListNode(0)\n        current = dummy\n        while heap:\n            node = heapq.heappop(heap)\n            current.next = node\n            current = current.next\n            if node.next:\n                heapq.heappush(heap, node.next)\n        return dummy.next"
  },
  {
    leetcodeNumber: 226,
    title: "Invert Binary Tree",
    topic: "Trees",
    difficulty: Difficulty.Easy,
    videoId: "",
    leetcodeUrl: "https://leetcode.com/problems/invert-binary-tree/",
    codeSnippet: "class Solution:\n    def invertTree(self, root: TreeNode) -> TreeNode:\n        if root is None:\n            return None\n        root.left, root.right = root.right, root.left\n        self.invertTree(root.left)\n        self.invertTree(root.right)\n        return root"
  },
  {
    leetcodeNumber: 104,
    title: "Maximum Depth of Binary Tree",
    topic: "Trees",
    difficulty: Difficulty.Easy,
    videoId: "",
    leetcodeUrl: "https://leetcode.com/problems/maximum-depth-of-binary-tree/",
    codeSnippet: "class Solution:\n    def maxDepth(self, root: TreeNode) -> int:\n        if root is None:\n            return 0\n        left_depth = self.maxDepth(root.left)\n        right_depth = self.maxDepth(root.right)\n        return max(left_depth, right_depth) + 1"
  },
  {
    leetcodeNumber: 100,
    title: "Same Tree",
    topic: "Trees",
    difficulty: Difficulty.Easy,
    videoId: "",
    leetcodeUrl: "https://leetcode.com/problems/same-tree/",
    codeSnippet: "class Solution:\n    def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:\n        if not p and not q:\n            return True\n        if not p or not q or p.val != q.val:\n            return False\n        return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)"
  },
  {
    leetcodeNumber: 572,
    title: "Subtree of Another Tree",
    topic: "Trees",
    difficulty: Difficulty.Easy,
    videoId: "",
    leetcodeUrl: "https://leetcode.com/problems/subtree-of-another-tree/",
    codeSnippet: "class Solution:\n    def isSubtree(self, s: TreeNode, t: TreeNode) -> bool:\n        if not s:\n            return False\n        if self.isSameTree(s, t):\n            return True\n        return self.isSubtree(s.left, t) or self.isSubtree(s.right, t)\n    def isSameTree(self, s: TreeNode, t: TreeNode) -> bool:\n        if not s and not t:\n            return True\n        if not s or not t or s.val != t.val:\n            return False\n        return self.isSameTree(s.left, t.left) and self.isSameTree(s.right, t.right)"
  },
  {
    leetcodeNumber: 235,
    title: "Lowest Common Ancestor of a Binary Search Tree",
    topic: "Trees",
    difficulty: Difficulty.Medium,
    videoId: "",
    leetcodeUrl: "https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/",
    codeSnippet: "class Solution:\n    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':\n        while root:\n            if p.val < root.val and q.val < root.val:\n                root = root.left\n            elif p.val > root.val and q.val > root.val:\n                root = root.right\n            else:\n                return root"
  },
  {
    leetcodeNumber: 102,
    title: "Binary Tree Level Order Traversal",
    topic: "Trees",
    difficulty: Difficulty.Medium,
    videoId: "",
    leetcodeUrl: "https://leetcode.com/problems/binary-tree-level-order-traversal/",
    codeSnippet: "class Solution:\n    def levelOrder(self, root: TreeNode) -> List[List[int]]:\n        if not root:\n            return []\n        result, queue = [], [root]\n        while queue:\n            level_size = len(queue)\n            level = []\n            for _ in range(level_size):\n                node = queue.pop(0)\n                level.append(node.val)\n                if node.left:\n                    queue.append(node.left)\n                if node.right:\n                    queue.append(node.right)\n            result.append(level)\n        return result"
  },
  {
    leetcodeNumber: 98,
    title: "Validate Binary Search Tree",
    topic: "Trees",
    difficulty: Difficulty.Medium,
    videoId: "",
    leetcodeUrl: "https://leetcode.com/problems/validate-binary-search-tree/",
    codeSnippet: "class Solution:\n    def isValidBST(self, root: TreeNode) -> bool:\n        def is_valid(node, low, high):\n            if not node:\n                return True\n            if not (low < node.val < high):\n                return False\n            return is_valid(node.left, low, node.val) and is_valid(node.right, node.val, high)\n        return is_valid(root, float('-inf'), float('inf'))"
  },
  {
    leetcodeNumber: 230,
    title: "Kth Smallest Element In a Bst",
    topic: "Trees",
    difficulty: Difficulty.Medium,
    videoId: "",
    leetcodeUrl: "https://leetcode.com/problems/kth-smallest-element-in-a-bst/",
    codeSnippet: "class Solution:\n    def kthSmallest(self, root: TreeNode, k: int) -> int:\n        def inorder_traversal(node):\n            return inorder_traversal(node.left) + [node.val] + inorder_traversal(node.right) if node else []\n        return inorder_traversal(root)[k - 1]"
  },
  {
    leetcodeNumber: 105,
    title: "Construct Binary Tree From Preorder And Inorder Traversal",
    topic: "Trees",
    difficulty: Difficulty.Medium,
    videoId: "",
    leetcodeUrl: "https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/",
    codeSnippet: "class Solution:\n    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:\n        if not preorder or not inorder:\n            return None\n        root_val = preorder[0]\n        root = TreeNode(root_val)\n        mid = inorder.index(root_val)\n        root.left = self.buildTree(preorder[1:mid+1], inorder[:mid])\n        root.right = self.buildTree(preorder[mid+1:], inorder[mid+1:])\n        return root"
  },
  {
    leetcodeNumber: 124,
    title: "Binary Tree Maximum Path Sum",
    topic: "Trees",
    difficulty: Difficulty.Hard,
    videoId: "",
    leetcodeUrl: "https://leetcode.com/problems/binary-tree-maximum-path-sum/",
    codeSnippet: "class Solution:\n    def maxPathSum(self, root: TreeNode) -> int:\n        def dfs(node):\n            nonlocal max_sum\n            if not node:\n                return 0\n            left = max(dfs(node.left), 0)\n            right = max(dfs(node.right), 0)\n            max_sum = max(max_sum, left + right + node.val)\n            return node.val + max(left, right)\n        max_sum = float('-inf')\n        dfs(root)\n        return max_sum"
  },
  {
    leetcodeNumber: 297,
    title: "Serialize and Deserialize Binary Tree",
    topic: "Trees",
    difficulty: Difficulty.Hard,
    videoId: "",
    leetcodeUrl: "https://leetcode.com/problems/serialize-and-deserialize-binary-tree/",
    codeSnippet: "class Codec:\n    def serialize(self, root: TreeNode) -> str:\n        def recur(node):\n            if not node:\n                return 'None,'\n            return str(node.val) + ',' + recur(node.left) + recur(node.right)\n        return recur(root)\n\n    def deserialize(self, data: str) -> TreeNode:\n        def recur(nodes):\n            if nodes[0] == 'None':\n                nodes.pop(0)\n                return None\n            node = TreeNode(int(nodes.pop(0)))\n            node.left = recur(nodes)\n            node.right = recur(nodes)\n            return node\n        node_list = data.split(',')[:-1]\n        return recur(node_list)"
  },
  {
    leetcodeNumber: 295,
    title: "Find Median from Data Stream",
    topic: "Heap",
    difficulty: Difficulty.Hard,
    videoId: "",
    leetcodeUrl: "https://leetcode.com/problems/find-median-from-data-stream/",
    codeSnippet: "import heapq\n\nclass MedianFinder:\n    def __init__(self):\n        self.small = []  # max heap (inverted min heap)\n        self.large = []  # min heap\n\n    def addNum(self, num: int) -> None:\n        heapq.heappush(self.small, -num)\n        if self.small and self.large and (-self.small[0] > self.large[0]):\n            heapq.heappush(self.large, -heapq.heappop(self.small))\n        if len(self.small) > len(self.large) + 1:\n            heapq.heappush(self.large, -heapq.heappop(self.small))\n        if len(self.large) > len(self.small):\n            heapq.heappush(self.small, -heapq.heappop(self.large))\n\n    def findMedian(self) -> float:\n        if len(self.small) > len(self.large):\n            return -self.small[0]\n        return (-self.small[0] + self.large[0]) / 2"
  },
  {
    leetcodeNumber: 39,
    title: "Combination Sum",
    topic: "Backtracking",
    difficulty: Difficulty.Medium,
    videoId: "",
    leetcodeUrl: "https://leetcode.com/problems/combination-sum/",
    codeSnippet: "class Solution:\n    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:\n        def backtrack(start, target, path):\n            if target == 0:\n                result.append(path)\n                return\n            for i in range(start, len(candidates)):\n                if candidates[i] > target:\n                    break\n                backtrack(i, target - candidates[i], path + [candidates[i]])\n        result = []\n        backtrack(0, target, [])\n        return result"
  },
  {
    leetcodeNumber: 79,
    title: "Word Search",
    topic: "Backtracking",
    difficulty: Difficulty.Medium,
    videoId: "",
    leetcodeUrl: "https://leetcode.com/problems/word-search/",
    codeSnippet: "class Solution:\n    def exist(self, board: List[List[str]], word: str) -> bool:\n        def backtrack(r, c, i):\n            if i == len(word):\n                return True\n            if not (0 <= r < len(board)) or not (0 <= c < len(board[0])) or board[r][c] != word[i]:\n                return False\n            tmp, board[r][c] = board[r][c], '#'  # mark as visited\n            found = (backtrack(r + 1, c, i + 1) or\n                     backtrack(r - 1, c, i + 1) or\n                     backtrack(r, c + 1, i + 1) or\n                     backtrack(r, c - 1, i + 1))\n            board[r][c] = tmp  # unmark\n            return found\n        for r in range(len(board)):\n            for c in range(len(board[0])):\n                if backtrack(r, c, 0):\n                    return True\n        return False"
  },
  {
    leetcodeNumber: 208,
    title: "Implement Trie Prefix Tree",
    topic: "Tries",
    difficulty: Difficulty.Medium,
    videoId: "",
    leetcodeUrl: "https://leetcode.com/problems/implement-trie-prefix-tree/",
    codeSnippet: "class TrieNode:\n    def __init__(self):\n        self.children = {}\n        self.isEndOfWord = False\n\nclass Trie:\n    def __init__(self):\n        self.root = TrieNode()\n\n    def insert(self, word: str) -> None:\n        node = self.root\n        for char in word:\n            if char not in node.children:\n                node.children[char] = TrieNode()\n            node = node.children[char]\n        node.isEndOfWord = True\n\n    def search(self, word: str) -> bool:\n        node = self.root\n        for char in word:\n            if char not in node.children:\n                return False\n            node = node.children[char]\n        return node.isEndOfWord\n\n    def startsWith(self, prefix: str) -> bool:\n        node = self.root\n        for char in prefix:\n            if char not in node.children:\n                return False\n            node = node.children[char]\n        return True"
  },
  {
    leetcodeNumber: 211,
    title: "Design Add And Search Words Data Structure",
    topic: "Tries",
    difficulty: Difficulty.Medium,
    videoId: "",
    leetcodeUrl: "https://leetcode.com/problems/design-add-and-search-words-data-structure/",
    codeSnippet: "class WordDictionary:\n    def __init__(self):\n        self.trie = TrieNode()\n\n    def addWord(self, word: str) -> None:\n        node = self.trie\n        for char in word:\n            if char not in node.children:\n                node.children[char] = TrieNode()\n            node = node.children[char]\n        node.isEndOfWord = True\n\n    def search(self, word: str) -> bool:\n        def dfs(j, node):\n            for i in range(j, len(word)):\n                char = word[i]\n                if char == '.':\n                    for child in node.children.values():\n                        if dfs(i + 1, child):\n                            return True\n                    return False\n                else:\n                    if char not in node.children:\n                        return False\n                    node = node.children[char]\n            return node.isEndOfWord\n        return dfs(0, self.trie)"
  },
  {
    leetcodeNumber: 212,
    title: "Word Search II",
    topic: "Tries",
    difficulty: Difficulty.Hard,
    videoId: "",
    leetcodeUrl: "https://leetcode.com/problems/word-search-ii/",
    codeSnippet: "class TrieNode:\n    def __init__(self):\n        self.children = {}\n        self.word = None\n\nclass Solution:\n    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:\n        def build_trie():\n            root = TrieNode()\n            for word in words:\n                node = root\n                for char in word:\n                    if char not in node.children:\n                        node.children[char] = TrieNode()\n                    node = node.children[char]\n                node.word = word\n            return root\n\n        def search(node, r, c):\n            char = board[r][c]\n            if char not in node.children:\n                return\n            node = node.children[char]\n            if node.word:\n                result.add(node.word)\n                node.word = None\n            board[r][c] = '#'\n            for dr, dc in directions:\n                nr, nc = r + dr, c + dc\n                if 0 <= nr < len(board) and 0 <= nc < len(board[0]):\n                    search(node, nr, nc)\n            board[r][c] = char\n\n        root = build_trie()\n        result = set()\n        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]\n        for r in range(len(board)):\n            for c in range(len(board[0])):\n                search(root, r, c)\n        return list(result)"
  },
  {
    leetcodeNumber: 200,
    title: "Number of Islands",
    topic: "Graphs",
    difficulty: Difficulty.Medium,
    videoId: "",
    leetcodeUrl: "https://leetcode.com/problems/number-of-islands/",
    codeSnippet: "class Solution:\n    def numIslands(self, grid: List[List[str]]) -> int:\n        def dfs(r, c):\n            if r < 0 or c < 0 or r >= len(grid) or c >= len(grid[0]) or grid[r][c] == '0':\n                return\n            grid[r][c] = '0'\n            for dr, dc in directions:\n                dfs(r + dr, c + dc)\n\n        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]\n        num_islands = 0\n        for r in range(len(grid)):\n            for c in range(len(grid[0])):\n                if grid[r][c] == '1':\n                    num_islands += 1\n                    dfs(r, c)\n        return num_islands"
  },
  {
    leetcodeNumber: 133,
    title: "Clone Graph",
    topic: "Graphs",
    difficulty: Difficulty.Medium,
    videoId: "",
    leetcodeUrl: "https://leetcode.com/problems/clone-graph/",
    codeSnippet: "class Node:\n    def __init__(self, val=0, neighbors=None):\n        self.val = val\n        self.neighbors = neighbors if neighbors is not None else []\n\nclass Solution:\n    def cloneGraph(self, node: 'Node') -> 'Node':\n        if not node:\n            return None\n        old_to_new = {}\n        def dfs(node):\n            if node in old_to_new:\n                return old_to_new[node]\n            copy = Node(node.val)\n            old_to_new[node] = copy\n            for neighbor in node.neighbors:\n                copy.neighbors.append(dfs(neighbor))\n            return copy\n        return dfs(node)"
  },
  {
    leetcodeNumber: 417,
    title: "Pacific Atlantic Water Flow",
    topic: "Graphs",
    difficulty: Difficulty.Hard,
    videoId: "",
    leetcodeUrl: "https://leetcode.com/problems/pacific-atlantic-water-flow/",
    codeSnippet: "class Solution:\n    def pacificAtlantic(self, matrix: List[List[int]]) -> List[List[int]]:\n        if not matrix or not matrix[0]:\n            return []\n\n        m, n = len(matrix), len(matrix[0])\n        pacific = set()\n        atlantic = set()\n\n        def dfs(r, c, visited, prev_height):\n            if (r < 0 or c < 0 or r >= m or c >= n or (r, c) in visited or matrix[r][c] < prev_height):\n                return\n            visited.add((r, c))\n            for dr, dc in directions:\n                dfs(r + dr, c + dc, visited, matrix[r][c])\n\n        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]\n\n        for r in range(m):\n            dfs(r, 0, pacific, matrix[r][0])\n            dfs(r, n - 1, atlantic, matrix[r][n - 1])\n\n        for c in range(n):\n            dfs(0, c, pacific, matrix[0][c])\n            dfs(m - 1, c, atlantic, matrix[m - 1][c])\n\n        return list(pacific & atlantic)"
  },
  {
    leetcodeNumber: 207,
    title: "Course Schedule",
    topic: "Graphs",
    difficulty: Difficulty.Medium,
    videoId: "",
    leetcodeUrl: "https://leetcode.com/problems/course-schedule/",
    codeSnippet: "class Solution:\n    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:\n        graph = collections.defaultdict(list)\n        in_degree = [0] * numCourses\n\n        for dest, src in prerequisites:\n            graph[src].append(dest)\n            in_degree[dest] += 1\n\n        queue = collections.deque([i for i in range(numCourses) if in_degree[i] == 0])\n        count = 0\n\n        while queue:\n            course = queue.popleft()\n            count += 1\n            for next_course in graph[course]:\n                in_degree[next_course] -= 1\n                if in_degree[next_course] == 0:\n                    queue.append(next_course)\n\n        return count == numCourses"
  },
  {
    leetcodeNumber: 261,
    title: "Graph Valid Tree",
    topic: "Graphs",
    difficulty: Difficulty.Medium,
    videoId: "",
    leetcodeUrl: "https://leetcode.com/problems/graph-valid-tree/",
    codeSnippet: "class Solution:\n    def validTree(self, n: int, edges: List[List[int]]) -> bool:\n        if len(edges) != n - 1:\n            return False\n\n        graph = collections.defaultdict(list)\n        for u, v in edges:\n            graph[u].append(v)\n            graph[v].append(u)\n\n        visited = set()\n\n        def dfs(node, parent):\n            visited.add(node)\n            for neighbor in graph[node]:\n                if neighbor not in visited:\n                    if not dfs(neighbor, node):\n                        return False\n                elif neighbor != parent:\n                    return False\n            return True\n\n        return dfs(0, -1) and len(visited) == n"
  },
  {
    leetcodeNumber: 323,
    title: "Number of Connected Components in an Undirected Graph",
    topic: "Graphs",
    difficulty: Difficulty.Medium,
    videoId: "",
    leetcodeUrl: "https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph/",
    codeSnippet: "class Solution:\n    def countComponents(self, n: int, edges: List[List[int]]) -> int:\n        def dfs(node):\n            visited.add(node)\n            for neighbor in graph[node]:\n                if neighbor not in visited:\n                    dfs(neighbor)\n\n        graph = collections.defaultdict(list)\n        for u, v in edges:\n            graph[u].append(v)\n            graph[v].append(u)\n\n        visited = set()\n        count = 0\n        for i in range(n):\n            if i not in visited:\n                dfs(i)\n                count += 1\n\n        return count"
  },
  {
    leetcodeNumber: 269,
    title: "Alien Dictionary",
    topic: "Graphs",
    difficulty: Difficulty.Hard,
    videoId: "",
    leetcodeUrl: "https://leetcode.com/problems/alien-dictionary/",
    codeSnippet: "class Solution:\n    def alienOrder(self, words: List[str]) -> str:\n        # Step 1: Build graph\n        graph = collections.defaultdict(set)\n        in_degree = collections.defaultdict(int)\n\n        for word in words:\n            for char in word:\n                in_degree[char] = 0\n\n        for i in range(len(words) - 1):\n            w1, w2 = words[i], words[i + 1]\n            min_len = min(len(w1), len(w2))\n            found_diff = False\n            for j in range(min_len):\n                if w1[j] != w2[j]:\n                    if w2[j] not in graph[w1[j]]:\n                        graph[w1[j]].add(w2[j])\n                        in_degree[w2[j]] += 1\n                    found_diff = True\n                    break\n            if not found_diff and len(w1) > len(w2):\n                return ''\n\n        # Step 2: Topological sort\n        zero_in_degree = collections.deque([char for char in in_degree if in_degree[char] == 0])\n        result = []\n\n        while zero_in_degree:\n            char = zero_in_degree.popleft()\n            result.append(char)\n            for neighbor in graph[char]:\n                in_degree[neighbor] -= 1\n                if in_degree[neighbor] == 0:\n                    zero_in_degree.append(neighbor)\n\n        return ''.join(result) if len(result) == len(in_degree) else ''"
  },
  {
    leetcodeNumber: 70,
    title: "Climbing Stairs",
    topic: "Dynamic Programming",
    difficulty: Difficulty.Easy,
    videoId: "",
    leetcodeUrl: "https://leetcode.com/problems/climbing-stairs/",
    codeSnippet: "class Solution:\n    def climbStairs(self, n: int) -> int:\n        if n <= 1:\n            return 1\n        dp = [0] * (n + 1)\n        dp[1] = 1\n        dp[2] = 2\n        for i in range(3, n + 1):\n            dp[i] = dp[i - 1] + dp[i - 2]\n        return dp[n]"
  },
  {
    leetcodeNumber: 198,
    title: "House Robber",
    topic: "Dynamic Programming",
    difficulty: Difficulty.Medium,
    videoId: "",
    leetcodeUrl: "https://leetcode.com/problems/house-robber/",
    codeSnippet: "class Solution:\n    def rob(self, nums: List[int]) -> int:\n        if not nums:\n            return 0\n        if len(nums) == 1:\n            return nums[0]\n        dp = [0] * len(nums)\n        dp[0] = nums[0]\n        dp[1] = max(nums[0], nums[1])\n        for i in range(2, len(nums)):\n            dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])\n        return dp[-1]"
  },
  {
    leetcodeNumber: 213,
    title: "House Robber II",
    topic: "Dynamic Programming",
    difficulty: Difficulty.Medium,
    videoId: "",
    leetcodeUrl: "https://leetcode.com/problems/house-robber-ii/",
    codeSnippet: "class Solution:\n    def rob(self, nums: List[int]) -> int:\n        def rob_linear(nums: List[int]) -> int:\n            if len(nums) == 1:\n                return nums[0]\n            dp = [0] * len(nums)\n            dp[0] = nums[0]\n            dp[1] = max(nums[0], nums[1])\n            for i in range(2, len(nums)):\n                dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])\n            return dp[-1]\n\n        if len(nums) == 1:\n            return nums[0]\n        return max(rob_linear(nums[:-1]), rob_linear(nums[1:]))"
  },
  {
    leetcodeNumber: 5,
    title: "Longest Palindromic Substring",
    topic: "Dynamic Programming",
    difficulty: Difficulty.Medium,
    videoId: "",
    leetcodeUrl: "https://leetcode.com/problems/longest-palindromic-substring/",
    codeSnippet: "class Solution:\n    def longestPalindrome(self, s: str) -> str:\n        n = len(s)\n        if n == 0:\n            return \"\"\n        dp = [[False] * n for _ in range(n)]\n        start = 0\n        max_length = 1\n\n        for i in range(n):\n            dp[i][i] = True\n\n        for length in range(2, n + 1):\n            for i in range(n - length + 1):\n                j = i + length - 1\n                if length == 2:\n                    dp[i][j] = (s[i] == s[j])\n                else:\n                    dp[i][j] = (s[i] == s[j]) and dp[i + 1][j - 1]\n                if dp[i][j] and length > max_length:\n                    start = i\n                    max_length = length\n\n        return s[start:start + max_length]"
  },
  {
    leetcodeNumber: 647,
    title: "Palindromic Substrings",
    topic: "Dynamic Programming",
    difficulty: Difficulty.Medium,
    videoId: "",
    leetcodeUrl: "https://leetcode.com/problems/palindromic-substrings/",
    codeSnippet: "class Solution:\n    def countSubstrings(self, s: str) -> int:\n        n = len(s)\n        count = 0\n        dp = [[False] * n for _ in range(n)]\n\n        for i in range(n):\n            dp[i][i] = True\n            count += 1\n\n        for length in range(2, n + 1):\n            for i in range(n - length + 1):\n                j = i + length - 1\n                if length == 2:\n                    dp[i][j] = (s[i] == s[j])\n                else:\n                    dp[i][j] = (s[i] == s[j]) and dp[i + 1][j - 1]\n                if dp[i][j]:\n                    count += 1\n\n        return count"
  },
  {
    leetcodeNumber: 91,
    title: "Decode Ways",
    topic: "Dynamic Programming",
    difficulty: Difficulty.Medium,
    videoId: "",
    leetcodeUrl: "https://leetcode.com/problems/decode-ways/",
    codeSnippet: "class Solution:\n    def numDecodings(self, s: str) -> int:\n        if not s or s[0] == '0':\n            return 0\n        dp = [0] * (len(s) + 1)\n        dp[0] = dp[1] = 1\n\n        for i in range(2, len(s) + 1):\n            if s[i - 1] != '0':\n                dp[i] += dp[i - 1]\n            if s[i - 2:i] >= '10' and s[i - 2:i] <= '26':\n                dp[i] += dp[i - 2]\n\n        return dp[-1]"
  },
  {
    leetcodeNumber: 322,
    title: "Coin Change",
    topic: "Dynamic Programming",
    difficulty: Difficulty.Medium,
    videoId: "",
    leetcodeUrl: "https://leetcode.com/problems/coin-change/",
    codeSnippet: "class Solution:\n    def coinChange(self, coins: List[int], amount: int) -> int:\n        dp = [float('inf')] * (amount + 1)\n        dp[0] = 0\n        for coin in coins:\n            for x in range(coin, amount + 1):\n                dp[x] = min(dp[x], dp[x - coin] + 1)\n        return dp[amount] if dp[amount] != float('inf') else -1"
  },
  {
    leetcodeNumber: 152,
    title: "Maximum Product Subarray",
    topic: "Dynamic Programming",
    difficulty: Difficulty.Medium,
    videoId: "",
    leetcodeUrl: "https://leetcode.com/problems/maximum-product-subarray/",
    codeSnippet: "class Solution:\n    def maxProduct(self, nums: List[int]) -> int:\n        if not nums:\n            return 0\n        max_product = min_product = result = nums[0]\n        for num in nums[1:]:\n            temp_max = max(num, num * max_product, num * min_product)\n            min_product = min(num, num * max_product, num * min_product)\n            max_product = temp_max\n            result = max(result, max_product)\n        return result"
  },
  {
    leetcodeNumber: 139,
    title: "Word Break",
    topic: "Dynamic Programming",
    difficulty: Difficulty.Medium,
    videoId: "",
    leetcodeUrl: "https://leetcode.com/problems/word-break/",
    codeSnippet: "class Solution:\n    def wordBreak(self, s: str, wordDict: List[str]) -> bool:\n        word_set = set(wordDict)\n        dp = [False] * (len(s) + 1)\n        dp[0] = True\n\n        for i in range(1, len(s) + 1):\n            for j in range(i):\n                if dp[j] and s[j:i] in word_set:\n                    dp[i] = True\n                    break\n\n        return dp[len(s)]"
  },
  {
    leetcodeNumber: 300,
    title: "Longest Increasing Subsequence",
    topic: "Dynamic Programming",
    difficulty: Difficulty.Medium,
    videoId: "",
    leetcodeUrl: "https://leetcode.com/problems/longest-increasing-subsequence/",
    codeSnippet: "class Solution:\n    def lengthOfLIS(self, nums: List[int]) -> int:\n        if not nums:\n            return 0\n        dp = [1] * len(nums)\n        for i in range(1, len(nums)):\n            for j in range(i):\n                if nums[i] > nums[j]:\n                    dp[i] = max(dp[i], dp[j] + 1)\n        return max(dp)"
  },
  {
    leetcodeNumber: 62,
    title: "Unique Paths",
    topic: "Dynamic Programming",
    difficulty: Difficulty.Medium,
    videoId: "",
    leetcodeUrl: "https://leetcode.com/problems/unique-paths/",
    codeSnippet: "class Solution:\n    def uniquePaths(self, m: int, n: int) -> int:\n        dp = [[1] * n for _ in range(m)]\n        for i in range(1, m):\n            for j in range(1, n):\n                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]\n        return dp[-1][-1]"
  },
  {
    leetcodeNumber: 1143,
    title: "Longest Common Subsequence",
    topic: "Dynamic Programming",
    difficulty: Difficulty.Medium,
    videoId: "",
    leetcodeUrl: "https://leetcode.com/problems/longest-common-subsequence/",
    codeSnippet: "class Solution:\n    def longestCommonSubsequence(self, text1: str, text2: str) -> int:\n        m, n = len(text1), len(text2)\n        dp = [[0] * (n + 1) for _ in range(m + 1)]\n\n        for i in range(1, m + 1):\n            for j in range(1, n + 1):\n                if text1[i - 1] == text2[j - 1]:\n                    dp[i][j] = dp[i - 1][j - 1] + 1\n                else:\n                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])\n\n        return dp[m][n]"
  },
  {
    leetcodeNumber: 53,
    title: "Maximum Subarray",
    topic: "Greedy",
    difficulty: Difficulty.Medium,
    videoId: "",
    leetcodeUrl: "https://leetcode.com/problems/maximum-subarray/",
    codeSnippet: "class Solution:\n    def maxSubArray(self, nums: List[int]) -> int:\n        max_sum = current_sum = nums[0]\n        for num in nums[1:]:\n            current_sum = max(num, current_sum + num)\n            max_sum = max(max_sum, current_sum)\n        return max_sum"
  },
  {
    leetcodeNumber: 55,
    title: "Jump Game",
    topic: "Greedy",
    difficulty: Difficulty.Medium,
    videoId: "",
    leetcodeUrl: "https://leetcode.com/problems/jump-game/",
    codeSnippet: "class Solution:\n    def canJump(self, nums: List[int]) -> bool:\n        max_reachable = 0\n        for i, jump in enumerate(nums):\n            if i > max_reachable:\n                return False\n            max_reachable = max(max_reachable, i + jump)\n        return True"
  },
  {
    leetcodeNumber: 57,
    title: "Insert Interval",
    topic: "Intervals",
    difficulty: Difficulty.Hard,
    videoId: "",
    leetcodeUrl: "https://leetcode.com/problems/insert-interval/",
    codeSnippet: "class Solution:\n    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:\n        result = []\n        i = 0\n        while i < len(intervals) and intervals[i][1] < newInterval[0]:\n            result.append(intervals[i])\n            i += 1\n        while i < len(intervals) and intervals[i][0] <= newInterval[1]:\n            newInterval[0] = min(newInterval[0], intervals[i][0])\n            newInterval[1] = max(newInterval[1], intervals[i][1])\n            i += 1\n        result.append(newInterval)\n        result.extend(intervals[i:])\n        return result"
  },
  {
    leetcodeNumber: 56,
    title: "Merge Intervals",
    topic: "Intervals",
    difficulty: Difficulty.Medium,
    videoId: "",
    leetcodeUrl: "https://leetcode.com/problems/merge-intervals/",
    codeSnippet: "class Solution:\n    def merge(self, intervals: List[List[int]]) -> List[List[int]]:\n        if not intervals:\n            return []\n        intervals.sort(key=lambda x: x[0])\n        merged = [intervals[0]]\n        for interval in intervals[1:]:\n            last = merged[-1]\n            if last[1] >= interval[0]:\n                last[1] = max(last[1], interval[1])\n            else:\n                merged.append(interval)\n        return merged"
  },
  {
    leetcodeNumber: 435,
    title: "Non Overlapping Intervals",
    topic: "Intervals",
    difficulty: Difficulty.Medium,
    videoId: "",
    leetcodeUrl: "https://leetcode.com/problems/non-overlapping-intervals/",
    codeSnippet: "class Solution:\n    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:\n        if not intervals:\n            return 0\n        intervals.sort(key=lambda x: x[1])\n        end = intervals[0][1]\n        count = 0\n        for i in range(1, len(intervals)):\n            if intervals[i][0] < end:\n                count += 1\n            else:\n                end = intervals[i][1]\n        return count"
  },
  {
    leetcodeNumber: 252,
    title: "Meeting Rooms",
    topic: "Intervals",
    difficulty: Difficulty.Easy,
    videoId: "",
    leetcodeUrl: "https://leetcode.com/problems/meeting-rooms/",
    codeSnippet: "class Solution:\n    def canAttendMeetings(self, intervals: List[List[int]]) -> bool:\n        intervals.sort(key=lambda x: x[0])\n        for i in range(1, len(intervals)):\n            if intervals[i][0] < intervals[i - 1][1]:\n                return False\n        return True"
  },
  {
    leetcodeNumber: 253,
    title: "Meeting Rooms II",
    topic: "Intervals",
    difficulty: Difficulty.Medium,
    videoId: "",
    leetcodeUrl: "https://leetcode.com/problems/meeting-rooms-ii/",
    codeSnippet: "class Solution:\n    def minMeetingRooms(self, intervals: List[List[int]]) -> int:\n        if not intervals:\n            return 0\n        start_times = sorted([i[0] for i in intervals])\n        end_times = sorted([i[1] for i in intervals])\n        room_count = end_pointer = 0\n        for start in start_times:\n            if start < end_times[end_pointer]:\n                room_count += 1\n            else:\n                end_pointer += 1\n        return room_count"
  },
  {
    leetcodeNumber: 48,
    title: "Rotate Image",
    topic: "Matrix",
    difficulty: Difficulty.Medium,
    videoId: "",
    leetcodeUrl: "https://leetcode.com/problems/rotate-image/",
    codeSnippet: "class Solution:\n    def rotate(self, matrix: List[List[int]]) -> None:\n        n = len(matrix)\n        # Transpose the matrix\n        for i in range(n):\n            for j in range(i, n):\n                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]\n        # Reverse each row\n        for i in range(n):\n            matrix[i].reverse()"
  },
  {
    leetcodeNumber: 54,
    title: "Spiral Matrix",
    topic: "Matrix",
    difficulty: Difficulty.Medium,
    videoId: "",
    leetcodeUrl: "https://leetcode.com/problems/spiral-matrix/",
    codeSnippet: "class Solution:\n    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:\n        res = []\n        while matrix:\n            res += matrix[0]\n            matrix = list(zip(*matrix[1:]))[::-1]\n        return res"
  },
  {
    leetcodeNumber: 73,
    title: "Set Matrix Zeroes",
    topic: "Matrix",
    difficulty: Difficulty.Medium,
    videoId: "",
    leetcodeUrl: "https://leetcode.com/problems/set-matrix-zeroes/",
    codeSnippet: "class Solution:\n    def setZeroes(self, matrix: List[List[int]]) -> None:\n        m, n = len(matrix), len(matrix[0])\n        rows, cols = set(), set()\n        # Find all rows and columns that need to be zeroed\n        for i in range(m):\n            for j in range(n):\n                if matrix[i][j] == 0:\n                    rows.add(i)\n                    cols.add(j)\n        # Zero out the rows\n        for row in rows:\n            for j in range(n):\n                matrix[row][j] = 0\n        # Zero out the columns\n        for col in cols:\n            for i in range(m):\n                matrix[i][col] = 0"
  },
  {
    leetcodeNumber: 191,
    title: "Number of 1 Bits",
    topic: "Binary",
    difficulty: Difficulty.Easy,
    videoId: "",
    leetcodeUrl: "https://leetcode.com/problems/number-of-1-bits/",
    codeSnippet: "class Solution:\n    def hammingWeight(self, n: int) -> int:\n        count = 0\n        while n:\n            count += n & 1\n            n >>= 1\n        return count"
  },
  {
    leetcodeNumber: 338,
    title: "Counting Bits",
    topic: "Binary",
    difficulty: Difficulty.Easy,
    videoId: "",
    leetcodeUrl: "https://leetcode.com/problems/counting-bits/",
    codeSnippet: "class Solution:\n    def countBits(self, n: int) -> List[int]:\n        dp = [0] * (n + 1)\n        for i in range(1, n + 1):\n            dp[i] = dp[i >> 1] + (i & 1)\n        return dp"
  },
  {
    leetcodeNumber: 190,
    title: "Reverse Bits",
    topic: "Binary",
    difficulty: Difficulty.Easy,
    videoId: "",
    leetcodeUrl: "https://leetcode.com/problems/reverse-bits/",
    codeSnippet: "class Solution:\n    def reverseBits(self, n: int) -> int:\n        res = 0\n        for _ in range(32):\n            res = (res << 1) | (n & 1)\n            n >>= 1\n        return res"
  },
  {
    leetcodeNumber: 268,
    title: "Missing Number",
    topic: "Binary",
    difficulty: Difficulty.Easy,
    videoId: "",
    leetcodeUrl: "https://leetcode.com/problems/missing-number/",
    codeSnippet: "class Solution:\n    def missingNumber(self, nums: List[int]) -> int:\n        n = len(nums)\n        return n * (n + 1) // 2 - sum(nums)"
  },
  {
    leetcodeNumber: 371,
    title: "Sum of Two Integers",
    topic: "Binary",
    difficulty: Difficulty.Medium,
    videoId: "",
    leetcodeUrl: "https://leetcode.com/problems/sum-of-two-integers/",
    codeSnippet: "class Solution:\n    def getSum(self, a: int, b: int) -> int:\n        while b:\n            carry = a & b\n            a ^= b\n            b = carry << 1\n        return a"
  },
];

  // {
  //   leetcodeNumber: 0,
  //   title: '',
  //   topic: '',
  //   difficulty: Difficulty.Easy,
  //   leetcodeUrl: '',
  //   videoId: '',
  //   codeSnippet: ``
  // }
