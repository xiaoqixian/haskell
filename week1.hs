-- Date:   Wed Jun 19 15:09:21 2024
-- Mail:   lunar_ubuntu@qq.com
-- Author: https://github.com/xiaoqixian

add :: Int -> Int -> Int
add x y = x + y

fib :: Int -> Int
fib 0 = 0
fib 1 = 1
fib n = fib (n-1) + fib (n-2)

fib2 :: Int -> Int
fib2 n 
  | n == 0 = 0
  | n == 1 = 1
  | otherwise = fib (n-1) + fib (n-2)

-- receive function as parameters
apply :: (Int -> Int -> Int) -> Int -> Int -> Int
apply func = func

len :: [a] -> Int
len [] = 0
len (_:tail) = 1 + len tail

lastElem :: [a] -> a
lastElem [x] = x
lastElem (_:tail) = lastElem tail

a = [1,2,3]

(+-) :: (Int, Int) -> Int -> (Int, Int)
(+-) (x, y) v = (x+v, y-v)

(x, y) = (0,0) +- 1
main = print (x, y)
