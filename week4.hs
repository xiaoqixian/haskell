-- Date:   Fri Jun 21 15:18:57 2024
-- Mail:   lunar_ubuntu@qq.com
-- Author: https://github.com/xiaoqixian

import Prelude(print, Int, String)
import Common

class Eq a where 
  (==) :: a -> a -> Bool
  (/=) :: a -> a -> Bool
  x /= y = not (x == y)

instance Eq a => Eq (Maybe a) where 
  Nothing == Nothing = True
  Just _ == Nothing = False
  Nothing == Just _ = False
  Just x == Just y = x == y

data Direction = Left | Right
instance Eq Direction where 
  (==) Left Left = True
  (==) Left Right = False
  (==) Right Left = False
  (==) Right Right = True

-- foo :: a -> a -> String
foo :: (Eq a) => a -> a -> String
foo x y = case x == y of 
  True -> "Same Direction"
  False -> "Different Direction"



a = Left == Right
main = print a
