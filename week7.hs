-- Date:   Sat Jun 22 20:53:07 2024
-- Mail:   lunar_ubuntu@qq.com
-- Author: https://github.com/xiaoqixian

import Prelude(foldr, print)

class Monoid m where
  munit :: m
  (<>) :: m -> m -> m

instance Monoid [a] where
  munit = []
  x <> [] = x
  [] <> x = x
  (x:xs) <> y = x:(xs <> y)

l1 = [1,2,3]
l2 = [4,5,6]

l3 = l1 <> l2
main = print l3
