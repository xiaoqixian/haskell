-- Date:   Thu Jun 20 22:24:16 2024
-- Mail:   lunar_ubuntu@qq.com
-- Author: https://github.com/xiaoqixian

data Option a = None | Some a

data List a = Empty | Cons a (List a)

data Tree a = Leaf a | Node (Tree a) a (Tree a)
