-- Date:   Fri Jun 21 17:48:22 2024
-- Mail:   lunar_ubuntu@qq.com
-- Author: https://github.com/xiaoqixian

module Common where

import Prelude(Show)

data Bool = True | False
  deriving Show

not :: Bool -> Bool
not True = False
not False = True

data Maybe a = Nothing | Just a
  deriving Show

data List a = Empty | Cons a (List a)

next :: List a -> (Maybe a, List a)
next Empty = (Nothing, Empty)
next (Cons x xs) = (Just x, xs)
