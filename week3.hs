-- Date:   Fri Jun 21 10:50:24 2024
-- Mail:   lunar_ubuntu@qq.com
-- Author: https://github.com/xiaoqixian

data List a = Empty | Cons a (List a)

filterList :: (t -> Bool) -> List t -> List t
filterList _ Empty = Empty
filterList p (Cons x xs)
  | p x = Cons x (filterList p xs)
  | otherwise = filterList p xs


fmap :: (a -> b) -> Maybe a -> Maybe b
fmap _ Nothing = Nothing
fmap f (Just x) = Just (f x)

