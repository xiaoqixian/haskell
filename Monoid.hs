class Monoid a where
    mempty :: a
    mappend :: a -> a -> a
    mconcat :: [a] -> a
