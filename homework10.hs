-- Homework 10
-- Implement a parser for a value of type `a` as a function which 
-- takes a String representing the input to be parsed, and succeeds
-- or fails; if it succeeds, it returns the parsed value along with 
-- whatever part of the input it did not use.
newtype Parser a
    = Parser { runParser :: String -> Maybe (a, String) }

satisfy :: (Char -> Bool) -> Parser Char
satisfy pred = Parser f
    where
        f [] = Nothing
        f (x:xs)
            | pred x = Just (x, xs)
            | otherwise = Nothing

-- Exercise 1
-- Implment a `Functor` for Parser
