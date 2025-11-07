module Matrices (Matrix2d, coeffs, newMatrix2d, randMatrix2d, addMatrices, multMatrices) where

import Common
import Nombres
import System.Random (Random, StdGen, randomR)

newtype Matrix2d = Matrix2d
  { coeffs :: [[Nombre]]
  }
  deriving (Eq, Show)

allSame :: (Eq a) => [a] -> Bool
allSame [] = True
allSame (x : xs) = all (== x) xs

isRectangle :: [[Nombre]] -> Maybe [[Nombre]]
isRectangle rows
  | all_rows_equal_len = Just rows
  | otherwise = Nothing
  where
    all_rows_equal_len = allSame [length row | row <- rows]

newMatrix2d :: [[Nombre]] -> Maybe Matrix2d
newMatrix2d rows =
  case is_rectangle of
    Just t -> Just (Matrix2d t)
    Nothing -> Nothing
  where
    is_rectangle :: Maybe [[Nombre]]
    is_rectangle = isRectangle rows

randList :: (Random a) => Int -> (a, a) -> StdGen -> Maybe ([a], StdGen)
randList n _ _ | n < 0 = Nothing
randList 0 _ key = Just ([], key)
randList n range key =
  case randList (n - 1) range new_key of
    Just (randNumbers, others_key) -> Just (randNumber : randNumbers, others_key)
    Nothing -> Nothing
  where
    (randNumber, new_key) = randomR range key

rand2dList :: (Random a) => (Int, Int) -> (a, a) -> StdGen -> Maybe ([[a]], StdGen)
rand2dList (_, n) _ _ | n < 0 = Nothing
rand2dList (m, _) _ _ | m < 0 = Nothing
rand2dList (0, _) _ key = Just ([], key)
rand2dList (m, n) range key = do
  (randRow, new_key) <- randList n range key
  (randRows, others_key) <- rand2dList (m - 1, n) range new_key
  return (randRow : randRows, others_key)

-- Convenience function to create a random 2d matrix kinda nice
randMatrix2d :: NodeId -> (Int, Int) -> (Double, Double) -> StdGen -> Maybe (Matrix2d, StdGen)
randMatrix2d name shape range key = do
  (rand_numbers, new_key) <- rand2dList shape range key
  result <- newMatrix2d [[createNombre (name ++ "_" ++ show i ++ "_" ++ show j, rand_numbers !! i !! j) | j <- [0 .. length (rand_numbers !! i) - 1]] | i <- [0 .. length rand_numbers - 1]]
  return (result, new_key)

addMatrices :: Matrix2d -> Matrix2d -> Maybe Matrix2d
addMatrices m1 m2
  | shape1 /= shape2 = Nothing
  | otherwise = newMatrix2d $ zipWith (zipWith (+)) (coeffs m1) (coeffs m2)
  where
    shape1 = (length (coeffs m1), length (head (coeffs m1)))
    shape2 = (length (coeffs m2), length (head (coeffs m2)))

multMatrices :: Matrix2d -> Matrix2d -> Maybe Matrix2d
multMatrices m1 m2
  | cols1 /= rows2 = Nothing
  -- problem : save intermediate compute into the graph...
  | otherwise = newMatrix2d [[sumNombre [coeffs m1 !! i !! k * coeffs m2 !! k !! j | k <- [0 .. rows1 - 1]] | j <- [0 .. cols2 - 1]] | i <- [0 .. rows1 - 1]]
  where
    rows1 = length (coeffs m1)
    cols1 = length (head (coeffs m1))
    rows2 = length (coeffs m2)
    cols2 = length (head (coeffs m2))
