module Mlp (Mlp, newMlp, forwardMlp, newRandomMlp) where

import Matrices
import System.Random (StdGen)

newtype Mlp = Mlp
  { layers :: [Matrix2d]
  }
  deriving (Eq, Show)

isCompatible :: [Matrix2d] -> Maybe [Matrix2d]
isCompatible [] = Just []
isCompatible [m] = Just [m]
isCompatible (m1 : m2 : ms)
  | cols1 /= rows2 = Nothing
  | otherwise = do
      rest <- isCompatible (m2 : ms)
      return (m1 : rest)
  where
    cols1 = length (head (coeffs m1))
    rows2 = length (coeffs m2)

newMlp :: [Matrix2d] -> Maybe Mlp
newMlp ms =
  case compatible_dims of
    Just t -> Just (Mlp t)
    Nothing -> Nothing
  where
    compatible_dims :: Maybe [Matrix2d]
    compatible_dims = isCompatible ms

data MlpOutput = MlpOutput
  { hidden_states :: [Matrix2d],
    output_tensor :: Matrix2d
  }
  deriving (Eq, Show)

removeFirstLayer :: Mlp -> Mlp
removeFirstLayer (Mlp []) = Mlp []
removeFirstLayer (Mlp mats) = Mlp (tail mats)

forwardMlp :: Mlp -> Matrix2d -> Maybe MlpOutput
forwardMlp (Mlp []) _ = Just (MlpOutput [] emptyMatrix2d)
forwardMlp (Mlp [mat]) inp =
  case first_mult of
    Nothing -> Nothing
    Just t -> Just (MlpOutput [] t)
  where
    first_mult :: Maybe Matrix2d
    first_mult = multMatrices mat inp
forwardMlp model inp =
  case first_mult of
    Nothing -> Nothing
    Just t ->
      let first_layer_hidden = applyTanh t
       in case forwardMlp (removeFirstLayer model) first_layer_hidden of
            Nothing -> Nothing
            Just rest -> Just $ MlpOutput (first_layer_hidden : hidden_states rest) (output_tensor rest)
  where
    first_mult :: Maybe Matrix2d
    first_mult = multMatrices (head (layers model)) inp

nameLayersBasedOnOrder :: Mlp -> Mlp
nameLayersBasedOnOrder model = Mlp [suffixParamsMatrix2d (show i) layer | (i, layer) <- zip [0 .. length (layers model) - 1] (layers model)]

newRandomMlp :: [(Int, Int, Double, Double, StdGen)] -> Maybe Mlp
newRandomMlp [] = Just (Mlp [])
newRandomMlp [(indim, outdim, minrange, maxrange, key)] = do
  (mat, _) <- randMatrix2d "layer" (indim, outdim) (minrange, maxrange) key
  newMlp [mat]
newRandomMlp dims_and_ranges = do
  let (indim, outdim, minrange, maxrange, key) = head dims_and_ranges
  (mat, _) <- randMatrix2d "layer" (indim, outdim) (minrange, maxrange) key
  rest <- newRandomMlp (tail dims_and_ranges)
  mlp <- newMlp (mat : layers rest)
  return (nameLayersBasedOnOrder mlp)
