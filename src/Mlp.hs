module Mlp (Mlp, newMlp, forwardMlp, newRandomMlp, MlpOutput (..), fitBatch) where

import qualified Data.HashMap.Strict as HM
import Graphs
import Matrices
import System.Random (StdGen)
import Tinygrad

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
    first_mult = multMatrices inp mat
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
    first_mult = multMatrices inp (head (layers model))

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

updateModelWithGraph :: Mlp -> Graph -> Mlp
updateModelWithGraph model graph = new_model
  where
    new_model = Mlp [updateMatrixWithGraph mat graph | mat <- layers model]

allParamsFromModel :: Mlp -> [Nombre]
allParamsFromModel model = concat [allParamsFromMatrix layer | layer <- layers model]

fitBatch :: Mlp -> (Matrix2d, Matrix2d) -> Double -> Mlp
fitBatch model (inp, labels) lr =
  case out of
    Nothing -> model
    Just (MlpOutput hidds ot) ->
      let ses = meanSquaredError ot labels
       in case ses of
            Nothing -> model
            Just mat ->
              let sum_ses = sumNombre (allParamsFromMatrix mat)
                  graph = Graph (HM.fromList [(nombre_id node, node) | node <- concat [[sum_ses], allParamsFromMatrix mat, allParamsFromMatrix ot, concat [allParamsFromMatrix hid | hid <- hidds], allParamsFromModel model]])
                  backwarded_graph = backward (nombre_id sum_ses) graph
                  grad_steped_graph = makeGradStep backwarded_graph lr
                  new_model = updateModelWithGraph model grad_steped_graph
               in new_model
  where
    out = forwardMlp model inp
