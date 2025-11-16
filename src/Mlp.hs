module Mlp (Mlp (..), newMlp, forwardMlp, newRandomMlp, MlpOutput (..), fitBatch, Weight, Bias, addBias) where

import qualified Data.HashMap.Strict as HM
import Graphs
import Matrices
import System.Random (StdGen)
import Tinygrad

type Weight = Matrix2d

type Bias = Matrix2d

newtype Mlp = Mlp
  { layers :: [(Weight, Bias)]
  }
  deriving (Eq, Show)

isCompatibleWeightBias :: (Weight, Bias) -> Maybe (Weight, Bias)
isCompatibleWeightBias (l, b)
  | length (coeffs b) /= 1 = Nothing
  | null (head (coeffs b)) = Nothing
  | length (head (coeffs l)) == length (head (coeffs b)) = Just (l, b)
  | otherwise = Nothing

isCompatible :: [(Weight, Bias)] -> Maybe [(Weight, Bias)]
isCompatible [] = Just []
isCompatible [(l, b)] =
  case isCompatibleWeightBias (l, b) of
    Nothing -> Nothing
    Just res -> Just [res]
isCompatible ((l1, b1) : rest) = do
  l1b1compat <- isCompatibleWeightBias (l1, b1)
  case rest of
    ((l2, _) : _) ->
      let cols1 = length (head (coeffs l1))
          rows2 = length (coeffs l2)
       in if cols1 /= rows2
            then Nothing
            else do
              restcomp <- isCompatible rest
              return (l1b1compat : restcomp)

newMlp :: [(Weight, Bias)] -> Maybe Mlp
newMlp ms =
  case compatible_dims of
    Just t -> Just (Mlp t)
    Nothing -> Nothing
  where
    compatible_dims :: Maybe [(Weight, Bias)]
    compatible_dims = isCompatible ms

data MlpOutput = MlpOutput
  { hidden_states :: [Matrix2d],
    output_tensor :: Matrix2d
  }
  deriving (Eq, Show)

addBias :: Matrix2d -> Bias -> Maybe Matrix2d
addBias h b
  | length (coeffs b) /= 1 = Nothing
  | otherwise = do
      let batch_size = length (coeffs h)
      broadcasted_bias <- newMatrix2d [head (coeffs b) | _ <- [1 .. (batch_size :: Int)]]
      addMatrices h broadcasted_bias

forwardMlp :: Mlp -> Matrix2d -> Maybe MlpOutput
forwardMlp (Mlp []) _ = Just (MlpOutput [] emptyMatrix2d)
forwardMlp (Mlp [(w, b)]) inp =
  case first_mult of
    Nothing -> Nothing
    Just t -> do
      res <- addBias t b
      Just (MlpOutput [] res)
  where
    first_mult :: Maybe Matrix2d
    first_mult = multMatrices inp w
forwardMlp (Mlp ((w1, b1) : others)) inp =
  case first_mult of
    Nothing -> Nothing
    Just t -> do
      res <- addBias t b1
      let first_layer_hidden = applyTanh res
       in case forwardMlp (Mlp others) first_layer_hidden of
            Nothing -> Nothing
            Just rest -> Just $ MlpOutput (first_layer_hidden : hidden_states rest) (output_tensor rest)
  where
    first_mult :: Maybe Matrix2d
    first_mult = multMatrices inp w1

nameLayersBasedOnOrder :: Mlp -> Mlp
nameLayersBasedOnOrder model = Mlp [(suffixParamsMatrix2d (show i) w, suffixParamsMatrix2d (show i) b) | (i, (w, b)) <- zip [0 .. length (layers model) - 1] (layers model)]

newRandomMlp :: [(Shape, Range, StdGen)] -> Maybe Mlp
newRandomMlp [] = Just (Mlp [])
newRandomMlp [((indim, outdim), (minrange, maxrange), key)] = do
  (w, new_key) <- randMatrix2d "weight" (indim, outdim) (minrange, maxrange) key
  (b, _) <- randMatrix2d "bias" (1, outdim) (minrange, maxrange) new_key
  newMlp [(w, b)]
newRandomMlp dims_and_ranges = do
  let ((indim, outdim), (minrange, maxrange), key) = head dims_and_ranges
  (w, new_key) <- randMatrix2d "weight" (indim, outdim) (minrange, maxrange) key
  (b, _) <- randMatrix2d "bias" (1, outdim) (minrange, maxrange) new_key
  rest <- newRandomMlp (tail dims_and_ranges)
  mlp <- newMlp ((w, b) : layers rest)
  return (nameLayersBasedOnOrder mlp)

updateModelWithGraph :: Mlp -> Graph -> Mlp
updateModelWithGraph model graph = new_model
  where
    new_model = Mlp [(updateMatrixWithGraph w graph, updateMatrixWithGraph b graph) | (w, b) <- layers model]

allParamsFromModel :: Mlp -> [Nombre]
allParamsFromModel model = concat [allParamsFromMatrix w ++ allParamsFromMatrix b | (w, b) <- layers model]

fitBatch :: Mlp -> (Matrix2d, Matrix2d) -> Double -> Maybe Mlp
fitBatch model (inp, labels) lr =
  case out of
    Nothing -> Nothing
    Just (MlpOutput _ ot) ->
      let ses = meanSquaredError ot labels
       in case ses of
            Nothing -> Nothing
            Just mat ->
              let sum_ses = sumNombre (allParamsFromMatrix mat)
                  graph = Graph (HM.fromList [(nombre_id node, node) | node <- sum_ses : allParamsFromModel model])
                  backwarded_graph = backward (nombre_id sum_ses) graph
                  grad_steped_graph = makeGradStep backwarded_graph lr
                  new_model = updateModelWithGraph model grad_steped_graph
               in Just new_model
  where
    out = forwardMlp model inp
