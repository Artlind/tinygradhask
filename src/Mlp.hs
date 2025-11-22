module Mlp (Mlp (..), newMlp, forwardMlp, newRandomMlp, MlpOutput (..), fitBatch, Weight, Bias, addBias, WithBias, LinearLayer, forwardLinear, forwardLinearBatch) where

import qualified Data.HashMap.Strict as HM
import Graphs
import Matrices
import System.Random (StdGen)
import Tinygrad

-- Types
type Weight = Matrix2d

type Bias = Maybe Matrix2d

type LinearLayer = (Weight, Bias)

type WithBias = Bool

newtype Mlp = Mlp
  { layers :: [LinearLayer]
  }
  deriving (Eq, Show)

-- Shape compatibility checks
isCompatibleWeightBias :: LinearLayer -> Maybe LinearLayer
isCompatibleWeightBias (w, Nothing) = Just (w, Nothing)
isCompatibleWeightBias (w, Just b)
  | length (coeffs b) /= 1 = Nothing
  | null (head (coeffs b)) = Nothing
  | length (head (coeffs w)) == length (head (coeffs b)) = Just (w, Just b)
  | otherwise = Nothing

isCompatible :: [LinearLayer] -> Maybe [LinearLayer]
isCompatible [] = Just []
isCompatible [(w, b)] =
  case isCompatibleWeightBias (w, b) of
    Nothing -> Nothing
    Just res -> Just [res]
isCompatible ((w1, b1) : rest) = do
  l1b1compat <- isCompatibleWeightBias (w1, b1)
  case rest of
    ((l2, _) : _) ->
      let cols1 = length (head (coeffs w1))
          rows2 = length (coeffs l2)
       in if cols1 /= rows2
            then Nothing
            else do
              restcomp <- isCompatible rest
              return (l1b1compat : restcomp)

-- Base Mlp utils
newMlp :: [LinearLayer] -> Maybe Mlp
newMlp ms =
  case compatible_dims of
    Just t -> Just (Mlp t)
    Nothing -> Nothing
  where
    compatible_dims :: Maybe [LinearLayer]
    compatible_dims = isCompatible ms

-- Forward
data MlpOutput = MlpOutput
  { hidden_states :: [Matrix2d],
    output_tensor :: Matrix2d
  }
  deriving (Eq, Show)

addBias :: Matrix2d -> Bias -> Maybe Matrix2d
addBias h Nothing = Just h
addBias h (Just b)
  | length (coeffs b) /= 1 = Nothing
  | otherwise = do
      let batch_size = length (coeffs h)
      broadcasted_bias <- newMatrix2d [head (coeffs b) | _ <- [1 .. (batch_size :: Int)]]
      addMatrices h broadcasted_bias

forwardLinear :: LinearLayer -> Matrix2d -> Maybe Matrix2d
forwardLinear (w, Nothing) inp = multMatrices inp w
forwardLinear (w, Just b) inp =
  case first_mult of
    Nothing -> Nothing
    Just t -> do
      addBias t (Just b)
  where
    first_mult :: Maybe Matrix2d
    first_mult = multMatrices inp w

forwardLinearBatch :: LinearLayer -> [Matrix2d] -> Maybe [Matrix2d]
forwardLinearBatch _ [] = Nothing
forwardLinearBatch layer [emb1] = do
  res <- forwardLinear layer emb1
  Just [res]
forwardLinearBatch layer (emb1 : others) = do
  res1 <- forwardLinear layer emb1
  rest <- forwardLinearBatch layer others
  Just (res1 : rest)

-- case first_mult of
--   Nothing -> Nothing
--   Just t -> do
--     addBias t (Just b)
-- where
--   first_mult :: Maybe Matrix2d
--   first_mult = multMatrices inp w

forwardMlp :: Mlp -> Matrix2d -> Maybe MlpOutput
forwardMlp (Mlp []) _ = Just (MlpOutput [] emptyMatrix2d)
forwardMlp (Mlp [l]) inp =
  case res of
    Nothing -> Nothing
    Just t -> Just (MlpOutput [] t)
  where
    res :: Maybe Matrix2d
    res = forwardLinear l inp
forwardMlp (Mlp (l1 : others)) inp =
  case res1 of
    Nothing -> Nothing
    Just t -> do
      let first_layer_hidden = applyTanh t
       in case forwardMlp (Mlp others) first_layer_hidden of
            Nothing -> Nothing
            Just rest -> Just $ MlpOutput (first_layer_hidden : hidden_states rest) (output_tensor rest)
  where
    res1 :: Maybe Matrix2d
    res1 = forwardLinear l1 inp

-- Random Mlp init
addSuffixParamsTobias :: String -> Bias -> Bias
addSuffixParamsTobias _ Nothing = Nothing
addSuffixParamsTobias name (Just b) = Just $ suffixParamsMatrix2d name b

nameLayersBasedOnOrder :: Mlp -> Mlp
nameLayersBasedOnOrder model = Mlp [(suffixParamsMatrix2d (show i) w, addSuffixParamsTobias (show i) b) | (i, (w, b)) <- zip [0 .. length (layers model) - 1] (layers model)]

newLinearLayer :: (Shape, Range, StdGen, WithBias) -> Maybe LinearLayer
newLinearLayer ((indim, outdim), (minrange, maxrange), key, with_bias) = do
  (w, new_key) <- randMatrix2d "weight" (indim, outdim) (minrange, maxrange) key
  if with_bias
    then do
      (b, _) <- randMatrix2d "bias" (1, outdim) (minrange, maxrange) new_key
      Just (w, Just b)
    else
      Just (w, Nothing)

defaultRangeForShape :: Shape -> Maybe Range
defaultRangeForShape (in_features, _)
  | in_features < 0 = Nothing
  | otherwise = Just (-(sqrt (fromIntegral in_features)), sqrt (fromIntegral in_features))

newRandomMlp :: [(Shape, StdGen, WithBias)] -> Maybe Mlp
newRandomMlp [] = Just (Mlp [])
newRandomMlp [(shape, key, with_bias)] = do
  range <- defaultRangeForShape shape
  layer <- newLinearLayer (shape, range, key, with_bias)
  newMlp [layer]
newRandomMlp ((shape, key, with_bias) : other_layers_params) = do
  -- let argslayer1 = head dims_and_ranges
  range <- defaultRangeForShape shape
  layer <- newLinearLayer (shape, range, key, with_bias)
  rest <- newRandomMlp other_layers_params
  mlp <- newMlp (layer : layers rest)
  return (nameLayersBasedOnOrder mlp)

-- Training
updateLinearLayerWithGraph :: LinearLayer -> Graph -> LinearLayer
updateLinearLayerWithGraph (w, Nothing) g = (updateMatrixWithGraph w g, Nothing)
updateLinearLayerWithGraph (w, Just b) g = (updateMatrixWithGraph w g, Just (updateMatrixWithGraph b g))

updateModelWithGraph :: Mlp -> Graph -> Mlp
updateModelWithGraph model graph = new_model
  where
    new_model = Mlp [updateLinearLayerWithGraph layer graph | layer <- layers model]

allParamsFromLinear :: LinearLayer -> [Nombre]
allParamsFromLinear (w, Nothing) = allParamsFromMatrix w
allParamsFromLinear (w, Just b) = allParamsFromMatrix w ++ allParamsFromMatrix b

allParamsFromModel :: Mlp -> [Nombre]
allParamsFromModel model = concat [allParamsFromLinear layer | layer <- layers model]

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
