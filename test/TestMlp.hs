module TestMlp (testSingleMlp, testMlp, testFitBatch, testFitBatchFrozenLayers) where

import Data.Bifunctor ()
import Data.Maybe (fromJust)
import Matrices
import Mlp
import System.Random (StdGen, mkStdGen)
import Tinygrad

testSingleMlp :: Bool
testSingleMlp = passed
  where
    key :: StdGen
    key = mkStdGen 42
    (w, new_key) = fromJust $ randMatrix2d "weight" ((3, 5) :: Shape) ((-1, 1) :: Range) key
    (b, _) = fromJust $ randMatrix2d "bias" ((1, 5) :: Shape) ((-1, 1) :: Range) new_key
    model = Mlp [(w :: Weight, Just b :: Bias)]

    key_rand_inputs :: StdGen
    key_rand_inputs = mkStdGen 44
    (rand_inputs, _) = fromJust $ randMatrix2d "inp" ((2, 3) :: Shape) ((-1, 1) :: Range) key_rand_inputs

    mlp_output :: MlpOutput
    mlp_output = fromJust $ forwardMlp (model :: Mlp) (rand_inputs :: Matrix2d)
    hs :: [Matrix2d]
    hs = hidden_states mlp_output
    ot :: Matrix2d
    ot = output_tensor mlp_output

    res_matmul :: Matrix2d
    res_matmul = fromJust $ multMatrices (rand_inputs :: Matrix2d) (w :: Matrix2d)
    res_computed_forward = fromJust $ addBias res_matmul (Just b)

    correct_n_hidden = null hs
    correct_ot_vals = all and [[value n1 == value n2 | (n1, n2) <- zip row1 row2] | (row1, row2) <- zip (coeffs res_computed_forward) (coeffs ot)]
    passed = correct_n_hidden && correct_ot_vals

testMlp :: Bool
testMlp = passed
  where
    shapes :: [Shape]
    shapes = [(3, 5), (5, 1)]
    ranges :: [Range]
    ranges = [(-1, 1), (-1, 1)]
    model :: Mlp
    model = fromJust $ newRandomMlp [(shape, range, key, True :: WithBias) | (shape, range, key) <- zip3 shapes ranges [mkStdGen 42, mkStdGen 43]]

    key_rand_inputs :: StdGen
    key_rand_inputs = mkStdGen 44
    rand_inputs :: Matrix2d
    (rand_inputs, _) = fromJust $ randMatrix2d "inp" ((2, 3) :: Shape) ((-1, 1) :: Range) key_rand_inputs

    mlp_output :: MlpOutput
    mlp_output = fromJust $ forwardMlp (model :: Mlp) (rand_inputs :: Matrix2d)
    hs :: [Matrix2d]
    hs = hidden_states mlp_output
    ot :: Matrix2d
    ot = output_tensor mlp_output

    correct_n_hidden = length hs == (length shapes - 1)
    correct_ot_dim = (length (coeffs ot), length (head (coeffs ot))) == (2, 1)
    passed = correct_n_hidden && correct_ot_dim

testFitBatch :: Bool
testFitBatch = passed
  where
    shapes :: [Shape]
    shapes = [(3, 5), (5, 1)]
    ranges :: [Range]
    ranges = [(-1, 1), (-1, 1)]
    model :: Mlp
    model = fromJust $ newRandomMlp [(shape, range, key, True :: WithBias) | (shape, range, key) <- zip3 shapes ranges [mkStdGen 42, mkStdGen 43]]

    batch_size :: Int
    batch_size = 3
    shape_input, shape_output :: Shape
    shape_input = (batch_size, fst (head shapes))
    shape_output = (batch_size, snd (last shapes))
    range_input_and_labels :: Range
    range_input_and_labels = (-1, 1)
    key_input, key_output :: StdGen
    key_input = mkStdGen 42
    key_output = mkStdGen 44
    rand_input :: Matrix2d
    (rand_input, _) = fromJust $ randMatrix2d "rand_input" shape_input range_input_and_labels key_input
    rand_labels :: Matrix2d
    (rand_labels, _) = fromJust $ randMatrix2d "rand_labels" shape_output range_input_and_labels key_output
    initial_forward :: MlpOutput
    initial_forward = fromJust $ forwardMlp model rand_input
    old_squared_errors :: Matrix2d
    old_squared_errors = fromJust $ meanSquaredError (output_tensor initial_forward) rand_labels
    old_sumed_squared_errors :: Nombre
    old_sumed_squared_errors = sumNombre (allParamsFromMatrix old_squared_errors)

    lr = 0.01
    new_model :: Mlp
    new_model = fromJust $ fitBatch model (rand_input, rand_labels) lr
    new_forward :: MlpOutput
    new_forward = fromJust $ forwardMlp new_model rand_input
    new_squared_errors :: Matrix2d
    new_squared_errors = fromJust $ meanSquaredError (output_tensor new_forward) rand_labels
    new_sumed_squared_errors :: Nombre
    new_sumed_squared_errors = sumNombre (allParamsFromMatrix new_squared_errors)

    passed = value old_sumed_squared_errors > value new_sumed_squared_errors

testFitBatchFrozenLayers :: Bool
testFitBatchFrozenLayers = passed
  where
    shapes :: [Shape]
    shapes = [(3, 5), (5, 1)]
    ranges :: [Range]
    ranges = [(-1, 1), (-1, 1)]
    model, model_frozen :: Mlp
    model = fromJust $ newRandomMlp [(shape, range, key, True :: WithBias) | (shape, range, key) <- zip3 shapes ranges [mkStdGen 42, mkStdGen 43]]
    model_frozen = fromJust $ newMlp ([(if i /= 0 then w else fromJust (newMatrix2d [[nombreNoGrad n | n <- row] | row <- coeffs w]), b) | (i, (w, b)) <- zip [(0 :: Int) ..] (layers model)])

    batch_size :: Int
    batch_size = 3
    shape_input, shape_output :: Shape
    shape_input = (batch_size, fst (head shapes))
    shape_output = (batch_size, snd (last shapes))
    range_input_and_labels :: Range
    range_input_and_labels = (-1, 1)
    key_input, key_output :: StdGen
    key_input = mkStdGen 42
    key_output = mkStdGen 44
    rand_input :: Matrix2d
    (rand_input, _) = fromJust $ randMatrix2d "rand_input" shape_input range_input_and_labels key_input
    rand_labels :: Matrix2d
    (rand_labels, _) = fromJust $ randMatrix2d "rand_labels" shape_output range_input_and_labels key_output
    initial_forward :: MlpOutput
    initial_forward = fromJust $ forwardMlp model rand_input
    old_squared_errors :: Matrix2d
    old_squared_errors = fromJust $ meanSquaredError (output_tensor initial_forward) rand_labels
    old_sumed_squared_errors :: Nombre
    old_sumed_squared_errors = sumNombre (allParamsFromMatrix old_squared_errors)

    lr = 0.01
    new_model :: Mlp
    new_model = fromJust $ fitBatch model (rand_input, rand_labels) lr
    new_forward :: MlpOutput
    new_forward = fromJust $ forwardMlp new_model rand_input
    new_squared_errors :: Matrix2d
    new_squared_errors = fromJust $ meanSquaredError (output_tensor new_forward) rand_labels
    new_sumed_squared_errors :: Nombre
    new_sumed_squared_errors = sumNombre (allParamsFromMatrix new_squared_errors)

    new_model_frozenw1 :: Mlp
    new_model_frozenw1 = fromJust $ fitBatch model_frozen (rand_input, rand_labels) lr
    new_forward_frozenw1 :: MlpOutput
    new_forward_frozenw1 = fromJust $ forwardMlp new_model_frozenw1 rand_input
    new_squared_errors_frozenw1 :: Matrix2d
    new_squared_errors_frozenw1 = fromJust $ meanSquaredError (output_tensor new_forward_frozenw1) rand_labels
    new_sumed_squared_errors_frozenw1 :: Nombre
    new_sumed_squared_errors_frozenw1 = sumNombre (allParamsFromMatrix new_squared_errors_frozenw1)

    -- checking that all weights are different except the first layer that we froze
    freezing_worked =
      and
        [ and
            [and [(value n_old == value n_new) == (i == 0) | (n_old, n_new) <- zip row_old row_new_frozen] | (row_old, row_new_frozen) <- zip (coeffs wi_old) (coeffs wi_new_frozen)]
          | ( i,
              (wi_old, _),
              (wi_new_frozen, _)
              ) <-
              zip3 [(0 :: Int) ..] (layers model_frozen) (layers new_model_frozenw1)
        ]

    passed =
      (value old_sumed_squared_errors > value new_sumed_squared_errors)
        && (value old_sumed_squared_errors > value new_sumed_squared_errors_frozenw1)
        && (value new_sumed_squared_errors_frozenw1 > value new_sumed_squared_errors)
        && freezing_worked
