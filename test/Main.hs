module Main (main) where

import Data.Bifunctor ()
import qualified Data.HashMap.Strict as HM
import Data.Maybe (fromJust, isJust, isNothing)
import Matrices
import Mlp
import System.Random (StdGen, mkStdGen)
import Tinygrad

testSimpleBackward :: Bool
testSimpleBackward = passed
  where
    a, b, c :: Nombre
    a = createNombre ("a", 2)
    b = createNombre ("b", 3)
    c = newNombreWithId ("c", a * b)

    graph, backwarded_graph :: Graph
    graph = Graph (HM.fromList [(nombre_id node, node) | node <- [a, b, c]])
    backwarded_graph = backward "c" graph

    computed_a_grad, computed_b_grad, expected_a_grad, expected_b_grad :: Double
    computed_a_grad = grad $ getNombreFromId "a" backwarded_graph
    computed_b_grad = grad $ getNombreFromId "b" backwarded_graph
    expected_a_grad = value b
    expected_b_grad = value a

    correct_a_grad = computed_a_grad == expected_a_grad
    correct_b_grad = computed_b_grad == expected_b_grad
    passed = correct_a_grad && correct_b_grad

testMoreComplexBackward :: Bool
testMoreComplexBackward = passed
  where
    a, b, c, d, e, f :: Nombre
    a = createNombre ("a", 2)
    b = createNombre ("b", 3)
    f = createNombre ("f", 4)
    c = newNombreWithId ("c", a * b)
    d = newNombreWithId ("d", c * c)
    e = newNombreWithId ("e", d / f)

    graph, backwarded_graph :: Graph
    graph = Graph (HM.fromList [(nombre_id node, node) | node <- [a, b, c, d, e, f]])
    backwarded_graph = backward "e" graph

    computed_a_grad, computed_b_grad, computed_c_grad, computed_d_grad, computed_f_grad :: Double
    computed_a_grad = grad $ getNombreFromId "a" backwarded_graph
    computed_b_grad = grad $ getNombreFromId "b" backwarded_graph
    computed_c_grad = grad $ getNombreFromId "c" backwarded_graph
    computed_d_grad = grad $ getNombreFromId "d" backwarded_graph
    computed_f_grad = grad $ getNombreFromId "f" backwarded_graph

    expected_a_grad, expected_b_grad, expected_c_grad, expected_d_grad, expected_f_grad :: Double
    expected_d_grad = 1 / value f
    expected_f_grad = -(value d / value f ** 2)
    expected_c_grad = expected_d_grad * 2 * value c
    expected_a_grad = value b * expected_c_grad
    expected_b_grad = value a * expected_c_grad

    correct_a_grad = computed_a_grad == expected_a_grad
    correct_b_grad = computed_b_grad == expected_b_grad
    correct_c_grad = computed_c_grad == expected_c_grad
    correct_d_grad = computed_d_grad == expected_d_grad
    correct_f_grad = computed_f_grad == expected_f_grad
    passed = correct_a_grad && correct_b_grad && correct_c_grad && correct_d_grad && correct_f_grad

testnewMatrix2d :: Bool
testnewMatrix2d = passed
  where
    not_rectangle, rectangle :: Maybe Matrix2d
    not_rectangle = newMatrix2d [[createNombre ("a_1_1", 1.0)], []]
    rectangle = newMatrix2d [[createNombre ("a_1_1", 1.0)], [createNombre ("a_2_1", 2.0)]]

    correct_not_rectangle = isNothing not_rectangle
    correct_rectangle = isJust rectangle
    passed = correct_not_rectangle && correct_rectangle

checkMatShapeAndRangeNoKey :: Maybe Matrix2d -> (Int, Int) -> (Double, Double) -> Bool
checkMatShapeAndRangeNoKey Nothing _ _ = False
checkMatShapeAndRangeNoKey (Just mat) (rows, cols) (minval, maxval) =
  length (coeffs mat) == rows
    && all (\row -> length row == cols && all (\n -> let v = value n in v >= minval && v <= maxval) row) (coeffs mat)

checkMatShapeAndRange :: Maybe (Matrix2d, StdGen) -> (Int, Int) -> (Double, Double) -> Bool
checkMatShapeAndRange Nothing _ _ = False
checkMatShapeAndRange (Just (mat, _)) shape range = checkMatShapeAndRangeNoKey (Just mat) shape range

testrandMatrix2d :: Bool
testrandMatrix2d = passed
  where
    randmat_nothing, randmat :: Maybe (Matrix2d, StdGen)
    wrong_shape, good_shape :: Shape
    wrong_shape = (3, -2)
    good_shape = (3, 2)
    range :: Range
    range = (-1, 1)
    key :: StdGen
    key = mkStdGen 42
    randmat_nothing = randMatrix2d "A" wrong_shape range key
    randmat = randMatrix2d "A" good_shape range key

    randmat_nothing_is_nothing = isNothing randmat_nothing
    randmat_not_nothing = isJust randmat
    has_correct_shape_and_range = checkMatShapeAndRange randmat good_shape range
    passed = randmat_not_nothing && randmat_nothing_is_nothing && has_correct_shape_and_range

testaddMatrices :: Bool
testaddMatrices = passed
  where
    -- It is ok to use the ugly fromJust here as we test randMatrix2d in another test
    shape1, shape2 :: Shape
    shape1 = (3, 2)
    shape2 = (3, 1)
    range :: Range
    low, high :: Double
    low = -1
    high = 1
    range = (low, high)
    key1, key2, key3 :: StdGen
    key1 = mkStdGen 42
    key2 = mkStdGen 43
    key3 = mkStdGen 42
    m1, m2, m3 :: Matrix2d
    (m1, _) = fromJust $ randMatrix2d "m1" shape1 range key1
    (m2, _) = fromJust $ randMatrix2d "m2" shape1 range key2
    (m3, _) = fromJust $ randMatrix2d "m3" shape2 range key3

    m4, m5 :: Maybe Matrix2d
    m4 = addMatrices m1 m2
    m5 = addMatrices m1 m3

    has_correct_shape_and_range = checkMatShapeAndRangeNoKey m4 shape1 (2 * low, 2 * high)
    passed = isNothing m5 && isJust m4 && has_correct_shape_and_range

testSumBackward :: Bool
testSumBackward = passed
  where
    a, b, c, d, e :: Nombre
    a = createNombre ("a", 2)
    b = createNombre ("b", 3)
    c = createNombre ("c", 4)
    d = newNombreWithId ("d", sumNombre [a, b, c])
    e = newNombreWithId ("e", d * d)

    graph, backwarded_graph :: Graph
    graph = Graph (HM.fromList [(nombre_id node, node) | node <- [a, b, c, d, e]])
    backwarded_graph = backward "d" graph

    computed_a_grad, computed_b_grad, computed_c_grad :: Double
    computed_a_grad = grad $ getNombreFromId "a" backwarded_graph
    computed_b_grad = grad $ getNombreFromId "b" backwarded_graph
    computed_c_grad = grad $ getNombreFromId "c" backwarded_graph

    expected_a_grad, expected_b_grad, expected_c_grad :: Double
    expected_a_grad = 1
    expected_b_grad = 1
    expected_c_grad = 1

    correct_a_grad = computed_a_grad == expected_a_grad
    correct_b_grad = computed_b_grad == expected_b_grad
    correct_c_grad = computed_c_grad == expected_c_grad

    backwarded_e_graph :: Graph
    backwarded_e_graph = backward "e" graph

    computed_a_grad_e, computed_b_grad_e, computed_c_grad_e :: Double
    computed_a_grad_e = grad $ getNombreFromId "a" backwarded_e_graph
    computed_b_grad_e = grad $ getNombreFromId "b" backwarded_e_graph
    computed_c_grad_e = grad $ getNombreFromId "c" backwarded_e_graph

    expected_a_grad_e, expected_b_grad_e, expected_c_grad_e :: Double
    expected_a_grad_e = 2 * value d -- d(a**2 + 2*ab + 2* ac + 2*bc)/da = 2a+2b+2c
    expected_b_grad_e = 2 * value d
    expected_c_grad_e = 2 * value d

    correct_a_grad_e = computed_a_grad_e == expected_a_grad_e
    correct_b_grad_e = computed_b_grad_e == expected_b_grad_e
    correct_c_grad_e = computed_c_grad_e == expected_c_grad_e
    passed = (value d == 9.0) && correct_a_grad && correct_b_grad && correct_c_grad && correct_a_grad_e && correct_b_grad_e && correct_c_grad_e

testDotProductBackward :: Bool
testDotProductBackward = passed
  where
    a, b, c, d, e :: Nombre
    a = createNombre ("a", 2)
    b = createNombre ("b", 3)
    c = createNombre ("c", 4)
    d = createNombre ("d", 5)
    e = newNombreWithId ("e", dotProduct [a, b] [c, d])
    f = newNombreWithId ("f", e * e)

    graph, backwarded_graph :: Graph
    graph = Graph (HM.fromList [(nombre_id node, node) | node <- [a, b, c, d, e, f]])
    backwarded_graph = backward "e" graph

    computed_a_grad, computed_b_grad, computed_c_grad, computed_d_grad :: Double
    computed_a_grad = grad $ getNombreFromId "a" backwarded_graph
    computed_b_grad = grad $ getNombreFromId "b" backwarded_graph
    computed_c_grad = grad $ getNombreFromId "c" backwarded_graph
    computed_d_grad = grad $ getNombreFromId "d" backwarded_graph

    expected_a_grad, expected_b_grad, expected_c_grad, expected_d_grad :: Double
    expected_a_grad = value c
    expected_b_grad = value d
    expected_c_grad = value a
    expected_d_grad = value b

    correct_a_grad = computed_a_grad == expected_a_grad
    correct_b_grad = computed_b_grad == expected_b_grad
    correct_c_grad = computed_c_grad == expected_c_grad
    correct_d_grad = computed_d_grad == expected_d_grad

    backwarded_graph_f :: Graph
    backwarded_graph_f = backward "f" graph

    computed_a_grad_f, computed_b_grad_f, computed_c_grad_f, computed_d_grad_f, computed_e_grad_f :: Double
    computed_a_grad_f = grad $ getNombreFromId "a" backwarded_graph_f
    computed_b_grad_f = grad $ getNombreFromId "b" backwarded_graph_f
    computed_c_grad_f = grad $ getNombreFromId "c" backwarded_graph_f
    computed_d_grad_f = grad $ getNombreFromId "d" backwarded_graph_f
    computed_e_grad_f = grad $ getNombreFromId "e" backwarded_graph_f

    expected_a_grad_f, expected_b_grad_f, expected_c_grad_f, expected_d_grad_f, expected_e_grad_f :: Double
    expected_e_grad_f = 2 * value e
    expected_a_grad_f = expected_e_grad_f * value c
    expected_b_grad_f = expected_e_grad_f * value d
    expected_c_grad_f = expected_e_grad_f * value a
    expected_d_grad_f = expected_e_grad_f * value b

    correct_a_grad_f = computed_a_grad_f == expected_a_grad_f
    correct_b_grad_f = computed_b_grad_f == expected_b_grad_f
    correct_c_grad_f = computed_c_grad_f == expected_c_grad_f
    correct_d_grad_f = computed_d_grad_f == expected_d_grad_f
    correct_e_grad_f = computed_e_grad_f == expected_e_grad_f
    passed = (value e == 23.0) && correct_a_grad && correct_b_grad && correct_c_grad && correct_d_grad && correct_a_grad_f && correct_b_grad_f && correct_c_grad_f && correct_d_grad_f && correct_e_grad_f

testtanH :: Bool
testtanH = passed
  where
    shape :: Shape
    shape = (3, 3)
    range :: Range
    range = (-1, 1)
    key :: StdGen
    key = mkStdGen 42
    m3, tanned_m3 :: Matrix2d
    (m3, _) = fromJust $ randMatrix2d "m3" shape range key
    tanned_m3 = fromJust $ newMatrix2d [[tanH n | n <- row] | row <- coeffs m3]

    summed :: Nombre
    summed = sumNombre (concat (coeffs tanned_m3))
    all_nombres :: [Nombre]
    all_nombres = [summed] ++ concat (coeffs tanned_m3) ++ concat (coeffs m3)
    graph, backwarded_graphsum :: Graph
    graph = Graph (HM.fromList [(nombre_id node, node) | node <- all_nombres])
    backwarded_graphsum = backward (nombre_id summed) graph

    passed = all and ([[grad (getNombreFromId (nombre_id n) backwarded_graphsum) == (1 - tanh (value n) ** 2) | n <- row] | row <- coeffs m3])

testSingleMlp :: Bool
testSingleMlp = passed
  where
    shape :: Shape
    shape = (3, 5)
    range :: Range
    range = (-1, 1)
    key :: StdGen
    key = mkStdGen 42
    mat :: Matrix2d
    (mat, _) = fromJust $ randMatrix2d "layer" shape range key
    model :: Mlp
    model = Mlp [mat]

    shape_rand_inputs :: Shape
    shape_rand_inputs = (2, 3)
    key_rand_inputs :: StdGen
    key_rand_inputs = mkStdGen 44
    rand_inputs :: Matrix2d
    (rand_inputs, _) = fromJust $ randMatrix2d "inp" shape_rand_inputs range key_rand_inputs

    mlp_output :: MlpOutput
    mlp_output = fromJust $ forwardMlp model rand_inputs
    hs :: [Matrix2d]
    hs = hidden_states mlp_output
    ot :: Matrix2d
    ot = output_tensor mlp_output

    res_matmul :: Matrix2d
    res_matmul = fromJust $ multMatrices rand_inputs mat

    correct_n_hidden = null hs
    correct_ot_vals = all and [[value n1 == value n2 | (n1, n2) <- zip row1 row2] | (row1, row2) <- zip (coeffs res_matmul) (coeffs ot)]
    passed = correct_n_hidden && correct_ot_vals

testMlp :: Bool
testMlp = passed
  where
    shapes :: [Shape]
    shapes = [(3, 5), (5, 1)]
    ranges :: [Range]
    ranges = [(-1, 1), (-1, 1)]
    model :: Mlp
    model = fromJust $ newRandomMlp [(shape, range, key) | (shape, range, key) <- zip3 shapes ranges [mkStdGen 42, mkStdGen 43]]

    shape_rand_inputs :: Shape
    shape_rand_inputs = (2, 3)
    range_rand_inputs :: Range
    range_rand_inputs = (-1, 1)
    key_rand_inputs :: StdGen
    key_rand_inputs = mkStdGen 44
    rand_inputs :: Matrix2d
    (rand_inputs, _) = fromJust $ randMatrix2d "inp" shape_rand_inputs range_rand_inputs key_rand_inputs

    mlp_output :: MlpOutput
    mlp_output = fromJust $ forwardMlp model rand_inputs
    hs :: [Matrix2d]
    hs = hidden_states mlp_output
    ot :: Matrix2d
    ot = output_tensor mlp_output

    correct_n_hidden = length hs == (2 * (length shapes - 1))
    correct_ot_dim = (length (coeffs ot), length (head (coeffs ot))) == (2, 1)
    passed = correct_n_hidden && correct_ot_dim

testMSE :: Bool
testMSE = passed
  where
    shape :: Shape
    shape = (3, 2)
    range :: Range
    range = (-1, 1)
    key1, key2 :: StdGen
    key1 = mkStdGen 42
    key2 = mkStdGen 43
    m1, m2, ses :: Matrix2d
    (m1, _) = fromJust $ randMatrix2d "m1" shape range key1
    (m2, _) = fromJust $ randMatrix2d "m2" shape range key2
    ses = fromJust $ meanSquaredError m1 m2

    correct_ses_dim = (length (coeffs ses), length (head (coeffs ses))) == shape
    correct_values = all and ([[value n3 == (value n1 - value n2) ** 2 | (n1, n2, n3) <- zip3 row1 row2 row3] | (row1, row2, row3) <- zip3 (coeffs m1) (coeffs m2) (coeffs ses)])
    sum_ses :: Nombre
    sum_ses = sumNombre (allParamsFromMatrix ses)

    graph, backwarded_graph :: Graph
    graph = Graph (HM.fromList [(nombre_id node, node) | node <- concat [allParamsFromMatrix m1, allParamsFromMatrix m2, allParamsFromMatrix ses, [sum_ses]]])
    backwarded_graph = backward (nombre_id sum_ses) graph

    correct_grads = all and ([[grad (getNombreFromId (nombre_id n1) backwarded_graph) == 2 * (value n1 - value n2) | (n1, n2) <- zip row1 row2] | (row1, row2) <- zip (coeffs m1) (coeffs m2)])
    passed = correct_ses_dim && correct_values && correct_grads

testFitBatch :: Bool
testFitBatch = passed
  where
    shapes :: [Shape]
    shapes = [(3, 5), (5, 1)]
    ranges :: [Range]
    ranges = [(-1, 1), (-1, 1)]
    model :: Mlp
    model = fromJust $ newRandomMlp [(shape, range, key) | (shape, range, key) <- zip3 shapes ranges [mkStdGen 42, mkStdGen 43]]

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
    old_suared_errors :: Matrix2d
    old_suared_errors = fromJust $ meanSquaredError (output_tensor initial_forward) rand_labels
    old_sumed_squared_errors :: Nombre
    old_sumed_squared_errors = sumNombre (allParamsFromMatrix old_suared_errors)

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

main :: IO ()
main = do
  if testSimpleBackwardPassed
    then putStrLn "PASSED test simplebackward"
    else putStrLn "FAILED test simplebackward"
  if testMoreComplexBackwardPassed
    then putStrLn "PASSED test morecomplexbackward"
    else putStrLn "FAILED test morecomplexbackward"
  if testnewMatrix2d
    then putStrLn "PASSED test testnewMatrix2d"
    else putStrLn "FAILED test testnewMatrix2d"
  if testrandMatrix2d
    then putStrLn "PASSED test testrandMatrix2d"
    else putStrLn "FAILED test testrandMatrix2d"
  if testaddMatrices
    then putStrLn "PASSED test testaddMatrices"
    else putStrLn "FAILED test testaddMatrices"
  if testSumBackward
    then putStrLn "PASSED test testSumBackward"
    else putStrLn "FAILED test testSumBackward"
  if testDotProductBackward
    then putStrLn "PASSED test testDotProductBackward"
    else putStrLn "FAILED test testDotProductBackward"
  if testtanH
    then putStrLn "PASSED test testtanH"
    else putStrLn "FAILED test testtanH"
  if testMlp
    then putStrLn "PASSED test testMlp"
    else putStrLn "FAILED test testMlp"
  if testSingleMlp
    then putStrLn "PASSED test testSingleMlp"
    else putStrLn "FAILED test testSingleMlp"
  if testMSE
    then putStrLn "PASSED test testMSE"
    else putStrLn "FAILED test testMSE"
  if testFitBatch
    then putStrLn "PASSED test testFitBatch"
    else putStrLn "FAILED test testFitBatch"
  where
    testSimpleBackwardPassed = testSimpleBackward
    testMoreComplexBackwardPassed = testMoreComplexBackward
