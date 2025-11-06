module Main (main) where

import qualified Data.HashMap.Strict as HM
import Data.Maybe (fromJust, isJust, isNothing)
import Matrices
import System.Random (StdGen, mkStdGen)
import Tinygrad

testSimpleBackward :: Bool
testSimpleBackward = passed
  where
    a = createNombre ("a", 2)
    b = createNombre ("b", 3)
    c = newNombreWithId ("c", a * b)
    graph = Graph (HM.fromList [(nombre_id node, node) | node <- [a, b, c]])
    backwarded_graph = backward "c" graph
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
    a = createNombre ("a", 2)
    b = createNombre ("b", 3)
    f = createNombre ("f", 4)
    c = newNombreWithId ("c", a * b)
    d = newNombreWithId ("d", c * c)
    e = newNombreWithId ("e", d / f)
    graph = Graph (HM.fromList [(nombre_id node, node) | node <- [a, b, c, d, e, f]])
    backwarded_graph = backward "e" graph
    computed_a_grad = grad $ getNombreFromId "a" backwarded_graph
    computed_b_grad = grad $ getNombreFromId "b" backwarded_graph
    computed_c_grad = grad $ getNombreFromId "c" backwarded_graph
    computed_d_grad = grad $ getNombreFromId "d" backwarded_graph
    computed_f_grad = grad $ getNombreFromId "f" backwarded_graph

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
    not_rectangle = newMatrix2d [[createNombre ("a_1_1", 1.0)], []]
    correct_not_rectangle = isNothing not_rectangle
    rectangle = newMatrix2d [[createNombre ("a_1_1", 1.0)], [createNombre ("a_2_1", 2.0)]]
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
    randmat_nothing = randMatrix2d "A" (3, -2) (-1, 1) (mkStdGen 42)
    randmat_nothing_is_nothing = isNothing randmat_nothing
    randmat = randMatrix2d "A" (3, 2) (-1, 1) (mkStdGen 42)
    randmat_not_nothing = isJust randmat
    has_correct_shape_and_range = checkMatShapeAndRange randmat (3, 2) (-1, 1)
    passed = randmat_not_nothing && randmat_nothing_is_nothing && has_correct_shape_and_range

testaddMatrices :: Bool
testaddMatrices = passed
  where
    -- It is ok to use the ugly fromJust here as we test randMatrix2d in another test
    (m1, _) = fromJust $ randMatrix2d "m1" (3, 2) (-1, 1) (mkStdGen 42)
    (m2, _) = fromJust $ randMatrix2d "m2" (3, 2) (-1, 1) (mkStdGen 43)
    (m3, _) = fromJust $ randMatrix2d "m3" (3, 1) (-1, 1) (mkStdGen 42)
    m4 = addMatrices m1 m2
    m5 = addMatrices m1 m3
    has_correct_shape_and_range = checkMatShapeAndRangeNoKey m4 (3, 2) (-2, 2)
    passed = isNothing m5 && isJust m4 && has_correct_shape_and_range

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
  where
    testSimpleBackwardPassed = testSimpleBackward
    testMoreComplexBackwardPassed = testMoreComplexBackward
