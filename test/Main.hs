module Main (main) where

import qualified Data.HashMap.Strict as HM
import MyLib

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
    c = newNombreWithId ("c", a * b)
    d = newNombreWithId ("d", c * c)
    graph = Graph (HM.fromList [(nombre_id node, node) | node <- [a, b, c, d]])
    backwarded_graph = backward "d" graph
    computed_a_grad = grad $ getNombreFromId "a" backwarded_graph
    computed_b_grad = grad $ getNombreFromId "b" backwarded_graph
    computed_c_grad = grad $ getNombreFromId "c" backwarded_graph
    expected_a_grad = 2 * value b * value b * value a
    expected_b_grad = 2 * value a * value a * value b
    expected_c_grad = 2 * value c
    correct_a_grad = computed_a_grad == expected_a_grad
    correct_b_grad = computed_b_grad == expected_b_grad
    correct_c_grad = computed_c_grad == expected_c_grad
    passed = correct_a_grad && correct_b_grad && correct_c_grad

main :: IO ()
main = do
  if testSimpleBackwardPassed
    then putStrLn "PASSED test simplebackward"
    else putStrLn "FAILED test simplebackward"
  if testMoreComplexBackwardPassed
    then putStrLn "PASSED test morecomplexbackward"
    else putStrLn "FAILED test morecomplexbackward"
  where
    testSimpleBackwardPassed = testSimpleBackward
    testMoreComplexBackwardPassed = testMoreComplexBackward
