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
