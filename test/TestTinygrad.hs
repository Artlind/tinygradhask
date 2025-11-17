module TestTinygrad (testSimpleBackward, testSimpleBackwardNoGrad, testMoreComplexBackward, testSumBackward, testDotProductBackward) where

import Data.Bifunctor ()
import qualified Data.HashMap.Strict as HM
import Data.Maybe (fromJust)
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
    computed_a_grad = grad (fromJust (getNombreFromId "a" backwarded_graph))
    computed_b_grad = grad (fromJust (getNombreFromId "b" backwarded_graph))
    expected_a_grad = value b
    expected_b_grad = value a

    correct_a_grad = computed_a_grad == expected_a_grad
    correct_b_grad = computed_b_grad == expected_b_grad
    passed = correct_a_grad && correct_b_grad

testSimpleBackwardNoGrad :: Bool
testSimpleBackwardNoGrad = passed
  where
    a, b, c :: Nombre
    a = createNombre ("a", 2)
    b = nombreNoGrad (createNombre ("b", 3))
    c = newNombreWithId ("c", a * b)

    graph, backwarded_graph :: Graph
    graph = Graph (HM.fromList [(nombre_id node, node) | node <- [a, b, c]])
    backwarded_graph = backward "c" graph

    computed_a_grad, computed_b_grad, expected_a_grad, expected_b_grad :: Double
    computed_a_grad = grad (fromJust (getNombreFromId "a" backwarded_graph))
    computed_b_grad = grad (fromJust (getNombreFromId "b" backwarded_graph))
    expected_a_grad = value b
    expected_b_grad = 0.0

    correct_a_grad = computed_a_grad == expected_a_grad
    correct_b_grad = computed_b_grad == expected_b_grad
    passed = correct_a_grad && correct_b_grad

testMoreComplexBackward :: Bool
testMoreComplexBackward = passed
  where
    a, b, e, f :: Nombre
    a = createNombre ("a", 2)
    b = createNombre ("b", 3)
    f = createNombre ("f", 4)
    e = newNombreWithId ("e", (a * a * b * b) / f)

    graph, backwarded_graph :: Graph
    graph = Graph (HM.fromList [(nombre_id node, node) | node <- [a, b, e, f]])
    backwarded_graph = backward "e" graph

    computed_a_grad, computed_b_grad, computed_f_grad :: Double
    computed_a_grad = grad (fromJust (getNombreFromId "a" backwarded_graph))
    computed_b_grad = grad (fromJust (getNombreFromId "b" backwarded_graph))
    computed_f_grad = grad (fromJust (getNombreFromId "f" backwarded_graph))

    expected_a_grad, expected_b_grad, expected_f_grad :: Double
    expected_f_grad = -(value a ** 2 * value b ** 2 / value f ** 2)
    expected_a_grad = 2 * value a * (value b ** 2) / value f
    expected_b_grad = 2 * value b * (value a ** 2) / value f

    correct_a_grad = computed_a_grad == expected_a_grad
    correct_b_grad = computed_b_grad == expected_b_grad
    correct_f_grad = computed_f_grad == expected_f_grad
    passed = correct_a_grad && correct_b_grad && correct_f_grad

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
    computed_a_grad = grad (fromJust (getNombreFromId "a" backwarded_graph))
    computed_b_grad = grad (fromJust (getNombreFromId "b" backwarded_graph))
    computed_c_grad = grad (fromJust (getNombreFromId "c" backwarded_graph))

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
    computed_a_grad_e = grad (fromJust (getNombreFromId "a" backwarded_e_graph))
    computed_b_grad_e = grad (fromJust (getNombreFromId "b" backwarded_e_graph))
    computed_c_grad_e = grad (fromJust (getNombreFromId "c" backwarded_e_graph))

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
    e = newNombreWithId ("e", fromJust (dotProduct [a, b] [c, d]))
    f = newNombreWithId ("f", e * e)

    graph, backwarded_graph :: Graph
    graph = Graph (HM.fromList [(nombre_id node, node) | node <- [a, b, c, d, e, f]])
    backwarded_graph = backward "e" graph

    computed_a_grad, computed_b_grad, computed_c_grad, computed_d_grad :: Double
    computed_a_grad = grad (fromJust (getNombreFromId "a" backwarded_graph))
    computed_b_grad = grad (fromJust (getNombreFromId "b" backwarded_graph))
    computed_c_grad = grad (fromJust (getNombreFromId "c" backwarded_graph))
    computed_d_grad = grad (fromJust (getNombreFromId "d" backwarded_graph))

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
    computed_a_grad_f = grad (fromJust (getNombreFromId "a" backwarded_graph_f))
    computed_b_grad_f = grad (fromJust (getNombreFromId "b" backwarded_graph_f))
    computed_c_grad_f = grad (fromJust (getNombreFromId "c" backwarded_graph_f))
    computed_d_grad_f = grad (fromJust (getNombreFromId "d" backwarded_graph_f))
    computed_e_grad_f = grad (fromJust (getNombreFromId "e" backwarded_graph_f))

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
