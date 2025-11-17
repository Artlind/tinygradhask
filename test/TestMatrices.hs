module TestMatrices (testnewMatrix2d, checkMatShapeAndRangeNoKey, checkMatShapeAndRange, testrandMatrix2d, testaddMatrices, testtanH, testMSE) where

import Data.Bifunctor ()
import qualified Data.HashMap.Strict as HM
import Data.Maybe (fromJust, isJust, isNothing)
import Matrices
import System.Random (StdGen, mkStdGen)
import Tinygrad

testnewMatrix2d :: Bool
testnewMatrix2d = passed
  where
    not_rectangle, rectangle :: Maybe Matrix2d
    not_rectangle = newMatrix2d [[createNombre ("a_1_1", 1.0)], []]
    rectangle = newMatrix2d [[createNombre ("a_1_1", 1.0)], [createNombre ("a_2_1", 2.0)]]

    correct_not_rectangle = isNothing not_rectangle
    correct_rectangle = isJust rectangle
    passed = correct_not_rectangle && correct_rectangle

checkMatShapeAndRangeNoKey :: Maybe Matrix2d -> Shape -> Range -> Bool
checkMatShapeAndRangeNoKey Nothing _ _ = False
checkMatShapeAndRangeNoKey (Just mat) (rows, cols) (minval, maxval) =
  length (coeffs mat) == rows
    && all (\row -> length row == cols && all (\n -> let v = value n in v >= minval && v <= maxval) row) (coeffs mat)

-- Util function checking if a matrix matches a shape and a range
checkMatShapeAndRange :: Maybe (Matrix2d, StdGen) -> Shape -> Range -> Bool
checkMatShapeAndRange Nothing _ _ = False
checkMatShapeAndRange (Just (mat, _)) shape range = checkMatShapeAndRangeNoKey (Just mat) shape range

testrandMatrix2d :: Bool
testrandMatrix2d = passed
  where
    wrong_shape, good_shape :: Shape
    wrong_shape = (3, -2)
    good_shape = (3, 2)
    range :: Range
    range = (-1, 1)
    key :: StdGen
    key = mkStdGen 42
    randmat_nothing, randmat :: Maybe (Matrix2d, StdGen)
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

testtanH :: Bool
testtanH = passed
  where
    key :: StdGen
    key = mkStdGen 42
    m3, tanned_m3 :: Matrix2d
    (m3, _) = fromJust $ randMatrix2d "m3" ((3, 3) :: Shape) ((-1, 1) :: Range) key
    tanned_m3 = fromJust $ newMatrix2d [[tanH n | n <- row] | row <- coeffs m3]

    summed :: Nombre
    summed = sumNombre (concat (coeffs tanned_m3))
    all_nombres :: [Nombre]
    all_nombres = [summed] ++ concat (coeffs tanned_m3) ++ concat (coeffs m3)
    graph, backwarded_graphsum :: Graph
    graph = Graph (HM.fromList [(nombre_id node, node) | node <- all_nombres])
    backwarded_graphsum = backward (nombre_id summed) graph

    passed = all and ([[grad (fromJust (getNombreFromId (nombre_id n) backwarded_graphsum)) == (1 - tanh (value n) ** 2) | n <- row] | row <- coeffs m3])

testMSE :: Bool
testMSE = passed
  where
    shape :: Shape
    shape = (3, 2)
    key1, key2 :: StdGen
    key1 = mkStdGen 42
    key2 = mkStdGen 43
    m1, m2, ses :: Matrix2d
    (m1, _) = fromJust $ randMatrix2d "m1" shape ((-1, 1) :: Range) key1
    (m2, _) = fromJust $ randMatrix2d "m2" shape ((-1, 1) :: Range) key2
    ses = fromJust $ meanSquaredError m1 m2

    correct_ses_dim = (length (coeffs ses), length (head (coeffs ses))) == shape
    correct_values = all and ([[value n3 == (value n1 - value n2) ** 2 | (n1, n2, n3) <- zip3 row1 row2 row3] | (row1, row2, row3) <- zip3 (coeffs m1) (coeffs m2) (coeffs ses)])
    sum_ses :: Nombre
    sum_ses = sumNombre (allParamsFromMatrix ses)

    graph, backwarded_graph :: Graph
    graph = Graph (HM.fromList [(nombre_id node, node) | node <- concat [allParamsFromMatrix m1, allParamsFromMatrix m2, allParamsFromMatrix ses, [sum_ses]]])
    backwarded_graph = backward (nombre_id sum_ses) graph

    correct_grads = all and ([[grad (fromJust (getNombreFromId (nombre_id n1) backwarded_graph)) == 2 * (value n1 - value n2) | (n1, n2) <- zip row1 row2] | (row1, row2) <- zip (coeffs m1) (coeffs m2)])
    passed = correct_ses_dim && correct_values && correct_grads
