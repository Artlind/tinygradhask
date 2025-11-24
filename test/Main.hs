module Main (main) where

import TestMatrices
import TestMlp
import TestTinygrad
import TestTransformer

main :: IO ()
main = do
  if testSimpleBackward
    then putStrLn "PASSED test simplebackward"
    else putStrLn "FAILED!!!!! test simplebackward"
  if testSimpleBackwardNoGrad
    then putStrLn "PASSED test testSimpleBackwardNoGrad "
    else putStrLn "FAILED!!!!! test testSimpleBackwardNoGrad "
  if testMoreComplexBackward
    then putStrLn "PASSED test morecomplexbackward"
    else putStrLn "FAILED!!!!! test morecomplexbackward"
  if testnewMatrix2d
    then putStrLn "PASSED test testnewMatrix2d"
    else putStrLn "FAILED!!!!! test testnewMatrix2d"
  if testrandMatrix2d
    then putStrLn "PASSED test testrandMatrix2d"
    else putStrLn "FAILED!!!!! test testrandMatrix2d"
  if testaddMatrices
    then putStrLn "PASSED test testaddMatrices"
    else putStrLn "FAILED!!!!! test testaddMatrices"
  if testSumBackward
    then putStrLn "PASSED test testSumBackward"
    else putStrLn "FAILED!!!!! test testSumBackward"
  if testDotProductBackward
    then putStrLn "PASSED test testDotProductBackward"
    else putStrLn "FAILED!!!!! test testDotProductBackward"
  if testtanH
    then putStrLn "PASSED test testtanH"
    else putStrLn "FAILED!!!!! test testtanH"
  if testMlp
    then putStrLn "PASSED test testMlp"
    else putStrLn "FAILED!!!!! test testMlp"
  if testSingleMlp
    then putStrLn "PASSED test testSingleMlp"
    else putStrLn "FAILED!!!!! test testSingleMlp"
  if testMSE
    then putStrLn "PASSED test testMSE"
    else putStrLn "FAILED!!!!! test testMSE"
  if testFitBatch
    then putStrLn "PASSED test testFitBatch"
    else putStrLn "FAILED!!!!! test testFitBatch"
  if testFitBatchFrozenLayers
    then putStrLn "PASSED test testFitBatchFrozenLayers "
    else putStrLn "FAILED!!!!! test testFitBatchFrozenLayers "
  if testnewRandomMultiHeadAttentionHead
    then putStrLn "PASSED test testnewRandomMultiHeadAttentionHead "
    else putStrLn "FAILED!!!!! test testnewRandomMultiHeadAttentionHead "
  if testforwardMultiHeadAttentionHead
    then putStrLn "PASSED test testforwardMultiHeadAttentionHead"
    else putStrLn "FAILED!!!!! test testforwardMultiHeadAttentionHead"
