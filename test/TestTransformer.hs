module TestTransformer (testnewRandomMultiHeadAttentionHead, testforwardMultiHeadAttentionHead) where

import Data.Maybe (fromJust)
import Matrices
import System.Random (mkStdGen)
import Transformer

testnewRandomMultiHeadAttentionHead :: Bool
testnewRandomMultiHeadAttentionHead = passed
  where
    hidden_dim = 12
    keyqueriesdim = 5
    valsdim = 8
    key = mkStdGen 42
    n_heads = 5
    multiheadattention = fromJust $ newRandomMultiHeadAttentionHead (hidden_dim, keyqueriesdim, valsdim, key, n_heads)
    (fpw, _) = final_proj multiheadattention
    corect_dims_fpw = (length (coeffs fpw) == (n_heads * valsdim)) && (length (head (coeffs fpw)) == hidden_dim)
    passed = (length (heads multiheadattention) == n_heads) && corect_dims_fpw

testforwardMultiHeadAttentionHead :: Bool
testforwardMultiHeadAttentionHead = passed
  where
    hidden_dim = 12
    keyqueriesdim = 5
    valsdim = 8
    key = mkStdGen 42
    n_heads = 5
    multiheadattention = fromJust $ newRandomMultiHeadAttentionHead (hidden_dim, keyqueriesdim, valsdim, key, n_heads)
    key_rand_inputs = mkStdGen 43
    sequence_length = 9
    batch_size = 4
    sequences = [fst (fromJust (randMatrix2d "inp" ((batch_size, hidden_dim) :: Shape) ((-1, 1) :: Range) key_rand_inputs)) | _ <- [1 .. (sequence_length :: Int)]]
    result = fromJust $ forwardMultiHeadAttention multiheadattention sequences
    passed = (length (coeffs (head result)) == batch_size) && (length result == sequence_length) && (length (head(coeffs (head result))) == hidden_dim)
