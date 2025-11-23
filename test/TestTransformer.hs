module TestTransformer (testnewRandomMultiHeadAttentionHead) where

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
