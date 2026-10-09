{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}

-- Accelerate equivalent of the CUDA map(*) + reduce(+) dot product program.
--
-- Design goals, matching the CUDA program as closely as Accelerate allows:
--   * Same computation: element-wise multiply (map) followed by a sum
--     reduction (reduce), on two Float vectors of length N.
--   * Same data shape: two host-side Float arrays of length N filled with
--     "rand()-like" values, generated and fully forced *before* timing
--     starts (mirroring the fact that the CUDA program calls rand() to
--     fill a[] and b[] before cudaEventRecord(start,...) is issued).
--   * Timing window matches the CUDA program's window as closely as
--     possible: it starts right where the CUDA program's start event is
--     recorded (immediately before the GPU buffers are allocated / data
--     is copied to the device) and ends right where the CUDA program's
--     stop event is recorded (after the result has been copied back to
--     the host and the device buffers have been freed).
--   * Single execution, no warm-up run: `run` is called exactly once.
--     See the README for an important caveat about what this means for
--     Accelerate specifically (JIT/PTX compilation is *included* in the
--     timed run, unlike CUDA which is compiled ahead-of-time by nvcc).

module Main where

import Data.Array.Accelerate                 as A
import Data.Array.Accelerate.LLVM.PTX         as PTX

import Control.DeepSeq                        (force)
import Control.Exception                      (evaluate)
import Data.Time.Clock                         (diffUTCTime, getCurrentTime)
import System.Environment                      (getArgs)
import System.Random                           (newStdGen, randomRs)
import Text.Printf                             (printf)

-- | dot product = fold (+) 0 . zipWith (*) a b
--   This is the direct Accelerate analogue of map_2kernel (the multiply)
--   followed by reduce_kernel (the sum reduction) in the CUDA source.
dotProduct :: Acc (Vector Float) -> Acc (Vector Float) -> Acc (Scalar Float)
dotProduct a b = A.fold (+) 0 (A.zipWith (*) a b)

-- | Generate N pseudo-random floats, in a similar magnitude to C's rand()
--   (POSIX RAND_MAX = 2^31 - 1), since the CUDA program does
--   a[i] = rand(); with an implicit int->float conversion.
genFloats :: Int -> IO [Float]
genFloats n = do
  gen <- newStdGen
  return $ Prelude.map Prelude.fromIntegral
          $ Prelude.take n (randomRs (0 :: Int, 2147483647) gen)

main :: IO ()
main = do
  args <- getArgs
  n <- case args of
         (x:_) -> return (read x :: Int)
         _     -> error "usage: dotprod-accelerate N"

  -- Host-side data generation happens BEFORE the timed region, exactly
  -- like rand() filling a[] / b[] before cudaEventRecord(start,...) in
  -- the CUDA program. `evaluate . force` makes sure the lists are fully
  -- built now, not lazily during the timed region.
  asL <- genFloats n
  bsL <- genFloats n
  _   <- evaluate (force asL)
  _   <- evaluate (force bsL)

  let vecA = A.fromList (Z :. n) asL :: Vector Float
      vecB = A.fromList (Z :. n) bsL :: Vector Float
  -- Force the list -> Array conversion now, so it is not counted below.
  _ <- evaluate (force vecA)
  _ <- evaluate (force vecB)

  -- Timed region: this is the single execution. It covers everything
  -- PTX.run does internally for this program -- host->device transfer,
  -- (first-time) compilation of the Accelerate program to PTX, kernel
  -- launch(es) for the fused map+fold, and device->host transfer of the
  -- scalar result. No prior call to `run`/`run1`/`runN` is made anywhere
  -- above this point, so there is no warm-up.
  t0 <- getCurrentTime
  let result = PTX.run (dotProduct (A.use vecA) (A.use vecB))
  _ <- evaluate (force result)   -- force full evaluation now
  t1 <- getCurrentTime

  let elapsedMs = realToFrac (diffUTCTime t1 t0) * 1000 :: Double
  printf "Accelerate\t%d\t%3.1f\n" n elapsedMs
