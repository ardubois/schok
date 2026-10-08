{-# LANGUAGE FlexibleContexts #-}

-- Accelerate dot product WITH a warm-up, before the timed run.
--
-- Differences from Main.hs (the no-warm-up version):
--   * The program is compiled once with `PTX.runN`, giving an ordinary
--     function  Vector Float -> Vector Float -> Scalar Float.
--   * That function is first applied to a SMALL dummy input (untimed).
--     This pays for everything that is a one-off cost in Accelerate:
--       - initialising the PTX backend / CUDA context
--       - JIT compilation of the fused zipWith/fold program to PTX
--       - loading the compiled module onto the device
--   * Only then is the same function applied to the real data, and only
--     that second application is timed. It therefore measures
--     host->device transfer + kernel execution + device->host transfer,
--     with no compilation cost.
--
-- The warm-up deliberately uses a small, fixed size rather than N: a
-- warm-up at size N would also pre-populate Accelerate's device memory
-- cache, so the timed run would skip device allocation, which the CUDA
-- program does pay for inside its timed window.

module Main where

import Data.Array.Accelerate                 as A
import Data.Array.Accelerate.LLVM.PTX         as PTX

import Control.DeepSeq                        (force)
import Control.Exception                      (evaluate)
import Data.Time.Clock                         (diffUTCTime, getCurrentTime)
import System.Environment                      (getArgs)
import System.Random                           (newStdGen, randomRs)
import Text.Printf                             (printf)

dotProduct :: Acc (Vector Float) -> Acc (Vector Float) -> Acc (Scalar Float)
dotProduct a b = A.fold (+) 0 (A.zipWith (*) a b)

genFloats :: Int -> IO [Float]
genFloats n = do
  gen <- newStdGen
  return $ Prelude.map Prelude.fromIntegral
          $ Prelude.take n (randomRs (0 :: Int, 2147483647) gen)

warmupSize :: Int
warmupSize = 1024

main :: IO ()
main = do
  args <- getArgs
  n <- case args of
         (x:_) -> return (read x :: Int)
         _     -> error "usage: dotprod-accelerate-warmup N"

  -- Host-side data, fully built before any timing.
  asL <- genFloats n
  bsL <- genFloats n
  _   <- evaluate (force asL)
  _   <- evaluate (force bsL)

  let vecA = A.fromList (Z :. n) asL :: Vector Float
      vecB = A.fromList (Z :. n) bsL :: Vector Float
  -- Force the list -> Array conversion now, so it is not counted below.
  _ <- evaluate vecA
  _ <- evaluate vecB

  let dot = PTX.runN dotProduct :: Vector Float -> Vector Float -> Scalar Float

  -- Warm-up (NOT timed): small dummy input, same compiled program.
  let wA = A.fromList (Z :. warmupSize) (Prelude.replicate warmupSize 1) :: Vector Float
      wB = A.fromList (Z :. warmupSize) (Prelude.replicate warmupSize 1) :: Vector Float
  _ <- evaluate (dot wA wB `A.indexArray` Z)

  -- Timed run: transfer + compute + transfer back, no compilation.
  t0 <- getCurrentTime
  let result = dot vecA vecB
  --_ <- evaluate (force result)
  --_ <- evaluate (A.indexArray result Z)
  __ <- evaluate (head . toList result)
  t1 <- getCurrentTime

  let elapsedMs = realToFrac (diffUTCTime t1 t0) * 1000 :: Double
  printf "Accelerate-warmup\t%d\t%3.1f\n" n elapsedMs
