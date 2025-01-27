import cv2
import numpy as np
import argparse
import sys
import os
import pdb


def pad(img):
  h, w = img.shape
  eximg = np.zeros((h+2, w+2))
  eximg[1:-1, 1:-1] = img
  eximg[0, 1:-1] = img[0, :]
  eximg[-1, 1:-1] = img[-1, :]
  eximg[1:-1, 0] = img[:, 0]
  eximg[1:-1, -1] = img[:, -1]
  return eximg


def get_cost(i, j, eximg, x, weights):
  # potential fonction corresponding to a gaussian markovian model (quadratic function)
  if np.isscalar(x):
    return np.sum(weights * (eximg[i-1:i+2, j-1:j+2] - x)**2)
  else:
    tmp = weights * (eximg[i-1:i+2, j-1:j+2] - x)**2
    return tmp.sum(axis=tuple(range(1, tmp.ndim)))


def ICM(args):
  # ICM : Iterated conditional mode algorithme
  NoisyIm = cv2.imread(args.image, 0)
  NoisyIm = NoisyIm.astype(float)
  height, width = NoisyIm.shape

  sigma2 = 5
  beta = args.beta
  alpha = 1./(2.0*sigma2)

  # weight metrics
  weights = np.array([[0, beta, 0],
                      [beta, alpha, beta],
                      [0, beta, 0]])

  # Each new image is used as the new reconstruction
  xs = np.arange(256)
  xs = xs[..., np.newaxis, np.newaxis]
  for iter in range(args.iter):
    eximg = pad(NoisyIm)
    print("iteration {}\n".format(iter+1))
    for i in range(1, height+1):
      for j in range(1, width+1):
        # 4-neighborhood system
        # Find the a local minimum of the COST corresponding to a Gibbs distribution
        costs = get_cost(i, j, eximg, xs, weights)
        xmin = np.argmin(costs)
        eximg[i][j] = xmin
    NoisyIm = eximg[1:-1, 1:-1]
    cv2.imwrite("data/iter_" + str(iter+1) + "_denoised_" +
                os.path.basename(args.image), NoisyIm)

    
def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--image", type=str, default="data/lennab.jpg",
                      help="Name of the noisy image")
  parser.add_argument("--iter", type=int, default=3,
                      help="Number of iterations for ICM")
  parser.add_argument("--beta", type=float, default=0.3,
                      help="Value of regularisation")
  args = parser.parse_args()
  sys.stdout.write(str(ICM(args)))


if __name__ == '__main__':
  main()
