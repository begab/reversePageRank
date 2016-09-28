package hu.u_szeged.graph;

import java.util.LinkedList;
import java.util.List;
import java.util.Set;

public class PageRankCalculator {

  public static final double DEFAULT_TELEPORT = 0.1d;
  public static final double DEFAULT_TOL = 1E-12;
  public static final int MAX_NUM_OF_ITERS = 300;
  private Set<Integer> teleportSet;
  private double teleport, tol;

  public PageRankCalculator() {
    this(DEFAULT_TELEPORT, DEFAULT_TOL, null);
  }

  public PageRankCalculator(double beta) {
    this(beta, DEFAULT_TOL, null);
  }

  public PageRankCalculator(double beta, double t) {
    this(beta, t, null);
  }

  public PageRankCalculator(double beta, double t, Set<Integer> favoredNodeIds) {
    tol = t;
    teleport = beta;
    teleportSet = favoredNodeIds;
  }

  public void setFavoredNodes(Set<Integer> favNodes) {
    teleportSet = favNodes;
  }

  private double[] initDistr(int n) {
    double unitRank = 1.0d / n;
    double[] d = new double[n];
    for (int i = 0; i < n; ++i) {
      d[i] = unitRank;
    }
    return d;
  }

  /**
   * Given a graph and its type of normalization it computes the stationary distribution for the Markov Chain corresponding to the weightig of the
   * graph.
   * 
   * @param g
   * @param softmaxNorm
   * @return
   */
  public double[] calculatePageRank(OwnGraph g, boolean softmaxNorm) {
    return calculatePageRank(g, softmaxNorm, false, false)[0];
  }

  /**
   * Given a graph and its type of normalization it computes the stationary distribution for the Markov Chain corresponding to the weightig of the
   * graph. <br/>
   * It can be specified if we want to get all the path towards the stationary distribution by the argument returnPerIterationRanks.
   * 
   * @param g
   * @param softmaxNorm
   * @param returnPerIterationRanks
   * @return
   */
  public double[][] calculatePageRank(OwnGraph g, boolean softmaxNorm, boolean returnPerIterationRanks) {
    return calculatePageRank(g, softmaxNorm, returnPerIterationRanks, false);
  }

  /**
   * Calculates the (personalized) page rank values of the given graph.
   * 
   * @param graph
   * @param softmaxnormalize
   * @param returnPerIterationRanks
   * @param debug
   * @return
   */
  public double[][] calculatePageRank(OwnGraph graph, boolean softmaxnormalize, boolean returnPerIterationRanks, boolean debug) {
    int numOfNodes = graph.getNumOfNodes(), iterations = 0;
    List<double[]> ranksPerIteration = new LinkedList<>();
    double[] ranks = initDistr(numOfNodes);
    double difference = Double.MAX_VALUE;
    while (++iterations < MAX_NUM_OF_ITERS && difference > tol) {
      if (returnPerIterationRanks) {
        ranksPerIteration.add(ranks);
      }
      double[] newRanks = new double[numOfNodes];
      double sum = 0.0d;
      for (int i = 0; i < numOfNodes; ++i) {
        int[] neighs = graph.getOutLinks(i);
        double[] weights = graph.getWeights(i).clone();
        if (softmaxnormalize) {
          graph.softmaxNormalizeWeights(weights, neighs[0]);
        } else {
          graph.normalizeWeights(weights, neighs[0]);
        }
        for (int n = 1; n <= neighs[0]; ++n) {
          double s = (1 - teleport) * weights[n] * ranks[i];
          newRanks[neighs[n]] += s;
          sum += s;
        }
      }
      difference = redistributeMissingPagerank(graph, sum, newRanks, ranks);
      ranks = newRanks;
    }
    ranksPerIteration.add(ranks);
    double[][] ranksToReturn = new double[ranksPerIteration.size()][];
    int t = 0;
    for (double[] r : ranksPerIteration) {
      ranksToReturn[t++] = r;
    }
    printInfo(iterations, ranks, debug);
    return ranksToReturn;
  }

  /**
   * Calculates the difference between two consecutive page rank vectors and redistributes the missing distribution
   * 
   * @param sum
   * @param newRanks
   * @param ranks
   * @return
   */
  private double redistributeMissingPagerank(OwnGraph graph, double sum, double[] newRanks, double[] ranks) {
    int numOfNodes = graph.getNumOfNodes(), denominator = numOfNodes;
    double difference = 0.0d;
    if (teleportSet != null && teleportSet.size() > 0) {
      denominator = teleportSet.size();
    }
    for (int i = 0; i < numOfNodes; ++i) {
      if (denominator == numOfNodes || teleportSet.contains(i)) {
        newRanks[i] += (1 - sum) / denominator; // either there is no teleport set defined or the node is in the teleport set
      }
      newRanks[i] = Math.max(newRanks[i], 0.0d);
      difference += Math.pow(ranks[i] - newRanks[i], 2.0d);
    }
    return difference;
  }

  private void printInfo(int iters, double[] rank, boolean debug) {
    if (iters == MAX_NUM_OF_ITERS) {
      // System.err.format("WARNING: Max number of PR iterations (i.e. %d) performed\n", MAX_NUM_OF_ITERS);
    }

    if (debug) {
      System.err.println(iters + " iterations performed");
      for (int i = 0; i < rank.length; ++i) {
        System.err.print(rank[i] + " ");
      }
      System.err.println();
    }
  }

}
