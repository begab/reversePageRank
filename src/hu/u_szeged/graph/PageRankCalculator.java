package hu.u_szeged.graph;

import java.util.LinkedList;
import java.util.List;
import java.util.Set;

public class PageRankCalculator {

  public static final double DEFAULT_TELEPORT = 0.01d;
  public static final double DEFAULT_TOL = 1E-9;
  public static final int MAX_NUM_OF_ITERS = 200;
  private Set<Integer> teleportSet;
  private double teleport, tol;

  public PageRankCalculator() {
    this(DEFAULT_TELEPORT, DEFAULT_TOL, null);
  }

  /**
   * 
   * @param beta
   *          teleport probability
   */
  public PageRankCalculator(double beta) {
    this(beta, DEFAULT_TOL, null);
  }

  /**
   * 
   * @param beta
   *          teleport probability
   * @param t
   *          tolerance
   */
  public PageRankCalculator(double beta, double t) {
    this(beta, t, null);
  }

  /**
   * 
   * @param beta
   *          teleport probability
   * @param favoredNodeIds
   *          from which set of nodes to possibly restart the random walk
   */
  public PageRankCalculator(double beta, Set<Integer> favoredNodeIds) {
    this(beta, DEFAULT_TOL, favoredNodeIds);
  }

  /**
   * 
   * @param beta
   *          teleport probability
   * @param t
   *          tolerance
   * @param favoredNodeIds
   *          from which set of nodes to possibly restart the random walk
   */
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
    return calculatePageRank(g, softmaxNorm, false)[0];
  }

  /**
   * Calculates the (personalized) page rank values of the given graph. <br/>
   * Given a graph and its type of normalization it computes the stationary distribution for the Markov Chain over the weighted graph. <br/>
   * It can be specified if we want to get all the path towards the stationary distribution by the argument returnPerIterationRanks.
   * 
   * @param graph
   * @param softmaxnormalize
   * @param returnPerIterationRanks
   * @param debug
   * @return
   */
  public double[][] calculatePageRank(OwnGraph graph, boolean softmaxnormalize, boolean returnPerIterationRanks) {
    if (softmaxnormalize) {
      graph.softmaxNormalizeWeights();
    } else {
      graph.normalizeWeights();
    }
    int numOfNodes = graph.getNumOfNodes(), iterations = 0;
    List<double[]> ranksPerIteration = new LinkedList<>();
    double[] ranks = initDistr(numOfNodes);
    double difference = Double.MAX_VALUE;
    while (++iterations < MAX_NUM_OF_ITERS && difference > tol) {
      if (returnPerIterationRanks) {
        ranksPerIteration.add(ranks);
      }
      double[] newRanks = new double[numOfNodes];
      double sum = 0.d;
      for (int i = 0; i < numOfNodes; ++i) {
        int[] neighs = graph.getOutLinks(i);
        double[] weights = graph.getWeights(i);
        for (int n = 1; n <= neighs[0]; ++n) {
          double s = (1 - teleport) * weights[n] * ranks[i];
          newRanks[neighs[n]] += s;
          sum += s;
        }
      }
      difference = redistributeMissingPagerank(graph.getNumOfNodes(), sum, newRanks, ranks);
      ranks = newRanks;
    }
    ranksPerIteration.add(ranks);
    double[][] ranksToReturn = new double[ranksPerIteration.size()][];
    int t = 0;
    for (double[] r : ranksPerIteration) {
      ranksToReturn[t++] = r;
    }
    // printInfo(iterations, ranks);
    if (softmaxnormalize) {
      graph.softmaxDenormalizeWeights();
    }
    return ranksToReturn;
  }

  public double[] calculateUnweightedPageRank(OwnGraph graph) {
    return calculateUnweightedPageRank(graph, false)[0];
  }

  public double[][] calculateUnweightedPageRank(OwnGraph graph, boolean returnPerIterationRanks) {
    double[] actualWeights = graph.getWeights().clone();
    double[] tempWeights = new double[actualWeights.length];
    graph.setWeights(tempWeights);
    double[][] prs = this.calculatePageRank(graph, true, returnPerIterationRanks);
    graph.setWeights(actualWeights);
    return prs;
  }

  /**
   * Calculates the difference between two consecutive page rank vectors and redistributes the missing distribution
   * 
   * @param sum
   * @param newRanks
   * @param ranks
   * @return
   */
  private double redistributeMissingPagerank(int nodeNumber, double sum, double[] newRanks, double[] ranks) {
    int denominator = nodeNumber;
    double difference = 0.0d;
    if (teleportSet != null && teleportSet.size() > 0) {
      denominator = teleportSet.size();
    }
    for (int i = 0; i < nodeNumber; ++i) {
      if (denominator == nodeNumber || teleportSet.contains(i)) {
        newRanks[i] += (1 - sum) / denominator; // either there is no teleport
                                                // set defined or the node is in
                                                // the teleport set
      }
      newRanks[i] = Math.max(newRanks[i], 0.0d);
      difference += Math.pow(ranks[i] - newRanks[i], 2.0d);
    }
    return difference;
  }

  private void printInfo(int iters, double[] rank) {
    if (iters == MAX_NUM_OF_ITERS) {
      // System.err.format("WARNING: Max number of PR iterations (i.e. %d)
      // performed\n", MAX_NUM_OF_ITERS);
    }

    System.err.println(iters + " iterations performed");
    for (int i = 0; i < rank.length; ++i) {
      System.err.print(rank[i] + " ");
    }
    System.err.println();
  }

  public static void main(String[] args) {
    OwnGraph g = new OwnGraph(6);
    int[] froms = { 0, 0, 1, 2, 2, 3, 4 };
    int[] tos = { 1, 2, 2, 3, 5, 4, 5 };
    // double[] weights = { .3, .7, .2, .8, .1, .9, .2, .8, .4 };
    for (int i = 0; i < froms.length; ++i) {
      g.addBidirectionalEdge(froms[i], tos[i]);
    }
    PageRankCalculator prc = new PageRankCalculator(0.2);
    prc.calculatePageRank(g, true);
  }

}