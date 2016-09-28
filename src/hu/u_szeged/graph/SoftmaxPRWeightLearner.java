package hu.u_szeged.graph;

import java.util.Set;

public class SoftmaxPRWeightLearner extends PRWeightLearner {

  public SoftmaxPRWeightLearner(double[] prs, OwnGraph g) {
    this(prs, g, PageRankCalculator.DEFAULT_TELEPORT, null);
  }

  public SoftmaxPRWeightLearner(double[] prs, OwnGraph g, double teleportProb) {
    this(prs, g, teleportProb, null);
  }

  public SoftmaxPRWeightLearner(double[] prs, OwnGraph g, double teleportProb, Set<Integer> favoredNodeIds) {
    super(prs, g, teleportProb, true, favoredNodeIds);
  }

  /**
   * Not the easiest method to debug, however, it is carefully tested with a minimal chance to contain any bug. <br />
   * The good part is however, that method is quite efficient wrt. speed and memory consumption.
   * 
   * @param buffer
   */
  public void getValueGradient(double[] buffer) {
    long time = System.currentTimeMillis();
    actualizePageRankValues();
    int actualFromNode = 0;
    double[] ws = graph.getWeights(actualFromNode).clone();
    int[] neighbors = graph.getOutLinks(actualFromNode);
    double[] partialSums = new double[neighbors[0]];
    double expSum = 0.0;
    for (int i = 0, k = 1; i <= graph.getNumOfEdges(); ++i, ++k) {
      if (i == cummulatedNeighsNum[actualFromNode + 1]) {
        for (i = cummulatedNeighsNum[actualFromNode]; i < cummulatedNeighsNum[actualFromNode + 1]; ++i) {
          int relativePos = i - cummulatedNeighsNum[actualFromNode];
          buffer[i] = (1 - teleportProbability) * prActual[actualFromNode] * ws[relativePos + 1] * partialSums[relativePos];
          buffer[i] /= (expSum * expSum);
        }
        while (actualFromNode < cummulatedNeighsNum.length - 2 && cummulatedNeighsNum[actualFromNode + 1] == cummulatedNeighsNum[actualFromNode + 2]) {
          actualFromNode += 1; // we have to skip through nodes in case they have no outgoing edges
        }
        if (++actualFromNode == graph.getNumOfNodes()) {
          break;
        }
        neighbors = graph.getOutLinks(actualFromNode).clone();
        partialSums = new double[neighbors[0]];
        expSum = 0;
        k = 1;
        ws = graph.getWeights(actualFromNode).clone();
      }
      expSum += (ws[k] = Math.exp(ws[k]));
      for (int j = 0; j < neighbors[0]; ++j) {
        partialSums[j] += ((prStar[neighbors[j + 1]] / prActual[neighbors[j + 1]])
            - (prStar[neighbors[i - cummulatedNeighsNum[actualFromNode] + 1]] / prActual[neighbors[i - cummulatedNeighsNum[actualFromNode] + 1]])) * ws[k];
      }
    }
    if ((System.currentTimeMillis() - time) / 1000.d > 180.0d) { // only print elapsed time if it was more than 3 minutes
      System.err.format("Softmax gradient in %f\n", (System.currentTimeMillis() - time) / 1000.d);
    }
  }

  public void getValueGradient2(double[] buffer) {
    long time = System.currentTimeMillis();
    actualizePageRankValues();
    int actualFromNode = 0;
    double[] ws = graph.getWeights(actualFromNode).clone();
    int[] neighbors = graph.getOutLinks(actualFromNode);
    double[] partialSums = new double[neighbors[0]];
    double expSum = 0.0;
    for (int i = 0, k = 1; i <= graph.getNumOfEdges(); ++i, ++k) {
      if (i == cummulatedNeighsNum[actualFromNode + 1]) {
        double expSumSquared = expSum * expSum;
        for (i = cummulatedNeighsNum[actualFromNode]; i < cummulatedNeighsNum[actualFromNode + 1]; ++i) {
          int relativeIndex = i - cummulatedNeighsNum[actualFromNode];
          buffer[i] = ((1 - teleportProbability) * prActual[actualFromNode] * ws[relativeIndex] * partialSums[relativeIndex]) / expSumSquared;
        }
        while (actualFromNode < cummulatedNeighsNum.length - 2 && cummulatedNeighsNum[actualFromNode + 1] == cummulatedNeighsNum[actualFromNode + 2]) {
          actualFromNode += 1; // we have to skip through nodes in case they have no outgoing edges
        }
        if (++actualFromNode == graph.getNumOfNodes()) {
          break;
        }
        neighbors = graph.getOutLinks(actualFromNode);
        partialSums = new double[neighbors[0]];
        ws = graph.getWeights(actualFromNode).clone();
        k = 1;
        expSum = 0;
      }
      expSum += (ws[k] = Math.exp(ws[k]));
      for (int j = 0; j < neighbors[0]; ++j) {
        partialSums[j] += ((prStar[neighbors[j + 1]] / prActual[neighbors[j + 1]])
            - (prStar[neighbors[i - cummulatedNeighsNum[actualFromNode] + 1]] / prActual[neighbors[i - cummulatedNeighsNum[actualFromNode] + 1]])) * ws[k];
      }
    }
    addRegularizationGradient(buffer);
    if ((System.currentTimeMillis() - time) / 1000.d > 180.0d) { // only print elapsed time if it was more than 3 minutes
      System.err.format("Softmax gradient in %f\n", (System.currentTimeMillis() - time) / 1000.d);
    }
  }

  protected void actualizePageRankValues() {
    prActual = prc.calculatePageRank(graph, true);
  }

  protected double[] projectVector(double[] vec) {
    // double expSum = 0.0d;
    // for (int i = 0; i < vec.length; ++i) {
    // expSum += Math.exp(vec[i]);
    // }
    // for (int i = 0; i < vec.length; ++i) {
    // vec[i] = Math.log(Math.exp(vec[i]) / expSum);
    // }
    return vec;
  }
}