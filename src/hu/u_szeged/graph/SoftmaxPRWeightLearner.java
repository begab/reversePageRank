package hu.u_szeged.graph;

import java.util.Arrays;
import java.util.Set;

import hu.u_szeged.utils.Utils;

public class SoftmaxPRWeightLearner extends PRWeightLearner {

  public SoftmaxPRWeightLearner(double[] prs, OwnGraph g) {
    this(prs, g, PageRankCalculator.DEFAULT_TELEPORT, null);
  }

  public SoftmaxPRWeightLearner(double[] prs, OwnGraph g, double teleportProb) {
    this(prs, g, teleportProb, null);
  }

  public SoftmaxPRWeightLearner(double[] prs, OwnGraph g, double teleportProb, Set<Integer> favoredNodeIds) {
    super(prs, g, teleportProb, favoredNodeIds);
  }

  public void getValueGradient(double[] buffer) {
    long time = System.currentTimeMillis();
    actualizePageRankValues();
    setPartialDerivatives(buffer, time);
  }

  protected double[] precalculateSum(int[] neighs, double[] ws, double[] oracleBaselineWeights) {
    double sum = 0.0d, regularizationSum = 0.0d;
    for (int n = 1; n <= neighs[0]; ++n) {
      sum += ws[n] * (prStar[neighs[n]] / prActual[neighs[n]]);
      if (regularizationWeight > 0 && ws[n] > 0) {
        if (regularization == RegularizationType.ORACLE) {
          regularizationSum += ws[n] * (ws[n] - oracleBaselineWeights[n]);
        } else if (regularization == RegularizationType.ENTROPY) {
          regularizationSum += ws[n] * (1 + Math.log(ws[n]));
        }
      }
    }
    return new double[] { sum, regularizationSum };
  }

  protected void setPartialDerivatives(double[] buffer, long time) {
    for (int i = 0, j = 0; i < graph.getNumOfNodes(); ++i) {
      int[] neighs = graph.getOutLinks(i);
      double[] ws = Arrays.copyOf(graph.getWeights(i), neighs[0] + 1);
      Utils.softmaxNormalize(ws, neighs[0]);
      double[] oracleBaselineWeights = null;
      if (regularization == RegularizationType.ORACLE && regularizationWeight > 0) {
        oracleBaselineWeights = calculateBaselineWeights(neighs);
      }
      double[] precalculatedSums = precalculateSum(neighs, ws, oracleBaselineWeights);
      double sum = precalculatedSums[0], regularizationSum = precalculatedSums[1];

      for (int n = 2; n <= neighs[0]; ++n, ++j) { // the first (unnormalized) value is constant 0, thus it suffices to start from index 2
        buffer[j] = (1 - teleportProbability) * prActual[i] * ws[n] * (prStar[neighs[n]] / prActual[neighs[n]] - sum);
        if (regularization == RegularizationType.ORACLE && regularizationWeight > 0 && ws[n] > 0) {
          buffer[j] -= regularizationWeight * ws[n] * (ws[n] - oracleBaselineWeights[n] - regularizationSum);
        } else if (regularization == RegularizationType.ENTROPY && regularizationWeight > 0 && ws[n] > 0) {
          buffer[j] += regularizationWeight * ws[n] * (1 + Math.log(ws[n]) - regularizationSum);
        }
      }
    }
    if ((System.currentTimeMillis() - time) / 1000.d > 180.0d) { // only print elapsed time if it was more than 3 minutes
      System.err.format("Softmax gradient in %f\n", (System.currentTimeMillis() - time) / 1000.d);
    }
  }

  public int getNumParameters() {
    int edgesWithVariableWeights = 0;
    for (int i = 0; i < graph.getNumOfNodes(); ++i) {
      int outdegree = graph.getOutDegree(i);
      if (outdegree > 1) {// one (unnormalized) edge weight per node is always assumed to be 0, thus not a parameter
        edgesWithVariableWeights += outdegree - 1;
      }
    }
    return edgesWithVariableWeights;
  }

  public void getParameters(double[] buffer) {
    int i = 0;
    for (int n = 0; n < graph.getNumOfNodes(); ++n) {
      int numOfNeighbors = graph.getNumOfNeighbors(n);
      double[] weights = graph.getWeights(n);
      for (int k = 2; k <= numOfNeighbors; ++k, i++) { // one (unnormalized) edge weight per node is always assumed to be 0, thus not a parameter
        buffer[i] = weights[k];
      }
    }
  }

  public void setParameters(double[] params) {
    int i = 0;
    for (int n = 0; n < graph.getNumOfNodes(); ++n) {
      int[] neighs = graph.getOutLinks(n);
      double[] weights = graph.getWeights(n);
      double sum = 0.0d;
      for (int k = 2; k <= neighs[0]; ++k) {
        sum += (weights[k] = params[i++]);
      }
      weights[0] = sum;
    }
  }

  protected void actualizePageRankValues() {
    prActual = prc.calculatePageRank(graph, true);
  }
}