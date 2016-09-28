package hu.u_szeged.graph;

import java.util.Set;

import hu.u_szeged.utils.Utils;

public class ProjectedPRWeightLearner extends PRWeightLearner {

  public ProjectedPRWeightLearner(double[] prs, OwnGraph g) {
    this(prs, g, PageRankCalculator.DEFAULT_TELEPORT, null);
  }

  public ProjectedPRWeightLearner(double[] prs, OwnGraph g, double teleportProb) {
    this(prs, g, teleportProb, null);
  }

  public ProjectedPRWeightLearner(double[] prs, OwnGraph g, double teleportProb, Set<Integer> favoredNodeIds) {
    super(prs, g, teleportProb, false, favoredNodeIds);
  }

  public void getValueGradient(double[] buffer) {
    long time = System.currentTimeMillis();
    actualizePageRankValues();
    int edge = 0;
    for (int i = 0; i < graph.getNumOfNodes(); ++i) {
      int[] neighs = graph.getOutLinks(i);
      for (int n = 1; n <= neighs[0]; ++n, ++edge) {
        buffer[edge] = (1 - teleportProbability) * (prActual[i] * prStar[neighs[n]] / prActual[neighs[n]]);
      }
    }
    // System.err.println(Arrays.toString(buffer));
    addRegularizationGradient(buffer);
    if ((System.currentTimeMillis() - time) / 1000.d > 180.0d) { // only print elapsed time if it was more than 3 minutes
      System.err.format("Gradient in %f\n", (System.currentTimeMillis() - time) / 1000.d);
    }
  }

  protected void actualizePageRankValues() {
    prActual = prc.calculatePageRank(graph, false);
  }

  /**
   * Projects a vector to the probability simplex as described in <a href="http://arxiv.org/pdf/1309.1541.pdf">http://arxiv.org/pdf/1309.1541.pdf</a>.
   */
  @Override
  protected double[] projectVector(double[] vec) {
    double etalonSum = 1.0d, compensation = (1.0 - etalonSum) / vec.length;
    int[] order = Utils.stableSort(vec);
    int t = 0;
    double cumSum = 0.0d, ut = 0.0;
    while (t < order.length) {
      cumSum += (ut = vec[order[order.length - 1 - t]]);
      if (ut + (etalonSum - cumSum) / (t + 1) > 0.0d) {
        t++;
      } else {
        cumSum -= ut;
        break;
      }
    }
    double l = (etalonSum - cumSum) / t;
    for (int i = 0; i < vec.length; ++i) {
      vec[i] = Math.max(vec[i] + l, 0) + compensation;
    }
    return vec;
  }
}