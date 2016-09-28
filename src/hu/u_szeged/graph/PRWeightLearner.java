package hu.u_szeged.graph;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.Set;

import cc.mallet.optimize.Optimizable;
import cc.mallet.optimize.OptimizationException;
import cc.mallet.optimize.Optimizer;
import cc.mallet.optimize.OwnLimitedMemoryBFGS;
import hu.u_szeged.graph.OwnGraph.WeightingStrategy;
import hu.u_szeged.utils.Utils;

public abstract class PRWeightLearner implements Optimizable.ByGradientValue {

  protected PrintStream log;

  private boolean extensiveSerialization;
  private String serializationPrefix;

  private boolean softmax;
  private double regularizationWeight;

  protected int[] cummulatedNeighsNum; // for efficient lookup of a specific edge by index (NB: edges are ordered lexicographically;e.g. (1,10)<(2,3))

  protected double[] prStar;
  protected double[] prInitial; // let's keep the original (i.e. unweighted) pagerank scores as well
  protected double[] prActual;

  protected double teleportProbability;
  protected OwnGraph graph;
  protected PageRankCalculator prc;

  protected abstract void actualizePageRankValues();

  protected abstract double[] projectVector(double[] vec);

  public PRWeightLearner(double[] prs, OwnGraph g, boolean softmaxNorm) {
    this(prs, g, PageRankCalculator.DEFAULT_TELEPORT, softmaxNorm, null);
  }

  public PRWeightLearner(double[] prs, OwnGraph g, double teleportProb, boolean softmaxNorm) {
    this(prs, g, teleportProb, softmaxNorm, null);
  }

  public PRWeightLearner(double[] prs, OwnGraph g, double teleportProb, boolean softmaxNorm, Set<Integer> favoredNodeIds) {
    log = System.err;
    prStar = prs;
    graph = g;
    softmax = softmaxNorm;
    teleportProbability = teleportProb;
    prc = new PageRankCalculator(teleportProbability);
    prc.setFavoredNodes(favoredNodeIds);
    prInitial = prc.calculatePageRank(graph, softmax);
    updateCummulatedNeighsNum();
  }

  public void updateCummulatedNeighsNum() {
    cummulatedNeighsNum = new int[graph.getNumOfNodes() + 1];
    for (int n = 1; n < cummulatedNeighsNum.length; ++n) {
      cummulatedNeighsNum[n] = cummulatedNeighsNum[n - 1] + graph.getNumOfNeighbors(n - 1);
    }
  }

  public boolean getSoftmax() {
    return softmax;
  }

  public double[] getActualPRvalues() {
    if (prActual == null) {
      prActual = prc.calculatePageRank(graph, softmax);
    }
    return prActual;
  }

  public void setExtensiveSerialization(String prefix) {
    if (prefix != null) {
      extensiveSerialization = true;
      serializationPrefix = prefix;
    }
  }

  public double getActualPRvalue(int i) {
    return getActualPRvalues()[i];
  }

  public double[] getInitialPRvalue() {
    return prInitial;
  }

  public double getInitialPRvalue(int i) {
    return prInitial[i];
  }

  public double[] getEtalonPRvalues() {
    return prStar;
  }

  public double getEtalonPRvalue(int i) {
    return prStar[i];
  }

  public double[] learnEdgeWeights(int numOfInitializations, boolean useAveragedModel) {
    return learnEdgeWeights(numOfInitializations, false, useAveragedModel)[0];
  }

  public double[][] learnEdgeWeights(int numOfInitializations, boolean returnAllParametersLearned, boolean averageModels) {
    double bestFinalObjVal = Double.NEGATIVE_INFINITY;
    double[] bestParameters = new double[getNumParameters()];
    double[][] toReturn = new double[(returnAllParametersLearned ? numOfInitializations : 0) + 3][];
    toReturn[0] = new double[2 * (numOfInitializations + 1)]; // contains the objective values and the index of the best initialization
    toReturn[toReturn.length - 2] = new double[getNumParameters()]; // the last row contains an averaged model over the different initializations
    toReturn[toReturn.length - 1] = new double[numOfInitializations + 1]; // stores the per initialization and the aggregated runtimes
    for (int i = 0; i < numOfInitializations; ++i) {
      if (i == 0) {
        graph.initWeights(WeightingStrategy.UNIFORM);
      } else {
        graph.initWeights(WeightingStrategy.RAND);
      }
      System.err.println(Arrays.toString(Arrays.copyOf(graph.getWeights(), 11)));
      long time = System.currentTimeMillis();
      double[] objVals = learnEdgeWeights();
      toReturn[toReturn.length - 1][i] = (System.currentTimeMillis() - time) / 1000.0d;
      toReturn[toReturn.length - 1][numOfInitializations] += toReturn[toReturn.length - 1][i];
      if (softmax) {
        graph.softmaxNormalizeWeights();
      } else {
        graph.normalizeWeights();
      }
      if (extensiveSerialization) {
        serializeWeights(String.format("%s_model%d.ser", serializationPrefix, i));
      }
      toReturn[0][2 * i] = objVals[0]; // this is the initial objective value
      toReturn[0][2 * i + 1] = objVals[1]; // this is the objective value obtained after the optimization

      double[] actualParameters = new double[getNumParameters()];
      getParameters(actualParameters);
      for (int k = 0; k < actualParameters.length; ++k) {
        toReturn[toReturn.length - 2][k] += actualParameters[k];
      }

      if (bestFinalObjVal < toReturn[0][2 * i + 1]) {
        toReturn[0][toReturn[0].length - 2] = i + 1; // store the index of the best run
        bestFinalObjVal = toReturn[0][2 * i + 1];
        bestParameters = actualParameters;
      }

      if (returnAllParametersLearned) {
        toReturn[i + 1] = actualParameters;
      }
      System.err.format("Random init. nr. %d is over with an initial and final objective value of %f and %f.\n", i + 1, objVals[0], objVals[1]);
    }
    for (int i = 0; i < toReturn[toReturn.length - 2].length; ++i) {
      toReturn[toReturn.length - 2][i] /= numOfInitializations;
    }
    if (averageModels) {
      graph.setWeights(toReturn[toReturn.length - 2]); // let's set the graph to the averaged model and not the best one
    } else {
      graph.setWeights(bestParameters);
    }
    double[] prFinal = prc.calculatePageRank(graph, false); // in this case weights should already be normalized
    toReturn[0][toReturn[0].length - 1] = getValue(prFinal); // store the objective value of the averaged model
    return toReturn;
  }

  private double[] learnEdgeWeights() {
    double initObjVal = getValue();
    for (int i = 0; i < 1; ++i) {
      Optimizer optimizer = new OwnLimitedMemoryBFGS(this);
      try {
        optimizer.optimize();
      } catch (OptimizationException | IllegalArgumentException e) {
        System.err.println(e.getLocalizedMessage()); // This condition does not necessarily mean that the optimizer has failed (but it might has)
      }
    }
    double finalObjective = getValue();
    return new double[] { initObjVal, finalObjective };
  }

  public void setLogFile(String outFile) {
    try {
      if (outFile == null) {
        log = null;
      } else {
        log = new PrintStream(new FileOutputStream(outFile));
      }
    } catch (FileNotFoundException e) {
      e.printStackTrace();
    }
  }

  public OwnGraph getGraph() {
    return graph;
  }

  public void setRegularizationWeight(double r) {
    regularizationWeight = r;
  }

  public int getNumParameters() {
    return graph.getNumOfEdges();
  }

  public void getParameters(double[] buffer) {
    int i = 0;
    for (int n = 0; n < graph.getNumOfNodes(); ++n) {
      int numOfNeighbors = graph.getNumOfNeighbors(n);
      double[] weights = graph.getWeights(n);
      for (int k = 1; k <= numOfNeighbors; ++k, i++) {
        buffer[i] = weights[k];
      }
    }
  }

  public void setParameters(double[] params) {
    int i = 0;
    for (int n = 0; n < graph.getNumOfNodes(); ++n) {
      int[] neighs = graph.getOutLinks(n);
      double[] weights = graph.getWeights(n);
      double[] newWeights = new double[neighs[0]];
      for (int k = 0; k < neighs[0]; ++k, ++i) {
        newWeights[k] = params[i];
      }
      double sum = 0.0d;
      newWeights = projectVector(newWeights);
      for (int j = 1; j <= neighs[0]; ++j) {
        sum += (weights[j] = newWeights[j - 1]);
      }
      weights[0] = sum;
    }
    // System.err.println("PARAMS set");
  }

  public double getParameter(int index) {
    try {
      throw new Exception("We shall not rely on elementwise getting of a parameter.");
    } catch (Exception e) {
      e.printStackTrace();
    }
    int from = Arrays.binarySearch(cummulatedNeighsNum, index);
    if (from < 0) {
      from = -from - 2;
    }
    index -= cummulatedNeighsNum[from];
    // System.err.format("%d\t%d\t%.4f\n", from, graph.getNeighbors(from)[index + 1], graph.getWeights(from)[index + 1]);
    return graph.getWeights(from)[index];
  }

  public void setParameter(int index, double value) {
    try {
      throw new Exception("We shall not rely on elementwise setting of parameters.");
    } catch (Exception e) {
      e.printStackTrace();
    }
    int from = Arrays.binarySearch(cummulatedNeighsNum, index);
    if (from < 0) {
      from = -from - 2;
    }
    index -= cummulatedNeighsNum[from];
    // System.err.format("%d\t%d\t%.4f\n", from, graph.getNeighbors(from)[index + 1], graph.getWeights(from)[index + 1]);
    double newWeight = value;
    graph.getWeights(from)[0] = graph.getWeights(from)[0] - graph.getWeights(from)[index] + newWeight;
    graph.getWeights(from)[index] = newWeight;
  }

  /**
   * This objective function implements the KL divergence between the expected and the actual distribution plus a regularization term.
   */
  public double getValue() {
    actualizePageRankValues();
    return getValue(prActual);
  }

  public double getValue(double[] alternativePr) {
    double negativeKLDivergence = 0.0d;
    for (int i = 0; i < prStar.length; ++i) {
      if (prStar[i] > 0 && alternativePr[i] > 0) {
        negativeKLDivergence -= prStar[i] * Math.log(prStar[i] / alternativePr[i]);
        // negativeCrossEntropy += prStar[i] * Math.log(alternativePr[i]);
      }
    }
    return negativeKLDivergence + calculateRegularization();
  }

  protected void addRegularizationGradient(double[] buffer) {
    if (regularizationWeight > 0.0d) {
      for (int n = 0, i = 0; n < graph.getNumOfNodes(); ++n) {
        double[] ws = graph.getWeights(n);
        for (int j = 1; j <= graph.getNumOfNeighbors(n); ++j, ++i) {
          buffer[i] -= regularizationWeight * ws[j];
        }
      }
    }
  }

  protected double calculateRegularization() {
    double regularization = 0.0d;
    for (int n = 0; n < graph.getNumOfNodes(); ++n) {
      double[] ws = graph.getWeights(n);
      double sqrdLength = 0.0d;
      for (int i = 1; i <= graph.getNumOfNeighbors(n); ++i) {
        sqrdLength += ws[i] * ws[i];
      }
      regularization += sqrdLength;
    }
    return -0.5d * regularizationWeight * regularization;
  }

  public OwnGraph getBaselineGraph() {
    OwnGraph baselineGraph = new OwnGraph(graph.getNumOfNodes());
    for (int i = 0; i < graph.getNumOfNodes(); ++i) {
      int[] neighbors = graph.getOutLinks(i);
      for (int j = 1; j <= neighbors[0]; ++j) {
        baselineGraph.addEdge(i, neighbors[j], prStar[neighbors[j]]);
      }
    }
    baselineGraph.normalizeWeights();
    return baselineGraph;
  }

  public double getBaselineValue() {
    return getBaselineValue(getBaselineGraph());
  }

  public double getBaselineValue(OwnGraph blg) {
    PRWeightLearner l = new SoftmaxPRWeightLearner(prStar, blg, this.teleportProbability);
    l.setRegularizationWeight(this.regularizationWeight);
    l.prActual = prc.calculatePageRank(blg, false);
    return l.getValue();
  }

  public void serializeWeights(String outFile) {
    Utils.serialize(outFile, graph.getWeights());
  }

  public void extensiveLog() {
    extensiveLog(Integer.MAX_VALUE, false);
  }

  public void extensiveLog(int maxToPrint, boolean relativize) {
    for (int n = 0; n < graph.getNumOfNodes(); ++n) {
      int[] neighs = graph.getOutLinks(n);
      double[] weights = graph.getWeights(n);
      double[] baselineWs = new double[neighs[0]];
      double sum = 0.0d;
      for (int i = 1; relativize && i <= neighs[0]; ++i) {
        sum += (baselineWs[i - 1] = getEtalonPRvalue(neighs[i]));
      }
      double[] weightsToRank = new double[neighs[0]];
      for (int i = 1; i <= neighs[0]; ++i) {
        weightsToRank[i - 1] = weights[i];
        if (relativize) {
          weightsToRank[i - 1] -= (baselineWs[i - 1] / sum);
        }
      }

      int[] order = Utils.stableSort(weightsToRank);
      double expected = weights[0] / neighs[0];
      log.format("%d\t%s\t%f\t%d\t%f\n", n, graph.getNodeLabel(n), expected, neighs[0], weights[0]);
      for (int i = 0; i < Math.min(order.length, maxToPrint); ++i) {
        int o = order[order.length - 1 - i];
        log.format("\t->%s\t%.9f\n", graph.getNodeLabel(neighs[o + 1]), weightsToRank[o]);
      }
      log.flush();
    }
  }
}
