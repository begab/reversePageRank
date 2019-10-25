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

  public static enum RegularizationType {
    NONE, ENTROPY, ORACLE
  };

  protected double[] prStar;
  protected double[] prInitial; // let's keep the original (i.e. unweighted) pagerank scores as well
  protected double[] prActual;

  protected RegularizationType regularization;
  protected double regularizationWeight;
  protected double teleportProbability;
  protected OwnGraph graph;
  protected PageRankCalculator prc;

  protected abstract void actualizePageRankValues();

  public PRWeightLearner(double[] prs, OwnGraph g) {
    this(prs, g, PageRankCalculator.DEFAULT_TELEPORT, null);
  }

  public PRWeightLearner(double[] prs, OwnGraph g, double teleportProb) {
    this(prs, g, teleportProb, null);
  }

  public PRWeightLearner(double[] prs, OwnGraph g, double teleportProb, Set<Integer> favoredNodeIds) {
    log = System.err;
    prStar = prs;
    graph = g;
    teleportProbability = teleportProb;
    prc = new PageRankCalculator(teleportProbability);
    prc.setFavoredNodes(favoredNodeIds);
    prInitial = prc.calculatePageRank(graph, true);
  }

  public double[] getActualPRvalues() {
    if (prActual == null) {
      prActual = prc.calculatePageRank(graph, true);
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

  public double[] getInitialPRvalues() {
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

  public double[] learnEdgeWeights(int numOfInitializations) {
    return learnEdgeWeights(numOfInitializations, false, false)[0];
  }

  public double[] learnEdgeWeights(int numOfInitializations, boolean useAveragedModel) {
    return learnEdgeWeights(numOfInitializations, false, useAveragedModel)[0];
  }

  public double[][] learnEdgeWeights(int numOfInitializations, boolean returnAllParametersLearned, boolean averageModels) {
    double bestFinalObjVal = Double.NEGATIVE_INFINITY;
    double[] bestParameters = new double[graph.getNumOfEdges()];
    double[][] toReturn = new double[(returnAllParametersLearned ? numOfInitializations : 0) + 3][];
    toReturn[0] = new double[2 * (numOfInitializations + 1)]; // contains the objective values and the index of the best initialization
    toReturn[toReturn.length - 2] = new double[graph.getNumOfEdges()]; // the last row contains an averaged model over the different initializations
    toReturn[toReturn.length - 1] = new double[numOfInitializations + 1]; // stores the per initialization and the aggregated run times
    for (int i = 0; i < numOfInitializations; ++i) {
      if (i == 0) {
        graph.initWeights(WeightingStrategy.UNIFORM);
      } else {
        graph.initWeights(WeightingStrategy.RAND);
      }
      if (!returnAllParametersLearned && !averageModels && i != numOfInitializations - 1) {
        // this happens when a single model is needed to be trained
        continue;
      }
      long time = System.currentTimeMillis();
      double[] objVals = learnEdgeWeights();
      toReturn[toReturn.length - 1][i] = (System.currentTimeMillis() - time) / 1000.0d;
      toReturn[toReturn.length - 1][numOfInitializations] += toReturn[toReturn.length - 1][i];
      graph.softmaxNormalizeWeights();
      if (extensiveSerialization) {
        serializeWeights(String.format("%s_model%d.ser", serializationPrefix, i));
      }
      toReturn[0][2 * i] = objVals[0]; // this is the initial objective value
      toReturn[0][2 * i + 1] = objVals[1]; // this is the objective value obtained after the optimization

      double[] actualParameters = graph.getWeights();
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
    double[] prFinal = prc.calculatePageRank(graph, false); // in this case weights need not be normalized (as it had already taken place)
    toReturn[0][toReturn[0].length - 1] = getValue(prFinal, false); // store the objective value of the averaged model
    return toReturn;
  }

  private double[] learnEdgeWeights() {
    double initObjVal = getValue();
    Optimizer optimizer = new OwnLimitedMemoryBFGS(this);
    try {
      optimizer.optimize();
    } catch (OptimizationException | IllegalArgumentException e) {
      System.err.println(e.getLocalizedMessage()); // This condition does not necessarily mean that the optimizer has failed (but it might has)
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

  public double getRegularizationWeight() {
    return regularizationWeight;
  }

  public void setRegularization(double r) {
    setRegularization(r, RegularizationType.ORACLE);
  }

  public void setRegularization(double r, RegularizationType rt) {
    regularizationWeight = r;
    regularization = rt;
    if (regularization == RegularizationType.NONE) {
      regularizationWeight = 0.0d;
    }
  }

  public String regularizationToString() {
    return String.format("_%s_%.8f", regularization, regularizationWeight);
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
    try {
      throw new Exception("We shall not rely on this method.");
    } catch (Exception e) {
      e.printStackTrace();
    }
  }

  public double getParameter(int index) {
    try {
      throw new Exception("We shall not rely on elementwise getting of a parameter.");
    } catch (Exception e) {
      e.printStackTrace();
    }
    return -1;
  }

  public void setParameter(int index, double value) {
    try {
      throw new Exception("We shall not rely on elementwise setting of parameters.");
    } catch (Exception e) {
      e.printStackTrace();
    }
  }

  /**
   * This objective function implements the KL divergence between the expected and the actual distribution plus a regularization term.
   */
  public double getValue() {
    actualizePageRankValues();
    return getValue(prActual, true);
  }

  /**
   * Returns hypothesized edge weights for a node with given certain neighborhood assuming edge weights are proportional to oracle scores.
   * 
   * @param neighs
   * @return
   */
  protected double[] calculateBaselineWeights(int[] neighs) {
    double sum = 0.0d;
    double[] oracleBasedWeights = new double[neighs[0] + 1];
    for (int i = 1; i <= neighs[0]; ++i) {
      sum += (oracleBasedWeights[i] = prStar[neighs[i]]);
    }
    for (int i = 1; i <= neighs[0]; ++i) {
      oracleBasedWeights[i] /= sum;
    }
    return oracleBasedWeights;
  }

  public double getValue(double[] alternativePr, boolean needsSoftmaxNormalization) {
    double negativeKLDivergence = 0.0d, regularizationScore = 0;
    for (int i = 0; i < prStar.length; ++i) {
      if (prStar[i] > 0 && alternativePr[i] > 0) {
        negativeKLDivergence -= prStar[i] * Math.log(prStar[i] / alternativePr[i]);
        // negativeCrossEntropy += prStar[i] * Math.log(alternativePr[i]);
        if (regularization == RegularizationType.ENTROPY) {
          regularizationScore -= calculateNegativeEntropy(i);
        }
      }
    }
    if (regularization == RegularizationType.ORACLE) {
      regularizationScore = calculateRegularization(needsSoftmaxNormalization);
    }
    return negativeKLDivergence - regularizationWeight * regularizationScore;
  }

  protected double calculateRegularization(boolean needsSoftmaxNormalization) {
    double r = 0;
    if (needsSoftmaxNormalization) {
      graph.softmaxNormalizeWeights();
    }
    for (int i = 0; i < graph.getNumOfNodes(); ++i) {
      int[] neighbors = graph.getOutLinks(i);
      double[] expectedWeights = calculateBaselineWeights(neighbors);
      double[] currentEdgeWeights = graph.getWeights(i);
      for (int n = 1; n <= neighbors[0]; ++n) {
        double diff = (expectedWeights[n] - currentEdgeWeights[n]);
        r += diff * diff;
      }
    }
    if (needsSoftmaxNormalization) {
      graph.softmaxDenormalizeWeights();
    }
    return .5 * r;
  }

  private double calculateNegativeEntropy(int node) {
    double entropy = 0.0d;
    if (regularizationWeight > 0) {
      int degree = graph.getOutDegree(node);
      double[] weights = Arrays.copyOf(graph.getWeights(node), degree + 1);
      Utils.softmaxNormalize(weights, degree);
      for (int n = 1; n < weights.length; ++n) {
        if (weights[n] > 0) {
          entropy += weights[n] * Math.log(weights[n]);
        }
      }
    }
    return entropy;
  }

  protected void addRegularizationGradient(double[] buffer) {
    try {
      throw new Exception("Method for calculating the gradient of the entropy based regularization is uninplemented.");
    } catch (Exception e) {
      e.printStackTrace();
    }
  }

  private OwnGraph getBaselineGraph() {
    OwnGraph baselineGraph = new OwnGraph(graph.getNumOfNodes());
    for (int i = 0; i < graph.getNumOfNodes(); ++i) {
      int[] neighbors = graph.getOutLinks(i);
      double[] baselineWeights = calculateBaselineWeights(neighbors);
      for (int j = 1; j <= neighbors[0]; ++j) {
        baselineGraph.addEdge(i, neighbors[j], baselineWeights[j]);
      }
    }
    return baselineGraph;
  }

  public double getBaselineValue() {
    OwnGraph baselineGraph = getBaselineGraph();
    double[] baselinePageranks = prc.calculatePageRank(baselineGraph, false);
    return getValue(baselinePageranks, false);
  }

  public double getBaselineValue(double[] baselineEdgeWeights) {
    OwnGraph baselineGraph = this.graph.clone();
    baselineGraph.setWeights(baselineEdgeWeights);
    double[] baselinePageranks = prc.calculatePageRank(baselineGraph, false);
    return getValue(baselinePageranks, false);
  }

  // public double getBaselineValue(OwnGraph blg) {
  // return getValue(prc.calculatePageRank(blg, false), false);
  // }

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
      double[] baselineWs = calculateBaselineWeights(neighs);
      double[] weightsToRank = new double[neighs[0]];
      for (int i = 1; i <= neighs[0]; ++i) {
        weightsToRank[i - 1] = weights[i] - (relativize ? baselineWs[i] : 0.0d);
      }

      int[] order = Utils.stableSort(weightsToRank);
      double expected = neighs[0] == 0 ? 0 : 1.0 / neighs[0];
      log.format("%d\t%s\t%f\t%d\n", n, graph.getNodeLabel(n), expected, neighs[0]);
      for (int i = 0; i < Math.min(order.length, maxToPrint); ++i) {
        int o = order[order.length - 1 - i];
        log.format("\t->%s\t%.6f\t%.6f\n", graph.getNodeLabel(neighs[o + 1]), weightsToRank[o], baselineWs[o + 1]);
      }
      log.flush();
    }
  }

  public static void performTest(int[][] testCase, boolean undirectedGraph) {
    performTest(testCase, undirectedGraph, 0.0d, 0.01d, RegularizationType.NONE);
  }

  public static void performTest(int[][] testCase, boolean directedGraph, double regularization, double teleportProb, RegularizationType rt) {
    int numOfNodes = testCase[0].length, sumEtalonRank = 0;
    double[] prStar = new double[numOfNodes];

    for (int i = 0; i < numOfNodes; ++i) {
      prStar[i] = testCase[0][i];
      sumEtalonRank += testCase[0][i];
    }
    for (int i = 0; i < numOfNodes; ++i) {
      prStar[i] /= (double) sumEtalonRank;
    }

    OwnGraph g = new OwnGraph(testCase[1], testCase[2], directedGraph, numOfNodes);
    g.initWeights(WeightingStrategy.UNIFORM);
    PRWeightLearner learner = new SoftmaxPRWeightLearner(prStar, g, teleportProb);
    learner.setRegularization(regularization, rt);
    learner.learnEdgeWeights(1, true);
    learner.extensiveLog();

    String weightMtx = g.returnWeightMatrix();
    System.err.println(weightMtx);
    // g.saveToDot(learner.getActualPRvalues(), prStar, "test1_wl.dot");
  }

  public static void main(String[] args) {
    int[][][] testCases = { { { 25, 25, 20, 10, 10, 10 }, { 0, 0, 1, 2, 2, 3, 4 }, { 1, 2, 2, 3, 5, 4, 5 } } }; // , // undirected
    // { { 30, 25, 5, 5, 15, 20 }, { 0, 0, 1, 2, 3, 3, 4 }, { 1, 5, 2, 3, 4, 5, 5 } }, // undirected
    // { { 30, 20, 15, 35 }, { 0, 0, 1, 2, 2, 3 }, { 1, 3, 2, 0, 3, 1 } } };// if directed no 'good' solution exists
    RegularizationType rt = RegularizationType.ORACLE;
    for (int i = 0; i < testCases.length; ++i) {
      // performTest(testCases[i], false, 0.0d, 0.01d, false);
      // System.err.println("~~~~~~~~~~");
      for (double reg : new double[] { 0.0 }) {
        performTest(testCases[i], false, reg, 0.1d, rt);
        System.err.println("==========");
      }
    }
  }
}