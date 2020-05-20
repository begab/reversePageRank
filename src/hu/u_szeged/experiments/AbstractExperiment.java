package hu.u_szeged.experiments;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;

import hu.u_szeged.graph.OwnGraph;
import hu.u_szeged.graph.PRWeightLearner;
import hu.u_szeged.graph.PRWeightLearner.RegularizationType;
import hu.u_szeged.graph.PageRankCalculator;
import hu.u_szeged.graph.SoftmaxPRWeightLearner;
import hu.u_szeged.utils.Utils;

public abstract class AbstractExperiment {

  protected OwnGraph g;
  protected double[] etalonDistr;
  protected Map<String, Integer> labelsToIds;
  protected Map<Integer, String> idsToLabels;
  protected boolean addSelfLoops;
  protected PRWeightLearner learner;

  private double[] choicerankParams;
  private boolean verbose;
  private String fileA, fileB;

  protected abstract void readNodes(String inFile);

  protected abstract void readEdges(String inFile);

  public AbstractExperiment() {
    labelsToIds = new HashMap<>();
    idsToLabels = new HashMap<>();
  }

  public AbstractExperiment(String file) {
    this(file, file, false);
  }

  public AbstractExperiment(String file, boolean selfLoops) {
    this(file, file, selfLoops);
  }

  public AbstractExperiment(String nodeFile, String edgeFile) {
    this(nodeFile, edgeFile, false);
  }

  public AbstractExperiment(String nodeFile, String edgeFile, boolean selfLoops) {
    addSelfLoops = selfLoops;
    fileA = nodeFile;
    fileB = edgeFile;
    labelsToIds = new HashMap<>();
    idsToLabels = new HashMap<>();
    SyntheticExperiment.RANDOM.setSeed(10l);
  }

  protected void init() {
    init(PageRankCalculator.DEFAULT_TELEPORT);
  }

  protected void init(double teleportProb) {
    readNodes(fileA);
    if (labelsToIds.size() > 0 && idsToLabels.size() == 0) {
      for (Entry<String, Integer> e : labelsToIds.entrySet()) {
        idsToLabels.put(e.getValue(), e.getKey());
      }
    }
    g = new OwnGraph(labelsToIds.size());
    choicerankParams = new double[g.getNumOfNodes()];
    readEdges(fileB);
    setGraphLabels();
    determineEtalon();
    learner = new SoftmaxPRWeightLearner(etalonDistr, g, teleportProb);
  }

  public void setVerbose(boolean b) {
    this.verbose = b;
  }

  protected void setEdges(Map<Integer, Set<Integer>> links) {
    for (Entry<Integer, Set<Integer>> e : links.entrySet()) {
      for (Integer to : e.getValue()) {
        g.addEdge(e.getKey(), to);
      }
    }
  }

  protected void setGraphLabels() {
    for (Entry<Integer, String> i2A : idsToLabels.entrySet()) {
      g.setNodeLabel(i2A.getValue(), i2A.getKey());
      if (addSelfLoops) {
        // self loops can be useful sometime, e.g. for a co-authorship
        // network, it can mean that an author is a trivial co-author of
        // him/herself.
        // the weight for this edge can be interpreted in the end in a
        // special way, i.e. how 'grateful' one should be to him/herself
        // for being cited.
        g.addEdge(i2A.getKey(), i2A.getKey());
      }
    }
    System.err.format("There are %d nodes and %d edges in the graph.\n", g.getNumOfNodes(), g.getNumOfEdges());
  }

  public PRWeightLearner getPRWeightLearner() {
    return learner;
  }

  public void determineEtalon() {
    if (etalonDistr == null) {
      System.err.println("WARNING: The default oracle values for the nodes are to be determined (which might not be a problem if it is intentional).");
      int totalEdges = 0;
      etalonDistr = new double[g.getNumOfNodes()];
      for (int i = 0; i < g.getNumOfNodes(); ++i) {
        int[] neighbors = g.getOutLinks(i);
        totalEdges += neighbors[0];
        for (int j = 1; j <= neighbors[0]; ++j) {
          etalonDistr[neighbors[j]]++;
        }
      }
      for (int i = 0; i < g.getNumOfNodes(); ++i) {
        etalonDistr[i] /= totalEdges;
      }
    }
  }

  protected void printMostProminentNodes() {
    printMostProminentNodes(Integer.MAX_VALUE);
  }

  protected void printMostProminentNodes(int t) {
    int[] sort = Utils.stableSort(etalonDistr);
    for (int i = 0; i < Math.min(t, sort.length); ++i) {
      int n = sort[sort.length - i - 1];
      System.err.format("%d\t%s\t%f\n", i + 1, g.getNodeLabel(n), etalonDistr[n]);
    }
  }

  public void setRegularizationWeight(double r) {
    learner.setRegularization(r, RegularizationType.ORACLE);
  }

  public void setRegularizationWeight(double r, RegularizationType rt) {
    learner.setRegularization(r, rt);
  }

  protected void learnWeights() {
    learnWeights(1, false);
  }

  protected void learnWeights(int initializations, boolean useAveragedModel) {
    learnWeights(initializations, useAveragedModel, "");
  }

  protected void learnWeights(int initializations, boolean useAveragedModel, String modelNamePrefix) {
    String className = this.getClass().getSimpleName();
    double[] objectiveLog = learner.learnEdgeWeights(initializations, false, useAveragedModel)[0];
    if (this.verbose) {// now the final verbose model gets printed to the location defined by the outFile variable
      String outFile = String.format("%s%s_%s.txt", modelNamePrefix, className, useAveragedModel ? "_avg" : "");
      learner.setLogFile(outFile);
      learner.extensiveLog(15);
      for (int i = 0; i < objectiveLog.length; i += 2) {
        System.err.format("%d\t%f\t%f\n", (i / 2) + 1, objectiveLog[i], objectiveLog[i + 1]);
      }
    }
  }

  private double[] getPredictedWeights(int nodeId, int[] ns, String mode, Random r) {
    double[] ws = new double[ns[0] + 1];
    if (mode.equals("jaccard")) {
      for (int i = 1; i <= ns[0]; ++i) {
        int[] neighbors = g.getOutLinks(ns[i]);
        ws[i] = Utils.determineOverlap(ns, neighbors);
        ws[i] /= (double) (ns[0] + neighbors[0] - ws[i]);
      }
    } else if (mode.equals("prlearn")) {
      ws = g.getWeights(nodeId);
    } else if (mode.equals("indegree")) {
      for (int i = 1; i <= ns[0]; ++i) {
        ws[i] = g.getIndegree(ns[i]);
      }
    } else if (mode.equals("uniform")) {
      for (int i = 1; i <= ns[0]; ++i) {
        ws[i] = r.nextDouble();
      }
    } else if (mode.equals("popularity")) {
      for (int i = 1; i <= ns[0]; ++i) {
        ws[i] = etalonDistr[ns[i]];
      }
    } else if (mode.equals("pagerank")) {
      for (int i = 1; i <= ns[0]; ++i) {
        ws[i] = learner.getInitialPRvalue(ns[i]);
      }
    } else if (mode.contains("choicerank")) {
      for (int i = 1; i <= ns[0]; ++i) {
        ws[i] = choicerankParams[ns[i]];
      }
    }
    double predictedNormalizer = 0;
    double[] predictedDistr = new double[ns[0] + 1];
    for (int i = 1; i <= ns[0]; ++i) {
      predictedNormalizer += ws[i];
    }
    for (int i = 1; predictedNormalizer > 0 && i <= ns[0]; ++i) {
      predictedDistr[i] = ws[i] / predictedNormalizer;
    }
    return predictedDistr;
  }

  protected double[] evaluateNode(int nodeId, Map<Integer, Double> etalonTransitions, String mode, Random r) {
    int[] ns = g.getOutLinks(nodeId);
    double[] predictedDistr = getPredictedWeights(nodeId, ns, mode, r);
    double[] etalonDistr = new double[ns[0] + 1];
    double max = 0, etalonNormalizer = 0;

    for (Double v : etalonTransitions.values()) {
      etalonNormalizer += v;
    }

    int argMaxNeighbor = -1, argmaxRank = -1;
    for (int i = 1; i <= ns[0]; ++i) {
      etalonDistr[i] = etalonTransitions.get(ns[i]) / etalonNormalizer;
      if (etalonDistr[i] > max) {
        max = etalonDistr[i];
        argMaxNeighbor = ns[i];
      }
    }

    int[] sort = Utils.stableSort(predictedDistr);
    Map<Integer, double[]> sortedPredictions = new HashMap<>();
    for (int i = sort.length - 1, rank = 0; i >= 0; --i) {
      if (sort[i] != 0) {
        sortedPredictions.put(ns[sort[i]], new double[] { ++rank, predictedDistr[sort[i]] });
        if (ns[sort[i]] == argMaxNeighbor) {
          argmaxRank = rank;
        }
      }
    }

    sort = Utils.stableSort(etalonDistr);
    Map<Integer, double[]> sortedEtalons = new HashMap<>();
    for (int i = sort.length - 1, rank = 0; i >= 0; --i) {
      if (sort[i] != 0) {
        sortedEtalons.put(ns[sort[i]], new double[] { ++rank, etalonDistr[sort[i]] });
      }
    }

    double kl = 0, rmse = 0, rankDisplacement = 0;
    for (Entry<Integer, double[]> etalon : sortedEtalons.entrySet()) {
      double[] preds = sortedPredictions.get(etalon.getKey());
      double etalonProb = etalon.getValue()[1];
      rankDisplacement += Math.abs(etalon.getValue()[0] - preds[0]);
      if (etalonProb > 0) {
        kl += etalonProb * Math.log(etalonProb / preds[1]);
        rmse += Math.pow(etalonProb - preds[1], 2.0);
      }
    }
    rmse = Math.sqrt(rmse / ns[0]);
    rankDisplacement /= Math.pow(ns[0], 2.0);
    double reciprocalRank = 1.0d / argmaxRank;
    return new double[] { kl, rmse, argmaxRank == 1 ? 1 : 0, reciprocalRank, rankDisplacement };
  }

  protected void readChoiceRankParams(String fileName) {
    try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(fileName)))) {
      String line;
      int i = 0;
      while ((line = br.readLine()) != null) {
        if (i < choicerankParams.length) {
          choicerankParams[i++] = Math.exp(Double.parseDouble(line));
        }
      }
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

}