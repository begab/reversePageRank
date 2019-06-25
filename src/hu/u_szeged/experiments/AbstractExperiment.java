package hu.u_szeged.experiments;

import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;
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

  private boolean verbose;
  private String fileA, fileB;

  protected abstract void readNodes(String inFile);

  protected abstract void readEdges(String infile);

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
    learnWeights(1, false, false);
  }

  protected void learnWeights(int initializations, boolean useAveragedModel) {
    learnWeights(initializations, useAveragedModel, false);
  }

  protected void learnWeights(int initializations, boolean useAveragedModel, boolean relativizeWeights) {
    learnWeights(initializations, useAveragedModel, relativizeWeights, "");
  }

  protected void learnWeights(int initializations, boolean useAveragedModel, boolean relativizeWeights, String modelNamePrefix) {
    String className = this.getClass().getSimpleName();
    String outFile = String.format("%s%s_%s.txt", modelNamePrefix, className, useAveragedModel ? "_avg" : "");
    double[] objectiveLog = learner.learnEdgeWeights(initializations, false, useAveragedModel)[0];
    learner.setLogFile(outFile);
    learner.extensiveLog(15, relativizeWeights); // now the final verbose model gets printed to the location defined by the outFile variable
    for (int i = 0; this.verbose && i < objectiveLog.length; i += 2) {
      System.err.format("%d\t%f\t%f\n", (i / 2) + 1, objectiveLog[i], objectiveLog[i + 1]);
    }
  }
}