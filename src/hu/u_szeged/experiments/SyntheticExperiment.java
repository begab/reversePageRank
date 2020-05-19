package hu.u_szeged.experiments;

import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.apache.commons.math3.random.JDKRandomGenerator;
import org.graphstream.algorithm.generator.BarabasiAlbertGenerator;
import org.graphstream.algorithm.generator.BaseGenerator;
import org.graphstream.algorithm.generator.DorogovtsevMendesGenerator;
import org.graphstream.algorithm.generator.RandomGenerator;
import org.graphstream.algorithm.generator.WattsStrogatzGenerator;
import org.graphstream.graph.Edge;
import org.graphstream.graph.Graph;
import org.graphstream.graph.implementations.SingleGraph;

import hu.u_szeged.graph.OwnGraph;
import hu.u_szeged.graph.OwnGraph.WeightingStrategy;
import hu.u_szeged.graph.PRWeightLearner;
import hu.u_szeged.graph.PageRankCalculator;
import hu.u_szeged.graph.SoftmaxPRWeightLearner;
import hu.u_szeged.utils.Utils;

public class SyntheticExperiment {

  public static Random RANDOM = new JDKRandomGenerator();
  public long seed;

  public SyntheticExperiment() {
    seed = 5;
    RANDOM.setSeed(seed);
  }

  public void incrementSeed() {
    setSeed(seed + 1);
  }

  public void setSeed(long s) {
    seed = s;
    RANDOM.setSeed(s);
  }

  /**
   * 
   * @param etalonSyntheticGraph
   * @param learner
   * @return The first and second elements are the initial and final (KL/cross entropy-based) objective values, while the third one is the squared sum
   *         of the differences of the individual edge weights between the etalon graph and the learned one.
   */
  private double[][] learnWeights(OwnGraph etalonSyntheticGraph, PRWeightLearner learner, int randInits) {
    long actualSeed = seed;
    learner.setRegularization(0.0d);
    learner.setLogFile(null);
    setSeed(actualSeed + 1);
    double[][] stats = learner.learnEdgeWeights(randInits, true, true);
    double[] initObjVals = new double[randInits + 1];
    double[] finalObjVals = new double[randInits + 1];
    double[] squaredDiffs = new double[randInits + 1];
    double[] absDiffs = new double[randInits + 1];
    // double[] bestParams = stats[(int) stats[0][randInits]];
    for (int k = 0; k <= randInits; ++k) {
      initObjVals[k] = -stats[0][2 * k]; // the objective value was multiplied by -1 previously so maximization can take place
      finalObjVals[k] = -stats[0][2 * k + 1];
      squaredDiffs[k] = getAverageSqrdDiff(etalonSyntheticGraph, stats[k + 1]);
      absDiffs[k] = getAverageAbsDiff(etalonSyntheticGraph, stats[k + 1]);
    }
    double[] times = stats[stats.length - 1];
    return new double[][] { initObjVals, finalObjVals, squaredDiffs, absDiffs, times, stats[1] };
  }

  private int determineOverlap(int[] indexA, int[] indexB) {
    int overlap = 0;
    for (int ii = 1, oi = 1; ii <= indexA[0] && oi <= indexB[0]; ++ii, ++oi) {
      if (indexA[ii] == indexB[oi]) {
        overlap++;
      } else if (indexA[ii] > indexB[oi]) {
        ii--;
      } else {
        oi--;
      }
    }
    return overlap;
  }

  private double[] determineJaccardBaselineSolution(OwnGraph etalonSyntheticGraph) {
    double[] unnormalizedEdgeWeights = new double[etalonSyntheticGraph.getNumOfEdges()];
    for (int n = 0, index = 0; n < etalonSyntheticGraph.getNumOfNodes(); ++n) {
      int[] neighs = etalonSyntheticGraph.getOutLinks(n);
      for (int i = 1; i <= neighs[0]; ++i) {
        int[] neighs2 = etalonSyntheticGraph.getOutLinks(neighs[i]);
        int overlap = determineOverlap(neighs, neighs2);
        unnormalizedEdgeWeights[index++] = overlap / (double) (neighs[0] + neighs2[0] - overlap);
      }
    }
    return unnormalizedEdgeWeights;
  }

  private double[] determineIndegreeBaselineSolution(OwnGraph etalonSyntheticGraph) {
    double[] unnormalizedEdgeWeights = new double[etalonSyntheticGraph.getNumOfEdges()];
    for (int n = 0, index = 0; n < etalonSyntheticGraph.getNumOfNodes(); ++n) {
      int[] neighs = etalonSyntheticGraph.getOutLinks(n);
      for (int i = 1; i <= neighs[0]; ++i) {
        unnormalizedEdgeWeights[index++] = etalonSyntheticGraph.getIndegree(neighs[i]);
      }
    }
    return unnormalizedEdgeWeights;
  }

  private double[][] determineBaselineScores(OwnGraph etalonSyntheticGraph) {
    return determineBaselineScores(etalonSyntheticGraph, null);
  }

  /***
   * Returns the means squared error and the mean average error based on unnormalized edge weights. <br />
   * If baselineEdgeWeights is set to null a uniform network weighting (i.e. 1/outdegree) is assumed.
   * 
   * @param etalonSyntheticGraph
   * @return Returns MSE and MAE scores in the first row of the output array and the baseline edge weights in the second row.
   */
  private double[][] determineBaselineScores(OwnGraph etalonSyntheticGraph, double[] baselineEdgeWeights) {
    double[] normalizedEdgeWeights = null;
    if (baselineEdgeWeights == null) {
      normalizedEdgeWeights = new double[etalonSyntheticGraph.getNumOfEdges()];
      for (int i = 0, j = 0; i < etalonSyntheticGraph.getNumOfNodes(); ++i) {
        int neighbors = etalonSyntheticGraph.getOutDegree(i);
        for (int e = 0; e < neighbors; ++e) {
          normalizedEdgeWeights[j++] = 1.0 / neighbors;
        }
      }
    } else {
      normalizedEdgeWeights = normalizeBaselineSolution(etalonSyntheticGraph, baselineEdgeWeights);
    }
    double mse = getAverageSqrdDiff(etalonSyntheticGraph, normalizedEdgeWeights);
    double mae = getAverageAbsDiff(etalonSyntheticGraph, normalizedEdgeWeights);
    return new double[][] { { mse, mae }, normalizedEdgeWeights };
  }

  private double[] normalizeBaselineSolution(OwnGraph etalonSyntheticGraph, double[] baselineEdgeWeights) {
    double[] params = new double[etalonSyntheticGraph.getNumOfEdges()];
    for (int n = 0, index = 0; n < etalonSyntheticGraph.getNumOfNodes(); ++n) {
      int[] neighs = etalonSyntheticGraph.getOutLinks(n);
      double[] baselinePredictions = new double[neighs[0]];
      double denominator = 0.0d;
      for (int i = 1; i <= neighs[0]; ++i) {
        baselinePredictions[i - 1] = baselineEdgeWeights[neighs[i]];// ;Math.exp(baselineEdgeWeights[neighs[i]]);
        denominator += baselinePredictions[i - 1];
      }
      for (double d : baselinePredictions) {
        params[index++] = denominator > 0 ? d / denominator : 1.0d / neighs[0];
      }
    }
    return params;
  }

  /**
   * 
   * Compares the relative proportion of cases for which top-k ranked edge weights match ground truth ordering.
   * 
   * @param etalonSyntheticGraph
   * @param params
   * @return
   */
  private double getPrecisionAtK(OwnGraph etalonSyntheticGraph, double[] params, int k) {
    double match = 0, denominator = 0;
    for (int i = 0, h = 0; i < etalonSyntheticGraph.getNumOfNodes(); ++i) {
      int neighbors = etalonSyntheticGraph.getOutDegree(i);
      Set<Integer> strongNeighbors = new HashSet<>();
      double max = 0.0d;
      double[] weights = etalonSyntheticGraph.getWeights(i);
      for (int n = 1; n <= neighbors; ++n) {
        if (weights[n] > max) {
          max = weights[n];
          strongNeighbors = new HashSet<Integer>();
        }
        if (weights[n] == max) {
          strongNeighbors.add(n - 1);
        }
      }
      double[] predictedWeights = Arrays.copyOfRange(params, h, h + neighbors);
      Map<Double, Integer> weightFreqs = new HashMap<>();
      for (double w : predictedWeights) {
        weightFreqs.put(w, weightFreqs.getOrDefault(w, 0) + 1);
      }
      int[] predictedWeightOrder = Utils.stableSort(predictedWeights);
      h += neighbors;
      denominator += Math.min(1, neighbors);
      for (int j = 0; j < k && j < neighbors; ++j) {
        if (strongNeighbors.contains(predictedWeightOrder[predictedWeightOrder.length - j - 1])) {
          match += 1.0d / weightFreqs.get(predictedWeights[predictedWeightOrder[predictedWeightOrder.length - j - 1]]);
        }
      }
    }
    return match / denominator;
  }

  private double getAverageSqrdDiff(OwnGraph etalonSyntheticGraph, double[] params) {
    return getAverageDiff(etalonSyntheticGraph, params, 2);
  }

  private double getAverageAbsDiff(OwnGraph etalonSyntheticGraph, double[] params) {
    return getAverageDiff(etalonSyntheticGraph, params, 1);
  }

  /**
   * 
   * @param etalonSyntheticGraph
   * @param params
   * @return
   */
  private double getAverageDiff(OwnGraph etalonSyntheticGraph, double[] params, int power) {
    double difference = 0.0d;
    for (int i = 0, paramId = 0; i < etalonSyntheticGraph.getNumOfNodes(); ++i) {
      int neighbors = etalonSyntheticGraph.getOutDegree(i);
      double[] weights = etalonSyntheticGraph.getWeights(i);
      for (int j = 1; j <= neighbors; ++j, ++paramId) {
        double weight = 1.0d / neighbors;
        if (params != null) {
          weight = params[paramId];
        }
        difference += Math.pow(Math.abs(weights[j] - weight), power);
      }
    }
    return difference / etalonSyntheticGraph.getNumOfEdges();
  }

  public static OwnGraph convertGraphStreamGraph(String generator, int numOfNodes, long seed) {
    BaseGenerator gen;
    int k = (int) (1.2 * Math.log(numOfNodes));
    if (k % 2 == 1) {
      k += 1;
    }
    // System.err.println(k);
    switch (generator) {
    case "ScaleFree":
      gen = new BarabasiAlbertGenerator(k);
      break;
    case "Random":
      gen = new RandomGenerator(k);
      break;
    case "DorogovtsevMendes":
      gen = new DorogovtsevMendesGenerator();
      break;
    case "WattsStrogatz":
      gen = new WattsStrogatzGenerator(numOfNodes, k, 0.05);
      break;
    default:
      gen = new BarabasiAlbertGenerator(k);
      System.err.format("WARNING: Note that the graph type provided in command line (%s) is not supported, a scale-free network will be generated instead.\n",
          generator);
    }

    Graph bag = new SingleGraph("");
    gen.setRandomSeed(seed);
    gen.addSink(bag);
    gen.begin();
    while (bag.getNodeCount() <= numOfNodes - (generator.equals("WattsStrogatz") ? 0 : 1) && gen.nextEvents())
      ;
    gen.end();
    // ConnectedComponents cc = new ConnectedComponents();
    // cc.init(bag);
    // System.err.println(cc.getConnectedComponentsCount());

    // bag.display();
    // try {
    // Thread.sleep(20000);
    // } catch (InterruptedException e1) {
    // // TODO Auto-generated catch block
    // e1.printStackTrace();
    // }
    // org.graphstream.algorithm.Toolkit.

    OwnGraph og = new OwnGraph(bag.getNodeCount());
    for (int n = 0; n < bag.getNodeCount(); ++n) {
      og.setNodeLabel(Integer.toString(n), n);
    }
    for (int i = 0; i < bag.getEdgeCount(); ++i) {
      Edge e = bag.getEdge(i);
      og.addBidirectionalEdge(e.getSourceNode().getIndex(), e.getTargetNode().getIndex());
    }
    return og;
  }

  public static void main(String[] args) {
    PageRankCalculator pc = new PageRankCalculator(0.1);
    SyntheticExperiment se = new SyntheticExperiment();
    WeightingStrategy degreeBasedWeights = WeightingStrategy.RAND;
    int numOfExperiments = 100, numOfRestarts = 50, numOfNodes = 5_000;

    args = new String[] { "DorogovtsevMendes" }; // DorogovtsevMendes WattsStrogatz Random ScaleFree
    System.err.format("Edge weights are drawn in a(n) %s manner.\n", degreeBasedWeights.toString());
    for (String gg : args) {
      try (PrintWriter logOut = new PrintWriter(String.format("%s_%s_new.txt", degreeBasedWeights.toString(), gg));
          PrintWriter accAtKWriter = new PrintWriter(String.format("%s_%s_new_accAtK.txt", degreeBasedWeights.toString(), gg));
          PrintWriter baselinesWriter = new PrintWriter(String.format("%s_%s_new_baselines.txt", degreeBasedWeights.toString(), gg))) {
        for (int e = 0; e < numOfExperiments; ++e) {
          se.incrementSeed();
          OwnGraph testGraph = convertGraphStreamGraph(gg, numOfNodes, e);
          testGraph.initWeights(degreeBasedWeights);
          System.err.println(Arrays.toString(testGraph.getOutLinks(0)));
          OwnGraph uninformedGraph = testGraph.copyGraph(WeightingStrategy.UNIFORM);
          System.err.println(Arrays.toString(Arrays.copyOf(testGraph.getWeights(), 11)));
          double[] etalon = pc.calculatePageRank(testGraph, false);
          PRWeightLearner learner = new SoftmaxPRWeightLearner(etalon, uninformedGraph.clone());
          // new ExactProjectedPRWeightLearner(etalon, uninformedGraph.clone()) };
          double[][] results = se.learnWeights(testGraph, learner, numOfRestarts);
          baselinesWriter.format("%d\tInversePR\t%.9f\t%.9f\t%.9f\n", e, results[1][0], results[2][0], results[3][0]);
          for (int k = 1; k < 6; ++k) {
            accAtKWriter.format("%d\tInversePR\t%d\t%.9f\n", e, k, se.getPrecisionAtK(testGraph, results[results.length - 1], k));
          }

          // int bestIndividualModelIndex = (int) -(results[0][numOfRestarts] + 1); // the index of the best experiment is stored as -(index+1)

          double finalObj = results[1][results[1].length - 1], finalAvgMSE = results[2][results[2].length - 1], finalAvgMAE = results[3][results[3].length - 1];
          for (int i = 0; i <= numOfRestarts; ++i) {
            logOut.format("%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t", results[0][i], results[1][i], results[2][i], results[3][i], results[4][i]);
          }
          logOut.format("%d\t%.9f\t%.9f\t%.9f\t%d\n", testGraph.getNumOfEdges(), finalObj, finalAvgMSE, finalAvgMAE, e + 1);
          logOut.flush();

          double[][] uniformBaseline = se.determineBaselineScores(testGraph);
          double objVal = -learner.getBaselineValue(uniformBaseline[1]);
          baselinesWriter.format("%d\tUniform\t%.9f\t%.9f\t%.9f\n", e, objVal, uniformBaseline[0][0], uniformBaseline[0][1]);
          for (int k = 1; k < 6; ++k) {
            accAtKWriter.format("%d\tUniform\t%d\t%.9f\n", e, k, se.getPrecisionAtK(testGraph, uniformBaseline[1], k));
          }
          double[][] indegreeBaseline = se.determineBaselineScores(testGraph, se.determineIndegreeBaselineSolution(testGraph));
          objVal = -learner.getBaselineValue(indegreeBaseline[1]);
          baselinesWriter.format("%d\tIndegree\t%.9f\t%.9f\t%.9f\n", e, objVal, indegreeBaseline[0][0], indegreeBaseline[0][1]);
          for (int k = 1; k < 6; ++k) {
            accAtKWriter.format("%d\tIndegree\t%d\t%.9f\n", e, k, se.getPrecisionAtK(testGraph, indegreeBaseline[1], k));
          }
          double[][] jaccardBaseline = se.determineBaselineScores(testGraph, se.determineJaccardBaselineSolution(testGraph));
          objVal = -learner.getBaselineValue(jaccardBaseline[1]);
          baselinesWriter.format("%d\tJaccard\t%.9f\t%.9f\t%.9f\n", e, objVal, jaccardBaseline[0][0], jaccardBaseline[0][1]);
          for (int k = 1; k < 6; ++k) {
            accAtKWriter.format("%d\tJaccard\t%d\t%.9f\n", e, k, se.getPrecisionAtK(testGraph, jaccardBaseline[1], k));
          }
          double[][] etalonProportionalBaseline = se.determineBaselineScores(testGraph, learner.getEtalonPRvalues());
          objVal = -learner.getBaselineValue(etalonProportionalBaseline[1]);
          baselinesWriter.format("%d\tPopularity\t%.9f\t%.9f\t%.9f\n", e, objVal, etalonProportionalBaseline[0][0], etalonProportionalBaseline[0][1]);
          for (int k = 1; k < 6; ++k) {
            accAtKWriter.format("%d\tPopularity\t%d\t%.9f\n", e, k, se.getPrecisionAtK(testGraph, etalonProportionalBaseline[1], k));
          }
          double[][] unweightedPagerankBaseline = se.determineBaselineScores(testGraph, learner.getInitialPRvalues());
          objVal = -learner.getBaselineValue(unweightedPagerankBaseline[1]);
          baselinesWriter.format("%d\tPageRank\t%.9f\t%.9f\t%.9f\n", e, objVal, unweightedPagerankBaseline[0][0], unweightedPagerankBaseline[0][1]);
          for (int k = 1; k < 6; ++k) {
            accAtKWriter.format("%d\tPageRank\t%d\t%.9f\n", e, k, se.getPrecisionAtK(testGraph, unweightedPagerankBaseline[1], k));
          }
        }
      } catch (IOException e) {
        e.printStackTrace();
      }
    }
  }
}