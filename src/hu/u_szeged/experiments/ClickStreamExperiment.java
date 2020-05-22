package hu.u_szeged.experiments;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.zip.GZIPInputStream;

import hu.u_szeged.graph.PRWeightLearner;

public class ClickStreamExperiment extends AbstractExperiment {
  /**
   * This is a modified smaller-scale version of the Wikipedia experiment <br />
   * that relies on the Wikipedia articles included in the ClickStream dataset <br />
   * (as opposed to the entire Wikipedia) as it was done in the ChoiceRank paper.
   */

  private Map<Integer, Integer> outLinks;
  private Map<Integer, Map<Integer, Double>> etalonTransitions;

  public ClickStreamExperiment(String f, double teleport) {
    super(f);
    init(teleport);
    learner.setRegularization(0, PRWeightLearner.RegularizationType.ORACLE);
  }

  @Override
  protected void readNodes(String inFile) {
    etalonTransitions = new HashMap<>();
    try (BufferedReader br = new BufferedReader(new InputStreamReader(new GZIPInputStream(new FileInputStream(inFile)), "UTF-8"))) {
      String line;
      br.readLine(); // skip the 1st line being the header
      int lineCounter = 0;
      while ((line = br.readLine()) != null) {
        ++lineCounter;
        String[] parts = line.split("\t");
        if (parts[2].trim().equals("link")) {
          if (lineCounter % 1500000 == 0) {
            System.err.println(lineCounter + " " + line);
            // break;
          }
          labelsToIds.putIfAbsent(parts[0], labelsToIds.size());
          labelsToIds.putIfAbsent(parts[1], labelsToIds.size());
          etalonTransitions.putIfAbsent(labelsToIds.get(parts[0]), new HashMap<>());
          etalonTransitions.get(labelsToIds.get(parts[0])).put(labelsToIds.get(parts[1]), Double.parseDouble(parts[3]));
        }
      }
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  @Override
  protected void readEdges(String inFile) {
    outLinks = new HashMap<>();
    for (Entry<Integer, Map<Integer, Double>> e : etalonTransitions.entrySet()) {
      Integer fromNode = e.getKey();
      for (Entry<Integer, Double> neighbor : e.getValue().entrySet()) {
        outLinks.put(fromNode, (int) (outLinks.getOrDefault(fromNode, 0) + neighbor.getValue()));
        g.addEdge(e.getKey(), neighbor.getKey(), neighbor.getValue());
      }
    }
  }

  public void determineEtalon() {
    etalonDistr = new double[g.getNumOfNodes()];
    double totalWeights = 0.0;
    for (int i = 0; i < g.getNumOfNodes(); ++i) {
      int[] neighbors = g.getOutLinks(i);
      double[] weights = g.getWeights(i);
      for (int j = 1; j <= neighbors[0]; ++j) {
        totalWeights += weights[j];
        etalonDistr[neighbors[j]] += weights[j];
      }
    }
    System.err.println(totalWeights);
    for (int i = 0; i < g.getNumOfNodes(); ++i) {
      etalonDistr[i] /= totalWeights;
    }
  }

  public static void main(String[] args) {
    try (PrintWriter out = new PrintWriter("clickstream.results")) {
      String[] evalMetrics = new String[] { "KL", "RMSE", "accuracy", "MRR", "displacement" };
      for (String m : evalMetrics) {
        out.write(m + "\t");
      }
      out.write("N\tmode\tteleport\tnum_models\tnodeID\n");
      double teleProb = 0.01;
      int ri = 1;
      ClickStreamExperiment ce = new ClickStreamExperiment("./data/2016_03_en_clickstream.tsv.gz", teleProb);
      ce.setVerbose(false);
      ce.learnWeights(ri, false);
      ce.readChoiceRankParams("./data/clickstream.params");

      Random r = new Random(1);

      System.err.println("==========" + teleProb + " " + ri);
      for (String mode : new String[] { "uniform", "indegree", "jaccard", "popularity", "pagerank", "prlearn", "choicerank" }) {
        double[] evals = new double[5];
        int counter = 0;
        for (int n = 0; n < ce.g.getNumOfNodes(); ++n) {
          int[] ns = ce.g.getOutLinks(n);
          if (ns[0] > 1) {
            counter++;
            if (mode.equals("prlearn") || teleProb == 0.01) {
              double[] nodeEvals = ce.evaluateNode(n, ce.etalonTransitions.get(n), mode, r);
              for (int e = 0; e < evals.length; ++e) {
                out.format("%.4f\t", nodeEvals[e]);
                evals[e] += nodeEvals[e];
              }
              out.format("%d\t%s\t%.2f\t%d\t%d\n", ns[0], mode, teleProb, 1, n);
            }
          }
        }
        for (int e = 0; e < evals.length; ++e) {
          evals[e] /= counter;
        }
        if (mode.equals("prlearn") || teleProb == 0.01) {
          System.err.format("%s\t%.2f\t%s\t%d\t%d\n", mode, teleProb, Arrays.toString(evals), counter, ce.g.getNumOfNodes());
        }
      }
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

}
