package hu.u_szeged.experiments;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import hu.u_szeged.graph.PRWeightLearner;

public class CitibikeExperiment extends AbstractExperiment {

  private Map<Integer, Integer> outLinks;
  private Map<Integer, Map<Integer, Double>> etalonTransitions;

  public CitibikeExperiment(String file, double teleportProb) {
    super(file);
    init(teleportProb);
    learner.setRegularization(0, PRWeightLearner.RegularizationType.ORACLE);
  }

  @Override
  protected void readNodes(String inFile) {
    etalonTransitions = new HashMap<>();
    try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(inFile)))) {
      String line;
      int max = 0;
      while ((line = br.readLine()) != null) {
        String parts[] = line.split("\t");
        labelsToIds.putIfAbsent(parts[0], Integer.parseInt(parts[0]));
        labelsToIds.putIfAbsent(parts[1], Integer.parseInt(parts[1]));
        etalonTransitions.putIfAbsent(Integer.parseInt(parts[0]), new HashMap<>());
        max = Math.max(max, Math.max(Integer.parseInt(parts[0]), Integer.parseInt(parts[1])));
      }
      for (int i = 0; i < max; ++i) {
        labelsToIds.putIfAbsent(Integer.toString(i), i);
        etalonTransitions.putIfAbsent(i, new HashMap<>());
      }
    } catch (IOException ie) {
      ie.printStackTrace();
    }
  }

  @Override
  protected void readEdges(String infile) {
    outLinks = new HashMap<>();
    try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(infile)))) {
      String line;
      while ((line = br.readLine()) != null) {
        String parts[] = line.split("\t");
        Integer fromNode = Integer.parseInt(parts[0]);
        outLinks.put(fromNode, outLinks.getOrDefault(fromNode, 0) + (int) Double.parseDouble(parts[2]));
        etalonTransitions.get(fromNode).put(Integer.parseInt(parts[1]), Double.parseDouble(parts[2]));
        g.addEdge(labelsToIds.get(parts[0]), labelsToIds.get(parts[1]), Double.parseDouble(parts[2]));
      }
    } catch (IOException ie) {
      ie.printStackTrace();
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
    for (int i = 0; i < g.getNumOfNodes(); ++i) {
      etalonDistr[i] /= totalWeights;
    }
  }

  public static void main(String[] args) {
    try (PrintWriter out = new PrintWriter("citibike.results")) {
      String[] evalMetrics = new String[] { "KL", "RMSE", "accuracy", "MRR", "displacement" };
      for (String m : evalMetrics) {
        out.write(m + "\t");
      }
      out.write("N\tmode\tteleport\tnum_models\tnodeID\n");
      for (double teleProb : new double[] { 0.2, 0.1, 0.05, 0.01 }) {
        for (int ri = 1; ri < 6; ++ri) { // number of random initializations to apply
          CitibikeExperiment ce = new CitibikeExperiment("citibike2015_edges.tsv", teleProb);
          ce.setVerbose(false);
          ce.learnWeights(ri, false);
          ce.readChoiceRankParams("citibike2015.params");

          Random r = new Random(1);

          System.err.println("==========" + teleProb + " " + ri);
          for (String mode : new String[] { "uniform", "indegree", "jaccard", "popularity", "pagerank", "prlearn", "choicerank" }) {
            double[] evals = new double[5];
            int counter = 0;
            for (int n = 0; n < ce.g.getNumOfNodes(); ++n) {
              int[] ns = ce.g.getOutLinks(n);
              if (ns[0] == 0) {
                continue;
              }
              counter++;

              if (mode.equals("prlearn") || (teleProb == 0.01 && ri == 5)) {
                double[] nodeEvals = ce.evaluateNode(n, ce.etalonTransitions.get(n), mode, r);
                for (int e = 0; e < evals.length; ++e) {
                  out.format("%.4f\t", nodeEvals[e]);
                  evals[e] += nodeEvals[e];
                }
                out.format("%d\t%s\t%.2f\t%d\t%d\n", ns[0], mode, teleProb, ri, n);
              }
            }
            for (int e = 0; e < evals.length; ++e) {
              evals[e] /= counter;
            }
            if (mode.equals("prlearn") || (teleProb == 0.01 && ri == 5)) {
              System.err.format("%s\t%.2f\t%d\t%s\t%d\t%d\n", mode, teleProb, ri, Arrays.toString(evals), counter, ce.g.getNumOfNodes());
            }
          }
        }
      }
    } catch (IOException e) {
      e.printStackTrace();
    }
  }
}
