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
import java.util.zip.GZIPInputStream;

import hu.u_szeged.graph.PRWeightLearner;

public class KosarakExperiment extends AbstractExperiment {

  private Map<Integer, Integer> outLinks;
  private Map<Integer, Map<Integer, Double>> etalonTransitions;

  public KosarakExperiment(String file, double teleportProb) {
    super(file);
    init(teleportProb);
    learner.setRegularization(0, PRWeightLearner.RegularizationType.ORACLE);
  }

  @Override
  protected void readNodes(String inFile) {
    etalonTransitions = new HashMap<>();

    try (BufferedReader br = new BufferedReader(new InputStreamReader(new GZIPInputStream(new FileInputStream(inFile))))) {
      String line;
      while ((line = br.readLine()) != null) {
        String parts[] = line.split(" ");
        if (parts.length > 1) {
          for (String label : parts) {
            if (!labelsToIds.containsKey(label)) {
              Integer id = labelsToIds.size();
              labelsToIds.putIfAbsent(label, id);
              etalonTransitions.putIfAbsent(id, new HashMap<>());
            }
          }
        }
      }
    } catch (IOException ie) {
      ie.printStackTrace();
    }
  }

  @Override
  protected void readEdges(String infile) {
    outLinks = new HashMap<>();
    try (BufferedReader br = new BufferedReader(new InputStreamReader(new GZIPInputStream(new FileInputStream(infile))))) {
      String line;
      while ((line = br.readLine()) != null) {
        String parts[] = line.split(" ");
        for (int i = 1; i < parts.length; ++i) {
          Integer from = labelsToIds.get(parts[i - 1]);
          Integer to = labelsToIds.get(parts[i]);
          outLinks.put(from, outLinks.getOrDefault(from, 0) + 1);
          etalonTransitions.get(from).put(to, etalonTransitions.get(from).getOrDefault(to, 0.) + 1);
          g.addEdge(from, to, 1);
        }
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
    System.err.println(totalWeights);
    for (int i = 0; i < g.getNumOfNodes(); ++i) {
      etalonDistr[i] /= totalWeights;
    }
  }

  public static void main(String[] args) {
    try (PrintWriter out = new PrintWriter("kosarak.results")) {
      String[] evalMetrics = new String[] { "KL", "RMSE", "accuracy", "MRR", "displacement" };
      for (String m : evalMetrics) {
        out.write(m + "\t");
      }
      out.write("N\tmode\tteleport\tnum_models\tnodeID\n");
      for (double teleProb : new double[] { 0.2, 0.1, 0.05, 0.01 }) {
        for (int ri = 1; ri < 6; ++ri) { // number of random initializations to apply
          KosarakExperiment ke = new KosarakExperiment("./data/kosarak.dat.gz", teleProb);
          ke.setVerbose(false);
          ke.learnWeights(ri, false);
          ke.readChoiceRankParams("./data/kosarak.params");

          Random r = new Random(1);

          System.err.println("==========" + teleProb + " " + ri);
          for (String mode : new String[] { "uniform", "indegree", "jaccard", "popularity", "pagerank", "prlearn", "choicerank" }) {
            double[] evals = new double[5];
            int counter = 0;
            for (int n = 0; n < ke.g.getNumOfNodes(); ++n) {
              int[] ns = ke.g.getOutLinks(n);
              if (ns[0] > 1) {
                counter++;

                if (mode.equals("prlearn") || (teleProb == 0.01 && ri == 5)) {
                  double[] nodeEvals = ke.evaluateNode(n, ke.etalonTransitions.get(n), mode, r);
                  for (int e = 0; e < evals.length; ++e) {
                    out.format("%.4f\t", nodeEvals[e]);
                    evals[e] += nodeEvals[e];
                  }
                  out.format("%d\t%s\t%.2f\t%d\t%d\n", ns[0], mode, teleProb, ri, n);
                }
              }
            }
            for (int e = 0; e < evals.length; ++e) {
              evals[e] /= counter;
            }
            if (mode.equals("prlearn") || (teleProb == 0.01 && ri == 5)) {
              System.err.format("%s\t%.2f\t%d\t%s\t%d\t%d\n", mode, teleProb, ri, Arrays.toString(evals), counter, ke.g.getNumOfNodes());
            }
          }
        }
      }
    } catch (IOException e) {
      e.printStackTrace();
    }
  }
}
