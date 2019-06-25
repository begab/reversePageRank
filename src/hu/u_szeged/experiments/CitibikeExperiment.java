package hu.u_szeged.experiments;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.Random;

import hu.u_szeged.graph.PRWeightLearner;
import hu.u_szeged.graph.PageRankCalculator;
import hu.u_szeged.utils.Utils;

public class CitibikeExperiment extends AbstractExperiment {

  private int[] argmaxNeighbors;
  private double[] pagerank;

  public CitibikeExperiment(String file, boolean useProjectedLearning, double teleportProb) {
    super(file, useProjectedLearning);
    init(teleportProb);
    learner.setRegularization(0, PRWeightLearner.RegularizationType.ORACLE);
  }

  @Override
  protected void readNodes(String inFile) {
    try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(inFile)))) {
      String line;
      int max = 0;
      while ((line = br.readLine()) != null) {
        String parts[] = line.split("\t");
        labelsToIds.putIfAbsent(parts[0], Integer.parseInt(parts[0]));
        labelsToIds.putIfAbsent(parts[1], Integer.parseInt(parts[1]));
        max = Math.max(max, Math.max(Integer.parseInt(parts[0]), Integer.parseInt(parts[1])));
      }
      for (int i = 0; i < max; ++i) {
        labelsToIds.putIfAbsent(Integer.toString(i), i);
      }
    } catch (IOException ie) {
      ie.printStackTrace();
    }
  }

  @Override
  protected void readEdges(String infile) {
    try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(infile)))) {
      String line;
      while ((line = br.readLine()) != null) {
        String parts[] = line.split("\t");
        g.addEdge(labelsToIds.get(parts[0]), labelsToIds.get(parts[1]), Double.parseDouble(parts[2]));
      }
    } catch (IOException ie) {
      ie.printStackTrace();
    }
  }

  public void determineEtalon() {
    etalonDistr = new double[g.getNumOfNodes()];
    argmaxNeighbors = new int[g.getNumOfNodes()];
    double totalWeights = 0.0;
    for (int i = 0; i < g.getNumOfNodes(); ++i) {
      int[] neighbors = g.getOutLinks(i);
      double[] weights = g.getWeights(i);
      int argmaxN = -1;
      double maxWeight = 0;
      for (int j = 1; j <= neighbors[0]; ++j) {
        if (weights[j] > maxWeight) {
          maxWeight = weights[j];
          argmaxN = neighbors[j];
        }
        totalWeights += weights[j];
        etalonDistr[neighbors[j]] += weights[j];
      }
      argmaxNeighbors[i] = argmaxN;
    }
    for (int i = 0; i < g.getNumOfNodes(); ++i) {
      etalonDistr[i] /= totalWeights;
    }
    try (PrintWriter out = new PrintWriter("citibike2015_edges_adapted.tsv")) {
      for (int i = 0; i < g.getNumOfNodes(); ++i) {
        int[] neigs = g.getOutLinks(i);
        for (int j = 1; j <= neigs[0]; ++j) {
          out.format("%d\t%d\t%f\n", i, neigs[j], etalonDistr[neigs[j]]);
        }
      }
    } catch (IOException io) {
      io.printStackTrace();
    }
    PageRankCalculator prc = new PageRankCalculator();
    pagerank = prc.calculatePageRank(g, false);
  }

  private int evaluateNode(double[] ws, int[] ns, int etalonargmax) {
    int rank = 1;
    double probForTrueArgMax = 0;
    for (int i = 1; i <= ns[0]; ++i) {
      if (etalonargmax == ns[i]) {
        probForTrueArgMax = ws[i];
      }
    }
    // System.err.println(etalonargmax + " " + probForTrueArgMax + " " + Arrays.toString(ns));
    for (int i = 1; i <= ns[0]; ++i) {
      if (ws[i] > probForTrueArgMax) {
        rank++;
      }
    }
    return rank;
  }

  public static void main(String[] args) {
    for (double teleProb : new double[] { 0.2, 0.1, 0.05, 0.01 }) {
      for (int ri = 1; ri < 6; ++ri) {
        CitibikeExperiment ce = new CitibikeExperiment("citibike2015_edges.tsv", false, teleProb);
        ce.setVerbose(false);
        ce.learnWeights(ri, true);
        Random r = new Random(1);

        double[] choicerankParams = new double[ce.pagerank.length];
        try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream("citibike_adapted_choicerank.params")))) {
          String line;
          int i = 0;
          while ((line = br.readLine()) != null) {
            if (i < choicerankParams.length) {
              choicerankParams[i++] = Double.parseDouble(line);
            }
          }
        } catch (IOException e) {
          e.printStackTrace();
        }

        System.err.println("==========");
        for (String mode : new String[] { "uniform", "indegree", "jaccard", "popularity", "pagerank", "prlearn", "choicerank" }) {
          // for (String mode : new String[] { "prlearn" }) {
          double precision = 0, mrr = 0, c = 0;
          for (int n = 0; n < ce.g.getNumOfNodes(); ++n) {
            int[] ns = ce.g.getOutLinks(n);
            if (ns[0] < 2) {
              continue;
            }
            c++;
            double[] ws = new double[ns[0] + 1];
            if (mode.equals("jaccard")) {
              for (int i = 1; i <= ns[0]; ++i) {
                int[] neighbors = ce.g.getOutLinks(ns[i]);
                ws[i] = Utils.determineOverlap(ns, neighbors);
                ws[i] /= (double) (ns[0] + neighbors[0] - ws[i]);
              }
            } else if (mode.equals("prlearn")) {
              ws = ce.g.getWeights(n);
            } else if (mode.equals("indegree")) {
              for (int i = 1; i <= ns[0]; ++i) {
                ws[i] = ce.g.getIndegree(ns[i]);
              }
            } else if (mode.equals("uniform")) {
              for (int i = 1; i <= ns[0]; ++i) {
                ws[i] = r.nextDouble();
              }
            } else if (mode.equals("popularity")) {
              for (int i = 1; i <= ns[0]; ++i) {
                ws[i] = ce.etalonDistr[ns[i]];
              }
            } else if (mode.equals("pagerank")) {
              for (int i = 1; i <= ns[0]; ++i) {
                ws[i] = ce.pagerank[ns[i]];
              }
            } else if (mode.contains("choicerank")) {
              for (int i = 1; i <= ns[0]; ++i) {
                ws[i] = choicerankParams[ns[i]];
              }
            }
            int rank = ce.evaluateNode(ws, ns, ce.argmaxNeighbors[n]);
            precision += rank == 1 ? 1 : 0;
            mrr += 1.0 / rank;
            // break;
          }
          System.err.format("%s\t%f\t%f\t%d\t%d\n", mode, (precision / c), (mrr / c), (int) c, ce.g.getNumOfNodes());
        }
      }
    }
  }
}
