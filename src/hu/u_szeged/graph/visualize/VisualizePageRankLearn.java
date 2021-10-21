package hu.u_szeged.graph.visualize;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;

import org.graphstream.graph.Edge;
import org.graphstream.graph.Graph;
import org.graphstream.graph.Node;
import org.graphstream.graph.implementations.SingleGraph;
import org.graphstream.ui.spriteManager.Sprite;
import org.graphstream.ui.spriteManager.SpriteManager;

import cc.mallet.optimize.OwnLimitedMemoryBFGS;
import hu.u_szeged.experiments.SyntheticExperiment;
import hu.u_szeged.graph.OwnGraph;
import hu.u_szeged.graph.PRWeightLearner;
import hu.u_szeged.graph.SoftmaxPRWeightLearner;

public class VisualizePageRankLearn {

  public static final int UNIT_NODE_SIZE = 250;
  public static final int UNIT_EDGE_WIDTH = 1;

  private static double[] calculatePrStar(int[] unnormalizedRanks) {
    int numOfNodes = unnormalizedRanks.length;
    double[] prStar = new double[numOfNodes];
    double sumEtalonRank = 0.0d;

    for (int i = 0; i < numOfNodes; ++i) {
      prStar[i] = unnormalizedRanks[i];
      sumEtalonRank += unnormalizedRanks[i];
    }
    for (int i = 0; i < numOfNodes; ++i) {
      prStar[i] /= (double) sumEtalonRank;
    }
    return prStar;
  }

  public static void main(String args[]) throws IOException {
    SyntheticExperiment.RANDOM.setSeed(10l);
    System.setProperty("org.graphstream.ui.renderer", "org.graphstream.ui.j2dviewer.J2DGraphRenderer");

    int[][] testCase = new int[][] { { 26, 24, 12, 9, 16, 13 }, { 0, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5 }, { 1, 0, 2, 0, 1, 5, 2, 4, 3, 5, 3, 4 } };
    // int[][] testCase = new int[][] { { 18192, 21089, 32239, 22480 }, { 0, 0, 1, 1, 1, 2, 2, 2, 3, 3 }, { 1, 2, 0, 2, 3, 0, 1, 3, 1, 2 } };
    double[] prStar = calculatePrStar(testCase[0]);
    FileWriter fw = new FileWriter("inversePR.txt");
    fw.write(String.format("The expected stationary distribution the random walk should converge to is: %s\n", Arrays.toString(prStar)));
    OwnGraph g = new OwnGraph(testCase[1], testCase[2], true, testCase[0].length);
    String originalMtx = g.returnWeightMatrix();
    fw.write(String.format("The original (i.e. uniformly weighted) transition matrix:\n%s\n", originalMtx));
    System.err.println(String.format("The original (i.e. uniformly weighted) transition matrix:\n%s\n", originalMtx));

    PRWeightLearner pwl = new SoftmaxPRWeightLearner(prStar, g);
    pwl.setRegularization(0d, PRWeightLearner.RegularizationType.NONE);

    Graph graph = new SingleGraph("PageRankLearn");
    SpriteManager sman = new SpriteManager(graph);
    graph.addAttribute("ui.stylesheet", "url('./style.css')");

    for (int i = 0; i < g.getNumOfNodes(); ++i) {
      Node n = graph.addNode(g.getNodeLabel(i));
      n.setAttribute("label", g.getNodeLabel(i));
      if (pwl.getActualPRvalue(i) > pwl.getEtalonPRvalue(i)) {
        n.addAttribute("ui.class", "big");
        n.addAttribute("ui.color", pwl.getEtalonPRvalue(i) / pwl.getActualPRvalue(i));
      } else {
        n.addAttribute("ui.class", "small");
        n.addAttribute("ui.color", pwl.getActualPRvalue(i) / pwl.getEtalonPRvalue(i));
      }
      n.addAttribute("ui.size", UNIT_NODE_SIZE * pwl.getActualPRvalue(i));
      Sprite s = sman.addSprite("S" + i);
      s.addAttribute("ui.size", UNIT_NODE_SIZE * pwl.getEtalonPRvalue(i));
      s.attachToNode(n.getId());
    }
    for (int i = 0; i < g.getNumOfNodes(); ++i) {
      int[] neighs = g.getOutLinks(i);
      for (int n = 1; n <= neighs[0]; ++n) {
        Edge e = graph.addEdge(i + "_" + neighs[n], i, neighs[n], true);
        e.addAttribute("ui.size", Math.max(0, UNIT_EDGE_WIDTH * g.getWeight(i, neighs[n])));
      }
    }

    double initObjVal = pwl.getValue();
    OwnLimitedMemoryBFGS optimizer = new OwnLimitedMemoryBFGS(pwl);
    optimizer.addGraph(graph);
    try {
      optimizer.optimize();
    } catch (Exception e) {
      e.printStackTrace();
      System.err.println(e.getLocalizedMessage());
    }
    double finalObjective = pwl.getValue();
    pwl.getGraph().softmaxNormalizeWeights();
    pwl.extensiveLog();
    String finalMtx = g.returnWeightMatrix();
    fw.write(String.format("Objective value changed from %.6f to %.6f.\n", -initObjVal, -finalObjective));
    fw.write(String.format("The learned weighted transition matrix:\n%s\n", finalMtx));
    System.err.println(String.format("The learned weighted transition matrix:\n%s\n", finalMtx));
    fw.write(String.format("The stationary distribution of the learned random walk is: %s\n", Arrays.toString(pwl.getActualPRvalues())));
    System.err.format("Objective value changed from %.6f to %.6f.\n", -initObjVal, -finalObjective);
    fw.close();
  }
}