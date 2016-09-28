package hu.u_szeged.graph.visualize;

import java.util.Arrays;

import org.graphstream.graph.Edge;
import org.graphstream.graph.Graph;
import org.graphstream.graph.Node;
import org.graphstream.graph.implementations.SingleGraph;
import org.graphstream.ui.spriteManager.Sprite;
import org.graphstream.ui.spriteManager.SpriteManager;

import cc.mallet.optimize.OwnLimitedMemoryBFGS;
import hu.u_szeged.graph.OwnGraph;
import hu.u_szeged.graph.PRWeightLearner;
import hu.u_szeged.graph.SoftmaxPRWeightLearner;

public class VisualizePageRankLearn {

  public static final int UNIT_NODE_SIZE = 250;
  public static final int UNIT_EDGE_WIDTH = 4;

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

  private static Graph createVisualizationGraph(OwnGraph g, PRWeightLearner pwl) {
    Graph graph = new SingleGraph("PageRankLearn");
    SpriteManager sman = new SpriteManager(graph);
    graph.addAttribute("ui.stylesheet", "url('./style.css')");

    // Copy the nodes from g to graph
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
    // Copy the edges from g to graph
    for (int i = 0; i < g.getNumOfNodes(); ++i) {
      int[] neighs = g.getOutLinks(i);
      for (int n = 1; n <= neighs[0]; ++n) {
        Edge e = graph.addEdge(i + "_" + neighs[n], i, neighs[n], true);
        e.addAttribute("ui.size", UNIT_EDGE_WIDTH * g.getWeight(i, neighs[n]));
      }
    }
    return graph;
  }

  private static void displayGraph(Graph visualizationGraph) {
    visualizationGraph.display();
    try {
      Thread.sleep(3000);
    } catch (InterruptedException e1) {
      e1.printStackTrace();
    }
  }

  public static void main(String args[]) {
    OwnGraph.RANDOM.setSeed(10l);
    System.setProperty("org.graphstream.ui.renderer", "org.graphstream.ui.j2dviewer.J2DGraphRenderer");

    int[][] testCase = new int[][] { { 20, 10, 20, 30, 20, 10 }, { 0, 0, 1, 2, 2, 3, 4 }, { 1, 2, 2, 3, 5, 4, 5 } };
    double[] prStar = calculatePrStar(testCase[0]);
    System.err.println(Arrays.toString(prStar));
    OwnGraph inputGraph = new OwnGraph(testCase[1], testCase[2], false, testCase[0].length);
    System.err.println("Weight matrix of g before optimization of graph.");
    inputGraph.printWeightMatrix();

    PRWeightLearner prWeightLearner = new SoftmaxPRWeightLearner(prStar, inputGraph, .1d); // new ProjectedPRWeightLearner(prStar, inputGraph, .1d);
    prWeightLearner.setRegularizationWeight(0.0d);

    Graph visualizationGraph = createVisualizationGraph(inputGraph, prWeightLearner);
    displayGraph(visualizationGraph);

    double initObjVal = prWeightLearner.getValue();
    OwnLimitedMemoryBFGS optimizer = new OwnLimitedMemoryBFGS(prWeightLearner);
    optimizer.addVisualizationGraph(visualizationGraph);
    try {
      optimizer.optimize();
    } catch (Exception e) {
      e.printStackTrace();
      System.err.println(e.getLocalizedMessage());
    }
    double finalObjective = prWeightLearner.getValue();
    prWeightLearner.getGraph().softmaxNormalizeWeights();
    prWeightLearner.extensiveLog();
    System.err.println("Weight matrix of g after optimization of graph.");
    inputGraph.printWeightMatrix();
    System.err.format("Objective value changed from %.6f to %.6f.\n", -initObjVal, -finalObjective);
  }
}