package hu.u_szeged.graph.reader;

import java.io.IOException;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

import org.graphstream.graph.Edge;
import org.graphstream.graph.Graph;
import org.graphstream.graph.Node;
import org.graphstream.graph.implementations.DefaultGraph;
import org.graphstream.stream.file.FileSource;
import org.graphstream.stream.file.FileSourceFactory;

import hu.u_szeged.graph.OwnGraph;
import hu.u_szeged.graph.PRWeightLearner;
import hu.u_szeged.graph.SoftmaxPRWeightLearner;

public class GraphReader {

  private static Random rnd = new Random(10l);
  private static double[] etalonWeights;

  public static OwnGraph readGraph(String f) {
    Graph g = new DefaultGraph("graph", false, true);
    try {
      FileSource fs = FileSourceFactory.sourceFor(f);
      fs.addSink(g);
      fs.readAll(f);
      fs.removeSink(g);
    } catch (IOException e) {
      e.printStackTrace();
    }
    OwnGraph ownGraph = new OwnGraph(g.getNodeCount());
    Iterator<? extends Node> nIt = g.getEachNode().iterator();
    int counter = 0;
    boolean warning = false;
    double weightSum = 0;
    List<Double> weights = new LinkedList<>();
    while (nIt.hasNext()) {
      Node n = nIt.next();
      String label = n.getAttribute("label");
      Double etalonNodeWeight = 1d;
      if (n.hasAttribute("weight")) {
        etalonNodeWeight = Double.parseDouble(n.getAttribute("weight"));
      } else {
        etalonNodeWeight = rnd.nextDouble();
        warning = true;
      }
      weightSum += etalonNodeWeight;
      weights.add(etalonNodeWeight);
      ownGraph.setNodeLabel(label == null ? Integer.toString(counter) : label, n.getIndex());
      counter++;
    }

    if (warning) {
      System.err.println("WARNING: no 'weight' attribute was present in the input file, hence etalon node importances were generated randomly.");
    }

    etalonWeights = new double[ownGraph.getNumOfNodes()]; // normalize the etalon weights to form a distribution
    int i = 0;
    for (double w : weights) {
      etalonWeights[i++] = w / weightSum;
    }

    Iterator<? extends Edge> eIt = g.getEachEdge().iterator();
    while (eIt.hasNext()) {
      Edge e = eIt.next();
      ownGraph.addEdge(e.getSourceNode().getIndex(), e.getTargetNode().getIndex());
    }
    System.err.format("Network from file %s with %d vertices and %d edges read in.\n", f, ownGraph.getNumOfNodes(), ownGraph.getNumOfEdges());

    return ownGraph;
  }

  public static void main(String[] args) {
    String inputGraphFile = "./airlines-sample.gexf";
    String outputFile = "output.log";

    if (args.length > 0) {
      inputGraphFile = args[0];
    }

    OwnGraph g = GraphReader.readGraph(inputGraphFile);

    PRWeightLearner pr = new SoftmaxPRWeightLearner(etalonWeights, g);
    pr.setLogFile(outputFile);

    double[] res = pr.learnEdgeWeights();

    System.err.format("The initial objective value improved from %f to %f\n" + "The predicted edge transition probabilities were written into file %s.", res[0],
        res[1], outputFile);

    g.softmaxNormalizeWeights();
    pr.extensiveLog();
  }
}
