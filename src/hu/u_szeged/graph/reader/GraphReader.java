package hu.u_szeged.graph.reader;

import java.io.IOException;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

import org.graphstream.graph.Edge;
import org.graphstream.graph.Graph;
import org.graphstream.graph.Node;
import org.graphstream.graph.implementations.DefaultGraph;
import org.graphstream.stream.file.FileSource;
import org.graphstream.stream.file.FileSourceFactory;

import hu.u_szeged.graph.OwnGraph;

public class GraphReader {

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
    double weightSum = 0;
    List<Double> weights = new LinkedList<>();
    while (nIt.hasNext()) {
      Node n = nIt.next();
      String label = n.getAttribute("label");
      Double etalonNodeWeight = 1d;
      if (n.hasAttribute("weight")) {
        etalonNodeWeight = Double.parseDouble(n.getAttribute("weight"));
      }
      weightSum += etalonNodeWeight;
      weights.add(etalonNodeWeight);
      ownGraph.setNodeLabel(label == null ? Integer.toString(counter) : label, n.getIndex());
      counter++;
    }
    Iterator<? extends Edge> eIt = g.getEachEdge().iterator();
    while (eIt.hasNext()) {
      Edge e = eIt.next();
      ownGraph.addEdge(e.getSourceNode().getIndex(), e.getTargetNode().getIndex());
    }
    return ownGraph;
  }

  public static void main(String[] args) {
    OwnGraph g = GraphReader.readGraph("/home/berend/Desktop/airlines-sample.gexf");
    System.err.println(g.getNumOfNodes() + " " + g.getNumOfEdges());
    for (int n = 0; n < 10; ++n) {
      int[] ids = g.getInLinks(n);
      System.err.println(n + " " + ids[0]);
    }
  }
}
