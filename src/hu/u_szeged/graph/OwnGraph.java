package hu.u_szeged.graph;

import java.io.FileNotFoundException;
import java.io.PrintStream;
import java.io.PrintWriter;
import java.io.Serializable;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Set;

import hu.u_szeged.utils.Utils;

public class OwnGraph implements Serializable, Cloneable {

  /**
   * 
   */
  private static final long serialVersionUID = -2201356524637857822L;

  private int numOfNodes;
  private int numOfEdges;
  private String[] nodeLabels;
  private Map<String, Integer> lookupForNodeId;
  private int[] indegrees;
  private int[][] neighbors;
  private double[][] weights;

  public OwnGraph(int nodeNum) {
    numOfNodes = nodeNum;
    lookupForNodeId = new HashMap<>();
    nodeLabels = new String[numOfNodes];
    indegrees = new int[numOfNodes];
    neighbors = new int[numOfNodes][2];
    weights = new double[numOfNodes][2];
  }

  public OwnGraph(int[] from, int[] to, int nodeNum) {
    this(from, to, null, true, nodeNum);
  }

  public OwnGraph(int[] from, int[] to, boolean directed, int nodeNum) {
    this(from, to, null, directed, nodeNum);
  }

  public OwnGraph(int[] from, int[] to, double[] ws, boolean directed, int nodeNum) {
    this(nodeNum);
    if (ws == null) {
      ws = new double[from.length];
      for (int i = 0; i < from.length; ++i) {
        ws[i] = 1.0d;
      }
    }
    for (int i = 0; i < numOfNodes; ++i) {
      setNodeLabel(Character.toString((char) (65 + i)), i);
    }
    for (int i = 0; i < from.length; ++i) {
      if (directed) {
        addEdge(from[i], to[i]);
      } else {
        addBidirectionalEdge(from[i], to[i]);
      }
    }
    normalizeWeights();
  }

  public OwnGraph clone() {
    OwnGraph copy = new OwnGraph(this.getNumOfNodes());
    for (int i = 0; i < this.getNumOfNodes(); ++i) {
      copy.setNodeLabel(this.getNodeLabel(i), i);
      int[] neighbors = this.getOutLinks(i);
      double[] weightsToClone = this.getWeights(i);
      for (int j = 1; j <= neighbors[0]; ++j) {
        copy.addEdge(i, neighbors[j], weightsToClone[j]);
      }
    }
    return copy;
  }

  public enum WeightingStrategy {
    UNIFORM, DEGREE_BASED, INVERSE_DEGREE_BASED, EXACT, RAND
  }

  public OwnGraph copyGraph(WeightingStrategy weightStrat) {
    OwnGraph copy = clone();
    if (weightStrat != WeightingStrategy.EXACT) {
      copy.initWeights(weightStrat);
    }
    return copy;
  }

  public int getIndegree(int i) {
    return indegrees[i];
  }

  public int getNumOfNeighbors(int i) {
    return neighbors[i][0];
  }

  public int getOutDegree(int i) {
    return neighbors[i][0];
  }

  public int getTotalDegree(int i) {
    return indegrees[i] + neighbors[i][0];
  }

  /**
   * Returns the in-links of a node. All the nodes are iterated over and the target node is looked for as an out-link. <br />
   * The zero-index element in the returned array contains the number of in-degrees for the target node.
   * 
   * @param target
   * @return
   */
  public int[] getInLinks(int target) {
    int[] inLinks = new int[indegrees[target] + 1];
    inLinks[0] = indegrees[target];
    for (int i = 0, j = 0; i < getNumOfNodes() && j < inLinks.length; ++i) {
      if (findNodeIndex(i, target) >= 0) {
        inLinks[++j] = i;
      }
    }
    return inLinks;
  }

  /**
   * Returns the out-links of a some target node.
   * 
   * @param target
   * @return
   */
  public int[] getOutLinks(int target) {
    return neighbors[target];
  }

  /**
   * Returns the labels of the out-links of some target node.
   * 
   * @param target
   * @return
   */
  public String[] getOutLinkLabels(int target) {
    int[] neighs = neighbors[target];
    String[] outLinkLabels = new String[neighs[0]];
    for (int i = 1; i <= neighs[0]; ++i) {
      outLinkLabels[i - 1] = nodeLabels[neighs[i]];
    }
    return outLinkLabels;
  }

  /**
   * Returns all the weights for all the nodes "unrolled" as one double array.
   * 
   * @return
   */
  public double[] getWeights() {
    double[] everyWeight = new double[numOfEdges];
    for (int i = 0, index = 0; i < numOfNodes; ++i) {
      int nbrs = getNumOfNeighbors(i);
      double[] ws = weights[i];
      for (int j = 1; j <= nbrs; ++j, index++) {
        everyWeight[index] = ws[j];
      }
    }
    return everyWeight;
  }

  public double getTotalWeights() {
    double totalWeight = 0.0d;
    for (int i = 0; i < getNumOfNodes(); ++i) {
      totalWeight += getWeights(i)[0];
    }
    return totalWeight;
  }

  public double[] getWeights(int target) {
    return weights[target];
  }

  public double getWeight(int from, int to) {
    int index = findNodeIndex(from, to);
    double weight = 0.0d;
    if (index > 0) {
      weight = weights[from][index];
    }
    return weight;
  }

  public int getNumOfEdges() {
    return numOfEdges;
  }

  public int getNumOfNodes() {
    return numOfNodes;
  }

  /**
   * Reassigns edge weights in the graph based on the values of newWeights.
   * 
   * @param newWeights
   * @param softmax
   */
  public void setWeights(double[] newWeights) {
    int i = 0;
    if (newWeights.length != numOfEdges) {
      System.err.println("FATAL error: the length of the new parameters should exactly match that of the number of edges in the graph.");
      System.exit(1);
    }
    for (int n = 0; n < numOfNodes; ++n) {
      int[] neighs = neighbors[n];
      double[] ws = new double[neighs[0] + 1];
      double sum = 0.0d;
      for (int k = 0; k < neighs[0]; ++k, ++i) {
        sum += (ws[k + 1] = newWeights[i]);
      }
      ws[0] = sum;
      weights[n] = ws;
    }
  }

  public void setEdgeWeight(int from, int to, double newWeight) {
    updateEdgeWeight(from, to, newWeight, true);
  }

  public void updateEdgeWeight(int from, int to, double newWeight) {
    updateEdgeWeight(from, to, newWeight, false);
  }

  public void updateEdgeWeight(int from, int to, double newWeight, boolean replaceOld) {
    /**
     * Updates the weight for the edge (from, to) with the weight given in newWeight.
     */
    if (from >= numOfNodes || to >= numOfNodes) {
      System.err.println("The edge for which the weight is to be set was not present in the graph previously.");
      return;
    }
    double[] nodeWeights = weights[from];
    int index = this.findNodeIndex(from, to);
    if (index < 0) {
      System.err.println("The edge for which the weight is to be set was not present in the graph previously.");
      return;
    }
    if (replaceOld) {
      nodeWeights[0] += (newWeight - nodeWeights[index]);
      nodeWeights[index] = newWeight;
    } else {
      nodeWeights[0] += newWeight;
      nodeWeights[index] += newWeight;
    }
  }

  public void initWeights(WeightingStrategy weightStrat) {
    for (int i = 0; i < getNumOfNodes(); ++i) {
      int[] neighs = getOutLinks(i);
      if (weightStrat == WeightingStrategy.RAND) {
        weights[i] = Utils.drawMultinomial(neighs[0], new double[] { 1.0d });
      } else if (weightStrat == WeightingStrategy.DEGREE_BASED || weightStrat == WeightingStrategy.INVERSE_DEGREE_BASED) {
        double[] neighsTotalDegree = new double[neighs[0]];
        int maxDegree = 0;
        for (int n = 1; n <= neighs[0]; ++n) {
          neighsTotalDegree[n - 1] = getIndegree(neighs[n]) + getNumOfNeighbors(neighs[n]);
          maxDegree = Math.max(maxDegree, (int) neighsTotalDegree[n - 1]);
        }
        for (int n = 1; weightStrat == WeightingStrategy.INVERSE_DEGREE_BASED && n <= neighs[0]; ++n) {
          neighsTotalDegree[n - 1] = maxDegree - neighsTotalDegree[n - 1] + 1;
        }
        weights[i] = Utils.drawMultinomial(neighs[0], neighsTotalDegree);
      } else if (weightStrat == WeightingStrategy.UNIFORM) {
        weights[i] = new double[neighs[0] + 1];
        for (int n = 0; n <= neighs[0]; ++n) {
          weights[i][n] = 0.0d;
        }
      }
    }
  }

  public void setNodeLabel(String label, int nodeId) {
    nodeLabels[nodeId] = label;
    lookupForNodeId.put(label, nodeId);
  }

  public String getNodeLabel(int n) {
    return nodeLabels[n];
  }

  public boolean containsNode(String s) {
    return lookupForNodeId.containsKey(s);
  }

  public boolean containsEdge(String from, String to) {
    return containsEdge(lookupForNodeId.getOrDefault(from, numOfNodes), lookupForNodeId.getOrDefault(to, numOfNodes));
  }

  public boolean containsEdge(int from, int to) {
    if (from < numOfNodes && to < numOfNodes) {
      return findNodeIndex(from, to) >= 0;
    }
    return false;
  }

  public Integer getNodeIdByLabel(String label) {
    return lookupForNodeId.get(label);
  }

  public boolean[] addBidirectionalEdge(int from, int to) {
    return addBidirectionalEdge(from, to, 1.0d);
  }

  public boolean[] addBidirectionalEdge(int from, int to, double weight) {
    boolean successA = addEdge(from, to, weight);
    boolean successB = addEdge(to, from, weight);
    return new boolean[] { successA, successB };
  }

  public boolean addEdge(int from, int to) {
    return addEdge(from, to, 1.0d);
  }

  /**
   * The edge is only added if it was not previously present in the network. <br />
   * In case the edge was present then its weight is going to be increased by the weight parameter.<br />
   * Returns true if the edge was not yet present in the graph.
   * 
   * @param from
   * @param to
   * @param weight
   */
  public boolean addEdge(int from, int to, double weight) {
    int index = findNodeIndex(from, to);
    if (index < 0) {
      index = -index - 1;
      numOfEdges++;
      neighbors[from][0]++;
      indegrees[to]++;
      weights[from][0] += weight;
      if (neighbors[from].length == neighbors[from][0]) {
        neighbors[from] = Arrays.copyOf(neighbors[from], 2 * neighbors[from].length);
        weights[from] = Arrays.copyOf(weights[from], 2 * weights[from].length);
      }
      System.arraycopy(neighbors[from], index, neighbors[from], index + 1, neighbors[from].length - index - 1);
      System.arraycopy(weights[from], index, weights[from], index + 1, weights[from].length - index - 1);
      neighbors[from][index] = to;
      weights[from][index] = weight;
    } else {
      weights[from][0] += weight;
      weights[from][index] += weight;
    }
    return index < 0;
  }

  /**
   * Finds the relative position of the 'to' node within the list of neighbors of the 'from' node. <br/>
   * It finds the index using binary search as the node IDs of the neighbors are guaranteed to be ordered. <br />
   * If the to node is found a strictly positive number is returned. Otherwise, a non-positive number is returned.
   * 
   * @param from
   * @param to
   * @return
   */
  public int findNodeIndex(int from, int to) {
    return Arrays.binarySearch(neighbors[from], 1, neighbors[from][0] + 1, to);
  }

  public void removeBidirectionalEdge(int from, int to) {
    removeEdge(from, to);
    removeEdge(to, from);
  }

  public void removeEdge(int from, int to) {
    int[] ns = neighbors[from];
    double[] ws = weights[from];
    int index = findNodeIndex(from, to);
    if (index > 0) {
      numOfEdges--;
      indegrees[to]--;
      ws[0] -= ws[index];
      while (index < ns[0]) {
        ns[index] = ns[index + 1];
        ws[index] = ws[index + 1];
        index++;
      }
      ns[index] = 0;
      ws[index] = 0;
      ns[0]--; // the out-degree of the node has to be decreased
    }
  }

  public void normalizeWeights() {
    for (int i = 0; i < numOfNodes; ++i) {
      normalizeWeights(weights[i], getNumOfNeighbors(i));
    }
  }

  /**
   * This method normalizes the array in place, meaning that the value it stores will be modified.
   *
   * @param w
   * @param length
   */
  public void normalizeWeights(double[] w, int length) {
    double sum = 0.0d;
    for (int j = 1; j <= length; ++j) {
      sum += (w[j] = w[j] / w[0]);
    }
    if (Double.isNaN(sum)) { // there was a division by 0, i.e. all neighbors had a relative weight of 0
      sum = 0.0d;
      for (int j = 1; j <= length; ++j) {
        sum += (w[j] = 1.0d / length);
      }
    }
    w[0] = sum;
  }

  public void softmaxNormalizeWeights() {
    for (int i = 0; i < numOfNodes; ++i) {
      Utils.softmaxNormalize(weights[i], getNumOfNeighbors(i));
    }
  }

  public void softmaxDenormalizeWeights() {
    for (int i = 0; i < numOfNodes; ++i) {
      Utils.softmaxDenormalize(weights[i], getNumOfNeighbors(i));
    }
  }

  /**
   * This method should be called once a graph is believed not to change. <br/>
   * When invoked the method frees up memory that is not storing anything useful, which could be useful before serialization of a graph happens.
   */
  public void finalizeGraph() {
    for (int i = 0; i < numOfNodes; ++i) {
      int[] neighs = neighbors[i];
      if (neighs.length > neighs[0] + 1) {
        neighbors[i] = Arrays.copyOf(neighs, neighs[0] + 1);
      }
      double[] ws = weights[i];
      if (ws.length > neighs[0] + 1) {
        weights[i] = Arrays.copyOf(ws, neighs[0] + 1);
      }
    }
  }

  /**
   * This method check if the edge outgoing edge weights for each node forms a proper distribution.
   * 
   * @return
   */
  public boolean checkEdgeWeights() {
    boolean edgeWeightsOK = true;
    for (int n = 0; edgeWeightsOK && n < numOfNodes; ++n) {
      // the sum of the weights of its outgoing edges should sum to 1 (modulo a small tolerance due to possible numerical issues)
      if (this.neighbors[n][0] > 0 && Math.abs(this.weights[n][0] - 1.0d) > 1e-10) {
        edgeWeightsOK = false;
      }
    }
    return edgeWeightsOK;
  }

  public String toString() {
    StringBuffer sb = new StringBuffer();
    for (int i = 0; i < numOfNodes; ++i) {
      sb.append(String.format("ID:%d\t%s\tIndeg:%d\tOutdeg:%d\tNeighs:", i, nodeLabels[i], indegrees[i], neighbors[i][0]));
      if (neighbors[i][0] > 0) {
        sb.append(String.format("%s\tWeights:%s\n", Arrays.toString(Arrays.copyOfRange(neighbors[i], 1, neighbors[i][0] + 1)), Arrays.toString(weights[i])));
      } else {
        sb.append(String.format("null\tWeights:%s\n", Arrays.toString(weights[i])));
      }
    }
    return sb.toString();
  }

  /**
   * This method prints the weight matrix corresponding to the actual state of the object in an Octave-compatible sparse matrix format.
   */
  public String returnWeightMatrix() {
    int[] froms = new int[numOfEdges], tos = new int[numOfEdges];
    double[] ws = new double[numOfEdges];
    for (int n = 0, i = 0; n < this.getNumOfNodes(); ++n) {
      int[] neighs = this.getOutLinks(n);
      double[] weights = this.getWeights(n);
      for (int j = 1; j <= neighs[0]; ++j, ++i) {
        froms[i] = n + 1;
        tos[i] = neighs[j] + 1;
        ws[i] = weights[j];
      }
    }
    return String.format("M=sparse(%s,%s,%s,%d,%d);\n", Arrays.toString(froms), Arrays.toString(tos), Arrays.toString(ws), numOfNodes, numOfNodes);
  }

  /**
   * This method prints the graph to a file in a format that Snap (i.e. the Stanford Network Analysis Project) can directly process.
   * 
   * @param file
   */
  public void toSnapFile(String file) {
    try (PrintWriter out = new PrintWriter(file)) {
      for (int i = 0; i < getNumOfNodes(); ++i) {
        int[] neighs = getOutLinks(i);
        for (int n = 1; n <= neighs[0]; ++n) {
          out.format("%d\t%d\t%s\t%s\n", i, neighs[n], getNodeLabel(i), getNodeLabel(neighs[n]));
        }
      }
    } catch (Exception e) {
      e.printStackTrace();
    }
  }

  public void saveToDot(double[] prs, double[] etalon, String fileName) {
    saveToDot(prs, etalon, fileName, 0, 1);
  }

  /**
   * Prints the actual graph to a format that GraphViz can process.
   * 
   * @param prs
   * @param etalon
   * @param fileName
   */
  public void saveToDot(double[] prs, double[] etalon, String fileName, double lowerWeightLimit, double upperWeightLimit) {
    try (PrintWriter out = new PrintWriter(fileName)) {
      out.println("digraph romeo_and_juliet{\n\toverlap = false;\n\tnode [shape=box,style=filled,width=.3, height=.3];");
      for (int i = 0; i < getNumOfNodes(); ++i) {
        out.format("\t%d [label=\"%s\n(%.1f/%.1f)\"]\n", i, getNodeLabel(i), prs[i] * 100, etalon[i] * 100);
      }
      for (int i = 0; i < getNumOfNodes(); ++i) {
        int[] ns = getOutLinks(i);
        double[] ws = getWeights(i);
        double maxWeight = 0.0d;
        for (int j = 1; j <= ns[0]; ++j) {
          if (ws[j] > maxWeight) {
            maxWeight = ws[j];
          }
        }
        double expected = 1.0d / ns[0];
        for (int j = 1; j <= ns[0]; ++j) {
          if (ws[j] > expected && ws[j] > lowerWeightLimit && ws[j] <= upperWeightLimit) {
            out.format("\t%d->%d [penwidth=%.4f style=\"filled\" gradientanlge=0 color=\"red;0.5:blue\"]\n", i, ns[j], 4 * ws[j] / maxWeight);
          }
        }
      }
      out.println("}");
    } catch (FileNotFoundException e) {
      e.printStackTrace();
    }
  }

  public void printInTikz(PrintStream log) {
    for (int n = 0; n < getNumOfNodes(); ++n) {
      int[] neighs = getOutLinks(n);
      double[] weights = getWeights(n);
      int[] order = Utils.stableSort(weights);
      for (int i = 0; i < order.length; ++i) {
        int o = order[order.length - 1 - i];
        if (o == 0 || weights[o] == 0) {
          continue;
        }
        log.format("%s/%s/%.3f,", getNodeLabel(n), getNodeLabel(neighs[o]), weights[o]);
      }
      log.flush();
    }
  }

  /**
   * Checks for the symmetry of the graph.
   * 
   * @return
   */
  public boolean isSymmetric() {
    for (int i = 0; i < getNumOfNodes(); ++i) {
      int[] neighs = getOutLinks(i);
      for (int n = 1; n <= neighs[0]; ++n) {
        int[] neighs2 = getOutLinks(neighs[n]);
        int index = Arrays.binarySearch(neighs2, 1, neighs2[0] + 1, i);
        if (index < 0) {
          return false;
        }
      }
    }
    return true;
  }

  /**
   * Calculates a breadth-first search.
   * 
   * @param source
   * @return
   */
  public int[][] bfs(int source) {
    return bfs(source, new HashSet<Integer>());
  }

  /**
   * Calculates a breadth-first search limited for the nodes contained in the set interestingNodes. <br />
   * In case the interestingNodes is empty then BFS for all the nodes are calculated.
   * 
   * @param source
   * @param interestingNodes
   * @return
   */
  public int[][] bfs(int source, Set<Integer> interestingNodes) {
    boolean limitedBFS = interestingNodes.size() > 0;
    int[][] distances = new int[2][this.getNumOfNodes()];
    for (int i = 0; i < this.getNumOfNodes(); ++i) {
      distances[0][i] = distances[1][i] = -1;
    }
    distances[0][source] = 0;
    LinkedList<Integer> queue = new LinkedList<>();
    queue.add(source);
    while (queue.size() > 0 && (!limitedBFS || interestingNodes.size() > 0)) {
      int node = queue.removeFirst();
      int[] neighbors = this.getOutLinks(node);
      for (int i = 1; i <= neighbors[0]; ++i) {
        if (distances[0][neighbors[i]] == -1) {
          interestingNodes.remove(neighbors[i]);
          distances[0][neighbors[i]] = distances[0][node] + 1;
          distances[1][neighbors[i]] = node;
          queue.addLast(neighbors[i]);
        }
      }
    }
    return distances;
  }

  public double[][] dijkstra(int source) {
    return dijkstra(source, new HashSet<>());
  }

  /**
   * Returns the shortest paths and distances in the graph assuming edge weights are probabilities. <br />
   * In case the interestingNodes is empty then BFS for all the nodes are calculated.
   * 
   * @param source
   * @return
   */
  public double[][] dijkstra(int source, Set<Integer> interestingNodes) {
    boolean limitedDisjktra = interestingNodes.size() > 0;
    double[][] shortestPaths = new double[2][this.getNumOfNodes()];// hack for returning the shortest path together with the distances simultaneously
    for (int i = 0; i < this.getNumOfNodes(); ++i) {
      shortestPaths[0][i] = Double.MAX_VALUE;
      shortestPaths[1][i] = -1;
    }
    shortestPaths[0][source] = 0;
    PriorityQueue<OwnNode> openNodes = new PriorityQueue<>();
    openNodes.add(new OwnNode(source, 0.d));
    while (openNodes.size() > 0 && (!limitedDisjktra || interestingNodes.size() > 0)) {
      OwnNode on = openNodes.poll();
      interestingNodes.remove(on.nodeId);
      int[] neighbors = this.getOutLinks(on.nodeId);
      double[] neighborWeigths = this.getWeights(on.nodeId);
      for (int i = 1; i <= neighbors[0]; ++i) {
        double distance = -Math.log(neighborWeigths[i]), updatedDistance = 0;
        if ((updatedDistance = shortestPaths[0][on.nodeId] + distance) < shortestPaths[0][neighbors[i]]) {
          shortestPaths[0][neighbors[i]] = updatedDistance;
          shortestPaths[1][neighbors[i]] = on.nodeId;
          openNodes.add(new OwnNode(neighbors[i], updatedDistance));
        }
      }
    }
    for (int i = 0; i < this.getNumOfNodes(); ++i) {
      shortestPaths[0][i] = Math.exp(-shortestPaths[0][i]);
    }
    return shortestPaths;
  }

  class OwnNode implements Comparable<OwnNode> {
    private int nodeId;
    private double value;

    public OwnNode(int n, double v) {
      this.nodeId = n;
      this.value = v;
    }

    @Override
    public int compareTo(OwnNode o) {
      if (this.value != o.value) {
        return this.value > o.value ? 1 : -1;
      } else {
        return this.nodeId > o.nodeId ? 1 : -1;
      }
    }

    @Override
    public boolean equals(Object o) {
      if (o instanceof OwnNode) {
        return ((OwnNode) o).nodeId == this.nodeId;
      }
      return false;
    }

    @Override
    public String toString() {
      return String.format("%f\t%d", this.value, this.nodeId);
    }
  }
}
