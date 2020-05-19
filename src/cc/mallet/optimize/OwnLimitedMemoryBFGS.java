package cc.mallet.optimize;

import java.util.Arrays;
import java.util.LinkedList;

import org.graphstream.graph.Edge;
import org.graphstream.graph.Graph;
import org.graphstream.graph.Node;
import org.graphstream.ui.view.Viewer;

import cc.mallet.optimize.Optimizable.ByGradientValue;
import cc.mallet.types.MatrixOps;
import hu.u_szeged.graph.OwnGraph;
import hu.u_szeged.graph.PRWeightLearner;
import hu.u_szeged.graph.visualize.VisualizePageRankLearn;
import hu.u_szeged.utils.Utils;

public class OwnLimitedMemoryBFGS extends LimitedMemoryBFGS {

  private double tolerance = .0001;
  private Graph graph;
  private Viewer v;
  private LineOptimizer.ByGradient lineMaximizer;
  private OptimizerEvaluator.ByGradient eval = null;
  private boolean debug = false;

  public OwnLimitedMemoryBFGS(ByGradientValue function) {
    super(function);
    lineMaximizer = new BackTrackLineSearch(function);
  }

  public void addGraph(Graph g) {
    graph = g;
    v = graph.display();
    updateGraph();
  }

  @SuppressWarnings("unchecked")
  public boolean optimize(int numIterations) {
    // double initialValue = ((Optimizable.ByGradientValue) super.getOptimizable()).getValue();
    if (g == null) { // first time through
      if (debug) {
        System.err.println("First time through L-BFGS");
      }
      iterations = 0;
      s = new LinkedList<Object>();
      y = new LinkedList<Object>();
      rho = new LinkedList<Double>();
      alpha = new double[m];

      for (int i = 0; i < m; i++) {
        alpha[i] = 0.0;
      }

      parameters = new double[optimizable.getNumParameters()];
      oldParameters = new double[optimizable.getNumParameters()];
      g = new double[optimizable.getNumParameters()];
      oldg = new double[optimizable.getNumParameters()];
      direction = new double[optimizable.getNumParameters()];

      optimizable.getParameters(parameters);
      System.arraycopy(parameters, 0, oldParameters, 0, parameters.length);

      optimizable.getValueGradient(g);
      System.arraycopy(g, 0, oldg, 0, g.length);
      System.arraycopy(g, 0, direction, 0, g.length);

      if (MatrixOps.absNormalize(direction) == 0) {
        if (debug) {
          System.err.println("L-BFGS initial gradient is zero; saying converged");
        }
        g = null;
        converged = true;
        return true;
      }
      if (debug) {
        System.err.println("direction.2norm: " + MatrixOps.twoNorm(direction));
      }
      MatrixOps.timesEquals(direction, 1.0 / MatrixOps.twoNorm(direction));

      // make initial jump
      if (debug) {
        System.err.println("before initial jump: \ndirection.2norm: " + MatrixOps.twoNorm(direction) + " \ngradient.2norm: " + MatrixOps.twoNorm(g)
            + "\nparameters.2norm: " + MatrixOps.twoNorm(parameters));
      }

      // TestMaximizable.testValueAndGradientInDirection (maxable, direction);
      step = lineMaximizer.optimize(direction, step);
      if (step == 0.0) {
        // could not step in this direction. give up and say converged.
        g = null; // reset search
        step = 1.0;
        throw new OptimizationException("Line search could not step in the current direction. "
            + "(This is not necessarily cause for alarm. Sometimes this happens close to the maximum," + " where the function may be very flat.)");
        // return false;
      }

      optimizable.getParameters(parameters);
      optimizable.getValueGradient(g);
      if (debug) {
        System.err.println("after initial jump: \ndirection.2norm: " + MatrixOps.twoNorm(direction) + " \ngradient.2norm: " + MatrixOps.twoNorm(g));
      }
    }

    for (int iterationCount = 0; iterationCount < numIterations; iterationCount++) {
      double value = optimizable.getValue();
      if (debug) {
        System.err.format("L-BFGS iteration=%d, value=%.8f g.twoNorm: %.8f oldg.twoNorm: %.8f\n", iterationCount, value, MatrixOps.twoNorm(g),
            MatrixOps.twoNorm(oldg));
      }

      // get difference between previous 2 gradients and parameters
      double sy = 0.0, yy = 0.0;

      for (int i = 0; i < oldParameters.length; i++) {
        // -inf - (-inf) = 0; inf - inf = 0
        if (Double.isInfinite(parameters[i]) && Double.isInfinite(oldParameters[i]) && (parameters[i] * oldParameters[i] > 0)) {
          oldParameters[i] = 0.0;
        } else {
          oldParameters[i] = parameters[i] - oldParameters[i];
        }

        if (Double.isInfinite(g[i]) && Double.isInfinite(oldg[i]) && (g[i] * oldg[i] > 0)) {
          oldg[i] = 0.0;
        } else {
          oldg[i] = g[i] - oldg[i];
        }

        sy += oldParameters[i] * oldg[i]; // si * yi
        yy += oldg[i] * oldg[i];
        direction[i] = g[i];
      }

      if (sy > 0) {
        throw new InvalidOptimizableException("sy = " + sy + " > 0");
      }

      double gamma = sy / yy; // scaling factor

      if (gamma > 0) {
        throw new InvalidOptimizableException("gamma = " + gamma + " > 0");
      }

      push(rho, 1.0 / sy);
      // These arrays are now the *differences* between parameters and gradient.
      push(s, oldParameters);
      push(y, oldg);

      assert (s.size() == y.size()) : "s.size: " + s.size() + " y.size: " + y.size();

      // This next section is where we calculate the new direction
      // First work backwards, from the most recent difference vectors
      for (int i = s.size() - 1; i >= 0; i--) {
        alpha[i] = ((Double) rho.get(i)).doubleValue() * MatrixOps.dotProduct((double[]) s.get(i), direction);
        MatrixOps.plusEquals(direction, (double[]) y.get(i), -1.0 * alpha[i]);
      }

      MatrixOps.timesEquals(direction, gamma); // Scale the direction by the ratio of s'y and y'y

      // Now work forwards, from the oldest to the newest difference vectors
      for (int i = 0; i < y.size(); i++) {
        double beta = (((Double) rho.get(i)).doubleValue()) * MatrixOps.dotProduct((double[]) y.get(i), direction);
        MatrixOps.plusEquals(direction, (double[]) s.get(i), alpha[i] - beta);
      }

      // Move the current values to the "last iteration" buffers and negate the search direction
      for (int i = 0; i < oldg.length; i++) {
        oldParameters[i] = parameters[i];
        oldg[i] = g[i];
        direction[i] *= -1.0;
      }

      if (debug) {
        System.err.format("before linesearch: direction.gradient.dotprod: %.8f\ndirection.2norm: %.8f\nparameters.2norm: %.8f",
            MatrixOps.dotProduct(direction, g), MatrixOps.twoNorm(direction), MatrixOps.twoNorm(parameters));
      }

      // Test whether the gradient is ok
      // TestMaximizable.testValueAndGradientInDirection (maxable, direction);

      // Do a line search in the current direction
      step = lineMaximizer.optimize(direction, step);

      if (step == 0.0) { // could not step in this direction.
        g = null; // reset search
        step = 1.0;
        // xxx Temporary test; passed OK
        // TestMaximizable.testValueAndGradientInDirection (maxable, direction);
        throw new OptimizationException("Line search could not step in the current direction. "
            + "(This is not necessarily cause for alarm. Sometimes this happens close to the maximum, where the function may be very flat.)");
        // return false;
      }
      optimizable.getParameters(parameters);
      optimizable.getValueGradient(g);
      if (debug) {
        System.err.println("after linesearch: direction.2norm: " + MatrixOps.twoNorm(direction));
      }
      double newValue = optimizable.getValue();
      updateGraph();
      if (graph != null) {
        System.err.format("Obj. val:\t%f->%f\n", -value, -newValue);
      }

      // Test for terminations
      if (2.0 * Math.abs(newValue - value) <= tolerance * (Math.abs(newValue) + Math.abs(value) + eps)) {
        if (debug) {
          System.err.println("Exiting L-BFGS on termination #1:\nvalue difference below tolerance (oldValue: " + value + " newValue: " + newValue);
        }
        converged = true;
        return true;
      }
      double gg = MatrixOps.twoNorm(g);
      if (gg < gradientTolerance) {
        if (debug) {
          System.err.println("Exiting L-BFGS on termination #2: \ngradient=" + gg + " < " + gradientTolerance);
        }
        converged = true;
        return true;
      }
      if (gg == 0.0) {
        if (debug) {
          System.err.println("Exiting L-BFGS on termination #3: \ngradient==0.0");
        }
        converged = true;
        return true;
      }
      if (debug) {
        System.err.println("Gradient = " + gg);
      }
      iterations++;
      if (iterations > maxIterations) {
        if (debug) {
          System.err.println("Too many iterations in L-BFGS.java. Continuing with current parameters.");
        }
        converged = true;
        return true;
        // throw new IllegalStateException ("Too many iterations.");
      }

      // end of iteration. call evaluator
      if (eval != null && !eval.evaluate(optimizable, iterationCount)) {
        if (debug) {
          System.err.println("Exiting L-BFGS on termination #4: evaluator returned false.");
        }
        converged = true;
        return false;
      }
    }
    return false;
  }

  /**
   * Pushes a new object onto the queue l
   * 
   * @param l
   *          linked list queue of Matrix obj's
   * @param toadd
   *          matrix to push onto queue
   */
  private void push(LinkedList<Object> l, double[] toadd) {
    assert (l.size() <= m);
    if (l.size() == m) {
      // remove oldest matrix and add newest to end of list.
      // to make this more efficient, actually overwrite
      // memory of oldest matrix

      // this overwrites the oldest matrix
      double[] last = (double[]) l.get(0);
      System.arraycopy(toadd, 0, last, 0, toadd.length);
      Object ptr = last;
      // this readjusts the pointers in the list
      for (int i = 0; i < l.size() - 1; i++) {
        l.set(i, (double[]) l.get(i + 1));
      }
      l.set(m - 1, ptr);
    } else {
      double[] newArray = new double[toadd.length];
      System.arraycopy(toadd, 0, newArray, 0, toadd.length);
      l.addLast(newArray);
    }
  }

  /**
   * Pushes a new object onto the queue l
   * 
   * @param l
   *          linked list queue of Double obj's
   * @param toadd
   *          double value to push onto queue
   */
  private void push(LinkedList<Double> l, double toadd) {
    assert (l.size() <= m);
    if (l.size() == m) { // pop old double and add new
      l.removeFirst();
      l.addLast(new Double(toadd));
    } else {
      l.addLast(new Double(toadd));
    }
  }

  private void updateGraph() {
    if (graph != null) {
      if (iterations == 0) {
        try {
          Thread.sleep(2000);
        } catch (InterruptedException e1) {
          e1.printStackTrace();
        }
      }
      updateEdges();
      try {
        Thread.sleep(500);
      } catch (InterruptedException e1) {
        e1.printStackTrace();
      }
      updateNodes();
    }
  }

  private void updateEdges() {
    OwnGraph og = ((PRWeightLearner) optimizable).getGraph();
    for (int i = 0, edgeIndex = 0; i < og.getNumOfNodes(); ++i) {
      int[] neighbors = og.getOutLinks(i);
      double[] weights = Arrays.copyOf(og.getWeights(i), og.getOutDegree(i) + 1);
      Utils.softmaxNormalize(weights, neighbors[0]);
      for (int n = 1; n <= neighbors[0]; ++n, ++edgeIndex) {
        // System.err.format("Edge %d weight set to %f\n", edgeIndex, VisualizePageRankLearn.UNIT_EDGE_WIDTH * weights[n]);
        boolean below = weights[n] * neighbors[0] < 1;
        Edge e = graph.getEdge(edgeIndex);
        if (weights[n] * neighbors[0] == 1.0d) {
        } else if (below) {
          e.setAttribute("ui.class", "below");
          e.setAttribute("ui.color", weights[n] * neighbors[0]);
        } else {
          e.setAttribute("ui.class", "larger");
          e.setAttribute("ui.color", 1.0d / (weights[n] * neighbors[0]));
        }
        double relativeWeight = Math.pow(1.2d, Math.abs(1 - weights[n] * neighbors[0]));
        e.setAttribute("ui.size", VisualizePageRankLearn.UNIT_EDGE_WIDTH * relativeWeight);
      }
    }
  }

  private void updateNodes() {
    for (int nodeId = 0; nodeId < graph.getNodeCount(); ++nodeId) {
      Node n = graph.getNode(nodeId);
      double actualPrOfNode = ((PRWeightLearner) optimizable).getActualPRvalue(nodeId);
      double etalonPrOfNode = ((PRWeightLearner) optimizable).getEtalonPRvalue(nodeId);
      if (actualPrOfNode > etalonPrOfNode) {
        n.setAttribute("ui.class", "big");
        n.setAttribute("ui.color", etalonPrOfNode / actualPrOfNode);
      } else {
        n.setAttribute("ui.class", "small");
        n.setAttribute("ui.color", actualPrOfNode / etalonPrOfNode);
      }
      n.setAttribute("ui.size", VisualizePageRankLearn.UNIT_NODE_SIZE * actualPrOfNode);
    }
  }
}