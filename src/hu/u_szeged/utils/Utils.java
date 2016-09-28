package hu.u_szeged.utils;

import java.io.BufferedOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;

import org.apache.commons.math3.distribution.GammaDistribution;
import org.apache.commons.math3.random.JDKRandomGenerator;

import hu.u_szeged.graph.OwnGraph;

public class Utils {

  public static final double SMALL = 1e-16;

  public static void serialize(String outFile, Object o) {
    try (ObjectOutputStream out = new ObjectOutputStream(new BufferedOutputStream(new FileOutputStream(outFile)))) {
      out.writeObject(o);
    } catch (IOException io) {
      io.printStackTrace();
    }
  }

  private static int partition(double[] array, int[] index, int l, int r) {
    double pivot = array[index[(l + r) / 2]];
    int help;
    while (l < r) {
      while ((array[index[l]] < pivot) && (l < r)) {
        l++;
      }
      while ((array[index[r]] > pivot) && (l < r)) {
        r--;
      }
      if (l < r) {
        help = index[l];
        index[l] = index[r];
        index[r] = help;
        l++;
        r--;
      }
    }
    if ((l == r) && (array[index[r]] > pivot)) {
      r--;
    }
    return r;
  }

  private static int partition(int[] array, int[] index, int l, int r) {
    double pivot = array[index[(l + r) / 2]];
    int help;
    while (l < r) {
      while ((array[index[l]] < pivot) && (l < r)) {
        l++;
      }
      while ((array[index[r]] > pivot) && (l < r)) {
        r--;
      }
      if (l < r) {
        help = index[l];
        index[l] = index[r];
        index[r] = help;
        l++;
        r--;
      }
    }
    if ((l == r) && (array[index[r]] > pivot)) {
      r--;
    }
    return r;
  }

  public static/* @pure@ */int[] stableSort(int[] array) {
    int[] index = new int[array.length];
    int[] newIndex = new int[array.length];
    int[] helpIndex;
    int numEqual;

    array = (int[]) array.clone();
    for (int i = 0; i < index.length; i++) {
      index[i] = i;
      if (Double.isNaN(array[i])) {
        array[i] = Integer.MAX_VALUE;
      }
    }
    quickSort(array, index, 0, array.length - 1);
    // Make sort stable
    int i = 0;
    while (i < index.length) {
      numEqual = 1;
      for (int j = i + 1; ((j < index.length) && eq(array[index[i]], array[index[j]])); j++)
        numEqual++;
      if (numEqual > 1) {
        helpIndex = new int[numEqual];
        for (int j = 0; j < numEqual; j++)
          helpIndex[j] = i + j;
        quickSort(index, helpIndex, 0, numEqual - 1);
        for (int j = 0; j < numEqual; j++)
          newIndex[i + j] = index[helpIndex[j]];
        i += numEqual;
      } else {
        newIndex[i] = index[i];
        i++;
      }
    }
    return newIndex;
  }

  public static/* @pure@ */int[] stableSort(double[] array) {
    int[] index = new int[array.length];
    int[] newIndex = new int[array.length];
    int[] helpIndex;
    int numEqual;

    array = (double[]) array.clone();
    for (int i = 0; i < index.length; i++) {
      index[i] = i;
      if (Double.isNaN(array[i])) {
        array[i] = Double.MAX_VALUE;
      }
    }
    quickSort(array, index, 0, array.length - 1);
    // Make sort stable
    int i = 0;
    while (i < index.length) {
      numEqual = 1;
      for (int j = i + 1; ((j < index.length) && eq(array[index[i]], array[index[j]])); j++)
        numEqual++;
      if (numEqual > 1) {
        helpIndex = new int[numEqual];
        for (int j = 0; j < numEqual; j++)
          helpIndex[j] = i + j;
        quickSort(index, helpIndex, 0, numEqual - 1);
        for (int j = 0; j < numEqual; j++)
          newIndex[i + j] = index[helpIndex[j]];
        i += numEqual;
      } else {
        newIndex[i] = index[i];
        i++;
      }
    }
    return newIndex;
  }

  private static void quickSort(/* @non_null@ */double[] array, /* @non_null@ */int[] index, int left, int right) {
    if (left < right) {
      int middle = partition(array, index, left, right);
      quickSort(array, index, left, middle);
      quickSort(array, index, middle + 1, right);
    }
  }

  private static void quickSort(/* @non_null@ */int[] array, /* @non_null@ */int[] index, int left, int right) {
    if (left < right) {
      int middle = partition(array, index, left, right);
      quickSort(array, index, left, middle);
      quickSort(array, index, middle + 1, right);
    }
  }

  public static/* @pure@ */boolean grOrEq(double a, double b) {
    return (b - a < SMALL);
  }

  public static/* @pure@ */boolean eq(double a, double b) {
    return (a - b < SMALL) && (b - a < SMALL);
  }

  /**
   * In case the length of the priors is 1, it is assumed that all the other priors have the same value.
   * 
   * @param l
   * @param priors
   * @return
   */
  public static double[] drawMultinomial(int l, double... priors) {
    double[] multinomial = new double[l + 1];
    double sum = 0.0d, cummulatedProb = 0.0d;
    for (int i = 1; i <= l; ++i) {
      GammaDistribution gd = new GammaDistribution((JDKRandomGenerator) OwnGraph.RANDOM, priors.length == 1 ? priors[0] : priors[i - 1], 1.0d);
      multinomial[i] = gd.sample();
      sum += multinomial[i];
    }
    for (int i = 1; i <= l; ++i) {
      multinomial[i] /= sum;
      cummulatedProb += multinomial[i];
    }
    multinomial[0] = cummulatedProb;
    return multinomial;
  }
}