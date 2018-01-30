package hu.u_szeged.utils;

import java.io.BufferedOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.AbstractMap;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentMap;
import java.util.stream.Collector;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.apache.commons.math3.distribution.GammaDistribution;
import org.apache.commons.math3.random.JDKRandomGenerator;

import hu.u_szeged.experiments.SyntheticExperiment;

public class Utils {

  public static final double SMALL = 1e-16;

  /**
   * Returns the elements present in both of the argument arrays (which are assumed to be ordered).
   * 
   * @param a
   * @param b
   * @return
   */
  public static List<Integer> determineOverlappingValues(int[] a, int[] b) {
    List<Integer> overlap = new LinkedList<>();
    for (int ii = 1, oi = 1; ii <= a[0] && oi <= b[0]; ++ii, ++oi) {
      if (a[ii] == b[oi]) {
        overlap.add(a[ii]);
      } else if (a[ii] > b[oi]) {
        ii--;
      } else {
        oi--;
      }
    }
    return overlap;
  }

  /**
   * Returns the number of elements present in both of the argument arrays (which are assumed to be ordered).
   * 
   * @param a
   * @param b
   * @return
   */
  public static int determineOverlap(int[] a, int[] b) {
    return determineOverlappingValues(a, b).size();
  }

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
      GammaDistribution gd = new GammaDistribution((JDKRandomGenerator) SyntheticExperiment.RANDOM, priors.length == 1 ? priors[0] : priors[i - 1], 1.0d);
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

  public static <K, V> Map.Entry<K, V> entry(K key, V value) {
    return new AbstractMap.SimpleEntry<>(key, value);
  }

  public static <K, U> Collector<Map.Entry<K, U>, ?, Map<K, U>> entriesToMap() {
    return Collectors.toMap((e) -> e.getKey(), (e) -> e.getValue());
  }

  public static <K, U> Collector<Map.Entry<K, U>, ?, ConcurrentMap<K, U>> entriesToConcurrentMap() {
    return Collectors.toConcurrentMap((e) -> e.getKey(), (e) -> e.getValue());
  }

  public static Map<String, String> createUnmodifiableMap(String[] k, String[] v) {
    Map.Entry<String, String>[] elements = new Map.Entry[k.length];
    for (int i = 0; i < k.length; ++i) {
      elements[i] = Utils.entry(k[i], v[i]);
    }
    return Collections.unmodifiableMap(Stream.of(elements).collect(Utils.entriesToMap()));
  }

  public static String listToString(List<String> q) {
    StringBuffer sb = new StringBuffer();
    for (String s : q) {
      sb.append(s + " ");
    }
    return sb.toString().trim();
  }

  /**
   * Turns (unnormalized) edge scores into a probability distribution. <br />
   * The scores array is modified in place without allocating a new array.
   * 
   * @param scores
   * @param length
   * @return
   */
  public static void softmaxNormalize(double[] scores, int length) {
    double maxExponent = Double.NEGATIVE_INFINITY;
    for (int i = 1; i <= length; ++i) {
      double d = scores[i];
      if (d > maxExponent) {
        maxExponent = d;
      }
    }
    double partition = 0.0d;
    for (int i = 1; i <= length; ++i) {
      partition += Math.exp(scores[i] - maxExponent);
    }
    for (int i = 1; i <= length; ++i) {
      scores[i] = Math.exp(scores[i] - maxExponent) / partition;
    }
    scores[0] = partition;
  }

  public static void softmaxDenormalize(double[] scores, int length) {
    if (length > 0) {
      double partition = scores[0];
      for (int i = 1; i <= length; ++i) {
        scores[i] = Math.log(scores[i] * partition);
      }
      double sum = 0, s = scores[1];
      for (int i = 1; i <= length; ++i) {
        sum += (scores[i] -= s);
      }
      scores[0] = sum;
    }
  }
}