package hu.u_szeged.experiments;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.ObjectInputStream;
import java.io.PrintWriter;
import java.io.Reader;
import java.net.URLDecoder;
import java.util.Arrays;
import java.util.Date;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Scanner;
import java.util.Set;
import java.util.zip.GZIPInputStream;

import hu.u_szeged.graph.OwnGraph;
import hu.u_szeged.graph.SoftmaxPRWeightLearner;
import hu.u_szeged.utils.Utils;

public class WikipediaExperiment extends AbstractExperiment {

  private String dir, lang, date;
  private Scanner scanner;

  public WikipediaExperiment(String directory, String language, String d) {
    this(directory, language, d, true);
  }

  /**
   * readEtalonNodeWeights indicates whether the node weights need to be initialized from log files.
   * 
   * @param directory
   * @param language
   * @param d
   * @param readEtalonWeights
   */
  public WikipediaExperiment(String directory, String language, String d, boolean readEtalonNodeWeights) {
    date = d;
    dir = directory;
    lang = language;
    init(readEtalonNodeWeights);
  }

  /**
   * If readEtalonNodeWeights is false it means that we only want to know the topology of the network and not the individual importance of nodes.
   * 
   * @param readEtalonNodeWeights
   */
  public void init(boolean readEtalonNodeWeights) {
    File serializedGraphFile = new File(String.format("%s/%s%sGraph.ser", dir, lang, date));
    File serializedEtalonsFile = readEtalonNodeWeights ? new File(String.format("%s/%s%sEtalon.ser", dir, lang, date)) : null;
    readSerializedGraph(serializedGraphFile, serializedEtalonsFile);
    if (g == null || (readEtalonNodeWeights && etalonDistr == null)) {
      readNodes(String.format("%s/%swiki-%s-page.sql.gz", dir, lang, date));
      readEdges(String.format("%s/%swiki-%s-pagelinks.sql.gz", dir, lang, date));
      calculateEtalonWeights();
      setGraphLabels();
      g.finalizeGraph();
      Utils.serialize(serializedGraphFile.getAbsolutePath(), g);
      if (readEtalonNodeWeights) {
        Utils.serialize(serializedEtalonsFile.getAbsolutePath(), etalonDistr);
      }
    }
    // try (PrintWriter out = new PrintWriter("wikipedia_edge_list.tsv")) {
    // for (int i = 0; i < g.getNumOfNodes(); ++i) {
    // String article = g.getNodeLabel(i);
    // int neighs[] = g.getOutLinks(i);
    // for (int n = 1; n <= neighs[0]; ++n) {
    // out.format("%s\t%s\t%.8E\n", article, g.getNodeLabel(neighs[n]), etalonDistr[neighs[n]]);
    // if (i % 25000 == 0)
    // System.err.format("%d\t%s\t%s\t%.8E\n", i, article, g.getNodeLabel(neighs[n]), etalonDistr[neighs[n]]);
    // }
    // }
    // } catch (IOException e) {
    // e.printStackTrace();
    // }
    learner = new SoftmaxPRWeightLearner(etalonDistr, g, 0.01d);
    System.err.format("There are %d, %d nodes and links in the %s/%s%s graph, respectively.\n", g.getNumOfNodes(), g.getNumOfEdges(), dir, lang, date);
    scanner = new Scanner(System.in);
  }

  private void setExtensiveSerialization(boolean es) {
    if (es) {
      learner.setExtensiveSerialization(String.format("%s/%s%s%s", dir, lang, date, "Weights"));
    }
  }

  private void readSerializedGraph(File serializedGraph, File serializedEtalons) {
    try (ObjectInputStream objectinputstream = new ObjectInputStream(new FileInputStream(serializedGraph))) {
      g = (OwnGraph) objectinputstream.readObject();
      if (serializedEtalons != null) {
        ObjectInputStream ois = new ObjectInputStream(new FileInputStream(serializedEtalons));
        etalonDistr = (double[]) ois.readObject();
        ois.close();
      }
    } catch (ClassNotFoundException | IOException e) {
      System.err.println("The graph has to be built from scratch.");
    }
  }

  /**
   * In the gzipped input files 'The second column is the title of the page retrieved, the third column is the number of requests, and the fourth
   * column is the size of the content returned.' <br />
   * See <a href="http://dumps.wikimedia.org/other/pagecounts-raw/">http://dumps. wikimedia.org/other/pagecounts-raw/</a> for more details.
   * 
   */
  private void calculateEtalonWeights() {
    etalonDistr = new double[idsToLabels.size()];
    Set<Integer> uniqueRequestedPages = new HashSet<>();
    int counter = 0;
    long totalRequests = 0;
    for (File f : new File(dir).listFiles()) {
      if (!f.getName().startsWith("pagecounts-") || !f.getName().contains(this.date.substring(0, 6))) {
        continue;
      }
      try (BufferedReader br = new BufferedReader(new InputStreamReader(new GZIPInputStream(new FileInputStream(f))))) {
        double maxFrequency = 0.0;
        String line, mostPopularSite = "";
        while ((line = br.readLine()) != null) {
          String[] parts = line.split(" ");
          if (parts.length == 4 && parts[0].equals(lang)) {
            try {
              String articleName = URLDecoder.decode(parts[1], "UTF-8");
              Integer wikiId = labelsToIds.get(articleName);
              if (wikiId != null) {
                uniqueRequestedPages.add(wikiId);
                int freq = Integer.parseInt(parts[2]);
                etalonDistr[wikiId] += freq;
                if (etalonDistr[wikiId] > maxFrequency) {
                  maxFrequency = etalonDistr[wikiId];
                  mostPopularSite = articleName;
                }
                totalRequests += freq;
              }
            } catch (IllegalArgumentException e) {
              System.err.format("WARNING: %s was undecodable.\n", parts[1]);
              continue;
            }
          }
        }
        System.err.format("File '%s' processed\t%d\t%d\t%d unique and total requests.\n", f.getName(), ++counter, uniqueRequestedPages.size(), totalRequests);
        System.err.format("Most popular aticle so far:\t%s\t%f\n", mostPopularSite, maxFrequency);
      } catch (IOException e) {
        e.printStackTrace();
      }
    }
    for (int i = 0; i < etalonDistr.length; ++i) {
      etalonDistr[i] = (etalonDistr[i] + 1) / (double) (totalRequests + etalonDistr.length); // Laplace smoothing
    }
  }

  /**
   * The original Wikipedia ids have to be ordered and made continuous so that they can function as node ids.
   * 
   * @return
   */
  private Map<Integer, Integer> reorderIds() {
    g = new OwnGraph(idsToLabels.size());
    int i = 0;
    int[] ids = new int[idsToLabels.size()];
    for (Integer id : idsToLabels.keySet()) {
      ids[i++] = id;
    }
    Map<Integer, String> orderedIdsToLabels = new HashMap<>();
    Map<Integer, Integer> mapping = new HashMap<>();
    int[] sortedOrder = Utils.stableSort(ids);
    for (int j = 0; j < sortedOrder.length; ++j) {
      mapping.put(ids[sortedOrder[j]], j);
      String label = idsToLabels.get(ids[sortedOrder[j]]);
      orderedIdsToLabels.put(j, label);
      labelsToIds.put(label, j);
      g.setNodeLabel(label, j);
    }
    idsToLabels = orderedIdsToLabels;
    return mapping;
  }

  protected void readNodes(String fileToProcess) {
    System.out.println(fileToProcess);
    boolean insertOn = false, stringFieldIsOn = false;
    try (Reader ir = new InputStreamReader(new GZIPInputStream(new FileInputStream(fileToProcess)))) {
      StringBuffer s = new StringBuffer();
      String[] fields = new String[15];
      char nextChar, prevChar = 0;
      int fieldId = 0, ic, counter = 0;
      while ((ic = ir.read()) != -1) {
        nextChar = (char) ic;
        if (!insertOn) {
          s.append(nextChar);
        }
        if (s.toString().endsWith("INSERT INTO `page` VALUES ")) {
          insertOn = true;
          s = new StringBuffer();
        } else if (insertOn) {
          if (!stringFieldIsOn && nextChar == ',') {
            if (prevChar != ')') {
              fields[fieldId++] = s.toString();
              s = new StringBuffer();
            }
          } else if (stringFieldIsOn && prevChar != '\\' && nextChar == '\'') {
            stringFieldIsOn = false;
          } else if (!stringFieldIsOn && nextChar == '\'') {
            stringFieldIsOn = true;
          } else if (!stringFieldIsOn && nextChar == ')') {
            fields[fieldId] = s.toString();
            if (fields[1].equals("0")) {
              int wikiId = Integer.parseInt(fields[0].substring(fields[0].charAt(0) == '(' ? 1 : 0));
              idsToLabels.put(wikiId, fields[2]);
              Integer prevVal = labelsToIds.get(fields[2]);
              if (prevVal == null) {
                labelsToIds.put(fields[2], wikiId);
              } else {
                System.err.println("Duplicate key problem\t" + prevVal + "\t" + wikiId + "\t" + fields[2]);
              }
              if (++counter % 500_000 == 0) {
                System.err.println(counter + "\t" + wikiId + "\t" + fields[2]);
              }
            }
            s = new StringBuffer();
            fields = new String[fields.length];
            fieldId = 0;
          } else if (nextChar != '\\' || (stringFieldIsOn && prevChar == '\\' && nextChar == '\\')) {
            s.append(nextChar);
          }
        }
        prevChar = stringFieldIsOn && prevChar == '\\' && nextChar == '\\' ? 0 : nextChar;
      }
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  protected void readEdges(String fileToProcess) {
    boolean insertOn = false, stringFieldIsOn = false;
    Map<Integer, Integer> wikiIdsToGraphIds = reorderIds();
    try (Reader ir = new InputStreamReader(new GZIPInputStream(new FileInputStream(fileToProcess)))) {
      int[] pageInLinks = new int[idsToLabels.size()];
      StringBuffer s = new StringBuffer();
      String[] fields = new String[4];
      char nextChar, prevChar = 0;
      int fieldId = 0, numOfLinks = 0, ic;
      while ((ic = ir.read()) != -1) {
        nextChar = (char) ic;
        if (!insertOn) {
          s.append(nextChar);
        }
        if (s.toString().endsWith("INSERT INTO `pagelinks` VALUES ")) {
          insertOn = true;
          s = new StringBuffer();
        } else if (insertOn) {
          if (!stringFieldIsOn && nextChar == ',') {
            if (prevChar != ')') {
              fields[fieldId++] = s.toString();
              s = new StringBuffer();
            }
          } else if (stringFieldIsOn && prevChar != '\\' && nextChar == '\'') {
            stringFieldIsOn = false;
          } else if (!stringFieldIsOn && nextChar == '\'') {
            stringFieldIsOn = true;
          } else if (!stringFieldIsOn && nextChar == ')') {
            fields[fieldId] = s.toString();
            Integer fromId = Integer.parseInt(fields[0].substring(fields[0].charAt(0) == '(' ? 1 : 0));
            Integer toId = labelsToIds.get(fields[2]);
            if (toId != null && (fromId = wikiIdsToGraphIds.get(fromId)) != null && fields[1].equals("0") && fields[3].equals("0")) {
              ++numOfLinks;
              pageInLinks[toId]++;
              g.addEdge(fromId, toId);
              if (numOfLinks % 500_000 == 0) {
                System.err.format("%d\t%d-->%d\t%s-->%s\n", numOfLinks, fromId, toId, idsToLabels.get(fromId), fields[2]);
              }
            }
            s = new StringBuffer();
            fields = new String[fields.length];
            fieldId = 0;
          } else if (nextChar != '\\' || (stringFieldIsOn && prevChar == '\\' && nextChar == '\\')) {
            s.append(nextChar);
          }
        }
        prevChar = stringFieldIsOn && prevChar == '\\' && nextChar == '\\' ? 0 : nextChar;
      }
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  protected void learnWeights(int initializations, boolean averageModels) {
    String modelLocation = String.format("%s/%s%s%sFinalWeights.ser", dir, lang, date, learner.regularizationToString());
    System.err.println(String.format("Model creation for path %s started at %s", modelLocation, new Date()));
    learner.learnEdgeWeights(initializations, false, averageModels);
    learner.serializeWeights(modelLocation);
    System.err.println("Baseline KL-divergence: " + learner.getBaselineValue());
  }

  private void printBaselineLine(PrintWriter writer, double[] values, int i) {
    int[] neighbors = g.getOutLinks(i);
    double[] weights = g.getWeights(i);
    writer.format("%d\t%s\t%d\n", i, g.getNodeLabel(i), g.getNumOfNeighbors(i));
    int[] ranking = Utils.stableSort(values);
    for (int j = 0; j < ranking.length; ++j) {
      int r = ranking[ranking.length - 1 - j] + 1; // +1 is for skipping the 0th
                                                   // meta-elements
      writer.format("\t->%s\t%f\t%f\t%d\n", g.getNodeLabel(neighbors[r]), weights[r], values[r - 1], neighbors[r]);
    }
  }

  private void loadChoicerankWeights(String gsPath) {
    try (BufferedReader br = new BufferedReader(new InputStreamReader(new GZIPInputStream(new FileInputStream(gsPath))))) {
      String line;
      while ((line = br.readLine()) != null) {
        String parts[] = line.split("\t");
        Integer from = g.getNodeIdByLabel(parts[0]);
        Integer to = g.getNodeIdByLabel(parts[1]);
        double weight = Double.parseDouble(parts[2]);
        g.setEdgeWeight(from, to, weight);
      }
    } catch (IOException e) {
      e.printStackTrace();
    }

  }

  /**
   * Sets graph weights to the Jaccard similarities between nodes.
   */
  private void determineJaccardWeights() {
    double[] newWeights = new double[g.getNumOfEdges()];
    for (int n = 0, k = 0; n < g.getNumOfNodes(); ++n) {
      double[] jaccardWeights = calculateJaccardScores(n);
      for (int i = 0; i < jaccardWeights.length; ++i, ++k) {
        newWeights[k] += jaccardWeights[i];
      }
    }
    g.setWeights(newWeights);
    g.normalizeWeights();
  }

  private double[] normalize(double[] d) {
    double sum = 0.0d;
    for (double v : d) {
      sum += v;
    }
    double[] normalized = new double[d.length];
    for (int i = 0; i < d.length; ++i) {
      normalized[i] = d[i] / sum;
    }
    return normalized;
  }

  private double[] calculateJaccardScores(int target) {
    int[] neighs = g.getOutLinks(target);
    double[] jaccards = new double[neighs[0]];
    for (int j = 1; j <= neighs[0]; ++j) {
      int[] outLinksForNeighbor = g.getOutLinks(neighs[j]);
      jaccards[j - 1] = Utils.determineOverlap(neighs, outLinksForNeighbor);
      jaccards[j - 1] /= (double) (neighs[0] + outLinksForNeighbor[0] - jaccards[j - 1]);
    }
    return normalize(jaccards);
  }

  private double[] calculatePageRankScores(int target) {
    int[] neighs = g.getOutLinks(target);
    double[] pageranks = new double[neighs[0]];
    for (int j = 1; j <= neighs[0]; ++j) {
      pageranks[j - 1] = learner.getInitialPRvalue(neighs[j]);
    }
    return normalize(pageranks);
  }

  private double[] calculatePopularityScores(int target) {
    int[] neighs = g.getOutLinks(target);
    double[] relativeClicks = new double[neighs[0]];
    for (int j = 1; j <= neighs[0]; ++j) {
      relativeClicks[j - 1] = etalonDistr[neighs[j]];
    }
    return normalize(relativeClicks);
  }

  private double[] calculatePMIScores(int target) {
    int[] neighs = g.getOutLinks(target);
    int[] inLinks = g.getInLinks(target);
    double[] pmis = new double[neighs[0]];
    for (int j = 1; j <= neighs[0]; ++j) {
      int[] inLinksForNeighbor = g.getInLinks(neighs[j]);
      int inLinksOverlap = Utils.determineOverlap(inLinks, inLinksForNeighbor);
      double pmiScore = Double.NEGATIVE_INFINITY;
      if (inLinksOverlap > 0) {
        pmiScore = Math.log((double) ((long) inLinksOverlap * g.getNumOfNodes()) / ((long) inLinks[0] * inLinksForNeighbor[0]));
        if (Double.isNaN(pmiScore)) {
          System.err.format("ERROR: PMI('%s','%s')=%f\n", g.getNodeLabel(target), g.getNodeLabel(neighs[j]), pmiScore);
        }
      }
      pmis[j - 1] = pmiScore;
    }
    return pmis;
  }

  private double[] calculateIndegreeScores(int target) {
    int[] neighs = g.getOutLinks(target);
    double[] indegrees = new double[neighs[0]];
    for (int j = 1; j <= neighs[0]; ++j) {
      indegrees[j - 1] = g.getIndegree(neighs[j]);
    }
    return normalize(indegrees);
  }

  public void runBaselines() {
    learner.setLogFile(String.format("%s/%s%sBaseline", dir, lang, date));
    System.err.println("Baseline value: " + learner.getBaselineValue());
    try (PrintWriter jaccardOut = new PrintWriter(String.format("%s/%s%sJaccardBaseline", dir, lang, date))) {
      PrintWriter indegreeOut = new PrintWriter(String.format("%s/%s%sWikiIndegreeBaseline", dir, lang, date));
      PrintWriter clickFreqOut = new PrintWriter(String.format("%s/%s%sWikiClicksBaseline", dir, lang, date));
      PrintWriter pmiOut = new PrintWriter(String.format("%s/%s%sWikiPMIBaseline", dir, lang, date));
      PrintWriter pageRankOut = new PrintWriter(String.format("%s/%s%sWikiPageRankBaseline", dir, lang, date));
      for (int i = 0; i < g.getNumOfNodes(); ++i) {
        printBaselineLine(jaccardOut, calculateJaccardScores(i), i);
        printBaselineLine(pageRankOut, calculatePageRankScores(i), i);
        printBaselineLine(clickFreqOut, calculatePopularityScores(i), i);
        printBaselineLine(pmiOut, calculatePMIScores(i), i);
        printBaselineLine(indegreeOut, calculateIndegreeScores(i), i);
      }
      indegreeOut.close();
      clickFreqOut.close();
      pageRankOut.close();
      pmiOut.close();
    } catch (FileNotFoundException e) {
      e.printStackTrace();
    }
  }

  private String[] getMostSimilars(String testCase, String approach) {
    return getMostSimilars(testCase, approach, Integer.MAX_VALUE);
  }

  private String[] getMostSimilars(String testCase, String approach, int k) {
    Integer nodeId = g.getNodeIdByLabel(testCase);
    if (nodeId != null) {
      int[] neighs = g.getOutLinks(nodeId);
      if (approach.contains("learning")) {
        double[] ws = g.getWeights(nodeId);
        double[] transformedWeights = new double[neighs[0]];
        for (int i = 1; i <= neighs[0]; ++i) {
          transformedWeights[i - 1] = ws[i];
        }
        return rankSimilarities(neighs, transformedWeights, k, null);
      } else if (approach.equals("jaccard")) {
        return rankSimilarities(neighs, calculateJaccardScores(nodeId), k, null);
      } else if (approach.equals("pagerank")) {
        return rankSimilarities(neighs, calculatePageRankScores(nodeId), k, null);
      } else if (approach.equals("popularity")) {
        return rankSimilarities(neighs, calculatePopularityScores(nodeId), k, null);
      } else if (approach.equals("pmi")) {
        return rankSimilarities(neighs, calculatePMIScores(nodeId), k, null);
      } else if (approach.equals("indegree")) {
        return rankSimilarities(neighs, calculateIndegreeScores(nodeId), k, null);
      }
    }
    return new String[0];
  }

  /***
   * Returns a 2d array of outgoing links and a simulated path of length 6 towards the most probable articles.
   * 
   * @param testCase
   * @param approach
   * @return
   */
  private String[][] predictForTestCase(String testCase, String approach) {
    String[][] results = new String[2][];
    results[0] = getMostSimilars(testCase, approach);
    results[1] = new String[2 * 6];
    if (results[0].length > 0) {
      Set<String> alreadyVisited = new HashSet<>();
      results[1][0] = results[0][0].equals(testCase) && results[0].length > 2 ? results[0][2] : results[0][0];
      results[1][1] = results[0][0].equals(testCase) && results[0].length > 2 ? results[0][3] : results[0][1];
      alreadyVisited.add(testCase);
      alreadyVisited.add(results[1][0]);
      for (int step = 1; step < 6; ++step) {
        String nextPage = "", score = "";
        String[] similarPages = getMostSimilars(results[1][2 * (step - 1)], approach);
        for (int idx = 0; idx < similarPages.length; idx += 2) {
          if (alreadyVisited.add(similarPages[idx])) {
            nextPage = similarPages[idx];
            score = similarPages[idx + 1];
            break;
          }
        }
        results[1][2 * step] = nextPage;
        results[1][2 * step + 1] = score;
      }
    }
    return results;
  }

  private void printMostSimilar(String q, int k, boolean relativize, String... types) {
    // Set<String> typesSet = new HashSet<>(Arrays.asList(types));
    Integer nodeId = g.getNodeIdByLabel(q);
    if (nodeId == null) {
      System.err.format("No article named '%s' was found.\n", q);
    } else {
      int[] neighs = g.getOutLinks(nodeId);
      double[] ws = g.getWeights(nodeId);
      double[] baselineWs = new double[neighs[0]];
      double sum = 0.0d;
      for (int i = 1; i <= neighs[0]; ++i) {
        sum += (baselineWs[i - 1] = learner.getEtalonPRvalue(neighs[i]));
      }
      double[] transformedWeights = new double[neighs[0]];
      for (int i = 1; i <= neighs[0]; ++i) {
        transformedWeights[i - 1] = ws[i] - (relativize ? (baselineWs[i - 1] / sum) : 0.0d);
      }
      rankSimilarities(neighs, transformedWeights, k, "learned");
      rankSimilarities(neighs, calculateJaccardScores(nodeId), k, "jaccard");
      rankSimilarities(neighs, calculatePageRankScores(nodeId), k, "pagerank");
      rankSimilarities(neighs, calculatePopularityScores(nodeId), k, "clicks");
      rankSimilarities(neighs, calculateIndegreeScores(nodeId), k, "indegree");
      if (neighs[0] < 100) { // otherwise it would likely to take to long to wait for it in a demo application
        rankSimilarities(neighs, calculatePMIScores(nodeId), k, "PMI");
      }
    }
  }

  /**
   * Returns a ranked array of article_name, similarity_score doubles.
   * 
   * @param neighbors
   * @param scores
   * @param k
   * @param rankingType
   * @return
   */
  private String[] rankSimilarities(int[] neighbors, double[] scores, int k, String rankingType) {
    boolean silent = rankingType == null;
    int n = Math.min(neighbors[0], k);
    String[] orderedNeighbors = new String[2 * n];
    if (!silent) {
      System.err.format("Top-%d pages (out of %d) according to %s ranking.\n", n, neighbors[0], rankingType);
    }
    int[] ranking = Utils.stableSort(scores);
    for (int j = 1; j <= n; ++j) {
      int r = ranking[ranking.length - j];
      orderedNeighbors[2 * (j - 1)] = g.getNodeLabel(neighbors[r + 1]);
      orderedNeighbors[2 * (j - 1) + 1] = Double.toString(scores[r]);
      if (!silent) {
        System.err.format("\t%d\t%s\t%.9f\t%d\n", j, g.getNodeLabel(neighbors[r + 1]).replace('_', ' '), scores[r], neighbors[r + 1]);
      }
    }
    return orderedNeighbors;
  }

  /**
   * Given a query article, it generates a sequence of chainLength articles by always choosing the most probable neighbor from the actual article.
   * 
   * @param q
   * @param chainLength
   */
  private void findAssociationChain(String q, int chainLength) {
    Integer nodeId = g.getNodeIdByLabel(q), prevNodeId = -1;
    Set<Integer> pagesVisited = new HashSet<>();
    pagesVisited.add(nodeId);
    if (nodeId != null) {
      System.err.print(q);
      for (int step = 0; step < chainLength; ++step) {
        int[] neighs = g.getOutLinks(nodeId);
        double[] weights = g.getWeights(nodeId);
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 1; i <= neighs[0]; ++i) {
          if (weights[i] > max && !g.getNodeLabel(neighs[i]).matches("\\d+") && !pagesVisited.contains(neighs[i])) {
            max = weights[i];
            nodeId = neighs[i];
            pagesVisited.add(neighs[i]);
          }
        }
        if (prevNodeId == nodeId) {
          System.err.println();
          return;
        } else {
          prevNodeId = nodeId;
          System.err.format("-->%s", g.getNodeLabel(nodeId));
        }
      }
      System.err.println();
    }
  }

  private boolean loadEdgeWeights(int modelId) {
    return loadEdgeWeights(modelId, false);
  }

  private boolean loadEdgeWeights(int modelId, boolean interactive) {
    File serializedWeights = null;
    if (modelId == -1) {
      serializedWeights = new File(String.format("%s/%s%s%sFinalWeights.ser", dir, lang, date, learner.regularizationToString()));
    } else {
      serializedWeights = new File(String.format("%s/%s%s%sWeights_model%d.ser", dir, lang, date, learner.regularizationToString(), modelId));
    }
    if (!interactive && !serializedWeights.exists()) {
      return false;
    }
    while (interactive && !serializedWeights.exists()) {
      System.err.format("The desired model file (i.e. %s) does not exists. The program exits now.\n", serializedWeights.getAbsolutePath());
      System.err.format(
          "Please type in either\n-a valid path to a file storing edge weights\n-'q' if you wish to quit\n-'learn' if you want to learn a new model.\n>>");
      String input = scanner.nextLine();
      if (input.equals("q")) {
        System.exit(1);
      } else if (input.equals("learn")) {
        learnWeights(1, false);
        serializedWeights = new File(String.format("%s/%s%s%sFinalWeights.ser", dir, lang, date, learner.regularizationToString()));
      } else {
        serializedWeights = new File(input);
      }
    }
    try (ObjectInputStream objectinputstream = new ObjectInputStream(new FileInputStream(serializedWeights))) {
      g.setWeights((double[]) objectinputstream.readObject());
    } catch (ClassNotFoundException | IOException e) {
      System.err.format("%s\nThe graph has to be built from scratch.", e.getLocalizedMessage());
    }
    return true;
  }

  public void query(int modelId, boolean relativize) {
    loadEdgeWeights(modelId, true);
    String query = "";
    System.err.print(">>");
    while (!(query = scanner.nextLine().replace(' ', '_').trim()).equals("q")) {
      if (query.length() == 0) {
        printStrongConnections(String.format("%s%s_%d_strongConnections.txt", lang, date, modelId));
      } else {
        findAssociationChain(query, 10);
        printMostSimilar(query, 5, relativize);
      }
      System.err.print("\n>>");
    }
    scanner.close();
  }

  private void printStrongConnections(String outFile) {
    System.err.println("Strong connections will be written out to file " + outFile);
    int[] strongestConnections = new int[g.getNumOfNodes()];
    for (int i = 0; i < g.getNumOfNodes(); ++i) {
      int[] ns = g.getOutLinks(i);
      double[] ws = g.getWeights(i);
      double max = 0.0d;
      int argMax = -1;
      for (int j = 1; j <= ns[0]; ++j) {
        if (ws[j] > max && i != ns[j]) {
          max = ws[j];
          argMax = ns[j];
        }
      }
      strongestConnections[i] = argMax;
    }
    try (PrintWriter out = new PrintWriter(outFile)) {
      for (int i = 0; i < g.getNumOfNodes(); ++i) {
        int sc = strongestConnections[i];
        if (sc != -1 && i == strongestConnections[sc] && i > sc) {
          out.format("%s\t%s\t%d\t%d\n", g.getNodeLabel(i), g.getNodeLabel(strongestConnections[i]), g.getOutDegree(i),
              g.getOutDegree(strongestConnections[i]));
        }
      }
    } catch (FileNotFoundException e) {
      e.printStackTrace();
    }
  }

  private void navigate(String wikiPageName, String targetWikiPage) {
    Integer nodeId = g.getNodeIdByLabel(wikiPageName), prevNodeId = nodeId;
    if (nodeId == null) {
      System.err.format("No such page named as '%s' is found in the %s Wikipedia project (date: %s).\n", wikiPageName, lang, date);
      return;
    }
    LinkedList<Integer> pagesVisited = new LinkedList<>();
    int input = -1;
    while (true) {
      String pageName = g.getNodeLabel(nodeId);
      if (pageName.equals(targetWikiPage)) {
        break;
      }
      System.err.format("'%s'\n===============================\n", pageName);
      System.err.format("-1: quit\n");
      if (pagesVisited.size() > 0) {
        System.err.format("0: %s\n", g.getNodeLabel(pagesVisited.getLast()));
      }
      int[] ns = g.getOutLinks(nodeId);
      int[] outdegs = new int[ns[0]];
      for (int n = 1; n <= ns[0]; ++n) {
        outdegs[n - 1] = g.getNumOfNeighbors(ns[n]);
      }
      int[] numOfLinksBasedOrder = Utils.stableSort(outdegs);
      int[] nbrs = new int[ns[0] + 1];
      nbrs[0] = prevNodeId;
      Integer targetPageId = null;
      for (int n = 1; n <= ns[0]; ++n) {
        int nid = ns[numOfLinksBasedOrder[ns[0] - n] + 1];
        nbrs[n] = nid;
        String neighboringPageName = g.getNodeLabel(nid);
        System.err.format("%4d: %60s\t%5d\t%5d\n", n, neighboringPageName, g.getNumOfNeighbors(nid), g.getIndegree(nid));
        if (neighboringPageName.equals(targetWikiPage)) {
          targetPageId = nid;
        }
      }
      if (targetPageId != null) {
        pagesVisited.add(targetPageId);
        break;
      }
      boolean okInput = false;
      while (!okInput) {
        try {
          input = scanner.nextInt();
          if (input == -1) {
            break;
          } else if (input >= 0 && input <= ns[0]) {
            nodeId = nbrs[input];
            okInput = true;
          }
        } catch (Exception e) {
          System.err.println("Incorrect input. Please try again.");
        }
      }
      prevNodeId = nodeId;
      pagesVisited.add(nodeId);
    }
    System.err.format("A path consisting of %d steps was found.\n%s", pagesVisited.size(), wikiPageName);
    for (Integer i : pagesVisited) {
      System.err.print("-->" + g.getNodeLabel(i));
    }
    System.err.println();
    scanner.close();
  }

  /**
   * 
   * @param newGraph
   * @return
   */
  private Map<String, Map<Integer, Set<Integer>>> determineLinkChanges(OwnGraph newGraph, String testDate) {
    // link (A_i,A_j) can disappear due to the deletion of article A_j itself or
    // just by deleting the link between the 2 articles
    Map<Integer, Set<Integer>> hardDeletions = new HashMap<>();
    Map<Integer, Set<Integer>> deletions = new HashMap<>();
    Map<Integer, Set<Integer>> insertions = new HashMap<>();
    int[][] linkChangesPerPage = new int[g.getNumOfNodes()][3];
    for (int id = 0; id < g.getNumOfNodes(); ++id) {
      Integer newId = newGraph.getNodeIdByLabel(g.getNodeLabel(id));
      Set<Integer> insertedLinkIds = new HashSet<>(), deletedLinkIds = new HashSet<>(), hardDeletedLinkIds = new HashSet<>();
      if (newId != null) {
        Set<String> oldNeighbors = new HashSet<String>(Arrays.asList(g.getOutLinkLabels(id)));
        Set<String> newNeighbors = new HashSet<String>(Arrays.asList(newGraph.getOutLinkLabels(newId)));
        for (String newNeighbor : newNeighbors) {
          Integer neighborIdInOldGraph = g.getNodeIdByLabel(newNeighbor);
          if (neighborIdInOldGraph != null && !oldNeighbors.contains(newNeighbor)) {
            insertedLinkIds.add(neighborIdInOldGraph); // only nodes that existed in the old graph as well are considered
          }
        }

        oldNeighbors.removeAll(newNeighbors);
        for (String deletedArticle : oldNeighbors) { // since a removeAll is performed by now, we are left with the deleted links
          int deletedNodeId = g.getNodeIdByLabel(deletedArticle);
          String[] deletedNeighbors = g.getOutLinkLabels(deletedNodeId);
          // a "pseudodeletion" is the case when a link to an article is
          // replaced by a link to which the original article redirected to
          boolean pseudoDeletion = deletedNeighbors.length == 1 && newNeighbors.contains(deletedNeighbors[0]);
          if (!pseudoDeletion && newGraph.containsNode(deletedArticle)) {
            deletedLinkIds.add(deletedNodeId);
          } else if (!pseudoDeletion) {
            deletedLinkIds.add(deletedNodeId);
            hardDeletedLinkIds.add(deletedNodeId);
          }
        }
        linkChangesPerPage[id] = new int[] { insertedLinkIds.size(), deletedLinkIds.size(), hardDeletedLinkIds.size() };
      }
      insertions.put(id, insertedLinkIds);
      deletions.put(id, deletedLinkIds);
      hardDeletions.put(id, hardDeletedLinkIds);
    }
    printLinkChangesNumberPerArticle(linkChangesPerPage, String.format("wiki_%s_%s_%s_per_page_deletions_insertions.tsv", lang, date, testDate));
    Map<String, Map<Integer, Set<Integer>>> changes = new HashMap<>();
    changes.put("insertions", insertions);
    changes.put("deletions", deletions);
    changes.put("hardDeletions", hardDeletions);
    return changes;
  }

  private void printLinkChangesNumberPerArticle(int[][] linkChangesPerPage, String outFile) {
    try (PrintWriter out = new PrintWriter(outFile)) {
      out.println("insertions\tdeletions\tpage deletions");
      for (int i = 0; i < g.getNumOfNodes(); ++i) {
        out.format("%d\t%d\t%d\n", linkChangesPerPage[i][0], linkChangesPerPage[i][1], linkChangesPerPage[i][2]);
      }
    } catch (FileNotFoundException e) {
      e.printStackTrace();
    }
  }

  /**
   * Method to help distance-wise normalization of pagerank values.
   * 
   * @param distances
   * @param scores
   * @return
   */
  private double[] calculatePartitionByDistance(int[] distances, double[] scores) {
    double[] denominator = new double[10];
    for (int i = 0; i < distances.length; ++i) {
      if (distances[i] < 9) {
        if (distances[i] == -1) {
          denominator[9] += scores[i]; // the last index is reserved for unreachable nodes
        } else {
          denominator[distances[i]] += scores[i];
        }
      }
    }
    return denominator;
  }

  /**
   * Calculates and returns the mean and the standard error of the mean.
   * 
   * @param sum
   * @param sumSquared
   * @param n
   * @return
   */
  private double[] calculateStatistics(double sum, double sumSquared, long n) {
    if (n > 1) {
      double mean = sum / n;
      double var = (n / (n - 1)) * (sumSquared / n - mean * mean);
      double std = Math.sqrt(var);
      return new double[] { mean, std / Math.sqrt(n) };
    }
    return new double[] { 0.d, 0.d };
  }

  /**
   * 
   * @param targetId
   * @param ng
   * @param nodes
   * @return
   */
  private Map<Integer, Integer> calculateDistanceOfCriticalNodesInNewGraph(int targetId, OwnGraph ng, Set<Integer> nodes) {
    Map<Integer, Integer> newDistances = new HashMap<>();
    int targetIdInNewGraph = ng.getNodeIdByLabel(g.getNodeLabel(targetId));
    Set<Integer> nodeIdsInNewGraph = new HashSet<>();
    Map<Integer, Integer> nodeId2NewId = new HashMap<>();
    for (Integer nodeId : nodes) {
      int newNodeId = ng.getNodeIdByLabel(g.getNodeLabel(nodeId));
      nodeIdsInNewGraph.add(newNodeId);
      nodeId2NewId.put(nodeId, newNodeId);
    }
    int[] distancesInNewGraph = ng.bfs(targetIdInNewGraph, nodeIdsInNewGraph)[0];
    for (Integer nodeId : nodes) {
      newDistances.put(nodeId, distancesInNewGraph[nodeId2NewId.get(nodeId)]);
    }
    return newDistances;
  }

  /**
   * 
   * @param targetId
   * @param insertStats
   * @param deleteStats
   * @param hardDeleted
   * @param deleted
   * @param inserted
   */
  private void updateStats(int targetId, double[][][] insertStats, double[][][] deleteStats, Set<Integer> hardDeleted, Set<Integer> deleted,
      Set<Integer> inserted) {
    int[] distancesFromArticle = g.bfs(targetId)[0];
    // Map<Integer, Integer> distancesInNewGraph =
    // calculateDistanceOfCriticalNodesInNewGraph(targetId, newGraph, deleted);

    // high restart probability is employed as we want to explore the close
    // vicinity of the start page
    // PageRankCalculator ppc = new PageRankCalculator(0.95,
    // Stream.of(targetId).collect(Collectors.toSet()));
    // double[] personalizedPRs = ppc.calculatePageRank(g, false);
    // double[] normalizersByDistance =
    // calculatePartitionByDistance(distancesFromeArticle, personalizedPRs);

    double[] dijsktraDistance = g.dijkstra(targetId)[0];

    // int[] neighbors = g.getOutLinks(targetId);
    // double[] neighborWeights = g.getWeights(targetId), weights = new
    // double[g.getNumOfNodes()];
    // for (int i = 1; i <= neighbors[0]; ++i) {
    // weights[neighbors[i]] = neighborWeights[i];
    // }

    double[] maxDijkstra = { Double.MIN_VALUE, Double.MIN_VALUE }; // both for control and real insertion
    String[] argMax = { null, null };
    int[] bestDistance = { -1, -1 };
    for (int i = 0; i < g.getNumOfNodes(); ++i) {
      int distance = distancesFromArticle[i];
      if (distance < 9) {
        distance = distance == -1 ? 9 : distance;
        double score = 0.0d; // personalizedPRs[i] / normalizersByDistance[distance];
        if (distance > 1) {
          boolean controlGroup = !inserted.contains(i);
          if (maxDijkstra[controlGroup ? 0 : 1] < dijsktraDistance[i]) {
            maxDijkstra[controlGroup ? 0 : 1] = dijsktraDistance[i];
            argMax[controlGroup ? 0 : 1] = g.getNodeLabel(i);
            bestDistance[controlGroup ? 0 : 1] = distance;
          }
          for (int dist : new int[] { 1, distance }) {
            insertStats[dist][controlGroup ? 0 : 1][0] += score;
            insertStats[dist][controlGroup ? 0 : 1][1] += score * score;
            insertStats[dist][controlGroup ? 0 : 1][2]++;
            insertStats[dist][controlGroup ? 0 : 1][3] += dijsktraDistance[i];
            insertStats[dist][controlGroup ? 0 : 1][4] += dijsktraDistance[i] * dijsktraDistance[i];
          }
        } else if (distance == 1) { // deleted pages are quite noisy (shall we at all keep track of them?)
          boolean controlGroup = !deleted.contains(i);
          if (hardDeleted.contains(i)) {
            distance = 0; // we want to distinguish between deleted links and deleted articles (1(0)->link(article) deletion)
          }
          int alternativeDistance = 9; // this corresponds to unaccessible distance
          int[] inLinksFrom = g.getInLinks(i);
          for (int n = 1; n <= inLinksFrom[0]; ++n) {
            if (inLinksFrom[n] != targetId && distancesFromArticle[inLinksFrom[n]] != -1) {
              alternativeDistance = Math.min(alternativeDistance, distancesFromArticle[inLinksFrom[n]] + 1);
            }
          }
          // if (!controlGroup) {
          // System.err.format("%s\t%s\t%d\t%f\t%f\t%d\n", g.getNodeLabel(targetId), g.getNodeLabel(i), alternativeDistance, score,
          // dijsktraDistance[i], neighbors[0]);
          // }
          for (int dist : new int[] { distance, alternativeDistance }) {
            deleteStats[dist][controlGroup ? 0 : 1][0] += score;
            deleteStats[dist][controlGroup ? 0 : 1][1] += score * score;
            deleteStats[dist][controlGroup ? 0 : 1][2]++;
            deleteStats[dist][controlGroup ? 0 : 1][3] += dijsktraDistance[i];
            deleteStats[dist][controlGroup ? 0 : 1][4] += dijsktraDistance[i] * dijsktraDistance[i];
            // if (dist > 1 && !controlGroup) {
            // System.err.println(g.getNodeLabel(targetId) + "->" + g.getNodeLabel(i) + "\t" + dist + "\t" + g.getOutDegree(targetId) + "\t" +
            // dijsktraDistance[i] + "\t" + deleted.contains(i) + "\t" + hardDeleted.contains(i));
            // }
          }
        }
      }
    }
    for (int i = 0; i < argMax.length; ++i) {
      System.err.format("[STRONG_PAIR_%s]\t%s\t%s\t%d\t%f\n", i == 0 ? "A" : "B", g.getNodeLabel(targetId), argMax[i], bestDistance[i], maxDijkstra[i]);
    }
  }

  /**
   * Prints the mean and standard error of the probability for arriving to nodes up to a certain distance from some a page. </br>
   * Mean is calculated for the interesting (link deleted/inserted) and the control group as well.
   * 
   * @param linkChanges
   */
  private void linkPlacementEvaluation(Map<String, Map<Integer, Set<Integer>>> linkChanges) {
    Map<Integer, Set<Integer>> deletedLinkIds = linkChanges.get("deletions");
    Map<Integer, Set<Integer>> hardDeletedLinkIds = linkChanges.get("hardDeletions");
    Map<Integer, Set<Integer>> insertedLinkIds = linkChanges.get("insertions");

    int[] counter = new int[4];
    for (int id = 0; id < g.getNumOfNodes(); ++id) {
      Set<Integer> deletedLinks = deletedLinkIds.get(id);
      Set<Integer> hardDeletedLinks = hardDeletedLinkIds.get(id);
      Set<Integer> insertedLinks = insertedLinkIds.get(id);
      if (deletedLinks.size() * insertedLinks.size() > 0) {
        counter[0]++;
      } else if (insertedLinks.size() > 0) {
        counter[1]++;
      } else if (deletedLinks.size() > 0) {
        counter[2]++;
      }
      if (hardDeletedLinks.size() > 0) {
        counter[3]++;
      }
    }
    System.err.format("%d\t%d\t%d\t%d\n", counter[0], counter[1], counter[2], counter[3]);

    double[][][] insertStats = new double[10][2][5], deleteStats = new double[10][2][5];
    for (int id = 0, processedDocs = 0; id < g.getNumOfNodes(); ++id) {
      Set<Integer> deletedLinks = deletedLinkIds.get(id);
      Set<Integer> hardDeletedLinks = hardDeletedLinkIds.get(id);
      Set<Integer> insertedLinks = insertedLinkIds.get(id);
      System.err.format("INFO\t%d\t%d\t%d\t%d\t%d\n", processedDocs, id, deletedLinks.size(), hardDeletedLinks.size(), insertedLinks.size());
      if (insertedLinks.size() > 0) {
        updateStats(id, insertStats, deleteStats, hardDeletedLinks, deletedLinks, insertedLinks);
        if (++processedDocs % 50 == 0) { // print statistics after having processed 50 documents with both deletions and insertions
          printStatistics(insertStats, "ins");
          printStatistics(deleteStats, "del");
        }
      }
    }
  }

  private void printStatistics(double[][][] stats, String type) {
    for (int d = 0; d < 10; ++d) {
      double[][] statsForDistance = stats[d];
      for (int i = 0; i < 2; ++i) {
        // double[] confidenceForPPR = calculateStatistics(statsForDistance[i][0], statsForDistance[i][1], (long) statsForDistance[i][2]);
        double[] confidenceForDijkstra = calculateStatistics(statsForDistance[i][3], statsForDistance[i][4], (long) statsForDistance[i][2]);
        // System.err.format("d=%d\t%s\t%.9f\t%.9f\t%d\n", d, i == 0 ? "A" : "B", confidenceForPPR[0], confidenceForPPR[1], (long)
        // statsForDistance[i][2]);
        System.err.format("%s\td=%d\t%s\t%.9f\t%.9f\t%d\n", type, d, i == 0 ? "A" : "B", confidenceForDijkstra[0], confidenceForDijkstra[1],
            (long) statsForDistance[i][2]);
      }
    }
    System.err.println("==============================");
  }

  public static void main(String[] args) {

    if (args.length < 4) {
      System.err.format("4 command line arguments (i.e. folder, language, date, learn/query?) need to be provided. Program exits now.");
      System.exit(1);
    }
    int modelId = -1;
    if (args[3].contains("=")) {
      modelId = Integer.parseInt(args[3].split("=")[1]);
    }

    String[] dates = args[2].split(":");
    String trainDate = dates[0], testDate = dates.length > 1 ? dates[1] : null;
    WikipediaExperiment we = new WikipediaExperiment(args[0], args[1], trainDate);

    we.setExtensiveSerialization(args.length > 5 ? Boolean.parseBoolean(args[5]) : false);
    if (args[3].toLowerCase().startsWith("learn=")) {
      we.learnWeights(modelId, true);
    } else if (args[3].toLowerCase().startsWith("query")) {
      we.query(modelId, args.length > 6 ? Boolean.parseBoolean(args[6]) : false);
    } else if (args[3].equalsIgnoreCase("navigate")) {
      we.navigate("Szeged", "Jane_Goodall");
    }

    if (args[3].startsWith("eval") || args[3].toLowerCase().startsWith("learn=")) {
      boolean modelNeedsToBeLoaded = args[3].startsWith("eval");
      Random r = new Random(1l);
      String mode = args[3].split("_")[1].split("=")[0];
      System.err.println(mode);
      String choicerankGraphFile = "./choicerank_wiki_03_2016edges.output.gz";
      if (mode.equals("choicerank")) {
        we.loadChoicerankWeights(choicerankGraphFile);
      } else if (mode.equals("invPR")) {
        boolean successfullyLoaded = false;
        if (modelNeedsToBeLoaded) {
          successfullyLoaded = we.loadEdgeWeights(modelId);
        }
        if (!successfullyLoaded) {
          we.learner.learnEdgeWeights(modelId);
        }
      }

      String f = "/home/berend/datasets/wikipedia_clickstream/2016_03_en_clickstream.tsv.gz";
      Map<Integer, Map<Integer, Integer>> etalonNeighbors = new HashMap<>();
      Map<Integer, Integer> totalOutclicks = new HashMap<>();
      try (BufferedReader br = new BufferedReader(new InputStreamReader(new GZIPInputStream(new FileInputStream(f))));) {
        String line;
        int cntr = 0;
        while ((line = br.readLine()) != null) {
          if (++cntr % 500000 == 0) {
            System.err.println(cntr + " many clickstream lines read in...");
          }
          String parts[] = line.split("\t");
          if (!parts[2].equals("link")) {
            continue;
          }
          Integer from = we.g.getNodeIdByLabel(parts[0]);
          Integer to = we.g.getNodeIdByLabel(parts[1]);
          if (from != null && to != null) {
            if (we.g.containsEdge(from, to)) {
              int clicks = Integer.parseInt(parts[3]);
              etalonNeighbors.putIfAbsent(from, new HashMap<Integer, Integer>());
              etalonNeighbors.get(from).put(to, clicks);
              totalOutclicks.put(from, totalOutclicks.getOrDefault(from, 0) + clicks);
            }
          }
        }
      } catch (IOException e) {
        e.printStackTrace();
      }
      System.err.println("===========" + etalonNeighbors.size() + "=============");

      Map<Integer, Integer> mostProbablePredictedTransition = new HashMap<>();
      Map<Integer, Integer> mostProbableEtalonTransition = new HashMap<>();
      try (PrintWriter out = new PrintWriter(String.format("eval_%s_%d.out", mode, modelId))) {
        out.write("article\tetalon_transition\tpredicted_transition\tN\tKL-divergence\tRMSE\tP@1\tMRR\tdisplacement\n");
        for (Entry<Integer, Map<Integer, Integer>> e : etalonNeighbors.entrySet()) {
          String currentArticle = we.g.getNodeLabel(e.getKey());
          int[] neighborhood = we.g.getOutLinks(e.getKey());
          int N = neighborhood[0];
          double[] probs = we.g.getWeights(e.getKey());
          if (mode.equals("invPR") || mode.equals("clickstream")) {
            double[] tempProbs = new double[N + 1];
            for (int i = 0; i < tempProbs.length; ++i) {
              tempProbs[i] = probs[i];
            }
            probs = tempProbs;
          } else if (mode.equals("jaccard")) {
            probs = we.calculateJaccardScores(e.getKey());
          } else if (mode.equals("pagerank")) {
            probs = we.calculatePageRankScores(e.getKey());
          } else if (mode.equals("popularity")) {
            probs = we.calculatePopularityScores(e.getKey());
          } else if (mode.equals("degree")) {
            probs = we.calculateIndegreeScores(e.getKey());
          } else if (mode.equals("uniform")) {
            probs = new double[probs.length];
            for (int i = 1; i <= N; ++i) {
              probs[i] = 1.0d / N;
              probs[0] += probs[i];
            }
          } else if (mode.equals("random")) {
            probs = new double[probs.length];
            for (int i = 1; i <= N; ++i) {
              probs[i] = r.nextDouble();
              probs[0] += probs[i];
            }
            for (int i = N; i >= 0; --i) {
              probs[i] /= probs[0];
            }
          }
          if (probs.length == neighborhood.length - 1) {
            double[] extendedProbs = new double[probs.length + 1];
            double sum = 0;
            for (int i = 0; i < probs.length; ++i) {
              extendedProbs[i + 1] = probs[i];
              sum += probs[i];
            }
            probs = extendedProbs;
            if (Math.abs(sum - 1) > 1e-6) {
              System.err.println("WARNING: transition probs do not sum up to 1 for " + currentArticle);
            }
          }

          double[] etalonTransitions = new double[probs.length];
          double maxEtalonTransition = 0;
          int argMaxNeighbor = -1;
          for (int i = 1; i <= N; ++i) {
            double freq = (double) e.getValue().getOrDefault(neighborhood[i], 0);
            etalonTransitions[i] = freq / totalOutclicks.get(e.getKey());
            if (maxEtalonTransition < etalonTransitions[i]) {
              maxEtalonTransition = etalonTransitions[i];
              argMaxNeighbor = neighborhood[i];
            }
          }

          int[] sort = Utils.stableSort(probs);
          Map<Integer, double[]> sortedPredictions = new HashMap<>();
          int argmaxRank = -1;
          String predictedTransition = null;
          for (int i = sort.length - 1, rank = 0; i >= 0; --i) {
            if (sort[i] != 0) {
              sortedPredictions.put(neighborhood[sort[i]], new double[] { ++rank, probs[sort[i]] });
              if (neighborhood[sort[i]] == argMaxNeighbor) {
                argmaxRank = rank;
              }
              if (rank == 1) {
                predictedTransition = we.g.getNodeLabel(neighborhood[sort[i]]);
                mostProbablePredictedTransition.put(e.getKey(), neighborhood[sort[i]]);
              }
            }
          }

          sort = Utils.stableSort(etalonTransitions);
          Map<Integer, double[]> sortedEtalons = new HashMap<>();
          for (int i = sort.length - 1, rank = 0; i >= 0; --i) {
            if (sort[i] != 0) {
              sortedEtalons.put(neighborhood[sort[i]], new double[] { ++rank, etalonTransitions[sort[i]] });
              if (rank == 1) {
                mostProbableEtalonTransition.put(e.getKey(), neighborhood[sort[i]]);
              }
            }
          }

          double kl = 0, rmse = 0, rankDisplacement = 0;
          for (Entry<Integer, double[]> etalon : sortedEtalons.entrySet()) {
            double[] preds = sortedPredictions.get(etalon.getKey());
            double etalonProb = etalon.getValue()[1];
            rankDisplacement += Math.abs(etalon.getValue()[0] - preds[0]);
            if (etalonProb > 0) {
              kl += etalonProb * Math.log(etalonProb / preds[1]);
              rmse += Math.pow(etalonProb - preds[1], 2.0);
            }
          }
          rmse = Math.sqrt(rmse / N);
          rankDisplacement /= Math.pow(N, 2.0);
          double reciprocalRank = 1.0d / argmaxRank;
          String etalonTransition = we.g.getNodeLabel(argMaxNeighbor);
          out.format("%s\t%s\t%s\t%d\t%.5f\t%.5f\t%d\t%.5f\t%.5f\n", currentArticle, predictedTransition, etalonTransition, N, kl, rmse,
              argmaxRank == 1 ? 1 : 0, reciprocalRank, rankDisplacement);
          out.flush();
        }

      } catch (IOException e) {
        e.printStackTrace();
      }
      System.err.println("#ETALON transitions = " + mostProbableEtalonTransition.size());
      Map<Integer, Integer> mutuallyStrongEtalons = new HashMap<>();
      for (Entry<Integer, Integer> mp : mostProbableEtalonTransition.entrySet()) {
        Integer reverse = mostProbableEtalonTransition.getOrDefault(mp.getValue(), -1);
        // System.err.println(mp.getKey() + " " + we.g.getNodeLabel(mp.getKey()) + " " + mp.getValue() + " " + we.g.getNodeLabel(mp.getValue()) + " "
        // + reverse);
        if (reverse.equals(mp.getKey())) {
          mutuallyStrongEtalons.put(mp.getKey(), mp.getValue());
        }
      }
      Map<Integer, Integer> mutuallyStrongPredictions = new HashMap<>();
      for (Entry<Integer, Integer> mp : mostProbablePredictedTransition.entrySet()) {
        Integer reverse = mostProbablePredictedTransition.getOrDefault(mp.getValue(), -1);
        if (reverse.equals(mp.getKey())) {
          mutuallyStrongPredictions.put(mp.getKey(), mp.getValue());
        }
      }

      System.err.format("%d and %d etalon and predicted strong pairs resp.", mutuallyStrongPredictions.size(), mutuallyStrongEtalons.size());
      try (PrintWriter out = new PrintWriter(String.format("strong_pairs_%s.out", mode))) {
        for (Entry<Integer, Integer> predictedStrongs : mutuallyStrongPredictions.entrySet()) {
          String from = we.g.getNodeLabel(predictedStrongs.getKey());
          String to = we.g.getNodeLabel(predictedStrongs.getValue());
          int etalonTo = mutuallyStrongEtalons.getOrDefault(predictedStrongs.getKey(), -1);
          out.format("%s\t%s\t%d\n", from, to, predictedStrongs.getValue() == etalonTo ? 1 : 0);
        }
      } catch (IOException e) {
        e.printStackTrace();
      }
    }
    if (testDate != null) {
      OwnGraph newGraph = new WikipediaExperiment(args[0], args[1], testDate, false).g;
      Map<String, Map<Integer, Set<Integer>>> linkChanges = we.determineLinkChanges(newGraph, testDate);
      if (modelId == -2) {
        System.err.println("Jaccard initialization of the graph weights...");
        we.determineJaccardWeights();
      } else {
        we.loadEdgeWeights(modelId);
      }
      we.linkPlacementEvaluation(linkChanges);
    }
  }
}
