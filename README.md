# Supplementary code for the paper _Efficient algorithm to compute Markov transitional probabilities for a desired PageRank_

This repo includes the source code for performing Reverse PageRank calculation as introduced in the [EPJ Data Science publication](https://epjdatascience.springeropen.com/articles/10.1140/epjds/s13688-020-00240-z).  
In this setting the network topology and the importance of the nodes (i.e. the stationary distribution of a random walk) are assumed to be known in advance and algorithm looks for a weighting of the edges which ensures that the random walk visits each node proportional to the importance of the nodes provided in advance.

## Running the algorithm (even if you are unfamiliar to Java)

If you are not proficient in programming Java, you can still follow the below steps as well to use the algorithm:

* (optional) [Install Java Runtime Environment](https://www.oracle.com/java/technologies/downloads/) if it is not available on your computer.
* Download the [reversePageRank.jar](reversePageRank.jar) file from this repository.
* Create an XML file in the [GEXF format](https://gephi.org/gexf/format/) (in your favorite programming language) for the network the transition probabilities you would like to learn. Make sure that every node has an attribute called `weight`, which will be treated as the expected stationary distribution for the given node. The `label` attributes can be used for identifying the individual nodes. (You can also find a sample GEXF file in the `data/` folder of this repo that you can immediately try out.)
* Assuming your network in the GEXF format is located in the file `my_network.gexf`, you can run the algorithm by invoking the command ``java -jar reversePageRank.jar my_network.gexf``. The learned transition probabilities to meet the desired stationary distribution will be written into a file named `output.log`, and the loss values of the initial solution and the one found by the algorithm as described in the article get printed to the standard output. 

## The quickest way of trying out the program (assuming Maven being installed)
Assuming access to Maven, the quickest way to give the algorithm a try and also to see a visualization is to enter  
```mvn package```  
in the command line.  
Note that running the above command creates a file named `inversePR.txt` in the current working directory with details of the learning procedure printed to it.

## Reading in graphs
Relying on the functionalities provided by [Graphstream](http://graphstream-project.org/doc/Tutorials/Reading-files-using-FileSource/), it is possible to work with networks in multiple popular network formats (including the DOT, TLP and GEXF formats).  
The `hu.u_szeged.graph.reader.GraphReader` is an easily extendable class which (on default) operates on the `./data/sample.gexf` sample input file in the GEXF format.

## Dependencies
Our source code relies on the following dependencies (also included in the pom.xml for Maven.)
* [Graphstream v1.3](http://graphstream-project.org/download/)
* [Mallet v2.0.9](http://mallet.cs.umass.edu/download.php)
* [Apache Commons Math v3.6](http://commons.apache.org/proper/commons-math/download_math.cgi)
