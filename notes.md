Notes

Data Set
  Mice Protein Expression Data Set: http://archive.ics.uci.edu/ml/datasets/Mice+Protein+Expression#
  Data Points: One sample/mouse
  Number of Data Points: 1080
  Features: Protein/Protein Modification
  Number of Features: 77
  Feature Values: Protein Expression Level
  Goal: "The aim is to identify subsets of proteins that are discriminant between the classes." 
  *Has missing values

Current Literature
  Self-Organizing Feature Maps Identify Proteins Critical to Learning in a Mouse Model of Down Syndrome
    URL: http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0129126
    Goal: "To identify biologically important differences in protein levels in mice exposed to context fear conditioning"
    What They Did (excerpts of abstract):
      They analyzed the data and determined that the drug memantine was effective in improving learning ability in trisomic mice.
      They designed a strategy based on the unsupervised clustering method, Self Organizing Maps (SOM) to determine which proteins
        and protein expression levels in mice were most important in learning ability.
      The SOM approach identified reduced subsets of proteins predicted to make the most critical contributions to normal learning,
        to failed learning and rescued learning, and provides a visual representation of the data that allows the user to extract 
        patterns that may underlie novel biological responses to the different kinds of learning and the response to memantine. 
      Results suggest that the application of SOM to new experimental data sets of complex protein profiles can be used to identify 
        common critical protein responses, which in turn may aid in identifying potentially more effective drug targets.
      
 
Our Project Goals
  Create a predictor that will classify a mouse's genotype (control or trisomy), treatment (saline or memantine), and behavior (stimulated to learn or not) based on its expression levels of 77 different genes.
  Identify subsets of proteins that most influential in determining different classes.
  Find the best way to handle missing values.
