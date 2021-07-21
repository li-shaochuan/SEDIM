
# Introduction
## SEDIM
High-throughput Molecular Data Imputation and Characterization with Surrogate-assisted Automatic Neural Architecture Search

Single-cell RNA sequencing (scRNA-seq) technologies have been heavily developed to probe gene expression profiles at single-cell resolution. Deep imputation methods have been proposed to address the related computational challenges (e.g.
the gene sparsity in single-cell data). In particular, the neural architectures of those deep imputation models have been proven to be critical for performance. However, deep imputation architectures are difficult to design and tune for those without rich
knowledge of deep neural networks and scRNA-seq. Therefore, Surrogate-assisted Evolutionary Deep Imputation Model (SEDIM) is proposed to automatically design the architectures of deep neural networks for imputing gene expression levels in
scRNA-seq data without any manual tuning. Moreover, the proposed SEDIM constructs an offline surrogate model, which can accelerate the computational efficiency of the architectural search. Comprehensive studies show that SEDIM
significantly improves the imputation and clustering performance compared to other benchmark methods. In addition, we also extensively explore the performance of SEDIM in other contexts and platforms including mass cytometry and
metabolic profiling in a comprehensive manner. Marker gene detection, gene ontology enrichment, and pathological analysis are conducted to provide novel insights into cell-type identification and the underlying mechanisms.


Authors

Xiangtao Li

School of Artificial Intelligence, Jilin University, Jilin, China; Department of Computer science, City University of Hong Kong, Hong Kong SAR


# FAQ:
## Is there any demo?

By default, we provide the origin code. SEDIM is written in python 3.7, and tensorflow-gpu==1.14.

## What is the methodology behind?

The approach is sequentially divided into three components (Phases A, B and C). 
In Phase A, a deep imputation model (DIM) is designed to learn patterns in scRNA-seq data for imputing dropout values;
In Phase B, an automatic neural architecture search algorithm based on evolutionary optimization
is proposed for automatically designing deep neural network evolution to impute gene valuesfrom
scRNA-seq data; In Phase C, a surrogate-assisted evolutionary deep imputation model (SEDIM) is
proposed to model the relationship between the hyperparameter configuration P and its performance
f(P) to accelerate the fitness evaluation during the evolution
## Where can I get data?

By searching for the keyword Jurkat, 293T, Neuron9k, CD14+,
CD56+, CD19+, CD34+, and CD4+ on the 10X Genomics platform including . GSE84133, GSE89232, GSE67602, can be downloaded from the public NCBI database
# Contact
If you have any questions, please feel free to contact us

lixt314@jlu.edu.cn
