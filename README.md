# SEDIM
High-throughput Molecular Data Imputation and Characterization with Surrogate-assisted Automatic Neural Architecture Search

Single-cell RNA sequencing (scRNA-seq) technologies have been heavily developed to probe gene expression profiles at single-cell resolution. Deep imputation methods have been proposed to address the related computational challenges (e.g.
the gene sparsity in single-cell data). In particular, the neural architectures of those deep imputation models have been proven to be critical for performance. However, deep imputation architectures are difficult to design and tune for those without rich
knowledge of deep neural networks and scRNA-seq. Therefore, Surrogate-assisted Evolutionary Deep Imputation Model (SEDIM) is proposed to automatically design the architectures of deep neural networks for imputing gene expression levels in
scRNA-seq data without any manual tuning. Moreover, the proposed SEDIM constructs an offline surrogate model, which can accelerate the computational efficiency of the architectural search. Comprehensive studies show that SEDIM
significantly improves the imputation and clustering performance compared to other benchmark methods. In addition, we also extensively explore the performance of SEDIM in other contexts and platforms including mass cytometry and
metabolic profiling in a comprehensive manner. Marker gene detection, gene ontology enrichment, and pathological analysis are conducted to provide novel insights into cell-type identification and the underlying mechanisms.


Authors

Xiangtao Li and Ka-Chun Wong

Department of Information Science and Technology, Northeast Normal University, Changchun, Jilin, China Department of Computer Science, City University of Hong Kong, Hong Kong

Contact

lixt314@nenu.edu.cn
