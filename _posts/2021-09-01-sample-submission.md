---
layout: post
title: 6.S898 blog post
authors: Yepeng Huang
# tags: [sample, template, tutorial]  # This should be the relevant areas related to your blog post
hide: true
---

<center><h1> Exploring neural retrieval for learning protein functional dependencies </h1></center>

<center>
<!-- <p><a href="https://github.com/jasperhyp/HomologRetrievalPretraining">Modeling protein sequences with homolog retrieval</a></p> -->
<em>Modeling protein sequences with homolog retrieval</em>
</center>

<center>
<p> Yepeng Huang</p>
</center>

## Table of Contents

- **To bold text**, use `<strong>`.
- *To italicize text*, use `<em>`.
- Abbreviations, like <abbr title="HyperText Markup Langage">HTML</abbr> should use `<abbr>`, with an optional `title` attribute for the full phrase.
- Citations, like <cite>&mdash; Mark otto</cite>, should use `<cite>`.
- <del>Deleted</del> text should use `<del>` and <ins>inserted</ins> text should use `<ins>`.
- Superscript <sup>text</sup> uses `<sup>` and subscript <sub>text</sub> uses `<sub>`.


<nav id="TOC">
<ul>
<li><a href="#background">Background</a><ul>
<li><a
href="#protein-modeling">Protein modeling</a></li>
<li><a href="#multiple-sequence-alignment">Multiple sequence alignment</a></li>
<li><a href="#protein-language-modeling">Protein language modeling</a></li>
<li><a href="#questions-in-mind">Questions in mind</a></li>
</ul></li>
<li><a href="Strategy">Strategy</a><ul>
<li><a href="#model-architecture">Model architecture</a></li>
<li><a href="#data-and-training">Data and training</a></li>
<li><a href="#baselines">Baselines</a></li>
<li><a href="#benchmark-tests">Benchmark tasks</a></li>
</ul></li>
<li><a href="#results">Results</a><ul>
</ul></li>
<li><a href="#discussion">Discussion</a><ul>
<li><a href="#implications">Implications</a></li>
<li><a href="#limitations">Limitations</a></li>
<li><a href="#future-directions">Future directions</a></li>
</ul>
</nav>

# Background
Proteins, linear chains of molecules known as amino acids (residues), carry out most functional roles in organisms. There are twenty common amino acids, represented by `A-Z` except for `B/J/O/U/X/Z`. Thus, a single protein (subunit) can be described by a string of letters, which is called a *protein sequence* (or *primary structure*). Meanwhile, amino acids interact locally and form shapes (*secondary structure*) in the 3-dimensional space. Compositions of these shapes then fold up and form the complete three-dimensional structure of a protein (*tertiary structure*). Sometimes the protein structure can be surprisingly complex or regular (see below for a figure showing the very exquisite protein structure of Dronpa, a photoactivatable fluorescent protein found in a stony coral).

<div align='center'>
<img width='100%' src='https://assets-global.website-files.com/621e749a546b7592125f38ed/62273f8719ed3b2a84c8fd13_Fig%201.svg'/>
<figcaption><font size="2">Going from protein sequences to protein structures. (Figure courtesy of the <a href="https://www.deepmind.com/blog/alphafold-using-ai-for-scientific-discovery-2020">AlphaFold1 blog post from DeepMind</a>.)</font></figcaption>
</div>

<br>

<div align='center'>
<img width='50%' src='https://upload.wikimedia.org/wikipedia/commons/1/17/Dronpa_structure_animation.gif'/>
<figcaption><font size="2">The very exquisite three-dimensional structure of Dropnpa, a photoactivatable fluorescent protein found in a stony coral. (Figure courtesy of <a href="https://www.wikiwand.com/en/Dronpa">Wikipedia</a>.)</font></figcaption>
</div>

<br>

As would be expected for all sequences, there are patterns in protein sequences and structures. The <abbr title="Sub-tertiary three-dimensional structural unit in a protein within which structures share some level of sequence similarity, typically determining biologically functional activity of proteins">fold</abbr> those <span style="color:yellow">sheets</span> and <span style="color:red">helices</span> (see the above plot for corresponding colors) and all sorts of repeats compose to form can be highly similar in different proteins, even with highly distinct protein sequences (called *remote homology*). We call these regions/structures *protein domains*, and those proteins *homologs*. 

Homology implies evolutionary relationships both *between* and *within* sequences. In the history evolution, there are numerous opportunities for a protein residue to be mutated, while only a small fraction of them survive selection. This is because mutations would oftentimes change the local structure, and the protein might thus become mulfunctional due to inability of <abbr title="protein-protein binding">binding</abbr> or <abbr title="protein-molecule docking">docking</abbr>. Therefore, interacting pairs of residues impose constraints on what mutations are permissible for each of them, which we call *coevolution*. Then, from homologous proteins that diverge from speciation events, which usually carry similar functions across species (*orthologs*), we can thus infer structural peoximity of amino acids, and thus structures.

<div align='center'>
<img width='80%' src='http://gremlin.bakerlab.org/img/covary_small.gif'/>
<figcaption><font size="2">Structural constraints between a pair of interacting amino acids lead to coevolution. (Figure courtesy of <a href="http://gremlin.bakerlab.org/gremlin_faq.php">GREMLIN</a>.)</font></figcaption>
</div>
<br>

Obviously, evolutionary constraints imposed on protein structures through contacts between amino acids in turn manifest in patterns in protein sequences. This leads to the classical approach of protein modeling, the **Potts model**. 

<div align='center'>
<img width='80%' src='https://susannvorberg.github.io/phd_thesis/img/intro/correlated-mutations-transparent.png'/>
<figcaption><font size="2">Protein structure constrains coevolutionary patterns among amino acids, while patterns in amino acids can then be used to infer protein structure. (Figure courtesy of <a href="https://susannvorberg.github.io/phd_thesis/introduction-to-contact-prediction.html">Susann Vorberg</a>.)</font></figcaption>
</div>
<br>

## Protein modeling
Traditionally, Potts model has been the dominant method for protein sequence modeling, which fits a family of aligned sequences to learn evolutionary patterns \citep{potts_model_example}. More recent self-supervised protein language models (PLMs) that learn surprisingly generalizable representations also learn from evolutionary signals in sequences through masked language modeling \citep{msa_transformer, esm-1v, esm-2}, but many of them do not use sequence alignments, i.e. do not explicitly model dependencies between (aligned positions in) sequences. We interpret this as learning the contextual evolutionary similarity of amino acids \textit{within} sequence. Another line of protein language models still utilize multiple sequence alignment (MSA), though, and they are the current state-of-the-art in mutation effect prediction \citep{tranception} and structure prediction\footnote{Note that very recently, an MSA-free language model OmegaFold \citep{omegafold} has claimed to achieve similar performance in structure prediction as AlphaFold.} \citep{af2}. A recent research also showed that MSA Transformer \citep{msa_transformer}, a classical model of this line, encodes detailed phylogenetic relationships \citep{msa-phylo}. This line of methods model dependencies (similarity) both \textit{between} and \textit{within} sequences, and should be more effective and efficient in predicting properties of families of sequences with some evolutionary history. 

## Multiple sequence alignment


## Protein language modeling


## Questions in mind
We consider 

# Strategy
## Model architecture


## Data and training


## Baslines


## Benchmark tasks


# Results
|| Param size | Training corpus | Retrieval corpus | Remote homology (Accuracy) | MIB (Accuracy) | ABR (Accuracy) | Fluorescence (Spearman) | Stability (Spearman) |
|--|--|--|--|--|--|--|--|--|
| MSA Transformer (ft) | 100M | UniRef50 (26M MSAs, 1192 sequences each on average) | / | 0.22 | 0.715 | 0.961 | 0.64 | 0.67 |
| Evoformer (no template, ft) | 88M | PDB (190K structure + MSAs) | / | 0.23 | 0.794 | 0.979 | 0.67 | 0.79 |
| ESM-1b (standard, ft) | 650M | UniRef50 (27M sequences) | / | 0.31 | 0.840 | 0.979 | 0.68 | 0.76 |
| ESM-2 (small, fixed) | 35M | UniRef50 (60M sequences) | / |  | 0.68 | 0.96 |  |  |
| ESM-Retrieval (fixed) | 42M ([9,12]) | UniRef50 subset (50K sequences) | UniRef50 subset (5M, k=4) |  |  |  |  |  |  |
| ESM-Retrieval (fixed) | 42M ([9,12]) | UniRef50 subset (50K sequences) | UniRef50 subset (5M, k=2) |  |  |  |  |  |  |
| ESM-Retrieval (fixed) | 42M ([12]) | UniRef50 subset (50K sequences) | UniRef50 subset (5M, k=4) |  |  |  |  |  |  |
| ESM-Retrieval (fixed) | 42M ([12]) | UniRef50 subset (50K sequences) | UniRef50 subset (5M, k=2) |  |  |  |  |  |  |
| ESM-Retrieval (fixed) | 42M ([9,12]) | UniRef50 subset (500K sequences) | UniRef50 subset (5M, k=2) |  |  |  |  |  |  |
| ESM-Retrieval (fixed) | 42M ([12]) | UniRef50 subset (500K sequences) | UniRef50 subset (5M, k=2) |  |  |  |  |  |  |

Notes:
1. The first three rows of numbers are from [Hu et al. (2022)](#revisiting-plm). The original paper reported validation metrics (as indicated by their code, where they only split datasets into `train` and `test`, and used `test` sets to decide the best model) for `mib`, `abr`, and `fold` although they claimed that it was test metrics being reported. To align with their setting for comparison, we also report validation metrics for those tasks. The `sta` and `flu` tasks have test metrics reported.
2. [Hu et al. (2022)](#revisiting-plm) finetune the whole model in the downstream, while due to limitations in computational resource, we only evaluate fixed embeddings (finetune only the predictor). Thus, the numbers of the first three rows and the last four rows are **not** directly comparable.
3. Again, due to resource constraints, we were not able to get results on the 5M retrieval model.

# Discussions


## Implications


## Limitations
Paralogs homologous proteins that diverge from gene duplication, which usually carry distinct functions within species

Retrieval-free inference  Relates to orphan proteins

## Future directions


## References

<a id="retro" href="https://proceedings.mlr.press/v162/borgeaud22a.html">Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Rutherford, Katie Millican, George Bm Van Den Driessche, Jean-Baptiste Lespiau, Bogdan Damoc, Aidan Clark,Diego De Las Casas, Aurelia Guy, Jacob Menick, Roman Ring, Tom Hennigan, Saffron Huang, Loren Maggiore, Chris Jones, Albin Cassirer, Andy Brock, Michela Paganini, Geoffrey Irving, Oriol Vinyals, Simon Osindero, Karen Simonyan, Jack Rae, Erich Elsen, and Laurent Sifre. Improving language models by retrieving from trillions of tokens. Proceedings of the 39th International Conference on Machine Learning, 2022.</a> 

<a id="revisiting-plm" href="https://arxiv.org/abs/2206.06583">Mingyang Hu, Fajie Yuan, Kevin K. Yang, Fusong Ju, Jin Su, Hui Wang, Fei Yang, and Qiuyang Ding. Exploring evolution-aware & -free protein language models as protein function predictors. Conference on Neural Information Processing Systems 2022.</a>

<a id="af2" href="https://www.nature.com/articles/s41586-021-03819-2">John Jumper, Richard Evans, Alexander Pritzel, Tim Green, Michael Figurnov, Olaf Ronneberger, Kathryn Tunyasuvunakool, Russ Bates, Augustin Zıdek, Anna Potapenko, Alex Bridgland, Clemens Meyer, Simon A. A. Kohl, Andrew J. Ballard, Andrew Cowie, Bernardino Romera-Paredes, Stanislav Nikolov, Rishub Jain, Jonas Adler, Trevor Back, Stig Petersen, David Reiman,Ellen Clancy, Michal Zielinski, Martin Steinegger, Michalina Pacholska, Tamas Berghammer, Sebastian Bodenstein, David Silver, Oriol Vinyals, Andrew W. Senior, Koray Kavukcuoglu, Pushmeet Kohli, and Demis Hassabis. Highly accurate protein structure prediction with alphafold. Nature, 596(7873):583–589, Aug 2021.</a>

<a id="knn-lm" href="https://arxiv.org/abs/1911.00172">Urvashi Khandelwal, Omer Levy, Dan Jurafsky, Luke Zettlemoyer, and Mike Lewis. Generalization through memorization: Nearest neighbor language models. arXiv preprint arXiv:1911.00172, 2019.</a>

<a id="esm2" href="https://www.biorxiv.org/content/10.1101/2022.07.20.500902v2">Zeming Lin, Halil Akin, Roshan Rao, Brian Hie, Zhongkai Zhu, Wenting Lu, Nikita Smetanin, Robert Verkuil, Ori Kabeli, Yaniv Shmueli, Allan dos Santos Costa, Maryam Fazel-Zarandi, Tom Sercu, Salvatore Candido, and Alexander Rives. Evolutionary-scale prediction of atomic level protein structure with a language model. bioRxiv, 2022.</a>

<a id="phylogeny" href="https://www.nature.com/articles/s41467-022-34032-y">Umberto Lupo, Damiano Sgarbossa, and Anne-Florence Bitbol. Protein language models trained on multiple sequence alignments learn phylogenetic relationships. Nature Communications, 13(1):6298, Oct 2022.</a>

<a id="esm-1v" href="https://www.biorxiv.org/content/10.1101/2021.07.09.450648v1.full">Joshua Meier, Roshan Rao, Robert Verkuil, Jason Liu, Tom Sercu, and Alexander Rives. Language models enable zero-shot prediction of the effects of mutations on protein function. bioRxiv, 2021.</a>

<a id="tranception" href="https://proceedings.mlr.press/v162/notin22a.html">Pascal Notin, Mafalda Dias, Jonathan Frazer, Javier Marchena Hurtado, Aidan N Gomez, Debora Marks, and Yarin Gal. Tranception: Protein fitness prediction with autoregressive transformers and inference-time retrieval. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato (eds.), Proceedings of the 39th International Conference on Machine Learning 2022.</a>

<a id="msa-transformer" href="https://proceedings.mlr.press/v139/rao21a.html">Roshan M Rao, Jason Liu, Robert Verkuil, Joshua Meier, John Canny, Pieter Abbeel, Tom Sercu, and Alexander Rives. MSA transformer. In Marina Meila and Tong Zhang (eds.), Proceedings of the 38th International Conference on Machine Learning 2021.</a>

<a id="esm-1b" href="https://www.pnas.org/doi/10.1073/pnas.2016239118">Alexander Rives, Joshua Meier, Tom Sercu, Siddharth Goyal, Zeming Lin, Jason Liu, Demi Guo, Myle Ott, C. Lawrence Zitnick, Jerry Ma, and Rob Fergus. Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences. Proceedings of the National Academy of Sciences, 118(15):e2016239118, Apr 2021.</a>

<a id="potts" href="https://www.pnas.org/doi/10.1073/pnas.0805923106">Martin Weigt, Robert A. White, Hendrik Szurmant, James A. Hoch, and Terence Hwa. Identification of direct residue contacts in protein–protein interaction by message passing. Proceedings of the National Academy of Sciences, 106(1):67–72, Jan 2009.</a>

<a id="pomegafold" href="https://www.biorxiv.org/content/10.1101/2022.07.21.500999v1">Ruidong Wu, Fan Ding, Rui Wang, Rui Shen, Xiwen Zhang, Shitong Luo, Chenpeng Su, Zuofan Wu, Qi Xie, Bonnie Berger, Jianzhu Ma, and Jian Peng. High-resolution de novo structure prediction from primary sequence. Jul 2022.</a>
