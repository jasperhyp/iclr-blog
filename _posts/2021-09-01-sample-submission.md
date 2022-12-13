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

## Contents

- **To bold text**, use `<strong>`.
- *To italicize text*, use `<em>`.
- Abbreviations, like <abbr title="HyperText Markup Langage">HTML</abbr> should use `<abbr>`, with an optional `title` attribute for the full phrase.
- Citations, like <cite>&mdash; Mark otto</cite>, should use `<cite>`.
- <del>Deleted</del> text should use `<del>` and <ins>inserted</ins> text should use `<ins>`.
- Superscript <sup>text</sup> uses `<sup>` and subscript <sub>text</sub> uses `<sub>`.


<nav id="TOC">
<ul>
<li><a href="#background">Background</a><ul>
<li><a href="#protein-modeling">Protein modeling</a></li>
<li><a href="#protein-language-modeling">Protein language modeling</a></li>
<li><a href="#questions-in-mind">Questions in mind</a></li>
</ul></li>
<li><a href="#strategy">Strategy</a><ul>
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
Proteins, linear chains of molecules known as amino acids (residues), carry out most functional roles in organisms. There are twenty common amino acids, represented by `A-Z` except for `B/J/O/U/X/Z`. Thus, a single protein (subunit) can be described by a string of letters, which is called a *protein sequence* (or *primary structure*). Meanwhile, amino acids interact locally and form shapes (*secondary structure*) in the 3-dimensional space ([Fig. 1](#sequence2structure)). Compositions of these shapes then fold up and form the complete three-dimensional structure of a protein (*tertiary structure*). Sometimes the protein structure can be surprisingly complex or regular (see [Fig. 2](#dronpa) for the very exquisite protein structure of Dronpa, a photoactivatable fluorescent protein found in a stony coral).

<div align='center'>
<img id='sequence2structure' width='100%' src='https://assets-global.website-files.com/621e749a546b7592125f38ed/62273f8719ed3b2a84c8fd13_Fig%201.svg'/>
<figcaption><font size="2">Figure 1. Going from protein sequences to protein structures. (Figure courtesy of the <a href="https://www.deepmind.com/blog/alphafold-using-ai-for-scientific-discovery-2020">AlphaFold1 blog post from DeepMind</a>.)</font></figcaption>
</div>

<br>

<div align='center'>
<img id='dronpa' width='40%' src='https://upload.wikimedia.org/wikipedia/commons/1/17/Dronpa_structure_animation.gif'/>
<figcaption><font size="2">Figure 2. The very exquisite three-dimensional structure of Dropnpa, a photoactivatable fluorescent protein found in a stony coral. (Figure courtesy of <a href="https://www.wikiwand.com/en/Dronpa">Wikipedia</a>.)</font></figcaption>
</div>

<br>

As would be expected for all sequences, there are patterns in protein sequences and structures. The <abbr title="Sub-tertiary three-dimensional structural unit in a protein within which structures share some level of sequence similarity, typically determining biologically functional activity of proteins.">fold</abbr> those <span style="color:#FDD100">sheets</span> and <span style="color:red">helices</span> (see the above plot for corresponding colors) and all sorts of repeats compose to form can be highly similar in different proteins, even with highly distinct protein sequences (called *remote homology*). We call these regions/structures *protein domains*, and those proteins *homologs*. 

Homology implies evolutionary relationships both *between* and *within* sequences. In the history evolution, there are numerous opportunities for a protein residue to be mutated, while only a small fraction of them survive selection. This is because mutations would oftentimes change the local structure, and the protein might thus become mulfunctional due to inability of <abbr title="protein-protein binding">binding</abbr> or <abbr title="protein-molecule docking">docking</abbr>. Therefore, interacting pairs of residues impose constraints on what mutations are permissible for each of them, which we call *coevolution* ([Fig. 3](#coevolution)). Then, from homologous proteins that diverge from speciation events, which usually carry similar functions across species (*orthologs*), we can thus infer structural proximity of amino acids, and thus structures.

<div align='center'>
<img id='coevolution' width='60%' src='http://gremlin.bakerlab.org/img/covary_small.gif'/>
<figcaption><font size="2">Figure 3. Structural constraints between a pair of interacting amino acids lead to coevolution. (Figure courtesy of <a href="http://gremlin.bakerlab.org/gremlin_faq.php">GREMLIN</a>.)</font></figcaption>
</div>
<br>

It is then obvious that evolutionary constraints imposed on protein structures through contacts between amino acids in turn manifest in patterns in protein sequences ([Fig. 4](#constraints2inference)). To better capture the within and between sequence dependencies, multiple sequence alignments (MSAs, see the letter matrix on the left below) are usually constructed using various alignment-based methods, which nicely align the positions in related sequences that potentially evolve from the same ancestry position. This leads to the classical approach of protein modeling, the **Potts model**. 

<div align='center'>
<img id='constraint2inference' width='60%' src='https://susannvorberg.github.io/phd_thesis/img/intro/correlated-mutations-transparent.png'/>
<figcaption><font size="2">Figure 4. Protein structure constrains coevolutionary patterns among amino acids, while patterns in amino acids can then be used to infer protein structure. (Figure courtesy of <a href="https://susannvorberg.github.io/phd_thesis/introduction-to-contact-prediction.html">Susann Vorberg</a>.)</font></figcaption>
</div>
<br>

## Classical protein modeling
Traditionally, Potts model has been the dominant method for protein sequence modeling, which fits a family of aligned sequences to learn evolutionary patterns ([Weigt et al., 2009](#potts)). In the Potts model, the input MSA is represented as $\mathbf X=\{\mathbf x_1, ..., \mathbf x_n\}$, where each $\mathbf x_k$ is a vector of $[x_{1k}, ..., x_{lk}]$ sampled from a distribution $p(\mathbf x)$. We can then model the single amino acid frequencies as $p_i(a)$, and pairwise frequencies (dependencies) as $q_{i,j}(a,b)$, where $i,j$ represent the position in the sequence, and $a,b$ represent amino acids. Then, solving for for the distribution $p(\mathbf x)$ with maximum entropy principle would (after some math) lead to the formulation of the Potts model:

$$p(\mathbf x|\mathbf v, \mathbf w)=\frac{1}{Z(\mathbf v, \mathbf w)}\exp\left(\sum_{i=1}^l v_i(x_i)\sum_{1\leq i<j\leq l}w_{ij}(x_i, x_j)\right),$$

which can then be optimized with various likelihood approximation methods.

<div align='center'>
<img id='input' width='60%' src='https://tianyu-lu.github.io/assets/images/potts_input.PNG'/>
<figcaption><font size="2">Figure 5. Example input to the Potts model. (Figure courtesy of <a href="https://tianyu-lu.github.io/communication/protein/ml/2021/04/09/Potts-Model-Visualized.html">Tianyu Lu</a>.)</font></figcaption>
</div>
<br>

> Does the input in [Fig. 5](#input) remind us of <abbr title="Attention mechanism...">something in deep learning?</abbr>

## Protein language modeling
More recently, the incredible success of deep learning in NLP also sparked a new trend in protein modeling ([Fig. 6](#plms)). Self-supervised protein language models (PLMs) based on architectures and learning methods from NLP are then established and proven to learn surprisingly generalizable representations of proteins that capture evolutionary signals in sequences. Commonly, masked language modeling ([Rives et al., 2021](#esm-1b)) or autoregressive modeling ([Notin et al., 2022](#tranception)) is used as the main objective, sometimes with regularizations ([Castro et al., 2022](#relso)). Although many of them do not use sequence alignments, i.e. do not explicitly model dependencies between (aligned positions in) sequences, the PLMs that utilize MSAs are still the current state-of-the-art in mutation effect prediction ([Notin et al., 2022](#tranception)) and structure prediction ([Jumper et al., 2021](#af2)) (though very recently, an MSA-free language model OmegaFold ([Wu et al., 2022](#omegafold)) has claimed to achieve similar performance in structure prediction as AlphaFold). This is likely due to the considerable benefit that between-sequence evolutionary information provides towards mutant fitness and structure understanding. A recent paper also showed that MSA Transformer ([Rao et al., 2020](#msa-transformer), as shown in [Fig. 7](#msa-transformer-fig)), a pioneering PLM that takes MSA into consideration, encodes detailed phylogenetic relationships ([Lupo et al., 2022](#phylogeny)). Intuitively, MSA-based models learn dependencies both *between* and *within* sequences, and should be more effective and efficient in predictions of propreties that have significant evolutionary underpinnings. 

<div align='center'>
<img id='plms' width='80%' src='https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs42256-022-00499-z/MediaObjects/42256_2022_499_Fig2_HTML.png'/>
<figcaption><font size="2">Figure 6. NLP methods and applications in the protein space. As is obvious, Transformer-based models are taking over everything these days. (Figure courtesy of <a href="https://www.nature.com/articles/s42256-022-00499-z">Ferruz et al. (2022)</a>.)</font></figcaption>
</div>
<br>

<div align='center'>
<img id='msa-transformer-fig' width='60%' src='https://www.biorxiv.org/content/biorxiv/early/2021/02/13/2021.02.12.430858/F1.large.jpg'/>
<figcaption><font size="2">Figure 7. With MSAs as input, a Transformer-based model can interleave row-wise and columns-wise attention in itself, and can be trained on a matrix-wise masked language modeling objective. (Figure courtesy of <a href="http://proceedings.mlr.press/v139/rao21a/rao21a.pdf">Rao et al. (2022)</a>.)</font></figcaption>
</div>
<br>

However, in a wide range of downstream function prediction tasks, such as fluorescence, metal ion binding, and antibiotic resistance (introduced later), better performances are observed from MSA-free language models ([Hu et al., 2022](#revisiting-plm)), suggesting that evolutionary constraints within-sequence might have contributed more to protein functions, and that MSA deters the learning of such knowledge. Meanwhile, there are also a few caveats in the use of MSA. Firstly, MSA is not available for a large fraction of the proteome, consisting of disordered regions, orphan proteins, antibodies2, and some others ([Wu et al., 2022](#omegafold)). Secondly, the fixed coordinate system that most MSA-based model constructed might not allow for predictions of effects of variants that do not align with such coordinate systems, such as insertion and deletion ([Notin et al., 2022](#tranception)). Thirdly, MSA-based models are very sensitive to the characteristics of the MSAs they are trained on, and in general, those models perform worse with fewer MSAs ([Hu et al., 2022](#revisiting-plm)). Again, echoing the first caveat, when a large enough set of homologous sequences cannot be retrieved, the model might suffer.

<div align='center'>
<img id='tranception-fig' width='80%' src='https://i.ibb.co/vk0BhSL/2022-12-12-230748.png'/>
<figcaption><font size="2">Figure 8. One possible solution is provided by Tranception, where the MSA is only retrieved at inference time as a prior to interpolate the conditional probability. The success of this model raises other questions, as specified below. (Figure courtesy of <a href="https://proceedings.mlr.press/v162/notin22a.html">Notin et al. (2022)</a>.)</font></figcaption>
</div>
<br>

## Questions in mind
An interesting experiment in [Hu et al. (2022)](#revisiting-plm) is using ESM-1b ([Rives et al., 2021](#esm-1b)), an MSA-free language model, to retrieve related (similar) sequences from respective databases, and then constructing MSA using an alignment algorithm (Famsa) for MSA-based models, which turns out to be promising. This ESM-1b retriever is finetuned in a siamese metric-learning-like fashion before retrieval. This attempt of homogeneous retrieval of related protein sequences (aka homologs), along with the success of retrieval-based PLMs such as Tranception ([Notin et al., 2022](#tranception), as shown in [Fig. 8](#tranception-fig)), inspired us to further explore homogeneous sequence retrieval to enhance protein language modeling. We further ask:

1. **Can we use a neural mechanism to cleverly construct MSA instead of constructing them a priori using alignment-based algorithms?**

2. **Are position alignments really needed? Can we generalize and replace MSA with the learning of some similarity, by cleverly integrating retrieved sequences directly into sequence modeling?**

3. **Can we further generalize homolog retrieval to heterogeneous spaces?**

We were not able to answer all those questions due to the time and computing resource constraints. So, we chose to address the second question -- arguably the most significant one. Noticing the recent success of retrieval-enhanced language models ([Khandelwal et al., 2019](#knn-lm)), especially RETRO from [Borgeaud et al. (2022)](#retro), which utilizes a frozen BERT model for retrieval and a cross-attention mechanism for integration, we explored adopting a similar cross-attention architecture for our homolog retrieval setup.

# Strategy
## Model architecture
We used a much smaller base model compared to the original RETRO. Specifically, we re-implemented an encoder-decoder architecture with ESM-2 ([Lin et al., 2022](#esm-2)) encoder based on RETRO's idea. 

**Nearest neighbor retrieval.** For efficiency, both retrieval corpus and query (training and evaluation) protein sequences are encoded using a frozen ESM-2 (35M). We did not use a larger model in order to reduce time cost. Tokens of $k=2,4$ nearest neighbors are retrieved for each sequence with Faiss.

**Encoder-decoder framework.** See [Fig. 9](#arch) for an overview of the architecture. The retrieved sequences are encoded with two Transformer encoder layers. To encode the contextual representation of neighbors, a cross-attention layer conditioned on the intermediate (before the first cross-attention layer in the decoder) activations of the sequence being modeled is used as the second encoder layer. The 35M 12-layer small ESM-2 encoder is used as the base model for (non-causal) decoder. We replace the individual self-attention layers with ESM-based RETRO blocks (basically interleaving a cross-attention layer) in one or both of the 9th and 12th layer in the decoder.

The encoder is trained from scratch, while the self-attention layers in the decoder are initialized with pretrained weights from ESM-2, and we froze all layers before the first cross-attention layer (can be 9th or 12th). A frozen pretrained ESM-2.

<div align='center'>
<img id='tranception-fig' width='80%' src='https://i.ibb.co/1KQCKp5/2022-12-12-231853.png'/>
<figcaption><font size="2">Figure 9. Our naive ESM-Retrieval model based on RETRO and ESM-2. (Figure courtesy of <a href="https://proceedings.mlr.press/v162/notin22a.html">Notin et al. (2022)</a>.)</font></figcaption>
</div>
<br>

## Data and training


## Baslines


## Benchmark tasks


# Results
|| Param size | Training corpus | Retrieval corpus | Fold (Accuracy) | MIB (Accuracy) | ABR (Accuracy) | Fluorescence (Spearman) | Stability (Spearman) |
|--|--|--|--|--|--|--|--|--|
| MSA Transformer (ft) | 100M | UniRef50 (26M MSAs, 1192 sequences each on average) | / | 0.22 | 0.72 | 0.96 | 0.64 | 0.67 |
| Evoformer (no template, ft) | 88M | PDB (190K structure + MSAs) | / | 0.23 | 0.79 | 0.98 | 0.67 | 0.79 |
| ESM-1b (standard, ft) | 650M | UniRef50 (27M sequences) | / | 0.31 | 0.84 | 0.98 | 0.68 | 0.76 |
| **ESM-2** (small, fixed) | 35M | UniRef50+ (60M sequences) | / | 0.27 | 0.68 | 0.96 | 0.32 | 0.45 |
| **ESM-Retrieval** (5e4\_9+12\_4, fixed) | 42M ([9,12]) | UniRef50 subset (50K sequences) | UniRef50 subset (500K, k=4) |  |  |  | / | / |
| **ESM-Retrieval** (5e4\_9+12\_4\_40K, fixed) | 42M ([9,12]) | UniRef50 subset (50K sequences) | UniRef50 subset (500K, k=4) |  |  |  | / | / |
| **ESM-Retrieval** (5e4\_9+12\_2, fixed) | 42M ([9,12]) | UniRef50 subset (50K sequences) | UniRef50 subset (500K, k=2) |  |  |  | / | / |
| **ESM-Retrieval** (5e4\_12\_4, fixed) | 42M ([12]) | UniRef50 subset (50K sequences) | UniRef50 subset (500K, k=4) |  |  |  | / | / |
| **ESM-Retrieval** (1e4\_9+12\_2, fixed) | 42M ([9,12]) | UniRef50 subset (10K sequences) | UniRef50 subset (500K, k=2) |  |  |  | / | / |

> Notes:
> 1. The first three rows of numbers are from [Hu et al. (2022)](#revisiting-plm). The original paper reported validation metrics (as indicated by their code, where they only split datasets into `train` and `test`, and used `test` sets to decide the best model) for `mib` and `abr` although they claimed that it was test metrics being reported. To align with their setting for comparison, we also report validation metrics for those tasks. The `fold`, `sta` and `flu` tasks have test metrics reported. 
> 2. [Hu et al. (2022)](#revisiting-plm) finetune the whole base model plus prediction head in the downstream, while due to limitations in computational resource, we only evaluate fixed embeddings. Since we are not finetuning, we had to use the average of all token embeddings as the sequence embedding, instead of using the embedding of the [CLS] token as in the paper. Also, we had to reduce batch size (and thus learning rate) in some cases due to extra GPU requirement imposed by retrieval. Thus, the numbers of the first three rows and the last five rows are **not** directly comparable.
> 3. Apart from 1 and 2, we kept all settings (dataset split, prediction head architecture, optimizer) the same as the original paper.
> 4. Again, due to resource constraints, we were not able to get results for the 5M retrieval model and some of the tasks listed above. We also suffered from unstable training, leading to NaNs, for some settings (not shown). Except for the 5e4\_9+12\_4\_40K setting (which runs about 40K passes), for each of the retrieval models, we also only run about 20K passes (batch size $\approx$ 25, depending on GPU availability at run time).
---

Our first interesting observation is that the plain ESM-2 already performed much better than MSA Transformer and Evoformer in Fold classification task, which shows that perhaps using MSA of sequences too similar might in turn hurt the capability of model to detect remote homology. Apart from this, all 


# Discussions


## Implications


## Limitations
Dealing with paralogs: homologous proteins that diverge from gene duplication, which usually carry distinct functions within species

Retrieval-free inference  Relates to orphan proteins

## Future directions
Investigate effect of scaling up both training corpus and retrieval corpus.

Finetune

Mutant effect prediction task

Reconstruct MSA

Knowledge-enhanced retrieval

Usage in structure prediction?


## References

<a id="retro" href="https://proceedings.mlr.press/v162/borgeaud22a.html">Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Rutherford, Katie Millican, George Bm Van Den Driessche, Jean-Baptiste Lespiau, Bogdan Damoc, Aidan Clark,Diego De Las Casas, Aurelia Guy, Jacob Menick, Roman Ring, Tom Hennigan, Saffron Huang, Loren Maggiore, Chris Jones, Albin Cassirer, Andy Brock, Michela Paganini, Geoffrey Irving, Oriol Vinyals, Simon Osindero, Karen Simonyan, Jack Rae, Erich Elsen, and Laurent Sifre. Improving language models by retrieving from trillions of tokens. Proceedings of the 39th International Conference on Machine Learning, 2022.</a> 

<a id="relso" href="https://www.nature.com/articles/s42256-022-00532-1">Egbert Castro, Abhinav Godavarthi, Julian Rubinfien, Kevin Givechian, Dhananjay Bhaskar, and Smita Krishnaswamy. Transformer-based protein generation with regularized latent space optimization. Nature Machine Intelligence 4:840–51, 2022.</a>

<a id="revisiting-plm" href="https://arxiv.org/abs/2206.06583">Mingyang Hu, Fajie Yuan, Kevin K. Yang, Fusong Ju, Jin Su, Hui Wang, Fei Yang, and Qiuyang Ding. Exploring evolution-aware & -free protein language models as protein function predictors. Conference on Neural Information Processing Systems 2022.</a>

<a id="af2" href="https://www.nature.com/articles/s41586-021-03819-2">John Jumper, Richard Evans, Alexander Pritzel, Tim Green, Michael Figurnov, Olaf Ronneberger, Kathryn Tunyasuvunakool, Russ Bates, Augustin Zıdek, Anna Potapenko, Alex Bridgland, Clemens Meyer, Simon A. A. Kohl, Andrew J. Ballard, Andrew Cowie, Bernardino Romera-Paredes, Stanislav Nikolov, Rishub Jain, Jonas Adler, Trevor Back, Stig Petersen, David Reiman,Ellen Clancy, Michal Zielinski, Martin Steinegger, Michalina Pacholska, Tamas Berghammer, Sebastian Bodenstein, David Silver, Oriol Vinyals, Andrew W. Senior, Koray Kavukcuoglu, Pushmeet Kohli, and Demis Hassabis. Highly accurate protein structure prediction with alphafold. Nature, 596(7873):583–589, Aug 2021.</a>

<a id="knn-lm" href="https://arxiv.org/abs/1911.00172">Urvashi Khandelwal, Omer Levy, Dan Jurafsky, Luke Zettlemoyer, and Mike Lewis. Generalization through memorization: Nearest neighbor language models. arXiv preprint arXiv:1911.00172, 2019.</a>

<a id="esm-2" href="https://www.biorxiv.org/content/10.1101/2022.07.20.500902v2">Zeming Lin, Halil Akin, Roshan Rao, Brian Hie, Zhongkai Zhu, Wenting Lu, Nikita Smetanin, Robert Verkuil, Ori Kabeli, Yaniv Shmueli, Allan dos Santos Costa, Maryam Fazel-Zarandi, Tom Sercu, Salvatore Candido, and Alexander Rives. Evolutionary-scale prediction of atomic level protein structure with a language model. bioRxiv, 2022.</a>

<a id="phylogeny" href="https://www.nature.com/articles/s41467-022-34032-y">Umberto Lupo, Damiano Sgarbossa, and Anne-Florence Bitbol. Protein language models trained on multiple sequence alignments learn phylogenetic relationships. Nature Communications, 13(1):6298, Oct 2022.</a>

<a id="esm-1v" href="https://www.biorxiv.org/content/10.1101/2021.07.09.450648v1.full">Joshua Meier, Roshan Rao, Robert Verkuil, Jason Liu, Tom Sercu, and Alexander Rives. Language models enable zero-shot prediction of the effects of mutations on protein function. bioRxiv, 2021.</a>

<a id="tranception" href="https://proceedings.mlr.press/v162/notin22a.html">Pascal Notin, Mafalda Dias, Jonathan Frazer, Javier Marchena Hurtado, Aidan N Gomez, Debora Marks, and Yarin Gal. Tranception: Protein fitness prediction with autoregressive transformers and inference-time retrieval. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato (eds.), Proceedings of the 39th International Conference on Machine Learning 2022.</a>

<a id="msa-transformer" href="https://proceedings.mlr.press/v139/rao21a.html">Roshan M Rao, Jason Liu, Robert Verkuil, Joshua Meier, John Canny, Pieter Abbeel, Tom Sercu, and Alexander Rives. MSA transformer. In Marina Meila and Tong Zhang (eds.), Proceedings of the 38th International Conference on Machine Learning 2021.</a>

<a id="esm-1b" href="https://www.pnas.org/doi/10.1073/pnas.2016239118">Alexander Rives, Joshua Meier, Tom Sercu, Siddharth Goyal, Zeming Lin, Jason Liu, Demi Guo, Myle Ott, C. Lawrence Zitnick, Jerry Ma, and Rob Fergus. Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences. Proceedings of the National Academy of Sciences, 118(15):e2016239118, Apr 2021.</a>

<a id="potts" href="https://www.pnas.org/doi/10.1073/pnas.0805923106">Martin Weigt, Robert A. White, Hendrik Szurmant, James A. Hoch, and Terence Hwa. Identification of direct residue contacts in protein–protein interaction by message passing. Proceedings of the National Academy of Sciences, 106(1):67–72, Jan 2009.</a>

<a id="pomegafold" href="https://www.biorxiv.org/content/10.1101/2022.07.21.500999v1">Ruidong Wu, Fan Ding, Rui Wang, Rui Shen, Xiwen Zhang, Shitong Luo, Chenpeng Su, Zuofan Wu, Qi Xie, Bonnie Berger, Jianzhu Ma, and Jian Peng. High-resolution de novo structure prediction from primary sequence. Jul 2022.</a>
