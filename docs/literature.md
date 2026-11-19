# Where this work sits

Synthetic data is judged on fidelity, utility and privacy. Data augmentation knows that an edit can flip a label
and repairs it by hand. Financial benchmarks know that expert references contain errors. Preference optimization
knows that a graded judgment says more than a binary one. Each of these literatures holds one piece of a single
question that none of them asks: after a generated case changes, is its supervision still right? This page records
what each line establishes, where it stops, and what the article adds. The complete list of 192 sources is in the
[bibliography](bibliography.md).

## The short version

| Line of work | What it establishes | Where it stops | What the article adds |
| --- | --- | --- | --- |
| Tabular generators and constraint layers: CTGAN, TabDDPM, Stoian et al. | Learned generators fit mixed-type distributions; layers enforce linear inequalities | No statement about the law of the corrected sample; equalities are outside the layer | Identities by a projection that returns the exact conditional law, with a monitor read off the correction |
| Gaussian conditioning: Cong, Chen and Zhou; Matheron's rule | The exact sampler for a normal law on hyperplanes | A sampler, not a data architecture | Accounting identities in the place of the hyperplanes, and where exactness holds in a copula |
| Synthetic data in finance: Assefa et al., Potluru et al. | Realism, similarity and privacy for financial tables and series | Generated records carry no labels that can go stale | Supervision validity as a fourth axis |
| Label validity under edits: counterfactual augmentation, contrast sets, CheckList, GSM-Symbolic | A minimal edit can flip the gold answer | Validity is restored by a person, or by one formula per template | A margin certificate that says which labels may be inherited, with a stated error |
| Financial reasoning benchmarks: FinQA, FinanceBench, FinanceReasoning | References as executable programs; measured error rates in expert gold answers | Benchmarks, fixed once written | Batch Zero: the reference is re-solved independently before a seed scales |
| Graded preferences: Llama 2 and 3, HelpSteer2-Preference, DPO variants | Strength-graded comparisons improve reward models and policies | What a grade is worth is left unpriced | The Fisher information of a five-level judgment and the cutpoints that maximize it |
| Leakage and clustered evaluation: Kapoor and Narayanan; Cameron and Miller; Miller | Leakage taxonomies; cluster-robust inference | Diagnoses after the fact | Information sets fixed in the task specification, closed-form leak sizes, and the optimal number of variants per seed |

## Timing

The framework dates from late 2023. CTGAN, TabDDPM, counterfactual augmentation, contrast sets and FinQA precede
it. Constraint layers for deep tabular generators, GSM-Symbolic, HelpSteer2-Preference, the graded variants of DPO
and FinanceReasoning appeared while the program was running, and each addresses one of the four objects.

## Key sources

### Generators and constraints

<a id="xu2019"></a>
**Xu, Skoularidou, Cuesta-Infante and Veeramachaneni (2019), "Modeling Tabular Data using Conditional GAN."** [arXiv](https://arxiv.org/abs/1907.00503)
CTGAN models multimodal and imbalanced mixed-type columns with mode-specific normalization and a conditional
vector. The conditional vector is encouraged by a penalty, and nothing ties one column to another by rule. It is
the right tool for the empirical distribution and the reason business-rule checks follow it in my architecture.

<a id="stoian2024"></a>
**Stoian, Dyrmishi, Cordy, Lukasiewicz and Giunchiglia (2024), "How Realistic Is Your Synthetic Data? Constraining Deep Generative Models for Tabular Data."** [arXiv](https://arxiv.org/abs/2402.04823)
Unconstrained deep generators violate linear background constraints in a large share of samples, and a
differentiable constraint layer removes the violations. The layer guarantees feasibility for linear inequalities
in any generator. It makes no statement about the distribution of the corrected sample and does not cover
equalities, which is the division of labor Section 2 draws: inequalities to layers and transformations, identities
to the projection.

<a id="alaa2022"></a>
**Alaa, van Breugel, Saveliev and van der Schaar (2022), "How Faithful is your Synthetic Data?"** [PMLR](https://proceedings.mlr.press/v162/alaa22a.html)
Sample-level precision, recall and authenticity score each synthetic record for fidelity, diversity and copying.
These are the sharpest forms of the three standard axes, and all three score inputs. None asks whether the label
attached to a generated record is still correct.

<a id="cong2017"></a>
**Cong, Chen and Zhou (2017), "Fast Simulation of Hyperplane-Truncated Multivariate Normal Distributions."** [doi](https://doi.org/10.1214/17-BA1052)
A normal law restricted to $`Gx = r`$ is sampled exactly by $`x = y + \Sigma G^\top(G\Sigma G^\top)^{-1}(r - Gy)`$.
This is the update of Section 2, and with [Wilson et al. (2021)](https://arxiv.org/abs/2011.04026) on Matheron's
rule it is the lineage the article cites. I put accounting identities in the place of the hyperplanes, state that
exactness holds in the latent coordinates of the copula, and read a generator monitor off the size of the
correction (D4).

### Synthetic data in finance

<a id="assefa2020"></a>
**Assefa et al. (2020), "Generating Synthetic Data in Finance: Opportunities, Challenges and Pitfalls."** [doi](https://doi.org/10.1145/3383455.3422554)
The field's statement of its problems: realism, the measurement of similarity, and privacy, for retail-banking
tables and order-book data. A generated record is treated as data to be released, never as a supervised case whose
answer can go stale.

<a id="potluru2023"></a>
**Potluru et al. (2023), "Synthetic Data Applications in Finance."** [arXiv](https://arxiv.org/abs/2401.00081)
A survey of tabular, event, time-series and unstructured financial data under fidelity, utility and privacy.
Constraints appear in passing and training data for language models does not. The fourth axis is absent from the
field's most complete map.

### Labels under perturbation

<a id="kaushik2020"></a>
**Kaushik, Hovy and Lipton (2020), "Learning the Difference That Makes a Difference with Counterfactually-Augmented Data."** [arXiv](https://arxiv.org/abs/1909.12434)
Minimal human revisions that flip a label break models trained on the originals. The paper establishes that a
small edit can change the gold answer. Each edit and its new label are written by a person, at minutes per item,
with no rule for when a label survives.

<a id="gardner2020"></a>
**Gardner et al. (2020), "Evaluating Models' Local Decision Boundaries via Contrast Sets."** [ACL Anthology](https://aclanthology.org/2020.findings-emnlp.117/)
Expert edits near a test instance's decision boundary expose large drops in accuracy. This is the boundary focus
of my diagnostic tasks, as an evaluation method. Relabeling is again manual, which does not carry to generated
variants at scale.

<a id="ribeiro2020"></a>
**Ribeiro, Wu, Guestrin and Singh (2020), "Beyond Accuracy: Behavioral Testing of NLP Models with CheckList."** [arXiv](https://arxiv.org/abs/2005.04118)
Invariance tests assert that a perturbation preserves the label and directional tests assert that it changes it.
The split is declared per perturbation type. The margin certificate derives it per variant, from the distance to
the decision boundary, and attaches an error bound.

<a id="mirzadeh2024"></a>
**Mirzadeh et al. (2024), "GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models."** [arXiv](https://arxiv.org/abs/2410.05229)
Symbolic templates recompute the answer for every instance. This is recomputation done right when the answer is
one formula. A financial case also carries evidence and a preference between two responses, which a template does
not recompute, and most of its variants do not need recomputing at all.

### Financial reasoning and the reliability of references

<a id="chen2021"></a>
**Chen et al. (2021), "FinQA: A Dataset of Numerical Reasoning over Financial Data."** [ACL Anthology](https://aclanthology.org/2021.emnlp-main.300/)
Expert-written questions paired with executable reasoning programs. A reference stored as a program over its
inputs can be recomputed when the inputs change, which is the link Section 3 maintains for conclusions, evidence
and preferences together.

<a id="islam2023"></a>
**Islam et al. (2023), "FinanceBench: A New Benchmark for Financial Question Answering."** [arXiv](https://arxiv.org/abs/2311.11944)
Questions over public filings with gold answers and evidence strings. A frontier model of late 2023 used with a
retrieval system answered incorrectly or declined to answer 81% of them. This is the state of financial reasoning
the program started from.

<a id="tang2025"></a>
**Tang et al. (2025), "FinanceReasoning."** [arXiv](https://arxiv.org/abs/2506.05828)
Re-examining four public financial reasoning sets, the authors updated 15.6% of the questions, between 9.72% and
30.00% per set. Expert-written references carry an error rate of that order. A synthetic pipeline multiplies the
rate by the number of variants per seed, which is the case for an independent re-solve before scale (D10).

<a id="wang2023"></a>
**Wang et al. (2023), "Self-Instruct: Aligning Language Models with Self-Generated Instructions."** [arXiv](https://arxiv.org/abs/2212.10560)
Generated instructions are filtered for diversity and format, and the authors' own audit found a large share of
incorrect outputs. Mainstream synthetic instruction data was built without verification of each answer, the gap
the recompute rule closes for financial tasks.

### Graded preferences

<a id="touvron2023"></a>
**Touvron et al. (2023), "Llama 2: Open Foundation and Fine-Tuned Chat Models."** [arXiv](https://arxiv.org/abs/2307.09288)
Pairwise comparisons with a four-level strength scale feed a margin ranking loss, and the reward model is far less
accurate on the weakest-preference pairs than on the strongest. Graded preference predates my protocol. The
reported accuracy gap is the reason the protocol attaches evidence and a decisive variable to each judgment: near
the boundary, strength alone carries little signal.

<a id="wang2024"></a>
**Wang et al. (2024), "HelpSteer2-Preference: Complementing Ratings with Preferences."** [arXiv](https://arxiv.org/abs/2410.01257)
Strength-graded preferences with written justifications from several annotators, and a strength-scaled
Bradley-Terry objective that improves reward models. It is the closest public protocol. It does not price what a
grade is worth, which D7 does: 1.4 to 2 binary judgments, with the cutpoints that maximize the figure.

<a id="amini2024"></a>
**Amini, Vieira and Cotterell (2024), "Direct Preference Optimization with an Offset."** [arXiv](https://arxiv.org/abs/2402.10571)
A per-pair offset that grows with the strength of the preference improves on DPO, most when pairs are few. The
authors call for collecting the degree of preference, which the five-level scale records.

<a id="just2024"></a>
**Just, Jin, Sahu, Phan and Jia (2024), "Data-Centric Human Preference with Rationales for Direct Preference Alignment."** [arXiv](https://arxiv.org/abs/2407.14477)
Adding the likelihood of a rationale to the preference objective speeds convergence. The rationales there are
machine-written. Mine are expert records that cite the evidence, the rule and the decisive variable, and they also
serve as audit pairs.

### Leakage and clustered evaluation

<a id="kapoor2023"></a>
**Kapoor and Narayanan (2023), "Leakage and the Reproducibility Crisis in Machine-Learning-Based Science."** [arXiv](https://arxiv.org/abs/2207.07048)
A taxonomy of leakage across scientific fields, including temporal leakage and non-independence between training
and test data. The taxonomy diagnoses published work. The three information sets and family-level splits of
Sections 4 and 5 prevent the same classes at the point where the task is specified.

<a id="glasserman2023"></a>
**Glasserman and Lin (2023), "Assessing Look-Ahead Bias in Stock Return Predictions Generated by GPT Sentiment Analysis."** [arXiv](https://arxiv.org/abs/2309.17322)
Anonymizing firms separates what a model remembers from what it infers. This is leakage through the model's
memory. Section 4 treats the other channel, leakage through the generator, and gives its size in closed form.

<a id="cameron2015"></a>
**Cameron and Miller (2015), "A Practitioner's Guide to Cluster-Robust Inference."** [pdf](https://cameron.econ.ucdavis.edu/research/Cameron_Miller_JHR_2015_February.pdf)
The variance inflation of clustered data and its correction for unequal cluster sizes. The design effect of
Section 5 is the equal-correlation case. Econometrics stops at the diagnosis; D10 turns the same algebra into the
number of variants a seed should yield.

<a id="miller2024"></a>
**Miller (2024), "Adding Error Bars to Evals."** [arXiv](https://arxiv.org/abs/2411.00640)
Clustered standard errors for grouped evaluation questions, with inflation factors on public benchmarks of the
same order as the 3.24 of Section 5. The paper confirms that the effect is large in practice for language-model
evaluation.
