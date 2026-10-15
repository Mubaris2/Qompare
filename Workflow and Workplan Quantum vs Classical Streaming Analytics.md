# Workflow and Workplan: Quantum vs Classical Approaches for Streaming Data Analytics

Team of 2 (Member A and Member B). Duration: 5 weeks plus Day 0 setup. Companion to the approved proposal.

## 1. Project Summary

**Goal.** Compare classical streaming ML against quantum and hybrid quantum-classical ML on streaming data, under one shared, fair, sliding-window protocol, and test whether quantum predictive advantages reported in offline papers survive streaming constraints (latency, incremental updates, encoding overhead, drift).

**Base papers.**

- P1: Domingos and Hulten (2000), Hoeffding Tree / VFDT. Classical baseline, reproduced.
- P2: Herbst, De Maio, Brandic (2024), variational quantum regression on Seoul Bike data (arXiv:2402.15542).
- P3: Tscharke, Issel, Debus (2023), quantum-kernel SVR for semisupervised anomaly detection (arXiv:2308.00583).

**Research questions.**

- RQ1: On equal data and window budgets, does a quantum model match or beat the best classical streaming/kernel model in predictive quality?
- RQ2: What does the quantum side cost per example (encode, fit, predict) compared with classical, and how does that cost scale with window size?
- RQ3: Do quantum models learn faster from fewer samples (sample efficiency), as the QML literature claims?
- RQ4: How do all models behave around concept drift?
- RQ5 (enhancement): Do incremental kernel caching and warm-started variational training cut quantum cost without hurting accuracy?

**Hypotheses (state them before running, so results are not rationalised afterwards).**

- H1: Quantum kernel SVR is competitive with RBF SVR on small anomaly-detection data, but not clearly better once hyperparameters of both are tuned fairly.
- H2: Quantum cost per example grows much faster with window size than classical cost.
- H3: Kernel caching cuts kernel evaluations by roughly W/H times without changing predictions.
- H4: Hoeffding Tree has far higher throughput and comparable accuracy on drifting classification streams.

A negative result (quantum does not win) is a valid and publishable outcome. The study is judged on rigour, not on quantum winning.

## 2. Experimental Design

### 2.1 Tasks and datasets

| Task | Data | Preprocessing | Models compared |
| --- | --- | --- | --- |
| T1 Regression | Seoul Bike Sharing Demand (UCI, 8,760 hourly rows) | Drop non-numeric or one-hot them, PCA to 7 features, scale to \[0, pi\], target to \[0, 1\] | VQR (Qiskit), SVR-RBF, River HoeffdingTreeRegressor, Random Forest (batch reference) |
| T2 Anomaly detection | One or two benchmark AD sets (ODDS or ADBench; check the datasets listed in P3's paper) | PCA to 4 to 6 features, scale to \[0, pi\], train on normal only | QSVR (PennyLane), SVR-RBF reconstruction scorer, One-Class SVM, small autoencoder |
| T3 Classification | Electricity (ELEC2, 45,312 rows, natural drift) | Use first 5,000 to 10,000 rows for any quantum run | Hoeffding Tree (reproduction), HoeffdingAdaptiveTree, Naive Bayes, optional QSVC (stretch) |

T3 is the P1 reproduction. T1 and T2 are the quantum-vs-classical comparisons.

### 2.2 Streaming protocol (shared by every model)

1. Order data by time (or by fixed shuffled order with a fixed seed where no time exists).
2. Fit scaler and PCA on the first window only. Apply frozen to the rest. This avoids leakage and mimics a real stream.
3. Loop: train on the last W samples, test on the next H samples, record metrics and timers, slide forward by H.
4. Window grid: W in {100, 200, 400}. Step H = 20 (T1, T3) or H = 50 (T2).
5. Drift scenarios: none, abrupt (feature mean shift at 50 percent of stream), gradual (linear shift over 20 percent of stream). Natural drift in ELEC2 is also used.
6. Seeds: 5 (fallback 3 for quantum, stated openly in the report).

Anomaly detection detail: a block of 50 samples may contain zero anomalies, so AUC is undefined per block. Compute AUC on rolling pooled blocks of 500 scored samples, and also on the full stream.

### 2.3 Metrics

- Quality: MSE and MAE (T1), AUC and average precision (T2), prequential accuracy and F1 (T3).
- Efficiency: t\_encode, t\_fit, t\_predict in seconds, throughput in examples per second, peak memory.
- Sample efficiency: metric versus number of samples seen (learning curve).
- Resource footprint: qubits, circuit depth, trainable parameters, circuit executions count; for classical: tree nodes or support vectors.
- Drift behaviour: metric drop at drift point and recovery time (samples until back within 5 percent of pre-drift level).

### 2.4 Fairness controls (the part reviewers will attack)

- Same data, same window, same order, same seeds for every model.
- Same hyperparameter tuning budget per model (for example 20 random-search trials on the first window or a held-out prefix, never on the test stream).
- Classical baselines get tuned too (SVR C, epsilon, gamma; tree grace period and delta). Do not compare tuned quantum against default classical.
- Report the strongest classical competitor, not only the convenient one.
- Separate noiseless simulation results from shot-noise results. Noiseless is an upper bound on what hardware can do.
- Report time for classical and simulated quantum with a warning: simulator wall-clock is not hardware wall-clock. Also report hardware-relevant proxies (circuit executions, circuit depth).

## 3. Enhancements (what is new beyond the papers)

**E1. Incremental kernel cache for QSVR.** When the window slides by H, only H new samples enter and H leave. Keep the kernel matrix; delete H rows and columns, compute only the new H rows and columns. Kernel evaluations per step fall from about W squared over 2 to about W times H. Verify the cached kernel equals a freshly computed kernel to numerical tolerance (unit test), and that predictions are identical.

**E2. Warm-started variational training for VQR.** Initialise each window's parameters from the previous window's trained parameters and run few optimiser iterations (for example 20 instead of 100). Compare accuracy and cost against cold-start retraining.

**E3. Encoding overhead decomposition.** Measure how much of the total time is state preparation (encoding) versus optimisation versus inference. P2 names encoding as a bottleneck but does not quantify it in a streaming loop.

**Simulation insight to exploit.** In noiseless statevector simulation, each sample's state can be computed once and the kernel is the squared overlap matrix, which is very fast. On real hardware or with finite shots, every kernel entry needs its own circuit executions. So report both the simulated time and the count of circuit executions a hardware run would need. The cache win is most meaningful in the circuit-count metric.

## 4. Team Structure and Isolation

Two people, three handoff files, no shared code editing.

**Member A: classical, data, harness, analysis.** Owns data prep, the evaluation harness, metrics, classical models, P1 reproduction, drift injection, stats and plots.

**Member B: quantum models and enhancements.** Owns P2 and P3 reproductions, quantum wrappers, E1 to E3, quantum interpretation.

**Repository layout.**

- common/ : A owns. Harness, metrics, timers, result validator. Frozen by Day 3.
- data/ : A owns. raw/ and processed/.
- classical/ : A owns.
- quantum/ : B owns.
- results/classical/ and results/quantum/ : each writes only its own.
- analysis/ : A owns code, reads both result folders.
- report/ : sections split by owner.

**Handoffs.**

- Data: data/processed/T\*\_stream.parquet (A to B).
- Interface: common/CONTRACT.md, frozen Day 1 (both).
- Results: results/\<owner>/\<task>\_\<model>.csv (each to analysis).

**Rules.** Branches a/\* and b/\*. Pull requests only for common/. A validator script runs on every pull request and rejects CSVs that break the schema. A ships a tiny mock parquet on Day 1 so B is never blocked.

## 5. Technical Contract

**Model interface (both implement).**

- name: string.
- fit(X, y=None): train on current window.
- predict(X): regression or classification output (T1, T3).
- score\_anomaly(X): higher means more anomalous (T2).
- timings: dict with t\_encode, t\_fit, t\_predict.
- meta: dict with n\_params, n\_qubits, circuit\_depth, n\_circuit\_evals (zeros for classical).

**Result CSV columns.** task, model, variant, seed, W, H, drift, step, n\_seen, metric\_name, metric\_value, t\_encode, t\_fit, t\_predict, n\_params, n\_qubits, n\_circuit\_evals, peak\_mem\_mb.

**Reproducibility.** Pinned requirements.txt, fixed seeds, config files (YAML) per experiment, one command to run each experiment, git commit hash stored in every result file.

## 6. Workflow (pipeline view)

1. **Acquire** raw data into data/raw (A).
2. **Prepare**: clean, PCA, scale, export parquet, plus drift-injected versions (A).
3. **Tune**: random search on a prefix of each stream, per model, equal budget (each owner).
4. **Stream-evaluate**: run the shared window loop, write result CSVs (each owner).
5. **Validate**: schema validator and unit tests (automatic).
6. **Analyse**: merge CSVs, tables, plots, statistical tests (A, with B interpreting quantum parts).
7. **Write**: report and slides (split by section).
8. **Package**: README, run script, archived results.

## 7. Detailed Workplan

### Day 0 (both, about 3 hours)

- Agree and freeze CONTRACT.md (tasks, protocol, schema, seeds).
- Create repository, folder layout, pinned environment, pre-commit and pull request validator.
- Send the paper list to the instructor. Do not start the baseline reproduction until approved; setup work is fine.
- Pick and write down hypotheses H1 to H4.

### Week 1: Foundations

**Member A**

- Download Seoul Bike, ELEC2, and chosen AD dataset. Write prep.py: clean, PCA, scale, export parquet (T1, T2, T3), plus mock data.
- Write common/harness.py (window loop), metrics.py, timing.py, validate\_results.py. Freeze by Day 3 and notify B.
- Reproduce Hoeffding Tree on ELEC2 with River: prequential accuracy over the stream, throughput, memory, tree size. Also run the batch tree reference.
- Write drift injectors (abrupt and gradual) with unit tests.

**Member B**

- Build environments for Qiskit, qiskit-machine-learning, qiskit-aer, PennyLane. Pin versions. Check API differences from the papers' era: primitives, kernel classes and optimiser interfaces change between releases.
- Run Qiskit and PennyLane tutorials end to end (links in Section 11).
- Reproduce P2 static: VQR on Bike, 7 PCA features, 400 train and 250 test, ZFeatureMap, one ansatz, SPSA and COBYLA. Log MSE, MAE, runtime.
- Wrap VQR in the Model interface against the mock parquet.

**Exit criteria.** A: harness frozen, Hoeffding Tree curve produced. B: one VQR run with MSE and MAE, wrapped in the interface.

### Week 2: Baselines and paper reproductions

**Member A**

- T1 classical: SVR-RBF, River HoeffdingTreeRegressor, Random Forest. Tune with equal budget.
- T2 classical: SVR-RBF reconstruction scorer (P3's classical analogue), One-Class SVM, small autoencoder.
- T3: HoeffdingAdaptiveTree, Naive Bayes, ADWIN-based drift alarms.
- Run all in batch mode first (paper parity), then through the harness.

**Member B**

- P2 mini-sweep: about 24 configurations (feature map Z or ZZ, optimiser SPSA or COBYLA, 2 ansatz types, 2 repetition counts) instead of 672. Check the qualitative trend: encoding and optimiser matter more than ansatz.
- P3 reproduction: quantum kernel from a feature map via state overlap, precomputed kernel to SVR, anomaly score as reconstruction loss, AUC on the T2 dataset. Compare with A's SVR-RBF numbers when available.
- Record numeric gaps to the papers and likely causes (data split, library version, seed, qubit count).

**Checkpoint 1 (end of Week 2, 1 hour).** Exchange CSVs, run validator. Confirm trends against papers. Fix final W and H grid and tuning budget. Decide quantum seed count from measured runtimes.

### Week 3: Streaming enhancements

**Member A**

- Run all classical models in the sliding-window protocol on T1, T2, T3 with and without drift.
- Add drift metrics: pre-drift level, drop, recovery time.
- Measure latency per example, throughput, memory for every model.
- Draft analysis notebook skeleton against classical results and mock quantum rows.

**Member B**

- Build streaming wrappers for VQR and QSVR.
- Implement E1 kernel cache; unit test equality with full recompute; count circuit executions.
- Implement E2 warm start for VQR; compare against cold start at equal iteration budget and at equal accuracy target.
- Implement E3 timers splitting encode, fit, predict.
- Optional ablation: finite shots (for example 1024) to show noise effect on QSVR AUC.

**Exit criteria.** Every model has one complete streaming run on at least one task, CSVs valid, E1 test passing.

### Week 4: Full runs and analysis

- Full grid: tasks x models x W x drift x seeds. Each owner runs only their models. Use a run log (what finished, seed, commit hash).
- If runtime exceeds plan, cut in this order: shot-noise ablation, T3 QSVC, quantum seeds from 5 to 3, quantum W=400. Never cut the reproductions or E1.

**Member A (analysis)**

- Tables: model x task x metric, mean and standard deviation across seeds.
- Plots: (1) metric vs samples seen, (2) latency vs W, (3) behaviour around drift, (4) encode share of total time, (5) circuit executions with and without cache.
- Statistics: paired Wilcoxon signed-rank across seeds or windows, effect sizes, multiple comparison correction (Holm).

**Member B**

- Interpret quantum results: qubits vs accuracy, shot noise, kernel concentration risk (kernels with many qubits can become nearly constant, see Thanasilp et al.), cache correctness.
- Prepare short appendix on classical simulability of the circuits used.

**Checkpoint 2 (end of Week 4).** Freeze results. Every claim in the report must trace to a CSV and a figure.

### Week 5: Writing and polish

| Section | Owner |
| --- | --- |
| Abstract, introduction, related work, datasets, classical methods, harness, metrics | A |
| Quantum methods, enhancements, NISQ limits, threats to validity | B |
| Results, discussion, conclusion | One drafts, the other reviews |
| Slides and demo | 50 percent each |

- Days 1 to 2: draft. Day 3: cross review (each reads the other's sections). Day 4: figures final, README, one-command reproduction check on a clean machine. Day 5: buffer, rehearsal.

## 8. Experiment Budget (plan before running)

- Classical runs are cheap (seconds to minutes each).
- Quantum runs: count kernel evaluations and optimiser iterations first. Example: QSVR with W=400 needs a 400 x 400 kernel; with cache each step needs about W x H new entries. Estimate total circuit-evaluation count and time one step before launching the grid.
- Keep qubits at 7 or fewer (4 to 6 for QSVR). Use statevector for noiseless runs.
- Use Colab, Kaggle, or a lab machine for parallel seeds. Each seed is an independent job.

## 9. Risks and Mitigations

| Risk | Likelihood | Mitigation |
| --- | --- | --- |
| Quantum simulation too slow | High | Cap qubits, subsample, cache, statevector trick, fewer seeds with disclosure |
| Cannot match paper numbers | Medium | Report gap and probable causes; trends matter more than exact values |
| Library API changes break paper code | Medium | Pin versions, read current docs, write thin wrappers |
| Schema drift between members | Medium | Frozen contract, validator on every pull request |
| Quantum loses everywhere | Medium | Valid finding; emphasise cost and sample efficiency analysis |
| Unfair comparison challenged | Medium | Equal tuning budget, strongest classical baseline, documented controls |
| Member unavailable | Low | Contract and mock data let the other continue; shared README of how to run each side |
| Instructor rejects a paper | Low | Alternates listed in Section 11 (EFDT, VHT, quantum autoencoder papers) |

## 10. Quality Gates

- Gate 0: papers approved.
- Gate 1 (Week 1): harness frozen; Hoeffding Tree reproduced; VQR runs once.
- Gate 2 (Week 2): both quantum papers reproduced at reduced scale; classical baselines tuned.
- Gate 3 (Week 3): E1 correctness test passes; all models stream.
- Gate 4 (Week 4): all result CSVs valid; analysis complete.
- Gate 5 (Week 5): fresh-machine reproduction works; every claim has a source figure or table.

## 11. Resources

### 11.1 Software and documentation

- River (streaming ML in Python): riverml.xyz and github.com/online-ml/river. Use HoeffdingTreeClassifier, HoeffdingAdaptiveTreeClassifier, HoeffdingTreeRegressor, drift.ADWIN, evaluate.progressive\_val\_score, datasets.Elec2, synth generators.
- MOA (Massive Online Analysis, Java, original home of Hoeffding Tree implementations): moa.cms.waikato.ac.nz. Useful as a cross-check of your River results.
- scikit-learn: SVR, OneClassSVM, RandomForest, PCA, kernel SVM with precomputed kernels (kernel='precomputed'), random search.
- Qiskit and Qiskit Aer: docs.quantum.ibm.com and github.com/Qiskit/qiskit. Feature maps (ZFeatureMap, ZZFeatureMap), ansatz circuits (EfficientSU2, RealAmplitudes, TwoLocal, PauliTwoDesign), SPSA and COBYLA optimisers.
- Qiskit Machine Learning: qiskit-community.github.io/qiskit-machine-learning. VQR, QSVR, FidelityQuantumKernel, tutorials on quantum kernels and regression.
- PennyLane: pennylane.ai and pennylane.ai/qml/demos. Search the demos for quantum embedding kernels, kernel-based training of quantum models with scikit-learn, and variational classifiers. Backends: default.qubit and lightning.qubit for speed.
- Python stack: pandas, pyarrow, numpy, scipy.stats (wilcoxon), statsmodels (multiple-comparison correction), matplotlib, seaborn, tracemalloc or memory\_profiler, pytest.
- Experiment tracking and reproducibility: Git and GitHub with Actions, DVC or plain checksummed data folder, MLflow or Weights and Biases (optional), YAML configs, Makefile or run\_all.sh.
- Writing: Overleaf or Word, Zotero for references.
- Compute: laptop, university lab machine, Google Colab, Kaggle notebooks. IBM Quantum has a free plan for real-hardware runs with small monthly limits (check current allowance); hardware is optional.

### 11.2 Datasets

- Seoul Bike Sharing Demand: UCI repository, archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand.
- Electricity (ELEC2): bundled in River as datasets.Elec2; original from Harries (1999) NSW market data, also on OpenML and the MOA site.
- Anomaly detection: ODDS (odds.cs.stonybrook.edu), ADBench (github.com/Minqi824/ADBench), and the dataset list in the appendix of P3. NSL-KDD (UNB Canadian Institute for Cybersecurity) as an optional network stream.
- Synthetic stream generators in River (SEA, Hyperplane, Agrawal) for controlled drift tests.

### 11.3 Core papers (read these)

- P1: Domingos and Hulten (2000), Mining High-Speed Data Streams, KDD.
- P2: Herbst, De Maio, Brandic (2024), arXiv:2402.15542.
- P3: Tscharke, Issel, Debus (2023), arXiv:2308.00583.
- Bifet and Gavalda (2007), Learning from Time-Changing Data with Adaptive Windowing (ADWIN).
- Gama, Zliobaite, Bifet, Pechenizkiy, Bouchachia (2014), A Survey on Concept Drift Adaptation, ACM Computing Surveys.
- Gama, Sebastiao, Rodrigues (2013), On evaluating stream learning algorithms (prequential evaluation).
- Bifet, Holmes, Kirkby, Pfahringer (2010), MOA: Massive Online Analysis, JMLR.
- Montiel et al. (2021), River: machine learning for streaming data in Python, JMLR.
- Bifet, Gavalda, Holmes, Pfahringer (2018), Machine Learning for Data Streams, MIT Press (book).

### 11.4 Quantum ML background and critical reading

- Havlicek et al. (2019), Supervised learning with quantum-enhanced feature spaces, Nature.
- Schuld and Killoran (2019), Quantum machine learning in feature Hilbert spaces, PRL.
- Huang et al. (2021), Power of data in quantum machine learning, Nature Communications.
- Kubler, Buchholz, Scholkopf (2021), The inductive bias of quantum kernels.
- Cerezo et al. (2021), Variational quantum algorithms, Nature Reviews Physics.
- Thanasilp et al. (2022), Exponential concentration in quantum kernel methods.
- Bowles, Ahmed, Schuld (2024), Better than classical? The subtle art of benchmarking quantum machine learning models (arXiv:2403.07059). Read this before designing your fairness controls.
- Preskill (2018), Quantum computing in the NISQ era and beyond.
- Schuld and Petruccione (2021), Machine Learning with Quantum Computers (book).
- Spall (1992), Multivariate stochastic approximation using simultaneous perturbation (SPSA).

### 11.5 Alternate or extra papers if the instructor asks for substitutions

- Manapragada, Webb, Salehi (2018), Extremely Fast Decision Tree (arXiv:1802.08780).
- Hulten, Spencer, Domingos (2001), CVFDT (Hoeffding Tree with drift handling).
- Quantum autoencoder for multivariate time-series anomaly detection (arXiv:2504.17548).
- Quantum algorithm for sparse online learning with truncated gradient descent (arXiv:2411.03925).

## 12. Report Outline

1. Abstract.
2. Introduction and research questions.
3. Background: streaming ML, concept drift, quantum kernels, variational circuits.
4. Base papers and what we reproduced (with match and gap tables).
5. Experimental setup: data, protocol, metrics, fairness controls.
6. Enhancements E1 to E3.
7. Results: quality, cost, sample efficiency, drift.
8. Discussion: when quantum helps, when classical wins, simulator caveats, kernel concentration.
9. Threats to validity and limitations.
10. Conclusion and future work (hardware runs, noise, larger qubit counts).
11. Appendix: configs, hyperparameters, run logs, reproduction instructions.

## 13. Final Checklist

- Paper list approved.
- Contract frozen and validator in place.
- Hoeffding Tree reproduction note (match or gap, with reasons).
- P2 and P3 reproduced at reduced scale, gaps documented.
- Classical baselines tuned with equal budget.
- E1 correctness test and cost measurement; E2 and E3 results.
- All result CSVs valid; analysis notebook runs top to bottom.
- Report, slides, README, one-command reproduction verified on a clean machine.
