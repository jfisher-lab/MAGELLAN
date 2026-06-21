# Srivatsan et al. 2019 Benchmark Dataset

We benchmark MAGELLAN on the dataset from [Srivatsan et al. 2020](https://www.science.org/doi/full/10.1126/science.aax6234). 

## Download the data
To generate the dataset, download the Supplementary Tables 3 and 5 from [Srivatsan et al. 2020](https://www.science.org/doi/full/10.1126/science.aax6234) and move them to the `benchmarks/bio_benchmarks/data/srivatsan` directory.

```bash
cd benchmarks/bio_benchmarks/data/srivatsan
mv aax6234-srivatsan-table-s5.txt Supplementary_Table_5.txt
mv aax6234-srivatsan-table-s3.txt Supplementary_Table_3.txt
```

To generate the dataset for from scratch, download the raw data from [NCBI GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE139944), specifically unpack the files:

| File type        | File                                                                |
| ---------------- | ------------------------------------------------------------------- |
| Counts           | `GSM4150378_sciPlex3_A549_MCF7_K562_screen_UMI.count.matrix`        |
| Gene annotations | `GSM4150378_sciPlex3_A549_MCF7_K562_screen_gene.annotations.txt` |
| Cell annotations | `GSM4150378_sciPlex3_A549_MCF7_K562_screen_cell.annotations.txt` |
| pData            | `GSM4150378_sciPlex3_pData.txt`            |

Then run the `1_gen_ann_data.py` script to generate the AnnData object.
```bash
uv run benchmarks/bio_benchmarks/data/srivatsan/utils/raw_counts/1_gen_ann_data.py
```

This will generate the `sciPlex3_MCF7_raw_counts.h5ad` file in the `ann_data` directory.

## Process the data

### Process for use by MAGELLAN

#### Clean the data

Supplementary Table 5 contains a header that is split across two lines and rows that break the number of columns.

```bash
uv run benchmarks/bio_benchmarks/data/srivatsan/utils/1_clean_srivatsan_table_5.py
```

#### Filter the data

As we are comparing to our BRCA model, we need to filter the data for the MCF7 cell lines, as well as genes that have corresponding nodes in the BRCA model.

```bash
uv run benchmarks/bio_benchmarks/data/srivatsan/utils/2_filter_srivatsan_table_5.py
```

#### Generate drug to perturbation mapping

The dataset measures the response to drugs, to simulate this in our model, we need to generate a drug to perturbed gene mapping.

We use the PubChem and ChEMBL APIs to generate a drug to perturbation mapping.

```bash
uv run benchmarks/bio_benchmarks/data/srivatsan/utils/3_annotate_chembl_ids.py
```

#### Convert to specification format

We filter the data for drugs that perturb nodes that are in the model AND where observations are made of nodes in the model. We assume that all drugs have maximal effect, mapping to a perturbation assuming a model granularity of 2:

 - INHIBITOR or ANTAGONIST = 0
 - ACTIVATOR or AGONIST = 2
 - Other = NA

We make a special case for TOREMIFENE CITRATE (CHEMBL1200675) as it is known to act as an antagonist in breast cancer (https://pubmed.ncbi.nlm.nih.gov/31643660/). 

We assume that genes with significant (q<0.05) and positive effect (normalized_effect>0) are upregulated, and genes with significant (q<0.05) and negative effect (normalized_effect<0) are downregulated, translated to our network expectation as 2 and 0 respectively. 

We also assume that basal activity as described in our literature curated specification is present (MCF7.basal) unless overridden by a drug perturbation.

```bash
uv run benchmarks/bio_benchmarks/data/srivatsan/utils/4_convert_to_spec_format.py
```

This will generate the `srivatsan_spec_master.csv` file in the `spec` directory.

## Run benchmark of MAGELLAN vs Srivatsan et al. 2019

We train a model on our literature curated specification and the Srivatsan et al. 2019 dataset, with a 20% holdback test set, entirely from Srivatsan et al. 2019 to test if the model is able to predict gene level perturbation responses, as well as the primarily overall behaviour response in our literature curated specification.

To train and evaluate the model, run the `train_bio.py` script.

```bash
uv run scripts/train_bio.py --config scripts/example_configs/brca_train_bio_config_lit_srivatsan.toml
```

This will train the model and evaluate it on the test set.

The results will be saved in the `results` directory. You can evaluate the model from the "metrics_summary_test.txt" file in the `results` directory.
