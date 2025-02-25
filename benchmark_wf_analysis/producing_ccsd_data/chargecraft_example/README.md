Because qcarchive can't calculate electronic properties for post-HF methods, we use our bespoke program
`ChargeCraft` to run this calculation. The following files are in this directory:

1. *properties_example.py*: the following contains the necassary script to run a chargrecraft calculation
on a small set of compounds to produce a set of CCSD properties. This file reads smiles from `set.smi` compounds. 