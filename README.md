# Effect-of-Sparsity-in-Explaining-IT-VIsual-Cortex

### Preprocessing Scripts:
1. prep_data_tinyimagenet.py
2. generate_image_embs.py
3. prepare_bold5000.py

### Reproduce Results Scripts:
1. train.py: Script to train the VGG-16 model on TinyImageNet dataset
2. calculate_ridge_results.py: Script to calculate results for the Ridge Regression model and evaluate on 2vs2 and R2 score.
3. calculate_ridge_results_filtered.py: Script to calculate results for the Ridge Regression model and evaluate with filtering on 2vs2 and R2 score.
4. calculate_rsa_results.py: Calculate RDM cosine similarity scores.
