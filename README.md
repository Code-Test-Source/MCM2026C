# MCM2026C
Coding of MCM 2026C

## Collaboration
### Directory structure
```
MCM2026C/
├── README.md
├── data/
│   ├── raw/                # Raw data files
│   ├── result/                # Result data files
│   └── processed/          # Processed data files
├── src/
│   ├── data/             # Data processing scripts
│   ├── model/           # Model training and evaluation scripts
|   ├── plot/     # Visualization scripts
|   ├── test/     # Testing scripts
│   └── analysis/            # analysis scripts
```
### Collaboration Guidelines
```plaintext
git pull # To fetch and merge changes from the remote repository
git add <file> # To stage changes for commit
git commit -m "docs:update readme" # To commit staged changes with a message
git push # To push committed changes to the remote repository
```
Commit message:
 - feat: new feature
 - fix: bug fix
 - docs: documentation update

### Logical Workflow

First delete all the example files in the directory when you push your first actual files in the directory.
1. Data Collection and Preprocessing
   - Collect raw data and store it in the `data/raw/` directory.
   - Write data preprocessing scripts in `src/data/` to clean and transform the raw data.
   - Save processed data in the `data/processed/` directory.
2. Model Development
   - Implement model training and evaluation scripts in `src/model/`.
   - Use processed data from `data/processed/` for training and testing the models.
   - Store model results in the `data/result/` directory.
3. Visualization
   - Create visualization scripts in `src/plot/` to generate plots and charts based on model results.
   - Save generated visualizations in an appropriate directory (e.g., `data/result/`).
4. Analysis
   - Write analysis scripts in `src/analysis/` to interpret model results and derive insights.
   - Regularly commit and push changes to the remote repository to keep the project up-to-date.
5. Test
   - Implement testing scripts in `src/test/` to try new ideas and without disturbing the model folder.
   - Ensure that all new code is tested before integrating it into the main workflow.