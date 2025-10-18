# CS698Y-assignment2

1. Train a simple model for your selected task; you may use libraries such as scikit-learn,
   pandas, numpy, etc.
2. Before training, conduct a bias evaluation:
   • Identify potential biases in the dataset (sampling, historical, labeler, measurement, etc.),
   identify its source.
   • Analyze whether certain groups are over- or under-represented or disproportionately
   affected, and how you can address it.
3. Implement corrective measures, to mitigate bias. Examples include:
   • Feature elimination: Remove features that encode sensitive attributes or proxies
   • Reweighting or resampling to balance representation
   • Calibration techniques to adjust predictions across groups
4. Fairness Evaluation:
   • Evaluate your model’s performance overall and across subgroups.
   • Use metrics such as accuracy, precision, recall, F1 score as appropriate.
   • Use fairness metrics (e.g., statistical parity, equal opportunity), justify them.
5. Document these metrics before and after implementing corrective measures.

### Installation

```bash

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
select python interpreter as venv/bin/python (in VS code)

```



### Dataset and References

- [DATASET (UCI Repository)](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)
- [Early Prediction of Student’s Performance in Higher Education: A Case Study](https://link.springer.com/chapter/10.1007/978-3-030-72657-7_16)
- [Predicting Student Dropouts with Machine Learning: An Empirical Study in Finnish Higher Education](https://www.sciencedirect.com/science/article/pii/S0160791X24000228)
