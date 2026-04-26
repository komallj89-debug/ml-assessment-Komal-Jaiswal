# Part B: Business Case Analysis
## Scenario: Promotion Effectiveness at a Fashion Retail Chain

---

## B1. Problem Formulation — 8 marks

### (a) ML Problem Formulation — 3 marks

**Target Variable:** `items_sold` — the number of items sold at a given store during a given promotion in a given month.

**Candidate Input Features:**
- **Store attributes:** store size, location type (urban/semi-urban/rural), store ID
- **Promotion attributes:** promotion type (Flat Discount, BOGO, Free Gift with Purchase, Category-Specific Offer, Loyalty Points Bonus)
- **Time attributes:** month, whether it is a weekend or festival period
- **Market context:** local competition density, monthly footfall

**Type of ML Problem:** This is a **supervised regression problem** — we are predicting a continuous numerical target (items sold) given a set of input features. Since the business objective is to *choose the best promotion per store per month*, it can also be framed as a **multi-class classification problem** (predict which promotion type will yield the highest items_sold), which may be more directly actionable. The regression framing is preferable initially as it retains more information (exact predicted volumes) and allows ranking promotions by their predicted outcomes.

---

### (b) Why Items Sold Over Sales Revenue — 3 marks

**Using items sold (sales volume) as the target is more reliable than total sales revenue for several reasons:**

1. **Price variation distorts revenue:** A store running a Flat Discount promotion inherently reduces the price per item, so revenue drops even if far more items are sold. Revenue conflates promotional effectiveness with pricing strategy, making it a misleading signal.

2. **Comparability across promotions:** Comparing revenue across promotion types is fundamentally unfair — a BOGO promotion may generate high item movement but low revenue per transaction, while a high-margin Category-Specific Offer generates high revenue on fewer items. Items sold is a neutral, promotion-agnostic metric.

3. **Broader Principle — Align the Target with the Business Objective:** In real-world ML projects, the target variable must directly reflect what the business is trying to optimise. If the goal is to increase store footfall and product movement (inventory turnover), then items sold is the right metric. Revenue is a downstream financial outcome influenced by many factors outside promotion choice (markdowns, pricing team decisions, seasonal demand). Choosing the wrong proxy target leads to models that optimise for something other than what the business needs — a fundamental pitfall in applied ML.

---

### (c) Alternative to a Single Global Model — 2 marks

**Proposed Strategy: Location-Stratified Models (Segmented Modelling)**

Rather than training one model across all 50 stores, we propose training **separate models per location type** (urban, semi-urban, rural) — or even per individual store if data volume permits.

**Justification:**
- Urban stores may respond better to BOGO promotions due to high footfall and deal-seeking customers, while rural stores with loyal customer bases may respond better to Loyalty Points Bonus.
- A single global model averages these effects, producing mediocre predictions for all location types.
- A segmented approach captures location-specific promotion elasticity. This is an instance of the **"no free lunch" principle** — no single model fits all subpopulations equally well when the data-generating process differs by group.
- Alternatively, location type and store-level features can be included as interaction terms in a single model, but explicit segmentation is simpler to interpret and easier to communicate to business stakeholders.

---

## B2. Data and EDA Strategy — 10 marks

### (a) Joining Tables and Data Grain — 4 marks

**How to Join the Four Tables:**

The four raw tables — transactions, store attributes, promotion details, and calendar — should be joined as follows:

1. **Transactions + Store Attributes:** Join on `store_id`. This adds store size, location type, and other static store characteristics to each transaction record.

2. **Transactions + Promotion Details:** Join on `promotion_type` (or a promotion ID if available). This enriches each transaction with promotion-specific metadata (e.g., discount depth, eligibility rules).

3. **Transactions + Calendar:** Join on `transaction_date`. This adds weekend flags, festival flags, and any national/regional holiday markers.

**Final Join Key:** `store_id + transaction_date + promotion_type`

**Grain of the Final Modelling Dataset:** One row = one store × one month × one promotion type. Before modelling, transactions should be aggregated to this grain — summing `items_sold` across all daily transactions within that store-month-promotion combination.

**Aggregations to Perform Before Modelling:**
- Sum of `items_sold` per store per month per promotion
- Count of promotion days (how many days the promotion ran)
- Average competition density over the month
- Binary flag: did any festival occur during this period

---

### (b) EDA Strategy — 4 marks

**At least four analyses/charts before modelling:**

1. **Promotion Type vs Average Items Sold (Bar Chart):** Plot the mean items_sold for each of the five promotion types across all stores. This reveals which promotions work best on average and whether BOGO outperforms Flat Discount, etc. Finding: If Loyalty Points Bonus consistently underperforms, we might reduce its weight or exclude it from certain store segments.

2. **Items Sold by Location Type (Box Plot):** Compare the distribution of items_sold across urban, semi-urban, and rural stores. Finding: If rural stores show much lower median sales with high variance, we may need to model them separately or add more robust store-level features. This directly informs the segmented modelling strategy from B1(c).

3. **Promotion Effectiveness Heatmap (Location Type × Promotion Type):** A heatmap of average items_sold indexed by location type (rows) and promotion type (columns). Finding: Interaction effects — e.g., if BOGO works well in urban areas but poorly in rural — will be visible here and will inform feature interaction engineering.

4. **Time Series Plot of Items Sold (Monthly Trend):** Plot total monthly items sold across all stores over the three-year period. Finding: Identifies seasonality (e.g., festival months spike), long-term trend, and structural breaks. This informs the temporal train-test split strategy and suggests adding lag features or seasonal indices.

**How Findings Influence Modelling:**
- Strong seasonality → add month and festival interaction features
- High location-type variance → consider stratified models
- Outlier stores → investigate and possibly apply store-level normalisation

---

### (c) Handling 80% No-Promotion Transactions — 2 marks

**Impact on the Model:**

If 80% of transactions occur without any promotion, the dataset is heavily imbalanced with respect to the promotion feature. A naive model may learn to predict "no promotion" outcomes most of the time, as this minimises training error. This leads to poor predictions for promoted periods, precisely the scenarios the business cares most about.

**Steps to Address This:**

1. **Stratified Sampling:** Ensure train-test splits maintain the promotion-vs-no-promotion ratio using stratified splitting so neither split is over-represented by no-promotion records.

2. **Separate Modelling:** Consider building a model trained *only on promoted transactions* to understand promotion effectiveness in isolation. A no-promotion baseline can be estimated separately and subtracted to compute the **promotion uplift**.

3. **Uplift Modelling / Treatment Effect Estimation:** Frame the problem as a causal inference task — estimate the incremental effect of each promotion type over the no-promotion counterfactual — rather than directly predicting raw items_sold.

4. **Weighted Loss Functions:** If retaining all data, assign higher sample weights to promoted transactions during model training so the model prioritises accuracy in those scenarios.

---

## B3. Model Evaluation and Deployment — 12 marks

### (a) Train-Test Split Strategy and Metrics — 4 marks

**Setting Up the Train-Test Split:**

With three years of monthly store-level data (50 stores × 36 months), the temporal split is critical. We should use the **most recent time period as the test set** — for example, the final 6 months (Year 3, months 7–12) as the test set and the preceding 30 months as training data. An even more rigorous approach is **walk-forward validation**: train on Year 1, test on first month of Year 2; retrain on Year 1 + tested month, test on next month — repeating until all of Year 3 is evaluated.

**Why Random Split is Inappropriate:** A random split would train on Year 3 data and test on Year 1, allowing the model to learn from the future. This inflates performance metrics and produces an unreliable estimate of real-world accuracy.

**Evaluation Metrics and Business Interpretation:**

| Metric | Formula | Business Interpretation |
|--------|---------|------------------------|
| **RMSE** | √(mean squared error) | Average error in predicted items_sold; penalises large mistakes more heavily — useful when large prediction errors (e.g., severely understocking) are very costly |
| **MAE** | Mean absolute error | Average absolute deviation in items_sold; intuitive for operations teams ("we're off by X items on average") |
| **R² (R-squared)** | 1 - SS_res/SS_tot | Proportion of variance in items_sold explained by the model; 0.75 means the model explains 75% of variation — a business-friendly summary metric |
| **Promotion Uplift Accuracy** | Custom metric | Whether the model correctly ranks promotions — i.e., does the model correctly identify the best promotion for a store even if absolute predictions are imperfect |

---

### (b) Explaining Different Recommendations for Store 12 — 4 marks

**Why Store 12 gets Loyalty Points Bonus in December but Flat Discount in March:**

Using **feature importance analysis**, we would investigate the following:

1. **Extract the top features driving December vs March predictions for Store 12.** Using SHAP (SHapley Additive exPlanations) values or the model's built-in feature importances, we can compute which features pushed the prediction toward each promotion type in each month.

2. **December likely shows:** High `is_festival` flag (Christmas/year-end), high `days_since_last_visit` suggesting customer re-engagement potential — conditions where Loyalty Points Bonus rewards returning customers and drives repeat purchases.

3. **March likely shows:** Lower festival activity, higher `competition_density` in the area — conditions where a direct price incentive (Flat Discount) is needed to attract price-sensitive customers away from competitors.

**Communicating to the Marketing Team:**

Prepare a simple summary table:

| Factor | December | March |
|--------|----------|-------|
| Festival Period | Yes | No |
| Competition Density | Low | High |
| Customer Recency | Moderate | High churn risk |
| Recommended Promotion | Loyalty Points Bonus | Flat Discount |
| Reason | Reward loyal base during peak season | Price competition pressure requires direct discount |

This narrative explanation, grounded in feature importance, translates the model's statistical reasoning into business logic that the marketing team can validate and trust.

---

### (c) End-to-End Deployment Process — 4 marks

**Step 1 — Save the Trained Model:**
Serialise the trained scikit-learn pipeline (including all preprocessing steps) using `joblib.dump(pipeline, 'promotion_model_v1.pkl')`. Store this artefact in a version-controlled model registry (e.g., MLflow, AWS S3 with versioning) alongside the training data snapshot and model performance metrics.

**Step 2 — Prepare Monthly Input Data:**
At the start of each month, the data engineering team runs an automated pipeline that: (i) extracts the latest store attributes and calendar flags, (ii) joins them to form one row per store for the upcoming month, (iii) validates the schema and checks for missing values or data drift. The resulting input dataframe is stored in a staging table.

**Step 3 — Generate Recommendations:**
The inference pipeline loads the saved model, passes the staged monthly data through it, and generates predicted items_sold for all five promotion types for each of the 50 stores. The promotion with the highest predicted items_sold is selected as the recommendation for that store. Output is written to a dashboard or sent as a structured report to the marketing team.

**Step 4 — Monitoring for Model Degradation:**

The following monitoring checks are implemented:

- **Prediction Drift:** Track the distribution of predicted items_sold over time. If the distribution shifts significantly (measured by Population Stability Index or KL-divergence), alert the team.
- **Actual vs Predicted Tracking:** Each month, once actual results are available, compute RMSE and MAE for that month's predictions. If rolling 3-month MAE exceeds a defined threshold (e.g., 20% above baseline), trigger a retraining workflow.
- **Feature Drift Detection:** Monitor input feature distributions (e.g., competition_density, store size mix) using statistical tests. Changes in the input distribution suggest the model was trained on a different data regime and may no longer generalise.
- **Scheduled Retraining:** Regardless of drift alerts, retrain the model quarterly by appending the latest 3 months of actuals to the training set, ensuring the model stays current with evolving consumer behaviour and competitive dynamics.
