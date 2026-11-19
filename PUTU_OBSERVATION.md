# Feedback on Baseline Model & Optuna Tuning (`04_baseline_model.ipynb` & `05_tune_baseline.ipynb`)

**To:** Ardelle  & Revel
**From:** Putu  
**Subject:** Code Review & Methodological Feedback on Baseline SVM & Optuna Tuning

---

Hi Ardelle,

I was reviewing your notebooks for the baseline SVM model and the Optuna hyperparameter tuning. 

In your conclusion in `05_tune_baseline.ipynb`, you noted that the default baseline scored **0.9269** while the Optuna-tuned model scored slightly lower at **0.9256**, concluding that hyperparameter tuning didn’t improve performance. 

I took a closer look at the code, and it turns out **your Optuna optimization actually worked and improved the model to ~0.9288 Macro F1**. However, a scoping bug in the final evaluation cell caused the notebook to evaluate the wrong model configuration. 

Here is what is happening and what you need to adjust:

---

### 1. The Variable Scope Bug in `05_tune_baseline.ipynb` (Cell 7)

During the final evaluation step on the test set, you properly re-initialized the SVM using `best_params`:
```python
best_svm = SVC(
    C=best_params['C'],
    kernel=best_params['kernel'],
    gamma=best_params['gamma'],
    random_state=42
)
```

However, you didn't re-initialize `tfidf` with the tuned parameters (`ngram_range`, `min_df`, `max_df`):
```python
# Here, Python silently reused the default `tfidf` from Cell 3!
X_train_val_vec = tfidf.fit_transform(X_temp)
X_test_final_vec = tfidf.transform(X_test)
```

Because `tfidf` inside your `objective()` function was a local variable, Cell 7 ended up using the global `TfidfVectorizer()` with default settings (`min_df=1`, `max_df=1.0`). In other words, you evaluated:
* **What you intended to evaluate:** Tuned TF-IDF + Tuned RBF SVM
* **What actually got evaluated:** Default TF-IDF + Tuned RBF SVM $\rightarrow$ **0.9256**

When you properly instantiate `TfidfVectorizer` with `best_params` before transforming the test data:
* **Tuned TF-IDF + Tuned RBF SVM:** $\rightarrow$ **0.9288** *(this successfully outperforms your baseline)*

---

### 2. Methodological Observations for Your Thesis

Aside from the bug fix, there are a couple of points regarding your methodology that you should address before we move on to deep learning:

1. **Comparing Linear vs. RBF Kernels:**
   In Notebook 04, your baseline was `SVC(kernel='linear')`. In Notebook 05, Optuna selected `kernel='rbf'`. Text data with TF-IDF is already high-dimensional and typically linearly separable. When you pair an RBF kernel with default TF-IDF (without the `min_df=4` filter), it easily overfits noisy features. It would be cleaner to either tune `LinearSVC` and `SVC(rbf)` separately, or report both so the comparison is apples-to-apples.

2. **Optuna Exploration in 20 Trials:**
   Looking at your tuning history in `svm_tuning.sqlite3`, Optuna only tested `linear` 6 times, and in all 6 instances it was paired with `bigram` (which yielded lower scores). The sampler prematurely converged to `rbf` and never evaluated `linear` with `unigram`. Consider increasing the trial count (e.g., 50 trials) or tuning Linear SVM explicitly.

3. **Validation Strategy:**
   Currently, Optuna optimizes on a single 16% validation split (5,585 samples). The difference between trials is tiny ($\pm 0.0005$, or just 2–3 reviews). To make your pre-thesis findings methodologically sound and defensible during examination, use **Stratified 5-Fold Cross-Validation** inside the objective function instead of a single split.

---

### Next Steps

1. In Cell 7 of `05_tune_baseline.ipynb`, rebuild the vectorizer before fitting:
   ```python
   ngram_param = (1, 1) if best_params['ngram_range'] == 'unigram' else (1, 2)
   best_tfidf = TfidfVectorizer(
       ngram_range=ngram_param,
       min_df=best_params['min_df'],
       max_df=best_params['max_df']
   )
   X_train_val_vec = best_tfidf.fit_transform(X_temp)
   X_test_final_vec = best_tfidf.transform(X_test)
   ```
2. Re-run the final test evaluation and update your Markdown commentary with the corrected results.
3. Once that is validated, let's schedule time to discuss your results and next steps.