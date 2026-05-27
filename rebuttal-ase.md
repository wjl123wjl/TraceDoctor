We sincerely thank our referees. 

**ReviewerA:**

**Q1. Statistical tests.**
- We perform Wilcoxon signed-rank tests to assess the sensitivity of VulFaith to the choice of auxiliary model. We conduct the tests on U_C and U_P because U_O is derived from the union of the two metrics rather than an independent observation. Replacing the default auxiliary model (GPT-5.4) with Claude Opus 4.6 yields p-values of 0.903 and 0.583 for U_C and U_P, respectively. Replacing GPT-5.4 with Gemini 3.1 Pro yields p-values of 0.104 and 0.194. All p-values exceed 0.05, indicating no statistically significant differences. These results provide statistical support for our conclusion that the reported faithfulness measurements are not sensitive to the choice of auxiliary model.

**Q2. Computational and human costs.**
- We quantify the computational cost of VulFaith by measuring the usage of the auxiliary LLM throughout the evaluation process. Using GPT-5.4 as the auxiliary model, VulFaith requires 1.72 repair attempts per code input on average, corresponding to 3,481 input tokens and 2,890 output tokens (6,371 tokens in total). Based on the GPT-5.4 API pricing, this corresponds to approximately \$0.052 per code input on average.


**Q3. Baselines.**
- Baselines face limitations when applied to vulnerability detection. For example, they yield substantially lower per-factor unfaithfulness U_P values of 6.57%–14.60% across models. These methods perturb reasoning traces but do not eliminate the influence of the corresponding insecure code patterns on model predictions. As a result, the perturbation often has limited impact on the prediction outcome, making per-factor unfaithfulness more difficult to reveal.

- In addition, existing approaches do not consider completeness unfaithfulness \(U_C\). Their evaluation protocols are not designed to identify decision-relevant factors omitted from reasoning traces and are therefore not directly applicable to evaluating U_C.


**Q4. Presentation.**
- Thanks. We'll revise the final version accordingly.

**ReviewerB:**

**Q1. Similarity technique.**
- The correspondence between factors is determined by the auxiliary model. Specifically, the auxiliary model jointly examines the original code, original reasoning trace, repaired code, and regenerated reasoning trace, and determines whether a regenerated factor corresponds to an original factor, represents a newly surfaced factor, or is introduced by the repair. Therefore, factor matching does not rely on a predefined similarity score or threshold.












