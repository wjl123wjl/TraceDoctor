We sincerely thank our referees. 

**ReviewerA:**

**Q1. Prioritizing high‑frequency errors & impact of variant count per error type.**
- We analyze error types from a known error set. However, we avoid prioritizing high-frequency errors within this set, as its distribution may differ from that of real-world test scenarios. Low-frequency error types may become more prominent in unseen environments. Our approach is flexible: users can filter out low-frequency errors and generate variants only for high-frequency types if desired.

- The number of variants per error type can influence performance improvements, as more training data typically leads to better model performance. While allocating more variants to specific errors may further improve results, we limit the total number of generated variants to ensure a fair comparison. Specifically, the fine-tuning data volume for our method does not exceed that of any baseline. Additionally, we allocate approximately equal numbers of variants per error type, assuming no prior knowledge of real-world error distributions.



**Q2. Parsing errors.**
- Parsing errors are defined as violations of the expected template structure, which cause logically identical logs to map to different templates. This structural inconsistency introduces noise that degrades downstream tasks. For example, if some instances treat a field as one variable and others as two, the logically identical logs yield different templates. This inconsistency disrupts anomaly detection models that rely on template frequencies or sequences. Logically identical logs may be mapped to different templates, which introduces noise into the training data and reduces detection accuracy.


**Q3. Threshold.**
- We experimented with thresholds of 0.8, 0.85, 0.9, and 0.95. Based on manual inspection, a threshold of 0.9 achieved the best performance. Lower thresholds such as 0.8 tended to over-merge conceptually different errors, reducing interpretability. A higher threshold of 0.95 failed to group near-duplicate errors that differed only slightly in phrasing. 




**ReviewerB:**

**Q1.Data leakage & .**
- We fine-tune using only 50 logs from LogHub-2k, while LogHub-2.0 contains over 40 million logs. We ensure there is no data leakage by de-duplicating these 50 logs from the evaluation set in LogHub-2.0. We'll clarify this.

- More importantly, we compare against directly fine-tuning on the same 50 logs without any variant generation (Table III, Non-Aug in Section IV-C). The results show minimal performance improvement, indicating that these logs alone are insufficient to enhance model quality, further confirming no leakage or unfair advantage.


**Q2.Fairness of fine-tuning data selection.**
- To address concerns about inconsistent data selection, we also substitute 50 logs in the baselines’ training data with the same 50 logs used by TraceDoctor. The baselines still underperform significantly. Combined with the Non-Aug results, this shows that our improvements are not due to specific log selection, but the effectiveness of our variant generation and error-driven fine-tuning strategy.


**Q3.No ground truth templates.**
- Fine-tuning-based log parsing methods inherently require labeled data (i.e., templates) during training. TraceDoctor follows this paradigm. In fact, most existing log parsing approaches, including non-fine-tuning methods, typically assume access to templates. Importantly, templates are only required during training. Once the model is fine-tuned, it can be deployed in real-world scenarios without relying on templates. We leave template-free fine-tuning as future work.

**Q4.Verifiability.**
- Our code is in:xxx, and prompts are in xxx.py, line xxx: 





**ReviewerC:**

**Q1.Prompt&Code.**
- Our code is in:xxx, and prompts are in xxx.py, line xxx: 

**Q2.Selection of generation strategies.**
- Log entries are composed of twocomponents: variables and constants.
Thus, we designed three strategies to manipulate each component type (variables or constants), plus one that alters the entire semantic structure: (1) Variable substitution: Modifies only instance-level values while preserving the syntactic and semantic template. (2) Constant-inclusive Rewriting: Rewrites selected constants based on semantic surprisal, enabling deeper but still semantically faithful variation.
(3) Semantic rewriting: Goes beyond structural edits to perform counterfactual rewriting, allowing broader exploration of error-triggering patterns beyond fixed templates.
Together, these three strategies span the full spectrum: from localized edits to high-level semantic changes. This design ensures high coverage of potential error triggers while maintaining the interpretability and quality of the generated logs.


- We considered and tested additional strategies, such as: Random token corruption: often produced invalid or unparseable logs with limited utility. Unconstrained log generation: lacked alignment with target error types and frequently generated irrelevant content.
Preliminary experiments showed that these alternatives failed to produce effective fine-tuning signals, whereas our proposed strategies generated targeted, high-quality variations with significantly better error recall and parser improvement. Thus, we excluded these alternatives based on empirical performance and control concerns.
We will clarify this in final version.



**Q3. Rationale for choosing DeepSeek-V3-0324 in Analyzer.**
- We adopt a commercial LLM in Analyzer and select DeepSeek-V3-0324 due to its strong performance, low cost, and fast response time. Our experiments show that most commercial LLMs are capable of understanding and summarizing natural language reasoning traces. For instance, we tested ChatGPT-4o and found that it produced nearly identical error types to DeepSeek-V3-0324. This indicates that the results are not sensitive to the choice of commercial LLMs. We ultimately choose DeepSeek-V3-0324 over ChatGPT-4o due to its lower cost and faster response time.

**Q4.Categories.**
- We do not manually specify the number of error categories. Instead, we adopt an automated strategy described in Section III-B, where the LLM first proposes a broad set of candidate types, and we then apply a semantic filtering process to remove redundant ones based on pairwise similarity. The final number of categories (29) is automatically derived through this process. Rather than tuning for a fixed category count, our method ensures semantic distinctiveness between retained types, thereby avoiding both over-fragmentation and over-merging. This mechanism implicitly controls the trade-off between granularity and performance
