We sincerely thank our referees. 

**ReviewerA:**

**Q1. Threshold.**
- We experimented with thresholds of 0.85, 0.9, and 0.95. Based on manual inspection, a threshold of 0.9 achieved the best performance. Lower thresholds like 0.85 tend to over-merge distinct error types. For example, (1) Pattern Over-segmentation Issues: The model excessively splits logs into overly fine-grained templates, losing structural integrity. (2) Pattern Under-consolidation Problems: The model fails to group structurally similar logs into unified templates.
These two are semantically different but have a similarity score of 0.87. A threshold of 0.85 would incorrectly merge them.



**Q2. Prioritizing high‑frequency errors & impact of variant count per type.**
- We analyze error types from a known error set. However, we avoid prioritizing high-frequency errors within this set, as its distribution may differ from that of unseen test scenarios. Low-frequency error types may become more prominent in unseen test scenarios. Our approach is flexible: users can filter out low-frequency errors and generate variants only for high-frequency types if desired.

- The number of variants per error type can influence performance improvements, as more training data typically leads to better model performance. While allocating more variants to specific errors may further improve results, we limit the total number of generated variants to ensure a fair comparison. Specifically, the fine-tuning data volume for our method does not exceed that of any baseline. Additionally, we allocate approximately equal numbers of variants per error type, assuming no prior knowledge of  error distributions of test scenarios.

**Q3. Error impacts.**
- Parsing errors are defined as violations of the expected template structure, which cause logically identical logs to map to different templates. This structural inconsistency introduces noise that degrades downstream tasks. For example, if some instances treat a field as one variable and others as two, the logically identical logs yield different templates. This inconsistency disrupts anomaly detection models that rely on template frequencies or sequences. Logically identical logs may be mapped to different templates, which introduces noise into the training data and reduces detection accuracy.



**ReviewerB:**

**Q1.Data leakage.**

- To mitigate potential data leakage, we select only 50 logs per system. This is significantly fewer than the 200 logs used by baselines. These 50 logs also cover fewer templates, with an average of 14, compared to 25~31 templates in baselines. In contrast, the evaluation set LogHub-2.0 contains over 40 million logs and on average 249 templates per system. This significant discrepancy in both data volume and template coverage makes it unlikely that potential leakage would influence model performance.
Additoinally, we explicitly remove any fine-tuning logs from LogHub-2.0 before evaluation, ensuring that no overlap exists between the fine-tuning and evaluation data. We will clarify this in final version.

- More importantly, we include an ablation (Non-Aug in Table III) that fine-tunes models directly on these 50 logs without any variant generation. Results show that Non-Aug performs substantially worse than TraceDoctor, indicating that these logs alone are insufficient to improve model performance and are unlikely to confer unfair advantage.


**Q2.Fine-tuning data selection.**
- To address concerns about inconsistent data selection, we also substitute 50 logs in the baselines’ training data with the same 50 logs used by TraceDoctor. TraceDoctor consistently outperforms all baselines. For example, on DeepSeek-LLaMA3-8B, TraceDoctor achieves an average PA of 0.889 and GA of 0.848, while the baselines perform as follows:
(1) LogParser: PA 0.790, GA 0.761 (2) Superlog: PA 0.746, GA 0.615 (3) LogLLM: PA 0.795, GA 0.787.

- It is worth noting that baselines are exposed to more data (200 logs per system) and more diverse templates (25–31 on average) than TraceDoctor (50 logs and 14 templates). Some baselines explicitly aim to increase template diversity. For example, LogParser adopts clustering to cover diverse templates. In contrast, TraceDoctor selects logs based on model errors, not template coverage.


**Q3.No ground truth templates.**
- Fine-tuning-based methods inherently require labels (templates) during training. TraceDoctor follows this paradigm. In fact, most existing parsing approaches, including non-fine-tuning methods, typically assume access to templates. Importantly, templates are only required during training. Once the model is fine-tuned, TraceDoctor can be deployed in real-world scenarios without relying on templates. We leave template-free fine-tuning as future work.

**Q4.Verifiability.**
- TraceDoctor is available: https://github.com/Trace-Doctor/TraceDoctor.




**ReviewerC:**

**Q1.Code&data&prompts.**
- All code, data and prompts are available: https://github.com/Trace-Doctor/TraceDoctor. 

**Q2.Selection of generation strategies.**
- We design these strategies to cover all components within a log that can be modified, including variables, constants, structure, and semantics. Based on this coverage, alternative variant generation methods can be viewed as subsets or combinations of our strategies.
Specifically, logs typically consist of variables and constants. To target these elements, we introduce two strategies: Variable substitution and Constant-inclusive rewriting. However, both strategies operate under the constraints of the original log structure and semantics. To enable broader perturbation beyond such constraints, we propose a third strategy, Semantic rewriting, which allows altering the overall semantics and structure of a log.
Together, these three strategies comprehensively span the key dimensions along which logs can be modified. Other generation approaches typically fall within this space and can be considered subsets or compositions of our strategies. We will clarify this in final version.

**Q3. Rationale for choosing DeepSeek-V3-0324 in Analyzer.**
- We adopt a commercial LLM in Analyzer and select DeepSeek-V3-0324 due to its strong performance, low cost, and fast response time. Our experiments show that most commercial LLMs are capable of understanding and summarizing natural language reasoning traces. For instance, we tested ChatGPT-4o and found that the results are not sensitive to the choice of commercial LLMs. For example, ChatGPT-4o produced nearly identical error types to DeepSeek-V3-0324. We ultimately choose DeepSeek-V3-0324 over ChatGPT-4o due to its lower cost and faster response time.

**Q4.Categories.**
- We determine the number of categories by identifying semantically different types of errors. As described in **Error categorization** (Section III-B), the LLM first proposes candidate error types. We then perform embedding-based similarity analysis to identify and remove semantically overlapping types. This process naturally determines the number of categories without manually tuning the category granularity. We do not explicitly control the granularity, as different datasets may exhibit different characteristics, and our goal is to develop an automatic and generalizable approach.
To showcase the resulting error categorization quality, we release the 29 identified types in our GitHub repository (high-level-error-types.md).




