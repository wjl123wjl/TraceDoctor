We sincerely thank our referees. We have open-sourced our code, data, and prompts at: https://github.com/Trace-Doctor/TraceDoctor.

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
To showcase the resulting error categorization quality, we release the 29 identified types in our GitHub repository (29.md).

We determine the number of categories by identifying semantically different types of errors. As described in **Error categorization** (Section III-B), the LLM first proposes candidate error types. Then, we perform
embedding-based similarity analysis to identify and remove semantically overlapping types. This process naturally determines the number of categories without manual tuning the category granularity. We do not explicitly tune the category granularity, as different datasets may exhibit different characteristic, and we aim an automatic and general approach. 


