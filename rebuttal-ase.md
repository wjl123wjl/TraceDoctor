We sincerely thank our referees. 

**ReviewerA:**

**Q1. Prioritizing high‑frequency errors & impact of variant count per error type.**
- We analyze error types from a known error set. However, we avoid prioritizing high-frequency errors within this set, as its distribution may differ from that of real-world test scenarios. Low-frequency error types may become more prominent in unseen environments. Our approach is flexible: users can filter out low-frequency errors and generate variants only for high-frequency types if desired.

- The number of variants per error type can influence performance improvements, as more training data typically leads to better model performance. While allocating more variants to specific errors may further improve results, we limit the total number of generated variants to ensure a fair comparison. Specifically, the fine-tuning data volume for our method does not exceed that of any baseline. Additionally, we allocate approximately equal numbers of variants per error type, assuming no prior knowledge of real-world error distributions.



**Q2. Impact of Parsing Errors.**
- Parsing errors are defined as violations of the expected template structure, which cause logically identical logs to map to different templates. This structural inconsistency introduces noise that degrades downstream tasks. For example, if some instances treat a field as one variable and others as two, the logically identical logs yield different templates. This inconsistency disrupts anomaly detection models that rely on template frequencies or sequences. Logically identical logs may be mapped to different templates, which introduces noise into the training data and reduces detection accuracy.





- More importantly, we focus on exploring the role of CFGs. It's crucial to distinguish CFG features from semantic features to clearly ascertain the role of CFGs. In contrast, related works can only identify significant function sub-parts encompassing a plethora of instructions. The conflation of CFGs with semantic features cannot reflect the independent role of CFGs.

- Explaining ML-BFSD solutions diverges notably from explaining malware classifiers. Malware classifiers, based on classification tasks, produce outputs compatible with various explanation methods like CFGExplainer. Conversely, ML-BFSD solutions, rooted in contrastive learning, yield multi-dimensional feature vectors. These vectors are not readily suited for explanation methods used in malware classification. We claim this challenge in Section 3.2 (**Labeling training data**).

- We'll discuss and cite the recommended papers.

**Q3. $\delta$CFG.**
- When basic blocks have the same number of direct successors, 
we compute the path counts from these successors to the final block in CFGs. Blocks corresponding to successors with more paths are prioritized for matching. In cases where path counts are equal, blocks are randomly chosen for matching. Experimental results show that we can effectively manipulate all function pairs.


- In most cases, the empty basic blocks are not disregarded. However, in a few rare instances, due to features with IDA disassembler, some empty basic blocks might disappear. To address this, we employ **grakel.kernels.WeisfeilerLehmanOptimalAssignment** to check whether the modified CFGs become identical/different. This approach filters out a few anomalies, ensuring function pairs' CFGs are manipulated as expected.


**ReviewerB:**

**Q1.What features now ML-BFSD rely on.**
- Table 11 reveals enhanced models have become more reliant on semantic features, and the over-reliance on CFGs has been mitigated. However, we do not wish for any particular semantic feature to be overly relied upon, as this could make feature brittle and lead to errors. Results show the importance of each semantic feature has increased, with varying degrees across different models. Currently, there is no evidence of over-reliance on any specific semantic feature, suggesting they are not as brittle as CFG features, which aligns with our desired outcome.

**Q2.Convincing with experiments&robustness.**
- We appreciate your suggestion and will design experiments to make our conclusions more convincing. E.g., we intend to improve the feature representation method in GEMINI and then observe the importance of these features. Additionally, we will devise new experiments to discuss the robustness of semantic features. E.g., we plan to use obfuscating instructions to equivalently replace existing instructions and observe their impact on models. 

**Q3.Why relying on them might be wrong.**
- Over-reliance on CFGs implies that GNNs focus on learning the relationships between connected basic blocks, or those indirectly connected but spanning fewer blocks, which are depicted by CFGs. They struggle to learn relationships between basic blocks that are not directly connected and span multiple blocks, impacting their ability to understand the semantics of functions. E.g., there might be data dependency relations between non-directly connected basic blocks, which CFG cannot represent. The over-reliance hinders GNNs from learning semantics.

**Q4.Introduction&DEXTER.**
- We'll revise introduction and evaluate DEXTER.

**Q5.Additional comments.**
- Yes, compilation variants.

- In Table 3, "0~20%" range represents the similarity score decrease ratio, not the proportion of function pairs. E.g., on jTrans, 97.4% of function pairs show a similarity score decrease within 20%, indicating that most pairs remain relatively unchanged despite CFG modifications. This demonstrates jTrans's insensitivity to these changes, suggesting a limited reliance on CFGs. Conversely, in "20%~40%" range, only 1.8% of pairs exhibit this level of decrease, highlighting that significant changes in similarity scores are rare and likely due to model inherent errors, without impacting main conclusions. 

- We'll refer to and cite the recommended work and revise Section 5.3. Previous works[52,70] show that metrics such as F-1 and ROC cannot adequately represent solution performance in real-world applications.



**ReviewerC:**

**Q1.Feature.**
- Our Explainer is based on LIME and LEMNA. They rely on model output changes by randomly deleting instructions. In this context, each instruction is represented by a binary value of 0 (removed) or 1 (retained). This binary representation is a specific requirement of these two explanation schemes. LIME and LEMNA necessitate that each dimension of the representation vector be a single scalar value, prohibiting the use of high-dimensional vectors, like embeddings, for instruction representation. Also, GNN embedding values of CFGs cannot be used.

- Due to the requirement for scalar values in each dimension of the vector, many abstract features (e.g. data flow dependencies) are hard to represent. Additionally, these methods require a fixed order of vector elements and do not support altering the sequence of elements. This means that observing the impact of changes in instruction order directly is hard. We focus on adapting existing explanation schemes to ML-BFSD. Improving existing explanation methods is not within our current scope and is left for future work.

**Q2.Table2&downstream tasks.**
- Explanation methods typically focus on relative importance scores rather than absolute values, emphasizing the comparative significance of features. E.g., if feature A has a score of 0.2 and feature B scores 0.4, it indicates that feature B is more important than A, rather than being exactly twice as important. Our use of LIME and LEMNA models for fitting may have influenced these outcomes. We plan to use more complex models for fitting and correct writing flaws. Additionally, we plan to apply the augmentation technique to real-world downstream applications.

**ReviewerD:**
- BinaryCorp-3M, derived from official ArchLinux packages and Arch User Repository, encompasses about 2000 projects, 10000 binaries, and 3.5 million functions. It is among the most extensive BCSD datasets available, notable for its function count, function type diversity, and project source variety. Unlike other datasets typically limited to a few projects, BinaryCorp-3M's functions span nearly 2000 projects, offering a closer representation of real-world scenarios. Results are expected to generalize to others.

- Sorry for confusion. We'll clarify D measures cosine similarity; a high D indicates that x and x' are similar.



