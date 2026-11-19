This is a very common question when reading fact-checking code.

In short: **We are NOT doing traditional neural network training (we are not updating Transformer weights with backpropagation or gradient descent). Instead, we are doing THRESHOLD CALIBRATION on the training split, and then doing INFERENCE on the testing split.**

Let's break down exactly what the data looks like, what "training" means here, and how testing is performed in [Complete_Major_Project_Benchmark_Evaluation.ipynb](file:///home/user/snap/Downloads/MAJOR%20PROJECT/Complete_Major_Project_Benchmark_Evaluation.ipynb).

---

### 1. How Our Data Model Looks Like

We download the official **SciFact dataset** from AllenAI. It provides two separate sets of data:
1. `claims_train.jsonl` $\rightarrow$ **Training Split** (used for calibration: 100 claims)
2. `claims_dev.jsonl` $\rightarrow$ **Testing Split** (used for evaluation: all 338 claims)

Every single item in both sets has this exact structure:

```python
{
    "id": 1,
    "claim": "1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants...",
    "gold_label": "SUPPORTS",   # Can be SUPPORTS or REFUTES
    "evidence_pieces": [
        {
            "id": "e_0",
            "text": "Genome-wide association studies have identified common variants associated with complex diseases.",
            "tag": "USEFUL"     # <--- True Gold Rationale (Ground truth from human annotators)
        },
        {
            "id": "e_1",
            "text": "However, these markers reflect nearby causal sites with moderate penetrance.",
            "tag": "NOISE"      # <--- Irrelevant Distractor Sentence
        },
        {
            "id": "e_2",
            "text": "Resequencing arrays allow mapping rare single nucleotide variants...",
            "tag": "NOISE"      # <--- Irrelevant Distractor Sentence
        }
        # ... up to 10 or 12 sentences per abstract
    ]
}
```

---

### 2. What Is Our "Model"?

We load a pre-trained Transformer cross-encoder:  
👉 **`cross-encoder/nli-deberta-v3-base`** (Cell 6 of the notebook).
* For encoder models like DeBERTa and RoBERTa, the technically precise term to use with your guide is: "Cross-Encoder NLI Classification Fine-Tuning" (fine-tuned on Natural Language Inference datasets like MNLI, FEVER, and SNLI).
* This model already knows English and scientific vocabulary.
* Its weights are **frozen** (`torch.no_grad()` is used throughout the entire notebook).
* It takes as input: `[CLS] Evidence Text [SEP] Claim [SEP]`
* It outputs 3 probabilities:
  1. $P(\text{CONTRADICTION}) \rightarrow$ represents **REFUTES**
  2. $P(\text{ENTAILMENT}) \rightarrow$ represents **SUPPORTS**
  3. $P(\text{NEUTRAL}) \rightarrow$ represents **NOT ENOUGH INFO**

---

### 3. What Does "Training" (Calibration) Mean Here?

If we are not training neural network weights, what is happening during the "training" phase in Cell 9 and Cell 12?

#### The Problem:
Our algorithms output continuous scores between $0.0$ and $1.0$ for every sentence:
* Sentence 1 gets score: $0.12$
* Sentence 2 gets score: $0.88$
* Sentence 3 gets score: $0.45$

We need a cutoff point (a **threshold** $t^*$) to say:
> *"Any sentence with a score $\ge t^*$ is `USEFUL`. Any sentence with a score $< t^*$ is `NOISE`."*

#### Where should we set $t^*$?
* If we set $t^* = 0.10$ (too low): We let in a lot of garbage noise.
* If we set $t^* = 0.95$ (too high): We throw away valid evidence.

#### The "Training" (Calibration) Process (Algorithm 2):
To prevent **Data Leakage**, we are forbidden from picking $t^*$ by looking at the test set. So we use the **Training Split (100 claims)**:

1. We run our attribution/saliency algorithm on all sentences across the 100 training claims.
2. We get a list of scores and their true ground-truth tags (`USEFUL` vs `NOISE`).
3. We test 100 candidate thresholds from $0.01$ to $0.99$:
   - For threshold $t = 0.40$: Calculate $F1_{\text{Useful}}$ and $F1_{\text{Noise}}$.
   - For threshold $t = 0.43$: Calculate $F1_{\text{Useful}}$ and $F1_{\text{Noise}}$.
   - For threshold $t = 0.45$: Calculate $F1_{\text{Useful}}$ and $F1_{\text{Noise}}$.
4. We pick the threshold $t^*$ that gives the highest balanced score:
   $$t^* = \arg\max_t \frac{1}{2} \left( F1_{\text{Useful}} + F1_{\text{Noise}} \right)$$

In our run:
* For the **Base Paper Baseline**, the calibrated threshold came out to: **$t^* = 0.4323$** (Cell 9).
* For our **Proposed Saliency Model**, the calibrated threshold came out to: **$\theta^* = 0.9697$** (Cell 12).

Once this threshold number is found, the "training" phase is **FINISHED**. The threshold is frozen.

---

### 4. How Is "Testing" Performed?

Now comes the real evaluation (Cell 10 for Baseline, Cell 13 for Proposed):
We take the **Testing Split (all 338 unseen test claims and 3,647 sentences)**.

For each test claim:
1. We run the algorithm to score each of its candidate sentences.
2. We apply our frozen threshold ($t^*$ or $\theta^*$):
   * Sentences $\ge$ threshold are marked as predicted `USEFUL`.
   * Sentences $<$ threshold are marked as predicted `NOISE`.
3. In the **Proposed Method**, we take only the `USEFUL` sentences and feed them to DeBERTa to predict `SUPPORTS` or `REFUTES`. (If all sentences were pruned, our Soft Fallback supplies the top-1 sentence).
4. In the **Baseline Method**, it evaluates the full noisy text and records Leave-One-Out differences.

Finally, we compare the model's predictions on the test set against the ground truth to calculate:
* **Evidence $F1_{\text{Useful}}$:** Did the model find the true gold sentences?
* **Evidence $F1_{\text{Noise}}$:** Did the model successfully reject the distractor sentences?
* **Claim Veracity Accuracy:** Did the model correctly classify the claim as True (`SUPPORTS`) or False (`REFUTES`)?
* **Latency:** How many seconds did it take per claim?

---

### Summary to Explain to Your Guide in 3 Sentences

> *"Sir, we are using a pre-trained DeBERTa-v3 cross-encoder as our fixed backbone verifier.*
> 
> *The 'training' phase in our notebook is **zero-leakage threshold calibration** (Algorithm 2): we use 100 training claims to learn the optimal cutoff threshold $t^*$ that best separates useful evidence from noise without touching the test data.*
> 
> *The 'testing' phase evaluates that calibrated threshold and our inference pipeline across **all 338 unseen test claims**, measuring real-world evidence discovery ($F1_{\text{Useful}}$), veracity accuracy, and latency."*

Here is the complete, crystal-clear explanation of how **$F1_{\text{Useful}}$** and **$F1_{\text{Noise}}$** are calculated, using a simple concrete numerical example.

---

### 1. The Core Idea: It's Just Binary Classification of Sentences

In scientific fact-checking, every sentence in an abstract belongs to one of two ground-truth classes:
* **Class 1: `USEFUL` (Positive)** $\rightarrow$ A sentence that actually proves or disproves the claim (the *Gold Rationale*).
* **Class 0: `NOISE` (Negative)** $\rightarrow$ An irrelevant background sentence (the *Distractor*).

Our model assigns an attribution/saliency score to every sentence and compares it against our threshold ($t^*$):
* If $\text{Score} \ge t^* \implies$ Model predicts **`USEFUL`**
* If $\text{Score} < t^* \implies$ Model predicts **`NOISE`**

To evaluate how well our model did, we look at the **Confusion Matrix**:

| | **Ground Truth is USEFUL** | **Ground Truth is NOISE** |
| :--- | :---: | :---: |
| **Model Predicts USEFUL** | **True Positive ($TP$)** | **False Positive ($FP$)** |
| **Model Predicts NOISE** | **False Negative ($FN$)** | **True Negative ($TN$)** |

---

### 2. Concrete Example with 5 Sentences

Suppose we are evaluating **one scientific claim**, and its retrieved abstract has **5 sentences**.

Here is the ground truth from human experts vs. what our model predicted:

| Sentence | Sentence Content | Ground Truth Tag | Model Attribution Score | Model Prediction (Threshold $t^* = 0.50$) | What is this? |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **$s_1$** | *"COVID-19 was discovered in 2019."* | `NOISE` | $0.15$ | `NOISE` | **$TN$** (Correctly rejected noise) |
| **$s_2$** | *"Drug X reduced viral load by 80%."* | `USEFUL` | $0.85$ | `USEFUL` | **$TP$** (Correctly found true proof) |
| **$s_3$** | *"We tested 300 patients in Boston."* | `NOISE` | $0.62$ | `USEFUL` | **$FP$** (Caught noise by mistake!) |
| **$s_4$** | *"Mortality dropped from 15% to 2%."* | `USEFUL` | $0.20$ | `NOISE` | **$FN$** (Missed this proof! False Alarm) |
| **$s_5$** | *"Statistical tests used p < 0.05."* | `NOISE` | $0.10$ | `NOISE` | **$TN$** (Correctly rejected noise) |

---

### 3. Calculating $F1_{\text{Useful}}$ (Did We Find the Proof?)

When calculating **$F1_{\text{Useful}}$**, we treat **`USEFUL`** as the positive class.

From our table above:
* $TP = 1$ ($s_2$)
* $FP = 1$ ($s_3$)
* $FN = 1$ ($s_4$)

#### Step A: Calculate Precision ($P_{\text{Useful}}$)
> *"Out of all sentences our model flagged as Useful, how many were actually Useful?"*
$$P_{\text{Useful}} = \frac{TP}{TP + FP} = \frac{1}{1 + 1} = \frac{1}{2} = 0.50 \quad (50\%)$$

#### Step B: Calculate Recall ($R_{\text{Useful}}$)
> *"Out of all the real Useful sentences in the document, how many did our model catch?"*
$$R_{\text{Useful}} = \frac{TP}{TP + FN} = \frac{1}{1 + 1} = \frac{1}{2} = 0.50 \quad (50\%)$$

#### Step C: Calculate $F1_{\text{Useful}}$
> The harmonic mean of Precision and Recall:
$$F1_{\text{Useful}} = 2 \times \frac{P_{\text{Useful}} \times R_{\text{Useful}}}{P_{\text{Useful}} + R_{\text{Useful}}} = 2 \times \frac{0.50 \times 0.50}{0.50 + 0.50} = \mathbf{0.50}$$

---

### 4. Calculating $F1_{\text{Noise}}$ (Did We Filter the Distractors?)

When calculating **$F1_{\text{Noise}}$**, we flip our perspective and treat **`NOISE`** as the positive class!

From our table:
* True Noise sentences correctly caught ($TN$ from earlier) = $2$ ($s_1, s_5$) $\rightarrow$ This is our **$TP_{\text{Noise}}$**.
* Sentences wrongly called Noise when they were useful ($FN$ from earlier) = $1$ ($s_4$) $\rightarrow$ This is our **$FP_{\text{Noise}}$**.
* Sentences that were Noise but we missed them ($FP$ from earlier) = $1$ ($s_3$) $\rightarrow$ This is our **$FN_{\text{Noise}}$**.

#### Step A: Calculate Precision ($P_{\text{Noise}}$)
> *"Out of all sentences our model threw into the trash as Noise, how many were truly Noise?"*
$$P_{\text{Noise}} = \frac{TP_{\text{Noise}}}{TP_{\text{Noise}} + FP_{\text{Noise}}} = \frac{2}{2 + 1} = \frac{2}{3} \approx 0.667$$

#### Step B: Calculate Recall ($R_{\text{Noise}}$)
> *"Out of all the garbage Noise sentences in the document, what fraction did we succeed in throwing away?"*
$$R_{\text{Noise}} = \frac{TP_{\text{Noise}}}{TP_{\text{Noise}} + FN_{\text{Noise}}} = \frac{2}{2 + 1} = \frac{2}{3} \approx 0.667$$

#### Step C: Calculate $F1_{\text{Noise}}$
$$F1_{\text{Noise}} = 2 \times \frac{0.667 \times 0.667}{0.667 + 0.667} = \mathbf{0.667}$$

---

### 5. Why Does the Base Paper Optimize $M(t) = 0.5 \times (F1_{\text{Useful}} + F1_{\text{Noise}})$ in Algorithm 2?

Notice the **imbalance**:
In our full SciFact dataset of 3,647 sentences:
* Only **366 sentences are `USEFUL` (10%)**
* A massive **3,281 sentences are `NOISE` (90%)**

#### The Trap:
If a lazy model just labels **EVERY SINGLE SENTENCE AS `NOISE`**:
* Its $F1_{\text{Noise}}$ would be **0.95 (95%)**!
* But its $F1_{\text{Useful}}$ would be **0.00 (0%)** because it caught zero proof!

To prevent this trap, the base paper's **Algorithm 2** balances both metrics:
$$M(t) = \frac{F1_{\text{Useful}} + F1_{\text{Noise}}}{2}$$
The algorithm sweeps through candidate thresholds and forces the system to find a cutoff that keeps $F1_{\text{Useful}}$ high while also filtering noise.

---

### 6. Now You Can Interpret Our Final Results Clearly!

Look at the numbers we got on all 338 test claims:

| Metric | Base Paper System | Our Proposed System | What This Means in Plain English |
| :--- | :---: | :---: | :--- |
| **Evidence $F1_{\text{Useful}}$** | **0.130** | **0.279** | **The base paper missed 87% of true rationales! Our system caught more than double (+114.6% increase).** |
| **Evidence $F1_{\text{Noise}}$** | **0.939** | **0.885** | Both systems reject ~90% of the garbage background sentences. |
| **Latency per claim** | **0.598s** | **0.333s** | Our system finds evidence **44.2% faster**. |

When your guide asks: *"How did you evaluate evidence usefulness?"*  
You can answer:
> *"Sir, we treat evidence detection as a sentence-level classification task: each sentence is either `USEFUL` (gold rationale) or `NOISE` (distractor). $F1_{\text{Useful}}$ measures how accurately the model isolates the true proof, while $F1_{\text{Noise}}$ measures its ability to discard distractors. Our model more than doubled the base paper's $F1_{\text{Useful}}$ from 0.130 to 0.279."*


These are two foundational questions about how pre-trained models and threshold calibration work in practice. Let's address both step-by-step.

---

### Question 1: *"How do we get $P(\text{CONTRADICTION})$, $P(\text{ENTAILMENT})$, and $P(\text{NEUTRAL})$ if we never defined them to our AI model?"*

#### The Answer:
**We did not have to define them because the pre-trained model already came with them built-in!**

Look closely at the model name we downloaded in Cell 6 of our notebook:
```python
model_name = "cross-encoder/nli-deberta-v3-base"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
```

Notice the letters **`nli`** in the name:
* **NLI** stands for **Natural Language Inference**.
* This model was trained by researchers on massive datasets (like MultiNLI, SNLI, and FEVER with over 1,000,000 sentence pairs).
* Its final neural network layer (the classification head) has **exactly 3 output neurons**:
  - **Neuron 0:** Logit for `CONTRADICTION` (meaning: False / Refutes)
  - **Neuron 1:** Logit for `ENTAILMENT` (meaning: True / Supports)
  - **Neuron 2:** Logit for `NEUTRAL` (meaning: Unrelated / No information)

#### What happens in code when we pass a sentence and a claim?
Look at Cell 12 in our notebook:
```python
with torch.no_grad():
    logits = model(**inputs).logits # Returns 3 raw numbers, e.g., [-2.1, 4.5, -1.8]
    probs = F.softmax(logits, dim=-1).cpu().numpy()[0] # Turns them into percentages summing to 1.0!

p_contra = float(probs[0]) # e.g., 0.01 (1%)
p_entail = float(probs[1]) # e.g., 0.98 (98%)
p_neutral = float(probs[2]) # e.g., 0.01 (1%)
```
The Softmax function converts those 3 raw numbers into probabilities that always add up to $1.0$ ($100\%$).  
Because the model was already pre-trained for NLI, it natively produces these three exact probabilities.

---

### Question 2: *"How do we calibrate the threshold? Do we use the base paper's Leave-One-Out scores, or do we calibrate a separate threshold for our proposed method?"*

#### The Answer:
**We calibrate TWO SEPARATE THRESHOLDS!**  
1. **$t^*$** is calibrated specifically for the **Base Paper Baseline** (using Leave-One-Out scores).
2. **$\theta^*$** is calibrated specifically for our **Proposed Method** (using Saliency scores).

Let's look at why this is required and how each one is calibrated in the notebook:

---

#### 1. Baseline Threshold Calibration ($t^*$ in Cell 9 of the Notebook)
In the Base Paper system:
* Every sentence $s_j$ gets an attribution score using **Algorithm 1 (Leave-One-Out)**:
  $$\phi(s_j) = |P(\text{Full}) - P(\text{Full} \setminus \{s_j\})|$$
* These scores are typically small numbers between $0.001$ and $0.50$.
* Algorithm 2 runs on the 100 training claims, sweeps candidate thresholds across these Leave-One-Out scores, and finds the best cutoff:
  $$t^* = 0.4323$$
* This threshold $t^* = 0.4323$ is used **only** when running the baseline.

---

#### 2. Proposed Threshold Calibration ($\theta^*$ in Cell 12 of the Notebook)
Our proposed method does **NOT** use Leave-One-Out scores!  
Our proposed method scores sentences using the **Saliency formula**:
$$\text{Saliency}(s_j) = P(\text{CONTRADICTION} \mid s_j, c) + P(\text{ENTAILMENT} \mid s_j, c)$$

Because the scoring formula is completely different, **we calibrate a new threshold ($\theta^*$) specifically for our saliency scores!**

Here is the exact code from Cell 12 showing how $\theta^*$ is calibrated on the 100 training claims:

```python
def calibrate_proposed_threshold(calib_items):
    scores, gold_tags = [], []
    for item in calib_items:
        claim = item["claim"]
        for p in item["evidence_pieces"]:
            # 1. Compute saliency for each sentence in the training set
            sal, _, _ = score_single_piece_proposed(claim, p["text"])
            scores.append(sal)
            gold_tags.append(1 if p["tag"] == "USEFUL" else 0)

    # 2. Test 100 candidate thresholds from min(scores) to max(scores)
    candidates = np.linspace(min(scores), max(scores), 100)
    best_theta, best_m = 0.30, -1.0

    for t in candidates:
        preds = (np.array(scores) >= t).astype(int)
        f1_u = f1_score(gold_tags, preds, pos_label=1, zero_division=0)
        f1_n = f1_score(gold_tags, preds, pos_label=0, zero_division=0)
        m = 0.5 * (f1_u + f1_n) # Maximizes balanced M(t)

        if m > best_m:
            best_m = m
            best_theta = t

    return best_theta
```

* For each of the 100 training claims, we compute $\text{Saliency}(s_j)$.
* We pair each saliency score with its ground-truth tag: $1$ if `USEFUL`, $0$ if `NOISE`.
* We sweep thresholds $t$ to find which cutoff maximizes $0.5 \times (F1_{\text{Useful}} + F1_{\text{Noise}})$.
* In our notebook run, the best threshold for our saliency model was:
  $$\theta^* = 0.9697$$

---

### Summary Table: Baseline vs. Proposed Thresholding

| Component | Base Paper Baseline | Our Proposed Improvisation |
| :--- | :---: | :---: |
| **Sentence Scoring Method** | Leave-One-Out Perturbation ($\phi = \|\Delta P\|$) | Active Saliency ($P_{\text{contra}} + P_{\text{entail}}$) |
| **Threshold Symbol** | **$t^*$** | **$\theta^*$** |
| **How it is Calibrated** | Calibrated on 100 training claims using Leave-One-Out scores | Calibrated on 100 training claims using Saliency scores |
| **Calibrated Value in Notebook** | **$t^* = 0.4323$** | **$\theta^* = 0.9697$** |
| **Where it is Applied** | Applied to baseline test claims in Cell 10 | Applied to proposed test claims in Cell 13 |

Now the separation is complete:
1. $P(\text{contra})$ and $P(\text{entail})$ come natively from the pre-trained NLI cross-encoder backbone.
2. The baseline uses its own calibrated threshold ($t^* = 0.4323$), and our proposed system uses its own calibrated threshold ($\theta^* = 0.9697$).