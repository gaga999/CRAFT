# CRAFT agent instructions

## Scope and language

- CRAFT is the project. Sibling repositories and papers are references or baselines; modify them only when the task calls for it. Preserve attribution when adapting code.
- Write all repository content in English, including code comments, docstrings, documentation, configuration descriptions, and commit messages. Conversation may follow the user's language.
- Keep README.md focused on orientation and quickstart, docs/ on research plans and design details, and this file on agent behavior. Read the context relevant to the task; avoid duplicating documentation.

## Engineering approach

- Prefer small, cohesive modules with clear interfaces and explicit data flow. Separate models, objectives, data handling, training, and evaluation as their responsibilities emerge; avoid speculative frameworks and premature abstractions.
- Keep code and documentation concise without sacrificing correctness or reproducibility. Explain non-obvious assumptions and decisions rather than narrating the implementation.
- Work toward the research outcome: inspect underlying equations, data, and reference implementations, challenge assumptions, and use appropriate numerical and statistical tools. Seek depth and evidence without expanding scope unnecessarily.
- Complete authorized work through implementation and relevant verification. Resolve routine choices independently; ask when ambiguity materially changes the scientific question or evaluation protocol.

## Scientific correctness

- Before implementing or changing a method, check its defining objective, required components, and important implementation details against primary papers or official implementations. Distinguish established findings, assumptions, and proposed experiments; cite consequential choices briefly in the relevant documentation.
- **Proactive expertise:** Identify and use applicable, well-known implementation tricks and training practices that materially affect results, even when the user's prompt does not mention them. Consult primary papers and strong reference implementations for details such as normalization, regularization, initialization, optimizer schedules, numerical stability, and gradient flow.
- **Research-quality baseline:** Deliver an implementation an experienced researcher would consider a credible baseline. Running successfully and matching the main equations are insufficient if established practices affecting learning, stability, or evaluation are missing.
- **Established practices by default:** Apply mature, well-supported practices whenever their assumptions fit. Explicitly justify consequential omissions rather than silently leaving them out. When reproducing a specified baseline, preserve its method and label additional improvements separately so comparisons remain interpretable.
- For contrastive learning, check the roles of encoder representations and loss embeddings, projection heads, similarity normalization, temperature, and positive/negative construction. SimCLR and MoCo v2 support nonlinear projection heads in their settings: do not silently force representation and contrastive spaces to coincide. Verify whether that evidence transfers to CRAFT's objective; treat departures as explicit, testable design choices.
- For RL, verify algorithm-specific components and update semantics. Standard actor-critic PPO includes a value estimator and value loss: omitting them requires an explicitly justified variant. Check advantage/return computation, bootstrap masks for termination versus truncation, policy log-probabilities, and action bounds as applicable.
- If a request conflicts with sound methodology, explain the specific failure mode, likely consequence, and a defensible alternative before implementing the affected choice—even when the request explicitly insists on it. If the user chooses an intentional deviation, record its rationale, risks, and validation or ablation plan. Instruction compliance is not evidence of scientific validity.

## Validation and reporting

- Match verification to the change. For learning code, use focused checks of shapes, finite losses/gradients, parameter updates, and relevant mathematical invariants, followed by a small training smoke test when feasible. Passing a smoke test does not establish learning performance.
- Make experiments reproducible with resolved configurations, seeds, dependency versions, and code revisions. Compare baselines under consistent evaluation protocols and compute budgets; use multiple seeds and report variability for performance claims.
- Prefer executable checks and measured results over self-assessment. Report what changed, what was actually verified, and remaining limitations. Never invent runs, citations, metrics, or successful checks; keep planned work distinct from completed work.

## References

Instruction design reviewed on 2026-11-09:

- [OpenAI: Rethinking skills and prompts (2026-09-11)](https://developers.openai.com/blog/rethinking-skills-and-prompts-for-gpt-6-astra)
- [OpenAI: Reasoning best practices (living documentation)](https://developers.openai.com/api/docs/guides/reasoning-best-practices)
- [Felix Rieseberg: How to be more ambitious (2026-10-13)](https://felixrieseberg.com/how-to-be-more-ambitious-a-tutorial-for-humans-with-models/)
