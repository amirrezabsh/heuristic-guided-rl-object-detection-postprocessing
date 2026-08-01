# Method

## Final Formulation

The final method treats adaptive object-detection post-processing as a one-step
episodic Markov decision process, equivalent to contextual-bandit RL.

For each image:

```text
state  -> action -> reward -> terminal
```

The detector is frozen. The policy only controls post-processing.

## State

The state is computed from candidate detector outputs, not from raw pixels.
Features include:

- number of candidate boxes
- confidence mean, max, and standard deviation
- box area statistics
- class-count statistics
- top-score statistics
- density, uncertainty, and overlap indicators used by heuristic baselines

## Action

The compact action space controls:

```text
confidence threshold
NMS IoU threshold
maximum detections
```

An extended action-space ablation also tested:

```text
minimum normalized box area
pre-NMS top-k candidate limit
```

The compact action space produced the best final result.

## Reward

The training reward is based on detection quality after applying the sampled
post-processing action:

```text
reward = mAP50-95 + 0.1 * mAP50 - latency_weight * predictions_per_image
```

Ground-truth annotations are used to compute reward, but the method does not
require labels for the best post-processing action. In this sense the method is
not label-free, but it is action-label-free.

## Baselines

The project evaluates:

- fixed standard post-processing
- random continuous post-processing
- direct continuous REINFORCE
- density/confidence/combined heuristic rules
- heuristic-guided residual REINFORCE

## Final Method

The best method is:

```text
final_action = heuristic_combined_action + RL_residual_correction
```

This keeps the useful expert prior from the heuristic and lets REINFORCE
optimize small corrections from reward feedback.
