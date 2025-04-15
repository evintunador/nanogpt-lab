# 2x faster training with Muon optimizer

https://aistudio.google.com/prompts/1NKyXjq3s136YZtPB5XI4vMQmFt5qlSXA

1. Forward Pass: Input data goes through the network using current weights to produce a prediction.
2. Loss Calculation: The prediction is compared to the ground truth using a loss function (e.g., Mean Squared Error,
    Cross-Entropy). This gives a single number representing how wrong the prediction was (between predicted number and ground truth number).
3. Backward Pass (Backpropagation): This is where the gradients are calculated. Starting from the loss, the algorithm
    works backward through the network, calculating the gradient of the loss with respect to each weight and bias (if you
    change a weight number, how does loss number change, becomes bigger or smaller, faster or slower change).
4. Optimizer's Role: After a single optimization step (after processing a batch of data and calculating all the
    necessary gradients - each forward pass), the optimizer applies an update rule (like SGD, Adam, RMSprop) to all
    the weights, using the gradients calculated for that step (eaach weight has customized update).


Each batch has slightly different landscape of loss.

Part 2: Common Optimizers - SGD Variants and AdamW
Plain SGD has limitations. The gradients can be noisy (especially with small batches), and progress can be
slow or get stuck in suboptimal areas of the loss landscape. Optimizers introduce techniques to improve this.

SGD with Momentum:

Idea: Add inertia to the updates. Instead of only using the current gradient, also consider the direction moved in previous steps.
This helps smooth out oscillations and accelerate movement in consistent directions.

Mechanism: Maintains a "velocity" or "momentum" vector m.

m_t = β * m_{t-1} + ∇L(θ_t) (Update momentum: decay old momentum β and add current gradient)

θ_{t+1} = θ_t - η * m_t (Update parameter using momentum)

β is the momentum coefficient (e.g., 0.9, 0.95).


# Adam
Adam adapts the learning rate for each parameter individually.

Steep Slopes (Large Gradients): If you take a large, fixed step on a very steep slope, you risk leaping right over the valley (the minimum loss) and ending up higher on the other side. You might even bounce back and forth, failing to settle in the valley (oscillating or diverging). Reducing the step size in steep areas helps ensure more careful, stable progress towards the minimum.

Gentle Slopes or Flat Plateaus (Small Gradients): If you're on a nearly flat area, taking tiny fixed steps means progress will be incredibly slow. You might get stuck or take forever to reach the valley. Increasing the step size in these flat regions helps accelerate progress.

Infrequent Features/Parameters (Infrequent Gradients): Some parameters might only get non-zero gradients occasionally (e.g., parameters related to rare words in an embedding layer). When they do get a gradient signal, we want to make sure they learn effectively from it. If we only take tiny steps, these parameters might barely update over the entire training process. Giving them a relatively larger update when their gradient is non-zero helps them learn.

Adam's Goal: To automatically adjust the step size for each parameter based on its own history of gradients. It aims to:

Dampen oscillations: Reduce the effective learning rate for parameters with consistently large gradients.

Accelerate progress: Increase the effective learning rate for parameters with consistently small or infrequent gradients.


https://aistudio.google.com/prompts/1NKyXjq3s136YZtPB5XI4vMQmFt5qlSXA