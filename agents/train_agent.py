import tensorflow as tf
from tf_agents.policies import policy_saver

# Register the 'action' method as a concrete function
agent.policy.action = tf.function(agent.policy.action)
print(
    f"Registered 'action' method as a concrete function: "
    f"{agent.policy.action}"
)

tf_policy_saver = policy_saver.PolicySaver(
    agent.policy,
    batch_size=1,  # Specify a batch size of 1
    use_nest_path_signatures=True,
    train_step=train_step_counter  # Include the train step counter
)
tf_policy_saver.save(policy_dir)
