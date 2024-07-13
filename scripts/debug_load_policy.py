import tensorflow as tf

# Define the directory where the policy is saved
POLICY_DIR = "/home/ubuntu/2D-birds/policy"

# Load the saved policy
saved_policy = tf.compat.v2.saved_model.load(POLICY_DIR)

# Print the signatures of the loaded policy
print(f"Signatures of the saved model: {saved_policy.signatures}")

# Check if the 'action' method is present in the signatures
if "action" in saved_policy.signatures:
    print(f"'action' method is in policy signatures: {saved_policy.signatures['action']}")
else:
    print("'action' method is NOT in policy signatures")

# Additional debugging: Print the available methods in the saved policy
print(f"Available methods in saved policy: {dir(saved_policy)}")

# Additional debugging: Print the concrete function for 'action' method if available
if hasattr(saved_policy, 'action'):
    print(f"Concrete function for 'action' method: {saved_policy.action}")
else:
    print("The 'action' method is not available as a concrete function.")
