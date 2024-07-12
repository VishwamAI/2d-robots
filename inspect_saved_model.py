import tensorflow as tf

# Load the saved model
policy_dir = 'policy'
saved_model = tf.saved_model.load(policy_dir)

# Print the available signatures of the saved model
print(f"Available signatures: {list(saved_model.signatures.keys())}")

# Print the details of the 'serving_default' signature if it exists
if 'serving_default' in saved_model.signatures:
    print(f"Serving default signature: {saved_model.signatures['serving_default']}")
else:
    print("The 'serving_default' signature is not available in the saved model.")
