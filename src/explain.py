import tensorflow as tf
import numpy as np
import cv2
from tensorflow import keras

def generate_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Computes a Grad-CAM heatmap for a given input image and model.
    Updated for strict Keras 3 / TensorFlow 2.16+ compatibility.
    """
    # 1. Ensure the model is 'called' to initialize the graph
    # We do a dummy forward pass if needed (though predict usually does this)
    _ = model(img_array)

    # 2. Identify the target layers
    last_conv_layer = model.get_layer(last_conv_layer_name)
    
    # 3. Build the Gradient Model
    # Note: We use model.input and model.output directly
    grad_model = keras.models.Model(
        inputs=model.input, 
        outputs=[last_conv_layer.output, model.output]
    )

    # 4. Record the gradients
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # 5. Calculate gradients of the class w.r.t the conv layer output
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # 6. Weight the channels by their gradient importance
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 7. Normalize the heatmap for visualization
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

def overlay_heatmap(img, heatmap, alpha=0.4):
    """
    Overlays a heatmap on top of the original spectrogram image.
    """
    img = np.uint8(255 * img)
    heatmap = np.uint8(255 * heatmap)

    jet = cv2.applyColorMap(cv2.resize(heatmap, (img.shape[1], img.shape[0])), cv2.COLORMAP_JET)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    superimposed_img = jet * alpha + img_rgb
    return keras.preprocessing.image.array_to_img(superimposed_img)
