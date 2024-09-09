import onnxruntime as ort
import cv2
import numpy as np

def preprocess_image(image, input_size):
    original_size = image.shape[:2]  # (height, width)

    input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    input_img = cv2.resize(input_img, (input_size, input_size))

    input_img = input_img / 255.0
    input_img = np.transpose(input_img, (2, 0, 1))  # Change to (channels, height, width)
    input_img = np.expand_dims(input_img, axis=0)  # Add batch dimension
    input_img = input_img.astype(np.float32)
    #input_img = input_img[np.newaxis, :, :, :].astype(np.float32)

    return input_img, original_size

def rescale_boxes(boxes, input_size, original_height, original_width):
    input_shape = np.array([input_size, input_size, input_size, input_size])
    boxes = np.divide(boxes, input_shape, dtype=np.float32)
    boxes *= np.array([original_width, original_height, original_width, original_height])
    return boxes

def postprocess_output(output, conf_threshold, original_size):
    output = output.squeeze() # shape is [84, 8400]

    boxes = output[:4, :]
    class_probs = output[4:, :]
    class_ids = np.argmax(class_probs, axis=0)
    class_scores = class_probs[class_ids, range(class_probs.shape[1])]



    scores_mask = class_scores > conf_threshold

    boxes = boxes[:, scores_mask]
    class_scores = class_scores[scores_mask]
    class_ids = class_ids[scores_mask]






    nms_mask = cv2.dnn.NMSBoxes(boxes.transpose(), class_scores, 0.2, 0.5)


    boxes = boxes[:, nms_mask]
    class_scores = class_scores[nms_mask]
    class_ids = class_ids[nms_mask]

    print(nms_mask)


    # Rescale boxes to original image dimensions
    boxes = rescale_boxes(boxes.transpose(), 640, original_size[0], original_size[1])



    return class_ids, boxes, class_scores




# Preprocessing
image = cv2.imread('image.jpg')
input_tensor, original_size = preprocess_image(image, input_size=640)


# Load ONNX model and run inference
onnx_model_path = '../models/yolov8s.onnx'
session = ort.InferenceSession(onnx_model_path)
input_name = session.get_inputs()[0].name
outputs = session.run(None, {input_name: input_tensor})

print(outputs[0].shape)

print(outputs[0][0, 0, 0])
print(outputs[0][0, 1, 0])


# Postprocessing
class_ids, boxes, confidences = postprocess_output(outputs[0], 0.2, original_size)

print(boxes)


# Draw the boxes on the image
image = cv2.imread('image.jpg')

for i, box in enumerate(boxes):
    x1, y1, x2, y2 = box.astype(int)
    label = f"Class {class_ids[i]}: {confidences[i]:.2f}"
    cv2.rectangle(image, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

# Save the image with bounding boxes
output_image_path = 'output_with_boxes2.jpg'
cv2.imwrite(output_image_path, image)

print(f"Image saved as {output_image_path}")
