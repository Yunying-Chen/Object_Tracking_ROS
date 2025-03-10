import os
import warnings


import cv2
import numpy

# warnings.filterwarnings("ignore")


class ONNXDetect:
    def __init__(self, onnx_path, session=None):
        self.session = session
        if self.session is None:
            assert onnx_path is not None
            assert os.path.exists(onnx_path)
            from onnxruntime import InferenceSession
            self.session = InferenceSession(onnx_path,
                                            providers=['CUDAExecutionProvider'])

        self.inputs = self.session.get_inputs()[0]
        self.confidence_threshold = 0.85
        self.iou_threshold = 0.85
        self.input_size = 640
        shape = (1, 3, self.input_size, self.input_size)
        image = numpy.zeros(shape, dtype='float32')
        for _ in range(10):
            self.session.run(output_names=None,
                             input_feed={self.inputs.name: image})

    def __call__(self, image):
        image, scale = self.resize(image, self.input_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose((2, 0, 1))[::-1]
        image = image.astype('float32') / 255
        image = image[numpy.newaxis, ...]

        outputs = self.session.run(output_names=None,
                                   input_feed={self.inputs.name: image})
        outputs = numpy.transpose(numpy.squeeze(outputs[0]))

        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = []
        scores = []
        class_indices = []

        # Iterate over each row in the outputs array
        for i in range(outputs.shape[0]):
            # Extract the class scores from the current row
            classes_scores = outputs[i][4:]

            # Find the maximum score among the class scores
            max_score = numpy.amax(classes_scores)

            # If the maximum score is above the confidence threshold
            if max_score >= self.confidence_threshold:
                # Get the class ID with the highest score
                class_id = numpy.argmax(classes_scores)

                # Extract the bounding box coordinates from the current row
                image, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # Calculate the scaled coordinates of the bounding box
                left = int((image - w / 2) / scale)
                top = int((y - h / 2) / scale)
                width = int(w / scale)
                height = int(h / scale)

                # Add the class ID, score, and box coordinates to the respective lists
                class_indices.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        # if len(boxes)>0:
        #     indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_threshold, self.iou_threshold)

        # Iterate over the selected indices after non-maximum suppression
        # nms_outputs = []
        # for i in indices:
        #     # Get the box, score, and class ID corresponding to the index
        #     box = boxes[i]
        #     score = scores[i]
        #     class_id = class_indices[i]
        #     nms_outputs.append([*box, score, class_id])
        # return nms_outputs

        # if indices: 
        #     best_index = numpy.argmax([scores[i] for i in indices])
        #     best_box = boxes[indices[best_index]]
        #     best_score = scores[indices[best_index]]
        #     best_class_id = class_indices[indices[best_index]]
        #     return [*best_box, best_score, best_class_id] 

        best_index = numpy.argmax([scores])
        best_box = boxes[best_index]
        best_score = scores[best_index]
        best_class_id = class_indices[best_index]
        
        return [*best_box, best_score, best_class_id] 

    @staticmethod
    def resize(image, input_size):
        shape = image.shape

        ratio = float(shape[0]) / shape[1]
        if ratio > 1:
            h = input_size
            w = int(h / ratio)
        else:
            w = input_size
            h = int(w * ratio)
        scale = float(h) / shape[0]
        resized_image = cv2.resize(image, (w, h))
        det_image = numpy.zeros((input_size, input_size, 3), dtype=numpy.uint8)
        det_image[:h, :w, :] = resized_image
        return det_image, scale




def onnx_inference(frame,model_path='./yolov8.onnx'):
    # Load model
    model = ONNXDetect(onnx_path=model_path)

    # frame = cv2.imread('./frame.jpg')
    image = frame.copy()
    outputs = model(image)
    for output in outputs:
        x, y, w, h, score, index = output
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
    cv2.imwrite('output.jpg', frame)





if __name__ == "__main__":
    onnx_inference()
