# Multimodal Human Recognition with MediaPipe and MobileNetV2

## To run the code, please either use "Visual Studio" or "Jupyter Notebook from Anaconda Navigator".

### Thank you.

<br>

## Code Explanation:

1. **Library Imports**: The code imports necessary libraries including OpenCV (`cv2`), NumPy (`numpy`), MediaPipe (`mediapipe`), and TensorFlow (`tensorflow`). It also imports specific modules from these libraries.

2. **Initializing MediaPipe Solutions**: MediaPipe solutions for pose and hand keypoint detection are initialized using `mp_pose.Pose()` and `mp_hands.Hands()` respectively.

3. **Loading MobileNetV2 Model**: The MobileNetV2 model pre-trained on ImageNet is loaded using TensorFlow's `MobileNetV2` function. The top layer is excluded (`include_top=False`) and the weights are set to 'imagenet'. The model is then set to non-trainable (`base_model.trainable = False`) to freeze its layers.

4. **Building Neural Network Model**: A custom neural network model is built on top of the MobileNetV2 base model using TensorFlow's Keras API. The model consists of a flattened layer, followed by two dense layers with ReLU activation and dropout.

5. **Simulated Known Faces Database**: A simulated database of known faces with randomly generated feature vectors is created. This is used for face recognition.

6. **Preprocessing Functions**: Helper functions are defined for preprocessing face images and extracting faces from frames using Haar cascade classifier.

7. **Recognition Function**: A function is defined for matching extracted face features with known faces in the database using cosine similarity.

8. **Real-time Processing Loop**: A loop captures frames from the video feed in real-time. Each frame is processed for pose and hand keypoints detection using MediaPipe, and landmarks are drawn on the frame. Faces are then extracted and processed for recognition. Recognized identities and similarity scores are displayed on the frame along with bounding boxes around the detected faces.

9. **Exiting**: The program exits the loop if the 'ESC' key is pressed, releasing resources used for pose and hand keypoints estimation, video capture, and closing all OpenCV windows.

## Key Points:
- Utilizes MediaPipe for real-time pose and hand keypoint detection.
- Integrates MobileNetV2 model for face feature extraction.
- Simulated known faces database for face recognition.
- Real-time processing loop for simultaneous pose, hand, and face recognition.
- Utilizes OpenCV for video capture and visualization.

The code performs real-time pose, hand, and face recognition using MediaPipe and MobileNetV2 model.
