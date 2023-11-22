import streamlit as st
import cv2
from PIL import Image
import torch
from torchvision import models, transforms
from av import VideoFrame
from aiortc import VideoStreamTrack
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
model = models.resnet18(pretrained=True)
model.eval()
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
class ObjectClassifier:
    def __init__(self):
        self.model = models.resnet18(pretrained=True)
        self.model.eval()

    def classify_image(self, image):
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)

        with torch.no_grad():
            output = self.model(input_batch)
            _, predicted_idx = torch.max(output, 1)
        return predicted_idx.item()

def classify_image(image):
    # Preprocess the image
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    with torch.no_grad():
        output = model(input_batch)
    _, predicted_idx = torch.max(output, 1)
    return predicted_idx.item()

def is_recyclable(label):
    return label == 1

def main():
    st.title("Recyclable Object Classifier")

    object_classifier = ObjectClassifier()

    class VideoTransformer(VideoTransformerBase):
        def transform(self, frame):
            image = frame.to_ndarray(format="bgr24")
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            st.image(pil_image, caption="Camera Feed", use_column_width=True)
            label = object_classifier.classify_image(pil_image)
            st.write(f"Predicted label: {label}")
            if is_recyclable(label):
                st.success("Recyclable!")
            else:
                st.error("Not recyclable!")

    webrtc_streamer(
        key="example",
        video_transformer_factory=VideoTransformer,
        async_transform=True,
    )

if __name__ == "__main__":
    main()
