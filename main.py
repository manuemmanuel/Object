import streamlit as st
import cv2
from PIL import Image
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from torch.autograd import Variable
from av import VideoFrame
from aiortc import VideoStreamTrack
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = fasterrcnn_resnet50_fpn(pretrained=True)
model = model.to(device).eval()

class ObjectClassifier:
    def __init__(self):
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.model = self.model.to(device).eval()

    def detect_objects(self, image):
        tensor_image = F.to_tensor(image).unsqueeze(0).to(device)

        with torch.no_grad():
            prediction = self.model(tensor_image)

        return prediction

def main():
    st.title("Recyclable Object Classifier")

    object_classifier = ObjectClassifier()

    class VideoTransformer(VideoTransformerBase):
        def transform(self, frame):
            image = frame.to_ndarray(format="bgr24")
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            st.image(pil_image, caption="Camera Feed", use_column_width=True)

            prediction = object_classifier.detect_objects(pil_image)

            for score, label, box in zip(
                prediction[0]["scores"],
                prediction[0]["labels"],
                prediction[0]["boxes"],
            ):
                if score > 0.5:
                    st.write(f"Detected object: {label} with confidence: {score}")

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
