# Brain-MRI-Metastasis-Segmentation-Assignment

Here’s a comprehensive structure for your `README.md` file in your GitHub repository.

---

# Brain MRI Metastasis Segmentation using Nested U-Net and Attention U-Net

This project focuses on brain metastasis segmentation from MRI images using two advanced neural network architectures: Nested U-Net (U-Net++) and Attention U-Net. We demonstrate the results through an interactive Streamlit application where users can upload MRI images and view the segmentation results.

## Table of Contents
- [Introduction](#introduction)
- [Architectures](#architectures)
  - [Nested U-Net (U-Net++)](#nested-u-net-u-net)
  - [Attention U-Net](#attention-u-net)
- [Streamlit UI Demonstration](#streamlit-ui-demonstration)
- [Setup Instructions](#setup-instructions)
- [Challenges in Brain Metastasis Segmentation](#challenges-in-brain-metastasis-segmentation)
- [How We Address the Challenges](#how-we-address-the-challenges)
- [Conclusion](#conclusion)

---

## Introduction

Metastasis in the brain is a critical concern for cancer patients, often requiring accurate diagnosis and treatment planning. Segmenting metastatic lesions in brain MRI images is challenging due to the irregular shape, size, and contrast of the lesions. This project uses state-of-the-art deep learning techniques to automatically segment metastasis from brain MRI images, providing a powerful tool to assist clinicians.

## Architectures

### Nested U-Net (U-Net++)
Nested U-Net, also known as U-Net++, is a more refined version of the traditional U-Net architecture. The primary innovation is the use of densely connected convolutional layers, which helps capture more detailed features and improve the accuracy of segmentation.

- **Skip Pathways**: Unlike traditional U-Net, which has direct connections between encoder and decoder layers, U-Net++ introduces nested skip pathways, allowing the network to capture finer spatial details and reduce the semantic gap between feature maps from different levels of the encoder and decoder.
- **Metastasis Segmentation Application**: For brain metastasis segmentation, the detailed and dense feature maps created by Nested U-Net make it particularly well-suited for segmenting lesions with varying shapes and sizes.

### Attention U-Net
Attention U-Net extends the traditional U-Net by incorporating attention mechanisms. The attention block helps the model focus on the most relevant parts of the image during the segmentation process.

- **Attention Mechanism**: This mechanism allows the network to give more weight to the important regions of the image, such as the metastasis lesions, while ignoring irrelevant parts, such as normal brain tissue.
- **Metastasis Segmentation Application**: Brain metastases can often be small and difficult to distinguish from surrounding tissue. The attention mechanism makes the Attention U-Net better suited for focusing on the metastasis regions, improving accuracy in detecting smaller lesions.

## Streamlit UI Demonstration

Below is a video demonstrating how our Streamlit UI works, allowing users to upload brain MRI images and view metastasis segmentation results:

[![Streamlit UI Demo]](https://github.com/user-attachments/assets/ca8a117d-710d-4212-8ef1-4338b23a3869)  
*Click on the image to watch the full video demonstration.*

The video shows:
1. Uploading a brain MRI image.
2. Segmentation results displayed using both the Nested U-Net and Attention U-Net architectures.
3. A comparison between the original image and the segmented results.

## Setup Instructions

Follow these steps to set up and run the project locally:

### Prerequisites
- Python 3.8+
- TensorFlow 2.0+
- Streamlit

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your_username/brain-mri-metastasis-segmentation.git
   cd brain-mri-metastasis-segmentation
   ```

2. **Create and Activate a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Required Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Pretrained Models**
   Download the pretrained models (U-Net++ and Attention U-Net) from the provided links or train them using the dataset you have. Place them in the root directory of the project.

5. **Run the Streamlit App**
   ```bash
   streamlit run app.py
   ```

### Using the App
- Upload an MRI image through the sidebar in the Streamlit interface.
- The app will display the original image alongside the segmentation results from both the Nested U-Net and Attention U-Net models.

## Challenges in Brain Metastasis Segmentation

Segmenting metastases in brain MRI images poses unique challenges:
- **Varied Appearance**: Metastasis lesions can differ significantly in size, shape, and intensity in MRI scans.
- **Class Imbalance**: There are often far fewer pixels belonging to metastatic lesions compared to normal brain tissue, making it difficult for models to learn effective segmentation.
- **Small Lesions**: Many metastases are small and hard to detect, often requiring high resolution and attention to small details in the image.

## How We Address the Challenges

1. **Nested U-Net**: The use of dense skip pathways enables the model to learn from multi-scale features, improving its ability to handle variations in lesion size and shape.
2. **Attention U-Net**: The attention mechanism helps the model focus on relevant parts of the image, ensuring better performance when segmenting small, challenging lesions.
3. **Custom Loss Functions**: We used the Dice coefficient loss, which is particularly effective in handling the class imbalance by focusing on the overlap between predicted and actual lesion areas.

## Conclusion

By employing advanced architectures like Nested U-Net and Attention U-Net, this project addresses the challenges associated with brain metastasis segmentation. The provided Streamlit interface allows users to interactively upload MRI images and view the segmentation results, offering a potential clinical tool for assisting in diagnosis and treatment planning.

---

