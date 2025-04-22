# Skin Disease Detection ML Model

This folder contains the machine learning model for skin disease detection.

## Setup Instructions

1. Install Python 3.8 or higher if not already installed

2. Install the required Python packages:
   ```
   pip install -r python_requirements.txt
   ```

3. Run the Flask server:
   ```
   python app.py
   ```

The server will start on port 7000 and will be accessible at http://localhost:7000

## Testing the Model

You can test the model API using the provided test script:

```
python test_model.py [path_to_image]
```

If you don't provide an image path, the script will look for a file named `test_image.jpg` in the current directory.

## API Endpoints

### POST /predict
Accepts an image file and returns the predicted skin disease.

**Request:**
- Form data with an 'image' field containing the image file

**Response:**
```json
{
  "prediction": "Disease Name",
  "confidence": "85.75%"
}
```

## Supported Disease Classes
- Actinic keratosis
- Atopic Dermatitis
- Benign keratosis
- Dermatofibroma
- Melanocytic nevus
- Melanoma
- Squamous cell carcinoma
- Tinea Ringworm Candidiasis
- Vascular lesion

## Implementation Notes

This model uses a pre-trained MobileNetV2 architecture from TensorFlow. In a production environment, you would fine-tune this model on a skin disease dataset for better accuracy. The current implementation provides a demonstration of the API functionality.
