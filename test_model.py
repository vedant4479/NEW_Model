import requests
import os
import sys

def test_model_api(image_path):
    """
    Test the model API by sending an image to the endpoint
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} does not exist")
        return
    
    print(f"Testing model API with image: {image_path}")
    
    try:
        # Prepare the image file
        with open(image_path, 'rb') as f:
            files = {'image': (os.path.basename(image_path), f, 'image/jpeg')}
            
            # Send the request to the API
            response = requests.post('http://localhost:7000/predict', files=files)
            
            # Check if the request was successful
            if response.status_code == 200:
                result = response.json()
                print("\nPrediction Result:")
                print(f"Disease: {result.get('prediction', 'Unknown')}")
                print(f"Confidence: {result.get('confidence', 'Unknown')}")
                print("\nAPI test successful!")
            else:
                print(f"Error: API returned status code {response.status_code}")
                print(response.text)
    
    except Exception as e:
        print(f"Error during API test: {str(e)}")

if __name__ == "__main__":
    # Check if an image path was provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Use a default image if none provided
        image_path = "test_image.jpg"
        
        # If the default image doesn't exist, inform the user
        if not os.path.exists(image_path):
            print(f"Please provide an image path as an argument or create a file named {image_path}")
            print("Usage: python test_model.py [image_path]")
            sys.exit(1)
    
    test_model_api(image_path)
