import requests

BASE_URL = "http://localhost:3000"

def test_detect_endpoint():
    # Path to your test images
    low_res_path = img_path
    high_res_path = img_path
    
    # Prepare the files
    files = {
        'low_res': ('low_res.png', open(low_res_path, 'rb'), 'image/png'),
        'high_res': ('high_res.png', open(high_res_path, 'rb'), 'image/png')
    }
    
    # Make the request
    response = requests.post(f"{BASE_URL}/detect", files=files)
    
    # Check if request was successful
    if response.status_code == 200:
        print("Detection Results:", response.json())
    else:
        print("Error:", response.status_code, response.text)

def test_recognize_endpoint():
    # Prepare the files and form data
    files = {
        'image': ('image.png', open(img_path, 'rb'), 'image/png')
    }
    
    # Send as form-data
    response = requests.post(
        f"{BASE_URL}/recognize",
        files=files,
        data={'output_file_name': "result.xlsx"}
    )
    # Check if request was successful
    if response.status_code == 200:
        # Save the response content as an Excel file
        with open("result.xlsx", 'wb') as f:
            f.write(response.content)
        print("Excel file saved as result.xlsx")
    else:
        print("Error:", response.status_code, response.text)

if __name__ == "__main__":
    print("Testing detect endpoint...")
    test_detect_endpoint()
    
    print("\nTesting recognize endpoint...")
    test_recognize_endpoint()