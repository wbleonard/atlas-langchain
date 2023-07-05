import requests

def save_web_page_as_html(url, file_path):
    # Send a GET request to the web page
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Save the content to a file
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(response.text)
        
        print(f"The web page at {url} has been saved as HTML at {file_path}.")
    else:
        print(f"Error: Failed to retrieve the web page. Status code: {response.status_code}")
