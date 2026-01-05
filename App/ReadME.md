# Te-QAS Flask Application

A web application for Telugu Question Answering System (Te-QAS) using XLM-RoBERTa model.

## Setup Instructions

1. Create a virtual environment:

    ```
    python -m venv venv
    ```

2. Activate the virtual environment:

    - Windows: `venv\\Scripts\\activate`
    - Linux/Mac: `source venv/bin/activate`

3. Install the requirements:

    ```
    pip install -r requirements.txt
    ```

4. Place your model files in the `models/final_xlmr_2.0_tel_3` directory.

5. Run the Flask application:

    ```
    python app.py
    ```

6. Open a web browser and navigate to:
    ```
    http://localhost:5001
    ```
