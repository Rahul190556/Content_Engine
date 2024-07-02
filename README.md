# Multi-PDF ChatBot using LlamaCpp :books:

This project is a Content Engine designed to analyze and compare multiple PDF documents. It specifically identifies and highlights their differences by utilizing Retrieval Augmented Generation (RAG) techniques to effectively retrieve, assess, and generate insights from the documents. The system uses LangChain, FAISS, and a local embedding model (LlamaCpp) to achieve this.


![Description](https://github.com/Rahul190556/Content_Engine/blob/cddad6a2d9afddb95a7e4b30fa7335ed9c63747d/Img3.png)


## Features

- **Document Parsing**: Extracts text and structure from uploaded PDF files. The project uses PyPDFLoader to read and extract text from PDF documents. This allows the system to handle multiple documents and convert them into a format suitable for analysis.

- **Vector Generation**: Uses a local embedding model to create embeddings for document content. The embedding model (LlamaCpp) generates vector representations of the document content, making it easier to perform similarity searches and comparisons.

- **Vector Store**: Utilizes FAISS to manage and query embeddings. FAISS is an efficient similarity search library that stores the embeddings generated by the embedding model and allows for quick retrieval based on similarity queries.

- **Conversational AI**: Employs LlamaCpp for contextual insights and conversation generation. The LlamaCpp model processes user queries and provides responses based on the content of the uploaded PDFs and the embeddings stored in FAISS.

- **Interactive UI**: Built with Streamlit to facilitate user interaction and display comparative insights. The Streamlit app provides an easy-to-use interface for uploading documents, asking questions, and viewing responses.

## Setup

### Backend Framework
- **LangChain**: A powerful toolkit for building LLM applications with a strong focus on retrieval-augmented generation.

### Frontend Framework
- **Streamlit**: An open-source app framework for Machine Learning and Data Science projects, allowing you to create interactive web applications easily.

### Vector Store
- **FAISS**: A library for efficient similarity search and clustering of dense vectors.

### Embedding Model
- **LlamaCpp**: A local embedding model for generating vectors from the PDF file content.

## Requirements

- Python 3.8 or higher
- CUDA Toolkit (for GPU support) or you can use CPU
- PyTorch (compatible with CUDA)
- Required Python packages (listed in `requirements.txt`)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/your-username/multi-pdf-chatbot.git
    cd multi-pdf-chatbot
    ```

2. Create a virtual environment:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Download the LlamaCpp model and place it in the `models` directory:
    ```sh
    mkdir models
    # Download the model and place it in the models directory
    ```

## Usage

1. Run the Streamlit app:
    ```sh
    streamlit run app.py
    ```

2. Upload PDF files through the sidebar.

3. Ask questions about the content of the uploaded PDFs and receive insights from the chatbot.

## Project Structure

- `app.py`: The main Streamlit application file.
- `requirements.txt`: The list of required Python packages.
- `models/`: Directory to store the LlamaCpp model. use this link to download the file that I used: "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/blob/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
- `data/`: Directory to store uploaded PDF files (if necessary).

## Functions

- **initialize_session_state**: Initializes session state variables.
- **conversation_chat**: Handles conversation with the LlamaCpp model.
- **display_chat_history**: Displays chat history in the Streamlit UI.
- **create_conversational_chain**: Creates the conversational chain using LlamaCpp and FAISS.
- **process_pdf_files**: Processes PDF files and extracts text.
- **main**: Main function to run the Streamlit app.

## Troubleshooting

- Ensure that the CUDA toolkit and PyTorch are properly installed and configured.
- Use this for PyTorch compatible with CUDA 11.8 version:
    ```sh
    pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 -f https://download.pytorch.org/whl/cu118/torch_stable.html
    ```
- Make sure the LlamaCpp model is correctly downloaded and placed in the `models` directory.
- If the app fails to load or crashes, check the Streamlit logs for detailed error messages.

## Contributing

Feel free to open issues or submit pull requests if you find any bugs or have suggestions for improvements.

## Acknowledgements

- [LangChain](https://github.com/hwchase17/langchain)
- [Streamlit](https://github.com/streamlit/streamlit)
- [FAISS](https://github.com/facebookresearch/faiss)
- [LlamaCpp](https://github.com/ggerganov/llama.cpp)

## Contact

If you have any questions or need further assistance, please contact [rahulshamr620607@gmail.com](mailto:rahulshamr620607@gmail.com).

---

Feel free to customize this README file as per your needs. Happy coding!
